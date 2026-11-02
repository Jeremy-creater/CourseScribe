"""Processing order, resuming after a crash, and what the user is told."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

from coursescribe.jobs import Job, JobStore, Stage
from coursescribe.pipeline import (
    ModelMissing,
    NoAudio,
    Pipeline,
    ProofreadBlocked,
    ProofreadResult,
    Stopped,
)
from coursescribe.ollama import OllamaClient
from coursescribe.proofread import BREAK_MARKER, TEXT_MARKER, TIMEOUT_MESSAGE, OllamaProofreader
from coursescribe.segments import Segment, segments_to_text
from coursescribe.text import content_chars
from fake_ollama import FakeOllama, Reply

NOW = datetime(2026, 11, 1, 19, 30)
SEGMENTS = [Segment(float(i), float(i + 1), f"第{i}句") for i in range(5)]
RAW = "第0句\n第1句\n第2句\n第3句\n第4句\n"


class Crash(BaseException):
    """Stands in for the process dying: nothing may catch and tidy up after it."""


@dataclass
class FakeTranscriber:
    segments: list[Segment] = field(default_factory=lambda: list(SEGMENTS))
    crash_after: int | None = None
    error: Exception | None = None
    starts: list[float] = field(default_factory=list)
    emitted: list[Segment] = field(default_factory=list)
    releases: int = 0
    events: list[str] | None = None

    def transcribe(self, audio: Path, model: str, start_at: float, on_segment, on_progress) -> None:
        self.starts.append(start_at)
        if self.error:
            raise self.error
        for segment in self.segments:
            if segment.start < start_at:
                continue
            if self.crash_after is not None and len(self.emitted) >= self.crash_after:
                raise Crash
            self.emitted.append(segment)
            on_segment(segment)
            on_progress(segment.end, 5.0)

    def release(self) -> None:
        self.releases += 1
        if self.events is not None:
            self.events.append("release")


@dataclass
class FakeProofreader:
    error: Exception | None = None
    calls: list[str] = field(default_factory=list)
    result: ProofreadResult = field(default_factory=lambda: ProofreadResult("校对后的正文。\n", 0, 3))
    events: list[str] | None = None

    def proofread(self, text: str, model: str, checkpoint: Path, on_progress) -> ProofreadResult:
        self.calls.append(text)
        if self.events is not None:
            self.events.append("proofread")
        if self.error:
            raise self.error
        on_progress(1, 3)
        return self.result

    def release(self, model: str) -> None:
        pass


class CrashingStore(JobStore):
    """Dies right after the Nth save, as a power cut would."""

    def __init__(self, folder: Path, crash_at_save: int) -> None:
        super().__init__(folder)
        self.left = crash_at_save

    def save(self, job: Job) -> None:
        super().save(job)
        self.left -= 1
        if self.left == 0:
            raise Crash


def _job(tmp_path: Path, store: JobStore) -> Job:
    source = tmp_path / "课.mp3"
    source.write_bytes(b"audio")
    job = Job.new_file(source, tmp_path / "out", NOW, "large-v3", "model-a")
    store.save(job)
    return job


def _pipeline(store, transcriber=None, proofreader=None, repair=lambda path: 5.0, **kw) -> Pipeline:
    return Pipeline(store, transcriber or FakeTranscriber(), proofreader or FakeProofreader(), repair, **kw)


def test_full_run_writes_both_files_and_finishes(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store).run(job)
    assert job.stage is Stage.DONE
    assert job.txt_path.read_text(encoding="utf-8") == RAW
    assert job.md_path.read_text(encoding="utf-8") == "# 课\n\n校对后的正文。\n"
    assert store.load_all()[0].stage is Stage.DONE


def test_transcriber_is_released_before_proofreading_starts(tmp_path):
    store = JobStore(tmp_path / "state")
    events: list[str] = []
    job = _job(tmp_path, store)
    _pipeline(store, FakeTranscriber(events=events), FakeProofreader(events=events)).run(job)
    assert events[:2] == ["release", "proofread"]


def test_source_file_is_left_untouched(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    before = (job.audio_path.read_bytes(), job.audio_path.stat().st_mtime_ns)
    _pipeline(store).run(job)
    assert (job.audio_path.read_bytes(), job.audio_path.stat().st_mtime_ns) == before


def test_txt_is_byte_identical_after_proofreading(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    proofreader = FakeProofreader()
    _pipeline(store, proofreader=proofreader).run(job)
    assert proofreader.calls == [RAW]
    assert job.txt_path.read_bytes() == RAW.encode("utf-8")


def test_crash_during_transcription_continues_after_the_last_segment(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    first = FakeTranscriber(crash_after=2)
    with pytest.raises(Crash):
        _pipeline(store, first).run(job)
    assert not job.txt_path.exists()

    reloaded = store.load_all()[0]
    second = FakeTranscriber()
    _pipeline(store, second).run(reloaded)
    assert second.starts == [2.0]
    assert [s.text for s in second.emitted] == ["第2句", "第3句", "第4句"]
    assert reloaded.txt_path.read_text(encoding="utf-8") == RAW


def test_half_written_last_checkpoint_line_is_dropped_and_redone(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    with pytest.raises(Crash):
        _pipeline(store, FakeTranscriber(crash_after=2)).run(job)
    checkpoint = store.segments_path(job)
    with checkpoint.open("ab") as stream:
        stream.write(b'{"start": 2.0, "end"')

    second = FakeTranscriber()
    _pipeline(store, second).run(store.load_all()[0])
    assert second.starts == [2.0]
    assert job.txt_path.read_text(encoding="utf-8") == RAW


@pytest.mark.parametrize("crash_at_save", range(1, 7))
def test_power_cut_after_any_state_save_still_ends_with_correct_files(tmp_path, crash_at_save):
    plain = JobStore(tmp_path / "state")
    job = _job(tmp_path, plain)
    try:
        _pipeline(CrashingStore(tmp_path / "state", crash_at_save)).run(job)
    except Crash:
        pass

    resumed = plain.load_all()[0]
    if resumed.stage is not Stage.DONE:
        _pipeline(plain).run(resumed)
    final = plain.load_all()[0]
    assert final.stage is Stage.DONE
    assert final.txt_path.read_text(encoding="utf-8") == RAW
    assert final.md_path.read_text(encoding="utf-8") == "# 课\n\n校对后的正文。\n"
    assert sorted(p.name for p in final.folder.iterdir()) == ["课.md", "课.txt"]


def test_finished_transcription_is_not_redone_when_proofreading_resumes(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, proofreader=FakeProofreader(error=ProofreadBlocked("校对服务没有启动"))).run(job)
    assert job.stage is Stage.WAITING_PROOFREAD

    transcriber = FakeTranscriber()
    _pipeline(store, transcriber).run(store.load_all()[0])
    assert transcriber.starts == []
    assert store.load_all()[0].stage is Stage.DONE


def test_blocked_proofreading_keeps_txt_and_writes_no_md(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, proofreader=FakeProofreader(error=ProofreadBlocked("校对服务没有启动，请先打开 Ollama"))).run(job)
    assert job.stage is Stage.WAITING_PROOFREAD
    assert job.message == "校对服务没有启动，请先打开 Ollama"
    assert job.txt_path.read_text(encoding="utf-8") == RAW
    assert not job.md_path.exists()
    assert store.load_all()[0].message == job.message


def test_unproofread_count_is_saved_and_reported(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, proofreader=FakeProofreader(result=ProofreadResult("正文\n", 2, 20))).run(job)
    saved = store.load_all()[0]
    assert (saved.unproofread_chunks, saved.total_chunks) == (2, 20)
    assert saved.stage is Stage.DONE
    assert "2" in saved.message and "没校对" in saved.message


def test_missing_transcription_model_waits_for_consent_and_keeps_progress(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, FakeTranscriber(error=ModelMissing("large-v3"))).run(job)
    assert job.stage is Stage.WAITING_MODEL
    assert "large-v3" in job.message
    assert not job.txt_path.exists()
    assert job in store.unfinished()


def test_recording_with_no_speech_fails_without_writing_files(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, FakeTranscriber(segments=[])).run(job)
    assert job.stage is Stage.FAILED
    assert "没有识别到" in job.message
    assert list(job.folder.iterdir()) == []


def test_unreadable_audio_is_reported_in_plain_words(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, FakeTranscriber(error=NoAudio("读不出声音"))).run(job)
    assert job.stage is Stage.FAILED
    assert job.message == "读不出声音"


def test_unexpected_error_marks_failed_without_a_traceback_in_the_message(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, FakeTranscriber(error=RuntimeError("cuda exploded at 0x1234"))).run(job)
    assert job.stage is Stage.FAILED
    assert "0x1234" not in job.message
    assert "出错" in job.message


def test_stopping_leaves_the_stage_for_the_next_start(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, FakeTranscriber(error=Stopped())).run(job)
    assert store.load_all()[0].stage is Stage.TRANSCRIBING


def test_interrupted_recording_is_repaired_before_transcribing(tmp_path):
    store = JobStore(tmp_path / "state")
    job = Job.new_recording(tmp_path / "out", NOW, "large-v3", "model-a")
    job.audio_path.write_bytes(b"partial")
    store.save(job)
    repaired: list[Path] = []

    def repair(path: Path) -> float:
        repaired.append(path)
        return 12.0

    _pipeline(store, repair=repair).run(store.load_all()[0])
    assert repaired == [job.audio_path]
    assert store.load_all()[0].stage is Stage.DONE


def test_interrupted_recording_with_no_usable_sound_fails_clearly(tmp_path):
    store = JobStore(tmp_path / "state")
    job = Job.new_recording(tmp_path / "out", NOW, "large-v3", "model-a")
    store.save(job)
    _pipeline(store, repair=lambda path: 0.0).run(job)
    assert job.stage is Stage.FAILED
    assert "录音" in job.message


def test_progress_messages_are_plain_words(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    seen: list[str] = []
    _pipeline(store, on_status=lambda j, text, fraction: seen.append(text)).run(job)
    assert "正在校对（第 1/3 段）" in seen
    assert any(text.startswith("正在转录") for text in seen)
    assert seen[-1] == "完成"


def test_checkpoints_are_removed_after_finishing(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store).run(job)
    assert not store.segments_path(job).exists()
    assert not store.proofread_path(job).exists()


THREE_PARTS = [Segment(float(i), float(i + 1), mark * 89) for i, mark in enumerate("甲乙丙")]


def _three_part_model(slow_part: str | None = None):
    """Gives back each chunk with a full stop; the chunk starting with ``slow_part`` answers too late."""

    def respond(prompt):
        if BREAK_MARKER in prompt:
            return "无"
        chunk = prompt.rsplit(TEXT_MARKER, 1)[1]
        return Reply("晚了", delay=5.0) if chunk[0] == slow_part else chunk + "。"

    return respond


def _real_proofreader(fake: FakeOllama) -> OllamaProofreader:
    client = OllamaClient(fake.url, timeout=0.5, unload_timeout=0.5)
    return OllamaProofreader(client, chunk_chars=100, retry_pause=0)


def test_slow_proofreading_service_waits_then_resumes_into_one_md(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    expected_txt = segments_to_text(THREE_PARTS).encode("utf-8")
    slow = FakeOllama(_three_part_model(slow_part="乙"))
    try:
        _pipeline(store, FakeTranscriber(segments=list(THREE_PARTS)), _real_proofreader(slow)).run(job)
    finally:
        slow.close()
    assert job.stage is Stage.WAITING_PROOFREAD
    assert job.message == TIMEOUT_MESSAGE
    saved = store.load_all()[0]
    assert (saved.stage, saved.message) == (Stage.WAITING_PROOFREAD, TIMEOUT_MESSAGE)
    assert list(job.folder.glob("*.md")) == []
    assert job.txt_path.read_bytes() == expected_txt
    assert store.proofread_path(job).exists()

    fixed = FakeOllama(_three_part_model())
    try:
        _pipeline(store, FakeTranscriber(segments=list(THREE_PARTS)), _real_proofreader(fixed)).run(saved)
    finally:
        fixed.close()
    final = store.load_all()[0]
    assert final.stage is Stage.DONE
    assert final.unproofread_chunks == 0
    assert [p.name for p in final.folder.glob("*.md")] == ["课.md"]
    markdown = final.md_path.read_text(encoding="utf-8")
    assert markdown.startswith("# 课\n\n")
    assert content_chars(markdown.removeprefix("# 课\n\n")) == content_chars(expected_txt.decode("utf-8"))
    assert final.txt_path.read_bytes() == expected_txt
