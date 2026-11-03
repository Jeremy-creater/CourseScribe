"""Processing order, resuming after a crash, and what the user is told."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

import coursescribe.pipeline as flow
from coursescribe.jobs import Job, JobStore, Stage
from coursescribe.pipeline import (
    TRANSCRIPT_EMPTY,
    TRANSCRIPT_ENCODING,
    TRANSCRIPT_HELD,
    TRANSCRIPT_MISSING,
    TRANSCRIPT_UNREADABLE,
    ModelMissing,
    NoAudio,
    Pipeline,
    ProofreadBlocked,
    ProofreadResult,
    Stopped,
)
from coursescribe.ollama import OllamaClient
from coursescribe.proofread import BREAK_MARKER, TEXT_MARKER, TIMEOUT_MESSAGE, OllamaProofreader
from coursescribe.segments import Segment, read_segments, segments_to_text
from coursescribe.text import content_chars
from coursescribe.transcribe import WhisperTranscriber
from fake_ollama import FakeOllama, Reply
from test_transcribe import Engines, _broken_driver, five_seconds, quiet_logs  # noqa: F401 (a fixture)

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
    fell_back_to_cpu: bool = False

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


def _job(tmp_path: Path, store: JobStore, name: str = "课.mp3") -> Job:
    source = tmp_path / name
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


# --- transcript moved away before proofreading (docs/plans/M2-moved-transcript-tests.md) ---


def test_transcript_deleted_before_proofreading_is_written_again_from_the_checkpoint(tmp_path):
    reference_store = JobStore(tmp_path / "reference" / "state")
    reference_proofreader = FakeProofreader()
    (tmp_path / "reference").mkdir()
    reference = _job(tmp_path / "reference", reference_store)
    _pipeline(reference_store, proofreader=reference_proofreader).run(reference)

    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, proofreader=FakeProofreader(error=ProofreadBlocked("校对服务没有启动"))).run(job)
    assert job.stage is Stage.WAITING_PROOFREAD
    job.txt_path.unlink()

    _pipeline(store).run(store.load_all()[0])
    failed = store.load_all()[0]
    assert failed.stage is Stage.FAILED and failed.can_retry
    assert failed.message == TRANSCRIPT_MISSING.format(path=job.txt_path)
    assert failed.retry_stage() is Stage.TRANSCRIBING

    failed.stage = failed.retry_stage()
    transcriber, proofreader = FakeTranscriber(), FakeProofreader()
    _pipeline(store, transcriber, proofreader).run(failed)
    final = store.load_all()[0]
    assert final.stage is Stage.DONE
    assert transcriber.starts == [] and transcriber.emitted == []  # the checkpoint already held all of it
    assert final.txt_path.read_bytes() == RAW.encode()
    assert proofreader.calls == reference_proofreader.calls == [RAW]
    assert final.md_path.read_text(encoding="utf-8") == reference.md_path.read_text(encoding="utf-8")
    assert sorted(p.name for p in final.folder.iterdir()) == ["课.md", "课.txt"]


def test_a_finished_transcript_whose_checkpoint_is_gone_is_transcribed_again(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    job.transcript_complete = True  # e.g. the program data folder was cleaned
    store.save(job)
    transcriber = FakeTranscriber()
    _pipeline(store, transcriber).run(store.load_all()[0])
    final = store.load_all()[0]
    assert final.stage is Stage.DONE
    assert transcriber.starts == [0.0]
    assert final.txt_path.read_bytes() == RAW.encode()


def test_a_job_saved_before_transcripts_were_marked_complete_still_transcribes(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    record = store.folder / f"{job.job_id}.json"
    raw = json.loads(record.read_text(encoding="utf-8"))
    del raw["transcript_complete"]
    record.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    old = store.load_all()[0]
    assert old.transcript_complete is False
    transcriber = FakeTranscriber()
    _pipeline(store, transcriber).run(old)
    assert store.load_all()[0].stage is Stage.DONE
    assert transcriber.starts == [0.0]


@pytest.mark.parametrize(
    "error",
    [OSError(28, "No space left on device"), PermissionError(13, "Permission denied"), ValueError("x")],
    ids=["disk-full", "refused", "other"],
)
def test_errors_after_the_transcript_was_read_are_not_blamed_on_the_transcript(tmp_path, error):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, proofreader=FakeProofreader(error=error)).run(job)
    assert job.stage is Stage.FAILED
    transcript_messages = {
        template.format(path=job.txt_path)
        for template in (TRANSCRIPT_MISSING, TRANSCRIPT_HELD, TRANSCRIPT_UNREADABLE, TRANSCRIPT_ENCODING, TRANSCRIPT_EMPTY)
    }
    assert job.message not in transcript_messages
    assert str(job.txt_path) not in job.message
    assert job.txt_path.read_bytes() == RAW.encode()


THREE_PARTS =[Segment(float(i), float(i + 1), mark * 89) for i, mark in enumerate("甲乙丙")]


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


# --- the GPU cannot be used this time (docs/plans/M2-gpu-fallback-tests.md) ---
# The real WhisperTranscriber, with stand-ins only for the engine and the graphics driver.

GENERIC_FAILURE = "处理时出错了。录音和已有文件都还在，详情见日志。"
_PROOFREAD_TAIL = ["正在准备校对", "正在校对（第 1/3 段）", "完成"]
PLAIN_STATUSES = ["正在准备转录", *[f"正在转录（{p}%）" for p in (20, 40, 60, 80, 100)], *_PROOFREAD_TAIL]


def _noted(percents):
    return [f"正在转录（{p}%）。{flow.CPU_FALLBACK_NOTE}" for p in percents]


def _fallback_at_load():
    return ["正在准备转录", *_noted((0, 20, 40, 60, 80, 100)), *_PROOFREAD_TAIL]


def _fallback_after_first_segment():
    return ["正在准备转录", "正在转录（20%）", *_noted((20, 40, 60, 80, 100)), *_PROOFREAD_TAIL]


@dataclass
class MemoryCheckingProofreader(FakeProofreader):
    """Notes, when proofreading starts, which transcription models are still in memory."""

    engines: Engines | None = None
    alive: list[list[bool]] = field(default_factory=list)

    def proofread(self, text, model, checkpoint, on_progress):
        gc.collect()
        self.alive.append([ref() is not None for _, ref in self.engines.refs])
        return super().proofread(text, model, checkpoint, on_progress)


def _statuses_by_job():
    seen: dict[str, list[str]] = {}
    return seen, lambda job, text, fraction: seen.setdefault(job.job_id, []).append(text)


@pytest.mark.parametrize(
    ("plan", "has_gpu", "expected"),
    [
        (["ok"], lambda: True, "plain"),
        (["ok"], lambda: False, "plain"),
        (["load_fails"], lambda: True, "at_load"),
        (["fails_after:1"], lambda: True, "after_first"),
        (["ok"], _broken_driver, "at_load"),
    ],
    ids=["gpu-ok", "no-gpu", "gpu-load-fails", "gpu-fails-after-one", "driver-broken"],
)
def test_status_says_it_will_be_slower_from_the_moment_the_cpu_takes_over(tmp_path, quiet_logs, plan, has_gpu, expected):
    expected = {"plain": PLAIN_STATUSES, "at_load": _fallback_at_load(), "after_first": _fallback_after_first_segment()}[expected]
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    engines = Engines(plan)
    proofreader = MemoryCheckingProofreader(engines=engines)
    seen, on_status = _statuses_by_job()
    _pipeline(store, WhisperTranscriber(engines, five_seconds, has_gpu), proofreader, on_status=on_status).run(job)
    assert seen[job.job_id] == expected
    assert store.load_all()[0].stage is Stage.DONE
    assert job.txt_path.read_text(encoding="utf-8") == RAW
    assert proofreader.alive and all(not any(alive) for alive in proofreader.alive)
    if expected == PLAIN_STATUSES:
        assert not any(word in text for text in seen[job.job_id] for word in ("CPU", "显卡", "慢"))


def test_after_a_job_on_the_cpu_the_next_job_has_the_gpu_and_no_note(tmp_path, quiet_logs):
    store = JobStore(tmp_path / "state")
    first, second = _job(tmp_path, store, "第一课.mp3"), _job(tmp_path, store, "第二课.mp3")
    engines = Engines(["load_fails", "ok"])
    proofreader = MemoryCheckingProofreader(engines=engines)
    seen, on_status = _statuses_by_job()
    pipeline = _pipeline(store, WhisperTranscriber(engines, five_seconds, lambda: True), proofreader, on_status=on_status)
    pipeline.run(first)
    pipeline.run(second)
    assert seen[first.job_id] == _fallback_at_load()
    assert seen[second.job_id] == PLAIN_STATUSES
    assert [job.stage for job in store.load_all()] == [Stage.DONE, Stage.DONE]
    assert first.txt_path.read_text(encoding="utf-8") == second.txt_path.read_text(encoding="utf-8") == RAW
    assert len(proofreader.alive) == 2 and all(not any(alive) for alive in proofreader.alive)


def test_the_fallback_note_and_the_start_failure_say_what_happens_in_plain_words():
    note, failed = flow.CPU_FALLBACK_NOTE, flow.MODEL_LOAD_FAILED.format(model="large-v3")
    assert all(word in note for word in ("显卡", "CPU", "慢", "下一个任务"))
    assert all(word in failed for word in ("large-v3", "内存", "重试"))
    for text in (note, failed):
        assert not any(word in text for word in ("CUDA", "显存", "驱动"))
    assert "显卡" not in failed  # true on a computer without one as well


def _stop_once_a_segment_is_saved(store, job):
    return lambda: bool(read_segments(store.segments_path(job)))


CUDA, CPU = ("large-v3", "cuda"), ("large-v3", "cpu")


@pytest.mark.parametrize(
    ("engines", "has_gpu", "stop", "stage", "loads", "kept"),
    [
        (lambda: Engines(["fails_after:2"], gpu_error=ValueError("bad")), True, False, Stage.FAILED, [CUDA], 2),
        (lambda: Engines(["load_fails"], cpu="fails_after:1"), True, False, Stage.FAILED, [CUDA, CPU], 1),
        (lambda: Engines(["load_fails"]), True, True, Stage.TRANSCRIBING, [CUDA, CPU], 1),
        (lambda: Engines(["ok"], cpu="missing"), False, False, Stage.WAITING_MODEL, [CPU], 0),
    ],
    ids=["gpu-plain-error", "cpu-plain-error", "closing-on-cpu", "cpu-model-missing"],
)
def test_transcription_that_ends_badly_is_told_as_it_is_and_gives_the_model_back(
    tmp_path, quiet_logs, engines, has_gpu, stop, stage, loads, kept
):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    engines = engines()
    should_stop = _stop_once_a_segment_is_saved(store, job) if stop else (lambda: False)
    transcriber = WhisperTranscriber(engines, five_seconds, lambda: has_gpu, should_stop)
    _pipeline(store, transcriber).run(job)
    saved = store.load_all()[0]
    assert saved.stage is stage
    if stage is Stage.FAILED:
        assert saved.message == GENERIC_FAILURE
    if stage is Stage.WAITING_MODEL:
        assert "large-v3" in saved.message
    assert saved.message != flow.MODEL_LOAD_FAILED.format(model="large-v3")
    assert engines.loads == loads  # a plain error on the GPU is no reason to use the CPU
    assert [s.text for s in read_segments(store.segments_path(job))] == [f"第{i}句" for i in range(kept)]
    assert bool(engines.refs) is (stage is not Stage.WAITING_MODEL)  # a model existed unless none could load
    assert not any(engines.alive())


def test_model_that_cannot_start_fails_retryably_and_retry_continues_where_it_stopped(tmp_path, quiet_logs):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    engines = Engines(["fails_after:2"], cpu="load_fails")
    proofreader = FakeProofreader()
    _pipeline(store, WhisperTranscriber(engines, five_seconds, lambda: True), proofreader).run(job)
    failed = store.load_all()[0]
    assert (failed.stage, failed.can_retry) == (Stage.FAILED, True)
    assert failed.message == flow.MODEL_LOAD_FAILED.format(model="large-v3")
    assert proofreader.calls == []
    assert not job.txt_path.exists()
    assert [s.text for s in read_segments(store.segments_path(job))] == ["第0句", "第1句"]
    assert not any(engines.alive())

    healthy = Engines(["ok"])
    failed.stage = failed.retry_stage()
    _pipeline(store, WhisperTranscriber(healthy, five_seconds, lambda: True)).run(failed)
    final = store.load_all()[0]
    assert final.stage is Stage.DONE
    assert healthy.loads == [("large-v3", "cuda")]
    assert healthy.seen_by("cuda") == [3.0]
    assert final.txt_path.read_text(encoding="utf-8") == RAW


def test_closed_while_on_the_cpu_restarts_on_the_gpu_without_repeating_or_missing(tmp_path, quiet_logs):
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    dying = Engines(["fails_after:1"], cpu="fails_after:1", cpu_error=Crash())
    with pytest.raises(Crash):
        _pipeline(store, WhisperTranscriber(dying, five_seconds, lambda: True)).run(job)

    restarted = store.load_all()[0]
    healthy = Engines(["ok"])
    seen, on_status = _statuses_by_job()
    _pipeline(store, WhisperTranscriber(healthy, five_seconds, lambda: True), on_status=on_status).run(restarted)
    final = store.load_all()[0]
    assert final.stage is Stage.DONE
    assert final.txt_path.read_text(encoding="utf-8") == RAW
    assert healthy.loads == [("large-v3", "cuda")]
    assert healthy.seen_by("cuda") == [3.0]
    assert not any(flow.CPU_FALLBACK_NOTE in text for text in seen[restarted.job_id])
    assert sorted(p.name for p in final.folder.iterdir()) == ["课.md", "课.txt"]
