"""The whole application behind the window, with stand-ins for models and devices."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from coursescribe.controller import AppController
from coursescribe.jobs import Job, JobStore, Stage
from coursescribe.pipeline import ModelMissing, ProofreadBlocked, ProofreadResult
from coursescribe.segments import Segment
from coursescribe.settings import Settings, SettingsStore

NOW = datetime(2026, 11, 1, 19, 30)


class FakeTranscriber:
    def __init__(self):
        self.models = []
        self.missing = set()
        self.gate = None

    def transcribe(self, audio, model, start_at, on_segment, on_progress):
        self.models.append(model)
        if model in self.missing:
            raise ModelMissing(model)
        if self.gate:
            self.gate.wait(5)
        on_segment(Segment(0.0, 1.0, f"{Path(audio).stem}的内容"))
        on_progress(1.0, 2.0)

    def release(self):
        pass


class FakeProofreader:
    def __init__(self):
        self.blocked = None
        self.error = None
        self.models = []

    def proofread(self, text, model, checkpoint, on_progress):
        self.models.append(model)
        if self.error:
            raise self.error
        if self.blocked:
            raise ProofreadBlocked(self.blocked)
        on_progress(1, 1)
        return ProofreadResult(text.strip() + "。\n", 0, 1)

    def release(self, model):
        pass


class FakeCapture:
    sample_rate = 48000
    channels = 2

    def __init__(self, fail=False):
        self.fail = fail

    def __enter__(self):
        if self.fail:
            raise RuntimeError("no device")
        return self

    def __exit__(self, *args):
        pass

    def read(self, frames):
        time.sleep(0.005)
        return np.full((frames, 2), 3000, dtype=np.int16)


class Rig:
    def __init__(self, tmp_path: Path, capture_fails=False):
        self.tmp_path = tmp_path
        self.store = JobStore(tmp_path / "state")
        self.settings = SettingsStore(tmp_path / "settings.json", Settings(tmp_path / "out", "large-v3", "model-a"))
        self.transcriber = FakeTranscriber()
        self.proofreader = FakeProofreader()
        self.downloads = []
        self.download_error = None
        self.states = []
        self.controller = AppController(
            store=self.store,
            settings=self.settings,
            transcriber=self.transcriber,
            proofreader=self.proofreader,
            open_capture=lambda: FakeCapture(capture_fails),
            download_model=self._download,
            stopping=threading.Event(),
            now=lambda: NOW,
        )
        self.controller.add_listener(self.states.append)

    def _download(self, name):
        if self.download_error:
            raise self.download_error
        self.downloads.append(name)
        self.transcriber.missing.discard(name)

    def source(self, name="第一课.mp3") -> Path:
        path = self.tmp_path / "input" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"audio")
        return path

    def wait(self, condition, seconds=10.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if condition():
                return
            time.sleep(0.005)
        raise AssertionError(f"Timed out; last state: {self.controller.state}")

    def wait_idle(self):
        self.wait(lambda: self.controller.state.pending == 0)


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path)
    yield made
    made.controller.shutdown()


def test_chosen_file_goes_all_the_way_with_no_other_action(rig):
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    folder = rig.tmp_path / "out" / "第一课"
    assert (folder / "第一课.txt").read_text(encoding="utf-8") == "第一课的内容\n"
    assert (folder / "第一课.md").read_text(encoding="utf-8") == "# 第一课\n\n第一课的内容。\n"
    assert rig.controller.state.last_folder == folder
    assert "完成" in rig.controller.state.notice


def test_several_files_are_processed_in_order(rig):
    rig.controller.start()
    rig.transcriber.gate = threading.Event()
    rig.controller.add_files([rig.source("a.mp3"), rig.source("b.mp4"), rig.source("c.mkv")])
    assert rig.controller.state.pending == 3
    rig.transcriber.gate.set()
    rig.wait_idle()
    assert [job.txt_path.stem for job in rig.store.load_all()] == ["a", "b", "c"]
    assert all(job.stage is Stage.DONE for job in rig.store.load_all())


def test_unsupported_and_missing_files_are_refused_with_a_reason(rig):
    rig.controller.start()
    notes = rig.tmp_path / "input"
    notes.mkdir()
    (notes / "notes.docx").write_bytes(b"x")
    rig.controller.add_files([notes / "notes.docx", notes / "gone.mp3", notes])
    assert rig.store.load_all() == []
    assert "notes.docx" in rig.controller.state.notice and "不支持" in rig.controller.state.notice


@pytest.mark.parametrize("name", ["a.MP3", "a.wav", "a.m4a", "a.mp4", "a.mkv", "a.flac", "a.mov", "a.webm"])
def test_common_media_types_are_accepted(rig, name):
    rig.controller.start()
    rig.controller.add_files([rig.source(name)])
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.DONE


def test_recording_start_stop_produces_all_three_files(rig):
    rig.controller.start()
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is True
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.5)
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is False
    rig.wait_idle()
    job = rig.store.load_all()[0]
    assert job.stage is Stage.DONE
    assert job.audio_path.stat().st_size > 44
    assert job.audio_path.suffix == ".wav" and job.audio_path.parent == job.folder
    assert job.txt_path.exists() and job.md_path.exists()


def test_recording_state_is_on_disk_while_recording(rig):
    rig.controller.start()
    rig.controller.toggle_recording()
    assert rig.store.unfinished()[0].stage is Stage.RECORDING
    rig.controller.toggle_recording()


def test_device_failure_at_start_tells_the_user_and_leaves_no_job(tmp_path):
    rig = Rig(tmp_path, capture_fails=True)
    rig.controller.start()
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is False
    assert "没能开始录音" in rig.controller.state.notice
    assert rig.store.unfinished() == []
    rig.controller.shutdown()


def test_unfinished_work_continues_at_the_next_start_including_recording(rig):
    source = rig.source()
    waiting = Job.new_file(source, rig.tmp_path / "out", NOW, "large-v3", "model-a")
    rig.store.save(waiting)
    interrupted = Job.new_recording(rig.tmp_path / "out", NOW, "large-v3", "model-a")
    from coursescribe.wavfile import WavWriter

    writer = WavWriter(interrupted.audio_path, 48000, 2)
    writer.write(np.full((48000, 2), 3000, dtype=np.int16))
    writer.close()
    rig.store.save(interrupted)

    rig.controller.start()
    rig.wait_idle()
    assert [job.stage for job in rig.store.load_all()] == [Stage.DONE, Stage.DONE]


def test_blocked_proofreading_offers_retry_and_retry_finishes_it(rig):
    rig.proofreader.blocked = "校对服务（Ollama）没有启动。"
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    job = rig.store.load_all()[0]
    assert job.stage is Stage.WAITING_PROOFREAD
    assert job.txt_path.exists() and not job.md_path.exists()
    assert rig.controller.state.waiting_proofread == 1
    assert "Ollama" in rig.controller.state.notice

    rig.proofreader.blocked = None
    rig.controller.retry_waiting()
    rig.wait_idle()
    rig.wait(lambda: rig.controller.state.waiting_proofread == 0)
    assert rig.store.load_all()[0].md_path.exists()


def test_retry_uses_the_proofreading_model_chosen_since(rig):
    rig.proofreader.blocked = "没有模型"
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    rig.proofreader.blocked = None
    rig.controller.update_settings(Settings(rig.tmp_path / "out", "large-v3", "model-b"))
    rig.controller.retry_waiting()
    rig.wait(lambda: rig.controller.state.waiting_proofread == 0)
    rig.wait_idle()
    assert rig.proofreader.models[-1] == "model-b"


def test_new_settings_apply_to_the_next_job_only(rig):
    rig.transcriber.gate = threading.Event()
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    rig.wait(lambda: rig.transcriber.models == ["large-v3"])
    new_out = rig.tmp_path / "other"
    rig.controller.update_settings(Settings(new_out, "medium", "model-b"))
    rig.controller.add_files([rig.source("b.mp3")])
    rig.transcriber.gate.set()
    rig.wait_idle()
    assert rig.transcriber.models == ["large-v3", "medium"]
    assert rig.proofreader.models == ["model-a", "model-b"]
    assert (new_out / "b" / "b.md").exists()
    assert (rig.tmp_path / "out" / "a" / "a.md").exists()


def test_model_is_not_downloaded_until_the_user_agrees(rig):
    rig.transcriber.missing.add("large-v3")
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    assert rig.downloads == []
    assert rig.controller.state.model_needed == "large-v3"
    assert rig.store.load_all()[0].stage is Stage.WAITING_MODEL

    rig.controller.download_needed_model()
    rig.wait(lambda: rig.controller.state.model_needed is None and rig.controller.state.pending == 0)
    assert rig.downloads == ["large-v3"]
    assert rig.store.load_all()[0].stage is Stage.DONE


def test_failed_download_keeps_waiting_and_says_so(rig):
    rig.transcriber.missing.add("large-v3")
    rig.download_error = OSError("no internet")
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    rig.controller.download_needed_model()
    rig.wait(lambda: "下载失败" in rig.controller.state.notice)
    rig.wait_idle()
    assert rig.controller.state.model_needed == "large-v3"
    assert rig.store.load_all()[0].stage is Stage.WAITING_MODEL


def test_progress_reaches_the_listener_in_plain_words(rig):
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    texts = [state.status for state in rig.states]
    assert any("正在转录" in text for text in texts)
    assert any("正在校对（第 1/1 段）" in text for text in texts)


def test_silence_warning_shows_and_clears(rig):
    rig.controller.start()
    rig.controller._on_silence(True)
    assert rig.controller.state.silence_warning is True
    rig.controller._on_silence(False)
    assert rig.controller.state.silence_warning is False


def test_shutdown_while_recording_saves_the_recording_for_next_time(rig):
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.25)
    rig.controller.shutdown()
    job = rig.store.load_all()[0]
    assert job.stage is Stage.TRANSCRIBING
    assert job.audio_path.stat().st_size > 44
    assert all(state.pending == 0 for state in rig.states)  # closing never starts the work


def test_settings_store_round_trip_and_defaults(tmp_path):
    default = Settings(tmp_path / "docs", "large-v3", "model-a")
    store = SettingsStore(tmp_path / "s.json", default)
    assert store.load() == default
    changed = Settings(tmp_path / "x", "medium", "model-b")
    store.save(changed)
    assert SettingsStore(tmp_path / "s.json", default).load() == changed


def test_damaged_settings_fall_back_to_defaults(tmp_path):
    (tmp_path / "s.json").write_text("{broken", encoding="utf-8")
    default = Settings(tmp_path / "docs", "large-v3", "model-a")
    assert SettingsStore(tmp_path / "s.json", default).load() == default


def test_default_folders_come_from_the_system(monkeypatch, tmp_path):
    from coursescribe.settings import app_home, default_settings

    monkeypatch.delenv("COURSESCRIBE_HOME", raising=False)
    home = app_home()
    assert home.name == "CourseScribe"
    settings = default_settings()
    assert settings.output_dir.name == "CourseScribe"
    assert settings.output_dir.parent.name in ("Documents", "文档")
    assert settings.transcribe_model == "large-v3"
    monkeypatch.setenv("COURSESCRIBE_HOME", str(tmp_path))
    assert app_home() == tmp_path


def test_clicking_retry_twice_does_not_process_a_job_twice(rig):
    rig.proofreader.blocked = "没有启动"
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    rig.proofreader.blocked = None
    rig.proofreader.models.clear()
    rig.controller.retry_waiting()
    rig.controller.retry_waiting()
    rig.wait_idle()
    rig.wait(lambda: rig.controller.state.waiting_proofread == 0)
    assert len(rig.proofreader.models) == 1
    assert rig.controller.state.pending == 0


def _disk_fails_after(monkeypatch, writes):
    from coursescribe import wavfile

    real_write = wavfile.WavWriter.write
    calls = []

    def write(self, frames):
        calls.append(1)
        if len(calls) > writes:
            raise OSError(28, "No space left on device")
        return real_write(self, frames)

    monkeypatch.setattr(wavfile.WavWriter, "write", write)


def test_recording_that_had_to_stop_by_itself_is_transcribed_and_explained(rig, monkeypatch):
    _disk_fails_after(monkeypatch, 4)
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.state.recording is False)
    rig.wait_idle()
    job = rig.store.load_all()[0]
    assert job.stage is Stage.DONE
    assert any("磁盘" in state.notice for state in rig.states)


def _state_when_recording_stopped(states):
    was_recording = False
    for state in states:
        if state.recording:
            was_recording = True
        elif was_recording:
            return state
    raise AssertionError("the recording never stopped")


def test_stopped_recording_is_never_shown_as_idle_before_it_is_queued(rig):
    """Otherwise the window briefly says nothing is happening, and anything waiting for idle sees the job unfinished."""
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.25)
    rig.controller.toggle_recording()
    rig.wait_idle()
    assert _state_when_recording_stopped(rig.states).pending == 1


def test_recording_that_stopped_by_itself_is_never_shown_as_idle_before_it_is_queued(rig, monkeypatch):
    _disk_fails_after(monkeypatch, 4)
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.state.recording is False)
    rig.wait(lambda: rig.store.load_all()[0].stage is Stage.DONE)
    stopped = _state_when_recording_stopped(rig.states)
    assert stopped.pending == 1
    assert "磁盘" in stopped.notice  # the reason is already there when the button changes back


def test_recording_file_that_could_not_be_finished_is_kept_for_repair(rig, monkeypatch):
    from coursescribe import wavfile

    def close(self):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(wavfile.WavWriter, "close", close)
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.25)
    rig.controller.shutdown()
    assert rig.store.load_all()[0].stage is Stage.RECORDING  # repaired before transcribing at the next start


def test_stop_ends_the_recording_even_if_the_file_cannot_be_finished(rig, monkeypatch):
    from coursescribe import wavfile

    def close(self):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(wavfile.WavWriter, "close", close)
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.25)
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is False
    rig.wait_idle()
    job = rig.store.load_all()[0]
    assert job.stage is Stage.DONE
    import wave

    with wave.open(str(job.audio_path), "rb") as stream:
        frames = stream.getnframes()
    assert frames > 0 and frames * 4 == job.audio_path.stat().st_size - 44  # header repaired, nothing lost


def _saved_job(rig, attempts=0, stage=Stage.TRANSCRIBING):
    job = Job.new_file(rig.source(), rig.tmp_path / "out", NOW, "large-v3", "model-a")
    job.stage = stage
    job.attempts = attempts
    rig.store.save(job)
    return job


def test_job_that_took_the_program_down_twice_is_paused_not_resumed(rig):
    job = _saved_job(rig, attempts=2)
    rig.controller.start()
    rig.wait(lambda: rig.controller.state.failed == 1)
    assert rig.transcriber.models == []
    saved = rig.store.load_all()[0]
    assert saved.job_id == job.job_id and saved.stage is Stage.FAILED
    assert "暂停" in rig.controller.state.notice and "重试" in rig.controller.state.notice


def test_job_that_took_the_program_down_once_is_resumed(rig):
    _saved_job(rig, attempts=1)
    rig.controller.start()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.DONE


def test_closing_the_program_normally_does_not_count_as_a_crash(tmp_path):
    first = Rig(tmp_path)
    _saved_job(first, attempts=2)
    first.store.mark_closed_cleanly()
    first.controller.start()
    first.wait_idle()
    assert first.store.load_all()[0].stage is Stage.DONE
    first.controller.shutdown()


def test_shutdown_records_a_clean_close_and_the_next_start_consumes_it(rig):
    rig.controller.start()
    rig.controller.shutdown()
    assert rig.store.take_closed_cleanly() is True
    assert rig.store.take_closed_cleanly() is False


def test_attempts_go_back_to_zero_after_a_run_that_ends_normally(rig):
    rig.proofreader.blocked = "没有启动"
    _saved_job(rig, attempts=1)
    rig.controller.start()
    rig.wait_idle()
    job = rig.store.load_all()[0]
    assert job.stage is Stage.WAITING_PROOFREAD and job.attempts == 0


def test_failed_job_retries_from_the_step_that_failed(rig):
    rig.proofreader.error = RuntimeError("boom")
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.FAILED
    assert rig.controller.state.failed == 1
    rig.proofreader.error = None
    rig.transcriber.models.clear()
    rig.controller.retry_failed()
    rig.wait_idle()
    job = rig.store.load_all()[0]
    assert job.stage is Stage.DONE
    assert rig.transcriber.models == []  # the .txt was kept, so no second transcription
    assert job.md_path.exists()
    assert rig.controller.state.failed == 0


def test_failed_file_job_without_a_txt_retries_the_transcription(rig):
    job = _saved_job(rig, stage=Stage.FAILED)
    rig.controller.start()
    rig.wait(lambda: rig.controller.state.failed == 1)
    rig.controller.retry_failed()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.DONE
    assert rig.transcriber.models == ["large-v3"]
    assert job.job_id == rig.store.load_all()[0].job_id


def test_failed_recording_retry_starts_by_checking_the_recording(rig):
    job = Job.new_recording(rig.tmp_path / "out", NOW, "large-v3", "model-a")
    from coursescribe.wavfile import WavWriter

    writer = WavWriter(job.audio_path, 48000, 2)
    writer.write(np.full((48000, 2), 3000, dtype=np.int16))
    writer.close()
    job.stage = Stage.FAILED
    rig.store.save(job)
    rig.controller.start()
    rig.wait(lambda: rig.controller.state.failed == 1)
    rig.controller.retry_failed()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.DONE


def test_ignored_failures_stop_offering_a_retry(rig):
    _saved_job(rig, stage=Stage.FAILED)
    rig.controller.start()
    rig.wait(lambda: rig.controller.state.failed == 1)
    rig.controller.dismiss_failed()
    assert rig.controller.state.failed == 0
    rig.controller.retry_failed()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.FAILED
    assert rig.transcriber.models == []


def test_recording_with_no_sound_is_not_offered_for_retry(rig):
    job = Job.new_recording(rig.tmp_path / "out", NOW, "large-v3", "model-a")
    from coursescribe.wavfile import WavWriter

    WavWriter(job.audio_path, 48000, 2).close()  # stopped before any sound arrived
    rig.store.save(job)
    rig.controller.start()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.FAILED
    assert rig.controller.state.failed == 0


def _unusable_output(rig):
    blocker = rig.tmp_path / "unplugged"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    current = rig.controller.settings
    rig.controller.update_settings(Settings(blocker / "out", current.transcribe_model, current.proofread_model))


def test_record_with_an_unusable_save_folder_says_why(rig):
    _unusable_output(rig)
    rig.controller.start()
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is False
    assert "保存文件夹" in rig.controller.state.notice and "设置" in rig.controller.state.notice
    assert rig.store.load_all() == []


def test_adding_files_with_an_unusable_save_folder_says_why(rig):
    _unusable_output(rig)
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3"), rig.source("b.mp3"), rig.tmp_path / "x.docx"])
    notice = rig.controller.state.notice
    assert "保存文件夹" in notice and "a.mp3" in notice and "b.mp3" in notice
    assert str(rig.controller.settings.output_dir) in notice
    folder_line, refused_line = notice.split("\n")
    assert "x.docx" not in folder_line and "x.docx" in refused_line
    assert rig.store.load_all() == []


def _job_records_fail(rig, fails):
    real_save = rig.store.save

    def save(job):
        if fails(job):
            raise OSError(28, "No space left on device")
        real_save(job)

    rig.store.save = save


def _output_contents(rig):
    folder = rig.controller.settings.output_dir
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_adding_a_file_when_job_records_cannot_be_written_says_why(rig):
    _job_records_fail(rig, lambda job: job.audio_path.name == "b.mp3")
    rig.transcriber.gate = threading.Event()  # a.mp3 must not finish and replace the notice first
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3"), rig.source("b.mp3")])
    notice = rig.controller.state.notice
    rig.transcriber.gate.set()
    assert "b.mp3" in notice and "程序自己的数据文件夹" in notice and "a.mp3" not in notice
    assert "保存文件夹" not in notice  # the save folder is fine; blaming it would mislead
    rig.wait_idle()
    assert [job.stage for job in rig.store.load_all()] == [Stage.DONE]
    assert _output_contents(rig) == ["a"]  # no empty folder left for b


def test_record_when_job_records_cannot_be_written_says_why(rig):
    _job_records_fail(rig, lambda job: True)
    rig.controller.start()
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is False
    assert "没有开始录音" in rig.controller.state.notice
    assert "程序自己的数据文件夹" in rig.controller.state.notice
    assert _output_contents(rig) == []


def test_worker_keeps_going_when_a_job_record_cannot_be_saved(rig):
    """Without this the window says "working" forever and nothing else is processed."""
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    rig.wait_idle()
    _job_records_fail(rig, lambda job: job.attempts == 1 and job.audio_path.name == "b.mp3")
    rig.controller.add_files([rig.source("b.mp3"), rig.source("c.mp3")])
    rig.wait_idle()
    stages = {job.audio_path.name: job.stage for job in rig.store.load_all()}
    assert stages == {"a.mp3": Stage.DONE, "b.mp3": Stage.TRANSCRIBING, "c.mp3": Stage.DONE}
    assert any("b" in s.notice and "程序自己的数据文件夹" in s.notice for s in rig.states)


def test_job_waiting_for_a_model_keeps_the_data_folder_notice(rig):
    rig.transcriber.missing.add("large-v3")
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    rig.wait_idle()
    _job_records_fail(rig, lambda job: job.attempts == 1)
    rig.controller.download_needed_model()
    rig.wait(lambda: "程序自己的数据文件夹" in rig.controller.state.notice)
    rig.wait_idle()
    assert "程序自己的数据文件夹" in rig.controller.state.notice
    assert rig.store.load_all()[0].attempts == 0


@pytest.mark.parametrize("stop_by_itself", [False, True])
def test_recording_stop_with_unwritable_job_records_does_not_stick(rig, monkeypatch, stop_by_itself):
    """The disk can fill during a recording; the button must still come back."""
    if stop_by_itself:
        _disk_fails_after(monkeypatch, 4)
    _job_records_fail(rig, lambda job: job.stage is not Stage.RECORDING)  # before: the self-stop does not wait
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.05)
    if not stop_by_itself:
        rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.state.recording is False)
    assert "程序自己的数据文件夹" in rig.controller.state.notice
    assert rig.store.load_all()[0].stage is Stage.RECORDING  # repaired and transcribed at the next start
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is True  # and recording again works


def test_finished_job_that_cannot_be_saved_says_why(rig):
    _job_records_fail(rig, lambda job: job.stage in (Stage.DONE, Stage.FAILED))  # the disk filled up during the job
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    rig.wait(lambda: "程序自己的数据文件夹" in rig.controller.state.notice)
    rig.wait_idle()
    assert "程序自己的数据文件夹" in rig.controller.state.notice
    assert rig.controller.state.pending == 0


def test_pending_count_survives_a_file_added_while_another_job_finishes(rig):
    """The window must not say "ready" while work is queued, nor "working" forever."""
    entered, release = threading.Event(), threading.Event()
    real_unfinished = rig.store.unfinished

    def unfinished():
        if threading.current_thread().name == "jobs" and not release.is_set():
            entered.set()
            release.wait(5)
        return real_unfinished()

    rig.store.unfinished = unfinished
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    assert entered.wait(5)  # a.mp3 is finishing
    rig.controller.add_files([rig.source("c.mp3")])
    release.set()
    rig.wait(lambda: rig.controller.state.notice.startswith("完成：c"))
    assert rig.controller.state.pending == 0
    assert all(state.pending >= 0 for state in rig.states)


def test_quitting_is_not_counted_as_a_crash_even_if_shutdown_is_cut_short(rig):
    """Closing waits for the job thread; the user or Windows may end the program meanwhile."""
    rig.transcriber.gate = threading.Event()
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait(lambda: rig.store.load_all()[0].attempts == 1)  # the evidence a crash would leave
    closer = threading.Thread(target=rig.controller.shutdown)
    closer.start()
    try:
        rig.wait(lambda: (rig.store.folder / "closed-cleanly").exists(), seconds=1.0)
    finally:
        rig.transcriber.gate.set()
        closer.join(5)


def test_closing_still_saves_the_recording_when_the_clean_close_note_cannot_be_written(rig):
    def full_disk():
        raise OSError(28, "No space left on device")

    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.25)
    rig.store.mark_closed_cleanly = full_disk
    rig.controller.shutdown()
    assert rig.store.load_all()[0].stage is Stage.TRANSCRIBING
    assert rig.controller.state.recording is False


def test_retried_paused_job_starts_a_fresh_crash_count(rig):
    _saved_job(rig, attempts=2)
    rig.controller.start()
    rig.wait(lambda: rig.controller.state.failed == 1)
    rig.transcriber.gate = threading.Event()
    rig.controller.retry_failed()
    try:
        rig.wait(lambda: rig.store.load_all()[0].attempts == 1)  # not 3: one more crash must not pause it
    finally:
        rig.transcriber.gate.set()
    rig.wait_idle()


def test_retry_pressed_while_that_job_is_still_finishing_is_not_lost(rig):
    entered, release = threading.Event(), threading.Event()
    real_unfinished = rig.store.unfinished

    def unfinished():
        if threading.current_thread().name == "jobs" and not release.is_set():
            entered.set()
            release.wait(5)
        return real_unfinished()

    rig.store.unfinished = unfinished
    rig.proofreader.error = RuntimeError("boom")
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    assert entered.wait(5)  # saved as failed, not yet handed back
    rig.controller.retry_failed()
    release.set()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.FAILED  # still failed and offered, never stranded
    assert rig.controller.state.failed == 1


@pytest.mark.parametrize(
    ("action", "stage"),
    [("retry_failed", Stage.FAILED), ("dismiss_failed", Stage.FAILED), ("retry_waiting", Stage.WAITING_PROOFREAD)],
)
def test_retry_and_ignore_with_unwritable_job_records_say_why(rig, action, stage):
    rig.controller.start()
    _saved_job(rig, stage=stage)
    _job_records_fail(rig, lambda job: True)
    getattr(rig.controller, action)()
    assert "程序自己的数据文件夹" in rig.controller.state.notice
    assert rig.store.load_all()[0].stage is stage
    assert rig.controller.state.failed == (1 if stage is Stage.FAILED else 0)  # still offered
    assert rig.controller.state.pending == 0


def test_unexpected_error_is_reported_in_plain_words(rig):
    rig.controller.report_unexpected_error()
    assert "日志" in rig.controller.state.notice
