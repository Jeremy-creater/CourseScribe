"""The whole application behind the window, with stand-ins for models and devices."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from coursescribe.controller import SLEEP_WARNING, AppController
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
        self.gate = None
        self.models = []

    def proofread(self, text, model, checkpoint, on_progress):
        self.models.append(model)
        if self.gate:
            self.gate.wait(5)
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


class FakeKeepAwake:
    """The system's power state as far as we are concerned: records results, not calls."""

    def __init__(self):
        self.held = False
        self.history: list[bool] = []
        self.violations: list[str] = []
        self.fail_hold: Exception | None = None
        self.fail_release: Exception | None = None
        self.hold_attempts = 0
        self.last_hold_failed = False
        self.on_hold = None  # called first, to stall a hold in tests
        self.on_release = None

    def hold(self):
        self.hold_attempts += 1
        if self.on_hold:
            self.on_hold()
        if self.fail_hold:
            self.last_hold_failed = True
            raise self.fail_hold
        self.last_hold_failed = False
        if self.held:
            self.violations.append("hold while held")
        self.held = True
        self.history.append(True)

    def release(self):
        if self.on_release:
            self.on_release()
        if not self.held:
            self.violations.append("release while released")
        self.held = False
        self.history.append(False)
        if self.fail_release:
            raise self.fail_release


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
        self.awake = FakeKeepAwake()
        self.controller = AppController(
            store=self.store,
            settings=self.settings,
            transcriber=self.transcriber,
            proofreader=self.proofreader,
            open_capture=lambda: FakeCapture(capture_fails),
            download_model=self._download,
            stopping=threading.Event(),
            keep_awake=self.awake,
            now=lambda: NOW,
        )
        self.controller.add_listener(self.states.append)
        self.closed = False
        close = self.controller.shutdown

        def shutdown():
            self.closed = True
            close()

        self.controller.shutdown = shutdown

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

    def check_keep_awake_follows_busy(self):
        """The law every scenario must keep: awake exactly while recording or working."""

        def follows():
            state = self.controller.state
            busy = (state.recording or state.pending > 0) and not self.closed
            return self.awake.held == busy or (busy and self.awake.last_hold_failed)

        self.wait(follows, seconds=2.0)
        assert self.awake.violations == []
        self.controller.shutdown()
        assert self.awake.held is False


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path)
    yield made
    made.check_keep_awake_follows_busy()


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
    from platformdirs import user_documents_dir

    from coursescribe.settings import app_home, default_settings

    monkeypatch.delenv("COURSESCRIBE_HOME", raising=False)
    home = app_home()
    assert home.name == "CourseScribe"
    settings = default_settings()
    assert settings.output_dir.name == "CourseScribe"
    # Documents, or the home folder when Documents is synced by OneDrive (test_sync_folder_desktop checks which).
    assert settings.output_dir.parent in (Path(user_documents_dir()), Path.home())
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


# --- keeping the computer awake (docs/plans/M2-keep-awake-tests.md) ---


def _gate(rig):
    rig.transcriber.gate = threading.Event()
    rig.proofreader.gate = rig.transcriber.gate
    return rig.transcriber.gate


def _stalled(when):
    """A hook that stalls the first call made on a matching thread until released."""
    entered, release = threading.Event(), threading.Event()

    def hook():
        if when() and not entered.is_set():
            entered.set()
            release.wait(5)

    return hook, entered, release


def test_awake_while_recording_and_until_its_transcript_is_done(rig):
    rig.controller.start()
    rig.controller.toggle_recording()
    assert rig.awake.held is True
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.05)
    gate = _gate(rig)
    rig.controller.toggle_recording()
    assert rig.awake.history == [True]  # no gap while the recording is handed to transcription
    gate.set()
    rig.wait_idle()
    assert rig.awake.history == [True, False]


def test_awake_until_every_queued_file_is_done(rig):
    gate = _gate(rig)
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3"), rig.source("b.mp3")])
    assert rig.awake.held is True
    gate.set()
    rig.wait_idle()
    assert rig.awake.history == [True, False]


def test_recording_during_a_job_keeps_one_hold(rig):
    gate = _gate(rig)
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.05)
    rig.controller.toggle_recording()
    assert rig.awake.history == [True]
    gate.set()
    rig.wait_idle()
    assert rig.awake.history == [True, False]


def test_recording_that_never_starts_never_holds(tmp_path):
    rig = Rig(tmp_path, capture_fails=True)
    rig.controller.start()
    rig.controller.toggle_recording()
    assert rig.awake.history == []
    rig.check_keep_awake_follows_busy()


def test_recording_stopped_with_unwritable_job_records_lets_the_computer_sleep(rig):
    rig.controller.start()
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.05)
    _job_records_fail(rig, lambda job: job.stage is not Stage.RECORDING)
    rig.controller.toggle_recording()
    assert rig.awake.history == [True, False]


@pytest.mark.parametrize("waiting_for", ["proofreading", "model"])
def test_jobs_left_waiting_let_the_computer_sleep(rig, waiting_for):
    if waiting_for == "proofreading":
        rig.proofreader.blocked = "Ollama 没有启动"
    else:
        rig.transcriber.missing.add("large-v3")
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    assert rig.awake.history == [True, False]


def test_paused_job_does_not_keep_the_computer_awake(rig):
    _saved_job(rig, attempts=2)
    rig.controller.start()
    rig.wait(lambda: rig.controller.state.failed == 1)
    assert rig.awake.history == []


def _waiting_for_proofreading(rig):
    rig.proofreader.blocked = "Ollama 没有启动"
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    rig.proofreader.blocked = None


def _failed_at_proofreading(rig):
    rig.proofreader.error = RuntimeError("boom")
    rig.controller.add_files([rig.source()])
    rig.wait_idle()
    rig.proofreader.error = None


def _waiting_for_model(rig):
    rig.transcriber.missing.add("large-v3")
    rig.controller.add_files([rig.source()])
    rig.wait_idle()


_RESUMES = {
    "retry_waiting": _waiting_for_proofreading,
    "retry_failed": _failed_at_proofreading,
    "download_needed_model": _waiting_for_model,
}


@pytest.mark.parametrize("action", sorted(_RESUMES))
def test_awake_while_a_retried_or_resumed_job_runs(rig, action):
    rig.controller.start()
    _RESUMES[action](rig)
    gate = _gate(rig)
    getattr(rig.controller, action)()
    rig.wait(lambda: rig.controller.state.pending == 1)
    assert rig.awake.held is True
    gate.set()
    rig.wait_idle()
    assert rig.awake.history[-2:] == [True, False]


def test_quick_record_presses_never_leave_the_computer_held(rig):
    rig.controller.start()
    for _ in range(20):
        rig.controller.toggle_recording()
    rig.wait(lambda: not rig.controller.state.recording)
    rig.wait_idle()
    assert rig.awake.held is False


def test_file_added_while_the_last_job_lets_go_keeps_the_computer_awake(rig):
    """Forces the worker to be inside "let the computer sleep" while the window adds a file."""
    hook, entered, release = _stalled(lambda: threading.current_thread().name == "jobs")
    rig.awake.on_release = hook
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    assert entered.wait(5)
    adder = threading.Thread(target=rig.controller.add_files, args=([rig.source("b.mp3")],))
    adder.start()
    time.sleep(0.1)
    release.set()
    adder.join(5)
    rig.wait_idle()
    assert rig.awake.history == [True, False, True, False]


def test_failed_hold_keeps_working_logs_and_says_so_once(rig, caplog):
    import logging

    rig.awake.fail_hold = OSError("power request refused")
    rig.controller.start()
    with caplog.at_level(logging.WARNING):
        rig.controller.toggle_recording()
        rig.wait(lambda: rig.controller.recording_seconds() >= 0.05)
        rig.controller.toggle_recording()
        rig.wait_idle()
        rig.controller.add_files([rig.source()])
        rig.wait_idle()
    assert [job.stage for job in rig.store.load_all()] == [Stage.DONE, Stage.DONE]
    first_recording = next(state for state in rig.states if state.recording)
    assert first_recording.notice == SLEEP_WARNING
    shown = [SLEEP_WARNING in state.notice for state in rig.states]
    assert sum(1 for before, now in zip([False] + shown, shown) if now and not before) == 1
    assert caplog.text.count("power request refused") == 2  # one line per attempt, one attempt per busy period
    assert rig.awake.hold_attempts == 2
    assert rig.awake.violations == []


def _record_briefly(rig):
    rig.controller.toggle_recording()
    rig.wait(lambda: rig.controller.recording_seconds() >= 0.05)
    rig.controller.toggle_recording()


def _add_a_file(rig):
    rig.controller.add_files([rig.source("b.mp3")])


def _retry_a_failed_file(rig):
    _saved_job(rig, stage=Stage.FAILED)
    rig.controller.retry_failed()


@pytest.mark.parametrize("second_busy", [_record_briefly, _add_a_file, _retry_a_failed_file])
def test_hold_is_tried_again_the_next_time_there_is_work(rig, second_busy):
    rig.awake.fail_hold = OSError("power request refused")
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    rig.wait_idle()
    rig.awake.fail_hold = None
    gate = _gate(rig)
    since = len(rig.states)
    second_busy(rig)
    rig.wait(lambda: rig.controller.state.pending >= 1)
    assert rig.awake.held is True
    assert rig.awake.hold_attempts == 2
    assert not any(SLEEP_WARNING in state.notice for state in rig.states[since:])
    gate.set()
    rig.wait_idle()


def test_hold_failing_in_an_unexpected_way_still_lets_the_work_finish(rig):
    rig.awake.fail_hold = AttributeError("function 'PowerCreateRequest' not found")
    gate = _gate(rig)
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    assert SLEEP_WARNING in rig.controller.state.notice
    gate.set()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.DONE
    assert rig.awake.violations == []


def test_failed_release_is_logged_not_shown_and_the_next_hold_works(rig, caplog):
    import logging

    rig.awake.fail_release = OSError("clear failed")
    rig.controller.start()
    with caplog.at_level(logging.WARNING):
        rig.controller.add_files([rig.source()])
        rig.wait_idle()
    assert SLEEP_WARNING not in rig.controller.state.notice
    assert "clear failed" in caplog.text
    rig.awake.fail_release = None
    rig.controller.toggle_recording()
    assert rig.awake.held is True
    rig.controller.toggle_recording()


def _add_good_and_bad(rig):
    rig.controller.add_files([rig.source("a.mp3"), rig.tmp_path / "x.docx"])


def _retry_waiting_first(rig):
    _saved_job(rig, stage=Stage.WAITING_PROOFREAD)
    rig.controller.retry_waiting()


@pytest.mark.parametrize("first_busy", [_add_good_and_bad, _retry_a_failed_file, _retry_waiting_first])
def test_sleep_warning_is_not_wiped_by_the_notice_of_the_same_action(rig, first_busy):
    rig.awake.fail_hold = OSError("power request refused")
    gate = _gate(rig)
    rig.controller.start()
    first_busy(rig)
    assert SLEEP_WARNING in rig.controller.state.notice
    gate.set()
    rig.wait_idle()


@pytest.mark.parametrize("situation", ["jobs queued", "recording", "idle"])
def test_closing_lets_the_computer_sleep(rig, situation):
    rig.controller.start()
    if situation == "jobs queued":
        gate = _gate(rig)
        rig.controller.add_files([rig.source("a.mp3"), rig.source("b.mp3")])
    elif situation == "recording":
        rig.controller.toggle_recording()
    rig.controller.shutdown()
    assert rig.awake.held is False
    assert rig.awake.history == ([] if situation == "idle" else [True, False])
    if situation == "jobs queued":
        gate.set()


def test_nothing_holds_after_closing_and_closing_twice_is_harmless(rig):
    rig.controller.start()
    rig.controller.shutdown()
    rig.controller.add_files([rig.source()])
    rig.controller.shutdown()
    assert rig.awake.history == []


def test_closing_while_a_hold_is_in_progress_ends_released(rig):
    hook, entered, release = _stalled(lambda: threading.current_thread().name == "adder")
    rig.awake.on_hold = hook
    rig.controller.start()
    adder = threading.Thread(target=rig.controller.add_files, args=([rig.source()],), name="adder")
    adder.start()
    assert entered.wait(5)
    closer = threading.Thread(target=rig.controller.shutdown)
    closer.start()
    time.sleep(0.1)
    release.set()
    adder.join(5)
    closer.join(5)
    assert rig.awake.held is False
    assert rig.awake.history[-1] is False


def test_awake_while_jobs_left_by_the_last_run_continue(rig):
    _saved_job(rig, attempts=1)
    gate = _gate(rig)
    rig.controller.start()
    assert rig.awake.held is True
    gate.set()
    rig.wait_idle()
    assert rig.awake.history == [True, False]


def test_nothing_to_continue_means_no_hold_at_start(rig):
    rig.controller.start()
    assert rig.awake.history == []


def test_sleep_warning_goes_away_when_the_work_is_over(rig):
    """A notice carried forward, like a failed download, must not keep saying the computer may sleep."""
    rig.awake.fail_hold = OSError("power request refused")
    rig.transcriber.missing.add("large-v3")
    rig.download_error = OSError("no internet")
    rig.controller.start()
    _saved_job(rig, stage=Stage.WAITING_MODEL)  # the download is the first work of this run
    rig.controller.download_needed_model()
    rig.wait(lambda: "下载失败" in rig.controller.state.notice)
    rig.wait_idle()
    assert "下载失败" in rig.controller.state.notice
    assert SLEEP_WARNING not in rig.controller.state.notice


def test_mac_keep_awake_can_be_used_and_never_fails():
    from coursescribe.platform import macos

    keep_awake = macos.create_keep_awake()
    assert [keep_awake.hold(), keep_awake.hold(), keep_awake.release(), keep_awake.release()] == [None] * 4


def test_unexpected_error_is_reported_in_plain_words(rig):
    rig.controller.report_unexpected_error()
    assert "日志" in rig.controller.state.notice


# --- long names (docs/plans/M2-long-names-tests.md) ---


def _u16(text):
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _fits(root):
    """How many characters a long name keeps in ``root``: 80, less when the save folder path is long (N7)."""
    return min(80, (243 - _u16(str(root))) // 2)


def test_long_file_name_goes_all_the_way_under_a_recognisable_name(rig):
    name = "讲" * _fits(rig.tmp_path / "out")
    source = rig.source("讲" * 100 + ".mp3")
    rig.controller.start()
    rig.controller.add_files([source])
    rig.wait_idle()
    folder = rig.tmp_path / "out" / name
    assert rig.controller.state.notice == "完成：" + name
    assert rig.controller.state.last_folder.name == name
    assert "讲" * 100 + "的内容" in (folder / f"{name}.txt").read_text(encoding="utf-8")  # read from the full path
    assert (folder / f"{name}.md").read_text(encoding="utf-8").splitlines()[0] == "# " + name
    assert sorted(p.name for p in folder.iterdir()) == sorted([f"{name}.txt", f"{name}.md"])
    assert source.read_bytes() == b"audio"


def _too_long_output(rig):
    root = rig.tmp_path / ("r" * (204 - _u16(str(rig.tmp_path / "r")) + 1))
    assert _u16(str(root)) == 204
    current = rig.controller.settings
    rig.controller.update_settings(Settings(root, current.transcribe_model, current.proofread_model))
    return root


@pytest.mark.parametrize("action", ["add_files", "record"])
def test_save_folder_path_too_long_says_why_and_leaves_nothing(rig, action):
    root = _too_long_output(rig)
    rig.controller.start()
    if action == "add_files":
        rig.controller.add_files([rig.source("a.mp3"), rig.source("b.mp3")])
    else:
        rig.controller.toggle_recording()
    notice = rig.controller.state.notice
    reason = f"保存文件夹的路径太长：{root}。请在设置里换一个路径短一些的文件夹。"
    assert "打不开" not in notice  # not blamed on an unplugged drive
    if action == "add_files":
        assert notice == "没有加入：a.mp3、b.mp3。" + reason
    else:
        assert notice == "没有开始录音。" + reason
    if action == "record":
        assert rig.controller.state.recording is False
    assert rig.controller.state.pending == 0
    assert rig.store.load_all() == []
    assert not root.exists()
    assert {p.name for p in rig.tmp_path.iterdir()} <= {"input", "settings.json", "state"}
    assert rig.awake.history == []


def test_cut_name_is_kept_when_the_job_resumes_after_a_restart(tmp_path, monkeypatch):
    from coursescribe import jobs

    name = "讲" * _fits(tmp_path / "out")
    first = Rig(tmp_path)  # never started and never shut down: closed before transcribing, like a crash
    first.controller.add_files([first.source("讲" * 100 + ".mp3")])
    saved = first.store.load_all()[0]
    assert saved.stage is Stage.TRANSCRIBING
    monkeypatch.setattr(jobs, "NAME_LIMIT", 40)
    assert len(jobs.job_name("讲" * 100, tmp_path / "out")) == 40  # a fresh name would now differ
    second = Rig(tmp_path)
    second.controller.start()
    second.wait_idle()
    out = tmp_path / "out"
    assert second.controller.state.notice == "完成：" + name
    assert sorted(p.name for p in out.iterdir()) == [name]
    assert sorted(p.name for p in (out / name).iterdir()) == sorted([f"{name}.txt", f"{name}.md"])
    assert second.store.load_all()[0].folder == saved.folder
    second.check_keep_awake_follows_busy()
