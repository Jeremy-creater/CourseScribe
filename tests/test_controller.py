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
        self.models = []

    def proofread(self, text, model, checkpoint, on_progress):
        self.models.append(model)
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
