"""The packaged program on the real desktop, with no help from the development setup.

Build it first: python -m PyInstaller packaging/coursescribe.spec --noconfirm
These open the real window, press the real shortcut and play speech out loud.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from coursescribe.jobs import Job, JobStore, Stage
from speech import speak_to_wav

pytestmark = pytest.mark.gui

EXE = Path(__file__).resolve().parents[1] / "dist" / "CourseScribe" / "CourseScribe.exe"
LESSON = "今天我们来学习数据库索引。索引可以让查询变快，但是写入数据的时候会慢一点。所以索引不是越多越好。"


class Packaged:
    def __init__(self, home: Path) -> None:
        if not EXE.exists():
            pytest.fail(f"Packaged program not found at {EXE}; build it first")
        self.home = home
        self.out = home / "out"
        (home / "settings.json").write_text(
            json.dumps({"output_dir": str(self.out), "transcribe_model": "large-v3", "proofread_model": "gemma4:12b"}),
            encoding="utf-8",
        )
        self.store = JobStore(home / "jobs")
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        system_root = os.environ["SystemRoot"]
        env = os.environ.copy()
        # Only Windows itself on the path: any developer CUDA or ffmpeg must not help.
        env["PATH"] = f"{system_root}\\System32;{system_root}"
        env["COURSESCRIBE_HOME"] = str(self.home)
        self.process = subprocess.Popen([str(EXE)], env=env)
        self.wait_for(lambda: (self.home / "logs" / "coursescribe.log").exists(), 60)
        time.sleep(3)  # the window and the shortcut are ready

    def kill(self) -> None:
        self.process.kill()
        self.process.wait(10)

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.process.pid)], capture_output=True)
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.kill()

    def wait_for(self, condition, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if condition():
                return
            if self.process.poll() is not None:
                pytest.fail(f"The program exited with code {self.process.returncode}\n{self.log()}")
            time.sleep(0.5)
        pytest.fail(f"Timed out. Jobs: {self.store.load_all()}\n{self.log()}")

    def wait_until_finished(self, seconds: float = 300) -> Job:
        self.wait_for(
            lambda: self.store.load_all() and all(j.stage in (Stage.DONE, Stage.FAILED) for j in self.store.load_all()),
            seconds,
        )
        return self.store.load_all()[-1]

    def log(self) -> str:
        path = self.home / "logs" / "coursescribe.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""


@pytest.fixture
def packaged(tmp_path):
    made = Packaged(tmp_path)
    yield made
    made.close()


@pytest.fixture(scope="module")
def spoken(tmp_path_factory):
    return speak_to_wav(LESSON, tmp_path_factory.mktemp("speech") / "lesson.wav")


def _press_shortcut() -> None:
    user32 = ctypes.windll.user32
    ctrl, alt, r, key_up = 0x11, 0x12, 0x52, 0x0002
    for key in (ctrl, alt, r):
        user32.keybd_event(key, 0, 0, 0)
    for key in (r, alt, ctrl):
        user32.keybd_event(key, 0, key_up, 0)


def _play(path: Path) -> threading.Thread:
    import soundcard

    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)

    def play() -> None:
        soundcard.default_speaker().play(samples.astype(np.float32) / 32768 * 0.5, samplerate=rate)

    thread = threading.Thread(target=play)
    thread.start()
    return thread


def _watch_connections(pid: int) -> set[str]:
    """Collect, in the background, every remote address the program connects to."""
    import psutil

    hosts: set[str] = set()

    def watch() -> None:
        try:
            process = psutil.Process(pid)
            while process.is_running():
                hosts.update(c.raddr.ip for c in process.net_connections() if c.raddr)
                time.sleep(0.05)
        except psutil.Error:
            pass  # the program has exited

    threading.Thread(target=watch, daemon=True).start()
    return hosts


def test_file_job_finishes_on_the_gpu_with_only_bundled_libraries(packaged, spoken):
    job = Job.new_file(spoken, packaged.out, datetime.now(), "large-v3", "gemma4:12b")
    packaged.store.save(job)
    packaged.start()
    remote_hosts = _watch_connections(packaged.process.pid)
    done = packaged.wait_until_finished()
    assert done.stage is Stage.DONE, f"{done.message}\n{packaged.log()}"
    assert remote_hosts <= {"127.0.0.1", "::1"}, "audio and text must never leave this computer"
    assert "127.0.0.1" in remote_hosts, "the watcher should at least have seen Ollama"
    assert "on cuda" in packaged.log()
    assert "on the CPU" not in packaged.log() and "using the CPU" not in packaged.log()
    assert "索引" in done.txt_path.read_text(encoding="utf-8")
    assert "。" in done.md_path.read_text(encoding="utf-8")


def test_shortcut_records_and_produces_all_three_files(packaged, spoken):
    packaged.start()
    _press_shortcut()
    packaged.wait_for(lambda: packaged.store.load_all(), 15)
    _play(spoken).join()
    time.sleep(1)
    _press_shortcut()
    done = packaged.wait_until_finished()
    assert done.stage is Stage.DONE, f"{done.message}\n{packaged.log()}"
    assert done.audio_path.stat().st_size > 100_000
    assert done.txt_path.exists() and done.md_path.exists()
    assert sorted(p.suffix for p in done.folder.iterdir()) == [".md", ".txt", ".wav"]


def test_force_closed_during_recording_continues_at_the_next_start(packaged, spoken):
    packaged.start()
    _press_shortcut()
    packaged.wait_for(lambda: packaged.store.load_all(), 15)
    _play(spoken).join()
    packaged.kill()
    job = packaged.store.load_all()[0]
    assert job.stage is Stage.RECORDING
    assert job.audio_path.stat().st_size > 100_000

    packaged.start()
    done = packaged.wait_until_finished()
    assert done.stage is Stage.DONE, f"{done.message}\n{packaged.log()}"
    assert done.txt_path.exists() and done.md_path.exists()
