"""Windows: system sound through WASAPI loopback, no settings or virtual devices."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

import numpy as np

_SAMPLE_RATE = 48000
_CHANNELS = 2
_COINIT_MULTITHREADED = 0
_KEEPER_BLOCK_SECONDS = 0.1


def _start_com_for_this_thread() -> None:
    ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)


class WindowsSystemAudio:
    """Records what the default speaker plays.

    Windows may deliver nothing while no program is playing, which would freeze the
    recording during a pause. A silent stream keeps sound flowing, so pauses stay
    in the recording and stopping never waits on a device with nothing to say.
    """

    sample_rate = _SAMPLE_RATE
    channels = _CHANNELS

    def __init__(self, speaker_name: str | None = None, keep_alive: bool = True) -> None:
        self._speaker_name = speaker_name
        self._keep_alive = keep_alive
        self._recorder = None
        self._keeper: threading.Thread | None = None
        self._stop_keeper = threading.Event()

    def __enter__(self) -> WindowsSystemAudio:
        import soundcard  # must come first: its import fails if this thread already started COM

        _start_com_for_this_thread()
        speaker = soundcard.get_speaker(self._speaker_name) if self._speaker_name else soundcard.default_speaker()
        microphone = soundcard.get_microphone(speaker.id, include_loopback=True)
        self._recorder = microphone.recorder(samplerate=_SAMPLE_RATE, channels=_CHANNELS)
        self._recorder.__enter__()
        if self._keep_alive:
            self._keeper = threading.Thread(target=self._play_silence, args=(speaker,), name="silence", daemon=True)
            self._keeper.start()
        return self

    def _play_silence(self, speaker) -> None:
        _start_com_for_this_thread()
        count = int(_SAMPLE_RATE * _KEEPER_BLOCK_SECONDS)
        block = np.zeros((count, _CHANNELS), dtype=np.float32)
        with speaker.player(samplerate=_SAMPLE_RATE, channels=_CHANNELS) as player:
            while not self._stop_keeper.is_set():
                player.play(block)

    def __exit__(self, *args) -> None:
        self._stop_keeper.set()
        if self._keeper is not None:
            self._keeper.join(1.0)
        if self._recorder is not None:
            self._recorder.__exit__(None, None, None)

    def read(self, frames: int) -> np.ndarray:
        block = self._recorder.record(numframes=frames)
        return (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)


def open_system_audio() -> WindowsSystemAudio:
    return WindowsSystemAudio()


def prepare_gpu_libraries() -> None:
    """Let the engine find the GPU libraries installed with, or packaged in, the program."""
    roots = [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        roots.append(Path(bundled) / "nvidia")
    folders = [str(folder) for root in roots if root.is_dir() for folder in root.glob("*/bin")]
    if folders:
        os.environ["PATH"] = os.pathsep.join([*folders, os.environ.get("PATH", "")])
