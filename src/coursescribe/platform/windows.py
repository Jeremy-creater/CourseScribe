"""Windows: system sound through WASAPI loopback, no settings or virtual devices."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

import numpy as np

# A program holding a file open can shut everyone else out for a moment (antivirus, indexer).
FILES_CAN_BE_HELD = True

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


HOTKEY_LABEL = "Ctrl+Alt+R"
_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_NOREPEAT = 0x4000
_VK_R = 0x52
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012
_HOTKEY_ID = 1


class GlobalHotkey:
    """Calls back when the shortcut is pressed, whichever program has the focus."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._thread = threading.Thread(target=self._listen, name="hotkey", daemon=True)
        self._ready = threading.Event()
        self._registered = False
        self._thread_id = 0

    def start(self) -> bool:
        """Returns False when another program already owns this shortcut."""
        self._thread.start()
        self._ready.wait(5)
        return self._registered

    def _listen(self) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        modifiers = _MOD_CONTROL | _MOD_ALT | _MOD_NOREPEAT
        self._registered = bool(user32.RegisterHotKey(None, _HOTKEY_ID, modifiers, _VK_R))
        self._ready.set()
        if not self._registered:
            return
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message == _WM_HOTKEY:
                self._callback()
        user32.UnregisterHotKey(None, _HOTKEY_ID)

    def stop(self) -> None:
        if self._registered and self._thread.is_alive():
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
            self._thread.join(2.0)


def create_global_hotkey(callback: Callable[[], None]) -> GlobalHotkey:
    return GlobalHotkey(callback)
