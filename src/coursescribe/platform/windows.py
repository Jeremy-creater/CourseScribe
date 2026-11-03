"""Windows: system sound through WASAPI loopback, no settings or virtual devices."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import winreg
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

import numpy as np

# A program holding a file open can shut everyone else out for a moment (antivirus, indexer).
FILES_CAN_BE_HELD = True

logger = logging.getLogger(__name__)

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

    Windows converts whatever the device's own format is to the 48 kHz stereo asked for,
    so a recording that moves to another device keeps the same format.
    """

    sample_rate = _SAMPLE_RATE
    channels = _CHANNELS

    def __init__(self, speaker_name: str | None = None, keep_alive: bool = True) -> None:
        self._speaker_name = speaker_name
        self._keep_alive = keep_alive
        self._speaker = None
        self._recorder = None
        self._keeper: threading.Thread | None = None
        self._stop_keeper = threading.Event()
        self.device_name = ""
        self.device_id = ""

    def __enter__(self) -> WindowsSystemAudio:
        import soundcard  # must come first: its import fails if this thread already started COM

        _start_com_for_this_thread()
        speaker = soundcard.get_speaker(self._speaker_name) if self._speaker_name else soundcard.default_speaker()
        self._speaker = speaker
        self.device_id = speaker.id
        try:
            self.device_name = speaker.name or ""
        except Exception:  # only used to tell the user where the sound went
            logger.warning("Could not read the output device's name", exc_info=True)
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
        try:
            with speaker.player(samplerate=_SAMPLE_RATE, channels=_CHANNELS) as player:
                while not self._stop_keeper.is_set():
                    player.play(block)
        except Exception:  # the device went away; the recording side notices and reconnects
            logger.warning("The silent stream stopped", exc_info=True)

    def __exit__(self, *args) -> None:
        self._stop_keeper.set()
        if self._keeper is not None:
            self._keeper.join(1.0)
        if self._recorder is not None:
            self._recorder.__exit__(None, None, None)

    def read(self, frames: int) -> np.ndarray:
        block = self._recorder.record(numframes=frames)
        return (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)

    def is_default(self) -> bool:
        """Whether this device is still the default output. A device chosen by name is never left."""
        if self._speaker_name:
            return True
        return self.default_id() == self.device_id

    def default_id(self) -> str:
        """The default output device's ID now. Raises when there is none (Windows: 0x80070490).

        Call on a thread that has started COM, such as the one that opened this capture.
        """
        import soundcard

        return soundcard.default_speaker().id


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


_POWER_REQUEST_SYSTEM_REQUIRED = 1
_POWER_REQUEST_EXECUTION_REQUIRED = 3  # keeps the program running on laptops with modern standby
_KEEP_AWAKE_KINDS = (_POWER_REQUEST_SYSTEM_REQUIRED, _POWER_REQUEST_EXECUTION_REQUIRED)
_KEEP_AWAKE_REASON = "CourseScribe 正在录音或处理任务"


class _ReasonContext(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("Flags", wintypes.DWORD), ("SimpleReasonString", wintypes.LPWSTR)]


class WindowsKeepAwake:
    """A power request: the system stays on, the screen may still turn off, and closing the lid still sleeps.

    Unlike SetThreadExecutionState it is not tied to a thread, so it can be released on
    another thread than the one that asked.
    """

    def __init__(self) -> None:
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.PowerCreateRequest.restype = wintypes.HANDLE
        self._kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(_ReasonContext)]
        self._kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
        self._kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._reason = _ReasonContext(0, 1, _KEEP_AWAKE_REASON)  # version 0, simple string
        self._handle = None
        self._lock = threading.Lock()

    def hold(self) -> None:
        with self._lock:
            if self._handle is not None:
                return
            handle = self._kernel32.PowerCreateRequest(ctypes.byref(self._reason))
            if not handle or handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            for kind in _KEEP_AWAKE_KINDS:
                if not self._kernel32.PowerSetRequest(handle, kind):
                    error = ctypes.WinError(ctypes.get_last_error())
                    self._kernel32.CloseHandle(handle)  # closing also clears what was set
                    raise error
            self._handle = handle

    def release(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is None:
                return
            failed = [kind for kind in _KEEP_AWAKE_KINDS if not self._kernel32.PowerClearRequest(handle, kind)]
            error = ctypes.WinError(ctypes.get_last_error()) if failed else None
            self._kernel32.CloseHandle(handle)  # closing ends the request even if clearing failed
            if error is not None:
                raise error


def create_keep_awake() -> WindowsKeepAwake:
    return WindowsKeepAwake()


_ONEDRIVE_VARIABLES = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
_ONEDRIVE_ACCOUNTS = r"Software\Microsoft\OneDrive\Accounts"


def cloud_sync_roots() -> list[str]:
    """OneDrive folders, as given: the variables are set at sign-in, the registry also after."""
    roots = [os.environ.get(name, "") for name in _ONEDRIVE_VARIABLES]
    try:
        accounts = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _ONEDRIVE_ACCOUNTS)
    except FileNotFoundError:
        return roots
    with accounts:
        for index in range(winreg.QueryInfoKey(accounts)[0]):
            try:
                with winreg.OpenKey(accounts, winreg.EnumKey(accounts, index)) as account:
                    folder, kind = winreg.QueryValueEx(account, "UserFolder")
            except FileNotFoundError:
                continue  # an account that is not set up yet has no folder
            if kind == winreg.REG_EXPAND_SZ:
                folder = winreg.ExpandEnvironmentStrings(folder)
            if isinstance(folder, str):
                roots.append(folder)
    return roots
