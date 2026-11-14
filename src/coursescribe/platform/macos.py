"""Mac implementations."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from coursescribe.platform.file_identity import mac_file_identity_key as file_identity_key
from coursescribe.platform.limits import MAC_LIMITS
from coursescribe.platform.mac_hotkey import HOTKEY_LABEL, create_global_hotkey  # noqa: F401

HOTKEY_PLATFORM = "mac"  # the name the shortcut rules use for this system
DEFAULT_HOTKEY = "ctrl+alt+r"  # canonical shortcut text


def open_system_audio():
    from coursescribe.platform.mac_audio import MacSystemAudio

    return MacSystemAudio()


def prepare_gpu_libraries() -> None:
    """Nothing to prepare: the Mac build runs the engine on the CPU."""

FILES_CAN_BE_HELD = False  # an open file never shuts others out; a refusal is real
LIMITS = MAC_LIMITS


def caffeinate_command(pid: int) -> list[str]:
    """``-i``: no sleep while idle (the screen may turn off, closing the lid still sleeps); ``-w``: ends with ``pid``."""
    return ["caffeinate", "-i", "-w", str(pid)]


class MacKeepAwake:
    """A ``caffeinate`` process that lives exactly as long as the request.

    The Mac itself ends it when this program ends, so a crash cannot leave the computer awake.
    """

    def __init__(self, spawn=subprocess.Popen) -> None:
        self._spawn = spawn
        self._process = None
        self._lock = threading.Lock()

    def hold(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self._process = self._spawn(
                caffeinate_command(os.getpid()),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def release(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def create_keep_awake() -> MacKeepAwake:
    return MacKeepAwake()


def cloud_sync_roots() -> list[str]:
    """iCloud Drive. Whether Documents itself lives there is confirmed in the Mac phase."""
    return [str(Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs")]
