"""Mac implementations. System audio capture arrives in the Mac phase of the plan."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from coursescribe.platform.limits import PathLimits


def open_system_audio():
    raise NotImplementedError("System audio capture on the Mac is not built yet")


def prepare_gpu_libraries() -> None:
    """Nothing to prepare: the Mac build runs the engine on the CPU."""


HOTKEY_LABEL = ""
FILES_CAN_BE_HELD = False  # an open file never shuts others out; a refusal is real
# The Mac counts each name (255 bytes) and the whole path (PATH_MAX 1024 less the terminating null) in UTF-8 bytes.
LIMITS = PathLimits(path_units=None, component_bytes=255, path_bytes=1023)


class _NoHotkey:
    def start(self) -> bool:
        return False

    def stop(self) -> None:
        pass


def create_global_hotkey(callback) -> _NoHotkey:
    return _NoHotkey()


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
