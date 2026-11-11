"""Mac implementations. System audio capture arrives in the Mac phase of the plan."""

from __future__ import annotations

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


class _NoKeepAwake:
    """Keeping the Mac awake arrives in the Mac phase of the plan."""

    def hold(self) -> None:
        pass

    def release(self) -> None:
        pass


def create_keep_awake() -> _NoKeepAwake:
    return _NoKeepAwake()


def cloud_sync_roots() -> list[str]:
    """iCloud Drive. Whether Documents itself lives there is confirmed in the Mac phase."""
    return [str(Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs")]
