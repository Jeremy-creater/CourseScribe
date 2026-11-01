"""Mac implementations. System audio capture arrives in the Mac phase of the plan."""

from __future__ import annotations


def open_system_audio():
    raise NotImplementedError("System audio capture on the Mac is not built yet")


def prepare_gpu_libraries() -> None:
    """Nothing to prepare: the Mac build runs the engine on the CPU."""


HOTKEY_LABEL = ""


class _NoHotkey:
    def start(self) -> bool:
        return False

    def stop(self) -> None:
        pass


def create_global_hotkey(callback) -> _NoHotkey:
    return _NoHotkey()
