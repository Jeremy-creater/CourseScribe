"""The real request that keeps the computer awake. Runs on the real desktop; no admin rights needed."""

from __future__ import annotations

import sys
import threading

import pytest

from coursescribe.platform import create_keep_awake

pytestmark = pytest.mark.gui

_SYSTEM_REQUIRED = 0x1
_DISPLAY_REQUIRED = 0x2


def _execution_state() -> int:
    """What the whole system is currently asked to stay on for (any program, not only us)."""
    if sys.platform != "win32":
        pytest.fail("Keeping the Mac awake is not built yet (see the M2 keep-awake test plan).")
    import ctypes
    from ctypes import wintypes

    state = wintypes.ULONG()
    system_execution_state = 16
    result = ctypes.WinDLL("powrprof").CallNtPowerInformation(
        system_execution_state, None, 0, ctypes.byref(state), ctypes.sizeof(state)
    )
    assert result == 0, f"CallNtPowerInformation failed with {result:#x}"
    return state.value


def _process_handles() -> int:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    count = wintypes.DWORD()
    assert kernel32.GetProcessHandleCount(kernel32.GetCurrentProcess(), ctypes.byref(count))
    return count.value


def test_hold_and_release_work_and_holding_twice_is_one_request():
    before = _execution_state()
    keep_awake = create_keep_awake()
    keep_awake.hold()
    assert _execution_state() & _SYSTEM_REQUIRED
    keep_awake.hold()
    keep_awake.release()
    assert _execution_state() & _SYSTEM_REQUIRED == before & _SYSTEM_REQUIRED
    keep_awake.release()  # releasing again is harmless


def test_release_on_another_thread_still_lets_the_computer_sleep():
    """Recording starts on the window's thread and the last job ends on the worker's."""
    before = _execution_state()
    keep_awake = create_keep_awake()
    for action in (keep_awake.hold, keep_awake.release):
        thread = threading.Thread(target=action)
        thread.start()
        thread.join()
    assert _execution_state() & _SYSTEM_REQUIRED == before & _SYSTEM_REQUIRED


def test_the_screen_may_still_turn_off():
    before = _execution_state()
    keep_awake = create_keep_awake()
    keep_awake.hold()
    try:
        assert _execution_state() & _DISPLAY_REQUIRED == before & _DISPLAY_REQUIRED
    finally:
        keep_awake.release()


def test_many_busy_periods_do_not_leak_system_handles():
    keep_awake = create_keep_awake()
    keep_awake.hold()
    keep_awake.release()
    before = _process_handles()
    for _ in range(200):
        keep_awake.hold()
        keep_awake.release()
    assert _process_handles() - before <= 2
