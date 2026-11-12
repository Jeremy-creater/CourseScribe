"""The real Mac keep-awake request, read back from the system's own power assertions."""

from __future__ import annotations

import re
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.gui

if sys.platform != "darwin":
    pytest.fail("These tests need a real Mac (pmset power assertions).", pytrace=False)

from coursescribe.platform import create_keep_awake


def _caffeinate_pids() -> set[int]:
    """Processes the system lists as keeping it awake while idle. Others (not ours) may be there already."""
    out = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, check=True).stdout
    return {
        int(m.group(1))
        for line in out.splitlines()
        if "PreventUserIdleSystemSleep" in line and (m := re.search(r"pid (\d+)\(caffeinate\)", line))
    }


def _wait_until(condition, seconds=5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.1)
    return condition()


def test_hold_shows_up_in_the_system_and_release_removes_it():
    before = _caffeinate_pids()
    keep_awake = create_keep_awake()
    keep_awake.hold()
    try:
        assert _wait_until(lambda: _caffeinate_pids() - before)
        ours = _caffeinate_pids() - before
        keep_awake.hold()  # holding twice is still one request
        assert _caffeinate_pids() - before == ours
    finally:
        keep_awake.release()
    assert _wait_until(lambda: not (_caffeinate_pids() & ours))
    keep_awake.release()  # releasing again is harmless


def test_killing_the_program_forcibly_still_lets_the_computer_sleep():
    before = _caffeinate_pids()
    code = (
        "from coursescribe.platform import create_keep_awake; import time;"
        "k = create_keep_awake(); k.hold(); print('held', flush=True); time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        assert _wait_until(lambda: _caffeinate_pids() - before)
        ours = _caffeinate_pids() - before
        child.kill()
        child.wait(timeout=5)
        assert _wait_until(lambda: not (_caffeinate_pids() & ours))
    finally:
        child.kill()
