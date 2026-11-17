"""The Mac keeps the computer awake with a ``caffeinate`` process that lives exactly as long as the request.

``caffeinate`` itself only exists on a Mac, so these run a real stand-in process (a sleeping Python) in its place.
What stays for the Mac: that the computer really does not sleep.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from coursescribe.platform.macos import MacKeepAwake, caffeinate_command

SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


def _alive(process: subprocess.Popen) -> bool:
    return process.poll() is None


@pytest.fixture
def keep_awake():
    spawned: list[subprocess.Popen] = []

    def spawn(command, **kwargs):
        process = subprocess.Popen(SLEEPER, **kwargs)
        spawned.append(process)
        return process

    request = MacKeepAwake(spawn=spawn)
    request.spawned = spawned
    yield request
    for process in spawned:
        if _alive(process):
            process.kill()
        process.wait(timeout=5)


def test_caffeinate_is_asked_to_stop_idle_sleep_and_to_end_with_this_program():
    command = caffeinate_command(4321)
    assert command[0] == "caffeinate"
    assert "-i" in command  # no sleep while idle; the screen may still turn off, closing the lid still sleeps
    assert command[command.index("-w") + 1] == "4321"  # the Mac ends it when this program ends, even a crash
    assert "-d" not in command and "-s" not in command


def test_hold_starts_one_process_and_release_ends_it(keep_awake):
    keep_awake.hold()
    (process,) = keep_awake.spawned
    assert _alive(process)
    keep_awake.release()
    assert process.poll() is not None


def test_holding_twice_is_still_one_process(keep_awake):
    keep_awake.hold()
    keep_awake.hold()
    assert len(keep_awake.spawned) == 1


def test_releasing_without_holding_does_nothing(keep_awake):
    keep_awake.release()
    keep_awake.hold()
    keep_awake.release()
    keep_awake.release()
    assert len(keep_awake.spawned) == 1


def test_it_can_be_held_again_after_a_release(keep_awake):
    for _ in range(3):
        keep_awake.hold()
        keep_awake.release()
    assert len(keep_awake.spawned) == 3
    assert all(not _alive(process) for process in keep_awake.spawned)


def test_a_process_that_died_on_its_own_is_replaced_by_the_next_hold(keep_awake):
    keep_awake.hold()
    keep_awake.spawned[0].kill()
    keep_awake.spawned[0].wait(timeout=5)
    keep_awake.hold()
    assert len(keep_awake.spawned) == 2
    assert _alive(keep_awake.spawned[1])
    keep_awake.release()
    assert not _alive(keep_awake.spawned[1])


def test_releasing_after_the_process_already_died_is_fine(keep_awake):
    keep_awake.hold()
    keep_awake.spawned[0].kill()
    keep_awake.spawned[0].wait(timeout=5)
    keep_awake.release()


def test_a_missing_caffeinate_is_reported_so_the_program_can_warn_the_user():
    request = MacKeepAwake(spawn=lambda command, **kwargs: subprocess.Popen(["no-such-program-coursescribe"], **kwargs))
    with pytest.raises(OSError):
        request.hold()
    request.release()  # nothing was held

