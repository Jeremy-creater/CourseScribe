"""Small helpers shared by several test files."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

GIVE_UP_AFTER = 200  # far beyond any sane number of tries; reaching it means the loop has no limit
WATCHED_THREAD = "watchdog-"


def hold_without_sharing(path, seconds):
    """Open ``path`` the way antivirus or another program can on Windows: nobody else may read it.

    Other systems have no such hold, so there the file is simply readable, which is the right outcome too.
    """
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    generic_read, no_sharing, open_existing = 0x80000000, 0, 3
    handle = kernel32.CreateFileW(str(path), generic_read, no_sharing, None, open_existing, 0, None)
    assert handle not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    threading.Timer(seconds, kernel32.CloseHandle, [handle]).start()


class KeptTrying(AssertionError):
    """Raised by DeniedFolder once the code under test keeps asking far too often.

    Not an OSError, so a retry loop that only catches OSError stops here instead of running forever.
    """


def _same_place(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


class DeniedFolder:
    """The system answers ``os.open`` in ``folder`` only (no right to create files, disk full, a clash).

    ``times=None`` refuses every time, ``times=k`` the first k times, ``times=0`` never (only records).
    ``error`` may be a function ``step(path)``: it raises to refuse, or returns None to let the real
    ``os.open`` answer. Calls elsewhere go to the real ``os.open`` untouched.
    """

    def __init__(self, monkeypatch, folder, error=PermissionError(13, "Permission denied"), times=None) -> None:
        self.folder = Path(folder)
        self.error = error
        self.times = times
        self.attempts: list[Path] = []
        self.kept_trying = False
        self._fixed = False
        self._lock = threading.Lock()
        real = os.open

        def fake(path, flags, mode=0o777, *args, **kwargs):
            if self._fixed or not _same_place(os.path.dirname(str(path)), self.folder):
                return real(path, flags, mode, *args, **kwargs)
            with self._lock:
                self.attempts.append(Path(path))
                count = len(self.attempts)
                if count > GIVE_UP_AFTER:
                    self.kept_trying = True
            if count > GIVE_UP_AFTER:
                raise KeptTrying(f"试了 {GIVE_UP_AFTER} 次还在试：建文件的循环没有上限")
            if not isinstance(self.error, BaseException):
                self.error(Path(path))  # a step function decides; an exception class here would fail loudly
            elif self.times is None or count <= self.times:
                raise type(self.error)(*self.error.args)
            return real(path, flags, mode, *args, **kwargs)

        monkeypatch.setattr("coursescribe.files.os.open", fake)

    def fix(self) -> None:
        """From now on every call goes to the real ``os.open``, without being recorded."""
        self._fixed = True


class Watchdog:
    """Runs calls on daemon threads so a hang fails the test instead of the whole run."""

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> Watchdog:
        return self

    def run(self, seconds, fn, *args):
        outcome: dict = {}

        def work():
            try:
                outcome["value"] = fn(*args)
            except BaseException as exc:  # noqa: BLE001  handed back to the test as it was
                outcome["error"] = exc

        thread = threading.Thread(target=work, daemon=True, name=f"{WATCHED_THREAD}{len(self._threads)}")
        self._threads.append(thread)
        thread.start()
        thread.join(seconds)
        if thread.is_alive():
            pytest.fail(f"{seconds} 秒还没结束：写文件卡住了")
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def __exit__(self, kind, error, trace) -> bool:
        for thread in self._threads:
            thread.join(5)
        if any(thread.is_alive() for thread in self._threads):
            if error is not None:
                raise pytest.fail.Exception("后台线程 5 秒后还没停") from error
            pytest.fail("后台线程 5 秒后还没停")
        return False


def finishes_within(seconds, fn, *args):
    with Watchdog() as dog:
        return dog.run(seconds, fn, *args)


class deny_new_files:  # noqa: N801  used like a function, as a context manager
    """Make ``folder`` refuse new files for real: an ACL on Windows, permissions elsewhere."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def __enter__(self):
        if os.name == "nt":
            user = os.environ.get("USERNAME", "")
            subprocess.run(["icacls", str(self.folder), "/deny", f"{user}:(WD,AD)"], check=True, capture_output=True)
        else:
            self.mode = self.folder.stat().st_mode
            self.folder.chmod(0o555)
        probe = self.folder / "probe.tmp"
        try:
            probe.write_bytes(b"x")
        except OSError:
            return self
        probe.unlink()
        self.__exit__()
        pytest.fail(f"Could not make {self.folder} read-only for this test")

    def __exit__(self, *args):
        if os.name == "nt":
            user = os.environ.get("USERNAME", "")
            subprocess.run(["icacls", str(self.folder), "/remove:d", user], check=True, capture_output=True)
        else:
            self.folder.chmod(self.mode)
