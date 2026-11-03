"""Small helpers shared by several test files."""

from __future__ import annotations

import os
import threading


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
