"""A global shortcut through Windows' RegisterHotKey.

Kept apart from ``windows.py`` so the registration can be exercised with a stand-in for
``user32`` on any computer; the real libraries are only looked up when none is given.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable
from ctypes import wintypes

from coursescribe.platform.hotkey_codes import UnsupportedShortcut, windows_codes
from coursescribe.shortcut import Shortcut

logger = logging.getLogger(__name__)

_MOD_NOREPEAT = 0x4000
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012
_HOTKEY_ID = 1


class GlobalHotkey:
    """Calls back when the shortcut is pressed, whichever program has the focus."""

    def __init__(self, shortcut: Shortcut, callback: Callable[[], None], user32=None, kernel32=None) -> None:
        self._shortcut = shortcut
        self._callback = callback
        self._user32 = user32
        self._kernel32 = kernel32
        self._thread = threading.Thread(target=self._listen, name="hotkey", daemon=True)
        self._ready = threading.Event()
        self._registered = False
        self._thread_id = 0
        self._started = False

    def start(self) -> bool:
        """Returns False when another program already owns this shortcut. A second call changes nothing."""
        if self._started:
            return self._registered
        self._started = True
        self._thread.start()
        self._ready.wait(5)
        return self._registered

    def _listen(self) -> None:
        try:
            self._listen_and_dispatch()
        finally:
            self._ready.set()  # whatever happened, never leave start() waiting

    def _listen_and_dispatch(self) -> None:
        user32 = self._user32 if self._user32 is not None else ctypes.windll.user32
        kernel32 = self._kernel32 if self._kernel32 is not None else ctypes.windll.kernel32
        try:
            modifiers, key = windows_codes(self._shortcut)
        except UnsupportedShortcut:
            logger.warning("This shortcut has no key code here", exc_info=True)
            return
        self._thread_id = kernel32.GetCurrentThreadId()
        self._registered = bool(user32.RegisterHotKey(None, _HOTKEY_ID, modifiers | _MOD_NOREPEAT, key))
        self._ready.set()
        if not self._registered:
            return
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message == _WM_HOTKEY:
                try:
                    self._callback()
                except Exception:  # one failing action must not end the listening
                    logger.exception("The shortcut's action failed")
        user32.UnregisterHotKey(None, _HOTKEY_ID)

    def stop(self) -> None:
        if self._registered and self._thread.is_alive():
            user32 = self._user32 if self._user32 is not None else ctypes.windll.user32
            user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
            self._thread.join(2.0)
        self._registered = False


def create_global_hotkey(shortcut: Shortcut, callback: Callable[[], None]) -> GlobalHotkey:
    return GlobalHotkey(shortcut, callback)
