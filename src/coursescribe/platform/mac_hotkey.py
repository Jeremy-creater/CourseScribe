"""A global shortcut through Carbon's RegisterEventHotKey: it needs no extra privacy permission."""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

HOTKEY_LABEL = "⌃⌥R"
_CARBON = "/System/Library/Frameworks/Carbon.framework/Carbon"
_EVENT_CLASS_KEYBOARD = 0x6B657962  # 'keyb'
_EVENT_HOTKEY_PRESSED = 5
_CONTROL_KEY = 0x1000
_OPTION_KEY = 0x0800
_KEY_R = 0x0F
_SIGNATURE = 0x43534352  # 'CSCR'
_NO_ERR = 0


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _HotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class MacHotkey:
    """Control+Option+R. ``start`` must run on the main thread, whose event loop delivers the press."""

    _next_id = 1

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._carbon = None
        self._hotkey_ref = ctypes.c_void_p()
        self._handler_ref = ctypes.c_void_p()
        self._handler = _HANDLER(self._handle)
        self._active = False

    def _on_pressed(self) -> None:
        try:
            self._callback()
        except Exception:  # never let an error cross back into the system's event code
            logger.exception("The shortcut's action failed")

    def _handle(self, _call_ref, _event, _data) -> int:
        self._on_pressed()
        return _NO_ERR

    def start(self) -> bool:
        if self._active:
            return True
        try:
            carbon = ctypes.CDLL(_CARBON)
            carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
            carbon.InstallEventHandler.argtypes = [
                ctypes.c_void_p, _HANDLER, ctypes.c_uint32, ctypes.POINTER(_EventTypeSpec),
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ]
            carbon.RegisterEventHotKey.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32, _HotKeyID, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
            carbon.RemoveEventHandler.argtypes = [ctypes.c_void_p]
            target = carbon.GetApplicationEventTarget()
            spec = _EventTypeSpec(_EVENT_CLASS_KEYBOARD, _EVENT_HOTKEY_PRESSED)
            status = carbon.InstallEventHandler(target, self._handler, 1, ctypes.byref(spec), None, ctypes.byref(self._handler_ref))
            if status != _NO_ERR:
                logger.warning("InstallEventHandler failed: %s", status)
                return False
            MacHotkey._next_id += 1
            status = carbon.RegisterEventHotKey(
                _KEY_R, _CONTROL_KEY | _OPTION_KEY, _HotKeyID(_SIGNATURE, MacHotkey._next_id), target, 0,
                ctypes.byref(self._hotkey_ref),
            )
        except (OSError, AttributeError):
            logger.warning("Carbon is not available", exc_info=True)
            return False
        if status != _NO_ERR:
            logger.warning("RegisterEventHotKey failed: %s", status)
            carbon.RemoveEventHandler(self._handler_ref)
            self._handler_ref = ctypes.c_void_p()
            return False
        self._carbon = carbon
        self._active = True
        return True

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._carbon.UnregisterEventHotKey(self._hotkey_ref)
        self._carbon.RemoveEventHandler(self._handler_ref)
        self._hotkey_ref = ctypes.c_void_p()
        self._handler_ref = ctypes.c_void_p()


def create_global_hotkey(callback: Callable[[], None]) -> MacHotkey:
    return MacHotkey(callback)
