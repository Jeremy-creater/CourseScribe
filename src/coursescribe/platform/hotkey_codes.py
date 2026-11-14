"""Native key codes for a recording shortcut.

Pure tables, no system calls, so they can be checked on any computer. Windows takes
RegisterHotKey modifier bits and a virtual-key code; the Mac takes Carbon modifier bits
and a kVK_ANSI_* / kVK_F* key code. A modifier or key a system lacks raises
``UnsupportedShortcut``: a wrong code would register some other key, so none is ever guessed.
"""

from __future__ import annotations

from coursescribe.shortcut import Shortcut


class UnsupportedShortcut(ValueError):
    """The shortcut has a modifier or key this system has no code for."""


# WinUser.h: MOD_ALT, MOD_CONTROL, MOD_SHIFT (MOD_NOREPEAT is added when registering).
_WINDOWS_MODIFIERS = {"alt": 0x0001, "ctrl": 0x0002, "shift": 0x0004}
# Carbon Events.h: cmdKey, shiftKey, optionKey, controlKey.
_MAC_MODIFIERS = {"cmd": 0x0100, "shift": 0x0200, "alt": 0x0800, "ctrl": 0x1000}

# Virtual-key codes: letters and digits are their ASCII capitals, F1..F12 are VK_F1..VK_F12.
_WINDOWS_KEYS = {
    **{chr(code).lower(): code for code in range(ord("A"), ord("Z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    **{f"f{n}": 0x70 + n - 1 for n in range(1, 13)},
}

# HIToolbox/Events.h kVK_ANSI_* and kVK_F*: these follow the physical keyboard, not the alphabet.
_MAC_KEYS = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07,
    "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10,
    "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "9": 0x19,
    "7": 0x1A, "8": 0x1C, "0": 0x1D, "o": 0x1F, "u": 0x20, "i": 0x22, "p": 0x23, "l": 0x25,
    "j": 0x26, "k": 0x28, "n": 0x2D, "m": 0x2E,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61,
    "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
}


def _codes(shortcut: Shortcut, modifiers: dict[str, int], keys: dict[str, int], system: str) -> tuple[int, int]:
    bits = 0
    for name in sorted(shortcut.modifiers):
        if name not in modifiers:
            raise UnsupportedShortcut(f"{system} has no {name!r} modifier key")
        bits |= modifiers[name]
    if shortcut.key not in keys:
        raise UnsupportedShortcut(f"{system} has no code for the key {shortcut.key!r}")
    return bits, keys[shortcut.key]


def windows_codes(shortcut: Shortcut) -> tuple[int, int]:
    """(modifier bits, virtual-key code), without MOD_NOREPEAT."""
    return _codes(shortcut, _WINDOWS_MODIFIERS, _WINDOWS_KEYS, "Windows")


def mac_codes(shortcut: Shortcut) -> tuple[int, int]:
    """(Carbon modifier bits, key code)."""
    return _codes(shortcut, _MAC_MODIFIERS, _MAC_KEYS, "Mac")
