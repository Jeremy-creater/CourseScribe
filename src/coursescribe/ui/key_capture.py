"""Turn a Qt key press into candidate shortcut text.

Pure: the platform name is passed in ("windows" or "mac"), nothing here reads the operating system. The key is
read by its Qt key code, never by the typed text, so an input method that turns the key into other characters
changes nothing. The rules about what is allowed stay in coursescribe.shortcut; this only says which text the
press stands for.

On a Mac Qt swaps two meanings: ControlModifier is the Command key and MetaModifier is the Control key. On
Windows MetaModifier is the Win key, which the rules refuse.
"""

from __future__ import annotations

from PySide6.QtCore import Qt

_MODIFIER = Qt.KeyboardModifier
# a key code that no allowed key has, so the rules answer "that key is not allowed"
NOT_A_KEY = "?"

_PURE_MODIFIER_KEYS = frozenset(
    int(k.value)
    for k in (
        Qt.Key.Key_Shift,
        Qt.Key.Key_Control,
        Qt.Key.Key_Meta,
        Qt.Key.Key_Alt,
        Qt.Key.Key_AltGr,
        Qt.Key.Key_Super_L,
        Qt.Key.Key_Super_R,
        Qt.Key.Key_CapsLock,
        Qt.Key.Key_unknown,  # what an input method often reports while composing
    )
)

# Shift + number row on a US keyboard: Qt reports the symbol, the shortcut means the digit
_SHIFTED_DIGITS = {
    Qt.Key.Key_Exclam: "1",
    Qt.Key.Key_At: "2",
    Qt.Key.Key_NumberSign: "3",
    Qt.Key.Key_Dollar: "4",
    Qt.Key.Key_Percent: "5",
    Qt.Key.Key_AsciiCircum: "6",
    Qt.Key.Key_Ampersand: "7",
    Qt.Key.Key_Asterisk: "8",
    Qt.Key.Key_ParenLeft: "9",
    Qt.Key.Key_ParenRight: "0",
}
_SHIFTED = {int(k.value): digit for k, digit in _SHIFTED_DIGITS.items()}

_F1 = int(Qt.Key.Key_F1.value)
_A, _Z = int(Qt.Key.Key_A.value), int(Qt.Key.Key_Z.value)
_0, _9 = int(Qt.Key.Key_0.value), int(Qt.Key.Key_9.value)

# keys that leave the recorder when pressed with nothing but (maybe) Shift
LEAVING_KEYS = frozenset(
    int(k.value) for k in (Qt.Key.Key_Escape, Qt.Key.Key_Tab, Qt.Key.Key_Backtab, Qt.Key.Key_Return, Qt.Key.Key_Enter)
)


def is_modifier_key(key: int) -> bool:
    return int(key) in _PURE_MODIFIER_KEYS


def is_leaving_press(key: int, modifiers) -> bool:
    """Esc, Tab, Shift+Tab and Enter on their own: they leave the recorder and are never recorded."""
    return int(key) in LEAVING_KEYS and not (modifiers & ~(_MODIFIER.ShiftModifier | _MODIFIER.KeypadModifier))


def _main_key(key: int, modifiers) -> str:
    if modifiers & _MODIFIER.KeypadModifier:
        return NOT_A_KEY  # the keypad is a different physical key from the number row
    if _A <= key <= _Z:
        return chr(key).lower()
    if _0 <= key <= _9:
        return chr(key)
    if _F1 <= key <= _F1 + 11:
        return f"f{key - _F1 + 1}"
    return _SHIFTED.get(key, NOT_A_KEY)


def candidate_text(key: int, modifiers, platform: str) -> str | None:
    """The shortcut text for one key press, or None when it is a modifier on its own (keep waiting)."""
    key = int(key)
    if key in _PURE_MODIFIER_KEYS:
        return None
    words: list[str] = []
    if platform == "mac":
        pairs = [(_MODIFIER.MetaModifier, "ctrl"), (_MODIFIER.AltModifier, "alt"), (_MODIFIER.ShiftModifier, "shift"), (_MODIFIER.ControlModifier, "cmd")]
    else:
        pairs = [(_MODIFIER.ControlModifier, "ctrl"), (_MODIFIER.AltModifier, "alt"), (_MODIFIER.ShiftModifier, "shift"), (_MODIFIER.MetaModifier, "win")]
    for flag, word in pairs:
        if modifiers & flag:
            words.append(word)
    words.append(_main_key(key, modifiers))
    return "+".join(words)
