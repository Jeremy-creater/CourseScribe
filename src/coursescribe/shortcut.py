"""Recording shortcut: parse, validate, serialize and display.

Pure logic. The platform is always passed in ("windows" or "mac"); this module
never looks at the operating system and imports nothing from Qt, the platform
package or the settings.
"""

from __future__ import annotations

from dataclasses import dataclass

MODIFIER_ORDER = ("ctrl", "alt", "shift", "cmd")
KEYS = frozenset(
    [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(d) for d in range(10)]
    + [f"f{n}" for n in range(1, 13)]
)
_WIN_WORDS = frozenset({"win", "meta", "super"})
_PLATFORMS = ("windows", "mac")

_MAC_GLYPH = {"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"}
_WIN_NAME = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "cmd": "Cmd"}

REASON_NOT_TEXT = "快捷键必须是文字。"
REASON_BLANK = "快捷键不能是空的。"
REASON_EMPTY_PART = "快捷键里有多余的“+”，每个“+”前后都要有按键。"
REASON_NO_MAIN = "还差一个主键，请再按一个字母、数字或 F1 到 F12。"
REASON_TWO_MAIN = "主键只能有一个。"
REASON_KEY_NOT_ALLOWED = "主键只能是字母 A 到 Z、数字 0 到 9 或 F1 到 F12。"
REASON_NEEDS_MODIFIER = "要带上 Ctrl、Alt 或 Command 中的至少一个，只有 Shift 或不带修饰键容易误触。"
REASON_WIN = "不支持 Win 键，请改用 Ctrl 或 Alt。"
REASON_CMD_ON_WINDOWS = "Command 键只在 Mac 上有，请改用 Ctrl 或 Alt。"


class InvalidShortcut(ValueError):
    """The text is not an allowed shortcut; `reason` is a Chinese sentence for the user."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Shortcut:
    modifiers: frozenset[str]
    key: str

    def serialize(self) -> str:
        """Canonical text: modifiers in the order ctrl, alt, shift, cmd, then the key."""
        return "+".join([m for m in MODIFIER_ORDER if m in self.modifiers] + [self.key])

    def display(self, platform: str) -> str:
        _require_platform(platform)
        mods = [m for m in MODIFIER_ORDER if m in self.modifiers]
        if platform == "mac":
            return "".join(_MAC_GLYPH[m] for m in mods) + self.key.upper()
        return "+".join([_WIN_NAME[m] for m in mods] + [self.key.upper()])


def _require_platform(platform: str) -> None:
    if platform not in _PLATFORMS:
        raise ValueError(f"unknown platform: {platform!r}")


def check(shortcut: Shortcut, platform: str) -> str | None:
    """Return the reason this shortcut is not allowed on the platform, or None."""
    _require_platform(platform)
    if platform == "windows" and "cmd" in shortcut.modifiers:
        return REASON_CMD_ON_WINDOWS
    if shortcut.key not in KEYS:
        return REASON_KEY_NOT_ALLOWED
    if not shortcut.modifiers & {"ctrl", "alt", "cmd"}:
        return REASON_NEEDS_MODIFIER
    return None


def parse(text: object, platform: str) -> Shortcut:
    """Parse text such as "ctrl+alt+r"; raise InvalidShortcut(reason) when not allowed.

    Total over `text`: any value gives a Shortcut or an InvalidShortcut.
    """
    _require_platform(platform)
    if not isinstance(text, str):
        raise InvalidShortcut(REASON_NOT_TEXT)
    if not text.strip():
        raise InvalidShortcut(REASON_BLANK)
    words = [part.strip().lower() for part in text.split("+")]
    if "" in words:
        raise InvalidShortcut(REASON_EMPTY_PART)
    word_set = set(words)
    if word_set & _WIN_WORDS:
        raise InvalidShortcut(REASON_WIN)
    modifiers = frozenset(word_set & set(MODIFIER_ORDER))
    mains = word_set - set(MODIFIER_ORDER)
    if not mains:
        raise InvalidShortcut(REASON_NO_MAIN)
    if len(mains) > 1:
        raise InvalidShortcut(REASON_TWO_MAIN)
    shortcut = Shortcut(modifiers, next(iter(mains)))
    reason = check(shortcut, platform)
    if reason is not None:
        raise InvalidShortcut(reason)
    return shortcut
