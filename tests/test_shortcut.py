"""Shortcut parsing, validation and display. Pure logic, no Qt, no system calls."""

from __future__ import annotations

import dataclasses
import itertools
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coursescribe.shortcut import InvalidShortcut, Shortcut, check, parse

PLATFORMS = ("windows", "mac")

# The reasons are written out here on purpose: they are the specification.
R_NOT_TEXT = "快捷键必须是文字。"
R_BLANK = "快捷键不能是空的。"
R_EMPTY_PART = "快捷键里有多余的“+”，每个“+”前后都要有按键。"
R_NO_MAIN = "还差一个主键，请再按一个字母、数字或 F1 到 F12。"
R_TWO_MAIN = "主键只能有一个。"
R_KEY_NOT_ALLOWED = "主键只能是字母 A 到 Z、数字 0 到 9 或 F1 到 F12。"
R_NEEDS_MODIFIER = "要带上 Ctrl、Alt 或 Command 中的至少一个，只有 Shift 或不带修饰键容易误触。"
R_WIN = "不支持 Win 键，请改用 Ctrl 或 Alt。"
R_CMD_ON_WINDOWS = "Command 键只在 Mac 上有，请改用 Ctrl 或 Alt。"

ALL_REASONS = {
    R_NOT_TEXT,
    R_BLANK,
    R_EMPTY_PART,
    R_NO_MAIN,
    R_TWO_MAIN,
    R_KEY_NOT_ALLOWED,
    R_NEEDS_MODIFIER,
    R_WIN,
    R_CMD_ON_WINDOWS,
}

KEYS = (
    [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(d) for d in range(10)]
    + [f"f{n}" for n in range(1, 13)]
)
MODS = ("ctrl", "alt", "shift", "cmd")  # canonical order
MAC_GLYPH = {"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"}
WIN_NAME = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "cmd": "Cmd"}


def _subsets():
    for n in range(len(MODS) + 1):
        for combo in itertools.combinations(MODS, n):
            yield frozenset(combo)


def _expected_valid(mods: frozenset, platform: str) -> bool:
    if platform == "windows" and "cmd" in mods:
        return False
    return bool(mods & {"ctrl", "alt", "cmd"})


def _canonical(mods: frozenset, key: str) -> str:
    return "+".join([m for m in MODS if m in mods] + [key])


def _parse_or_reason(text, platform):
    try:
        return parse(text, platform), None
    except InvalidShortcut as exc:
        return None, exc.reason


# 1. valid table ------------------------------------------------------------

VALID_TABLE = [
    ("ctrl+alt+r", "ctrl+alt+r", "Ctrl+Alt+R", "⌃⌥R"),
    (" CTRL + Alt+R ", "ctrl+alt+r", "Ctrl+Alt+R", "⌃⌥R"),
    ("alt+ctrl+r", "ctrl+alt+r", "Ctrl+Alt+R", "⌃⌥R"),
    ("ctrl+ctrl+alt+r", "ctrl+alt+r", "Ctrl+Alt+R", "⌃⌥R"),
    ("ctrl+shift+r", "ctrl+shift+r", "Ctrl+Shift+R", "⌃⇧R"),
    ("cmd+shift+4", "shift+cmd+4", None, "⇧⌘4"),
    ("ctrl+alt+shift+cmd+f12", "ctrl+alt+shift+cmd+f12", None, "⌃⌥⇧⌘F12"),
    ("alt+f5", "alt+f5", "Alt+F5", "⌥F5"),
    ("ctrl+0", "ctrl+0", "Ctrl+0", "⌃0"),
]


@pytest.mark.parametrize("text, canonical, win_label, mac_label", VALID_TABLE)
def test_valid_table(text, canonical, win_label, mac_label):
    shortcut = parse(text, "mac")
    assert shortcut.serialize() == canonical
    assert shortcut.display("mac") == mac_label
    if win_label is not None:
        windows = parse(text, "windows")
        assert windows == shortcut
        assert windows.display("windows") == win_label


def test_cmd_serializes_after_shift_in_canonical_order():
    assert parse("cmd+shift+ctrl+alt+a", "mac").serialize() == "ctrl+alt+shift+cmd+a"


# 2. invalid table ------------------------------------------------------------

# (input, platforms where it is invalid, reason)
INVALID_TABLE = [
    ("", PLATFORMS, R_BLANK),
    ("   ", PLATFORMS, R_BLANK),
    ("\t\n", PLATFORMS, R_BLANK),
    (None, PLATFORMS, R_NOT_TEXT),
    (42, PLATFORMS, R_NOT_TEXT),
    (["ctrl", "r"], PLATFORMS, R_NOT_TEXT),
    (b"ctrl+alt+r", PLATFORMS, R_NOT_TEXT),
    ("ctrl+alt", PLATFORMS, R_NO_MAIN),
    ("shift", PLATFORMS, R_NO_MAIN),
    ("r", PLATFORMS, R_NEEDS_MODIFIER),
    ("f5", PLATFORMS, R_NEEDS_MODIFIER),
    ("shift+r", PLATFORMS, R_NEEDS_MODIFIER),
    ("shift+f5", PLATFORMS, R_NEEDS_MODIFIER),
    ("ctrl+a+b", PLATFORMS, R_TWO_MAIN),
    ("a+b", PLATFORMS, R_TWO_MAIN),
    ("ctrl+;", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+space", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+f13", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+f0", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+é", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+Ｒ", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+٣", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+ab", PLATFORMS, R_KEY_NOT_ALLOWED),
    ("ctrl+alt+", PLATFORMS, R_EMPTY_PART),
    ("+", PLATFORMS, R_EMPTY_PART),
    ("ctrl++r", PLATFORMS, R_EMPTY_PART),
    ("+ctrl+r", PLATFORMS, R_EMPTY_PART),
    ("win+r", PLATFORMS, R_WIN),
    ("meta+r", PLATFORMS, R_WIN),
    ("super+r", PLATFORMS, R_WIN),
    ("ctrl+win+r", PLATFORMS, R_WIN),
    ("cmd+win+r", PLATFORMS, R_WIN),
    ("cmd+r", ("windows",), R_CMD_ON_WINDOWS),
    ("ctrl+cmd+f5", ("windows",), R_CMD_ON_WINDOWS),
]


def _invalid_cases():
    for text, platforms, reason in INVALID_TABLE:
        for platform in platforms:
            yield pytest.param(text, platform, reason, id=f"{text!r}-{platform}")


@pytest.mark.parametrize("text, platform, reason", list(_invalid_cases()))
def test_invalid_table_gives_exact_reason(text, platform, reason):
    shortcut, got = _parse_or_reason(text, platform)
    assert shortcut is None
    assert got == reason
    assert str(InvalidShortcut(reason)) == reason


def test_each_rule_has_its_own_non_empty_reason():
    assert all(r.strip() for r in ALL_REASONS)
    assert len(ALL_REASONS) == 9
    used = {reason for _, _, reason in INVALID_TABLE}
    assert used == ALL_REASONS


def test_win_words_are_never_silently_command_on_mac():
    _, win_reason = _parse_or_reason("win+r", "mac")
    cmd = parse("cmd+r", "mac")
    assert win_reason is not None
    assert cmd.modifiers == frozenset({"cmd"})
    for word in ("win", "meta", "super"):
        shortcut, reason = _parse_or_reason(f"{word}+alt+r", "mac")
        assert shortcut is None and reason == R_WIN


def test_windows_reasons_name_the_right_key():
    _, win = _parse_or_reason("win+r", "windows")
    _, cmd = _parse_or_reason("cmd+r", "windows")
    assert "Win" in win and "不支持" in win
    assert "Command" in cmd and "Mac" in cmd
    assert win != cmd


def test_parse_is_case_insensitive_for_win_words_too():
    assert _parse_or_reason("WIN+R", "windows")[1] == R_WIN
    assert _parse_or_reason("Meta + R", "mac")[1] == R_WIN


# 3. shift boundary -----------------------------------------------------------


def test_shift_is_only_an_extra():
    for platform in PLATFORMS:
        assert _parse_or_reason("shift", platform)[1] is not None
        assert _parse_or_reason("shift+r", platform)[1] == R_NEEDS_MODIFIER
        assert _parse_or_reason("shift+f5", platform)[1] == R_NEEDS_MODIFIER
        assert parse("ctrl+shift+r", platform).serialize() == "ctrl+shift+r"
        assert parse("alt+shift+r", platform).serialize() == "alt+shift+r"


# 4. exhaustive round trip and idempotence, plus 1-A display -------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_exhaustive_round_trip_idempotence_and_display(platform):
    seen_serialized = {}
    seen_display = {}
    valid_count = 0
    for mods in _subsets():
        for key in KEYS:
            text = _canonical(mods, key)
            shortcut, reason = _parse_or_reason(text, platform)
            if not _expected_valid(mods, platform):
                assert shortcut is None, text
                assert reason in ALL_REASONS, text
                continue
            valid_count += 1
            assert reason is None, text
            assert shortcut == Shortcut(mods, key)
            assert shortcut.serialize() == text
            # round trip and idempotence
            again = parse(shortcut.serialize(), platform)
            assert again == shortcut
            assert again.serialize() == shortcut.serialize()
            assert check(shortcut, platform) is None
            # no two different shortcuts share a string
            assert seen_serialized.setdefault(text, shortcut) == shortcut
            # display against an independently built expectation
            if platform == "windows":
                expected = "+".join([WIN_NAME[m] for m in MODS if m in mods] + [key.upper()])
            else:
                expected = "".join([MAC_GLYPH[m] for m in MODS if m in mods]) + key.upper()
            assert shortcut.display(platform) == expected
            assert expected not in seen_display, expected
            seen_display[expected] = text
    # windows: subsets of {ctrl,alt,shift} containing ctrl or alt = 6;
    # mac: all 16 subsets except none and shift-only = 14.
    assert valid_count == (6 if platform == "windows" else 14) * len(KEYS)
    assert len(KEYS) == 48


def test_different_orderings_collapse_to_one_shortcut():
    for platform in PLATFORMS:
        mods = ["ctrl", "alt", "shift"]
        results = {
            parse("+".join(list(p) + ["f7"]), platform) for p in itertools.permutations(mods)
        }
        assert len(results) == 1


def test_check_validates_hand_built_shortcuts():
    assert check(Shortcut(frozenset({"ctrl"}), "r"), "windows") is None
    assert check(Shortcut(frozenset({"shift"}), "r"), "mac") == R_NEEDS_MODIFIER
    assert check(Shortcut(frozenset(), "r"), "mac") == R_NEEDS_MODIFIER
    assert check(Shortcut(frozenset({"ctrl"}), "space"), "mac") == R_KEY_NOT_ALLOWED
    assert check(Shortcut(frozenset({"cmd"}), "r"), "windows") == R_CMD_ON_WINDOWS
    assert check(Shortcut(frozenset({"cmd"}), "r"), "mac") is None
    assert check(Shortcut(frozenset({"ctrl"}), "R"), "mac") == R_KEY_NOT_ALLOWED


def test_unknown_platform_is_a_programming_error():
    with pytest.raises(ValueError):
        parse("ctrl+r", "linux")
    with pytest.raises(ValueError):
        Shortcut(frozenset({"ctrl"}), "r").display("linux")


# 5. random text totality -------------------------------------------------------

FRAGMENTS = ["ctrl", "alt", "shift", "cmd", "win", "meta", "super", "r", "a", "4", "f5", "f12",
             "f13", "space", ";", "é", "Ｒ", "", " ", "+", "ctrl ", " alt", "CMD", "Shift"]


def _random_case(rng: random.Random):
    kind = rng.randrange(6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.randrange(-5, 5000)
    if kind == 2:
        return [rng.choice(FRAGMENTS) for _ in range(rng.randrange(4))]
    if kind == 3:
        return "".join(chr(rng.randrange(0, 0x3100)) for _ in range(rng.randrange(0, 30)))
    parts = [rng.choice(FRAGMENTS) for _ in range(rng.randrange(1, 6))]
    text = "+".join(parts)
    return "".join(c.upper() if rng.random() < 0.3 else c for c in text)


def test_random_input_is_total_and_valid_results_obey_the_rules():
    rng = random.Random(20261114)
    valid_seen = 0
    for _ in range(2000):
        value = _random_case(rng)
        for platform in PLATFORMS:
            shortcut, reason = _parse_or_reason(value, platform)
            if shortcut is None:
                assert isinstance(reason, str) and reason in ALL_REASONS
                continue
            valid_seen += 1
            assert reason is None
            mods = shortcut.modifiers
            assert mods & {"ctrl", "alt", "cmd"}
            assert shortcut.key in KEYS
            assert mods <= set(MODS)
            if platform == "windows":
                assert "cmd" not in mods
            assert parse(shortcut.serialize(), platform) == shortcut
    assert valid_seen > 50  # the fragments really do hit valid shortcuts


# 6. long input -----------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_very_long_inputs(platform):
    start = time.perf_counter()
    merged = parse("ctrl+" * 10000 + "r", platform)
    assert merged.serialize() == "ctrl+r"
    assert _parse_or_reason("ctrl+" + "a" * 10000, platform)[1] == R_KEY_NOT_ALLOWED
    assert _parse_or_reason("+" * 10000, platform)[1] == R_EMPTY_PART
    assert _parse_or_reason(" " * 10000, platform)[1] == R_BLANK
    assert _parse_or_reason("ctrl+a" * 10000, platform)[1] is not None
    assert time.perf_counter() - start < 1.0


def test_one_long_key_word_is_invalid_with_a_reason():
    for platform in PLATFORMS:
        shortcut, reason = _parse_or_reason("a" * 10000, platform)
        assert shortcut is None
        assert reason in ALL_REASONS


def test_parsing_time_grows_linearly():
    def cost(n):
        text = "+".join(["ctrl", "alt"] * n) + "+r"
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            parse(text, "mac")
            best = min(best, time.perf_counter() - start)
        return best

    small, large = cost(5000), cost(40000)
    assert large < max(small, 0.002) * 40  # 8x input, far below quadratic (64x)


# 7. determinism ------------------------------------------------------------------


def test_same_input_same_result_twice():
    inputs = [row[0] for row in VALID_TABLE] + [row[0] for row in INVALID_TABLE[:11]]
    for platform in PLATFORMS:
        for text in inputs:
            assert _parse_or_reason(text, platform) == _parse_or_reason(text, platform)


# 8. frozen and equal ----------------------------------------------------------------


def test_equal_hash_equal_and_frozen():
    a = parse("ctrl+alt+r", "windows")
    b = parse("alt+ctrl+r", "windows")
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1
    assert a != parse("ctrl+alt+t", "windows")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.key = "t"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.modifiers = frozenset()  # type: ignore[misc]


# 9. module purity ----------------------------------------------------------------------


def test_import_loads_no_qt_platform_or_settings_modules():
    src = str(Path(__file__).resolve().parent.parent / "src")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import coursescribe.shortcut;"
        "bad = [m for m in sys.modules if m.split('.')[0] == 'PySide6'"
        " or m in ('coursescribe.platform', 'coursescribe.settings')"
        " or m.startswith('coursescribe.platform.')];"
        "print('BAD', bad); sys.exit(1 if bad else 0)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, src], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_module_does_not_read_the_operating_system():
    text = (Path(__file__).resolve().parent.parent / "src/coursescribe/shortcut.py").read_text(
        encoding="utf-8"
    )
    for needle in ("sys.platform", "platform.system", "os.name", "import platform"):
        assert needle not in text
