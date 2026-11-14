"""The recording-shortcut row of the settings window (WI-4), off-screen with real widgets and real QKeyEvents.

The platform is injected into the dialog, so both the Windows and the Mac reading of the keys run on any
machine. On Mac Qt swaps two meanings: ControlModifier is the Command key and MetaModifier is the Control key.
Expected values below are written out independently of the code under test (tables, not calls into it); only the
reason sentences are imported from the rules module, because they are the rules' own wording.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialogButtonBox, QLabel

from coursescribe import shortcut as rules
from coursescribe.platform import DEFAULT_HOTKEY, HOTKEY_PLATFORM
from coursescribe.settings import Settings, SettingsStore
from coursescribe.ui import key_capture
from coursescribe.ui.settings_dialog import SettingsDialog
from test_ui import Screen, _driven, qt_app  # noqa: F401  (qt_app is a fixture)

# ---------------------------------------------------------------- independent key tables

LETTERS = {chr(c).lower(): c for c in range(0x41, 0x5B)}
DIGITS = {str(d): 0x30 + d for d in range(10)}
FKEYS = {f"f{n}": 0x01000030 + n - 1 for n in range(1, 13)}
ALLOWED = {**LETTERS, **DIGITS, **FKEYS}
assert len(ALLOWED) == 48

# Shift+number-row symbols on a US keyboard, and the digit they stand for
SYMBOLS = {
    0x21: "1",  # Key_Exclam
    0x40: "2",  # Key_At
    0x23: "3",  # Key_NumberSign
    0x24: "4",  # Key_Dollar
    0x25: "5",  # Key_Percent
    0x5E: "6",  # Key_AsciiCircum
    0x26: "7",  # Key_Ampersand
    0x2A: "8",  # Key_Asterisk
    0x28: "9",  # Key_ParenLeft
    0x29: "0",  # Key_ParenRight
}

KEY_ESC, KEY_TAB, KEY_BACKTAB, KEY_RETURN, KEY_ENTER = 0x01000000, 0x01000001, 0x01000002, 0x01000004, 0x01000005
KEY_SHIFT, KEY_CONTROL, KEY_META, KEY_ALT = 0x01000020, 0x01000021, 0x01000022, 0x01000023
KEY_CAPSLOCK, KEY_SUPER_L, KEY_SUPER_R, KEY_ALTGR, KEY_UNKNOWN = 0x01000024, 0x01000053, 0x01000054, 0x01001103, 0x01FFFFFF
MODIFIER_KEYS = {KEY_SHIFT, KEY_CONTROL, KEY_META, KEY_ALT, KEY_CAPSLOCK, KEY_SUPER_L, KEY_SUPER_R, KEY_ALTGR, KEY_UNKNOWN}
BARE_LEAVERS = {KEY_ESC, KEY_TAB, KEY_BACKTAB, KEY_RETURN, KEY_ENTER}
KEY_SPACE, KEY_LEFT = 0x20, 0x01000012
# not allowed as a main key: space, punctuation, arrows, Backspace, Delete, Home
NOT_ALLOWED_KEYS = [KEY_SPACE, KEY_LEFT, 0x3B, 0x2C, 0x2E, 0x2F, 0x2D, 0x3D, 0x5B, 0x5D, 0x01000003, 0x01000007, 0x01000010]

C, A, S, M = (
    Qt.KeyboardModifier.ControlModifier,
    Qt.KeyboardModifier.AltModifier,
    Qt.KeyboardModifier.ShiftModifier,
    Qt.KeyboardModifier.MetaModifier,
)
KEYPAD = Qt.KeyboardModifier.KeypadModifier
NONE = Qt.KeyboardModifier.NoModifier
ORDER = [(C, "C"), (A, "A"), (S, "S"), (M, "M")]
SUBSETS = [frozenset(name for i, (_, name) in enumerate(ORDER) if mask >> i & 1) for mask in range(16)]
WHICH = dict((name, bit) for bit, name in ORDER)


def flags(names, keypad=False):
    out = NONE
    for n in sorted(names):
        out = out | WHICH[n]
    return out | KEYPAD if keypad else out


def event(key, names=(), text="", auto=False, keypad=False):
    return QKeyEvent(QEvent.Type.KeyPress, key, flags(names, keypad), 0, 0, 0, text, auto, 1)


def press(dialog, key, names=(), text="", auto=False, keypad=False):
    made = event(key, names, text, auto, keypad)
    dialog.hotkey_box.keyPressEvent(made)
    return made


def expected_words(platform, names):
    """The shortcut words the pressed modifiers stand for, in canonical order; None when Win is involved."""
    if platform == "windows":
        if "M" in names:
            return None
        table = {"C": "ctrl", "A": "alt", "S": "shift"}
    else:
        table = {"M": "ctrl", "A": "alt", "S": "shift", "C": "cmd"}
    words = {table[n] for n in names if n in table}
    return [w for w in ("ctrl", "alt", "shift", "cmd") if w in words]


def label_of(platform, text):
    """Independent display of a canonical text."""
    words = text.split("+")
    mods, key = words[:-1], words[-1].upper()
    if platform == "windows":
        return "+".join([{"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift"}[m] for m in mods] + [key])
    return "".join({"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"}[m] for m in mods) + key


def oracle(platform, key, names, auto=False, keypad=False):
    """('ignore',) | ('reject',) | ('ok', canonical_text) for one key press, written without the code under test."""
    if auto or key in MODIFIER_KEYS:
        return ("ignore",)
    if key in BARE_LEAVERS and set(names) <= {"S"}:
        return ("ignore",)
    name = None
    if not keypad:
        if key in SYMBOLS:
            name = SYMBOLS[key]
        else:
            name = next((n for n, code in ALLOWED.items() if code == key), None)
    words = expected_words(platform, names)
    if name is None or words is None or not set(words) & {"ctrl", "alt", "cmd"}:
        return ("reject",)
    return ("ok", "+".join(words + [name]))


def make(tmp_path, platform="windows", stored="ctrl+alt+r", default="ctrl+alt+r", **links):
    current = Settings(tmp_path / "out", "large-v3", "model-a", hotkey=stored)
    return SettingsDialog(
        current, ["model-a"], choose_folder=lambda: None, hotkey_platform=platform, default_hotkey=default, **links
    )


def shown(dialog, platform, text):
    return dialog.hotkey_box.text() == label_of(platform, text)


PLATFORMS = ["windows", "mac"]


# ---------------------------------------------------------------- 1 valid keys

def test_windows_ctrl_alt_r_is_recorded(tmp_path, qt_app):
    dialog = make(tmp_path, "windows", stored="alt+f5")
    press(dialog, 0x52, "CA")
    assert dialog.hotkey_box.text() == "Ctrl+Alt+R"
    assert dialog.hotkey_reason.text() == ""
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


def test_mac_command_shift_4_uses_the_swapped_qt_modifiers(tmp_path, qt_app):
    dialog = make(tmp_path, "mac")
    press(dialog, 0x34, "CS")  # on a Mac Qt reports the Command key as ControlModifier
    assert dialog.hotkey_box.text() == "⇧⌘4"
    assert dialog.hotkey_reason.text() == ""
    assert dialog.result_settings().hotkey == "shift+cmd+4"


def test_mac_control_is_the_meta_modifier(tmp_path, qt_app):
    dialog = make(tmp_path, "mac", stored="alt+f5")
    press(dialog, 0x52, "MA")
    assert dialog.hotkey_box.text() == "⌃⌥R"
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


# ---------------------------------------------------------------- 2 invalid keys (and the stored value stays)

INVALID = [
    ("Shift+R", 0x52, "S", rules.REASON_NEEDS_MODIFIER),
    ("bare R", 0x52, "", rules.REASON_NEEDS_MODIFIER),
    ("bare F5", FKEYS["f5"], "", rules.REASON_NEEDS_MODIFIER),
    ("Ctrl+Space", KEY_SPACE, "C", rules.REASON_KEY_NOT_ALLOWED),
    ("Ctrl+Tab", KEY_TAB, "C", rules.REASON_KEY_NOT_ALLOWED),
    ("Ctrl+Return", KEY_RETURN, "C", rules.REASON_KEY_NOT_ALLOWED),
    ("Ctrl+Esc", KEY_ESC, "C", rules.REASON_KEY_NOT_ALLOWED),
    ("Ctrl+Left", KEY_LEFT, "C", rules.REASON_KEY_NOT_ALLOWED),
    ("Ctrl+;", 0x3B, "C", rules.REASON_KEY_NOT_ALLOWED),
    ("Ctrl+,", 0x2C, "C", rules.REASON_KEY_NOT_ALLOWED),
]


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("label, key, names, reason", INVALID, ids=[i[0] for i in INVALID])
def test_an_invalid_key_shows_its_reason_and_keeps_the_stored_value(tmp_path, qt_app, platform, label, key, names, reason):
    dialog = make(tmp_path, platform)
    press(dialog, key, names)
    assert dialog.hotkey_reason.text() == reason
    assert shown(dialog, platform, "ctrl+alt+r")
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


def test_the_windows_key_is_refused_with_its_own_reason(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    press(dialog, 0x52, "M")
    assert dialog.hotkey_reason.text() == rules.REASON_WIN
    assert dialog.hotkey_box.text() == "Ctrl+Alt+R"
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


# ---------------------------------------------------------------- 3 modifier keys alone wait

@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("key", [KEY_CONTROL, KEY_ALT, KEY_SHIFT, KEY_META])
def test_a_modifier_alone_changes_nothing_and_says_nothing(tmp_path, qt_app, platform, key):
    dialog = make(tmp_path, platform)
    press(dialog, key, "C" if key == KEY_CONTROL else "")
    assert dialog.hotkey_reason.text() == ""
    assert shown(dialog, platform, "ctrl+alt+r")
    press(dialog, 0x52, "S")  # an invalid press, reason shown
    reason = dialog.hotkey_reason.text()
    assert reason
    press(dialog, key, "")
    assert dialog.hotkey_reason.text() == reason  # the waiting press does not clear or change it
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


def test_ctrl_alt_held_then_r_records(tmp_path, qt_app):
    dialog = make(tmp_path, "windows", stored="alt+f5")
    press(dialog, KEY_CONTROL, "C")
    press(dialog, KEY_ALT, "CA")
    assert dialog.result_settings().hotkey == "alt+f5"
    press(dialog, 0x52, "CA")
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


# ---------------------------------------------------------------- 4 the last valid one wins

@pytest.mark.parametrize("platform", PLATFORMS)
def test_the_last_valid_key_wins_and_an_invalid_one_never_erases_it(tmp_path, qt_app, platform):
    dialog = make(tmp_path, platform, stored="alt+f5")
    ctrl_names = "C" if platform == "windows" else "M"
    steps = [
        ((0x41, ctrl_names), "ctrl+a"),
        ((0x42, ctrl_names + "A"), "ctrl+alt+b"),
        ((0x43, "S"), None),
        ((0x44, ctrl_names + "S"), "ctrl+shift+d"),
        ((KEY_SPACE, ctrl_names), None),
    ]
    last = "alt+f5"
    for (key, names), valid in steps:
        press(dialog, key, names)
        if valid is None:
            assert dialog.hotkey_reason.text() != ""
        else:
            last = valid
            assert dialog.hotkey_reason.text() == ""
        assert dialog.result_settings().hotkey == last
        assert shown(dialog, platform, last)


# ---------------------------------------------------------------- 4-A exhaustive: 16 modifier subsets x 48 keys

@pytest.mark.parametrize("platform", PLATFORMS)
def test_every_modifier_subset_with_every_allowed_key(tmp_path, qt_app, platform):
    dialog = make(tmp_path, platform, stored="alt+f12")
    value = "alt+f12"
    checked = 0
    for names in SUBSETS:
        for key_name, code in ALLOWED.items():
            before = value
            press(dialog, code, names)
            words = expected_words(platform, names)
            if words is None:
                value, reason = before, rules.REASON_WIN
            elif not set(words) & {"ctrl", "alt", "cmd"}:
                value, reason = before, rules.REASON_NEEDS_MODIFIER
            else:
                value, reason = "+".join(words + [key_name]), ""
            assert dialog.result_settings().hotkey == value, (platform, sorted(names), key_name)
            assert dialog.hotkey_box.text() == label_of(platform, value), (platform, sorted(names), key_name)
            assert dialog.hotkey_reason.text() == reason, (platform, sorted(names), key_name)
            checked += 1
    assert checked == 16 * 48


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("key", NOT_ALLOWED_KEYS)
def test_keys_outside_the_allowed_set_are_refused_with_every_modifier_subset(tmp_path, qt_app, platform, key):
    dialog = make(tmp_path, platform)
    for names in SUBSETS:
        press(dialog, key, names)
        if platform == "windows" and "M" in names:
            continue  # the Win key is refused first, with its own reason
        assert dialog.hotkey_reason.text() == rules.REASON_KEY_NOT_ALLOWED, sorted(names)
        assert dialog.result_settings().hotkey == "ctrl+alt+r"
        assert shown(dialog, platform, "ctrl+alt+r")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_keypad_digit_is_not_the_number_row_key(tmp_path, qt_app, platform):
    dialog = make(tmp_path, platform)
    ctrl = "C" if platform == "windows" else "M"
    press(dialog, 0x35, ctrl, keypad=True)
    assert dialog.hotkey_reason.text() == rules.REASON_KEY_NOT_ALLOWED
    assert dialog.result_settings().hotkey == "ctrl+alt+r"
    press(dialog, 0x35, ctrl)  # the same key on the number row works
    assert dialog.result_settings().hotkey == "ctrl+5"


# ---------------------------------------------------------------- 4-B shifted number-row symbols

@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("code, digit", sorted(SYMBOLS.items()))
def test_shifted_number_row_symbols_are_recorded_as_digits(tmp_path, qt_app, platform, code, digit):
    dialog = make(tmp_path, platform, stored="alt+f5")
    ctrl = "C"  # Windows: Ctrl. Mac: Command (Qt reports it as ControlModifier)
    press(dialog, code, ctrl + "S", text=chr(code))
    expected = "ctrl+shift+" + digit if platform == "windows" else "shift+cmd+" + digit
    assert dialog.result_settings().hotkey == expected
    assert dialog.hotkey_box.text() == label_of(platform, expected)
    assert dialog.hotkey_reason.text() == ""


def test_a_shifted_symbol_without_ctrl_or_command_is_refused(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    press(dialog, 0x24, "S", text="$")
    assert dialog.hotkey_reason.text() == rules.REASON_NEEDS_MODIFIER
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


# ---------------------------------------------------------------- 5 random sequences: always valid

@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("start", ["ctrl+alt+r", "alt+f5", "ctrl+shift+f5"])
@pytest.mark.parametrize("seed", [1, 2])
def test_random_key_sequences_always_leave_a_valid_shortcut(tmp_path, qt_app, platform, start, seed):
    rng = random.Random(f"{platform}-{start}-{seed}")
    dialog = make(tmp_path, platform, stored=start)
    pool = (
        list(ALLOWED.values()) * 3
        + list(SYMBOLS)
        + sorted(MODIFIER_KEYS)
        + sorted(BARE_LEAVERS)
        + NOT_ALLOWED_KEYS
    )
    value, reason_expected = start, False
    for step in range(2000):
        if rng.random() < 0.1:
            key = rng.randrange(0, 0x01FFFFFF)
        else:
            key = rng.choice(pool)
        names = {n for n in "CASM" if rng.random() < 0.4}
        auto = rng.random() < 0.15
        keypad = rng.random() < 0.08
        text = rng.choice(["", "r", "ｒ", "る", "あ"])
        made = press(dialog, key, names, text=text, auto=auto, keypad=keypad)
        verdict = oracle(platform, key, names, auto, keypad)
        if verdict[0] == "ok":
            value, reason_expected = verdict[1], False
        elif verdict[0] == "reject":
            reason_expected = True
        else:
            assert not made.isAccepted() or key in MODIFIER_KEYS or auto  # leavers are passed on, not swallowed
        got = dialog.result_settings().hotkey
        assert got == value, (step, key, sorted(names), auto, keypad)
        rules.parse(got, platform)  # valid on this platform, never raises
        assert (dialog.hotkey_reason.text() != "") == reason_expected, (step, key, sorted(names))
        assert dialog.hotkey_box.text() == label_of(platform, value)


# ---------------------------------------------------------------- 6 restore default

@pytest.mark.parametrize("platform, label", [("windows", "Ctrl+Alt+R"), ("mac", "⌃⌥R")])
def test_restore_default_twice(tmp_path, qt_app, platform, label):
    dialog = make(tmp_path, platform, stored="alt+f5", default="ctrl+alt+r")
    ctrl = "C" if platform == "windows" else "M"
    press(dialog, 0x41, ctrl)
    press(dialog, 0x42, "S")  # leaves a reason
    assert dialog.hotkey_reason.text() != ""
    dialog.hotkey_reset_button.click()
    assert dialog.hotkey_box.text() == label_of(platform, "ctrl+alt+r") == label
    assert dialog.hotkey_reason.text() == ""
    assert dialog.result_settings().hotkey == "ctrl+alt+r"
    dialog.hotkey_reset_button.click()
    assert dialog.hotkey_box.text() == label
    assert dialog.hotkey_reason.text() == ""
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


def test_restore_default_uses_the_default_given_to_the_dialog(tmp_path, qt_app):
    dialog = make(tmp_path, "windows", stored="alt+f5", default="ctrl+shift+f9")
    dialog.hotkey_reset_button.click()
    assert dialog.result_settings().hotkey == "ctrl+shift+f9"
    assert dialog.hotkey_box.text() == "Ctrl+Shift+F9"


# ---------------------------------------------------------------- 7 opening shows the stored value

@pytest.mark.parametrize(
    "platform, stored, label",
    [
        ("windows", "ctrl+shift+f5", "Ctrl+Shift+F5"),
        ("mac", "ctrl+shift+f5", "⌃⇧F5"),
        ("mac", "cmd+shift+4", "⇧⌘4"),
        ("windows", "alt+r", "Alt+R"),
    ],
)
def test_opening_shows_the_stored_shortcut_and_returns_it_untouched(tmp_path, qt_app, platform, stored, label):
    dialog = make(tmp_path, platform, stored=stored)
    assert dialog.hotkey_box.text() == label
    assert dialog.hotkey_reason.text() == ""
    assert dialog.result_settings().hotkey == rules.parse(stored, platform).serialize()


def test_a_stored_value_that_is_not_valid_here_shows_the_default(tmp_path, qt_app):
    dialog = make(tmp_path, "windows", stored="cmd+shift+4", default="ctrl+alt+r")
    assert dialog.hotkey_box.text() == "Ctrl+Alt+R"
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


def test_the_dialog_defaults_come_from_the_platform_package(tmp_path, qt_app):
    dialog = SettingsDialog(Settings(tmp_path, "large-v3", "model-a"), ["model-a"])
    assert dialog.hotkey_box.text() == rules.parse(DEFAULT_HOTKEY, HOTKEY_PLATFORM).display(HOTKEY_PLATFORM)
    assert dialog.result_settings().hotkey == rules.parse(DEFAULT_HOTKEY, HOTKEY_PLATFORM).serialize()


def test_result_settings_keeps_the_other_fields(tmp_path, qt_app):
    dialog = make(tmp_path, "windows", stored="alt+f5")
    result = dialog.result_settings()
    assert result == Settings(tmp_path / "out", "large-v3", "model-a", hotkey="alt+f5")


# ---------------------------------------------------------------- 8 cancel / save through the real window

def _record_ctrl_alt_t(dialog):
    ctrl = "M" if HOTKEY_PLATFORM == "mac" else "C"
    press(dialog, 0x54, ctrl + "A")


def test_cancel_changes_nothing_and_save_stores_the_new_shortcut(tmp_path, qt_app, monkeypatch):
    from coursescribe.ui import main_window

    made = Screen(tmp_path, qt_app)
    made.rig.controller.start()
    try:
        controller = made.rig.controller
        settings_file = tmp_path / "settings.json"
        first = controller.settings

        monkeypatch.setattr(main_window, "SettingsDialog", _driven(_record_ctrl_alt_t, 0))
        made.window.settings_button.click()
        assert controller.settings == first
        assert not settings_file.exists()

        monkeypatch.setattr(main_window, "SettingsDialog", _driven(_record_ctrl_alt_t, 1))
        made.window.settings_button.click()
        assert controller.settings.hotkey == "ctrl+alt+t"
        assert SettingsStore(settings_file, made.rig.settings.defaults).load().hotkey == "ctrl+alt+t"
    finally:
        made.close()


# ---------------------------------------------------------------- 9 auto-repeat does not flash

@pytest.mark.parametrize("platform", PLATFORMS)
def test_auto_repeat_changes_nothing(tmp_path, qt_app, platform):
    dialog = make(tmp_path, platform, stored="alt+f5")
    ctrl = "C" if platform == "windows" else "M"
    press(dialog, 0x41, ctrl)
    assert dialog.result_settings().hotkey == "ctrl+a"
    for _ in range(10):
        press(dialog, 0x41, ctrl, auto=True)
    assert dialog.result_settings().hotkey == "ctrl+a"
    assert dialog.hotkey_reason.text() == ""
    press(dialog, 0x42, "S", auto=True)  # an invalid key held down: no reason appears
    assert dialog.hotkey_reason.text() == ""
    assert dialog.result_settings().hotkey == "ctrl+a"
    press(dialog, 0x42, "S")  # the first real press does show it
    assert dialog.hotkey_reason.text() == rules.REASON_NEEDS_MODIFIER
    reason = dialog.hotkey_reason.text()
    press(dialog, 0x43, ctrl, auto=True)  # a valid repeat does not clear it or record
    assert dialog.hotkey_reason.text() == reason
    assert dialog.result_settings().hotkey == "ctrl+a"


# ---------------------------------------------------------------- 10 input method on: the key code decides

@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("text", ["", "ｒ", "る", "r", "R", "\x12"])
def test_the_key_code_decides_whatever_the_text_says(tmp_path, qt_app, platform, text):
    dialog = make(tmp_path, platform, stored="alt+f5")
    names = "CA" if platform == "windows" else "MA"
    press(dialog, 0x52, names, text=text)
    assert dialog.result_settings().hotkey == "ctrl+alt+r"
    assert dialog.hotkey_reason.text() == ""


# ---------------------------------------------------------------- 11 keyboard users are not trapped

@pytest.mark.parametrize("key, names", [(KEY_ESC, ""), (KEY_TAB, ""), (KEY_BACKTAB, "S"), (KEY_TAB, "S"), (KEY_RETURN, ""), (KEY_ENTER, "")])
def test_leaving_keys_are_neither_recorded_nor_shown_as_errors_and_are_passed_on(tmp_path, qt_app, key, names):
    dialog = make(tmp_path, "windows")
    made = press(dialog, key, names)
    assert not made.isAccepted()  # left for the dialog / focus chain to handle
    assert dialog.hotkey_reason.text() == ""
    assert dialog.hotkey_box.text() == "Ctrl+Alt+R"
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


@pytest.mark.parametrize("key", [KEY_ESC, KEY_TAB, KEY_RETURN])
def test_with_ctrl_the_same_keys_follow_the_invalid_key_rule(tmp_path, qt_app, key):
    dialog = make(tmp_path, "windows")
    made = press(dialog, key, "C")
    assert made.isAccepted()
    assert dialog.hotkey_reason.text() == rules.REASON_KEY_NOT_ALLOWED
    assert dialog.result_settings().hotkey == "ctrl+alt+r"


def test_escape_in_the_recorder_cancels_the_dialog(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    dialog.show()
    dialog.hotkey_box.setFocus()
    QTest.keyClick(dialog.hotkey_box, Qt.Key.Key_Escape)
    assert dialog.result() == 0
    assert not dialog.isVisible()


def test_tab_in_the_recorder_moves_the_focus_on(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    dialog.show()
    dialog.activateWindow()
    dialog.hotkey_box.setFocus()
    assert dialog.focusWidget() is dialog.hotkey_box
    QTest.keyClick(dialog.hotkey_box, Qt.Key.Key_Tab)
    assert dialog.focusWidget() is not dialog.hotkey_box
    assert dialog.hotkey_box.text() == "Ctrl+Alt+R"
    assert dialog.hotkey_reason.text() == ""
    dialog.close()


def test_enter_in_the_recorder_never_saves_an_invalid_value(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    press(dialog, 0x52, "S")  # invalid, reason shown
    press(dialog, KEY_RETURN, "")
    press(dialog, KEY_ENTER, "")
    press(dialog, KEY_RETURN, "C")
    assert dialog.result_settings().hotkey == "ctrl+alt+r"
    rules.parse(dialog.result_settings().hotkey, "windows")


def test_save_and_cancel_stay_reachable_by_keyboard(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    box = dialog.findChild(QDialogButtonBox)
    for kind in (QDialogButtonBox.StandardButton.Ok, QDialogButtonBox.StandardButton.Cancel):
        assert box.button(kind).focusPolicy() != Qt.FocusPolicy.NoFocus
        assert box.button(kind).isEnabled()
    assert dialog.hotkey_reset_button.focusPolicy() != Qt.FocusPolicy.NoFocus


# ---------------------------------------------------------------- 12 notes

def test_the_note_keeps_the_old_sentence_and_adds_the_shortcut_line(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    texts = [label.text() for label in dialog.findChildren(QLabel)]
    old, new = "修改后，从下一个任务开始生效。", "快捷键保存后立即生效。"
    holders = [t for t in texts if old in t and new in t]
    assert holders, texts
    assert holders[0].index(old) < holders[0].index(new)


def test_the_reason_label_wraps_and_the_recorder_is_read_only(tmp_path, qt_app):
    dialog = make(tmp_path, "windows")
    assert dialog.hotkey_reason.wordWrap()
    assert dialog.hotkey_box.isReadOnly()


# ---------------------------------------------------------------- the pure key -> text step

@pytest.mark.parametrize("platform", PLATFORMS)
def test_candidate_text_of_modifier_keys_is_none(platform):
    for key in MODIFIER_KEYS:
        assert key_capture.candidate_text(key, flags("C"), platform) is None


@pytest.mark.parametrize("platform", PLATFORMS)
def test_candidate_text_feeds_the_rules_and_an_unknown_key_is_refused_by_them(platform):
    text = key_capture.candidate_text(0x52, flags("CA"), platform)
    assert rules.parse(text, platform).key == "r"
    bad = key_capture.candidate_text(0x01000099, flags("C"), platform)
    with pytest.raises(rules.InvalidShortcut) as error:
        rules.parse(bad, platform)
    assert error.value.reason == rules.REASON_KEY_NOT_ALLOWED
