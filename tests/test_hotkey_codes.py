"""Native key codes for the recording shortcut, and registering it through injected system libraries.

The expected tables below are written out by hand from the Win32 headers (WinUser.h:
MOD_*, VK_*) and Carbon's Events.h / HIToolbox (cmdKey ..., kVK_ANSI_*, kVK_F*), not derived
from the code under test. The fakes keep real state and refuse duplicates by the numbers
they are given, so a wrong code or a hard-coded key shows up as a wrong number.
"""

from __future__ import annotations

import ctypes
import itertools
import queue
import re
import threading
from pathlib import Path

import pytest

from coursescribe.platform import hotkey_codes
from coursescribe.platform.hotkey_codes import UnsupportedShortcut, mac_codes, windows_codes
from coursescribe.platform.mac_hotkey import MacHotkey
from coursescribe.platform.win_hotkey import GlobalHotkey
from coursescribe.shortcut import MODIFIER_ORDER, Shortcut, parse

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
FUNCTION_KEYS = [f"f{n}" for n in range(1, 13)]
ALL_KEYS = list(LETTERS) + list(DIGITS) + FUNCTION_KEYS

# ---- Win32 ---------------------------------------------------------------------------------
WIN_MODIFIER = {"alt": 0x0001, "ctrl": 0x0002, "shift": 0x0004}
WIN_NOREPEAT = 0x4000
WIN_KEYS = {
    **{c: 0x41 + i for i, c in enumerate(LETTERS)},  # VK_A .. VK_Z
    **{d: 0x30 + int(d) for d in DIGITS},  # '0' .. '9'
    **{f"f{n}": 0x70 + n - 1 for n in range(1, 13)},  # VK_F1 .. VK_F12
}

# ---- Carbon --------------------------------------------------------------------------------
MAC_MODIFIER = {"cmd": 0x0100, "shift": 0x0200, "alt": 0x0800, "ctrl": 0x1000}  # cmdKey shiftKey optionKey controlKey
MAC_KEYS = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07,
    "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10,
    "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "9": 0x19,
    "7": 0x1A, "8": 0x1C, "0": 0x1D, "o": 0x1F, "u": 0x20, "i": 0x22, "p": 0x23, "l": 0x25,
    "j": 0x26, "k": 0x28, "n": 0x2D, "m": 0x2E,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61,
    "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
}


def _subsets():
    for size in range(len(MODIFIER_ORDER) + 1):
        for combo in itertools.combinations(MODIFIER_ORDER, size):
            yield frozenset(combo)


ALL_SHORTCUTS = [Shortcut(mods, key) for mods in _subsets() for key in ALL_KEYS]


def test_the_expected_tables_cover_every_allowed_key():
    assert set(WIN_KEYS) == set(ALL_KEYS) == set(MAC_KEYS)
    assert len(ALL_SHORTCUTS) == 16 * 48
    assert len(set(MAC_KEYS.values())) == 48 and len(set(WIN_KEYS.values())) == 48


# ---- 1. Windows table -----------------------------------------------------------------------
def test_windows_codes_match_the_win32_table_for_every_allowed_combination():
    checked = 0
    for shortcut in ALL_SHORTCUTS:
        if "cmd" in shortcut.modifiers:
            continue
        bits = sum(WIN_MODIFIER[m] for m in shortcut.modifiers)
        assert windows_codes(shortcut) == (bits, WIN_KEYS[shortcut.key]), shortcut.serialize()
        checked += 1
    assert checked == 8 * 48


# ---- 2. Mac table ---------------------------------------------------------------------------
def test_mac_codes_match_the_carbon_table_for_every_allowed_combination():
    for shortcut in ALL_SHORTCUTS:
        bits = sum(MAC_MODIFIER[m] for m in shortcut.modifiers)
        assert mac_codes(shortcut) == (bits, MAC_KEYS[shortcut.key]), shortcut.serialize()


def test_the_default_shortcut_maps_to_the_numbers_that_were_hard_coded_before():
    default = parse("ctrl+alt+r", "mac")
    assert mac_codes(default) == (0x1000 | 0x0800, 0x0F)
    assert windows_codes(default) == (0x0002 | 0x0001, 0x52)


# ---- 3. injective and total -----------------------------------------------------------------
def test_mappings_never_raise_for_valid_combinations_and_never_collide():
    windows = {windows_codes(s) for s in ALL_SHORTCUTS if "cmd" not in s.modifiers}
    mac = {mac_codes(s) for s in ALL_SHORTCUTS}
    assert len(windows) == 8 * 48
    assert len(mac) == 16 * 48


# ---- 4. unsupported -> a named error that names the thing -------------------------------------
def test_windows_has_no_command_key_and_says_so():
    with pytest.raises(UnsupportedShortcut) as info:
        windows_codes(Shortcut(frozenset({"cmd", "ctrl"}), "r"))
    assert "cmd" in str(info.value)


@pytest.mark.parametrize("key", ["space", "f13", ";", "é", "", "R", "kp_1"])
def test_keys_outside_the_table_raise_a_named_error_on_both_systems(key):
    shortcut = Shortcut(frozenset({"ctrl"}), key)  # built around the validation on purpose
    for convert in (windows_codes, mac_codes):
        with pytest.raises(UnsupportedShortcut) as info:
            convert(shortcut)
        assert repr(key) in str(info.value) or key in str(info.value)


def test_an_unknown_modifier_raises_instead_of_being_dropped():
    odd = Shortcut(frozenset({"ctrl", "meta"}), "r")
    for convert in (windows_codes, mac_codes):
        with pytest.raises(UnsupportedShortcut) as info:
            convert(odd)
        assert "meta" in str(info.value)


def test_the_named_error_is_a_value_error_so_callers_can_catch_it_broadly():
    assert issubclass(UnsupportedShortcut, ValueError)


# ---- 5. default and labels --------------------------------------------------------------------
def test_the_default_is_valid_and_has_the_right_label_on_each_system():
    from coursescribe import platform as native

    assert parse(native.DEFAULT_HOTKEY, native.HOTKEY_PLATFORM).serialize() == "ctrl+alt+r"
    shortcut = parse("ctrl+alt+r", native.HOTKEY_PLATFORM)
    assert native.hotkey_label(shortcut) == native.HOTKEY_LABEL
    assert parse("ctrl+alt+r", "windows").display("windows") == "Ctrl+Alt+R"
    assert parse("ctrl+alt+r", "mac").display("mac") == "⌃⌥R"
    expected = {"windows": "Ctrl+Alt+R", "mac": "⌃⌥R"}[native.HOTKEY_PLATFORM]
    assert native.HOTKEY_LABEL == expected


def test_the_label_follows_the_shortcut_not_a_fixed_text():
    from coursescribe import platform as native

    other = parse("ctrl+shift+f5", native.HOTKEY_PLATFORM)
    assert native.hotkey_label(other) == other.display(native.HOTKEY_PLATFORM)
    assert native.hotkey_label(other) != native.HOTKEY_LABEL


def test_the_platform_package_exports_the_new_names():
    from coursescribe import platform as native

    for name in ("DEFAULT_HOTKEY", "HOTKEY_PLATFORM", "HOTKEY_LABEL", "hotkey_label", "create_global_hotkey"):
        assert name in native.__all__ and hasattr(native, name)


# ---- 6. no operating-system checks outside platform/ -------------------------------------------
def test_nothing_outside_the_platform_folder_asks_which_system_it_runs_on():
    root = Path(__file__).resolve().parents[1] / "src" / "coursescribe"
    pattern = re.compile(r"sys\.platform|platform\.system|os\.name")
    hits = []
    files = 0
    for path in root.rglob("*.py"):
        if "platform" in path.relative_to(root).parts[:-1]:
            continue
        files += 1
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(root)}:{number}")
    assert files > 10  # the scan really looked at the code
    assert hits == []


def test_the_pure_tables_import_without_windows_or_ctypes_loaders():
    source = Path(hotkey_codes.__file__).read_text(encoding="utf-8")
    assert "winreg" not in source and "windll" not in source and "CDLL" not in source


# ---- fake Carbon ------------------------------------------------------------------------------
class FakeCarbon:
    """Remembers installed handlers and registered (keycode, modifiers); refuses duplicates by those numbers."""

    def __init__(self) -> None:
        self.hotkeys: dict[int, tuple[int, int]] = {}
        self.handlers: dict[int, object] = {}
        self.attempts: list[tuple[int, int]] = []
        self._next = 100
        self.refuse_install = False

    def GetApplicationEventTarget(self):
        return 1

    def InstallEventHandler(self, target, handler, count, spec, data, ref):
        if self.refuse_install:
            return -50
        self._next += 1
        ref._obj.value = self._next
        self.handlers[self._next] = handler
        return 0

    def RegisterEventHotKey(self, keycode, modifiers, hotkey_id, target, options, ref):
        self.attempts.append((keycode, modifiers))
        if (keycode, modifiers) in self.hotkeys.values():
            return -9878  # eventHotKeyExistsErr
        self._next += 1
        ref._obj.value = self._next
        self.hotkeys[self._next] = (keycode, modifiers)
        return 0

    def UnregisterEventHotKey(self, ref):
        self.hotkeys.pop(ref.value, None)
        return 0

    def RemoveEventHandler(self, ref):
        self.handlers.pop(ref.value, None)
        return 0

    def press(self, keycode, modifiers) -> bool:
        """What the system does on a key press: only a registered combination reaches a handler."""
        if (keycode, modifiers) not in self.hotkeys.values():
            return False
        for handler in list(self.handlers.values()):
            handler(None, None, None)
        return True


def _mac(text: str, carbon: FakeCarbon, callback=lambda: None) -> MacHotkey:
    return MacHotkey(parse(text, "mac"), callback, load_carbon=lambda: carbon)


# ---- 3-A / 3-B on the Mac side ---------------------------------------------------------------
@pytest.mark.parametrize("key", ALL_KEYS)
def test_the_mac_registration_uses_the_keycode_of_the_given_shortcut(key):
    carbon = FakeCarbon()
    hotkey = _mac(f"ctrl+alt+{key}", carbon)
    assert hotkey.start() is True
    assert carbon.attempts == [(MAC_KEYS[key], 0x1000 | 0x0800)]
    hotkey.stop()
    assert carbon.hotkeys == {} and carbon.handlers == {}


def test_combinations_that_differ_only_in_modifiers_or_key_can_coexist():
    carbon = FakeCarbon()
    first = _mac("ctrl+alt+shift+t", carbon)
    second = _mac("ctrl+alt+shift+y", carbon)
    third = _mac("ctrl+alt+t", carbon)
    assert [first.start(), second.start(), third.start()] == [True, True, True]
    # control 0x1000 + option 0x0800 + shift 0x0200 = 0x1A00; without shift 0x1800
    assert sorted(carbon.attempts) == [(0x10, 0x1A00), (0x11, 0x1800), (0x11, 0x1A00)]
    assert sorted(carbon.hotkeys.values()) == [(0x10, 0x1A00), (0x11, 0x1800), (0x11, 0x1A00)]


def test_the_same_numbers_twice_are_refused_and_leave_the_first_untouched():
    carbon = FakeCarbon()
    pressed = []
    first = _mac("ctrl+alt+t", carbon, lambda: pressed.append("first"))
    second = _mac("alt+ctrl+t", carbon, lambda: pressed.append("second"))  # same keys, other spelling
    assert first.start() is True
    assert second.start() is False
    assert len(carbon.hotkeys) == 1 and len(carbon.handlers) == 1  # the failed one left nothing behind
    second.stop()  # harmless
    assert carbon.press(MAC_KEYS["t"], 0x1800) is True
    assert pressed == ["first"]
    first.stop()
    assert carbon.hotkeys == {} and carbon.handlers == {}
    third = _mac("ctrl+alt+t", carbon)
    assert third.start() is True


def test_start_and_stop_twice_are_harmless_and_the_key_can_be_taken_again():
    carbon = FakeCarbon()
    hotkey = _mac("ctrl+alt+t", carbon)
    assert hotkey.start() is True
    assert hotkey.start() is True
    assert len(carbon.hotkeys) == 1
    hotkey.stop()
    hotkey.stop()
    assert carbon.hotkeys == {} and carbon.handlers == {}
    assert _mac("ctrl+alt+t", carbon).start() is True


def test_after_stop_a_key_press_no_longer_calls_back():
    carbon = FakeCarbon()
    pressed = []
    hotkey = _mac("ctrl+alt+r", carbon, lambda: pressed.append(1))
    assert hotkey.start() is True
    assert carbon.press(0x0F, 0x1800) is True
    hotkey.stop()
    assert carbon.press(0x0F, 0x1800) is False
    assert pressed == [1]


def test_a_failed_registration_removes_the_event_handler_that_was_installed():
    carbon = FakeCarbon()
    carbon.hotkeys[1] = (0x0F, 0x1800)  # somebody else holds it
    hotkey = _mac("ctrl+alt+r", carbon)
    assert hotkey.start() is False
    assert carbon.handlers == {}


def test_a_refused_event_handler_means_not_started_and_nothing_registered():
    carbon = FakeCarbon()
    carbon.refuse_install = True
    assert _mac("ctrl+alt+r", carbon).start() is False
    assert carbon.hotkeys == {}


def test_a_callback_that_raises_does_not_reach_the_system_handler():
    carbon = FakeCarbon()
    hotkey = _mac("ctrl+alt+r", carbon, lambda: 1 / 0)
    assert hotkey.start() is True
    assert carbon.press(0x0F, 0x1800) is True  # no exception came out of the handler
    hotkey.stop()


def test_a_missing_carbon_library_is_a_clear_false_not_a_crash():
    def missing():
        raise OSError("no such framework")

    hotkey = MacHotkey(parse("ctrl+alt+r", "mac"), lambda: None, load_carbon=missing)
    assert hotkey.start() is False
    hotkey.stop()  # harmless


def test_a_shortcut_with_no_mac_code_is_refused_without_touching_the_system():
    carbon = FakeCarbon()
    hotkey = MacHotkey(Shortcut(frozenset({"ctrl"}), "space"), lambda: None, load_carbon=lambda: carbon)
    assert hotkey.start() is False
    assert carbon.attempts == [] and carbon.hotkeys == {} and carbon.handlers == {}


# ---- fake user32 ------------------------------------------------------------------------------
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012


class FakeUser32:
    """Registration is by (modifiers without MOD_NOREPEAT, virtual key); GetMessageW waits for a message."""

    def __init__(self) -> None:
        self.hotkeys: dict[int, tuple[int, int]] = {}
        self.attempts: list[tuple[int, int]] = []  # exactly what RegisterHotKey was given
        self.messages: queue.Queue[int] = queue.Queue()

    def RegisterHotKey(self, hwnd, hotkey_id, modifiers, vk):
        self.attempts.append((modifiers, vk))
        if (modifiers & ~WIN_NOREPEAT, vk) in [(m & ~WIN_NOREPEAT, v) for m, v in self.hotkeys.values()]:
            return 0
        self.hotkeys[hotkey_id] = (modifiers, vk)
        return 1

    def UnregisterHotKey(self, hwnd, hotkey_id):
        self.hotkeys.pop(hotkey_id, None)
        return 1

    def GetMessageW(self, ref, hwnd, low, high):
        message = self.messages.get(timeout=5)
        ref._obj.message = message
        return 0 if message == WM_QUIT else 1

    def PostThreadMessageW(self, thread_id, message, wparam, lparam):
        self.messages.put(message)
        return 1

    def press(self, modifiers, vk) -> bool:
        if (modifiers, vk) not in [(m & ~WIN_NOREPEAT, v) for m, v in self.hotkeys.values()]:
            return False
        self.messages.put(WM_HOTKEY)
        return True


class FakeKernel32:
    def GetCurrentThreadId(self):
        return threading.get_ident() & 0xFFFFFFFF


def _windows(text: str, user32: FakeUser32, callback=lambda: None) -> GlobalHotkey:
    return GlobalHotkey(parse(text, "windows"), callback, user32=user32, kernel32=FakeKernel32())


@pytest.mark.parametrize("key", ALL_KEYS)
def test_the_windows_registration_uses_the_virtual_key_of_the_given_shortcut(key):
    user32 = FakeUser32()
    hotkey = _windows(f"ctrl+alt+{key}", user32)
    assert hotkey.start() is True
    assert user32.attempts == [(0x0002 | 0x0001 | WIN_NOREPEAT, WIN_KEYS[key])]
    hotkey.stop()
    assert user32.hotkeys == {}


def test_windows_combinations_that_differ_in_modifiers_or_key_coexist():
    user32 = FakeUser32()
    first = _windows("ctrl+alt+shift+t", user32)
    second = _windows("ctrl+alt+shift+y", user32)
    third = _windows("ctrl+alt+t", user32)
    try:
        assert [first.start(), second.start(), third.start()] == [True, True, True]
    finally:
        for hotkey in (first, second, third):
            hotkey.stop()
    assert sorted(user32.attempts) == sorted([
        (0x0007 | WIN_NOREPEAT, 0x54), (0x0007 | WIN_NOREPEAT, 0x59), (0x0003 | WIN_NOREPEAT, 0x54),
    ])


def test_windows_same_combination_is_refused_leaves_nothing_and_can_be_retaken():
    user32 = FakeUser32()
    pressed = []
    first = _windows("ctrl+alt+t", user32, lambda: pressed.append("first"))
    second = _windows("alt+ctrl+t", user32, lambda: pressed.append("second"))
    assert first.start() is True
    assert second.start() is False
    second.stop()  # harmless
    assert len(user32.hotkeys) == 1
    assert user32.press(0x0003, 0x54) is True
    first.stop()
    assert pressed == ["first"]
    assert user32.hotkeys == {}
    third = _windows("ctrl+alt+t", user32)
    try:
        assert third.start() is True
    finally:
        third.stop()


def test_windows_start_and_stop_twice_are_harmless():
    user32 = FakeUser32()
    hotkey = _windows("ctrl+alt+t", user32)
    assert hotkey.start() is True
    assert hotkey.start() is True
    assert len(user32.attempts) == 1
    hotkey.stop()
    hotkey.stop()
    assert user32.hotkeys == {}


def test_windows_a_callback_that_raises_is_contained_and_later_presses_still_work():
    user32 = FakeUser32()
    done = threading.Event()
    calls = []

    def callback():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        done.set()

    hotkey = _windows("ctrl+alt+t", user32, callback)
    assert hotkey.start() is True
    try:
        user32.press(0x0003, 0x54)
        user32.press(0x0003, 0x54)
        assert done.wait(3)
    finally:
        hotkey.stop()


def test_windows_has_no_command_key_so_a_command_shortcut_does_not_register():
    user32 = FakeUser32()
    hotkey = GlobalHotkey(Shortcut(frozenset({"cmd"}), "r"), lambda: None, user32=user32, kernel32=FakeKernel32())
    assert hotkey.start() is False
    assert user32.attempts == []
    hotkey.stop()


def test_the_ctypes_byref_trick_the_fakes_rely_on_holds():
    """If this ever failed, the fakes would be reading the wrong object, not the code under test."""
    value = ctypes.c_void_p()
    assert ctypes.byref(value)._obj is value
