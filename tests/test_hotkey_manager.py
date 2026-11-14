"""A new recording shortcut takes effect at once (WI-5).

The registrar is a hand-written in-memory system with real state: one registration per key combination (a second
one is refused, as Carbon does), combinations "taken by another program", and a count of live handles. Nothing here
asserts that a mock was called; every check reads what is registered and what the user is shown. The real system
paths are the gui tests in test_mac_hotkey_desktop.py and test_recording_desktop.py.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

from coursescribe.controller import SETTINGS_NOT_SAVED
from coursescribe.hotkey_manager import ApplyResult, HotkeyManager, Outcome
from coursescribe.platform import HOTKEY_PLATFORM
from coursescribe.shortcut import parse
from coursescribe.ui import main_window
from coursescribe.ui.main_window import MainWindow
from coursescribe.ui.settings_dialog import SettingsDialog
from test_controller import Rig
from test_ui import _driven, qt_app  # noqa: F401  (qt_app is a fixture)

SRC = Path(__file__).resolve().parents[1] / "src" / "coursescribe"
DRAG = "也可以把文件拖到这里"


def sc(text: str):
    return parse(text, "mac")


R, T, Y, U = sc("ctrl+alt+r"), sc("ctrl+alt+t"), sc("ctrl+alt+y"), sc("ctrl+alt+u")


def label_of(shortcut) -> str:
    return shortcut.display("mac")


# ------------------------------------------------------------------------------------- the fake system


class FakeHandle:
    def __init__(self, system: "FakeSystem", shortcut, callback) -> None:
        self.system, self.shortcut, self.callback, self.active = system, shortcut, callback, False

    def start(self) -> bool:
        system = self.system
        system.starts += 1
        system.threads.append(threading.get_ident())
        system.attempts.append(self.shortcut)
        if self.shortcut in system.raising:
            raise OSError("the system refused to say")
        if self.shortcut in system.taken_elsewhere or self.shortcut in system.held:
            return False  # held here (Carbon refuses a duplicate in one process) or by another program
        system.held.add(self.shortcut)
        self.active = True
        return True

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.system.stops += 1
        self.system.held.discard(self.shortcut)
        if self.shortcut in self.system.steal_on_release:
            self.system.taken_elsewhere.add(self.shortcut)  # another program grabs it the moment it is free

    def press(self) -> None:
        assert self.active
        self.callback()


class FakeSystem:
    def __init__(self) -> None:
        self.held: set = set()
        self.taken_elsewhere: set = set()
        self.raising: set = set()
        self.steal_on_release: set = set()
        self.starts = 0
        self.stops = 0
        self.attempts: list = []
        self.threads: list[int] = []
        self.handles: list[FakeHandle] = []

    def create(self, shortcut, callback) -> FakeHandle:
        handle = FakeHandle(self, shortcut, callback)
        self.handles.append(handle)
        return handle

    @property
    def allocated_handles(self) -> int:
        return sum(1 for handle in self.handles if handle.active)

    def live_handle(self, shortcut):
        return next((h for h in self.handles if h.active and h.shortcut == shortcut), None)


class Seen:
    """What the observer was told: (label, note, and what was really live at that moment)."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str, object]] = []
        self.manager = None

    def __call__(self, label: str, note: str) -> None:
        self.entries.append((label, note, self.manager.live))
        expected = label_of(self.manager.live) if self.manager.live else ""
        assert label == expected, f"told {label!r} while {self.manager.live} was live"


def build(started: str | None = "ctrl+alt+r", **system_kwargs):
    system = FakeSystem()
    for name, value in system_kwargs.items():
        setattr(system, name, value)
    seen = Seen()
    pressed: list[int] = []
    manager = HotkeyManager(system.create, lambda: pressed.append(1), "mac", label_of, on_change=seen)
    seen.manager = manager
    if started:
        manager.apply(sc(started))
    return system, manager, seen, pressed


# ------------------------------------------------------------------------------------- 1 free combination


def test_a_free_combination_replaces_the_old_one():
    system, manager, seen, _ = build()
    old = system.live_handle(R)
    result = manager.apply(T)
    assert result.outcome is Outcome.APPLIED
    assert system.held == {T}
    assert old.active is False
    assert manager.live == T and manager.label == label_of(T) and result.label == label_of(T)
    assert seen.entries[-1][:2] == (label_of(T), "")


def test_the_registered_callback_is_the_one_given_to_the_manager():
    system, manager, _, pressed = build()
    system.live_handle(R).press()
    assert pressed == [1]


# ------------------------------------------------------------------------------------- 2 taken: old stays


def test_a_taken_combination_keeps_the_old_one_and_says_why():
    system, manager, seen, _ = build(taken_elsewhere={T})
    before = len(seen.entries)
    result = manager.apply(T)
    assert result.outcome is Outcome.KEPT_OLD
    assert label_of(T) in result.message and "被别的程序或系统占用" in result.message
    assert label_of(R) in result.message
    assert system.held == {R} and manager.live == R and manager.label == label_of(R)
    assert result.label == label_of(R)
    assert [e[0] for e in seen.entries[before:]] == [label_of(R)]  # never showed the new label


# ------------------------------------------------------------------------------------- 3 both fail, recover


def test_when_neither_can_be_registered_there_is_no_shortcut_until_a_free_one_is_applied():
    system, manager, seen, _ = build(taken_elsewhere={T}, steal_on_release={R})
    result = manager.apply(T)
    assert result.outcome is Outcome.NONE
    assert "现在没有快捷键" in result.message
    assert system.held == set() and manager.live is None and manager.label == ""
    assert seen.entries[-1][0] == ""
    assert system.allocated_handles == 0
    again = manager.apply(Y)
    assert again.outcome is Outcome.APPLIED
    assert system.held == {Y} and manager.label == label_of(Y)


# ------------------------------------------------------------------------------------- 4 same = no-op (5-A)


def test_applying_the_current_combination_does_nothing_at_all():
    system, manager, _, _ = build()
    before = system.live_handle(R)
    starts, stops = system.starts, system.stops
    for same in (R, parse("ctrl+alt+r", "mac"), parse("alt+ctrl+r", "mac"), parse(" CTRL + Alt+R ", "mac")):
        result = manager.apply(same)
        assert result.outcome is Outcome.APPLIED
        assert (system.starts, system.stops) == (starts, stops)
        assert system.live_handle(R) is before
    assert system.held == {R}


def test_applying_the_current_one_is_safe_when_another_program_would_grab_it_on_release():
    system, manager, _, _ = build(steal_on_release={R})
    manager.apply(R)
    manager.apply(R)
    assert system.held == {R} and manager.live == R


def test_applying_the_current_one_never_says_it_is_taken_even_though_the_system_refuses_duplicates():
    system, manager, _, _ = build()
    for _ in range(3):
        assert manager.apply(R).outcome is Outcome.APPLIED
    assert manager.apply(R).message == ""


# ------------------------------------------------------------------------------------- 5-C retry with none


def test_with_no_live_shortcut_the_same_combination_is_tried_again():
    system, manager, _, _ = build(started=None, taken_elsewhere={R})
    assert manager.apply(R).outcome is Outcome.NONE
    assert manager.live is None
    system.taken_elsewhere.clear()  # the other program closed
    result = manager.apply(R)
    assert result.outcome is Outcome.APPLIED
    assert system.held == {R} and manager.live == R


def test_a_start_up_refusal_gives_the_plain_message_and_no_shortcut():
    system, manager, seen, _ = build(started=None, taken_elsewhere={R})
    result = manager.apply(R)
    assert result.outcome is Outcome.NONE
    assert result.message == "快捷键被别的程序占用了，请用按钮"
    assert seen.entries[-1][:2] == ("", "快捷键被别的程序占用了，请用按钮")


# ------------------------------------------------------------------------------------- 5 raising registrar


def test_a_registrar_that_raises_counts_as_a_failure_and_the_old_one_stays():
    system, manager, seen, _ = build(raising={T})
    result = manager.apply(T)
    assert result.outcome is Outcome.KEPT_OLD
    assert system.held == {R} and manager.live == R
    assert label_of(T) in result.message


def test_raising_while_putting_the_old_one_back_ends_with_no_shortcut_and_no_leak():
    system, manager, seen, _ = build()
    system.raising = {T, R}
    result = manager.apply(T)
    assert result.outcome is Outcome.NONE
    assert system.held == set() and system.allocated_handles == 0 and manager.live is None
    assert seen.entries[-1][0] == ""
    system.raising = set()
    assert manager.apply(U).outcome is Outcome.APPLIED


def test_a_registrar_whose_factory_raises_is_a_failure_too():
    calls = []

    def create(shortcut, callback):
        calls.append(shortcut)
        if shortcut == T:
            raise OSError("no such library")
        return FakeSystem.create(system, shortcut, callback)

    system = FakeSystem()
    manager = HotkeyManager(create, lambda: None, "mac", label_of)
    manager.apply(R)
    result = manager.apply(T)
    assert result.outcome is Outcome.KEPT_OLD and system.held == {R}


# ------------------------------------------------------------------------------------- 6 random sequences


def test_random_sequences_keep_at_most_one_registration_and_an_honest_label():
    pool = [R, T, Y, U]
    rng = random.Random(20261114)
    for _ in range(2000):
        system = FakeSystem()
        seen = Seen()
        manager = HotkeyManager(system.create, lambda: None, "mac", label_of, on_change=seen)
        seen.manager = manager
        closed = False
        for _step in range(rng.randint(1, 12)):
            system.taken_elsewhere = {s for s in pool if rng.random() < 0.3}
            system.raising = {s for s in pool if rng.random() < 0.15}
            system.steal_on_release = {s for s in pool if rng.random() < 0.15}
            action = rng.random()
            old = manager.live
            if action < 0.08:
                manager.shutdown()
                closed = True
            elif action < 0.3 and old is not None:
                manager.apply(old)
                assert manager.live == old
            else:
                manager.apply(rng.choice(pool))
                if closed:
                    assert manager.live is None  # after shutdown, applying does nothing
            assert len(system.held) <= 1
            assert system.held == ({manager.live} if manager.live else set())
            assert system.allocated_handles == len(system.held)  # 5-E: checked at every step
            assert manager.label == (label_of(manager.live) if manager.live else "")
        manager.shutdown()
        assert system.held == set() and system.allocated_handles == 0 and manager.live is None


def test_a_failed_apply_with_a_restorable_old_one_leaves_the_old_one_live():
    rng = random.Random(7)
    pool = [R, T, Y, U]
    for _ in range(500):
        system = FakeSystem()
        manager = HotkeyManager(system.create, lambda: None, "mac", label_of)
        manager.apply(R)
        new = rng.choice([T, Y, U])
        system.taken_elsewhere = {new}
        system.raising = {s for s in pool if s not in (R, new) and rng.random() < 0.5}
        manager.apply(new)
        assert manager.live == R and system.held == {R}


# ------------------------------------------------------------------------------------- 7 shutdown / 5-E


def test_shutdown_stops_only_the_current_one_and_is_safe_to_repeat():
    system, manager, _, _ = build()
    manager.apply(T)
    manager.shutdown()
    manager.shutdown()
    assert system.held == set() and system.allocated_handles == 0 and manager.live is None
    assert system.stops == 2  # the first one when it was replaced, the second one now


def test_shutdown_before_anything_was_registered_is_harmless():
    system, manager, _, _ = build(started=None)
    manager.shutdown()
    assert system.starts == 0 and manager.live is None


def test_after_shutdown_applying_does_nothing_and_does_not_raise():
    system, manager, _, _ = build()
    manager.shutdown()
    starts = system.starts
    result = manager.apply(T)
    assert system.starts == starts and system.held == set() and manager.live is None
    assert isinstance(result, ApplyResult)


def test_a_stop_that_raises_does_not_leave_a_second_registration():
    system, manager, _, _ = build()
    handle = system.live_handle(R)
    real_stop = handle.stop

    def broken():
        real_stop()
        raise OSError("already gone")

    handle.stop = broken
    manager.apply(T)
    assert system.held == {T} and manager.live == T
    manager.shutdown()
    assert system.allocated_handles == 0


def test_no_handle_is_left_active_after_a_long_run_of_a_b_a_a():
    system, manager, _, _ = build()
    for shortcut in (T, R, R, T, T, Y, R):
        manager.apply(shortcut)
        assert system.allocated_handles == 1
    manager.shutdown()
    assert system.allocated_handles == 0


# ------------------------------------------------------------------------------------- text from the settings


def test_apply_text_reads_the_stored_text_and_leaves_things_alone_when_it_is_not_valid():
    system, manager, _, _ = build()
    assert manager.apply_text("ctrl+alt+t").outcome is Outcome.APPLIED
    assert system.held == {T}
    result = manager.apply_text("banana")
    assert result.outcome is Outcome.KEPT_OLD and system.held == {T}


# ------------------------------------------------------------------------------------- the window


def make_window(tmp_path, qt_app, system, **window_kwargs):
    """A real window over a real controller, with the manager wired as app.py wires it."""
    rig = Rig(tmp_path)
    holder = {}
    manager = HotkeyManager(
        system.create, lambda: holder["window"].hotkey_pressed.emit(), HOTKEY_PLATFORM,
        lambda s: s.display(HOTKEY_PLATFORM),
        on_change=lambda label, note: holder["window"].set_hotkey_hint(label, note),
    )
    window = MainWindow(
        rig.controller,
        hotkey_label="",
        choose_files=lambda: [],
        confirm=lambda question: True,
        open_folder=lambda folder: None,
        list_proofread_models=lambda: ["model-a"],
        on_settings_saved=lambda settings: manager.apply_saved(settings, rig.controller.update_settings),
    )
    holder["window"] = window
    manager.apply_text(rig.controller.settings.hotkey)
    rig.controller.start()
    window.show()
    return rig, window, manager


def close(rig, window, qt_app, manager):
    manager.shutdown()
    window.close()
    rig.controller._listeners.clear()
    qt_app.processEvents()
    import shiboken6

    shiboken6.delete(window)
    qt_app.processEvents()


def press_text(dialog, text: str, auto: bool = False) -> None:
    """Press the combination the way a keyboard on this system reports it (Qt swaps Ctrl and Command on a Mac)."""
    *mods, key = text.split("+")
    table = {
        "windows": {"ctrl": Qt.KeyboardModifier.ControlModifier, "alt": Qt.KeyboardModifier.AltModifier,
                    "shift": Qt.KeyboardModifier.ShiftModifier},
        "mac": {"ctrl": Qt.KeyboardModifier.MetaModifier, "alt": Qt.KeyboardModifier.AltModifier,
                "shift": Qt.KeyboardModifier.ShiftModifier, "cmd": Qt.KeyboardModifier.ControlModifier},
    }[HOTKEY_PLATFORM]
    flags = Qt.KeyboardModifier.NoModifier
    for mod in mods:
        flags = flags | table[mod]
    event = QKeyEvent(QEvent.Type.KeyPress, ord(key.upper()), flags, 0, 0, 0, "", auto, 1)
    dialog.hotkey_box.keyPressEvent(event)


def settings_file(rig) -> dict:
    return json.loads((rig.tmp_path / "settings.json").read_text(encoding="utf-8"))


def save_with(script):
    return _driven(script, 1)


def test_the_window_hint_changes_at_once_and_still_offers_dragging(tmp_path, qt_app):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        window.set_hotkey_hint("⌃⌥T")
        assert "⌃⌥T" in window.hotkey_label.text() and DRAG in window.hotkey_label.text()
        window.set_hotkey_hint("", "现在没有快捷键，请用按钮")
        assert "现在没有快捷键" in window.hotkey_label.text() and DRAG in window.hotkey_label.text()
        window.set_hotkey_hint("")
        assert "可以把文件拖到这里" in window.hotkey_label.text()
    finally:
        close(rig, window, qt_app, manager)


def test_the_start_up_refusal_shows_the_existing_text(tmp_path, qt_app):
    system = FakeSystem(); system.taken_elsewhere = {R}
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        assert window.hotkey_label.text() == "快捷键被别的程序占用了，请用按钮。也可以把文件拖到这里"
        assert manager.live is None
        assert window.record_button.isEnabled()
    finally:
        close(rig, window, qt_app, manager)


def test_saving_a_free_combination_applies_it_and_shows_it(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        old = system.live_handle(R)
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        assert system.held == {T} and old.active is False
        assert label_of(T) in window.hotkey_label.text() and DRAG in window.hotkey_label.text()
        assert settings_file(rig)["hotkey"] == "ctrl+alt+t"
        assert set(system.threads) == {threading.get_ident()}  # registered on the window's own (main) thread
    finally:
        close(rig, window, qt_app, manager)


def test_saving_a_taken_combination_keeps_the_old_one_and_names_the_new_one(tmp_path, qt_app, monkeypatch):
    system = FakeSystem(); system.taken_elsewhere = {T}
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        assert system.held == {R}
        text = window.hotkey_label.text()
        assert label_of(T) in text and "被别的程序或系统占用" in text and label_of(R) in text and DRAG in text
    finally:
        close(rig, window, qt_app, manager)


def test_a_taken_combination_is_not_left_in_the_settings_file(tmp_path, qt_app, monkeypatch):
    system = FakeSystem(); system.taken_elsewhere = {T}
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        assert settings_file(rig)["hotkey"] == "ctrl+alt+r"  # what is live is what is saved and shown next time
        assert rig.controller.settings.hotkey == "ctrl+alt+r"
    finally:
        close(rig, window, qt_app, manager)


def test_when_nothing_can_be_registered_the_chosen_combination_stays_saved(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        system.taken_elsewhere = {T, R}  # the old one can no longer be re-registered either
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        assert manager.live is None
        assert settings_file(rig)["hotkey"] == "ctrl+alt+t"
    finally:
        close(rig, window, qt_app, manager)


def test_saving_when_nothing_can_be_registered_says_so_and_the_buttons_still_work(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        system.taken_elsewhere = {T}
        system.steal_on_release = {R}
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        assert system.held == set()
        assert "没有快捷键" in window.hotkey_label.text() and DRAG in window.hotkey_label.text()
        assert window.record_button.isEnabled() and window.files_button.isEnabled()
        rig.wait(lambda: not rig.controller.state.recording)
        window.record_button.click()
        rig.wait(lambda: rig.controller.state.recording)
        window.record_button.click()
        rig.wait(lambda: not rig.controller.state.recording)
    finally:
        close(rig, window, qt_app, manager)


def test_the_hint_only_ever_shows_what_is_really_registered(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        seen = []
        original = window.set_hotkey_hint

        def spy(label, note=""):
            seen.append((label, manager.live))
            original(label, note)

        window.set_hotkey_hint = spy
        system.raising = {T}
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        assert seen and all(label == (label_of(live) if live else "") for label, live in seen)
        assert seen[-1][0] == label_of(R)
    finally:
        close(rig, window, qt_app, manager)


def test_an_invalid_key_after_a_valid_one_saves_the_valid_one(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        def script(dialog):
            press_text(dialog, "ctrl+alt+t")
            press_text(dialog, "shift+r")  # not allowed: the value stays
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(script))
        window.settings_button.click()
        assert settings_file(rig)["hotkey"] == "ctrl+alt+t"
        assert set(system.attempts) == {R, T} and system.held == {T}
        assert label_of(T) in window.hotkey_label.text()
    finally:
        close(rig, window, qt_app, manager)


def test_an_invalid_key_alone_leaves_the_stored_shortcut_and_does_not_re_register(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        handle = system.live_handle(R)
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "shift+r")))
        window.settings_button.click()
        assert settings_file(rig)["hotkey"] == "ctrl+alt+r"
        assert system.attempts == [R] and system.live_handle(R) is handle
    finally:
        close(rig, window, qt_app, manager)


def test_saving_other_settings_leaves_the_shortcut_alone_and_says_nothing_about_it(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        handle = system.live_handle(R)
        before = window.hotkey_label.text()
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: d.auto_notes_box.setChecked(True)))
        window.settings_button.click()
        assert system.live_handle(R) is handle and system.starts == 1 and system.stops == 0
        assert window.hotkey_label.text() == before and "占用" not in before
        assert settings_file(rig)["auto_notes"] is True
    finally:
        close(rig, window, qt_app, manager)


def test_when_the_settings_cannot_be_written_the_new_shortcut_still_works_this_session(tmp_path, qt_app, monkeypatch):
    from test_notes_job import DiskFault

    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        DiskFault(monkeypatch, tmp_path, OSError(28, "No space left on device"))
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        rig.wait(lambda: window.notice_label.isVisible())
        assert window.notice_label.text() == SETTINGS_NOT_SAVED.format(path=tmp_path / "settings.json")
        assert system.held == {T}
        assert rig.controller.settings.hotkey == "ctrl+alt+t"
        assert label_of(T) in window.hotkey_label.text()
    finally:
        close(rig, window, qt_app, manager)


def test_saving_a_new_shortcut_during_a_recording_does_not_interrupt_it(tmp_path, qt_app, monkeypatch):
    system = FakeSystem()
    rig, window, manager = make_window(tmp_path, qt_app, system)
    try:
        window.record_button.click()
        rig.wait(lambda: rig.controller.state.recording)
        monkeypatch.setattr(main_window, "SettingsDialog", save_with(lambda d: press_text(d, "ctrl+alt+t")))
        window.settings_button.click()
        assert rig.controller.state.recording
        system.live_handle(T).press()  # the new combination stops it
        rig.wait(lambda: not rig.controller.state.recording)
        system.live_handle(T).press()  # and starts it again
        rig.wait(lambda: rig.controller.state.recording)
        system.live_handle(T).press()
        rig.wait(lambda: not rig.controller.state.recording)
    finally:
        close(rig, window, qt_app, manager)


# ------------------------------------------------------------------------------------- structure


def test_the_wiring_keeps_logic_out_of_the_window_and_the_platform_check_out_of_the_app():
    app = (SRC / "ui" / "app.py").read_text(encoding="utf-8")
    assert "hotkey_label.setText" not in app
    assert "sys.platform" not in app
    assert "HotkeyManager" in app and "shutdown()" in app
    assert "主线程" in app  # the call site says why it must stay on the main thread
    for name in ("main_window.py", "settings_dialog.py"):
        text = (SRC / "ui" / name).read_text(encoding="utf-8")
        assert "hotkey_manager" not in text and "create_global_hotkey" not in text


def test_the_manager_is_pure_it_imports_no_window_system_or_platform_code():
    import subprocess
    import sys

    code = (
        "import sys; import coursescribe.hotkey_manager;"
        "bad=[m for m in sys.modules if m.startswith('PySide6') or m.startswith('coursescribe.platform')"
        " or m=='coursescribe.settings'];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=SRC.parent,
                         env={**os.environ, "PYTHONPATH": str(SRC.parent)}, check=True).stdout
    assert out.strip() == "[]"
