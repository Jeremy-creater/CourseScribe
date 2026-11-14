"""The real Carbon global shortcut on a real Mac (registering it changes nothing on the system)."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.gui

if sys.platform != "darwin":
    pytest.fail("These tests need a real Mac (Carbon global shortcuts).", pytrace=False)

from coursescribe.platform import macos
from coursescribe.shortcut import parse

DEFAULT = parse("ctrl+alt+r", "mac")
OTHER = parse("ctrl+alt+shift+t", "mac")


def test_the_shortcut_registers_and_releases():
    hotkey = macos.create_global_hotkey(DEFAULT, lambda: None)
    try:
        assert hotkey.start() is True
    finally:
        hotkey.stop()
    hotkey.stop()  # stopping twice is harmless


def test_a_second_registration_of_the_same_keys_is_refused_until_the_first_is_released():
    first = macos.create_global_hotkey(DEFAULT, lambda: None)
    second = macos.create_global_hotkey(DEFAULT, lambda: None)
    try:
        assert first.start() is True
        assert second.start() is False
        first.stop()
        assert second.start() is True
    finally:
        first.stop()
        second.stop()


def test_the_label_names_the_keys():
    assert macos.HOTKEY_LABEL == "⌃⌥R"


def test_the_pressed_callback_runs_when_the_system_reports_the_shortcut():
    called = []
    hotkey = macos.create_global_hotkey(DEFAULT, lambda: called.append(1))
    try:
        assert hotkey.start()
        hotkey._on_pressed()  # what the Carbon handler calls
    finally:
        hotkey.stop()
    assert called == [1]


def test_a_callback_that_raises_does_not_escape_into_the_system_handler():
    hotkey = macos.create_global_hotkey(DEFAULT, lambda: 1 / 0)
    try:
        assert hotkey.start()
        assert hotkey._on_pressed() is None
    finally:
        hotkey.stop()


# ---- a shortcut other than the default (matrix cases 7-10; nothing is pressed) ------------------


def test_a_non_default_shortcut_registers_releases_and_registers_again():
    hotkey = macos.create_global_hotkey(OTHER, lambda: None)
    try:
        assert hotkey.start() is True
        hotkey.stop()
        assert hotkey.start() is True
        hotkey.stop()
        again = macos.create_global_hotkey(OTHER, lambda: None)
        try:
            assert again.start() is True
        finally:
            again.stop()
    finally:
        hotkey.stop()


def test_shortcuts_that_differ_in_one_modifier_or_key_do_not_collide():
    shortcuts = [parse(text, "mac") for text in ("ctrl+alt+shift+t", "ctrl+alt+shift+y", "ctrl+alt+t")]
    hotkeys = [macos.create_global_hotkey(shortcut, lambda: None) for shortcut in shortcuts]
    try:
        assert [hotkey.start() for hotkey in hotkeys] == [True, True, True]
    finally:
        for hotkey in hotkeys:
            hotkey.stop()


def test_a_second_instance_of_a_non_default_shortcut_is_refused_until_the_first_stops():
    first = macos.create_global_hotkey(OTHER, lambda: None)
    second = macos.create_global_hotkey(OTHER, lambda: None)
    third = macos.create_global_hotkey(OTHER, lambda: None)
    try:
        assert first.start() is True
        assert second.start() is False
        first.stop()
        assert third.start() is True
    finally:
        for hotkey in (first, second, third):
            hotkey.stop()


def test_stopping_twice_is_harmless_and_the_shortcut_can_be_taken_again():
    hotkey = macos.create_global_hotkey(OTHER, lambda: None)
    assert hotkey.start() is True
    hotkey.stop()
    hotkey.stop()
    again = macos.create_global_hotkey(OTHER, lambda: None)
    try:
        assert again.start() is True
    finally:
        again.stop()


def test_a_refused_registration_leaves_the_first_one_and_the_system_intact():
    a = macos.create_global_hotkey(OTHER, lambda: None)
    b = macos.create_global_hotkey(OTHER, lambda: None)
    c = macos.create_global_hotkey(parse("ctrl+alt+shift+y", "mac"), lambda: None)
    try:
        assert a.start() is True
        assert b.start() is False
        b.stop()
        a.stop()
        assert c.start() is True  # the failed one did not tear down a shared handler
        again = macos.create_global_hotkey(OTHER, lambda: None)
        try:
            assert again.start() is True
        finally:
            again.stop()
    finally:
        for hotkey in (a, b, c):
            hotkey.stop()


def test_the_label_follows_the_shortcut_on_the_real_system():
    assert macos.hotkey_label(DEFAULT) == "⌃⌥R"
    assert macos.hotkey_label(OTHER) == "⌃⌥⇧T"
