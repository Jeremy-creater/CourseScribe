"""The real Carbon global shortcut on a real Mac (registering it changes nothing on the system)."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.gui

if sys.platform != "darwin":
    pytest.fail("These tests need a real Mac (Carbon global shortcuts).", pytrace=False)

from coursescribe.platform import macos


def test_the_shortcut_registers_and_releases():
    hotkey = macos.create_global_hotkey(lambda: None)
    try:
        assert hotkey.start() is True
    finally:
        hotkey.stop()
    hotkey.stop()  # stopping twice is harmless


def test_a_second_registration_of_the_same_keys_is_refused_until_the_first_is_released():
    first = macos.create_global_hotkey(lambda: None)
    second = macos.create_global_hotkey(lambda: None)
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
    hotkey = macos.create_global_hotkey(lambda: called.append(1))
    try:
        assert hotkey.start()
        hotkey._on_pressed()  # what the Carbon handler calls
    finally:
        hotkey.stop()
    assert called == [1]


def test_a_callback_that_raises_does_not_escape_into_the_system_handler():
    hotkey = macos.create_global_hotkey(lambda: 1 / 0)
    try:
        assert hotkey.start()
        assert hotkey._on_pressed() is None
    finally:
        hotkey.stop()
