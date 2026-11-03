"""Real system audio on the real desktop. These play a short quiet tone."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from coursescribe.platform import open_system_audio
from coursescribe.recorder import RecordingSession
from coursescribe.wavfile import repair_wav

pytestmark = pytest.mark.gui

CHILD = Path(__file__).with_name("record_child.py")
TONE_HZ = 1000


def _play_tone(seconds: float) -> None:
    import soundcard

    rate = 48000
    count = int(rate * seconds)
    tone = 0.15 * np.sin(np.arange(count) * 2 * np.pi * TONE_HZ / rate)
    soundcard.default_speaker().play(np.column_stack([tone, tone]), samplerate=rate)


def _tone_share(path: Path) -> float:
    """How much of the recording's energy sits at the test tone."""
    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).astype(np.float64)
    mono = data[::2]
    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    frequencies = np.fft.rfftfreq(len(mono), 1 / rate)
    band = (frequencies > TONE_HZ - 10) & (frequencies < TONE_HZ + 10)
    return float(spectrum[band].sum() / spectrum.sum())


def test_system_sound_is_captured_without_any_setup(tmp_path):
    target = tmp_path / "real.wav"
    session = RecordingSession(open_system_audio, target)
    session.start()
    player = threading.Thread(target=_play_tone, args=(2.0,))
    player.start()
    player.join()
    time.sleep(0.3)
    seconds = session.stop()
    assert 1.8 < seconds < 5
    assert _tone_share(target) > 0.02


def test_recording_survives_the_process_being_killed(tmp_path):
    target = tmp_path / "killed.wav"
    child = subprocess.Popen(
        [sys.executable, str(CHILD), str(target)], stdout=subprocess.PIPE, text=True, env=os.environ.copy()
    )
    try:
        assert child.stdout.readline().strip() == "recording"
        time.sleep(4.0)
    finally:
        child.kill()
    child.wait(10)

    seconds = repair_wav(target)
    assert seconds > 2.0, "at most the last few seconds may be lost"
    import av

    with av.open(str(target)) as media:
        assert sum(frame.samples for frame in media.decode(audio=0)) == pytest.approx(seconds * 48000, rel=0.01)


def _wait_until_readable(path: Path, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            with open(path, "rb"):
                return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_repair_waits_out_a_short_real_hold_and_reports_a_long_one(tmp_path):
    """A program that opens the recording without sharing (Windows only) must not make it count as empty."""
    from coursescribe.platform import FILES_CAN_BE_HELD
    from helpers import hold_without_sharing
    from test_recording import _crashed_wav

    briefly = _crashed_wav(tmp_path / "briefly.wav", 2.0)
    hold_without_sharing(briefly, 0.3)
    assert repair_wav(briefly) == pytest.approx(2.0)

    long_held = _crashed_wav(tmp_path / "long.wav", 2.0)
    before = long_held.read_bytes()
    hold_without_sharing(long_held, 3.0)
    if FILES_CAN_BE_HELD:
        started = time.monotonic()
        with pytest.raises(PermissionError):
            repair_wav(long_held)
        assert time.monotonic() - started < 2
        _wait_until_readable(long_held, 5.0)
        assert long_held.read_bytes() == before
    assert repair_wav(long_held) == pytest.approx(2.0)
    for path in (briefly, long_held):
        with wave.open(str(path), "rb") as stream:
            assert stream.getnframes() == 96000


def _press_hotkey() -> None:
    import ctypes

    user32 = ctypes.windll.user32
    ctrl, alt, r, key_up = 0x11, 0x12, 0x52, 0x0002
    for key in (ctrl, alt, r):
        user32.keybd_event(key, 0, 0, 0)
    for key in (r, alt, ctrl):
        user32.keybd_event(key, 0, key_up, 0)


def test_global_hotkey_fires_while_another_program_has_focus():
    from coursescribe.platform import create_global_hotkey

    pressed = threading.Event()
    hotkey = create_global_hotkey(pressed.set)
    assert hotkey.start(), "Ctrl+Alt+R is taken by another program"
    try:
        _press_hotkey()
        assert pressed.wait(3)
    finally:
        hotkey.stop()


def test_second_registration_reports_the_shortcut_is_taken():
    from coursescribe.platform import create_global_hotkey

    first = create_global_hotkey(lambda: None)
    assert first.start()
    try:
        second = create_global_hotkey(lambda: None)
        assert second.start() is False
    finally:
        first.stop()
