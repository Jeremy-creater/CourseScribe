"""Real Mac system sound: plays a short quiet tone with afplay and checks it is recorded."""

from __future__ import annotations

import subprocess
import sys
import wave

import numpy as np
import pytest

pytestmark = pytest.mark.gui

if sys.platform != "darwin":
    pytest.fail("These tests need a real Mac (system sound recording through Core Audio).", pytrace=False)


def _tone(path, seconds=2.0, hz=440, rate=44100):
    t = np.arange(int(rate * seconds))
    samples = (0.1 * 32767 * np.sin(2 * np.pi * hz * t / rate)).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def test_the_tone_the_mac_plays_is_in_the_recording(tmp_path):
    from coursescribe.platform import open_system_audio

    tone = tmp_path / "tone.wav"
    _tone(tone)
    source = open_system_audio()
    with source:
        player = subprocess.Popen(["afplay", str(tone)])
        blocks = [source.read(4800) for _ in range(20)]
        player.wait(timeout=10)
    sound = np.concatenate(blocks)
    assert sound.shape[1] == 2
    assert int(np.abs(sound).max()) > 1000
    spectrum = np.abs(np.fft.rfft(sound[:, 0].astype(float)))
    assert abs(np.argmax(spectrum) * 48000 / len(sound) - 440) < 20
