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
