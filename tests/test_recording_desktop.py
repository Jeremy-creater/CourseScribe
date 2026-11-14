"""Real system audio on the real desktop. These play a short quiet tone."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import platformdirs
import pytest

from coursescribe.platform import open_system_audio
from coursescribe.recorder import _SWITCHED, RecordingSession
from coursescribe.wavfile import repair_wav

pytestmark = pytest.mark.gui

CHILD = Path(__file__).with_name("record_child.py")
SET_DEFAULT_OUTPUT = Path(__file__).with_name("set_default_output.py")
TONE_HZ = 1000
# Where the original default output devices are kept while a test has changed them.
RESTORE_MARKER = Path(platformdirs.user_cache_dir("CourseScribe", appauthor=False)) / "tests" / "default-output.json"
_VIRTUAL_DEVICE_WORDS = ("broadcast", "virtual", "vb-audio", "voicemeeter", "steam streaming")


def _play_tone(seconds: float, hz: float = TONE_HZ) -> None:
    import soundcard

    rate = 48000
    count = int(rate * seconds)
    tone = 0.15 * np.sin(np.arange(count) * 2 * np.pi * hz / rate)
    soundcard.default_speaker().play(np.column_stack([tone, tone]), samplerate=rate)


def _tone_share(path: Path, hz: float = TONE_HZ, part: slice = slice(None)) -> float:
    """How much of the recording's energy (or of ``part`` of its frames) sits at the test tone."""
    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).astype(np.float64)
    mono = data[::2][part]
    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    frequencies = np.fft.rfftfreq(len(mono), 1 / rate)
    band = (frequencies > hz - 10) & (frequencies < hz + 10)
    return float(spectrum[band].sum() / spectrum.sum())


def _default_outputs(*args: str) -> dict[str, str]:
    """Runs the helper in its own process; returns the default output ID for each role afterwards."""
    done = subprocess.run(
        [sys.executable, str(SET_DEFAULT_OUTPUT), *args], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, f"set_default_output {args} failed: {done.stderr}"
    return json.loads(done.stdout)


def _put_back(saved: dict[str, str]) -> None:
    """Restores every role it can, then deletes the marker. If any role fails, keeps the marker and says what to do."""
    problems = []
    for role, device_id in saved.items():
        try:
            _default_outputs("--set", device_id, "--role", role)
        except Exception as exc:  # keep going: the other roles can still be put back
            problems.append(f"{role}: {exc}")
    try:
        now = _default_outputs("--show")
    except Exception as exc:
        problems.append(f"show: {exc}")
    else:
        problems += [f"{role} is {now.get(role)!r}, was {device_id!r}" for role, device_id in saved.items()
                     if now.get(role) != device_id]
    assert not problems, (
        f"Could not put the default output device back ({'; '.join(problems)}). The original setting is in "
        f"{RESTORE_MARKER}. Pick your output device in the taskbar sound menu, then delete this file."
    )
    RESTORE_MARKER.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def default_output_restored():
    """Only for tests that change the default output: first undo a run that was killed before restoring."""
    if RESTORE_MARKER.exists():
        _put_back(json.loads(RESTORE_MARKER.read_text(encoding="utf-8")))
    yield


@contextmanager
def _default_output_may_change():
    """Remembers the default output devices (also on disk, in case the run is killed) and puts them back."""
    saved = _default_outputs("--show")
    RESTORE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    RESTORE_MARKER.write_text(json.dumps(saved), encoding="utf-8")
    try:
        yield saved
    finally:
        _put_back(saved)


def _two_enabled_outputs():
    """The current default A and another enabled device B, real hardware first."""
    import ctypes

    import soundcard

    ctypes.windll.ole32.CoInitializeEx(None, 0)  # this (main) thread has not started COM yet
    speakers = soundcard.all_speakers()  # enabled ones only
    a = soundcard.default_speaker()
    others = [speaker for speaker in speakers if speaker.id != a.id]
    assert others, f"需要至少两个可用的输出设备，现在只有：{[speaker.name for speaker in speakers]}"
    real = [s for s in others if not any(word in s.name.lower() for word in _VIRTUAL_DEVICE_WORDS)]
    b = (real or others)[0]
    print(f"\nA = {a.name} ({a.id})\nB = {b.name} ({b.id})")
    return a, b


def _until(condition, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


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
    from coursescribe.shortcut import parse

    pressed = threading.Event()
    hotkey = create_global_hotkey(parse("ctrl+alt+r", "windows"), pressed.set)
    assert hotkey.start(), "Ctrl+Alt+R is taken by another program"
    try:
        _press_hotkey()
        assert pressed.wait(3)
    finally:
        hotkey.stop()


def test_second_registration_reports_the_shortcut_is_taken():
    from coursescribe.platform import create_global_hotkey
    from coursescribe.shortcut import parse

    default = parse("ctrl+alt+r", "windows")
    first = create_global_hotkey(default, lambda: None)
    assert first.start()
    try:
        second = create_global_hotkey(default, lambda: None)
        assert second.start() is False
    finally:
        first.stop()


def test_a_non_default_shortcut_registers_is_refused_twice_and_frees_up_after_stop():
    from coursescribe.platform import create_global_hotkey
    from coursescribe.shortcut import parse

    other = parse("ctrl+alt+shift+t", "windows")
    first = create_global_hotkey(other, lambda: None)
    second = create_global_hotkey(other, lambda: None)
    third = create_global_hotkey(other, lambda: None)
    try:
        assert first.start()
        assert second.start() is False
        first.stop()
        assert third.start()  # each start uses a new object: GlobalHotkey.start() works once
    finally:
        for hotkey in (first, second, third):
            hotkey.stop()


# Following the default output device (plan M2, item 14).
# G1 and G2 change the system's default output device (marker changes_system_settings):
# run them only with Jeremy's approval, e.g. -m "gui and changes_system_settings".
# G3 changes nothing.


@pytest.mark.changes_system_settings
@pytest.mark.usefixtures("default_output_restored")
def test_g1_is_default_follows_the_system_setting_there_and_back():
    a, b = _two_enabled_outputs()
    with _default_output_may_change():
        with open_system_audio() as capture:
            assert capture.is_default() is True
            assert capture.device_name == a.name and capture.device_name, (a.name, b.name)
            assert (capture.sample_rate, capture.channels) == (48000, 2)
            _default_outputs("--set", b.id)
            assert _until(lambda: capture.is_default() is False, 1.0), (a.name, b.name)
            assert capture.default_id() == b.id
            with open_system_audio() as newer:
                assert newer.device_name == b.name
            _default_outputs("--set", a.id)
            assert _until(lambda: capture.is_default() is True, 1.0), (a.name, b.name)
    assert not RESTORE_MARKER.exists()


@pytest.mark.changes_system_settings
@pytest.mark.usefixtures("default_output_restored")
def test_g2_recording_follows_the_new_default_into_the_same_file(tmp_path):
    a, b = _two_enabled_outputs()
    target = tmp_path / "real.wav"
    notices = []
    with _default_output_may_change():
        started = time.monotonic()
        session = RecordingSession(open_system_audio, target, on_error=notices.append)
        session.start()
        try:
            _play_tone(2.0, hz=1000)
            _default_outputs("--set", b.id)
            time.sleep(1.5)
            _play_tone(2.0, hz=1500)
        finally:
            seconds = session.stop()
        elapsed = time.monotonic() - started
    assert notices.count(_SWITCHED.format(name=b.name)) == 1, (notices, a.name, b.name)
    assert abs(seconds - elapsed) < 1.0, (seconds, elapsed)
    with wave.open(str(target), "rb") as stream:
        assert (stream.getframerate(), stream.getnchannels()) == (48000, 2)
        frames = stream.getnframes()
    assert _tone_share(target, 1000, slice(0, frames // 2)) > 0.02
    assert _tone_share(target, 1500, slice(frames // 2, None)) > 0.02  # silence here before the fix
    assert repair_wav(target) == pytest.approx(seconds)
    assert _until(lambda: not [t for t in threading.enumerate() if t.name == "silence" and t.is_alive()], 2.0)


def test_g3_a_device_chosen_by_id_is_never_followed_away():
    from coursescribe.platform.windows import WindowsSystemAudio

    before = _default_outputs("--show")
    a, b = _two_enabled_outputs()
    for speaker in (b, a):
        with WindowsSystemAudio(speaker_name=speaker.id) as capture:
            blocks = [capture.read(12000) for _ in range(4)]
            assert capture.is_default() is True, speaker.name
            assert capture.device_name == speaker.name
        assert [block.shape for block in blocks] == [(12000, 2)] * 4
    assert _default_outputs("--show") == before
