"""Writing recordings to disk as they happen, and surviving interruption."""

from __future__ import annotations

import logging
import random
import shutil
import struct
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from coursescribe.recorder import (
    _ENDED_DISK,
    _ENDED_FULL,
    _RECONNECTED,
    _SWITCH_FAILED,
    _SWITCHED,
    _SWITCHED_BACK,
    RecordingFailed,
    RecordingSession,
    SilenceMonitor,
)
from coursescribe.wavfile import WavWriter, repair_wav

RATE = 48000


def _tone(seconds: float, level: float = 0.2, channels: int = 2) -> np.ndarray:
    count = int(RATE * seconds)
    wave_form = (np.sin(np.arange(count) * 2 * np.pi * 440 / RATE) * level * 32767).astype(np.int16)
    return np.column_stack([wave_form] * channels)


def _read(path):
    with wave.open(str(path), "rb") as stream:
        return (
            stream.getframerate(),
            stream.getnchannels(),
            np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16),
        )


def _crashed_wav(path, seconds=2.0, extra=b""):
    """A recording cut off by a crash: all the sound is there, the header sizes still say empty."""
    writer = WavWriter(path, RATE, 2)
    writer.write(_tone(seconds))
    writer.close()
    raw = bytearray(path.read_bytes())
    raw[4:8] = (36).to_bytes(4, "little")
    raw[40:44] = (0).to_bytes(4, "little")
    path.write_bytes(bytes(raw) + extra)
    return path


class _Held:
    def __init__(self, times):
        self.times = times
        self.attempts = 0
        self.released = False

    def release(self):
        self.released = True


def _held(monkeypatch, path, times=None, error=None):
    """Another program (antivirus, a player) has ``path`` open and Windows refuses to share it.

    The first ``times`` opens are refused; ``None`` means until ``.release()``.
    ``error`` replaces the refusal with another failure, such as ``OSError(5, ...)`` for an unplugged drive.
    """
    real_open = Path.open
    target = Path(path)
    held = _Held(times)

    def open_(self, *args, **kwargs):
        if Path(self) == target:
            held.attempts += 1
            if held.attempts > 100:
                raise AssertionError("waited forever")
            if not held.released and (held.times is None or held.attempts <= held.times):
                if error is not None:
                    raise type(error)(*error.args)
                raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_)
    monkeypatch.setattr("coursescribe.files._REPLACE_WAIT", 0.001)
    return held


class _Fault:
    def __init__(self, at):
        self.at = at


class _Faulty:
    """A real open file whose change named by ``fault.at`` fails as on a full disk."""

    def __init__(self, stream, fault):
        self._stream = stream
        self._fault = fault
        self._writes = 0

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stream.close()

    def _check(self, action):
        if self._fault.at == action:
            raise OSError(28, "No space left on device")

    def truncate(self, *args):
        self._check("truncate")
        return self._stream.truncate(*args)

    def write(self, data):
        self._writes += 1
        self._check(f"size{self._writes}")
        self._check("write")
        return self._stream.write(data)


def _disk_full_during_repair(monkeypatch, path, at):
    """The disk fills up at ``at``: "truncate", "size1", "size2", or "write" for every write.

    Set ``.at = None`` to free space.
    """
    real_open = Path.open
    target = Path(path)
    fault = _Fault(at)

    def open_(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        return _Faulty(stream, fault) if Path(self) == target else stream

    monkeypatch.setattr(Path, "open", open_)
    return fault


# What WavWriter leaves when a recording stops before any sound arrives.
_EMPTY_RECORDING = struct.pack(
    "<4sI4s4sIHHIIHH4sI", b"RIFF", 36, b"WAVE", b"fmt ", 16, 1, 2, RATE, RATE * 4, 4, 16, b"data", 0
)


class TestWavWriter:
    def test_closed_file_reads_back_exactly(self, tmp_path):
        target = tmp_path / "a.wav"
        audio = _tone(0.5)
        writer = WavWriter(target, RATE, 2)
        writer.write(audio)
        writer.close()
        rate, channels, data = _read(target)
        assert (rate, channels) == (RATE, 2)
        assert np.array_equal(data, audio.reshape(-1))

    def test_refuses_to_replace_an_existing_file(self, tmp_path):
        target = tmp_path / "a.wav"
        target.write_bytes(b"precious")
        with pytest.raises(FileExistsError):
            WavWriter(target, RATE, 2)
        assert target.read_bytes() == b"precious"

    def test_close_twice_is_harmless(self, tmp_path):
        writer = WavWriter(tmp_path / "a.wav", RATE, 2)
        writer.close()
        writer.close()

    def test_sound_reaches_disk_within_a_second_of_audio_without_closing(self, tmp_path):
        target = tmp_path / "a.wav"
        writer = WavWriter(target, RATE, 2)
        writer.write(_tone(1.0))
        copy = tmp_path / "copy.wav"  # what a crash would leave behind
        copy.write_bytes(target.read_bytes())
        _, _, data = _read(copy)
        assert len(data) == RATE * 2
        writer.close()

    def test_write_stops_at_the_size_limit_and_says_so(self, tmp_path):
        target = tmp_path / "a.wav"
        writer = WavWriter(target, RATE, 2, max_data_bytes=RATE * 4)  # one second
        assert writer.write(_tone(0.75)) is True
        assert writer.write(_tone(0.5)) is False
        writer.close()
        _, _, data = _read(target)
        assert len(data) == RATE * 2

    def test_default_limit_keeps_the_sizes_inside_the_wav_header(self, tmp_path):
        from coursescribe.wavfile import MAX_DATA_BYTES

        assert MAX_DATA_BYTES + 36 <= 0xFFFFFFFF and MAX_DATA_BYTES % 4 == 0

    def test_frames_written_counts_seconds(self, tmp_path):
        writer = WavWriter(tmp_path / "a.wav", RATE, 2)
        writer.write(_tone(0.5))
        assert writer.seconds == pytest.approx(0.5)
        writer.close()


class TestRepairWav:
    def _crashed_file(self, tmp_path, extra=b""):
        return _crashed_wav(tmp_path / "a.wav", 2.0, extra)

    def test_restores_the_sizes_and_reports_the_length(self, tmp_path):
        target = self._crashed_file(tmp_path)
        assert repair_wav(target) == pytest.approx(2.0)
        _, _, data = _read(target)
        assert len(data) == 2 * RATE * 2

    def test_half_written_last_frame_is_trimmed(self, tmp_path):
        target = self._crashed_file(tmp_path, extra=b"\x01\x02\x03")
        assert repair_wav(target) == pytest.approx(2.0)
        assert (target.stat().st_size - 44) % 4 == 0

    def test_healthy_file_is_unchanged(self, tmp_path):
        target = tmp_path / "a.wav"
        writer = WavWriter(target, RATE, 2)
        writer.write(_tone(1.0))
        writer.close()
        before = target.read_bytes()
        assert repair_wav(target) == pytest.approx(1.0)
        assert target.read_bytes() == before

    @pytest.mark.parametrize(
        "content",
        [
            b"",
            b"RIFF",
            _EMPTY_RECORDING[:43],
            b"not a wav file at all" * 10,
            _EMPTY_RECORDING,
            _EMPTY_RECORDING[:22] + b"\x00\x00" + _EMPTY_RECORDING[24:] + b"\x01\x02\x03\x04",
            _EMPTY_RECORDING[:24] + b"\x00\x00\x00\x00" + _EMPTY_RECORDING[28:] + b"\x01\x02\x03\x04",
            _EMPTY_RECORDING[:32] + b"\x00\x00" + _EMPTY_RECORDING[34:] + b"\x01\x02\x03\x04",
        ],
        ids=["empty", "riff-only", "short-header", "not-wav", "no-data", "no-channels", "no-rate", "no-block"],
    )
    def test_unusable_file_means_no_sound(self, tmp_path, content):
        target = tmp_path / "a.wav"
        target.write_bytes(content)
        assert repair_wav(target) == 0.0
        assert target.read_bytes() == content

    def test_repairing_a_crashed_file_twice_gives_the_same_file(self, tmp_path):
        target = _crashed_wav(tmp_path / "a.wav", 2.0, extra=b"\x01\x02\x03")
        assert repair_wav(target) == pytest.approx(2.0)
        once = target.read_bytes()
        assert repair_wav(target) == pytest.approx(2.0)
        assert target.read_bytes() == once

    def test_less_than_one_frame_of_sound_is_cut_to_an_empty_recording(self, tmp_path):
        target = tmp_path / "a.wav"
        target.write_bytes(_EMPTY_RECORDING + b"\x01\x02\x03")
        assert repair_wav(target) == 0.0
        assert target.read_bytes() == _EMPTY_RECORDING

    def test_file_past_the_size_limit_is_cut_to_it_instead_of_crashing(self, tmp_path):
        target = self._crashed_file(tmp_path)
        assert repair_wav(target, max_data_bytes=RATE * 4) == pytest.approx(1.0)
        _, _, data = _read(target)
        assert len(data) == RATE * 2

    def test_empty_recording_constant_is_what_the_writer_leaves(self, tmp_path):
        WavWriter(tmp_path / "e.wav", RATE, 2).close()
        assert (tmp_path / "e.wav").read_bytes() == _EMPTY_RECORDING

    def test_missing_file_is_reported_not_taken_for_silence(self, tmp_path):
        """A recording moved away or deleted is not empty: putting it back and retrying must work."""
        target = tmp_path / "gone.wav"
        with pytest.raises(FileNotFoundError):
            repair_wav(target)
        assert not target.exists()

    def test_briefly_held_file_is_repaired_once_let_go(self, tmp_path, monkeypatch):
        monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
        target = _crashed_wav(tmp_path / "a.wav", 2.0)
        held = _held(monkeypatch, target, times=3)
        assert repair_wav(target) == pytest.approx(2.0)
        assert held.attempts == 4
        with wave.open(str(target), "rb") as stream:
            assert stream.getnframes() == 2 * RATE

    def test_file_held_for_long_is_reported_and_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
        target = _crashed_wav(tmp_path / "a.wav", 2.0)
        before = target.read_bytes()
        held = _held(monkeypatch, target)
        with pytest.raises(PermissionError):
            repair_wav(target)
        assert held.attempts == 20  # about a second of patience, no more
        held.release()
        assert target.read_bytes() == before

    def test_refusal_where_files_cannot_be_held_is_reported_at_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", False)
        target = _crashed_wav(tmp_path / "a.wav", 2.0)
        before = target.read_bytes()
        held = _held(monkeypatch, target)
        with pytest.raises(PermissionError):
            repair_wav(target)
        assert held.attempts == 1
        held.release()
        assert target.read_bytes() == before

    @pytest.mark.parametrize("at", ["truncate", "size1", "size2"])
    def test_disk_full_midway_keeps_the_sound_and_repairing_again_ends_as_if_once(self, tmp_path, monkeypatch, at):
        extra = b"\x01\x02\x03"
        target = _crashed_wav(tmp_path / "a.wav", 2.0, extra)
        reference = _crashed_wav(tmp_path / "ref.wav", 2.0, extra)
        sound = target.read_bytes()[44 : 44 + 2 * RATE * 4]
        assert repair_wav(reference) == pytest.approx(2.0)

        fault = _disk_full_during_repair(monkeypatch, target, at)
        with pytest.raises(OSError) as caught:
            repair_wav(target)
        assert caught.value.errno == 28
        after = target.read_bytes()
        assert len(after) >= 44 + 2 * RATE * 4
        assert after[44 : 44 + 2 * RATE * 4] == sound

        fault.at = None
        assert repair_wav(target) == pytest.approx(2.0)
        assert target.read_bytes() == reference.read_bytes()


class TestSilenceMonitor:
    def _feed(self, monitor, level, seconds):
        changes = []
        for _ in range(int(seconds * 4)):
            change = monitor.feed(_tone(0.25, level=level))
            if change is not None:
                changes.append(change)
        return changes

    def test_warns_once_after_continuous_silence(self):
        monitor = SilenceMonitor(RATE, alert_after=5.0)
        assert self._feed(monitor, 0.0, 4.5) == []
        assert self._feed(monitor, 0.0, 10.0) == [True]

    def test_sound_clears_the_warning_and_restarts_the_count(self):
        monitor = SilenceMonitor(RATE, alert_after=5.0)
        assert self._feed(monitor, 0.0, 6.0) == [True]
        assert self._feed(monitor, 0.2, 1.0) == [False]
        assert self._feed(monitor, 0.0, 4.5) == []

    def test_short_pause_in_speech_never_warns(self):
        monitor = SilenceMonitor(RATE, alert_after=5.0)
        for _ in range(5):
            assert self._feed(monitor, 0.0, 3.0) == []
            assert self._feed(monitor, 0.2, 1.0) == []

    def test_very_quiet_noise_counts_as_silence(self):
        monitor = SilenceMonitor(RATE, alert_after=1.0)
        assert self._feed(monitor, 0.0002, 2.0) == [True]

    def test_quiet_speech_is_not_silence(self):
        monitor = SilenceMonitor(RATE, alert_after=1.0)
        assert self._feed(monitor, 0.02, 2.0) == []


class FakeCapture:
    sample_rate = RATE
    channels = 2
    device_name = "假设备"
    device_id = "fake"

    def is_default(self):
        return True

    def default_id(self):
        return "fake"

    def __init__(self, level=0.2, fail_after=None, stuck_after=None, realtime=0.002):
        self.level = level
        self.fail_after = fail_after
        self.stuck_after = stuck_after
        self.realtime = realtime
        self.reads = 0
        self.exited = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited.set()

    def read(self, frames):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("device removed")
        if self.stuck_after is not None and self.reads > self.stuck_after:
            threading.Event().wait(30)
        time.sleep(self.realtime)
        return _tone(frames / RATE, level=self.level)


def _wait_for(condition, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


class TestRecordingSession:
    def test_records_until_stopped_and_file_holds_it_all(self, tmp_path):
        capture = FakeCapture()
        session = RecordingSession(lambda: capture, tmp_path / "a.wav")
        session.start()
        assert _wait_for(lambda: capture.reads >= 5)
        seconds = session.stop()
        rate, channels, data = _read(tmp_path / "a.wav")
        assert (rate, channels) == (RATE, 2)
        assert seconds == pytest.approx(len(data) / 2 / RATE)
        assert seconds >= 1.0
        assert capture.exited.is_set()
        assert session.finished is True

    def test_device_that_will_not_open_fails_at_start_with_plain_words(self, tmp_path):
        def broken():
            raise RuntimeError("0x88890008")

        session = RecordingSession(broken, tmp_path / "a.wav")
        with pytest.raises(RecordingFailed) as caught:
            session.start()
        assert "0x" not in str(caught.value) and "录" in str(caught.value)

    def test_device_lost_midway_reconnects_and_keeps_recording_into_the_same_file(self, tmp_path):
        captures = [FakeCapture(fail_after=4), FakeCapture()]
        notices, ended = [], []
        session = RecordingSession(
            lambda: captures.pop(0), tmp_path / "a.wav", on_error=notices.append, on_ended=ended.append,
            reconnect_wait=0.01,
        )
        session.start()
        assert _wait_for(lambda: session.seconds >= 2.0)
        assert "重新连接" in notices[0]
        assert ended == []
        seconds = session.stop()
        _, _, data = _read(tmp_path / "a.wav")
        assert seconds == pytest.approx(len(data) / 2 / RATE) and seconds >= 2.0

    def test_device_that_never_comes_back_ends_the_recording_and_keeps_it(self, tmp_path):
        opened = []

        def open_capture():
            if opened:
                raise RuntimeError("no device")
            opened.append(1)
            return FakeCapture(fail_after=4)

        ended = []
        session = RecordingSession(open_capture, tmp_path / "a.wav", on_ended=ended.append,
                                   reconnect_wait=0.01, reconnect_for=0.1)
        session.start()
        assert _wait_for(lambda: bool(ended))
        assert "录音中断" in ended[0] and "转" in ended[0]
        assert session.stop() == pytest.approx(1.0)
        _, _, data = _read(tmp_path / "a.wav")
        assert len(data) == RATE * 2

    def test_device_back_with_another_format_ends_the_recording(self, tmp_path):
        other = FakeCapture()
        other.channels = 1
        captures = [FakeCapture(fail_after=4), other]
        ended = []
        session = RecordingSession(lambda: captures.pop(0), tmp_path / "a.wav", on_ended=ended.append,
                                   reconnect_wait=0.01, reconnect_for=0.1)
        session.start()
        assert _wait_for(lambda: bool(ended))
        assert session.stop() == pytest.approx(1.0)
        assert "转" in ended[0]
        assert other.exited.is_set()  # the device opened in the wrong format is let go

    def test_stop_while_reconnecting_ends_quietly(self, tmp_path):
        """Stop can arrive while the device is reopening, which can take a while."""
        opening, reopening, release = threading.Event(), threading.Event(), threading.Event()
        second = FakeCapture()

        def open_capture():
            if not opening.is_set():
                opening.set()
                return FakeCapture(fail_after=4)
            reopening.set()
            release.wait(5)
            return second

        notices, ended = [], []
        session = RecordingSession(open_capture, tmp_path / "a.wav", on_error=notices.append,
                                   on_ended=ended.append, reconnect_wait=0.01)
        session.start()
        assert reopening.wait(5)
        stopper = threading.Thread(target=session.stop)
        stopper.start()
        assert _wait_for(lambda: session._stop.is_set())
        release.set()
        stopper.join(5)
        assert not stopper.is_alive()
        assert notices == [] and ended == []  # no "reconnected, still recording" after Stop
        assert second.exited.is_set()  # the device reopened meanwhile is let go
        _, _, data = _read(tmp_path / "a.wav")
        assert len(data) == RATE * 2

    def test_disk_error_ends_the_recording_and_stop_never_raises(self, tmp_path, monkeypatch):
        from coursescribe import wavfile

        real_write = wavfile.WavWriter.write
        calls = []

        def failing_write(self, frames):
            calls.append(1)
            if len(calls) > 4:
                raise OSError(28, "No space left on device")
            return real_write(self, frames)

        def failing_close(self):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(wavfile.WavWriter, "write", failing_write)
        monkeypatch.setattr(wavfile.WavWriter, "close", failing_close)
        ended = []
        session = RecordingSession(lambda: FakeCapture(), tmp_path / "a.wav", on_ended=ended.append)
        session.start()
        assert _wait_for(lambda: bool(ended))
        assert "磁盘" in ended[0]
        assert session.stop() == pytest.approx(1.0)
        assert session.stop() == pytest.approx(1.0)
        assert session.finished is False  # the header may be behind: repair before use
        assert repair_wav(tmp_path / "a.wav") == pytest.approx(1.0)
        _, _, data = _read(tmp_path / "a.wav")
        assert len(data) == RATE * 2  # what was recorded before the disk failed is all there

    def test_recording_stops_itself_at_the_file_size_limit(self, tmp_path):
        ended = []
        limit = RATE * 4 * 2  # two seconds of stereo 16-bit sound
        session = RecordingSession(lambda: FakeCapture(), tmp_path / "a.wav", on_ended=ended.append,
                                   max_data_bytes=limit)
        session.start()
        assert _wait_for(lambda: bool(ended))
        assert "上限" in ended[0]
        assert session.stop() == pytest.approx(2.0)
        _, _, data = _read(tmp_path / "a.wav")
        assert len(data) == RATE * 2 * 2

    def test_stop_returns_even_if_the_device_hangs(self, tmp_path):
        capture = FakeCapture(stuck_after=2)
        session = RecordingSession(lambda: capture, tmp_path / "a.wav", stop_timeout=0.2)
        session.start()
        assert _wait_for(lambda: capture.reads >= 3)
        started = time.time()
        assert session.stop() == pytest.approx(0.5)
        assert time.time() - started < 2

    def test_silence_warning_reaches_the_listener(self, tmp_path):
        changes = []
        session = RecordingSession(
            lambda: FakeCapture(level=0.0),
            tmp_path / "a.wav",
            on_silence=changes.append,
            silence_alert_after=0.5,
        )
        session.start()
        assert _wait_for(lambda: changes == [True])
        session.stop()

    def test_stop_twice_gives_the_same_answer(self, tmp_path):
        session = RecordingSession(lambda: FakeCapture(), tmp_path / "a.wav")
        session.start()
        first = session.stop()
        assert session.stop() == first


# ---------------------------------------------------------------------------
# Following the default output device (plan M2, item 14)
# ---------------------------------------------------------------------------

BLOCK = 12000  # frames the session asks for at a time: 0.25 s
_NOT_FOUND = "Error 0x80070490"  # what Windows says when there is no default output device
_IN_USE = "Error 0x8889000a"  # AUDCLNT_E_DEVICE_IN_USE: another program holds the device exclusively
_VALUES = {"A": 1000, "B": 2000, "C": 3000}


class FakeLoopback:
    """One opened loopback recorder: hears its device's sound only while that device is the default."""

    def __init__(self, system: FakeSoundSystem, device_id: str) -> None:
        self.system = system
        self.id = device_id
        self.device_id = device_id
        self.device_name = system.names.get(device_id, "")
        self.sample_rate, self.channels = system.formats.get(device_id, (RATE, 2))
        self.open_delay = system.open_delay.get(device_id, 0.0)
        self.exited = threading.Event()
        self.exit_count = 0
        self.frames_read = 0
        self._entered = False

    def __enter__(self):
        system = self.system
        with system.lock:
            self._entered = True
            system.open_now += 1
            system.max_open = max(system.max_open, system.open_now)
        return self

    def __exit__(self, *args):
        system = self.system
        system.event_closing(self.id).set()
        gate = system.close_gate.get(self.id)
        if gate is not None:
            gate.wait(10)
        with system.lock:
            self.exit_count += 1
            if self._entered and self.exit_count == 1:
                system.open_now -= 1
                index = system.captures.index(self)
                newer = [c for c in system.captures[index + 1 :] if c._entered and c.exit_count == 0]
                if newer:  # let go after its successor was up: a completed switch
                    system.switches.append((newer[-1].open_delay, system.close_delay.get(self.id, 0.0)))
                    system.switch_at.append(system.reads)
            system.advance(system.close_delay.get(self.id, 0.0))
        self.exited.set()

    def is_default(self) -> bool:
        current = self.system.default
        if current is None:
            raise RuntimeError(_NOT_FOUND)
        return current == self.id

    def default_id(self) -> str:
        current = self.system.default
        if current is None:
            raise RuntimeError(_NOT_FOUND)
        return current

    def read(self, frames: int) -> np.ndarray:
        system = self.system
        if self.exit_count:
            raise RuntimeError("read after the capture was closed")
        with system.lock:
            system.reads += 1
            number = system.reads
            system.open_at_read.append(system.open_now)
            if self.id in system.gone:
                raise RuntimeError("device removed")
            current = system.default
            value = system.value.get(self.id, 0) if current == self.id else 0
            self.frames_read += frames
            system.read_log.append((self.id, value, current))
        block = np.full((frames, self.channels), value, dtype=np.int16)
        action = system.script.get(number)
        if action is not None:
            action()
        if system.read_sleep:
            time.sleep(system.read_sleep)
        return block


class FakeSoundSystem:
    """The operating system's side: which output is the default, and what each loopback hears.

    Programs play on the default device, so a loopback on any other device hears only silence,
    which is what Windows does (checked on the real machine by the gui test G2).
    """

    def __init__(
        self,
        default: str | None = "A",
        names: dict[str, str] | None = None,
        value: dict[str, int] | None = None,
        refuse: dict[str, Exception] | None = None,
        formats: dict[str, tuple[int, int]] | None = None,
        open_delay: dict[str, float] | None = None,
        close_delay: dict[str, float] | None = None,
        open_gate: dict[str, threading.Event] | None = None,
        script: dict[int, Callable[[], None]] | None = None,
        read_sleep: float = 0.001,
    ) -> None:
        self.default = default
        self.names = names if names is not None else {"A": "扬声器", "B": "耳机", "C": "显示器"}
        self.value = value if value is not None else dict(_VALUES)
        self.refuse = refuse if refuse is not None else {}
        self.formats = formats or {}
        self.open_delay = open_delay or {}
        self.close_delay = close_delay or {}
        self.open_gate = open_gate or {}
        self.opening: dict[str, threading.Event] = {}
        self.close_gate: dict[str, threading.Event] = {}  # closing this id waits for the event
        self.closing: dict[str, threading.Event] = {}  # set when a close of this id begins
        self.after_open: dict[str, Callable[[], None]] = {}  # run right after this id opened
        self.gone: set[str] = set()
        self.script = script if script is not None else {}
        self.read_sleep = read_sleep
        self.lock = threading.RLock()
        self.reads = 0
        self.attempts: list[str] = []
        self.attempt_log: list[tuple[str, int, str]] = []  # (id, reads so far, "refused" or "opened")
        self.opened: list[str] = []
        self.captures: list[FakeLoopback] = []
        self.open_now = 0
        self.max_open = 0
        self.open_at_read: list[int] = []
        self.read_log: list[tuple[str, int, str | None]] = []  # (device read, value, default at the time)
        self.switches: list[tuple[float, float]] = []
        self.switch_at: list[int] = []  # reads so far when each switch completed
        self._ms = 0

    def advance(self, seconds: float) -> None:
        with self.lock:
            self._ms += round(seconds * 1000)

    def clock(self) -> float:
        return self._ms / 1000

    def event_opening(self, device_id: str) -> threading.Event:
        with self.lock:
            return self.opening.setdefault(device_id, threading.Event())

    def event_closing(self, device_id: str) -> threading.Event:
        with self.lock:
            return self.closing.setdefault(device_id, threading.Event())

    def open(self) -> FakeLoopback:
        """Like ``open_system_audio``: whatever is the default at this moment."""
        target = self.default
        if target is None:
            raise RuntimeError(_NOT_FOUND)
        with self.lock:
            self.attempts.append(target)
            reads = self.reads
        self.event_opening(target).set()
        gate = self.open_gate.get(target)
        if gate is not None:
            gate.wait(10)
        self.advance(self.open_delay.get(target, 0.0))
        refusal = self.refuse.get(target)
        if refusal is not None:
            self.attempt_log.append((target, reads, "refused"))
            raise type(refusal)(*refusal.args)
        with self.lock:
            self.gone.discard(target)
            capture = FakeLoopback(self, target)
            self.captures.append(capture)
            self.opened.append(target)
            self.attempt_log.append((target, reads, "opened"))
        hook = self.after_open.get(target)
        if hook is not None:
            hook()
        return capture

    def of(self, device_id: str) -> list[FakeLoopback]:
        return [capture for capture in self.captures if capture.id == device_id]


def _runs(path) -> list[tuple[int, int]]:
    """The recording as (sample value, frame count) stretches, both channels being the same."""
    rate, channels, data = _read(path)
    assert (rate, channels) == (RATE, 2)
    frames = data.reshape(-1, 2)
    assert np.array_equal(frames[:, 0], frames[:, 1])
    values = frames[:, 0]
    if len(values) == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(values)]])
    return [(int(values[s]), int(e - s)) for s, e in zip(starts, ends)]


def _switch_to(system: FakeSoundSystem, device_id: str | None) -> Callable[[], None]:
    def action():
        system.default = device_id

    return action


def _together(*actions: Callable[[], None]) -> Callable[[], None]:
    def action():
        for each in actions:
            each()

    return action


def _record(system: FakeSoundSystem, folder: Path, until: float, **options):
    """Records from the fake system until ``until`` seconds are written, then stops."""
    notices, ended = [], []
    session = RecordingSession(
        system.open, folder / "a.wav", on_error=notices.append, on_ended=ended.append, clock=system.clock, **options
    )
    session.start()
    try:
        assert _wait_for(lambda: session.seconds >= until or bool(ended), 10)
    finally:
        seconds = session.stop()  # also when failing: a session left running would disturb later tests
    return session, seconds, notices, ended


def _start(request, session: RecordingSession) -> RecordingSession:
    """Starts ``session`` and makes sure it is stopped when the test ends, whatever happens."""
    session.start()
    request.addfinalizer(session.stop)
    return session


def _frames(path) -> int:
    _, _, data = _read(path)
    return len(data) // 2


def _records(caplog, level, text):
    return [
        r for r in caplog.records
        if r.name == "coursescribe.recorder" and r.levelno == level and text in r.getMessage()
    ]


class TestFollowingTheDefaultOutput:
    def test_messages_are_plain_words(self):
        assert "{name}" in _SWITCHED and "同一个文件" in _SWITCHED
        assert "录不到" in _SWITCH_FAILED and "再试" in _SWITCH_FAILED
        assert "正常" in _SWITCHED_BACK
        for text in (_SWITCHED, _SWITCH_FAILED, _SWITCHED_BACK):
            assert "0x" not in text and "WASAPI" not in text and "loopback" not in text.lower()

    def test_d0_new_default_while_the_old_device_stays_is_followed(self, tmp_path):
        system = FakeSoundSystem(names={"A": "扬声器", "B": "耳机"})
        system.script[8] = _switch_to(system, "B")
        _, seconds, notices, ended = _record(system, tmp_path, 3.0)
        runs = _runs(tmp_path / "a.wav")
        assert runs[0] == (1000, 96000)
        assert len(runs) == 2 and runs[1][0] == 2000, runs
        assert notices == [_SWITCHED.format(name="耳机")]
        assert ended == []
        assert system.opened == ["A", "B"]
        assert set(system.open_at_read) == {1}  # A was let go before B was first read
        assert system.max_open == 2
        assert [path.name for path in tmp_path.iterdir()] == ["a.wav"]
        assert [capture.exit_count for capture in system.captures] == [1, 1]
        assert seconds * RATE == pytest.approx(_frames(tmp_path / "a.wav"))

    @pytest.mark.parametrize("n", [1, 3, 4, 5, 8])
    def test_d1_found_within_a_second_with_the_old_device_silent_meanwhile(self, tmp_path, n):
        system = FakeSoundSystem()
        system.script[n] = _switch_to(system, "B")
        _record(system, tmp_path, 4.0)
        runs = _runs(tmp_path / "a.wav")
        z = ((n + 3) // 4 * 4 - n) * BLOCK
        assert runs[:-1] == [(1000, n * BLOCK)] + ([(0, z)] if z else [])
        assert runs[-1][0] == 2000
        assert z <= 48000
        if n in (4, 8):
            assert z == 0

    @pytest.mark.parametrize(
        ("open_delay", "close_delay", "gap", "capped"),
        [
            (0.0, 0.0, 0, False),
            (0.5, 0.0, 24000, False),
            (1.999, 0.0, 95952, False),
            (10.0, 0.0, 96000, True),
            (0.5, 0.25, 36000, False),
            (1.5, 1.0, 96000, True),
        ],
    )
    def test_d2_the_switch_gap_is_filled_with_silence_up_to_two_seconds(
        self, tmp_path, caplog, open_delay, close_delay, gap, capped
    ):
        caplog.set_level(logging.DEBUG, logger="coursescribe.recorder")
        system = FakeSoundSystem(open_delay={"B": open_delay}, close_delay={"A": close_delay})
        system.script[8] = _switch_to(system, "B")
        _, seconds, _, _ = _record(system, tmp_path, 4.0)
        runs = _runs(tmp_path / "a.wav")
        assert runs[:-1] == [(1000, 96000)] + ([(0, gap)] if gap else [])
        assert runs[-1][0] == 2000
        assert seconds * RATE == pytest.approx(_frames(tmp_path / "a.wav"))
        warnings = [r for r in caplog.records if r.name == "coursescribe.recorder" and r.levelno == logging.WARNING]
        assert len(warnings) == (1 if capped else 0)

    def test_d3_back_to_the_first_device(self, tmp_path):
        system = FakeSoundSystem()
        system.script[8] = _switch_to(system, "B")
        system.script[16] = _switch_to(system, "A")
        _, _, notices, _ = _record(system, tmp_path, 5.0)
        runs = _runs(tmp_path / "a.wav")
        assert [value for value, _ in runs if value] == [1000, 2000, 1000]
        assert notices == [_SWITCHED.format(name="耳机"), _SWITCHED.format(name="扬声器")]
        assert system.opened == ["A", "B", "A"]
        assert [capture.exit_count for capture in system.captures] == [1, 1, 1]

    def test_d4a_several_changes_within_a_second_follow_only_the_last(self, tmp_path):
        system = FakeSoundSystem()
        system.script[5] = _switch_to(system, "B")
        system.script[6] = _switch_to(system, "C")
        _, _, notices, _ = _record(system, tmp_path, 3.0)
        assert system.attempts == ["A", "C"]
        assert [value for value, _ in _runs(tmp_path / "a.wav") if value] == [1000, 3000]
        assert notices == [_SWITCHED.format(name="显示器")]

    def test_d4b_away_and_back_within_a_second_changes_nothing(self, tmp_path):
        system = FakeSoundSystem()
        system.script[5] = _switch_to(system, "B")
        system.script[7] = _switch_to(system, "A")
        _, _, notices, _ = _record(system, tmp_path, 3.0)
        assert system.attempts == ["A"]
        assert notices == []
        runs = _runs(tmp_path / "a.wav")
        assert runs[:2] == [(1000, 60000), (0, 24000)]
        assert len(runs) == 3 and runs[2][0] == 1000

    @pytest.mark.parametrize(("open_delay", "gap"), [(None, 0), (0.5, 24000)], ids=["a", "b-open-delay"])
    def test_d5_device_that_will_not_open_keeps_the_old_one_says_so_once_and_retries(
        self, tmp_path, caplog, open_delay, gap
    ):
        caplog.set_level(logging.DEBUG, logger="coursescribe.recorder")
        system = FakeSoundSystem(
            refuse={"B": RuntimeError(_IN_USE)}, open_delay={"B": open_delay} if open_delay else None
        )
        a_exits_meanwhile = []
        system.script[8] = _switch_to(system, "B")
        system.script[30] = lambda: a_exits_meanwhile.append(system.captures[0].exit_count)
        system.script[40] = lambda: system.refuse.pop("B")
        _, _, notices, ended = _record(system, tmp_path, 13.0)
        assert a_exits_meanwhile == [0]  # A stayed open all along
        assert system.attempts.count("B") == 3  # at 2.0 s, 7.0 s and 12.0 s
        assert notices == [_SWITCH_FAILED, _SWITCHED.format(name="耳机")]
        assert "0x" not in "".join(notices)
        assert ended == []
        runs = _runs(tmp_path / "a.wav")
        assert runs[:2] == [(1000, 96000), (0, 40 * BLOCK + gap)]
        assert len(runs) == 3 and runs[2][0] == 2000
        assert len(_records(caplog, logging.WARNING, "0x8889000a")) == 1
        assert len(_records(caplog, logging.DEBUG, "0x8889000a")) >= 1
        read = sum(capture.frames_read for capture in system.captures)
        assert _frames(tmp_path / "a.wav") == read + gap  # failed opens add no silence

    @pytest.mark.parametrize("other_format", [(44100, 2), (48000, 1)])
    def test_d6_device_in_another_format_is_never_written_into_the_file(self, tmp_path, caplog, other_format):
        caplog.set_level(logging.DEBUG, logger="coursescribe.recorder")
        system = FakeSoundSystem(formats={"B": other_format})
        a_exits_meanwhile = []
        system.script[8] = _switch_to(system, "B")
        system.script[50] = lambda: a_exits_meanwhile.append(system.captures[0].exit_count)
        _, _, notices, ended = _record(system, tmp_path, 13.0)
        runs = _runs(tmp_path / "a.wav")  # also checks the header is still 48000 Hz stereo
        assert 2000 not in [value for value, _ in runs]
        assert system.attempts.count("B") >= 3
        assert [capture.exit_count for capture in system.of("B")] == [1] * len(system.of("B"))
        assert a_exits_meanwhile == [0]
        assert notices == [_SWITCH_FAILED]
        assert ended == []
        said = [
            r for r in caplog.records
            if r.name == "coursescribe.recorder" and r.levelno == logging.WARNING
            and str(other_format) in r.getMessage() and str((RATE, 2)) in r.getMessage()
        ]
        assert len(said) == 1

    def test_d7_back_to_the_recorded_device_after_a_failure_then_away_again(self, tmp_path):
        system = FakeSoundSystem(refuse={"B": RuntimeError(_IN_USE)})
        system.script[8] = _switch_to(system, "B")
        system.script[14] = _switch_to(system, "A")
        system.script[22] = _switch_to(system, "B")
        _, _, notices, _ = _record(system, tmp_path, 7.0)
        assert notices == [_SWITCH_FAILED, _SWITCHED_BACK, _SWITCH_FAILED]
        assert system.attempts.count("B") == 2
        assert system.opened == ["A"]
        runs = _runs(tmp_path / "a.wav")
        assert runs[:3] == [(1000, 96000), (0, 6 * BLOCK), (1000, 8 * BLOCK)]
        assert len(runs) == 4 and runs[3][0] == 0

    def test_d8_change_while_nothing_plays_is_followed_by_identity(self, tmp_path):
        changes = []
        system = FakeSoundSystem(value={"A": 0, "B": 0})
        system.script[8] = _switch_to(system, "B")
        _, _, notices, _ = _record(system, tmp_path, 4.0, silence_alert_after=0.5, on_silence=changes.append)
        assert system.opened == ["A", "B"]
        assert notices == [_SWITCHED.format(name="耳机")]
        assert changes == [True]

    def test_d9a_silence_warning_ends_once_the_sound_is_followed(self, tmp_path):
        changes = []
        system = FakeSoundSystem(refuse={"B": RuntimeError(_IN_USE)})
        system.script[1] = _switch_to(system, "B")
        system.script[12] = lambda: system.refuse.pop("B")
        _, _, notices, _ = _record(system, tmp_path, 8.0, silence_alert_after=1.0, on_silence=changes.append)
        assert changes == [True, False]
        assert notices == [_SWITCH_FAILED, _SWITCHED.format(name="耳机")]

    def test_d9b_filled_silence_does_not_raise_the_silence_warning(self, tmp_path):
        changes = []
        system = FakeSoundSystem(open_delay={"B": 1.5})
        system.script[8] = _switch_to(system, "B")
        _record(system, tmp_path, 5.0, silence_alert_after=1.0, on_silence=changes.append)
        assert changes == []
        runs = _runs(tmp_path / "a.wav")
        assert runs[:2] == [(1000, 96000), (0, 72000)]
        assert len(runs) == 3 and runs[2][0] == 2000

    def test_d10_no_output_device_at_all_keeps_recording_quietly(self, tmp_path, caplog):
        caplog.set_level(logging.DEBUG, logger="coursescribe.recorder")
        system = FakeSoundSystem()
        system.script[4] = _switch_to(system, None)
        system.script[20] = _switch_to(system, "A")
        _, _, notices, ended = _record(system, tmp_path, 6.0)
        assert notices == [] and ended == []
        assert system.opened == ["A"]
        assert len(_records(caplog, logging.WARNING, "0x80070490")) == 1

    def test_d11a_old_device_unplugged_and_new_default_says_where_it_went(self, tmp_path):
        system = FakeSoundSystem()
        system.script[5] = _together(_switch_to(system, "B"), lambda: system.gone.add("A"))
        _, _, notices, ended = _record(system, tmp_path, 3.0, reconnect_wait=0.01)
        assert notices == [_SWITCHED.format(name="耳机")]
        assert [value for value, _ in _runs(tmp_path / "a.wav") if value] == [1000, 2000]
        assert ended == []

    def test_d11b_same_device_back_keeps_the_reconnected_message(self, tmp_path):
        system = FakeSoundSystem()
        system.script[5] = lambda: system.gone.add("A")
        _, _, notices, ended = _record(system, tmp_path, 3.0, reconnect_wait=0.01)
        assert notices == [_RECONNECTED]
        assert ended == []

    def test_d12a_stop_while_the_new_device_opens(self, request, tmp_path, monkeypatch):
        errors = []
        monkeypatch.setattr(threading, "excepthook", errors.append)
        gate = threading.Event()
        system = FakeSoundSystem(open_gate={"B": gate})
        system.script[8] = _switch_to(system, "B")
        notices, ended = [], []
        session = RecordingSession(system.open, tmp_path / "a.wav", on_error=notices.append,
                                   on_ended=ended.append, clock=system.clock)
        request.addfinalizer(gate.set)  # a failing test must not leave the thread waiting
        _start(request, session)
        assert system.event_opening("B").wait(5)
        stopper = threading.Thread(target=session.stop)
        started = time.time()
        stopper.start()
        assert _wait_for(lambda: session._stop.is_set())
        gate.set()
        stopper.join(5)
        assert not stopper.is_alive() and time.time() - started < 5
        assert _runs(tmp_path / "a.wav") == [(1000, 96000)]
        assert notices == [] and ended == []
        assert [capture.exit_count for capture in system.captures] == [1, 1]
        assert session.finished is True
        assert errors == []

    def test_d12b_stop_gives_up_on_a_hung_open_and_nothing_follows(self, request, tmp_path, monkeypatch):
        errors = []
        monkeypatch.setattr(threading, "excepthook", errors.append)
        gate = threading.Event()
        system = FakeSoundSystem(open_gate={"B": gate}, open_delay={"B": 0.5})
        system.script[8] = _switch_to(system, "B")
        notices, ended = [], []
        session = RecordingSession(system.open, tmp_path / "a.wav", on_error=notices.append,
                                   on_ended=ended.append, clock=system.clock, stop_timeout=0.2)
        request.addfinalizer(gate.set)  # a failing test must not leave the thread waiting
        _start(request, session)
        assert system.event_opening("B").wait(5)
        started = time.time()
        assert session.stop() == pytest.approx(2.0)
        assert time.time() - started < 2
        before = (tmp_path / "a.wav").read_bytes()
        gate.set()
        session._thread.join(5)
        assert not session._thread.is_alive()
        assert (tmp_path / "a.wav").read_bytes() == before
        assert [capture.exit_count for capture in system.captures] == [1, 1]
        assert notices == [] and ended == []
        assert errors == []

    def test_d13_killed_during_a_switch_leaves_a_repairable_start_of_the_recording(self, request, tmp_path):
        gate = threading.Event()
        system = FakeSoundSystem(open_gate={"B": gate}, open_delay={"B": 0.5})
        system.script[8] = _switch_to(system, "B")
        session = RecordingSession(system.open, tmp_path / "a.wav", clock=system.clock)
        request.addfinalizer(gate.set)  # a failing test must not leave the thread waiting
        _start(request, session)
        assert system.event_opening("B").wait(5)
        shutil.copyfile(tmp_path / "a.wav", tmp_path / "crash1.wav")
        gate.set()
        assert _wait_for(lambda: session.seconds >= 4.0)
        shutil.copyfile(tmp_path / "a.wav", tmp_path / "crash2.wav")
        session.stop()
        _, _, final = _read(tmp_path / "a.wav")
        assert repair_wav(tmp_path / "crash1.wav") >= 1.0
        rate, channels, first = _read(tmp_path / "crash1.wav")
        assert (rate, channels) == (RATE, 2)
        assert np.array_equal(first, final[: len(first)])
        repair_wav(tmp_path / "crash2.wav")
        _, _, second = _read(tmp_path / "crash2.wav")
        assert np.array_equal(second, final[: len(second)])
        assert 2000 in second

    def test_d14a_disk_full_while_filling_the_gap_ends_the_recording(self, request, tmp_path, monkeypatch):
        system = FakeSoundSystem(open_delay={"B": 0.5})
        system.script[8] = _switch_to(system, "B")
        disk = _disk_full_during_repair(monkeypatch, tmp_path / "a.wav", None)
        system.after_open["B"] = lambda: setattr(disk, "at", "write")  # the disk fills up as B opens
        ended = []
        session = RecordingSession(system.open, tmp_path / "a.wav", on_ended=ended.append, clock=system.clock)
        _start(request, session)
        assert _wait_for(lambda: bool(ended))
        assert ended == [_ENDED_DISK]
        assert session.stop() == pytest.approx(2.0)
        assert [capture.exit_count for capture in system.captures] == [1, 1]

    def test_d14b_size_limit_inside_the_filled_gap_ends_the_recording(self, request, tmp_path):
        system = FakeSoundSystem(open_delay={"B": 0.5})
        system.script[8] = _switch_to(system, "B")
        ended = []
        session = RecordingSession(system.open, tmp_path / "a.wav", on_ended=ended.append, clock=system.clock,
                                   max_data_bytes=(96000 + 12000) * 4)
        _start(request, session)
        assert _wait_for(lambda: bool(ended))
        assert ended == [_ENDED_FULL]
        session.stop()
        assert _runs(tmp_path / "a.wav") == [(1000, 96000), (0, 12000)]
        assert [capture.exit_count for capture in system.captures] == [1, 1]

    def test_d15_device_without_a_name(self, tmp_path):
        system = FakeSoundSystem(names={"A": "扬声器", "B": ""})
        system.script[8] = _switch_to(system, "B")
        _, _, notices, _ = _record(system, tmp_path, 3.0)
        assert notices == [_SWITCHED.format(name="新的输出设备")]

    def test_d16_a_third_device_during_a_failure_is_tried_at_once(self, tmp_path):
        system = FakeSoundSystem(refuse={"B": RuntimeError(_IN_USE)})
        system.script[8] = _switch_to(system, "B")
        system.script[14] = _switch_to(system, "C")
        # The file reaches the disk once per second of sound, so a copy at read 21 holds blocks 1-20.
        system.script[21] = lambda: shutil.copyfile(tmp_path / "a.wav", tmp_path / "before21.wav")
        _, _, notices, _ = _record(system, tmp_path, 6.0)
        assert system.attempts == ["A", "B", "C"]
        assert [device for device, _, _ in system.read_log[16:19]] == ["C", "C", "C"]  # reads 17-19
        _, _, early = _read(tmp_path / "before21.wav")
        assert len(early) == 20 * BLOCK * 2
        assert 3000 in early  # C was tried at 4.0 s, not after B's five-second wait
        assert notices == [_SWITCH_FAILED, _SWITCHED.format(name="显示器")]

    @pytest.mark.parametrize("path", ["failed", "switched", "back"])
    def test_r1_nothing_is_checked_switched_or_said_once_stop_is_asked(self, request, tmp_path, monkeypatch, path):
        """Stop arrives while a block is being read; the check due after that block must not happen."""
        errors = []
        monkeypatch.setattr(threading, "excepthook", errors.append)
        holder = {}

        def stop_now():
            holder["session"]._stop.set()

        system = FakeSoundSystem()
        if path == "failed":
            system.refuse["B"] = RuntimeError(_IN_USE)
            system.script[8] = _together(_switch_to(system, "B"), stop_now)
            expected_notices, expected_attempts = [], ["A"]
        elif path == "switched":
            system.script[8] = _together(_switch_to(system, "B"), stop_now)
            expected_notices, expected_attempts = [], ["A"]
        else:
            system.refuse["B"] = RuntimeError(_IN_USE)
            system.script[8] = _switch_to(system, "B")
            system.script[16] = _together(_switch_to(system, "A"), stop_now)
            expected_notices, expected_attempts = [_SWITCH_FAILED], ["A", "B"]
        notices, ended = [], []
        session = RecordingSession(system.open, tmp_path / "a.wav", on_error=notices.append,
                                   on_ended=ended.append, clock=system.clock)
        holder["session"] = session
        _start(request, session)
        session._thread.join(5)
        assert not session._thread.is_alive()
        session.stop()
        assert notices == expected_notices
        assert system.attempts == expected_attempts
        assert ended == [] and errors == []
        assert [capture.exit_count for capture in system.captures] == [1] * len(system.captures)

    def test_r2_default_back_between_the_two_questions_is_no_change(self, tmp_path):
        """is_default() says no, but by the time default_id() answers it is this device again."""
        system = FakeSoundSystem()

        def stale_answer():
            system.captures[0].is_default = lambda: False

        system.script[1] = stale_answer
        _, _, notices, ended = _record(system, tmp_path, 3.0)
        assert system.attempts == ["A"]  # not reopened
        assert notices == [] and ended == []
        runs = _runs(tmp_path / "a.wav")
        assert len(runs) == 1 and runs[0][0] == 1000

    def test_r3_reconnect_during_a_failed_switch_tells_about_the_refused_device_once_more(self, tmp_path):
        system = FakeSoundSystem(refuse={"B": RuntimeError(_IN_USE)})
        system.script[8] = _switch_to(system, "B")
        system.script[10] = _together(_switch_to(system, "A"), lambda: system.gone.add("A"))
        system.script[14] = _switch_to(system, "B")
        _, _, notices, ended = _record(system, tmp_path, 12.0, reconnect_wait=0.01)
        assert notices == [_SWITCH_FAILED, _RECONNECTED, _SWITCH_FAILED]
        assert system.attempts.count("B") >= 3  # retried after the second failure, without telling again
        assert ended == []

    def test_d12c_stop_gives_up_while_the_old_device_closes_during_a_switch(self, request, tmp_path, monkeypatch):
        errors = []
        monkeypatch.setattr(threading, "excepthook", errors.append)
        gate = threading.Event()
        system = FakeSoundSystem(open_delay={"B": 0.5})
        system.close_gate["A"] = gate
        system.script[8] = _switch_to(system, "B")
        notices, ended = [], []
        session = RecordingSession(system.open, tmp_path / "a.wav", on_error=notices.append,
                                   on_ended=ended.append, clock=system.clock, stop_timeout=0.2)
        request.addfinalizer(gate.set)
        _start(request, session)
        assert system.event_closing("A").wait(5)
        assert session.stop() == pytest.approx(2.0)
        before = (tmp_path / "a.wav").read_bytes()
        gate.set()
        session._thread.join(5)
        assert not session._thread.is_alive()
        assert (tmp_path / "a.wav").read_bytes() == before  # no silence added after the file was finished
        assert notices == [] and ended == []
        assert [capture.exit_count for capture in system.captures] == [1, 1]
        assert errors == []

    def test_a_read_that_returns_after_stop_gave_up_is_not_written(self, request, tmp_path, monkeypatch):
        errors = []
        monkeypatch.setattr(threading, "excepthook", errors.append)
        gate, hanging = threading.Event(), threading.Event()
        system = FakeSoundSystem()
        system.script[6] = lambda: (hanging.set(), gate.wait(10))
        notices, ended = [], []
        session = RecordingSession(system.open, tmp_path / "a.wav", on_error=notices.append,
                                   on_ended=ended.append, clock=system.clock, stop_timeout=0.2)
        request.addfinalizer(gate.set)
        _start(request, session)
        assert hanging.wait(5)
        assert session.stop() == pytest.approx(1.25)
        before = (tmp_path / "a.wav").read_bytes()
        gate.set()
        session._thread.join(5)
        assert not session._thread.is_alive()
        assert (tmp_path / "a.wav").read_bytes() == before
        assert notices == [] and ended == []
        assert system.captures[0].exit_count == 1
        assert errors == []


# --- P1: random switching scripts ---------------------------------------------

_P1_SCRIPTS = 150
_P1_SEED = 20261104
_RETRY_BLOCKS = 20  # five seconds of sound
_FOLLOW_BLOCKS = 8  # two seconds of sound


def _p1_plan(rng: random.Random) -> dict:
    total = rng.randint(30, 80)
    changes = [(rng.randint(1, total), rng.choice("ABC")) for _ in range(rng.randint(0, 6))]
    windows, formats = {}, {}
    for device in "BC":
        roll = rng.random()
        if roll < 0.2:
            formats[device] = (44100, 2)
        elif roll < 0.6:
            low = rng.randint(0, total)
            windows[device] = (low, rng.randint(low + 1, total + 20))
    return {
        "total": total,
        "changes": changes,
        "refuse_windows": windows,
        "formats": formats,
        "open_delay": {device: rng.choice([0.0, 0.3, 3.0]) for device in "ABC"},
        "close_delay": {device: rng.choice([0.0, 0.2]) for device in "ABC"},
        "copy_at": rng.randint(1, total),
    }


def _p1_refused(plan: dict, device: str, reads: int) -> bool:
    """Whether opening ``device`` is refused once ``reads`` reads (and their scripts) have happened."""
    window = plan["refuse_windows"].get(device)
    return window is not None and window[0] <= reads < window[1]


def _gap_frames(open_delay: float, close_delay: float) -> int:
    return min(round((open_delay + close_delay) * RATE), 2 * RATE)


def _p1_run(plan: dict, folder: Path):
    system = FakeSoundSystem(
        formats=plan["formats"], open_delay=plan["open_delay"], close_delay=plan["close_delay"], read_sleep=0
    )
    for device, (low, _) in plan["refuse_windows"].items():
        if low == 0:
            system.refuse[device] = RuntimeError(_IN_USE)
    actions: dict[int, list[Callable[[], None]]] = {}
    for at, device in plan["changes"]:
        actions.setdefault(at, []).append(_switch_to(system, device))
    for device, (low, high) in plan["refuse_windows"].items():
        if low > 0:
            actions.setdefault(low, []).append(lambda d=device: system.refuse.__setitem__(d, RuntimeError(_IN_USE)))
        actions.setdefault(high, []).append(lambda d=device: system.refuse.pop(d, None))
    copy = folder / "copy.wav"
    actions.setdefault(plan["copy_at"], []).append(lambda: shutil.copyfile(folder / "a.wav", copy))
    holder = {}
    # Stop is asked for from inside the last read, so every run reads exactly ``total`` blocks.
    actions.setdefault(plan["total"], []).append(lambda: holder["session"]._stop.set())
    system.script = {at: _together(*listed) for at, listed in actions.items()}
    notices, ended = [], []
    session = RecordingSession(system.open, folder / "a.wav", on_error=notices.append, on_ended=ended.append,
                               clock=system.clock)
    holder["session"] = session
    session.start()
    try:
        session._thread.join(10)
        assert not session._thread.is_alive()
    finally:
        seconds = session.stop()
    return system, seconds, notices, ended, copy


def _p1_check(plan: dict, folder: Path) -> dict[str, int]:
    """Runs one script and checks the laws. Returns how many default stretches Law 4 checked, per device."""
    system, seconds, notices, ended, copy = _p1_run(plan, folder)
    try:
        return _p1_laws(plan, folder, system, seconds, notices, ended, copy)
    except AssertionError as failure:
        raise AssertionError(f"reads={system.reads}\n{failure}") from failure


def _p1_laws(plan, folder, system, seconds, notices, ended, copy) -> dict[str, int]:
    assert system.reads == plan["total"]  # the same script always runs the same way
    checked = {"A": 0, "B": 0, "C": 0}
    rate, channels, data = _read(folder / "a.wav")
    frames = len(data) // 2
    good = {device for device in "ABC" if plan["formats"].get(device, (RATE, 2)) == (RATE, 2)}
    assert ended == []

    # Law 1: the same file in the first format, as long as reported
    assert (rate, channels) == (RATE, 2)
    assert round(seconds * RATE) == frames

    # Law 2: every frame read is written once, plus the silence of each completed switch
    gaps = [_gap_frames(*switch) for switch in system.switches]
    assert frames == sum(capture.frames_read for capture in system.captures) + sum(gaps)
    switch_gap = dict(zip(system.switch_at, gaps))
    expected = []
    for number, (_, value, _) in enumerate(system.read_log, start=1):
        expected.append(np.full(BLOCK * 2, value, dtype=np.int16))
        if switch_gap.get(number):
            expected.append(np.zeros(switch_gap[number] * 2, dtype=np.int16))
    assert np.array_equal(data, np.concatenate(expected))

    # Law 3: sound comes in the order the devices were opened, never from a device in another format
    by_value = {value: device for device, value in _VALUES.items()}
    # One capture can give two stretches of sound with silence between (the default went elsewhere and
    # came back without a switch), so repeats are merged on both sides.
    heard = [by_value[value] for value, _ in _runs(folder / "a.wav") if value]
    heard = [device for i, device in enumerate(heard) if i == 0 or heard[i - 1] != device]
    order = [device for device in system.opened if device in good]
    order = [device for i, device in enumerate(order) if i == 0 or order[i - 1] != device]
    remaining = iter(order)
    assert all(device in remaining for device in heard), (heard, order)

    # Law 4: a default that stays long enough on a device that opens is followed
    log = system.read_log
    start = 0
    while start < len(log):
        end = start
        while end + 1 < len(log) and log[end + 1][2] == log[start][2]:
            end += 1
        device = log[start][2]
        first, last = start + 1, end + 1  # read numbers, 1-based
        if device in good:
            before_last = [entry for entry in system.attempt_log if entry[1] <= last]
            begin = first
            if before_last and before_last[-1][0] == device and before_last[-1][2] == "refused":
                begin = max(first, before_last[-1][1] + _RETRY_BLOCKS)
            steady = not any(_p1_refused(plan, device, r) for r in range(begin - 1, last + 1))
            if steady and last - begin + 1 >= _FOLLOW_BLOCKS:
                checked[device] += 1
                assert log[end][1] == _VALUES[device], ("not followed", device, first, last, begin)
        start = end + 1

    # Law 5: one capture at every read, at most two while switching, each closed exactly once
    assert all(capture.exit_count == 1 for capture in system.captures)
    assert system.max_open <= 2
    assert set(system.open_at_read) == {1}

    # Law 6: a copy taken at any moment repairs to the start of the final file
    if repair_wav(copy) == 0.0:
        # Only in the first second: until then not even the header has reached the disk.
        assert plan["copy_at"] <= 4, "an empty copy after the first second of sound"
        early = np.zeros(0, dtype=np.int16)
    else:
        _, _, early = _read(copy)
    assert np.array_equal(early, data[: len(early)])

    # What the user is told matches what happened
    switched = [notice for notice in notices if notice.startswith(_SWITCHED.split("{name}")[0])]
    assert len(switched) == len(system.switches)
    failures = [entry for entry in system.attempt_log if entry[2] == "refused"]
    failures += [entry for entry in system.attempt_log if entry[2] == "opened" and entry[0] not in good]
    assert notices.count(_SWITCH_FAILED) <= len(failures)
    return checked


def test_p1_random_switching_keeps_the_file_the_devices_and_the_notices_in_step(tmp_path):
    rng = random.Random(_P1_SEED)
    checked = {"A": 0, "B": 0, "C": 0}
    for index in range(_P1_SCRIPTS):
        plan = _p1_plan(rng)
        folder = tmp_path / f"s{index}"
        folder.mkdir()
        try:
            for device, count in _p1_check(plan, folder).items():
                checked[device] += count
        except AssertionError as failure:
            raise AssertionError(f"script {index}: {plan}\n{failure}") from failure
    # Law 4 must really have been put to the test on the devices the recording moves to.
    assert checked["B"] > 0 and checked["C"] > 0, checked
