"""Writing recordings to disk as they happen, and surviving interruption."""

from __future__ import annotations

import struct
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from coursescribe.recorder import RecordingFailed, RecordingSession, SilenceMonitor
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
        return self._stream.write(data)


def _disk_full_during_repair(monkeypatch, path, at):
    """The disk fills up at ``at``: "truncate", "size1" or "size2". Set ``.at = None`` to free space."""
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
