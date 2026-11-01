"""Writing recordings to disk as they happen, and surviving interruption."""

from __future__ import annotations

import threading
import time
import wave

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

    def test_frames_written_counts_seconds(self, tmp_path):
        writer = WavWriter(tmp_path / "a.wav", RATE, 2)
        writer.write(_tone(0.5))
        assert writer.seconds == pytest.approx(0.5)
        writer.close()


class TestRepairWav:
    def _crashed_file(self, tmp_path, extra=b""):
        target = tmp_path / "a.wav"
        writer = WavWriter(target, RATE, 2)
        writer.write(_tone(2.0))
        writer.close()
        raw = bytearray(target.read_bytes())
        raw[4:8] = (36).to_bytes(4, "little")  # sizes still say "empty"
        raw[40:44] = (0).to_bytes(4, "little")
        target.write_bytes(bytes(raw) + extra)
        return target

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

    @pytest.mark.parametrize("content", [b"", b"RIFF", b"not a wav file at all" * 10])
    def test_unusable_file_means_no_sound(self, tmp_path, content):
        target = tmp_path / "a.wav"
        target.write_bytes(content)
        assert repair_wav(target) == 0.0
        assert target.read_bytes() == content

    def test_missing_file_means_no_sound(self, tmp_path):
        assert repair_wav(tmp_path / "gone.wav") == 0.0


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

    def test_device_that_will_not_open_fails_at_start_with_plain_words(self, tmp_path):
        def broken():
            raise RuntimeError("0x88890008")

        session = RecordingSession(broken, tmp_path / "a.wav")
        with pytest.raises(RecordingFailed) as caught:
            session.start()
        assert "0x" not in str(caught.value) and "录" in str(caught.value)

    def test_device_lost_midway_keeps_what_was_recorded_and_reports(self, tmp_path):
        errors = []
        session = RecordingSession(lambda: FakeCapture(fail_after=4), tmp_path / "a.wav", on_error=errors.append)
        session.start()
        assert _wait_for(lambda: bool(errors))
        assert "录音中断" in errors[0]
        assert session.stop() == pytest.approx(1.0)

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
