"""The Mac system-sound source, driven by a fake Core Audio session that delivers real sample bytes."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from coursescribe.platform.mac_audio import MacSystemAudio


class FakeBuffer:
    def __init__(self, samples: np.ndarray) -> None:
        self.data = samples.astype(np.float32).tobytes()


class FakeSession:
    """Stands in for catap's RecordingSession: the test pushes buffers through on_buffer."""

    def __init__(self, on_buffer, fail_on_start: BaseException | None = None) -> None:
        self.on_buffer = on_buffer
        self.started = 0
        self.stopped = 0
        self.fail_on_start = fail_on_start

    def start(self) -> None:
        if self.fail_on_start:
            raise self.fail_on_start
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def push(self, frames: np.ndarray) -> None:
        self.on_buffer(FakeBuffer(frames))


def make(fail: BaseException | None = None, wait: float = 0.3):
    holder: dict[str, FakeSession] = {}

    def factory(on_buffer):
        holder["s"] = FakeSession(on_buffer, fail)
        return holder["s"]

    return MacSystemAudio(session_factory=factory, read_timeout=wait), holder


def stereo(left: list[float], right: list[float]) -> np.ndarray:
    return np.column_stack([left, right])


def test_it_reports_the_format_the_recorder_expects():
    audio, _ = make()
    assert (audio.sample_rate, audio.channels) == (48000, 2)


def test_read_returns_the_exact_samples_as_16_bit_stereo():
    audio, h = make()
    with audio:
        h["s"].push(stereo([0.0, 0.5, -0.5, 1.0], [0.25, -0.25, 0.0, -1.0]))
        out = audio.read(4)
    assert out.dtype == np.int16 and out.shape == (4, 2)
    assert out[:, 0].tolist() == [0, 16383, -16383, 32767]
    assert out[:, 1].tolist() == [8191, -8191, 0, -32767]


def test_read_clips_samples_beyond_full_scale():
    audio, h = make()
    with audio:
        h["s"].push(stereo([3.0, -3.0], [3.0, -3.0]))
        out = audio.read(2)
    assert out.max() == 32767 and out.min() == -32767


def test_blocks_are_cut_to_the_size_asked_and_nothing_is_lost_or_repeated():
    audio, h = make()
    with audio:
        h["s"].push(stereo([0.1, 0.2, 0.3], [0, 0, 0]))
        h["s"].push(stereo([0.4, 0.5], [0, 0]))
        first = audio.read(2)
        second = audio.read(3)
    joined = np.concatenate([first, second])[:, 0]
    assert joined.tolist() == [int(v * 32767) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]


def test_read_waits_for_sound_that_arrives_later():
    audio, h = make(wait=2.0)
    with audio:
        threading.Timer(0.1, lambda: h["s"].push(stereo([0.5, 0.5], [0.5, 0.5]))).start()
        out = audio.read(2)
    assert out[:, 0].tolist() == [16383, 16383]


def test_a_stalled_stream_gives_silence_instead_of_hanging_forever():
    audio, _ = make(wait=0.1)
    with audio:
        start = time.monotonic()
        out = audio.read(480)
    assert time.monotonic() - start < 1.0
    assert out.shape == (480, 2) and not out.any()


def test_a_short_stall_keeps_the_sound_already_received():
    audio, h = make(wait=0.1)
    with audio:
        h["s"].push(stereo([0.5], [0.5]))
        out = audio.read(3)
    assert out.shape == (3, 2)
    assert out[0].tolist() == [16383, 16383] and not out[1:].any()


def test_leaving_stops_the_session_once_even_after_a_read_error():
    audio, h = make()
    with pytest.raises(RuntimeError):
        with audio:
            raise RuntimeError("boom")
    assert h["s"].started == 1 and h["s"].stopped == 1


def test_a_start_failure_is_reported_with_a_way_forward():
    from coursescribe.recorder import RecordingFailed

    audio, h = make(fail=OSError("permission denied"))
    with pytest.raises(RecordingFailed) as info:
        audio.__enter__()
    assert "permission" in str(info.value).lower() or "权限" in str(info.value)
    assert h["s"].stopped == 0


def test_the_global_tap_never_leaves_the_default_output():
    audio, _ = make()
    with audio:
        assert audio.is_default() is True
        assert audio.default_id() == audio.device_id


def test_a_buffer_with_an_odd_byte_count_does_not_corrupt_the_stream():
    audio, h = make()
    with audio:
        bad = FakeBuffer(stereo([0.5], [0.5]))
        bad.data = bad.data + b"\x00\x00"  # a torn trailing sample
        h["s"].on_buffer(bad)
        h["s"].push(stereo([0.25], [0.25]))
        out = audio.read(2)
    assert out[:, 0].tolist() == [16383, 8191]
