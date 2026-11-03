"""Transcription logic around the engine, using a stand-in for the model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from coursescribe.models import MODELS, model_size_label
from coursescribe.pipeline import SOURCE_HELD, SOURCE_MISSING, ModelMissing, NoAudio, Stopped
from coursescribe.transcribe import SAMPLE_RATE, WhisperTranscriber

AUDIO = Path("lesson.wav")


@dataclass
class RawSegment:
    start: float
    end: float
    text: str


class FakeModel:
    def __init__(self, segments=(), error=None):
        self.segments = segments
        self.error = error
        self.seen_seconds = None
        self.options = None

    def transcribe(self, audio, **options):
        self.seen_seconds = len(audio) / SAMPLE_RATE
        self.options = options

        def generate():
            for segment in self.segments:
                yield segment
            if self.error:
                raise self.error

        return generate(), None


def _transcriber(model=None, seconds=100, **kwargs):
    loads = []

    def load(name, device):
        loads.append((name, device))
        return model or FakeModel()

    transcriber = WhisperTranscriber(
        load_model=kwargs.pop("load_model", load),
        decode=kwargs.pop("decode", lambda path: np.zeros(seconds * SAMPLE_RATE, dtype=np.float32)),
        has_gpu=kwargs.pop("has_gpu", lambda: True),
        **kwargs,
    )
    transcriber.loads = loads
    return transcriber


def _run(transcriber, start_at=0.0):
    segments, progress = [], []
    transcriber.transcribe(AUDIO, "large-v3", start_at, segments.append, lambda p, d: progress.append((p, d)))
    return segments, progress


def test_segments_come_out_in_order_and_cleaned():
    model = FakeModel([RawSegment(0.0, 2.0, " 第一句 "), RawSegment(2.0, 4.0, "第二句")])
    segments, progress = _run(_transcriber(model))
    assert [(s.start, s.end, s.text) for s in segments] == [(0.0, 2.0, "第一句"), (2.0, 4.0, "第二句")]
    assert progress == [(2.0, 100.0), (4.0, 100.0)]


def test_resuming_decodes_only_the_rest_and_keeps_real_timestamps():
    model = FakeModel([RawSegment(1.0, 2.0, "后面的话")])
    segments, progress = _run(_transcriber(model), start_at=60.0)
    assert model.seen_seconds == 40.0
    assert (segments[0].start, segments[0].end) == (61.0, 62.0)
    assert progress == [(62.0, 100.0)]


def test_resuming_at_the_very_end_does_nothing():
    model = FakeModel([RawSegment(0.0, 1.0, "不应该出现")])
    segments, _ = _run(_transcriber(model, seconds=10), start_at=10.0)
    assert segments == []
    assert model.seen_seconds is None


def test_engine_is_set_up_for_chinese_with_english_terms_and_skips_silence():
    model = FakeModel()
    _run(_transcriber(model))
    assert model.options["language"] == "zh"
    assert "英文" in model.options["initial_prompt"]
    assert model.options["vad_filter"] is True
    assert model.options["condition_on_previous_text"] is False


def test_prefers_gpu_when_present():
    transcriber = _transcriber()
    _run(transcriber)
    assert transcriber.loads == [("large-v3", "cuda")]
    assert transcriber.device == "cuda"


def test_uses_cpu_when_there_is_no_gpu():
    transcriber = _transcriber(has_gpu=lambda: False)
    _run(transcriber)
    assert transcriber.loads == [("large-v3", "cpu")]
    assert transcriber.device == "cpu"


def test_falls_back_to_cpu_when_gpu_cannot_load():
    good = FakeModel([RawSegment(0, 1, "好")])

    def load(name, device):
        if device == "cuda":
            raise RuntimeError("CUDA driver too old")
        return good

    segments, _ = _run(_transcriber(load_model=load))
    assert [s.text for s in segments] == ["好"]


def test_falls_back_to_cpu_when_gpu_libraries_fail_at_first_use():
    broken = FakeModel(error=RuntimeError("Library cublas64_12.dll is not found"))
    good = FakeModel([RawSegment(0, 1, "好")])

    def load(name, device):
        return broken if device == "cuda" else good

    transcriber = _transcriber(load_model=load)
    segments, _ = _run(transcriber)
    assert [s.text for s in segments] == ["好"]
    assert transcriber.device == "cpu"


def test_gpu_failure_after_some_segments_continues_on_cpu_without_repeating():
    broken = FakeModel([RawSegment(0, 1, "前面")], error=RuntimeError("CUDA out of memory"))
    good = FakeModel([RawSegment(1, 2, "后面")])

    def load(name, device):
        return broken if device == "cuda" else good

    segments, _ = _run(_transcriber(load_model=load))
    assert [s.text for s in segments] == ["前面", "后面"]
    assert good.seen_seconds == 99.0


def test_other_engine_errors_are_not_hidden():
    with pytest.raises(ValueError):
        _run(_transcriber(FakeModel(error=ValueError("bad"))))


def test_same_model_is_loaded_only_once_across_files():
    transcriber = _transcriber()
    _run(transcriber)
    _run(transcriber)
    assert len(transcriber.loads) == 1


def test_another_model_replaces_the_loaded_one():
    transcriber = _transcriber()
    _run(transcriber)
    transcriber.transcribe(AUDIO, "medium", 0.0, lambda s: None, lambda p, d: None)
    assert [name for name, _ in transcriber.loads] == ["large-v3", "medium"]


def test_release_drops_the_model_and_next_use_loads_again():
    transcriber = _transcriber()
    _run(transcriber)
    transcriber.release()
    assert transcriber._model is None
    _run(transcriber)
    assert len(transcriber.loads) == 2


def test_missing_model_is_reported_without_touching_the_network():
    def load(name, device):
        raise ModelMissing(name)

    with pytest.raises(ModelMissing) as caught:
        _run(_transcriber(load_model=load))
    assert caught.value.model == "large-v3"


def test_missing_source_file_is_reported_in_plain_words():
    def decode(path):
        raise FileNotFoundError(path)

    with pytest.raises(NoAudio, match="找不到"):
        _run(_transcriber(decode=decode))


def test_unreadable_source_file_is_reported_in_plain_words():
    def decode(path):
        raise ValueError("Invalid data found when processing input")

    with pytest.raises(NoAudio, match="读不出"):
        _run(_transcriber(decode=decode))


def _decode_test_audio(path):
    """Reads the real file, so a missing one is reported by the file system; only the media engine is left out."""
    Path(path).read_bytes()
    return np.ones(SAMPLE_RATE, dtype=np.float32)


def test_missing_source_file_says_where_it_was_and_what_to_do(tmp_path):
    audio = tmp_path / "第一课.mp3"
    transcriber = _transcriber(decode=_decode_test_audio)
    with pytest.raises(NoAudio) as caught:
        transcriber.transcribe(audio, "large-v3", 0.0, lambda s: None, lambda p, d: None)
    message = str(caught.value)
    assert message == SOURCE_MISSING.format(path=audio)
    assert str(audio) in message and "放回" in message and "重试" in message
    assert not audio.exists()
    assert transcriber.loads == []


class _HeldDecoder:
    """Another program has the source file open and Windows refuses to share it, ``times`` times."""

    def __init__(self, times=None, error=None):
        self.times = times
        self.error = error
        self.attempts = 0

    def __call__(self, path):
        self.attempts += 1
        if self.attempts > 100:
            raise AssertionError("waited forever")
        if self.times is None or self.attempts <= self.times:
            raise self.error or PermissionError(13, "Permission denied")
        return np.ones(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def quick_waits(monkeypatch):
    monkeypatch.setattr("coursescribe.files._REPLACE_WAIT", 0.001)


def test_source_file_held_briefly_is_waited_for(monkeypatch, quick_waits):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    decoder = _HeldDecoder(times=3)
    segments, _ = _run(_transcriber(FakeModel([RawSegment(0, 1, "第一句")]), decode=decoder))
    assert [s.text for s in segments] == ["第一句"]
    assert decoder.attempts == 4


@pytest.mark.parametrize(("can_be_held", "attempts"), [(True, 20), (False, 1)], ids=["windows", "mac"])
def test_source_file_that_stays_held_is_called_held_not_soundless(monkeypatch, quick_waits, can_be_held, attempts):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", can_be_held)
    decoder = _HeldDecoder()
    transcriber = _transcriber(decode=decoder)
    with pytest.raises(NoAudio) as caught:
        _run(transcriber)
    message = str(caught.value)
    assert message == SOURCE_HELD.format(path=AUDIO)
    assert "读不出" not in message and "没有声音" not in message
    assert decoder.attempts == attempts
    assert transcriber.loads == []


def test_source_on_an_unplugged_drive_is_not_called_held(monkeypatch, quick_waits):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    decoder = _HeldDecoder(error=OSError(5, "Input/output error"))
    with pytest.raises(NoAudio) as caught:
        _run(_transcriber(decode=decoder))
    assert str(caught.value) == f"读不出这个文件的声音：{AUDIO.name}。"
    assert decoder.attempts == 1


class _EngineArgumentError(ValueError):
    """What the media engine (PyAV) raises on Windows for a missing or held file with a non-ASCII name."""

    errno = 22


def _engine_cannot_open(path):
    raise _EngineArgumentError(f"Invalid argument: '{path}' returned 22")


def test_missing_source_with_a_chinese_name_is_found_missing_even_though_the_engine_says_invalid(tmp_path):
    audio = tmp_path / "第一课.mp3"
    with pytest.raises(NoAudio) as caught:
        _run_on(_transcriber(decode=_engine_cannot_open), audio)
    assert str(caught.value) == SOURCE_MISSING.format(path=audio)
    assert not audio.exists()


def test_held_source_with_a_chinese_name_is_found_held_even_though_the_engine_says_invalid(tmp_path, monkeypatch):
    from test_recording import _held

    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    audio = tmp_path / "第一课.mp3"
    audio.write_bytes(b"audio")
    held = _held(monkeypatch, audio)
    with pytest.raises(NoAudio) as caught:
        _run_on(_transcriber(decode=_engine_cannot_open), audio)
    assert str(caught.value) == SOURCE_HELD.format(path=audio)
    assert held.attempts == 20
    held.release()
    assert audio.read_bytes() == b"audio"


def test_readable_source_the_engine_calls_invalid_is_reported_as_unreadable(tmp_path):
    audio = tmp_path / "第一课.mp3"
    audio.write_bytes(b"audio")
    with pytest.raises(NoAudio) as caught:
        _run_on(_transcriber(decode=_engine_cannot_open), audio)
    assert str(caught.value) == "读不出这个文件的声音：第一课.mp3。"
    assert audio.read_bytes() == b"audio"


def _run_on(transcriber, audio):
    transcriber.transcribe(audio, "large-v3", 0.0, lambda s: None, lambda p, d: None)


def test_file_with_no_sound_is_reported():
    with pytest.raises(NoAudio, match="声音"):
        _run(_transcriber(decode=lambda path: np.zeros(0, dtype=np.float32)))


def test_stop_request_ends_after_the_current_segment():
    model = FakeModel([RawSegment(0, 1, "一"), RawSegment(1, 2, "二"), RawSegment(2, 3, "三")])
    seen = []
    transcriber = _transcriber(model, should_stop=lambda: len(seen) >= 1)
    with pytest.raises(Stopped):
        transcriber.transcribe(AUDIO, "large-v3", 0.0, seen.append, lambda p, d: None)
    assert [s.text for s in seen] == ["一"]


def test_catalog_has_the_default_and_every_entry_states_its_size():
    assert "large-v3" in MODELS
    assert all(size > 0 for size in MODELS.values())
    assert model_size_label("large-v3") == "约 3.1 GB"
    assert model_size_label("unknown-model") == "大小未知"
