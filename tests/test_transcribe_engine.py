"""The real transcription model on generated speech."""

from __future__ import annotations

import difflib
import shutil
import subprocess

import pytest

from coursescribe.pipeline import NoAudio
from coursescribe.segments import segments_to_text
from coursescribe.text import content_chars
from coursescribe.transcribe import WhisperTranscriber, is_model_downloaded
from no_skip_plugin import require_module
from speech import as_video, convert, speak_to_wav

pytestmark = pytest.mark.engine

MODEL = "large-v3"
SPOKEN = "今天我们来学习数据库索引。索引可以让查询变快，但是写入数据的时候会慢一点。所以索引不是越多越好。"


@pytest.fixture(scope="module")
def spoken_wav(tmp_path_factory):
    require_module("faster_whisper", "run: python -m pip install -e .[dev]")
    if not is_model_downloaded(MODEL):
        pytest.fail(f"Model {MODEL} is not downloaded; open the app once and agree to download it")
    return speak_to_wav(SPOKEN, tmp_path_factory.mktemp("speech") / "spoken.wav")


@pytest.fixture(scope="module")
def transcriber():
    engine = WhisperTranscriber()
    yield engine
    engine.release()


def _text(engine, path, start_at=0.0):
    segments = []
    engine.transcribe(path, MODEL, start_at, segments.append, lambda p, d: None)
    return segments, segments_to_text(segments)


def _similarity(text: str) -> float:
    return difflib.SequenceMatcher(None, content_chars(SPOKEN), content_chars(text), autojunk=False).ratio()


def test_wav_is_transcribed_accurately(transcriber, spoken_wav):
    segments, text = _text(transcriber, spoken_wav)
    assert _similarity(text) > 0.85, text
    assert segments[0].start < segments[-1].end
    assert "数据库" in text


@pytest.mark.parametrize("suffix", ["mp3", "m4a", "mp4", "mkv"])
def test_common_formats_are_read_without_system_ffmpeg(transcriber, spoken_wav, tmp_path, suffix, monkeypatch):
    if suffix in ("mp4", "mkv"):
        sample = as_video(spoken_wav, tmp_path / f"sample.{suffix}")
    else:
        sample = convert(spoken_wav, tmp_path / f"sample.{suffix}")
    monkeypatch.setenv("PATH", "")  # the engine's own decoder must be enough
    _, text = _text(transcriber, sample)
    assert _similarity(text) > 0.8, text


def test_resuming_transcribes_only_the_later_part(transcriber, spoken_wav):
    whole, _ = _text(transcriber, spoken_wav)
    middle = whole[-1].end / 2
    later, _ = _text(transcriber, spoken_wav, start_at=middle)
    assert later and all(segment.start >= middle for segment in later)
    assert len(content_chars(segments_to_text(later))) < len(content_chars(segments_to_text(whole)))


def test_file_with_no_audio_track_is_reported(transcriber, tmp_path):
    video = tmp_path / "silent.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.fail("ffmpeg is needed to make test samples")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=1", str(video)],
        check=True,
    )
    with pytest.raises(NoAudio):
        transcriber.transcribe(video, MODEL, 0.0, lambda s: None, lambda p, d: None)


def test_not_a_media_file_is_reported(transcriber, tmp_path):
    fake = tmp_path / "fake.mp3"
    fake.write_bytes(b"this is not audio")
    with pytest.raises(NoAudio):
        transcriber.transcribe(fake, MODEL, 0.0, lambda s: None, lambda p, d: None)


def _gpu_memory_used_mb() -> int:
    done = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(done.stdout.split()[0])


def test_release_gives_gpu_memory_back(spoken_wav):
    if not shutil.which("nvidia-smi"):
        pytest.fail("This check needs an NVIDIA GPU; run it on the Windows computer")
    before = _gpu_memory_used_mb()
    engine = WhisperTranscriber()
    _text(engine, spoken_wav)
    assert engine.device == "cuda"
    during = _gpu_memory_used_mb()
    engine.release()
    after = _gpu_memory_used_mb()
    assert during - before > 1000
    assert after - before < 300
