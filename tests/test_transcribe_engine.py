"""The real transcription model on generated speech."""

from __future__ import annotations

import difflib
import gc
import json
import shutil
import subprocess
import urllib.error
import urllib.request
import wave
from datetime import datetime

import pytest

import coursescribe.pipeline as flow
from coursescribe.jobs import Job, JobStore, Stage
from coursescribe.pipeline import SOURCE_MISSING, NoAudio, Pipeline
from coursescribe.segments import read_segments, segments_to_text
from coursescribe.text import content_chars
from coursescribe.transcribe import WhisperTranscriber, _is_gpu_failure, is_model_downloaded, load_whisper
from coursescribe.wavfile import repair_wav
from no_skip_plugin import require_module
from speech import as_video, convert, speak_to_wav
from test_pipeline import FakeProofreader

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


def test_source_file_that_is_gone_says_where_it_was_with_the_real_decoder(transcriber, tmp_path):
    gone = tmp_path / "不存在.mp3"
    with pytest.raises(NoAudio) as caught:
        transcriber.transcribe(gone, MODEL, 0.0, lambda s: None, lambda p, d: None)
    assert str(caught.value) == SOURCE_MISSING.format(path=gone)
    assert not gone.exists()


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


# --- the GPU cannot be used this time (docs/plans/M2-gpu-fallback-tests.md) ---

_OLLAMA_PS = "http://127.0.0.1:11434/api/ps"
_FREE_ENOUGH_MB = 3072  # the desktop itself uses about 1.5 GB


def _ollama_loaded_models() -> list[str]:
    """Models Ollama holds on this computer now; Ollama not running means none."""
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through the system proxy
    try:
        with direct.open(_OLLAMA_PS, timeout=5) as response:
            return [model.get("name", "?") for model in json.load(response).get("models", [])]
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return []
        pytest.fail(f"could not ask Ollama which models it holds ({exc}); close Ollama first")


def _require_an_idle_gpu() -> None:
    if not shutil.which("nvidia-smi"):
        pytest.fail("This check needs an NVIDIA GPU; run it on the Windows computer")
    loaded = _ollama_loaded_models()
    used = _gpu_memory_used_mb()
    if loaded or used > _FREE_ENOUGH_MB:
        pytest.fail(f"close Ollama and other GPU programs first (Ollama holds {loaded}; {used} MiB of GPU memory in use)")


def test_full_gpu_memory_moves_to_the_cpu_with_a_note_and_the_next_job_has_the_gpu_again(
    spoken_wav, tmp_path, transcriber
):
    transcriber.release()  # the model the tests above share would otherwise count as another program
    _require_an_idle_gpu()
    hogs = []

    def fill() -> tuple[str, bool]:
        """Load copies of the model onto the GPU until it has no room; the error is the real one."""
        for _ in range(8):
            try:
                hogs.append(load_whisper(MODEL, "cuda"))
            except Exception as exc:
                return str(exc), _is_gpu_failure(exc)
        pytest.fail("could not fill GPU memory")

    def job_in(folder: str) -> Job:
        job = Job.new_file(spoken_wav, tmp_path / folder, datetime(2026, 11, 3, 9, 0), MODEL, "model-a")
        store.save(job)
        return job

    store = JobStore(tmp_path / "state")
    engine = engine2 = None
    try:
        # (a) the model cannot even be loaded onto the GPU
        text, recognised = fill()
        assert recognised, text
        engine = WhisperTranscriber()
        seen = []

        def on_status(job, status, fraction):
            saved = len(read_segments(store.segments_path(job)))
            seen.append((job.job_id, status, saved, engine.device, engine.fell_back_to_cpu))

        pipeline = Pipeline(store, engine, FakeProofreader(), repair_wav, on_status=on_status)
        first = job_in("a")
        pipeline.run(first)
        assert store.load_all()[0].stage is Stage.DONE
        noted = [row for row in seen if row[1].startswith("正在转录") and flow.CPU_FALLBACK_NOTE in row[1]]
        assert noted, [row[1] for row in seen]
        assert noted[0][2] == 0  # said before the first segment was saved
        transcribing = [row for row in seen if row[1].startswith("正在转录")]
        assert all(row[3:] == ("cpu", True) for row in transcribing)
        assert _similarity(first.txt_path.read_text(encoding="utf-8")) > 0.85

        # (b) the model is on the GPU, then the memory fills before it is first used. Silence loads it
        # without running it (the speech filter leaves nothing); once it has run, it keeps its working
        # memory and a full GPU no longer stops it, so this must not be speech.
        hogs.clear()
        gc.collect()
        engine2 = WhisperTranscriber()
        silence = tmp_path / "silence.wav"
        with wave.open(str(silence), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(bytes(2 * 16000 * 2))
        assert _text(engine2, silence)[0] == []
        assert engine2.device == "cuda"
        text, recognised = fill()
        assert recognised, text
        events, segments = [], []

        def on_segment(segment):
            events.append("segment")
            segments.append(segment)

        engine2.transcribe(spoken_wav, MODEL, 0.0, on_segment, lambda p, d: events.append("progress"))
        assert (engine2.device, engine2.fell_back_to_cpu) == ("cpu", True)
        assert events[0] == "progress"
        assert _similarity(segments_to_text(segments)) > 0.85

        # (c) the memory is free again: the next job is on the GPU and says nothing about it
        hogs.clear()
        engine2.release()
        gc.collect()
        seen.clear()
        again = job_in("c")
        pipeline.run(again)
        assert [job.stage for job in store.load_all() if job.job_id == again.job_id] == [Stage.DONE]
        assert (engine.device, engine.fell_back_to_cpu) == ("cuda", False)
        assert not any(flow.CPU_FALLBACK_NOTE in row[1] for row in seen)
        assert _similarity(again.txt_path.read_text(encoding="utf-8")) > 0.85
    finally:
        hogs.clear()
        for used in (engine, engine2):
            if used is not None:
                used.release()
        gc.collect()
    assert hogs == []
