"""Turn audio into timed text with faster-whisper, on the GPU when possible."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from pathlib import Path

from coursescribe.pipeline import ModelMissing, NoAudio, Stopped
from coursescribe.segments import Segment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # what the engine works on
_INITIAL_PROMPT = "这是一节中文网课的录音，其中夹杂英文技术词汇。英文词汇保持英文原文，不要音译成中文。"
_GPU_FAILURE_WORDS = ("cuda", "cublas", "cudnn", "out of memory", "gpu")


def load_whisper(name: str, device: str):
    """Load a model already on this computer. This never downloads anything."""
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    try:
        path = download_model(name, local_files_only=True)
    except FileNotFoundError as exc:  # the Hugging Face "not in cache" error is one of these
        raise ModelMissing(name) from exc
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(path, device=device, compute_type=compute_type, local_files_only=True)


def is_model_downloaded(name: str) -> bool:
    from faster_whisper.utils import download_model

    try:
        download_model(name, local_files_only=True)
    except FileNotFoundError:
        return False
    return True


def download_whisper_model(name: str) -> None:
    """Fetch a model from the internet. Call only after the user has agreed."""
    from faster_whisper.utils import download_model

    download_model(name)


def _decode_with_engine(path: Path):
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def _gpu_available() -> bool:
    import ctranslate2

    return ctranslate2.get_cuda_device_count() > 0


def _is_gpu_failure(error: Exception) -> bool:
    text = str(error).lower()
    return any(word in text for word in _GPU_FAILURE_WORDS)


class WhisperTranscriber:
    def __init__(
        self,
        load_model: Callable = load_whisper,
        decode: Callable = _decode_with_engine,
        has_gpu: Callable[[], bool] = _gpu_available,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        self._load_model = load_model
        self._decode = decode
        self._has_gpu = has_gpu
        self._should_stop = should_stop
        self._model = None
        self._model_name: str | None = None
        self.device: str | None = None
        self._gpu_failed = False

    def release(self) -> None:
        """Give back the model's memory, so proofreading can have the GPU."""
        self._model = None
        self._model_name = None
        gc.collect()

    def _ensure_model(self, name: str):
        if self._model is not None and self._model_name == name:
            return self._model
        self.release()
        if self._has_gpu() and not self._gpu_failed:
            try:
                self._model = self._load_model(name, "cuda")
                self.device = "cuda"
            except ModelMissing:
                raise
            except Exception as exc:
                logger.warning("GPU model load failed, using the CPU: %s", exc)
                self._gpu_failed = True
        if self._model is None:
            self._model = self._load_model(name, "cpu")
            self.device = "cpu"
        self._model_name = name
        logger.info("Transcription model %s on %s", name, self.device)
        return self._model

    def _read_audio(self, path: Path):
        try:
            audio = self._decode(path)
        except FileNotFoundError as exc:
            raise NoAudio(f"找不到原文件：{path.name}。它可能被移动或删除了。") from exc
        except Exception as exc:
            logger.warning("Could not decode %s: %s", path.name, exc)
            raise NoAudio(f"读不出这个文件的声音：{path.name}。") from exc
        if len(audio) == 0:
            raise NoAudio(f"这个文件里没有声音：{path.name}。")
        return audio

    def transcribe(
        self,
        audio: Path,
        model: str,
        start_at: float,
        on_segment: Callable[[Segment], None],
        on_progress: Callable[[float, float], None],
    ) -> None:
        samples = self._read_audio(audio)
        duration = len(samples) / SAMPLE_RATE
        position = start_at
        while position < duration:
            engine = self._ensure_model(model)
            try:
                self._run(engine, samples, position, duration, on_segment, on_progress)
                return
            except (Stopped, ModelMissing):
                raise
            except Exception as exc:
                if self.device != "cuda" or not _is_gpu_failure(exc):
                    raise
                logger.warning("GPU failed while transcribing, continuing on the CPU: %s", exc)
                self._gpu_failed = True
                self.release()
                position = max(position, self._last_end)

    def _run(self, engine, samples, start_at, duration, on_segment, on_progress) -> None:
        self._last_end = start_at
        pieces, _ = engine.transcribe(
            samples[int(start_at * SAMPLE_RATE) :],
            language="zh",
            initial_prompt=_INITIAL_PROMPT,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        for piece in pieces:
            segment = Segment(piece.start + start_at, piece.end + start_at, piece.text.strip())
            on_segment(segment)
            self._last_end = segment.end
            on_progress(segment.end, duration)
            if self._should_stop():
                raise Stopped
