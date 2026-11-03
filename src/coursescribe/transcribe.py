"""Turn audio into timed text with faster-whisper, on the GPU when possible."""

from __future__ import annotations

import errno
import gc
import logging
from collections.abc import Callable
from pathlib import Path

from coursescribe.files import _patiently
from coursescribe.pipeline import (
    MODEL_LOAD_FAILED,
    SOURCE_HELD,
    SOURCE_MISSING,
    EngineFailed,
    ModelMissing,
    NoAudio,
    Stopped,
)
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
    """Each call tries the GPU once; if it fails, the rest of that call runs on the CPU."""

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
        self.device: str | None = None  # where the last model that loaded runs; kept after release()
        self._fell_back = False
        self._caller_failed = False

    @property
    def fell_back_to_cpu(self) -> bool:
        """True when this call moved to the CPU because the GPU could not be used."""
        return self._fell_back

    def release(self) -> None:
        """Give back the model's memory, so proofreading can have the GPU."""
        self._model = None
        self._model_name = None
        gc.collect()

    def _ensure_model(self, name: str, device: str):
        if self._model is not None and self._model_name == name and self.device == device:
            return self._model
        self.release()  # a model kept from a GPU failure is dropped here, so the GPU is tried again
        self._model = self._load_model(name, device)
        self.device = device
        self._model_name = name
        logger.info("Transcription model %s on %s", name, device)
        return self._model

    def _gpu_present(self) -> bool | None:
        """What the driver says; None when asking it failed, which counts as a GPU failure."""
        try:
            return bool(self._has_gpu())
        except Exception as exc:
            logger.warning("Could not ask the graphics driver about the GPU, using the CPU: %s", str(exc))
            return None

    def _decode_or_say_why(self, path: Path):
        try:
            return self._decode(path)
        except OSError:
            raise
        except Exception as exc:
            if getattr(exc, "errno", None) != errno.EINVAL:
                raise
            # On Windows the media engine reports a missing or held file with a non-ASCII name
            # as an invalid argument; opening it here (read only) gives the real reason.
            with path.open("rb"):
                pass
            raise

    def _read_audio(self, path: Path):
        try:
            audio = _patiently(lambda: self._decode_or_say_why(path))  # another program may have it open briefly
        except FileNotFoundError as exc:
            raise NoAudio(SOURCE_MISSING.format(path=path)) from exc
        except PermissionError as exc:
            logger.warning("Source file %s is held: %s", path.name, exc)
            raise NoAudio(SOURCE_HELD.format(path=path)) from exc
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
        self._fell_back = False
        samples = self._read_audio(audio)
        duration = len(samples) / SAMPLE_RATE
        if start_at >= duration:
            return
        position = start_at
        gpu = self._gpu_present()
        if gpu:
            position = self._transcribe_on_gpu(model, samples, position, duration, on_segment, on_progress)
            if position is None:
                return
            self.release()  # the broken GPU model goes before the CPU one is loaded
            if position >= duration:
                return  # the GPU had given everything before it failed
        engine = self._load_on_cpu(model)
        if gpu is not False:
            self._fell_back = True
            on_progress(position, duration)  # so the note shows now, not after the first slow segment
        self._run(engine, samples, position, duration, on_segment, on_progress)

    def _transcribe_on_gpu(self, model, samples, start_at, duration, on_segment, on_progress) -> float | None:
        """None when the GPU did it all; otherwise where the CPU has to continue."""
        try:
            engine = self._ensure_model(model, "cuda")
        except ModelMissing:
            raise
        except Exception as exc:
            logger.warning("GPU model load failed, using the CPU: %s", str(exc))
            return start_at
        try:
            self._run(engine, samples, start_at, duration, on_segment, on_progress)
            return None
        except (Stopped, ModelMissing):
            raise
        except Exception as exc:
            if self._caller_failed or not _is_gpu_failure(exc):
                raise
            logger.warning("GPU failed while transcribing, continuing on the CPU: %s", str(exc))
        return max(start_at, self._last_end)

    def _load_on_cpu(self, model: str):
        try:
            return self._ensure_model(model, "cpu")
        except ModelMissing:
            raise
        except Exception as exc:
            logger.error("Transcription model %s could not start on the CPU", model, exc_info=True)
            raise EngineFailed(MODEL_LOAD_FAILED.format(model=model)) from exc

    def _run(self, engine, samples, start_at, duration, on_segment, on_progress) -> None:
        self._last_end = start_at
        self._caller_failed = False
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
            try:
                on_segment(segment)
                self._last_end = segment.end
                on_progress(segment.end, duration)
            except Exception:
                self._caller_failed = True  # e.g. a full disk; its path may say "CUDA" without the GPU being at fault
                raise
            if self._should_stop():
                raise Stopped
