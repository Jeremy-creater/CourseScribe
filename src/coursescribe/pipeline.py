"""The one processing flow: transcribe -> write .txt -> proofread -> write .md.

Every step saves its stage first, so a restart picks up exactly where it stopped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from coursescribe.files import commit_new_text
from coursescribe.jobs import Job, JobStore, Stage
from coursescribe.segments import Segment, append_segment, read_segments, segments_to_text

logger = logging.getLogger(__name__)

StatusCallback = Callable[[Job, str, float | None], None]


class ModelMissing(Exception):
    """The transcription model is not on this computer; downloading needs consent."""

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


class NoAudio(Exception):
    """The audio cannot be read. The message is shown to the user as it is."""


class ProofreadBlocked(Exception):
    """Proofreading cannot run now. The message tells the user how to fix it."""


class Stopped(Exception):
    """The program is closing; the job continues at the next start."""


@dataclass(frozen=True)
class ProofreadResult:
    markdown: str
    unproofread_chunks: int
    total_chunks: int


class Transcriber(Protocol):
    def transcribe(
        self,
        audio: Path,
        model: str,
        start_at: float,
        on_segment: Callable[[Segment], None],
        on_progress: Callable[[float, float], None],
    ) -> None: ...

    def release(self) -> None: ...


class Proofreader(Protocol):
    def proofread(
        self, text: str, model: str, checkpoint: Path, on_progress: Callable[[int, int], None]
    ) -> ProofreadResult: ...

    def release(self, model: str) -> None: ...


class Pipeline:
    def __init__(
        self,
        store: JobStore,
        transcriber: Transcriber,
        proofreader: Proofreader,
        repair_recording: Callable[[Path], float],
        on_status: StatusCallback | None = None,
    ) -> None:
        self.store = store
        self.transcriber = transcriber
        self.proofreader = proofreader
        self.repair_recording = repair_recording
        self.on_status = on_status or (lambda job, text, fraction: None)

    def run(self, job: Job) -> None:
        """Advance ``job`` as far as it can go. Only a crash escapes this method."""
        try:
            if job.stage is Stage.RECORDING:
                self._finish_interrupted_recording(job)
            if job.stage in (Stage.TRANSCRIBING, Stage.WAITING_MODEL):
                self._transcribe(job)
            if job.stage in (Stage.PROOFREADING, Stage.WAITING_PROOFREAD):
                self._proofread(job)
        except Stopped:
            logger.info("Stopped job %s at %s", job.job_id, job.stage.value)
        except ModelMissing as exc:
            self._set(job, Stage.WAITING_MODEL, f"这台电脑上还没有转录模型 {exc.model}，下载后才能继续。")
        except ProofreadBlocked as exc:
            self._set(job, Stage.WAITING_PROOFREAD, str(exc))
        except NoAudio as exc:
            self._set(job, Stage.FAILED, str(exc))
        except Exception:
            logger.exception("Job %s failed", job.job_id)
            self._set(job, Stage.FAILED, "处理时出错了。录音和已有文件都还在，详情见日志。")

    def _set(self, job: Job, stage: Stage, message: str = "") -> None:
        job.stage = stage
        job.message = message
        self.store.save(job)
        if message:
            self.on_status(job, message, None)

    def _finish_interrupted_recording(self, job: Job) -> None:
        if self.repair_recording(job.audio_path) <= 0:
            self._set(job, Stage.FAILED, "这次录音没有留下可用的声音。")
            return
        self._set(job, Stage.TRANSCRIBING)

    def _transcribe(self, job: Job) -> None:
        checkpoint = self.store.segments_path(job)
        done = read_segments(checkpoint)
        start_at = done[-1].end if done else 0.0
        job.message = ""
        self.on_status(job, "正在准备转录", None)

        def on_progress(position: float, duration: float) -> None:
            fraction = min(position / duration, 1.0) if duration > 0 else None
            percent = f"（{fraction:.0%}）" if fraction is not None else ""
            self.on_status(job, f"正在转录{percent}", fraction)

        self.transcriber.transcribe(
            job.audio_path,
            job.transcribe_model,
            start_at,
            lambda segment: append_segment(checkpoint, segment),
            on_progress,
        )
        segments = read_segments(checkpoint)
        if not segments:
            self._set(job, Stage.FAILED, "没有识别到说话的声音，所以没有生成文字稿。")
            self.transcriber.release()
            return
        job.txt_path = commit_new_text(job.txt_path, segments_to_text(segments))
        self.transcriber.release()
        self._set(job, Stage.PROOFREADING)

    def _proofread(self, job: Job) -> None:
        text = job.txt_path.read_bytes().decode("utf-8")
        job.message = ""
        self.on_status(job, "正在准备校对", None)

        def on_progress(done: int, total: int) -> None:
            self.on_status(job, f"正在校对（第 {done}/{total} 段）", done / total if total else None)

        try:
            result = self.proofreader.proofread(
                text, job.proofread_model, self.store.proofread_path(job), on_progress
            )
        finally:
            self.proofreader.release(job.proofread_model)
        job.md_path = commit_new_text(job.md_path, f"# {job.md_path.stem}\n\n{result.markdown}")
        job.unproofread_chunks = result.unproofread_chunks
        job.total_chunks = result.total_chunks
        message = f"有 {result.unproofread_chunks} 段没校对，保留了原文。" if result.unproofread_chunks else ""
        self._set(job, Stage.DONE, message)
        self.store.remove_working_files(job)
        self.on_status(job, "完成", 1.0)
