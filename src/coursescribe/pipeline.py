"""The one processing flow: transcribe -> write .txt -> proofread -> write .md.
A notes job has a flow of its own: read the .md -> the notes writer -> write 名称-整理笔记.md.

Every step saves its stage first, so a restart picks up exactly where it stopped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from coursescribe.files import UnknownEncoding, commit_new_text, read_transcript, unique_path
from coursescribe.jobs import Job, JobKind, JobStore, Stage, notes_file_for, path_fits
from coursescribe.notes import NotesMeta, lecture_date
from coursescribe.notes_writer import NotesBlocked, NotesFailed, OllamaNotesWriter
from coursescribe.segments import Segment, append_segment, read_segments, segments_to_text
from coursescribe.vocabulary import WordListUnreadable

logger = logging.getLogger(__name__)

StatusCallback = Callable[[Job, str, float | None], None]

RECORDING_HELD = (
    "录音文件 {path} 暂时打不开（可能正被别的程序占用，比如杀毒软件或播放器）。"
    "录音还在。关掉占用它的程序，或等一会儿，再点“重试”。"
)
RECORDING_MISSING = (
    "找不到录音文件 {path}，可能被移走或删除了，或者它所在的移动硬盘没插上。"
    "把它放回原处或插回硬盘后点“重试”；不需要了就点“忽略”。"
)
RECORDING_REPAIR_FAILED = "录音文件 {path} 修复时出错（可能磁盘满了）。录音还在。腾出磁盘空间后点“重试”。"

TRANSCRIPT_MISSING = (
    "找不到文字稿 {path}，可能被移走、改名或删除了。把它放回原处后点“重试”；"
    "找不回来也可以直接点“重试”，程序会重新生成文字稿再校对。"
)
TRANSCRIPT_HELD = (
    "文字稿 {path} 暂时打不开（可能正被别的程序占用，比如正在编辑它的软件）。"
    "关掉那个程序，或等一会儿，再点“重试”。"
)
TRANSCRIPT_UNREADABLE = "读不了文字稿 {path}（可能它所在的硬盘被拔掉了，或没有权限）。处理好后点“重试”。"
TRANSCRIPT_ENCODING = (
    "文字稿 {path} 的编码认不出来（可能被另存成了别的编码）。用文本编辑器打开它，另存为 UTF-8 编码，再点“重试”。"
)
TRANSCRIPT_EMPTY = (
    "文字稿 {path} 是空的，没有可以校对的内容。把内容恢复后点“重试”；"
    "或者把这个空文件删掉再点“重试”，程序会重新生成文字稿。"
)

SOURCE_MISSING = (
    "找不到要转录的文件 {path}，可能被移走、改名或删除了。把它放回原处后点“重试”；不需要了就点“忽略”。"
)
SOURCE_HELD = (
    "要转录的文件 {path} 暂时打不开（可能正被别的程序占用）。关掉占用它的程序，或等一会儿，再点“重试”。"
)

CPU_FALLBACK_NOTE = "显卡这次没能用上，改用电脑的处理器（CPU）转录，会慢一些；下一个任务会再试显卡。"
MODEL_LOAD_FAILED = "转录模型 {model} 没能启动（可能电脑内存不够）。关掉占用内存多的程序后点“重试”；详情见日志。"


NOTES_WORKING = "正在整理笔记"
NOTES_SOURCE_MISSING = "找不到校对稿 {path}，可能被移走、改名或删除了。把它放回原处后点“重试”；不需要了就点“忽略”。"
NOTES_SOURCE_HELD = (
    "校对稿 {path} 暂时打不开（可能正被别的程序占用，比如正在编辑它的软件）。关掉那个程序，或等一会儿，再点“重试”。"
)
NOTES_SOURCE_UNREADABLE = "读不了校对稿 {path}（可能它所在的硬盘被拔掉了，或没有权限）。处理好后点“重试”。"
NOTES_SOURCE_ENCODING = (
    "校对稿 {path} 的编码认不出来（可能被另存成了别的编码）。用文本编辑器打开它，另存为 UTF-8 编码，再点“重试”。"
)
NOTES_WRITE_FAILED = (
    "笔记没能保存到 {folder}（可能磁盘满了，或这个文件夹是只读的、没有权限）。"
    "模型写好的内容已经留着，处理好后点“重试”，不用再等模型重写。校对稿没有变。"
)
NOTES_PATH_TOO_LONG = (
    "笔记文件的路径太长：{path}。请把校对稿挪到路径短一些的文件夹，或把它的名字改短，再对它生成整理笔记；"
    "也可以删掉旁边不要的旧笔记后点“重试”。校对稿没有变。"
)
NOTES_UNEXPECTED = "整理笔记时出错了。校对稿和已有文件都还在，详情见日志。可以点“重试”。"


class ModelMissing(Exception):
    """The transcription model is not on this computer; downloading needs consent."""

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


class NoAudio(Exception):
    """The audio cannot be read. The message is shown to the user as it is."""


class EngineFailed(Exception):
    """The transcription model could not start at all. The message is shown to the user as it is."""


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

    @property
    def fell_back_to_cpu(self) -> bool:
        """The current or last transcription moved to the CPU because the GPU could not be used."""
        ...


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
        notes_writer: OllamaNotesWriter | None = None,
    ) -> None:
        self.store = store
        self.notes_writer = notes_writer if notes_writer is not None else OllamaNotesWriter()
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
            if job.stage in (Stage.NOTES, Stage.WAITING_NOTES):
                self._notes(job)
        except Stopped:
            logger.info("Stopped job %s at %s", job.job_id, job.stage.value)
        except ModelMissing as exc:
            self._set(job, Stage.WAITING_MODEL, f"这台电脑上还没有转录模型 {exc.model}，下载后才能继续。")
        except ProofreadBlocked as exc:
            self._set(job, Stage.WAITING_PROOFREAD, str(exc))
        except NotesBlocked as exc:
            self._set(job, Stage.WAITING_NOTES, str(exc))
        except (NoAudio, EngineFailed, WordListUnreadable, NotesFailed) as exc:
            self._set(job, Stage.FAILED, str(exc))
        except Exception:
            logger.exception("Job %s failed", job.job_id)
            if job.kind is JobKind.NOTES:
                self._set(job, Stage.FAILED, NOTES_UNEXPECTED)
            else:
                self._set(job, Stage.FAILED, "处理时出错了。录音和已有文件都还在，详情见日志。")

    def _set(self, job: Job, stage: Stage, message: str = "", can_retry: bool = True) -> None:
        job.stage = stage
        job.message = message
        job.can_retry = can_retry
        self.store.save(job)
        if message:
            self.on_status(job, message, None)

    def _finish_interrupted_recording(self, job: Job) -> None:
        try:
            seconds = self.repair_recording(job.audio_path)
        except FileNotFoundError as error:
            logger.warning("Recording of job %s is missing: %s", job.job_id, error)
            self._set(job, Stage.FAILED, RECORDING_MISSING.format(path=job.audio_path))
            return
        except PermissionError as error:
            logger.warning("Recording of job %s is held: %s", job.job_id, error)
            self._set(job, Stage.FAILED, RECORDING_HELD.format(path=job.audio_path))
            return
        except OSError:
            logger.exception("Could not repair recording of job %s", job.job_id)
            self._set(job, Stage.FAILED, RECORDING_REPAIR_FAILED.format(path=job.audio_path))
            return
        if seconds <= 0:
            self._set(job, Stage.FAILED, "这次录音没有留下可用的声音。", can_retry=False)
            return
        self._set(job, Stage.TRANSCRIBING)

    def _transcribe(self, job: Job) -> None:
        try:
            self._transcribe_and_write(job)
        finally:
            self.transcriber.release()  # however it ended: one model at a time

    def _transcribe_and_write(self, job: Job) -> None:
        checkpoint = self.store.segments_path(job)
        done = read_segments(checkpoint)
        job.message = ""
        if job.transcript_complete and done:
            # the transcript was written once and then lost: the checkpoint holds all of it
            segments = done
        else:
            self._run_transcriber(job, checkpoint, done[-1].end if done else 0.0)
            segments = read_segments(checkpoint)
        if not segments:
            self._set(job, Stage.FAILED, "没有识别到说话的声音，所以没有生成文字稿。", can_retry=False)
            return
        job.txt_path = commit_new_text(job.txt_path, segments_to_text(segments))
        job.transcript_complete = True
        self._set(job, Stage.PROOFREADING)

    def _run_transcriber(self, job: Job, checkpoint: Path, start_at: float) -> None:
        self.on_status(job, "正在准备转录", None)

        def on_progress(position: float, duration: float) -> None:
            fraction = min(position / duration, 1.0) if duration > 0 else None
            percent = f"（{fraction:.0%}）" if fraction is not None else ""
            note = f"。{CPU_FALLBACK_NOTE}" if self.transcriber.fell_back_to_cpu else ""
            self.on_status(job, f"正在转录{percent}{note}", fraction)

        self.transcriber.transcribe(
            job.audio_path,
            job.transcribe_model,
            start_at,
            lambda segment: append_segment(checkpoint, segment),
            on_progress,
        )

    def _read_transcript(self, job: Job) -> str | None:
        """The transcript as the user left it, or None after telling them why it cannot be proofread."""
        try:
            text = read_transcript(job.txt_path)
        except (OSError, UnknownEncoding) as error:
            if isinstance(error, FileNotFoundError):
                problem = TRANSCRIPT_MISSING
            elif isinstance(error, PermissionError):
                problem = TRANSCRIPT_HELD
            elif isinstance(error, OSError):
                problem = TRANSCRIPT_UNREADABLE
            else:
                problem = TRANSCRIPT_ENCODING
            logger.warning("Transcript of job %s cannot be read: %s", job.job_id, error)
        else:
            if text.strip():
                return text
            logger.warning("Transcript of job %s is empty", job.job_id)
            problem = TRANSCRIPT_EMPTY
        self._set(job, Stage.FAILED, problem.format(path=job.txt_path))
        return None

    def _proofread(self, job: Job) -> None:
        text = self._read_transcript(job)
        if text is None:
            return
        job.message = ""
        self.on_status(job, "正在准备校对", None)

        def on_progress(done: int, total: int) -> None:
            self.on_status(job, f"正在校对（第 {done}/{total} 段）", done / total if total else None)

        try:
            result = self.proofreader.proofread(
                text, job.proofread_model, self.store.proofread_path(job), on_progress
            )
        except WordListUnreadable:
            raise  # refused before the service was asked anything: there is no model to release
        except BaseException:
            self.proofreader.release(job.proofread_model)
            raise
        self.proofreader.release(job.proofread_model)
        job.md_path = commit_new_text(job.md_path, f"# {job.md_path.stem}\n\n{result.markdown}")
        job.unproofread_chunks = result.unproofread_chunks
        job.total_chunks = result.total_chunks
        message = f"有 {result.unproofread_chunks} 段没校对，保留了原文。" if result.unproofread_chunks else ""
        self._set(job, Stage.DONE, message)
        self.store.remove_working_files(job)
        self.on_status(job, "完成", 1.0)

    def _read_source(self, job: Job) -> str | None:
        """The proofread .md as the user left it, or None after telling them why it cannot be read."""
        try:
            return read_transcript(job.source_md)
        except (OSError, UnknownEncoding) as error:
            if isinstance(error, FileNotFoundError):
                problem = NOTES_SOURCE_MISSING
            elif isinstance(error, PermissionError):
                problem = NOTES_SOURCE_HELD
            elif isinstance(error, OSError):
                problem = NOTES_SOURCE_UNREADABLE
            else:
                problem = NOTES_SOURCE_ENCODING
            logger.warning("The proofread text of notes job %s cannot be read: %s", job.job_id, error)
        self._set(job, Stage.FAILED, problem.format(path=job.source_md))
        return None

    def _notes(self, job: Job) -> None:
        """Organised notes beside the .md. The .md and everything already there stay as they are."""
        job.message = ""
        self.on_status(job, NOTES_WORKING, None)
        source = job.source_md
        text = self._read_source(job)
        if text is None:
            return
        # The day the job was made, kept in the job: a restart after midnight gives the same notes.
        meta = NotesMeta(source.stem, lecture_date(source.stem, date.fromtimestamp(job.created_ns / 1e9)))
        try:
            result = self.notes_writer.write(text, meta, job.notes_model, self.store.notes_path(job))
        except NotesFailed:
            # A usable kept reply never ends here, so what is left is one for other input. Kept, it
            # would make "重试" take the model as done and never use the one set now.
            self._forget_notes_reply(job)
            raise
        if job.notes_path is None:
            if not source.parent.is_dir():
                # Chosen in a folder that is gone, the base name would be free; once the folder is
                # back, older notes the same byte for byte would pass as this job's.
                self._set(job, Stage.FAILED, NOTES_SOURCE_MISSING.format(path=source))
                return
            # Chosen only now and saved before writing: notes made again go to "(2)" even when the
            # same word for word, while this job, restarted, writes to the same name and keeps one file.
            chosen = unique_path(notes_file_for(source))
            if not path_fits(chosen):
                self._set(job, Stage.FAILED, NOTES_PATH_TOO_LONG.format(path=chosen))
                return
            job.notes_path = chosen
            self._set(job, Stage.NOTES)
        if not source.parent.is_dir():
            # commit_new_text would make the folder again, holding nothing but the notes
            self._set(job, Stage.FAILED, NOTES_SOURCE_MISSING.format(path=source))
            return
        try:
            job.notes_path = commit_new_text(job.notes_path, result.markdown)
        except OSError as error:
            logger.warning("Could not save the notes of job %s: %s", job.job_id, error)
            self._set(job, Stage.FAILED, NOTES_WRITE_FAILED.format(folder=source.parent))
            return
        job.removed_quotes = result.removed_quotes
        job.removed_concepts = result.removed_concepts
        job.rules_note = result.rules_note
        self._set(job, Stage.DONE, "")
        self.store.remove_working_files(job)
        self.on_status(job, "完成", 1.0)

    def _forget_notes_reply(self, job: Job) -> None:
        checkpoint = self.store.notes_path(job)
        try:
            checkpoint.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove the notes checkpoint %s: %s", checkpoint.name, exc)
