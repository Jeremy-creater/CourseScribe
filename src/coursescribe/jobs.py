"""What a job is and how its state survives a restart."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from coursescribe.files import atomic_write_json, read_json, unique_dir
from coursescribe.platform import LIMITS

logger = logging.getLogger(__name__)


class JobKind(str, Enum):
    RECORDING = "recording"
    FILE = "file"
    NOTES = "notes"


class Stage(str, Enum):
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    WAITING_MODEL = "waiting_model"
    PROOFREADING = "proofreading"
    WAITING_PROOFREAD = "waiting_proofread"
    NOTES = "notes"
    WAITING_NOTES = "waiting_notes"
    DONE = "done"
    FAILED = "failed"


FINISHED_STAGES = frozenset({Stage.DONE, Stage.FAILED})
_CLOSED_CLEANLY = "closed-cleanly"

NAME_LIMIT = 80
PATH_LIMIT = 259  # Windows MAX_PATH, less the terminating null
MIN_NAME = 20
FALLBACK_NAME = "未命名"
# Longest path a job makes: root\name (99)\name (99).txt = root + 2 * name + 16
_NAME_BUDGET = PATH_LIMIT - 16
_EXTENDERS = frozenset("\u200d\ufe0e\ufe0f")


class SaveFolderTooLong(Exception):
    """The save folder leaves too little room in a path for any job name. Not an OSError on purpose."""

    def __init__(self, root: Path) -> None:
        super().__init__(f"Save folder path is too long for job names: {root}")
        self.root = root


def _units(text: str) -> int:
    """Length as Windows counts it, in UTF-16 units."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _bytes(text: str) -> int:
    return len(text.encode("utf-8", "surrogateescape"))


NOTES_SUFFIX = "-整理笔记"
# The longest name a job name ends up in: <name>-整理笔记 (99).md
_NOTES_TAIL_BYTES = _bytes(f"{NOTES_SUFFIX} (99).md")


def path_fits(path: Path) -> bool:
    """Whether the system can store ``path``: Windows counts the whole path, the Mac each name and the whole path."""
    text = str(path)
    limits = LIMITS
    if limits.path_units is not None and _units(text) > limits.path_units:
        return False
    if limits.path_bytes is not None and _bytes(text) > limits.path_bytes:
        return False
    if limits.component_bytes is not None and any(_bytes(part) > limits.component_bytes for part in path.parts):
        return False
    return True


def _byte_budget(root: Path) -> int | None:
    """The most UTF-8 bytes a job name may have on the Mac so every name and the longest path fit; None elsewhere."""
    budgets = []
    if LIMITS.component_bytes is not None:
        budgets.append(LIMITS.component_bytes - _NOTES_TAIL_BYTES)
    if LIMITS.path_bytes is not None:
        budgets.append((LIMITS.path_bytes - _bytes(str(root)) - 2 - _NOTES_TAIL_BYTES) // 2)  # root/name/name+tail
    return min(budgets) if budgets else None


_NOTES_NAME = re.compile(re.escape(NOTES_SUFFIX) + r"( \(\d+\))?\.md$", re.IGNORECASE)


def notes_file_for(source_md: Path) -> Path:
    """Where the organised notes of a proofread ``.md`` go: beside it, ``名称-整理笔记.md``."""
    return source_md.with_name(f"{source_md.stem}{NOTES_SUFFIX}.md")


def is_notes_file(path: Path) -> bool:
    """Whether ``path`` is named like organised notes (``…-整理笔记.md``, ``…-整理笔记 (2).md``)."""
    return _NOTES_NAME.search(path.name) is not None


def _extends(char: str) -> bool:
    """Whether ``char`` belongs to the character before it: marks, joiners, variation and skin tone."""
    return unicodedata.combining(char) > 0 or char in _EXTENDERS or "\U0001f3fb" <= char <= "\U0001f3ff"


def job_name(raw: str, root: Path) -> str:
    """The name for a job's folder and files: ``raw`` cut so every path the job makes fits."""
    limit = NAME_LIMIT
    if LIMITS.path_units is not None:
        limit = min(limit, (LIMITS.path_units - 16 - _units(str(root))) // 2)
    byte_limit = _byte_budget(root)
    if limit < MIN_NAME or (byte_limit is not None and byte_limit < 3 * MIN_NAME):  # 3 bytes: a Chinese character
        raise SaveFolderTooLong(root)
    name = raw
    if _units(raw) > limit or (byte_limit is not None and _bytes(raw) > byte_limit):
        end = units = size = 0
        for char in raw:
            units += _units(char)
            size += _bytes(char)
            if units > limit or (byte_limit is not None and size > byte_limit):
                break
            end += 1
        # never split a character from its marks; a space or dot cut off here could expose a joiner
        while end and (_extends(raw[end]) or raw[end - 1] in "\u200d ."):
            end -= 1
        name = raw[:end]
    return name.rstrip(" .") or FALLBACK_NAME  # Windows drops these from the end of a name


_last_stamp = 0
_stamp_lock = threading.Lock()


def _next_creation_stamp() -> int:
    """Strictly increasing, because the system clock can repeat between two calls."""
    global _last_stamp
    with _stamp_lock:
        _last_stamp = max(_last_stamp + 1, time.time_ns())
        return _last_stamp


@dataclass
class Job:
    job_id: str
    kind: JobKind
    stage: Stage
    folder: Path
    audio_path: Path
    txt_path: Path
    md_path: Path
    transcribe_model: str
    proofread_model: str
    created_ns: int
    message: str = ""
    unproofread_chunks: int = 0
    total_chunks: int = 0
    model_download_requested: bool = False
    attempts: int = 0  # runs started and not ended normally; more than one means it crashed the program
    can_retry: bool = True
    dismissed: bool = False
    # the whole transcription is in the checkpoint, so a lost .txt is written again without transcribing
    transcript_complete: bool = False
    source_md: Path | None = None  # a notes job's proofread .md, as a full path
    notes_path: Path | None = None  # chosen just before the notes are written, and saved first
    notes_model: str = ""
    removed_quotes: int = 0
    removed_concepts: int = 0
    rules_note: str | None = None

    @classmethod
    def new_recording(cls, root: Path, now: datetime, transcribe_model: str, proofread_model: str) -> Job:
        name = job_name(f"{now:%Y-%m-%d %H-%M} 录音", root)
        folder = unique_dir(root / name)
        return cls._create(JobKind.RECORDING, Stage.RECORDING, folder, name, folder / f"{name}.wav", transcribe_model, proofread_model)

    @classmethod
    def new_file(cls, source: Path, root: Path, now: datetime, transcribe_model: str, proofread_model: str) -> Job:
        del now  # file jobs are named after the source, not the time
        name = job_name(source.stem, root)
        folder = unique_dir(root / name)
        return cls._create(JobKind.FILE, Stage.TRANSCRIBING, folder, name, source, transcribe_model, proofread_model)

    @classmethod
    def new_notes(cls, source_md: Path, notes_model: str) -> Job:
        """A job that only notes the .md down: nothing is created, read or written here.

        A notes job works on ``source_md`` and ``notes_path`` only; the other paths are filled
        in so code that shows any job (its folder, its name) needs no special case.
        """
        source = Path(os.path.abspath(source_md))
        job = cls._create(JobKind.NOTES, Stage.NOTES, source.parent, source.stem, source, "", "")
        job.md_path = source
        job.source_md = source
        job.notes_model = notes_model
        return job

    @classmethod
    def _create(
        cls,
        kind: JobKind,
        stage: Stage,
        folder: Path,
        name: str,
        audio: Path,
        transcribe_model: str,
        proofread_model: str,
    ) -> Job:
        return cls(
            job_id=uuid.uuid4().hex,
            kind=kind,
            stage=stage,
            folder=folder,
            audio_path=audio,
            txt_path=folder / f"{name}.txt",
            md_path=folder / f"{name}.md",
            transcribe_model=transcribe_model,
            proofread_model=proofread_model,
            created_ns=_next_creation_stamp(),
        )

    def retry_stage(self) -> Stage:
        """Where a failed job picks up again: keep the transcript if there is one."""
        if self.kind is JobKind.NOTES:
            return Stage.NOTES  # the .txt beside a proofread .md is not this job's to proofread
        if self.txt_path.is_file():
            return Stage.PROOFREADING
        return Stage.RECORDING if self.kind is JobKind.RECORDING else Stage.TRANSCRIBING

    @property
    def label(self) -> str:
        """What the notices call this job: the .md of a notes job, the folder of the others."""
        if self.kind is JobKind.NOTES and self.source_md is not None:
            return self.source_md.name
        return self.folder.name

    @property
    def offers_retry(self) -> bool:
        return self.stage is Stage.FAILED and self.can_retry and not self.dismissed

    def to_dict(self) -> dict:
        raw = asdict(self)
        raw["kind"] = self.kind.value
        raw["stage"] = self.stage.value
        for key in ("folder", "audio_path", "txt_path", "md_path"):
            raw[key] = str(raw[key])
        for key in _OPTIONAL_PATHS:
            raw[key] = None if raw[key] is None else str(raw[key])
        return raw

    @classmethod
    def from_dict(cls, raw: dict) -> Job:
        values = dict(raw)
        values["kind"] = JobKind(raw["kind"])
        values["stage"] = Stage(raw["stage"])
        for key in ("folder", "audio_path", "txt_path", "md_path"):
            values[key] = Path(raw[key])
        for key in _OPTIONAL_PATHS:
            if values.get(key) is not None:  # older versions wrote no such key: the default stands
                values[key] = Path(values[key])
        return cls(**values)


_OPTIONAL_PATHS = ("source_md", "notes_path")
_WORKING_FILES = (".proofread.json", ".notes.json")


class JobStore:
    """One JSON file per job, plus that job's working files, in a private folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def save(self, job: Job) -> None:
        atomic_write_json(self.folder / f"{job.job_id}.json", job.to_dict())

    def load(self, job_id: str) -> Job | None:
        """One job as it is on disk now; None when it is gone or damaged (damaged is logged, as in load_all)."""
        path = self.folder / f"{job_id}.json"
        try:
            return Job.from_dict(read_json(path))
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Ignoring damaged job state %s: %s", path.name, exc)
            return None

    def load_all(self) -> list[Job]:
        if not self.folder.is_dir():
            return []
        jobs = []
        for path in self.folder.glob("*.json"):
            if path.name.endswith(_WORKING_FILES):
                continue  # a job's working file, not a job
            try:
                jobs.append(Job.from_dict(read_json(path)))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                logger.warning("Ignoring damaged job state %s: %s", path.name, exc)
        return sorted(jobs, key=lambda job: job.created_ns)

    def unfinished(self) -> list[Job]:
        return [job for job in self.load_all() if job.stage not in FINISHED_STAGES]

    def failed_to_offer(self) -> list[Job]:
        return [job for job in self.load_all() if job.offers_retry]

    def mark_closed_cleanly(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / _CLOSED_CLEANLY).touch()

    def take_closed_cleanly(self) -> bool:
        """Whether the program was closed normally last time. Asking clears the answer."""
        marker = self.folder / _CLOSED_CLEANLY
        if not marker.exists():
            return False
        marker.unlink(missing_ok=True)
        return True

    def segments_path(self, job: Job) -> Path:
        return self.folder / f"{job.job_id}.segments.jsonl"

    def proofread_path(self, job: Job) -> Path:
        return self.folder / f"{job.job_id}.proofread.json"

    def notes_path(self, job: Job) -> Path:
        return self.folder / f"{job.job_id}.notes.json"

    def remove_working_files(self, job: Job) -> None:
        """Each file on its own: the job is finished, so one that cannot go is only logged."""
        for path in (self.segments_path(job), self.proofread_path(job), self.notes_path(job)):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove the working file %s: %s", path.name, exc)
