"""What a job is and how its state survives a restart."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from coursescribe.files import atomic_write_json, read_json, unique_dir

logger = logging.getLogger(__name__)


class JobKind(str, Enum):
    RECORDING = "recording"
    FILE = "file"


class Stage(str, Enum):
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    WAITING_MODEL = "waiting_model"
    PROOFREADING = "proofreading"
    WAITING_PROOFREAD = "waiting_proofread"
    DONE = "done"
    FAILED = "failed"


FINISHED_STAGES = frozenset({Stage.DONE, Stage.FAILED})
_CLOSED_CLEANLY = "closed-cleanly"


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

    @classmethod
    def new_recording(cls, root: Path, now: datetime, transcribe_model: str, proofread_model: str) -> Job:
        name = f"{now:%Y-%m-%d %H-%M} 录音"
        folder = unique_dir(root / name)
        return cls._create(JobKind.RECORDING, Stage.RECORDING, folder, name, folder / f"{name}.wav", transcribe_model, proofread_model)

    @classmethod
    def new_file(cls, source: Path, root: Path, now: datetime, transcribe_model: str, proofread_model: str) -> Job:
        del now  # file jobs are named after the source, not the time
        name = source.stem
        folder = unique_dir(root / name)
        return cls._create(JobKind.FILE, Stage.TRANSCRIBING, folder, name, source, transcribe_model, proofread_model)

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
        if self.txt_path.is_file():
            return Stage.PROOFREADING
        return Stage.RECORDING if self.kind is JobKind.RECORDING else Stage.TRANSCRIBING

    @property
    def offers_retry(self) -> bool:
        return self.stage is Stage.FAILED and self.can_retry and not self.dismissed

    def to_dict(self) -> dict:
        raw = asdict(self)
        raw["kind"] = self.kind.value
        raw["stage"] = self.stage.value
        for key in ("folder", "audio_path", "txt_path", "md_path"):
            raw[key] = str(raw[key])
        return raw

    @classmethod
    def from_dict(cls, raw: dict) -> Job:
        values = dict(raw)
        values["kind"] = JobKind(raw["kind"])
        values["stage"] = Stage(raw["stage"])
        for key in ("folder", "audio_path", "txt_path", "md_path"):
            values[key] = Path(raw[key])
        return cls(**values)


class JobStore:
    """One JSON file per job, plus that job's working files, in a private folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def save(self, job: Job) -> None:
        atomic_write_json(self.folder / f"{job.job_id}.json", job.to_dict())

    def load_all(self) -> list[Job]:
        if not self.folder.is_dir():
            return []
        jobs = []
        for path in self.folder.glob("*.json"):
            if path.name.endswith(".proofread.json"):
                continue
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

    def remove_working_files(self, job: Job) -> None:
        self.segments_path(job).unlink(missing_ok=True)
        self.proofread_path(job).unlink(missing_ok=True)
