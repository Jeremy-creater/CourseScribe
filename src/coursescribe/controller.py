"""What the application does. The window only shows this state and calls these methods."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from coursescribe.jobs import Job, JobStore, Stage
from coursescribe.models import model_size_label
from coursescribe.pipeline import Pipeline, Proofreader, Transcriber
from coursescribe.recorder import AudioCapture, RecordingFailed, RecordingSession
from coursescribe.settings import Settings, SettingsStore
from coursescribe.wavfile import repair_wav

logger = logging.getLogger(__name__)

MEDIA_SUFFIXES = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
     ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts"}
)


@dataclass(frozen=True)
class AppState:
    recording: bool = False
    silence_warning: bool = False
    status: str = "准备好了"
    fraction: float | None = None
    notice: str = ""
    pending: int = 0
    waiting_proofread: int = 0
    model_needed: str | None = None
    downloading: bool = False
    last_folder: Path | None = None


class AppController:
    def __init__(
        self,
        store: JobStore,
        settings: SettingsStore,
        transcriber: Transcriber,
        proofreader: Proofreader,
        open_capture: Callable[[], AudioCapture],
        download_model: Callable[[str], None],
        stopping: threading.Event,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.store = store
        self.settings_store = settings
        self.settings = settings.load()
        self._open_capture = open_capture
        self._download_model = download_model
        self._stopping = stopping
        self._now = now
        self._pipeline = Pipeline(store, transcriber, proofreader, repair_wav, on_status=self._on_job_status)
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._queued_ids: set[str] = set()
        self._worker = threading.Thread(target=self._work, name="jobs", daemon=True)
        self._lock = threading.RLock()
        self._listeners: list[Callable[[AppState], None]] = []
        self.state = AppState()
        self._session: RecordingSession | None = None
        self._recording_job: Job | None = None

    def add_listener(self, listener: Callable[[AppState], None]) -> None:
        self._listeners.append(listener)

    def _publish(self, **changes) -> None:
        with self._lock:
            self.state = replace(self.state, **changes)
            state = self.state
        for listener in self._listeners:
            listener(state)

    def start(self) -> None:
        """Begin working, picking up whatever the last run left unfinished."""
        self._worker.start()
        for job in self.store.unfinished():
            self._enqueue(job)

    def shutdown(self) -> None:
        self._stop_recording()
        self._stopping.set()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(2.0)

    def _enqueue(self, job: Job) -> None:
        with self._lock:
            if job.job_id in self._queued_ids:
                return
            self._queued_ids.add(job.job_id)
            pending = self.state.pending + 1
        self._publish(pending=pending)
        self._queue.put(job)

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if self._stopping.is_set():
                continue
            self._run_job(job)

    def _run_job(self, job: Job) -> None:
        try:
            if job.stage is Stage.WAITING_MODEL and job.model_download_requested:
                self._download(job)
            self._pipeline.run(job)
        except Exception:  # keeps the worker alive for the next job
            logger.exception("Could not process job %s", job.job_id)
        self._finish_job(job)

    def _download(self, job: Job) -> None:
        job.model_download_requested = False
        self._publish(downloading=True, status=f"正在下载转录模型 {job.transcribe_model}", fraction=None, notice="")
        try:
            self._download_model(job.transcribe_model)
        except Exception as exc:
            logger.warning("Model download failed: %s", exc)
            self._publish(notice="下载失败。请检查网络后再试一次。")
        finally:
            self._publish(downloading=False)

    def _finish_job(self, job: Job) -> None:
        with self._lock:
            self._queued_ids.discard(job.job_id)
            pending = self.state.pending - 1
            notice = self.state.notice
        changes: dict = {"pending": pending}
        if job.stage is Stage.DONE:
            name = job.folder.name
            notice = f"完成：{name}。{job.message}" if job.message else f"完成：{name}"
            changes["last_folder"] = job.folder
        elif job.stage is Stage.FAILED:
            notice = f"{job.folder.name}：{job.message}"
        elif job.stage in (Stage.WAITING_PROOFREAD, Stage.WAITING_MODEL) and "下载失败" not in notice:
            notice = job.message
        changes["notice"] = notice
        changes.update(self._waiting_counts())
        if pending == 0:
            changes.update(status="准备好了", fraction=None)
        self._publish(**changes)

    def _waiting_counts(self) -> dict:
        unfinished = self.store.unfinished()
        needing_model = [job.transcribe_model for job in unfinished if job.stage is Stage.WAITING_MODEL]
        return {
            "waiting_proofread": sum(job.stage is Stage.WAITING_PROOFREAD for job in unfinished),
            "model_needed": needing_model[0] if needing_model else None,
        }

    def _on_job_status(self, job: Job, text: str, fraction: float | None) -> None:
        self._publish(status=text, fraction=fraction)

    def add_files(self, paths: Iterable[Path]) -> None:
        """Queue media files. Anything else is refused, and the notice says why."""
        refused: list[str] = []
        for path in paths:
            path = Path(path)
            if not path.is_file():
                refused.append(f"{path.name}（不是文件）")
            elif path.suffix.lower() not in MEDIA_SUFFIXES:
                refused.append(f"{path.name}（不支持这种格式）")
            else:
                self._enqueue_new_file(path)
        if refused:
            self._publish(notice="没有加入：" + "、".join(refused))

    def _enqueue_new_file(self, path: Path) -> None:
        settings = self.settings
        job = Job.new_file(path, settings.output_dir, self._now(), settings.transcribe_model, settings.proofread_model)
        self.store.save(job)
        self._enqueue(job)

    def toggle_recording(self) -> None:
        if self.state.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        settings = self.settings
        job = Job.new_recording(settings.output_dir, self._now(), settings.transcribe_model, settings.proofread_model)
        self.store.save(job)
        session = RecordingSession(
            self._open_capture,
            job.audio_path,
            on_silence=self._on_silence,
            on_error=self._on_recording_error,
            on_ended=self._on_recording_ended,
        )
        try:
            session.start()
        except RecordingFailed as exc:
            job.stage = Stage.FAILED
            job.message = str(exc)
            self.store.save(job)
            if not any(job.folder.iterdir()):
                job.folder.rmdir()  # we made this empty folder moments ago
            self._publish(notice=str(exc))
            return
        self._session = session
        self._recording_job = job
        self._publish(recording=True, silence_warning=False, notice="")

    def _stop_recording(self) -> None:
        with self._lock:
            session, job = self._session, self._recording_job
            if session is None or job is None:
                return
            self._session = None
            self._recording_job = None
        seconds = session.stop()
        self._publish(recording=False, silence_warning=False)
        if seconds <= 0:
            job.stage = Stage.FAILED
            job.message = "这次录音没有留下可用的声音。"
            self.store.save(job)
            self._publish(notice=job.message)
            return
        job.stage = Stage.TRANSCRIBING
        self.store.save(job)
        if not self._stopping.is_set():
            self._enqueue(job)

    def recording_seconds(self) -> float:
        session = self._session
        return session.seconds if session else 0.0

    def _on_silence(self, alerting: bool) -> None:
        self._publish(silence_warning=alerting)

    def _on_recording_error(self, message: str) -> None:
        self._publish(notice=message)

    def _on_recording_ended(self, message: str) -> None:
        """The recording had to stop by itself: finish it as if the user had pressed Stop."""
        self._stop_recording()
        self._publish(notice=message)

    def retry_waiting(self) -> None:
        """Try again the jobs waiting for proofreading, with the model set now."""
        for job in self.store.unfinished():
            if job.stage is Stage.WAITING_PROOFREAD:
                job.proofread_model = self.settings.proofread_model
                self.store.save(job)
                self._enqueue(job)
        self._publish(notice="")

    def download_needed_model(self) -> None:
        """The user has agreed to the download; waiting jobs continue after it."""
        first = True
        for job in self.store.unfinished():
            if job.stage is Stage.WAITING_MODEL:
                job.model_download_requested = first
                first = False
                self._enqueue(job)

    def needed_model_size(self) -> str:
        return model_size_label(self.state.model_needed or "")

    def update_settings(self, settings: Settings) -> None:
        """Takes effect from the next job; jobs already created keep theirs."""
        self.settings = settings
        self.settings_store.save(settings)
