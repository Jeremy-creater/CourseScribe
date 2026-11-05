"""What the application does. The window only shows this state and calls these methods."""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

from coursescribe.files import unique_path
from coursescribe.jobs import Job, JobKind, JobStore, SaveFolderTooLong, Stage, is_notes_file, notes_file_for, path_fits
from coursescribe.models import model_size_label
from coursescribe.notes_writer import OllamaNotesWriter
from coursescribe.pipeline import Pipeline, Proofreader, Transcriber
from coursescribe.recorder import AudioCapture, RecordingFailed, RecordingSession
from coursescribe.settings import Settings, SettingsStore
from coursescribe.wavfile import repair_wav

logger = logging.getLogger(__name__)

MEDIA_SUFFIXES = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
     ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts"}
)


def _same_file(path: Path) -> str:
    """One spelling for every way of naming the same file; a file no longer there compares by its path."""
    return os.path.normcase(os.path.realpath(path))


def _remove_if_empty(folder: Path) -> None:
    try:
        if not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass  # gone already, or not ours to clean up


SAVE_FOLDER_TOO_LONG = "保存文件夹的路径太长：{root}。请在设置里换一个路径短一些的文件夹。"

SLEEP_WARNING = "没能阻止电脑自动睡眠，录音或处理中途电脑可能会睡着。需要的话，请在系统设置里暂时关掉自动睡眠。"


class KeepAwake(Protocol):
    def hold(self) -> None: ...

    def release(self) -> None: ...


NOTES_ALREADY = "{name}：已经在整理了。"
NOTES_ALREADY_WAITING = "{name}：已经在整理了，正在等 Ollama。处理好后点“继续整理笔记”。"
NOTES_NOT_A_FILE = "没有整理：{name}（找不到这个文件，可能被移走或删除了）"
NOTES_NOT_MD = "没有整理：{name}（只能对校对稿 .md 生成整理笔记）"
NOTES_IS_NOTES = "没有整理：{name}（这已经是整理笔记了，请选它的校对稿）"
NOTES_PATH_TOO_LONG_BEFORE = (
    "没有整理：{name}（笔记文件的路径会太长：{path}）。请把校对稿挪到路径短一些的文件夹，或把它的名字改短，"
    "再生成整理笔记。校对稿没有变。"
)

_MAX_ATTEMPTS = 2
_PAUSED = "这个任务连续两次让程序意外退出，已暂停。可以点“重试”再试一次，或点“忽略”。"


@dataclass(frozen=True)
class AppState:
    recording: bool = False
    silence_warning: bool = False
    status: str = "准备好了"
    fraction: float | None = None
    notice: str = ""
    pending: int = 0
    waiting_proofread: int = 0
    waiting_notes: int = 0
    failed: int = 0
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
        keep_awake: KeepAwake,
        now: Callable[[], datetime] = datetime.now,
        notes_writer: OllamaNotesWriter | None = None,
    ) -> None:
        self.store = store
        self.settings_store = settings
        self.settings = settings.load()
        self._open_capture = open_capture
        self._download_model = download_model
        self._stopping = stopping
        self._now = now
        self._pipeline = Pipeline(
            store, transcriber, proofreader, repair_wav, on_status=self._on_job_status, notes_writer=notes_writer
        )
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._queued_ids: set[str] = set()
        self._worker = threading.Thread(target=self._work, name="jobs", daemon=True)
        self._lock = threading.RLock()
        self._listeners: list[Callable[[AppState], None]] = []
        self.state = AppState()
        self._session: RecordingSession | None = None
        self._recording_job: Job | None = None
        self._keep_awake = keep_awake
        self._awake = False  # the system has our request to stay awake
        self._closed = False
        self._sleep_warned = False  # the user hears about a failed request once per run
        self._sleep_warning_shown = False  # ...and it stays in the notices of that busy period
        self._hold_failed = False  # tried and failed in this busy period; next period tries again

    def add_listener(self, listener: Callable[[AppState], None]) -> None:
        self._listeners.append(listener)

    def _publish(self, **changes) -> None:
        with self._lock:
            state = replace(self.state, **changes)
            busy = (state.recording or state.pending > 0) and not self._closed
            self._follow_busy(busy)
            if self._sleep_warning_shown and SLEEP_WARNING not in state.notice:
                notice = f"{state.notice}\n{SLEEP_WARNING}" if state.notice else SLEEP_WARNING
                state = replace(state, notice=notice)
            elif not self._sleep_warning_shown and SLEEP_WARNING in state.notice:
                # the work is over; a notice carried forward no longer needs the warning
                notice = state.notice.replace(f"\n{SLEEP_WARNING}", "").replace(SLEEP_WARNING, "")
                state = replace(state, notice=notice)
            self.state = state
        for listener in self._listeners:
            listener(state)

    def _follow_busy(self, busy: bool) -> None:
        """Keep the computer awake exactly while recording or working. Called under the lock."""
        if busy and not self._awake and not self._hold_failed:
            try:
                self._keep_awake.hold()
                self._awake = True
            except Exception as exc:  # the work matters more; it goes on without the request
                logger.warning("Could not keep the computer awake: %s", exc)
                self._hold_failed = True
                if not self._sleep_warned:
                    self._sleep_warned = True
                    self._sleep_warning_shown = True
        elif not busy:
            self._sleep_warning_shown = False
            self._hold_failed = False
            if self._awake:
                self._awake = False
                try:
                    self._keep_awake.release()
                except Exception as exc:
                    logger.warning("Could not let the computer sleep again: %s", exc)

    def start(self) -> None:
        """Begin working, picking up whatever the last run left unfinished.

        A job that was running when the program died twice in a row is paused instead,
        or it would take the program down at every start.
        """
        closed_cleanly = self.store.take_closed_cleanly()
        self._worker.start()
        paused = []
        for job in self.store.unfinished():
            if closed_cleanly and job.attempts:
                job.attempts = 0
                self.store.save(job)
            if job.attempts >= _MAX_ATTEMPTS:
                job.stage = Stage.FAILED
                job.message = _PAUSED
                self.store.save(job)
                paused.append(job)
                continue
            self._enqueue(job)
        changes = self._waiting_counts()
        if paused:
            changes["notice"] = "、".join(job.label for job in paused) + "：" + _PAUSED
        self._publish(**changes)

    def shutdown(self) -> None:
        self._stopping.set()  # first, so the stopped recording is kept for next time, not started
        try:
            self.store.mark_closed_cleanly()  # before the waiting below, which Windows may cut short
        except OSError as exc:
            logger.warning("Could not note the clean close: %s", exc)
        with self._lock:
            self._closed = True
            self._follow_busy(False)
        self._stop_recording()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(2.0)

    def _enqueue(self, job: Job, **changes) -> None:
        """Queue a job. Other changes are published together with it, so no state in between is ever shown."""
        with self._lock:
            queued = job.job_id not in self._queued_ids
            if queued:
                self._queued_ids.add(job.job_id)
                changes["pending"] = len(self._queued_ids)
            if changes:
                self._publish(**changes)  # inside the lock, or a finishing job could publish an older count
        if queued:
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
        job.attempts += 1
        if not self._save_running(job):
            # left as it was on disk, so the next start picks it up
            self._finish_job(job, notice=f"{job.label}：{self._data_problem()}这个任务先放着，下次打开程序会接着做。")
            return
        try:
            if job.stage is Stage.WAITING_MODEL and job.model_download_requested:
                self._download(job)
            self._pipeline.run(job)
        except Exception:  # keeps the worker alive for the next job
            logger.exception("Could not process job %s", job.job_id)
        job.attempts = 0
        if self._save_running(job):
            self._finish_job(job)
        else:
            self._finish_job(job, notice=f"{job.label}：{self._data_problem()}")

    def _save_running(self, job: Job) -> bool:
        try:
            self.store.save(job)
            return True
        except OSError as exc:
            logger.warning("Could not save job %s: %s", job.job_id, exc)
            return False

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

    def _finish_job(self, job: Job, notice: str | None = None) -> None:
        """``notice`` replaces the usual one, for a problem the job's own stage does not describe."""
        changes: dict = self._waiting_counts()  # reads files: before the lock, not between count and publish
        if job.stage is Stage.DONE and notice is None:
            changes["last_folder"] = job.folder
        with self._lock:
            self._queued_ids.discard(job.job_id)
            pending = len(self._queued_ids)
            if notice is None:
                notice = self._usual_notice(job, self.state.notice)
            changes.update(pending=pending, notice=notice)
            if pending == 0:
                changes.update(status="准备好了", fraction=None)
            self._publish(**changes)

    @staticmethod
    def _usual_notice(job: Job, current: str) -> str:
        if job.stage is Stage.DONE and job.kind is JobKind.NOTES and job.notes_path is not None:
            return f"笔记已生成：{job.notes_path.name}"
        if job.stage is Stage.DONE:
            name = job.folder.name
            return f"完成：{name}。{job.message}" if job.message else f"完成：{name}"
        if job.stage is Stage.FAILED:
            return f"{job.label}：{job.message}"
        waiting = (Stage.WAITING_PROOFREAD, Stage.WAITING_MODEL, Stage.WAITING_NOTES)
        if job.stage in waiting and "下载失败" not in current:
            return job.message
        return current

    def _waiting_counts(self) -> dict:
        unfinished = self.store.unfinished()
        needing_model = [job.transcribe_model for job in unfinished if job.stage is Stage.WAITING_MODEL]
        return {
            "waiting_proofread": sum(job.stage is Stage.WAITING_PROOFREAD for job in unfinished),
            "waiting_notes": sum(job.stage is Stage.WAITING_NOTES for job in unfinished),
            "model_needed": needing_model[0] if needing_model else None,
            "failed": len(self.store.failed_to_offer()),
        }

    def _on_job_status(self, job: Job, text: str, fraction: float | None) -> None:
        self._publish(status=text, fraction=fraction)

    def add_files(self, paths: Iterable[Path]) -> None:
        """Queue media files. Anything else is refused, and the notice says why."""
        refused: list[str] = []
        no_folder: list[str] = []
        too_long: list[str] = []
        no_record: list[str] = []
        settings = self.settings
        for path in paths:
            path = Path(path)
            if not path.is_file():
                refused.append(f"{path.name}（不是文件）")
                continue
            if path.suffix.lower() not in MEDIA_SUFFIXES:
                refused.append(f"{path.name}（不支持这种格式）")
                continue
            try:
                job = Job.new_file(path, settings.output_dir, self._now(), settings.transcribe_model, settings.proofread_model)
            except SaveFolderTooLong:
                too_long.append(path.name)
                continue
            except OSError as exc:
                logger.warning("Could not create a job folder: %s", exc)
                no_folder.append(path.name)
                continue
            if self._save_new(job):
                self._enqueue(job)
            else:
                no_record.append(path.name)
        notices = []
        if too_long:
            notices.append("没有加入：" + "、".join(too_long) + "。" + SAVE_FOLDER_TOO_LONG.format(root=settings.output_dir))
        if no_folder:
            notices.append("没有加入：" + "、".join(no_folder) + "。" + self._folder_problem())
        if no_record:
            notices.append("没有加入：" + "、".join(no_record) + "。" + self._data_problem())
        if refused:
            notices.append("没有加入：" + "、".join(refused))
        if notices:
            self._publish(notice="\n".join(notices))

    def _data_problem(self) -> str:
        return f"程序自己的数据文件夹 {self.store.folder} 写不进去（可能磁盘满了，或没有权限）。"

    def _save_new(self, job: Job) -> bool:
        """Save a job just created. If that fails, the folder made for it is removed again."""
        try:
            self.store.save(job)
            return True
        except OSError as exc:
            logger.warning("Could not save new job %s: %s", job.job_id, exc)
            _remove_if_empty(job.folder)
            return False

    def _folder_problem(self) -> str:
        return (
            f"保存文件夹 {self.settings.output_dir} 打不开（可能是移动硬盘没插上，或没有权限）。"
            "请插上硬盘，或在设置里换一个保存文件夹。"
        )

    def toggle_recording(self) -> None:
        if self.state.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        settings = self.settings
        try:
            job = Job.new_recording(settings.output_dir, self._now(), settings.transcribe_model, settings.proofread_model)
        except SaveFolderTooLong:
            self._publish(notice="没有开始录音。" + SAVE_FOLDER_TOO_LONG.format(root=settings.output_dir))
            return
        except OSError as exc:
            logger.warning("Could not create a recording folder: %s", exc)
            self._publish(notice="没有开始录音。" + self._folder_problem())
            return
        if not self._save_new(job):
            self._publish(notice="没有开始录音。" + self._data_problem())
            return
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
            _remove_if_empty(job.folder)  # we made this folder moments ago
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
        stopped = {"recording": False, "silence_warning": False}
        if seconds <= 0:
            job.stage = Stage.FAILED
            job.message = "这次录音没有留下可用的声音。"
            job.can_retry = False
            self._save_running(job)
            self._publish(notice=job.message, **stopped)
            return
        # a file that could not be closed properly stays a recording, so it is repaired first
        job.stage = Stage.TRANSCRIBING if session.finished else Stage.RECORDING
        if not self._save_running(job):
            # the record still says "recording", so the next start repairs and transcribes it
            self._publish(notice=f"录音已停止。{self._data_problem()}下次打开程序时会接着转录。", **stopped)
            return
        if self._stopping.is_set():
            self._publish(**stopped)
        else:
            self._enqueue(job, **stopped)  # never "not recording and nothing to do" while handing over

    def recording_seconds(self) -> float:
        session = self._session
        return session.seconds if session else 0.0

    def _on_silence(self, alerting: bool) -> None:
        self._publish(silence_warning=alerting)

    def _on_recording_error(self, message: str) -> None:
        self._publish(notice=message)

    def _on_recording_ended(self, message: str) -> None:
        """The recording had to stop by itself: finish it as if the user had pressed Stop."""
        self._publish(notice=message)  # first: the job may finish before a later notice
        self._stop_recording()

    def report_unexpected_error(self) -> None:
        self._publish(
            notice="程序出了点意外，详情已记入日志。如果有按钮没反应，请关掉程序重新打开，没做完的任务会接着做。"
        )

    def retry_waiting(self) -> None:
        """Try again the jobs waiting for proofreading, with the model set now."""
        waiting = [job for job in self.store.unfinished() if job.stage is Stage.WAITING_PROOFREAD]
        self._change_jobs(waiting, self._retry_proofreading, queue=True)

    def retry_failed(self) -> None:
        """Run the failed jobs again from the step where each one stopped."""
        self._change_jobs(self.store.failed_to_offer(), self._retry_from_failed_step, queue=True)

    def dismiss_failed(self) -> None:
        self._change_jobs(self.store.failed_to_offer(), self._dismiss, queue=False, then=self._forget_notes_reply)

    def _retry_proofreading(self, job: Job) -> None:
        job.proofread_model = self.settings.proofread_model

    def _continue_notes(self, job: Job) -> None:
        job.notes_model = self.settings.notes_model

    def _retry_from_failed_step(self, job: Job) -> None:
        job.stage = job.retry_stage()
        job.message = ""
        job.attempts = 0
        # Without a kept reply the model is what failed, and the message says to pick another in the
        # settings. With one, the model did its part: the same model reuses it instead of writing again.
        if job.kind is JobKind.NOTES and not self.store.notes_path(job).exists():
            job.notes_model = self.settings.notes_model

    @staticmethod
    def _dismiss(job: Job) -> None:
        job.dismissed = True

    def _forget_notes_reply(self, job: Job) -> None:
        """The kept reply holds the lesson in the model's words: not kept once the job is dismissed."""
        if job.kind is JobKind.NOTES:
            self.store.remove_working_files(job)

    def _change_jobs(
        self,
        jobs: list[Job],
        change: Callable[[Job], None],
        queue: bool,
        then: Callable[[Job], None] | None = None,
    ) -> None:
        not_saved = []
        for job in jobs:
            with self._lock:
                if job.job_id in self._queued_ids:
                    continue  # the worker still has it and will save its own copy
            change(job)
            if not self._save_running(job):
                not_saved.append(job.label)
                continue
            if then is not None:
                then(job)
            if queue:
                self._enqueue(job)
        notice = "没有处理：" + "、".join(not_saved) + "。" + self._data_problem() if not_saved else ""
        self._publish(notice=notice, **self._waiting_counts())

    def make_notes(self, md_path: Path) -> None:
        """Organised notes for a proofread .md: "生成整理笔记", or a .md dropped on the window.

        Nothing is created before the job runs. The same .md has at most one notes job that is
        unfinished or offering "重试": clicking again says so, or retries the failed one.
        """
        path = Path(os.path.abspath(md_path))
        if not path.is_file():
            self._publish(notice=NOTES_NOT_A_FILE.format(name=path.name))
            return
        if path.suffix.lower() != ".md":
            self._publish(notice=NOTES_NOT_MD.format(name=path.name))
            return
        if is_notes_file(path):
            self._publish(notice=NOTES_IS_NOTES.format(name=path.name))
            return
        chosen = unique_path(notes_file_for(path))  # the name this job would write now
        if not path_fits(chosen):
            self._publish(notice=NOTES_PATH_TOO_LONG_BEFORE.format(name=path.name, path=chosen))
            return
        key = _same_file(path)
        with self._lock:  # the check and the save together, or two clicks at once make two jobs
            mine = [
                job
                for job in self.store.load_all()
                if job.kind is JobKind.NOTES and job.source_md is not None and _same_file(job.source_md) == key
            ]
            for job in mine:
                # Running, queued, or just saved as failed by the worker, which has not let go of it yet.
                if job.job_id in self._queued_ids:
                    self._publish(notice=NOTES_ALREADY.format(name=job.label))
                    return
            for job in mine:
                if job.stage in (Stage.NOTES, Stage.WAITING_NOTES):
                    template = NOTES_ALREADY_WAITING if job.stage is Stage.WAITING_NOTES else NOTES_ALREADY
                    self._publish(notice=template.format(name=job.label))
                    return
            retrying = [job for job in mine if job.offers_retry]
            if retrying:
                job = retrying[0]
                self._retry_from_failed_step(job)
            else:
                job = Job.new_notes(path, self.settings.notes_model)
            if not self._save_running(job):
                self._publish(notice=f"没有整理：{path.name}。{self._data_problem()}")
                return
            self._enqueue(job, notice="", **self._waiting_counts())

    def continue_notes(self) -> None:
        """"继续整理笔记": try again the notes jobs waiting for Ollama, with the notes model set now."""
        waiting = [job for job in self.store.unfinished() if job.stage is Stage.WAITING_NOTES]
        self._change_jobs(waiting, self._continue_notes, queue=True)

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
