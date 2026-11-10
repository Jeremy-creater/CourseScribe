"""Notes in the settings (12-7): the lock around "继续/重试/忽略", organised notes made by themselves
once a proofread .md is written, settings that cannot be saved, and the program's real assembly.

The service is the shared FakeOllama (a real local HTTP server) talked to by the real notes writer;
the only other stand-ins sit at the system's edge (the transcription engine and proofreader doubles
already used by the controller tests, a refused new file, a failed job record, the power state)."""

from __future__ import annotations

import copy
import logging
import random
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from coursescribe.controller import (
    NOTES_ALREADY,
    NOTES_ALREADY_WAITING,
    NOTES_NOT_A_FILE,
    NOTES_NOT_AUTO,
    NOTES_PATH_TOO_LONG_BEFORE,
    NOTES_UNREACHABLE,
    SETTINGS_NOT_SAVED,
)
from coursescribe.jobs import NOTES_SUFFIX, Job, JobKind, JobStore, Stage, is_notes_file
from coursescribe.notes_rules import RULES_UNREADABLE
from coursescribe.notes_writer import NOTES_BUSY_MESSAGE, NOTES_MODEL_MISSING, NOTES_REFUSED
from coursescribe.pipeline import NOTES_WORKING, ProofreadBlocked, ProofreadResult
from coursescribe.settings import Settings, SettingsStore
from coursescribe.vocabulary import WordListUnreadable
from test_controller import FakeKeepAwake, _fits, _job_records_fail, _saved_job, _snapshot, _u16
from test_controller import FakeProofreader as RigProofreader
from test_notes import SRC
from test_notes_job import (  # noqa: F401  (ollama, svc and rigs are fixtures)
    FINISHED_BYTES,
    NAME,
    NOTES_FILE,
    SRC_BYTES,
    DiskFault,
    TimedTranscriber,
    gated,
    lesson,
    names,
    notes_files,
    notes_jobs,
    ollama,
    rigs,
    same_file,
    set_notes_model,
    svc,
    temp_files,
)
from test_notes_writer import BUSY, GOOD, REFUSED
from test_pipeline import Crash

BODY = SRC.split("\n\n", 1)[1]
LESSON_FILE = f"{NAME}.mp3"


# ---------------------------------------------------------------- tools


class LessonProofreader(RigProofreader):
    """The proofreading double: whatever it is given, it answers with the body of SRC, so a file
    job named after NAME writes an .md byte for byte SRC_BYTES. It keeps what it was given."""

    def __init__(self, unproofread: int = 0, total: int = 1) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.spans: list[tuple[float, float]] = []
        self.unproofread = unproofread
        self.total = total

    def proofread(self, text, model, checkpoint, on_progress):
        start = time.perf_counter()
        self.texts.append(text)
        self.models.append(model)
        if self.gate:
            self.gate.wait(5)
        if self.error:
            raise self.error
        if self.blocked:
            raise ProofreadBlocked(self.blocked)
        on_progress(1, 1)
        self.spans.append((start, time.perf_counter()))
        return ProofreadResult(BODY, self.unproofread, self.total)


def set_auto(rig, on: bool) -> None:
    rig.controller.update_settings(replace(rig.controller.settings, auto_notes=on))


def record_briefly(rig, seconds: float = 0.3) -> None:
    rig.controller.toggle_recording()
    assert rig.controller.state.recording is True
    rig.wait(lambda: rig.controller.recording_seconds() >= seconds)
    rig.controller.toggle_recording()


def wait_notes(rig, count: int) -> None:
    """Exactly ``count`` notes jobs, each finished or waiting, and nothing queued; 10 seconds at most."""

    def ready() -> bool:
        jobs = notes_jobs(rig.store)
        settled = all(job.stage in (Stage.DONE, Stage.FAILED, Stage.WAITING_NOTES) for job in jobs)
        return len(jobs) == count and settled and rig.controller.state.pending == 0

    rig.wait(ready, seconds=10.0)


def at_status(rig, text: str, action) -> list[bool]:
    """Run ``action`` once, on the thread that publishes the first state whose status is ``text``."""
    fired: list[bool] = []

    def listener(state) -> None:
        if not fired and state.status == text:
            fired.append(True)
            action()

    rig.controller.add_listener(listener)
    return fired


class one_shot:  # noqa: N801  used like a function
    """Replace ``store.<name>`` so ``hook`` runs once, on the first call from this (the main) thread.

    ``failed_to_offer``/``unfinished``/``load``: after the real call has read, before it returns.
    ``save``: before the real save, and only for a job ``when`` accepts. ``fired`` counts the hook
    runs; ``returned`` is a deep copy of what that call returned (``save``: of the job to be saved)."""

    def __init__(self, store, name: str, hook, when=lambda *args: True) -> None:
        self.fired = 0
        self.returned = None
        main = threading.current_thread()
        real = getattr(store, name)

        def mine(args) -> bool:
            return self.fired == 0 and threading.current_thread() is main and when(*args)

        def wrapper(*args):
            if name == "save":
                if mine(args):
                    self.fired += 1
                    self.returned = copy.deepcopy(args[0])
                    hook()
                return real(*args)
            result = real(*args)
            if mine(args):
                self.fired += 1
                self.returned = copy.deepcopy(result)
                hook()
            return result

        setattr(store, name, wrapper)


def on_thread(action, then=None) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def work() -> None:
        try:
            action()
            if then is not None:
                then()
        except BaseException as exc:  # handed back to the test
            errors.append(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread, errors


def data_problem(rig) -> str:
    return f"程序自己的数据文件夹 {rig.store.folder} 写不进去（可能磁盘满了，或没有权限）。"


def record(rig, job_id: str) -> Job | None:
    """The record on disk, read the way the program lists jobs."""
    return next((job for job in rig.store.load_all() if job.job_id == job_id), None)


def source_jobs(rig) -> list[Job]:
    return [job for job in rig.store.load_all() if job.kind is not JobKind.NOTES]


def first_index(states, test) -> int:
    return next(i for i, state in enumerate(states) if test(state))


def lesson_rig(rigs, root=None, **kwargs):
    kwargs.setdefault("proofreader", LessonProofreader())
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    rig = rigs(root, **kwargs)
    rig.controller.start()
    return rig


def generation_times(fake) -> list[float]:
    return [when for when, path, prompt in fake.log if prompt is not None]


# ---------------------------------------------------------------- L: the lock around changing listed jobs


def _failed_notes(rig, md, svc) -> Job:
    rig.controller.make_notes(md)
    rig.wait_idle()
    jobs = notes_jobs(rig.store)
    assert [job.stage for job in jobs] == [Stage.FAILED]
    return jobs[0]


@pytest.mark.parametrize("row", ["retry", "dismiss", "continue-proofreading", "continue-notes", "dismissed-first",
                                 "reread-copy", "now-waiting"])
def test_another_action_after_the_list_was_read_is_never_undone_by_the_old_copy(tmp_path, svc, rigs, row):
    """L1 (a)-(g)."""
    md = lesson(tmp_path)
    inserted: dict = {}

    def insert(action):
        def hook():
            thread, errors = on_thread(action, rig.wait_idle)
            inserted.update(thread=thread, errors=errors)
            thread.join(2)

        return hook

    if row in ("retry", "dismiss", "dismissed-first"):
        svc.serve(REFUSED, then=GOOD)
        rig = lesson_rig(rigs)
        job = _failed_notes(rig, md, svc)
        action = rig.controller.dismiss_failed if row == "dismissed-first" else lambda: rig.controller.make_notes(md)
        shot = one_shot(rig.store, "failed_to_offer", insert(action))
        (rig.controller.dismiss_failed if row == "dismiss" else rig.controller.retry_failed)()
        rig.wait_idle()
        before_stage = Stage.FAILED
    elif row == "now-waiting":
        svc.serve(REFUSED, BUSY, BUSY)
        rig = lesson_rig(rigs)
        job = _failed_notes(rig, md, svc)
        shot = one_shot(rig.store, "failed_to_offer", insert(lambda: rig.controller.make_notes(md)))
        rig.controller.retry_failed()
        rig.wait_idle()
        before_stage = Stage.FAILED
    elif row == "continue-notes":
        svc.serve(BUSY, BUSY, then=GOOD)
        rig = lesson_rig(rigs)
        rig.controller.make_notes(md)
        rig.wait_idle()
        job = notes_jobs(rig.store)[0]
        assert job.stage is Stage.WAITING_NOTES
        shot = one_shot(rig.store, "unfinished", insert(rig.controller.continue_notes))
        rig.controller.continue_notes()
        rig.wait_idle()
        before_stage = Stage.WAITING_NOTES
    elif row == "continue-proofreading":
        rig = lesson_rig(rigs)
        set_auto(rig, True)
        rig.proofreader.blocked = "没有启动"
        rig.controller.add_files([rig.source(LESSON_FILE)])
        rig.wait_idle()
        job = source_jobs(rig)[0]
        assert job.stage is Stage.WAITING_PROOFREAD
        rig.proofreader.blocked = None
        shot = one_shot(rig.store, "unfinished", insert(rig.controller.retry_waiting))
        rig.controller.retry_waiting()
        wait_notes(rig, 1)
        before_stage = Stage.WAITING_PROOFREAD
    else:  # reread-copy
        rig = lesson_rig(rigs)
        rig.proofreader.error = RuntimeError("坏了")
        rig.controller.add_files([rig.source(LESSON_FILE)])
        rig.wait_idle()
        job = source_jobs(rig)[0]
        assert job.stage is Stage.FAILED and job.offers_retry and job.txt_path.is_file()
        original = job.txt_path.read_text(encoding="utf-8")
        rig.proofreader.error = None
        second = job.folder / f"{NAME} (2).txt"

        def change_it():
            second.write_bytes("第二份文字稿\n".encode("utf-8"))
            other = JobStore(rig.store.folder)
            on_disk = next(j for j in other.load_all() if j.job_id == job.job_id)
            other.save(replace(on_disk, txt_path=second))

        shot = one_shot(rig.store, "failed_to_offer", change_it)
        rig.controller.retry_failed()
        rig.wait_idle()
        before_stage = Stage.FAILED

    assert shot.fired == 1
    listed = [j for j in shot.returned if j.job_id == job.job_id]
    assert [j.stage for j in listed] == [before_stage]
    if "thread" in inserted:
        inserted["thread"].join(5)
        assert not inserted["thread"].is_alive()
        assert inserted["errors"] == []
    assert rig.controller.state.pending == 0
    gens = len(svc.gens)
    state = rig.controller.state

    if row == "retry":
        jobs = notes_jobs(rig.store)
        assert [(j.stage, j.notes_path.name) for j in jobs] == [(Stage.DONE, NOTES_FILE)]
        assert notes_files(md) == [NOTES_FILE]
        assert gens == 2
    elif row == "dismiss":
        saved = record(rig, job.job_id)
        assert (saved.stage, saved.dismissed) == (Stage.DONE, False)
        assert notes_files(md) == [NOTES_FILE] and (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
        assert state.failed == 0
    elif row == "continue-proofreading":
        assert len(rig.proofreader.models) == 2
        assert record(rig, job.job_id).stage is Stage.DONE
        made = record(rig, job.job_id).md_path
        assert len(notes_jobs(rig.store)) == 1 and notes_files(made) == [NOTES_FILE]
    elif row == "continue-notes":
        assert notes_files(md) == [NOTES_FILE]
        assert record(rig, job.job_id).stage is Stage.DONE
    elif row == "dismissed-first":
        saved = record(rig, job.job_id)
        assert (saved.stage, saved.dismissed) == (Stage.FAILED, True)
        assert gens == 1
        assert state.failed == 0
        assert notes_files(md) == []
    elif row == "reread-copy":
        saved = record(rig, job.job_id)
        assert saved.txt_path.name == f"{NAME} (2).txt"
        assert "第二份文字稿" in rig.proofreader.texts[-1]
        assert original.strip() not in rig.proofreader.texts[-1]
        assert saved.stage is Stage.DONE
    else:  # now-waiting
        assert record(rig, job.job_id).stage is Stage.WAITING_NOTES
        assert gens == 3
        assert (state.waiting_notes, state.failed) == (1, 0)
        assert notes_files(md) == []


def _armed_service(svc) -> tuple[threading.Event, threading.Event]:
    """After a refusal, the service answers GOOD at once; once armed, only after the gate opens."""
    armed, gate = threading.Event(), threading.Event()

    def respond(prompt):
        if armed.is_set():
            gate.wait(5)
        return GOOD

    svc.fake.respond = respond
    return armed, gate


def _race_make_notes(rig, md, svc, inserted: dict, armed: threading.Event, job_id: str):
    def hook():
        thread, errors = on_thread(lambda: rig.controller.make_notes(md))
        inserted.update(thread=thread, errors=errors)
        deadline = time.time() + 2
        done = False
        while time.time() < deadline:
            record = rig.store.load_all()
            done = not thread.is_alive() and any(j.job_id == job_id and j.stage is Stage.DONE for j in record)
            if done:
                break
            time.sleep(0.01)
        if done:
            rig.wait_idle()
        else:
            armed.set()  # waiting for the lock: the worker must not finish before the thread gets it

    return hook


def _idle_with_notes_on_disk(rig) -> list:
    seen = []

    def listener(state):
        if state.pending == 0:
            seen.append([j.stage for j in rig.store.load_all() if j.kind is JobKind.NOTES])

    rig.controller.add_listener(listener)
    return seen


@pytest.mark.parametrize("moment", ["before-saving", "after-reading-again"])
def test_a_click_between_checking_and_queuing_waits_for_the_lock(tmp_path, svc, rigs, moment):
    """L2 (before the save) and L2 (b) (after the record is read again)."""
    svc.serve(REFUSED)
    md = lesson(tmp_path)
    rig = lesson_rig(rigs)
    job = _failed_notes(rig, md, svc)
    armed, gate = _armed_service(svc)
    inserted: dict = {}
    hook = _race_make_notes(rig, md, svc, inserted, armed, job.job_id)
    idle = _idle_with_notes_on_disk(rig)
    if moment == "before-saving":
        shot = one_shot(rig.store, "save", hook, when=lambda j: j.kind is JobKind.NOTES and j.stage is Stage.NOTES)
    else:
        shot = one_shot(rig.store, "load", hook)
    rig.controller.retry_failed()
    thread = inserted.get("thread")
    if thread is not None:
        thread.join(5)
    gate.set()
    rig.wait_idle()

    assert shot.fired == 1
    assert not thread.is_alive()
    assert shot.returned.job_id == job.job_id
    assert shot.returned.stage is (Stage.NOTES if moment == "before-saving" else Stage.FAILED)
    assert inserted["errors"] == []
    jobs = notes_jobs(rig.store)
    assert [(j.stage, j.notes_path.name) for j in jobs] == [(Stage.DONE, NOTES_FILE)]
    assert notes_files(md) == [NOTES_FILE]
    assert len(svc.gens) == 2
    assert NOTES_ALREADY.format(name=md.name) in [state.notice for state in rig.states]
    assert not any(Stage.NOTES in stages for stages in idle)


@pytest.mark.parametrize("row", ["already", "already-waiting", "not-recorded"])
def test_a_refusal_to_make_notes_is_published_after_the_lock_is_let_go(tmp_path, svc, rigs, row):
    """L3 (a)-(c)."""
    md = lesson(tmp_path)
    gate = None
    if row == "already":
        gate = gated(svc)
    elif row == "already-waiting":
        svc.serve(BUSY, BUSY)
    rig = lesson_rig(rigs)
    if row == "already":
        rig.controller.make_notes(md)
        rig.wait(lambda: len(svc.gens) == 1)
        refusal = NOTES_ALREADY.format(name=md.name)
    elif row == "already-waiting":
        rig.controller.make_notes(md)
        rig.wait_idle()
        refusal = NOTES_ALREADY_WAITING.format(name=md.name)
    else:
        _job_records_fail(rig, lambda j: j.kind is JobKind.NOTES)
        refusal = f"没有整理：{md.name}。{data_problem(rig)}"

    main = threading.current_thread()
    other: dict = {}

    def listener(state):
        if not other and threading.current_thread() is main and state.notice == refusal:
            thread, errors = on_thread(lambda: rig.controller.make_notes(tmp_path / "没有.md"))
            other.update(thread=thread, errors=errors)
            thread.join(1.0)
            other["in_time"] = not thread.is_alive()

    rig.controller.add_listener(listener)
    rig.controller.make_notes(md)
    assert "thread" in other
    other["thread"].join(5)
    assert not other["thread"].is_alive()
    assert other["in_time"] is True
    assert other["errors"] == []
    assert rig.controller.state.notice == NOTES_NOT_A_FILE.format(name="没有.md")
    assert len(notes_jobs(rig.store)) == (0 if row == "not-recorded" else 1)
    if gate is not None:
        gate.set()
    rig.wait_idle()


def test_a_record_gone_damaged_or_still_with_the_worker_is_skipped(tmp_path, svc, rigs, caplog):
    """L4 (a)-(d)."""
    # (a) and (b): the record is deleted or damaged after the list was read
    for row in ("deleted", "damaged"):
        root = tmp_path / row
        svc.serve(REFUSED, then=GOOD)
        md = lesson(root)
        rig = lesson_rig(rigs, root)
        job = _failed_notes(rig, md, svc)
        job_file = rig.store.folder / f"{job.job_id}.json"

        def spoil(job_file=job_file, row=row):
            if row == "deleted":
                job_file.unlink()
            else:
                job_file.write_bytes(b"{")

        asked = len(svc.gens)
        shot = one_shot(rig.store, "failed_to_offer", spoil)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="coursescribe.jobs"):
            rig.controller.retry_failed()
            rig.wait_idle()
        assert shot.fired == 1, row
        if row == "deleted":
            assert not job_file.exists()
        else:
            assert job_file.read_bytes() == b"{"
            assert any(r.name == "coursescribe.jobs" and job_file.name in r.getMessage() for r in caplog.records)
        assert len(svc.gens) == asked, row
        assert notes_files(md) == [], row

    # (c): "忽略" while the worker still holds the job it has just saved as failed
    root = tmp_path / "held"
    svc.serve(REFUSED, then=GOOD)
    md = lesson(root)
    rig = lesson_rig(rigs, root)
    other: dict = {}

    def dismiss_now():
        thread, errors = on_thread(rig.controller.dismiss_failed)
        other.update(thread=thread, errors=errors)
        thread.join(2)

    at_status(rig, NOTES_REFUSED, dismiss_now)
    rig.controller.make_notes(md)
    rig.wait(lambda: "thread" in other)
    other["thread"].join(5)
    assert not other["thread"].is_alive() and other["errors"] == []
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert (job.stage, job.dismissed) == (Stage.FAILED, False)
    assert rig.controller.state.failed == 1
    asked = len(svc.gens)
    rig.controller.retry_failed()
    rig.wait_idle()
    assert len(svc.gens) == asked + 1
    assert record(rig, job.job_id).stage is Stage.DONE
    assert notes_files(md) == [NOTES_FILE]

    # (d): of two failed jobs, the first one's record cannot be written
    root = tmp_path / "two"
    svc.serve(REFUSED, REFUSED)
    first_md = lesson(root, name="第一讲")
    second_md = lesson(root, name="第二讲")
    rig = lesson_rig(rigs, root)
    rig.controller.make_notes(first_md)
    rig.controller.make_notes(second_md)
    rig.wait_idle()
    first, second = notes_jobs(rig.store)
    assert first.source_md == first_md and (first.stage, second.stage) == (Stage.FAILED, Stage.FAILED)
    armed, gate = _armed_service(svc)
    armed.set()
    _job_records_fail(rig, lambda j: j.job_id == first.job_id)
    rig.controller.retry_failed()
    notice = rig.controller.state.notice
    gate.set()
    rig.wait_idle()
    assert notice.startswith(f"没有处理：{first.label}。") and "程序自己的数据文件夹" in notice
    assert record(rig, second.job_id).stage is Stage.DONE
    assert notes_files(second_md) == ["第二讲-整理笔记.md"]
    assert record(rig, first.job_id).stage is Stage.FAILED


# ---------------------------------------------------------------- A: notes made by themselves


@pytest.mark.parametrize("kind", ["file", "recording"])
def test_with_the_switch_on_a_finished_lesson_gets_its_notes(tmp_path, svc, rigs, kind):
    """A1 (a) a file job, (b) a recording."""
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    set_notes_model(rig, "model-c")
    kept: dict = {}

    def keep():
        job = source_jobs(rig)[0]
        kept.update(txt=_snapshot(job.txt_path), md=_snapshot(job.md_path))

    at_status(rig, "完成", keep)
    if kind == "file":
        rig.controller.add_files([rig.source(LESSON_FILE)])
    else:
        record_briefly(rig)
    wait_notes(rig, 1)

    lesson_job = source_jobs(rig)[0]
    notes = notes_jobs(rig.store)
    assert lesson_job.stage is Stage.DONE
    assert [(j.stage, j.source_md, j.notes_model) for j in notes] == [(Stage.DONE, lesson_job.md_path, "model-c")]
    md = lesson_job.md_path
    assert (_snapshot(lesson_job.txt_path), _snapshot(md)) == (kept["txt"], kept["md"])
    assert [g["model"] for g in svc.gens] == ["model-c"]
    state = rig.controller.state
    assert (state.last_folder, state.pending) == (md.parent, 0)
    if kind == "file":
        assert md.read_bytes() == SRC_BYTES
        assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
        notices = [s.notice for s in rig.states]
        done = notices.index(f"完成：{NAME}")
        made = next(i for i, n in enumerate(notices) if n.startswith(f"笔记已生成：{NOTES_FILE}"))
        assert done < made
        assert names(md.parent) == sorted([f"{NAME}.txt", f"{NAME}.md", NOTES_FILE])
    else:
        name = "2026-11-01 19-30 录音"
        assert lesson_job.kind is JobKind.RECORDING
        notes_file = md.parent / f"{name}{NOTES_SUFFIX}.md"
        lines = notes_file.read_text(encoding="utf-8").splitlines()
        assert lines[0] == f"# {name}" and lines[2] == "日期：2026-11-01"
        assert names(md.parent) == sorted([f"{name}.wav", f"{name}.txt", f"{name}.md", notes_file.name])


def _no_notes_for(rig, svc, name: str) -> None:
    jobs = rig.store.load_all()
    assert [j.stage for j in jobs] == [Stage.DONE] and jobs[0].kind is JobKind.FILE
    assert svc.fake.log == []
    assert not any(n.startswith(f"{name}{NOTES_SUFFIX}") for n in names(jobs[0].folder))
    assert rig.controller.state.notice == f"完成：{name}"


def test_without_the_switch_no_notes_are_made(tmp_path, svc, rigs):
    """A2 (a) by default, (b) switched on and off again, (c) settings saved by an older version."""
    root = tmp_path / "fresh"
    rig = lesson_rig(rigs, root)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    _no_notes_for(rig, svc, NAME)

    root = tmp_path / "on-off"
    rig = lesson_rig(rigs, root)
    set_auto(rig, True)
    assert rigs(root).controller.settings.auto_notes is True
    set_auto(rig, False)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    _no_notes_for(rig, svc, NAME)
    assert rigs(root).controller.settings.auto_notes is False

    root = tmp_path / "old"
    root.mkdir()
    old = {"format": 2, "output_dir": str(root / "D盘" / "课"), "transcribe_model": "small", "proofread_model": "qwen3:8b"}
    import json

    (root / "settings.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    rig = lesson_rig(rigs, root)
    settings = rig.controller.settings
    assert settings.auto_notes is False
    assert (settings.output_dir, settings.transcribe_model, settings.proofread_model) == (
        root / "D盘" / "课", "small", "qwen3:8b"
    )
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    _no_notes_for(rig, svc, NAME)


@pytest.mark.parametrize("unproofread", [0, 1])
def test_the_finished_notice_stays_until_the_notes_have_their_own_result(tmp_path, svc, rigs, unproofread):
    """A3 (a) and (b)."""
    rig = lesson_rig(rigs, proofreader=LessonProofreader(unproofread=unproofread, total=2 if unproofread else 1))
    set_auto(rig, True)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    wait_notes(rig, 1)
    lesson_job = source_jobs(rig)[0]
    finished = f"完成：{NAME}。有 1 段没校对，保留了原文。" if unproofread else f"完成：{NAME}"
    states = rig.states
    handover = first_index(states, lambda s: s.notice.startswith("完成："))
    made = first_index(states, lambda s: s.notice.startswith("笔记已生成："))
    assert handover < made
    assert [s.notice for s in states[handover:made]] == [finished] * (made - handover)
    assert states[handover].pending == 1
    assert states[handover].last_folder == lesson_job.folder
    assert NOTES_WORKING in [s.status for s in states]
    assert notes_jobs(rig.store)[0].stage is Stage.DONE


def test_handing_over_to_the_notes_never_shows_idle_or_lets_the_computer_sleep(tmp_path, svc, rigs):
    """A3 (c)."""
    gate = gated(svc)
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    since = len(rig.states)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait(lambda: len(svc.gens) == 1)
    seen = rig.states[since:]
    assert seen and all(s.pending > 0 for s in seen)
    # Queuing publishes the count before the worker sets a status (older behaviour, see Deferred);
    # from the worker's first status on, idle must never show again.
    working = since + first_index(seen, lambda s: s.status != "准备好了")
    assert all(s.notice == "" and s.fraction is None for s in rig.states[since:working])
    assert all(s.status != "准备好了" for s in rig.states[working:])
    gate.set()
    wait_notes(rig, 1)
    last = rig.states[-1]
    assert all(s.pending > 0 for s in rig.states[since:-1])
    assert all(s.status != "准备好了" for s in rig.states[working:-1])
    assert working < first_index(rig.states, lambda s: s.notice.startswith("完成："))
    assert last.pending == 0
    assert rig.awake.history == [True, False]
    assert rig.awake.violations == []


def _long_out(tmp_path) -> Path:
    base = _u16(str(tmp_path)) + 1
    extra = max(1, 83 - base)
    if (base + extra) % 2 == 0:
        extra += 1
    out = tmp_path / ("o" * extra)
    rounded = _u16(str(out))
    if rounded > 203:
        pytest.fail(f"The temporary folder {tmp_path} is too long for this test; run pytest with a shorter --basetemp")
    assert rounded % 2 == 1 and rounded >= 83
    return out


@pytest.mark.parametrize("row", ["moved", "not-recorded", "already", "too-long"])
def test_notes_that_cannot_start_say_why_under_the_finished_notice(tmp_path, svc, rigs, row):
    """A4 (a)-(d)."""
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    kept: dict = {}
    source = rig.source(LESSON_FILE)
    if row == "too-long":
        out = _long_out(tmp_path)
        rig.controller.update_settings(replace(rig.controller.settings, output_dir=out))
        size = _fits(out)
        assert _u16(str(out)) + 2 * size == 243
        (out / ("讲" * size)).mkdir(parents=True)
        source = rig.source("讲" * 100 + ".mp3")

    def at_finish():
        job = source_jobs(rig)[0]
        md = job.md_path
        kept["md"] = md
        if row == "moved":
            moved = md.with_name("moved.md")
            md.rename(moved)
            kept["snap"] = _snapshot(moved)
            return
        kept["snap"] = _snapshot(md)
        if row == "already":
            rig.controller.make_notes(md)
        elif row == "too-long":
            (md.parent / f"{md.stem}{NOTES_SUFFIX}.md").write_bytes(b"older notes\n")

    if row == "not-recorded":
        _job_records_fail(rig, lambda j: j.kind is JobKind.NOTES)
    at_status(rig, "完成", at_finish)
    rig.controller.add_files([source])
    wait_notes(rig, 1 if row == "already" else 0)

    job = source_jobs(rig)[0]
    md = kept["md"]
    assert job.stage is Stage.DONE and job.md_path == md
    snap_of = md.with_name("moved.md") if row == "moved" else md
    assert _snapshot(snap_of) == kept["snap"]
    if row == "moved":
        reason = NOTES_NOT_A_FILE.format(name=md.name)
    elif row == "not-recorded":
        reason = f"没有整理：{md.name}。{data_problem(rig)}"
    elif row == "already":
        reason = NOTES_ALREADY.format(name=md.name)
    else:
        numbered = md.parent / f"{md.stem}{NOTES_SUFFIX} (2).md"
        assert (_u16(str(md)), _u16(str(md.parent / f"{md.stem}{NOTES_SUFFIX}.md")), _u16(str(numbered))) == (
            252, 257, 261
        )
        reason = NOTES_PATH_TOO_LONG_BEFORE.format(name=md.name, path=numbered)
    states = rig.states
    handover = states[first_index(states, lambda s: s.notice.startswith("完成："))]
    assert handover.notice == f"完成：{job.folder.name}\n{reason}"
    assert handover.last_folder == job.folder
    if row == "already":
        notes = notes_jobs(rig.store)
        assert [j.stage for j in notes] == [Stage.DONE]
        assert notes_files(md) == [NOTES_FILE]
        return
    state = rig.controller.state
    assert state.notice == f"完成：{job.folder.name}\n{reason}"
    assert state.last_folder == job.folder
    assert notes_jobs(rig.store) == []
    assert svc.fake.log == [] and state.pending == 0
    if row == "not-recorded":
        after = states.index(handover)
        assert all(s.pending == 0 for s in states[after:])
        rig.controller.shutdown()
        again = rigs(tmp_path, proofreader=LessonProofreader())
        again.controller.start()
        again.wait_idle()
        assert notes_jobs(again.store) == []


def test_a_lesson_that_cannot_be_read_at_the_handover_does_not_stop_the_next_one(tmp_path, svc, rigs, monkeypatch):
    """A4 (e): the drive or network drops as the .md is written; the error is raised, not returned."""
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    kept: dict = {}
    is_file = Path.is_file

    def unreachable(path, *args, **kwargs):
        if "md" in kept and path.name == kept["md"].name:
            raise PermissionError(13, "Access is denied", str(path))
        return is_file(path, *args, **kwargs)

    def at_finish():
        kept["md"] = source_jobs(rig)[0].md_path
        monkeypatch.setattr(Path, "is_file", unreachable)

    at_status(rig, "完成", at_finish)
    rig.controller.add_files([rig.source(LESSON_FILE), rig.source("x.mp3")])
    wait_notes(rig, 1)

    jobs = {job.folder.name: job for job in source_jobs(rig)}
    assert set(jobs) == {NAME, "x"}
    assert all(job.stage is Stage.DONE for job in jobs.values())
    md = kept["md"]
    assert jobs[NAME].md_path == md
    states = rig.states
    handover = states[first_index(states, lambda s: s.notice.startswith(f"完成：{NAME}"))]
    assert handover.notice == f"完成：{NAME}\n" + NOTES_UNREACHABLE.format(name=md.name)
    notes = notes_jobs(rig.store)
    assert [(j.stage, j.source_md) for j in notes] == [(Stage.DONE, jobs["x"].md_path)]
    assert rig.controller.state.pending == 0
    assert rig.awake.history == [True, False]


def test_finished_notes_jobs_never_make_notes_of_their_own(tmp_path, svc, rigs):
    """A5 (a) made, (b) retried, (c) continued, (d) resumed at the next start."""
    for row in ("made", "retried", "continued", "resumed"):
        root = tmp_path / row
        md = lesson(root)
        if row == "made":
            svc.serve()
        elif row == "retried":
            svc.serve(REFUSED, then=GOOD)
        elif row == "continued":
            svc.serve(BUSY, BUSY, then=GOOD)
        else:
            svc.serve()
        if row == "resumed":
            first = rigs(root, proofreader=LessonProofreader())
            set_auto(first, True)
            first.store.save(Job.new_notes(md, "model-a"))
            rig = rigs(root, proofreader=LessonProofreader())
            assert rig.controller.settings.auto_notes is True
            rig.controller.start()
        else:
            rig = lesson_rig(rigs, root)
            set_auto(rig, True)
            rig.controller.make_notes(md)
        rig.wait_idle()
        if row == "retried":
            rig.controller.retry_failed()
        elif row == "continued":
            rig.controller.continue_notes()
        rig.wait_idle()
        time.sleep(0.3)
        rig.wait_idle()
        jobs = notes_jobs(rig.store)
        assert [j.stage for j in jobs] == [Stage.DONE], row
        assert notes_files(md) == [NOTES_FILE], row
        assert len(svc.gens) == {"made": 1, "retried": 2, "continued": 3, "resumed": 1}[row], row
        assert rig.controller.state.pending == 0, row


@pytest.mark.parametrize("row", ["waiting", "failed", "resumed", "clicked-twice"])
def test_proofreading_that_waits_or_fails_makes_no_notes_until_it_is_done(tmp_path, svc, rigs, row):
    """A6 (a)-(d)."""
    if row == "resumed":
        first = rigs(proofreader=LessonProofreader())
        set_auto(first, True)
        job = _saved_job(first, stage=Stage.PROOFREADING)
        job.txt_path.write_text("第一课的内容\n", encoding="utf-8")
        rig = rigs(proofreader=LessonProofreader())
        rig.controller.start()
        wait_notes(rig, 1)
        made = record(rig, job.job_id).md_path
        assert notes_files(made) == [f"第一课{NOTES_SUFFIX}.md"]
        return
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    if row == "failed":
        rig.proofreader.error = WordListUnreadable("词表读不了")
    else:
        rig.proofreader.blocked = "没有启动"
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    assert notes_jobs(rig.store) == []
    assert svc.fake.log == []
    state = rig.controller.state
    if row == "failed":
        assert state.failed == 1 and "词表读不了" in state.notice
        rig.proofreader.error = None
        rig.controller.retry_failed()
    else:
        assert state.waiting_proofread == 1 and state.notice == "没有启动"
        rig.proofreader.blocked = None
        if row == "clicked-twice":
            rig.proofreader.gate = threading.Event()
            rig.controller.retry_waiting()
            rig.wait(lambda: len(rig.proofreader.models) == 2)
            rig.controller.retry_waiting()
            assert rig.controller.state.pending == 1
            rig.proofreader.gate.set()
        else:
            rig.controller.retry_waiting()
    wait_notes(rig, 1)
    job = source_jobs(rig)[0]
    assert job.stage is Stage.DONE
    assert notes_files(job.md_path) == [NOTES_FILE]
    if row == "clicked-twice":
        assert len(rig.proofreader.models) == 2


def test_changing_the_settings_as_a_lesson_finishes_counts_for_that_lesson(tmp_path, svc, rigs):
    """A7 (a)-(d)."""
    rows = [("turned-on", False), ("turned-off", True), ("model-changed", True), ("two-queued", False)]
    for row, initially in rows:
        root = tmp_path / row
        svc.serve()
        rig = lesson_rig(rigs, root)
        set_auto(rig, initially)
        if row == "turned-on" or row == "two-queued":
            at_status(rig, "完成", lambda rig=rig: set_auto(rig, True))
        elif row == "turned-off":
            at_status(rig, "完成", lambda rig=rig: set_auto(rig, False))
        else:
            at_status(rig, "完成", lambda rig=rig: set_notes_model(rig, "model-b"))
        files = [rig.source(LESSON_FILE)] + ([rig.source("第二课.mp3")] if row == "two-queued" else [])
        rig.controller.add_files(files)
        expected = {"turned-on": 1, "turned-off": 0, "model-changed": 1, "two-queued": 2}[row]
        wait_notes(rig, expected)
        notes = notes_jobs(rig.store)
        assert len(notes) == expected, row
        saved = SettingsStore(root / "settings.json", rig.settings.defaults).load()
        if row == "model-changed":
            assert [j.notes_model for j in notes] == ["model-b"]
            assert [g["model"] for g in svc.gens] == ["model-b"]
            assert saved.notes_model == "model-b" and saved.auto_notes is True
        else:
            assert saved.auto_notes is (row != "turned-off"), row


def test_notes_join_the_end_of_the_one_queue_one_model_at_a_time(tmp_path, svc, rigs):
    """A8."""
    transcriber = TimedTranscriber()
    rig = lesson_rig(rigs, transcriber=transcriber)
    set_auto(rig, True)
    transcriber.gate = threading.Event()
    rig.controller.add_files([rig.source("甲.mp3"), rig.source("乙.mp3")])
    transcriber.gate.set()
    wait_notes(rig, 2)

    assert [kind for kind, _ in transcriber.times] == ["start", "release", "start", "release"]
    (a_start, a_end), (b_start, b_end) = (
        (transcriber.times[0][1], transcriber.times[1][1]),
        (transcriber.times[2][1], transcriber.times[3][1]),
    )
    (a_check, a_checked), (b_check, b_checked) = rig.proofreader.spans
    log = svc.fake.log
    asked = [i for i, (_, _, prompt) in enumerate(log) if prompt is not None]
    assert len(asked) == 2
    unloaded = [
        next(when for when, path, prompt in log[i + 1:] if prompt is None and path not in ("/api/tags", "/api/show"))
        for i in asked
    ]
    notes_spans = [(log[i][0], end) for i, end in zip(asked, unloaded)]
    order = [a_start, a_end, a_check, a_checked, b_start, b_end, b_check, b_checked]
    for start, end in notes_spans:
        order += [start, end]
    assert order == sorted(order), order
    notices = [s.notice for s in rig.states]
    first = next(i for i, n in enumerate(notices) if n.startswith(f"笔记已生成：甲{NOTES_SUFFIX}.md"))
    second = next(i for i, n in enumerate(notices) if n.startswith(f"笔记已生成：乙{NOTES_SUFFIX}.md"))
    assert first < second
    out = rig.controller.settings.output_dir
    assert (out / "甲" / f"甲{NOTES_SUFFIX}.md").read_text(encoding="utf-8").startswith("# 甲\n")
    assert (out / "乙" / f"乙{NOTES_SUFFIX}.md").read_text(encoding="utf-8").startswith("# 乙\n")


def test_notes_are_made_of_the_md_the_pipeline_really_wrote(tmp_path, svc, rigs):
    """A9."""
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    folder = rig.controller.settings.output_dir / NAME
    theirs = folder / f"{NAME}.md"
    at_status(rig, "正在校对（第 1/1 段）", lambda: theirs.write_bytes("别人的\n".encode("utf-8")))
    rig.controller.add_files([rig.source(LESSON_FILE)])
    wait_notes(rig, 1)
    job = source_jobs(rig)[0]
    assert job.md_path.name == f"{NAME} (2).md"
    notes_file = folder / f"{NAME} (2){NOTES_SUFFIX}.md"
    assert notes_file.read_text(encoding="utf-8").splitlines()[0] == f"# {NAME} (2)"
    assert notes_jobs(rig.store)[0].source_md == job.md_path
    assert theirs.read_bytes() == "别人的\n".encode("utf-8")
    assert not (folder / NOTES_FILE).exists()


@pytest.mark.parametrize("row", ["busy", "model-missing"])
def test_notes_that_wait_for_ollama_leave_the_proofread_md_alone(tmp_path, svc, rigs, row):
    """A10."""
    if row == "busy":
        svc.serve(BUSY, BUSY, then=GOOD)
    else:
        svc.serve(models=["model-a"])
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    wait_notes(rig, 1)
    job = source_jobs(rig)[0]
    assert job.stage is Stage.DONE and job.md_path.read_bytes() == SRC_BYTES
    notes = notes_jobs(rig.store)
    assert [j.stage for j in notes] == [Stage.WAITING_NOTES]
    state = rig.controller.state
    assert (state.waiting_notes, state.waiting_proofread) == (1, 0)
    if row == "busy":
        assert state.notice == NOTES_BUSY_MESSAGE
    else:
        assert state.notice == NOTES_MODEL_MISSING.format(model=rig.controller.settings.notes_model)
        set_notes_model(rig, "model-a")
    assert rig.awake.held is False
    rig.controller.continue_notes()
    rig.wait_idle()
    assert record(rig, notes[0].job_id).stage is Stage.DONE
    assert notes_files(job.md_path) == [NOTES_FILE]
    assert job.md_path.read_bytes() == SRC_BYTES


@pytest.mark.parametrize("moment", ["after-saving-notes", "before-saving-notes", "after-done", "while-closing"])
def test_closed_at_any_step_of_the_handover_never_makes_two_notes_jobs(tmp_path, svc, rigs, monkeypatch, moment):
    """A11 (a)-(d)."""
    caught: list[BaseException] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: caught.append(args.exc_value))
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    if moment in ("after-saving-notes", "before-saving-notes"):
        real_save = rig.store.save
        crashed: list[bool] = []

        def save(job):
            if job.kind is JobKind.NOTES and not crashed:
                crashed.append(True)
                if moment == "after-saving-notes":
                    real_save(job)
                raise Crash
            real_save(job)

        rig.store.save = save
    elif moment == "after-done":

        def die():
            raise Crash

        at_status(rig, "完成", die)
    if moment == "while-closing":
        rig.proofreader.gate = threading.Event()
        rig.controller.add_files([rig.source(LESSON_FILE)])
        rig.wait(lambda: len(rig.proofreader.models) == 1)
        closer = threading.Thread(target=rig.controller.shutdown, daemon=True)
        closer.start()
        rig.wait(lambda: rig.stopping.is_set())
        rig.proofreader.gate.set()
        closer.join(10)
        assert not closer.is_alive()
        rig.controller._worker.join(5)
        assert not rig.controller._worker.is_alive()
        assert svc.gens == []
    else:
        rig.controller.add_files([rig.source(LESSON_FILE)])
        rig.wait(lambda: caught != [])
        rig.controller._worker.join(5)
        assert not rig.controller._worker.is_alive()
        assert len(caught) == 1 and isinstance(caught[0], Crash)

    again = rigs(tmp_path, proofreader=LessonProofreader())
    assert again.controller.settings.auto_notes is True
    again.controller.start()
    again.wait_idle()
    job = source_jobs(again)[0]
    assert job.stage is Stage.DONE and job.md_path.read_bytes() == SRC_BYTES
    notes = notes_jobs(again.store)
    if moment in ("after-saving-notes", "while-closing"):
        assert [j.stage for j in notes] == [Stage.DONE]
        assert notes_files(job.md_path) == [NOTES_FILE]
    else:
        assert notes == []
        assert notes_files(job.md_path) == []
        assert temp_files(job.folder) == []
        assert "出错" not in again.controller.state.notice


def test_a_finished_lesson_whose_last_save_fails_says_no_notes_will_come(tmp_path, svc, rigs):
    """A12."""
    rig = lesson_rig(rigs)
    set_auto(rig, True)
    _job_records_fail(rig, lambda j: j.kind is not JobKind.NOTES and j.stage is Stage.DONE and j.attempts == 0)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    md = rig.controller.settings.output_dir / NAME / f"{NAME}.md"
    assert md.read_bytes() == SRC_BYTES
    assert notes_jobs(rig.store) == []
    assert svc.fake.log == []
    notice = rig.controller.state.notice
    assert notice == f"{NAME}：{data_problem(rig)}{NOTES_NOT_AUTO}"
    assert notice.count("程序自己的数据文件夹") == 1 and "没有整理" not in notice
    assert "没有自动生成" in NOTES_NOT_AUTO and "手动生成" in NOTES_NOT_AUTO and "重试" not in NOTES_NOT_AUTO
    assert "WI" not in NOTES_NOT_AUTO and "12-" not in NOTES_NOT_AUTO


# ---------------------------------------------------------------- PA: any sequence of actions

PA_ROUNDS = 48  # the matrix starts at 8; 48 is the fewest that reach the waited-then-continued floor


def test_any_actions_give_each_proofread_md_one_auto_note_exactly_when_the_switch_was_on(tmp_path, svc, rigs):
    """PA."""
    rng = random.Random(20270405)
    seen = Counter()
    for round_ in range(PA_ROUNDS):
        root = tmp_path / f"r{round_}"
        rig = lesson_rig(rigs, root)
        auto = False
        blocked = None
        expected: dict[str, bool] = {}
        waited: set[str] = set()
        added = 0
        for step in range(12):
            draw = rng.random()
            if draw < 0.25:
                added += 1
                rig.controller.add_files([rig.source(f"第{added}课.mp3")])
                action = "add"
            elif draw < 0.35:
                rig.wait_idle()
                record_briefly(rig)
                action = "record"
            elif draw < 0.55:
                rig.wait_idle()
                auto = not auto
                set_auto(rig, auto)
                action = "switch"
            elif draw < 0.70:
                rig.wait_idle()
                blocked = None if blocked else "没有启动"
                rig.proofreader.blocked = blocked
                action = "block"
            elif draw < 0.80:
                rig.controller.retry_waiting()
                action = "continue"
            elif draw < 0.90:
                rig.wait_idle()
                action = "wait"
            else:
                rig.wait_idle()
                rig.controller.shutdown()
                if auto:
                    seen["restart while on"] += 1
                rig = rigs(root, proofreader=LessonProofreader())
                rig.proofreader.blocked = blocked
                assert rig.controller.settings.auto_notes is auto, (round_, step)
                rig.controller.start()
                action = "restart"
            rig.wait_idle()
            case = (round_, step, action, auto, blocked)

            jobs = rig.store.load_all()
            lessons = [j for j in jobs if j.kind is not JobKind.NOTES]
            notes = [j for j in jobs if j.kind is JobKind.NOTES]
            for job in lessons:
                if job.stage is Stage.WAITING_PROOFREAD:
                    waited.add(job.job_id)
                if job.stage is Stage.DONE and job.job_id not in expected:
                    expected[job.job_id] = auto
                    seen["finished while on" if auto else "finished while off"] += 1
                    if auto and job.kind is JobKind.RECORDING:
                        seen["recording finished while on"] += 1
                    if auto and job.job_id in waited:
                        seen["waited, continued, finished while on"] += 1
            for job in lessons:
                mine = [n for n in notes if n.source_md is not None and same_file(n.source_md, job.md_path)]
                if job.stage is Stage.DONE:
                    assert len(mine) == int(expected[job.job_id]), case
                else:
                    assert mine == [], case
            assert len(notes) == sum(expected.values()), case
            for note in notes:
                assert not is_notes_file(note.source_md), case
                done = [n for n in notes if same_file(n.source_md, note.source_md) and n.stage is Stage.DONE]
                assert len(notes_files(note.source_md)) == len(done), case
            assert rig.awake.violations == [], case
    floors = {
        "finished while on": 10,
        "finished while off": 10,
        "recording finished while on": 2,
        "waited, continued, finished while on": 3,
        "restart while on": 2,
    }
    print("\nPA coverage:", dict(seen))
    for name, floor in floors.items():
        assert seen[name] >= floor, (name, dict(seen))


# ---------------------------------------------------------------- U: settings that cannot be saved


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"), PermissionError(13, "Permission denied")])
def test_settings_that_cannot_be_saved_still_count_this_time_and_say_so(tmp_path, svc, rigs, monkeypatch, caplog, error):
    """U1 (a)-(c)."""
    rig = lesson_rig(rigs)
    set_auto(rig, False)
    settings_file = rig.tmp_path / "settings.json"
    before = settings_file.read_bytes()
    not_saved = SETTINGS_NOT_SAVED.format(path=settings_file)
    fault = DiskFault(monkeypatch, rig.tmp_path, error)

    # (a)
    new = replace(rig.controller.settings, proofread_model="model-b", auto_notes=True)
    raised = None
    with caplog.at_level(logging.WARNING, logger="coursescribe.controller"):
        try:
            rig.controller.update_settings(new)
        except OSError as exc:  # the window would turn this into "程序出了点意外"
            raised = exc
    assert raised is None
    assert rig.controller.state.notice == not_saved
    assert rig.controller.settings == new
    assert settings_file.read_bytes() == before
    assert temp_files(rig.tmp_path) == []
    assert any(r.name == "coursescribe.controller" and r.levelno == logging.WARNING for r in caplog.records)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    wait_notes(rig, 1)
    assert rig.proofreader.models[-1] == "model-b"
    assert [j.stage for j in notes_jobs(rig.store)] == [Stage.DONE]

    # (b)
    again = replace(new, auto_notes=False)
    rig.controller.update_settings(again)
    assert rig.controller.state.notice == not_saved
    fault.fix()
    rig.controller.update_settings(again)
    assert rig.controller.state.notice == ""
    assert SettingsStore(settings_file, rig.settings.defaults).load() == again

    # (c)
    rig.controller.add_files([rig.source("x.mp3")])
    rig.wait_idle()
    assert rig.controller.state.notice == "完成：x"
    rig.controller.update_settings(replace(rig.controller.settings, notes_model="model-c"))
    assert rig.controller.state.notice == "完成：x"
    assert SettingsStore(settings_file, rig.settings.defaults).load().notes_model == "model-c"
    assert not any("程序出了点意外" in s.notice for s in rig.states)


def test_settings_saved_again_lose_the_not_saved_notice_while_the_sleep_warning_shows(tmp_path, svc, rigs, monkeypatch):
    """U1 (d)."""
    from coursescribe.controller import SLEEP_WARNING

    rig = lesson_rig(rigs)
    set_auto(rig, False)
    settings_file = rig.tmp_path / "settings.json"
    not_saved = SETTINGS_NOT_SAVED.format(path=settings_file)
    rig.awake.fail_hold = OSError("PowerCreateRequest failed")
    rig.proofreader.gate = threading.Event()
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait(lambda: len(rig.proofreader.models) == 1)
    assert SLEEP_WARNING in rig.controller.state.notice

    fault = DiskFault(monkeypatch, rig.tmp_path, OSError(28, "No space left on device"))
    rig.controller.update_settings(replace(rig.controller.settings, notes_model="model-b"))
    assert not_saved in rig.controller.state.notice and SLEEP_WARNING in rig.controller.state.notice
    fault.fix()
    rig.controller.update_settings(replace(rig.controller.settings, notes_model="model-c"))
    assert not_saved not in rig.controller.state.notice
    assert SLEEP_WARNING in rig.controller.state.notice
    assert SettingsStore(settings_file, rig.settings.defaults).load().notes_model == "model-c"

    rig.proofreader.gate.set()
    rig.wait_idle()
    assert [job.stage for job in source_jobs(rig)] == [Stage.DONE]
    assert rig.controller.state.notice == f"完成：{NAME}"


# ---------------------------------------------------------------- E: the program's own assembly


def test_the_program_assembles_a_controller_that_reads_its_home(tmp_path, monkeypatch):
    """E1."""
    import json

    from coursescribe.ui import app

    home = tmp_path / "家"
    home.mkdir()
    monkeypatch.setenv("COURSESCRIBE_HOME", str(home))
    settings = {"format": 2, "output_dir": str(tmp_path / "out"), "transcribe_model": "small", "proofread_model": "m",
                "notes_model": "model-x", "auto_notes": True}
    (home / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    (home / "整理规则.txt").mkdir()
    controller = app.build_controller(home, threading.Event(), keep_awake=FakeKeepAwake())
    try:
        assert controller.settings == Settings(tmp_path / "out", "small", "m", "model-x", True)
        controller.start()
        controller.make_notes(lesson(tmp_path))
        deadline = time.time() + 10
        while controller.state.pending and time.time() < deadline:
            time.sleep(0.01)
        assert controller.state.pending == 0
    finally:
        controller.shutdown()
    jobs = JobStore(home / "jobs").load_all()
    assert [(j.kind, j.stage, j.notes_model) for j in jobs] == [(JobKind.NOTES, Stage.FAILED, "model-x")]
    assert jobs[0].message == RULES_UNREADABLE.format(path=home / "整理规则.txt")
    source = Path(app.__file__).read_text(encoding="utf-8")
    run = source[source.index("def run("):]
    assert "build_controller(" in run
    assert "notes_rules=rules_path()" in run
