"""The main window for organised notes (12-8), the controller's side: which proofread .md the
"生成整理笔记" button stands for, files dropped or chosen going each their own way, and what the
user is told when notes are done.

The service is the shared FakeOllama (a real local HTTP server) talked to by the real notes writer;
the other stand-ins sit at the system's edge (the transcription and proofreading doubles of the
controller tests, a refused new file, a failed job record, a file the system cannot look at)."""

from __future__ import annotations

import os
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from coursescribe.controller import (
    MEDIA_SUFFIXES,
    NOT_ADDED_TXT,
    NOTES_ALREADY,
    NOTES_ALREADY_WAITING,
    NOTES_DRAFT,
    NOTES_IS_NOTES,
    NOTES_NOT_A_FILE,
    NOTES_NOT_MD,
    NOTES_PATH_TOO_LONG_BEFORE,
    NOTES_UNREACHABLE,
    AppState,
    notes_done_notice,
)
from coursescribe.jobs import NOTES_SUFFIX, PATH_LIMIT, Job, JobKind, Stage, is_notes_file, notes_file_for
from coursescribe.models import DEFAULT_NOTES_MODEL
from coursescribe.notes import finish_notes
from coursescribe.notes_rules import RULES_EMPTY_NOTE, NotesRules
from coursescribe.notes_writer import NOTES_BUSY_MESSAGE, NOTES_EMPTY_SOURCE, NOTES_REFUSED
from coursescribe.pipeline import NOTES_SOURCE_ENCODING
from notes_replies import META
from test_controller import _job_records_fail, _saved_job, _snapshot, _too_long_output, _unusable_output
from test_notes import SRC
from test_notes_auto import (
    LESSON_FILE,
    LessonProofreader,
    at_status,
    data_problem,
    first_index,
    lesson_rig,
    record_briefly,
    set_auto,
    source_jobs,
    wait_notes,
)
from test_notes_job import (  # noqa: F401  (ollama, svc and rigs are fixtures)
    DISK_FULL,
    FINISHED_BYTES,
    NAME,
    NOTES_FILE,
    SRC_BYTES,
    TITLE_ONLY,
    DiskFault,
    gated,
    lesson,
    long_lesson,
    names,
    new_job,
    notes_files,
    notes_jobs,
    ollama,
    rigs,
    snapshots,
    spellings,
    svc,
    unchanged,
)
from test_notes_writer import BUSY, GOOD, MADE_UP_QUOTE, MADE_UP_TAG, REFUSED, RULES

NOT_READABLE = "{name}（读不到这个文件，可能是它所在的硬盘或网络断开了）"  # written out here on purpose

# What the model answers, by how much of it survives the check against SRC (each is checked with
# the real finish_notes before it is used).
NONE_REMOVED = "## 核心宗旨\n索引像目录。\n\n## 一\n写入会变慢。\n"
QUOTE_ONLY = "## 一\n写入会变慢。\n\n" + MADE_UP_QUOTE
TAG_ONLY = "## 一\n写入会变慢。 " + MADE_UP_TAG + "\n"
UNEQUAL = GOOD.replace("## 引述\n", "## 引述\n" + MADE_UP_QUOTE.split("\n", 1)[1] + "\n\n", 1)


# ---------------------------------------------------------------- tools


def held(rig, svc):
    """Nothing queued on ``rig`` can finish until the returned ``release()`` is called (or 5 s pass):
    transcription stops at its gate and the service holds every generation."""
    rig.transcriber.gate = threading.Event()
    gate = gated(svc)
    transcribing = rig.transcriber.gate

    def release() -> None:
        gate.set()
        transcribing.set()

    return release


def notices_during(rig, since: int, until: int | None = None) -> list[str]:
    """The notices published from ``since`` on, without the repeats next to each other."""
    seen: list[str] = []
    for state in rig.states[since:until]:
        if not seen or seen[-1] != state.notice:
            seen.append(state.notice)
    return seen


_NOTES_NAME = re.compile(r"-整理笔记( \(\d+\))?\.md$", re.IGNORECASE)


def expected_jobs(batch) -> list[tuple[JobKind, str]]:
    """The reference rule for one call of add_files on a controller with no jobs: the jobs."""
    return _reference(batch)[0]


def expected_notice(rig, batch) -> str:
    """The reference rule for one call of add_files: the notice once the call has returned and no
    job has had a result yet. Written from the decisions in the plan; uses no code under test.
    Call it before add_files."""
    jobs, lines = _reference(batch)
    if lines:
        return "\n".join(lines)
    if any(kind is JobKind.NOTES for kind, _ in jobs):
        return ""
    return rig.controller.state.notice


def _reference(batch) -> tuple[list[tuple[JobKind, str]], list[str]]:
    jobs: list[tuple[JobKind, str]] = []
    not_added: list[str] = []
    not_noted: list[str] = []
    noted: dict[str, str] = {}
    for path in batch:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".md":
            if not path.is_file():
                not_noted.append(NOTES_NOT_A_FILE.format(name=path.name))
            elif _NOTES_NAME.search(path.name):
                not_noted.append(NOTES_IS_NOTES.format(name=path.name))
            else:
                # Reason for the edit (WI-2): the file is the same file when the disk says so (the path exists
                # here, checked above). The old string rule (normcase) does nothing on a Mac volume that ignores case.
                found = os.stat(path)
                key = (found.st_dev, found.st_ino)
                if key in noted:
                    not_noted.append(NOTES_ALREADY.format(name=noted[key]))
                else:
                    noted[key] = path.name
                    jobs.append((JobKind.NOTES, path.name))
        elif not path.is_file():
            not_added.append(f"{path.name}（不是文件）")
        elif suffix == ".txt":
            not_added.append(NOT_ADDED_TXT.format(name=path.name))
        elif suffix not in MEDIA_SUFFIXES:
            not_added.append(f"{path.name}（不支持这种格式）")
        else:
            jobs.append((JobKind.FILE, path.name))
    lines = ["没有加入：" + "、".join(not_added)] if not_added else []
    return jobs, lines + not_noted


def queued(rig) -> list[tuple[JobKind, str]]:
    """Every job on record as (kind, name of what it was made from), in the order they were made."""
    return [
        (job.kind, (job.source_md if job.kind is JobKind.NOTES else job.audio_path).name)
        for job in rig.store.load_all()
    ]


def done_notice(name: str, quotes: int, concepts: int) -> str:
    """The four shapes of the finished notice, written out from the table in the plan."""
    if quotes and concepts:
        removed = f"删掉了 {quotes} 条对不上原文的引述、{concepts} 个原文里没有的概念。"
    elif quotes:
        removed = f"删掉了 {quotes} 条对不上原文的引述。"
    elif concepts:
        removed = f"删掉了 {concepts} 个原文里没有的概念。"
    else:
        removed = ""
    return f"笔记已生成：{name}。{removed}笔记是初稿，请自己看一遍。"


def unreachable(monkeypatch, *file_names: str) -> None:
    """The system cannot look at files with these names: asking raises, as a dropped drive does."""
    is_file = Path.is_file

    def answer(path, *args, **kwargs):
        if path.name in file_names:
            raise PermissionError(13, "Access is denied", str(path))
        return is_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", answer)


def add_safely(rig, paths) -> list[BaseException]:
    """add_files, with whatever it raises handed back: the window would turn it into "程序出了点意外"."""
    errors: list[BaseException] = []
    try:
        rig.controller.add_files(paths)
    except Exception as exc:
        errors.append(exc)
    return errors


def pending_grew(states, before: AppState) -> list[AppState]:
    """The states in which more jobs were pending than in the state before."""
    grew = []
    for state in states:
        if state.pending > before.pending:
            grew.append(state)
        before = state
    return grew


def new_notices(states, before: AppState) -> list[str]:
    """Every notice that was not empty and replaced a different one."""
    changed = []
    for state in states:
        if state.notice and state.notice != before.notice:
            changed.append(state.notice)
        before = state
    return changed


def finish(rig, name: str) -> Path:
    """Add a media file and wait: the .md its job says it wrote."""
    rig.controller.add_files([rig.source(name)])
    rig.wait_idle()
    return next(job for job in source_jobs(rig) if job.audio_path.name == name).md_path


# ---------------------------------------------------------------- LM: AppState.last_md


@pytest.mark.parametrize("kind", ["file", "recording"])
def test_a_finished_lesson_is_the_one_the_notes_button_stands_for(tmp_path, svc, rigs, kind):
    """LM1 (a) a file job, (b) a recording."""
    rig = lesson_rig(rigs)
    if kind == "file":
        rig.controller.add_files([rig.source(LESSON_FILE)])
        name = NAME
    else:
        record_briefly(rig)
        name = "2026-11-01 19-30 录音"
    rig.wait_idle()

    job = source_jobs(rig)[0]
    state = rig.controller.state
    assert job.stage is Stage.DONE
    assert state.last_md == job.md_path
    assert state.last_md.name == f"{name}.md" and state.last_md.is_file()
    if kind == "file":
        assert state.last_md.read_bytes() == SRC_BYTES
    states = rig.states
    first = first_index(states, lambda s: s.last_md is not None)
    assert states[first].notice == f"完成：{name}"
    assert states[first].last_folder == states[first].last_md.parent
    assert all(s.last_md is None for s in states[:first])
    # pointing at it is not starting on it
    assert notes_jobs(rig.store) == []
    assert notes_files(job.md_path) == []
    assert svc.fake.log == []


def test_last_md_is_the_name_the_pipeline_really_wrote(tmp_path, svc, rigs):
    """LM2."""
    rig = lesson_rig(rigs)
    folder = rig.controller.settings.output_dir / NAME
    theirs = folder / f"{NAME}.md"
    at_status(rig, "正在校对（第 1/1 段）", lambda: theirs.write_bytes("别人的\n".encode("utf-8")))
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    assert rig.controller.state.last_md == folder / f"{NAME} (2).md"

    rig.controller.make_notes(rig.controller.state.last_md)
    rig.wait_idle()
    notes_file = folder / f"{NAME} (2){NOTES_SUFFIX}.md"
    assert notes_file.read_text(encoding="utf-8").splitlines()[0] == f"# {NAME} (2)"
    assert theirs.read_bytes() == "别人的\n".encode("utf-8")
    assert not (folder / NOTES_FILE).exists()


def test_notes_jobs_never_move_last_md_and_a_newer_lesson_replaces_it(tmp_path, svc, rigs):
    """LM3."""
    rig = lesson_rig(rigs)
    controller = rig.controller
    first = finish(rig, "甲.mp3")
    assert controller.state.last_md == first

    old = lesson(tmp_path, name="旧课")
    controller.make_notes(old)
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE]
    assert (controller.state.last_md, controller.state.last_folder) == (first, old.parent)

    second = finish(rig, "乙.mp3")
    assert second != first
    assert controller.state.last_md == second

    controller.make_notes(first)
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store, first)] == [Stage.DONE]
    assert controller.state.last_md == second

    svc.serve(REFUSED)
    controller.make_notes(second)
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store, second)] == [Stage.FAILED]
    assert controller.state.last_md == second

    # no recording or file job has finished here: notes alone give the button nothing to stand for
    svc.serve()
    other = lesson_rig(rigs, tmp_path / "二")
    older = lesson(tmp_path / "二", name="旧课")
    other.controller.make_notes(older)
    other.wait_idle()
    assert [job.stage for job in notes_jobs(other.store)] == [Stage.DONE]
    assert (other.controller.state.last_md, other.controller.state.last_folder) == (None, older.parent)


def test_last_md_is_set_only_at_the_moment_a_lesson_is_really_done(tmp_path, svc, rigs):
    """LM4 (a) waited for proofreading, (b) failed, (d) waits for a model, (c) the last save failed."""
    rig = lesson_rig(rigs)
    controller = rig.controller
    out = controller.settings.output_dir
    first = finish(rig, "甲.mp3")
    assert controller.state.last_md == first

    # (a)
    rig.proofreader.blocked = "没有启动"
    controller.add_files([rig.source("乙.mp3")])
    rig.wait_idle()
    assert controller.state.waiting_proofread == 1
    assert controller.state.last_md == first
    assert not (out / "乙" / "乙.md").exists()
    rig.proofreader.blocked = None
    controller.retry_waiting()
    rig.wait_idle()
    assert controller.state.last_md == out / "乙" / "乙.md"

    # (b)
    rig.proofreader.error = RuntimeError("坏了")
    controller.add_files([rig.source("丙.mp3")])
    rig.wait_idle()
    assert controller.state.failed == 1
    assert controller.state.last_md == out / "乙" / "乙.md"
    rig.proofreader.error = None
    controller.retry_failed()
    rig.wait_idle()
    assert controller.state.last_md == out / "丙" / "丙.md"

    # (d)
    rig.transcriber.missing.add("large-v3")
    controller.add_files([rig.source("戊.mp3")])
    rig.wait_idle()
    assert controller.state.model_needed == "large-v3"
    assert controller.state.last_md == out / "丙" / "丙.md"

    # (c): the .md is on disk, so the button must work, though the record could not be saved
    rig = lesson_rig(rigs, tmp_path / "c")
    controller = rig.controller
    out = controller.settings.output_dir
    first = finish(rig, "甲.mp3")
    assert controller.state.last_md == first
    _job_records_fail(rig, lambda j: j.kind is not JobKind.NOTES and j.stage is Stage.DONE and j.attempts == 0)
    controller.add_files([rig.source("丁.mp3")])
    rig.wait_idle()
    last = out / "丁" / "丁.md"
    state = controller.state
    assert state.last_md == last
    assert last.is_file()
    assert state.last_folder == first.parent
    assert state.notice == f"丁：{data_problem(rig)}"
    states = rig.states
    at = first_index(states, lambda s: s.last_md == last)
    assert states[at].pending == 0
    assert states[at].notice == f"丁：{data_problem(rig)}"
    assert states[at].last_folder == states[at - 1].last_folder == first.parent
    assert all(s.last_md != last for s in states[:at])


def test_last_md_is_empty_after_a_restart_until_a_lesson_finishes_again(tmp_path, svc, rigs):
    """LM5."""
    rig = lesson_rig(rigs)
    first = finish(rig, "甲.mp3")
    assert rig.controller.state.last_md == first
    rig.controller.shutdown()

    again = lesson_rig(rigs, tmp_path)
    again.wait_idle()
    assert [job.stage for job in source_jobs(again)] == [Stage.DONE]  # on disk, and still it does not count
    assert (again.controller.state.last_md, again.controller.state.last_folder) == (None, None)
    assert all(s.last_md is None for s in again.states)
    second = finish(again, "乙.mp3")
    assert again.controller.state.last_md == second

    # a job the last run left half done counts as finished in this one
    root = tmp_path / "续"
    root.mkdir()
    before = rigs(root, proofreader=LessonProofreader())
    job = _saved_job(before, stage=Stage.PROOFREADING)
    job.txt_path.write_text("第一课的内容\n", encoding="utf-8")
    resumed = lesson_rig(rigs, root)
    resumed.wait_idle()
    saved = source_jobs(resumed)[0]
    assert saved.stage is Stage.DONE and saved.md_path.is_file()
    assert resumed.controller.state.last_md == saved.md_path


PLM_ROUNDS = 20
PLM_ACTIONS = ("add", "record", "notes", "block", "continue", "restart", "nothing")
PLM_WEIGHTS = {  # (proofreading is blocked, a job waits for proofreading): one weight per action
    (False, False): (0.30, 0.06, 0.17, 0.20, 0.05, 0.12, 0.10),
    (True, False): (0.54, 0.06, 0.10, 0.10, 0.05, 0.10, 0.05),
    (True, True): (0.18, 0.02, 0.10, 0.45, 0.10, 0.10, 0.05),
    (False, True): (0.10, 0, 0.05, 0.05, 0.40, 0.40, 0),
}


def _last_md_only_changes_with_a_finished_lesson(rig, case) -> None:
    previous = AppState()
    for state in rig.states:
        if state.last_md != previous.last_md:
            assert state.notice.startswith("完成："), case
            assert state.last_folder == state.last_md.parent, case
        previous = state


def test_whatever_is_done_last_md_is_the_lesson_finished_last_in_this_run(tmp_path, svc, rigs):
    """PLM."""
    rng = random.Random(20270501)
    seen = Counter()
    for round_ in range(PLM_ROUNDS):
        root = tmp_path / f"r{round_}"
        rig = lesson_rig(rigs, root)
        expected = None
        done: set[str] = set()
        added = 0
        for step in range(14):
            rig.wait_idle()
            blocked = rig.proofreader.blocked is not None
            waiting = any(job.stage is Stage.WAITING_PROOFREAD for job in rig.store.load_all())
            action = rng.choices(PLM_ACTIONS, PLM_WEIGHTS[(blocked, waiting)])[0]
            finished = [job for job in source_jobs(rig) if job.stage is Stage.DONE]
            if action == "notes" and not finished:
                action = "nothing"
            case = (round_, step, action, blocked)
            before = expected
            notes_before = sum(job.stage is Stage.DONE for job in notes_jobs(rig.store))
            if action == "add":
                added += 1
                rig.controller.add_files([rig.source(f"第{added}课.mp3")])
            elif action == "record":
                record_briefly(rig, 0.1)
            elif action == "notes":
                rig.controller.make_notes(rng.choice(finished).md_path)
            elif action == "block":
                rig.proofreader.blocked = None if blocked else "没有启动"
            elif action == "continue":
                rig.controller.retry_waiting()
            elif action == "restart":
                _last_md_only_changes_with_a_finished_lesson(rig, case)
                rig.controller.shutdown()
                rig = lesson_rig(rigs, root, proofreader=rig.proofreader)
            rig.wait_idle()

            lessons = source_jobs(rig)
            new = [job for job in lessons if job.stage is Stage.DONE and job.job_id not in done]
            if new:
                last = max(new, key=lambda job: job.created_ns)
                expected = last.md_path
            elif action == "restart":
                expected = None
            done.update(job.job_id for job in new)
            assert rig.controller.state.last_md == expected, case
            if expected is not None:
                assert expected.is_file() and not is_notes_file(expected), case

            still_waiting = any(job.stage is Stage.WAITING_PROOFREAD for job in lessons)
            if new and action == "add":
                seen["set as the file was added"] += 1
            if new and last.kind is JobKind.RECORDING:
                seen["set by a recording"] += 1
            if new and action == "continue":
                seen["set after waiting and continuing"] += 1
            if new and action == "restart":
                seen["set by a job resumed at the restart"] += 1
            if action == "restart" and not new and before is not None:
                seen["emptied by a restart"] += 1
            if action == "restart" and blocked and still_waiting and expected is None:
                seen["restart while blocked, still waiting"] += 1
            if action == "notes":
                assert sum(job.stage is Stage.DONE for job in notes_jobs(rig.store)) == notes_before + 1, case
                seen["notes done, unchanged" if expected is not None else "notes done, still empty"] += 1
        _last_md_only_changes_with_a_finished_lesson(rig, (round_, "end"))
    floors = {
        "set as the file was added": 25,
        "set by a recording": 3,
        "set after waiting and continuing": 3,
        "set by a job resumed at the restart": 3,
        "emptied by a restart": 8,
        "restart while blocked, still waiting": 1,
        "notes done, unchanged": 8,
        "notes done, still empty": 3,
    }
    print("\nPLM coverage:", dict(seen))
    for name, floor in floors.items():
        assert seen[name] >= floor, (name, dict(seen))


# ---------------------------------------------------------------- X: add_files sends each file its own way


def test_a_proofread_md_added_like_a_file_gets_its_notes_and_no_job_folder(tmp_path, svc, rigs):
    """X1."""
    md = lesson(tmp_path)
    before = snapshots(md.parent)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.add_files([md])
    returned = len(rig.states)
    assert any(s.pending == 1 and s.notice == "" for s in rig.states[since:returned])
    rig.wait_idle()

    jobs = rig.store.load_all()
    assert [(j.kind, j.stage, j.source_md, j.notes_model) for j in jobs] == [
        (JobKind.NOTES, Stage.DONE, md, DEFAULT_NOTES_MODEL)
    ]
    assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
    notice = rig.controller.state.notice
    assert notice == notes_done_notice(jobs[0])
    assert notice.startswith(f"笔记已生成：{NOTES_FILE}。")
    assert not (tmp_path / "out").exists()
    assert unchanged(md.parent, before)
    assert len(svc.gens) == 1


def test_a_mixed_batch_goes_each_its_own_way_with_the_refusals_in_one_notice(tmp_path, svc, rigs):
    """X2, and the empty batch of X7."""
    rig = rigs()
    rig.controller.start()
    release = held(rig, svc)
    published = len(rig.states)
    rig.controller.add_files([])
    rig.controller.add_files(iter(()))
    assert len(rig.states) == published
    assert rig.store.load_all() == []

    a = rig.source("a.mp3")
    md = lesson(tmp_path)
    txt = md.with_suffix(".txt")
    notes = md.parent / "课-整理笔记.md"
    notes.write_bytes(FINISHED_BYTES)
    docx = rig.source("讲义.docx")
    folder = tmp_path / "input"
    batch = [a, md, txt, notes, tmp_path / "没有.mp4", tmp_path / "没有.md", folder, docx, md]
    kept = {path: _snapshot(path) for path in (notes, docx, txt)}
    reference = expected_notice(rig, batch)
    rig.controller.add_files(path for path in batch)

    assert queued(rig) == [(JobKind.FILE, "a.mp3"), (JobKind.NOTES, md.name)]
    state = rig.controller.state
    assert state.pending == 2
    first_line = (
        f"没有加入：{NAME}.txt（这是文字稿，整理笔记要用它的校对稿 .md）、没有.mp4（不是文件）、"
        "input（不是文件）、讲义.docx（不支持这种格式）"
    )
    assert state.notice.split("\n") == [
        first_line,
        "没有整理：课-整理笔记.md（这已经是整理笔记了，请选它的校对稿）",
        "没有整理：没有.md（找不到这个文件，可能被移走或删除了）",
        f"{NAME}.md：已经在整理了。",
    ]
    assert state.notice.split("\n")[1:] == [
        NOTES_IS_NOTES.format(name=notes.name),
        NOTES_NOT_A_FILE.format(name="没有.md"),
        NOTES_ALREADY.format(name=md.name),
    ]
    assert state.notice == reference

    release()
    rig.wait_idle()
    jobs = rig.store.load_all()
    assert [job.stage for job in jobs] == [Stage.DONE, Stage.DONE]
    assert jobs[0].txt_path.is_file() and jobs[0].md_path.is_file()
    assert notes_files(md) == [NOTES_FILE]
    assert len(svc.gens) == 1
    assert {path: _snapshot(path) for path in kept} == kept


def test_each_kind_of_file_that_cannot_be_used_gets_its_own_reason(tmp_path, svc, rigs):
    """X3 (a)-(g)."""
    folder = tmp_path / "杂"
    folder.mkdir()

    def made(name: str, data: bytes = b"x") -> Path:
        path = folder / name
        path.write_bytes(data)
        return path

    txt = made("课.txt")
    upper_txt = made("课乙.TXT")
    notes = [made("课-整理笔记.md", FINISHED_BYTES), made("课-整理笔记 (2).md", FINISHED_BYTES),
             made("另-整理笔记.MD", FINISHED_BYTES)]
    like_md = folder / "子.md"
    like_md.mkdir()
    others = [made("讲义.docx"), made("README"), made("课.md.bak")]
    odd_txt = made("第（3）讲、复习.txt")
    listing = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    rig = rigs()
    rig.controller.start()

    rows = [(path, "没有加入：" + NOT_ADDED_TXT.format(name=path.name)) for path in (txt, upper_txt, odd_txt)]
    rows.append((folder / "没有.txt", "没有加入：没有.txt（不是文件）"))
    rows += [(path, NOTES_IS_NOTES.format(name=path.name)) for path in notes]
    rows.append((like_md, NOTES_NOT_A_FILE.format(name="子.md")))
    rows += [(path, f"没有加入：{path.name}（不支持这种格式）") for path in others]
    assert rows[0][1] == "没有加入：课.txt（这是文字稿，整理笔记要用它的校对稿 .md）"
    for path, notice in rows:
        rig.controller.add_files([path])
        assert rig.controller.state.notice == notice, path
        assert rig.store.load_all() == [] and rig.controller.state.pending == 0, path
    rig.controller.make_notes(txt)
    assert rig.controller.state.notice == NOTES_NOT_MD.format(name="课.txt")  # the other way in says what it did
    assert rig.store.load_all() == [] and rig.controller.state.pending == 0
    assert svc.fake.log == [] and rig.awake.history == []
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert [p for p in after if not p.startswith("state")] == listing


def test_the_same_md_twice_makes_one_job_and_says_so_for_the_others(tmp_path, svc, rigs):
    """X4 (a)-(f)."""
    rig = rigs()
    controller = rig.controller
    controller.start()

    # (a) three times in one batch
    gate = gated(svc)
    md = lesson(tmp_path, name="课一")
    controller.add_files([md, md, md])
    assert len(notes_jobs(rig.store, md)) == 1
    assert controller.state.notice.split("\n") == [NOTES_ALREADY.format(name=md.name)] * 2

    # (b) every way of naming one file
    md2 = lesson(tmp_path, name="课二")
    ways = spellings(md2)
    controller.add_files(ways)
    assert len(notes_jobs(rig.store, md2)) == 1
    assert controller.state.notice.split("\n") == [NOTES_ALREADY.format(name=md2.name)] * (len(ways) - 1)
    gate.set()
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE, Stage.DONE]
    assert len(svc.gens) == 2

    # (c) already waiting for Ollama
    svc.serve(BUSY, BUSY)
    md3 = lesson(tmp_path, name="课三")
    controller.add_files([md3])
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store, md3)] == [Stage.WAITING_NOTES]
    asked = len(svc.fake.log)
    controller.add_files([md3, md3])
    assert controller.state.notice.split("\n") == [NOTES_ALREADY_WAITING.format(name=md3.name)] * 2
    assert [job.stage for job in notes_jobs(rig.store, md3)] == [Stage.WAITING_NOTES]
    assert len(svc.fake.log) == asked and controller.state.pending == 0

    # (d) failed last time: the first one retries it, the second is told so
    svc.serve(REFUSED)
    md4 = lesson(tmp_path, name="课四")
    controller.add_files([md4])
    rig.wait_idle()
    failed = notes_jobs(rig.store, md4)
    assert [job.stage for job in failed] == [Stage.FAILED]
    gate = gated(svc)
    controller.add_files([md4, md4])
    assert [job.job_id for job in notes_jobs(rig.store, md4)] == [failed[0].job_id]
    assert controller.state.pending == 1
    assert controller.state.notice == NOTES_ALREADY.format(name=md4.name)
    gate.set()
    rig.wait_idle()
    assert [(job.job_id, job.stage) for job in notes_jobs(rig.store, md4)] == [(failed[0].job_id, Stage.DONE)]
    assert notes_files(md4) == ["课四-整理笔记.md"]
    assert len(svc.gens) == 1

    # (e) eight batches at once, each with the same .md
    gate = gated(svc)
    md5 = lesson(tmp_path, name="课五")
    sources = [rig.source(f"a{i}.mp3") for i in range(8)]
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def add(index: int) -> None:
        try:
            barrier.wait(5)
            controller.add_files([md5, sources[index]])
        except BaseException as exc:  # a thread's exception is otherwise only printed
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(i,), daemon=True) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(notes_jobs(rig.store, md5)) == 1
    assert sorted(job.audio_path.name for job in source_jobs(rig)) == [f"a{i}.mp3" for i in range(8)]
    gate.set()
    rig.wait_idle()
    assert len(svc.gens) == 1

    # (f) once the notes are done, the same .md is a new job by the rule of make_notes
    svc.serve()
    md6 = lesson(tmp_path, name="课六")
    controller.add_files([md6])
    rig.wait_idle()
    controller.add_files([md6])
    rig.wait_idle()
    assert [job.notes_path.name for job in notes_jobs(rig.store, md6)] == ["课六-整理笔记.md", "课六-整理笔记 (2).md"]
    assert len(svc.gens) == 2

    for one in (md, md2, md3, md4, md5, md6):
        finished = [job for job in notes_jobs(rig.store, one) if job.stage is Stage.DONE]
        assert len(notes_files(one)) == len(finished), one


def test_the_refusals_are_published_once_and_nothing_later_in_the_batch_wipes_them(tmp_path, svc, rigs):
    """X5."""
    rig = rigs()
    controller = rig.controller
    controller.start()
    finish(rig, "旧.mp3")
    assert controller.state.notice == "完成：旧"
    release = held(rig, svc)
    a = rig.source("a.mp3")
    md = lesson(tmp_path)
    notes = md.parent / "课1-整理笔记.md"
    notes.write_bytes(FINISHED_BYTES)
    txt = md.parent / "x.txt"
    txt.write_bytes(b"x")
    before = controller.state
    since = len(rig.states)
    controller.add_files([tmp_path / "没有1.md", a, notes, md, txt, tmp_path / "没有2.md"])
    during = rig.states[since:]
    final = "\n".join([
        "没有加入：" + NOT_ADDED_TXT.format(name="x.txt"),
        NOTES_NOT_A_FILE.format(name="没有1.md"),
        NOTES_IS_NOTES.format(name="课1-整理笔记.md"),
        NOTES_NOT_A_FILE.format(name="没有2.md"),
    ])

    assert controller.state.notice == final
    assert controller.state.pending == 2
    seen = notices_during(rig, since, since + len(during))
    assert set(seen) <= {"完成：旧", "", final}
    assert seen[-1] == final
    assert new_notices(during, before) == [final]
    counts = [before.pending] + [s.pending for s in during]
    assert counts == sorted(counts)
    assert next(s for s in during if s.pending == 1).notice == "完成：旧"  # queuing the media file leaves it
    assert next(s for s in during if s.pending == 2).notice == ""  # queuing the .md clears it, in that state
    assert (during[-1].notice, during[-1].pending) == (final, 2)
    release()
    rig.wait_idle()


def test_a_failure_already_shown_is_not_wiped_while_a_later_file_is_slow_to_look_at(tmp_path, svc, rigs, monkeypatch):
    """X5b."""
    rig = rigs()
    controller = rig.controller
    controller.start()
    controller.add_files([tmp_path / "没有.docx"])
    earlier = "没有加入：没有.docx（不是文件）"
    assert controller.state.notice == earlier
    release = held(rig, svc)
    md = lesson(tmp_path, name="只有标题", data="# 只有标题\n".encode("utf-8"))
    kept = _snapshot(md)
    slow = rig.source("慢.mp3")
    waited: list[bool] = []
    is_file = Path.is_file

    def slowly(path, *args, **kwargs):
        if path.name == "慢.mp3" and not waited:
            deadline = time.time() + 3
            while controller.state.failed != 1 and time.time() < deadline:
                time.sleep(0.005)
            waited.append(controller.state.failed == 1)
        return is_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", slowly)
    before = controller.state
    since = len(rig.states)
    controller.add_files([md, slow])
    during = rig.states[since:]
    state = controller.state
    failure = f"{md.name}：{NOTES_EMPTY_SOURCE}"

    assert waited == [True]  # the first file was queued, and had failed, before the second was looked at
    assert state.notice == failure
    assert (state.failed, state.pending) == (1, 1)
    jobs = rig.store.load_all()
    assert [(j.kind, j.stage is Stage.DONE) for j in jobs if j.kind is JobKind.FILE] == [(JobKind.FILE, False)]
    assert [j.audio_path.name for j in jobs if j.kind is JobKind.FILE] == ["慢.mp3"]
    assert [(j.stage, j.source_md) for j in notes_jobs(rig.store)] == [(Stage.FAILED, md)]
    grew = during.index(pending_grew(during, before)[0])
    assert during[grew].notice == ""
    assert all(s.notice == earlier for s in during[:grew])
    failed = first_index(during, lambda s: s.notice == failure)
    assert all(s.notice == failure for s in during[failed:])
    assert svc.gens == []
    release()
    rig.wait_idle()
    assert _snapshot(md) == kept and notes_files(md) == []
    assert len(rig.store.load_all()) == 2


@pytest.mark.parametrize("row", ["no-save-folder", "save-folder-too-long", "notes-not-recorded", "nothing-recorded",
                                 "md-unreachable", "media-unreachable", "others-unreachable"])
def test_one_file_with_a_problem_does_not_stop_the_others_in_the_batch(tmp_path, svc, rigs, monkeypatch, row):
    """X6 (a)-(g)."""
    rig = rigs()
    controller = rig.controller
    a = rig.source("a.mp3")
    md = lesson(tmp_path)
    batch = [a, md]
    folder_problem = too_long_problem = ""
    if row == "no-save-folder":
        _unusable_output(rig)
        folder_problem = (
            f"保存文件夹 {controller.settings.output_dir} 打不开（可能是移动硬盘没插上，或没有权限）。"
            "请插上硬盘，或在设置里换一个保存文件夹。"
        )
    elif row == "save-folder-too-long":
        root = _too_long_output(rig)
        too_long_problem = f"保存文件夹的路径太长：{root}。请在设置里换一个路径短一些的文件夹。"
    elif row == "notes-not-recorded":
        _job_records_fail(rig, lambda j: j.kind is JobKind.NOTES)
    elif row == "nothing-recorded":
        _job_records_fail(rig, lambda j: True)
    elif row == "md-unreachable":
        unreachable(monkeypatch, "掉线.md")
        batch = [tmp_path / "掉线.md", a, md]
    elif row == "media-unreachable":
        unreachable(monkeypatch, "掉线.mp4")
        batch = [tmp_path / "掉线.mp4", a, md]
    else:
        unreachable(monkeypatch, "掉线.txt", "掉线.docx")
        batch = [tmp_path / "掉线.txt", tmp_path / "掉线.docx", a, md]
    controller.start()
    release = held(rig, svc)
    errors = add_safely(rig, batch)
    notice = controller.state.notice
    jobs = queued(rig)
    pending = controller.state.pending
    out = controller.settings.output_dir

    assert errors == []
    both = [(JobKind.FILE, "a.mp3"), (JobKind.NOTES, md.name)]
    if row == "no-save-folder":
        assert notice == "没有加入：a.mp3。" + folder_problem
        assert jobs == [(JobKind.NOTES, md.name)]  # notes need no save folder
    elif row == "save-folder-too-long":
        assert notice == "没有加入：a.mp3。" + too_long_problem
        assert jobs == [(JobKind.NOTES, md.name)]
    elif row == "notes-not-recorded":
        assert notice == f"没有整理：{md.name}。" + data_problem(rig)
        assert jobs == [(JobKind.FILE, "a.mp3")]
    elif row == "nothing-recorded":
        assert notice.split("\n") == ["没有加入：a.mp3。" + data_problem(rig), f"没有整理：{md.name}。" + data_problem(rig)]
        assert jobs == [] and pending == 0
        assert rig.awake.history == []
        assert not out.exists() or list(out.iterdir()) == []
    elif row == "md-unreachable":
        assert notice == NOTES_UNREACHABLE.format(name="掉线.md")
        assert jobs == both
    elif row == "media-unreachable":
        assert notice == "没有加入：" + NOT_READABLE.format(name="掉线.mp4")
        assert jobs == both
    else:
        assert notice == "没有加入：" + "、".join(NOT_READABLE.format(name=name) for name in ("掉线.txt", "掉线.docx"))
        assert jobs == both and pending == 2
    assert not any("程序出了点意外" in s.notice for s in rig.states)
    release()
    rig.wait_idle()
    assert [job.stage for job in rig.store.load_all()] == [Stage.DONE] * len(jobs)
    if (JobKind.NOTES, md.name) in jobs:
        assert notes_files(md) == [NOTES_FILE]
    if row == "others-unreachable":
        assert names(out) == ["a"]


@pytest.mark.parametrize("row", ["upper-case", "title-only", "gbk", "too-long-with-one-that-fits", "odd-names", "many"])
def test_the_edges_of_what_can_be_added(tmp_path, svc, rigs, row):
    """X7 (a)-(f)."""
    rig = rigs()
    controller = rig.controller
    controller.start()
    if row == "upper-case":
        md = lesson(tmp_path, name="大写")
        md = md.rename(md.with_suffix(".MD"))
        controller.add_files([md])
        rig.wait_idle()
        assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE]
        assert notes_files(md) == ["大写-整理笔记.md"]
    elif row == "title-only":
        md = lesson(tmp_path, data=TITLE_ONLY)
        controller.add_files([md])
        rig.wait_idle()
        assert [job.stage for job in notes_jobs(rig.store)] == [Stage.FAILED]
        assert controller.state.notice == f"{md.name}：{NOTES_EMPTY_SOURCE}"
        assert controller.state.failed == 1
        assert svc.gens == []
    elif row == "gbk":
        md = lesson(tmp_path, data=SRC.encode("gbk"))
        controller.add_files([md])
        rig.wait_idle()
        assert [job.stage for job in notes_jobs(rig.store)] == [Stage.FAILED]
        assert NOTES_SOURCE_ENCODING.format(path=md) in controller.state.notice
        assert svc.gens == []
    elif row == "too-long-with-one-that-fits":
        too_long, base = long_lesson(tmp_path / "甲", PATH_LIMIT + 1)
        fits, _ = long_lesson(tmp_path / "乙", PATH_LIMIT)
        release = held(rig, svc)
        controller.add_files([too_long, fits])
        assert controller.state.notice == NOTES_PATH_TOO_LONG_BEFORE.format(name=too_long.name, path=base)
        assert [job.source_md for job in notes_jobs(rig.store)] == [fits]
        release()
        rig.wait_idle()
        assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE]
        assert len(notes_files(fits)) == 1 and notes_files(too_long) == []
    elif row == "odd-names":
        odd = [tmp_path / ("长" * 300 + ".md"), tmp_path / "含\0空.md"]
        reference = []
        for path in odd:
            controller.make_notes(path)
            reference.append(controller.state.notice)
            assert reference[-1] in (NOTES_NOT_A_FILE.format(name=path.name), NOTES_UNREACHABLE.format(name=path.name))
        controller.add_files([tmp_path / "没有.docx"])  # so the next notice is seen to be a new one
        errors = add_safely(rig, odd)
        assert errors == []
        assert controller.state.notice.split("\n") == reference
        assert rig.store.load_all() == []
    else:
        missing = [tmp_path / f"没有{i}.docx" for i in range(200)]
        a = rig.source("a.mp3")
        md = lesson(tmp_path)
        release = held(rig, svc)
        before = controller.state
        since = len(rig.states)
        started = time.perf_counter()
        controller.add_files(missing + [a, md])
        took = time.perf_counter() - started
        during = rig.states[since:]
        notice = controller.state.notice
        assert queued(rig) == [(JobKind.FILE, "a.mp3"), (JobKind.NOTES, md.name)]
        assert notice.startswith("没有加入：没有0.docx（不是文件）、")
        assert notice == "没有加入：" + "、".join(f"没有{i}.docx（不是文件）" for i in range(200))
        assert took < 2.0
        assert new_notices(during, before) == [notice]
        release()
        rig.wait_idle()


X8_REFUSED = ["missing", "folder", "notes-file", "notes-file-numbered", "too-long", "too-long-numbered",
              "in-progress", "waiting"]
X8_QUEUED = ["upper-case", "fits", "title-only", "failed-before", "done-before"]


def _one_way_in(root: Path, rigs, svc, row: str, way: str) -> dict:
    """Prepare ``row`` under ``root``, hand the .md in by ``way``, and say what could be observed."""
    root.mkdir()
    svc.serve()
    md = lesson(root)
    folder = md.parent
    target = md
    if row == "missing":
        target = folder / "没有.md"
    elif row == "folder":
        target = folder / "子.md"
        target.mkdir()
    elif row in ("notes-file", "notes-file-numbered"):
        target = folder / ("课-整理笔记.md" if row == "notes-file" else "课-整理笔记 (2).md")
        target.write_bytes(FINISHED_BYTES)
    elif row == "upper-case":
        target = lesson(root, name="大写")
        target = target.rename(target.with_suffix(".MD"))
    elif row == "too-long":
        target, _ = long_lesson(root / "g", PATH_LIMIT + 1)
    elif row == "too-long-numbered":
        target, base = long_lesson(root / "h", PATH_LIMIT)
        base.write_bytes("旧的\n".encode("utf-8"))
    elif row == "fits":
        target, _ = long_lesson(root / "h", PATH_LIMIT)
    elif row == "title-only":
        target = lesson(root, name="只有标题", data=TITLE_ONLY)
    rig = rigs(root)
    controller = rig.controller
    controller.start()
    gate = None
    if row == "in-progress":
        gate = gated(svc)
        controller.make_notes(md)
        rig.wait(lambda: len(svc.gens) == 1)
    elif row == "waiting":
        svc.serve(BUSY, BUSY)
        controller.make_notes(md)
        rig.wait_idle()
    elif row == "failed-before":
        svc.serve(REFUSED)
        controller.make_notes(md)
        rig.wait_idle()
        svc.serve()
    elif row == "done-before":
        controller.make_notes(md)
        rig.wait_idle()

    def plain(text: str) -> str:
        return text.replace(str(root), "<根>")

    before = controller.state
    since = len(rig.states)
    if way == "make_notes":
        controller.make_notes(target)
    else:
        controller.add_files([target])
    returned = plain(controller.state.notice)
    if gate is not None:
        gate.set()
    rig.wait_idle()
    grew = pending_grew(rig.states[since:], before)
    state = controller.state
    seen = {
        "jobs": [
            (job.stage, plain(job.message), job.notes_model, job.notes_path.name if job.notes_path else None)
            for job in notes_jobs(rig.store)
        ],
        "notice when queued": grew[0].notice if grew else None,
        "notice at the end": plain(state.notice),
        "counts": (state.pending, state.failed, state.waiting_notes),
        "notes files": sorted(
            str(path.relative_to(root)) for path in root.rglob(f"*{NOTES_SUFFIX}*") if "state" not in path.parts
        ),
        "generations": len(svc.gens),
    }
    if row in X8_REFUSED:
        seen["notice on return"] = returned  # the call is synchronous; a queued job may already have a result
    rig.controller.shutdown()
    return seen


@pytest.mark.parametrize("row", X8_REFUSED + X8_QUEUED)
def test_an_md_added_like_a_file_is_treated_exactly_as_make_notes_treats_it(tmp_path, svc, rigs, row):
    """X8."""
    made = _one_way_in(tmp_path / "m", rigs, svc, row, "make_notes")
    added = _one_way_in(tmp_path / "f", rigs, svc, row, "add_files")
    assert added == made
    if row in X8_QUEUED:
        assert added["notice when queued"] == ""  # cleared in the very state that shows it queued
    else:
        assert added["notice when queued"] is None
        assert added["notice on return"] != ""


def test_an_added_md_survives_ollama_trouble_and_the_program_closing(tmp_path, svc, rigs):
    """X9 (a) Ollama is busy, (b) closed with both still queued."""
    svc.serve(BUSY, BUSY)
    md = lesson(tmp_path)
    kept = _snapshot(md)
    rig = rigs()
    rig.controller.start()
    rig.controller.add_files([md, rig.source("a.mp3")])
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.WAITING_NOTES]
    assert [job.stage for job in source_jobs(rig)] == [Stage.DONE]
    assert rig.controller.state.waiting_notes == 1
    assert _snapshot(md) == kept
    svc.serve()
    rig.controller.continue_notes()
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE]
    assert notes_files(md) == [NOTES_FILE]

    root = tmp_path / "b"
    root.mkdir()
    md = lesson(root)
    rig = rigs(root)
    rig.controller.start()
    release = held(rig, svc)
    rig.controller.add_files([rig.source("a.mp3"), md])
    assert queued(rig) == [(JobKind.FILE, "a.mp3"), (JobKind.NOTES, md.name)]  # on record before anything runs
    closer = threading.Thread(target=rig.controller.shutdown, daemon=True)
    closer.start()
    rig.wait(lambda: rig.stopping.is_set())
    release()
    closer.join(10)
    assert not closer.is_alive()
    rig.controller._worker.join(5)
    assert not rig.controller._worker.is_alive()

    svc.serve()
    again = rigs(root)
    again.controller.start()
    again.wait_idle()
    assert [job.stage for job in again.store.load_all()] == [Stage.DONE, Stage.DONE]
    assert notes_files(md) == [NOTES_FILE]
    assert names(again.controller.settings.output_dir) == ["a"]


PX_ROUNDS = 60
PX_KINDS = ("media", "no-media", "md", "same-md", "txt", "notes-file", "docx", "folder", "no-md")
PX_WEIGHTS = (0.2, 0.08, 0.2, 0.12, 0.1, 0.1, 0.08, 0.06, 0.06)
PX_SEED_NOTICE = "没有加入：旧.docx（不是文件）"


def _px_batch(rng: random.Random, forced_empty: bool) -> list[tuple]:
    """What one batch is made of, as (kind, ...) for each item; no file is made here."""
    batch: list[tuple] = []
    suffixes = sorted(MEDIA_SUFFIXES)
    for index in range(0 if forced_empty else rng.randint(0, 8)):
        kind = rng.choices(PX_KINDS, PX_WEIGHTS)[0]
        earlier = [item[1] for item in batch if item[0] == "md"]
        if kind == "same-md" and not earlier:
            kind = "md"
        if kind in ("media", "no-media"):
            suffix = rng.choice(suffixes)
            batch.append((kind, f"{'音' if kind == 'media' else '没有'}{index}{suffix.upper() if rng.random() < 0.5 else suffix}"))
        elif kind == "md":
            batch.append((kind, f"课{index}.md"))
        elif kind == "same-md":
            batch.append((kind, rng.choice(earlier), rng.randrange(3)))
        elif kind == "txt":
            batch.append((kind, f"稿{index}.txt"))
        elif kind == "notes-file":
            batch.append((kind, f"课{index}-整理笔记{rng.choice(['', ' (2)'])}.md"))
        elif kind == "docx":
            batch.append((kind, f"讲义{index}.docx"))
        elif kind == "folder":
            batch.append((kind, f"夹{index}"))
        else:
            batch.append((kind, f"没有{index}.md"))
    return batch


def _px_paths(root: Path, batch: list[tuple]) -> list[Path]:
    """Make the batch real under ``root``."""
    folder = root / "素材"
    folder.mkdir()
    paths = []
    for item in batch:
        kind, name = item[0], item[1]
        path = folder / name
        if kind == "same-md":
            ways = spellings(path)
            path = ways[item[2] % len(ways)]
        elif kind == "folder":
            path.mkdir()
        elif kind == "md":
            path.write_bytes(SRC_BYTES)
        elif kind in ("media", "txt", "notes-file", "docx"):
            path.write_bytes(b"x")
        paths.append(path)
    return paths


def _px_rig(rigs, root: Path, svc):
    root.mkdir(parents=True)
    rig = rigs(root)
    rig.controller.start()
    rig.controller.add_files([root / "旧.docx"])
    assert rig.controller.state.notice == PX_SEED_NOTICE
    return rig, held(rig, svc)


def _px_refusals(notice: str) -> tuple[list[str], list[str]]:
    """The notice of a batch as (what was not added, the other lines)."""
    if notice in ("", PX_SEED_NOTICE):
        return [], []
    lines = notice.split("\n")
    if lines[0].startswith("没有加入："):
        return lines[0][len("没有加入："):].split("、"), lines[1:]
    return [], lines


def test_any_batch_puts_every_path_in_exactly_one_place_whatever_else_is_in_it(tmp_path, svc, rigs):
    """PX."""
    rng = random.Random(20270502)
    seen = Counter()
    for round_ in range(PX_ROUNDS):
        batch = _px_batch(rng, forced_empty=round_ == 0)
        order = list(range(len(batch)))
        rng.shuffle(order)
        first, release_first = _px_rig(rigs, tmp_path / f"r{round_}" / "a", svc)
        second, release_second = _px_rig(rigs, tmp_path / f"r{round_}" / "b", svc)
        paths = _px_paths(first.tmp_path, batch)
        second_paths = _px_paths(second.tmp_path, batch)
        shuffled = [second_paths[i] for i in order]
        case = (round_, batch)

        jobs = expected_jobs(paths)
        notice = expected_notice(first, paths)
        before = first.controller.state
        since = len(first.states)
        first.controller.add_files(paths)
        during = first.states[since:]
        state = first.controller.state
        assert queued(first) == jobs, case
        assert state.notice == notice, case
        assert state.pending == len(jobs), case
        not_added, other_lines = _px_refusals(state.notice)
        assert len(jobs) + len(not_added) + len(other_lines) == len(batch), case
        refused = bool(not_added or other_lines)
        assert new_notices(during, before) == ([notice] if refused else []), case
        if not batch:
            assert during == [], case

        second.controller.add_files(shuffled)
        assert sorted((kind.value, name.casefold()) for kind, name in queued(second)) == sorted(
            (kind.value, name.casefold()) for kind, name in jobs
        ), case
        shuffled_not_added, shuffled_lines = _px_refusals(second.controller.state.notice)
        assert len(second.controller.state.notice.split("\n")) == len(state.notice.split("\n")), case
        assert sorted(shuffled_not_added) == sorted(not_added), case
        assert sorted(line.casefold() for line in shuffled_lines) == sorted(line.casefold() for line in other_lines), case

        kinds = {kind for kind, _ in jobs}
        if kinds == {JobKind.FILE, JobKind.NOTES} and refused:
            seen["media, a proofread .md and a refusal"] += 1
        if any(item[0] == "same-md" for item in batch):
            seen["the same .md twice"] += 1
        if batch and not refused:
            seen["nothing refused"] += 1
        if not batch:
            seen["empty"] += 1

        release_first()
        release_second()
        if round_ % 5 == 0:
            first.wait_idle()
            second.wait_idle()
            noted = 0
            for rig in (first, second):
                assert all(job.stage is Stage.DONE for job in rig.store.load_all()), case
                for job in notes_jobs(rig.store):
                    noted += 1
                    assert len(notes_files(job.source_md)) == 1, case
            assert len(svc.gens) == noted, case
        for rig in (first, second):
            rig.controller.shutdown()
            rig.controller._worker.join(5)
            assert not rig.controller._worker.is_alive(), case
    floors = {"media, a proofread .md and a refusal": 10, "the same .md twice": 8, "nothing refused": 3, "empty": 1}
    print("\nPX coverage:", dict(seen))
    for name, floor in floors.items():
        assert seen[name] >= floor, (name, dict(seen))


# ---------------------------------------------------------------- F: the notice once notes are done


@pytest.mark.parametrize("row", ["nothing-removed", "a-quote", "a-concept", "both"])
def test_the_finished_notice_says_what_was_removed_and_that_the_notes_are_a_draft(tmp_path, svc, rigs, row):
    """F1."""
    reply = {"nothing-removed": NONE_REMOVED, "a-quote": QUOTE_ONLY, "a-concept": TAG_ONLY, "both": UNEQUAL}[row]
    result = finish_notes(reply, SRC, META)
    x, y = result.removed_quotes, result.removed_concepts
    if row == "both":
        assert x > 0 and y > 0 and x != y  # or swapping them would go unseen
    else:
        assert (x, y) == {"nothing-removed": (0, 0), "a-quote": (1, 0), "a-concept": (0, 1)}[row]
    svc.serve(reply)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    rig.controller.make_notes(md)
    rig.wait_idle()

    notice = rig.controller.state.notice
    expected = {
        "nothing-removed": f"笔记已生成：{NOTES_FILE}。笔记是初稿，请自己看一遍。",
        "a-quote": f"笔记已生成：{NOTES_FILE}。删掉了 {x} 条对不上原文的引述。笔记是初稿，请自己看一遍。",
        "a-concept": f"笔记已生成：{NOTES_FILE}。删掉了 {y} 个原文里没有的概念。笔记是初稿，请自己看一遍。",
        "both": f"笔记已生成：{NOTES_FILE}。删掉了 {x} 条对不上原文的引述、{y} 个原文里没有的概念。笔记是初稿，请自己看一遍。",
    }[row]
    assert notice.startswith(f"笔记已生成：{NOTES_FILE}")
    assert notice == expected
    assert notice == done_notice(NOTES_FILE, x, y)
    assert ("删掉了" in notice) == (row != "nothing-removed")
    assert ("概念" in notice) == (y > 0)
    assert ("引述" in notice) == (x > 0)
    assert notice.count(NOTES_DRAFT) == 1
    assert (md.parent / NOTES_FILE).read_bytes() == result.markdown.encode("utf-8")
    job = notes_jobs(rig.store)[0]
    assert (job.removed_quotes, job.removed_concepts) == (x, y)


def test_the_finished_notice_says_when_the_shipped_rules_were_used(tmp_path, svc, rigs):
    """F2 (a)-(c)."""
    note = RULES_EMPTY_NOTE.format(path=tmp_path / "整理规则.txt")
    for row, reply, counts in (("removed", GOOD, (2, 2)), ("clean", NONE_REMOVED, (0, 0))):
        root = tmp_path / row
        root.mkdir()
        svc.serve(reply)
        md = lesson(root)
        rig = rigs(root, notes_writer=svc.writer(rules=lambda: NotesRules(RULES, note)))
        rig.controller.start()
        rig.controller.make_notes(md)
        rig.wait_idle()
        assert rig.controller.state.notice == done_notice(NOTES_FILE, *counts) + "\n" + note, row

    job = Job.new_notes(tmp_path / "课.md", "m")
    job.stage = Stage.DONE
    job.notes_path = notes_file_for(job.source_md)
    for nothing in (None, ""):
        job.rules_note = nothing
        text = notes_done_notice(job)
        assert text == "笔记已生成：课-整理笔记.md。笔记是初稿，请自己看一遍。"
        assert "\n" not in text and text.endswith(NOTES_DRAFT)


@pytest.mark.parametrize("row", ["resumed", "retried-with-the-kept-reply", "made-by-itself", "numbered"])
def test_the_counts_in_the_notice_come_from_the_job_record(tmp_path, svc, rigs, monkeypatch, row):
    """F3 (a)-(d)."""
    result = finish_notes(UNEQUAL, SRC, META)
    x, y = result.removed_quotes, result.removed_concepts
    assert x > 0 and y > 0 and x != y
    svc.serve(respond=lambda prompt: UNEQUAL)
    file_name = NOTES_FILE
    if row == "resumed":
        md = lesson(tmp_path)
        new_job(rigs().store, md)
        rig = rigs()
        rig.controller.start()
        rig.wait_idle()
    elif row == "retried-with-the-kept-reply":
        md = lesson(tmp_path)
        first = rigs()
        first.controller.start()
        fault = DiskFault(monkeypatch, md.parent, DISK_FULL)
        first.controller.make_notes(md)
        first.wait_idle()
        assert [job.stage for job in notes_jobs(first.store)] == [Stage.FAILED]
        fault.fix()
        first.controller.shutdown()
        asked = len(svc.gens)
        rig = rigs()
        rig.controller.start()
        rig.controller.retry_failed()
        rig.wait_idle()
        assert len(svc.gens) == asked == 1  # the counts come from the kept reply, checked again
    elif row == "made-by-itself":
        rig = lesson_rig(rigs)
        set_auto(rig, True)
        rig.controller.add_files([rig.source(LESSON_FILE)])
        wait_notes(rig, 1)
    else:
        md = lesson(tmp_path)
        rig = rigs()
        rig.controller.start()
        for _ in range(2):
            rig.controller.make_notes(md)
            rig.wait_idle()
        file_name = f"{NAME}-整理笔记 (2).md"

    expected = f"笔记已生成：{file_name}。删掉了 {x} 条对不上原文的引述、{y} 个原文里没有的概念。笔记是初稿，请自己看一遍。"
    assert rig.controller.state.notice == expected
    job = notes_jobs(rig.store)[-1]
    assert (job.stage, job.removed_quotes, job.removed_concepts) == (Stage.DONE, x, y)
    if row == "made-by-itself":
        notices = [s.notice for s in rig.states]
        assert notices.index(f"完成：{NAME}") < notices.index(expected)


def test_only_finished_notes_get_the_draft_notice(tmp_path, svc, rigs):
    """F4."""
    rig = lesson_rig(rigs, tmp_path / "一")
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    seen = [rig.controller.state.notice]
    assert seen[-1] == f"完成：{NAME}"

    rig = lesson_rig(rigs, tmp_path / "二", proofreader=LessonProofreader(unproofread=1, total=2))
    rig.controller.add_files([rig.source(LESSON_FILE)])
    rig.wait_idle()
    seen.append(rig.controller.state.notice)
    assert seen[-1] == f"完成：{NAME}。有 1 段没校对，保留了原文。"

    svc.serve(REFUSED)
    refused = lesson(tmp_path, name="拒绝")
    rig.controller.make_notes(refused)
    rig.wait_idle()
    seen.append(rig.controller.state.notice)
    assert seen[-1] == f"{refused.name}：{NOTES_REFUSED}"

    svc.serve(BUSY, BUSY)
    rig.controller.make_notes(lesson(tmp_path, name="等待"))
    rig.wait_idle()
    seen.append(rig.controller.state.notice)
    assert seen[-1] == NOTES_BUSY_MESSAGE
    assert not any(NOTES_DRAFT in notice or "删掉了" in notice for notice in seen)

    svc.serve()
    rig.controller.make_notes(lesson(tmp_path))
    rig.wait_idle()
    assert rig.controller.state.notice == (
        f"笔记已生成：{NOTES_FILE}。删掉了 2 条对不上原文的引述、2 个原文里没有的概念。笔记是初稿，请自己看一遍。"
    )


PF_ROUNDS = 1200
_PF_NAME_CHARS = "课程数据库索引第一讲 ab(2)。、删掉了条个0123"
_PF_QUOTES = re.compile(r"(\d+) 条对不上原文的引述")
_PF_CONCEPTS = re.compile(r"(\d+) 个原文里没有的概念")


def test_the_two_counts_can_be_read_back_from_any_finished_notice(tmp_path):
    """PF."""
    rng = random.Random(20270503)
    seen = Counter()
    for round_ in range(PF_ROUNDS):
        name = "".join(rng.choice(_PF_NAME_CHARS) for _ in range(rng.randint(1, 20)))
        job = Job.new_notes(tmp_path / f"{name}.md", "m")
        job.stage = Stage.DONE
        base = notes_file_for(job.source_md)
        number = rng.choice([None, 2, 3, 17])
        job.notes_path = base if number is None else base.with_name(f"{base.stem} ({number}).md")
        job.removed_quotes = rng.choice([0, 0, 1, 2, 9, 10, 99, 1000, rng.randint(0, 10**6)])
        job.removed_concepts = rng.choice([0, 0, 1, 2, 9, 10, 99, 1000, rng.randint(0, 10**6)])
        said = "".join(rng.choice(_PF_NAME_CHARS + "删掉了 3 条对不上原文的引述") for _ in range(rng.randint(1, 30)))
        job.rules_note = rng.choice([None, "", said])
        x, y = job.removed_quotes, job.removed_concepts
        case = (round_, name, x, y, job.rules_note)

        text = notes_done_notice(job)
        head = f"笔记已生成：{job.notes_path.name}。"
        assert text.startswith(head), case
        first, *more = text[len(head):].split("\n")
        quotes = _PF_QUOTES.search(first)
        concepts = _PF_CONCEPTS.search(first)
        assert (int(quotes.group(1)) if quotes else 0) == x, case
        assert (int(concepts.group(1)) if concepts else 0) == y, case
        assert (quotes is None, concepts is None) == (x == 0, y == 0), case
        assert ("删掉了" in first) == (x + y > 0), case
        assert first.endswith(NOTES_DRAFT), case
        assert more == ([job.rules_note] if job.rules_note else []), case
        assert notes_done_notice(Job.from_dict(job.to_dict())) == text, case
        assert notes_done_notice(job) == text, case
        seen[(x > 0, y > 0)] += 1
        if job.rules_note:
            seen["rules note"] += 1
    print("\nPF coverage:", dict(seen))
    for cell in ((False, False), (True, False), (False, True), (True, True)):
        assert seen[cell] >= 40, (cell, dict(seen))
    assert seen["rules note"] >= 60, dict(seen)
