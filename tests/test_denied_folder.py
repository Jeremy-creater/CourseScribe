"""A folder that refuses new files must make saving fail quickly, never hang (M2 bug fix).

On Windows, ``tempfile.mkstemp`` retried a refused file creation about two billion times. The
refusal is injected at ``os.open``, the call that really creates the file, so any retry loop around
it runs as it would on a real refusal; the gui cases at the end refuse for real with a folder ACL."""

from __future__ import annotations

import importlib
import os
import random
import re
import stat
import threading
import time
from pathlib import Path

import pytest

import coursescribe.files as files
from coursescribe.files import atomic_write_json, commit_new_text, create_once, read_json
from coursescribe.jobs import _NAME_BUDGET, MIN_NAME, PATH_LIMIT, JobStore, SaveFolderTooLong, Stage, _units, job_name, path_fits
from coursescribe.pipeline import ProofreadBlocked
from helpers import WATCHED_THREAD, DeniedFolder, Watchdog, deny_new_files, finishes_within
from test_controller import Rig, _snapshot
from test_pipeline import FakeProofreader, _job, _pipeline

TEMP_NAME = re.compile(r"^\.coursescribe-.+\.tmp$")
TEMP_NAME_LIMIT = 26  # the path budget in jobs.py leaves room for a temporary name this long, no longer
GENERIC_FAILURE = "处理时出错了。录音和已有文件都还在，详情见日志。"


def denied_error(errno=13):
    return PermissionError(errno, "Permission denied")


def clash_error():
    return FileExistsError(17, "File exists")


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def temp_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob(".coursescribe-*.tmp"))


@pytest.fixture
def folder(tmp_path) -> Path:
    made = tmp_path / "存"
    made.mkdir()
    return made


# ---------------------------------------------------------------- F1


def test_a_folder_that_refuses_new_files_gives_up_quickly_without_waiting(folder, monkeypatch):
    """F1: the bug. Windows refused every new file and the save never came back."""
    denied = DeniedFolder(monkeypatch, folder)
    sleeps: list[float] = []
    real_sleep = time.sleep

    def sleep(seconds):
        if threading.current_thread().name.startswith(WATCHED_THREAD):
            sleeps.append(seconds)  # the call under test runs on the watchdog's thread
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)
    started = time.perf_counter()
    with pytest.raises(PermissionError) as caught:
        finishes_within(5, create_once, folder / "a.txt", b"x")
    took = time.perf_counter() - started
    assert caught.value.errno == 13
    assert 2 <= len(denied.attempts) <= 10
    assert denied.kept_trying is False
    assert all(TEMP_NAME.match(p.name) for p in denied.attempts), denied.attempts
    assert sleeps == []
    assert took < 0.5
    assert names(folder) == []


# ---------------------------------------------------------------- F2


def _always_clash(path):
    raise clash_error()


def _nine_refusals_then_a_real_clash(denied_ref):
    def step(path):
        count = len(denied_ref[0].attempts)
        if count <= 9:
            raise denied_error()
        if count == 10:
            path.write_bytes(b"other")  # someone else's file; the real os.open then reports the clash
        return None

    return step


def _nine_clashes_then_a_refusal(denied_ref):
    def step(path):
        count = len(denied_ref[0].attempts)
        if count <= 9:
            raise clash_error()
        if count == 10:
            raise denied_error()
        return None

    return step


@pytest.mark.parametrize("row", ["a-always-clash", "b-refusals-then-clash", "c-clashes-then-refusal"])
def test_endless_or_mixed_refusals_give_up_after_ten_with_the_last_error(folder, monkeypatch, row):
    """F2."""
    ref: list[DeniedFolder] = []
    if row.startswith("a"):
        step, expected = _always_clash, FileExistsError
    elif row.startswith("b"):
        step, expected = _nine_refusals_then_a_real_clash(ref), FileExistsError
    else:
        step, expected = _nine_clashes_then_a_refusal(ref), PermissionError
    denied = DeniedFolder(monkeypatch, folder, error=step)
    ref.append(denied)
    with pytest.raises(OSError) as caught:
        finishes_within(5, create_once, folder / "a.txt", b"x")
    assert type(caught.value) is expected, repr(caught.value)
    if expected is PermissionError:
        assert caught.value.errno == 13
    assert len(denied.attempts) == 10
    assert denied.kept_trying is False
    if row.startswith("b"):
        tenth = denied.attempts[9]
        assert names(folder) == [tenth.name]
        assert tenth.read_bytes() == b"other"
    else:
        assert names(folder) == []


# ---------------------------------------------------------------- F3, F4


def test_two_refusals_then_the_file_is_written_as_usual(folder, monkeypatch):
    """F3."""
    denied = DeniedFolder(monkeypatch, folder, times=2)
    target = folder / "课.txt"
    assert finishes_within(5, commit_new_text, target, "第一行\n第二行\n") == target
    assert target.read_bytes() == "第一行\n第二行\n".encode("utf-8")
    assert len(denied.attempts) == 3
    assert len({p.name for p in denied.attempts}) == 3
    assert names(folder) == ["课.txt"]


def test_a_real_clash_takes_a_new_name_and_leaves_the_other_file_alone(folder, monkeypatch):
    """F4."""
    planted: list[Path] = []

    def step(path):
        if not planted:
            path.write_bytes(b"someone else's")
            planted.append(path)
        return None

    denied = DeniedFolder(monkeypatch, folder, error=step)
    finishes_within(5, atomic_write_json, folder / "job.json", {"a": "中"})
    assert read_json(folder / "job.json") == {"a": "中"}
    assert len(denied.attempts) == 2
    assert names(folder) == sorted(["job.json", planted[0].name])
    assert planted[0].read_bytes() == b"someone else's"


# ---------------------------------------------------------------- F5


@pytest.mark.parametrize("errno", [28, 5], ids=["disk-full", "drive-gone"])
def test_other_errors_are_raised_at_once(folder, monkeypatch, errno):
    """F5 (a)(b)."""
    denied = DeniedFolder(monkeypatch, folder, error=OSError(errno, os.strerror(errno)))
    with pytest.raises(OSError) as caught:
        finishes_within(5, commit_new_text, folder / "a.txt", "x")
    assert caught.value.errno == errno
    assert not isinstance(caught.value, PermissionError)
    assert len(denied.attempts) == 1
    assert names(folder) == []


def test_a_missing_folder_is_reported_as_missing(tmp_path):
    """F5 (c): nothing stands in here."""
    with pytest.raises(FileNotFoundError):
        finishes_within(5, create_once, tmp_path / "不存在" / "a.txt", b"x")
    assert not (tmp_path / "不存在").exists()


# ---------------------------------------------------------------- F6


@pytest.mark.parametrize("way", ["create_once", "commit_new_text", "atomic_write_json"])
def test_every_public_way_of_writing_gives_up_in_a_refusing_folder(folder, monkeypatch, way):
    """F6."""
    if way == "atomic_write_json":
        atomic_write_json(folder / "job.json", {"v": 1})
    denied = DeniedFolder(monkeypatch, folder)
    calls = {
        "create_once": (create_once, folder / "词表.txt", b"x"),
        "commit_new_text": (commit_new_text, folder / "课.md", "# 课\n"),
        "atomic_write_json": (atomic_write_json, folder / "job.json", {"v": 2}),
    }
    with pytest.raises(PermissionError) as caught:
        finishes_within(5, *calls[way])
    assert isinstance(caught.value, OSError)
    assert denied.kept_trying is False
    assert len(denied.attempts) <= 10
    if way == "atomic_write_json":
        assert read_json(folder / "job.json") == {"v": 1}
        assert names(folder) == ["job.json"]
    else:
        assert names(folder) == []


# ---------------------------------------------------------------- F7


def _sample_bytes(rng: random.Random, turn: int) -> bytes:
    kind = turn % 6
    if kind == 0:
        return b""
    if kind == 1:
        return b"\n"
    if kind == 2:
        return b"".join(rng.choice([b"\r\n", b"\n", "行".encode("utf-8"), b"a"]) for _ in range(rng.randint(1, 50)))
    if kind == 3:
        return b"".join(rng.choice([b"\x1a", b"\x00", b"\n", b"x"]) for _ in range(rng.randint(1, 50)))
    return rng.randbytes(rng.randint(0, 4096))


def test_written_bytes_are_exactly_the_bytes_given(tmp_path):
    """F7: generated inputs, one round of a megabyte."""
    rng = random.Random(20271101)
    for turn in range(100):
        data = rng.randbytes(1024 * 1024) if turn == 99 else _sample_bytes(rng, turn)
        sub = tmp_path / str(turn)
        sub.mkdir()
        finishes_within(5, create_once, sub / "a.bin", data)
        assert (sub / "a.bin").read_bytes() == data, f"turn {turn}: {data[:80]!r}"
        assert names(sub) == ["a.bin"], f"turn {turn}"
        if os.name != "nt":  # Windows uses the mode only for the read-only attribute
            assert stat.S_IMODE(os.stat(sub / "a.bin").st_mode) == 0o600, f"turn {turn}"


# ---------------------------------------------------------------- F8


def test_threads_writing_into_one_folder_all_succeed_and_leave_no_temporary_files(folder):
    """F8."""
    barrier = threading.Barrier(8, timeout=5)
    errors: list[BaseException] = []

    def work(i):
        try:
            barrier.wait()
            for j in range(50):
                atomic_write_json(folder / f"{i}-{j}.json", {"i": i, "j": j})
        except BaseException as exc:  # noqa: BLE001  reported below, including a broken barrier
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    if any(thread.is_alive() for thread in threads):
        pytest.fail("线程 10 秒还没结束")
    assert errors == []
    for i in range(8):
        for j in range(50):
            assert read_json(folder / f"{i}-{j}.json") == {"i": i, "j": j}
    assert len(names(folder)) == 400
    assert temp_files(folder) == []


# ---------------------------------------------------------------- F9


@pytest.mark.parametrize("times", [0, 3], ids=["recorded-only", "three-refusals"])
def test_temporary_paths_fit_when_the_save_folder_is_at_the_limit(tmp_path, monkeypatch, times):
    """F9: the save folder leaves names only MIN_NAME long; the temporary file must still fit."""
    edge = _NAME_BUDGET - 2 * MIN_NAME
    assert edge == 203
    k = edge - _units(str(tmp_path)) - 1
    if k < 1:
        pytest.fail(f"The temporary folder {tmp_path} is too long for this test; run pytest with a shorter --basetemp")
    root = tmp_path / ("r" * k)
    assert _units(str(root)) == edge
    name = job_name("课" * 30, root)
    assert _units(name) == MIN_NAME
    with pytest.raises(SaveFolderTooLong):
        job_name("课" * 30, Path(str(root) + "r"))
    folder = root / f"{name} (99)"
    folder.mkdir(parents=True)
    target = folder / f"{name} (99).txt"
    assert _units(str(target)) == PATH_LIMIT
    seen = DeniedFolder(monkeypatch, folder, times=times)
    assert finishes_within(5, commit_new_text, target, "x") == target
    assert target.read_bytes() == b"x"
    for p in seen.attempts:
        assert path_fits(p), (p, _units(str(p)))
        assert _units(p.name) <= TEMP_NAME_LIMIT, p.name
    if times == 0:
        assert len(seen.attempts) >= 1
    else:
        assert len(seen.attempts) == 4
    assert names(folder) == [target.name]


# ---------------------------------------------------------------- F10


@pytest.fixture
def reloaded_files():
    """Reload coursescribe.files on demand, as a restart would; afterwards the original module comes back.

    ``reload`` runs the module again in the same namespace, so other modules holding its functions
    would see the new globals and new classes; the namespace is put back as it was."""
    saved = dict(vars(files))

    def reload():
        return importlib.reload(files)

    try:
        yield reload
    finally:
        vars(files).clear()
        vars(files).update(saved)
        assert files.create_once is saved["create_once"]


def test_after_a_restart_temporary_names_do_not_repeat(folder, monkeypatch, reloaded_files):
    """F10: files left by the last run, killed mid-save, must not block this run.

    Both runs start from a freshly loaded module: earlier tests have already made many temporary
    files, so a seeded or counting name generator would be far along its sequence in the first run
    and would not repeat after a single reload."""
    first = reloaded_files()
    denied = DeniedFolder(monkeypatch, folder)
    with pytest.raises(PermissionError):
        finishes_within(5, first.create_once, folder / "a.txt", b"x")
    old = [p.name for p in denied.attempts]
    denied.fix()
    assert len(old) == 10
    assert len(set(old)) == 10

    mod = reloaded_files()
    for k, left in enumerate(old):
        (folder / left).write_bytes(f"old-{k}".encode())
    seen = DeniedFolder(monkeypatch, folder, times=0)
    finishes_within(5, mod.create_once, folder / "a.txt", b"y")
    assert (folder / "a.txt").read_bytes() == b"y"
    assert {p.name for p in seen.attempts}.isdisjoint(old)
    assert names(folder) == sorted(["a.txt", *old])
    for k, left in enumerate(old):
        assert (folder / left).read_bytes() == f"old-{k}".encode()


# ---------------------------------------------------------------- P1

ANSWERS = ("deny", "clash", "ok", "other")
TRIES = 10


def _script(rng: random.Random, turn: int) -> list[str]:
    if turn % 10 == 0:  # a giving-up round: ten refusals or clashes, then answers that would succeed
        return [rng.choice(("deny", "clash")) for _ in range(TRIES)] + ["ok"] * 5
    return rng.choices(ANSWERS, weights=(0.35, 0.2, 0.3, 0.15), k=rng.randint(1, 15))


def _reference(script: list[str]) -> tuple[str, type | None, int | None, int]:
    """What must happen, worked out without the code under test: (outcome, error type, errno, tries)."""
    answers = (script + ["ok"] * TRIES)[:TRIES]
    for position, answer in enumerate(answers, 1):
        if answer == "ok":
            return "ok", None, None, position
        if answer == "other":
            return "raised", OSError, 28, position
    if answers[-1] == "deny":
        return "raised", PermissionError, 13, TRIES
    return "raised", FileExistsError, 17, TRIES


def test_any_run_of_answers_ends_as_the_reference_rule_says(tmp_path):
    """P1: generated runs of system answers; also checks only our own temporary file is ever removed."""
    rng = random.Random(20271102)
    tally = {"ok": 0, "gave_up": 0, "gave_up_deny": 0, "gave_up_clash": 0, "mixed": 0, "other": 0, "clash_then_ok": 0}
    for turn in range(300):
        script = _script(rng, turn)
        data = rng.randbytes(rng.randint(0, 64))
        sub = tmp_path / str(turn)
        sub.mkdir()
        planted: dict[str, bytes] = {}
        ref: list[DeniedFolder] = []

        def step(path, script=script, planted=planted, ref=ref):
            index = len(ref[0].attempts) - 1
            answer = script[index] if index < len(script) else "ok"
            if answer == "deny":
                raise denied_error()
            if answer == "other":
                raise OSError(28, "No space left on device")
            if answer == "clash":
                body = f"other-{index}".encode()
                path.write_bytes(body)
                planted[path.name] = body
            return None

        outcome, error_type, errno, tries = _reference(script)
        where = f"turn {turn}: script {script}, expected {outcome} {error_type} {errno} after {tries}"
        with pytest.MonkeyPatch.context() as patch:
            denied = DeniedFolder(patch, sub, error=step)
            ref.append(denied)
            try:
                finishes_within(5, create_once, sub / "a.txt", data)
                actual = ("ok", None, None)
            except OSError as exc:
                actual = ("raised", type(exc), exc.errno)
        assert actual == (outcome, error_type, errno), f"{where}; got {actual}"
        assert len(denied.attempts) == tries, f"{where}; tried {len(denied.attempts)}"
        assert denied.kept_trying is False, where
        expected_names = sorted([*planted, *(["a.txt"] if outcome == "ok" else [])])
        assert names(sub) == expected_names, where
        for name, body in planted.items():
            assert (sub / name).read_bytes() == body, where
        if outcome == "ok":
            assert (sub / "a.txt").read_bytes() == data, where

        used = (script + ["ok"] * TRIES)[:tries]
        if outcome == "ok":
            tally["ok"] += 1
            tally["clash_then_ok"] += "clash" in used
        elif error_type is OSError:
            tally["other"] += 1
        else:
            tally["gave_up"] += 1
            tally["gave_up_deny"] += used[-1] == "deny"
            tally["gave_up_clash"] += used[-1] == "clash"
            tally["mixed"] += used[0] != used[-1]
    assert tally["ok"] >= 100, tally
    assert tally["gave_up"] >= 3 and tally["gave_up_deny"] >= 1 and tally["gave_up_clash"] >= 1, tally
    assert tally["mixed"] >= 1, tally
    assert tally["other"] >= 40, tally
    assert tally["clash_then_ok"] >= 30, tally


# ---------------------------------------------------------------- controller


def test_a_job_whose_folder_refuses_fails_the_others_finish_and_retry_completes(tmp_path, monkeypatch):
    """C1: the worker must not hang on one job; after the folder is fixed, retrying finishes it."""
    rig = Rig(tmp_path)
    rig.controller.add_files([rig.source("a.mp3"), rig.source("b.mp3")])
    folder_a, folder_b = tmp_path / "out" / "a", tmp_path / "out" / "b"
    assert folder_a.is_dir() and folder_b.is_dir()
    denied = DeniedFolder(monkeypatch, folder_a)
    try:
        rig.controller.start()
        rig.wait_idle()
        jobs = {job.folder.name: job for job in rig.store.load_all()}
        a, b = jobs["a"], jobs["b"]
        assert a.stage is Stage.FAILED and a.can_retry
        assert a.message == GENERIC_FAILURE
        assert b.stage is Stage.DONE
        assert (folder_b / "b.txt").is_file() and (folder_b / "b.md").is_file()
        assert rig.controller.state.pending == 0
        assert denied.kept_trying is False
        assert len(denied.attempts) <= 10
        rig.wait(lambda: rig.awake.held is False, seconds=2.0)
        assert not (folder_a / "a.txt").exists() and not (folder_a / "a.md").exists()
        assert temp_files(folder_a) == []

        denied.fix()
        rig.controller.retry_failed()
        rig.wait_idle()
        again = {job.folder.name: job for job in rig.store.load_all()}["a"]
        assert again.stage is Stage.DONE
        assert names(folder_a) == ["a.md", "a.txt"]
        assert rig.controller.state.pending == 0
    finally:
        denied.fix()
        rig.check_keep_awake_follows_busy()


def test_a_data_folder_that_refuses_new_files_is_explained_when_adding(tmp_path, monkeypatch):
    """C2: JobStore.save goes through atomic_write_json."""
    rig = Rig(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    denied = DeniedFolder(monkeypatch, state)
    try:
        finishes_within(5, rig.controller.add_files, [rig.source("第一课.mp3")])
        notice = rig.controller.state.notice
        assert notice.startswith("没有加入：第一课.mp3。"), notice
        assert "程序自己的数据文件夹" in notice and str(state) in notice
        assert rig.controller.state.pending == 0
        assert denied.kept_trying is False
        assert not (tmp_path / "out" / "第一课").exists()
        assert names(state) == []
    finally:
        denied.fix()
        rig.check_keep_awake_follows_busy()


# ---------------------------------------------------------------- real desktop (gui, by hand)


@pytest.mark.gui
def test_a_really_refused_data_folder_when_adding_a_file(tmp_path):
    """G3."""
    rig = Rig(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    source = rig.source("第一课.mp3")
    try:
        with Watchdog() as dog, deny_new_files(state):  # leaving: the rights come back first, then the join
            started = time.perf_counter()
            dog.run(10, rig.controller.add_files, [source])
            took = time.perf_counter() - started
            notice = rig.controller.state.notice
            assert notice.startswith("没有加入：第一课.mp3。"), notice
            assert "程序自己的数据文件夹" in notice and str(state) in notice
            assert rig.controller.state.pending == 0
            assert took < 2
            assert not (tmp_path / "out" / "第一课").exists()
            assert names(state) == []
    finally:
        rig.controller.shutdown()


@pytest.mark.gui
def test_a_really_refused_save_folder_when_writing_the_proofread_md(tmp_path):
    """G4."""
    store = JobStore(tmp_path / "state")
    job = _job(tmp_path, store)
    _pipeline(store, proofreader=FakeProofreader(error=ProofreadBlocked("校对服务没有启动"))).run(job)
    assert job.stage is Stage.WAITING_PROOFREAD and job.txt_path.is_file()
    txt_before = _snapshot(job.txt_path)
    with Watchdog() as dog, deny_new_files(job.folder):  # leaving: the rights come back first, then the join
        started = time.perf_counter()
        dog.run(10, _pipeline(store).run, job)
        took = time.perf_counter() - started
        assert job.stage is Stage.FAILED and job.can_retry
        assert job.message == GENERIC_FAILURE
        assert took < 2
        assert _snapshot(job.txt_path) == txt_before
        assert not job.md_path.exists()
        assert temp_files(job.folder) == []
    job.stage = job.retry_stage()
    _pipeline(store).run(job)
    assert job.stage is Stage.DONE
    assert job.md_path.is_file()
