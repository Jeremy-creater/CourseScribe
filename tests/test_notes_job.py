"""Notes jobs (12-6): the job record, the notes step of the pipeline, and what the controller
does for "生成整理笔记", "继续整理笔记", "重试" and "忽略". The service is a real local HTTP
server (FakeOllama) talked to by the real client and the real notes writer; the only other
stand-ins are at the system's edge (a held file, a refused new file, a failed job record, the
power state). The integration and gui cases at the end use the real Ollama and the real desktop."""

from __future__ import annotations

import codecs
import json
import logging
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import pytest

from coursescribe.controller import (
    NOTES_ALREADY,
    NOTES_ALREADY_WAITING,
    NOTES_IS_NOTES,
    NOTES_NOT_A_FILE,
    NOTES_NOT_MD,
    NOTES_PATH_TOO_LONG_BEFORE,
)
from coursescribe.jobs import (
    NOTES_SUFFIX,
    PATH_LIMIT,
    Job,
    JobKind,
    JobStore,
    Stage,
    is_notes_file,
    notes_file_for,
    path_fits,
)
from coursescribe.models import DEFAULT_NOTES_MODEL
from coursescribe.notes import EmptyNotes, NotesMeta, finish_notes
from coursescribe.notes_rules import NotesRules
from coursescribe.notes_writer import (
    NOTES_BUSY_MESSAGE,
    NOTES_CUT_OFF,
    NOTES_EMPTY_SOURCE,
    NOTES_MODEL_MISSING,
    NOTES_REFUSED,
    NOTES_TOO_LONG,
    OllamaNotesWriter,
)
from coursescribe.ollama import DEFAULT_URL, OllamaClient
from coursescribe.pipeline import (
    NOTES_PATH_TOO_LONG,
    NOTES_SOURCE_ENCODING,
    NOTES_SOURCE_HELD,
    NOTES_SOURCE_MISSING,
    NOTES_SOURCE_UNREADABLE,
    NOTES_UNEXPECTED,
    NOTES_WORKING,
    NOTES_WRITE_FAILED,
    Pipeline,
)
from coursescribe.platform import FILES_CAN_BE_HELD
from coursescribe.proofread import OllamaProofreader
from fake_ollama import FakeOllama
from helpers import hold_without_sharing
from notes_replies import META, make_reply
from test_controller import FakeTranscriber as RigTranscriber
from test_controller import Rig, _job_records_fail, _snapshot, _u16
from test_notes import FINISHED, SRC, header
from test_notes_writer import (
    BUSY,
    CUT,
    GOOD,
    LESSON,
    LESSON_META,
    MADE_UP_QUOTE,
    REFUSED,
    RULES,
    generations,
    reset,
    sent,
    unloads,
)
from test_pipeline import Crash, CrashingStore, FakeProofreader, FakeTranscriber, _three_part_model
from test_proofread import NOTES_MODEL, _ready_notes_model
from test_recording import _held

NAME = "2026.10.30 数据库索引"
NOTES_FILE = f"{NAME}-整理笔记.md"
SRC_BYTES = SRC.encode("utf-8")
SRC_CRLF = SRC.replace("\n", "\r\n")
FINISHED_BYTES = FINISHED.encode("utf-8")
TXT_BYTES = "原始文字稿\n".encode("utf-8")
MODELS = ("model-a", "model-b", "model-c", DEFAULT_NOTES_MODEL)
NOW = datetime(2026, 11, 1, 19, 30)
DISK_FULL = OSError(28, "No space left on device")
READ_ONLY = PermissionError(13, "Permission denied")
TOO_LONG_SOURCE = "# 课\n\n" + "索" * 100  # with num_ctx=407, reply_budget=50 and two-character rules (12-5 W5)
TITLE_ONLY = f"# {NAME}\n".encode("utf-8")


def good_rules() -> NotesRules:
    return NotesRules(RULES, None)


def two_char_rules() -> NotesRules:
    return NotesRules("规则", None)


# ---------------------------------------------------------------- tools


@dataclass
class Service:
    fake: FakeOllama
    client: OllamaClient

    def serve(self, *script, models=MODELS, then=None, respond=None) -> FakeOllama:
        """A new script for the shared fake and empty records; without a script it answers GOOD."""
        reset(self.fake, *script, then=then)
        if respond is not None:
            self.fake.respond = respond
        self.fake.models = list(models)
        return self.fake

    def writer(self, rules=good_rules, **kwargs) -> OllamaNotesWriter:
        kwargs.setdefault("retry_pause", 0)
        return OllamaNotesWriter(self.client, rules, **kwargs)

    @property
    def gens(self) -> list[dict]:
        return generations(self.fake)


@pytest.fixture(scope="module")
def ollama():
    fake = FakeOllama(lambda prompt: GOOD, models=MODELS)
    yield Service(fake, OllamaClient(fake.url, timeout=5, unload_timeout=2))
    fake.close()


@pytest.fixture
def svc(ollama) -> Service:
    ollama.serve()
    return ollama


def lesson(root: Path, name: str = NAME, data: bytes = SRC_BYTES, folder_name: str = "课程") -> Path:
    """A proofread .md and its .txt in ``root/课程``, made up here."""
    folder = root / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    md = folder / f"{name}.md"
    md.write_bytes(data)
    (folder / f"{name}.txt").write_bytes(TXT_BYTES)
    return md


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def temp_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob(".coursescribe-*.tmp"))


def snapshots(folder: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: _snapshot(p) for p in folder.iterdir() if p.is_file()}


def unchanged(folder: Path, before: dict[str, tuple[bytes, int]]) -> bool:
    return all((folder / name).is_file() and _snapshot(folder / name) == snap for name, snap in before.items())


class Flow:
    """A pipeline whose transcriber and proofreader must never be used by a notes job."""

    def __init__(self, store: JobStore, writer: OllamaNotesWriter) -> None:
        self.statuses: list[str] = []
        self.transcriber = FakeTranscriber()
        self.proofreader = FakeProofreader()
        self.pipeline = Pipeline(
            store,
            self.transcriber,
            self.proofreader,
            lambda path: 5.0,
            on_status=lambda job, text, fraction: self.statuses.append(text),
            notes_writer=writer,
        )

    def run(self, job: Job) -> Job:
        self.pipeline.run(job)
        assert self.transcriber.starts == [] and self.proofreader.calls == []
        return job


def new_job(store: JobStore, md: Path, model: str = "model-a") -> Job:
    job = Job.new_notes(md, model)
    store.save(job)
    return job


def retry(job: Job) -> Job:
    job.stage = job.retry_stage()
    job.message = ""
    return job


class DiskFault:
    """The system refuses new files in ``folder`` only (disk full, read-only); elsewhere files are made for real."""

    def __init__(self, monkeypatch, folder: Path, error: OSError) -> None:
        self.folder = folder
        self.error: OSError | None = error
        real = tempfile.mkstemp

        def mkstemp(*args, **kwargs):
            where = kwargs.get("dir")
            if self.error is not None and where is not None and Path(where) == self.folder:
                raise type(self.error)(*self.error.args)
            return real(*args, **kwargs)

        monkeypatch.setattr("coursescribe.files.tempfile.mkstemp", mkstemp)

    def fix(self) -> None:
        self.error = None


def same_file(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def notes_jobs(store: JobStore, md: Path | None = None) -> list[Job]:
    return [
        job
        for job in store.load_all()
        if job.kind is JobKind.NOTES and (md is None or (job.source_md is not None and same_file(job.source_md, md)))
    ]


def notes_files(md: Path) -> list[str]:
    return [name for name in names(md.parent) if name.startswith(f"{md.stem}{NOTES_SUFFIX}")]


def case_blind() -> bool:
    return os.path.normcase("A") == "a"


def spellings(md: Path) -> list[Path]:
    """Ways a user or the system may name the same file."""
    ways = [md, md.parent / ".." / md.parent.name / md.name]
    if case_blind():
        ways.append(Path(str(md).upper()))
    return ways


def long_lesson(root: Path, units: int) -> tuple[Path, Path]:
    """A lesson whose notes path at the base name is exactly ``units`` long, as Windows counts it."""
    folder = root / "长"
    folder.mkdir(parents=True)
    k = units - _u16(str(folder / f"{NOTES_SUFFIX}.md"))
    if k < 1:
        pytest.fail(f"The temporary folder {folder} is too long for this test; run pytest with a shorter --basetemp")
    stem = "课" * k
    md = folder / f"{stem}.md"
    md.write_bytes(SRC_BYTES)
    base = folder / f"{stem}{NOTES_SUFFIX}.md"
    assert _u16(str(base)) == units
    return md, base


@pytest.fixture
def rigs(tmp_path, svc):
    """Rigs with the real notes writer; each ends with the keep-awake law checked."""
    made: list[Rig] = []

    def make(root: Path | None = None, **kwargs) -> Rig:
        kwargs.setdefault("notes_writer", svc.writer())
        rig = Rig(root or tmp_path, **kwargs)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.check_keep_awake_follows_busy()


def set_notes_model(rig: Rig, model: str) -> None:
    rig.controller.update_settings(replace(rig.controller.settings, notes_model=model))


def saw_pending_one(rig: Rig, since: int) -> bool:
    """A click since ``since`` was shown with one job pending. Read from the published states, not
    the state now: the worker may already have finished the job and published 0 after it."""
    return any(state.pending == 1 for state in rig.states[since:])


def gated(svc: Service) -> threading.Event:
    """The service holds every generation until the gate opens, then answers GOOD."""
    gate = threading.Event()

    def respond(prompt):
        gate.wait(5)
        return GOOD

    svc.serve(respond=respond)
    return gate


# ---------------------------------------------------------------- jobs: the record of a notes job

OLD_KEYS = {
    "job_id", "kind", "stage", "folder", "audio_path", "txt_path", "md_path", "transcribe_model",
    "proofread_model", "created_ns", "message", "unproofread_chunks", "total_chunks",
    "model_download_requested", "attempts", "can_retry", "dismissed", "transcript_complete",
}
NEW_DEFAULTS = {
    "source_md": None, "notes_path": None, "notes_model": "", "removed_quotes": 0, "removed_concepts": 0,
    "rules_note": None,
}
_NAME_CHARS = "课程数据库索引第一讲 ab(2)"


def _random_name(rng: random.Random) -> str:
    return "".join(rng.choice(_NAME_CHARS) for _ in range(rng.randint(1, 12))).strip() or "课"


def _random_job(rng: random.Random, root: Path, index: int) -> Job:
    kind = rng.choice(list(JobKind))
    model = f"qwen{rng.randint(1, 9)}:{rng.randint(1, 30)}b"
    if kind is JobKind.NOTES:
        job = Job.new_notes(root / _random_name(rng) / f"{_random_name(rng)}.md", model)
    elif kind is JobKind.RECORDING:
        job = Job.new_recording(root / f"r{index}", NOW, "large-v3", "model-a")
    else:
        job = Job.new_file(root / f"s{index}.mp3", root / f"f{index}", NOW, "small", "model-b")
    job.stage = rng.choice(list(Stage))
    job.source_md = None if rng.random() < 0.3 else root / _random_name(rng) / f"{_random_name(rng)} (2)-整理笔记.md"
    job.notes_path = None if rng.random() < 0.3 else root / _random_name(rng) / f"{_random_name(rng)}-整理笔记 (2).md"
    job.notes_model = model
    job.removed_quotes = rng.randint(0, 50)
    job.removed_concepts = rng.randint(0, 50)
    job.rules_note = None if rng.random() < 0.5 else f"规则文件 {root / '整理规则.txt'} 里没有“可用”的规则"
    job.message = rng.choice(["", "处理时出错了。", f"“{_random_name(rng)}”：等一会儿"])
    job.attempts = rng.randint(0, 3)
    job.can_retry = rng.random() < 0.5
    job.dismissed = rng.random() < 0.5
    job.transcript_complete = rng.random() < 0.5
    job.model_download_requested = rng.random() < 0.5
    job.unproofread_chunks = rng.randint(0, 5)
    job.total_chunks = rng.randint(0, 9)
    return job


def _json_ready(value) -> bool:
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_ready(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_ready(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def test_any_job_record_survives_json_and_the_store_unchanged(tmp_path):
    """J1."""
    rng = random.Random(20270301)
    jobs = [_random_job(rng, tmp_path, i) for i in range(200)]
    for job in jobs:
        raw = job.to_dict()
        assert _json_ready(raw), (job, raw)
        assert set(raw) == OLD_KEYS | set(NEW_DEFAULTS), job
        for key in ("source_md", "notes_path"):
            value = getattr(job, key)
            assert raw[key] == (None if value is None else str(value)), job
        assert Job.from_dict(json.loads(json.dumps(raw))) == job, job

    store = JobStore(tmp_path / "state")
    stored = jobs[:20]
    for job in stored:
        store.save(job)
    assert JobStore(tmp_path / "state").load_all() == sorted(stored, key=lambda job: job.created_ns)
    assert names(store.folder) == sorted(f"{job.job_id}.json" for job in stored)


def test_an_old_job_file_reads_with_the_new_fields_at_their_defaults(tmp_path):
    """J2 (a): a record as the version before notes wrote it."""
    out = tmp_path / "out" / "第一课"
    old = {
        "job_id": "abc", "kind": "file", "stage": "done", "folder": str(out),
        "audio_path": str(tmp_path / "input" / "第一课.mp3"), "txt_path": str(out / "第一课.txt"),
        "md_path": str(out / "第一课.md"), "transcribe_model": "large-v3", "proofread_model": "model-a",
        "created_ns": 1761000000000000000, "message": "有 2 段没校对，保留了原文。", "unproofread_chunks": 2,
        "total_chunks": 5, "model_download_requested": False, "attempts": 0, "can_retry": True,
        "dismissed": False, "transcript_complete": True,
    }
    state = tmp_path / "state"
    state.mkdir()
    (state / "abc.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    before = (state / "abc.json").read_bytes()
    store = JobStore(state)
    loaded = store.load_all()
    assert (state / "abc.json").read_bytes() == before
    assert len(loaded) == 1
    job = loaded[0]
    assert job.to_dict() == {**old, **NEW_DEFAULTS}
    assert (job.kind, job.stage, job.folder, job.unproofread_chunks, job.transcript_complete) == (
        JobKind.FILE, Stage.DONE, out, 2, True,
    )
    assert job.source_md is None and job.notes_path is None and job.rules_note is None
    assert (job.notes_model, job.removed_quotes, job.removed_concepts) == ("", 0, 0)
    store.save(job)
    saved = json.loads((state / "abc.json").read_text(encoding="utf-8"))
    assert saved == {**old, **NEW_DEFAULTS}


def test_records_missing_any_of_the_new_keys_read_as_the_defaults(tmp_path):
    """J2 (b)."""
    rng = random.Random(20270302)
    for i in range(50):
        job = _random_job(rng, tmp_path, i)
        raw = job.to_dict()
        assert _json_ready(raw), raw
        dropped = [key for key in NEW_DEFAULTS if rng.random() < 0.5]
        for key in dropped:
            del raw[key]
        expected = replace(job, **{key: NEW_DEFAULTS[key] for key in dropped})
        assert Job.from_dict(json.loads(json.dumps(raw))) == expected, dropped


def test_working_files_are_never_taken_for_jobs_but_a_broken_record_is_reported(tmp_path, caplog):
    """J3."""
    rng = random.Random(20270303)
    checkpoints = (
        json.dumps({"version": 1, "fingerprint": "f" * 64, "reply": "## 核心宗旨\n索引"}).encode("utf-8"),
        b"{",
        b"",
    )
    for round_ in range(30):
        root = tmp_path / f"r{round_}"
        store = JobStore(root / "state")
        jobs = [_random_job(rng, root, i) for i in range(rng.randint(1, 5))]
        for job in jobs:
            store.save(job)
            if rng.random() < 0.6:
                store.notes_path(job).write_bytes(rng.choice(checkpoints))
            if rng.random() < 0.5:
                store.proofread_path(job).write_bytes(b'{"chunks": {}}')
            if rng.random() < 0.5:
                store.segments_path(job).write_bytes(b'{"start": 0}\n')
        (store.folder / "broken.json").write_bytes(b"{not json")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="coursescribe.jobs"):
            loaded = store.load_all()
        assert {job.job_id for job in loaded} == {job.job_id for job in jobs}
        warnings = [r for r in caplog.records if r.name == "coursescribe.jobs" and r.levelno >= logging.WARNING]
        assert len(warnings) == 1 and "broken.json" in warnings[0].getMessage(), [r.getMessage() for r in warnings]


def test_a_notes_job_only_notes_the_md_down_and_touches_nothing(tmp_path, monkeypatch):
    """J4."""
    md = lesson(tmp_path)
    folder = md.parent
    listing = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    job = Job.new_notes(md, "model-a")
    assert (job.kind, job.stage) == (JobKind.NOTES, Stage.NOTES)
    assert job.source_md == md and job.folder == folder
    assert job.audio_path == job.md_path == md
    assert job.txt_path == folder / f"{NAME}.txt"
    assert job.notes_path is None and job.notes_model == "model-a"
    assert job.transcribe_model == job.proofread_model == ""
    assert job.label == f"{NAME}.md"

    monkeypatch.chdir(tmp_path)
    relative = Job.new_notes(Path("课程/../课程/x.md"), "m")
    assert relative.source_md.is_absolute()
    assert relative.source_md == Path(os.path.abspath("课程/../课程/x.md"))

    nowhere = Job.new_notes(tmp_path / "不存在" / "x.md", "m")
    assert nowhere.source_md == tmp_path / "不存在" / "x.md"
    assert not (tmp_path / "不存在").exists()
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == listing

    for txt in (False, True):
        transcript = folder / f"{NAME}.txt"
        if not txt:
            transcript.unlink(missing_ok=True)
        else:
            transcript.write_bytes(TXT_BYTES)
        job.stage = Stage.FAILED
        assert job.retry_stage() is Stage.NOTES, txt
        recording = Job.new_recording(tmp_path / f"rec{txt}", NOW, "large-v3", "model-a")
        media = Job.new_file(tmp_path / "第一课.mp3", tmp_path / f"out{txt}", NOW, "large-v3", "model-a")
        for other, plain in ((recording, Stage.RECORDING), (media, Stage.TRANSCRIBING)):
            other.stage = Stage.FAILED
            if txt:
                other.txt_path.write_bytes(TXT_BYTES)
            assert other.retry_stage() is (Stage.PROOFREADING if txt else plain)
            assert other.label == other.folder.name


def test_the_notes_name_and_length_helpers():
    """J5."""
    assert notes_file_for(Path("d/课.md")) == Path("d/课-整理笔记.md")
    assert notes_file_for(Path("d/a.b.MD")) == Path("d/a.b-整理笔记.md")
    for name in ("课-整理笔记.md", "课-整理笔记 (2).md", "课-整理笔记 (13).MD"):
        assert is_notes_file(Path("d") / name), name
    for name in (
        "课.md", "整理笔记.md", "课-整理笔记 (x).md", "课-整理笔记(2).md", "课-整理笔记 (2) 备份.md", "课-整理笔记.txt",
        "整理笔记方法.md",
    ):
        assert not is_notes_file(Path("d") / name), name
    for text, fits in (("x" * 259, True), ("x" * 260, False), ("😀" + "x" * 257, True), ("😀" + "x" * 258, False)):
        assert _u16(text) == (259 if fits else 260)
        assert path_fits(Path(text)) is fits, text
    assert PATH_LIMIT == 259


@pytest.mark.parametrize("kind", ["notes", "file"])
@pytest.mark.parametrize("blocked", ["segments", "proofread", "notes", None])
def test_removing_working_files_handles_each_file_on_its_own(tmp_path, caplog, kind, blocked):
    """J6."""
    store = JobStore(tmp_path / "state")
    store.folder.mkdir()
    if kind == "notes":
        job = Job.new_notes(lesson(tmp_path), "model-a")
    else:
        job = Job.new_file(tmp_path / "第一课.mp3", tmp_path / "out", NOW, "large-v3", "model-a")
    where = {"segments": store.segments_path(job), "proofread": store.proofread_path(job), "notes": store.notes_path(job)}
    if blocked is not None:
        for name, path in where.items():
            if name == blocked:
                path.mkdir()
                (path / "inside").write_bytes(b"x")
            else:
                path.write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="coursescribe.jobs"):
        store.remove_working_files(job)
    warnings = [r for r in caplog.records if r.name == "coursescribe.jobs" and r.levelno >= logging.WARNING]
    if blocked is None:
        assert warnings == []
        assert names(store.folder) == []
        return
    assert names(store.folder) == [where[blocked].name]
    assert where[blocked].is_dir()
    assert len(warnings) == 1 and where[blocked].name in warnings[0].getMessage()


def test_a_proofread_file_job_whose_working_file_cannot_go_still_finishes(tmp_path, caplog):
    """R6 (b)."""
    store = JobStore(tmp_path / "state")
    store.folder.mkdir()
    job = Job.new_file(tmp_path / "第一课.mp3", tmp_path / "out", NOW, "large-v3", "model-a")
    job.txt_path.write_bytes(TXT_BYTES)
    job.transcript_complete = True
    job.stage = Stage.PROOFREADING
    store.save(job)
    store.segments_path(job).mkdir()  # something real in the way of removing it
    (store.segments_path(job) / "inside").write_bytes(b"x")
    pipeline = Pipeline(store, FakeTranscriber(), FakeProofreader(), lambda path: 5.0)
    with caplog.at_level(logging.WARNING):
        pipeline.run(job)
    assert job.stage is Stage.DONE and job.message == ""
    assert store.load_all()[0].stage is Stage.DONE
    assert job.md_path.read_text(encoding="utf-8") == f"# {job.md_path.stem}\n\n校对后的正文。\n"
    assert job.txt_path.read_bytes() == TXT_BYTES
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].name == "coursescribe.jobs", [r.getMessage() for r in warnings]
    assert store.segments_path(job).name in warnings[0].getMessage()
    assert store.segments_path(job).is_dir()


# ---------------------------------------------------------------- pipeline: the notes step


def test_a_typical_notes_job_writes_the_notes_beside_the_md(tmp_path, svc):
    """N1."""
    md = lesson(tmp_path)
    folder = md.parent
    before = snapshots(folder)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    flow = Flow(store, svc.writer())
    flow.run(job)

    assert job.stage is Stage.DONE and job.message == ""
    assert job.notes_path == md.with_name(NOTES_FILE)
    assert job.notes_path.read_bytes() == FINISHED_BYTES
    assert (job.removed_quotes, job.removed_concepts, job.rules_note) == (2, 2, None)
    assert store.load_all() == [job]
    assert flow.statuses == [NOTES_WORKING, "完成"]
    gens = svc.gens
    assert len(gens) == 1 and gens[0]["prompt"].startswith(RULES) and gens[0]["model"] == "model-a"
    assert unloads(svc.fake) == [{"model": "model-a", "keep_alive": 0}]
    assert sent(svc.fake)[-1] == ("/api/generate", False)  # released after the one generation
    assert names(folder) == sorted([NOTES_FILE, f"{NAME}.md", f"{NAME}.txt"])
    assert unchanged(folder, before)
    assert names(store.folder) == [f"{job.job_id}.json"]


def test_the_rules_note_and_the_two_counts_are_kept_as_they_are(tmp_path, svc):
    """N1b."""
    reply = GOOD.replace("## 引述\n", "## 引述\n" + MADE_UP_QUOTE.split("\n", 1)[1] + "\n\n", 1)
    expected = finish_notes(reply, SRC, META)
    assert expected.removed_quotes != expected.removed_concepts  # or swapping them would go unseen
    note = "规则文件里没有可用的规则，用了自带规则。"
    svc.serve(reply)
    md = lesson(tmp_path)
    store = JobStore(tmp_path / "state")
    job = Flow(store, svc.writer(rules=lambda: NotesRules(RULES, note))).run(new_job(store, md))
    assert job.stage is Stage.DONE
    assert (job.removed_quotes, job.removed_concepts) == (expected.removed_quotes, expected.removed_concepts)
    assert job.rules_note == note
    saved = store.load_all()[0]
    assert (saved.removed_quotes, saved.removed_concepts, saved.rules_note) == (
        expected.removed_quotes, expected.removed_concepts, note,
    )
    assert job.notes_path.read_bytes() == expected.markdown.encode("utf-8")


def test_notes_again_go_to_a_new_numbered_file_even_when_word_for_word_the_same(tmp_path, svc):
    """N2."""
    md = lesson(tmp_path)
    folder = md.parent
    store = JobStore(tmp_path / "state")
    expected = [NOTES_FILE, f"{NAME}-整理笔记 (2).md", f"{NAME}-整理笔记 (3).md"]
    kept: dict[str, tuple[bytes, int]] = {}
    for name in expected:
        svc.serve()
        job = Flow(store, svc.writer()).run(new_job(store, md))
        assert job.stage is Stage.DONE and job.notes_path.name == name
        assert job.notes_path.read_bytes() == FINISHED_BYTES
        assert len(svc.gens) == 1
        assert unchanged(folder, kept)
        kept[name] = _snapshot(job.notes_path)
    (folder / NOTES_FILE).unlink()
    del kept[NOTES_FILE]
    svc.serve()
    job = Flow(store, svc.writer()).run(new_job(store, md))
    assert job.notes_path.name == NOTES_FILE and job.notes_path.read_bytes() == FINISHED_BYTES
    assert len(svc.gens) == 1
    assert unchanged(folder, kept)


def test_waiting_for_ollama_writes_nothing_and_continuing_leaves_one_file(tmp_path, svc):
    """N3 (a)."""
    svc.serve(BUSY, BUSY)
    md = lesson(tmp_path)
    before = snapshots(md.parent)
    store = JobStore(tmp_path / "state")
    job = Flow(store, svc.writer()).run(new_job(store, md))
    assert job.stage is Stage.WAITING_NOTES and job.message == NOTES_BUSY_MESSAGE
    saved = store.load_all()[0]
    assert (saved.stage, saved.message, saved.notes_path) == (Stage.WAITING_NOTES, NOTES_BUSY_MESSAGE, None)
    assert names(md.parent) == sorted(before) and unchanged(md.parent, before)

    svc.serve()
    final = Flow(store, svc.writer()).run(store.load_all()[0])
    assert final.stage is Stage.DONE
    assert notes_files(md) == [NOTES_FILE]
    assert unchanged(md.parent, before)


def test_a_notes_model_ollama_does_not_have_waits_without_asking(tmp_path, svc):
    """N3 (b)."""
    svc.serve(models=("model-a",))
    md = lesson(tmp_path)
    store = JobStore(tmp_path / "state")
    job = Flow(store, svc.writer()).run(new_job(store, md, "model-b"))
    assert job.stage is Stage.WAITING_NOTES
    assert job.message == NOTES_MODEL_MISSING.format(model="model-b")
    assert svc.gens == []
    assert notes_files(md) == []
    assert md.read_bytes() == SRC_BYTES


@pytest.mark.parametrize("case", ["md-moved", "folder-moved", "folder-moved-while-writing"])
def test_a_moved_md_or_folder_fails_without_making_the_folder_again(tmp_path, svc, case):
    """N4."""
    md = lesson(tmp_path)
    folder = md.parent
    away = tmp_path / "别处"
    if case == "folder-moved-while-writing":

        def respond(prompt):
            shutil.move(str(folder), str(away))
            return GOOD

        svc.serve(respond=respond)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    if case == "md-moved":
        md.rename(folder / "moved.md")
    elif case == "folder-moved":
        folder.rename(away)
    Flow(store, svc.writer()).run(job)

    assert job.stage is Stage.FAILED and job.can_retry and job.offers_retry
    assert job.message == NOTES_SOURCE_MISSING.format(path=job.source_md)
    assert store.load_all()[0].message == job.message
    if case == "folder-moved-while-writing":
        assert len(svc.gens) == 1
    else:
        assert svc.fake.log == []
    if case == "md-moved":
        assert names(folder) == sorted(["moved.md", f"{NAME}.txt"])
        assert (folder / "moved.md").read_bytes() == SRC_BYTES
        (folder / "moved.md").rename(md)
    else:
        assert not folder.exists()
        assert names(away) == sorted([f"{NAME}.md", f"{NAME}.txt"])
        assert (away / f"{NAME}.md").read_bytes() == SRC_BYTES
        away.rename(folder)

    svc.serve()
    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE
    assert len(svc.gens) == (0 if case == "folder-moved-while-writing" else 1)
    assert notes_files(md) == [NOTES_FILE]
    assert (folder / NOTES_FILE).read_bytes() == FINISHED_BYTES
    assert md.read_bytes() == SRC_BYTES


def test_a_folder_moved_while_writing_beside_old_notes_still_gets_a_new_numbered_file(tmp_path, svc):
    """N4 (d). With the folder gone, no name is chosen: chosen there, the base name would be free,
    and once the folder is back the old notes, byte for byte the same, would pass as written."""
    md = lesson(tmp_path)
    folder = md.parent
    old = folder / NOTES_FILE
    old.write_bytes(FINISHED_BYTES)
    old_snapshot = _snapshot(old)
    away = tmp_path / "别处"

    def respond(prompt):
        shutil.move(str(folder), str(away))
        return GOOD

    svc.serve(respond=respond)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    Flow(store, svc.writer()).run(job)

    assert job.stage is Stage.FAILED and job.offers_retry
    assert job.message == NOTES_SOURCE_MISSING.format(path=md)
    assert job.notes_path is None and store.load_all()[0].notes_path is None
    assert store.notes_path(job).is_file()
    assert not folder.exists()
    assert names(away) == sorted([NOTES_FILE, f"{NAME}.md", f"{NAME}.txt"])
    away.rename(folder)

    svc.serve()
    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE
    assert job.notes_path.name == f"{NAME}-整理笔记 (2).md"
    assert job.notes_path.read_bytes() == FINISHED_BYTES
    assert _snapshot(old) == old_snapshot
    assert svc.gens == []  # the kept reply, not a new request
    assert notes_files(md) == sorted([NOTES_FILE, f"{NAME}-整理笔记 (2).md"])


def test_a_folder_moved_after_the_name_was_kept_fails_without_making_the_folder_again(tmp_path, svc):
    """N4 (e). The name was chosen and saved by a run that was then closed; on the run after the
    restart the folder goes while the model writes, so the check right before writing must catch it."""
    md = lesson(tmp_path)
    folder = md.parent
    away = tmp_path / "别处"

    def respond(prompt):
        shutil.move(str(folder), str(away))
        return GOOD

    svc.serve(respond=respond)
    store = JobStore(tmp_path / "state")
    job = Job.new_notes(md, "model-a")
    job.notes_path = folder / NOTES_FILE
    store.save(job)
    Flow(store, svc.writer()).run(job)

    assert job.stage is Stage.FAILED and job.offers_retry
    assert job.message == NOTES_SOURCE_MISSING.format(path=md)
    assert store.load_all()[0].notes_path == folder / NOTES_FILE
    assert store.notes_path(job).is_file()
    assert not folder.exists()
    assert names(away) == sorted([f"{NAME}.md", f"{NAME}.txt"])
    away.rename(folder)

    svc.serve()
    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE
    assert job.notes_path == folder / NOTES_FILE
    assert job.notes_path.read_bytes() == FINISHED_BYTES
    assert svc.gens == []  # the kept reply, not a new request
    assert notes_files(md) == [NOTES_FILE]


def test_a_failed_reply_drops_the_kept_reply_and_a_checkpoint_that_cannot_go_is_only_logged(tmp_path, svc, caplog):
    """R2 at the pipeline: NotesFailed removes this job's checkpoint; one that cannot be removed
    (something real in the way) is logged, and the job still fails with the writer's message."""
    svc.serve(CUT, CUT)
    md = lesson(tmp_path)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    store.notes_path(job).mkdir(parents=True)
    (store.notes_path(job) / "inside").write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="coursescribe.pipeline"):
        Flow(store, svc.writer()).run(job)
    assert (job.stage, job.message) == (Stage.FAILED, NOTES_CUT_OFF) and job.offers_retry
    assert store.load_all()[0].stage is Stage.FAILED
    warnings = [r for r in caplog.records if r.name == "coursescribe.pipeline" and r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and store.notes_path(job).name in warnings[0].getMessage()
    assert store.notes_path(job).is_dir()
    assert notes_files(md) == []


@pytest.mark.parametrize("case", ["held", "unplugged"])
def test_a_held_or_unplugged_md_says_so_and_works_once_free(tmp_path, svc, monkeypatch, case):
    """N5."""
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    md = lesson(tmp_path)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    error = None if case == "held" else OSError(5, "Input/output error")
    held = _held(monkeypatch, md, error=error)
    Flow(store, svc.writer()).run(job)
    assert job.stage is Stage.FAILED and job.offers_retry
    if case == "held":
        assert job.message == NOTES_SOURCE_HELD.format(path=md)
        assert held.attempts == 20  # waited a moment before saying so
    else:
        assert job.message == NOTES_SOURCE_UNREADABLE.format(path=md)
        assert held.attempts == 1
    assert svc.fake.log == []
    assert notes_files(md) == []

    held.release()
    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE
    assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
    assert md.read_bytes() == SRC_BYTES


_ENCODED = {
    "utf-8-bom": codecs.BOM_UTF8 + SRC_CRLF.encode("utf-8"),
    "utf-16-bom": codecs.BOM_UTF16_LE + SRC_CRLF.encode("utf-16-le"),
    "gbk": SRC_CRLF.encode("gbk"),
    "utf-16-unmarked": SRC_CRLF.encode("utf-16-le"),  # no byte order mark
}


@pytest.mark.parametrize("encoding", list(_ENCODED))
def test_an_md_saved_in_another_encoding_is_read_when_certain_and_refused_otherwise(tmp_path, svc, encoding):
    """N6."""
    plain_store = JobStore(tmp_path / "plain" / "state")
    Flow(plain_store, svc.writer()).run(new_job(plain_store, lesson(tmp_path / "plain")))
    plain_prompt = svc.gens[0]["prompt"]

    svc.serve()
    data = _ENCODED[encoding]
    md = lesson(tmp_path / "other", data=data)
    store = JobStore(tmp_path / "other" / "state")
    job = Flow(store, svc.writer()).run(new_job(store, md))
    assert md.read_bytes() == data
    if encoding.endswith("-bom"):
        assert job.stage is Stage.DONE
        assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
        assert [g["prompt"] for g in svc.gens] == [plain_prompt]
        return
    assert job.stage is Stage.FAILED and job.offers_retry
    assert job.message == NOTES_SOURCE_ENCODING.format(path=md)
    assert svc.fake.log == []
    assert notes_files(md) == []
    if encoding == "gbk":
        md.write_bytes(SRC_BYTES)
        Flow(store, svc.writer()).run(retry(job))
        assert job.stage is Stage.DONE
        assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES


@pytest.mark.parametrize(
    "data, too_long",
    [
        (b"", False),
        (TITLE_ONLY, False),
        ("﻿# 课\r\n\r\n  \r\n".encode("utf-8"), False),
        (TOO_LONG_SOURCE.encode("utf-8"), True),
    ],
    ids=["empty", "title-only", "bom-title-blank-lines", "too-long"],
)
def test_an_empty_or_too_long_md_fails_without_asking(tmp_path, svc, data, too_long):
    """N7."""
    md = lesson(tmp_path, data=data)
    before = snapshots(md.parent)
    store = JobStore(tmp_path / "state")
    writer = svc.writer(rules=two_char_rules, num_ctx=407, reply_budget=50) if too_long else svc.writer()
    job = Flow(store, writer).run(new_job(store, md))
    assert job.stage is Stage.FAILED and job.offers_retry
    assert job.message == (NOTES_TOO_LONG.format(chars=100, limit=99) if too_long else NOTES_EMPTY_SOURCE)
    assert svc.fake.log == []
    assert names(md.parent) == sorted(before) and unchanged(md.parent, before)
    assert names(store.folder) == [f"{job.job_id}.json"]


@pytest.mark.parametrize("case", ["disk-full", "read-only", "disk-full-twice"])
def test_notes_that_cannot_be_saved_keep_the_reply_and_retry_without_the_model(tmp_path, svc, monkeypatch, case):
    """N8."""
    md = lesson(tmp_path)
    folder = md.parent
    before = snapshots(folder)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    fault = DiskFault(monkeypatch, folder, READ_ONLY if case == "read-only" else DISK_FULL)
    Flow(store, svc.writer()).run(job)

    expected = NOTES_WRITE_FAILED.format(folder=folder)
    assert job.stage is Stage.FAILED and job.offers_retry and job.message == expected
    assert "磁盘满" in expected and "只读" in expected and "不用再等模型重写" in expected
    assert job.notes_path == folder / NOTES_FILE
    assert store.load_all()[0].notes_path == folder / NOTES_FILE
    assert store.notes_path(job).is_file()
    assert names(folder) == sorted(before) and unchanged(folder, before) and temp_files(folder) == []
    if case == "disk-full-twice":
        for _ in range(2):
            Flow(store, svc.writer()).run(retry(job))
            assert (job.stage, job.message) == (Stage.FAILED, expected)
        assert names(folder) == sorted(before)

    fault.fix()
    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE
    assert len(svc.gens) == 1  # over the whole case
    assert (folder / NOTES_FILE).read_bytes() == FINISHED_BYTES
    assert unchanged(folder, before)


def test_a_notes_path_that_just_fits_is_written(tmp_path, svc):
    """N9 (a)."""
    md, base = long_lesson(tmp_path, PATH_LIMIT)
    store = JobStore(tmp_path / "state")
    job = Flow(store, svc.writer()).run(new_job(store, md))
    assert job.stage is Stage.DONE and job.notes_path == base
    assert base.read_text(encoding="utf-8").startswith(f"# {md.stem}\n")


def test_a_numbered_notes_path_that_does_not_fit_fails_and_retries_without_the_model(tmp_path, svc):
    """N9 (b)."""
    md, base = long_lesson(tmp_path, PATH_LIMIT)
    base.write_bytes("旧的\n".encode("utf-8"))
    old = _snapshot(base)
    store = JobStore(tmp_path / "state")
    job = Flow(store, svc.writer()).run(new_job(store, md))
    numbered = base.with_name(f"{md.stem}{NOTES_SUFFIX} (2).md")
    assert _u16(str(numbered)) == PATH_LIMIT + 4
    assert job.stage is Stage.FAILED and job.offers_retry
    assert job.message == NOTES_PATH_TOO_LONG.format(path=numbered)
    assert job.notes_path is None and store.load_all()[0].notes_path is None
    assert store.notes_path(job).is_file()
    assert len(svc.gens) == 1
    assert _snapshot(base) == old and names(md.parent) == sorted([md.name, base.name])

    base.unlink()
    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE and job.notes_path == base
    assert len(svc.gens) == 1


def test_a_checkpoint_that_cannot_be_kept_does_not_stop_the_notes(tmp_path, svc, monkeypatch, caplog):
    """N10."""
    monkeypatch.setattr("coursescribe.files._REPLACE_WAIT", 0.001)
    md = lesson(tmp_path)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    store.notes_path(job).mkdir(parents=True)  # something real in the way of the checkpoint
    with caplog.at_level(logging.WARNING):
        Flow(store, svc.writer()).run(job)
    assert job.stage is Stage.DONE
    assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
    assert store.load_all()[0].stage is Stage.DONE
    warnings = [r.name for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings.count("coursescribe.notes_writer") == 1, warnings
    assert warnings.count("coursescribe.jobs") == 1, warnings
    assert store.notes_path(job).is_dir()


def test_an_unexpected_error_gives_the_general_notes_message_and_can_be_retried(tmp_path, svc, caplog):
    """N11."""

    def boom() -> NotesRules:
        raise RuntimeError("cuda exploded at 0x1234")

    md = lesson(tmp_path)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    with caplog.at_level(logging.ERROR):
        Flow(store, svc.writer(rules=boom)).run(job)
    assert job.stage is Stage.FAILED and job.can_retry and job.offers_retry
    assert job.message == NOTES_UNEXPECTED
    assert "0x1234" not in job.message and "录音" not in job.message
    assert svc.fake.log == []
    assert any(r.exc_info and "0x1234" in str(r.exc_info[1]) for r in caplog.records)
    assert notes_files(md) == [] and md.read_bytes() == SRC_BYTES

    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE


def test_the_notes_date_comes_from_the_name_or_the_day_the_job_was_made(tmp_path, svc):
    """N12."""
    md = lesson(tmp_path / "a")
    store = JobStore(tmp_path / "a" / "state")
    job = Flow(store, svc.writer()).run(new_job(store, md))
    assert job.notes_path.read_text(encoding="utf-8").split("\n")[2] == "日期：2026-10-30"

    svc.serve(BUSY, BUSY)
    md = lesson(tmp_path / "b", name="数据库索引")
    store = JobStore(tmp_path / "b" / "state")
    job = new_job(store, md)
    made_on = date.fromtimestamp(job.created_ns / 1e9).isoformat()
    Flow(store, svc.writer()).run(job)
    assert job.stage is Stage.WAITING_NOTES
    svc.serve()
    final = Flow(store, svc.writer()).run(store.load_all()[0])
    assert final.stage is Stage.DONE
    assert final.notes_path.read_text(encoding="utf-8").split("\n")[2] == f"日期：{made_on}"


class CountingStore(JobStore):
    def __init__(self, folder: Path) -> None:
        super().__init__(folder)
        self.saves = 0

    def save(self, job: Job) -> None:
        super().save(job)
        self.saves += 1


class CrashBeforeStore(JobStore):
    """Dies right before the Nth save: that save never happens."""

    def __init__(self, folder: Path, crash_at_save: int) -> None:
        super().__init__(folder)
        self.left = crash_at_save

    def save(self, job: Job) -> None:
        self.left -= 1
        if self.left == 0:
            raise Crash
        super().save(job)


def test_closed_before_or_after_any_save_ends_with_one_file_and_one_question(tmp_path, svc):
    """N13."""
    md = lesson(tmp_path / "clean")
    plain = JobStore(tmp_path / "clean" / "state")
    job = new_job(plain, md)
    counting = CountingStore(plain.folder)
    Flow(counting, svc.writer()).run(job)
    assert job.stage is Stage.DONE
    saves = counting.saves
    assert saves >= 2  # at least the chosen name and DONE

    for crashing, when in ((CrashingStore, "after"), (CrashBeforeStore, "before")):
        for n in range(1, saves + 1):
            svc.serve()
            root = tmp_path / f"{when}-{n}"
            md = lesson(root)
            plain = JobStore(root / "state")
            job = new_job(plain, md)
            with pytest.raises(Crash):
                Flow(crashing(plain.folder, n), svc.writer()).run(job)
            resumed = plain.load_all()[0]
            already = resumed.notes_path.read_bytes() if resumed.notes_path and resumed.notes_path.exists() else None
            if resumed.stage is not Stage.DONE:
                Flow(plain, svc.writer()).run(resumed)
            final = plain.load_all()[0]
            where = f"{when} save {n}"
            assert final.stage is Stage.DONE, where
            assert names(md.parent) == sorted([NOTES_FILE, md.name, f"{NAME}.txt"]), where
            assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES, where
            assert len(svc.gens) == 1, where
            if when == "before" and n == saves:
                # the file was written and DONE not saved yet: the restart finds it and keeps its name
                assert already == FINISHED_BYTES and final.notes_path == resumed.notes_path, where
            left = [f"{job.job_id}.json"]
            if when == "after" and n == saves:
                left.append(f"{job.job_id}.notes.json")  # DONE saved, then closed: left over (Deferred)
            assert names(plain.folder) == sorted(left), where


_CHILD = r"""
import os
import sys
from pathlib import Path

import coursescribe.files as files
from coursescribe.jobs import JobStore
from coursescribe.notes_rules import NotesRules
from coursescribe.notes_writer import OllamaNotesWriter
from coursescribe.ollama import OllamaClient
from coursescribe.pipeline import Pipeline

url, state, job_id, rules = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
real_link = os.link


def link(source, target, *args, **kwargs):
    real_link(source, target, *args, **kwargs)
    if str(target).endswith("-整理笔记.md"):
        os._exit(3)  # killed the moment the notes file is published


files.os.link = link
store = JobStore(state)
job = next(job for job in store.load_all() if job.job_id == job_id)
writer = OllamaNotesWriter(OllamaClient(url, timeout=5, unload_timeout=2), lambda: NotesRules(rules, None), retry_pause=0)
Pipeline(store, None, None, lambda path: 5.0, notes_writer=writer).run(job)
"""


def test_killed_the_moment_the_notes_file_is_published(tmp_path, svc):
    """N14: a real process, really killed."""
    md = lesson(tmp_path)
    folder = md.parent
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, svc.fake.url, str(store.folder), job.job_id, RULES],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert child.returncode == 3, child.stderr
    assert (folder / NOTES_FILE).read_bytes() == FINISHED_BYTES
    halfway = store.load_all()[0]
    assert halfway.stage is Stage.NOTES and halfway.notes_path == folder / NOTES_FILE
    left = set(names(folder)) - {NOTES_FILE, md.name, f"{NAME}.txt"}
    assert all(re.fullmatch(r"\.coursescribe-.*\.tmp", name) for name in left), left

    Flow(store, svc.writer()).run(halfway)
    assert halfway.stage is Stage.DONE and halfway.notes_path == folder / NOTES_FILE
    assert notes_files(md) == [NOTES_FILE]
    assert len(svc.gens) == 1  # the child's; the restart used the kept reply


# ---------------------------------------------------------------- controller: make, continue, retry, dismiss


def test_making_notes_queues_a_job_that_finishes_without_making_any_folder(tmp_path, svc, rigs):
    """C1."""
    md = lesson(tmp_path)
    folder = md.parent
    before = snapshots(folder)
    folders = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()

    jobs = rig.store.load_all()
    assert len(jobs) == 1
    job = jobs[0]
    assert (job.kind, job.stage, job.notes_model) == (JobKind.NOTES, Stage.DONE, DEFAULT_NOTES_MODEL)
    assert DEFAULT_NOTES_MODEL == "qwen3.5:9b"
    assert (folder / NOTES_FILE).read_bytes() == FINISHED_BYTES
    state = rig.controller.state
    assert state.notice.startswith(f"笔记已生成：{NOTES_FILE}")
    assert state.last_folder == folder
    assert (state.failed, state.waiting_notes, state.waiting_proofread, state.pending) == (0, 0, 0, 0)
    assert NOTES_WORKING in [s.status for s in rig.states]
    assert not (tmp_path / "out").exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == sorted(folders + ["state"])
    assert unchanged(folder, before)


def test_making_notes_again_while_they_are_being_made_does_nothing_but_say_so(tmp_path, svc, rigs):
    """C2 (a)-(d)."""
    gate = gated(svc)
    md = lesson(tmp_path)
    other = lesson(tmp_path, name="另一课")
    rig = rigs()
    rig.controller.start()
    rig.controller.make_notes(md)
    assert rig.controller.state.pending == 1
    rig.wait(lambda: len(svc.gens) == 1)

    for spelling in (md, md.parent / ".." / md.parent.name / md.name):
        rig.controller.make_notes(spelling)
        assert rig.controller.state.notice == NOTES_ALREADY.format(name=md.name), spelling
        assert len(notes_jobs(rig.store)) == 1 and len(svc.gens) == 1
    upper = Path(str(md).upper())
    rig.controller.make_notes(upper)
    if case_blind():
        assert rig.controller.state.notice == NOTES_ALREADY.format(name=md.name)
    else:
        assert rig.controller.state.notice == NOTES_NOT_A_FILE.format(name=upper.name)
    assert len(notes_jobs(rig.store)) == 1 and len(svc.gens) == 1

    rig.controller.make_notes(other)
    assert rig.controller.state.notice == "" and rig.controller.state.pending == 2
    gate.set()
    rig.wait_idle()
    assert notes_files(md) == [NOTES_FILE]
    assert notes_files(other) == ["另一课-整理笔记.md"]
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE, Stage.DONE]


def test_making_notes_again_while_they_wait_for_ollama_says_how_to_continue(tmp_path, svc, rigs):
    """C2 (e)."""
    svc.serve(BUSY, BUSY)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.WAITING_NOTES]
    asked = len(svc.fake.log)
    rig.controller.make_notes(md)
    assert rig.controller.state.notice == NOTES_ALREADY_WAITING.format(name=md.name)
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.WAITING_NOTES]
    assert len(svc.fake.log) == asked and rig.controller.state.pending == 0


def test_making_notes_again_after_they_are_done_saves_a_numbered_copy(tmp_path, svc, rigs):
    """C2 (f)."""
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    first = _snapshot(md.parent / NOTES_FILE)
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    jobs = notes_jobs(rig.store, md)
    assert len(jobs) == 2 and jobs[0].job_id != jobs[1].job_id
    assert [job.stage for job in jobs] == [Stage.DONE, Stage.DONE]
    numbered = f"{NAME}-整理笔记 (2).md"
    assert jobs[1].notes_path.name == numbered
    assert rig.controller.state.notice.startswith(f"笔记已生成：{numbered}")
    assert (md.parent / numbered).read_bytes() == FINISHED_BYTES
    assert _snapshot(md.parent / NOTES_FILE) == first


def test_making_notes_after_a_failure_retries_that_job_with_the_model_set_now(tmp_path, svc, rigs):
    """C3, first group."""
    svc.serve(CUT, CUT)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    failed = notes_jobs(rig.store)
    assert [(job.stage, job.message) for job in failed] == [(Stage.FAILED, NOTES_CUT_OFF)]
    assert not rig.store.notes_path(failed[0]).exists()

    svc.serve(models=MODELS)
    set_notes_model(rig, "model-c")
    rig.controller.make_notes(md)
    rig.wait_idle()
    jobs = notes_jobs(rig.store)
    assert [job.job_id for job in jobs] == [failed[0].job_id]
    assert jobs[0].stage is Stage.DONE and jobs[0].notes_model == "model-c"
    assert [g["model"] for g in svc.gens] == ["model-c"]
    assert rig.controller.state.failed == 0
    assert notes_files(md) == [NOTES_FILE]


def test_making_notes_after_a_dismissed_failure_starts_a_new_job(tmp_path, svc, rigs):
    """C3, second group."""
    svc.serve(CUT, CUT)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    rig.controller.dismiss_failed()
    svc.serve()
    rig.controller.make_notes(md)
    rig.wait_idle()
    jobs = notes_jobs(rig.store)
    assert [(job.stage, job.dismissed) for job in jobs] == [(Stage.FAILED, True), (Stage.DONE, False)]
    assert notes_files(md) == [NOTES_FILE]


def test_making_notes_for_a_paused_job_runs_it_from_a_clean_count(tmp_path, svc, rigs):
    """C3, third group."""
    md = lesson(tmp_path)
    rig = rigs()
    paused = Job.new_notes(md, "model-a")
    paused.attempts = 2
    rig.store.save(paused)
    seen: list[int] = []
    rig.controller.add_listener(
        lambda state: seen.append(rig.store.load_all()[0].attempts) if state.status == NOTES_WORKING else None
    )
    rig.controller.start()
    assert rig.store.load_all()[0].stage is Stage.FAILED
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    assert seen == [1]
    final = rig.store.load_all()[0]
    assert (final.job_id, final.stage, final.attempts) == (paused.job_id, Stage.DONE, 0)
    assert notes_files(md) == [NOTES_FILE]


def test_continuing_notes_uses_the_notes_model_set_now(tmp_path, svc, rigs):
    """C4."""
    svc.serve(models=("model-a",))
    md = lesson(tmp_path)
    rig = rigs()
    set_notes_model(rig, "model-b")
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    missing = NOTES_MODEL_MISSING.format(model="model-b")
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.WAITING_NOTES]
    assert (rig.controller.state.waiting_notes, rig.controller.state.notice) == (1, missing)

    rig.controller.continue_notes()
    rig.wait_idle()
    assert [(job.stage, job.message) for job in notes_jobs(rig.store)] == [(Stage.WAITING_NOTES, missing)]
    assert rig.controller.state.notice == missing and svc.gens == []

    set_notes_model(rig, "model-a")
    since = len(rig.states)
    rig.controller.continue_notes()
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert (job.stage, job.notes_model) == (Stage.DONE, "model-a")
    assert [g["model"] for g in svc.gens] == ["model-a"]
    assert rig.controller.state.waiting_notes == 0
    assert notes_files(md) == [NOTES_FILE]


def test_continue_proofreading_and_continue_notes_each_take_only_their_own_jobs(tmp_path, svc, rigs):
    """C5 (a)(b)."""
    svc.serve(BUSY, BUSY)
    md = lesson(tmp_path)
    rig = rigs()
    rig.proofreader.blocked = "没有启动"
    rig.controller.start()
    rig.controller.add_files([rig.source()])
    rig.controller.make_notes(md)
    rig.wait_idle()

    def stages():
        jobs = rig.store.load_all()
        return [job.stage for job in jobs if job.kind is JobKind.FILE], [job.stage for job in notes_jobs(rig.store)]

    assert stages() == ([Stage.WAITING_PROOFREAD], [Stage.WAITING_NOTES])
    assert (rig.controller.state.waiting_proofread, rig.controller.state.waiting_notes) == (1, 1)

    asked, proofread = len(svc.fake.log), len(rig.proofreader.models)
    rig.controller.retry_waiting()
    rig.wait_idle()
    assert len(rig.proofreader.models) == proofread + 1
    assert len(svc.fake.log) == asked
    assert stages() == ([Stage.WAITING_PROOFREAD], [Stage.WAITING_NOTES])

    svc.serve()
    rig.controller.continue_notes()
    rig.wait_idle()
    assert len(rig.proofreader.models) == proofread + 1
    assert stages() == ([Stage.WAITING_PROOFREAD], [Stage.DONE])
    assert (rig.controller.state.waiting_proofread, rig.controller.state.waiting_notes) == (1, 0)


def test_at_start_a_waiting_notes_job_continues_with_its_own_model(tmp_path, svc, rigs):
    """C5 (c)."""
    md = lesson(tmp_path)
    rig = rigs()
    waiting = Job.new_notes(md, "model-a")
    waiting.stage = Stage.WAITING_NOTES
    rig.store.save(waiting)
    set_notes_model(rig, "model-b")
    rig.controller.start()
    rig.wait_idle()
    assert [g["model"] for g in svc.gens] == ["model-a"]
    assert notes_jobs(rig.store)[0].stage is Stage.DONE


def test_files_that_cannot_have_notes_are_refused_with_the_reason(tmp_path, svc, rigs):
    """C6."""
    md = lesson(tmp_path)
    folder = md.parent
    (folder / "子.md").mkdir()
    (folder / "x.MP3").write_bytes(b"audio")
    (folder / "课-整理笔记.md").write_bytes(FINISHED_BYTES)
    (folder / "课-整理笔记 (2).md").write_bytes(FINISHED_BYTES)
    long_md, _ = long_lesson(tmp_path / "g", PATH_LIMIT + 1)
    numbered_md, base = long_lesson(tmp_path / "h", PATH_LIMIT)
    base.write_bytes("旧的\n".encode("utf-8"))
    listing = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    rig = rigs()
    rig.controller.start()

    cases = [
        (folder / "没有.md", NOTES_NOT_A_FILE.format(name="没有.md")),
        (folder / "子.md", NOTES_NOT_A_FILE.format(name="子.md")),
        (folder / f"{NAME}.txt", NOTES_NOT_MD.format(name=f"{NAME}.txt")),
        (folder / "x.MP3", NOTES_NOT_MD.format(name="x.MP3")),
        (folder / "课-整理笔记.md", NOTES_IS_NOTES.format(name="课-整理笔记.md")),
        (folder / "课-整理笔记 (2).md", NOTES_IS_NOTES.format(name="课-整理笔记 (2).md")),
        (long_md, NOTES_PATH_TOO_LONG_BEFORE.format(name=long_md.name, path=notes_file_for(long_md))),
        (
            numbered_md,
            NOTES_PATH_TOO_LONG_BEFORE.format(
                name=numbered_md.name, path=base.with_name(f"{numbered_md.stem}{NOTES_SUFFIX} (2).md")
            ),
        ),
    ]
    for path, notice in cases:
        rig.controller.make_notes(path)
        assert rig.controller.state.notice == notice, path
        assert "重试" not in notice
    assert rig.store.load_all() == [] and rig.controller.state.pending == 0
    assert svc.fake.log == [] and rig.awake.history == []
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert [p for p in after if not p.startswith("state")] == listing

    upper = lesson(tmp_path, name="大写")
    upper = upper.rename(upper.with_suffix(".MD"))
    rig.controller.make_notes(upper)
    rig.wait_idle()
    assert notes_files(upper) == ["大写-整理笔记.md"]

    base.unlink()
    rig.controller.make_notes(numbered_md)
    rig.wait_idle()
    assert base.is_file() and notes_jobs(rig.store, numbered_md)[0].stage is Stage.DONE

    title = lesson(tmp_path, name="只有标题", data=f"# {NAME}\n\n".encode("utf-8"))
    svc.serve()
    rig.controller.make_notes(title)
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store, title)] == [Stage.FAILED]
    assert rig.controller.state.notice == f"{title.name}：{NOTES_EMPTY_SOURCE}"
    assert svc.gens == []


def test_a_refused_request_fails_and_retrying_twice_runs_once_with_the_model_set_now(tmp_path, svc, rigs):
    """C7 (a)."""
    svc.serve(REFUSED)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    assert rig.controller.state.failed == 1
    assert rig.controller.state.notice == f"{md.name}：{NOTES_REFUSED}"

    set_notes_model(rig, "model-c")
    svc.serve()
    rig.controller.retry_failed()
    rig.controller.retry_failed()
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert job.stage is Stage.DONE
    assert [g["model"] for g in svc.gens] == ["model-c"]
    assert notes_files(md) == [NOTES_FILE]


def test_dismissing_a_failed_notes_job_removes_its_kept_reply(tmp_path, svc, rigs, monkeypatch):
    """C7 (b)."""
    md = lesson(tmp_path)
    DiskFault(monkeypatch, md.parent, DISK_FULL)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert job.stage is Stage.FAILED and rig.store.notes_path(job).is_file()
    rig.controller.dismiss_failed()
    assert rig.controller.state.failed == 0
    assert not rig.store.notes_path(job).exists()
    assert md.read_bytes() == SRC_BYTES
    asked = len(svc.fake.log)
    rig.controller.retry_failed()
    rig.wait_idle()
    assert len(svc.fake.log) == asked
    assert [(j.stage, j.dismissed) for j in notes_jobs(rig.store)] == [(Stage.FAILED, True)]


def test_a_paused_notes_job_says_so_and_retries_from_the_notes_step(tmp_path, svc, rigs):
    """C7 (c)."""
    md = lesson(tmp_path)
    rig = rigs()
    paused = Job.new_notes(md, "model-a")
    paused.attempts = 2
    rig.store.save(paused)
    rig.controller.start()
    notice = rig.controller.state.notice
    assert notice.startswith(f"{md.name}：") and "暂停" in notice
    assert rig.store.load_all()[0].stage is Stage.FAILED and svc.fake.log == []
    rig.controller.retry_failed()
    rig.wait_idle()
    assert rig.store.load_all()[0].stage is Stage.DONE
    assert notes_files(md) == [NOTES_FILE] and len(svc.gens) == 1


@pytest.mark.parametrize("how", ["retry", "make-notes"])
def test_with_a_kept_reply_retrying_keeps_the_model_and_does_not_ask_again(tmp_path, svc, rigs, monkeypatch, how):
    """C7 (d)."""
    md = lesson(tmp_path)
    fault = DiskFault(monkeypatch, md.parent, DISK_FULL)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    assert notes_jobs(rig.store)[0].stage is Stage.FAILED
    assert len(svc.gens) == 1
    fault.fix()
    set_notes_model(rig, "model-c")
    if how == "retry":
        rig.controller.retry_failed()
    else:
        rig.controller.make_notes(md)
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert job.stage is Stage.DONE and job.notes_model == DEFAULT_NOTES_MODEL
    assert len(svc.gens) == 1
    assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES


def test_a_kept_reply_that_no_longer_matches_is_dropped_when_the_model_fails(tmp_path, svc, rigs, monkeypatch):
    """C7 (e). Without dropping it, "重试" would keep taking the model as done and never use the one set now."""
    md = lesson(tmp_path)
    fault = DiskFault(monkeypatch, md.parent, DISK_FULL)
    rig = rigs()
    rig.controller.start()
    rig.controller.make_notes(md)
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert job.stage is Stage.FAILED and rig.store.notes_path(job).is_file()
    fault.fix()
    md.write_bytes(SRC_BYTES + "\n补充的一段话。\n".encode("utf-8"))  # the kept reply is for other input now

    svc.serve(CUT, CUT)
    rig.controller.retry_failed()
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert (job.stage, job.message) == (Stage.FAILED, NOTES_CUT_OFF)
    assert rig.controller.state.notice == f"{md.name}：{NOTES_CUT_OFF}"
    assert not rig.store.notes_path(job).exists()

    set_notes_model(rig, "model-c")
    svc.serve()
    rig.controller.retry_failed()
    rig.wait_idle()
    job = notes_jobs(rig.store)[0]
    assert (job.stage, job.notes_model) == (Stage.DONE, "model-c")
    assert [g["model"] for g in svc.gens] == ["model-c"]
    assert notes_files(md) == [NOTES_FILE]


def test_making_notes_for_a_failed_job_that_cannot_be_recorded_says_why_and_keeps_the_failure(tmp_path, svc, rigs):
    """R6 (a)."""
    svc.serve(REFUSED)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    rig.controller.make_notes(md)
    rig.wait_idle()
    before = notes_jobs(rig.store)
    assert [job.stage for job in before] == [Stage.FAILED]
    asked = len(svc.fake.log)
    _job_records_fail(rig, lambda job: True)
    rig.controller.make_notes(md)
    rig.wait_idle()
    notice = rig.controller.state.notice
    assert notice.startswith(f"没有整理：{md.name}。")
    assert "程序自己的数据文件夹" in notice and str(rig.store.folder) in notice
    after = notes_jobs(rig.store)
    assert [(job.job_id, job.stage, job.message) for job in after] == [
        (before[0].job_id, Stage.FAILED, NOTES_REFUSED)
    ]
    assert len(svc.fake.log) == asked and rig.controller.state.pending == 0
    assert notes_files(md) == []


def test_continuing_notes_twice_while_the_first_runs_asks_only_once(tmp_path, svc, rigs):
    """R6 (c)."""
    svc.serve(BUSY, BUSY)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    rig.controller.make_notes(md)
    rig.wait_idle()
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.WAITING_NOTES]

    gate = gated(svc)
    rig.controller.continue_notes()
    rig.wait(lambda: len(svc.gens) == 1)
    rig.controller.continue_notes()
    assert rig.controller.state.pending == 1  # held at the gate: the worker cannot finish in between
    gate.set()
    rig.wait_idle()
    assert len(svc.gens) == 1
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE]
    assert notes_files(md) == [NOTES_FILE]


def test_a_new_notes_job_that_cannot_be_recorded_says_why_and_removes_nothing(tmp_path, svc, rigs):
    """C8 (a)."""
    md = lesson(tmp_path)
    before = names(md.parent)
    rig = rigs()
    rig.controller.start()
    _job_records_fail(rig, lambda job: True)
    rig.controller.make_notes(md)
    notice = rig.controller.state.notice
    assert notice.startswith(f"没有整理：{md.name}。")
    assert "程序自己的数据文件夹" in notice and str(rig.store.folder) in notice
    assert rig.store.load_all() == [] and svc.fake.log == [] and rig.controller.state.pending == 0
    assert names(md.parent) == before


def test_a_notes_job_whose_start_cannot_be_recorded_waits_for_the_next_start(tmp_path, svc, rigs):
    """C8 (b)."""
    md = lesson(tmp_path)
    rig = rigs()
    _job_records_fail(rig, lambda job: job.attempts == 1)
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    notice = rig.controller.state.notice
    assert notice.startswith(f"{md.name}：") and "程序自己的数据文件夹" in notice and "下次打开程序会接着做" in notice
    assert svc.fake.log == [] and notes_files(md) == []
    rig.check_keep_awake_follows_busy()

    again = rigs()
    again.controller.start()
    again.wait_idle()
    assert notes_jobs(again.store)[0].stage is Stage.DONE
    assert notes_files(md) == [NOTES_FILE]


@pytest.mark.parametrize("action", ["retry", "continue", "dismiss"])
def test_retry_continue_or_dismiss_that_cannot_be_recorded_names_the_md(tmp_path, svc, rigs, action):
    """C8 (c)."""
    svc.serve(*((BUSY, BUSY) if action == "continue" else (REFUSED,)))
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    before = notes_jobs(rig.store)[0]
    if action == "retry":
        _job_records_fail(rig, lambda j: j.kind is JobKind.NOTES and j.stage is Stage.NOTES)
        rig.controller.retry_failed()
    elif action == "continue":
        _job_records_fail(rig, lambda j: j.kind is JobKind.NOTES and j.stage is Stage.WAITING_NOTES)
        rig.controller.continue_notes()
    else:
        _job_records_fail(rig, lambda j: j.dismissed)
        rig.controller.dismiss_failed()
    rig.wait_idle()
    assert rig.controller.state.notice.startswith(f"没有处理：{md.name}。")
    after = notes_jobs(rig.store)[0]
    assert (after.stage, after.dismissed) == (before.stage, before.dismissed)


def test_the_computer_stays_awake_while_notes_are_made_and_not_while_they_wait(tmp_path, svc, rigs):
    """C9."""
    gate = gated(svc)
    md = lesson(tmp_path)
    rig = rigs()
    rig.controller.start()
    rig.controller.make_notes(md)
    rig.wait(lambda: len(svc.gens) == 1)
    assert rig.awake.held is True and rig.controller.state.pending == 1
    gate.set()
    rig.wait_idle()
    assert rig.awake.held is False

    svc.serve(BUSY, BUSY)
    other = lesson(tmp_path, name="另一课")
    since = len(rig.states)
    rig.controller.make_notes(other)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    assert rig.awake.held is False and rig.controller.state.waiting_notes == 1
    gate = gated(svc)
    rig.controller.continue_notes()
    rig.wait(lambda: len(svc.gens) == 1)
    assert rig.awake.held is True
    gate.set()
    rig.wait_idle()
    assert rig.awake.held is False and rig.awake.violations == []
    assert rig.awake.history == [True, False] * 3


def test_many_clicks_at_once_make_one_job(tmp_path, svc, rigs, monkeypatch):
    """C10."""
    holder = {"gate": threading.Event()}

    def respond(prompt):
        holder["gate"].wait(5)
        return GOOD

    svc.serve(respond=respond)
    folder = lesson(tmp_path).parent
    monkeypatch.chdir(folder)
    rig = rigs()
    rig.controller.start()
    for round_ in range(5):
        md = lesson(tmp_path, name=f"第{round_}课")
        ways = [md, folder / ".." / folder.name / md.name, Path(md.name)]
        holder["gate"] = threading.Event()
        asked = len(svc.gens)
        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def click(path):
            try:
                barrier.wait()
                rig.controller.make_notes(path)
            except BaseException as exc:  # a thread's exception is otherwise only printed
                errors.append(exc)

        threads = [threading.Thread(target=click, args=(ways[i % len(ways)],)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert errors == [], round_
        assert not any(thread.is_alive() for thread in threads), round_
        holder["gate"].set()
        rig.wait_idle()
        assert len(notes_jobs(rig.store, md)) == 1, round_
        assert len(svc.gens) - asked == 1, round_
        assert notes_files(md) == [f"{md.stem}-整理笔记.md"], round_


class TimedTranscriber(RigTranscriber):
    def __init__(self):
        super().__init__()
        self.times: list[tuple[str, float]] = []

    def transcribe(self, audio, model, start_at, on_segment, on_progress):
        self.times.append(("start", time.perf_counter()))
        super().transcribe(audio, model, start_at, on_segment, on_progress)

    def release(self):
        self.times.append(("release", time.perf_counter()))
        super().release()


def test_notes_take_turns_with_transcribing_and_proofreading_one_model_at_a_time(tmp_path, svc, rigs):
    """C11."""
    proofread = _three_part_model()
    svc.serve(respond=lambda prompt: GOOD if "【校对稿开始】" in prompt else proofread(prompt))
    md = lesson(tmp_path)
    transcriber = TimedTranscriber()
    rig = rigs(
        transcriber=transcriber,
        proofreader=OllamaProofreader(svc.client, chunk_chars=100, retry_pause=0),
    )
    rig.controller.start()
    rig.controller.add_files([rig.source("a.mp3")])
    rig.controller.make_notes(md)
    rig.controller.add_files([rig.source("b.mp3")])
    rig.wait_idle()
    jobs = rig.store.load_all()
    assert [job.stage for job in jobs] == [Stage.DONE] * 3
    assert [job.kind for job in jobs] == [JobKind.FILE, JobKind.NOTES, JobKind.FILE]

    bodies = iter(svc.fake.bodies)
    events: list[tuple[float, str, str | None]] = []
    for when, path, _ in svc.fake.log:
        if path == "/api/tags":
            events.append((when, "tags", None))
            continue
        _, raw = next(bodies)
        body = json.loads(raw)
        if path == "/api/show":
            continue
        if "prompt" in body:
            events.append((when, "notes" if "【校对稿开始】" in body["prompt"] else "proofread", body["model"]))
        else:
            events.append((when, "unload", body["model"]))

    def proofreading(after: float) -> tuple[float, float]:
        start = next(t for t, kind, _ in events if kind == "proofread" and t > after)
        end = next(t for t, kind, model in events if kind == "unload" and model == "model-a" and t > start)
        return start, end

    assert [kind for kind, _ in transcriber.times] == ["start", "release", "start", "release"]
    a_transcribe = (transcriber.times[0][1], transcriber.times[1][1])
    b_transcribe = (transcriber.times[2][1], transcriber.times[3][1])
    a_proofread = proofreading(a_transcribe[1])
    asked = next(i for i, (_, kind, _) in enumerate(events) if kind == "notes")
    notes_start = max(t for t, kind, _ in events[:asked] if kind == "tags")
    notes_end = next(t for t, kind, model in events[asked:] if kind == "unload" and model == DEFAULT_NOTES_MODEL)
    b_proofread = proofreading(b_transcribe[1])
    spans = [a_transcribe, a_proofread, (notes_start, notes_end), b_transcribe, b_proofread]
    assert all(start <= end for start, end in spans), spans
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), spans


@pytest.mark.parametrize("moment", ["after-the-last-save", "on-the-failure-status"])
def test_a_click_just_as_the_worker_saves_a_failure_is_not_lost(tmp_path, svc, rigs, moment):
    """C12."""
    svc.serve(REFUSED, GOOD)
    md = lesson(tmp_path)
    rig = rigs()
    clicked: list[str] = []

    def click():
        clicked.append("")
        rig.controller.make_notes(md)
        clicked[0] = rig.controller.state.notice

    if moment == "after-the-last-save":
        real_save = rig.store.save

        def save(job):
            real_save(job)
            if not clicked and job.kind is JobKind.NOTES and job.stage is Stage.FAILED and job.attempts == 0:
                click()

        rig.store.save = save
    else:
        rig.controller.add_listener(lambda state: click() if not clicked and state.status == NOTES_REFUSED else None)
    rig.controller.start()
    since = len(rig.states)
    rig.controller.make_notes(md)
    assert saw_pending_one(rig, since)
    rig.wait_idle()
    assert clicked == [NOTES_ALREADY.format(name=md.name)]
    assert not any(job.stage is Stage.NOTES for job in rig.store.load_all())  # never NOTES with nothing to run it

    for _ in range(3):
        if notes_jobs(rig.store)[0].stage is Stage.DONE:
            break
        rig.controller.make_notes(md)
        rig.wait_idle()
        assert not any(job.stage is Stage.NOTES for job in rig.store.load_all())
    jobs = rig.store.load_all()
    assert [job.stage for job in jobs] == [Stage.DONE]
    assert notes_files(md) == [NOTES_FILE]
    assert len(svc.gens) == 2


# ---------------------------------------------------------------- messages

PIPELINE_MESSAGES = {
    "NOTES_SOURCE_MISSING": NOTES_SOURCE_MISSING,
    "NOTES_SOURCE_HELD": NOTES_SOURCE_HELD,
    "NOTES_SOURCE_UNREADABLE": NOTES_SOURCE_UNREADABLE,
    "NOTES_SOURCE_ENCODING": NOTES_SOURCE_ENCODING,
    "NOTES_WRITE_FAILED": NOTES_WRITE_FAILED,
    "NOTES_PATH_TOO_LONG": NOTES_PATH_TOO_LONG,
    "NOTES_UNEXPECTED": NOTES_UNEXPECTED,
}
CONTROLLER_MESSAGES = {
    "NOTES_ALREADY": NOTES_ALREADY,
    "NOTES_ALREADY_WAITING": NOTES_ALREADY_WAITING,
    "NOTES_NOT_A_FILE": NOTES_NOT_A_FILE,
    "NOTES_NOT_MD": NOTES_NOT_MD,
    "NOTES_IS_NOTES": NOTES_IS_NOTES,
    "NOTES_PATH_TOO_LONG_BEFORE": NOTES_PATH_TOO_LONG_BEFORE,
}
_FIELDS = {
    "NOTES_SOURCE_MISSING": ["path"], "NOTES_SOURCE_HELD": ["path"], "NOTES_SOURCE_UNREADABLE": ["path"],
    "NOTES_SOURCE_ENCODING": ["path"], "NOTES_WRITE_FAILED": ["folder"], "NOTES_PATH_TOO_LONG": ["path"],
    "NOTES_UNEXPECTED": [], "NOTES_ALREADY": ["name"], "NOTES_ALREADY_WAITING": ["name"],
    "NOTES_NOT_A_FILE": ["name"], "NOTES_NOT_MD": ["name"], "NOTES_IS_NOTES": ["name"],
    "NOTES_PATH_TOO_LONG_BEFORE": ["name", "path"],
}


def _fields(template: str) -> list[str]:
    return [field for _, field, _, _ in string.Formatter().parse(template) if field is not None]


def _shown(template: str) -> str:
    """The words the user reads: the template without its placeholders."""
    return "".join(literal for literal, _, _, _ in string.Formatter().parse(template))


def _pattern(template: str) -> re.Pattern:
    parts = []
    for literal, field, _, _ in string.Formatter().parse(template):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append("(.+)")
    return re.compile("".join(parts), re.DOTALL)


def test_the_notes_messages_follow_the_rules_and_are_the_ones_shown(tmp_path, svc, monkeypatch):
    """M1."""
    every = {**PIPELINE_MESSAGES, **CONTROLLER_MESSAGES}
    for key, template in every.items():
        assert _fields(template) == _FIELDS[key], key
        filled = template.format(path=Path("D:/课/课.md"), folder=Path("D:/课"), name="课.md")
        for code in ("Notes", "WI", "12-"):
            assert code not in filled, (key, code)
    for key, template in PIPELINE_MESSAGES.items():
        assert "重试" in template and "继续整理笔记" not in template, key
        assert "校对稿" in template, key
        assert set(re.findall(r"[A-Za-z]+", _shown(template))) <= {"UTF"}, key
    for key, template in CONTROLLER_MESSAGES.items():
        assert set(re.findall(r"[A-Za-z]+", _shown(template))) <= {"Ollama", "md"}, key
    assert "重试" not in NOTES_PATH_TOO_LONG_BEFORE
    assert "继续整理笔记" in NOTES_ALREADY_WAITING
    assert "UTF-8" in NOTES_SOURCE_ENCODING
    assert NOTES_WORKING == "正在整理笔记"

    shown: list[str] = []
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)

    def pipeline_message(prepare, writer=None) -> str:
        root = tmp_path / f"p{len(shown)}"
        md = lesson(root)
        store = JobStore(root / "state")
        job = new_job(store, md)
        with pytest.MonkeyPatch.context() as patch:
            prepare(patch, md)
            svc.serve()
            Flow(store, writer or svc.writer()).run(job)
        assert job.stage is Stage.FAILED, job.message
        return job.message

    shown.append(pipeline_message(lambda patch, md: md.unlink()))
    shown.append(pipeline_message(lambda patch, md: _held(patch, md)))
    shown.append(pipeline_message(lambda patch, md: _held(patch, md, error=OSError(5, "Input/output error"))))
    shown.append(pipeline_message(lambda patch, md: md.write_bytes(SRC.encode("gbk"))))
    shown.append(pipeline_message(lambda patch, md: DiskFault(patch, md.parent, DISK_FULL)))
    shown.append(pipeline_message(lambda patch, md: None, svc.writer(rules=lambda: 1 / 0)))
    md, base = long_lesson(tmp_path / "long", PATH_LIMIT)
    base.write_bytes(b"x")
    store = JobStore(tmp_path / "long" / "state")
    svc.serve()
    shown.append(Flow(store, svc.writer()).run(new_job(store, md)).message)

    rig = Rig(tmp_path / "c", notes_writer=svc.writer())
    try:
        rig.controller.start()
        folder = lesson(tmp_path / "c").parent
        (folder / "x.txt").write_bytes(b"x")
        (folder / "x-整理笔记.md").write_bytes(b"x")
        for path in (folder / "没有.md", folder / "x.txt", folder / "x-整理笔记.md"):
            rig.controller.make_notes(path)
            shown.append(rig.controller.state.notice)
        long_md, _ = long_lesson(tmp_path / "c2", PATH_LIMIT + 1)
        rig.controller.make_notes(long_md)
        shown.append(rig.controller.state.notice)
        gate = gated(svc)
        md = folder / f"{NAME}.md"
        rig.controller.make_notes(md)
        rig.wait(lambda: len(svc.gens) == 1)
        rig.controller.make_notes(md)
        shown.append(rig.controller.state.notice)
        gate.set()
        rig.wait_idle()
        svc.serve(BUSY, BUSY)
        waiting = lesson(tmp_path / "c", name="另一课")
        rig.controller.make_notes(waiting)
        rig.wait_idle()
        rig.controller.make_notes(waiting)
        shown.append(rig.controller.state.notice)
    finally:
        rig.check_keep_awake_follows_busy()

    patterns = [_pattern(template) for template in every.values()]
    assert len(shown) == 13
    for message in shown:
        assert any(p.fullmatch(message) for p in patterns), message
    matched = {key for key, template in every.items() if any(_pattern(template).fullmatch(m) for m in shown)}
    assert matched == set(every), set(every) - matched


# ---------------------------------------------------------------- properties

_P1_SOURCES = (
    ("ok", 0.4), ("bom", 0.1), ("utf16", 0.1), ("missing", 0.08), ("held", 0.08), ("gbk", 0.08),
    ("title", 0.08), ("toolong", 0.08),
)
_P1_SERVICES = (
    ("ok", 0.5), ("busy2", 0.15), ("cut2", 0.1), ("missing_model", 0.1), ("refused", 0.1), ("busy_then_ok", 0.05),
)
_P1_DISKS = ((None, 0.7), ("full", 0.15), ("read-only", 0.15))
_P1_EXISTING = ((None, 0.6), ("same", 0.2), ("other", 0.2))
_P1_SCRIPTS = {
    "ok": (GOOD,), "busy2": (BUSY, BUSY), "cut2": (CUT, CUT), "missing_model": (), "refused": (REFUSED,),
    "busy_then_ok": (BUSY, GOOD),
}
_P1_DATA = {
    "ok": SRC_BYTES,
    "bom": codecs.BOM_UTF8 + SRC_CRLF.encode("utf-8"),
    "utf16": codecs.BOM_UTF16_LE + SRC_CRLF.encode("utf-16-le"),
    "held": SRC_BYTES,
    "gbk": SRC.encode("gbk"),
    "title": TITLE_ONLY,
    "toolong": TOO_LONG_SOURCE.encode("utf-8"),
}


def _pick(rng: random.Random, table):
    roll, total = rng.random(), 0.0
    for value, weight in table:
        total += weight
        if roll < total:
            return value
    return table[-1][0]


def _p1_reference(source: str, service: str, disk: str | None, md: Path) -> tuple[Stage, str, bool]:
    """(stage, message, the model gave a usable reply) by the stated rules, without the code under test."""
    early = {
        "missing": NOTES_SOURCE_MISSING.format(path=md),
        "held": NOTES_SOURCE_HELD.format(path=md),
        "gbk": NOTES_SOURCE_ENCODING.format(path=md),
        "title": NOTES_EMPTY_SOURCE,
        "toolong": NOTES_TOO_LONG.format(chars=100, limit=99),
    }
    if source in early:
        return Stage.FAILED, early[source], False
    if service == "busy2":
        return Stage.WAITING_NOTES, NOTES_BUSY_MESSAGE, False
    if service == "missing_model":
        return Stage.WAITING_NOTES, NOTES_MODEL_MISSING.format(model="model-x"), False
    if service == "cut2":
        return Stage.FAILED, NOTES_CUT_OFF, False
    if service == "refused":
        return Stage.FAILED, NOTES_REFUSED, False
    if disk is not None:
        return Stage.FAILED, NOTES_WRITE_FAILED.format(folder=md.parent), True
    return Stage.DONE, "", True


def test_any_md_any_service_any_disk_ends_by_the_rules_and_leaves_the_originals(tmp_path, svc):
    """P1."""
    rng = random.Random(20270310)
    seen = {"done": 0, "fixed": 0, "same-numbered": 0, "waiting": 0, "missing": 0, "held": 0, "gbk": 0, "marked": 0}
    for round_ in range(150):
        source, service = _pick(rng, _P1_SOURCES), _pick(rng, _P1_SERVICES)
        disk, existing = _pick(rng, _P1_DISKS), _pick(rng, _P1_EXISTING)
        where = (round_, source, service, disk, existing)
        root = tmp_path / f"r{round_}"
        folder = root / "课程"
        folder.mkdir(parents=True)
        md = folder / f"{NAME}.md"
        if source != "missing":
            md.write_bytes(_P1_DATA[source])
        (folder / f"{NAME}.txt").write_bytes(TXT_BYTES)
        if existing is not None:
            (folder / NOTES_FILE).write_bytes(FINISHED_BYTES if existing == "same" else "别人的\n".encode("utf-8"))
        before = snapshots(folder)
        store = JobStore(root / "state")
        job = new_job(store, md, "model-x" if service == "missing_model" else "model-a")
        stage, message, replied = _p1_reference(source, service, disk, md)
        with pytest.MonkeyPatch.context() as patch:
            if source == "held":
                patch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
                _held(patch, md)
            fault = DiskFault(patch, folder, DISK_FULL if disk == "full" else READ_ONLY) if disk else None
            svc.serve(*_P1_SCRIPTS[service])
            writer = svc.writer(rules=two_char_rules, num_ctx=407, reply_budget=50) if source == "toolong" else svc.writer()
            Flow(store, writer).run(job)
            assert (job.stage, job.message) == (stage, message), where
            if source in ("missing", "held", "gbk", "title", "toolong"):
                assert svc.fake.log == [], where
            if stage is not Stage.DONE:
                assert names(folder) == sorted(before), where
                assert store.notes_path(job).exists() is replied, where
            if message.startswith("笔记没能保存"):
                fault.fix()
                svc.serve()
                Flow(store, writer).run(retry(job))
                assert job.stage is Stage.DONE, where
                assert svc.gens == [], where  # the kept reply, not the model again
                seen["fixed"] += 1
        assert unchanged(folder, before), where
        assert temp_files(folder) == [], where
        new = sorted(set(names(folder)) - set(before))
        if job.stage is Stage.DONE:
            expected = NOTES_FILE if existing is None else f"{NAME}-整理笔记 (2).md"
            assert new == [expected] and job.notes_path == folder / expected, where
            assert (folder / expected).read_bytes() == FINISHED_BYTES, where
            assert not store.notes_path(job).exists(), where
            seen["done"] += 1
            seen["same-numbered"] += existing == "same"
            seen["marked"] += source in ("bom", "utf16")
        else:
            assert new == [], where
            seen["waiting"] += job.stage is Stage.WAITING_NOTES
            if source in ("missing", "held", "gbk"):
                seen[source] += 1
    print(seen)
    assert seen["done"] >= 25 and seen["fixed"] >= 8 and seen["same-numbered"] >= 5, seen
    assert seen["waiting"] >= 8 and seen["marked"] >= 5, seen
    assert min(seen["missing"], seen["held"], seen["gbk"]) >= 3, seen


def _usable_reply(rng: random.Random) -> tuple[str, bytes]:
    """Another reply the model might give that the checks keep something of, and the notes it gives."""
    while True:
        reply, _ = make_reply(rng, clean_titles=True)
        reply = reply.strip()
        if not reply or "<think>" in reply:
            continue
        try:
            return reply, finish_notes(reply, SRC, META).markdown.encode("utf-8")
        except EmptyNotes:
            continue


def test_notes_made_again_and_again_take_the_first_free_name_and_never_touch_older_ones(tmp_path, svc):
    """P2."""
    rng = random.Random(20270311)
    for round_ in range(20):
        root = tmp_path / f"r{round_}"
        md = lesson(root)
        folder = md.parent
        store = JobStore(root / "state")
        kept: dict[str, tuple[bytes, int]] = {}
        done = deleted = 0
        for task in range(rng.randint(1, 8)):
            where = (round_, task)
            if kept and rng.random() < 0.3:
                victim = rng.choice(sorted(kept))
                (folder / victim).unlink()
                del kept[victim]
                deleted += 1
            reply, expected = _usable_reply(rng) if rng.random() < 0.2 else (GOOD, FINISHED_BYTES)
            svc.serve(reply)
            number, name = 1, NOTES_FILE
            while (folder / name).exists():
                number += 1
                name = f"{NAME}-整理笔记 ({number}).md"
            job = Flow(store, svc.writer()).run(new_job(store, md))
            assert job.stage is Stage.DONE and job.notes_path == folder / name, where
            assert job.notes_path.read_bytes() == expected, where
            assert unchanged(folder, kept), where
            kept[name] = _snapshot(job.notes_path)
            done += 1
            assert len(notes_files(md)) == done - deleted, where


_P3_ACTIONS = (
    ("make", 0.35), ("continue", 0.15), ("retry", 0.15), ("dismiss", 0.1), ("wait", 0.15), ("restart", 0.1),
)


def test_any_clicks_keep_one_open_job_per_md_and_one_file_per_finished_job(tmp_path, svc):
    """P3."""
    rng = random.Random(20270312)
    replies = {"rng": random.Random(0)}

    def respond(prompt):
        roll = replies["rng"].random()
        return GOOD if roll < 0.5 else BUSY if roll < 0.7 else CUT if roll < 0.85 else REFUSED

    made_jobs = 0
    for round_ in range(10):
        root = tmp_path / f"r{round_}"
        mds = [lesson(root), lesson(root, name="2026.10.31 另一课")]
        replies["rng"] = random.Random(rng.random())
        svc.serve(respond=respond)
        made = {md: 0 for md in mds}
        rig = Rig(root, notes_writer=svc.writer())
        rig.controller.start()
        try:
            for step in range(12):
                action = _pick(rng, _P3_ACTIONS)
                where = (round_, step, action)
                if action == "make":
                    md = rng.choice(mds)
                    rig.wait_idle()
                    if not any(j.stage in (Stage.NOTES, Stage.WAITING_NOTES) or j.offers_retry for j in notes_jobs(rig.store, md)):
                        made[md] += 1
                        made_jobs += 1
                    rig.controller.make_notes(rng.choice(spellings(md)))
                elif action == "continue":
                    rig.controller.continue_notes()
                elif action == "retry":
                    rig.controller.retry_failed()
                elif action == "dismiss":
                    rig.controller.dismiss_failed()
                elif action == "restart":
                    rig.check_keep_awake_follows_busy()
                    rig = Rig(root, notes_writer=svc.writer())
                    rig.controller.start()
                rig.wait_idle()
                for md in mds:
                    mine = notes_jobs(rig.store, md)
                    assert len(mine) == made[md], where
                    assert sum(j.stage in (Stage.NOTES, Stage.WAITING_NOTES) or j.offers_retry for j in mine) <= 1, where
                    assert len(notes_files(md)) == sum(j.stage is Stage.DONE for j in mine), where
                    assert md.read_bytes() == SRC_BYTES, where
                jobs = rig.store.load_all()
                state = rig.controller.state
                assert state.waiting_notes == sum(j.stage is Stage.WAITING_NOTES for j in jobs), where
                assert state.failed == sum(j.offers_retry for j in jobs), where
                assert rig.awake.violations == [], where
        finally:
            rig.check_keep_awake_follows_busy()
    assert made_jobs >= 10


# ---------------------------------------------------------------- integration: the real Ollama


def _loaded_models() -> list[str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(DEFAULT_URL + "/api/ps", timeout=10) as response:
        return [item.get("name", "") for item in json.loads(response.read().decode("utf-8")).get("models", [])]


@pytest.mark.integration
def test_real_ollama_makes_notes_from_a_proofread_md(tmp_path, monkeypatch):
    """I1."""
    _ready_notes_model(OllamaClient())
    monkeypatch.setenv("COURSESCRIBE_HOME", str(tmp_path / "家"))  # no rules file: the shipped rules
    folder = tmp_path / "课程"
    folder.mkdir()
    md = folder / "2027.01.15 数据库索引入门.md"
    md.write_bytes(LESSON.encode("utf-8"))
    rig = Rig(tmp_path, notes_writer=OllamaNotesWriter())
    try:
        assert rig.controller.settings.notes_model == NOTES_MODEL
        rig.controller.start()
        rig.controller.make_notes(md)
        assert rig.controller.state.pending == 1
        rig.wait(lambda: rig.controller.state.pending == 0, seconds=600)
        job = rig.store.load_all()[0]
        assert job.stage is Stage.DONE, job.message
        notes = folder / "2027.01.15 数据库索引入门-整理笔记.md"
        assert notes.read_text(encoding="utf-8").startswith(header(LESSON_META))
        assert job.removed_quotes >= 0 and job.removed_concepts >= 0
        print(f"\nremoved quotes: {job.removed_quotes}, removed concepts: {job.removed_concepts}")
        assert rig.controller.state.notice.startswith("笔记已生成：")
        deadline = time.time() + 10
        while NOTES_MODEL in _loaded_models() and time.time() < deadline:
            time.sleep(0.5)
        assert NOTES_MODEL not in _loaded_models()
        assert md.read_bytes() == LESSON.encode("utf-8")
        assert names(rig.store.folder) == [f"{job.job_id}.json"]
    finally:
        rig.check_keep_awake_follows_busy()


# ---------------------------------------------------------------- gui: the real desktop


@pytest.mark.gui
def test_an_md_really_held_by_another_program(tmp_path, svc):
    """G1."""
    md = lesson(tmp_path)
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    hold_without_sharing(md, 3.0)
    started = time.perf_counter()
    Flow(store, svc.writer()).run(job)
    took = time.perf_counter() - started
    if FILES_CAN_BE_HELD:
        assert job.stage is Stage.FAILED and job.message == NOTES_SOURCE_HELD.format(path=md)
        assert took < 2
        deadline = time.time() + 10
        while True:
            try:
                md.read_bytes()
                break
            except PermissionError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
        Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE
    assert md.read_bytes() == SRC_BYTES


class deny_new_files:  # noqa: N801  used like a function, as a context manager
    """Make ``folder`` refuse new files for real: an ACL on Windows, permissions elsewhere."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def __enter__(self):
        if os.name == "nt":
            user = os.environ.get("USERNAME", "")
            subprocess.run(["icacls", str(self.folder), "/deny", f"{user}:(WD,AD)"], check=True, capture_output=True)
        else:
            self.mode = self.folder.stat().st_mode
            self.folder.chmod(0o555)
        probe = self.folder / "probe.tmp"
        try:
            probe.write_bytes(b"x")
        except OSError:
            return self
        probe.unlink()
        self.__exit__()
        pytest.fail(f"Could not make {self.folder} read-only for this test")

    def __exit__(self, *args):
        if os.name == "nt":
            user = os.environ.get("USERNAME", "")
            subprocess.run(["icacls", str(self.folder), "/remove:d", user], check=True, capture_output=True)
        else:
            self.folder.chmod(self.mode)


@pytest.mark.gui
def test_a_really_read_only_folder(tmp_path, svc):
    """G2."""
    md = lesson(tmp_path)
    folder = md.parent
    store = JobStore(tmp_path / "state")
    job = new_job(store, md)
    with deny_new_files(folder):
        Flow(store, svc.writer()).run(job)
        assert job.stage is Stage.FAILED and job.message == NOTES_WRITE_FAILED.format(folder=folder)
        assert names(folder) == sorted([md.name, f"{NAME}.txt"])
    Flow(store, svc.writer()).run(retry(job))
    assert job.stage is Stage.DONE and len(svc.gens) == 1
