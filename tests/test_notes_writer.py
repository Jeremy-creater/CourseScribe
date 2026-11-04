"""Writing organised notes with the model (12-5): the order of steps, the service's errors,
the checkpoint of the raw reply, and releasing the model. The service is a real local HTTP
server (FakeOllama) talked to by the real client; the integration cases use the real Ollama."""

from __future__ import annotations

import codecs
import json
import logging
import math
import random
import re
import string
import subprocess
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pytest

from coursescribe.files import decode_user_text
from coursescribe.notes import (
    EmptyNotes,
    NotesMeta,
    build_prompt,
    finish_notes,
    overflow,
    source_body,
    structure_notes,
)
from coursescribe.notes_rules import (
    RULES_EMPTY_NOTE,
    RULES_ENCODING,
    RULES_HELD,
    RULES_SHIPPED_MISSING,
    RULES_UNREADABLE,
    NotesRules,
    NotesRulesUnreadable,
    load_notes_rules,
    shipped_rules,
)
from coursescribe.notes_writer import (
    NOTES_ALL_REMOVED,
    NOTES_BUSY_MESSAGE,
    NOTES_CUT_OFF,
    NOTES_DOWN,
    NOTES_EMPTY_REPLY,
    NOTES_EMPTY_SOURCE,
    NOTES_MODEL_MISSING,
    NOTES_NUM_CTX,
    NOTES_REFUSED,
    NOTES_REPLY_BUDGET,
    NOTES_RULES_TOO_LONG,
    NOTES_TIMEOUT,
    NOTES_TIMEOUT_MESSAGE,
    NOTES_TOO_LONG,
    GeneratedNotes,
    NotesBlocked,
    NotesFailed,
    OllamaNotesWriter,
)
from coursescribe.ollama import (
    DEFAULT_URL,
    OllamaBusy,
    OllamaClient,
    OllamaDown,
    OllamaError,
    OllamaIncomplete,
    OllamaModelMissing,
    OllamaTimeout,
)
from fake_ollama import FakeOllama, Reply
from notes_replies import META, make_reply
from test_notes import (
    _LONG_PARAGRAPH,
    FINISH_REPLY,
    FINISHED,
    PROMPT_HEAD,
    PROMPT_TAIL,
    SRC,
    _lesson_variants,
    _make_lesson,
    header,
    headings,
    links_resolve_once,
    outline_lines,
)
from test_notes_rules import make_rules_text
from test_proofread import NOTES_MODEL, _closed_port_url, _ready_notes_model
from test_recording import _held

MODEL = "model-a"
GOOD = FINISH_REPLY
GOOD_RESULT = GeneratedNotes(FINISHED, 2, 2, None)
RULES_FILE = "// 说明\n## 核心宗旨\n用一两句话说这节课最重要的意思。\n".encode("utf-8")
RULES = "## 核心宗旨\n用一两句话说这节课最重要的意思。"
SHIPPED_FILE = "// x\n自带规则QZX\n".encode("utf-8")
LITERAL_PROMPT = (
    "## 核心宗旨\n用一两句话说这节课最重要的意思。"
    + PROMPT_HEAD
    + "索引就像一本书的目录，没有它，就只能一页一页地翻。\n大家记住：索引不是万能的！写入会因此变慢一点。\n\n"
    "ACID 是事务的四个特性，第 3 个是隔离性。\n联合索引要遵守最左前缀\n原则。B+ 树的叶子节点，连成一条链表。"
    + PROMPT_TAIL
)

CUT = Reply("## 核心宗旨\n写到一半", fields={"done_reason": "length"})
NOT_DONE = Reply("## 核心宗旨\n写到一半", fields={"done": False})
BUSY = Reply("x", status=500)
REFUSED = Reply("bad request", status=400)
MISSING = Reply("model 'model-b' not found", status=404)
# Ollama quits while writing: the service closes itself and this request is hung up on.
GONE = Reply(GOOD, close_service=True)

EMPTY_REPLIES = (
    "",
    "  \r\n\t\n",
    "# 只有标题",
    "---\ntags: [x]\n---\n# 标题\n日期：2026-10-30\n来源：网课",
    "## 本课脉络\n- [[#一]]\n- [[#二]]",
    "```markdown\n\n```",
    "。，！\n——",
)
MADE_UP_QUOTE = "## 引述\n> 这句话原文里根本没有出现过"
MADE_UP_TAG = "#编造概念"
REMOVED_REPLIES = (MADE_UP_QUOTE, MADE_UP_TAG, "## 引述\n- “索引永远让查询更快”\n\n#全表扫描")

WAITING = (NOTES_DOWN, NOTES_TIMEOUT_MESSAGE, NOTES_BUSY_MESSAGE, NOTES_MODEL_MISSING)
FAILING = (
    NOTES_EMPTY_SOURCE, NOTES_TOO_LONG, NOTES_RULES_TOO_LONG, NOTES_CUT_OFF,
    NOTES_EMPTY_REPLY, NOTES_ALL_REMOVED, NOTES_REFUSED,
)
LOGGER = "coursescribe.notes_writer"


# ---------------------------------------------------------------- tools


class Script:
    """The fake model's answers to generation requests, in order. Past the end: ``then``,
    or a loud refusal (418), so a request the test did not expect shows up as a wrong result."""

    def __init__(self, replies, then: Reply | str | None = None) -> None:
        self.replies = list(replies)
        self.then = then

    def __call__(self, *args) -> Reply | str:
        if self.replies:
            return self.replies.pop(0)
        if self.then is not None:
            return self.then
        return Reply("the test script has no more replies", status=418)


@pytest.fixture
def service():
    made = []

    def start(*script, **kwargs) -> FakeOllama:
        fake = FakeOllama(Script(script) if script else (lambda prompt: GOOD), **kwargs)
        made.append(fake)
        return fake

    yield start
    for fake in made:
        fake.close()


@dataclass
class Setup:
    rules_path: Path
    shipped: Path
    checkpoint: Path

    def rules(self) -> NotesRules:
        return load_notes_rules(self.rules_path, shipped=self.shipped)


@pytest.fixture
def setup(tmp_path) -> Setup:
    rules_path = tmp_path / "整理规则.txt"
    rules_path.write_bytes(RULES_FILE)
    shipped = tmp_path / "自带规则.txt"
    shipped.write_bytes(SHIPPED_FILE)
    folder = tmp_path / "job"
    folder.mkdir()
    return Setup(rules_path, shipped, folder / "job.notes.json")


def writer_for(target, rules, **kwargs) -> OllamaNotesWriter:
    url = target.url if isinstance(target, FakeOllama) else target
    client = OllamaClient(
        url, timeout=kwargs.pop("client_timeout", 5.0), unload_timeout=kwargs.pop("unload_timeout", 2.0)
    )
    kwargs.setdefault("retry_pause", 0)
    return OllamaNotesWriter(client, rules, **kwargs)


def client_for(fake: FakeOllama) -> OllamaClient:
    """One client for many writes, as the long property tests make thousands of them."""
    return OllamaClient(fake.url, timeout=5.0, unload_timeout=2.0)


def writer_with(client: OllamaClient, rules, **kwargs) -> OllamaNotesWriter:
    kwargs.setdefault("retry_pause", 0)
    return OllamaNotesWriter(client, rules, **kwargs)


def reset(fake: FakeOllama, *script, then=None) -> None:
    fake.log.clear()
    fake.requests.clear()
    fake.prompts.clear()
    fake.bodies.clear()
    fake.respond = Script(script, then) if script or then is not None else (lambda prompt: GOOD)


def generations(fake: FakeOllama) -> list[dict]:
    return [body for body in fake.requests if "prompt" in body]


def unloads(fake: FakeOllama) -> list[dict]:
    return [body for body in fake.requests if "prompt" not in body]


def sent(fake: FakeOllama) -> list[tuple[str, bool]]:
    """(path, carries a prompt) of every request, in order."""
    return [(path, prompt is not None) for _, path, prompt in fake.log]


def temp_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob(".coursescribe-*.tmp"))


def outcome(call):
    """What a write gave: ("ok", result), ("blocked", exc) or ("failed", exc)."""
    try:
        return "ok", call()
    except NotesBlocked as exc:
        return "blocked", exc
    except NotesFailed as exc:
        return "failed", exc


def assert_released_last(fake: FakeOllama, ended: float, model: str = MODEL) -> None:
    assert unloads(fake) == [{"model": model, "keep_alive": 0}]
    when, path, prompt = fake.log[-1]
    assert (path, prompt) == ("/api/generate", None)
    assert when < ended


def records(caplog, level: int, text: str = "") -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level and text in r.getMessage()]


def supplement(first: str, second: str) -> str:
    assert second.startswith(first)
    extra = second[len(first):]
    assert extra.startswith("\n")
    assert len(extra) <= 59
    return extra


def test_the_reply_groups_are_what_they_claim_to_be():
    """Guards the material: empty replies are empty to structure_notes, removed ones only to finish_notes."""
    for reply in EMPTY_REPLIES:
        with pytest.raises(EmptyNotes):
            structure_notes(reply.strip(), META)
    for reply in REMOVED_REPLIES:
        structure_notes(reply, META)
        with pytest.raises(EmptyNotes):
            finish_notes(reply, SRC, META)


# ---------------------------------------------------------------- W1


@pytest.mark.parametrize("thinking", [False, True], ids=["plain", "thinking"])
def test_a_typical_lesson_becomes_notes_and_the_model_is_released(service, setup, thinking):
    """W1."""
    fake = service(GOOD, capabilities=("completion", "thinking") if thinking else ("completion",))
    result = writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    ended = time.perf_counter()

    assert result == GeneratedNotes(FINISHED, 2, 2, None)
    assert fake.prompts == [LITERAL_PROMPT]
    assert sent(fake) == [
        ("/api/tags", False), ("/api/show", False), ("/api/generate", True), ("/api/generate", False),
    ]
    expected = {
        "model": MODEL, "prompt": LITERAL_PROMPT, "stream": False,
        "options": {"num_ctx": 32768, "temperature": 0.1, "num_predict": 8192},
    }
    if thinking:
        expected["think"] = False
    assert generations(fake) == [expected]
    assert_released_last(fake, ended)
    saved = json.loads(setup.checkpoint.read_text("utf-8"))
    assert set(saved) == {"version", "fingerprint", "reply"}
    assert saved["version"] == 1
    assert isinstance(saved["fingerprint"], str) and re.fullmatch(r"[0-9a-f]{64}", saved["fingerprint"])
    assert saved["reply"] == FINISH_REPLY.strip()
    assert [p.name for p in setup.checkpoint.parent.iterdir()] == [setup.checkpoint.name]
    assert (NOTES_NUM_CTX, NOTES_REPLY_BUDGET, NOTES_TIMEOUT) == (32768, 8192, 1800)


# ---------------------------------------------------------------- W2


def test_empty_rules_use_the_shipped_ones_and_say_so_also_when_replayed(service, setup, caplog):
    """W2(a)."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    setup.rules_path.write_bytes("// 只有说明\n".encode("utf-8"))
    fake = service(GOOD)
    result = writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)

    assert result.markdown == FINISHED
    assert fake.prompts[0].startswith("自带规则QZX" + PROMPT_HEAD)
    assert result.rules_note == RULES_EMPTY_NOTE.format(path=setup.rules_path)
    # One fixed sentence in the log; the note itself, with the full path (and so the Windows
    # user name), only goes to the result for the window.
    assert len(records(caplog, logging.INFO, "shipped notes rules")) == 1
    assert str(setup.rules_path.parent) not in caplog.text
    assert "QZX" not in caplog.text

    caplog.clear()
    log_length = len(fake.log)
    again = writer_for(_closed_port_url(), setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    assert again == result
    assert len(fake.log) == log_length
    assert len(records(caplog, logging.INFO, "shipped notes rules")) == 1
    assert str(setup.rules_path.parent) not in caplog.text
    assert setup.rules_path.read_bytes() == "// 只有说明\n".encode("utf-8")


def test_without_a_rules_file_the_shipped_rules_are_used_quietly(service, setup, caplog):
    """W2(b)."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    setup.rules_path.unlink()
    fake = service(GOOD)
    result = writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    assert result.markdown == FINISHED
    assert fake.prompts[0].startswith("自带规则QZX" + PROMPT_HEAD)
    assert result.rules_note is None
    assert records(caplog, logging.INFO, str(setup.rules_path)) == []
    assert records(caplog, logging.INFO, "shipped notes rules") == []
    assert not setup.rules_path.exists()


def test_the_default_rules_are_read_afresh_from_the_program_folder(service, tmp_path, monkeypatch):
    """W2(c)(d)."""
    home = tmp_path / "家"
    home.mkdir()
    monkeypatch.setenv("COURSESCRIBE_HOME", str(home))
    (home / "整理规则.txt").write_bytes("我的规则\n".encode("utf-8"))
    fake = service(GOOD, GOOD)
    writer = OllamaNotesWriter(OllamaClient(fake.url, timeout=5, unload_timeout=2), retry_pause=0)

    first = writer.write(SRC, META, MODEL, tmp_path / "一.notes.json")
    assert first.markdown == FINISHED
    assert fake.prompts[0].startswith("我的规则" + PROMPT_HEAD)

    (home / "整理规则.txt").write_bytes("我的新规则\n".encode("utf-8"))
    second = writer.write(SRC, META, MODEL, tmp_path / "二.notes.json")
    assert second.markdown == FINISHED
    assert fake.prompts[1].startswith("我的新规则" + PROMPT_HEAD)
    assert [p.name for p in home.iterdir()] == ["整理规则.txt"]
    assert (home / "整理规则.txt").read_bytes() == "我的新规则\n".encode("utf-8")


# ---------------------------------------------------------------- W3


@pytest.mark.parametrize("ollama", ["running", "not-running"])
@pytest.mark.parametrize("case", ["encoding", "folder", "no-shipped", "after-good-rules", "held"])
def test_rules_that_cannot_be_read_fail_without_asking_the_service(service, setup, monkeypatch, case, ollama):
    """W3."""
    fake = service(GOOD)
    path, shipped = setup.rules_path, setup.shipped
    before = None
    if case == "after-good-rules":
        assert writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
        before = setup.checkpoint.read_bytes()
        reset(fake)
    if case in ("encoding", "after-good-rules"):
        path.write_bytes("café".encode("latin-1"))
        expected = RULES_ENCODING.format(path=path)
    elif case == "folder":
        path.unlink()
        path.mkdir()
        expected = RULES_UNREADABLE.format(path=path)
    elif case == "no-shipped":
        path.unlink()
        shipped.unlink()
        expected = RULES_SHIPPED_MISSING.format(path=shipped)
    else:
        monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
        _held(monkeypatch, path)
        expected = RULES_HELD.format(path=path)
    writer = writer_for(fake if ollama == "running" else _closed_port_url(), setup.rules)

    with pytest.raises(NotesFailed) as caught:
        writer.write(SRC, META, MODEL, setup.checkpoint)
    assert str(caught.value) == expected
    assert isinstance(caught.value.__cause__, NotesRulesUnreadable)
    assert str(caught.value) == str(caught.value.__cause__)
    assert not isinstance(caught.value, NotesBlocked)
    assert fake.log == []
    if case == "after-good-rules":
        assert setup.checkpoint.read_bytes() == before
        path.write_bytes(RULES_FILE)
        assert writer.write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
        assert fake.log == []
    else:
        assert not setup.checkpoint.exists()


# ---------------------------------------------------------------- W4


@pytest.mark.parametrize(
    "source",
    ["", "# 2026.10.30 数据库索引", "﻿# 课\r\n\r\n  \r\n", "# 课\n\n。。。——", "# 课\n```\n```"],
)
def test_an_empty_lesson_is_refused_before_anything_else(service, setup, source):
    """W4."""
    fake = service(GOOD)
    with pytest.raises(NotesFailed) as caught:
        writer_for(fake, setup.rules).write(source, META, MODEL, setup.checkpoint)
    assert str(caught.value) == NOTES_EMPTY_SOURCE
    assert str(caught.value).startswith("这份校对稿是空的")
    assert fake.log == []
    assert not setup.checkpoint.exists()


def test_an_empty_lesson_is_reported_before_broken_rules(service, setup):
    """W4, both broken."""
    fake = service(GOOD)
    setup.rules_path.unlink()
    setup.rules_path.mkdir()
    with pytest.raises(NotesFailed) as caught:
        writer_for(fake, setup.rules).write("", META, MODEL, setup.checkpoint)
    assert str(caught.value) == NOTES_EMPTY_SOURCE
    assert fake.log == []
    assert not setup.checkpoint.exists()


# ---------------------------------------------------------------- W5

LESSON_100 = "# 课\n\n" + "索" * 100


def two_char_rules() -> NotesRules:
    return NotesRules("规则", None)


def test_a_lesson_that_does_not_fit_is_refused_with_the_right_numbers(service, setup, tmp_path):
    """W5."""
    fake = service("## 核心宗旨\n索索索索。")
    assert len(source_body(LESSON_100)) == 100 and len(LESSON_100) == 105

    fits = writer_for(fake, two_char_rules, num_ctx=408, reply_budget=50).write(
        LESSON_100, META, MODEL, setup.checkpoint
    )
    assert fits.markdown.startswith(header(META))
    [request] = generations(fake)
    assert request["options"]["num_ctx"] == 408
    assert request["options"]["num_predict"] == 50
    assert len(request["prompt"]) + 50 + 128 <= 408
    saved = setup.checkpoint.read_bytes()

    cases = [
        (407, NOTES_TOO_LONG.format(chars=100, limit=99), ("约 100 字", "约 99 字")),
        (309, NOTES_TOO_LONG.format(chars=100, limit=1), ("约 1 字",)),
        (308, NOTES_RULES_TOO_LONG.format(rules=2), ("整理规则", "改短")),
        (200, NOTES_RULES_TOO_LONG.format(rules=2), ("整理规则", "改短")),
    ]
    for num_ctx, expected, words in cases:
        reset(fake)
        checkpoint = tmp_path / f"other-{num_ctx}.notes.json"
        with pytest.raises(NotesFailed) as caught:
            writer_for(fake, two_char_rules, num_ctx=num_ctx, reply_budget=50).write(
                LESSON_100, META, MODEL, checkpoint
            )
        message = str(caught.value)
        assert message == expected, num_ctx
        assert all(word in message for word in words)
        assert not re.search(r"-\d", message)
        assert "约 0 字" not in message
        assert fake.log == []
        assert not checkpoint.exists()

    # The fit is measured before the checkpoint is looked at: reply_budget is not in the fingerprint.
    reset(fake)
    with pytest.raises(NotesFailed) as caught:
        writer_for(fake, two_char_rules, num_ctx=408, reply_budget=51).write(
            LESSON_100, META, MODEL, setup.checkpoint
        )
    assert str(caught.value) == NOTES_TOO_LONG.format(chars=100, limit=99)
    assert fake.log == []
    assert setup.checkpoint.read_bytes() == saved


# ---------------------------------------------------------------- W6


@pytest.mark.parametrize("case", ["not-listed", "listed-as-latest", "404-on-generate"])
def test_a_missing_model_waits_and_says_ollama_pull(service, setup, case):
    """W6."""
    if case == "not-listed":
        fake = service(GOOD, models=("model-a",))
    elif case == "listed-as-latest":
        fake = service(GOOD, models=("model-b:latest",))
    else:
        fake = service(MISSING, models=("model-b",))
    result = outcome(lambda: writer_for(fake, setup.rules).write(SRC, META, "model-b", setup.checkpoint))
    ended = time.perf_counter()

    assert all(path != "/api/pull" for path, _ in sent(fake))
    if case == "listed-as-latest":
        assert result == ("ok", GOOD_RESULT)
        return
    kind, exc = result
    assert kind == "blocked"
    assert str(exc) == NOTES_MODEL_MISSING.format(model="model-b")
    assert all(word in str(exc) for word in ("ollama pull model-b", "继续整理笔记", "不会自己下载"))
    assert isinstance(exc.__cause__, OllamaModelMissing)
    assert not setup.checkpoint.exists()
    if case == "not-listed":
        assert generations(fake) == [] and unloads(fake) == []
    else:
        assert len(generations(fake)) == 1
        assert_released_last(fake, ended, "model-b")


# ---------------------------------------------------------------- W7


def test_ollama_not_running_waits_without_pausing(setup):
    """W7(a)."""
    started = time.perf_counter()
    with pytest.raises(NotesBlocked) as caught:
        writer_for(_closed_port_url(), setup.rules, retry_pause=10).write(SRC, META, MODEL, setup.checkpoint)
    assert time.perf_counter() - started < 8
    assert str(caught.value) == NOTES_DOWN
    assert all(word in str(caught.value) for word in ("没有启动", "校对稿已保存", "继续整理笔记"))
    assert isinstance(caught.value.__cause__, OllamaDown)
    assert not setup.checkpoint.exists()


def test_ollama_quitting_while_writing_waits_and_no_release_is_tried(service, setup, caplog):
    """W7(b). A service confirmed gone holds no model, so releasing is not even tried
    (it would only wait for one more refused connection, about 2 s on Windows)."""
    caplog.set_level(logging.INFO)
    fake = service(GONE)
    started = time.perf_counter()
    with pytest.raises(NotesBlocked) as caught:
        writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    assert time.perf_counter() - started < 3.5
    assert str(caught.value) == NOTES_DOWN
    assert isinstance(caught.value.__cause__, OllamaDown)
    assert [path for path, prompt in sent(fake) if prompt] == ["/api/generate"]
    assert unloads(fake) == []
    assert records(caplog, logging.INFO, "unload") == []
    assert not setup.checkpoint.exists()


# ---------------------------------------------------------------- W8


def test_a_slow_model_times_out_by_the_notes_timeout_and_is_not_asked_again(service, setup):
    """W8(a)."""
    fake = service(Reply(GOOD, delay=1.0))
    started = time.perf_counter()
    with pytest.raises(NotesBlocked) as caught:
        writer_for(fake, setup.rules, client_timeout=5, timeout=0.3).write(SRC, META, MODEL, setup.checkpoint)
    ended = time.perf_counter()
    assert ended - started < 0.9
    assert str(caught.value) == NOTES_TIMEOUT_MESSAGE
    assert isinstance(caught.value.__cause__, OllamaTimeout)
    assert len(generations(fake)) == 1
    assert_released_last(fake, ended)
    assert not setup.checkpoint.exists()


def test_the_notes_timeout_is_used_rather_than_the_clients(service, setup):
    """W8(b)(c)."""
    fake = service(Reply(GOOD, delay=0.5))
    writer = writer_for(fake, setup.rules, client_timeout=0.2, timeout=3)
    assert writer.write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    assert OllamaNotesWriter(OllamaClient(fake.url)).timeout == 1800


def test_without_a_client_the_writer_talks_to_the_local_default():
    """W8(c')."""
    writer = OllamaNotesWriter()
    assert type(writer.client) is OllamaClient
    assert writer.client.url == DEFAULT_URL.rstrip("/")
    assert writer.client.timeout == 600.0


# ---------------------------------------------------------------- W9


@pytest.mark.parametrize(
    ("case", "script", "tags", "succeeds"),
    [
        ("500-then-good", (BUSY, GOOD), (), True),
        ("500-twice", (BUSY, BUSY), (), False),
        ("hung-up-then-good", (Reply(broken="no-reply"), GOOD), (), True),
        ("short-body-twice", (Reply(broken="short-body"), Reply(broken="short-body")), (), False),
        ("429-then-not-json", (Reply("x", status=429), Reply(raw=b"not json")), (), False),
        ("tags-500-once", (GOOD,), (BUSY,), True),
        ("tags-500-twice", (GOOD,), (BUSY, BUSY), False),
        ("500-then-cut-then-good", (BUSY, CUT, GOOD), (), True),
    ],
)
def test_a_temporary_error_is_sent_again_once_then_waits(service, setup, case, script, tags, succeeds):
    """W9."""
    fake = service(*script, tags_reply=Script(tags, then=Reply()))
    result = outcome(
        lambda: writer_for(fake, setup.rules, retry_pause=0.2 if case == "500-then-good" else 0).write(
            SRC, META, MODEL, setup.checkpoint
        )
    )
    ended = time.perf_counter()
    if succeeds:
        assert result == ("ok", GOOD_RESULT)
    else:
        kind, exc = result
        assert kind == "blocked"
        assert str(exc) == NOTES_BUSY_MESSAGE
        assert "重启 Ollama" in str(exc) and "继续整理笔记" in str(exc)
        assert isinstance(exc.__cause__, OllamaBusy)
        assert not setup.checkpoint.exists()
    if case == "500-then-good":
        first, second = [entry for entry in fake.log if entry[2] is not None]
        assert first[2] == second[2]
        assert second[0] - first[0] >= 0.2
    if case == "500-then-cut-then-good":
        # the resend is not an attempt: the cut-off reply still leaves a second attempt, asking for shorter notes
        first, again, second = fake.prompts
        assert first == again == LITERAL_PROMPT
        assert "精简" in supplement(first, second)
    if case.startswith("tags"):
        assert [path for path, _ in sent(fake)].count("/api/tags") == 2
    if case == "tags-500-twice":
        assert generations(fake) == [] and unloads(fake) == []
    else:
        assert_released_last(fake, ended)


# ---------------------------------------------------------------- W10


def test_a_refused_request_fails_without_trying_again(service, setup):
    """W10."""
    fake = service(REFUSED)
    with pytest.raises(NotesFailed) as caught:
        writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    ended = time.perf_counter()
    assert str(caught.value) == NOTES_REFUSED
    cause = caught.value.__cause__
    assert isinstance(cause, OllamaError)
    assert not isinstance(cause, (OllamaDown, OllamaTimeout, OllamaModelMissing, OllamaBusy, OllamaIncomplete))
    assert len(generations(fake)) == 1
    assert_released_last(fake, ended)
    assert not setup.checkpoint.exists()


# ---------------------------------------------------------------- W11


def test_a_cut_off_reply_is_asked_again_for_shorter_notes(service, setup):
    """W11(a)."""
    fake = service(CUT, GOOD)
    assert writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    first, second = fake.prompts
    assert first == LITERAL_PROMPT
    assert "精简" in supplement(first, second)


@pytest.mark.parametrize("cut", [CUT, NOT_DONE], ids=["length", "not-done"])
def test_cut_off_twice_fails(service, setup, cut):
    """W11(b)(c)."""
    fake = service(cut, cut)
    with pytest.raises(NotesFailed) as caught:
        writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    ended = time.perf_counter()
    assert str(caught.value) == NOTES_CUT_OFF
    assert "截断" in str(caught.value) and "重试" in str(caught.value)
    assert len(generations(fake)) == 2
    assert_released_last(fake, ended)
    assert not setup.checkpoint.exists()


# ---------------------------------------------------------------- W12


@pytest.mark.parametrize("empty", ["", "<think>想了很多</think>"], ids=["empty", "only-thinking"])
def test_an_empty_reply_is_asked_again_as_it_was(service, setup, empty):
    """W12(a)(c)."""
    fake = service(empty, GOOD)
    assert writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    first, second = fake.prompts
    assert first == second == LITERAL_PROMPT


@pytest.mark.parametrize("empty", EMPTY_REPLIES)
def test_empty_twice_fails(service, setup, empty):
    """W12(b)."""
    fake = service(empty, empty)
    with pytest.raises(NotesFailed) as caught:
        writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    assert str(caught.value) == NOTES_EMPTY_REPLY
    assert len(generations(fake)) == 2
    assert not setup.checkpoint.exists()


# ---------------------------------------------------------------- W13


def test_a_reply_with_nothing_backed_by_the_source_is_asked_again_word_for_word(service, setup):
    """W13(a)."""
    fake = service(MADE_UP_QUOTE, GOOD)
    assert writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    first, second = fake.prompts
    assert "逐字" in supplement(first, second)


@pytest.mark.parametrize(
    ("script", "expected"),
    [(reply, NOTES_ALL_REMOVED) for reply in [(r, r) for r in REMOVED_REPLIES]]
    + [
        (("", MADE_UP_TAG), NOTES_ALL_REMOVED),
        ((MADE_UP_TAG, ""), NOTES_EMPTY_REPLY),
        ((CUT, MADE_UP_QUOTE), NOTES_ALL_REMOVED),
    ],
)
def test_the_last_attempt_decides_the_message(service, setup, script, expected):
    """W13(b)-(e)."""
    fake = service(*script)
    with pytest.raises(NotesFailed) as caught:
        writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    assert str(caught.value) == expected
    assert len(generations(fake)) == 2
    assert not setup.checkpoint.exists()
    assert "对不上原文" in NOTES_ALL_REMOVED and "对不上原文" not in NOTES_EMPTY_REPLY


# ---------------------------------------------------------------- W14


def test_a_matching_checkpoint_is_used_without_the_service(service, setup):
    """W14."""
    fake = service(GOOD)
    first = writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    assert first == GOOD_RESULT
    data, modified = setup.checkpoint.read_bytes(), setup.checkpoint.stat().st_mtime_ns
    log_length = len(fake.log)

    second = writer_for(_closed_port_url(), setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    third = writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint)
    other = NotesMeta("另一个名字", date(2027, 1, 2))
    fourth = writer_for(fake, setup.rules).write(SRC, other, MODEL, setup.checkpoint)

    assert second == third == GOOD_RESULT
    assert fourth.markdown.startswith(header(other))
    assert fourth.markdown.split("\n")[5:] == FINISHED.split("\n")[5:]
    assert len(fake.log) == log_length
    assert setup.checkpoint.read_bytes() == data
    assert setup.checkpoint.stat().st_mtime_ns == modified


# ---------------------------------------------------------------- W15


def _dump(value) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


_SPOILED = {
    "bad-json": lambda good: b"{",
    "a-list": lambda good: b"[]",
    "reply-a-number": lambda good: _dump({**good, "reply": 123}),
    "no-reply": lambda good: _dump({k: v for k, v in good.items() if k != "reply"}),
    "version-2": lambda good: _dump({**good, "version": 2}),
    "other-fingerprint": lambda good: _dump({**good, "fingerprint": "x"}),
    "gbk": lambda good: json.dumps(good, ensure_ascii=False).encode("gbk"),
    "empty-file": lambda good: b"",
    "empty-reply": lambda good: _dump({**good, "reply": ""}),
    "made-up-reply": lambda good: _dump({**good, "reply": MADE_UP_QUOTE}),
}


@pytest.mark.parametrize("case", [*_SPOILED, "held", "a-folder"])
def test_a_checkpoint_that_cannot_be_used_is_as_good_as_none(service, setup, monkeypatch, caplog, case):
    """W15."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    fake = service(GOOD, GOOD)
    writer = writer_for(fake, setup.rules)
    assert writer.write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    good = json.loads(setup.checkpoint.read_text("utf-8"))
    held = None
    if case == "a-folder":
        setup.checkpoint.unlink()
        setup.checkpoint.mkdir()
    elif case == "held":
        monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
        held = _held(monkeypatch, setup.checkpoint)
    else:
        setup.checkpoint.write_bytes(_SPOILED[case](good))
    prompts_before = len(fake.prompts)

    started = time.perf_counter()
    assert writer.write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    elapsed = time.perf_counter() - started
    assert len(fake.prompts) - prompts_before == 1
    if held is not None:
        held.release()
    if case == "a-folder":
        assert setup.checkpoint.is_dir() and list(setup.checkpoint.iterdir()) == []
        assert len(records(caplog, logging.WARNING)) == 1
        assert elapsed < 5
    else:
        assert json.loads(setup.checkpoint.read_text("utf-8")) == good
        assert good["reply"] == FINISH_REPLY.strip()
    assert temp_files(setup.checkpoint.parent) == []


# ---------------------------------------------------------------- W16


def test_only_a_real_change_asks_the_model_again(service, setup, tmp_path):
    """W16."""
    rng = random.Random(20270204)
    fake = service(models=("model-a", "model-b"))
    client = client_for(fake)
    long_groups = gbk_groups = 0
    for group in range(40):
        while True:
            rules_text = None
            while rules_text is None:
                _, rules_text = make_rules_text(rng)
            lesson = _make_lesson(rng)
            if overflow(rules_text, lesson.expected_body(), NOTES_NUM_CTX, NOTES_REPLY_BUDGET) == 0:
                break
        long_groups += _LONG_PARAGRAPH in lesson.body
        path = tmp_path / f"规则{group}.txt"
        checkpoint = tmp_path / "job" / f"{group}.notes.json"
        utf8 = rules_text.encode("utf-8")
        path.write_bytes(utf8)
        assert load_notes_rules(path, shipped=setup.shipped).text == rules_text, ascii(rules_text)
        context = (group, ascii(rules_text[:100]), ascii(lesson.text()[:200]))

        def organise(source=lesson.text(), meta=META, model=MODEL, rules_bytes=utf8, **settings) -> int:
            path.write_bytes(rules_bytes)
            before = len(fake.prompts)
            writer = writer_with(client, lambda: load_notes_rules(path, shipped=setup.shipped), **settings)
            result = writer.write(source, meta, model, checkpoint)
            assert result.markdown == finish_notes(GOOD, source, meta).markdown, context
            return len(fake.prompts) - before

        assert organise() == 1, context
        for variant in _lesson_variants(lesson, rng):
            assert organise(source=variant.text()) == 0, (context, ascii(variant.text()[:200]))
        try:
            gbk = rules_text.encode("gb2312")
        except UnicodeEncodeError:
            gbk = None
        if gbk is not None and decode_user_text(gbk) == rules_text:
            gbk_groups += 1
            assert organise(rules_bytes=gbk) == 0, context
        assert organise(rules_bytes=codecs.BOM_UTF16_LE + rules_text.encode("utf-16-le")) == 0, context
        commented = "// 说明\r\n" + rules_text.replace("\n", "\r\n") + "\r\n  // 另一行说明\r\n"
        assert organise(rules_bytes=commented.encode("utf-8")) == 0, context
        assert organise(meta=NotesMeta("另一个名字", date(2027, 1, 2))) == 0, context
        assert organise(reply_budget=4096) == 0, context
        assert organise(timeout=60) == 0, context
        assert organise(retry_pause=1) == 0, context

        body = list(lesson.body)
        last_z = replace(lesson, body=(*body[:-1], body[-1] + "Z"))
        comma = next((i for i, line in enumerate(body) if "，" in line), None)
        if comma is None:
            body[0] += "，"
        else:
            body[comma] = body[comma].replace("，", ",", 1)
        changed_comma = replace(lesson, body=tuple(body))
        for change in (
            {"source": last_z.text()},
            {"source": changed_comma.text()},
            {"rules_bytes": (rules_text + "Z").encode("utf-8")},
            {"model": "model-b"},
            {"num_ctx": 32769},
        ):
            assert organise(**change) == 1, (context, change.keys())
            assert organise() == 1, (context, change.keys())
    assert long_groups >= 2
    assert gbk_groups >= 1


# ---------------------------------------------------------------- W17


def test_a_full_disk_still_gives_the_notes(service, setup, monkeypatch, caplog):
    """W17(a)."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    fake = service(GOOD, GOOD)

    def full(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("coursescribe.files.os.fsync", full)
    writer = writer_for(fake, setup.rules)
    assert writer.write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    assert not setup.checkpoint.exists()
    assert temp_files(setup.checkpoint.parent) == []
    assert len(records(caplog, logging.WARNING)) == 1

    monkeypatch.undo()
    assert writer.write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    assert len(fake.prompts) == 2
    assert setup.checkpoint.exists()


def test_a_file_in_the_way_of_the_checkpoint_folder_still_gives_the_notes(service, tmp_path, caplog):
    """W17(b)."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    blocker = tmp_path / "挡着"
    blocker.write_bytes(b"x")
    rules = tmp_path / "整理规则.txt"
    rules.write_bytes(RULES_FILE)
    fake = service(GOOD)
    writer = writer_for(fake, lambda: load_notes_rules(rules, shipped=rules))
    assert writer.write(SRC, META, MODEL, blocker / "job.notes.json") == GOOD_RESULT
    assert blocker.read_bytes() == b"x"
    assert len(records(caplog, logging.WARNING)) == 1


# ---------------------------------------------------------------- W18

_RELEASED = {"success", "cut-twice", "timeout", "500-twice", "400", "404", "unload-500", "unload-slow", "unload-gone"}


@pytest.mark.parametrize(
    "case",
    [
        "success", "cut-twice", "timeout", "500-twice", "400", "404", "checkpoint", "empty-source",
        "too-long", "not-listed", "tags-500-twice", "unload-500", "unload-slow", "unload-gone",
        "gone", "show-500-twice",
    ],
)
def test_the_model_is_released_once_whenever_it_was_asked(service, setup, caplog, case):
    """W18(a)-(n)."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    settings: dict = {}
    source = SRC
    script: tuple = (GOOD,)
    kwargs: dict = {}
    expected: tuple = ("ok", GOOD_RESULT)
    if case == "cut-twice":
        script, expected = (CUT, CUT), ("failed", NOTES_CUT_OFF)
    elif case == "timeout":
        script, expected = (Reply(GOOD, delay=1.0),), ("blocked", NOTES_TIMEOUT_MESSAGE)
        settings["timeout"] = 0.3
    elif case == "500-twice":
        script, expected = (BUSY, BUSY), ("blocked", NOTES_BUSY_MESSAGE)
    elif case == "400":
        script, expected = (REFUSED,), ("failed", NOTES_REFUSED)
    elif case == "404":
        script, expected = (MISSING,), ("blocked", NOTES_MODEL_MISSING.format(model=MODEL))
    elif case == "empty-source":
        source, expected = "# 课", ("failed", NOTES_EMPTY_SOURCE)
    elif case == "too-long":
        settings["num_ctx"] = 8000
        expected = ("failed", NOTES_RULES_TOO_LONG.format(rules=len(RULES)))
    elif case == "not-listed":
        kwargs["models"] = ("other",)
        expected = ("blocked", NOTES_MODEL_MISSING.format(model=MODEL))
    elif case == "tags-500-twice":
        kwargs["tags_reply"] = Script((BUSY, BUSY), then=Reply())
        expected = ("blocked", NOTES_BUSY_MESSAGE)
    elif case == "unload-500":
        kwargs["unload_reply"] = lambda: BUSY
    elif case == "unload-slow":
        kwargs["unload_reply"] = lambda: Reply(delay=5)
        settings["unload_timeout"] = 0.3
    elif case == "unload-gone":
        kwargs["unload_reply"] = lambda: Reply(delay=5, close_service=True)
    elif case == "gone":  # the service is confirmed gone: it holds no model, nothing to release
        script, expected = (GONE,), ("blocked", NOTES_DOWN)
    elif case == "show-500-twice":  # no generate request was ever sent: nothing was loaded
        kwargs["show_reply"] = Script((BUSY, BUSY), then=Reply())
        expected = ("blocked", NOTES_BUSY_MESSAGE)
    fake = service(*script, **kwargs)
    if case == "checkpoint":
        assert writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
        reset(fake)
    caplog.clear()

    started = time.perf_counter()
    kind, result = outcome(lambda: writer_for(fake, setup.rules, **settings).write(source, META, MODEL, setup.checkpoint))
    ended = time.perf_counter()

    assert (kind, result if kind == "ok" else str(result)) == expected
    if case in _RELEASED:
        assert_released_last(fake, ended)
    else:
        assert unloads(fake) == []
    if case.startswith("unload"):
        assert len(records(caplog, logging.INFO, "unload")) == 1
    if case not in _RELEASED:  # not even tried
        assert records(caplog, logging.INFO, "unload") == []
    if case == "show-500-twice":
        assert generations(fake) == []
    if case == "unload-slow":
        assert ended - started < 1.5


def test_release_swallows_a_service_that_is_not_running(caplog):
    """W18(o)."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    writer = writer_for(_closed_port_url(), lambda: NotesRules(RULES, None))
    assert writer.release(MODEL) is None
    assert len(records(caplog, logging.INFO, "unload")) == 1


# ---------------------------------------------------------------- W19

MARK = "4721"
MARK_SOURCE = "# 课\n\n今天的暗号是蓝鲸4721号。索引让查询变快。"
MARK_REPLY = "## 核心宗旨\n蓝鲸4721号\n\n## 引述\n> 蓝鲸4721号是编的吗"
MARK_EMPTY = "## 本课脉络\n- [[#蓝鲸4721号]]"
MARK_MADE_UP = "## 引述\n> 蓝鲸4721号是编的吗"


def mark_rules() -> NotesRules:
    return NotesRules("## 核心宗旨\n蓝鲸4721号", None)


def test_logs_and_messages_never_carry_the_lesson(service, tmp_path, monkeypatch, caplog):
    """W19."""
    caplog.set_level(logging.DEBUG)
    raised: list[Exception] = []
    count = iter(range(1000))

    def run(fake_or_url, *, checkpoint=None, **settings):
        checkpoint = checkpoint or tmp_path / f"{next(count)}.notes.json"
        kind, result = outcome(
            lambda: writer_for(fake_or_url, mark_rules, **settings).write(MARK_SOURCE, META, MODEL, checkpoint)
        )
        if kind != "ok":
            raised.append(result)
        return kind, result

    # W1 twice: the second is a replay from the checkpoint.
    fake = service(MARK_REPLY)
    checkpoint = tmp_path / "w1.notes.json"
    assert run(fake, checkpoint=checkpoint)[0] == "ok"
    first_log = caplog.text
    caplog.clear()
    assert run(_closed_port_url(), checkpoint=checkpoint)[0] == "ok"
    assert any("checkpoint" in r.getMessage().lower() for r in caplog.records if r.name == LOGGER)
    replay_log = caplog.text
    caplog.clear()
    good_checkpoint = json.loads(checkpoint.read_text("utf-8"))

    texts = [first_log, replay_log]
    scenarios = [
        # one character too many for the lesson to fit
        lambda: run(
            service(MARK_REPLY),
            num_ctx=len(mark_rules().text) + len(source_body(MARK_SOURCE)) + 50 + 256 - 1,
            reply_budget=50,
        ),
        lambda: run(service(GONE)),
        lambda: run(service(Reply(MARK_REPLY, delay=1.0)), timeout=0.3),
        lambda: run(service(BUSY, BUSY)),
        lambda: run(service(REFUSED)),
        lambda: run(service(*[Reply(MARK_REPLY, fields={"done_reason": "length"})] * 2)),
        lambda: run(service(MARK_EMPTY, MARK_EMPTY)),
        lambda: run(service(MARK_MADE_UP, MARK_MADE_UP)),
        lambda: run(service(MARK_REPLY, unload_reply=lambda: BUSY)),
    ]
    for scenario in scenarios:
        scenario()
        texts.append(caplog.text)
        caplog.clear()
    # W15(a)(j): a broken checkpoint, and one whose saved reply has nothing backed by the source.
    for spoiled in (b"{", _dump({**good_checkpoint, "reply": MARK_MADE_UP})):
        path = tmp_path / f"spoiled-{len(texts)}.notes.json"
        path.write_bytes(spoiled)
        assert run(service(MARK_REPLY), checkpoint=path)[0] == "ok"
        texts.append(caplog.text)
        caplog.clear()
    # W17(a): the checkpoint cannot be saved.
    def full(fd):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr("coursescribe.files.os.fsync", full)
        assert run(service(MARK_REPLY))[0] == "ok"
    texts.append(caplog.text)

    assert len(raised) == 8
    for text in texts:
        assert MARK not in text
    for exc in raised:
        assert MARK not in str(exc)


# ---------------------------------------------------------------- W20


@pytest.mark.parametrize(
    ("changes", "error", "word"),
    [
        ({"source": None}, TypeError, "source"),
        ({"source": b"x"}, TypeError, "source"),
        ({"source": Path("课.md")}, TypeError, "source"),
        ({"meta": None}, TypeError, "meta"),
        ({"meta": ("课", date(2026, 1, 1))}, TypeError, "meta"),
        ({"model": None}, TypeError, "model"),
        ({"model": ""}, ValueError, "model"),
        ({"model": "  "}, ValueError, "model"),
        ({"checkpoint": "job.json"}, TypeError, "checkpoint"),
    ],
)
def test_wrong_arguments_are_refused_before_anything_is_read(service, tmp_path, changes, error, word):
    """W20(a)-(d)."""
    fake = service(GOOD)
    folder = tmp_path / "整理规则.txt"
    folder.mkdir()
    before = sorted(tmp_path.rglob("*"))
    writer = writer_for(fake, lambda: load_notes_rules(folder, shipped=folder))
    arguments = {"source": SRC, "meta": META, "model": MODEL, "checkpoint": tmp_path / "job.notes.json"} | changes
    with pytest.raises(error) as caught:
        writer.write(**arguments)
    assert word in str(caught.value)
    assert not isinstance(caught.value, (NotesFailed, NotesBlocked))
    assert fake.log == []
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_ctx", 0), ("num_ctx", True), ("num_ctx", 1.5), ("reply_budget", 0), ("reply_budget", -1),
        ("timeout", 0), ("timeout", -1), ("timeout", math.inf), ("timeout", math.nan), ("retry_pause", -1),
    ],
)
def test_wrong_settings_are_refused_when_the_writer_is_made(name, value):
    """W20(e)."""
    with pytest.raises(ValueError, match=name):
        OllamaNotesWriter(OllamaClient(DEFAULT_URL), lambda: NotesRules(RULES, None), **{name: value})


def test_waiting_and_failing_are_separate_and_not_service_errors():
    """W20(f)."""
    for kind in (NotesBlocked, NotesFailed):
        assert not issubclass(kind, OllamaError)
    assert not issubclass(NotesBlocked, NotesFailed)
    assert not issubclass(NotesFailed, NotesBlocked)


# ---------------------------------------------------------------- W21


def test_a_long_lesson_and_a_very_long_reply(service, setup):
    """W21."""
    source = "# 课\n\n" + "索引就像一本书的目录，能让查询变快。\n" * 1200
    reply = "## 核心宗旨\n" + "索引就像一本书的目录。\n" * 20000
    fake = service(reply)
    started = time.perf_counter()
    result = writer_for(fake, setup.rules).write(source, META, MODEL, setup.checkpoint)
    elapsed = time.perf_counter() - started
    assert result.markdown.split("\n").count("索引就像一本书的目录。") == 20000
    assert fake.prompts == [build_prompt(RULES, source)]
    assert len(json.loads(setup.checkpoint.read_text("utf-8"))["reply"]) == len(reply.strip())
    assert elapsed < 5


# ---------------------------------------------------------------- W22

_CHILD = r"""
import os
import sys
from datetime import date
from pathlib import Path

from coursescribe.notes import NotesMeta
from coursescribe.notes_rules import load_notes_rules
from coursescribe.notes_writer import OllamaNotesWriter
from coursescribe.ollama import OllamaClient

url, rules, shipped, checkpoint, moment, source = sys.argv[1:]
if moment == "temp-written":
    real_fsync = os.fsync

    def fsync(fd):
        real_fsync(fd)
        os._exit(3)

    os.fsync = fsync
elif moment == "replaced":
    real_replace = os.replace

    def replace_(a, b):
        real_replace(a, b)
        os._exit(3)

    os.replace = replace_
writer = OllamaNotesWriter(
    OllamaClient(url), rules=lambda: load_notes_rules(Path(rules), shipped=Path(shipped)), retry_pause=0
)
meta = NotesMeta("2026.10.30 数据库索引", date(2026, 10, 30))
writer.write(Path(source).read_bytes().decode("utf-8"), meta, "model-a", Path(checkpoint))
"""


@pytest.mark.parametrize("moment", ["generating", "temp-written", "replaced"])
def test_killed_at_any_step_the_next_run_gives_the_same_notes(service, setup, tmp_path, moment):
    """W22."""
    assert META.name == "2026.10.30 数据库索引"
    delay = {"seconds": 10 if moment == "generating" else 0}
    fake = service()
    fake.respond = lambda prompt: Reply(GOOD, delay=delay["seconds"])
    source = tmp_path / "source.md"
    source.write_bytes(SRC.encode("utf-8"))
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, fake.url, str(setup.rules_path), str(setup.shipped),
         str(setup.checkpoint), moment, str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        if moment == "generating":
            deadline = time.monotonic() + 30
            while not fake.prompts and time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.02)
            assert fake.prompts, proc.communicate()
            proc.kill()
            proc.communicate(timeout=30)
        else:
            _, err = proc.communicate(timeout=60)
            assert proc.returncode == 3, err.decode("utf-8", "replace")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    delay["seconds"] = 0
    before = len(fake.prompts)

    assert writer_for(fake, setup.rules).write(SRC, META, MODEL, setup.checkpoint) == GOOD_RESULT
    assert len(fake.prompts) - before == (0 if moment == "replaced" else 1)
    others = [
        p.name for p in setup.checkpoint.parent.iterdir()
        if p.name != "job.notes.json" and not re.fullmatch(r"\.coursescribe-.*\.tmp", p.name)
    ]
    assert others == []


# ---------------------------------------------------------------- M1


def _placeholders(template: str) -> list[str]:
    return [field for _, field, _, _ in string.Formatter().parse(template) if field is not None]


def _pattern(template: str) -> re.Pattern:
    out = ""
    for literal, field, _, _ in string.Formatter().parse(template):
        out += re.escape(literal)
        if field is not None:
            out += r"\d+" if field in ("chars", "limit", "rules") else r".+"
    return re.compile(out)


def _check_message_texts() -> None:
    """M1(a)."""
    for template in WAITING:
        message = template.format(model="m")
        assert "校对稿已保存" in message and "继续整理笔记" in message, template
        assert "重试" not in message, template
    for template in FAILING:
        message = template.format(chars=100, limit=99, rules=2)
        assert "继续整理笔记" not in message, template
        assert ("重试" in message) == (template is not NOTES_TOO_LONG), template
        assert ("校对稿没有变" in message) == (template is not NOTES_EMPTY_SOURCE), template
    for template in WAITING + FAILING:
        bare = re.sub(r"\{\w+\}", "", template)
        assert set(re.findall(r"[A-Za-z]+", bare)) <= {"Ollama", "ollama", "pull"}, template
        assert "12-" not in template and "Notes" not in template
    assert _placeholders(NOTES_MODEL_MISSING) == ["model", "model"]
    assert "ollama pull {model}" in NOTES_MODEL_MISSING
    assert sorted(_placeholders(NOTES_TOO_LONG)) == ["chars", "limit"]
    assert _placeholders(NOTES_RULES_TOO_LONG) == ["rules"]
    for template in WAITING + FAILING:
        if template not in (NOTES_MODEL_MISSING, NOTES_TOO_LONG, NOTES_RULES_TOO_LONG):
            assert _placeholders(template) == [], template
    assert "显存" in NOTES_TIMEOUT_MESSAGE and "显存" in NOTES_BUSY_MESSAGE
    # The two attempts may have failed in different ways: the message names the last one only.
    for template in (NOTES_CUT_OFF, NOTES_EMPTY_REPLY, NOTES_ALL_REMOVED):
        assert "最后一次" in template, template
        assert "两次都这样" not in template and "两次都没有写出" not in template, template


def test_the_messages_say_what_to_do_and_are_the_ones_raised(service, setup, tmp_path):
    """M1(a)(b)."""
    _check_message_texts()
    raised: list[Exception] = []
    count = iter(range(1000))

    def run(target, source=SRC, rules=setup.rules, model=MODEL, **settings):
        kind, result = outcome(
            lambda: writer_for(target, rules, **settings).write(
                source, META, model, tmp_path / f"{next(count)}.notes.json"
            )
        )
        assert kind != "ok"
        raised.append(result)

    broken = tmp_path / "坏规则.txt"
    broken.write_bytes("café".encode("latin-1"))
    run(service(), rules=lambda: load_notes_rules(broken, shipped=setup.shipped))
    run(service(), rules=lambda: load_notes_rules(tmp_path, shipped=setup.shipped))
    run(service(), rules=lambda: load_notes_rules(tmp_path / "没有.txt", shipped=tmp_path / "也没有.txt"))
    run(service(), source="# 课")
    run(service(), source=LESSON_100, rules=two_char_rules, num_ctx=407, reply_budget=50)
    run(service(), source=LESSON_100, rules=two_char_rules, num_ctx=308, reply_budget=50)
    run(service(), model="model-b")
    run(service(MISSING))
    run(service(GONE))
    run(service(Reply(GOOD, delay=1.0)), timeout=0.3)
    run(service(BUSY, BUSY))
    run(service(REFUSED))
    run(service(CUT, CUT))
    run(service("", ""))
    run(service(MADE_UP_QUOTE, MADE_UP_QUOTE))

    rules_messages = (RULES_ENCODING, RULES_UNREADABLE, RULES_SHIPPED_MISSING, RULES_HELD)
    seen = set()
    for exc in raised:
        allowed = WAITING if isinstance(exc, NotesBlocked) else FAILING + rules_messages
        matches = [t for t in allowed if _pattern(t).fullmatch(str(exc))]
        assert matches, (type(exc).__name__, str(exc))
        seen.update(matches)
    assert set(WAITING + FAILING) <= seen


# ---------------------------------------------------------------- P1

_P1_TOKENS = (
    ("ok", 0.28), ("empty", 0.10), ("blank", 0.10), ("madeup", 0.10), ("cut", 0.15),
    ("busy", 0.15), ("refused", 0.04), ("gone", 0.03), ("slow", 0.03), ("down", 0.02),
)
_P1_REPLIES = {
    "ok": GOOD, "empty": "", "blank": "## 本课脉络\n- [[#一]]", "madeup": MADE_UP_QUOTE, "cut": CUT,
    "busy": BUSY, "refused": REFUSED, "gone": MISSING, "slow": Reply(GOOD, delay=0.5), "down": GONE,
}
# Each "down" round costs about 4 s on Windows (two refused connections). The matrix allows 3;
# lowered to 2 by its own rule because the unit lane went well past 60 s.
# Since no release is tried after OllamaDown it is about 2 s; the lane is still past 90 s, so 2 stays.
_P1_DOWN_ROUNDS = 2
_P1_KEPT = {"cut": NOTES_CUT_OFF, "empty": NOTES_EMPTY_REPLY, "blank": NOTES_EMPTY_REPLY, "madeup": NOTES_ALL_REMOVED}


def _p1_draw(rng: random.Random) -> str:
    roll = rng.random()
    total = 0.0
    for token, chance in _P1_TOKENS:
        total += chance
        if roll < total:
            return token
    return _P1_TOKENS[-1][0]


def _p1_reference(tokens: list[str]):
    """The stated rule, written here without the code under test.

    Returns None while it needs another token; otherwise ((kind, message), sends), where each
    send is (attempt, is a resend, what the previous attempt was).
    """
    sends: list[tuple[int, bool, str | None]] = []
    position = 0
    last = None
    for attempt in range(2):
        if position >= len(tokens):
            return None
        token = tokens[position]
        position += 1
        sends.append((attempt, False, last))
        if token == "busy":
            if position >= len(tokens):
                return None
            token = tokens[position]
            position += 1
            sends.append((attempt, True, last))
            if token == "busy":
                return ("blocked", NOTES_BUSY_MESSAGE), sends
        if token == "down":
            return ("blocked", NOTES_DOWN), sends
        if token == "slow":
            return ("blocked", NOTES_TIMEOUT_MESSAGE), sends
        if token == "gone":
            return ("blocked", NOTES_MODEL_MISSING.format(model=MODEL)), sends
        if token == "refused":
            return ("failed", NOTES_REFUSED), sends
        if token == "ok":
            return ("ok", None), sends
        last = token
    return ("failed", _P1_KEPT[last]), sends


def _p1_script(rng: random.Random, downs_left: int):
    tokens: list[str] = []
    while True:
        token = _p1_draw(rng)
        while token == "down" and (downs_left <= 0 or (tokens and tokens[-1] == "busy")):
            token = _p1_draw(rng)
        tokens.append(token)
        reference = _p1_reference(tokens)
        if reference is not None:
            return tokens, reference


def test_any_run_of_service_behaviour_ends_as_the_reference_rule_says(service, setup, tmp_path):
    """P1."""
    rng = random.Random(20270201)
    fake = service(GOOD)
    client = client_for(fake)
    nowhere = OllamaClient(_closed_port_url())
    other = tmp_path / "other.notes.json"
    assert writer_for(fake, setup.rules).write(SRC + "\n多出来的一句话。\n", META, MODEL, other).markdown
    other_bytes = other.read_bytes()
    folder = tmp_path / "p1"
    folder.mkdir()
    seen: Counter = Counter()
    downs = 0
    for round_ in range(300):
        tokens, ((kind, message), sends) = _p1_script(rng, _P1_DOWN_ROUNDS - downs)
        has_down = "down" in tokens
        downs += has_down
        start = rng.choice(("none", "other", "broken"))
        checkpoint = folder / f"{round_}.notes.json"
        if start == "other":
            checkpoint.write_bytes(other_bytes)
        elif start == "broken":
            checkpoint.write_bytes(b"{")
        before = checkpoint.read_bytes() if checkpoint.exists() else None
        if fake.closed:
            fake = service()
            client = client_for(fake)
        reset(fake, *[_P1_REPLIES[t] for t in tokens])
        # A refused connection takes about 2 s on Windows, so a shorter timeout would turn "down" into a timeout.
        timeout = 0.15 if "slow" in tokens else 5.0
        writer = writer_with(client, setup.rules, timeout=timeout)
        context = (round_, tokens, start)

        got, result = outcome(lambda: writer.write(SRC, META, MODEL, checkpoint))
        ended = time.perf_counter()

        if kind == "ok":
            assert (got, result) == ("ok", GOOD_RESULT), context
            saved = json.loads(checkpoint.read_text("utf-8"))
            assert saved["version"] == 1 and saved["reply"] == FINISH_REPLY.strip(), context
            replay = writer_with(nowhere, setup.rules).write(SRC, META, MODEL, checkpoint)
            assert replay == GOOD_RESULT, context
            seen["ok"] += 1
            seen["ok after busy"] += "busy" in tokens and tokens[-1] == "ok" and tokens[-2] == "busy"
        else:
            assert (got, str(result)) == (kind, message), context
            if message == NOTES_DOWN:
                assert isinstance(result.__cause__, OllamaDown), context
            assert (checkpoint.read_bytes() if checkpoint.exists() else None) == before, context
            seen[message] += 1

        prompts = list(fake.prompts)
        assert len(prompts) == len(sends) <= 4, context
        for index, (attempt, resend, previous) in enumerate(sends):
            if resend:
                assert prompts[index] == prompts[index - 1], context
            elif attempt == 0:
                assert prompts[index] == LITERAL_PROMPT, context
            elif previous in ("empty", "blank"):
                assert prompts[index] == LITERAL_PROMPT, context
            else:
                word = "精简" if previous == "cut" else "逐字"
                assert word in supplement(LITERAL_PROMPT, prompts[index]), context
        if has_down:
            assert unloads(fake) == [], context
        elif sends:
            assert_released_last(fake, ended)
        else:
            assert unloads(fake) == [], context

    assert seen["ok"] >= 90, seen
    assert seen[NOTES_CUT_OFF] >= 8, seen
    assert seen[NOTES_EMPTY_REPLY] >= 8, seen
    assert seen[NOTES_ALL_REMOVED] >= 8, seen
    assert seen[NOTES_BUSY_MESSAGE] >= 5, seen
    assert seen[NOTES_TIMEOUT_MESSAGE] >= 3, seen
    assert seen[NOTES_MODEL_MISSING.format(model=MODEL)] >= 3, seen
    assert seen[NOTES_REFUSED] >= 3, seen
    assert seen[NOTES_DOWN] >= _P1_DOWN_ROUNDS - 1, seen
    assert seen["ok after busy"] >= 10, seen


# ---------------------------------------------------------------- P2


def test_the_service_is_asked_exactly_when_the_lesson_fits(service, tmp_path):
    """P2."""
    rng = random.Random(20270203)
    fake = service()
    client = client_for(fake)
    seen: Counter = Counter()
    for round_ in range(300):
        rules_text = None
        while rules_text is None:
            _, rules_text = make_rules_text(rng)
        lesson = _make_lesson(rng)
        body = lesson.expected_body()
        budget = rng.randint(1, 300)
        edge = len(rules_text) + len(body) + budget + 256
        roll = rng.random()
        if roll < 0.6:
            num_ctx = edge + rng.randint(-250, 250)
        else:
            num_ctx = (edge, edge - 1, edge - len(body), edge - len(body) + 1)[min(3, int((roll - 0.6) / 0.1))]
        num_ctx = max(1, num_ctx)
        over = max(0, edge - num_ctx)
        limit = len(body) - over
        context = (round_, num_ctx, budget, over, ascii(rules_text[:80]), ascii(lesson.text()[:120]))

        reset(fake, CUT, GOOD)
        writer = writer_with(
            client, lambda text=rules_text: NotesRules(text, None), num_ctx=num_ctx, reply_budget=budget
        )
        got, result = outcome(lambda: writer.write(lesson.text(), META, MODEL, tmp_path / f"{round_}.notes.json"))

        assert (over == 0) == bool(fake.prompts), context
        for request in generations(fake):
            assert len(request["prompt"]) + budget + 128 <= num_ctx, context
            assert request["options"]["num_ctx"] == num_ctx and request["options"]["num_predict"] == budget
        if over == 0:
            assert fake.prompts[0] == build_prompt(rules_text, lesson.text()), context
            assert got == "ok", context
            seen["fits"] += 1
        else:
            assert fake.log == [], context
            assert got == "failed", context
            if limit >= 1:
                assert str(result) == NOTES_TOO_LONG.format(chars=len(body), limit=limit), context
                seen["too long"] += 1
            else:
                assert str(result) == NOTES_RULES_TOO_LONG.format(rules=len(rules_text)), context
                seen["rules too long"] += 1
            assert not re.search(r"-\d", str(result)), context
        seen["limit 1"] += over > 0 and limit == 1
        seen["over 1"] += over == 1
    assert seen["fits"] >= 60, seen
    assert seen["too long"] >= 60, seen
    assert seen["rules too long"] >= 20, seen
    assert seen["limit 1"] >= 5, seen
    assert seen["over 1"] >= 5, seen


# ---------------------------------------------------------------- P3


def test_the_notes_are_exactly_what_finish_notes_makes_of_the_reply(service, setup, tmp_path):
    """P3."""
    rng = random.Random(20270205)
    fake = service()
    client = client_for(fake)
    seen: Counter = Counter()
    for round_ in range(150):
        reply, _ = make_reply(rng)
        stripped = reply.strip()
        if stripped == "":
            expected = ("failed", NOTES_EMPTY_REPLY)
        else:
            try:
                structure_notes(stripped, META)
            except EmptyNotes:
                expected = ("failed", NOTES_EMPTY_REPLY)
            else:
                try:
                    finished = finish_notes(stripped, SRC, META)
                except EmptyNotes:
                    expected = ("failed", NOTES_ALL_REMOVED)
                else:
                    expected = ("ok", GeneratedNotes(
                        finished.markdown, finished.removed_quotes, finished.removed_concepts, None
                    ))
        reset(fake, reply, reply)
        got, result = outcome(
            lambda: writer_with(client, setup.rules).write(SRC, META, MODEL, tmp_path / f"{round_}.notes.json")
        )
        assert (got, result if got == "ok" else str(result)) == expected, (round_, ascii(reply[:300]))
        assert len(fake.prompts) == (1 if got == "ok" else 2), round_
        seen[expected[0] if got == "ok" else expected[1]] += 1
    assert seen["ok"] >= 100, seen
    assert seen[NOTES_EMPTY_REPLY] >= 5, seen


# ---------------------------------------------------------------- integration: the real Ollama

# A made-up lesson written for these tests; nothing here comes from a real class.
LESSON_PARAGRAPHS = (
    "大家好，今天这节课我们来讲数据库索引的入门知识。很多同学写过增删改查的代码，但是不太清楚为什么同样一条查询，"
    "有时候一眨眼就返回，有时候要等上好几秒。其实答案往往就藏在索引里。这节课我会先用一个生活中的例子帮大家建立直觉，"
    "再一起拆开看看索引在磁盘上到底是怎么组织起来的，知道它为什么能这么快，又为什么会让写入变慢，"
    "然后看一个真实的订单表案例，最后讲讲怎么判断一个索引到底有没有被用上。",
    "先说一个图书馆找书的例子。假设图书馆里有十万本书，全都随便堆在地上，你要找一本讲天文的书，只能从第一本开始"
    "一本一本地看书名，运气不好要翻遍整个图书馆。可是真实的图书馆不是这样的，它有一套分类编号，书架上按编号排好，"
    "同一类的书都放在相邻的几排书架上，每一排的侧面还贴着编号范围，管理员把新书上架的时候也要按编号插到正确的位置，"
    "前台还有一个检索目录。你先查目录，知道书在哪个区、哪一排，再直接走过去拿，几分钟就找到了。",
    "数据库里的索引，就相当于图书馆的检索目录。没有索引的时候，数据库要做的事情叫全表扫描，也就是把表里的每一行都读出来，"
    "不管有没有用都要读，"
    "挨个比较条件是不是满足。表小的时候感觉不到，表一旦有几百万行，全表扫描就会非常慢。有了索引，"
    "就不用再把整张表从头到尾读一遍了，就像你不会为了找一本书把整个图书馆的书都搬出来看一遍，"
    "数据库可以先在索引里定位到满足条件的那几行，再回到表里把它们取出来。",
    "那索引在磁盘上长什么样呢？大多数关系型数据库用的是 B+ 树。你可以把 B+ 树想象成一棵很矮很胖的树，"
    "和我们平时画的二叉树不一样，"
    "每个节点里放很多个键，从根节点往下走三四层就能到达叶子节点。叶子节点里按顺序存着所有的键，"
    "每个键旁边还记着这一行数据在表里放在哪里，或者干脆就把整行数据放在叶子节点里，这取决于数据库具体怎么实现，"
    "并且相邻的叶子节点之间用指针连起来，所以既能快速定位一个值，也能方便地做范围查询。",
    "接下来看一个案例。我们有一张订单表，叫 orders，里面有订单号、用户编号、下单时间、金额和状态这几列，"
    "数据是这几年慢慢积累下来的，"
    "一共大约八百万行。业务上最常见的查询是：查某个用户最近一个月的订单，按下单时间倒序排列。"
    "这个页面是用户最常打开的页面之一，高峰的时候每秒钟要被查询好几百次，所以它慢一点，整个系统都会跟着慢下来。"
    "一开始这张表只有主键索引，这条查询平均要跑四秒多，用户在页面上点一下要转好久的圈。",
    "我们给这张表加了一个联合索引，包含用户编号和下单时间两列，用户编号在前，下单时间在后。加完以后，"
    "我们在测试环境里把同一条查询连续跑了一百次，取了中间的结果，又在线上观察了一整天，"
    "同样的查询降到了二十毫秒左右。为什么这么快？因为索引里的数据先按用户编号排序，同一个用户内部再按下单时间排序，"
    "相当于每个用户都有一份按时间排好的订单清单，"
    "数据库直接定位到这个用户，再顺着时间往回读一个月的数据就行了，根本不用碰别的用户的订单。",
    "这里就引出一个非常重要的规则，叫最左前缀原则。联合索引是按照列的先后顺序来排序的，查询条件必须从最左边的列开始用，"
    "中间也不能跳过，"
    "索引才能发挥作用。比如刚才那个索引，如果你的查询只按下单时间来筛选，不带用户编号，那这个索引基本就用不上了，"
    "这就好比一本电话簿先按姓排、同姓的再按名字排，你只知道名字却不知道姓，就只能从第一页翻到最后一页，"
    "数据库还是会退回到全表扫描或者扫描整个索引。",
    "那怎么知道一条查询到底有没有用上索引呢？要学会看执行计划。在 MySQL 里，你在查询语句前面加上 EXPLAIN 这个关键字，"
    "数据库就会告诉你它打算怎么执行这条查询：用了哪个索引，预计要扫描多少行，有没有额外的排序。养成习惯，"
    "尤其是在新功能上线之前，或者表里的数据量明显变大以后，因为数据量变了，数据库选择的执行方案也可能跟着变，"
    "每次写完一条重要的查询，都先看一眼执行计划，再决定要不要调整。",
    "讲到这里，有的同学可能会想，既然索引这么好用，那我给每一列都建一个索引不就行了吗？这里我要特别强调一句话，"
    "这句话也是很多人踩过坑以后才真正明白的，"
    "大家一定要记住：索引不是越多越好，每多一个索引，写入就多一份负担。因为每次插入、更新或者删除数据的时候，"
    "哪怕只是改了一行里的一个字段，只要这个字段在某个索引里，这个索引就得跟着调整，"
    "数据库不光要改表里的数据，还要同步维护每一个相关的索引。",
    "除了写入变慢，索引还会占用额外的存储空间，一张大表的索引有时候比数据本身还要大。另外，索引太多的时候，"
    "尤其是好几个索引都包含同一列的时候，"
    "优化器在挑选执行方案时也可能选错，反而用了一个效果不好的索引。所以建索引之前，要先想清楚这张表主要被怎么查询，"
    "是按用户查、按时间查，还是按状态查，哪些条件经常一起出现，哪些列几乎从来不会出现在查询条件里，"
    "哪些查询最频繁、最需要快，再有针对性地去建。",
    "还有几种常见的情况会让索引失效，大家写查询的时候要留意。比如在索引列上做了函数运算，像把下单时间转换成日期字符串再比较；"
    "比如对两个不同的列用 OR 把条件连起来，而其中一列根本没有索引；"
    "比如字符串列和数字做比较，导致发生隐式类型转换；再比如用了以百分号开头的模糊匹配。这些写法都会让数据库"
    "在执行的时候"
    "没办法利用索引的有序性，只好一行一行地去检查。",
    "最后我们总结一下今天的内容。索引就像图书馆的检索目录，它用额外的空间换来了查询的速度；常见的实现是 B+ 树；"
    "联合索引要遵守最左前缀原则；判断索引有没有生效要看执行计划；而且索引不是越多越好，要根据实际的查询来设计。"
    "希望大家把这几条记下来，以后在工作中遇到慢查询的时候，先回头对照一遍，大部分问题都能找到原因。"
    "下节课我们会讲事务和锁，大家回去可以先在自己的数据库里用 EXPLAIN 试一试。",
)
LESSON = "# 2027.01.15 数据库索引入门\n\n" + "\n\n".join(LESSON_PARAGRAPHS) + "\n"
LESSON_META = NotesMeta("2027.01.15 数据库索引入门", date(2027, 1, 15))
_QUOTE_ORDINAL = re.compile(r"^(?:(?:[0-9]+|[一二三四五六七八九十]+)[.、．]|[（(](?:[0-9]+|[一二三四五六七八九十]+)[）)])\s*")
_INTEGRATION_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")


def _is_quote_title(title: str) -> bool:
    text = unicodedata.normalize("NFKC", title).replace("**", "").strip()
    text = _QUOTE_ORDINAL.sub("", text).rstrip(":：").strip()
    return text == "引述"


def _quote_section_lines(markdown: str) -> list[list[str]]:
    """The lines under each 引述 heading, up to the next heading of the same or a higher level, outside code."""
    lines = markdown.split("\n")
    found = headings(markdown)
    sections = []
    for position, (index, level, title) in enumerate(found):
        if not _is_quote_title(title):
            continue
        end = next((i for i, lvl, _ in found[position + 1:] if lvl <= level), len(lines))
        kept, fence = [], False
        for line in lines[index + 1:end]:
            if _INTEGRATION_FENCE.match(line):
                fence = not fence
                continue
            if not fence:
                kept.append(line)
        sections.append(kept)
    return sections


def _quote_lines_outside_sections(markdown: str) -> list[str]:
    """`>` lines outside code that are not under a 引述 heading: 12-3 never checks them."""
    lines = markdown.split("\n")
    found = headings(markdown)
    inside: set[int] = set()
    for position, (index, level, title) in enumerate(found):
        if _is_quote_title(title):
            end = next((i for i, lvl, _ in found[position + 1:] if lvl <= level), len(lines))
            inside.update(range(index, end))
    out, fence = [], False
    for number, line in enumerate(lines):
        if _INTEGRATION_FENCE.match(line):
            fence = not fence
            continue
        if not fence and number not in inside and line.lstrip(" ").startswith(">"):
            out.append(line)
    return out


def _kept_quotes(sections: list[list[str]]) -> int:
    """Quotes kept, counted as 12-3 counts them: a run of `>` lines is one, a line with quote marks is one."""
    count = 0
    for lines in sections:
        previous = False
        for line in lines:
            quoted = line.lstrip(" ").startswith(">")
            if quoted and not previous:
                count += 1
            elif not quoted and re.search(r"[“「『\"]", line):
                count += 1
            previous = quoted
    return count


def _no_proxy_post(path: str, body: dict, timeout: float) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        DEFAULT_URL + path, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _loaded_models() -> list[str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(DEFAULT_URL + "/api/ps", timeout=10) as response:
        return [item.get("name", "") for item in json.loads(response.read().decode("utf-8")).get("models", [])]


@pytest.mark.integration
def test_real_ollama_writes_notes_for_a_made_up_lesson(tmp_path):
    """I1."""
    client = OllamaClient()
    _ready_notes_model(client)
    rules = lambda: load_notes_rules(tmp_path / "没有.txt")  # noqa: E731  the shipped rules
    checkpoint = tmp_path / "job.notes.json"
    print(f"\nlesson: {len(source_body(LESSON))} characters in {len(LESSON_PARAGRAPHS)} paragraphs")
    started = time.perf_counter()
    result = OllamaNotesWriter(client, rules).write(LESSON, LESSON_META, NOTES_MODEL, checkpoint)
    elapsed = time.perf_counter() - started
    markdown = result.markdown
    print(f"took {elapsed:.1f} s; notes {len(markdown)} characters; removed {result.removed_quotes} quotes, "
          f"{result.removed_concepts} concept tags")
    print(markdown)

    deadline = time.monotonic() + 10
    while NOTES_MODEL in _loaded_models() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert NOTES_MODEL not in _loaded_models()

    assert markdown.startswith(header(LESSON_META))
    found = headings(markdown)
    core = [(index, level) for index, level, title in found if unicodedata.normalize("NFKC", title) == "核心宗旨"]
    assert core, found
    core_index, level = core[0]
    same_level = [title for _, lvl, title in found if lvl == level and title != "本课脉络"]
    assert len(same_level) >= 2, found
    after_core = [title for index, lvl, title in found if lvl == level and index > core_index and title != "本课脉络"]
    assert links_resolve_once(markdown, LESSON_META.name) == []
    assert len(outline_lines(markdown)) == len(after_core)
    assert [title for _, _, title in found].count("本课脉络") <= 1

    again = finish_notes(markdown, LESSON, LESSON_META)
    assert (again.removed_quotes, again.removed_concepts) == (0, 0)

    sections = _quote_section_lines(markdown)
    print(f"quotes kept in 引述 sections: {_kept_quotes(sections)}")
    for line in _quote_lines_outside_sections(markdown):
        print(f"> line outside any 引述 section (not checked): {line}")
    if not sections:
        print(f"no 引述 section left; {result.removed_quotes} quotes were removed")
    for lines in sections:
        assert any(line.lstrip(" ").startswith(">") for line in lines), lines
        for line in lines:
            if line.strip() and not line.lstrip(" ").startswith(">") and not re.search(r"[“”「」『』\"]", line):
                print(f"引述 section line that is not a quote: {line}")

    assert result.rules_note is None
    replay = OllamaNotesWriter(OllamaClient(_closed_port_url()), rules).write(
        LESSON, LESSON_META, NOTES_MODEL, checkpoint
    )
    assert replay == result


@pytest.mark.integration
def test_real_ollama_counts_no_more_tokens_than_characters():
    """I2."""
    client = OllamaClient(timeout=600)
    _ready_notes_model(client)
    parts: list[str] = []
    i = 0
    while sum(map(len, parts)) < 20000:
        i += 1
        parts.append(
            f"第{i}个例子：表 t{i} 在列 c{i % 7} 上建了索引，查询从 {i * 3 + 40} 毫秒降到 {i % 9 + 2} 毫秒，"
            "但每次写入多花了一点时间。"
        )
    lesson = "# 长课\n\n" + "".join(parts)
    rules = shipped_rules()
    assert overflow(rules, source_body(lesson), NOTES_NUM_CTX, NOTES_REPLY_BUDGET) == 0
    prompt = build_prompt(rules, lesson)
    body = {
        "model": NOTES_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_ctx": NOTES_NUM_CTX, "num_predict": 1, "temperature": 0.1},
    }
    if "thinking" in _no_proxy_post("/api/show", {"model": NOTES_MODEL}, 30).get("capabilities", []):
        body["think"] = False
    started = time.perf_counter()
    try:
        answer = _no_proxy_post("/api/generate", body, 600)
    finally:
        client.unload(NOTES_MODEL)
    count = answer.get("prompt_eval_count", 0)
    print(f"\nprompt {len(prompt)} characters, prompt_eval_count {count}, "
          f"ratio {count / len(prompt):.3f}, {time.perf_counter() - started:.1f} s")
    assert 0 < count <= len(prompt) + 128
    assert count >= len(prompt) * 0.3


@pytest.mark.integration
def test_real_ollama_without_the_model_waits_and_downloads_nothing(tmp_path):
    """I3."""
    _ready_notes_model(OllamaClient())
    model = "coursescribe-no-such-model:0.1b"
    writer = OllamaNotesWriter(rules=lambda: load_notes_rules(tmp_path / "没有.txt"))
    started = time.perf_counter()
    with pytest.raises(NotesBlocked) as caught:
        writer.write(LESSON, LESSON_META, model, tmp_path / "job.notes.json")
    assert time.perf_counter() - started < 10
    assert str(caught.value) == NOTES_MODEL_MISSING.format(model=model)
    assert model not in OllamaClient().models()
