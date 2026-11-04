"""The rules for organised notes: the user's file, the shipped copy, and how they are read and created.

M2 item 12-4 (docs/plans/M2-notes-4-rules-tests.md).
"""

from __future__ import annotations

import ast
import fnmatch
import os
import random
import re
import subprocess
import sys
import threading
import time
import tomllib
from datetime import date
from pathlib import Path

import pytest

from coursescribe import files, vocabulary
from coursescribe.notes import lecture_date
from coursescribe.notes_rules import (
    FILE_NAME,
    RULES_CREATE_FAILED,
    RULES_EMPTY_NOTE,
    RULES_ENCODING,
    RULES_FOLDER_IN_THE_WAY,
    RULES_HELD,
    RULES_SHIPPED_MISSING,
    RULES_UNREADABLE,
    SHIPPED_RULES,
    NotesRules,
    NotesRulesUnreadable,
    ensure_notes_rules,
    load_notes_rules,
    rules_path,
    shipped_rules,
)
from coursescribe.text import content_chars
from coursescribe.vocabulary import WORD_LIST_ENCODING, WordListUnreadable, load_vocabulary, parse_vocabulary
from test_recording import _held
from test_vocabulary import _ENCODINGS, _REFUSED, _decodes_as_utf8, _random_word, _snapshot

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "coursescribe"
SHIPPED_HEADER = [
    "// CourseScribe 整理规则。程序把下面的规则和校对稿一起交给模型，让它写整理笔记。",
    "// 以 // 开头的行是说明，不交给模型。改完保存，从下一次整理开始生效。",
    "// 想用回程序自带的规则：删掉这个文件，或者把内容清空。",
    "// 笔记的标题、日期、本课脉络由程序自己写；引述和概念标签会和原文核对，对不上的会被删掉。",
]
_BREAK = re.compile(r"\r\n|\r|\n")
_COMMENT_LEAD = " \t　"  # half-width space, tab, ideographic space: only these three


def expected_rules(text: str) -> str | None:
    """The rules the model gets from ``text``, worked out here by the rules in the plan; None when there are none."""
    lines = [line for line in _BREAK.split(text) if not line.lstrip(_COMMENT_LEAD).startswith("//")]
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    joined = "\n".join(lines[start:end])
    return joined if content_chars(joined) else None


def _shipped(folder: Path, text: str = "自带规则\n") -> Path:
    path = folder / "自带.txt"
    path.write_bytes(text.encode("utf-8"))
    return path


# ---------------------------------------------------------------- the rules text generator

_RULE_JOINERS = ("，", "。", "、", "：", " ")
_LOOKALIKE_JOINERS = (" ", ",", ".", ":")  # Chinese punctuation in GBK would stop the bytes being valid UTF-8


def make_rules_text(rng: random.Random) -> tuple[str, str | None]:
    """A rules file as a user might write it, and the rules expected from it (None when it has none)."""
    lookalike = rng.random() < 0.25
    joiners = _LOOKALIKE_JOINERS if lookalike else _RULE_JOINERS
    comment_leads = ("", " ", "  ", "\t", " \t") + (() if lookalike else ("　",))
    blank_lines = ("  \t",) + (() if lookalike else ("　",))

    def word() -> str:
        return _random_word(rng, lookalike)

    def paragraph() -> str:
        out = word()
        for _ in range(rng.randint(0, 5)):
            out += rng.choice(joiners) + word()
        return out

    def comment() -> str:
        return rng.choice(comment_leads) + rng.choice(("//", "///", f"// {word()}"))

    makers = (
        paragraph,
        lambda: f"## {word()}",
        lambda: f"# {word()}",
        lambda: f"#{word()} #{word()}",
        lambda: f"> {word()}",
        lambda: f"- {word()}",
        lambda: f"1. {word()}",
        lambda: f"`#{word()}`",
        lambda: f"{word()} // {word()}",
        lambda: "https://example.com/a//b",
        comment,
        comment,
        lambda: "",
        lambda: rng.choice(blank_lines),
    )
    only_comments = rng.random() < 0.1
    lines = []
    kinds = []
    for _ in range(rng.randint(1, 30)):
        if only_comments:
            kind = rng.choice(("comment", "blank", "space"))
            line = {"comment": comment, "blank": lambda: "", "space": lambda: rng.choice(blank_lines)}[kind]()
        else:
            maker = rng.choice(makers)
            kind = "comment" if maker is comment else "other"
            line = maker()
        lines.append(line + " " * rng.randint(0, 2))
        kinds.append(kind)
    first = next((i for i, (line, kind) in enumerate(zip(lines, kinds)) if kind != "comment" and line.strip()), None)
    if first is not None and rng.random() < 0.2:
        lines[first] = rng.choice(("\t", "  ")) + lines[first]
    text = ""
    for number, line in enumerate(lines):
        ending = rng.choice(("\n", "\r\n", "\r"))
        if number == len(lines) - 1 and rng.random() < 0.5:
            ending = ""
        text += line + ending
    return text, expected_rules(text)


def _has_chinese(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


# ---------------------------------------------------------------- files.decode_user_text


def test_there_is_one_decoder_for_the_word_list_and_the_rules():
    """D2."""
    assert callable(getattr(files, "decode_user_text", None))
    assert not hasattr(vocabulary, "_decode_word_list") or vocabulary._decode_word_list is files.decode_user_text


def _word_list_outcome(path: Path):
    try:
        return "read", load_vocabulary(path)
    except WordListUnreadable as exc:
        assert str(exc) == WORD_LIST_ENCODING.format(path=path), str(exc)
        return "refused", None


def _rules_outcome(path: Path, shipped: Path):
    try:
        return "read", load_notes_rules(path, shipped=shipped)
    except NotesRulesUnreadable as exc:
        assert str(exc) == RULES_ENCODING.format(path=path), str(exc)
        return "refused", None


def test_the_word_list_and_the_rules_agree_on_every_string_of_bytes(tmp_path):
    """D1: the same bytes are refused by both or read by both into the same text."""
    rng = random.Random(20261231)
    shipped = _shipped(tmp_path)
    words = tmp_path / "词表.txt"
    rules = tmp_path / "整理规则.txt"
    samples: list[tuple[bytes, str | None, str | None]] = []  # bytes, the text written, the rules expected
    for _ in range(200):
        text, expected = make_rules_text(rng)
        for encode in _ENCODINGS.values():
            samples.append((encode(text), text, expected or "自带规则"))
    for data in _REFUSED.values():
        samples.append((data, None, None))
    fixed = {
        b"": ("", "自带规则"),
        b"\xef\xbb\xbf": ("", "自带规则"),
        "学习\n".encode("gbk"): ("学习\n", "学习"),
        "浏电保\n".encode("gbk"): ("浏电保\n", "浏电保"),
        "привет\n".encode("utf-8"): ("привет\n", "привет"),
        # Outside both the Western ranges and GB2312, so GBK is tried; it does not read them cleanly.
        "café € naïve\n".encode("utf-8"): ("café € naïve\n", "café € naïve"),
        "规则😀\n".encode("utf-8"): ("规则😀\n", "规则😀"),
        # Punctuation only: read as UTF-8 (the word list shows it), but no rules, so the shipped ones.
        "–—…\n".encode("utf-8"): ("–—…\n", "自带规则"),
        "–—… a\n".encode("utf-8"): ("–—… a\n", "–—… a"),
    }
    for data, (text, expected) in fixed.items():
        samples.append((data, text, expected))

    for data, text, expected in samples:
        words.write_bytes(data)
        rules.write_bytes(data)
        before = (_snapshot(words), _snapshot(rules))
        word_kind, word_list = _word_list_outcome(words)
        rules_kind, got = _rules_outcome(rules, shipped)
        assert word_kind == rules_kind, (data, word_kind, rules_kind)
        if text is None:  # the refused strings of the word list's own tests
            assert rules_kind == "refused", data
        else:  # generated in an accepted encoding, or one of the fixed strings both read
            assert rules_kind == "read", (data, text)
            assert got.text == expected, (data, text)
            assert word_list == parse_vocabulary(text), (data, text)
        assert (_snapshot(words), _snapshot(rules)) == before


# ---------------------------------------------------------------- load_notes_rules


def test_a_typical_rules_file_is_read_without_its_notes_to_the_user(tmp_path):
    """R1."""
    shipped = _shipped(tmp_path)
    path = tmp_path / FILE_NAME
    path.write_bytes(
        "// 我的说明\n\n## 核心宗旨\n写一两句话。  \n\n#标签 只用原文里的词\n  // 缩进的说明\n> 引述照抄\n\n".encode("utf-8")
    )
    before = _snapshot(path)
    assert load_notes_rules(path, shipped=shipped) == NotesRules(
        "## 核心宗旨\n写一两句话。  \n\n#标签 只用原文里的词\n> 引述照抄", None
    )
    assert _snapshot(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([FILE_NAME, "自带.txt"])


_EDGES = {
    "a-comment-line-goes-whole": ("规则一\n// 说明\n规则二", "规则一\n规则二", "utf-8"),
    "blank-lines-beside-it-stay": ("规则一\n\n// 说明\n\n规则二", "规则一\n\n\n规则二", "utf-8"),
    "indented-comments": ("\t// 制表符\n　// 全角空格\n  // 两个空格\n///三条\n规则", "规则", "utf-8"),
    "slashes-mid-line": ("规则 // 行中间的不算", "规则 // 行中间的不算", "utf-8"),
    "an-address": ("https://example.com/a//b", "https://example.com/a//b", "utf-8"),
    "one-slash-or-full-width": ("/ 一条斜杠\n／／全角斜杠", "/ 一条斜杠\n／／全角斜杠", "utf-8"),
    "hash-lines-are-rules": (
        "# 一级标题\n## 核心宗旨\n#标签\n####### 七个", "# 一级标题\n## 核心宗旨\n#标签\n####### 七个", "utf-8"
    ),
    "line-breaks": ("规则一\r\n规则二\r规则三\n", "规则一\n规则二\n规则三", "utf-8"),
    "blank-lines-around": ("\n\n  \n　\n规则\n\t\n\n", "规则", "utf-8"),
    "line-separator-is-not-a-break": ("规则 同一行", "规则 同一行", "utf-8"),
    "no-nfkc-no-trailing-strip": ("ＡＢＣ　全角\t制表符  ", "ＡＢＣ　全角\t制表符  ", "utf-8"),
    "utf-16": ("// 说明\n规则", "规则", "utf-16-le-bom"),
    "the-first-line-keeps-its-indent": ("\n　　缩进的规则\n  - 子项", "　　缩进的规则\n  - 子项", "utf-8"),
    "a-bom-inside-stays": ("规则﻿一", "规则﻿一", "utf-8"),
    "no-break-space-is-not-a-comment-lead": (" // 不是说明\n规则", " // 不是说明\n规则", "utf-8"),
    "em-space-is-not-a-comment-lead": (" // x", " // x", "utf-8"),
}


@pytest.mark.parametrize("name", list(_EDGES))
def test_comment_lines_and_tidying_at_the_edges(tmp_path, name):
    """R2."""
    text, expected, encoding = _EDGES[name]
    shipped = _shipped(tmp_path)
    path = tmp_path / FILE_NAME
    path.write_bytes(_ENCODINGS[encoding](text))
    before = _snapshot(path)
    assert load_notes_rules(path, shipped=shipped) == NotesRules(expected, None)
    assert _snapshot(path) == before


def test_a_control_character_in_the_rules_is_refused(tmp_path):
    """R2 (j): a line separator is text, a vertical tab is not."""
    shipped = _shipped(tmp_path)
    path = tmp_path / FILE_NAME
    path.write_bytes("规则 同一行\x0b".encode("utf-8"))
    before = _snapshot(path)
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path, shipped=shipped)
    assert str(caught.value) == RULES_ENCODING.format(path=path)
    assert _snapshot(path) == before


_NO_RULES = {
    "empty": b"",
    "utf-8-mark-only": b"\xef\xbb\xbf",
    "utf-16-mark-only": b"\xff\xfe",
    "blank-lines": b"\r\n\r\n  \t\n",
    "comments-only-gbk": "// 只有说明\n  // 还是说明\n".encode("gbk"),
    "punctuation-only": "。。。\n——\n".encode("utf-8"),
    "ideographic-space": "　\n".encode("utf-8"),
    "a-hash-alone": "// 说明\n#".encode("utf-8"),
}


@pytest.mark.parametrize("name", list(_NO_RULES))
def test_a_file_without_rules_uses_the_shipped_ones_and_says_so(tmp_path, name):
    """R3."""
    shipped = _shipped(tmp_path, "// 自带的说明\n自带规则\n")
    path = tmp_path / FILE_NAME
    path.write_bytes(_NO_RULES[name])
    before = _snapshot(path)
    got = load_notes_rules(path, shipped=shipped)
    assert got == NotesRules("自带规则", RULES_EMPTY_NOTE.format(path=path))
    assert str(path) in got.note
    assert _snapshot(path) == before  # the shipped rules are not written into it


@pytest.mark.parametrize("text", ["一", "a", "1"])
def test_one_letter_or_digit_is_a_rule(tmp_path, text):
    """R3, the other side of the edge."""
    shipped = _shipped(tmp_path, "// 自带的说明\n自带规则\n")
    path = tmp_path / FILE_NAME
    path.write_bytes(text.encode("utf-8"))
    assert load_notes_rules(path, shipped=shipped) == NotesRules(text, None)


def test_without_a_file_the_shipped_rules_are_used_and_nothing_is_created(tmp_path):
    """R4."""
    shipped = shipped_rules()
    assert content_chars(shipped)
    assert load_notes_rules(tmp_path / FILE_NAME) == NotesRules(shipped, None)
    assert load_notes_rules(tmp_path / "无" / FILE_NAME) == NotesRules(shipped, None)
    assert list(tmp_path.iterdir()) == []


def test_rules_read_back_the_same_in_every_encoding_and_line_break(tmp_path):
    """R5: the round trip, and reading changes nothing."""
    rng = random.Random(20261230)
    shipped = _shipped(tmp_path)
    path = tmp_path / FILE_NAME
    none = hashes = comments = returns = indented = gbk_also_utf8 = 0
    for _ in range(300):
        text, expected = make_rules_text(rng)
        lines = _BREAK.split(text)
        none += expected is None
        hashes += any(line.startswith("#") for line in lines)
        comments += any(line.lstrip(_COMMENT_LEAD).startswith("//") for line in lines)
        returns += "\r" in text
        indented += expected is not None and expected[0].isspace()
        gbk_also_utf8 += _has_chinese(text) and _decodes_as_utf8(text.encode("gbk"))
        want = NotesRules(expected, None) if expected is not None else NotesRules(
            "自带规则", RULES_EMPTY_NOTE.format(path=path)
        )
        for name, encode in _ENCODINGS.items():
            path.write_bytes(encode(text))
            before = _snapshot(path)
            names = sorted(tmp_path.iterdir())
            got = load_notes_rules(path, shipped=shipped)
            assert got == want, (name, text)
            assert not got.text.startswith("﻿"), (name, text)
            assert _snapshot(path) == before
            assert sorted(tmp_path.iterdir()) == names
    assert none >= 15
    assert hashes >= 150
    assert comments >= 150
    assert returns >= 150
    assert indented >= 30
    assert gbk_also_utf8 >= 30


@pytest.mark.parametrize("name", list(_REFUSED))
def test_rules_in_an_encoding_that_cannot_be_told_are_refused_not_replaced(tmp_path, name):
    """R6."""
    path = tmp_path / FILE_NAME
    path.write_bytes(_REFUSED[name])
    before = _snapshot(path)
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path)
    message = str(caught.value)
    assert message == RULES_ENCODING.format(path=path)
    for part in (str(path), "UTF-8", "重试", "改个名字", "校对稿没有变"):
        assert part in message
    assert "Ollama" not in message
    assert _snapshot(path) == before
    assert [p.name for p in tmp_path.iterdir()] == [FILE_NAME]


def test_rules_held_for_a_moment_are_waited_for(tmp_path, monkeypatch):
    """R7 (a)."""
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    path = tmp_path / FILE_NAME
    data = "## 核心宗旨\n写一句话。\n".encode("gbk")
    path.write_bytes(data)
    held = _held(monkeypatch, path, times=3)
    assert load_notes_rules(path).text == "## 核心宗旨\n写一句话。"
    assert held.attempts == 4
    assert path.read_bytes() == data


@pytest.mark.parametrize(("can_be_held", "attempts"), [(True, 20), (False, 1)], ids=["windows", "elsewhere"])
def test_rules_held_for_long_are_reported_as_held(tmp_path, monkeypatch, can_be_held, attempts):
    """R7 (b)."""
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", can_be_held)
    path = tmp_path / FILE_NAME
    data = "## 核心宗旨\n写一句话。\n".encode("gbk")
    path.write_bytes(data)
    held = _held(monkeypatch, path)
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path)
    assert str(caught.value) == RULES_HELD.format(path=path)
    assert held.attempts == attempts
    held.release()
    assert load_notes_rules(path) == NotesRules("## 核心宗旨\n写一句话。", None)
    assert path.read_bytes() == data


def test_rules_on_a_drive_that_fails_are_reported_at_once(tmp_path, monkeypatch):
    """R8."""
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    path = tmp_path / FILE_NAME
    path.write_bytes("规则".encode("utf-8"))
    before = _snapshot(path)
    held = _held(monkeypatch, path, error=OSError(5, "Input/output error"))
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path)
    assert str(caught.value) == RULES_UNREADABLE.format(path=path)
    assert held.attempts == 1
    assert isinstance(caught.value.__cause__, OSError)
    held.release()  # reading the file back goes through the same stand-in
    assert _snapshot(path) == before


def test_a_folder_named_like_the_rules_is_not_opened(tmp_path, monkeypatch):
    """R9."""
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    path = tmp_path / FILE_NAME
    path.mkdir()
    held = _held(monkeypatch, path, times=0)
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path)
    assert str(caught.value) == RULES_UNREADABLE.format(path=path)
    assert held.attempts == 0
    assert list(path.iterdir()) == []


def test_broken_shipped_rules_are_reported_only_when_they_are_needed(tmp_path):
    """R10."""
    missing = tmp_path / "没有.txt"
    gbk = tmp_path / "gbk.txt"
    gbk.write_bytes("规则".encode("gbk"))
    empty = tmp_path / "空.txt"
    empty.write_bytes("// 只有说明\n".encode("utf-8"))
    path = tmp_path / FILE_NAME

    for shipped in (missing, empty):  # (a), (d)
        with pytest.raises(NotesRulesUnreadable) as caught:
            load_notes_rules(path, shipped=shipped)
        assert str(caught.value) == RULES_SHIPPED_MISSING.format(path=shipped)

    path.write_bytes("我的规则".encode("utf-8"))  # (b)
    assert load_notes_rules(path, shipped=missing) == NotesRules("我的规则", None)

    path.write_bytes(b"")  # (c)
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path, shipped=gbk)
    assert str(caught.value) == RULES_SHIPPED_MISSING.format(path=gbk)

    for shipped in (missing, gbk, empty):  # (e)
        with pytest.raises(NotesRulesUnreadable) as caught:
            shipped_rules(shipped)
        assert str(caught.value) == RULES_SHIPPED_MISSING.format(path=shipped)


def test_changed_rules_are_read_again_the_next_time(tmp_path):
    """R11."""
    shipped = _shipped(tmp_path)
    path = tmp_path / FILE_NAME
    path.write_bytes("规则一".encode("utf-8"))
    assert load_notes_rules(path, shipped=shipped) == NotesRules("规则一", None)
    path.write_bytes("规则二".encode("utf-8"))  # same length, within the same second
    assert load_notes_rules(path, shipped=shipped) == NotesRules("规则二", None)
    path.write_bytes(b"")
    assert load_notes_rules(path, shipped=shipped) == NotesRules("自带规则", RULES_EMPTY_NOTE.format(path=path))
    path.unlink()
    assert load_notes_rules(path, shipped=shipped) == NotesRules("自带规则", None)


def test_a_very_large_rules_file_is_read_whole_and_quickly(tmp_path):
    """R12."""
    lines = ["// 说明" if i % 100 == 0 else f"第{i:06d}条：只写原文里有的内容。" for i in range(100_000)]
    text = "\r\n".join(lines)
    expected = expected_rules(text)
    comments = sum(1 for line in lines if line.startswith("//"))
    path = tmp_path / FILE_NAME
    results = []
    for encoding in ("gbk", "utf-16-le-bom"):
        path.write_bytes(_ENCODINGS[encoding](text))
        before = _snapshot(path)
        started = time.perf_counter()
        got = load_notes_rules(path)
        assert time.perf_counter() - started < 5, encoding
        assert got.text == expected, encoding
        assert got.text.count("\n") + 1 == 100_000 - comments
        assert _snapshot(path) == before
        results.append(got.text)
    assert results[0] == results[1]


def test_the_rules_live_in_the_program_data_folder(tmp_path, monkeypatch):
    """R13."""
    monkeypatch.setenv("COURSESCRIBE_HOME", str(tmp_path / "家"))
    assert FILE_NAME == "整理规则.txt"
    assert rules_path() == tmp_path / "家" / "整理规则.txt"
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- ensure_notes_rules


def test_the_first_time_the_shipped_file_is_copied_byte_for_byte(tmp_path):
    """E1."""
    path = tmp_path / "家" / FILE_NAME
    ensure_notes_rules(path)
    assert path.is_file()
    assert path.read_bytes() == SHIPPED_RULES.read_bytes()
    assert load_notes_rules(path) == NotesRules(shipped_rules(), None)
    assert sorted(p.name for p in path.parent.iterdir()) == [FILE_NAME]

    shipped = tmp_path / "别的自带.txt"
    shipped.write_bytes(b"// x\r\n\xe8\xa7\x84\xe5\x88\x99\r\n")
    path2 = tmp_path / "二" / FILE_NAME
    ensure_notes_rules(path2, shipped=shipped)
    assert path2.read_bytes() == b"// x\r\n\xe8\xa7\x84\xe5\x88\x99\r\n"


def test_creating_twice_leaves_the_one_file_untouched(tmp_path):
    """E2."""
    path = tmp_path / FILE_NAME
    ensure_notes_rules(path)
    assert path.is_file()
    before = _snapshot(path)
    ensure_notes_rules(path)
    ensure_notes_rules(path)
    assert _snapshot(path) == before
    assert [p.name for p in tmp_path.iterdir()] == [FILE_NAME]


def test_an_existing_file_is_never_touched_whatever_it_holds(tmp_path):
    """E3."""
    assert SHIPPED_RULES.is_file()
    contents = [
        "我的规则\n".encode("utf-8"),
        "我的规则\r\n".encode("gbk"),
        b"\xff\xfe" + "我的规则".encode("utf-16-le"),
        "café".encode("latin-1"),
        b"",
        "// 只有说明\n".encode("utf-8"),
        SHIPPED_RULES.read_bytes()[:-5],
    ]
    for number, data in enumerate(contents):
        path = tmp_path / str(number) / FILE_NAME
        path.parent.mkdir()
        path.write_bytes(data)
        before = _snapshot(path)
        assert ensure_notes_rules(path) is None
        assert _snapshot(path) == before, data
        assert [p.name for p in path.parent.iterdir()] == [FILE_NAME]
    fresh = tmp_path / "新" / FILE_NAME
    ensure_notes_rules(fresh)
    assert fresh.is_file()


def test_a_folder_in_the_way_is_explained(tmp_path):
    """E4."""
    path = tmp_path / FILE_NAME
    path.mkdir()
    with pytest.raises(NotesRulesUnreadable) as caught:
        ensure_notes_rules(path)
    message = str(caught.value)
    assert message == RULES_FOLDER_IN_THE_WAY.format(path=path)
    for part in (str(path), "文件夹", "打开整理规则"):
        assert part in message
    assert "这次没有整理笔记" not in message
    assert list(path.iterdir()) == []
    assert [p.name for p in tmp_path.iterdir()] == [FILE_NAME]


def _no_space(*args):
    raise OSError(28, "No space left on device")


def test_a_full_disk_leaves_nothing_and_it_can_be_created_later(tmp_path, monkeypatch):
    """E5."""
    path = tmp_path / FILE_NAME
    monkeypatch.setattr("coursescribe.files.os.fsync", _no_space)
    with pytest.raises(NotesRulesUnreadable) as caught:
        ensure_notes_rules(path)
    assert str(caught.value) == RULES_CREATE_FAILED.format(path=path)
    assert caught.value.__cause__.errno == 28
    assert not path.exists()
    assert list(path.parent.iterdir()) == []
    monkeypatch.undo()
    ensure_notes_rules(path)
    assert path.read_bytes() == SHIPPED_RULES.read_bytes()
    assert [p.name for p in tmp_path.iterdir()] == [FILE_NAME]


def test_a_file_where_the_data_folder_should_be_is_explained(tmp_path):
    """E6."""
    home = tmp_path / "家"
    home.write_bytes(b"x")
    path = home / FILE_NAME
    with pytest.raises(NotesRulesUnreadable) as caught:
        ensure_notes_rules(path)
    assert str(caught.value) == RULES_CREATE_FAILED.format(path=path)
    assert home.read_bytes() == b"x"
    assert [p.name for p in tmp_path.iterdir()] == ["家"]


def test_a_file_another_program_creates_meanwhile_is_kept(tmp_path, monkeypatch):
    """E7: the target appears between the check and the publishing."""
    path = tmp_path / FILE_NAME
    theirs = "别人写的\n".encode("gbk")
    real_fsync = os.fsync

    def fsync_then_someone_writes(descriptor):
        real_fsync(descriptor)
        path.write_bytes(theirs)

    monkeypatch.setattr("coursescribe.files.os.fsync", fsync_then_someone_writes)
    ensure_notes_rules(path)
    assert path.is_file()
    assert path.read_bytes() == theirs
    assert [p.name for p in tmp_path.iterdir()] == [FILE_NAME]


def test_two_threads_creating_at_once_leave_one_whole_file(tmp_path):
    """E8."""
    assert SHIPPED_RULES.is_file()
    shipped = SHIPPED_RULES.read_bytes()
    for round_ in range(20):
        path = tmp_path / f"轮{round_}" / FILE_NAME
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def create():
            barrier.wait()
            try:
                ensure_notes_rules(path)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=create) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert path.parent.is_dir()
        assert [p.name for p in path.parent.iterdir()] == [FILE_NAME]
        assert path.read_bytes() == shipped


_DIE_AT = """
import os, sys
from pathlib import Path
import coursescribe.notes_rules as rules

real = getattr(os, sys.argv[2])

def die(*args, **kwargs):
    real(*args, **kwargs)
    os._exit(3)

setattr(os, sys.argv[2], die)
rules.ensure_notes_rules(Path(sys.argv[1]))
"""
_TEMPORARY = re.compile(r"\.coursescribe-.*\.tmp")


@pytest.mark.parametrize("moment", ["fsync", "link"])
def test_a_program_killed_while_creating_leaves_no_half_file(tmp_path, moment):
    """E9: the process really dies before publishing (fsync) and after it (link)."""
    path = tmp_path / "家" / FILE_NAME
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run([sys.executable, "-c", _DIE_AT, str(path), moment], env=env, capture_output=True, timeout=60)
    assert done.returncode == 3, done.stderr.decode("utf-8", "replace")
    if moment == "fsync":
        assert not path.exists()
    else:
        assert path.read_bytes() == SHIPPED_RULES.read_bytes()
    ensure_notes_rules(path)
    assert path.read_bytes() == SHIPPED_RULES.read_bytes()
    others = [p.name for p in path.parent.iterdir() if p.name != FILE_NAME]
    assert all(_TEMPORARY.fullmatch(name) for name in others), others


def test_without_the_shipped_file_nothing_is_created(tmp_path):
    """E10."""
    path = tmp_path / "家" / FILE_NAME
    with pytest.raises(NotesRulesUnreadable) as caught:
        ensure_notes_rules(path, shipped=tmp_path / "没有.txt")
    assert str(caught.value) == RULES_SHIPPED_MISSING.format(path=tmp_path / "没有.txt")
    assert not (tmp_path / "家").exists()
    assert list(tmp_path.iterdir()) == []


def test_an_existing_file_is_fine_on_a_full_disk(tmp_path, monkeypatch):
    """E11."""
    path = tmp_path / FILE_NAME
    path.write_bytes("我的规则\n".encode("utf-8"))
    before = _snapshot(path)
    monkeypatch.setattr("coursescribe.files.os.fsync", _no_space)
    assert ensure_notes_rules(path) is None
    assert _snapshot(path) == before
    assert [p.name for p in tmp_path.iterdir()] == [FILE_NAME]
    monkeypatch.undo()
    fresh = tmp_path / "新" / FILE_NAME
    ensure_notes_rules(fresh)
    assert fresh.is_file()


def test_an_existing_file_is_fine_without_the_shipped_file(tmp_path):
    """E12."""
    path = tmp_path / FILE_NAME
    path.write_bytes("我的规则\n".encode("utf-8"))
    before = _snapshot(path)
    assert ensure_notes_rules(path, shipped=tmp_path / "没有.txt") is None
    assert _snapshot(path) == before
    assert [p.name for p in tmp_path.iterdir()] == [FILE_NAME]
    fresh = tmp_path / "新" / FILE_NAME
    ensure_notes_rules(fresh)
    assert fresh.is_file()


# ---------------------------------------------------------------- notes_rules.txt and packaging


def test_the_shipped_file_is_plain_utf_8_and_starts_with_its_instructions():
    """P1."""
    assert SHIPPED_RULES.is_file()
    data = SHIPPED_RULES.read_bytes()
    text = data.decode("utf-8")
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    assert text.split("\n")[:4] == SHIPPED_HEADER


_TAG_LIKE = re.compile(r"(?:^|\s)#[A-Za-z0-9_一-鿿]", re.MULTILINE)


def test_the_shipped_rules_are_generic_short_and_ask_for_the_structure():
    """P2."""
    text = shipped_rules()
    assert 0 < len(text) <= 1500
    assert not any(line.lstrip(_COMMENT_LEAD).startswith("//") for line in text.split("\n"))
    at = -1
    for part in ("核心宗旨", "案例", "类比", "引述", "提示词", "结尾总结"):
        found = text.find(part, at + 1)
        assert found > at, part
        at = found
    first_mentions = [text.index(part) for part in ("核心宗旨", "案例", "类比", "引述", "提示词", "结尾总结")]
    assert first_mentions == sorted(first_mentions)
    assert ">" in text
    assert "本课脉络" in text
    assert _TAG_LIKE.search(re.sub(r"`[^`\n]*`", "", text)) is None
    assert lecture_date(text, date(2001, 2, 3)) == date(2001, 2, 3)
    for part in ("http", "www.", "[[", ":\\", "/Users/"):
        assert part not in text, part


def test_the_wheel_carries_the_shipped_rules():
    """P3."""
    patterns = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"][
        "package-data"
    ]["coursescribe"]
    assert any(fnmatch.fnmatchcase("notes_rules.txt", pattern) for pattern in patterns), patterns
    assert any(fnmatch.fnmatchcase("ui/chevron.svg", pattern) for pattern in patterns), patterns
    assert (SRC / "notes_rules.txt").is_file()


def _top_level_values(tree: ast.Module, name: str) -> list[ast.expr]:
    values = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            values.append(statement.value)
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name) and statement.target.id == name:
            values.append(statement.value)
    return values


def _datas_expressions(tree: ast.Module) -> list[ast.expr]:
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Analysis"
    ]
    assert len(calls) == 1
    datas = [keyword.value for keyword in calls[0].keywords if keyword.arg == "datas"]
    assert len(datas) == 1
    expressions = list(datas)
    seen: set[str] = set()
    index = 0
    while index < len(expressions):
        for node in ast.walk(expressions[index]):
            if isinstance(node, ast.Name) and node.id not in seen:
                seen.add(node.id)
                expressions += _top_level_values(tree, node.id)
        index += 1
    return expressions


def _data_items(expressions: list[ast.expr], file_name: str) -> list[ast.Tuple]:
    return [
        node for expression in expressions for node in ast.walk(expression)
        if isinstance(node, ast.Tuple) and len(node.elts) == 2 and file_name in ast.unparse(node.elts[0])
    ]


def test_the_packaged_program_gets_the_shipped_rules_where_the_module_looks():
    """P4."""
    tree = ast.parse((REPO / "packaging" / "coursescribe.spec").read_text(encoding="utf-8"))
    expressions = _datas_expressions(tree)
    items = [
        item for item in _data_items(expressions, "notes_rules.txt")
        if isinstance(item.elts[1], ast.Constant) and item.elts[1].value == "coursescribe"
    ]
    assert len(items) == 1
    source = eval(  # noqa: S307 - the spec's own expression, with only ROOT and str
        compile(ast.Expression(items[0].elts[0]), "spec", "eval"), {"__builtins__": {}, "ROOT": REPO, "str": str}
    )
    assert Path(source).resolve() == (SRC / "notes_rules.txt").resolve()
    assert Path(source).is_file()
    assert _data_items(expressions, "chevron.svg")
    assert SHIPPED_RULES.is_absolute()
    assert SHIPPED_RULES.resolve() == (SRC / "notes_rules.txt").resolve()


# ---------------------------------------------------------------- after the final review (12-4)


def test_gbk_rules_that_happen_to_be_valid_utf_8_are_read_as_gbk(tmp_path):
    """R16: "学习" in GBK is also valid UTF-8; it is read as the GBK the user saved."""
    data = "学习\n".encode("gbk")
    assert _decodes_as_utf8(data)  # the premise
    path = tmp_path / FILE_NAME
    path.write_bytes(data)
    before = _snapshot(path)
    assert load_notes_rules(path) == NotesRules("学习", None)
    assert _snapshot(path) == before


def _temporaries_refuse_removal(monkeypatch, delete: bool) -> list[str]:
    """Removing a .coursescribe-*.tmp file fails, as when antivirus holds it; ``delete`` removes it first."""
    real_unlink = Path.unlink
    refused: list[str] = []

    def unlink(self, missing_ok=False):
        if _TEMPORARY.fullmatch(self.name):
            refused.append(self.name)
            if delete:
                real_unlink(self, missing_ok=missing_ok)
            raise PermissionError(13, "Access is denied")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    return refused


_CREATORS = {
    "notes-rules": (ensure_notes_rules, lambda: SHIPPED_RULES.read_bytes()),
    "word-list": (vocabulary.ensure_word_list, lambda: vocabulary.TEMPLATE.encode("utf-8")),
}


@pytest.mark.parametrize("delete", [True, False], ids=["removed-then-refused", "refused"])
@pytest.mark.parametrize("creator", list(_CREATORS))
def test_a_temporary_file_that_cannot_be_removed_does_not_fail_a_created_file(tmp_path, monkeypatch, creator, delete):
    """E13: the file is published; only the cleaning up fails, so the user is not told it could not be created."""
    ensure, expected = _CREATORS[creator]
    path = tmp_path / "家" / FILE_NAME
    refused = _temporaries_refuse_removal(monkeypatch, delete)
    assert ensure(path) is None
    assert refused  # the cleaning up really was refused
    monkeypatch.undo()
    assert path.read_bytes() == expected()
    others = [p.name for p in path.parent.iterdir() if p.name != FILE_NAME]
    if delete:
        assert others == []
    else:
        assert len(others) == 1 and _TEMPORARY.fullmatch(others[0]), others


@pytest.mark.parametrize("creator", list(_CREATORS))
def test_when_publishing_and_cleaning_up_both_fail_the_real_cause_is_reported(tmp_path, monkeypatch, creator):
    """E14: a full disk while publishing is what the user hears about, not the refused clean-up after it."""
    ensure, _ = _CREATORS[creator]
    path = tmp_path / FILE_NAME

    def no_link(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("coursescribe.files.os.link", no_link)
    _held(monkeypatch, path, error=OSError(28, "No space left on device"))  # the fallback open("xb") fails too
    refused = _temporaries_refuse_removal(monkeypatch, delete=False)
    if creator == "notes-rules":
        with pytest.raises(NotesRulesUnreadable) as caught:
            ensure(path)
        assert str(caught.value) == RULES_CREATE_FAILED.format(path=path)
        cause = caught.value.__cause__
    else:
        with pytest.raises(OSError) as caught:
            ensure(path)
        cause = caught.value
    assert refused
    assert cause.errno == 28, repr(cause)
    monkeypatch.undo()
    assert not path.exists()


def _stat_refused(monkeypatch, path: Path) -> None:
    """Looking at ``path`` itself is refused, as with a folder the user has no right to list."""
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if Path(self) == path:
            raise PermissionError(13, "Access is denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


def test_rules_that_cannot_even_be_looked_at_are_reported_as_unreadable(tmp_path, monkeypatch):
    """R14."""
    path = tmp_path / FILE_NAME
    path.write_bytes("我的规则".encode("utf-8"))
    before = _snapshot(path)
    _stat_refused(monkeypatch, path)
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path)
    assert str(caught.value) == RULES_UNREADABLE.format(path=path)
    assert isinstance(caught.value.__cause__, PermissionError)
    monkeypatch.undo()
    assert _snapshot(path) == before


def test_a_place_that_cannot_even_be_looked_at_is_reported_as_not_creatable(tmp_path, monkeypatch):
    """E15."""
    path = tmp_path / FILE_NAME
    _stat_refused(monkeypatch, path)
    with pytest.raises(NotesRulesUnreadable) as caught:
        ensure_notes_rules(path)
    assert str(caught.value) == RULES_CREATE_FAILED.format(path=path)
    assert isinstance(caught.value.__cause__, PermissionError)
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_a_file_where_the_data_folder_should_be_is_named_in_the_message(tmp_path):
    """E6 (b): the message names the cause the user actually has."""
    home = tmp_path / "家"
    home.write_bytes(b"x")
    with pytest.raises(NotesRulesUnreadable) as caught:
        ensure_notes_rules(home / FILE_NAME)
    assert "同名文件" in str(caught.value)


def test_a_control_character_is_named_with_a_way_out(tmp_path):
    """R6 (b): the encoding message also covers invisible control characters."""
    path = tmp_path / FILE_NAME
    path.write_bytes("规则\x0b\n".encode("utf-8"))
    with pytest.raises(NotesRulesUnreadable) as caught:
        load_notes_rules(path)
    message = str(caught.value)
    for part in ("控制字符", "新文件", "UTF-8", "重试", "改个名字", "校对稿没有变"):
        assert part in message, part


def test_the_note_for_punctuation_only_rules_says_so(tmp_path):
    """R3 (b): the note does not claim the file is empty or only notes when it is only punctuation."""
    path = tmp_path / FILE_NAME
    path.write_bytes("。。。\n".encode("utf-8"))
    note = load_notes_rules(path).note
    assert note is not None
    assert "只有标点" in note
    assert str(path) in note
