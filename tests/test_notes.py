"""Organised notes: header, outline and clean-up of the model's reply, as pure string logic."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from coursescribe.notes import (
    EmptyNotes,
    EmptySource,
    NotesMeta,
    NotesResult,
    assemble_notes,
    build_prompt,
    finish_notes,
    lecture_date,
    notes_body,
    overflow,
    source_body,
    structure_notes,
)
from coursescribe.text import content_chars
from notes_quotes import BAD_TAGS, GOOD_TAGS, INTRO, INVENTED, SOURCE_SENTENCES, make_quoted
from notes_replies import META, NAME, make_reply
from test_notes_rules import make_rules_text
from test_vocabulary import _random_word

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FALLBACK = date(2001, 2, 3)
MARGIN = 256  # written out on purpose: changing the margin in the code must break these tests
COURSE = NotesMeta("课", date(2026, 10, 30))
COURSE_HEADER = "# 课\n\n日期：2026-10-30\n校对稿：[[课]]\n来源：\n"

_ADDRESS = {
    "%": "%25", " ": "%20", "#": "%23", "|": "%7C", "^": "%5E", "[": "%5B", "]": "%5D",
    "(": "%28", ")": "%29", "<": "%3C", ">": "%3E",
}
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t](.*))?$")
_FULLWIDTH = str.maketrans("0123456789.-_", "０１２３４５６７８９．－＿")
_LINK_BREAKERS = str.maketrans("#|^[]", "＃｜＾［］")


def header(meta: NotesMeta) -> str:
    """The five header lines the program must write, built here from the stated rules."""
    if any(ch in meta.name for ch in "#|^[]"):
        shown = "".join("\\" + ch if ch in "\\[]" else ch for ch in meta.name)
        address = "".join(_ADDRESS.get(ch, ch) for ch in meta.name)
        link = f"[{shown}]({address}.md)"
    else:
        link = f"[[{meta.name}]]"
    return f"# {meta.name}\n\n日期：{meta.date:%Y-%m-%d}\n校对稿：{link}\n来源：\n"


def headings(text: str) -> list[tuple[int, int, str]]:
    """(line index, level, text) of each heading outside code, by the stated rules."""
    found = []
    fence = None
    for index, line in enumerate(text.split("\n")):
        match = _FENCE.match(line)
        if fence is not None:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) \
                    and not match.group(2).strip():
                fence = None
            continue
        if match:
            fence = match.group(1)
            continue
        match = _HEADING.match(line)
        if match:
            title = (match.group(2) or "").strip(" \t")
            if title.endswith("#"):
                bare = title.rstrip("#")
                if not bare:
                    title = ""
                elif bare[-1] in " \t":
                    title = bare.rstrip(" \t")
            found.append((index, len(match.group(1)), title))
    return found


def key(title: str) -> str:
    return unicodedata.normalize("NFKC", title).casefold()


def outline_lines(result: str) -> list[str]:
    return [line for line in result.split("\n") if line.startswith("- [[#")]


def section_lines(result: str, level: int = 2) -> list[str]:
    lines = result.split("\n")
    return [lines[i] for i, lvl, text in headings(result) if lvl == level and text != "本课脉络"]


def links_resolve_once(result: str, name: str) -> list[str]:
    """Problems with the outline's link targets, checked against the result's own headings."""
    titles = [key(text) for _, _, text in headings(result)]
    problems = []
    for line in outline_lines(result):
        target = line[len("- [[#"):-len("]]")]
        if titles.count(key(target)) != 1:
            problems.append(f"{target!r} is the text of {titles.count(key(target))} headings")
        if key(target) == key(name):
            problems.append(f"{target!r} is the name")
    return problems


def samples(seed: int, count: int = 200, clean_titles: bool = False):
    rng = random.Random(seed)
    return [make_reply(rng, clean_titles) for _ in range(count)]


def assert_coverage(batch) -> None:
    expects = [expect for _, expect in batch]
    counts = {
        "wrapped": sum(e.wrapped for e in expects),
        "unclosed": sum(e.unclosed for e in expects),
        "level1": sum(e.level1 for e in expects),
        "model_outline": sum(e.model_outline for e in expects),
        "empty": sum(e.empty for e in expects),
        "separator": sum(e.separator for e in expects),
    }
    floors = {"wrapped": 10, "unclosed": 5, "level1": 10, "model_outline": 60, "empty": 5, "separator": 20}
    assert all(counts[name] >= floor for name, floor in floors.items()), counts
    # And not so many empty or one-section samples that the laws check almost nothing.
    counts["several_sections"] = sum(len(e.sections) >= 2 and not e.empty for e in expects)
    assert counts["empty"] <= 30 and counts["several_sections"] >= 120, counts


def leftover_removed(result: str, meta: NotesMeta, expect) -> list[str]:
    """Lines the reply had to lose that are still in the result, header aside."""
    rest = result[len(header(meta)):].split("\n")
    return [line for line in expect.removed if line not in expect.body_lines and line in rest]


# ---------------------------------------------------------------- lecture_date


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2026.10.30 第三课", date(2026, 10, 30)),
        ("2026-11-01 19-30 录音", date(2026, 11, 1)),
        ("第三课 2026_10_30", date(2026, 10, 30)),
        ("2026年10月30日 讲座", date(2026, 10, 30)),
        ("2026年1月5 讲座", date(2026, 1, 5)),
        ("2026.1.5", date(2026, 1, 5)),
        ("２０２６．１０．３０ 全角", date(2026, 10, 30)),
        ("v2026.10.30-final", date(2026, 10, 30)),
        ("2026.10.30 补课 2026.11.02", date(2026, 10, 30)),
        ("2028.02.29 闰年", date(2028, 2, 29)),
    ],
)
def test_lecture_date_reads_the_date_in_the_name(name, expected):
    result = lecture_date(name, FALLBACK)
    assert result == expected
    assert type(result) is date


def test_lecture_date_falls_back_when_the_name_has_no_real_date():
    no_date = [
        "2026.02.29", "2026.13.01", "2026.00.10", "2026.10.32", "2026.10-30", "12026.10.30", "2026.10.301",
        "1999.12.31 千年虫", "2100.01.01", "10.30 课", "20261030", "11-01-2026", "", "录音", "第 3 课",
    ]
    wrong = {name: lecture_date(name, FALLBACK) for name in no_date if lecture_date(name, FALLBACK) != FALLBACK}
    assert wrong == {}
    assert lecture_date("2026.02.30 改到 2026.03.01", FALLBACK) == date(2026, 3, 1)
    assert lecture_date("2000.01.01", FALLBACK) == date(2000, 1, 1)
    assert lecture_date("2099.12.31", FALLBACK) == date(2099, 12, 31)


_FRAGMENTS = ("第三课", " ", "-", ".", "_", "录音", "v", "Claude", "＃", "（下）", "【直播】")


def _written_date(day: date, rng: random.Random) -> str:
    style = rng.choice((".", "-", "_", "年月日", "年月"))
    month = f"{day.month:02d}" if rng.random() < 0.5 else str(day.month)
    dd = f"{day.day:02d}" if rng.random() < 0.5 else str(day.day)
    if style in (".", "-", "_"):
        text = f"{day.year}{style}{month}{style}{dd}"
    elif style == "年月日":
        text = f"{day.year}年{month}月{dd}日"
    else:
        text = f"{day.year}年{month}月{dd}"
    if rng.random() < 0.2:
        text = text.translate(_FULLWIDTH)
    return text


def _filler(rng: random.Random) -> str:
    return "".join(rng.choice(_FRAGMENTS) for _ in range(rng.randint(0, 6)))


def test_lecture_date_round_trips_any_written_date():
    rng = random.Random(20261211)
    for _ in range(300):
        day = date(2000, 1, 1) + timedelta(days=rng.randrange(36525))
        name = _filler(rng) + _written_date(day, rng) + _filler(rng)
        assert lecture_date(name, FALLBACK) == day, repr(name)
    for _ in range(200):
        name = _filler(rng)
        fallback = date(2000, 1, 1) + timedelta(days=rng.randrange(36525))
        assert lecture_date(name, fallback) == fallback, repr((name, fallback))


@pytest.mark.parametrize(
    ("name", "fallback"),
    [
        ("2026.10.30", datetime(2026, 11, 4, 10, 30)),
        ("录音", None),
        ("录音", "2026-11-04"),
        (None, date(2026, 11, 4)),
        (Path("2026.10.30 课"), date(2026, 11, 4)),
    ],
)
def test_lecture_date_refuses_wrong_argument_types(name, fallback):
    with pytest.raises(TypeError):
        lecture_date(name, fallback)


# ---------------------------------------------------------------- overflow


def test_overflow_at_the_edge_of_the_context():
    rules = "规" * 200
    text = "课" * 1000
    assert overflow(rules, text, 1956, 500) == 0
    assert overflow(rules, text, 1955, 500) == 1
    assert overflow(rules, text, 1957, 500) == 0
    assert overflow(rules, "", 956, 500) == 0
    assert overflow(rules, "", 955, 500) == 1
    big = overflow("规" * 1500, "课" * 12000, 16384, 4096)
    assert big == 12000 + 1500 + 4096 + MARGIN - 16384 == 1468
    assert type(big) is int
    assert overflow("", "😀" * 10 + "a\r\n", 1, 1) == 13 + 1 + MARGIN - 1
    assert overflow("", "", 16384, 32768) == 16640


def test_overflow_is_exact_and_monotonic():
    rng = random.Random(20261212)
    alphabet = "规课a 。\n😀\r"

    def text() -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 3000)))

    for _ in range(500):
        r, t, extra = text(), text(), text()
        n, b, d = rng.randint(1, 40000), rng.randint(1, 8000), rng.randint(1, 5000)
        case = repr((len(r), len(t), len(extra), n, b, d))
        base = overflow(r, t, n, b)
        assert base == max(0, len(r) + len(t) + b + MARGIN - n), case
        longer = overflow(r, t + extra, n, b)
        assert longer >= base, case
        if base > 0:
            assert longer - base == len(extra), case
        assert overflow(r + extra, t, n, b) >= base, case
        assert overflow(r, t, n, b + d) >= base, case
        assert overflow(r, t, n + d, b) <= base, case


@pytest.mark.parametrize(
    ("changed", "value", "error"),
    [
        ("num_ctx", 0, ValueError),
        ("num_ctx", -1, ValueError),
        ("num_ctx", True, ValueError),
        ("num_ctx", 16384.0, ValueError),
        ("num_ctx", "16384", ValueError),
        ("reply_budget", 0, ValueError),
        ("reply_budget", -1, ValueError),
        ("reply_budget", True, ValueError),
        ("reply_budget", 1.5, ValueError),
        ("rules", None, TypeError),
        ("rules", b"rules", TypeError),
        ("text", None, TypeError),
        ("text", ["课"], TypeError),
    ],
)
def test_overflow_refuses_bad_arguments(changed, value, error):
    arguments = {"rules": "规", "text": "课", "num_ctx": 16384, "reply_budget": 4096}
    arguments[changed] = value
    with pytest.raises(error) as raised:
        overflow(**arguments)
    if error is ValueError:
        assert changed in str(raised.value)


# ---------------------------------------------------------------- structure: examples

LITERAL_REPLY = """---
tags: [数据库]
---
# 数据库索引入门
日期：2026年10月30日

## 核心宗旨
索引是用空间换时间。

## 本课脉络
- [[#一、为什么需要索引]]
- [[#二、B+ 树]]

## 一、为什么需要索引
没有索引时要全表扫描。
#数据库 #索引

## 二、B+ 树
### 案例
查用户表。

## 案例
```sql
## 不是标题
SELECT 1;
```

## 结尾总结
记住最左前缀。
"""

LITERAL_RESULT = """# 2026.10.30 数据库索引

日期：2026-10-30
校对稿：[[2026.10.30 数据库索引]]
来源：

## 核心宗旨
索引是用空间换时间。

## 本课脉络

- [[#一、为什么需要索引]]
- [[#二、B+ 树]]
- [[#案例（2）]]
- [[#结尾总结]]

## 一、为什么需要索引
没有索引时要全表扫描。
#数据库 #索引

## 二、B+ 树
### 案例
查用户表。

## 案例（2）
```sql
## 不是标题
SELECT 1;
```

## 结尾总结
记住最左前缀。
"""


def test_a_typical_reply_becomes_the_expected_notes():
    assert structure_notes(LITERAL_REPLY, NotesMeta("2026.10.30 数据库索引", date(2026, 10, 30))) == LITERAL_RESULT


@pytest.mark.parametrize(
    ("name", "day", "expected"),
    [
        ("2026.10.30 数据库索引", date(2026, 10, 30),
         {0: "# 2026.10.30 数据库索引", 1: "", 2: "日期：2026-10-30", 3: "校对稿：[[2026.10.30 数据库索引]]", 4: "来源："}),
        ("2026-11-01 19-30 录音", date(2026, 1, 5),
         {0: "# 2026-11-01 19-30 录音", 1: "", 2: "日期：2026-01-05", 3: "校对稿：[[2026-11-01 19-30 录音]]", 4: "来源："}),
        ("第三课：数据库｜索引［上］", date(2026, 10, 30), {3: "校对稿：[[第三课：数据库｜索引［上］]]"}),
        ("C# 入门 [1]", date(2026, 10, 30),
         {0: "# C# 入门 [1]", 3: "校对稿：[C# 入门 \\[1\\]](C%23%20入门%20%5B1%5D.md)"}),
        ("100% 讲透 a^b (上)", date(2026, 10, 30), {3: "校对稿：[100% 讲透 a^b (上)](100%25%20讲透%20a%5Eb%20%28上%29.md)"}),
        ("a|b <c>", date(2026, 10, 30), {3: "校对稿：[a|b <c>](a%7Cb%20%3Cc%3E.md)"}),
    ],
)
def test_the_header_is_written_by_the_program(name, day, expected):
    result = structure_notes("## 核心宗旨\n内容", NotesMeta(name, day))
    lines = result.split("\n")
    assert len(lines) >= 8, result
    assert {index: lines[index] for index in expected} == expected
    assert lines[5] == ""
    assert lines[6] == "## 核心宗旨"
    assert "\r" not in result
    assert result.endswith("\n") and not result.endswith("\n\n")


def test_notes_body_drops_what_the_model_wrote_about_the_notes():
    rows = [
        ("---\ntags: [x]\n---\n## 一\n正文", "## 一\n正文"),
        ("---\ntags: [x]\n## 一\n正文", "---\ntags: [x]\n## 一\n正文"),
        ("---\ntags: [x]\n正文", "---\ntags: [x]\n正文"),
        ("---", "---"),
        ("# 模型标题\n## 一\n正文", "## 一\n正文"),
        ("好的，以下是整理笔记：\n# 模型标题\n## 一\n正文", "好的，以下是整理笔记：\n## 一\n正文"),
        ("# 核心宗旨\n甲\n# 一、背景\n乙", "# 核心宗旨\n甲\n# 一、背景\n乙"),
        ("# 课\n# 核心宗旨\n甲\n# 一、背景\n乙", "# 核心宗旨\n甲\n# 一、背景\n乙"),
        ("# 核心宗旨\n内容", "内容"),
        (
            "# 标题\n日期：2026-10-30\n**日期**：2026年10月30日\n- 来源：网课\n> 校对稿：[[x]]\nDate: 2026-10-30\n"
            "2026年10月30日\n这节课讲索引。\n## 一\n正文",
            "这节课讲索引。\n## 一\n正文",
        ),
        ("## 一\n日期：2026-01-01 那天发布了新版本\n来源：官方文档", "## 一\n日期：2026-01-01 那天发布了新版本\n来源：官方文档"),
        ("```\n日期：2026-01-01\n```\n## 一\n正文", "```\n日期：2026-01-01\n```\n## 一\n正文"),
        ("# 标题\n2026年10月30日这节课讲索引。\n## 一\n正文", "2026年10月30日这节课讲索引。\n## 一\n正文"),
        ("---\ntags: [x]\n...\n## 一\n正文", "## 一\n正文"),
        ("---   \ntags: [x]\n---\n## 一\n正文", "## 一\n正文"),
        (
            "# 标题\n时间：19:30\nsource: x\nSOURCE：y\n__日期__：2026-10-30\n* 来源：网课\n+ 来源：网课\n## 一\n正文",
            "## 一\n正文",
        ),
    ]
    for title in ("一、本课脉络", "**脉络**", "课程脉络：", "本节脉络", "2. 本课脉络", "1、本课脉络", "（二）本课脉络", "本课脉络:"):
        rows.append((f"## 核心宗旨\n甲\n## {title}\n- [[#一]]\n\n## 一\n乙", "## 核心宗旨\n甲\n## 一\n乙"))
    rows += [
        (
            "## 核心宗旨\n甲\n### 本课脉络\n- x\n#### 细目\n- y\n### 要点\n丙\n## 一\n乙",
            "## 核心宗旨\n甲\n#### 细目\n- y\n### 要点\n丙\n## 一\n乙",
        ),
        ("## 脉络\n- a\n## 一\n乙\n## 本课脉络\n- b", "## 一\n乙"),
        ("## 历史脉络\n甲", "## 历史脉络\n甲"),
        ("# 本课脉络\n- [[#一]]\n- [[#二]]\n## 一\n甲\n## 二\n乙", "## 一\n甲\n## 二\n乙"),
        ("## 一\n甲\n# 总结\n乙", "## 一\n甲\n# 总结\n乙"),
        ("# 模型标题\n---\n## 一\n正文\n---\n## 二\n乙", "---\n## 一\n正文\n---\n## 二\n乙"),
        ("---\n## 一\n正文\n---\n乙", "---\n## 一\n正文\n---\n乙"),
    ]
    wrong = [(reply, notes_body(reply, "课"), expected) for reply, expected in rows if notes_body(reply, "课") != expected]
    assert wrong == []


def test_the_outline_lists_the_sections_after_the_core_idea():
    rows = [
        ("## 一\n甲\n## 二\n乙", "## 本课脉络\n\n- [[#一]]\n- [[#二]]\n\n## 一\n甲\n## 二\n乙\n"),
        ("## 背景\n甲\n## 核心宗旨\n乙\n## 一\n丙", "## 背景\n甲\n## 核心宗旨\n乙\n\n## 本课脉络\n\n- [[#一]]\n\n## 一\n丙\n"),
        ("## 一、**核心宗旨**\n乙\n## 二\n丙", "## 一、**核心宗旨**\n乙\n\n## 本课脉络\n\n- [[#二]]\n\n## 二\n丙\n"),
        ("## 核心宗旨\n乙", "## 核心宗旨\n乙\n"),
        ("只是一段话。", "只是一段话。\n"),
        ("### 一\n甲\n### 二\n乙", "### 本课脉络\n\n- [[#一]]\n- [[#二]]\n\n### 一\n甲\n### 二\n乙\n"),
        ("# 一\n甲\n# 二\n乙", "# 本课脉络\n\n- [[#一]]\n- [[#二]]\n\n# 一\n甲\n# 二\n乙\n"),
        ("## 一\n### 1.1\n甲\n## 二\n乙", "## 本课脉络\n\n- [[#一]]\n- [[#二]]\n\n## 一\n### 1.1\n甲\n## 二\n乙\n"),
        ("这节课讲索引。\n\n\n## 一\n甲", "这节课讲索引。\n\n## 本课脉络\n\n- [[#一]]\n\n## 一\n甲\n"),
        ("## 背景\n### 核心宗旨\n乙\n## 一\n丙", "## 本课脉络\n\n- [[#背景]]\n- [[#一]]\n\n## 背景\n### 核心宗旨\n乙\n## 一\n丙\n"),
        ("## 核心宗旨\n甲\n### 要点\n乙\n## 一\n丙", "## 核心宗旨\n甲\n### 要点\n乙\n\n## 本课脉络\n\n- [[#一]]\n\n## 一\n丙\n"),
        ("## 1. 核心宗旨\n乙\n## 2. 背景\n丙", "## 1. 核心宗旨\n乙\n\n## 本课脉络\n\n- [[#2. 背景]]\n\n## 2. 背景\n丙\n"),
        ("## 核心宗旨以外\n甲\n## 一\n乙", "## 本课脉络\n\n- [[#核心宗旨以外]]\n- [[#一]]\n\n## 核心宗旨以外\n甲\n## 一\n乙\n"),
        ("## 核心宗旨（上）\n甲\n## 一\n乙", "## 本课脉络\n\n- [[#核心宗旨（上）]]\n- [[#一]]\n\n## 核心宗旨（上）\n甲\n## 一\n乙\n"),
    ]
    wrong = []
    for reply, expected in rows:
        result = structure_notes(reply, COURSE)
        if not result.startswith(COURSE_HEADER + "\n") or result[len(COURSE_HEADER) + 1:] != expected:
            wrong.append((reply, result, expected))
    assert wrong == []


def test_odd_headings_and_code_blocks_are_read_correctly():
    rows = [
        ("## 一\n#标签 #概念\n## 二", ["一", "二"], ["#标签 #概念"]),
        ("## 一\n```python\n# 注释\n## 也不是\n```\n## 二", ["一", "二"], []),
        ("## 一\n~~~\n## 不是\n~~~\n## 二\n````\n```\n## 不是\n````\n## 三", ["一", "二", "三"], []),
        ("## 一\n````\n```\n## 不是\n", ["一"], []),
        ("## 一\n```\n## 之后\n## 还是代码", ["一"], []),
        ("## 一\n　## 全角空格\n    ## 缩进代码\n## 二", ["一", "二"], ["　## 全角空格", "    ## 缩进代码"]),
        ("##\t制表符\n甲\n## 案例 ##\n乙", ["制表符", "案例"], ["##\t制表符", "## 案例 ##"]),
        ("## 一\n## 二", ["一", "二"], []),
        ("####### 七个井号\n## 一\n## 二", ["一", "二"], ["####### 七个井号"]),
    ]
    wrong = []
    for reply, targets, kept in rows:
        result = structure_notes(reply, COURSE)
        lines = result.split("\n")
        expected = [f"- [[#{t}]]" for t in targets]
        if not result.startswith(COURSE_HEADER) or outline_lines(result) != expected \
                or any(line not in lines for line in kept):
            wrong.append((reply, result, expected))
    assert wrong == []


def test_section_names_become_linkable_and_unique():
    rows = [
        ("## C#\n甲\n## a|b\n乙\n## [重点]\n丙\n## 2^10\n丁", ["## C＃", "## a｜b", "## ［重点］", "## 2＾10"]),
        ("## 案例\n甲\n## 案例\n乙\n## 案例\n丙", ["## 案例", "## 案例（2）", "## 案例（3）"]),
        ("## 案例\n甲\n## 案例\n乙\n## 案例（2）\n丙", ["## 案例", "## 案例（3）", "## 案例（2）"]),
        ("## Prompt\n甲\n## prompt\n乙\n## Ｐｒｏｍｐｔ\n丙", ["## Prompt", "## prompt（2）", "## Ｐｒｏｍｐｔ（3）"]),
        ("## 一\n### 案例\n甲\n## 案例\n乙", ["## 一", "## 案例（2）"]),
        ("##\n甲\n## \n乙", ["## 小节", "## 小节（2）"]),
        ("## ##\n甲\n## 二\n乙", ["## 小节", "## 二"]),
        ("## #\n甲\n## 二\n乙", ["## 小节", "## 二"]),
        ("## C#\n甲\n## C＃\n乙", ["## C＃", "## C＃（2）"]),
        ("## 案例 ##\n甲\n## 案例\n乙", ["## 案例 ##", "## 案例（2）"]),
        ("## 一\n### C#\n甲\n## 二", ["## 一", "## 二"]),
        ("## 课\n甲\n## 二\n乙", ["## 课（2）", "## 二"]),
        ("## 课（2）\n甲\n## 课\n乙", ["## 课（2）", "## 课（3）"]),
    ]
    wrong = []
    for reply, expected in rows:
        result = structure_notes(reply, COURSE)
        titles = [headings(line)[0][2] for line in expected]
        problems = links_resolve_once(result, COURSE.name)
        if not result.startswith(COURSE_HEADER) or section_lines(result) != expected \
                or outline_lines(result) != [f"- [[#{t}]]" for t in titles] or problems:
            wrong.append((reply, result, expected, problems))
    assert wrong == []
    assert "### 案例" in structure_notes("## 一\n### 案例\n甲\n## 案例\n乙", COURSE).split("\n")
    assert "### C#" in structure_notes("## 一\n### C#\n甲\n## 二", COURSE).split("\n")


def test_a_renamed_core_idea_is_still_the_core_when_redone():
    reply = "## 背景\n### 核心宗旨\n乙\n## 核心宗旨\n丙\n## 一\n丁"
    once = structure_notes(reply, COURSE)
    assert once == COURSE_HEADER + "\n## 背景\n### 核心宗旨\n乙\n## 核心宗旨（2）\n丙\n\n## 本课脉络\n\n- [[#一]]\n\n## 一\n丁\n"
    assert structure_notes(once, COURSE) == once


def test_a_section_named_like_the_title_obsidian_reads_is_renamed():
    meta = NotesMeta("课 ##", date(2026, 10, 30))
    result = structure_notes("## 课\n甲\n## 二\n乙", meta)
    assert section_lines(result) == ["## 课（2）", "## 二"]
    assert links_resolve_once(result, "课") == []


def test_line_breaks_bom_and_odd_characters():
    same = "# 课\n\n日期：2026-10-30\n校对稿：[[课]]\n来源：\n\n## 本课脉络\n\n- [[#一]]\n- [[#二]]\n\n## 一\n甲\n## 二\n乙\n"
    replies = {
        "a": "\ufeff# 标题\r\n## 一\r\n甲\r\n## 二\r\n乙\r\n",
        "b": "# 标题\r## 一\r甲\r## 二\r乙",
        "c": "# 标题\n## 一\r\n甲\r## 二\n乙",
        "d": "## 一\n甲\u2028乙\n丙\x0c丁\n尾随空格  \n中间\ufeff字\n## 二\n戊",
        "e": "## 一\n甲\n\n\n\n",
        "f": "## 一\n甲\n\n\n乙\n```\na\n\n\n\nb\n```\n## 二\n丙",
    }
    results = {row: structure_notes(reply, COURSE) for row, reply in replies.items()}
    assert all(result.startswith(COURSE_HEADER) and "\r" not in result for result in results.values()), results
    assert results["a"] == results["b"] == results["c"] == same
    d_lines = results["d"].split("\n")
    assert all(line in d_lines for line in ("甲\u2028乙", "丙\x0c丁", "尾随空格  ", "中间\ufeff字")), d_lines
    assert results["e"].endswith("## 一\n甲\n") and not results["e"].endswith("\n\n")
    assert results["f"].endswith("## 一\n甲\n\n\n乙\n```\na\n\n\n\nb\n```\n## 二\n丙\n")


def test_a_reply_wrapped_in_a_markdown_code_block_is_unwrapped():
    rows = [
        ("```markdown\n# 标题\n## 一\n甲\n```", "## 一\n甲"),
        ("\n```MD\n## 一\n甲\n```\n\n", "## 一\n甲"),
        ("~~~markdown\n## 一\n```sql\nSELECT 1;\n```\n~~~", "## 一\n```sql\nSELECT 1;\n```"),
        ("```\n## 一\n甲\n```", "```\n## 一\n甲\n```"),
        ("好的：\n```markdown\n## 一\n```", "好的：\n```markdown\n## 一\n```"),
        ("```markdown\n## 一\n甲", "```markdown\n## 一\n甲"),
        ("```markdown\n## 一\n甲\n~~~\n## 之后\n```", "## 一\n甲\n~~~\n## 之后"),
    ]
    wrong = [(reply, notes_body(reply, "课"), expected) for reply, expected in rows if notes_body(reply, "课") != expected]
    assert wrong == []
    assert outline_lines(structure_notes("```markdown\n## 一\n甲\n~~~\n## 之后\n```", COURSE)) == ["- [[#一]]"]


@pytest.mark.parametrize(
    "reply",
    [
        "", "  \r\n\t\n", "\ufeff", "# 只有标题", "---\ntags: [x]\n---\n# 标题\n日期：2026-10-30\n来源：网课",
        "## 本课脉络\n- [[#一]]\n- [[#二]]", "```markdown\n\n```", "。，！\n——",
    ],
)
def test_a_reply_with_nothing_in_it_is_refused(reply):
    assert issubclass(EmptyNotes, ValueError)
    with pytest.raises(EmptyNotes):
        structure_notes(reply, META)


@pytest.mark.parametrize("body", ["", "\n\n"])
def test_an_empty_body_is_refused(body):
    with pytest.raises(EmptyNotes):
        assemble_notes(body, META)


@pytest.mark.parametrize("name", ["", "   ", "课\n二", "课\r", "D:/课程/课", "课\\二", "/tmp/课"])
def test_notes_meta_refuses_a_name_that_could_be_a_path(name):
    with pytest.raises(ValueError, match="name"):
        NotesMeta(name, date(2026, 10, 30))


def test_notes_meta_refuses_a_date_with_a_time_or_no_date():
    with pytest.raises(TypeError):
        NotesMeta("课", datetime(2026, 10, 30, 9, 0))
    with pytest.raises(TypeError):
        NotesMeta("课", "2026-10-30")
    with pytest.raises((TypeError, ValueError)):
        NotesMeta(None, date(2026, 10, 30))


# ---------------------------------------------------------------- structure: generated replies


@pytest.mark.parametrize("name", [NAME, "C# 入门 [1]", "课 ##", "2026-11-01 19-30 录音"])
def test_tidying_twice_changes_nothing(name):
    batch = samples(20261213)
    assert_coverage(batch)
    meta = NotesMeta(name, date(2026, 10, 30))
    for reply, expect in batch:
        if expect.empty:
            with pytest.raises(EmptyNotes):
                structure_notes(reply, meta)
            continue
        once = structure_notes(reply, meta)
        assert once.startswith(header(meta)), repr(reply)
        assert leftover_removed(once, meta, expect) == [], repr(reply)
        assert structure_notes(once, meta) == once, repr(reply)
        assert assemble_notes(notes_body(once, meta.name), meta) == once, repr(reply)


def test_the_outline_matches_the_real_sections():
    batch = samples(20261214)
    assert_coverage(batch)
    for reply, expect in batch:
        if expect.empty:
            with pytest.raises(EmptyNotes):
                structure_notes(reply, META)
            continue
        result = structure_notes(reply, META)
        case = repr(reply)
        k, c = len(expect.sections), expect.core
        wanted = k - (c + 1)
        found = headings(result)
        outlines = [(i, lvl) for i, lvl, text in found if text == "本课脉络"]
        if wanted == 0:
            assert outlines == [] and outline_lines(result) == [], case
            continue
        level = expect.sections[0][2]
        assert len(outlines) == 1 and outlines[0][1] == level, case
        sections = [(i, text) for i, lvl, text in found if lvl == level and i != outlines[0][0] and i != 0]
        assert len(sections) == k, case
        lines = result.split("\n")
        at = outlines[0][0] + 1
        assert lines[at] == "", case
        at += 1
        while lines[at].startswith("- [[#"):
            at += 1
        while not lines[at].strip():
            at += 1
        assert at == sections[c + 1][0], case
        targets = [text for _, text in sections[c + 1:]]
        assert outline_lines(result) == [f"- [[#{t}]]" for t in targets], case
        assert all(t and not set(t) & set("#|^[]") for t in targets), case
        assert links_resolve_once(result, NAME) == [], case


def test_body_lines_keep_their_order_and_text():
    batch = samples(20261215)
    assert_coverage(batch)
    for reply, expect in batch:
        if expect.empty:
            with pytest.raises(EmptyNotes):
                structure_notes(reply, META)
            continue
        result = structure_notes(reply, META)
        case = repr(reply)
        lines = result.split("\n")[5:]
        rest = "\n".join(lines)
        outline = [i for i, lvl, text in headings(rest) if text == "本课脉络"]
        if outline:
            start = end = outline[0]
            end += 1
            if end < len(lines) and lines[end] == "":
                end += 1
            while end < len(lines) and lines[end].startswith("- [[#"):
                end += 1
            lines = lines[:start] + lines[end:]
        kept = [line for line in lines if line.strip()]
        assert len(kept) == len(expect.body_lines), case
        titles = {position: (title, level) for position, title, level in expect.sections}
        for position, (got, want) in enumerate(zip(kept, expect.body_lines)):
            if position not in titles:
                assert got == want, case
                continue
            title, level = titles[position]
            parsed = headings(got)
            assert parsed and parsed[0][1] == level, case
            base = title.translate(_LINK_BREAKERS) or "小节"
            numbered = re.fullmatch(re.escape(base) + r"(（\d+）)?", parsed[0][2])
            assert numbered, (case, got)
            if key(title) == key(NAME):
                assert numbered.group(1), (case, got)
        assert leftover_removed(result, META, expect) == [], case


def _drop_section(body: str, level: int, count: int, pick: int) -> tuple[str, str]:
    """``body`` without its section number ``pick``, and that section's title."""
    found = headings(body)
    sections = [(i, text) for i, lvl, text in found if lvl == level]
    assert min(lvl for _, lvl, _ in found) == level and len(sections) == count, body
    start, title = sections[pick]
    end = next((i for i, lvl, _ in found if i > start and lvl <= level), None)
    lines = body.split("\n")
    return "\n".join(lines[:start] + (lines[end:] if end is not None else [])), title


def test_the_outline_follows_the_body_it_is_given():
    rng = random.Random(20261216)
    qualifying = 0
    for _ in range(200):
        reply, expect = make_reply(rng, clean_titles=True)
        k, c = len(expect.sections), expect.core
        if expect.empty or k - (c + 1) < 2:
            continue
        qualifying += 1
        case = repr(reply)
        body = notes_body(reply, NAME)
        whole = assemble_notes(body, META)
        assert structure_notes(reply, META) == whole, case
        body2, title = _drop_section(body, expect.sections[0][2], k, rng.randrange(c + 1, k))
        fewer = assemble_notes(body2, META)
        expected = [line for line in outline_lines(whole) if line != f"- [[#{title}]]"]
        assert len(expected) == len(outline_lines(whole)) - 1, case
        assert outline_lines(fewer) == expected, case
        if not expected:
            assert "本课脉络" not in fewer, case
    assert qualifying >= 40


_DIGEST_SCRIPT = """
import hashlib, random, sys
sys.path.insert(0, sys.argv[1])
from coursescribe.notes import EmptyNotes, structure_notes
from notes_replies import META, make_reply
rng = random.Random(20261217)
digest = hashlib.sha256()
for _ in range(50):
    reply, _ = make_reply(rng)
    try:
        result = structure_notes(reply, META)
    except EmptyNotes:
        result = "EMPTY"
    data = result.encode("utf-8")
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
print(digest.hexdigest())
"""


def _results_here() -> tuple[list[str], list]:
    rng = random.Random(20261217)
    results, expects = [], []
    for _ in range(50):
        reply, expect = make_reply(rng)
        try:
            results.append(structure_notes(reply, META))
        except EmptyNotes:
            results.append("EMPTY")
        expects.append(expect)
    return results, expects


def test_the_same_reply_always_gives_the_same_bytes():
    outputs = []
    for seed in ("1", "2"):
        done = subprocess.run(
            [sys.executable, "-c", _DIGEST_SCRIPT, str(TESTS_DIR)],
            cwd=REPO_ROOT, env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True,
        )
        outputs.append(done.stdout.strip())
    first, expects = _results_here()
    second, _ = _results_here()
    digest = hashlib.sha256()
    for result in first:
        data = result.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    assert outputs[0] == outputs[1] == digest.hexdigest()
    assert first == second
    for result, expect in zip(first, expects):
        if result == "EMPTY":
            continue
        assert result.startswith(header(META)), result
        assert leftover_removed(result, META, expect) == [], result


def test_very_long_replies_finish_quickly():
    many = "".join("## 案例\n" + "索" * 150 + "\n" for _ in range(2000))
    started = time.perf_counter()
    result = structure_notes(many, META)
    assert time.perf_counter() - started < 10
    assert section_lines(result) == ["## 案例"] + [f"## 案例（{n}）" for n in range(2, 2001)]
    assert len(outline_lines(result)) == 2000

    odd = "## 前\n甲\n" + "#" * 200000 + "\n" + " " * 200000 + "## x\n" + "*" * 200000 + "\n## 后\n乙"
    started = time.perf_counter()
    result = structure_notes(odd, META)
    assert time.perf_counter() - started < 10
    assert outline_lines(result) == ["- [[#前]]", "- [[#后]]"]

    code = "## 一\n```\n" + "代码行\n" * 100000
    started = time.perf_counter()
    result = structure_notes(code, META)
    assert time.perf_counter() - started < 10
    assert result.startswith(header(META))


# ---------------------------------------------------------------- finish: quotes and concepts checked against the source

SRC = (
    "# 2026.10.30 数据库索引\n"
    "\n"
    "索引就像一本书的目录，没有它，就只能一页一页地翻。\n"
    "大家记住：索引不是万能的！写入会因此变慢一点。\n"
    "\n"
    "ACID 是事务的四个特性，第 3 个是隔离性。\n"
    "联合索引要遵守最左前缀\n"
    "原则。B+ 树的叶子节点，连成一条链表。\n"
)

FINISH_REPLY = """# 数据库索引入门

## 核心宗旨
索引是用空间换时间。
#索引 #空间换时间

## 一、为什么需要索引
没有索引时要全表扫描。 #全表扫描 #索引

## 引述
> “索引不是万能的”——讲师

> 索引就像一本书的目录……只能一页一页地翻

- “联合索引要遵守最左前缀原则”
- “索引永远让查询更快”——讲师

> 目录

## 结尾总结
记住最左前缀。 #最左前缀
"""

FINISHED = """# 2026.10.30 数据库索引

日期：2026-10-30
校对稿：[[2026.10.30 数据库索引]]
来源：

## 核心宗旨
索引是用空间换时间。
#索引

## 本课脉络

- [[#一、为什么需要索引]]
- [[#引述]]
- [[#结尾总结]]

## 一、为什么需要索引
没有索引时要全表扫描。 #索引

## 引述
> “索引不是万能的”——讲师

> 索引就像一本书的目录……只能一页一页地翻

- “联合索引要遵守最左前缀原则”

## 结尾总结
记住最左前缀。 #最左前缀
"""

FINISHED_WITHOUT_SOURCE = """# 2026.10.30 数据库索引

日期：2026-10-30
校对稿：[[2026.10.30 数据库索引]]
来源：

## 核心宗旨
索引是用空间换时间。

## 本课脉络

- [[#一、为什么需要索引]]
- [[#结尾总结]]

## 一、为什么需要索引
没有索引时要全表扫描。

## 结尾总结
记住最左前缀。
"""

ANCHOR = "> “索引不是万能的”——讲师"
QUOTE_HEAD = f"## 核心宗旨\n保留的一句。\n\n## 引述\n{ANCHOR}\n\n"
QUOTE_TAIL = "\n\n## 结尾总结\n记住最左前缀。"
_AREA_NAMES = ("引述", "六、引述", "**引述**", "引述：")
_TAG_COUNT = re.compile(r"(?<!\S)#[^\s#，。、：；！？]+")


def finish(reply: str, source: str = SRC) -> NotesResult:
    return finish_notes(reply, source, META)


def finish_lines(result: NotesResult) -> list[str]:
    return result.markdown.split("\n")


def redone(result: NotesResult, source: str = SRC) -> NotesResult:
    return finish_notes(result.markdown, source, META)


def quote_rows_wrong(kept: list[str], removed: list[str], head: str = QUOTE_HEAD, source: str = SRC) -> list:
    """Items put after the anchor in a quote section that did not stay or go as stated."""
    wrong = []
    for item in kept:
        result = finish(head + item + QUOTE_TAIL, source)
        if (result.removed_quotes, result.removed_concepts) != (0, 0) \
                or f"## 引述\n{ANCHOR}\n\n{item}\n\n## 结尾总结\n" not in result.markdown:
            wrong.append(("kept", item, result))
    for item in removed:
        result = finish(head + item + QUOTE_TAIL, source)
        lines = finish_lines(result)
        if (result.removed_quotes, result.removed_concepts) != (1, 0) \
                or f"## 引述\n{ANCHOR}\n\n## 结尾总结\n" not in result.markdown \
                or any(line in lines for line in item.split("\n") if line.strip()):
            wrong.append(("removed", item, result))
    return wrong


def outside_code(text: str) -> list[str]:
    """The lines of ``text`` that are not fences or inside fenced code, by the stated rules."""
    kept = []
    fence = None
    for line in text.split("\n"):
        match = _FENCE.match(line)
        if fence is not None:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) \
                    and not match.group(2).strip():
                fence = None
            continue
        if match:
            fence = match.group(1)
            continue
        kept.append(line)
    return kept


def tag_count(text: str) -> int:
    return sum(len(_TAG_COUNT.findall(line)) for line in outside_code(text))


def body_lines(result: str) -> list[str]:
    """Non-blank lines after the five header lines, without the program's outline."""
    lines = result.split("\n")[5:]
    outline = [i for i, _, text in headings("\n".join(lines)) if text == "本课脉络"]
    if outline:
        start = end = outline[0]
        end += 1
        if end < len(lines) and lines[end] == "":
            end += 1
        while end < len(lines) and lines[end].startswith("- [[#"):
            end += 1
        lines = lines[:start] + lines[end:]
    return [line for line in lines if line.strip()]


def quote_areas(result: str, names: tuple[str, ...] = _AREA_NAMES) -> list[list[tuple[str, bool]]]:
    """Each quote section's lines below its heading, with whether each line is a heading."""
    lines = result.split("\n")
    found = headings(result)
    at_heading = {i for i, _, _ in found}
    areas = []
    for number, (index, level, text) in enumerate(found):
        if re.sub(r"（\d+）$", "", text) in names:
            end = next((i for i, lvl, _ in found[number + 1:] if lvl <= level), len(lines))
            areas.append([(lines[i], i in at_heading) for i in range(index + 1, end)])
    return areas


def count_block(lines: list[str], item: tuple[str, ...]) -> int:
    """How often ``item`` stands in ``lines`` as a whole quote: a > quote must not run on."""
    size = len(item)
    quoted = item[0].startswith(">")
    count = 0
    for at in range(len(lines) - size + 1):
        if tuple(lines[at:at + size]) != item:
            continue
        if quoted and ((at > 0 and lines[at - 1].startswith(">"))
                       or (at + size < len(lines) and lines[at + size].startswith(">"))):
            continue
        count += 1
    return count


def quoted_samples(seed: int, count: int = 200):
    rng = random.Random(seed)
    return [make_quoted(rng) for _ in range(count)]


def assert_quote_coverage(batch) -> None:
    expects = [expect for _, _, expect in batch]
    floors = {
        "ellipsis": 30, "multiline": 30, "attribution": 40, "callout": 10, "linebreak": 30, "short": 30,
        "title_only": 15, "decoy": 40, "code_in_quotes": 20, "two_areas": 15, "sub_area": 20, "born_empty": 5,
    }
    counts = {name: sum(name in e.marks for e in expects) for name in floors}
    counts["emptied"] = sum(bool(e.emptied) for e in expects)
    counts["empty"] = sum(e.empty for e in expects)
    counts["with_removed_quotes"] = sum(e.quotes_removed > 0 for e in expects)
    counts["kept_items"] = sum(len(e.kept_items) for e in expects)
    counts["removed_tags"] = sum(e.concepts_removed for e in expects)
    counts["kept_tags"] = sum(e.concepts_found - e.concepts_removed for e in expects)
    floors.update(emptied=15, with_removed_quotes=100, kept_items=400, removed_tags=200, kept_tags=200)
    assert all(counts[name] >= floor for name, floor in floors.items()), counts
    assert 3 <= counts["empty"] <= 20, counts


# ---------------------------------------------------------------- finish: quote examples


def test_each_way_of_writing_a_quote_is_checked():
    kept = [
        "> 索引不是万能的",
        ">索引不是万能的",
        "> 「索引不是万能的」—— 讲师",
        "> 『写入会因此变慢一点』",
        "“索引不是万能的”",
        "- 讲师说：“索引不是万能的”",
        '- "ACID 是事务的四个特性"',
        "> 联合索引要遵守\n> 最左前缀原则",
        "> [!quote] 讲师原话\n> 索引不是万能的",
        "> [!quote]+ 索引永远让查询更快",
        "- “索引不是万能的”和“写入会因此变慢一点”",
        "> 索引不是万能的\n>\n> 写入会因此变慢一点",
        "- “索引不是万能的——讲师",
        ">> 索引不是万能的",
        "> 索引不是万能的\n>\n> ——讲师",
    ]
    removed = [
        "> 讲师说：“索引不是万能的”",
        "- “索引不是万能的”，“索引永远让查询更快”",
        "> 索引不是万能的\n>\n> 索引永远让查询更快",
        "- “讲师说「索引不是万能的」”",
        "- “编造出来的一句话",
        "> [!quote] 讲师原话\n> 索引永远让查询更快",
        "> 联合索引要遵守\n> 写入会因此变慢一点",
        "- “编造的第一句话”和“编造的第二句话”",
    ]
    assert quote_rows_wrong(kept, removed) == []


def test_an_attribution_at_the_end_is_not_checked():
    kept = [
        "> 索引不是万能的——讲师",
        "> 索引不是万能的 —— 李老师（数据库课）",
        "> 索引不是万能的 — 讲师",
        "> 索引不是万能的 -- 讲师",
        "> 索引不是万能的－－讲师",
        "> 索引不是万能的―讲师",
        "> 索引不是万能的\n> ——讲师",
        "> 索引不是万能的——一二三四五六七八九十甲乙",
    ]
    removed = [
        "> 索引不是万能的——一二三四五六七八九十甲乙丙",
        "> 索引不是万能的——讲师，2026",
        "> 索引不是万能的——这是我编出来的一句很长很长的结论",
        "> 索引不是万能的-讲师",
        "> 索引不是万能的——永远加索引——讲师",
        "> 索引不是万能的——讲师说“永远加索引”",
        "> 索引不是万能的——讲师。永远加索引",
        "> 索引不是万能的——讲师！永远加索引",
    ]
    assert quote_rows_wrong(kept, removed) == []


def test_each_part_around_an_ellipsis_is_found_in_order():
    kept = [
        "> 索引就像一本书的目录……只能一页一页地翻",
        "> 索引就像一本书的目录…只能一页一页地翻",
        "> 索引就像一本书的目录...只能一页一页地翻",
        "> 索引就像一本书的目录⋯⋯只能一页一页地翻",
        "> ……索引不是万能的……",
        "> 目录……链表",
        "- “索引就像……地翻”",
    ]
    removed = [
        "> 只能一页一页地翻……索引就像一本书的目录",
        "> 索引就像一本书的目录……永远不用翻",
        "> 链表……链表",
        "> 索……引",
        "> 索引就像一本书的目录..只能一页一页地翻",
    ]
    assert quote_rows_wrong(kept, removed) == []


def test_only_letters_and_digits_are_compared():
    kept = [
        "> 索引不是万能的，写入会因此变慢一点",
        "> 索 引 不 是 万 能 的",
        "> acid 是事务的四个特性",
        "> ＡＣＩＤ是事务的四个特性",
        "> 第３个是隔离性",
        "> 一页一页地翻。大家记住",
        "> 变慢一点。ACID 是事务",
        "> 联合索引要遵守最左前缀原则",
        "> 索引不是",
        "> ACID",
    ]
    removed = [
        "> 第三个是隔离性",
        "> 索引不是萬能的",
        "> 索引不",
        "> 索引",
        "> B+树",
        "> “”——讲师",
        "> 。。。。",
        "> 😀😀😀😀",
        "- “索引”",
    ]
    assert quote_rows_wrong(kept, removed) == []


def test_the_first_level_one_title_of_the_source_is_not_the_source():
    head = f"## 核心宗旨\n保留的一句。\n#数据库\n\n## 引述\n{ANCHOR}\n\n"
    tail = "\n\n索引不是万能的。"
    rows = [
        (SRC, "> 数据库索引", (1, 1)),
        ("数据库索引就像一本书的目录。", "> 数据库索引", (0, 0)),
        ("## 数据库索引\n\n索引就像目录。", "> 数据库索引", (0, 0)),
        ("﻿\n\n# 数据库索引\n索引就像目录。", "> 数据库索引", (1, 1)),
        ("# 课\n\n# 数据库索引\n索引就像目录。", "> 数据库索引", (0, 0)),
        ("#数据库索引\n索引就像目录。", "> 数据库索引", (0, 0)),
        ("# 课\n\n联合索\n引要遵\n守，数据库", "> 联合索引要遵守", (0, 0)),
        ("索引就像一本书的目录没有它就只能一页一页地翻", "> 索引就像一本书的目录", (0, 1)),
    ]
    wrong = []
    for source, item, counts in rows:
        result = finish(head + item + QUOTE_TAIL, source + tail)
        quote_kept = f"## 引述\n{ANCHOR}\n\n{item}\n\n## 结尾总结\n" in result.markdown
        quote_gone = f"## 引述\n{ANCHOR}\n\n## 结尾总结\n" in result.markdown and item not in finish_lines(result)
        tag_kept = "保留的一句。\n#数据库\n" in result.markdown
        tag_gone = "#数据库" not in result.markdown
        if (result.removed_quotes, result.removed_concepts) != counts \
                or not (quote_gone if counts[0] else quote_kept) or not (tag_gone if counts[1] else tag_kept):
            wrong.append((source, result))
    assert wrong == []


def test_repeated_quotes_and_several_quote_sections():
    wrong = []
    a = finish("## 引述\n> 索引不是万能的\n\n> 索引不是万能的\n\n## 结尾总结\n乙。")
    if (a.removed_quotes, a.removed_concepts) != (0, 0) or finish_lines(a).count("> 索引不是万能的") != 2:
        wrong.append(("a", a))
    b = finish("## 引述\n> 索引不是万能的\n\n> 索引永远让查询更快\n\n> 索引永远让查询更快\n\n## 结尾总结\n乙。")
    if b.removed_quotes != 2 or "> 索引永远让查询更快" in finish_lines(b):
        wrong.append(("b", b))
    c = finish("## 引述\n> 编造出来的一句话\n\n## 一\n甲。\n\n## 引述\n> 索引不是万能的")
    if c.removed_quotes != 1 or section_lines(c.markdown) != ["## 一", "## 引述"] \
            or outline_lines(c.markdown) != ["- [[#一]]", "- [[#引述]]"]:
        wrong.append(("c", c))
    d = finish("## 引述\n### 一\n> 编造出来的一句话\n### 二\n> 索引不是万能的\n## 结尾总结\n乙。")
    if d.removed_quotes != 1 or "### 一" not in finish_lines(d) or "> 编造出来的一句话" in finish_lines(d):
        wrong.append(("d", d))
    e = finish("## 引述\n> 索引不是万能的\n## 案例\n> 编造出来的一句话")
    if (e.removed_quotes, e.removed_concepts) != (0, 0) or "> 编造出来的一句话" not in finish_lines(e):
        wrong.append(("e", e))
    f = finish("## 一\n甲。\n### 引述\n> 编造出来的一句话\n## 二\n> 编造出来的另一句话")
    if f.removed_quotes != 1 or "### 引述" in finish_lines(f) or "> 编造出来的另一句话" not in finish_lines(f):
        wrong.append(("f", f))
    assert wrong == []


def test_which_headings_start_a_quote_section():
    quote = "\n> 编造出来的一句话\n> 第二行\n\n## 结尾总结\n乙。"
    areas = [
        "## 引述", "## 六、引述", "## （六）引述", "## **引述**", "## 引述：", "## 引述:", "## 引述（2）",
        "## 引述 ##", "##  引述  ", "## 一\n甲。\n### 引述",
    ]
    replies = [title + quote for title in areas]
    replies += ["# 一\n甲。\n# 引述\n> 编造出来的一句话\n\n# 结尾总结\n乙。", "## 引述 #索引\n> 编造出来的一句话\n\n## 二\n乙。"]
    others = ["## 引述与金句", "## 课堂引述", "## 引用", "## 金句", "## 案例", "　## 引述", "```\n## 引述\n```"]
    wrong = []
    for reply in replies:
        result = finish(reply)
        if (result.removed_quotes, result.removed_concepts) != (1, 0) \
                or "引述" in result.markdown or "编造出来的一句话" in result.markdown:
            wrong.append(("area", reply, result))
    for title in others:
        result = finish(title + quote)
        lines = finish_lines(result)
        if (result.removed_quotes, result.removed_concepts) != (0, 0) \
                or "> 编造出来的一句话" not in lines or "> 第二行" not in lines:
            wrong.append(("not an area", title, result))
    assert wrong == []


def test_blank_lines_after_removing():
    rows = [
        ("## 引述\n\n> 编造出来的一句话\n\n> 索引不是万能的\n\n## 结尾总结\n乙。",
         "in", "## 引述\n\n> 索引不是万能的\n\n## 结尾总结", None, False),
        ("## 引述\n> 编造出来的一句话\n> 也是编造的第二行\n\n> 索引不是万能的\n## 结尾总结\n乙。",
         "in", "## 引述\n\n> 索引不是万能的\n## 结尾总结", None, False),
        ("## 引述\n> 索引不是万能的\n\n- “编造的第一句话”\n- “编造的第二句话”\n\n\n## 结尾总结\n乙。",
         "in", "## 引述\n> 索引不是万能的\n\n## 结尾总结", (2, 0), False),
        ("## 引述\n> 编造的第一句话\n\n> 索引不是万能的\n\n> 编造的第二句话\n## 结尾总结\n乙。",
         "in", "## 引述\n\n> 索引不是万能的\n\n## 结尾总结", (2, 0), False),
        ("## 引述\n> 编造的第一句话\n\n> 编造的第二句话\n\n- 这句话说明索引要慎用\n## 结尾总结\n乙。",
         "in", "## 引述\n\n- 这句话说明索引要慎用\n## 结尾总结", None, False),
        ("## 引述\n> 索引不是万能的\n- “编造出来的一句话”\n> 联合索引要遵守\n\n## 结尾总结\n乙。",
         "in", "## 引述\n> 索引不是万能的\n\n> 联合索引要遵守\n\n## 结尾总结", (1, 0), True),
        ("## 引述\n> 索引不是万能的\n#空间换时间\n> 联合索引要遵守\n\n## 结尾总结\n乙。",
         "in", "## 引述\n> 索引不是万能的\n\n> 联合索引要遵守\n\n## 结尾总结", (0, 1), True),
        ("## 一\n> 甲\n#空间换时间\n> 乙", "end", "## 一\n> 甲\n\n> 乙\n", (0, 1), True),
        ("## 一\n甲。\n\n#空间换时间\n\n乙。", "end", "## 一\n甲。\n\n乙。\n", (0, 1), False),
        ("## 一\n甲\n#编造\n\n## 引述\n> 编造\n## 下\n乙。", "in", "## 一\n甲\n## 下\n乙。\n", (1, 1), True),
    ]
    wrong = []
    for reply, how, part, counts, again in rows:
        result = finish(reply)
        placed = part in result.markdown if how == "in" else result.markdown.endswith(part)
        if not placed or (counts is not None and (result.removed_quotes, result.removed_concepts) != counts) \
                or (again and redone(result) != NotesResult(result.markdown, 0, 0)):
            wrong.append((reply, result))
    assert wrong == []


def test_code_is_never_checked():
    rows = [
        ("## 引述\n```text\n> 编造出来的一句话\n“编造出来的一句话”\n#桃花\n```\n\n## 结尾总结\n乙。",
         ["```text", "> 编造出来的一句话", "“编造出来的一句话”", "#桃花", "```"]),
        ("## 引述\n~~~\n> 编造出来的一句话\n", ["## 引述", "~~~", "> 编造出来的一句话"]),
        ("## 引述\n- 用 `“编造出来的一句话”` 举例\n\n## 结尾总结\n乙。", ["- 用 `“编造出来的一句话”` 举例"]),
        ("## 一\n用 `#include` 引入头文件\n\n## 二\n乙。", ["用 `#include` 引入头文件"]),
        ("## 一\n```python\n# 注释 #桃花\nx = 1  #灯塔\n```", ["```python", "# 注释 #桃花", "x = 1  #灯塔", "```"]),
    ]
    wrong = []
    for reply, lines in rows:
        result = finish(reply)
        if (result.removed_quotes, result.removed_concepts) != (0, 0) or "\n".join(lines) not in result.markdown:
            wrong.append((reply, result))
    # Outside code the same quote goes: the rows above keep it only because it is code.
    made_up = finish("## 核心宗旨\n索引要慎用。\n\n## 引述\n> 编造出来的一句话\n\n## 结尾总结\n乙。")
    if made_up.removed_quotes != 1:
        wrong.append(("outside code", made_up))
    assert wrong == []


# ---------------------------------------------------------------- finish: concept examples


def test_bad_concept_tags_are_removed_and_good_ones_kept():
    rows = [
        ("#索引 #最左前缀", "#索引 #最左前缀", 0),
        ("#索引 #空间换时间", "#索引", 1),
        ("#空间换时间 #索引", "#索引", 1),
        ("#空间换时间", None, 1),
        ("- #空间换时间", None, 1),
        ("> #空间换时间", None, 1),
        ("概念： #空间换时间", "概念：", 1),
        ("没有索引时要全表扫描。 #全表扫描 #索引", "没有索引时要全表扫描。 #索引", 1),
        ("#空间换时间 #空间换时间", None, 2),
        ("#acid #ＡＣＩＤ #隔离性 #B_树 #B-树 #最左/前缀 #链表", "#acid #ＡＣＩＤ #隔离性 #B_树 #B-树 #最左/前缀 #链表", 0),
        ("#2026年 #数据库", None, 2),
        ("#索引　#空间换时间", "#索引", 1),
        ("甲\t#空间换时间", "甲", 1),
        ("#最左/桃花", None, 1),
        ("#索引-桃花", None, 1),
        ("#B+桃花", "#B+桃花", 0),
    ]
    wrong = []
    for line, expected, removed in rows:
        result = finish("## 一\n甲。\n" + line + "\n\n## 二\n乙。")
        part = "## 一\n甲。\n\n## 二\n" if expected is None else "## 一\n甲。\n" + expected + "\n\n## 二\n"
        if (result.removed_quotes, result.removed_concepts) != (0, removed) or part not in result.markdown:
            wrong.append((line, result))
    assert wrong == []


def test_hashes_that_are_not_concept_tags_are_left_alone():
    lines = [
        "C# 入门", "#3 号索引", "#2026", "见 [[#一]]", "https://example.com/a#桃花", "扫描。#桃花", "＃桃花", "#",
        "a#桃花", "#+桃花", "用 `x #桃花` 举例",
    ]
    wrong = []
    for line in lines:
        reply = "## 一\n甲。\n" + line + "\n\n## 二\n乙。"
        result = finish(reply)
        # A lone # is an empty level-one heading, which 12-2 already renames; it must not change further.
        if result != NotesResult(structure_notes(reply, META), 0, 0) or (line != "#" and line not in finish_lines(result)):
            wrong.append((line, result))
    made_up = finish("## 一\n甲。\n#空间换时间\n\n## 二\n乙。")
    if made_up.removed_concepts != 1 or "#空间换时间" in made_up.markdown:
        wrong.append(("a real tag", made_up))
    assert wrong == []


def test_tags_in_headings_and_quotes():
    wrong = []
    a = finish("## 一 #空间换时间\n甲。\n\n## 二\n乙。")
    if (a.removed_quotes, a.removed_concepts) != (0, 1) or section_lines(a.markdown) != ["## 一", "## 二"] \
            or outline_lines(a.markdown) != ["- [[#一]]", "- [[#二]]"]:
        wrong.append(("a", a))
    b = finish("## #空间换时间\n甲。\n\n## 二\n乙。")
    if (b.removed_quotes, b.removed_concepts) != (0, 1) or section_lines(b.markdown) != ["## 小节", "## 二"] \
            or outline_lines(b.markdown) != ["- [[#小节]]", "- [[#二]]"]:
        wrong.append(("b", b))
    c = finish("## 引述\n> 索引不是万能的 #索引\n\n## 二\n乙。")
    if (c.removed_quotes, c.removed_concepts) != (0, 0) or "> 索引不是万能的 #索引" not in finish_lines(c):
        wrong.append(("c", c))
    d = finish("## 引述\n> 索引不是万能的 #空间换时间\n\n## 二\n乙。")
    if (d.removed_quotes, d.removed_concepts) != (0, 1) or "> 索引不是万能的" not in finish_lines(d):
        wrong.append(("d", d))
    e = finish("## 引述\n> 索引不是万能的\n> #空间换时间\n\n## 二\n乙。")
    if (e.removed_quotes, e.removed_concepts) != (0, 1) or "## 引述\n> 索引不是万能的\n\n## 二\n" not in e.markdown:
        wrong.append(("e", e))
    f = finish("## 引述\n> 编造出来的一句话 #索引\n\n## 一\n甲。")
    if (f.removed_quotes, f.removed_concepts) != (1, 0) or "编造" in f.markdown:
        wrong.append(("f", f))
    g = finish("## 一\n#索引，#空间换时间。\n\n## 二\n乙。")
    if (g.removed_quotes, g.removed_concepts) != (0, 0) or "#索引，#空间换时间。" not in finish_lines(g):
        wrong.append(("g", g))
    h = finish("## 引述\n> 编造出来的一句话 #桃花\n\n## 二\n乙。")
    if (h.removed_quotes, h.removed_concepts) != (1, 1) or "编造" in h.markdown:
        wrong.append(("h", h))
    assert wrong == []


def test_what_12_2_drops_is_not_counted():
    reply = (
        "---\ntags: [桃花, 索引]\n---\n# 数据库索引入门\n日期：2026-10-30\n\n## 本课脉络\n- [[#引述]]\n\n"
        "## 一\n甲。 #桃花 #索引\n\n## 引述\n> 编造出来的一句话"
    )
    assert finish(reply) == NotesResult(header(META) + "\n## 本课脉络\n\n- [[#一]]\n\n## 一\n甲。 #索引\n", 1, 1)


def test_a_line_that_would_become_something_else_goes_whole():
    rows = [
        ("## 一\n甲。\n#空间换时间 # 小结\n乙。\n\n## 二\n丙。", "## 一\n甲。\n乙。\n\n## 二\n", ["- [[#一]]", "- [[#二]]"]),
        ("## 一\n甲。\n#空间换时间 ```\n## 二\n乙。", "## 一\n甲。\n## 二\n乙。\n", ["- [[#一]]", "- [[#二]]"]),
        ("## 引述\n> 索引不是万能的\n\n#空间换时间 > 编造出来的一句话\n\n## 结尾总结\n乙。",
         "## 引述\n> 索引不是万能的\n\n## 结尾总结\n", None),
        ("这节课讲索引。\n#空间换时间 日期：2026-10-30\n## 一\n甲。", "来源：\n\n这节课讲索引。\n", None),
        ("这节课讲索引。\n日期 #空间换时间：2026-10-30\n## 一\n甲。", "来源：\n\n这节课讲索引。\n", None),
        ("这节课讲索引。\n#空间换时间 2026-10-30\n## 一\n甲。", "来源：\n\n这节课讲索引。\n", None),
        ("## 一\n#空间换时间 来源：官方文档\n## 二\n乙。", "## 一\n来源：官方文档\n## 二\n", None),
        ("## 一\n甲。\n## 脉络 #空间换时间\n- [[#一]]\n来源：书\n## 二\n乙。", "## 一\n甲。\n## 二\n", None),
        ("## 一\n#空间换时间 #索引 正文\n## 二\n乙。", "## 一\n#索引 正文\n## 二\n", None),
    ]
    wrong = []
    for reply, part, outline in rows:
        result = finish(reply)
        if (result.removed_quotes, result.removed_concepts) != (0, 1) or part not in result.markdown \
                or (outline is not None and outline_lines(result.markdown) != outline) \
                or redone(result) != NotesResult(result.markdown, 0, 0):
            wrong.append((reply, result))
    lines = finish_lines(finish("这节课讲索引。\n#空间换时间 2026-10-30\n## 一\n甲。"))
    if "2026-10-30" in lines:
        wrong.append(("date line", lines))
    assert wrong == []


# ---------------------------------------------------------------- finish: whole results


def test_a_typical_reply_is_finished_exactly():
    result = finish_notes(FINISH_REPLY, SRC, META)
    assert result == NotesResult(FINISHED, 2, 2)
    assert type(result.removed_quotes) is int and type(result.removed_concepts) is int


def test_the_way_the_source_is_stored_does_not_matter():
    same = [
        SRC.replace("\n", "\r\n"),
        SRC.replace("\n", "\r"),
        "﻿" + SRC,
        SRC.replace("，", ", ").replace("。", ".\n"),
    ]
    assert [finish_notes(FINISH_REPLY, source, META) for source in same] == [NotesResult(FINISHED, 2, 2)] * 4
    nothing = ["", "   \r\n\t", "﻿", "# 2026.10.30 数据库索引", "# 2026.10.30 数据库索引\n\n"]
    results = [finish_notes(FINISH_REPLY, source, META) for source in nothing]
    assert results == [NotesResult(FINISHED_WITHOUT_SOURCE, 5, 5)] * 5


def test_quote_sections_left_empty_go_and_an_empty_result_is_refused():
    wrong = []
    a = finish("## 核心宗旨\n索引要慎用。\n\n## 引述\n> 编造出来的一句话\n\n> 索引永远让查询更快\n\n## 结尾总结\n记住最左前缀。")
    expected = header(META) + "\n## 核心宗旨\n索引要慎用。\n\n## 本课脉络\n\n- [[#结尾总结]]\n\n## 结尾总结\n记住最左前缀。\n"
    if a != NotesResult(expected, 2, 0):
        wrong.append(("a", a))
    b = finish("## 一\n甲。\n\n## 引述\n\n## 二\n乙。")
    if b.removed_quotes or b.removed_concepts or "引述" in b.markdown \
            or outline_lines(b.markdown) != ["- [[#一]]", "- [[#二]]"]:
        wrong.append(("b", b))
    c = finish("## 引述\n以下是课上的原话：\n> 编造出来的一句话\n\n## 二\n乙。")
    if (c.removed_quotes, c.removed_concepts) != (1, 0) or "## 引述\n以下是课上的原话：\n\n## 二\n" not in c.markdown:
        wrong.append(("c", c))
    d = finish("## 引述\n### 讲师原话\n> 编造出来的一句话\n\n## 二\n乙。")
    if (d.removed_quotes, d.removed_concepts) != (1, 0) or "引述" in d.markdown or "讲师原话" in d.markdown:
        wrong.append(("d", d))
    e = finish("## 案例\n查用户表。\n### 引述\n> 编造出来的一句话\n## 二\n乙。")
    if (e.removed_quotes, e.removed_concepts) != (1, 0) or "## 案例\n查用户表。\n## 二\n" not in e.markdown:
        wrong.append(("e", e))
    f = finish("## 引述\n#空间换时间\n\n## 二\n乙。")
    if (f.removed_quotes, f.removed_concepts) != (0, 1) or "引述" in f.markdown:
        wrong.append(("f", f))
    g = finish("这节课讲索引。\n## 引述\n> 编造出来的一句话")
    if g != NotesResult(header(META) + "\n这节课讲索引。\n", 1, 0):
        wrong.append(("g", g))
    i = finish("## 引述 #空间换时间\n> 编造出来的一句话\n\n## 二\n乙。")
    if (i.removed_quotes, i.removed_concepts) != (1, 1) or "引述" in i.markdown \
            or redone(i) != NotesResult(i.markdown, 0, 0):
        wrong.append(("i", i))
    assert wrong == []
    for reply in ("# 数据库索引入门\n## 引述\n> 编造出来的一句话", "## 引述\n#空间换时间", "## 引述\n> 索引"):
        with pytest.raises(EmptyNotes):
            finish(reply)


def test_a_line_dropped_for_turning_into_a_fence_does_not_hide_later_headings():
    result = finish("## 引述\n> 索引不是万能的\n\n#空间换时间 ```\n## 二\n> 编造出来的一句话")
    assert (result.removed_quotes, result.removed_concepts) == (0, 1)
    assert "## 二\n> 编造出来的一句话\n" in result.markdown
    assert redone(result) == NotesResult(result.markdown, 0, 0)


def test_a_line_of_good_tags_inside_a_quote_is_not_a_blank_line():
    kept = finish("## 引述\n> 索引不是\n> #索引\n> 万能的\n\n## 结尾总结\n乙。")
    assert (kept.removed_quotes, kept.removed_concepts) == (0, 0)
    assert "## 引述\n> 索引不是\n> #索引\n> 万能的\n\n## 结尾总结\n" in kept.markdown
    gone = finish("## 引述\n> 索引不是万能的\n> #索引\n> ACID 是事务的四个特性\n\n## 二\n乙。")
    assert (gone.removed_quotes, gone.removed_concepts) == (1, 0)
    assert "## 引述" not in gone.markdown and "> #索引" not in finish_lines(gone)


@pytest.mark.parametrize("fence", ["```", "```python"])
def test_any_code_block_keeps_a_quote_section(fence):
    result = finish(f"## 一\n甲。\n\n## 引述\n{fence}\n```\n\n## 二\n乙。")
    assert (result.removed_quotes, result.removed_concepts) == (0, 0)
    assert f"## 引述\n{fence}\n```\n" in result.markdown
    assert redone(result) == NotesResult(result.markdown, 0, 0)


def test_two_quote_sections_of_the_same_name():
    kept = finish("## 引述\n> 索引不是万能的\n\n## 一\n甲。\n\n## 引述\n> 联合索引要遵守")
    assert (kept.removed_quotes, kept.removed_concepts) == (0, 0)
    assert section_lines(kept.markdown) == ["## 引述", "## 一", "## 引述（2）"]
    assert "- [[#引述（2）]]" in outline_lines(kept.markdown)
    assert redone(kept) == NotesResult(kept.markdown, 0, 0)
    one_gone = finish("## 引述\n> 索引不是万能的\n\n## 一\n甲。\n\n## 引述\n> 编造出来的一句话")
    assert (one_gone.removed_quotes, one_gone.removed_concepts) == (1, 0)
    assert section_lines(one_gone.markdown) == ["## 引述", "## 一"]


def test_the_way_the_reply_is_stored_does_not_matter():
    replies = [FINISH_REPLY.replace("\n", "\r\n"), FINISH_REPLY.replace("\n", "\r"), "\ufeff" + FINISH_REPLY]
    assert [finish(reply) for reply in replies] == [NotesResult(FINISHED, 2, 2)] * 3


def test_a_quote_section_inside_a_quote_section():
    inner_only = finish("## 引述\n### 引述\n> 编造出来的一句话\n## 二\n乙。")
    assert (inner_only.removed_quotes, inner_only.removed_concepts) == (1, 0)
    assert "引述" not in inner_only.markdown
    both = finish("## 引述\n> 索引不是万能的\n### 引述\n> 编造出来的一句话\n## 二\n乙。")
    assert (both.removed_quotes, both.removed_concepts) == (1, 0)
    assert "### 引述" not in finish_lines(both) and "## 引述\n> 索引不是万能的\n" in both.markdown
    assert redone(both) == NotesResult(both.markdown, 0, 0)


def test_more_ways_of_writing_quotes():
    kept = [
        "    > 编造出来的一句话",  # four spaces: not a > line, and no quotation marks
        "- 用 ``“编造`的话”`` 举例",
        "> [!quote]- 讲师原话\n> 索引不是万能的",
        "> 索引就像一本书的目录⋯只能一页一页地翻",
        "> 索引不是万能的――讲师",
    ]
    removed = ["   > 编造出来的一句话", "- `“编造出来的一句话”"]
    assert quote_rows_wrong(kept, removed) == []


@pytest.mark.parametrize(
    ("changed", "value"),
    [
        ("reply", None), ("reply", b"## x"), ("reply", Path("a.md")),
        ("source", None), ("source", SRC.encode()), ("source", Path("a.md")), ("source", ["索引"]),
        ("meta", None), ("meta", "2026.10.30 数据库索引"), ("meta", {"name": "课", "date": date(2026, 10, 30)}),
    ],
)
def test_finish_notes_refuses_wrong_argument_types(changed, value):
    arguments = {"reply": FINISH_REPLY, "source": SRC, "meta": META}
    arguments[changed] = value
    with pytest.raises(TypeError) as raised:
        finish_notes(**arguments)
    assert changed in str(raised.value)


def test_the_result_cannot_be_changed():
    result = finish_notes(FINISH_REPLY, SRC, META)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.removed_quotes = 0
    assert result == NotesResult(result.markdown, 2, 2)
    assert isinstance(result.markdown, str)


# ---------------------------------------------------------------- finish: generated replies


def _headings_match(line: str, title: str, level: int) -> bool:
    parsed = headings(line)
    return bool(parsed) and parsed[0][1] == level and re.fullmatch(re.escape(title) + r"(（\d+）)?", parsed[0][2]) is not None


def test_finished_notes_keep_and_drop_what_the_generator_says():
    batch = quoted_samples(20261221)
    assert_quote_coverage(batch)
    for reply, source, expect in batch:
        case = repr((reply, source))
        if expect.empty:
            with pytest.raises(EmptyNotes):
                finish_notes(reply, source, META)
            continue
        result = finish_notes(reply, source, META)
        assert (result.removed_quotes, result.removed_concepts) == (expect.quotes_removed, expect.concepts_removed), case
        assert result.markdown.startswith(header(META)), case
        got = body_lines(result.markdown)
        assert len(got) == len(expect.lines), (case, got, expect.lines)
        titles = {position: (title, level) for position, title, level in expect.sections}
        for position, (line, wanted) in enumerate(zip(got, expect.lines)):
            if position in titles:
                assert _headings_match(line, *titles[position]), (case, line)
            else:
                assert line == wanted, (case, line, wanted)
        lines = finish_lines(result)
        for item, _ in expect.removed_items:
            assert all(line in expect.lines or line not in lines for line in item), (case, item)
        for item in dict.fromkeys(expect.kept_items):
            assert count_block(lines, item) == expect.kept_items.count(item), (case, item)
        for title in dict.fromkeys(expect.emptied):
            kept = sum(1 for other, _, gone in expect.areas if other == title and not gone)
            same = [text for _, _, text in headings(result.markdown) if re.fullmatch(re.escape(title) + r"(（\d+）)?", text)]
            assert len(same) == kept, (case, title)
            if not kept:
                assert not any(re.fullmatch(r"- \[\[#" + re.escape(title) + r"(（\d+）)?\]\]", line)
                               for line in outline_lines(result.markdown)), (case, title)
        assert links_resolve_once(result.markdown, NAME) == [], case
        assert tag_count(result.markdown) == expect.concepts_found - expect.concepts_removed, case


def test_kept_and_removed_add_up_to_what_was_found():
    rng = random.Random(20261222)
    checked = 0
    for _ in range(200):
        reply, source, expect = make_quoted(rng)
        if expect.empty or expect.empty_without_source:
            continue
        checked += 1
        case = repr((reply, source))
        real = finish_notes(reply, source, META)
        none = finish_notes(reply, "", META)
        assert (none.removed_quotes, none.removed_concepts) == (expect.quotes_found, expect.concepts_found), case
        areas = [[line for line, _ in area] for area in quote_areas(real.markdown)]
        kept = sum(count_block(area, item) for item in dict.fromkeys(expect.kept_items) for area in areas)
        assert real.removed_quotes + kept == none.removed_quotes, case
        assert real.removed_concepts + tag_count(real.markdown) == none.removed_concepts, case
        # With nothing to back them every quote goes, so a quote section stays only for its intro line or code
        # (the matrix's "no quote section at all" contradicts its own rule that code and other lines keep one).
        for area in quote_areas(none.markdown):
            assert any(line == INTRO or _FENCE.match(line) for line, _ in area), (case, area)
            text = "\n".join(line for line, _ in area)
            assert not any(line.startswith(">") or any(mark in line for mark in "“「『")
                           for line in outside_code(text)), (case, area)
    assert checked >= 150


def test_finishing_twice_removes_nothing_more():
    batch = quoted_samples(20261223)
    assert_quote_coverage(batch)
    for reply, source, expect in batch:
        case = repr((reply, source))
        if expect.empty:
            with pytest.raises(EmptyNotes):
                finish_notes(reply, source, META)
            continue
        once = finish_notes(reply, source, META)
        assert (once.removed_quotes, once.removed_concepts) == (expect.quotes_removed, expect.concepts_removed), case
        assert finish_notes(once.markdown, source, META) == NotesResult(once.markdown, 0, 0), case


def _extra(rng: random.Random) -> str:
    return "".join(rng.choice(SOURCE_SENTENCES + INVENTED) + rng.choice(("。", "！", "，", ",", " ", "")) for _ in range(rng.randint(0, 6)))


def _untagged(line: str) -> tuple[str, frozenset[str]]:
    """A line as compared when tags may have stayed: its text without tags and blanks, and its tags."""
    return "".join(_TAG_COUNT.sub("", line).split()), frozenset(_TAG_COUNT.findall(line))


def _is_subsequence(small: list[tuple[str, frozenset[str]]], big: list[tuple[str, frozenset[str]]]) -> bool:
    """Whether each line of ``small`` comes, in order, in ``big`` with the same text and at least its tags."""
    rest = iter(big)
    return all(any(text == other and tags <= more for other, more in rest) for text, tags in small)


def test_a_longer_source_never_removes_more():
    rng = random.Random(20261224)
    saved = 0
    for _ in range(200):
        reply, source, expect = make_quoted(rng)
        if expect.empty:
            continue
        case = repr((reply, source))
        before = finish_notes(reply, source, META)
        kept = [key for key in (_untagged(line) for line in body_lines(before.markdown) if not headings(line)) if key != ("", frozenset())]
        extra = _extra(rng)
        title = f"# {NAME}"
        inserted = source.replace(title, title + "\n" + extra + "\n", 1) if expect.has_title else extra + "\n" + source
        for longer in (source + "\n" + extra, inserted):
            after = finish_notes(reply, longer, META)
            assert after.removed_quotes <= before.removed_quotes, (case, longer)
            assert after.removed_concepts <= before.removed_concepts, (case, longer)
            now = [_untagged(line) for line in body_lines(after.markdown) if not headings(line)]
            assert _is_subsequence(kept, now), (case, longer)
        missing = sorted({made_up for _, needs in expect.removed_items if needs for made_up in needs})
        if not missing:
            continue
        saved += 1
        made_up = rng.choice(missing)
        after = finish_notes(reply, source + "\n\n" + made_up + "。", META)
        rescued = [item for item, needs in expect.removed_items if needs is not None and needs <= {made_up}]
        assert after.removed_quotes == expect.quotes_removed - len(rescued), case
        assert after.removed_concepts <= before.removed_concepts, case
        lines = finish_lines(after)
        assert all(count_block(lines, item) >= 1 for item in rescued), case
    assert saved >= 60


def _rewritten(rng: random.Random, source: str) -> str:
    """``source`` with its punctuation, blanks, line breaks, width and case changed; its title kept."""
    text = source.lstrip("﻿").replace("\r\n", "\n")
    lines = text.split("\n")
    first = next(index for index, line in enumerate(lines) if line.strip())
    head = ""
    if lines[first].startswith("# "):
        head = "\n".join(lines[:first + 1]) + "\n"
        text = "\n".join(lines[first + 1:])
    out = []
    for ch in text:
        if not content_chars(ch):
            if rng.random() < 0.5:
                ch = rng.choice(("，", ",", "。", ".", " ", "\n", "\n\n", ""))
        else:
            if ch.isascii() and rng.random() < 0.3:
                ch = ch.swapcase()
            if ch.isascii() and ch.isalnum() and rng.random() < 0.3:
                ch = chr(ord(ch) + 0xFEE0)
        out.append(ch)
        if rng.random() < 0.05:
            out.append(rng.choice((" ", "\n")))
    rewritten = (head + "".join(out)).replace("\n", rng.choice(("\r\n", "\r")))
    return "﻿" + rewritten if rng.random() < 0.5 else rewritten


def test_how_the_source_is_written_does_not_change_the_result():
    rng = random.Random(20261225)
    batch = [make_quoted(rng) for _ in range(200)]
    assert_quote_coverage(batch)
    for reply, source, expect in batch:
        if expect.empty:
            continue
        case = repr((reply, source))
        result = finish_notes(reply, source, META)
        assert (result.removed_quotes, result.removed_concepts) == (expect.quotes_removed, expect.concepts_removed), case
        other = _rewritten(rng, source)
        assert finish_notes(reply, other, META) == result, (case, other)


FULL = "# 2026.10.30 数据库索引\n\n索引不是万能的。标签和概念。"
NO_CONCEPT = "# 2026.10.30 数据库索引\n\n索引不是万能的。标签。"


def test_a_source_that_backs_everything_changes_nothing():
    rng = random.Random(20261226)
    checked = with_tags = 0
    for _ in range(200):
        reply, expect = make_reply(rng)
        if expect.empty:
            with pytest.raises(EmptyNotes):
                structure_notes(reply, META)
            with pytest.raises(EmptyNotes):
                finish_notes(reply, FULL, META)
            continue
        plain = structure_notes(reply, META)
        areas = quote_areas(plain, ("引述",))
        if any(not content_chars("\n".join(line for line, heading in area if not heading)) for area in areas):
            continue
        if any(a.startswith(">") and b.startswith(">") for area in areas for (a, _), (b, _) in zip(area, area[1:])):
            continue
        checked += 1
        case = repr(reply)
        assert finish_notes(reply, FULL, META) == NotesResult(plain, 0, 0), case
        lines = plain.split("\n")
        changed = lines.count("#标签 #概念")
        with_tags += changed > 0
        expected = "\n".join("#标签" if line == "#标签 #概念" else line for line in lines)
        assert finish_notes(reply, NO_CONCEPT, META) == NotesResult(expected, 0, changed), case
    assert checked >= 120 and with_tags >= 50, (checked, with_tags)


_FINISH_DIGEST_SCRIPT = """
import hashlib, random, sys
sys.path.insert(0, sys.argv[1])
from coursescribe.notes import EmptyNotes, finish_notes
from notes_quotes import META, make_quoted
rng = random.Random(20261227)
digest = hashlib.sha256()
for _ in range(30):
    reply, source, _ = make_quoted(rng)
    try:
        result = finish_notes(reply, source, META)
        text = f"{result.markdown}|{result.removed_quotes}|{result.removed_concepts}"
    except EmptyNotes:
        text = "EMPTY"
    data = text.encode("utf-8")
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
print(digest.hexdigest())
"""


def test_finishing_gives_the_same_bytes_whatever_the_hash_seed():
    outputs = []
    for seed in ("1", "2"):
        done = subprocess.run(
            [sys.executable, "-c", _FINISH_DIGEST_SCRIPT, str(TESTS_DIR)],
            cwd=REPO_ROOT, env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True,
        )
        outputs.append(done.stdout.strip())
    rng = random.Random(20261227)
    digest = hashlib.sha256()
    for _ in range(30):
        reply, source, expect = make_quoted(rng)
        try:
            result = finish_notes(reply, source, META)
        except EmptyNotes:
            text = "EMPTY"
        else:
            assert (result.removed_quotes, result.removed_concepts) == (expect.quotes_removed, expect.concepts_removed)
            text = f"{result.markdown}|{result.removed_quotes}|{result.removed_concepts}"
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    assert outputs[0] == outputs[1] == digest.hexdigest()


def _timed(reply: str, source: str) -> NotesResult:
    started = time.perf_counter()
    result = finish_notes(reply, source, META)
    assert time.perf_counter() - started < 10, reply[:80]
    return result


def test_very_long_sources_and_replies_finish_quickly():
    rng = random.Random(20261228)
    pieces = []
    size = 0
    while size < 500_000:
        piece = rng.choice(SOURCE_SENTENCES) + rng.choice(("。", "，", "\n", "\n\n"))
        pieces.append(piece)
        size += len(piece)
    source = "".join(pieces)
    said = content_chars(source)
    items = [f"> {said[at:at + rng.randint(4, 20)]}" for at in (rng.randrange(len(said) - 20) for _ in range(1000))]
    items += [f"> {rng.choice(INVENTED)}" for _ in range(1000)]
    rng.shuffle(items)
    tags = ["#" + rng.choice(GOOD_TAGS) for _ in range(2500)] + ["#" + rng.choice(BAD_TAGS) for _ in range(2500)]
    rng.shuffle(tags)
    sections = [f"## 小节{n}\n正文。\n" + "\n".join(" ".join(tags[at:at + 5]) for at in range(n * 25, n * 25 + 25, 5))
                for n in range(200)]
    reply = "## 引述\n" + "\n\n".join(items) + "\n\n" + "\n\n".join(sections)
    assert (lambda r: (r.removed_quotes, r.removed_concepts))(_timed(reply, source)) == (1000, 2500)

    parts = [said[at:at + 4] for at in range(0, 20000 * 6, 6)]
    long_quote = "## 引述\n> " + "……".join(parts) + "\n\n## 结尾\n乙。"
    assert _timed(long_quote, source).removed_quotes == 0

    assert _timed("## 一\n甲。\n\n## 引述\n> " + "索" * 5000 + "引", "索" * 500_000).removed_quotes == 1

    odd = ["“" * 200_000, "—" * 200_000, "> " + ">" * 200_000, "." * 200_000, "#" * 200_000, "#" + "a" * 200_000,
           "- " + "“a”" * 100_000]
    for line in odd:
        assert isinstance(_timed(QUOTE_HEAD + line + QUOTE_TAIL, SRC), NotesResult)

    for item in ("> 索引" + "—" * 200_000 + "讲", "> " + "-" * 200_000, "> 索引" + "." * 200_000 + "x", "> " + "——索" * 70_000):
        assert _timed(QUOTE_HEAD + item + QUOTE_TAIL, SRC).removed_quotes == 1, item[:20]


# ---------------------------------------------------------------- source_body and build_prompt (12-4)
# docs/plans/M2-notes-4-rules-tests.md

PROMPT_HEAD = "\n\n【校对稿开始】下面是这节课的原文，笔记里的引述和概念只能来自这里。\n"
PROMPT_TAIL = "\n【校对稿结束】请按上面的整理规则写这节课的笔记，只返回笔记正文。"

_SOURCE_BODIES = {
    "a-title": ("# 课\n\n索引就像目录。\n", "索引就像目录。"),
    "no-title": ("索引就像目录。\n\n写入会变慢。", "索引就像目录。\n\n写入会变慢。"),
    "a-second-level-heading-is-text": ("## 课\n索引。", "## 课\n索引。"),
    "blank-lines-and-a-bom-first": ("﻿\n\n# 课\r\n索引。\r\n", "索引。"),
    "only-the-first-title": ("# 课\n\n# 第二个\n索引。", "# 第二个\n索引。"),
    "no-space-after-the-hash": ("#课\n索引。", "#课\n索引。"),
    "four-spaces-is-not-a-heading": ("    # 缩进\n索引。", "    # 缩进\n索引。"),
    "an-empty-title": ("#\n索引。", "索引。"),
    "everything-else-as-written": (
        "# 课\n\n第一段  \n\t缩进\n\n\n第二段 同一行\n```\n# 代码\n```",
        "第一段  \n\t缩进\n\n\n第二段 同一行\n```\n# 代码\n```",
    ),
    "carriage-returns-only": ("# 课\r\r索引。\r", "索引。"),
    "the-first-line-keeps-its-tab": ("# 课\n\n\t第一段\n末段", "\t第一段\n末段"),
    "a-bom-inside-stays": ("# 课\n索﻿引", "索﻿引"),
    "only-one-leading-bom-goes": ("﻿﻿# 课\n正文", "﻿# 课\n正文"),
}


@pytest.mark.parametrize("name", list(_SOURCE_BODIES))
def test_source_body_keeps_everything_but_the_title_and_the_blank_edges(name):
    """S1."""
    source, expected = _SOURCE_BODIES[name]
    assert source_body(source) == expected


@pytest.mark.parametrize("source", ["# 课", "# 课\n\n  \n", ""])
def test_source_body_of_only_a_title_is_empty(source):
    """S1 (i)."""
    assert source_body(source) == ""


def test_a_typical_prompt():
    """B1."""
    rules = "## 核心宗旨\n用一两句话说这节课最重要的意思。"
    source = "﻿# 2026.10.30 数据库索引\r\n\r\n索引就像一本书的目录。\r\n\r\n## 写入\r\n写入会因此变慢一点。  \r\n\r\n"
    prompt = build_prompt(rules, source)
    assert prompt == (
        "## 核心宗旨\n用一两句话说这节课最重要的意思。"
        "\n\n【校对稿开始】下面是这节课的原文，笔记里的引述和概念只能来自这里。\n"
        "索引就像一本书的目录。\n\n## 写入\n写入会因此变慢一点。  "
        "\n【校对稿结束】请按上面的整理规则写这节课的笔记，只返回笔记正文。"
    )
    assert "\r" not in prompt
    assert "2026.10.30 数据库索引" not in prompt
    assert not prompt.startswith("﻿")
    assert prompt[prompt.index(PROMPT_HEAD) + len(PROMPT_HEAD)] != "﻿"


@pytest.mark.parametrize("name", list(_SOURCE_BODIES))
def test_the_source_goes_into_the_prompt_as_source_body_gives_it(name):
    """B2."""
    source, expected = _SOURCE_BODIES[name]
    assert build_prompt("规则", source) == "规则" + PROMPT_HEAD + expected + PROMPT_TAIL


_BLANKS = ("", "  ", "\t", "　")
_LONG_PARAGRAPH = "索引就像目录。" * 3000  # over 20 000 characters


@dataclasses.dataclass(frozen=True)
class _Lesson:
    """A proofread file in parts, so the same lesson can be written several ways."""

    before: tuple[str, ...]  # blank lines before the title, or before the body without one
    title: str | None
    after: tuple[str, ...]  # blank lines after the title
    body: tuple[str, ...]  # first and last lines are not blank; the first is not a level 1 heading
    trailing: tuple[str, ...]
    newline: str
    bom: bool

    def text(self) -> str:
        head = (f"# {self.title}", *self.after) if self.title is not None else ()
        return ("﻿" if self.bom else "") + self.newline.join((*self.before, *head, *self.body, *self.trailing))

    def expected_body(self) -> str:
        return "\n".join(self.body)


def _lesson_paragraph(rng: random.Random) -> str:
    out = "".join(_random_word(rng) + rng.choice(("", " ", "，", "。")) for _ in range(rng.randint(1, 8)))
    extra = rng.random()
    if extra < 0.15:
        out += " " + _random_word(rng)
    elif extra < 0.3:
        out += "﻿" + _random_word(rng)
    elif extra < 0.45:
        out += rng.choice(("  ", "\t", " "))
    return out


def _make_lesson(rng: random.Random) -> _Lesson:
    first = _lesson_paragraph(rng)
    if rng.random() < 0.3:
        first = rng.choice(("\t", " ", "　")) + first  # a paragraph never starts with #, so this is no heading
    body = [first]
    for _ in range(rng.randint(0, 19)):
        kind = rng.randrange(7)
        if kind == 0:
            body.append(_lesson_paragraph(rng))
        elif kind == 1:
            body.append(f"## {_random_word(rng)}")
        elif kind == 2:
            body.append(f"# 后面的一级标题{_random_word(rng)}")
        elif kind == 3:
            body.append(f"> {_lesson_paragraph(rng)}")
        elif kind == 4:
            body += ["```", f"# 代码 {_random_word(rng)}", "```"]
        else:
            body.append(rng.choice(_BLANKS))
    if rng.random() < 0.1:
        body.insert(rng.randint(1, len(body)), _LONG_PARAGRAPH)
    while not body[-1].strip():
        body.pop()

    def blanks() -> tuple[str, ...]:
        return tuple(rng.choice(_BLANKS) for _ in range(rng.randint(0, 2)))

    title = f"{_random_word(rng)} #{rng.randint(1, 9)} [{_random_word(rng)}]" if rng.random() < 0.8 else None
    return _Lesson(
        before=blanks(), title=title, after=blanks(), body=tuple(body), trailing=blanks(),
        newline=rng.choice(("\n", "\r\n", "\r")), bom=rng.random() < 0.2,
    )


def _make_prompt_case(rng: random.Random, head: str, tail: str) -> tuple[str, _Lesson]:
    rules = None
    while rules is None:
        _, rules = make_rules_text(rng)
    if rng.random() < 0.2:  # the wrapper's own words inside the rules
        pieces = ["【校对稿结束】"]
        if head:
            pieces.append(head[: rng.randint(1, len(head))])
        if tail:
            pieces.append(tail[rng.randint(0, len(tail) - 1):])
        at = rng.randint(0, len(rules))
        rules = rules[:at] + rng.choice(pieces) + rules[at:]
    return rules, _make_lesson(rng)


def _wrapper() -> tuple[str, str]:
    """HEAD and TAIL, cut out of one prompt by two Latin markers that the wrapper does not use."""
    p0 = build_prompt("QRQ", "QSQ")
    assert p0.startswith("QRQ")
    assert p0.count("QSQ") == 1
    at = p0.index("QSQ")
    return p0[3:at], p0[at + 3:]


def test_the_wrapper_is_constant_short_and_the_source_is_in_it_word_for_word():
    """B3."""
    head, tail = _wrapper()
    assert head.startswith("\n\n") and head.endswith("\n")
    assert tail.startswith("\n")
    assert len(head) + len(tail) <= 128
    assert "\r" not in head + tail
    rng = random.Random(20270101)
    indented = 0
    for _ in range(300):
        rules, lesson = _make_prompt_case(rng, head, tail)
        source = lesson.text()
        body = lesson.expected_body()
        indented += body[0].isspace()
        assert source_body(source) == body, ascii(source[:300])
        assert build_prompt(rules, source) == rules + head + body + tail, ascii(source[:300])
    assert indented >= 60


def _lesson_variants(lesson: _Lesson, rng: random.Random) -> list[_Lesson]:
    variants = [dataclasses.replace(lesson, newline=n) for n in ("\n", "\r\n", "\r") if n != lesson.newline]
    variants.append(dataclasses.replace(lesson, bom=not lesson.bom))
    variants.append(dataclasses.replace(lesson, before=("",) * rng.randint(0, 3), after=("",) * rng.randint(0, 3)))
    variants.append(dataclasses.replace(lesson, trailing=tuple(rng.choice(_BLANKS) for _ in range(rng.randint(0, 3)))))
    if lesson.title is not None:
        variants.append(dataclasses.replace(lesson, title="另一个名字 #2"))
        variants.append(dataclasses.replace(lesson, title=None))
    return variants


def test_how_the_source_is_written_does_not_change_the_prompt():
    """B4."""
    rng = random.Random(20270102)
    head, tail = _wrapper()
    for _ in range(200):
        rules, lesson = _make_prompt_case(rng, head, tail)
        prompt = build_prompt(rules, lesson.text())
        for variant in _lesson_variants(lesson, rng):
            assert build_prompt(rules, variant.text()) == prompt, (ascii(lesson.text()[:200]), ascii(variant.text()[:200]))


_SOURCE = "# 课\n正文"


@pytest.mark.parametrize(
    ("rules", "source", "error", "word"),
    [
        (None, _SOURCE, TypeError, "rules"),
        (b"rules", _SOURCE, TypeError, "rules"),
        ("规则", None, TypeError, "source"),
        ("规则", ["正文"], TypeError, "source"),
        ("规则", Path("课.md"), TypeError, "source"),
        ("", _SOURCE, ValueError, "rules"),
        (" \n\t", _SOURCE, ValueError, "rules"),
        ("。。。", _SOURCE, ValueError, "rules"),
        (None, "", TypeError, "rules"),  # (f) the type first, then the content
        ("", None, TypeError, "source"),  # (g) every type check before any content check
    ],
)
def test_build_prompt_refuses_wrong_arguments_in_order(rules, source, error, word):
    """B5 (a)(b)(c)(f)(g)."""
    with pytest.raises(error) as caught:
        build_prompt(rules, source)
    assert type(caught.value) is error
    assert word in str(caught.value)


def test_braces_and_percent_signs_are_kept_as_written():
    """B5 (d)."""
    assert build_prompt("规则 {source} {0} %s", "# 课\n正文 {}") == "规则 {source} {0} %s" + PROMPT_HEAD + "正文 {}" + PROMPT_TAIL


@pytest.mark.parametrize("source", ["", "# 课", "﻿# 课\r\n\r\n  \r\n", "# 课\n\n。。。——", "# 课\n```\n```"])
def test_an_empty_source_is_its_own_error(source):
    """B5 (e)."""
    assert issubclass(EmptySource, ValueError)
    with pytest.raises(EmptySource) as caught:
        build_prompt("规则", source)
    assert str(caught.value) == "这份校对稿是空的"


def test_the_length_estimate_covers_the_real_prompt():
    """B6: a written-down check that overflow(rules, source_body(s), ...) with its margin covers build_prompt."""
    head, tail = _wrapper()
    cases = random.Random(20270101)
    sizes = random.Random(20270103)
    fits = misses = 0
    for _ in range(300):
        rules, lesson = _make_prompt_case(cases, head, tail)
        source = lesson.text()
        body = source_body(source)
        budget = sizes.randint(1, 8000)
        num_ctx = len(rules) + len(body) + budget + 256 + sizes.randint(-50, 50)
        if overflow(rules, body, num_ctx, budget) == 0:
            fits += 1
            assert len(build_prompt(rules, source)) + budget + 128 <= num_ctx
        else:
            misses += 1
    assert fits >= 100 and misses >= 100


def test_very_long_rules_and_sources_make_a_prompt_quickly():
    """B7."""
    rules = "规则。\n" * 50_000
    body = "\n".join(["索引就像目录。"] * 100_000)
    prompts = []
    for newline in ("\n", "\r\n"):
        source = "# 课" + newline * 2 + ("索引就像目录。" + newline) * 100_000
        started = time.perf_counter()
        prompt = build_prompt(rules, source)
        assert time.perf_counter() - started < 2, ascii(newline)
        assert prompt == rules + PROMPT_HEAD + body + PROMPT_TAIL
        prompts.append(prompt)
    assert prompts[0] == prompts[1]
