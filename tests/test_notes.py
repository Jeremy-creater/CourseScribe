"""Organised notes: header, outline and clean-up of the model's reply, as pure string logic."""

from __future__ import annotations

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
    NotesMeta,
    assemble_notes,
    lecture_date,
    notes_body,
    overflow,
    structure_notes,
)
from notes_replies import META, NAME, make_reply

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


@pytest.mark.parametrize("name", ["", "   ", "课\n二", "课\r", "D:/课程/课", "课\\二", "/Users/x/课"])
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
