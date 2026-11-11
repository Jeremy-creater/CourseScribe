"""Seeded random model replies for organised notes, with what tidying them must keep and drop.

The generator writes down its own expectations while it builds a reply and never
calls the code under test. ``content_chars`` is only used to decide whether the
kept lines say anything at all.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from datetime import date

from coursescribe.notes import NotesMeta
from coursescribe.text import content_chars

NAME = "2026.10.30 数据库索引"
META = NotesMeta(NAME, date(2026, 10, 30))

SECTION_TITLES = (
    "核心宗旨", "一、为什么需要索引", "案例", "案例", "案例（2）", "类比", "引述", "提示词",
    "C#", "a|b", "[重点]", "2^10", "Prompt", "prompt", "Ｐｒｏｍｐｔ", "", "结尾总结", "历史脉络", NAME,
)
CLEAN_TITLES = (
    "核心宗旨", "一、为什么需要索引", "案例", "类比", "引述", "提示词", "结尾总结", "历史脉络",
    "二、B+ 树", "三、最左前缀",
)
CLEAN_SUBTITLES = ("要点", "细节", "小结")
MODEL_TITLES = ("# 数据库索引", "# 2026.10.30 数据库索引", "# C# 入门")
HEADER_LINES = (
    "日期：2026-10-30", "**日期**：2026年10月30日", "- 来源：网课", "> 校对稿：[[x]]", "2026-10-30",
    "Date: 2026-10-30",
)
OUTLINE_TITLES = ("本课脉络", "一、本课脉络", "**脉络**", "课程脉络：", "2. 本课脉络", "（二）本课脉络")
FRONT_MATTER = ("---", "tags: [x]", "date: 2026-10-30", "---")
OPENING = "好的，以下是整理笔记："
PARAGRAPHS = ("索引能让查询变快。", "写入会因此变慢一点。", "先看执行计划，再决定加不加索引。")
OUTLINE_LINK = "- [[#x]]"
_PROGRAM_LINE = "\x00program"  # marks a dropped line the program itself may write again


@dataclass
class Expect:
    body_lines: list[str]  # kept non-blank lines, in order, before any renaming
    sections: list[tuple[int, str, int]]  # (position in body_lines, original title, level)
    core: int  # index into sections of the first 核心宗旨, or -1
    removed: list[str]
    empty: bool
    wrapped: bool
    unclosed: bool
    level1: bool
    model_outline: bool
    separator: bool


def _heading(level: int, title: str) -> str:
    return "#" * level + (" " + title if title else "")


def _content(rng: random.Random, level: int, fences: tuple[str, ...], subtitles: tuple[str, ...]) -> list[str]:
    """One item of a section's content, as lines."""
    kind = rng.randrange(13)
    if kind == 0:
        return [rng.choice(PARAGRAPHS)]
    if kind == 1:
        return ["#标签 #概念"]
    if kind == 2:
        return [""]
    if kind == 3:
        return ["> “索引不是万能的”——讲师"]
    if kind == 4:
        fence = rng.choice(fences)
        return [fence, "## 不是标题", "# 也不是", "日期：2026-01-01", fence]
    if kind == 5:
        return [_heading(level + 1, rng.choice(subtitles))]
    if kind == 6:
        return ["　## 全角空格开头"]
    if kind == 7:
        return ["    ## 缩进代码"]
    if kind == 8:
        return ["- 列表项"]
    if kind == 9:
        return ["行尾带两个空格  "]
    if kind == 10:
        return ["甲\u2028乙"]
    if kind == 11:
        return ["日期：2026-01-01"]
    return ["---"]


def _name_key(title: str) -> str:
    """What makes two section names the same name: width, case, and the characters links cannot hold."""
    text = unicodedata.normalize("NFKC", title.translate(str.maketrans("#|^[]", "＃｜＾［］"))) or "小节"
    return text.casefold()


def _without_same_names(rng: random.Random, titles: list[str]) -> list[str]:
    """``titles`` with a different pick wherever a name came up twice (same-name sections are merged)."""
    seen: set[str] = set()
    for index, title in enumerate(titles):
        while _name_key(title) in seen:
            title = rng.choice(SECTION_TITLES)
        seen.add(_name_key(title))
        titles[index] = title
    return titles


def make_reply(rng: random.Random, clean_titles: bool = False, duplicates: bool = False) -> tuple[str, Expect]:
    """A model reply and what tidying it must give."""
    roll = rng.random()
    level = 2 if roll < 0.7 else 3 if roll < 0.9 else 1
    empty_sample = rng.random() < 0.05
    wrapped = rng.random() < 0.1
    fences = ("~~~",) if wrapped else ("```", "~~~")

    # Each line is (text, kept, section title or None).
    head: list[tuple[str, bool, str | None]] = []
    if rng.random() < 0.3:
        head += [(line, False, None) for line in FRONT_MATTER]
    if not empty_sample and rng.random() < 0.1:
        head.append((OPENING, True, None))
    if level != 1 and rng.random() < 0.7:
        head.append((rng.choice(MODEL_TITLES), False, None))
        if not empty_sample and rng.random() < 0.3:
            head.append(("---", True, None))
    for _ in range(rng.randint(0, 3)):
        head.append((rng.choice(HEADER_LINES), False, None))

    blocks: list[list[tuple[str, bool, str | None]]] = []
    if not empty_sample:
        count = rng.randint(2, 12) if level == 1 else rng.randint(0, 12)
        if clean_titles:
            titles = rng.sample(CLEAN_TITLES, min(count, len(CLEAN_TITLES)))
            subtitles = CLEAN_SUBTITLES
        else:
            titles = [rng.choice(SECTION_TITLES) for _ in range(count)]
            if not duplicates:
                titles = _without_same_names(rng, titles)
            subtitles = SECTION_TITLES
            while level == 1 and titles[0] == NAME:
                titles[0] = rng.choice(SECTION_TITLES)
        for title in titles:
            block = [(_heading(level, title), True, title)]
            for _ in range(rng.randint(0, 4)):
                block += [(line, True, None) for line in _content(rng, level, fences, subtitles)]
            blocks.append(block)

    unclosed = not empty_sample and rng.random() < 0.08
    tail = [(rng.choice(fences), True, None), ("## 之后", True, None)] if unclosed else []

    model_outline = False
    if rng.random() < 0.5:
        # Only before a section heading, or at the very end. Not right before the
        # unclosed fence: removing an outline runs to the next heading outside
        # code, and there is none after that fence, so it would take the code too.
        positions = list(range(len(blocks))) + ([] if unclosed else [len(blocks)])
        if positions:
            model_outline = True
            depth = rng.choice((level, level + 1))
            outline_heading = _heading(depth, rng.choice(OUTLINE_TITLES))
            # Dropped, but not listed as removed when it is exactly the line the
            # program writes for its own outline: that line may come back.
            same_as_program = outline_heading == _heading(level, "本课脉络")
            outline = [(outline_heading, False, _PROGRAM_LINE if same_as_program else None)]
            outline += [(OUTLINE_LINK, False, None) for _ in range(rng.randint(0, 4))]
            blocks.insert(rng.choice(positions), outline)

    lines = head + [line for block in blocks for line in block] + tail
    if wrapped:
        lines = [("```markdown", False, None)] + lines + [("```", False, None)]

    body_lines: list[str] = []
    sections: list[tuple[int, str, int]] = []
    removed: list[str] = []
    for text, kept, title in lines:
        if not kept:
            if title != _PROGRAM_LINE:
                removed.append(text)
            continue
        if not text.strip():
            continue
        body_lines.append(text)
        if title is not None:
            sections.append((len(body_lines) - 1, title, level))
    core = next((index for index, (_, title, _) in enumerate(sections) if title == "核心宗旨"), -1)

    mode = rng.choice(("\n", "\r\n", "mixed"))
    pieces: list[str] = []
    for index, (text, _, _) in enumerate(lines):
        pieces.append(text)
        if index < len(lines) - 1 or rng.random() < 0.5:
            pieces.append(rng.choice(("\n", "\r\n")) if mode == "mixed" else mode)
    reply = "".join(pieces)
    if rng.random() < 0.2:
        reply = "\ufeff" + reply

    expect = Expect(
        body_lines=body_lines,
        sections=sections,
        core=core,
        removed=removed,
        empty=content_chars("\n".join(body_lines)) == "",
        wrapped=wrapped,
        unclosed=unclosed,
        level1=level == 1,
        model_outline=model_outline,
        separator="---" in body_lines,
    )
    return reply, expect
