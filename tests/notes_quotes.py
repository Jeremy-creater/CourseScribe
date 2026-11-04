"""Seeded random replies with quotes and concept tags, the source they are checked against, and what must stay.

The generator writes down its own expectations while it builds a reply and never
calls the code under test. ``content_chars`` is only used to pick fragments of the
source and to check the pools below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from coursescribe.text import content_chars
from notes_replies import META, NAME

SOURCE_SENTENCES = (
    "索引就像一本书的目录没有它就只能一页一页地翻",
    "ACID 是事务的四个特性",
    "其中第 3 个特性是隔离性",
    "联合索引要遵守最左前缀原则",
    "B+ 树的叶子节点连成一条链表",
    "先看执行计划再决定加不加索引",
    "写入数据时索引也要跟着更新",
    "主键索引保证每一行都能被唯一找到",
    "区分度低的列不适合单独建索引",
    "覆盖索引可以避免回表查询",
    "范围查询之后的列用不上联合索引",
    "索引太多会占用更多的磁盘空间",
)
INVENTED = (
    "春天园里开满桃花",
    "海边灯塔照亮夜航船",
    "山谷溪水流向远方",
    "老猫趴窗台晒暖阳",
    "雪落松林鸟儿归巢",
    "微风吹过麦田泛起金浪",
    "渔夫摇橹小舟穿芦苇",
    "孩童放风筝跑过草坡",
)
GOOD_TAGS = ("索引", "最左前缀", "ACID", "acid", "隔离性", "链表", "B_树", "最左/前缀")
BAD_TAGS = ("桃花", "灯塔", "夜航船", "数据库")
REPLY_TITLE = "# 数据库索引入门"
INTRO = "以下是课上的原话："

_ENDINGS = ("。", "！", "？", "，")
_REPLACEMENTS = ("，", "、", "。", " ", "")
_ELLIPSES = ("……", "…", "...")
_PLAIN_TITLES = ("核心宗旨", "一、为什么需要索引", "案例", "类比", "提示词", "结尾总结")
_AREA_TITLES = ("引述", "六、引述", "**引述**", "引述：")
_SENTENCE_PREFIX = object()
_TAG_PREFIXES = ("", "- ", _SENTENCE_PREFIX, "概念： ")


def _check_pools() -> None:
    """The pools must keep the promises the expectations rely on."""
    for sentence in SOURCE_SENTENCES:
        assert len(content_chars(sentence)) >= 10, sentence
        assert not any(part in sentence for part in ("数据库", "2026", "—", "-", "…", ".", "#", ">")), sentence
        assert not any(mark in sentence for mark in "“”「」『』\""), sentence
    known = set(content_chars("".join(SOURCE_SENTENCES) + NAME))
    for sentence in INVENTED:
        chars = content_chars(sentence)
        # Disjoint characters: no piece of an invented sentence can be in the source.
        assert not set(chars) & known, sentence
        # No repeated character, and none inside another: half of one never meets the source or another.
        assert len(set(chars)) == len(chars) >= 4, sentence
        assert not any(chars in content_chars(other) for other in INVENTED if other != sentence), sentence
    said = content_chars("".join(SOURCE_SENTENCES))
    assert all(content_chars(tag) in said for tag in GOOD_TAGS)
    assert all(content_chars(tag) not in said for tag in BAD_TAGS)
    assert all(content_chars(tag) in content_chars("".join(INVENTED)) for tag in BAD_TAGS[:3])


_check_pools()


@dataclass
class QuoteExpect:
    lines: list[str]  # kept non-blank lines of the body, in order, before any renaming
    sections: list[tuple[int, str, int]]  # (position in lines, original title, level) of each kept heading
    quotes_found: int
    quotes_removed: int
    concepts_found: int
    concepts_removed: int
    kept_items: list[tuple[str, ...]]  # every kept quote, as its lines
    removed_items: list[tuple[tuple[str, ...], frozenset[str] | None]]  # lines and the sentences that would save it
    areas: list[tuple[str, int, bool]]  # (title, level, emptied) of each quote area
    empty: bool
    empty_without_source: bool
    marks: set[str] = field(default_factory=set)
    has_title: bool = True

    @property
    def emptied(self) -> list[str]:
        return [title for title, _, gone in self.areas if gone]


# Each line of a reply is (text, expected text or None when it must go, (title, level) when a heading).
Line = tuple


class _Source:
    """One generated source: its text, and its body as written with plain line breaks."""

    def __init__(self, rng: random.Random) -> None:
        order = rng.sample(SOURCE_SENTENCES, len(SOURCE_SENTENCES))
        paragraphs = []
        at = 0
        while at < len(order):
            count = rng.randint(1, 3)
            group = []
            for sentence in order[at:at + count]:
                if rng.random() < 0.3:
                    cut = rng.randint(1, len(sentence) - 1)
                    sentence = sentence[:cut] + "\n" + sentence[cut:]
                group.append(sentence + rng.choice(_ENDINGS))
            paragraphs.append("".join(group))
            at += count
        self.body = "\n\n".join(paragraphs)
        self.has_title = rng.random() < 0.9
        text = self.body
        if self.has_title:
            text = f"# {NAME}\n\n" + text
            if rng.random() < 0.1:
                text = "\n" * rng.randint(1, 2) + text
        newline = rng.choice(("\n", "\r\n"))
        text = text.replace("\n", newline)
        if rng.random() < 0.1:
            text = "﻿" + text
        self.text = text
        self.positions = [index for index, ch in enumerate(self.body) if content_chars(ch)]

    def fragment(self, rng: random.Random, low: int, high: int, start: int = 0, stop: int | None = None) -> tuple[str, int]:
        """The raw text of ``low``-``high`` consecutive content characters, and where it ends among them."""
        stop = len(self.positions) if stop is None else stop
        count = rng.randint(low, min(high, stop - start))
        first = rng.randint(start, stop - count)
        return self.body[self.positions[first]:self.positions[first + count - 1] + 1], first + count


def _disguise(rng: random.Random, raw: str) -> str:
    """``raw`` with its punctuation, spacing, width and case changed, and its letters and digits kept."""
    out = []
    gap = False
    for ch in raw:
        if not content_chars(ch):
            if not gap:
                out.append(rng.choice(_REPLACEMENTS))
            gap = True
            continue
        gap = False
        if ch.isascii() and rng.random() < 0.3:
            ch = ch.swapcase()
        if ch.isascii() and ch.isalnum() and rng.random() < 0.3:
            ch = chr(ord(ch) + 0xFEE0)
        out.append(ch)
        if rng.random() < 0.08:
            out.append(" ")
    return "".join(out)


def _good_text(rng: random.Random, source: _Source, marks: set[str]) -> str:
    raw, _ = source.fragment(rng, 4, 20)
    if "\n" in raw:
        marks.add("linebreak")
    return _disguise(rng, raw)


def _split(rng: random.Random, text: str) -> list[str]:
    """``text`` cut into 2-3 pieces that each have something other than blanks."""
    for _ in range(20):
        cuts = sorted(rng.sample(range(1, len(text)), min(rng.randint(1, 2), len(text) - 1)))
        pieces = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
        if all(piece.strip() for piece in pieces):
            return pieces
    return [text]


_GOOD_FORMS = tuple(range(13)) + (8,)  # quotes split over lines twice as often, to meet the coverage floor


def _good_item(rng: random.Random, source: _Source, marks: set[str]) -> list[str]:
    form = rng.choice(_GOOD_FORMS)
    if form == 11:
        marks.add("ellipsis")
        size = len(source.positions)
        first, end = source.fragment(rng, 2, 12, 0, size - 2)
        second, _ = source.fragment(rng, max(2, 4 - len(content_chars(first))), 12, end, size)
        if "\n" in first + second:
            marks.add("linebreak")
        return [f"> {_disguise(rng, first)}{rng.choice(_ELLIPSES)}{_disguise(rng, second)}"]
    text = _good_text(rng, source, marks)
    if form in (1, 2, 3, 4, 9):
        marks.add("attribution")
    if form == 0:
        return [f"> {text}"]
    if form == 1:
        return [f"> “{text}”——讲师"]
    if form == 2:
        return [f"> {text} —— 李老师"]
    if form == 3:
        return [f"> {text} — 讲师"]
    if form == 4:
        return [f"> {text} -- 讲师"]
    if form == 5:
        return [f"- “{text}”"]
    if form == 6:
        return [f"「{text}」"]
    if form == 7:
        return [f"- 讲师说：“{text}”"]
    if form == 8:
        pieces = _split(rng, text)
        if len(pieces) > 1:
            marks.add("multiline")
        return [f"> {piece}" for piece in pieces]
    if form == 9:
        return [f"> {text}", "> ——讲师"]
    if form == 10:
        marks.add("callout")
        return ["> [!quote] 讲师原话", f"> {text}"]
    return [f"- “{text}”和“{_good_text(rng, source, marks)}”"]


_BAD_FORMS = tuple(range(8)) + (3,)  # too-short quotes twice as often, to meet the coverage floor


def _bad_item(rng: random.Random, source: _Source, marks: set[str]) -> tuple[list[str], frozenset[str] | None]:
    form = rng.choice(_BAD_FORMS)
    made_up = rng.choice(INVENTED)
    if form == 0:
        return [f"> {made_up}"], frozenset((made_up,))
    if form == 1:
        return [f"- “{made_up}”——讲师"], frozenset((made_up,))
    if form == 2:
        text = _disguise(rng, source.fragment(rng, 4, 12)[0])
        return [f"> {text}{made_up[rng.randint(1, len(made_up) - 2):]}"], None
    if form == 3:
        marks.add("short")
        return [f"> {_disguise(rng, source.fragment(rng, 1, 3)[0])}"], None
    if form == 4:
        if source.has_title:
            marks.add("title_only")
        return ["> 数据库索引"], None
    text = _disguise(rng, source.fragment(rng, 4, 12)[0])
    if form == 5:
        marks.add("ellipsis")
        return [f"> {text}{rng.choice(_ELLIPSES)}{made_up}"], frozenset((made_up,))
    if form == 6:
        return [f"- “{text}”，“{made_up}”"], frozenset((made_up,))
    return [f"> {text}", ">", f"> {made_up}"], frozenset((made_up,))


def _tag_line(rng: random.Random) -> tuple[str, str | None, int, int]:
    """A line of tags, what it must become (None: dropped), and how many tags it has and loses."""
    tags = [(rng.choice(GOOD_TAGS), True) if rng.random() < 0.5 else (rng.choice(BAD_TAGS), False)
            for _ in range(rng.randint(1, 4))]
    prefix = rng.choice(_TAG_PREFIXES)
    if prefix is _SENTENCE_PREFIX:
        prefix = rng.choice(SOURCE_SENTENCES) + "。 "
    line = prefix + " ".join("#" + tag for tag, _ in tags)
    kept = ["#" + tag for tag, good in tags if good]
    # A bad tag goes with the blanks before it, or after it at the start of the line.
    expected = " ".join(kept) if not prefix else prefix.rstrip(" ") + "".join(" " + tag for tag in kept)
    if not content_chars(expected):
        expected = None
    return line, expected, len(tags), len(tags) - len(kept)


class _Builder:
    def __init__(self, rng: random.Random, source: _Source) -> None:
        self.rng = rng
        self.source = source
        self.marks: set[str] = set()
        self.found = self.removed = 0
        self.tags_found = self.tags_removed = 0
        self.kept_items: list[tuple[str, ...]] = []
        self.removed_items: list[tuple[tuple[str, ...], frozenset[str] | None]] = []
        self.areas: list[tuple[str, int, bool]] = []
        self.with_extras = False  # an area holds something that is not a quote

    def plain_item(self) -> list[Line]:
        rng = self.rng
        kind = rng.randrange(4)
        if kind == 0:
            sentence = rng.choice(SOURCE_SENTENCES + INVENTED) + "。"
            return [(sentence, sentence, None)]
        if kind == 1:
            self.marks.add("decoy")
            made_up = rng.choice(INVENTED)
            decoy = rng.choice((f"> “{made_up}”——讲师", f"- “{made_up}”"))
            return [(decoy, decoy, None)]
        if kind == 2:
            line, expected, found, removed = _tag_line(rng)
            self.tags_found += found
            self.tags_removed += removed
            return [(line, expected, None)]
        made_up = rng.choice(INVENTED)
        return [(text, text, None) for text in ("```", f"> {made_up}", f"“{made_up}”", "#桃花", "```")]

    def area(self, level: int) -> list[Line]:
        rng = self.rng
        title = rng.choice(_AREA_TITLES)
        # Each element: (lines, starts with >), lines as (text, expected).
        elements: list[tuple[list[tuple[str, str | None]], bool]] = []
        count = 0 if rng.random() < 0.1 else rng.randint(1, 6)
        good = 0
        for _ in range(count):
            self.found += 1
            if rng.random() < 0.6:
                lines = _good_item(rng, self.source, self.marks)
                self.kept_items.append(tuple(lines))
                elements.append(([(text, text) for text in lines], lines[0].startswith(">")))
                good += 1
            else:
                lines, needs = _bad_item(rng, self.source, self.marks)
                self.removed += 1
                self.removed_items.append((tuple(lines), needs))
                elements.append(([(text, None) for text in lines], lines[0].startswith(">")))
        extras = False
        if rng.random() < 0.3:
            elements.append(([(INTRO, INTRO)], False))
            extras = True
        if rng.random() < 0.2:
            self.marks.add("code_in_quotes")
            made_up = rng.choice(INVENTED)
            code = ["```", f"> {made_up}", f"“{made_up}”", "```"]
            elements.append(([(text, text) for text in code], False))
            extras = True
        rng.shuffle(elements)
        emptied = good == 0 and not extras
        if count == 0 and not extras:
            self.marks.add("born_empty")
        self.with_extras |= extras
        self.areas.append((title, level, emptied))
        heading = "#" * level + " " + title
        out: list[Line] = [(heading, None if emptied else heading, (title, level))]
        out += [("", "", None)] * rng.randint(0, 2)
        for number, (lines, quoted) in enumerate(elements):
            if number:
                blanks = rng.randint(0, 2)
                if quoted or elements[number - 1][1]:
                    blanks = max(1, blanks)  # two > quotes with nothing between are one quote
                out += [("", "", None)] * blanks
            out += [(text, None if emptied else expected, None) for text, expected in lines]
        return out


def make_quoted(rng: random.Random) -> tuple[str, str, QuoteExpect]:
    """A model reply, the source it is checked against, and what finishing it must give."""
    source = _Source(rng)
    builder = _Builder(rng, source)
    only_areas = rng.random() < 0.2  # raised from 0.05 so enough samples end up empty
    titles = [] if only_areas else rng.sample(_PLAIN_TITLES, rng.randint(1, 4))
    area_count = 0 if rng.random() < 0.1 else rng.randint(1, 2)
    if area_count == 2:
        builder.marks.add("two_areas")

    blocks: list[list[Line]] = []
    for title in titles:
        block: list[Line] = [(f"## {title}", f"## {title}", (title, 2))]
        for number in range(rng.randint(1, 3)):
            if number and rng.random() < 0.5:
                block.append(("", "", None))
            block += builder.plain_item()
        blocks.append(block)
    subs: list[int] = []
    tops = 0
    for _ in range(area_count):
        free = [number for number in range(len(blocks)) if number not in subs]
        if free and rng.random() < 0.2:
            subs.append(rng.choice(free))
        else:
            tops += 1
    for host in subs:
        builder.marks.add("sub_area")
        blocks[host] += [("", "", None)] * rng.randint(0, 1) + builder.area(3)
    plain = list(blocks)
    for _ in range(tops):
        plain.insert(rng.randint(0, len(plain)), builder.area(2))

    lines: list[Line] = []
    if rng.random() < 0.5:
        lines.append((REPLY_TITLE, None, None))
        lines += [("", "", None)] * rng.randint(0, 1)
    for number, block in enumerate(plain):
        if number and rng.random() < 0.7:
            lines.append(("", "", None))
        lines += block

    kept: list[str] = []
    sections: list[tuple[int, str, int]] = []
    for text, expected, heading in lines:
        if expected is None or not expected.strip():
            continue
        if heading is not None:
            sections.append((len(kept), heading[0], heading[1]))
        kept.append(expected)

    newline = rng.choice(("\n", "\r\n"))
    reply = newline.join(text for text, _, _ in lines)
    if rng.random() < 0.1:
        reply = "﻿" + reply
    marks = builder.marks
    if source.has_title:
        marks.add("has_title")
    expect = QuoteExpect(
        lines=kept,
        sections=sections,
        quotes_found=builder.found,
        quotes_removed=builder.removed,
        concepts_found=builder.tags_found,
        concepts_removed=builder.tags_removed,
        kept_items=builder.kept_items,
        removed_items=builder.removed_items,
        areas=builder.areas,
        empty=content_chars("\n".join(kept)) == "",
        empty_without_source=not titles and not builder.with_extras,
        marks=marks,
        has_title=source.has_title,
    )
    return reply, source.text, expect


__all__ = ["BAD_TAGS", "GOOD_TAGS", "INVENTED", "META", "NAME", "SOURCE_SENTENCES", "QuoteExpect", "make_quoted"]
