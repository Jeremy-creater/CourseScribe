"""Organised notes: tidy the model's reply and write the program's own header and outline.

Pure string logic: no files, network, model or interface, and no clock.
"""

from __future__ import annotations

import bisect
import datetime
import re
import unicodedata
from dataclasses import dataclass

from coursescribe.text import content_chars

PROMPT_MARGIN = 256  # tokens left for the prompt wrapper and chat template (12-4)

OUTLINE_TITLE = "本课脉络"
_CORE_TITLE = "核心宗旨"
_MODEL_OUTLINE_TITLES = frozenset(("本课脉络", "课程脉络", "本节脉络", "脉络"))
_LINK_BREAKERS = "#|^[]"
_LINKABLE = str.maketrans(_LINK_BREAKERS, "＃｜＾［］")
_ADDRESS = {
    "%": "%25", " ": "%20", "#": "%23", "|": "%7C", "^": "%5E", "[": "%5B", "]": "%5D",
    "(": "%28", ")": "%29", "<": "%3C", ">": "%3E",
}
_EMPTY_SECTION = "小节"

_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)$")
_FRONT_MATTER_OPEN = re.compile(r"---[ \t]*$")
_FRONT_MATTER_CLOSE = re.compile(r"(?:---|\.\.\.)[ \t]*$")
_NUMERAL = r"(?:[0-9]+|[一二三四五六七八九十百千零〇两]+)"
_ORDINAL = re.compile(rf"{_NUMERAL}[.、．]|[（(]{_NUMERAL}[）)]")
_RENAMED = re.compile(r"\(\d+\)$")  # the （n） added for uniqueness, after NFKC
_HEADER_LABEL = re.compile(r"(?:日期|时间|来源|校对稿|date|source)\s*[:：]", re.IGNORECASE)
_DATE = re.compile(
    r"(?<!\d)(20\d\d)(?:([.\-_])(\d{1,2})\2(\d{1,2})(?!\d)|年(\d{1,2})月(\d{1,2})(?:日|(?!\d)))"
)


class EmptyNotes(ValueError):
    """The reply has no real content once the header lines are removed."""


class EmptySource(ValueError):
    """The proofread text has no real content once its title is removed."""


@dataclass(frozen=True)
class NotesMeta:
    name: str  # the proofread file's name without .md, which is the job name
    date: datetime.date

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or any(ch in self.name for ch in "\n\r/\\"):
            raise ValueError(f"name must be a file name, not empty or a path: {self.name!r}")
        if not isinstance(self.date, datetime.date) or isinstance(self.date, datetime.datetime):
            raise TypeError(f"date must be a date without a time: {self.date!r}")


# ---------------------------------------------------------------- dates and sizes


def _first_date(text: str) -> tuple[datetime.date, int, int] | None:
    """The first real date in already normalised ``text``, with where it is."""
    position = 0
    while (match := _DATE.search(text, position)) is not None:
        year = int(match.group(1))
        month, day = (match.group(3), match.group(4)) if match.group(2) else (match.group(5), match.group(6))
        try:
            return datetime.date(year, int(month), int(day)), match.start(), match.end()
        except ValueError:
            position = match.start() + 1
    return None


def lecture_date(name: str, fallback: datetime.date) -> datetime.date:
    """The date written in ``name``, or ``fallback`` when it has none."""
    if not isinstance(name, str):
        raise TypeError(f"name must be text: {name!r}")
    if not isinstance(fallback, datetime.date) or isinstance(fallback, datetime.datetime):
        raise TypeError(f"fallback must be a date without a time: {fallback!r}")
    found = _first_date(unicodedata.normalize("NFKC", name))
    return found[0] if found else fallback


def _positive_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a whole number of at least 1: {value!r}")
    return value


def overflow(rules: str, text: str, num_ctx: int, reply_budget: int) -> int:
    """How many tokens too many, counting one per character; 0 when it fits."""
    if not isinstance(rules, str):
        raise TypeError(f"rules must be text: {rules!r}")
    if not isinstance(text, str):
        raise TypeError("text must be text")
    _positive_int(num_ctx, "num_ctx")
    _positive_int(reply_budget, "reply_budget")
    return max(0, len(rules) + len(text) + reply_budget + PROMPT_MARGIN - num_ctx)


# ---------------------------------------------------------------- reading Markdown structure


def _lines(text: str) -> list[str]:
    if text.startswith("\ufeff"):
        text = text[1:]
    return _LINE_BREAK.split(text)


def _closes(line: str, fence: str) -> bool:
    match = _FENCE.match(line)
    return (
        bool(match) and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence)
        and not match.group(2).strip()
    )


def _in_code(lines: list[str]) -> list[bool]:
    """For each line, whether it is a fence or inside a fenced code block."""
    flags = []
    fence = None
    for line in lines:
        if fence is not None:
            flags.append(True)
            if _closes(line, fence):
                fence = None
            continue
        match = _FENCE.match(line)
        if match:
            fence = match.group(1)
        flags.append(match is not None)
    return flags


def _heading(line: str) -> tuple[int, str, str] | None:
    """(level, text, run of #) when ``line`` is an ATX heading, else None."""
    indent = len(line) - len(line.lstrip(" "))
    if indent > 3:
        return None
    rest = line[indent:]
    level = len(rest) - len(rest.lstrip("#"))
    if not 1 <= level <= 6:
        return None
    after = rest[level:]
    if after and after[0] not in " \t":
        return None
    text = after.strip(" \t")
    if text.endswith("#"):
        bare = text.rstrip("#")
        if not bare:
            text = ""
        elif bare[-1] in " \t":
            text = bare.rstrip(" \t")
    return level, text, rest[:level]


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line index, level, text) of each heading outside code."""
    found = []
    for index, (line, code) in enumerate(zip(lines, _in_code(lines))):
        if not code and (heading := _heading(line)) is not None:
            found.append((index, heading[0], heading[1]))
    return found


def _normalised_title(text: str) -> str:
    """Heading text as compared when looking for the core idea or an outline."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("**", "").replace("__", "").replace("*", "").strip()
    text = _ORDINAL.sub("", text, count=1) if _ORDINAL.match(text) else text
    return re.sub(r"\s+", "", text).rstrip(":：")


def _is_core(text: str) -> bool:
    """Whether a section heading is the core idea.

    A number the program added to make the name unique, as in 核心宗旨（2）, still
    counts, so the outline lands in the same place when finished notes are redone.
    """
    return _RENAMED.sub("", _normalised_title(text)) == _CORE_TITLE


def _blank(line: str) -> bool:
    return not line.strip()


def _trim(lines: list[str]) -> list[str]:
    start = 0
    while start < len(lines) and _blank(lines[start]):
        start += 1
    end = len(lines)
    while end > start and _blank(lines[end - 1]):
        end -= 1
    return lines[start:end]


# ---------------------------------------------------------------- the source and the prompt

# The program's own words around the source. Together far under PROMPT_MARGIN, so at least
# 128 of it is left for Ollama's chat template.
_PROMPT_HEAD = "\n\n【校对稿开始】下面是这节课的原文，笔记里的引述和概念只能来自这里。\n"
_PROMPT_TAIL = "\n【校对稿结束】请按上面的整理规则写这节课的笔记，只返回笔记正文。"


def source_body(source: str) -> str:
    """The proofread text the model is given and quotes are checked against.

    Without one leading byte order mark, the ``# title`` on the first line that is not
    blank, and the blank lines at both ends; everything else as written, with "\\n" breaks.
    """
    lines = _lines(source)
    first = next((index for index, line in enumerate(lines) if not _blank(line)), None)
    if first is not None and (heading := _heading(lines[first])) is not None and heading[0] == 1:
        lines = lines[first + 1:]
    return "\n".join(_trim(lines))


def build_prompt(rules: str, source: str) -> str:
    """The rules, then the source without its title, then one line asking for the notes."""
    if not isinstance(source, str):
        raise TypeError(f"source must be text: {type(source).__name__}")
    if not isinstance(rules, str):
        raise TypeError(f"rules must be text: {type(rules).__name__}")
    if content_chars(rules) == "":
        raise ValueError("rules have no content")
    body = source_body(source)
    if content_chars(body) == "":
        raise EmptySource("这份校对稿是空的")
    return "".join((rules, _PROMPT_HEAD, body, _PROMPT_TAIL))


# ---------------------------------------------------------------- notes_body


def _unwrapped(lines: list[str]) -> list[str]:
    filled = [index for index, line in enumerate(lines) if not _blank(line)]
    if len(filled) < 2:
        return lines
    first, last = filled[0], filled[-1]
    match = _FENCE.match(lines[first])
    if not match or match.group(2).strip().lower() not in ("markdown", "md") or not _closes(lines[last], match.group(1)):
        return lines
    return lines[first + 1:last]


def _without_front_matter(lines: list[str]) -> list[str]:
    first = next((index for index, line in enumerate(lines) if not _blank(line)), None)
    if first is None or not _FRONT_MATTER_OPEN.match(lines[first]):
        return lines
    for index in range(first + 1, len(lines)):
        if _heading(lines[index]) is not None:
            return lines
        if _FRONT_MATTER_CLOSE.match(lines[index]):
            return lines[:first] + lines[index + 1:]
    return lines


def _without_model_outlines(lines: list[str]) -> list[str]:
    found = _headings(lines)
    drop: set[int] = set()
    for number, (index, _, text) in enumerate(found):
        if _normalised_title(text) in _MODEL_OUTLINE_TITLES:
            end = found[number + 1][0] if number + 1 < len(found) else len(lines)
            drop.update(range(index, end))
    return [line for index, line in enumerate(lines) if index not in drop]


def _without_model_title(lines: list[str], name: str) -> list[str]:
    found = _headings(lines)
    if not found or found[0][1] != 1:
        return lines
    index = found[0][0]
    only = sum(1 for _, level, _ in found if level == 1) == 1
    if only or lines[index] == f"# {name}":
        return lines[:index] + lines[index + 1:]
    return lines


def _is_header_line(line: str) -> bool:
    text = line.lstrip(">-*+ \t").replace("**", "").replace("__", "").lstrip(">-*+ \t")
    if _HEADER_LABEL.match(text):
        return True
    text = unicodedata.normalize("NFKC", text)
    found = _first_date(text)
    return found is not None and content_chars(text[:found[1]] + text[found[2]:]) == ""


def _without_header_lines(lines: list[str]) -> list[str]:
    found = _headings(lines)
    end = found[0][0] if found else len(lines)
    code = _in_code(lines)
    return [line for index, line in enumerate(lines) if index >= end or code[index] or not _is_header_line(line)]


def notes_body(reply: str, name: str) -> str:
    """The model's notes without the header, title and outline it wrote itself."""
    lines = _unwrapped(_lines(reply))
    lines = _without_front_matter(lines)
    lines = _without_model_outlines(lines)
    lines = _without_model_title(lines, name)
    lines = _without_header_lines(lines)
    return "\n".join(_trim(lines))


# ---------------------------------------------------------------- assemble_notes


def _key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _source_link(name: str) -> str:
    if not any(ch in name for ch in _LINK_BREAKERS):
        return f"[[{name}]]"
    # NotesMeta already refuses a backslash; it stays escaped here as Markdown requires.
    shown = "".join("\\" + ch if ch in "\\[]" else ch for ch in name)
    address = "".join(_ADDRESS.get(ch, ch) for ch in name)
    return f"[{shown}]({address}.md)"


def _header(meta: NotesMeta) -> str:
    return f"# {meta.name}\n\n日期：{meta.date.isoformat()}\n校对稿：{_source_link(meta.name)}\n来源："


def _sections(found: list[tuple[int, int, str]], level: int) -> list[tuple[int, str]]:
    return [(index, text) for index, lvl, text in found if lvl == level]


def _section_names(found: list[tuple[int, int, str]], level: int, name: str) -> dict[int, str]:
    """Linkable, unique text for each section heading, by line index, in order."""
    sections = _sections(found, level)
    others = {_key(text) for _, lvl, text in found if lvl != level}
    # The program's own title, both as the name and as the text Obsidian reads
    # from "# {name}" (a name ending in " ##" reads without the #).
    reserved = {_key(name), _key(_heading(f"# {name}")[1])}
    linkable = [(index, text.translate(_LINKABLE) or _EMPTY_SECTION) for index, text in sections]
    taken = reserved | others | {_key(text) for _, text in linkable}
    earlier: set[str] = set()
    names = {}
    for index, text in linkable:
        if _key(text) in reserved or _key(text) in earlier or _key(text) in others:
            number = 2
            while _key(f"{text}（{number}）") in taken:
                number += 1
            text = f"{text}（{number}）"
            taken.add(_key(text))
        earlier.add(_key(text))
        names[index] = text
    return names


def _merge_same_name(lines: list[str], found: list[tuple[int, int, str]], level: int) -> list[str]:
    """``lines`` with each later section of the same name joined to the first, in order.

    The body of a later section (its sub-headings too) follows the first one's, after a
    blank line. Nothing changes when no two sections share a name.
    """
    starts = [index for index, _ in _sections(found, level)]
    spans = [lines[a:b] for a, b in zip(starts, starts[1:] + [len(lines)])]
    # A code block left open runs to the end of the text, so the last section stays where it is.
    open_end = _in_code(lines + [""])[-1]
    first_of: dict[str, int] = {}
    merged: list[list[str]] = []
    for index, span in zip(starts, spans):
        text = _heading(span[0])[1].translate(_LINKABLE) or _EMPTY_SECTION
        at = first_of.setdefault(_key(text), len(merged))
        if at == len(merged) or (open_end and span is spans[-1]):
            merged.append(span)
            continue
        body = _trim(span[1:])
        if body:
            merged[at] = _trim(merged[at]) + [""] + body + [""]
    if len(merged) == len(spans):
        return lines
    return lines[: starts[0]] + [line for span in merged for line in span]


def assemble_notes(body: str, meta: NotesMeta) -> str:
    """The finished notes: the program's header, then ``body`` with an outline of its sections."""
    if content_chars(body) == "":
        raise EmptyNotes("the notes have no content")
    lines = _trim(_lines(body))
    found = _headings(lines)
    outline: list[str] = []
    split = len(lines)
    if found:
        level = min(lvl for _, lvl, _ in found)
        if (joined := _merge_same_name(lines, found, level)) is not lines:
            lines = _trim(joined)
            found = _headings(lines)
        names = _section_names(found, level, meta.name)
        sections = list(names)
        core = next((n for n, (_, text) in enumerate(_sections(found, level)) if _is_core(text)), -1)
        for index, text in names.items():
            heading = _heading(lines[index])
            if text != heading[1]:
                lines[index] = f"{heading[2]} {text}"
        after = sections[core + 1:]  # core is -1 without a core idea: every section
        if after:
            split = after[0]
            outline = [f"{'#' * level} {OUTLINE_TITLE}", ""] + [f"- [[#{names[index]}]]" for index in after]
    blocks = ["\n".join(_trim(lines[:split])), "\n".join(outline), "\n".join(lines[split:])]
    return "\n\n".join([_header(meta)] + [block for block in blocks if block]) + "\n"


def structure_notes(reply: str, meta: NotesMeta) -> str:
    """Tidy ``reply`` without checking quotes and concepts against the source."""
    return assemble_notes(notes_body(reply, meta.name), meta)


# ---------------------------------------------------------------- finish_notes


@dataclass(frozen=True)
class NotesResult:
    markdown: str  # the finished notes, as assemble_notes writes them
    removed_quotes: int  # quotes removed, counted by item
    removed_concepts: int  # concept tags removed, counted by occurrence


_QUOTE_TITLE = "引述"
_SHORTEST_QUOTE = 4  # letters and digits; shorter quotes cannot be checked
_LONGEST_ATTRIBUTION = 12
_QUOTE_LINE = re.compile(r" {0,3}>")
_TAG = re.compile(r"(?<!\S)#([\w/-]+)")  # \w is str.isalnum() and _
_BACKTICKS = re.compile(r"`+")
_CALLOUT = re.compile(r"\[![^\W\d_]+\][+-]?")
_DASHES = re.compile(r"—+|―+|-{2,}")
_LEADING_DASHES = re.compile(rf"\s*(?:{_DASHES.pattern})")
_ELLIPSIS = re.compile(r"\.{3,}|⋯+")
_OPENER = re.compile(r"[“「『\"]")
_CLOSERS = {"“": "”", "「": "」", "『": "』", '"': '"'}
_QUOTE_MARKS = frozenset("“”「」『』\"")
_NOT_IN_A_NAME = frozenset("。！？!?，,；;") | _QUOTE_MARKS


def _code_spans(line: str) -> list[tuple[int, int]]:
    """(start, end) of each inline code span: a run of backticks up to the next run as long."""
    if "`" not in line:
        return []
    runs = [match.span() for match in _BACKTICKS.finditer(line)]
    following: list[int | None] = [None] * len(runs)
    last: dict[int, int] = {}
    for number in range(len(runs) - 1, -1, -1):
        size = runs[number][1] - runs[number][0]
        following[number] = last.get(size)
        last[size] = number
    spans = []
    number = 0
    while number < len(runs):
        closing = following[number]
        if closing is None:
            number += 1  # no partner: these backticks are plain text
            continue
        spans.append((runs[number][0], runs[closing][1]))
        number = closing + 1
    return spans


def _tags(line: str) -> list[re.Match[str]]:
    """The concept tags in ``line``: #name after the line start or a blank, outside inline code."""
    if "#" not in line:
        return []
    spans = _code_spans(line)
    starts = [start for start, _ in spans]
    found = []
    for match in _TAG.finditer(line):
        inside = bisect.bisect_right(starts, match.start()) - 1
        if inside >= 0 and match.start() < spans[inside][1]:
            continue
        if any(ch.isalpha() for ch in match.group(1)):  # #3 and #2026 are not tags
            found.append(match)
    return found


def _without_tags(line: str, tags: list[re.Match[str]]) -> str:
    """``line`` without ``tags``, each with the blanks before it, or after it at the start of the line.

    A tag after only removed tags and blanks is at the start of the line too, so
    "#编造 #编造 #索引" becomes "#索引" without a leading blank.
    """
    pieces = []
    started = False  # whether something other than blanks is kept before the tag
    at = 0
    for match in tags:
        start, end = match.span()
        while start > 0 and line[start - 1].isspace():
            start -= 1
        before = line[at:max(start, at)]
        started = started or bool(before.strip())
        if started:
            pieces.append(before)
        else:
            while end < len(line) and line[end].isspace():
                end += 1
        at = max(at, end)
    pieces.append(line[at:])
    return "".join(pieces)


def _source_chars(source: str) -> str:
    """What quotes and tags are compared with: the source's letters and digits, without its title."""
    return content_chars(source_body(source))


# ---------------------------------------------------------------- finish_notes: concept tags


def _changes_kind(before: str, after: str, opening: bool) -> bool:
    """Whether removing tags turned a line into a heading, a fence, a > quote or a header line."""
    if _heading(before) is None and (
        _heading(after) is not None or _FENCE.match(after) is not None
        or (_QUOTE_LINE.match(after) is not None and _QUOTE_LINE.match(before) is None)
    ):
        return True
    return opening and not _is_header_line(before) and _is_header_line(after)


def _filter_concepts(lines: list[str], code: list[bool], drop: list[bool], said: str) -> int:
    """Remove the tags the source does not back, in place; mark lines to drop. Returns how many went."""
    known: dict[str, bool] = {}

    def backed(name: str) -> bool:
        if name not in known:
            key = content_chars(name)
            known[name] = bool(key) and key in said
        return known[name]

    rows = [index for index, _, _ in _headings(lines)]
    opening = rows[0] if rows else len(lines)  # lines before the first heading
    removed = 0
    for index, line in enumerate(lines):
        if code[index]:
            continue
        bad = [match for match in _tags(line) if not backed(match.group(1))]
        if not bad:
            continue
        removed += len(bad)
        lines[index] = after = _without_tags(line, bad)
        heading = _heading(after)
        if _changes_kind(line, after, index < opening):
            drop[index] = True
        elif heading is not None and _normalised_title(heading[1]) in _MODEL_OUTLINE_TITLES:
            # Now it reads as the model's outline, so it goes as 12-2 drops one: down to the next heading.
            following = bisect.bisect_right(rows, index)
            end = rows[following] if following < len(rows) else len(lines)
            drop[index:end] = [True] * (end - index)
        elif heading is None and not content_chars(after):
            drop[index] = True
    return removed


# ---------------------------------------------------------------- finish_notes: quotes


def _is_quote_title(text: str) -> bool:
    bare = _without_tags(text, _tags(text))
    return _RENAMED.sub("", _normalised_title(bare)) == _QUOTE_TITLE


def _quote_areas(rows: list[tuple[int, int, str]], total: int) -> list[tuple[int, int]]:
    """(heading, end) of each quote section: down to the next heading as high or higher."""
    areas: list[list[int]] = []
    open_areas: list[tuple[int, int]] = []  # (position in areas, level), levels rising
    for index, level, text in rows:
        while open_areas and open_areas[-1][1] >= level:
            areas[open_areas.pop()[0]][1] = index
        if _is_quote_title(text):
            open_areas.append((len(areas), level))
            areas.append([index, total])
    return [(heading, end) for heading, end in areas]


def _is_attribution(rest: str) -> bool:
    """Whether what follows a dash is a short name, as in ——讲师, rather than more of the quote."""
    rest = rest.strip()
    return 1 <= len(rest) <= _LONGEST_ATTRIBUTION and not any(ch in _NOT_IN_A_NAME for ch in rest)


def _said(sentence: str, said: str) -> bool:
    """Whether one quoted sentence, tags already gone, stands in the source."""
    text = unicodedata.normalize("NFKC", sentence)
    dashes = list(_DASHES.finditer(text))
    if dashes and _is_attribution(text[dashes[-1].end():]):
        text = text[:dashes[-1].start()]  # the last dash only, once: what is before it is checked
    parts = [part for part in map(content_chars, _ELLIPSIS.split(text)) if part]
    if sum(map(len, parts)) < _SHORTEST_QUOTE:
        return False
    at = 0
    for part in parts:
        found = said.find(part, at)
        if found < 0:
            return False
        at = found + len(part)
    return True


def _block_sentences(block: list[str]) -> list[str]:
    """The sentences a run of > lines says: each paragraph whole, without a callout title or a lone name.

    Whether a line is blank is decided before its tags go, so "> #索引" does not
    split a paragraph; the tags go line by line, before the lines are joined.
    """
    raw = [line.lstrip(" >") for line in block]
    texts = [_without_tags(line, _tags(line)).lstrip(" >") for line in block]
    if _CALLOUT.match(texts[0]):
        texts[0] = ""  # a callout's title is a label
    sentences = []
    paragraph: list[str] = []
    for blank, text in zip([not line.strip() for line in raw] + [True], texts + [""]):
        if not blank:
            paragraph.append(text)
            continue
        joined = unicodedata.normalize("NFKC", "".join(paragraph))
        if joined.strip():  # a callout title or a line of tags alone says nothing to check
            leading = _LEADING_DASHES.match(joined)
            if not (leading and _is_attribution(joined[leading.end():])):
                sentences.append(joined)
        paragraph = []
    return sentences


def _line_sentences(line: str) -> list[str]:
    """What stands between quotation marks in ``line``, outermost only; an open quote runs to the end."""
    text = _without_tags(line, _tags(line))
    pieces = []
    at = 0
    for start, end in _code_spans(text):
        code = "".join(" " if set(unicodedata.normalize("NFKC", ch)) & _QUOTE_MARKS else ch for ch in text[start:end])
        pieces += [text[at:start], code]
        at = end
    text = unicodedata.normalize("NFKC", "".join(pieces) + text[at:])
    sentences = []
    at = 0
    while (opener := _OPENER.search(text, at)) is not None:
        closer = text.find(_CLOSERS[opener.group()], opener.end())
        if closer < 0:
            sentences.append(text[opener.end():])
            break
        sentences.append(text[opener.end():closer])
        at = closer + 1
    return sentences


def _check_quotes(lines: list[str], code: list[bool], drop: list[bool], said: str) -> int:
    """Mark the quotes in quote sections the source does not back, then the sections left empty."""
    # The code flags of the reply as written: a line dropped for turning into a fence must not open code.
    rows = []
    for index, line in enumerate(lines):
        if not code[index] and not drop[index] and (heading := _heading(line)) is not None:
            rows.append((index, heading[0], heading[1]))
    is_heading = [False] * len(lines)
    for index, _, _ in rows:
        is_heading[index] = True
    areas = _quote_areas(rows, len(lines))
    inside = [False] * len(lines)
    for heading, end in areas:
        inside[heading + 1:end] = [True] * (end - heading - 1)

    def candidate(index: int) -> bool:
        return inside[index] and not code[index] and not drop[index] and not is_heading[index]

    removed = 0
    index = 0
    while index < len(lines):
        if not candidate(index):
            index += 1
            continue
        start = index
        if _QUOTE_LINE.match(lines[index]):
            while index < len(lines) and candidate(index) and _QUOTE_LINE.match(lines[index]):
                index += 1
            sentences = _block_sentences(lines[start:index])
        else:
            index += 1
            sentences = _line_sentences(lines[start])
        if not all(_said(sentence, said) for sentence in sentences):
            drop[start:index] = [True] * (index - start)
            removed += 1

    for heading, end in areas:
        # Any code, even an empty block, keeps the section.
        if all(drop[i] or is_heading[i] or (not code[i] and not content_chars(lines[i])) for i in range(heading + 1, end)):
            drop[heading:end] = [True] * (end - heading)
    return removed


# ---------------------------------------------------------------- finish_notes: removing lines


def _without_marked(lines: list[str], code: list[bool], drop: list[bool]) -> list[str]:
    """``lines`` without the marked ones, leaving the blank lines around each removed run tidy."""

    def quote_line(index: int) -> bool:
        return not code[index] and _QUOTE_LINE.match(lines[index]) is not None

    kept = []
    index = 0
    while index < len(lines):
        if not drop[index]:
            kept.append(lines[index])
            index += 1
            continue
        # A removed run: marked lines, with the blank lines between two of them.
        end = index
        while True:
            while end < len(lines) and drop[end]:
                end += 1
            after = end
            while after < len(lines) and not drop[after] and _blank(lines[after]):
                after += 1
            if after < len(lines) and drop[after]:
                end = after
                continue
            break
        if index > 0 and _blank(lines[index - 1]):
            index = after  # a blank line before it stays, so the ones after it go
            continue
        if after == end and 0 < index and end < len(lines) and quote_line(index - 1) and quote_line(end):
            kept.append("")  # two > quotes must not run together into one
        index = end
    return kept


def finish_notes(reply: str, source: str, meta: NotesMeta) -> NotesResult:
    """The finished notes, with quotes and concept tags the source does not back removed and counted."""
    if not isinstance(reply, str):
        raise TypeError(f"reply must be text: {type(reply).__name__}")
    if not isinstance(source, str):
        raise TypeError(f"source must be text: {type(source).__name__}")
    if not isinstance(meta, NotesMeta):
        raise TypeError(f"meta must be NotesMeta: {type(meta).__name__}")
    lines = notes_body(reply, meta.name).split("\n")
    code = _in_code(lines)
    said = _source_chars(source)
    drop = [False] * len(lines)
    # Tags first: a heading like "## 引述 #编造" is a quote section once its bad tag is gone.
    concepts = _filter_concepts(lines, code, drop, said)
    quotes = _check_quotes(lines, code, drop, said)
    body = "\n".join(_without_marked(lines, code, drop))
    return NotesResult(assemble_notes(body, meta), quotes, concepts)
