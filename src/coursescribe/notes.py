"""Organised notes: tidy the model's reply and write the program's own header and outline.

Pure string logic: no files, network, model or interface, and no clock.
"""

from __future__ import annotations

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
