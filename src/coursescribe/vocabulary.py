"""The user's word list, a plain text file they edit themselves.

A line is either a common word ("践行") or a fix for a word the engine keeps
mishearing ("见行 → 践行"). Fixes are applied by rule before proofreading;
common words are given to the proofreading model as a hint.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FILE_NAME = "词表.txt"
TEMPLATE = (
    "# CourseScribe 词表。每行一条，# 开头的行是说明，不起作用。\n"
    "# 常用词：直接写，例如\n"
    "#   践行\n"
    "# 总被听错的词：写“听错的 → 正确的”，例如\n"
    "#   见行 → 践行\n"
    "# 改完保存即可，从下一次校对开始生效。\n"
)
_ARROW = re.compile(r"\s*(?:→|->|=>)\s*")


@dataclass(frozen=True)
class Vocabulary:
    terms: tuple[str, ...]
    fixes: tuple[tuple[str, str], ...]

    def all_terms(self) -> tuple[str, ...]:
        """Common words plus the correct side of every fix, each once, in order."""
        seen: dict[str, None] = dict.fromkeys(self.terms)
        for _, written in self.fixes:
            seen.setdefault(written)
        return tuple(seen)


EMPTY = Vocabulary((), ())


def parse_vocabulary(text: str) -> Vocabulary:
    terms: list[str] = []
    fixes: list[tuple[str, str]] = []
    for line in text.lstrip("﻿").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if _ARROW.search(line):
            parts = _ARROW.split(line)
            if len(parts) == 2 and all(parts):
                fixes.append((parts[0], parts[1]))
            continue
        terms.append(line)
    return Vocabulary(tuple(terms), tuple(fixes))


def apply_fixes(text: str, vocabulary: Vocabulary) -> str:
    if not vocabulary.fixes:
        return text
    spelling = dict(vocabulary.fixes)
    pattern = re.compile("|".join(re.escape(heard) for heard in sorted(spelling, key=len, reverse=True)))
    return pattern.sub(lambda match: spelling[match.group(0)], text)


def load_vocabulary(path: Path) -> Vocabulary:
    try:
        return parse_vocabulary(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return EMPTY
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read the word list: %s", exc)
        return EMPTY


def ensure_word_list(path: Path) -> None:
    """Create the file with instructions, so the user has something to open. Never overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(TEMPLATE)
    except FileExistsError:
        pass
