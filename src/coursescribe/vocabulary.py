"""The user's word list, a plain text file they edit themselves.

A line is either a common word ("践行") or a fix for a word the engine keeps
mishearing ("见行 → 践行"). Fixes are applied by rule before proofreading;
common words are given to the proofreading model as a hint.
"""

from __future__ import annotations

import functools
import logging
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from coursescribe import files

logger = logging.getLogger(__name__)

FILE_NAME = "词表.txt"
TEMPLATE = (
    "# CourseScribe 词表。每行一条，# 开头的行是说明，不起作用。\n"
    "# 常用词：直接写，例如\n"
    "#   践行\n"
    "# 总被听错的词：写“听错的 → 正确的”，例如\n"
    "#   见行 → 践行\n"
    "# 英文区分大小写，只换完整的词；要换几种写法就各写一行。\n"
    "# 改完保存即可，从下一次校对开始生效。\n"
)
_ARROW = re.compile(r"\s*(?:→|->|=>)\s*")
_BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"

WORD_LIST_ENCODING = (
    "词表 {path} 的编码认不出来，这次没有校对，文字稿已保存。在“设置 → 打开词表…”里打开它，另存为 UTF-8 编码，"
    "再点“重试”；不想用词表的话，把这个文件改个名字再点“重试”。"
)
WORD_LIST_HELD = (
    "词表 {path} 暂时打不开（可能正被别的程序占用），这次没有校对，文字稿已保存。关掉那个程序，或等一会儿，再点“重试”。"
)
WORD_LIST_UNREADABLE = (
    "读不了词表 {path}（可能没有权限，或同名的是一个文件夹），这次没有校对，文字稿已保存。处理好后点“重试”。"
)
WORD_LIST_CREATE_FAILED = "词表 {path} 建不了（可能磁盘满了，或这个文件夹不能写）。腾出空间后再点“打开词表…”。"


class WordListUnreadable(Exception):
    """The word list is there but cannot be read for sure. The message is shown to the user as it is."""


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
    for line in text.lstrip(_BOM).splitlines():
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


# --- replacing ---


def _is_word_char(ch: str) -> bool:
    """Letters, digits, "_" and "-" join English into one word; Chinese and punctuation do not."""
    return ch.isascii() and (ch.isalnum() or ch in "_-")


def _is_ascii_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


class _Matcher:
    """Finds the misheard words of a word list, longest first, English only as whole words."""

    def __init__(self, vocabulary: Vocabulary) -> None:
        self.spelling = dict(vocabulary.fixes)  # the last line for the same word counts
        self.lengths = sorted({len(heard) for heard in self.spelling}, reverse=True)
        self.firsts = frozenset(heard[0] for heard in self.spelling)
        self.correct = frozenset(self.spelling.values())
        self.correct_lengths = sorted({len(word) for word in self.correct}, reverse=True)

    def match_at(self, text: str, index: int) -> str | None:
        if text[index] not in self.firsts:
            return None
        for length in self.lengths:
            heard = text[index : index + length]
            if len(heard) == length and heard in self.spelling and self._whole(text, index, heard):
                return heard
        return None

    @staticmethod
    def _whole(text: str, index: int, heard: str) -> bool:
        end = index + len(heard)
        if _is_ascii_alnum(heard[0]) and index > 0 and _is_word_char(text[index - 1]):
            return False
        if _is_ascii_alnum(heard[-1]) and end < len(text) and _is_word_char(text[end]):
            return False
        return True

    def scan(self, text: str) -> Iterator[tuple[int, str]]:
        """Every match, left to right; text a match covers is not searched again."""
        index = 0
        while index < len(text):
            heard = self.match_at(text, index)
            if heard is None:
                index += 1
                continue
            yield index, heard
            index += len(heard)

    def is_free(self, text: str, index: int, heard: str) -> bool:
        """Whether this match lies outside every occurrence of a correct word."""
        end = index + len(heard)
        for length in self.correct_lengths:
            if length < len(heard):
                break
            for start in range(max(0, end - length), index + 1):
                if text[start : start + length] in self.correct:
                    return False
        return True


@functools.lru_cache(maxsize=8)
def _matcher(vocabulary: Vocabulary) -> _Matcher:
    return _Matcher(vocabulary)


def _replace(text: str, matches: Iterator[tuple[int, str]], spelling: dict[str, str]) -> str:
    out: list[str] = []
    last = 0
    for index, heard in matches:
        out.append(text[last:index])
        out.append(spelling[heard])
        last = index + len(heard)
    out.append(text[last:])
    return "".join(out)


def apply_fixes(text: str, vocabulary: Vocabulary) -> str:
    """Replace each misheard word once, in one pass; what is written in is never replaced again."""
    if not vocabulary.fixes:
        return text
    matcher = _matcher(vocabulary)
    return _replace(text, matcher.scan(text), matcher.spelling)


def fix_reply(sent: str, reply: str, vocabulary: Vocabulary) -> str:
    """Change back only the fixes the model undid; the rest of its reply stays exactly as it is.

    A fix counts as undone when the reply has more free matches of the misheard word than
    the text sent had. Counting rather than aligning: short repeated words align wrongly.
    """
    if not vocabulary.fixes:
        return reply
    matcher = _matcher(vocabulary)
    before = Counter(heard for index, heard in matcher.scan(sent) if matcher.is_free(sent, index, heard))
    free = [(index, heard) for index, heard in matcher.scan(reply) if matcher.is_free(reply, index, heard)]
    after = Counter(heard for _, heard in free)
    undone = {heard for heard, count in after.items() if count > before[heard]}
    if not undone:
        return reply
    return _replace(reply, iter([(i, h) for i, h in free if h in undone]), matcher.spelling)


# --- reading ---


def load_vocabulary(path: Path) -> Vocabulary:
    """Read the word list; this never changes, creates or deletes a file.

    No file is an empty list. A file that cannot be read for sure raises WordListUnreadable,
    so proofreading never goes ahead with a misread list.
    """
    if path.is_dir():
        raise WordListUnreadable(WORD_LIST_UNREADABLE.format(path=path))
    try:
        data = files._patiently(lambda: files._read_bytes_once(path))
    except FileNotFoundError:
        return EMPTY
    except PermissionError as exc:
        logger.warning("The word list is held: %s", exc)
        raise WordListUnreadable(WORD_LIST_HELD.format(path=path)) from exc
    except OSError as exc:
        logger.warning("Could not read the word list: %s", exc)
        raise WordListUnreadable(WORD_LIST_UNREADABLE.format(path=path)) from exc
    text = files.decode_user_text(data)
    if text is None:
        logger.warning("The word list is in an encoding that cannot be told for sure")
        raise WordListUnreadable(WORD_LIST_ENCODING.format(path=path))
    return parse_vocabulary(text)


def ensure_word_list(path: Path) -> None:
    """Create the file with instructions, so the user has something to open. Never overwrites.

    Written whole or not at all: a full disk leaves neither a half file nor a temporary one.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    files.create_once(path, TEMPLATE.encode("utf-8"))
