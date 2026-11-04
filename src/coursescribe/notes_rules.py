"""The rules for organised notes: a text file the user may edit, and a generic copy shipped with the program.

Lines starting with // (after half-width spaces, tabs or ideographic spaces) are notes to the
user and are not given to the model. # is not a comment here: the rules themselves are
Markdown, with headings such as "## 核心宗旨" and the way to write a tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from coursescribe import files
from coursescribe.notes import _LINE_BREAK, _trim
from coursescribe.settings import app_home
from coursescribe.text import content_chars

logger = logging.getLogger(__name__)

FILE_NAME = "整理规则.txt"
SHIPPED_RULES = Path(__file__).absolute().with_name("notes_rules.txt")

RULES_ENCODING = (
    "整理规则 {path} 的编码认不出来，或者文件里有看不见的控制字符，这次没有整理笔记，校对稿没有变。"
    "在“设置 → 打开整理规则…”里打开它，另存为 UTF-8 编码，再点“重试”；有控制字符的话，可以把文字复制到新文件再保存。"
    "想用程序自带的规则，把这个文件改个名字再点“重试”。"
)
RULES_HELD = (
    "整理规则 {path} 暂时打不开（可能正被别的程序占用），这次没有整理笔记，校对稿没有变。"
    "关掉那个程序，或等一会儿，再点“重试”。"
)
RULES_UNREADABLE = (
    "读不了整理规则 {path}（可能没有权限，或同名的是一个文件夹），这次没有整理笔记，校对稿没有变。"
    "处理好后点“重试”；想用程序自带的规则，把它改个名字再点“重试”。"
)
RULES_CREATE_FAILED = (
    "整理规则 {path} 建不了（可能磁盘满了、这个文件夹不能写，或者那个位置被一个同名文件占着）。"
    "处理好后再点“打开整理规则…”。"
)
RULES_FOLDER_IN_THE_WAY = (
    "整理规则 {path} 建不了：那里已经有一个同名的文件夹。把这个文件夹改个名字或移走，再点“打开整理规则…”。"
)
RULES_SHIPPED_MISSING = "程序自带的整理规则 {path} 不见了或读不了，请重新安装 CourseScribe。"
RULES_EMPTY_NOTE = "整理规则 {path} 里面没有可用的规则（空的、只有说明或只有标点），这次用的是程序自带的规则。"

_COMMENT_LEAD = " \t　"  # only these; a no-break space before // keeps the line a rule


class NotesRulesUnreadable(Exception):
    """The rules cannot be read or created. The message is shown to the user as it is."""


@dataclass(frozen=True)
class NotesRules:
    text: str  # the rules given to the model; never empty
    note: str | None  # None, or RULES_EMPTY_NOTE when the user's file has no rules and the shipped ones are used


def rules_path() -> Path:
    """Where the user's rules live, beside the word list. Nothing is created."""
    return app_home() / FILE_NAME


def _rules(text: str) -> str:
    """``text`` without its comment lines and the blank lines at both ends; nothing else changes."""
    lines = [line for line in _LINE_BREAK.split(text) if not line.lstrip(_COMMENT_LEAD).startswith("//")]
    return "\n".join(_trim(lines))


def shipped_rules(shipped: Path = SHIPPED_RULES) -> str:
    """The rules that come with the program. Missing or broken means the installation is incomplete."""
    try:
        rules = _rules(shipped.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read the shipped notes rules: %s", exc)
        raise NotesRulesUnreadable(RULES_SHIPPED_MISSING.format(path=shipped)) from exc
    if content_chars(rules) == "":
        raise NotesRulesUnreadable(RULES_SHIPPED_MISSING.format(path=shipped))
    return rules


def load_notes_rules(path: Path, *, shipped: Path = SHIPPED_RULES) -> NotesRules:
    """Read the user's rules afresh; this never changes, creates or deletes a file.

    No file means the shipped rules. A file that cannot be read for sure raises
    NotesRulesUnreadable rather than falling back, so notes are never written by rules
    the user did not choose without them knowing.
    """
    try:
        is_dir = path.is_dir()
    except OSError as exc:  # even looking at it is refused
        logger.warning("Could not look at the notes rules: %s", exc)
        raise NotesRulesUnreadable(RULES_UNREADABLE.format(path=path)) from exc
    if is_dir:
        raise NotesRulesUnreadable(RULES_UNREADABLE.format(path=path))
    try:
        data = files._patiently(lambda: files._read_bytes_once(path))
    except FileNotFoundError:
        return NotesRules(shipped_rules(shipped), None)
    except PermissionError as exc:
        logger.warning("The notes rules are held: %s", exc)
        raise NotesRulesUnreadable(RULES_HELD.format(path=path)) from exc
    except OSError as exc:
        logger.warning("Could not read the notes rules: %s", exc)
        raise NotesRulesUnreadable(RULES_UNREADABLE.format(path=path)) from exc
    text = files.decode_user_text(data)
    if text is None:
        logger.warning("The notes rules are in an encoding that cannot be told for sure")
        raise NotesRulesUnreadable(RULES_ENCODING.format(path=path))
    rules = _rules(text)
    if content_chars(rules) == "":
        return NotesRules(shipped_rules(shipped), RULES_EMPTY_NOTE.format(path=path))
    return NotesRules(rules, None)


def ensure_notes_rules(path: Path, *, shipped: Path = SHIPPED_RULES) -> None:
    """Copy the shipped file to ``path`` byte for byte, so the user has something to open.

    An existing file is never touched, whatever it holds, and nothing else is done then.
    Written whole or not at all: a failure leaves neither a half file nor a temporary one.
    """
    try:
        is_dir, exists = path.is_dir(), path.exists()
    except OSError as exc:  # even looking at it is refused
        logger.warning("Could not look at where the notes rules go: %s", exc)
        raise NotesRulesUnreadable(RULES_CREATE_FAILED.format(path=path)) from exc
    if is_dir:
        raise NotesRulesUnreadable(RULES_FOLDER_IN_THE_WAY.format(path=path))
    if exists:
        return
    try:
        data = shipped.read_bytes()
    except OSError as exc:
        logger.warning("Could not read the shipped notes rules: %s", exc)
        raise NotesRulesUnreadable(RULES_SHIPPED_MISSING.format(path=shipped)) from exc
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        files.create_once(path, data)
    except OSError as exc:
        logger.warning("Could not create the notes rules: %s", exc)
        raise NotesRulesUnreadable(RULES_CREATE_FAILED.format(path=path)) from exc
