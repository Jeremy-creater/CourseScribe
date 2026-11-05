"""Files on disk: crash-safe writing that never overwrites a finished file, and reading
the text files the user saves (transcripts, the word list, the notes rules) without guessing."""

from __future__ import annotations

import codecs
import functools
import json
import logging
import os
import secrets
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from coursescribe.platform import FILES_CAN_BE_HELD

logger = logging.getLogger(__name__)

_REPLACE_TRIES = 20
_REPLACE_WAIT = 0.05  # seconds; one second in all is far longer than a reader holds a small file
# Where a held file shuts others out, it is reported the same way as a real refusal
# (reads do not even carry the Windows error code), so every refusal gets a short wait.
_FILES_CAN_BE_HELD = FILES_CAN_BE_HELD

# Not tempfile.mkstemp: on Windows it retries a refused creation up to two billion times, so a folder
# that denies new files hung the program. A clash or a refusal (a file of that name being deleted)
# gets a new random name, at most this many tries in all; the last error is raised.
_CREATE_TRIES = 10
# The name is ".coursescribe-" + 8 random characters + ".tmp", 26 in all, as the path budget in jobs.py expects.
_TEMP_PREFIX = ".coursescribe-"
_TEMP_SUFFIX = ".tmp"
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)


def _numbered(path: Path, number: int) -> Path:
    return path.with_name(f"{path.stem} ({number}){path.suffix}")


def unique_path(path: Path) -> Path:
    """Return ``path``, or the first free ``name (2).ext`` beside it."""
    candidate = path
    number = 2
    while candidate.exists():
        candidate = _numbered(path, number)
        number += 1
    return candidate


def unique_dir(path: Path) -> Path:
    """Create and return a new folder at ``path``, numbering it when taken."""
    path.parent.mkdir(parents=True, exist_ok=True)
    number = 2
    candidate = path
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.name} ({number})")
            number += 1


def _create_temp(folder: Path) -> tuple[int, Path]:
    """Create a new, empty temporary file in ``folder`` that only we can read; give up after a few tries."""
    tries = 0
    while True:
        tries += 1
        temp = folder / f"{_TEMP_PREFIX}{secrets.token_hex(4)}{_TEMP_SUFFIX}"
        try:
            return os.open(temp, _CREATE_FLAGS, 0o600), temp
        except (FileExistsError, PermissionError):
            if tries >= _CREATE_TRIES:
                raise  # the last error: no waiting, a new name is all a clash or a passing refusal needs


def _write_temp(folder: Path, data: bytes) -> Path:
    """Write ``data`` durably to a temporary file in ``folder``."""
    descriptor, temp = _create_temp(folder)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _publish(temp: Path, target: Path, data: bytes) -> None:
    """Give ``target`` its content, raising FileExistsError rather than replacing."""
    try:
        os.link(temp, target)  # unlike a rename, fails when the target exists
    except FileExistsError:
        raise
    except OSError:  # drives without hard links, such as exFAT
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def _remove_temp(temp: Path) -> None:
    """Clean up after publishing; a refusal (antivirus holding it) leaves a hidden file, not a failure."""
    try:
        temp.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove the temporary file %s: %s", temp.name, exc)


def create_once(path: Path, data: bytes) -> None:
    """Give ``path`` exactly ``data`` if nothing is there; whatever is there already stays.

    Written whole or not at all. The folder must exist. A failure to write raises OSError;
    failing to remove the temporary file afterwards does not.
    """
    temp = _write_temp(path.parent, data)
    try:
        _publish(temp, path, data)
    except FileExistsError:
        pass  # another program created it meanwhile; theirs stays
    finally:
        _remove_temp(temp)


def commit_new_text(path: Path, text: str) -> Path:
    """Atomically publish ``text`` at ``path`` without replacing another file.

    If ``path`` already holds exactly this text, an earlier run got this far
    before being interrupted, so the existing file is the result. Any other
    existing file belongs to someone else and the text goes to a numbered name.
    """
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _write_temp(path.parent, data)
    try:
        target = path
        number = 2
        while True:
            if target.exists() and target.read_bytes() == data:
                return target
            try:
                _publish(temp, target, data)
                return target
            except FileExistsError:
                target = _numbered(path, number)
                number += 1
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace ``path`` with ``value``; a crash leaves the old or the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    temp = _write_temp(path.parent, data)
    try:
        _patiently(lambda: os.replace(temp, path))
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _patiently(action: Callable[[], Any]) -> Any:
    """Windows refuses while another program (antivirus, indexer, our own save) has the file open."""
    for _ in range(_REPLACE_TRIES - 1):
        try:
            return action()
        except PermissionError:
            if not _FILES_CAN_BE_HELD:
                raise  # elsewhere a refusal is real and does not clear by waiting
            time.sleep(_REPLACE_WAIT)
    return action()


class UnknownEncoding(ValueError):
    """The bytes are not in an encoding that can be told for sure."""


_MARKS = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


def _read_bytes_once(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read()


def read_transcript(path: Path) -> str:
    """Read a transcript the user may have re-saved, exactly as written; this never changes the file.

    Only encodings that can be told for sure are accepted: UTF-8, and UTF-8 or UTF-16 marked
    by a byte order mark. Anything else raises UnknownEncoding rather than being guessed, because
    a wrong guess would be proofread into a normal-looking but garbled .md.
    """
    data = _patiently(lambda: _read_bytes_once(path))
    body, encoding = data, "utf-8"
    for mark, marked_encoding in _MARKS:
        if data.startswith(mark):
            body, encoding = data[len(mark) :], marked_encoding
            break
    try:
        text = body.decode(encoding)
    except UnicodeDecodeError as exc:
        raise UnknownEncoding(f"{path.name} is not {encoding}") from exc
    if "\x00" in text:  # UTF-16 without a mark often decodes as UTF-8 full of NULs
        raise UnknownEncoding(f"{path.name} has NUL characters")
    return text


# --- files the user edits: the word list and the notes rules ---

_CHINESE = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0x3000, 0x303F), (0xFF00, 0xFFEF))
_WESTERN = ((0x0000, 0x007F), (0x00A0, 0x024F), (0x2000, 0x206F), (0x2190, 0x21FF))
_ALLOWED_CONTROLS = frozenset("\t\r\n")


def _in(ch: str, ranges) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in ranges)


def _has_chinese(text: str) -> bool:
    return any(_in(ch, _CHINESE) for ch in text)


@functools.lru_cache(maxsize=4096)
def _in_gb2312(ch: str) -> bool:
    try:
        ch.encode("gb2312")
    except UnicodeEncodeError:
        return False
    return True


def _as_gbk(data: bytes) -> str | None:
    """Notepad's "ANSI" on Chinese Windows; only text inside GB2312 is believed, the rest is usually a misreading."""
    try:
        text = data.decode("gbk")
        text.encode("gb2312")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return None
    return text


def decode_user_text(data: bytes) -> str | None:
    """The text of a file the user edits (the word list, the notes rules), or None when its encoding
    cannot be told for sure. Same on every system."""
    text: str | None = None
    for mark, encoding in _MARKS:
        if data.startswith(mark):
            try:
                text = data[len(mark) :].decode(encoding)
            except UnicodeDecodeError:
                return None
            break
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = _as_gbk(data)
        else:
            # GBK bytes can happen to be valid UTF-8 (学习, 之, 浏电保). Read that way they give characters
            # a Chinese word list does not have: outside GB2312 and not Western. Then GBK is believed instead,
            # when it reads cleanly into Chinese. A UTF-8 file whose characters are all in GB2312 stays UTF-8.
            if any(not _in(ch, _WESTERN) and not _in_gb2312(ch) for ch in text):
                gbk = _as_gbk(data)
                if gbk is not None and _has_chinese(gbk):
                    text = gbk
    if text is None:
        return None
    if any(unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROLS for ch in text):
        return None  # NULs from UTF-16 without a mark, or bytes that were never text
    return text


def read_json(path: Path) -> Any:
    return json.loads(_patiently(lambda: path.read_text(encoding="utf-8")))
