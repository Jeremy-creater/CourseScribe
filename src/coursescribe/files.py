"""Crash-safe file writing. Finished files never overwrite anything."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from coursescribe.platform import FILES_CAN_BE_HELD

_REPLACE_TRIES = 20
_REPLACE_WAIT = 0.05  # seconds; one second in all is far longer than a reader holds a small file
# Where a held file shuts others out, it is reported the same way as a real refusal
# (reads do not even carry the Windows error code), so every refusal gets a short wait.
_FILES_CAN_BE_HELD = FILES_CAN_BE_HELD


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


def _write_temp(folder: Path, data: bytes) -> Path:
    """Write ``data`` durably to a temporary file in ``folder``."""
    descriptor, name = tempfile.mkstemp(dir=folder, prefix=".coursescribe-", suffix=".tmp")
    temp = Path(name)
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


def read_json(path: Path) -> Any:
    return json.loads(_patiently(lambda: path.read_text(encoding="utf-8")))
