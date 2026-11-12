"""One key for every way of naming the same file. Plain functions: no Windows-only import, so they run anywhere.

Keys are compared with ``==`` only and never stored.
"""

from __future__ import annotations

import os
import unicodedata


def windows_file_identity_key(path) -> str:
    """Windows ignores letter case in names: today's rule, kept exactly."""
    return os.path.normcase(os.path.realpath(path))


def _spelling_key(path) -> tuple[str, str]:
    """For a file that is not there (or cannot be asked about): the resolved spelling, composed and case-folded.

    ``casefold`` folds a sharp s to ``ss`` although a Mac disk keeps the two apart; that only reaches files
    that do not exist, because a file that exists is judged by the disk. NFC, not NFKC: a fullwidth letter
    is another name.
    """
    try:
        text = os.path.realpath(path)
    except (OSError, ValueError):  # a name too long for the system, or one with a null character in it
        text = os.fspath(path)
    return ("spelling", unicodedata.normalize("NFC", text).casefold())


def mac_file_identity_key(path) -> tuple:
    """A Mac volume may or may not ignore letter case, so a file that exists is asked of the disk.

    Two names for one file (case, a link, a second hard link, composed or decomposed letters) give the same
    device and file number. Asked every time: a new file under an old name is a different file.
    """
    try:
        found = os.stat(path)
    except (OSError, ValueError):  # gone, not allowed to look, drive dropped, name too long
        return _spelling_key(path)
    return ("disk", found.st_dev, found.st_ino)
