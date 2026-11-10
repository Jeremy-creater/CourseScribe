"""User settings. Default locations come from the system, never a fixed path."""

from __future__ import annotations

import logging
import os
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir, user_documents_dir

from coursescribe.files import atomic_write_json, read_json
from coursescribe.models import DEFAULT_NOTES_MODEL, DEFAULT_PROOFREAD_MODEL, DEFAULT_TRANSCRIBE_MODEL
from coursescribe.platform import cloud_sync_roots

logger = logging.getLogger(__name__)

APP_NAME = "CourseScribe"
_FORMAT = 2  # from this format on, a saved folder is always one the user chose


def app_home() -> Path:
    """Where job state, settings and logs live. Tests may redirect it."""
    override = os.environ.get("COURSESCRIBE_HOME")
    return Path(override) if override else Path(user_data_dir(APP_NAME, appauthor=False))


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    transcribe_model: str
    proofread_model: str
    notes_model: str = DEFAULT_NOTES_MODEL
    auto_notes: bool = False  # organised notes made by themselves once a proofread .md is written


def _parts(path: str | Path | None) -> tuple[str, ...] | None:
    """A folder as comparable pieces, or None when it is not a usable full path."""
    text = "" if path is None else str(path).strip()
    if not text or not os.path.isabs(text):
        return None
    normal = unicodedata.normalize("NFC", os.path.normpath(text)).casefold()
    return tuple(part for part in normal.split(os.sep) if part)


def _within(path: tuple[str, ...], folder: tuple[str, ...]) -> bool:
    return path[: len(folder)] == folder


def _same_folder(a: str | Path | None, b: str | Path | None) -> bool:
    first, second = _parts(a), _parts(b)
    return first is not None and first == second


def choose_output_dir(documents: str | Path | None, home: Path, sync_roots: Iterable[str] | None) -> Path:
    """Documents/CourseScribe, unless that would be uploaded by OneDrive or iCloud.

    When it is not known where Documents is or what is synced, the home folder is
    the safe choice: recordings and transcripts must never leave this computer.
    """
    fallback = home / APP_NAME
    target = _parts(documents)
    if target is None or sync_roots is None:
        return fallback
    target += (APP_NAME.casefold(),)
    for root in sync_roots:
        folder = _parts(root)
        if folder is not None and _within(target, folder):
            return fallback
    return Path(documents) / APP_NAME


def _ask_system(question, what: str):
    try:
        return question()
    except Exception as exc:  # the program must still start; the safe default follows
        logger.warning("Could not find %s: %s", what, exc)
        return None


def _documents() -> str | None:
    found = _ask_system(user_documents_dir, "the Documents folder")
    if found is not None and _parts(found) is None:
        logger.warning("Could not find the Documents folder: the system gave %r", found)
        return None
    return found


def _defaults_for(documents: str | None) -> Settings:
    roots = _ask_system(cloud_sync_roots, "the synced folders")
    return Settings(choose_output_dir(documents, Path.home(), roots), DEFAULT_TRANSCRIBE_MODEL, DEFAULT_PROOFREAD_MODEL)


def default_settings() -> Settings:
    return _defaults_for(_documents())


def open_settings(path: Path) -> SettingsStore:
    """The settings store, knowing where older versions saved by default."""
    documents = _documents()
    old_default = Path(documents) / APP_NAME if documents is not None else None
    return SettingsStore(path, _defaults_for(documents), old_default_dir=old_default)


class SettingsStore:
    def __init__(self, path: Path, defaults: Settings, old_default_dir: Path | None = None) -> None:
        self.path = path
        self.defaults = defaults
        # Older versions always saved the folder, so their default looks like a choice.
        self.old_default_dir = old_default_dir

    def load(self) -> Settings:
        try:
            raw = read_json(self.path)
            folder = self._folder(raw)
            transcribe, proofread = _text(raw["transcribe_model"]), _text(raw["proofread_model"])
        except FileNotFoundError:
            return self.defaults
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Using default settings: %s", exc)
            return self.defaults
        # Added later, one by one: a file without them is an older version's, and one gone wrong
        # costs only itself, not the folder and models the user chose.
        return Settings(folder, transcribe, proofread, self._notes_model(raw), self._auto_notes(raw))

    def _notes_model(self, raw: dict) -> str:
        if "notes_model" not in raw:
            return self.defaults.notes_model
        value = raw["notes_model"]
        if isinstance(value, str) and value.strip():
            return value
        logger.warning("Using the default notes_model: the saved one is %r", value)
        return self.defaults.notes_model

    def _auto_notes(self, raw: dict) -> bool:
        if "auto_notes" not in raw:
            return self.defaults.auto_notes
        value = raw["auto_notes"]
        if type(value) is bool:  # not isinstance: 1 and 0 are ints that pass as bools
            return value
        logger.warning("Using the default auto_notes: the saved one is %r", value)
        return self.defaults.auto_notes

    def _folder(self, raw: dict) -> Path:
        if "output_dir" not in raw:
            return self.defaults.output_dir
        saved = raw["output_dir"]
        if not isinstance(saved, str) or _parts(saved) is None:
            raise ValueError(f"Saved folder is not a full path: {saved!r}")
        if "format" not in raw and self._was_old_default(saved):
            return self.defaults.output_dir
        return Path(saved)

    def _was_old_default(self, saved: str) -> bool:
        if self.old_default_dir is None:
            # Documents is unknown this time: any CourseScribe folder may be the old default, maybe synced.
            return _parts(saved)[-1:] == (APP_NAME.casefold(),)
        return _same_folder(saved, self.old_default_dir)

    def save(self, settings: Settings) -> None:
        data: dict = {
            "format": _FORMAT,
            "transcribe_model": settings.transcribe_model,
            "proofread_model": settings.proofread_model,
            # new keys need no new format: older versions read only the keys they know
            "notes_model": settings.notes_model,
            "auto_notes": settings.auto_notes,
        }
        if not _same_folder(settings.output_dir, self.defaults.output_dir):
            data["output_dir"] = str(settings.output_dir)
        atomic_write_json(self.path, data)


def _text(value) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected text, got {value!r}")
    return value
