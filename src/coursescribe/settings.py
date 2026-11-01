"""User settings. Default locations come from the system, never a fixed path."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir, user_documents_dir

from coursescribe.files import atomic_write_json, read_json
from coursescribe.models import DEFAULT_PROOFREAD_MODEL, DEFAULT_TRANSCRIBE_MODEL

logger = logging.getLogger(__name__)

APP_NAME = "CourseScribe"


def app_home() -> Path:
    """Where job state, settings and logs live. Tests may redirect it."""
    override = os.environ.get("COURSESCRIBE_HOME")
    return Path(override) if override else Path(user_data_dir(APP_NAME, appauthor=False))


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    transcribe_model: str
    proofread_model: str


def default_settings() -> Settings:
    return Settings(Path(user_documents_dir()) / APP_NAME, DEFAULT_TRANSCRIBE_MODEL, DEFAULT_PROOFREAD_MODEL)


class SettingsStore:
    def __init__(self, path: Path, defaults: Settings) -> None:
        self.path = path
        self.defaults = defaults

    def load(self) -> Settings:
        try:
            raw = read_json(self.path)
            return Settings(Path(raw["output_dir"]), raw["transcribe_model"], raw["proofread_model"])
        except FileNotFoundError:
            return self.defaults
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Using default settings: %s", exc)
            return self.defaults

    def save(self, settings: Settings) -> None:
        atomic_write_json(
            self.path,
            {
                "output_dir": str(settings.output_dir),
                "transcribe_model": settings.transcribe_model,
                "proofread_model": settings.proofread_model,
            },
        )
