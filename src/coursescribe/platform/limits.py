"""How long names and paths may be on a system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathLimits:
    """A limit that is ``None`` does not apply on that system."""

    path_units: int | None  # the whole path, in UTF-16 units (Windows)
    component_bytes: int | None  # each name in the path, in UTF-8 bytes (Mac)
    path_bytes: int | None  # the whole path, in UTF-8 bytes (Mac)
