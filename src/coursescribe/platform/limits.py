"""How long names and paths may be on a system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathLimits:
    """A limit that is ``None`` does not apply on that system."""

    path_units: int | None  # the whole path, in UTF-16 units (Windows)
    component_bytes: int | None  # each name in the path, in UTF-8 bytes (Mac)
    path_bytes: int | None  # the whole path, in UTF-8 bytes (Mac)


# Windows counts the whole path, in UTF-16 units (MAX_PATH less the terminating null).
WINDOWS_LIMITS = PathLimits(path_units=259, component_bytes=None, path_bytes=None)
# The Mac counts each name (255 bytes) and the whole path (PATH_MAX 1024 less the terminating null) in UTF-8 bytes.
MAC_LIMITS = PathLimits(path_units=None, component_bytes=255, path_bytes=1023)
