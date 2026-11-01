"""Command line entry point: ``python -m coursescribe``."""

from __future__ import annotations

import sys

from coursescribe import __version__


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--version" in args:
        print(f"CourseScribe {__version__}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
