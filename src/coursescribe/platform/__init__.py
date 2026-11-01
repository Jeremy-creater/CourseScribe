"""Everything that depends on the operating system lives under this folder."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from coursescribe.platform.windows import open_system_audio, prepare_gpu_libraries
else:
    from coursescribe.platform.macos import open_system_audio, prepare_gpu_libraries

__all__ = ["open_system_audio", "prepare_gpu_libraries"]
