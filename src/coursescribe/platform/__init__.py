"""Everything that depends on the operating system lives under this folder."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from coursescribe.platform.windows import (
        FILES_CAN_BE_HELD,
        LIMITS,
        cloud_sync_roots,
        file_identity_key,
        DEFAULT_HOTKEY,
        HOTKEY_LABEL,
        HOTKEY_PLATFORM,
        create_global_hotkey,
        hotkey_label,
        create_keep_awake,
        open_system_audio,
        prepare_gpu_libraries,
    )
else:
    from coursescribe.platform.macos import (
        FILES_CAN_BE_HELD,
        LIMITS,
        cloud_sync_roots,
        file_identity_key,
        DEFAULT_HOTKEY,
        HOTKEY_LABEL,
        HOTKEY_PLATFORM,
        create_global_hotkey,
        hotkey_label,
        create_keep_awake,
        open_system_audio,
        prepare_gpu_libraries,
    )

__all__ = [
    "DEFAULT_HOTKEY",
    "HOTKEY_PLATFORM",
    "FILES_CAN_BE_HELD",
    "LIMITS",
    "HOTKEY_LABEL",
    "hotkey_label",
    "cloud_sync_roots",
    "create_global_hotkey",
    "create_keep_awake",
    "file_identity_key",
    "open_system_audio",
    "prepare_gpu_libraries",
]
