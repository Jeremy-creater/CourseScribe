"""Records everything the Mac plays, through a Core Audio tap (macOS 14.2 or later)."""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
_FRAME_BYTES = CHANNELS * 4  # 32-bit float samples, as the tap delivers them
_DEFAULT_READ_TIMEOUT = 1.0
_DEVICE_ID = "system-output-mix"

_PERMISSION_HELP = (
    "无法开始录系统声音。请在“系统设置 → 隐私与安全性”里允许本程序（或你用来启动它的终端）录制“系统音频”，"
    "然后再试一次。原因：{reason}"
)


def _default_session_factory(on_buffer):
    from catap import record_system_audio

    return record_system_audio(on_buffer=on_buffer)


class MacSystemAudio:
    """The whole system's sound mix. The tap is not tied to one output device, so switching
    speakers or headphones needs no reconnecting: ``is_default`` is always true.

    Delivers 48 kHz stereo 16-bit like the Windows source. When the system sends nothing for
    ``read_timeout`` seconds, ``read`` returns silence so stopping never waits on a quiet device.
    """

    sample_rate = SAMPLE_RATE
    channels = CHANNELS

    def __init__(self, session_factory=_default_session_factory, read_timeout: float = _DEFAULT_READ_TIMEOUT) -> None:
        self._factory = session_factory
        self._timeout = read_timeout
        self._session = None
        self._pending: deque[np.ndarray] = deque()
        self._available = 0
        self._cond = threading.Condition()
        self.device_name = ""
        self.device_id = _DEVICE_ID

    def __enter__(self) -> MacSystemAudio:
        from coursescribe.recorder import RecordingFailed

        session = self._factory(self._on_buffer)
        try:
            session.start()
        except Exception as exc:
            logger.warning("Could not start the system audio tap", exc_info=True)
            raise RecordingFailed(_PERMISSION_HELP.format(reason=exc)) from exc
        self._session = session
        return self

    def __exit__(self, *args) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.stop()

    def _on_buffer(self, buffer) -> None:
        data = buffer.data
        usable = len(data) - len(data) % _FRAME_BYTES  # drop a torn trailing frame
        if not usable:
            return
        block = np.frombuffer(data[:usable], dtype=np.float32).reshape(-1, CHANNELS)
        with self._cond:
            self._pending.append(block)
            self._available += len(block)
            self._cond.notify_all()

    def read(self, frames: int) -> np.ndarray:
        with self._cond:
            self._cond.wait_for(lambda: self._available >= frames, timeout=self._timeout)
            taken: list[np.ndarray] = []
            need = frames
            while need and self._pending:
                block = self._pending.popleft()
                if len(block) > need:
                    self._pending.appendleft(block[need:])
                    block = block[:need]
                taken.append(block)
                need -= len(block)
            self._available -= frames - need
        if need:
            taken.append(np.zeros((need, CHANNELS), dtype=np.float32))
        mixed = np.concatenate(taken) if taken else np.zeros((0, CHANNELS), dtype=np.float32)
        return (np.clip(mixed, -1.0, 1.0) * 32767).astype(np.int16)

    def is_default(self) -> bool:
        return True

    def default_id(self) -> str:
        return self.device_id
