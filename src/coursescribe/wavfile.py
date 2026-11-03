"""A WAV file that stays readable while it grows, and can be repaired after a crash."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from coursescribe.files import _patiently

_HEADER_SIZE = 44
_BITS = 16
_RIFF_SIZE_AT = 4
_DATA_SIZE_AT = 40
_FLUSH_EVERY_SECONDS = 1.0
# The header stores sizes in 32 bits; stop a little short so every block size fits.
MAX_DATA_BYTES = (0xFFFFFFFF - 36) // 64 * 64  # about 6.2 hours of 48 kHz stereo


def _header(rate: int, channels: int, data_bytes: int) -> bytes:
    block = channels * _BITS // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_bytes, b"WAVE",
        b"fmt ", 16, 1, channels, rate, rate * block, block, _BITS,
        b"data", data_bytes,
    )


class WavWriter:
    """Appends 16-bit audio and forces it to disk about once per second of sound."""

    def __init__(self, path: Path, rate: int, channels: int, max_data_bytes: int = MAX_DATA_BYTES) -> None:
        self.rate = rate
        self.channels = channels
        self._max_data_bytes = max_data_bytes
        self._stream = path.open("xb")
        self._stream.write(_header(rate, channels, 0))
        self._data_bytes = 0
        self._unflushed_frames = 0

    @property
    def seconds(self) -> float:
        return self._data_bytes / (self.channels * _BITS // 8) / self.rate

    def write(self, frames: np.ndarray) -> bool:
        """Append ``frames``. Returns False once the file is full; the rest is not written."""
        frame_bytes = self.channels * _BITS // 8
        room = (self._max_data_bytes - self._data_bytes) // frame_bytes
        fits = frames[:room]
        self._stream.write(fits.astype("<i2").tobytes())
        self._data_bytes += len(fits) * frame_bytes
        self._unflushed_frames += len(fits)
        full = len(fits) < len(frames) or self._data_bytes >= self._max_data_bytes
        if full or self._unflushed_frames >= self.rate * _FLUSH_EVERY_SECONDS:
            self._flush()
        return not full

    def _flush(self) -> None:
        self._stream.seek(_RIFF_SIZE_AT)
        self._stream.write(struct.pack("<I", 36 + self._data_bytes))
        self._stream.seek(_DATA_SIZE_AT)
        self._stream.write(struct.pack("<I", self._data_bytes))
        self._stream.seek(0, os.SEEK_END)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._unflushed_frames = 0

    def close(self) -> None:
        if self._stream.closed:
            return
        try:
            self._flush()
        finally:
            self._stream.close()


def repair_wav(path: Path, max_data_bytes: int = MAX_DATA_BYTES) -> float:
    """Make a recording cut off by a crash readable. Returns its length in seconds.

    Returns 0.0 when the file is there but has no usable sound. A file that is gone,
    held by another program or cannot be written raises its OSError instead, so it
    is never taken for an empty recording. Repairing twice is harmless, so a held
    file is simply tried again.
    """
    return _patiently(lambda: _repair(path, max_data_bytes))


def _repair(path: Path, max_data_bytes: int) -> float:
    with path.open("r+b") as stream:
        head = stream.read(_HEADER_SIZE)
        if len(head) < _HEADER_SIZE or head[:4] != b"RIFF" or head[8:12] != b"WAVE" or head[36:40] != b"data":
            return 0.0
        channels, rate = struct.unpack("<HI", head[22:28])
        block = struct.unpack("<H", head[32:34])[0]
        if not (channels and rate and block):
            return 0.0
        size = stream.seek(0, os.SEEK_END)
        data_bytes = min(size - _HEADER_SIZE, max_data_bytes) // block * block
        stream.truncate(_HEADER_SIZE + data_bytes)
        stream.seek(_RIFF_SIZE_AT)
        stream.write(struct.pack("<I", 36 + data_bytes))
        stream.seek(_DATA_SIZE_AT)
        stream.write(struct.pack("<I", data_bytes))
    return data_bytes / block / rate
