"""A WAV file that stays readable while it grows, and can be repaired after a crash."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

_HEADER_SIZE = 44
_BITS = 16
_RIFF_SIZE_AT = 4
_DATA_SIZE_AT = 40
_FLUSH_EVERY_SECONDS = 1.0


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

    def __init__(self, path: Path, rate: int, channels: int) -> None:
        self.rate = rate
        self.channels = channels
        self._stream = path.open("xb")
        self._stream.write(_header(rate, channels, 0))
        self._data_bytes = 0
        self._unflushed_frames = 0

    @property
    def seconds(self) -> float:
        return self._data_bytes / (self.channels * _BITS // 8) / self.rate

    def write(self, frames: np.ndarray) -> None:
        self._stream.write(frames.astype("<i2").tobytes())
        self._data_bytes += frames.size * _BITS // 8
        self._unflushed_frames += len(frames)
        if self._unflushed_frames >= self.rate * _FLUSH_EVERY_SECONDS:
            self._flush()

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
        self._flush()
        self._stream.close()


def repair_wav(path: Path) -> float:
    """Make a recording cut off by a crash readable. Returns its length in seconds.

    Returns 0.0, changing nothing, when the file has no usable sound.
    """
    try:
        size = path.stat().st_size
        with path.open("r+b") as stream:
            head = stream.read(_HEADER_SIZE)
            if len(head) < _HEADER_SIZE or head[:4] != b"RIFF" or head[8:12] != b"WAVE" or head[36:40] != b"data":
                return 0.0
            channels, rate = struct.unpack("<HI", head[22:28])
            block = struct.unpack("<H", head[32:34])[0]
            if not (channels and rate and block):
                return 0.0
            data_bytes = (size - _HEADER_SIZE) // block * block
            stream.truncate(_HEADER_SIZE + data_bytes)
            stream.seek(_RIFF_SIZE_AT)
            stream.write(struct.pack("<I", 36 + data_bytes))
            stream.seek(_DATA_SIZE_AT)
            stream.write(struct.pack("<I", data_bytes))
    except OSError:
        return 0.0
    return data_bytes / block / rate
