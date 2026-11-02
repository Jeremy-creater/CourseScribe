"""One recording: pull sound from the system and write it straight to disk."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from coursescribe.wavfile import MAX_DATA_BYTES, WavWriter

logger = logging.getLogger(__name__)

_BLOCK_SECONDS = 0.25
_SILENCE_LEVEL = 0.001  # peak, as a share of full scale: inaudible
SILENCE_ALERT_SECONDS = 30.0


class RecordingFailed(Exception):
    """Recording could not start. The message is shown to the user."""


class AudioCapture(Protocol):
    sample_rate: int
    channels: int

    def __enter__(self) -> AudioCapture: ...
    def __exit__(self, *args) -> None: ...
    def read(self, frames: int) -> np.ndarray: ...


class SilenceMonitor:
    """Says when sound has been absent for a long time, and when it returns."""

    def __init__(self, rate: int, alert_after: float = SILENCE_ALERT_SECONDS) -> None:
        self.rate = rate
        self.alert_after = alert_after
        self._silent_seconds = 0.0
        self._alerting = False

    def feed(self, frames: np.ndarray) -> bool | None:
        """Returns True when the alert begins, False when it ends, otherwise None."""
        if frames.size and np.abs(frames.astype(np.int32)).max() / 32768 >= _SILENCE_LEVEL:
            self._silent_seconds = 0.0
            if self._alerting:
                self._alerting = False
                return False
            return None
        self._silent_seconds += len(frames) / self.rate
        if not self._alerting and self._silent_seconds >= self.alert_after:
            self._alerting = True
            return True
        return None


class _WriteFailed(Exception):
    """The disk refused the sound; reconnecting the device would not help."""


_ENDED_DISK = "录音写不进磁盘了（可能磁盘满了），已自动停止。已录下的部分会转成文字。"
_ENDED_DEVICE = "录音中断了：声音设备断开后没能重新连接，已自动停止。已录下的部分会转成文字。"
_ENDED_FULL = "录音已达到单个文件的上限（约 6 小时），已自动停止，开始转成文字。"
_RECONNECTED = "声音设备断开过一次，已重新连接，接着录。"


class RecordingSession:
    """Records until stopped. Survives the device dropping out; ends itself only when it must.

    ``on_error`` gets notices the user should see while recording goes on;
    ``on_ended`` is called once if the recording had to stop by itself.
    """

    def __init__(
        self,
        open_capture: Callable[[], AudioCapture],
        path: Path,
        on_silence: Callable[[bool], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_ended: Callable[[str], None] | None = None,
        silence_alert_after: float = SILENCE_ALERT_SECONDS,
        stop_timeout: float = 3.0,
        reconnect_for: float = 60.0,
        reconnect_wait: float = 1.0,
        max_data_bytes: int = MAX_DATA_BYTES,
    ) -> None:
        self._open_capture = open_capture
        self.path = path
        self._on_silence = on_silence or (lambda alerting: None)
        self._on_error = on_error or (lambda message: None)
        self._on_ended = on_ended or (lambda message: None)
        self._alert_after = silence_alert_after
        self._stop_timeout = stop_timeout
        self._reconnect_for = reconnect_for
        self._reconnect_wait = reconnect_wait
        self._max_data_bytes = max_data_bytes
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="recording", daemon=True)
        self._writer: WavWriter | None = None
        self._write_lock = threading.Lock()
        self._seconds = 0.0

    @property
    def seconds(self) -> float:
        with self._write_lock:
            return self._writer.seconds if self._writer else self._seconds

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()
        if self._start_error is not None:
            logger.error("Could not start recording: %s", self._start_error)
            raise RecordingFailed("没能开始录音。请确认电脑有可用的声音输出设备，然后再试一次。")

    def _run(self) -> None:
        try:
            capture = self._open_capture()
            source = capture.__enter__()
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            return
        try:
            self._writer = WavWriter(self.path, source.sample_rate, source.channels, self._max_data_bytes)
        except Exception as exc:
            _close_quietly(capture)
            self._start_error = exc
            self._ready.set()
            return
        sound_format = (source.sample_rate, source.channels)
        self._ready.set()
        monitor = SilenceMonitor(source.sample_rate, self._alert_after)
        while True:
            try:
                full = self._pump(source, monitor)
            except _WriteFailed:
                logger.exception("Recording could not be written")
                _close_quietly(capture)
                self._on_ended(_ENDED_DISK)
                return
            except Exception:
                logger.exception("The sound device stopped delivering")
                _close_quietly(capture)
                capture, source = self._reconnect(sound_format)
                if capture is None:
                    if not self._stop.is_set():
                        self._on_ended(_ENDED_DEVICE)
                    return
                self._on_error(_RECONNECTED)
                continue
            _close_quietly(capture)
            if full:
                self._on_ended(_ENDED_FULL)
            return

    def _reconnect(self, sound_format: tuple[int, int]):
        """The same device format again, or (None, None) when it does not come back in time."""
        deadline = time.monotonic() + self._reconnect_for
        while not self._stop.wait(self._reconnect_wait):
            try:
                capture = self._open_capture()
                source = capture.__enter__()
            except Exception as exc:
                logger.info("Sound device not back yet: %s", exc)
            else:
                if (source.sample_rate, source.channels) == sound_format:
                    logger.info("Sound device reconnected")
                    return capture, source
                logger.warning("Sound device came back in another format; ending the recording")
                _close_quietly(capture)
                return None, None
            if time.monotonic() >= deadline:
                break
        return None, None

    def _pump(self, source: AudioCapture, monitor: SilenceMonitor) -> bool:
        """Copy sound to the file until stopped. Returns True if the file filled up."""
        frames_per_block = int(source.sample_rate * _BLOCK_SECONDS)
        while not self._stop.is_set():
            frames = source.read(frames_per_block)
            with self._write_lock:
                if self._writer is None:  # stop() gave up waiting on a hung device
                    return False
                try:
                    room_left = self._writer.write(frames)
                except OSError as exc:
                    raise _WriteFailed from exc
            if not room_left:
                return True
            change = monitor.feed(frames)
            if change is not None:
                self._on_silence(change)
        return False

    def stop(self) -> float:
        """Finish the file and return its length, even if the device hangs or the disk fails."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(self._stop_timeout)
        with self._write_lock:
            if self._writer is not None:
                self._seconds = self._writer.seconds
                try:
                    self._writer.close()
                except Exception:
                    logger.exception("Could not finish the recording file; repairing it on the next start")
                self._writer = None
        return self._seconds


def _close_quietly(capture) -> None:
    try:
        capture.__exit__(None, None, None)
    except Exception:
        logger.warning("Could not close the sound device", exc_info=True)
