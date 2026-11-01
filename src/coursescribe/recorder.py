"""One recording: pull sound from the system and write it straight to disk."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from coursescribe.wavfile import WavWriter

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


class RecordingSession:
    def __init__(
        self,
        open_capture: Callable[[], AudioCapture],
        path: Path,
        on_silence: Callable[[bool], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        silence_alert_after: float = SILENCE_ALERT_SECONDS,
        stop_timeout: float = 3.0,
    ) -> None:
        self._open_capture = open_capture
        self.path = path
        self._on_silence = on_silence or (lambda alerting: None)
        self._on_error = on_error or (lambda message: None)
        self._alert_after = silence_alert_after
        self._stop_timeout = stop_timeout
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
            with capture as source:
                self._writer = WavWriter(self.path, source.sample_rate, source.channels)
                self._ready.set()
                self._pump(source)
        except Exception as exc:
            if self._ready.is_set():
                logger.exception("Recording stopped by an error")
                self._on_error("录音中断了，已录下的部分都保存了。可以点“停止”继续处理这部分。")
            else:
                self._start_error = exc
                self._ready.set()

    def _pump(self, source: AudioCapture) -> None:
        monitor = SilenceMonitor(source.sample_rate, self._alert_after)
        frames_per_block = int(source.sample_rate * _BLOCK_SECONDS)
        while not self._stop.is_set():
            frames = source.read(frames_per_block)
            with self._write_lock:
                if self._writer is None:  # stop() gave up waiting on a hung device
                    return
                self._writer.write(frames)
            change = monitor.feed(frames)
            if change is not None:
                self._on_silence(change)

    def stop(self) -> float:
        """Finish the file and return its length, even if the device has hung."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(self._stop_timeout)
        with self._write_lock:
            if self._writer is not None:
                self._seconds = self._writer.seconds
                self._writer.close()
                self._writer = None
        return self._seconds
