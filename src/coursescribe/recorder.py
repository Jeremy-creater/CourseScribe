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
    device_name: str  # the device's name in the system, "" when unknown
    device_id: str  # the system's ID for this device, as default_id() gives it

    def __enter__(self) -> AudioCapture: ...
    def __exit__(self, *args) -> None: ...
    def read(self, frames: int) -> np.ndarray: ...

    def is_default(self) -> bool:
        """Whether this device is still the system's default output (always True for a chosen one)."""
        ...

    def default_id(self) -> str:
        """The system's default output device now. Raises when there is none."""
        ...


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
_SWITCHED = "电脑的声音换到了“{name}”，录音已跟着换过去，接着录进同一个文件。"
_SWITCH_FAILED = (
    "电脑的声音换到了别的设备，但录音没能跟过去，现在可能录不到声音。"
    "请把声音换回原来的设备，或者关掉独占这个设备的程序；程序每隔几秒会自动再试。"
)
_SWITCHED_BACK = "声音回到了正在录的设备，录音正常。"
_NEW_DEVICE = "新的输出设备"


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
        clock: Callable[[], float] = time.monotonic,
        switch_check_every: float = 1.0,
        switch_retry_after: float = 5.0,
        max_gap_fill: float = 2.0,
    ) -> None:
        self._clock = clock
        self._switch_check_every = switch_check_every
        self._switch_retry_after = switch_retry_after
        self._max_gap_fill = max_gap_fill
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
        self._finished = True
        # Used only by the recording thread:
        self._capture: AudioCapture | None = None
        self._source: AudioCapture | None = None
        self._format = (0, 0)
        self._since_check = 0  # frames written since the default output was last checked
        self._since_attempt = 0  # frames written since the last try to move to another device
        self._failed_target: str | None = None  # the default we could not move to, while that lasts
        self._asking_failed = False

    @property
    def seconds(self) -> float:
        with self._write_lock:
            return self._writer.seconds if self._writer else self._seconds

    @property
    def finished(self) -> bool:
        """False when the file could not be closed properly: repair its header before use."""
        return self._finished

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
        self._capture, self._source = capture, source
        self._format = (source.sample_rate, source.channels)
        self._ready.set()
        monitor = SilenceMonitor(source.sample_rate, self._alert_after)
        while True:
            try:
                full = self._pump(monitor)
            except _WriteFailed:
                logger.exception("Recording could not be written")
                _close_quietly(self._capture)
                self._on_ended(_ENDED_DISK)
                return
            except Exception:
                logger.exception("The sound device stopped delivering")
                lost = self._source.device_name
                _close_quietly(self._capture)
                capture, source = self._reconnect(self._format)
                if capture is None:
                    if not self._stop.is_set():
                        self._on_ended(_ENDED_DEVICE)
                    return
                if self._stop.is_set():  # Stop came while the device was reopening
                    _close_quietly(capture)
                    return
                self._capture, self._source = capture, source
                self._failed_target = None
                name = source.device_name
                self._on_error(_RECONNECTED if name == lost else _SWITCHED.format(name=name or _NEW_DEVICE))
                continue
            _close_quietly(self._capture)
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

    def _pump(self, monitor: SilenceMonitor) -> bool:
        """Copy sound to the file until stopped. Returns True if the file filled up."""
        rate = self._format[0]
        frames_per_block = int(rate * _BLOCK_SECONDS)
        check_every = round(self._switch_check_every * rate)
        while not self._stop.is_set():
            frames = self._source.read(frames_per_block)
            room_left = self._write(frames)
            if room_left is None:  # stop() gave up waiting on a hung device
                return False
            if not room_left:
                return True
            change = monitor.feed(frames)
            if change is not None:
                self._on_silence(change)
            if self._stop.is_set():  # Stop came during that block: no more checks, switches or notices
                break
            # Counted in sound written, not wall time: the silent stream keeps sound flowing.
            self._since_check += len(frames)
            self._since_attempt += len(frames)
            if self._since_check >= check_every:
                self._since_check = 0
                if self._follow_default():
                    return True
        return False

    def _write(self, frames: np.ndarray) -> bool | None:
        """Append to the file: True while there is room, False once full, None after stop()."""
        with self._write_lock:
            if self._writer is None:
                return None
            try:
                return self._writer.write(frames)
            except OSError as exc:
                raise _WriteFailed from exc

    def _follow_default(self) -> bool:
        """Move to the system's default output if it changed. Returns True if the file filled up.

        A device the user chose by name always says it is the default, so it is never left.
        """
        try:
            # Asked once more when not: the default may have come back between the two questions.
            target = None if self._source.is_default() else self._source.default_id()
            if target == self._source.device_id:
                target = None
        except Exception as exc:  # no output device at all, or the sound service restarting
            if not self._asking_failed:
                logger.warning("Could not tell which output device is the default; staying: %s", exc)
                self._asking_failed = True
            return False
        self._asking_failed = False
        if target is None:
            if self._failed_target is not None:
                self._failed_target = None
                logger.info("The default output is the recorded device again")
                self._tell(_SWITCHED_BACK)
            return False
        retry_after = round(self._switch_retry_after * self._format[0])
        if target == self._failed_target and self._since_attempt < retry_after:
            return False
        return self._switch_to(target)

    def _switch_to(self, target: str) -> bool:
        """Open the new default while the old one still records; let go of the old one once it works."""
        rate, channels = self._format
        self._since_attempt = 0
        started = self._clock()
        try:
            capture = self._open_capture()
            source = capture.__enter__()
        except Exception as exc:
            self._switch_failed(target, "Could not open the new default output device: %s", exc)
            return False
        if self._stop.is_set():  # Stop came while it opened: let go of it and record nothing more
            _close_quietly(capture)
            return False
        new_format = (source.sample_rate, source.channels)
        if new_format != self._format:
            _close_quietly(capture)
            self._switch_failed(
                target, "The new default output device came in another format: %s, not %s", new_format, self._format
            )
            return False
        before = self._source.device_name
        _close_quietly(self._capture)
        self._capture, self._source = capture, source
        self._failed_target = None
        # Nothing was recorded while opening and closing: silence keeps the length true to the clock.
        gap = round((self._clock() - started) * rate)
        most = round(self._max_gap_fill * rate)
        if gap > most:
            logger.warning("Switching output devices took %.1f s; filling only %.1f s", gap / rate, most / rate)
            gap = most
        logger.info("Followed the default output device from %r to %r", before, source.device_name)
        if gap > 0:
            room_left = self._write(np.zeros((gap, channels), dtype=np.int16))
            if room_left is None:
                return False
            if not room_left:
                return True
        self._tell(_SWITCHED.format(name=source.device_name or _NEW_DEVICE))
        return False

    def _switch_failed(self, target: str, message: str, *args) -> None:
        """Keep recording the old device. Tell the user once per new target; warn once per failing stretch."""
        if self._failed_target is None:
            logger.warning(message, *args)
        else:
            logger.debug(message, *args)
        if target != self._failed_target:
            self._tell(_SWITCH_FAILED)
        self._failed_target = target

    def _tell(self, message: str) -> None:
        """A notice about following the default output; none once Stop was asked for."""
        if not self._stop.is_set():
            self._on_error(message)

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
                    logger.exception("Could not finish the recording file; it is repaired before use")
                    self._finished = False
                self._writer = None
        return self._seconds


def _close_quietly(capture) -> None:
    try:
        capture.__exit__(None, None, None)
    except Exception:
        logger.warning("Could not close the sound device", exc_info=True)
