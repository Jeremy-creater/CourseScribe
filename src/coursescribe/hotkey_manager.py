"""Owns the one live recording shortcut: switching to a new one, keeping the old one when that fails.

No window and no operating-system code here. The system part is a factory, ``create(shortcut, callback)``,
that returns an object with ``start() -> bool`` and ``stop()``. Because Mac refuses a second registration of
the same keys in one program, the old shortcut is released before the new one is tried and put back if the
new one fails.

Threading: on a Mac ``start`` must run on the main thread, so ``apply`` must be called from there (the settings
window closes on that thread, and so does start-up).
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from coursescribe.shortcut import InvalidShortcut, Shortcut, parse

logger = logging.getLogger(__name__)

TAKEN_AT_START = "快捷键被别的程序占用了，请用按钮"


class Outcome(enum.Enum):
    APPLIED = "applied"  # the requested shortcut is the live one
    KEPT_OLD = "kept_old"  # the request failed; the old one still works
    NONE = "none"  # nothing is registered now
    CLOSED = "closed"  # shut down; nothing was done


@dataclass(frozen=True)
class ApplyResult:
    outcome: Outcome
    label: str  # the live shortcut's label, "" when there is none
    message: str  # what to tell the user, "" when all is well


class HotkeyManager:
    def __init__(
        self,
        create: Callable[[Shortcut, Callable[[], None]], object],
        callback: Callable[[], None],
        platform: str,
        label_of: Callable[[Shortcut], str],
        on_change: Callable[[str, str], None] | None = None,
    ) -> None:
        self._create = create
        self._callback = callback
        self._platform = platform
        self._label_of = label_of
        self._on_change = on_change
        self._handle = None
        self.live: Shortcut | None = None
        self._closed = False

    @property
    def label(self) -> str:
        return self._label_of(self.live) if self.live is not None else ""

    def apply_text(self, text: object) -> ApplyResult:
        try:
            shortcut = parse(text, self._platform)
        except InvalidShortcut as exc:
            return ApplyResult(Outcome.KEPT_OLD, self.label, exc.reason)
        return self.apply(shortcut)

    def apply_saved(self, settings, keep: Callable[[object], None]) -> ApplyResult:
        """Apply the saved shortcut; when it was refused and the old one stays live, `keep` the live one instead.

        Otherwise the settings window and the next start would show a combination that is not the one working.
        When nothing could be registered the user's choice stays saved, so a later start can try it again.
        """
        result = self.apply_text(settings.hotkey)
        if result.outcome is Outcome.KEPT_OLD and self.live is not None:
            keep(replace(settings, hotkey=self.live.serialize()))
        return result

    def apply(self, shortcut: Shortcut) -> ApplyResult:
        if self._closed:
            return ApplyResult(Outcome.CLOSED, "", "")
        if self.live is not None and shortcut == self.live:
            return ApplyResult(Outcome.APPLIED, self.label, "")  # nothing to do: no stop, no new registration
        old = self.live
        self._release()
        if self._register(shortcut):
            return self._settle(Outcome.APPLIED, "")
        wanted = self._label_of(shortcut)
        if old is None:
            return self._settle(Outcome.NONE, TAKEN_AT_START)
        if self._register(old):
            return self._settle(Outcome.KEPT_OLD, f"快捷键 {wanted} 被别的程序或系统占用，继续用 {self._label_of(old)}")
        return self._settle(
            Outcome.NONE,
            f"现在没有快捷键：{wanted} 被别的程序或系统占用，原来的 {self._label_of(old)} 也没能恢复，请用按钮",
        )

    def shutdown(self) -> None:
        self._closed = True
        self._release()

    def _settle(self, outcome: Outcome, message: str) -> ApplyResult:
        if self._on_change is not None:
            self._on_change(self.label, message)
        return ApplyResult(outcome, self.label, message)

    def _release(self) -> None:
        handle, self._handle, self.live = self._handle, None, None
        if handle is not None:
            self._stop(handle)

    @staticmethod
    def _stop(handle) -> None:
        try:
            handle.stop()
        except Exception:
            logger.exception("Could not release the shortcut")

    def _register(self, shortcut: Shortcut) -> bool:
        handle = None
        try:
            handle = self._create(shortcut, self._callback)
            started = bool(handle.start())
        except Exception:
            logger.exception("The system could not register the shortcut")
            started = False
        if not started:
            if handle is not None:
                self._stop(handle)
            return False
        self._handle, self.live = handle, shortcut
        return True
