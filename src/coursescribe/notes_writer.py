"""Write organised notes with the model: one call, checked, and its raw reply kept for a restart.

Order: the proofread text has content -> read the rules -> it fits the model's context ->
a checkpoint of the same input gives the reply without asking -> the model is installed ->
ask, at most twice (once more when the reply is cut off, empty, or has nothing the source
backs) -> keep the raw reply -> release the model. Nothing here writes the notes file.
Logs never carry the lesson, the rules or the reply: they may be pasted into a report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from coursescribe.files import atomic_write_json, read_json
from coursescribe.notes import (
    EmptyNotes,
    NotesMeta,
    NotesResult,
    build_prompt,
    finish_notes,
    overflow,
    source_body,
    structure_notes,
)
from coursescribe.notes_rules import NotesRules, NotesRulesUnreadable, load_notes_rules, rules_path
from coursescribe.ollama import (
    OllamaBusy,
    OllamaClient,
    OllamaDown,
    OllamaError,
    OllamaIncomplete,
    OllamaModelMissing,
    OllamaTimeout,
)
from coursescribe.text import content_chars

logger = logging.getLogger(__name__)

T = TypeVar("T")

NOTES_NUM_CTX = 32768  # starting value; item 13 settles it
NOTES_REPLY_BUDGET = 8192  # tokens kept for the reply, also sent as num_predict; item 13 settles it
NOTES_TIMEOUT = 1800.0  # seconds one call may take; item 13 settles it

NOTES_EMPTY_SOURCE = "这份校对稿是空的：去掉开头的标题以后没有文字，所以没有整理笔记。在校对稿里补上内容后可以点“重试”。"
NOTES_TOO_LONG = (
    "这节课约 {chars} 字，笔记模型一次最多能整理约 {limit} 字，所以没有整理笔记，校对稿没有变。"
    "太长的课以后会支持分段整理；现在可以把校对稿另存一份、删短一些，再对新的那份整理笔记。"
)
NOTES_RULES_TOO_LONG = (
    "整理规则约 {rules} 字，太长了：加上给笔记留的位置，笔记模型一次放不下任何课文，所以没有整理笔记，校对稿没有变。"
    "在“设置 → 打开整理规则…”里把规则改短，再点“重试”。"
)
NOTES_DOWN = "整理笔记要用的服务（Ollama）没有启动。校对稿已保存，没有变。请先打开 Ollama，然后点“继续整理笔记”。"
NOTES_TIMEOUT_MESSAGE = (
    "整理笔记的服务（Ollama）太久没有写完，可能正忙，或显卡内存（显存）不够跑得很慢。校对稿已保存，没有变。"
    "请稍后点“继续整理笔记”；一直这样的话，请关掉其他占用显卡的程序，或在设置里换一个小一点的笔记模型。"
)
NOTES_BUSY_MESSAGE = (
    "整理笔记的服务（Ollama）暂时出错了，可能是显卡内存（显存）不够或服务不稳定。校对稿已保存，没有变。"
    "请稍后点“继续整理笔记”；一直这样的话，请重启 Ollama（在屏幕右下角的托盘图标里退出 Ollama 再打开），"
    "或在设置里换一个小一点的笔记模型。"
)
NOTES_MODEL_MISSING = (
    "Ollama 里没有笔记模型 {model}，程序不会自己下载。校对稿已保存，没有变。"
    "请在命令行运行 ollama pull {model}，或在设置里换成已有的模型，然后点“继续整理笔记”。"
)
# The two attempts may fail in different ways: these name the last one only.
NOTES_CUT_OFF = (
    "笔记模型试了两次都没写成，最后一次写到长度上限被截断了，所以没有保存半截的笔记，校对稿没有变。"
    "可以点“重试”；一直这样的话，可能是这节课内容太多，可以在设置里换一个笔记模型。"
)
NOTES_EMPTY_REPLY = (
    "笔记模型试了两次都没写成，最后一次没有写出内容，所以没有生成笔记，校对稿没有变。"
    "可以点“重试”，或在设置里换一个笔记模型。"
)
NOTES_ALL_REMOVED = (
    "笔记模型试了两次都没写成，最后一次写的引述和概念标签都对不上原文，核对后什么都没剩下，所以没有生成笔记，校对稿没有变。"
    "可以点“重试”，或在“设置 → 打开整理规则…”里检查规则，或在设置里换一个笔记模型。"
)
NOTES_REFUSED = (
    "整理笔记的服务（Ollama）拒绝了这次请求，所以没有生成笔记，校对稿没有变，详情见日志。"
    "可以点“重试”，或在设置里换一个笔记模型。"
)


_ATTEMPTS = 2
_CHECKPOINT_VERSION = 1
_FINGERPRINT_KIND = "notes-1"
# What the reply of one attempt was when it could not be used.
_CUT, _EMPTY, _REMOVED = "cut off", "empty", "all removed"
_FAILED = {_CUT: NOTES_CUT_OFF, _EMPTY: NOTES_EMPTY_REPLY, _REMOVED: NOTES_ALL_REMOVED}
# Added to the prompt for the second attempt. At most 59 characters, so the wrapper (69) and
# this stay within the 128 that PROMPT_MARGIN keeps for them: a lesson that fits still fits.
_AGAIN = {
    _CUT: "\n注意：上一次的笔记写到长度上限被截断了。这一次请写得精简一些，每节只写要点。",
    _REMOVED: "\n注意：上一次的引述和概念标签都对不上原文。这一次引述必须逐字照抄原文，概念只用原文里出现过的词。",
    _EMPTY: "",  # nothing went wrong in the prompt; the same request is sent again
}


class NotesBlocked(Exception):
    """A problem with the service: the task waits for "继续整理笔记". The message is shown as it is."""


class NotesFailed(Exception):
    """This attempt at notes failed: "重试" or "忽略". The message is shown as it is."""


@dataclass(frozen=True)
class GeneratedNotes:
    markdown: str  # what finish_notes gives, word for word
    removed_quotes: int
    removed_concepts: int
    rules_note: str | None  # NotesRules.note as it is, for the completion message


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


class OllamaNotesWriter:
    def __init__(
        self,
        client: OllamaClient | None = None,
        rules: Callable[[], NotesRules] = lambda: load_notes_rules(rules_path()),
        *,
        num_ctx: int = NOTES_NUM_CTX,
        reply_budget: int = NOTES_REPLY_BUDGET,
        timeout: float = NOTES_TIMEOUT,
        retry_pause: float = 3.0,
    ) -> None:
        if not _is_count(num_ctx):
            raise ValueError(f"num_ctx must be a whole number of at least 1, not {num_ctx!r}")
        if not _is_count(reply_budget):
            raise ValueError(f"reply_budget must be a whole number of at least 1, not {reply_budget!r}")
        if not (_is_number(timeout) and timeout > 0):
            raise ValueError(f"timeout must be a positive number of seconds, not {timeout!r}")
        if not (_is_number(retry_pause) and retry_pause >= 0):
            raise ValueError(f"retry_pause must be a number of seconds, at least 0, not {retry_pause!r}")
        self.client = client if client is not None else OllamaClient()
        self._rules = rules  # read at every write, so an edited file counts from the next notes
        self.num_ctx = num_ctx
        self.reply_budget = reply_budget
        self.timeout = timeout
        self.retry_pause = retry_pause

    def release(self, model: str) -> None:
        """Free the model's memory now. Never raises: a failed release must not replace the result."""
        try:
            self.client.unload(model)
        except OllamaError as exc:
            logger.info("Could not unload %s: %s", model, exc)

    def write(self, source: str, meta: NotesMeta, model: str, checkpoint: Path) -> GeneratedNotes:
        """The notes for ``source``, or NotesBlocked (the service) or NotesFailed (this attempt).

        Only ``checkpoint`` is read and written; the caller saves the notes and removes it.
        """
        _check_arguments(source, meta, model, checkpoint)
        body = source_body(source)
        if content_chars(body) == "":
            raise NotesFailed(NOTES_EMPTY_SOURCE)
        try:
            rules = self._rules()
        except NotesRulesUnreadable as exc:
            logger.info("The notes rules cannot be read")
            raise NotesFailed(str(exc)) from exc
        if rules.note is not None:
            # Not the note itself: it holds the full path, and so the Windows user name. It goes to the result.
            logger.info("The notes rules file has no usable rules; the shipped notes rules were used")
        # Before the checkpoint: whether a lesson fits depends on the settings now, not on earlier runs.
        over = overflow(rules.text, body, self.num_ctx, self.reply_budget)
        if over > 0:
            logger.info("The lesson does not fit: %d characters, %d tokens too many", len(body), over)
            limit = len(body) - over
            if limit >= 1:
                raise NotesFailed(NOTES_TOO_LONG.format(chars=len(body), limit=limit))
            raise NotesFailed(NOTES_RULES_TOO_LONG.format(rules=len(rules.text)))

        fingerprint = _fingerprint(model, self.num_ctx, rules.text, body)
        saved = _saved_reply(checkpoint, fingerprint)
        if saved is not None:
            kind, result, _ = _judge(saved, source, meta)
            if result is not None:
                logger.info("Used the reply saved in the notes checkpoint; Ollama was not asked")
                return _generated(result, rules)
            logger.info("The reply saved in the notes checkpoint is %s; asking the model again", kind)

        asked = False  # a generate request was sent, so the model may be in memory
        gone = False  # the service was confirmed gone: it holds no model
        try:
            if model not in self._ask(self.client.models):
                raise OllamaModelMissing(f"{model} is not among Ollama's models")
            # generate() first asks what the model can do; asked here on its own (the client keeps the
            # answer), a failure of that question is known to come before any generate request.
            self._ask(lambda: self.client._thinks(model))
            last: str | None = None
            cause: Exception | None = None
            for attempt in range(1, _ATTEMPTS + 1):
                prompt = build_prompt(rules.text, source) + (_AGAIN[last] if last else "")
                asked = True
                try:
                    reply = self._ask(
                        lambda: self.client.generate(
                            model, prompt, num_ctx=self.num_ctx, num_predict=self.reply_budget, timeout=self.timeout
                        )
                    )
                except OllamaIncomplete as exc:
                    last, cause = _CUT, exc
                else:
                    last, result, cause = _judge(reply, source, meta)
                    if result is not None:
                        _save(checkpoint, fingerprint, reply)
                        return _generated(result, rules)
                logger.info("Notes attempt %d of %d: the reply was %s", attempt, _ATTEMPTS, last)
            raise NotesFailed(_FAILED[last]) from cause
        except OllamaDown as exc:
            gone = True
            logger.info("Ollama is down: %s", exc)
            raise NotesBlocked(NOTES_DOWN) from exc
        except OllamaTimeout as exc:
            logger.info("Ollama did not finish the notes in time: %s", exc)
            raise NotesBlocked(NOTES_TIMEOUT_MESSAGE) from exc
        except OllamaBusy as exc:
            logger.info("Ollama failed for now: %s", exc)
            raise NotesBlocked(NOTES_BUSY_MESSAGE) from exc
        except OllamaModelMissing as exc:
            logger.info("The notes model is missing: %s", exc)
            raise NotesBlocked(NOTES_MODEL_MISSING.format(model=model)) from exc
        except OllamaError as exc:
            logger.warning("Ollama refused the notes request: %s", exc)
            raise NotesFailed(NOTES_REFUSED) from exc
        finally:
            # One model at a time: free it on every way out, before the caller goes on, unless it was
            # never asked for or the service is gone (a release would only wait for a refused connection).
            if asked and not gone:
                self.release(model)

    def _ask(self, request: Callable[[], T]) -> T:
        """Send ``request`` once more after a pause if the service failed for now; that is not an attempt.

        A timeout is not sent again: it has already waited a long time.
        """
        try:
            return request()
        except OllamaBusy as exc:
            logger.warning("Ollama failed for now, asking again: %s", exc)
            time.sleep(self.retry_pause)
            return request()


def _check_arguments(source: object, meta: object, model: object, checkpoint: object) -> None:
    if not isinstance(source, str):
        raise TypeError(f"source must be text: {type(source).__name__}")
    if not isinstance(meta, NotesMeta):
        raise TypeError(f"meta must be NotesMeta: {type(meta).__name__}")
    if not isinstance(model, str):
        raise TypeError(f"model must be text: {type(model).__name__}")
    if not model.strip():
        raise ValueError("model must name a model")
    if not isinstance(checkpoint, Path):
        raise TypeError(f"checkpoint must be a Path: {type(checkpoint).__name__}")


def _judge(reply: str, source: str, meta: NotesMeta) -> tuple[str | None, NotesResult | None, Exception | None]:
    """(what was wrong, None, why) for a reply that cannot be used; (None, the notes, None) otherwise.

    finish_notes only removes more than structure_notes, so when it leaves nothing,
    structure_notes tells whether the reply had nothing to begin with or lost it all in the checks.
    """
    if reply == "":
        return _EMPTY, None, None
    try:
        return None, finish_notes(reply, source, meta), None
    except EmptyNotes as removed:
        try:
            structure_notes(reply, meta)
        except EmptyNotes as empty:
            return _EMPTY, None, empty
        return _REMOVED, None, removed


def _generated(result: NotesResult, rules: NotesRules) -> GeneratedNotes:
    return GeneratedNotes(result.markdown, result.removed_quotes, result.removed_concepts, rules.note)


def _fingerprint(model: str, num_ctx: int, rules: str, body: str) -> str:
    """Covers what the model is given and how: re-saving, renaming or a new date does not change it."""
    key = json.dumps([_FINGERPRINT_KIND, model, num_ctx, rules, body], ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _saved_reply(checkpoint: Path, fingerprint: str) -> str | None:
    """The reply kept for this very input, or None; a checkpoint that cannot be used is no checkpoint."""
    try:
        if not checkpoint.is_file():
            return None
        state = read_json(checkpoint)
    except (OSError, ValueError) as exc:  # held, not UTF-8, not JSON
        logger.info("The notes checkpoint cannot be read (%s); asking the model", type(exc).__name__)
        return None
    if (
        not isinstance(state, dict)
        or state.get("version") != _CHECKPOINT_VERSION
        or state.get("fingerprint") != fingerprint
        or not isinstance(state.get("reply"), str)
    ):
        logger.info("The notes checkpoint is for other input or broken; asking the model")
        return None
    return state["reply"]


def _save(checkpoint: Path, fingerprint: str, reply: str) -> None:
    """Keep the raw reply. Failing loses only the shortcut on a restart, never these notes."""
    try:
        atomic_write_json(checkpoint, {"version": _CHECKPOINT_VERSION, "fingerprint": fingerprint, "reply": reply})
    except OSError as exc:
        logger.warning("Could not save the notes checkpoint: %s", exc)
