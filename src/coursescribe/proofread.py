"""Proofreading: fix the words without ever losing them.

Order: remove fillers -> fix known terms -> proofread chunk by chunk, checking
each reply for lost text -> remove fillers again, now that punctuation shows
where clauses end -> ask only for paragraph positions, so the model never
touches the body text while paragraphing.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path

from coursescribe.files import atomic_write_json, read_json
from coursescribe.ollama import (
    OllamaClient,
    OllamaDown,
    OllamaError,
    OllamaModelMissing,
)
from coursescribe.pipeline import ProofreadBlocked, ProofreadResult
from coursescribe.text import chunk_text, clean_fillers, content_chars, correct_terms, split_sentences

logger = logging.getLogger(__name__)

TEXT_MARKER = "\n---\n"
BREAK_MARKER = "只返回句子编号"

_CHUNK_CHARS = 800  # small enough to redo quickly, and well inside the model's context
_ATTEMPTS = 2
_MIN_PARAGRAPH_CHARS = 40
_BREAK_WINDOW_CHARS = 3000

_FILLERS = frozenset("嗯呃啊呢哦额唉哎呀嘛")
_MAX_LOST_RUN = 2  # characters: a longer gap is lost or invented text
_MAX_REPLACED = 6
_MAX_TERM_LENGTH = 12
_MAX_CHANGED_RATIO = 0.05
_MIN_CHANGES_ALLOWED = 3
_FORMATTING = ("#", "*", "`", ">", "- ")

_PROMPT = (
    "你是中文转录稿校对员。下面是一段网课的语音转录文字，请只做三件事：\n"
    "1. 按语义加标点（逗号、句号、问号等）并断句；\n"
    "2. 只修正明显的错别字和同音识别错误，拿不准就保持原样；英文术语保持英文原文，不要翻译；\n"
    "3. 逐字保留原文的每一句话和说话顺序，不得删除、增加、合并、改写或总结任何内容。\n"
    "不要加标题、列表或任何格式。直接返回处理后的正文，不要任何说明或前缀。"
)
_RETRY_NOTE = "\n注意：上一次的结果漏掉或改动了原文内容。这一次必须一字不漏。"
_BREAK_PROMPT = (
    "下面是一段讲课转录稿，已经按句子编号，每行一句。请根据话题转换，判断应该在哪些句子之后分段。"
    "每段应包含完整的一层意思，不要一两句话就分一段。\n"
    f"{BREAK_MARKER}，用逗号分隔，例如：3,8,15。不要返回任何正文或解释；如果不需要分段，返回“无”。\n\n"
)


def is_lossless(original: str, corrected: str) -> bool:
    """Whether ``corrected`` keeps the content of ``original``.

    Compares letters and digits only. Allows dropped fillers, fixed typos and a
    misheard term written in English. Rejects any gap, addition or rewrite
    beyond a couple of characters, and any formatting the source did not have.
    """
    if any(mark in corrected and mark not in original for mark in _FORMATTING):
        return False
    before = content_chars(original)
    after = content_chars(corrected)
    changed = 0
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        old, new = before[i1:i2], after[j1:j2]
        if operation == "replace":
            if (old.isascii() or new.isascii()) and max(len(old), len(new)) <= _MAX_TERM_LENGTH:
                changed += 1
            elif len(old) <= _MAX_REPLACED and abs(len(old) - len(new)) <= 1:
                changed += max(len(old), len(new))
            else:
                return False
        else:
            real = [ch for ch in old + new if ch not in _FILLERS]
            if len(real) > _MAX_LOST_RUN:
                return False
            changed += len(real)
    return changed <= max(_MIN_CHANGES_ALLOWED, len(before) * _MAX_CHANGED_RATIO)


_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)*")
_MAX_DIGITS = 7


def parse_breaks(reply: str, sentence_count: int) -> list[int]:
    """Read paragraph positions from a reply; everything else in it is ignored."""
    text = unicodedata.normalize("NFKC", reply)
    found: set[int] = set()
    for match in _NUMBER.finditer(text):
        token = match.group()
        negative = match.start() > 0 and text[match.start() - 1] == "-"
        if "." in token or negative or len(token) > _MAX_DIGITS:
            continue
        if 1 <= int(token) < sentence_count:
            found.add(int(token))
    return sorted(found)


def apply_breaks(sentences: list[str], breaks: Iterable[int], min_chars: int) -> str:
    """Join sentences, with a blank line after each position that leaves both sides long enough."""
    paragraphs: list[str] = []
    start = 0
    for position in sorted(set(breaks)):
        paragraph = "".join(sentences[start:position]).strip()
        rest = "".join(sentences[position:]).strip()
        if len(paragraph) >= min_chars and len(rest) >= min_chars:
            paragraphs.append(paragraph)
            start = position
    last = "".join(sentences[start:]).strip()
    if last:
        paragraphs.append(last)
    return "\n\n".join(paragraphs)


def group_windows(sentences: list[str], limit: int) -> list[list[str]]:
    """Group whole sentences into windows; a very long sentence gets its own."""
    windows: list[list[str]] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) > limit:
            windows.append(current)
            current, size = [], 0
        current.append(sentence)
        size += len(sentence)
    if current:
        windows.append(current)
    return windows


def _needs_space(left: str, right: str) -> bool:
    return bool(left and right) and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum()


def _join_lines(chunk: str) -> str:
    joined = ""
    for line in chunk.splitlines():
        line = line.strip()
        joined += (" " if _needs_space(joined, line) else "") + line
    return joined


class OllamaProofreader:
    def __init__(self, client: OllamaClient | None = None, chunk_chars: int = _CHUNK_CHARS) -> None:
        self.client = client or OllamaClient()
        self.chunk_chars = chunk_chars

    def release(self, model: str) -> None:
        try:
            self.client.unload(model)
        except OllamaError as exc:
            logger.info("Could not unload %s: %s", model, exc)

    def proofread(
        self, text: str, model: str, checkpoint: Path, on_progress: Callable[[int, int], None]
    ) -> ProofreadResult:
        try:
            self._check_ready(model)
            return self._run(text, model, checkpoint, on_progress)
        except OllamaDown as exc:
            logger.info("Ollama is down: %s", exc)
            raise ProofreadBlocked(
                "校对服务（Ollama）没有启动。文字稿已保存。请先打开 Ollama，然后点“继续校对”。"
            ) from exc
        except OllamaModelMissing as exc:
            raise ProofreadBlocked(_missing_model_message(model)) from exc

    def _check_ready(self, model: str) -> None:
        if model not in self.client.models():
            raise OllamaModelMissing(model)

    def _run(self, text: str, model: str, checkpoint: Path, on_progress) -> ProofreadResult:
        prepared = correct_terms(clean_fillers(text))
        chunks = [c for c in chunk_text(prepared, self.chunk_chars) if content_chars(c)]
        state = _load_state(checkpoint, _fingerprint(prepared, self.chunk_chars))

        for index, chunk in enumerate(chunks):
            key = str(index)
            if key not in state["chunks"]:
                state["chunks"][key] = self._proofread_chunk(_join_lines(chunk), model)
                atomic_write_json(checkpoint, state)
            on_progress(index + 1, len(chunks))

        results = [state["chunks"][str(index)] for index in range(len(chunks))]
        failed = sum(not item["ok"] for item in results)
        if chunks and failed == len(chunks):
            checkpoint.unlink(missing_ok=True)
            raise ProofreadBlocked(
                "这次校对一段都没有成功，所以没有生成校对稿。文字稿已保存。"
                "请确认 Ollama 正常运行，或在设置里换一个模型，然后点“继续校对”。"
            )
        paragraphs = self._paragraphs(results, model, state, checkpoint)
        markdown = "\n\n".join(paragraphs)
        return ProofreadResult(f"{markdown}\n" if markdown else "", failed, len(chunks))

    def _proofread_chunk(self, chunk: str, model: str) -> dict:
        for attempt in range(_ATTEMPTS):
            prompt = _PROMPT + (_RETRY_NOTE if attempt else "") + TEXT_MARKER + chunk
            try:
                reply = self.client.generate(model, prompt)
            except (OllamaDown, OllamaModelMissing):
                raise
            except OllamaError as exc:
                logger.warning("Proofreading attempt %d failed: %s", attempt + 1, exc)
                continue
            if reply and is_lossless(chunk, reply):
                return {"text": correct_terms(clean_fillers(_join_lines(reply))), "ok": True}
            logger.warning("Proofreading attempt %d changed the content; rejected", attempt + 1)
        return {"text": chunk, "ok": False}

    def _paragraphs(self, results: list[dict], model: str, state: dict, checkpoint: Path) -> list[str]:
        """Consecutive proofread chunks are paragraphed together; a failed one stands alone."""
        paragraphs: list[str] = []
        run = ""
        run_index = 0
        for item in results:
            if item["ok"]:
                run += (" " if _needs_space(run, item["text"]) else "") + item["text"]
                continue
            if run:
                paragraphs.append(self._paragraph_run(run, run_index, model, state, checkpoint))
                run_index += 1
                run = ""
            paragraphs.append(item["text"])
        if run:
            paragraphs.append(self._paragraph_run(run, run_index, model, state, checkpoint))
        return paragraphs

    def _paragraph_run(self, run: str, run_index: int, model: str, state: dict, checkpoint: Path) -> str:
        sentences = split_sentences(run)
        breaks: list[int] = []
        offset = 0
        for window_index, window in enumerate(group_windows(sentences, _BREAK_WINDOW_CHARS)):
            key = f"{run_index}/{window_index}"
            if key not in state["breaks"]:
                state["breaks"][key] = self._window_breaks(window, model)
                atomic_write_json(checkpoint, state)
            breaks.extend(offset + position for position in state["breaks"][key])
            offset += len(window)
        return apply_breaks(sentences, breaks, _MIN_PARAGRAPH_CHARS)

    def _window_breaks(self, window: list[str], model: str) -> list[int]:
        if len(window) < 2:
            return []
        numbered = "\n".join(f"{i}. {' '.join(s.split())}" for i, s in enumerate(window, start=1))
        try:
            reply = self.client.generate(model, _BREAK_PROMPT + numbered)
        except (OllamaDown, OllamaModelMissing):
            raise
        except OllamaError as exc:
            logger.warning("Paragraph request failed, leaving this part unbroken: %s", exc)
            return []
        return parse_breaks(reply, len(window))


def _missing_model_message(model: str) -> str:
    return (
        f"校对服务里没有模型 {model}。文字稿已保存。"
        f"请在命令行运行 ollama pull {model}，或在设置里换一个模型，然后点“继续校对”。"
    )


def _fingerprint(prepared: str, chunk_chars: int) -> str:
    return hashlib.sha256(f"{chunk_chars}\n{prepared}".encode("utf-8")).hexdigest()


def _load_state(checkpoint: Path, fingerprint: str) -> dict:
    empty = {"fingerprint": fingerprint, "chunks": {}, "breaks": {}}
    if not checkpoint.exists():
        return empty
    try:
        state = read_json(checkpoint)
    except (ValueError, OSError):
        return empty
    if not isinstance(state, dict) or state.get("fingerprint") != fingerprint:
        return empty
    return state
