"""Proofreading: fix the words without ever losing them.

Order: remove fillers -> fix known terms -> proofread chunk by chunk, checking
each reply for lost text -> remove fillers again, now that punctuation shows
where clauses end -> ask only for paragraph positions, so the model never
touches the body text while paragraphing.
"""

from __future__ import annotations

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
from coursescribe.text import chunk_text, clean_fillers, clean_tics, content_chars, correct_terms, split_sentences
from coursescribe.vocabulary import EMPTY, Vocabulary, apply_fixes

logger = logging.getLogger(__name__)

TEXT_MARKER = "\n---\n"
BREAK_MARKER = "只返回句子编号"

_CHUNK_CHARS = 800  # small enough to redo quickly, and well inside the model's context
_ATTEMPTS = 2
_MIN_PARAGRAPH_CHARS = 40
_MAX_PARAGRAPH_CHARS = 350  # longer reads as a wall of text
_BREAK_WINDOW_CHARS = 3000

_FILLERS = frozenset("嗯呃啊呢哦额唉哎呀嘛哈")
# Tics the clean copy may drop (PRD), as compared: letters and digits only, lower case.
_TIC_WORDS = ("对不对", "对吧", "对啊", "是吧", "yeah", "这个", "那个", "然后")
_MAX_LOST_RUN = 2  # characters: a longer gap is lost or invented text
_MAX_STUTTER = 6  # a doubled run longer than this is a repeated passage, not a stutter
_MAX_REPLACED = 6
_MAX_TERM_LENGTH = 12
_MAX_CHANGED_RATIO = 0.05
_MIN_CHANGES_ALLOWED = 3
_MIN_SEARCH_WIDTH = 40
_SEARCH_WIDTH_RATIO = 0.1
_FORMATTING = ("#", "*", "`", ">", "- ")

_PROMPT = (
    "你是中文讲课转录稿的编辑。把下面的口语转录整理成可直接阅读的文字稿，要求：\n"
    "1. 按语义加标点（逗号、句号、问号等）并断句；\n"
    "2. 修正错别字和同音识别错误，拿不准就保持原样；英文术语保持英文原文，不要翻译；\n"
    "3. 删除口头禅和无意义的词：嗯、呃、啊、呢、哈、呀、对吧、对不对、对啊、yeah，"
    "以及作停顿用的“这个”“那个”“然后呢”；口吃式的紧挨着重复只保留一遍；\n"
    "4. 指代 AI、软件、工具、模型时用“它”，不用“他”；\n"
    "5. 除以上几点，逐字保留原文的每一句话和说话顺序，不得总结、概括、改写、合并或增加任何内容。\n"
    "不要加标题、列表或任何格式。直接返回处理后的正文，不要任何说明或前缀。"
)
_TERMS_NOTE = "\n本课的常用词：{terms}。转录里读音相同或相近的错字，改成这些词。"
_RETRY_NOTE = "\n注意：上一次的结果漏掉或改动了原文内容。这一次除口头禅外必须一字不漏。"
_BREAK_PROMPT = (
    "下面是一段讲课转录稿，已经按句子编号，每行一句。请根据话题转换，判断应该在哪些句子之后分段。"
    "每段应包含完整的一层意思，通常 3 到 6 句、100 到 300 字，不要一两句话就分一段，也不要整页一段。\n"
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
    limit = max(_MIN_CHANGES_ALLOWED, len(before) * _MAX_CHANGED_RATIO)
    differences = _differences(before, after, max(_MIN_SEARCH_WIDTH, int(len(before) * _SEARCH_WIDTH_RATIO)))
    if differences is None:
        return False
    changed = 0
    for old, new in differences:
        if old and new:
            if (old.isascii() or new.isascii()) and max(len(old), len(new)) <= _MAX_TERM_LENGTH:
                changed += 1
            elif len(old) <= _MAX_REPLACED and abs(len(old) - len(new)) <= 1:
                changed += max(len(old), len(new))
            else:
                return False
        elif not new and len(old) <= _MAX_STUTTER and old * 2 in before:
            continue  # a stutter: the same words said twice in a row, kept once
        else:
            real = _without_tics("".join(ch for ch in old + new if ch not in _FILLERS))
            if len(real) > _MAX_LOST_RUN:
                return False
            changed += len(real)
    return changed <= limit


def _without_tics(run: str) -> str:
    previous = None
    while previous != run:
        previous = run
        for word in _TIC_WORDS:
            run = run.replace(word, "")
    return run


def _differences(before: str, after: str, width: int) -> list[tuple[str, str]] | None:
    """The smallest set of (old, new) edits, or None if it needs more than ``width``.

    A true edit-distance alignment: a quicker matcher pairs up the wrong copy when
    a teacher repeats a passage, and then reports intact text as lost. Only cells
    near the diagonal are computed, which keeps this fast and bounds the work.
    """
    if abs(len(before) - len(after)) > width:
        return None
    outside = width + 1
    # moves[i][j - low(i)]: 0 match, 1 substitute, 2 delete from before, 3 insert from after
    moves: list[list[int]] = []
    row = {j: j for j in range(0, min(width, len(after)) + 1)}
    for i, old_char in enumerate(before, start=1):
        low, high = max(0, i - width), min(len(after), i + width)
        current: dict[int, int] = {}
        chosen = []
        for j in range(low, high + 1):
            options = [(row.get(j, outside) + 1, 2), (current.get(j - 1, outside) + 1, 3)]
            if j > 0:
                same = old_char == after[j - 1]
                options.insert(0, (row.get(j - 1, outside) + (0 if same else 1), 0 if same else 1))
            else:
                options[0] = (i, 2)
            best, move = min(options, key=lambda option: option[0])
            current[j] = best
            chosen.append(move)
        if min(current.values(), default=outside) > width:
            return None
        moves.append(chosen)
        row = current
    if not before:
        return [("", after)] if after else []
    if row.get(len(after), outside) > width:
        return None

    edits: list[tuple[str, str]] = []
    old_run: list[str] = []
    new_run: list[str] = []
    i, j = len(before), len(after)
    while i > 0 or j > 0:
        move = 3 if i == 0 else moves[i - 1][j - max(0, i - width)]
        if move == 0:
            if old_run or new_run:
                edits.append(("".join(reversed(old_run)), "".join(reversed(new_run))))
                old_run, new_run = [], []
            i, j = i - 1, j - 1
            continue
        if move in (1, 2):
            old_run.append(before[i - 1])
            i -= 1
        if move in (1, 3):
            new_run.append(after[j - 1])
            j -= 1
    if old_run or new_run:
        edits.append(("".join(reversed(old_run)), "".join(reversed(new_run))))
    edits.reverse()
    return edits


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


def apply_breaks(sentences: list[str], breaks: Iterable[int], min_chars: int, max_chars: int | None = None) -> str:
    """Join sentences, with a blank line after each position that leaves both sides long enough.

    A paragraph still longer than ``max_chars`` is split further at sentence ends.
    """
    groups: list[list[str]] = []
    start = 0
    for position in sorted(set(breaks)):
        paragraph = "".join(sentences[start:position]).strip()
        rest = "".join(sentences[position:]).strip()
        if len(paragraph) >= min_chars and len(rest) >= min_chars:
            groups.append(sentences[start:position])
            start = position
    groups.append(sentences[start:])
    paragraphs: list[str] = []
    for group in groups:
        for part in _split_long(group, max_chars) if max_chars else [group]:
            text = "".join(part).strip()
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _split_long(group: list[str], max_chars: int) -> list[list[str]]:
    total = sum(len(s) for s in group)
    if total <= max_chars:
        return [group]
    pieces = -(-total // max_chars)  # even sizes read better than one full piece and a stub
    target = total / pieces
    parts: list[list[str]] = []
    current: list[str] = []
    size = 0
    for sentence in group:
        if current and size + len(sentence) > target and len(parts) < pieces - 1:
            parts.append(current)
            current, size = [], 0
        current.append(sentence)
        size += len(sentence)
    parts.append(current)
    return parts


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
    def __init__(
        self,
        client: OllamaClient | None = None,
        chunk_chars: int = _CHUNK_CHARS,
        vocabulary: Callable[[], Vocabulary] = lambda: EMPTY,
    ) -> None:
        self.client = client or OllamaClient()
        self.chunk_chars = chunk_chars
        self._load_vocabulary = vocabulary
        self._vocabulary = EMPTY

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
        self._vocabulary = self._load_vocabulary()  # read for each task, so edits apply to the next one
        prepared = self._tidy(text)
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

    def _tidy(self, text: str) -> str:
        """Everything done by rule: word-list fixes, tics, fillers, English term spelling."""
        return correct_terms(clean_fillers(clean_tics(apply_fixes(text, self._vocabulary))))

    def _prompt(self) -> str:
        terms = self._vocabulary.all_terms()
        return _PROMPT + (_TERMS_NOTE.format(terms="、".join(terms)) if terms else "")

    def _proofread_chunk(self, chunk: str, model: str) -> dict:
        for attempt in range(_ATTEMPTS):
            prompt = self._prompt() + (_RETRY_NOTE if attempt else "") + TEXT_MARKER + chunk
            try:
                reply = self.client.generate(model, prompt)
            except (OllamaDown, OllamaModelMissing):
                raise
            except OllamaError as exc:
                logger.warning("Proofreading attempt %d failed: %s", attempt + 1, exc)
                continue
            if reply and is_lossless(chunk, reply):
                return {"text": self._tidy(_join_lines(reply)), "ok": True}
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
        return apply_breaks(sentences, breaks, _MIN_PARAGRAPH_CHARS, _MAX_PARAGRAPH_CHARS)

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
