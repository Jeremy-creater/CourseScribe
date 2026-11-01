"""Pure text helpers for proofreading: no model, network or interface involved."""

from __future__ import annotations

import re
import unicodedata

_DELIMITER_RUN = re.compile(r"([，,。.！!？?；;：:、…\s]+)")
_STRENGTH = {"？": 6, "?": 6, "！": 5, "!": 5, "。": 4, ".": 4, "…": 4, "；": 3, ";": 3, "：": 2, ":": 2, "，": 1, ",": 1, "、": 1}
_QUESTION_MARKS = frozenset("？?")
_ALWAYS_FILLER = re.compile("[嗯呃]")
_CLAUSE_END_PARTICLES = frozenset("啊呢")
_QUESTION_WORDS = (
    "怎么", "为什么", "为何", "什么", "谁", "哪", "几", "多少", "能不能", "是否", "吗",
    "要不要", "是不是", "有没有", "还是",
)
# "你呢" asks "and you?" even when the transcript has no question mark.
_ASKING_ABOUT = re.compile(r"^(?:那|那么)?(?:你|我|他|她|它|你们|我们|他们|她们|大家|别人)呢$")
_INTERJECTIONS = ("天啊",)


def _keeps_final_particle(clause: str) -> bool:
    return (
        any(word in clause for word in _QUESTION_WORDS)
        or _ASKING_ABOUT.match(clause) is not None
        or clause.endswith(_INTERJECTIONS)
    )


def _clean_clause(clause: str, delimiter: str) -> str:
    clause = _ALWAYS_FILLER.sub("", clause)
    if any(mark in delimiter for mark in _QUESTION_MARKS):
        return clause
    while clause and clause[-1] in _CLAUSE_END_PARTICLES and not _keeps_final_particle(clause):
        clause = clause[:-1]
    return clause


def _strength(delimiter: str) -> int:
    return max((_STRENGTH.get(ch, 0) for ch in delimiter), default=0)


def clean_fillers(text: str) -> str:
    """Remove meaningless 嗯/呃 anywhere, and 啊/呢 at the end of a clause.

    A line break or a space counts as a clause end, since a raw transcript has
    almost no punctuation. Real questions and real words stay. Only filler
    characters are ever lost; a clause reduced to nothing takes one of the two
    delimiters around it with it, keeping the stronger one.
    """
    parts = _DELIMITER_RUN.split(text)
    out: list[str] = []  # clause, delimiter, clause, delimiter, ...
    for index in range(0, len(parts), 2):
        clause = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        cleaned = _clean_clause(clause, delimiter)
        if cleaned or not clause:
            out.extend((cleaned, delimiter))
        elif out and _strength(delimiter) > _strength(out[-1]):
            out[-1] = delimiter
    return "".join(out)


# Each pair is (how it may appear, how it should be written).
_TERMS = (
    ("claude code", "Claude Code"),
    ("claude", "Claude"),
    ("chat gpt", "ChatGPT"),
    ("chatgpt", "ChatGPT"),
    ("openai", "OpenAI"),
    ("github", "GitHub"),
    ("git", "Git"),
    ("python", "Python"),
    ("ollama", "Ollama"),
    ("vs code", "VS Code"),
    ("vscode", "VS Code"),
    ("macos", "macOS"),
)
# Beyond letters and digits, "." "/" "_" "-" mean a path or an identifier,
# where the original spelling matters (a real "~/.claude" must stay lowercase).
_NOT_NEXT_TO = "A-Za-z0-9./_-"
_TERM_PATTERN = re.compile(
    "|".join(
        rf"(?<![{_NOT_NEXT_TO}]){re.escape(seen)}(?![{_NOT_NEXT_TO}])"
        for seen, _ in sorted(_TERMS, key=lambda pair: len(pair[0]), reverse=True)
    ),
    re.IGNORECASE,
)
_TERM_SPELLING = {seen: written for seen, written in _TERMS}


def correct_terms(text: str) -> str:
    """Fix the spelling of known English terms in the built-in list."""
    return _TERM_PATTERN.sub(lambda match: _TERM_SPELLING[match.group(0).lower()], text)


_TERMINATORS = "。！？!?…"
_CLOSERS = "”’」』）)\"'》"


def _ends_sentence(text: str, index: int) -> bool:
    ch = text[index]
    if ch in _TERMINATORS:
        return True
    # "." ends a sentence only before a space or the end, so "3.5" stays whole.
    return ch == "." and (index + 1 == len(text) or text[index + 1].isspace())


def split_sentences(text: str) -> list[str]:
    """Split after each sentence end. ``"".join(result) == text`` always holds."""
    sentences: list[str] = []
    start = 0
    index = 0
    size = len(text)
    while index < size:
        if not _ends_sentence(text, index):
            index += 1
            continue
        while index < size and _ends_sentence(text, index):
            index += 1
        while index < size and text[index] in _CLOSERS:
            index += 1
        while index < size and text[index].isspace():
            index += 1
        sentences.append(text[start:index])
        start = index
    if start < size:
        sentences.append(text[start:])
    return sentences


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _hard_cut(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    while len(text) > limit:
        cut = limit
        if _is_ascii_word_char(text[cut - 1]) and _is_ascii_word_char(text[cut]):
            word_start = cut
            while word_start > 0 and _is_ascii_word_char(text[word_start - 1]):
                word_start -= 1
            if word_start > 0:
                cut = word_start
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


def _atoms(text: str, limit: int) -> list[str]:
    atoms: list[str] = []
    for line in text.splitlines(keepends=True):
        if len(line) <= limit:
            atoms.append(line)
            continue
        for sentence in split_sentences(line):
            atoms.extend([sentence] if len(sentence) <= limit else _hard_cut(sentence, limit))
    return atoms


def chunk_text(text: str, limit: int) -> list[str]:
    """Cut ``text`` into pieces of at most ``limit`` characters, losing nothing.

    Cuts prefer line ends, then sentence ends, then the limit itself.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    chunks: list[str] = []
    current = ""
    for atom in _atoms(text, limit):
        if current and len(current) + len(atom) > limit:
            chunks.append(current)
            current = ""
        current += atom
    if current:
        chunks.append(current)
    return chunks


def content_chars(text: str) -> str:
    """The letters, digits and Chinese characters of ``text``, in a uniform form."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalnum())
