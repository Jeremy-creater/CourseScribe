"""The user's word list: common words, and fixes for words the engine mishears."""

from __future__ import annotations

import random
import re
import time

import pytest

from coursescribe.vocabulary import (
    EMPTY,
    TEMPLATE,
    WORD_LIST_ENCODING,
    WORD_LIST_HELD,
    WORD_LIST_UNREADABLE,
    Vocabulary,
    WordListUnreadable,
    apply_fixes,
    ensure_word_list,
    fix_reply,
    load_vocabulary,
    parse_vocabulary,
)
from test_recording import _held


def test_reads_words_and_fixes_and_skips_comments():
    vocabulary = parse_vocabulary("# 说明\n践行\n见行 → 践行\nCloud -> Claude\n\n  定投  \n人之见 => 人至践\n")
    assert vocabulary.terms == ("践行", "定投")
    assert vocabulary.fixes == (("见行", "践行"), ("Cloud", "Claude"), ("人之见", "人至践"))


def test_the_written_form_of_a_fix_also_counts_as_a_common_word():
    assert parse_vocabulary("见行 → 践行\n").all_terms() == ("践行",)
    assert parse_vocabulary("践行\n见行 → 践行\n").all_terms() == ("践行",)


def test_lines_that_are_not_a_clear_fix_are_ignored():
    vocabulary = parse_vocabulary("→ 践行\n见行 →\n→\n")
    assert vocabulary == Vocabulary((), ())


def test_fixes_replace_every_occurrence_longest_first():
    vocabulary = parse_vocabulary("见行 → 践行\n人之见则无敌 → 人至践则无敌\n")
    text = "我们社群有一句话叫人之见则无敌\n见是见行的见\n大家要见行\n"
    assert apply_fixes(text, vocabulary) == "我们社群有一句话叫人至践则无敌\n见是践行的见\n大家要践行\n"


def test_fixes_are_stable_when_applied_twice():
    vocabulary = parse_vocabulary("见行 → 践行\n")
    once = apply_fixes("见行", vocabulary)
    assert apply_fixes(once, vocabulary) == once == "践行"


def test_no_word_list_changes_nothing():
    assert apply_fixes("见行", Vocabulary((), ())) == "见行"


def test_a_missing_file_is_an_empty_list_and_a_folder_is_reported_as_unreadable(tmp_path):
    assert load_vocabulary(tmp_path / "missing.txt") == Vocabulary((), ())
    folder = tmp_path / "folder.txt"
    folder.mkdir()
    # Rewritten for M2 item 11 (PRD: an unreadable word list is reported, not used as empty).
    with pytest.raises(WordListUnreadable) as caught:
        load_vocabulary(folder)
    assert str(caught.value) == WORD_LIST_UNREADABLE.format(path=folder)


def test_creating_the_file_writes_the_template_once_and_never_overwrites(tmp_path):
    path = tmp_path / "home" / "词表.txt"
    ensure_word_list(path)
    assert path.read_text(encoding="utf-8") == TEMPLATE
    assert load_vocabulary(path) == Vocabulary((), ())  # the template is only comments
    path.write_text("践行\n", encoding="utf-8")
    ensure_word_list(path)
    assert path.read_text(encoding="utf-8") == "践行\n"


def test_a_file_saved_by_notepad_with_a_bom_reads_cleanly(tmp_path):
    path = tmp_path / "词表.txt"
    path.write_bytes("﻿践行\r\n见行 → 践行\r\n".encode("utf-8"))
    assert load_vocabulary(path) == Vocabulary(("践行",), (("见行", "践行"),))


# --- M2 item 11: reading, encodings, whole-word fixes (docs/plans/M2-vocabulary-tests.md) ---

# All in GB2312. 浏洹浒澶 have GBK lead bytes E4/E5, the start of a three-byte UTF-8 character,
# so GBK lists such as "浏电保" are also valid UTF-8 (review M1).
_CHINESE = "践行定投见人之至则无敌克劳德学习浏洹浒澶电版保报安典变"
_ASCII_WORDS = ("java", "Claude", "dot net", "C++", ".NET", "5G")
_PUNCTUATION = ("，", "。", "“", "”")
_ARROWS = ("→", "->", "=>")
_ENCODINGS = {
    "utf-8": lambda text: text.encode("utf-8"),
    "utf-8-bom": lambda text: b"\xef\xbb\xbf" + text.encode("utf-8"),
    "utf-16-le-bom": lambda text: b"\xff\xfe" + text.encode("utf-16-le"),
    "utf-16-be-bom": lambda text: b"\xfe\xff" + text.encode("utf-16-be"),
    "gbk": lambda text: text.encode("gbk"),
}


def _snapshot(path):
    return path.read_bytes(), path.stat().st_mtime_ns


# Three GB2312 characters whose GBK bytes (lead E4/E5, then B0-BF leads) are also one valid
# UTF-8 sequence each; a list of only these and ASCII is valid UTF-8 in GBK (review M1).
_UTF8_LOOKALIKES = ("浏电保", "洹颁啊", "浒典报", "浏版安", "澶变安")


def _random_word(rng: random.Random, lookalike: bool = False) -> str:
    if lookalike:
        if rng.random() < 0.6:
            return "".join(rng.choice(_UTF8_LOOKALIKES) for _ in range(rng.randint(1, 2)))
        return rng.choice(_ASCII_WORDS)
    if rng.random() < 0.6:
        word = "".join(rng.choice(_CHINESE) for _ in range(rng.randint(1, 4)))
        return word + (rng.choice(_PUNCTUATION) if rng.random() < 0.2 else "")
    return rng.choice(_ASCII_WORDS)


def _random_word_list(rng: random.Random) -> str:
    lookalike = rng.random() < 0.25
    arrows = ("->", "=>") if lookalike else _ARROWS
    lines = []
    for _ in range(rng.randint(1, 40)):
        kind = rng.randrange(4)
        if kind == 0:
            line = "# " + _random_word(rng, lookalike)
        elif kind == 1:
            line = ""
        elif kind == 2:
            line = _random_word(rng, lookalike)
        else:
            line = f"{_random_word(rng, lookalike)} {rng.choice(arrows)} {_random_word(rng, lookalike)}"
        lines.append(line + rng.choice(("\n", "\r\n")))
    return "".join(lines)


def test_the_generated_chinese_is_what_the_round_trip_law_assumes():
    for ch in _CHINESE + "".join(_PUNCTUATION) + "→":
        ch.encode("gb2312")
    assert any(0xE0 <= ch.encode("gbk")[0] <= 0xEF for ch in _CHINESE)  # widened after review M1


def test_every_accepted_encoding_reads_back_the_same_word_list(tmp_path):
    rng = random.Random(20261111)
    path = tmp_path / "词表.txt"
    gbk_also_utf8 = 0
    for _ in range(300):
        text = _random_word_list(rng)
        expected = parse_vocabulary(text)
        if _has_chinese_text(text) and _decodes_as_utf8(text.encode("gbk")):
            gbk_also_utf8 += 1
        for name, encode in _ENCODINGS.items():
            path.write_bytes(encode(text))
            before = _snapshot(path)
            got = load_vocabulary(path)
            assert got == expected, (name, text)
            assert all("\ufeff" not in item for item in got.terms), (name, text)
            assert _snapshot(path) == before
            assert [p.name for p in tmp_path.iterdir()] == ["词表.txt"]
    assert gbk_also_utf8 >= 30  # the generator really reaches the GBK-that-is-valid-UTF-8 case


def _has_chinese_text(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _decodes_as_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def test_a_notepad_ansi_word_list_is_read(tmp_path):
    path = tmp_path / "词表.txt"
    path.write_bytes("践行\r\n见行 → 践行\r\nCloud -> Claude\r\n".encode("gbk"))
    assert load_vocabulary(path) == Vocabulary(("践行",), (("见行", "践行"), ("Cloud", "Claude")))


def _broken_utf8() -> bytes:
    data = "践行定".encode("utf-8")
    return data[:3] + data[4:]


_REFUSED = {
    "utf-16-le-no-bom": "第一句".encode("utf-16-le"),
    "utf-16-ascii-no-bom": "abc".encode("utf-16-le"),
    "latin-1": "café -> Café".encode("latin-1"),
    "shift-jis": "日本語 → 日本".encode("shift_jis"),
    "big5": "見行 → 踐行".encode("big5"),
    "broken-utf-8": _broken_utf8(),
    "utf-8-bom-on-gbk": b"\xef\xbb\xbf" + "践行".encode("gbk"),
    "utf-16-bom-odd": b"\xff\xfe" + b"\x00",
    "bell": "践行\x07\n".encode("utf-8"),
    "form-feed": "践行\x0c\n".encode("utf-8"),
    "utf-32": "践行".encode("utf-32"),
}
_GOES_THROUGH_THE_GB2312_CHECK = ("shift-jis", "big5", "broken-utf-8")


@pytest.mark.parametrize("name", list(_REFUSED))
def test_an_encoding_that_cannot_be_told_for_sure_is_refused_and_explained(tmp_path, name):
    data = _REFUSED[name]
    if name in _GOES_THROUGH_THE_GB2312_CHECK:
        data.decode("gbk")  # the premise: GBK alone would accept this, so only the GB2312 check refuses it
    path = tmp_path / "词表.txt"
    path.write_bytes(data)
    with pytest.raises(WordListUnreadable) as caught:
        load_vocabulary(path)
    message = str(caught.value)
    assert message == WORD_LIST_ENCODING.format(path=path)
    assert str(path) in message and "UTF-8" in message and "重试" in message and "改个名字" in message
    assert "Ollama" not in message and "处理时出错了" not in message
    assert path.read_bytes() == data


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"java -> Java\n", Vocabulary((), (("java", "Java"),))),
        (b"", EMPTY),
        (b"\xef\xbb\xbf", EMPTY),
        (b"\r\n\r\n  \t\n", EMPTY),
        (TEMPLATE.encode("gbk"), EMPTY),
        ("café -> Café\n".encode("utf-8"), Vocabulary((), (("café", "Café"),))),
        ("学习\n".encode("gbk"), Vocabulary(("学习",), ())),
        ("之\n".encode("gbk"), Vocabulary(("之",), ())),
        ("привет\n".encode("utf-8"), Vocabulary(("привет",), ())),
    ],
    ids=[
        "ascii", "empty", "bom-only", "blank", "template-gbk", "utf-8-latin", "gbk-valid-utf-8", "gbk-one-char",
        "utf-8-cyrillic",
    ],
)
def test_encodings_that_can_be_told_are_read(tmp_path, data, expected):
    path = tmp_path / "词表.txt"
    path.write_bytes(data)
    assert load_vocabulary(path) == expected
    assert path.read_bytes() == data


def _in_gb2312(ch: str) -> bool:
    try:
        ch.encode("gb2312")
    except UnicodeEncodeError:
        return False
    return True


def test_cyrillic_is_outside_the_western_range_and_inside_gb2312():
    """The premise of the utf-8-cyrillic row: neither Chinese nor Western, and nothing that marks a misreading."""
    assert all(0x0400 <= ord(ch) <= 0x04FF for ch in "привет")
    assert all(_in_gb2312(ch) for ch in "привет")


def test_a_gbk_list_that_is_also_valid_utf_8_is_read_as_gbk(tmp_path):
    """Review M1: 浏电保 in GBK is E4 AF B5 E7 B1 A3, which UTF-8 reads as 䯵籣, outside GB2312."""
    data = "浏电保\njava -> Java\n".encode("gbk")
    misread = data.decode("utf-8")  # the premise: strict UTF-8 accepts it too
    assert any(not _in_gb2312(ch) for ch in misread)
    path = tmp_path / "词表.txt"
    path.write_bytes(data)
    assert load_vocabulary(path) == Vocabulary(("浏电保",), (("java", "Java"),))
    assert path.read_bytes() == data


def test_a_utf_8_list_whose_characters_are_all_in_gb2312_stays_utf_8(tmp_path):
    data = "浏览器\n电报 -> 电报机\n".encode("utf-8")
    path = tmp_path / "词表.txt"
    path.write_bytes(data)
    assert load_vocabulary(path) == Vocabulary(("浏览器",), (("电报", "电报机"),))


def test_a_missing_word_list_is_empty_and_nothing_is_created(tmp_path):
    missing = tmp_path / "没有.txt"
    no_folder = tmp_path / "无" / "词表.txt"
    assert load_vocabulary(missing) == EMPTY
    assert load_vocabulary(no_folder) == EMPTY
    assert not missing.exists() and not no_folder.parent.exists()


def _word_list(tmp_path):
    path = tmp_path / "词表.txt"
    path.write_bytes("见行 → 践行\n".encode("utf-8"))
    return path


def test_a_word_list_held_for_a_moment_is_waited_for(tmp_path, monkeypatch):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    path = _word_list(tmp_path)
    data = path.read_bytes()
    held = _held(monkeypatch, path, times=3)
    assert load_vocabulary(path) == Vocabulary((), (("见行", "践行"),))
    assert held.attempts == 4
    assert path.read_bytes() == data


@pytest.mark.parametrize(("can_be_held", "attempts"), [(True, 20), (False, 1)], ids=["windows", "elsewhere"])
def test_a_word_list_held_for_long_is_reported_as_held(tmp_path, monkeypatch, can_be_held, attempts):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", can_be_held)
    path = _word_list(tmp_path)
    data = path.read_bytes()
    held = _held(monkeypatch, path)
    with pytest.raises(WordListUnreadable) as caught:
        load_vocabulary(path)
    assert str(caught.value) == WORD_LIST_HELD.format(path=path)
    assert held.attempts == attempts
    held.release()
    assert path.read_bytes() == data


def test_a_word_list_on_an_unplugged_drive_is_reported_as_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    path = _word_list(tmp_path)
    data = path.read_bytes()
    held = _held(monkeypatch, path, error=OSError(5, "Input/output error"))
    with pytest.raises(WordListUnreadable) as caught:
        load_vocabulary(path)
    assert str(caught.value) == WORD_LIST_UNREADABLE.format(path=path)
    assert held.attempts == 1
    held.release()
    assert path.read_bytes() == data


def test_a_folder_named_like_the_word_list_is_not_called_held(tmp_path, monkeypatch):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)
    monkeypatch.setattr("coursescribe.files._REPLACE_WAIT", 0.001)
    path = tmp_path / "词表.txt"
    path.mkdir()
    with pytest.raises(WordListUnreadable) as caught:
        load_vocabulary(path)
    assert str(caught.value) == WORD_LIST_UNREADABLE.format(path=path)
    assert list(path.iterdir()) == []


def test_a_very_long_word_list_is_read_whole_and_fixes_stay_fast(tmp_path):
    lines = [f"词{i:05d}错 → 词{i:05d}\n" for i in range(5000)]
    lines += [f"常用{i:05d}\n" for i in range(5000)]
    lines.append("践" * 200_000 + "\n")
    text = "".join(lines)
    results = []
    for name in ("gbk", "utf-16-le-bom"):
        path = tmp_path / f"{name}.txt"
        data = _ENCODINGS[name](text)
        path.write_bytes(data)
        results.append(load_vocabulary(path))
        assert path.read_bytes() == data
    vocabulary = results[0]
    assert results[1] == vocabulary
    assert len(vocabulary.fixes) == 5000 and len(vocabulary.terms) == 5001
    filler = "我们今天讲的内容很重要。" * 2500  # about 30 000 characters on each side
    lesson = filler + "词04999错" + filler + "词00000错" + "\n"
    started = time.perf_counter()
    fixed = apply_fixes(lesson, vocabulary)
    elapsed = time.perf_counter() - started
    assert fixed == filler + "词04999" + filler + "词00000" + "\n"
    assert elapsed < 2, elapsed


# --- replacing: whole English words only, once, left to right ---

_WORD_RUN = re.compile(r"([A-Za-z0-9_-]+)")


def _whole_word_oracle(text: str, spelling: dict[str, str]) -> str:
    parts = _WORD_RUN.split(text)
    return "".join(spelling.get(part, part) if index % 2 else part for index, part in enumerate(parts))


def test_english_fixes_change_whole_words_only():
    vocabulary = parse_vocabulary("java → Java\ncloud → Claude\ngpt → GPT\n")
    spelling = dict(vocabulary.fixes)
    pieces = (
        "java", "cloud", "gpt",
        "javascript", "xjava", "java2", "java_8", "java-based", "cloudflare", "icloud", "gpts",
        "-", "_", "re-", "my_", "-x", "_v2",
        "Java", "CLOUD",
        "用", "的", "编", "程", "讲",
        " ", "，", "。", ".", "!", "'", "/", "\n",
        "3", "5",
    )
    rng = random.Random(20261112)
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 30)))
        assert apply_fixes(text, vocabulary) == _whole_word_oracle(text, spelling), text


@pytest.mark.parametrize(
    ("word_list", "text", "expected"),
    [
        ("java → Java", "用了 java 和 javascript", "用了 Java 和 javascript"),
        ("java → Java", "java编程", "Java编程"),
        ("java → Java", "学java。", "学Java。"),
        ("java → Java", "还有 java.", "还有 Java."),
        ("java → Java", "java's", "Java's"),
        ("java → Java", "java/python", "Java/python"),
        ("java → Java", "Java编程", "Java编程"),
        ("java → Java", "xjava java2 java_8 java-based", "xjava java2 java_8 java-based"),
        ("java → Java", "re-java", "re-java"),
        ("java → Java", "my_java", "my_java"),
        ("Cloud -> Claude", "Cloudflare 和 Cloud", "Cloudflare 和 Claude"),
        ("Cloud -> Claude", "cloud", "cloud"),
        ("c 加加 → C++", "用c 加加写", "用C++写"),
        ("c 加加 → C++", "用 c 加加写", "用 C++写"),
        ("c 加加 → C++", "abc 加加", "abc 加加"),
        ("c 加加 → C++", "c 加加加", "C++加"),
        ("dot net → .NET", "用 dot net 开发", "用 .NET 开发"),
        ("dot net → .NET", "dot network", "dot network"),
        ("dot net → .NET", "dotnet", "dotnet"),
        ("c sharp → C#", "学c sharp。", "学C#。"),
        ("5 g → 5G", "5 g网络", "5G网络"),
        ("5 g → 5G", "15 g", "15 g"),
        ("5 g → 5G", "5 gb", "5 gb"),
        ("a.b → AB", "axb", "axb"),
        ("a.b → AB", "a.b", "AB"),
        ("怪符号 → \\1\\g<0>$&", "一个怪符号", "一个\\1\\g<0>$&"),
        ("py → Python\npython → Python", "py 和 python 和 pypy", "Python 和 Python 和 pypy"),
    ],
)
def test_english_word_edges_case_and_symbols(word_list, text, expected):
    assert apply_fixes(text, parse_vocabulary(word_list + "\n")) == expected


def test_chinese_fixes_replace_inside_running_text():
    vocabulary = parse_vocabulary("见行 → 践行\n定头 → 定投\n克劳德 → Claude\n")
    pieces = ("见行", "定头", "克劳德", "我", "们", "要", "大", "家", "，", "。", "\n", " ")
    rng = random.Random(20261113)
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 40)))
        expected = text.replace("见行", "践行").replace("定头", "定投").replace("克劳德", "Claude")
        assert apply_fixes(text, vocabulary) == expected, text


@pytest.mark.parametrize(
    ("lines", "text", "expected"),
    [
        (["见 → 建", "见行 → 践行"], "见行见", "践行建"),
        (["甲乙 → 子", "乙丙 → 丑"], "甲乙丙", "子丙"),
        (["甲 → 乙", "乙 → 丙"], "甲乙", "乙丙"),
        (["行 → 践行"], "行动", "践行动"),
        (["践行 → 践行"], "践行", "践行"),
        (["claude → Claude", "claude code → Claude Code"], "用 claude code 和 claude", "用 Claude Code 和 Claude"),
        (["claude → Claude", "claude code → Claude Code"], "claude coder", "Claude coder"),
    ],
    ids=["longest-first", "leftmost-first", "no-chain", "fix-contains-word", "same-both-sides", "english-overlap", "fall-back"],
)
def test_order_and_overlap_do_not_depend_on_line_order(lines, text, expected):
    for order in (lines, list(reversed(lines))):
        assert apply_fixes(text, parse_vocabulary("\n".join(order) + "\n")) == expected, order


def test_the_last_line_for_the_same_word_counts():
    assert apply_fixes("见行", parse_vocabulary("见行 → 践行\n见行 → 建行\n")) == "建行"


def test_a_cycle_of_fixes_changes_each_character_once():
    vocabulary = parse_vocabulary("甲 → 乙\n乙 → 丙\n丙 → 甲\n")
    table = str.maketrans("甲乙丙", "乙丙甲")
    rng = random.Random(20261114)
    for _ in range(300):
        text = "".join(rng.choice("甲乙丙丁") for _ in range(rng.randint(0, 50)))
        assert apply_fixes(text, vocabulary) == text.translate(table), text


# --- fix_reply: only what the model undid is changed back ---

_REPLY_SETS = (
    ("见行 → 践行",),
    ("定头 → 定投",),
    ("克劳德 → Claude",),
    ("java → Java",),
    ("行 → 践行",),
    ("A → B", "B → C"),
    ("甲 → 乙", "乙 → 丙", "丙 → 甲"),
)
_REPLY_FILLER = ("我", "们", "要", "大", "家", "用", "和", " ")
_ADDED_MARKS = "，。、！？"


def _is_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in "_-")


def _occurrences(text: str, word: str) -> list[int]:
    found, start = [], text.find(word)
    while start != -1:
        found.append(start)
        start = text.find(word, start + 1)
    return found


def _inside_a_correct_word(text: str, corrects) -> set[int]:
    """Insertion points strictly inside an occurrence of a correct word."""
    inside = set()
    for word in corrects:
        for start in _occurrences(text, word):
            inside.update(range(start + 1, start + len(word)))
    return inside


def _add_marks(rng, text: str, corrects) -> str:
    inside = _inside_a_correct_word(text, corrects)
    allowed = [
        k for k in range(len(text) + 1)
        if k not in inside
        and not (k > 0 and _is_word_char(text[k - 1]))
        and not (k < len(text) and _is_word_char(text[k]))
    ]
    chosen = sorted(rng.sample(allowed, min(len(allowed), rng.randint(1, 3)))) if allowed else []
    for k in reversed(chosen):
        text = text[:k] + rng.choice(_ADDED_MARKS) + text[k:]
    return text


def _undo_spots(text: str, undoable) -> list[tuple[int, str, str]]:
    """(position, written, heard) for each occurrence of a fix's written side that can be written back as heard."""
    spots = []
    for heard, written in undoable:
        for start in _occurrences(text, written):
            end = start + len(written)
            if heard[0].isascii() and heard[0].isalnum() and start > 0 and _is_word_char(text[start - 1]):
                continue
            if heard[-1].isascii() and heard[-1].isalnum() and end < len(text) and _is_word_char(text[end]):
                continue
            spots.append((start, written, heard))
    return spots


def _undo(text: str, spots) -> str:
    taken, chosen = set(), []
    for start, written, heard in sorted(spots):
        span = set(range(start, start + len(written)))
        if span & taken:
            continue
        taken |= span
        chosen.append((start, written, heard))
    for start, written, heard in reversed(chosen):
        text = text[:start] + heard + text[start + len(written) :]
    return text


def test_fix_reply_changes_back_only_what_the_model_undid():
    rng = random.Random(20261115)
    for _ in range(300):
        sets = rng.sample(_REPLY_SETS, rng.randint(1, 3))
        lines = [line for chosen in sets for line in chosen]
        vocabulary = parse_vocabulary("\n".join(lines) + "\n")
        fixes = dict(vocabulary.fixes)
        corrects = set(fixes.values())
        heard_words = list(fixes)
        x = "".join(rng.choice(heard_words + list(_REPLY_FILLER)) for _ in range(rng.randint(1, 20)))
        s = apply_fixes(x, vocabulary)
        context = (lines, x, s)
        # (1) an untouched reply, or one with a full stop added, is kept as it is
        assert fix_reply(s, s, vocabulary) == s, context
        assert fix_reply(s, s + "。", vocabulary) == s + "。", context
        # (1') only punctuation added
        r = _add_marks(rng, s, corrects)
        assert fix_reply(s, r, vocabulary) == r, (*context, r)
        undoable = [(heard, written) for heard, written in fixes.items() if heard not in corrects]
        spots_in_s = _undo_spots(s, undoable)
        if not spots_in_s:
            continue
        # (2) one undone fix is changed back
        one = _undo(s, [rng.choice(spots_in_s)])
        assert fix_reply(s, one, vocabulary) == s, (*context, one)
        # (2') several undone, with punctuation added
        spots_in_r = _undo_spots(r, undoable)
        some = rng.sample(spots_in_r, rng.randint(1, len(spots_in_r)))
        r2 = _undo(r, some)
        assert fix_reply(s, r2, vocabulary) == r, (*context, r, r2)


@pytest.mark.parametrize(
    ("word_list", "sent", "reply", "expected"),
    [
        ("A → B\nB → C", "用B和C和B", "用A和C和B", "用B和C和B"),
        ("A → B\nB → C", "用B和B和B", "用A和B和B", "用B和B和B"),
        ("A → B\nB → C", "用B和B和B", "用A和A和B", "用B和B和B"),
        # The matrix's ("BB", "AB") cannot hold: "A" before the letter "B" is not a whole word (R2).
        # Same intent, adjacent repeats of a short word, where the edge rule does not apply:
        ("子 → 丑\n丑 → 寅", "丑丑", "子丑", "丑丑"),
        ("A → B\nB → C", "B B", "A B", "B B"),
        ("A → B\nB → C", "用B和C", "用B和B", "用B和B"),
        ("A → B\nB → C", "用B和C", "用B和C。", "用B和C。"),
        ("行 → 践行", "我们要践行动", "我们要践行动。", "我们要践行动。"),
        ("行 → 践行", "我们要践行动", "我们要行动。", "我们要践行动。"),
        ("见行 → 践行", "我们讲践行", "我们讲见行。", "我们讲践行。"),
        ("见行 → 践行", "讲践行和践行", "讲见行和见行", "讲践行和践行"),
        ("见行 → 践行", "讲践行。好", "好。讲见行", "好。讲践行"),
        ("java → Java", "用 Java 和 javascript", "用 java 和 javascript。", "用 Java 和 javascript。"),
        ("甲 → 乙\n乙 → 丙\n丙 → 甲", "乙", "甲", "甲"),
    ],
)
def test_fix_reply_exact_results(word_list, sent, reply, expected):
    assert fix_reply(sent, reply, parse_vocabulary(word_list + "\n")) == expected


# --- creating the file ---


def test_creating_the_word_list_twice_leaves_one_complete_file(tmp_path):
    path = tmp_path / "home" / "词表.txt"
    ensure_word_list(path)
    assert path.read_bytes() == TEMPLATE.encode("utf-8") and "\r" not in TEMPLATE
    assert load_vocabulary(path) == EMPTY
    assert sorted(p.name for p in path.parent.iterdir()) == ["词表.txt"]
    before = _snapshot(path)
    ensure_word_list(path)
    assert _snapshot(path) == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["词表.txt"]


@pytest.mark.parametrize(
    "data",
    ["践行\r\n".encode("gbk"), b"\xff\xfe" + "践行\n".encode("utf-16-le"), "café".encode("latin-1"), b""],
    ids=["gbk", "utf-16", "latin-1", "empty"],
)
def test_an_existing_word_list_is_never_touched(tmp_path, data):
    path = tmp_path / "词表.txt"
    path.write_bytes(data)
    before = _snapshot(path)
    ensure_word_list(path)
    assert _snapshot(path) == before
    assert [p.name for p in tmp_path.iterdir()] == ["词表.txt"]


def test_a_full_disk_leaves_no_half_word_list_and_a_later_try_works(tmp_path, monkeypatch):
    path = tmp_path / "词表.txt"

    def no_space(descriptor):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("coursescribe.files.os.fsync", no_space)
    with pytest.raises(OSError):
        ensure_word_list(path)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    ensure_word_list(path)
    assert path.read_bytes() == TEMPLATE.encode("utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["词表.txt"]


def test_a_word_list_another_program_creates_at_the_same_moment_is_kept(tmp_path, monkeypatch):
    """The target appears between the check and the publish: the other writer's file stays, no temp is left."""
    import coursescribe.files as files

    path = tmp_path / "词表.txt"
    theirs = "践行\n".encode("gbk")
    real_write_temp = files._write_temp

    def write_temp_then_another_program_writes(folder, data):
        temp = real_write_temp(folder, data)
        path.write_bytes(theirs)
        return temp

    monkeypatch.setattr(files, "_write_temp", write_temp_then_another_program_writes)
    ensure_word_list(path)
    assert path.read_bytes() == theirs
    assert [p.name for p in tmp_path.iterdir()] == ["词表.txt"]


def test_the_template_says_english_is_case_sensitive_and_whole_word():
    assert "# 英文区分大小写，只换完整的词；要换几种写法就各写一行。\n" in TEMPLATE
