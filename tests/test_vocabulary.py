"""The user's word list: common words, and fixes for words the engine mishears."""

from __future__ import annotations

from coursescribe.vocabulary import (
    TEMPLATE,
    Vocabulary,
    apply_fixes,
    ensure_word_list,
    load_vocabulary,
    parse_vocabulary,
)


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


def test_missing_or_unreadable_file_means_an_empty_list(tmp_path):
    assert load_vocabulary(tmp_path / "missing.txt") == Vocabulary((), ())
    folder = tmp_path / "folder.txt"
    folder.mkdir()
    assert load_vocabulary(folder) == Vocabulary((), ())


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
