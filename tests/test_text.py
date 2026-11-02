"""Text helpers. All samples here are written for these tests."""

from __future__ import annotations

import random

import pytest

from coursescribe.segments import Segment, segments_to_text
from coursescribe.text import (
    chunk_text,
    clean_fillers,
    clean_tics,
    content_chars,
    correct_terms,
    split_sentences,
)


def _seg(text: str) -> Segment:
    return Segment(0.0, 1.0, text)


class TestSegmentsToText:
    def test_one_line_per_segment_with_final_newline(self):
        assert segments_to_text([_seg(" 今天讲缓存 "), _seg("然后讲索引")]) == "今天讲缓存\n然后讲索引\n"

    def test_empty_segments_are_dropped(self):
        assert segments_to_text([_seg("今天"), _seg("  "), _seg("讲课")]) == "今天\n讲课\n"

    def test_punctuation_only_segment_joins_the_previous_line(self):
        assert segments_to_text([_seg("今天"), _seg("。"), _seg("讲课")]) == "今天。\n讲课\n"

    def test_leading_punctuation_only_segment_is_dropped(self):
        assert segments_to_text([_seg("……"), _seg("讲课")]) == "讲课\n"

    def test_no_segments_means_no_text(self):
        assert segments_to_text([]) == ""

    def test_repeated_content_is_kept(self):
        assert segments_to_text([_seg("重要"), _seg("重要")]) == "重要\n重要\n"


class TestCleanFillers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("我们开始讲啊，缓存很重要。", "我们开始讲，缓存很重要。"),
            ("其实呢，这个问题不难。", "其实，这个问题不难。"),
            ("然后呢，看第二页。", "然后，看第二页。"),
            ("嗯，今天讲索引。", "今天讲索引。"),
            ("呃，今天讲索引。", "今天讲索引。"),
            ("今天嗯讲呃索引。", "今天讲索引。"),
            ("今天讲索引，嗯。", "今天讲索引。"),
            ("今天讲索引。嗯。下面开始。", "今天讲索引。下面开始。"),
            ("好的，啊，继续。", "好的，继续。"),
            ("讲完了啊", "讲完了"),
            ("讲完了呢啊。", "讲完了。"),
        ],
    )
    def test_removes_meaningless_fillers(self, raw, expected):
        assert clean_fillers(raw) == expected

    @pytest.mark.parametrize(
        "kept",
        [
            "这是为什么呢？",
            "这是为什么呢，我们来看。",
            "那你呢，有什么想法。",
            "我们呢？",
            "你呢。",
            "怎么办呢。",
            "这个结果是对还是错呢，大家想想。",
            "也就是说，索引要先建好。",
            "也就是说索引要先建好。",
            "就是说，索引要先建好。",
            "这是真的啊？",
            "天啊，太慢了。",
            "呢喃这个词很少见。",
            "啊哈时刻是一种感觉。",
            "我们开始吧。",
            "这样可以，对吧？",
            "索引不是万能的。",
        ],
    )
    def test_keeps_real_questions_and_real_words(self, kept):
        assert clean_fillers(kept) == kept

    def test_line_breaks_mark_clause_ends_and_survive(self):
        assert clean_fillers("然后呢\n我们看第二页啊\n结束") == "然后\n我们看第二页\n结束"

    def test_a_line_of_only_filler_disappears(self):
        assert clean_fillers("开始\n嗯\n结束\n") == "开始\n结束\n"

    def test_space_between_chinese_clauses_is_a_boundary(self):
        assert clean_fillers("好的啊 我们开始") == "好的 我们开始"

    def test_english_is_not_touched(self):
        assert clean_fillers("use Claude Code, ok?") == "use Claude Code, ok?"

    def test_empty_text(self):
        assert clean_fillers("") == ""

    def test_is_stable_when_run_twice(self):
        once = clean_fillers("嗯，然后呢，啊，我们开始啊。")
        assert clean_fillers(once) == once

    def test_only_filler_characters_are_ever_lost(self):
        rng = random.Random(7)
        pieces = ["索引", "缓存", "呢", "啊", "嗯", "呃", "，", "。", "？", "\n", " ", "为什么", "Claude", "吧"]
        for _ in range(500):
            raw = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
            kept = [c for c in content_chars(raw) if c not in "呢啊嗯呃"]
            cleaned = [c for c in content_chars(clean_fillers(raw)) if c not in "呢啊嗯呃"]
            assert cleaned == kept, raw


class TestCorrectTerms:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("我们用claude code写代码", "我们用Claude Code写代码"),
            ("chat gpt和chatgpt", "ChatGPT和ChatGPT"),
            ("把代码放到github上", "把代码放到GitHub上"),
            ("用python写", "用Python写"),
            ("GITHUB", "GitHub"),
        ],
    )
    def test_fixes_known_terms(self, raw, expected):
        assert correct_terms(raw) == expected

    @pytest.mark.parametrize(
        "kept",
        [
            "配置在 ~/.claude 目录",
            "pip install my-python-tool",
            "digit 和 legit 不是 Git 的意思",
            "pythonic 风格",
            "src/python/main.py",
        ],
    )
    def test_leaves_paths_and_longer_words_alone(self, kept):
        assert correct_terms(kept) == kept

    def test_already_correct_text_is_unchanged(self):
        text = "Claude Code 和 GitHub"
        assert correct_terms(correct_terms(text)) == text


class TestSplitSentences:
    def test_keeps_every_character(self):
        text = "第一句。第二句！第三句？\n  没有结尾"
        assert "".join(split_sentences(text)) == text

    def test_splits_after_each_terminator_run(self):
        assert split_sentences("真的吗？！对。") == ["真的吗？！", "对。"]

    def test_closing_quote_stays_with_its_sentence(self):
        assert split_sentences("他说“好。”然后走了。") == ["他说“好。”", "然后走了。"]

    @pytest.mark.parametrize("text", ["版本 3.5 发布", "用 v1.2 版本", "example.com 网站"])
    def test_a_dot_inside_a_token_does_not_split(self, text):
        assert split_sentences(text) == [text]

    def test_english_full_stop_followed_by_space_splits(self):
        assert split_sentences("Use it. Then stop.") == ["Use it. ", "Then stop."]

    def test_empty_text_has_no_sentences(self):
        assert split_sentences("") == []


class TestChunkText:
    def test_no_chunk_exceeds_the_limit_and_nothing_is_lost(self):
        rng = random.Random(3)
        for _ in range(200):
            text = "".join(
                rng.choice(["短句。", "这一行比较长一点没有标点", "\n", "English words here. ", "问题？"])
                for _ in range(rng.randint(0, 80))
            )
            chunks = chunk_text(text, 50)
            assert "".join(chunks) == text
            assert all(0 < len(chunk) <= 50 for chunk in chunks)

    def test_prefers_to_cut_at_line_ends(self):
        assert chunk_text("甲甲甲\n乙乙乙\n丙丙丙\n", 8) == ["甲甲甲\n乙乙乙\n", "丙丙丙\n"]

    def test_long_line_is_cut_at_a_sentence_end(self):
        assert chunk_text("甲甲甲。乙乙乙乙乙。丙", 6) == ["甲甲甲。", "乙乙乙乙乙。", "丙"]

    def test_long_line_without_any_punctuation_is_cut_at_the_limit(self):
        assert chunk_text("甲" * 7, 3) == ["甲甲甲", "甲甲甲", "甲"]

    def test_a_cut_does_not_split_an_english_word(self):
        assert chunk_text("甲甲Claude甲", 7) == ["甲甲", "Claude甲"]

    def test_a_word_longer_than_the_limit_is_still_cut(self):
        assert chunk_text("abcdefg", 3) == ["abc", "def", "g"]

    def test_empty_text_has_no_chunks(self):
        assert chunk_text("", 10) == []

    def test_limit_must_be_positive(self):
        with pytest.raises(ValueError):
            chunk_text("x", 0)


class TestContentChars:
    def test_ignores_punctuation_space_and_width_and_case(self):
        assert content_chars("Ａ b，。 c\n1") == content_chars("a B c1")


class TestCleanTics:
    """Verbal tics go; everything said for its meaning stays (PRD: clean copy)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("资产涨了三倍了\n对吧\n然后我们看\n", "资产涨了三倍了\n然后我们看\n"),
            ("因为你过去也要学\n对不对\n那么现在\n", "因为你过去也要学\n那么现在\n"),
            ("开始\nyeah\n这个\n那个\n对啊\n结束\n", "开始\n结束\n"),
            ("资产至少涨了三倍了对吧\n", "资产至少涨了三倍了\n"),
            ("留下的40%里面对吧\n真正在践行的\n", "留下的40%里面\n真正在践行的\n"),
            ("开始\n然后呢\n结束\n", "开始\n结束\n"),
            ("这样可以，对吧？下面继续。", "这样可以。下面继续。"),
            ("这样可以，对吧，下面继续。", "这样可以，下面继续。"),
        ],
    )
    def test_removes_tag_questions_and_pause_words(self, raw, expected):
        assert clean_tics(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("你可以观察到\n观察到的\n绝大多数人\n", "你可以\n观察到的\n绝大多数人\n"),
            ("所以你要改变的\n改变的更多的是\n", "所以你要\n改变的更多的是\n"),
        ],
    )
    def test_a_stutter_across_a_pause_is_said_once(self, raw, expected):
        assert clean_tics(raw) == expected

    @pytest.mark.parametrize(
        "kept",
        [
            "行动最重要\n行动最重要\n",  # repeated on purpose
            "不断的重复\n不断的重复\n",
            "这个问题不难\n",
            "那个时候我还小\n",
            "对吧台的设计很满意\n",
            "他说的是好的\n好的我们继续\n",  # two-character overlap is too short to call a stutter
            "对\n就是这样\n",
            "然后我们看第二页\n",
        ],
    )
    def test_keeps_words_that_carry_meaning(self, kept):
        assert clean_tics(kept) == kept

    def test_is_stable_when_run_twice(self):
        once = clean_tics("开始了对吧\n对不对\n你可以观察到\n观察到的人\n")
        assert clean_tics(once) == once

    def test_never_adds_or_reorders_characters(self):
        rng = random.Random(11)
        pieces = ["索引", "对吧", "对不对", "这个", "然后呢", "观察到", "观察到的", "\n", "，", "？", "行动", "yeah"]
        for _ in range(500):
            raw = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 20)))
            cleaned = content_chars(clean_tics(raw))
            it = iter(content_chars(raw))
            assert all(ch in it for ch in cleaned), raw
