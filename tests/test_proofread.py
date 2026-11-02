"""Proofreading rules, exercised through a real HTTP client and a fake service."""

from __future__ import annotations

import itertools
import json
import socket
import time
import urllib.error
from pathlib import Path

import pytest

from coursescribe.models import DEFAULT_PROOFREAD_MODEL
from coursescribe.ollama import (
    OllamaBusy,
    OllamaClient,
    OllamaDown,
    OllamaError,
    OllamaModelMissing,
    OllamaTimeout,
)
from coursescribe.pipeline import ProofreadBlocked
from coursescribe.proofread import (
    BREAK_MARKER,
    BUSY_MESSAGE,
    TEXT_MARKER,
    TIMEOUT_MESSAGE,
    OllamaProofreader,
    apply_breaks,
    group_windows,
    is_lossless,
    parse_breaks,
)
from coursescribe.text import content_chars
from coursescribe.vocabulary import parse_vocabulary
from fake_ollama import FakeOllama, Reply

SOURCE = "今天我们讲数据库索引\n索引可以让查询变快\n但是写入会变慢一点\n"
PROOFED = "今天我们讲数据库索引。索引可以让查询变快，但是写入会变慢一点。"


def chunk_of(prompt: str) -> str:
    return prompt.rsplit(TEXT_MARKER, 1)[1]


def echo_with_stop(prompt: str) -> str:
    """A well-behaved model: gives back the same words with a full stop."""
    if BREAK_MARKER in prompt:
        return "无"
    return chunk_of(prompt) + "。"


@pytest.fixture
def service(request):
    made = []

    def start(respond=echo_with_stop, **kwargs) -> FakeOllama:
        fake = FakeOllama(respond, **kwargs)
        made.append(fake)
        return fake

    yield start
    for fake in made:
        fake.close()


def _proofreader(fake, tmp_path=None, **kwargs) -> OllamaProofreader:
    client = OllamaClient(
        fake.url, timeout=kwargs.pop("timeout", 5.0), unload_timeout=kwargs.pop("unload_timeout", 15.0)
    )
    kwargs.setdefault("retry_pause", 0)
    return OllamaProofreader(client, **kwargs)


def _closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


class TestIsLossless:
    ORIGINAL = "今天我们讲数据库索引索引可以让查询变快但是写入会变慢一点所以要按实际情况来选择"

    @pytest.mark.parametrize(
        "corrected",
        [
            "今天我们讲数据库索引。索引可以让查询变快，但是写入会变慢一点，所以要按实际情况来选择。",
            "今天我们讲数据库索引，索引可以让查询变快，但是写入会变慢一点，所以要按实际情况来选泽。",
        ],
    )
    def test_accepts_punctuation_and_a_fixed_typo(self, corrected):
        assert is_lossless(self.ORIGINAL, corrected)

    def test_accepts_removing_fillers(self):
        assert is_lossless("今天嗯我们讲啊数据库索引", "今天我们讲数据库索引。")

    def test_accepts_writing_a_misheard_term_in_english(self):
        assert is_lossless("我们用克劳德写代码然后再看一下结果怎么样", "我们用 Claude 写代码，然后再看一下结果怎么样。")

    @pytest.mark.parametrize(
        ("name", "corrected"),
        [
            ("dropped sentence", "今天我们讲数据库索引。索引可以让查询变快，所以要按实际情况来选择。"),
            ("dropped three characters", "今天我们讲数据库索引索引可以让查询变快入会变慢一点所以要按实际情况来选择"),
            ("summary", "这段话讲了索引的优缺点。"),
            ("preface", "好的以下是校对后的内容今天我们讲数据库索引索引可以让查询变快但是写入会变慢一点所以要按实际情况来选择"),
            ("rewrite", "今天的主题是数据库的索引功能它能够加速查询但写入速度略有下降因此需要结合场景来决定"),
            ("repeat", "今天我们讲数据库索引索引可以让查询变快但是写入会变慢一点所以要按实际情况来选择今天我们讲数据库索引"),
            ("empty", ""),
        ],
    )
    def test_rejects_lost_or_invented_content(self, name, corrected):
        assert not is_lossless(self.ORIGINAL, corrected), name

    @pytest.mark.parametrize("added", ["# 数据库索引\n", "**今天**", "- 今天", "`今天`"])
    def test_rejects_added_formatting(self, added):
        assert not is_lossless("今天", added if "今天" in added else added + "今天")

    def test_formatting_already_in_the_source_is_fine(self):
        assert is_lossless("用 `git` 命令", "用 `git` 命令。")

    def test_repeated_teaching_passages_are_compared_in_place(self):
        passage = "索引就像一本书的目录可以帮助我们快速找到内容如果没有索引就只能一条一条的查找"
        original = passage * 8
        corrected = passage.replace("一条的", "一条地") + passage * 7
        assert is_lossless(original, corrected)
        assert is_lossless(original, "，".join([passage] * 8))

    def test_a_lost_passage_among_repeats_is_still_caught(self):
        passage = "索引就像一本书的目录"
        assert not is_lossless(passage * 8, passage * 7)
        assert not is_lossless(passage * 8, passage * 9)

    def test_very_different_lengths_are_rejected_quickly(self):
        started = time.perf_counter()
        assert not is_lossless("甲" * 5000, "甲" * 100)
        assert not is_lossless("甲乙" * 2000, "丙丁" * 2000)
        assert time.perf_counter() - started < 1.0

    def test_real_reply_on_a_repeating_lecture_is_not_mistaken_for_lost_text(self):
        """Self-written lesson text and a real model reply that an earlier comparison rejected."""
        sample = json.loads((Path(__file__).parent / "fixtures" / "repeating_lecture.json").read_text(encoding="utf-8"))
        assert is_lossless(sample["original"], sample["corrected"])

    def test_nothing_to_check(self):
        assert is_lossless("", "")


class TestCleanCopyEdits:
    """The clean copy may drop tics and stutters, nothing else (PRD)."""

    @pytest.mark.parametrize(
        ("original", "corrected"),
        [
            ("因为你过去也是要学编程的对不对那么现在来看", "因为你过去也是要学编程的。那么现在来看，"),
            ("然后呢这个这个编程领域只能算是一个子集", "编程领域只能算是一个子集。"),
            ("你可以观察到观察到的绝大多数人只是获取了信息", "你可以观察到的绝大多数人，只是获取了信息。"),
            ("然后他就像你的研究生一样派发出一堆的subagent", "然后它就像你的研究生一样，派发出一堆的 subagent。"),
            ("yeah然后我们继续讲", "然后我们继续讲。"),
            ("你要用client你不要用这个GUI", "你要用CLI，你不要用这个GUI。"),  # a misheard English term
        ],
    )
    def test_accepts_dropped_tics_stutters_and_pronoun_fixes(self, original, corrected):
        assert is_lossless(original, corrected)

    @pytest.mark.parametrize(
        "corrected",
        [
            "所以今天我们用AI去辅助学习。就是信息获取了之后消化的能力没有变。",  # a sentence lost
            "所以今天我们用AI去辅助学习，然而有一件事情变了。",  # meaning reversed
        ],
    )
    def test_still_rejects_lost_or_changed_meaning(self, corrected):
        original = "所以今天我们用AI去辅助学习然而有一件事情没有变就是信息获取了之后消化的能力没有变"
        assert not is_lossless(original, corrected)


class TestParseBreaks:
    @pytest.mark.parametrize(
        ("reply", "count", "expected"),
        [
            ("3,8,15", 20, [3, 8, 15]),
            ("在第 3 句和第 8 句之后", 20, [3, 8]),
            ("３，８", 20, [3, 8]),
            ("8,3,3", 20, [3, 8]),
            ("无", 20, []),
            ("0,20,21,5", 20, [5]),
            ("1.5,-2,4", 20, [4]),
            ("9" * 50, 20, []),
        ],
    )
    def test_reads_only_valid_positions(self, reply, count, expected):
        assert parse_breaks(reply, count) == expected


class TestApplyBreaks:
    SENTENCES = ["甲" * 50 + "。", "乙" * 50 + "。", "丙" * 50 + "。"]

    def test_inserts_blank_lines_and_keeps_all_text(self):
        result = apply_breaks(self.SENTENCES, [1, 2], min_chars=40)
        assert result == "\n\n".join(self.SENTENCES)

    def test_a_break_that_would_leave_a_short_paragraph_is_ignored(self):
        sentences = ["短。", "乙" * 50 + "。"]
        assert apply_breaks(sentences, [1], min_chars=40) == "".join(sentences)

    def test_no_sentences(self):
        assert apply_breaks([], [1], min_chars=40) == ""

    def test_a_paragraph_too_long_to_read_is_split_at_sentence_ends(self):
        sentences = ["甲" * 60 + "。"] * 10
        result = apply_breaks(sentences, [], min_chars=40, max_chars=300)
        paragraphs = result.split("\n\n")
        assert len(paragraphs) > 1
        assert all(len(p) <= 300 for p in paragraphs)
        assert "".join(paragraphs) == "".join(sentences)

    def test_one_very_long_sentence_stays_whole(self):
        assert apply_breaks(["甲" * 500 + "。"], [], min_chars=40, max_chars=300) == "甲" * 500 + "。"


def test_windows_hold_whole_sentences_up_to_the_limit():
    assert group_windows(["甲甲", "乙乙", "丙丙丙丙丙丙", "丁"], 4) == [["甲甲", "乙乙"], ["丙丙丙丙丙丙"], ["丁"]]


class TestOllamaClient:
    def test_refuses_any_address_outside_this_computer(self):
        with pytest.raises(ValueError):
            OllamaClient("http://example.com:11434")

    def test_system_proxy_is_bypassed(self, service, monkeypatch):
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        fake = service(lambda prompt: "好的")
        assert OllamaClient(fake.url).generate("model-a", "问") == "好的"

    def test_connection_refused_is_reported_as_down(self):
        with pytest.raises(OllamaDown):
            OllamaClient(_closed_port_url()).models()

    def test_slow_reply_is_a_timeout(self, service):
        fake = service(lambda prompt: Reply("晚了", delay=1.0))
        with pytest.raises(OllamaTimeout):
            OllamaClient(fake.url, timeout=0.1).generate("model-a", "问")

    def test_thinking_is_turned_off_only_for_models_that_support_it(self, service):
        plain = service(lambda prompt: "好")
        OllamaClient(plain.url).generate("model-a", "问")
        assert "think" not in plain.requests[-1]

        thinking = service(lambda prompt: "好", capabilities=("completion", "thinking"))
        OllamaClient(thinking.url).generate("model-a", "问")
        assert thinking.requests[-1]["think"] is False

    def test_unload_asks_the_service_to_free_memory_now(self, service):
        fake = service()
        OllamaClient(fake.url).unload("model-a")
        assert fake.requests[-1]["keep_alive"] == 0

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            (Reply("x", status=500), OllamaBusy),
            (Reply("x", status=429), OllamaBusy),
            (Reply("x", status=404), OllamaModelMissing),
            (Reply(raw=b"<html>busy</html>"), OllamaBusy),
            (Reply(raw=b"[1,2]"), OllamaBusy),
        ],
        ids=["500", "429", "404", "not-json", "json-list"],
    )
    def test_each_kind_of_answer_is_its_own_error(self, service, reply, expected):
        fake = service(lambda prompt: reply)
        with pytest.raises(OllamaError) as caught:
            OllamaClient(fake.url).generate("model-a", "问")
        assert caught.type is expected

    def test_a_bad_request_is_not_a_temporary_error(self, service):
        fake = service(lambda prompt: Reply("x", status=400))
        with pytest.raises(OllamaError) as caught:
            OllamaClient(fake.url).generate("model-a", "问")
        assert caught.type is OllamaError
        assert not isinstance(caught.value, OllamaBusy | OllamaTimeout)

    def test_an_unreadable_model_list_is_a_temporary_error(self, service):
        fake = service(tags_reply=lambda: Reply(raw=b"[]"))
        with pytest.raises(OllamaError) as caught:
            OllamaClient(fake.url).models()
        assert caught.type is OllamaBusy

    @pytest.mark.parametrize("broken", ["no-reply", "short-body"])
    def test_a_connection_dropped_after_asking_is_a_temporary_error(self, service, broken):
        fake = service(lambda prompt: Reply(broken=broken))
        with pytest.raises(OllamaError) as caught:
            OllamaClient(fake.url, timeout=5.0).generate("model-a", "问")
        assert caught.type is OllamaBusy

    def test_a_timeout_wrapped_by_urllib_is_a_timeout(self, service, monkeypatch):
        client = OllamaClient(service().url)

        def open_(*args, **kwargs):
            raise urllib.error.URLError(TimeoutError("timed out"))

        monkeypatch.setattr(client._opener, "open", open_)
        with pytest.raises(OllamaError) as caught:
            client.models()
        assert caught.type is OllamaTimeout

    @pytest.mark.parametrize("status", [500, 400])
    def test_the_services_own_reason_is_kept_in_the_error(self, service, status):
        fake = service(lambda prompt: Reply("model runner has unexpectedly stopped", status=status))
        with pytest.raises(OllamaError) as caught:
            OllamaClient(fake.url).generate("model-a", "问")
        assert str(status) in str(caught.value)
        assert "model runner has unexpectedly stopped" in str(caught.value)

    def test_the_reason_kept_is_short(self, service):
        fake = service(lambda prompt: Reply("长" * 5000, status=500))
        with pytest.raises(OllamaError) as caught:
            OllamaClient(fake.url).generate("model-a", "问")
        assert "长" in str(caught.value)
        assert len(str(caught.value).encode("utf-8")) < 700

    def test_unloading_gives_up_quickly_when_the_service_is_stuck(self, service, caplog):
        fake = service(unload_reply=lambda: Reply(delay=10.0))
        reader = OllamaProofreader(OllamaClient(fake.url, timeout=30, unload_timeout=0.3))
        started = time.perf_counter()
        with caplog.at_level("INFO", logger="coursescribe.proofread"):
            reader.release("model-a")
        assert time.perf_counter() - started < 2.0
        assert "Could not unload" in caplog.text


class TestProofread:
    def test_good_run_produces_proofread_markdown(self, service, tmp_path):
        fake = service(lambda p: "无" if BREAK_MARKER in p else PROOFED)
        result = _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        assert result.markdown == PROOFED + "\n"
        assert (result.unproofread_chunks, result.total_chunks) == (0, 1)

    def test_fillers_are_gone_even_when_only_the_model_added_punctuation(self, service, tmp_path):
        fake = service(lambda p: "无" if BREAK_MARKER in p else "好的啊，我们开始讲呢。")
        result = _proofreader(fake).proofread("好的啊我们开始讲呢\n", "model-a", tmp_path / "c.json", lambda d, t: None)
        assert result.markdown == "好的，我们开始讲。\n"

    def test_terms_are_corrected_before_the_model_sees_them(self, service, tmp_path):
        fake = service()
        _proofreader(fake).proofread("我们用github存代码\n", "model-a", tmp_path / "c.json", lambda d, t: None)
        assert "GitHub" in chunk_of(fake.prompts[0])

    def test_english_words_on_separate_lines_stay_apart(self, service, tmp_path):
        fake = service()
        _proofreader(fake).proofread("use the\ncache now\n", "model-a", tmp_path / "c.json", lambda d, t: None)
        assert "the cache" in chunk_of(fake.prompts[0])

    def test_lost_text_is_retried_then_accepted(self, service, tmp_path):
        replies = iter(["今天", PROOFED])

        def respond(prompt):
            return "无" if BREAK_MARKER in prompt else next(replies)

        fake = service(respond)
        result = _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        assert result.markdown == PROOFED + "\n"
        assert result.unproofread_chunks == 0

    @pytest.mark.parametrize(
        "bad_reply",
        [
            Reply("这段讲索引。"),  # lost text
            Reply("   "),  # empty
            Reply("x", status=400),  # a request the service will never accept
        ],
        ids=["lost-text", "empty", "bad-request"],
    )
    def test_bad_chunk_keeps_its_original_words_and_is_counted(self, service, tmp_path, bad_reply):
        good = "好段落一句话。" * 30
        bad = "这一段模型总是处理不好"
        text = good + "\n" + bad * 30 + "\n"

        def respond(prompt):
            if BREAK_MARKER in prompt:
                return "无"
            return bad_reply if bad in prompt else chunk_of(prompt)

        fake = service(respond)
        result = _proofreader(fake, chunk_chars=300).proofread(
            text, "model-a", tmp_path / "c.json", lambda d, t: None
        )
        assert (result.unproofread_chunks, result.total_chunks) == (2, 3)
        assert content_chars(result.markdown) == content_chars(text)
        assert sum(bad in prompt for prompt in fake.prompts) == 4  # two chunks, two tries each

    def test_every_chunk_failing_writes_no_markdown_and_waits(self, service, tmp_path):
        fake = service(lambda prompt: Reply("   "))
        with pytest.raises(ProofreadBlocked, match="校对"):
            _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)

    def test_retrying_after_every_chunk_failed_asks_the_model_again(self, service, tmp_path):
        fake = service(lambda prompt: Reply("   "))
        with pytest.raises(ProofreadBlocked):
            _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        fixed = service(lambda p: "无" if BREAK_MARKER in p else PROOFED)
        result = _proofreader(fixed).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        assert result.unproofread_chunks == 0

    def test_service_down_blocks_and_says_what_to_do(self, tmp_path):
        reader = OllamaProofreader(OllamaClient(_closed_port_url()))
        with pytest.raises(ProofreadBlocked, match="Ollama"):
            reader.proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)

    def test_missing_model_blocks_and_names_the_model(self, service, tmp_path):
        fake = service(models=("other",))
        with pytest.raises(ProofreadBlocked, match="model-a"):
            _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        assert fake.prompts == []

    def test_service_dying_halfway_keeps_finished_chunks_for_later(self, service, tmp_path):
        text = "第一段内容在这里。\n" * 40
        checkpoint = tmp_path / "c.json"

        class DyingClient(OllamaClient):
            calls = 0

            def generate(self, model, prompt):
                DyingClient.calls += 1
                if DyingClient.calls == 3:
                    raise OllamaDown("connection lost")
                return super().generate(model, prompt)

        with pytest.raises(ProofreadBlocked, match="Ollama"):
            OllamaProofreader(DyingClient(service().url), chunk_chars=100).proofread(
                text, "model-a", checkpoint, lambda d, t: None
            )

        fresh = service()
        result = _proofreader(fresh, chunk_chars=100).proofread(text, "model-a", checkpoint, lambda d, t: None)
        asked = [p for p in fresh.prompts if BREAK_MARKER not in p]
        assert len(asked) == result.total_chunks - 2
        assert content_chars(result.markdown) == content_chars(text)

    def test_checkpoint_from_different_text_is_not_reused(self, service, tmp_path):
        checkpoint = tmp_path / "c.json"
        fake = service()
        _proofreader(fake).proofread("第一份文字内容\n", "model-a", checkpoint, lambda d, t: None)
        result = _proofreader(fake).proofread("第二份文字内容\n", "model-a", checkpoint, lambda d, t: None)
        assert "第二份" in result.markdown and "第一份" not in result.markdown

    def test_damaged_checkpoint_is_ignored(self, service, tmp_path):
        checkpoint = tmp_path / "c.json"
        checkpoint.write_text("{broken", encoding="utf-8")
        result = _proofreader(service()).proofread(SOURCE, "model-a", checkpoint, lambda d, t: None)
        assert content_chars(result.markdown) == content_chars(SOURCE)

    def test_paragraphs_come_only_from_positions_the_model_returns(self, service, tmp_path):
        one = "甲" * 50 + "。"
        two = "乙" * 50 + "。"

        def respond(prompt):
            # A hostile reply: body text in the break answer must not reach the output.
            return "1 这里是模型自己编的话" if BREAK_MARKER in prompt else one + two

        fake = service(respond)
        result = _proofreader(fake).proofread("甲" * 50 + "乙" * 50 + "\n", "model-a", tmp_path / "c.json", lambda d, t: None)
        assert result.markdown == f"{one}\n\n{two}\n"

    def test_paragraph_request_failing_still_finishes_with_full_text(self, service, tmp_path):
        def respond(prompt):
            return Reply("x", status=400) if BREAK_MARKER in prompt else PROOFED

        result = _proofreader(service(respond)).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        assert result.markdown == PROOFED + "\n"
        assert result.unproofread_chunks == 0

    def test_progress_counts_each_chunk(self, service, tmp_path):
        seen = []
        result = _proofreader(service(), chunk_chars=200).proofread(
            "这是一句话的内容。\n" * 60, "model-a", tmp_path / "c.json", lambda d, t: seen.append((d, t))
        )
        assert seen[-1] == (result.total_chunks, result.total_chunks)
        assert [d for d, _ in seen] == list(range(1, result.total_chunks + 1))

    def test_tics_and_word_list_fixes_are_applied_before_the_model_sees_the_text(self, service, tmp_path):
        fake = service()
        vocabulary = parse_vocabulary("见行 → 践行\n")
        _proofreader(fake, vocabulary=lambda: vocabulary).proofread(
            "我们讲见行\n对不对\n这很重要\n", "model-a", tmp_path / "c.json", lambda d, t: None
        )
        chunk = chunk_of(fake.prompts[0])
        assert "践行" in chunk and "见行" not in chunk and "对不对" not in chunk

    def test_common_words_are_given_to_the_model(self, service, tmp_path):
        fake = service()
        vocabulary = parse_vocabulary("定投\n见行 → 践行\n")
        _proofreader(fake, vocabulary=lambda: vocabulary).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        polish = next(p for p in fake.prompts if BREAK_MARKER not in p)
        assert "定投" in polish.split(TEXT_MARKER)[0] and "践行" in polish.split(TEXT_MARKER)[0]

    def test_no_word_list_means_no_word_hint(self, service, tmp_path):
        fake = service()
        _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        assert "常用词" not in fake.prompts[0]

    def test_a_fix_the_model_undoes_is_applied_again(self, service, tmp_path):
        fake = service(lambda p: "无" if BREAK_MARKER in p else "我们讲见行，这很重要。")
        vocabulary = parse_vocabulary("见行 → 践行\n")
        result = _proofreader(fake, vocabulary=lambda: vocabulary).proofread(
            "我们讲见行\n这很重要\n", "model-a", tmp_path / "c.json", lambda d, t: None
        )
        assert "践行" in result.markdown and "见行" not in result.markdown

    def test_a_chunk_the_model_fails_still_loses_its_tics(self, service, tmp_path):
        fake = service(lambda p: "完全不同的一段话。" if BREAK_MARKER not in p and "第一段" in chunk_of(p) else echo_with_stop(p))
        text = "第一段内容在这里\n对吧\n" + "甲" * 400 + "\n"
        result = _proofreader(fake, chunk_chars=300).proofread(text, "model-a", tmp_path / "c.json", lambda d, t: None)
        assert result.unproofread_chunks >= 1
        assert "对吧" not in result.markdown and "第一段内容在这里" in result.markdown

    def test_prompt_asks_for_a_clean_copy(self, service, tmp_path):
        fake = service()
        _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        polish = next(p for p in fake.prompts if BREAK_MARKER not in p)
        assert "口头禅" in polish and "它" in polish and "总结" in polish

    def test_prompt_requires_verbatim_text_and_english_terms(self, service, tmp_path):
        fake = service()
        _proofreader(fake).proofread(SOURCE, "model-a", tmp_path / "c.json", lambda d, t: None)
        polish = next(p for p in fake.prompts if BREAK_MARKER not in p)
        assert "逐字保留" in polish and "英文" in polish


LINES = "第一段内容在这里。\n" * 40  # four chunks of 100 characters, all alike
THREE = "甲" * 89 + "\n" + "乙" * 89 + "\n" + "丙" * 89 + "\n"  # three chunks of 100, one break window
LOST = Reply("这段讲索引。")
TIMEOUT = 0.5  # client timeout in these tests; SLOW answers well after it
SLOW = Reply("晚了", delay=5.0)
RETRY_NOTE_WORDS = "上一次的结果漏掉"


def polish_prompts(fake) -> list[str]:
    return [p for p in fake.prompts if BREAK_MARKER not in p]


def break_prompts(fake) -> list[str]:
    return [p for p in fake.prompts if BREAK_MARKER in p]


def polish_times(fake) -> list[float]:
    return [t for t, path, prompt in fake.log if prompt is not None and BREAK_MARKER not in prompt]


def tags_requests(fake) -> int:
    return sum(path == "/api/tags" for _, path, _ in fake.log)


def saved(checkpoint: Path) -> dict:
    """The checkpoint as saved, or an empty one if nothing was saved."""
    if not checkpoint.exists():
        return {"chunks": {}, "breaks": {}}
    return json.loads(checkpoint.read_text(encoding="utf-8"))


def nth_polish(bad, from_index: int = 0, until: int | None = None):
    """A good model, except that proofreading requests number from_index..until (from 0) get ``bad``."""
    count = itertools.count()

    def respond(prompt):
        if BREAK_MARKER in prompt:
            return "无"
        index = next(count)
        if index >= from_index and (until is None or index < until):
            return bad
        return chunk_of(prompt) + "。"

    return respond


def in_turn(replies, breaks="无"):
    """Proofreading requests get ``replies`` one after another, then the last one again."""
    left = list(replies)

    def respond(prompt):
        if BREAK_MARKER in prompt:
            return breaks
        return left.pop(0) if len(left) > 1 else left[0]

    return respond


def three_part_model(faults: dict[str, list[Reply]] | None = None, accepted: list[str] | None = None):
    """A good model for THREE; ``faults`` maps 甲/乙/丙/"break" to replies given there first."""
    left = {place: list(replies) for place, replies in (faults or {}).items()}

    def respond(prompt):
        place = "break" if BREAK_MARKER in prompt else chunk_of(prompt)[0]
        if left.get(place):
            return left[place].pop(0)
        if place == "break":
            return "2"
        if accepted is not None:
            accepted.append(place)
        return chunk_of(prompt) + "。"

    return respond


def _run(reader, text, checkpoint):
    return reader.proofread(text, "model-a", checkpoint, lambda d, t: None)


class TestTemporaryServiceErrors:
    """A busy or slow service makes the task wait; it never marks a chunk unproofread (PRD)."""

    @pytest.mark.parametrize("message", [TIMEOUT_MESSAGE, BUSY_MESSAGE], ids=["timeout", "busy"])
    def test_messages_explain_their_terms_in_plain_words(self, message):
        assert "显存" in message
        assert message.count("显存") == message.count("显卡内存（显存）")
        assert "继续校对" in message

    def test_busy_message_says_how_to_restart_ollama(self):
        assert "托盘图标里退出 Ollama 再打开" in BUSY_MESSAGE

    def test_a_chunk_timing_out_waits_and_keeps_the_chunks_before_it(self, service, tmp_path):
        checkpoint = tmp_path / "c.json"
        fake = service(nth_polish(SLOW, from_index=1, until=2))
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(fake, timeout=TIMEOUT, chunk_chars=100), LINES, checkpoint)
        assert str(caught.value) == TIMEOUT_MESSAGE
        chunks = saved(checkpoint)["chunks"]
        assert set(chunks) == {"0"} and chunks["0"]["ok"] is True
        assert len(polish_prompts(fake)) == 2  # the slow chunk was asked once, not again

    def test_one_service_error_is_sent_again_after_a_pause_and_succeeds(self, service, tmp_path):
        fake = service(in_turn([Reply("x", status=500), PROOFED]))
        result = _run(_proofreader(fake, retry_pause=0.3), SOURCE, tmp_path / "c.json")
        assert result.markdown == PROOFED + "\n"
        assert (result.unproofread_chunks, result.total_chunks) == (0, 1)
        first, second = polish_prompts(fake)
        assert first == second and RETRY_NOTE_WORDS not in second
        sent = polish_times(fake)
        assert sent[1] - sent[0] >= 0.3

    def test_a_service_error_does_not_use_up_the_models_second_try(self, service, tmp_path):
        fake = service(in_turn([Reply("x", status=500), LOST, PROOFED]))
        result = _run(_proofreader(fake, retry_pause=0.05), SOURCE, tmp_path / "c.json")
        assert result.markdown == PROOFED + "\n"
        assert result.unproofread_chunks == 0
        prompts = polish_prompts(fake)
        assert len(prompts) == 3
        assert prompts[0] == prompts[1] and RETRY_NOTE_WORDS not in prompts[1]
        assert RETRY_NOTE_WORDS in prompts[2]
        sent = polish_times(fake)
        assert sent[1] - sent[0] < 2.0

    @pytest.mark.parametrize(
        "bad",
        [Reply("x", status=500), Reply("x", status=503), Reply(raw=b"not json")],
        ids=["500", "503", "not-json"],
    )
    def test_two_service_errors_in_a_row_wait_after_sending_twice(self, service, tmp_path, bad):
        checkpoint = tmp_path / "c.json"
        fake = service(nth_polish(bad, from_index=1))
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(fake, chunk_chars=100), LINES, checkpoint)
        assert str(caught.value) == BUSY_MESSAGE
        assert len(polish_prompts(fake)) == 3  # chunk 1 once, chunk 2 twice
        assert set(saved(checkpoint)["chunks"]) == {"0"}

    def test_a_service_error_then_a_timeout_waits_with_the_timeout_message(self, service, tmp_path):
        checkpoint = tmp_path / "c.json"
        replies = iter([Reply("x", status=500), SLOW])
        count = itertools.count()

        def respond(prompt):
            if BREAK_MARKER in prompt:
                return "无"
            return next(replies) if next(count) >= 1 else chunk_of(prompt) + "。"

        fake = service(respond)
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(fake, timeout=TIMEOUT, chunk_chars=100), LINES, checkpoint)
        assert str(caught.value) == TIMEOUT_MESSAGE
        assert len(polish_prompts(fake)) == 3
        assert set(saved(checkpoint)["chunks"]) == {"0"}

    def test_a_timeout_after_lost_text_still_waits_instead_of_failing_the_chunk(self, service, tmp_path):
        checkpoint = tmp_path / "c.json"
        fake = service(in_turn([LOST, SLOW]))
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(fake, timeout=TIMEOUT), SOURCE, checkpoint)
        assert str(caught.value) == TIMEOUT_MESSAGE
        assert saved(checkpoint)["chunks"] == {}

        fixed = service(lambda p: "无" if BREAK_MARKER in p else PROOFED)
        result = _run(_proofreader(fixed), SOURCE, checkpoint)
        assert result.unproofread_chunks == 0
        assert result.markdown == PROOFED + "\n"

    @pytest.mark.parametrize(
        ("tags_reply", "message", "asked"),
        [
            (lambda: SLOW, TIMEOUT_MESSAGE, 1),
            (lambda: Reply("x", status=500), BUSY_MESSAGE, 2),
        ],
        ids=["timeout", "server-error"],
    )
    def test_model_list_failing_waits_without_touching_the_checkpoint(
        self, service, tmp_path, tags_reply, message, asked
    ):
        checkpoint = tmp_path / "c.json"
        stopping = service(nth_polish(Reply("x", status=500), from_index=2))
        with pytest.raises(ProofreadBlocked):  # stops at the third chunk, leaving two saved
            _run(_proofreader(stopping, chunk_chars=100), LINES, checkpoint)
        assert set(saved(checkpoint)["chunks"]) == {"0", "1"}
        before = checkpoint.read_bytes()

        fake = service(tags_reply=tags_reply)
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(fake, timeout=TIMEOUT, chunk_chars=100), LINES, checkpoint)
        assert str(caught.value) == message
        assert fake.prompts == []
        assert tags_requests(fake) == asked
        assert checkpoint.read_bytes() == before

    def test_model_list_failing_once_is_asked_again_and_carries_on(self, service, tmp_path):
        baseline = _run(_proofreader(service(), chunk_chars=100), LINES, tmp_path / "base.json")
        answers = iter([Reply("x", status=500)])
        fake = service(tags_reply=lambda: next(answers, Reply()))
        result = _run(_proofreader(fake, chunk_chars=100), LINES, tmp_path / "c.json")
        assert result.markdown == baseline.markdown
        assert tags_requests(fake) == 2

    def test_paragraph_request_failing_for_now_waits_and_resuming_asks_only_for_paragraphs(
        self, service, tmp_path
    ):
        one = "甲" * 50 + "。"
        two = "乙" * 50 + "。"
        text = "甲" * 50 + "乙" * 50 + "\n"
        checkpoint = tmp_path / "c.json"
        busy = service(lambda p: Reply("x", status=500) if BREAK_MARKER in p else one + two)
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(busy), text, checkpoint)
        assert str(caught.value) == BUSY_MESSAGE
        state = saved(checkpoint)
        assert set(state["chunks"]) == {"0"} and state["chunks"]["0"]["ok"] is True
        assert state["breaks"] == {}

        fixed = service(lambda p: "1" if BREAK_MARKER in p else one + two)
        result = _run(_proofreader(fixed), text, checkpoint)
        assert result.markdown == f"{one}\n\n{two}\n"
        assert polish_prompts(fixed) == []
        assert len(break_prompts(fixed)) == 1

    def test_a_later_paragraph_window_failing_keeps_the_windows_before_it(self, service, tmp_path):
        text = "第一段内容在这里。\n" * 400  # proofread text long enough for two paragraph windows

        def model(break_replies):
            count = itertools.count()

            def respond(prompt):
                if BREAK_MARKER not in prompt:
                    return chunk_of(prompt) + "。"
                index = next(count)
                return break_replies[index] if index < len(break_replies) else "5"

            return respond

        clean = service(model([]))
        baseline = _run(_proofreader(clean), text, tmp_path / "base.json")
        assert len(break_prompts(clean)) == 2  # the case needs two windows

        checkpoint = tmp_path / "c.json"
        busy = service(model(["5", Reply("x", status=500), Reply("x", status=500)]))
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(busy), text, checkpoint)
        assert str(caught.value) == BUSY_MESSAGE
        assert set(saved(checkpoint)["breaks"]) == {"0/0"}

        fixed = service(model([]))
        result = _run(_proofreader(fixed), text, checkpoint)
        assert result.markdown == baseline.markdown
        assert len(break_prompts(fixed)) == 1
        assert polish_prompts(fixed) == []

    def test_waiting_after_only_failed_chunks_keeps_the_checkpoint(self, service, tmp_path):
        checkpoint = tmp_path / "c.json"
        fake = service(three_part_model({"甲": [LOST] * 2, "乙": [SLOW]}))
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(fake, timeout=TIMEOUT, chunk_chars=100), THREE, checkpoint)
        assert str(caught.value) == TIMEOUT_MESSAGE
        chunks = saved(checkpoint)["chunks"]
        assert set(chunks) == {"0"} and chunks["0"]["ok"] is False

        fixed = service(three_part_model())
        result = _run(_proofreader(fixed, chunk_chars=100), THREE, checkpoint)
        assert result.unproofread_chunks == 1
        assert [chunk_of(p)[0] for p in polish_prompts(fixed)] == ["乙", "丙"]


class TestResumingGivesTheSameResult:
    """However a temporary error interrupts, resuming ends with what one clean run gives."""

    PLACES = ["甲", "乙", "丙", "break"]

    @staticmethod
    def _baseline(service, tmp_path) -> str:
        result = _run(_proofreader(service(three_part_model()), chunk_chars=100), THREE, tmp_path / "base.json")
        assert result.unproofread_chunks == 0
        return result.markdown

    @pytest.mark.parametrize("place", PLACES)
    def test_a_timeout_anywhere_waits_then_resumes_to_the_same_result(self, service, tmp_path, place):
        baseline = self._baseline(service, tmp_path)
        checkpoint = tmp_path / "c.json"
        accepted: list[str] = []
        faulty = service(three_part_model({place: [SLOW]}, accepted))
        with pytest.raises(ProofreadBlocked) as caught:
            _run(_proofreader(faulty, timeout=TIMEOUT, chunk_chars=100), THREE, checkpoint)
        assert str(caught.value) == TIMEOUT_MESSAGE

        fixed = service(three_part_model(accepted=accepted))
        result = _run(_proofreader(fixed, chunk_chars=100), THREE, checkpoint)
        assert result.markdown == baseline
        assert result.unproofread_chunks == 0
        assert sorted(accepted) == sorted("甲乙丙")  # each chunk proofread once over both runs
        assert len(polish_prompts(fixed)) == 3 - self.PLACES.index(place)

    @pytest.mark.parametrize("place", PLACES)
    def test_one_service_error_anywhere_does_not_stop_and_gives_the_same_result(self, service, tmp_path, place):
        baseline = self._baseline(service, tmp_path)
        accepted: list[str] = []
        fake = service(three_part_model({place: [Reply("x", status=500)]}, accepted))
        result = _run(_proofreader(fake, chunk_chars=100), THREE, tmp_path / "c.json")
        assert result.markdown == baseline
        assert result.unproofread_chunks == 0
        assert sorted(accepted) == sorted("甲乙丙")

    def test_each_request_gets_its_own_second_chance(self, service, tmp_path):
        baseline = self._baseline(service, tmp_path)
        fake = service(three_part_model({"甲": [Reply("x", status=500)], "丙": [Reply("x", status=500)]}))
        result = _run(_proofreader(fake, chunk_chars=100), THREE, tmp_path / "c.json")
        assert result.markdown == baseline
        assert result.unproofread_chunks == 0


@pytest.mark.integration
def test_real_ollama_proofreads_without_losing_content(tmp_path):
    text = (
        "大家好今天我们来讲数据库索引\n索引其实就是一种帮助查询变快的结构\n"
        "如果你用过 Python 写程序的话应该听说过字典\n字典查找很快原因和索引差不多\n"
        "但是索引也有代价每次写入数据的时候都要更新它\n所以不是建得越多越好\n"
    )
    client = OllamaClient()
    try:
        installed = client.models()
    except OllamaDown:
        pytest.fail("Ollama is not running. Start Ollama, then run this test again.")
    model = DEFAULT_PROOFREAD_MODEL
    if model not in installed:
        pytest.fail(f"Model {model} is missing. Run: ollama pull {model}")
    result = OllamaProofreader(client).proofread(text, model, tmp_path / "c.json", lambda d, t: None)
    client.unload(model)
    assert result.unproofread_chunks == 0
    assert "。" in result.markdown
    assert is_lossless(text, result.markdown)


@pytest.mark.integration
def test_real_ollama_timing_out_waits_and_resumes_to_complete_text(tmp_path):
    """A one-second limit is far too short for a real model: the task must wait, then finish."""
    text = (
        "大家好今天我们来讲数据库索引\n索引其实就是一种帮助查询变快的结构\n"
        "如果你用过 Python 写程序的话应该听说过字典\n字典查找很快原因和索引差不多\n"
        "但是索引也有代价每次写入数据的时候都要更新它\n所以不是建得越多越好\n"
    ) * 4
    checkpoint = tmp_path / "c.json"
    hurried = OllamaClient(timeout=1.0, unload_timeout=15)
    try:
        installed = hurried.models()
    except OllamaDown:
        pytest.fail("Ollama is not running. Start Ollama, then run this test again.")
    model = DEFAULT_PROOFREAD_MODEL
    if model not in installed:
        pytest.fail(f"Model {model} is missing. Run: ollama pull {model}")
    reader = OllamaProofreader(hurried)
    with pytest.raises(ProofreadBlocked) as caught:
        reader.proofread(text, model, checkpoint, lambda d, t: None)
    assert str(caught.value) == TIMEOUT_MESSAGE
    assert saved(checkpoint)["chunks"] == {}  # the chunk that timed out is not recorded

    started = time.perf_counter()
    reader.release(model)
    assert time.perf_counter() - started < 20

    patient = OllamaClient()
    result = OllamaProofreader(patient).proofread(text, model, checkpoint, lambda d, t: None)
    patient.unload(model)
    assert result.unproofread_chunks == 0
    assert is_lossless(text, result.markdown)
