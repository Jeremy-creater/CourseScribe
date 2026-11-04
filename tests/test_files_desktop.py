"""Files held open by another program on the real desktop (Windows refuses to share them)."""

from __future__ import annotations

import time
import wave

import numpy as np
import pytest

from coursescribe.files import read_transcript
from coursescribe.notes_rules import RULES_HELD, NotesRules, NotesRulesUnreadable, load_notes_rules
from coursescribe.pipeline import SOURCE_HELD, NoAudio
from coursescribe.platform import FILES_CAN_BE_HELD
from coursescribe.vocabulary import WORD_LIST_HELD, Vocabulary, WordListUnreadable, load_vocabulary
from helpers import hold_without_sharing
from test_recording_desktop import _wait_until_readable

pytestmark = pytest.mark.gui


def test_transcript_read_waits_out_a_short_real_hold_and_reports_a_long_one(tmp_path):
    data = "第一课的内容\n第二句\n".encode()
    transcript = tmp_path / "课.txt"
    transcript.write_bytes(data)

    hold_without_sharing(transcript, 0.3)
    assert read_transcript(transcript) == "第一课的内容\n第二句\n"
    _wait_until_readable(transcript, 5.0)

    hold_without_sharing(transcript, 3.0)
    if FILES_CAN_BE_HELD:
        started = time.monotonic()
        with pytest.raises(PermissionError):
            read_transcript(transcript)
        assert time.monotonic() - started < 2
        _wait_until_readable(transcript, 5.0)
    assert read_transcript(transcript) == "第一课的内容\n第二句\n"
    assert transcript.read_bytes() == data


def test_word_list_read_waits_out_a_short_real_hold_and_reports_a_long_one(tmp_path):
    """M2 item 11 (G1): the stand-in for a held word list matches a real Windows hold."""
    data = "见行 → 践行\r\n".encode("gbk")
    word_list = tmp_path / "词表.txt"
    word_list.write_bytes(data)
    expected = Vocabulary((), (("见行", "践行"),))

    hold_without_sharing(word_list, 0.3)
    assert load_vocabulary(word_list) == expected
    _wait_until_readable(word_list, 5.0)

    hold_without_sharing(word_list, 3.0)
    if FILES_CAN_BE_HELD:
        started = time.monotonic()
        with pytest.raises(WordListUnreadable) as caught:
            load_vocabulary(word_list)
        assert str(caught.value) == WORD_LIST_HELD.format(path=word_list)
        assert time.monotonic() - started < 2
        _wait_until_readable(word_list, 5.0)
    assert load_vocabulary(word_list) == expected
    assert word_list.read_bytes() == data


def test_rules_read_waits_out_a_short_real_hold_and_reports_a_long_one(tmp_path):
    """M2 item 12-4 (G1): the stand-in for held rules matches a real Windows hold."""
    data = "## 核心宗旨\r\n写一句话。\r\n".encode("gbk")
    rules = tmp_path / "整理规则.txt"
    rules.write_bytes(data)
    expected = NotesRules("## 核心宗旨\n写一句话。", None)

    hold_without_sharing(rules, 0.3)
    assert load_notes_rules(rules) == expected
    _wait_until_readable(rules, 5.0)

    hold_without_sharing(rules, 3.0)
    if FILES_CAN_BE_HELD:
        started = time.monotonic()
        with pytest.raises(NotesRulesUnreadable) as caught:
            load_notes_rules(rules)
        assert str(caught.value) == RULES_HELD.format(path=rules)
        assert time.monotonic() - started < 2
        _wait_until_readable(rules, 5.0)
    assert load_notes_rules(rules) == expected
    assert rules.read_bytes() == data


@pytest.mark.engine
@pytest.mark.parametrize("name", ["a.wav", "第一课.wav"], ids=["ascii", "chinese"])
def test_source_file_held_by_another_program_is_called_held_by_the_real_decoder(tmp_path, name):
    from coursescribe.transcribe import WhisperTranscriber

    source = tmp_path / name
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        tone = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 8000).astype(np.int16)
        stream.writeframes(tone.tobytes())
    transcriber = WhisperTranscriber()  # the real media decoder; no model is loaded to read a file

    hold_without_sharing(source, 3.0)
    if FILES_CAN_BE_HELD:
        with pytest.raises(NoAudio) as caught:
            transcriber._read_audio(source)
        assert str(caught.value) == SOURCE_HELD.format(path=source)
        _wait_until_readable(source, 5.0)
    assert len(transcriber._read_audio(source)) > 0
