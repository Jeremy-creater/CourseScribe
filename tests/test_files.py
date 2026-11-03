from __future__ import annotations

import json

import pytest

from coursescribe.files import (
    UnknownEncoding,
    atomic_write_json,
    commit_new_text,
    read_json,
    read_transcript,
    unique_dir,
    unique_path,
)
from test_recording import _held


def test_unique_path_returns_the_name_when_free(tmp_path):
    assert unique_path(tmp_path / "a.txt") == tmp_path / "a.txt"


def test_unique_path_adds_the_first_free_number(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a (2).txt").write_text("x", encoding="utf-8")
    assert unique_path(tmp_path / "a.txt") == tmp_path / "a (3).txt"


def test_unique_dir_creates_a_new_folder_next_to_an_existing_one(tmp_path):
    first = unique_dir(tmp_path / "root" / "lesson")
    second = unique_dir(tmp_path / "root" / "lesson")
    assert first.name == "lesson" and first.is_dir()
    assert second.name == "lesson (2)" and second.is_dir()


def test_commit_writes_utf8_text_without_translating_line_ends(tmp_path):
    target = tmp_path / "a.txt"
    assert commit_new_text(target, "第一行\n第二行\n") == target
    assert target.read_bytes() == "第一行\n第二行\n".encode("utf-8")


def test_commit_never_overwrites_a_different_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("user's own file", encoding="utf-8")
    written = commit_new_text(target, "new")
    assert target.read_text(encoding="utf-8") == "user's own file"
    assert written == tmp_path / "a (2).txt"
    assert written.read_text(encoding="utf-8") == "new"


def test_commit_of_identical_content_is_recognised_as_already_done(tmp_path):
    target = tmp_path / "a.txt"
    first = commit_new_text(target, "same")
    assert commit_new_text(target, "same") == first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]


def test_commit_leaves_no_temporary_files(tmp_path):
    commit_new_text(tmp_path / "a.txt", "x")
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_commit_failure_leaves_no_partial_target(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"

    def broken_fsync(fd):
        raise OSError("disk full")

    monkeypatch.setattr("coursescribe.files.os.fsync", broken_fsync)
    with pytest.raises(OSError):
        commit_new_text(target, "x")
    assert list(tmp_path.iterdir()) == []


def test_commit_works_on_drives_without_hard_links(tmp_path, monkeypatch):
    def no_links(src, dst):
        raise OSError("hard links not supported")

    monkeypatch.setattr("coursescribe.files.os.link", no_links)
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    written = commit_new_text(target, "new")
    assert written == tmp_path / "a (2).txt"
    assert written.read_text(encoding="utf-8") == "new"
    assert target.read_text(encoding="utf-8") == "old"


def test_atomic_json_round_trip_and_replace(tmp_path):
    target = tmp_path / "deep" / "state.json"
    atomic_write_json(target, {"a": "中"})
    atomic_write_json(target, {"a": "文"})
    assert read_json(target) == {"a": "文"}
    assert [p.name for p in target.parent.iterdir()] == ["state.json"]


def test_interrupted_json_write_keeps_the_old_content(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"v": 1})

    calls = []

    def broken_replace(src, dst):
        calls.append(dst)
        raise OSError("power cut")

    monkeypatch.setattr("coursescribe.files.os.replace", broken_replace)
    with pytest.raises(OSError):
        atomic_write_json(target, {"v": 2})
    assert len(calls) == 1  # only a file held open by someone else is worth waiting for
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def _held_open():
    """What Windows reports while another program has the file open."""
    return PermissionError(13, "Permission denied")


@pytest.fixture
def files_can_be_held(monkeypatch):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", True)


def test_json_save_waits_out_a_reader_that_briefly_holds_the_file(tmp_path, monkeypatch, files_can_be_held):
    """Windows refuses to replace a file another program has open, such as antivirus."""
    import os

    target = tmp_path / "job.json"
    atomic_write_json(target, {"a": 1})
    real_replace = os.replace
    refusals = []

    def busy_then_free(src, dst):
        if len(refusals) < 2:
            refusals.append(dst)
            raise _held_open()
        real_replace(src, dst)

    monkeypatch.setattr("coursescribe.files.os.replace", busy_then_free)
    atomic_write_json(target, {"a": 2})
    assert read_json(target) == {"a": 2}
    assert len(refusals) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["job.json"]


def test_json_save_gives_up_on_a_file_that_stays_locked(tmp_path, monkeypatch, files_can_be_held):
    target = tmp_path / "job.json"
    atomic_write_json(target, {"a": 1})

    def always_busy(src, dst):
        raise _held_open()

    monkeypatch.setattr("coursescribe.files.os.replace", always_busy)
    monkeypatch.setattr("coursescribe.files._REPLACE_WAIT", 0.001)
    with pytest.raises(PermissionError):
        atomic_write_json(target, {"a": 2})
    assert read_json(target) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["job.json"]


def test_json_save_does_not_wait_where_files_cannot_be_held(tmp_path, monkeypatch):
    """On a Mac a refusal is a read-only folder or a privacy block, which will not clear by waiting."""
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", False)
    target = tmp_path / "job.json"
    calls = []

    def denied(src, dst):
        calls.append(dst)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("coursescribe.files.os.replace", denied)
    with pytest.raises(PermissionError):
        atomic_write_json(target, {"a": 2})
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def _briefly_held(monkeypatch, times):
    from pathlib import Path

    real_read = Path.read_text
    refusals = []

    def read_text(self, *args, **kwargs):
        if len(refusals) < times:
            refusals.append(self)
            raise _held_open()
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr("coursescribe.files._REPLACE_WAIT", 0.001)
    return refusals


def test_json_read_waits_out_a_brief_hold(tmp_path, monkeypatch, files_can_be_held):
    target = tmp_path / "job.json"
    atomic_write_json(target, {"a": 1})
    refusals = _briefly_held(monkeypatch, 2)
    assert read_json(target) == {"a": 1}
    assert len(refusals) == 2


def test_json_read_gives_up_on_a_file_that_stays_held(tmp_path, monkeypatch, files_can_be_held):
    target = tmp_path / "job.json"
    atomic_write_json(target, {"a": 1})
    _briefly_held(monkeypatch, 1000)
    with pytest.raises(PermissionError):
        read_json(target)


# --- reading back a transcript the user may have moved, held or re-saved (docs/plans/M2-moved-transcript-tests.md) ---

TRANSCRIPT_TEXTS = [
    "第一句，讲 Python 的 list。\n第二句 😀\n",
    "a",
    "\r\n中文\r\n",
    "  首尾有空白  \n",
    "中" * 200_000,
]
TRANSCRIPT_ENCODINGS = {
    "utf-8": lambda text: text.encode("utf-8"),
    "utf-8 with BOM": lambda text: text.encode("utf-8-sig"),
    "utf-16 default": lambda text: text.encode("utf-16"),
    "utf-16 little-endian with BOM": lambda text: b"\xff\xfe" + text.encode("utf-16-le"),
    "utf-16 big-endian with BOM": lambda text: b"\xfe\xff" + text.encode("utf-16-be"),
}


def _transcript_file(tmp_path, data):
    path = tmp_path / "课.txt"
    path.write_bytes(data)
    return path, path.stat().st_mtime_ns


@pytest.mark.parametrize("encoding", TRANSCRIPT_ENCODINGS)
@pytest.mark.parametrize("text", TRANSCRIPT_TEXTS, ids=["mixed", "one-char", "crlf", "spaces", "long"])
def test_transcript_in_any_accepted_encoding_reads_back_exactly_and_is_left_alone(tmp_path, text, encoding):
    data = TRANSCRIPT_ENCODINGS[encoding](text)
    path, mtime = _transcript_file(tmp_path, data)
    assert read_transcript(path) == text
    assert path.read_bytes() == data
    assert path.stat().st_mtime_ns == mtime
    assert [p.name for p in tmp_path.iterdir()] == ["课.txt"]


@pytest.mark.parametrize(
    "data",
    [
        "第一句，讲 Python。".encode("gbk"),
        "第一句".encode("utf-16-le"),
        "abc".encode("utf-16-le"),
        "café".encode("latin-1"),
        b"\xef\xbb\xbf" + "中".encode("gbk"),
        b"\xff\xfe" + b"\x00",
    ],
    ids=["gbk", "utf-16-no-bom", "ascii-utf-16-no-bom", "latin-1", "utf-8-bom-over-gbk", "utf-16-bom-odd-bytes"],
)
def test_transcript_in_an_encoding_that_cannot_be_told_for_sure_is_refused(tmp_path, data):
    path, mtime = _transcript_file(tmp_path, data)
    with pytest.raises(UnknownEncoding) as caught:
        read_transcript(path)
    assert not isinstance(caught.value, OSError)
    assert path.read_bytes() == data
    assert path.stat().st_mtime_ns == mtime


@pytest.mark.parametrize(
    ("data", "text"),
    [
        ("Hello, world\n".encode("gbk"), "Hello, world\n"),
        (b"", ""),
        (b"\xef\xbb\xbf", ""),
    ],
    ids=["plain-ascii", "empty", "bom-only"],
)
def test_transcript_whose_encoding_is_certain_is_read_even_at_the_edges(tmp_path, data, text):
    path, _ = _transcript_file(tmp_path, data)
    assert read_transcript(path) == text
    assert path.read_bytes() == data


def test_transcript_that_is_gone_is_reported_and_not_made_up(tmp_path):
    gone = tmp_path / "gone.txt"
    with pytest.raises(FileNotFoundError):
        read_transcript(gone)
    assert not gone.exists()


def test_transcript_held_briefly_is_waited_for(tmp_path, monkeypatch, files_can_be_held):
    data = "第一课的内容\n".encode()
    path, _ = _transcript_file(tmp_path, data)
    held = _held(monkeypatch, path, times=3)
    assert read_transcript(path) == "第一课的内容\n"
    assert held.attempts == 4
    assert path.read_bytes() == data


def test_transcript_held_for_long_is_reported_after_about_a_second(tmp_path, monkeypatch, files_can_be_held):
    data = "第一课的内容\n".encode()
    path, _ = _transcript_file(tmp_path, data)
    held = _held(monkeypatch, path)
    with pytest.raises(PermissionError):
        read_transcript(path)
    assert held.attempts == 20
    held.release()
    assert path.read_bytes() == data


def test_transcript_refused_where_files_cannot_be_held_is_reported_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr("coursescribe.files._FILES_CAN_BE_HELD", False)
    data = "第一课的内容\n".encode()
    path, _ = _transcript_file(tmp_path, data)
    held = _held(monkeypatch, path)
    with pytest.raises(PermissionError):
        read_transcript(path)
    assert held.attempts == 1
    held.release()
    assert path.read_bytes() == data


def test_transcript_on_an_unplugged_drive_is_reported_at_once(tmp_path, monkeypatch, files_can_be_held):
    data = "第一课的内容\n".encode()
    path, _ = _transcript_file(tmp_path, data)
    held = _held(monkeypatch, path, error=OSError(5, "Input/output error"))
    with pytest.raises(OSError) as caught:
        read_transcript(path)
    assert caught.value.errno == 5 and not isinstance(caught.value, PermissionError)
    assert held.attempts == 1  # not a hold, so no waiting
    held.release()
    assert path.read_bytes() == data


@pytest.mark.parametrize(
    ("data", "expected"),
    [(b"a\x0cb\n", "a\x0cb\n"), (b"\xd1\xa7\xcf\xb0\n", "ѧϰ\n")],
    ids=["control-character-kept", "gbk-that-is-valid-utf-8-not-reread"],
)
def test_transcripts_keep_their_own_rules_beside_the_word_list(tmp_path, data, expected):
    """M2 item 11 (T0): the word list's stricter decoding does not reach transcripts."""
    path = tmp_path / "t.txt"
    path.write_bytes(data)
    assert read_transcript(path) == expected
    assert path.read_bytes() == data
