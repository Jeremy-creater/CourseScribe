from __future__ import annotations

import json

import pytest

from coursescribe.files import atomic_write_json, commit_new_text, read_json, unique_dir, unique_path


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

    def broken_replace(src, dst):
        raise OSError("power cut")

    monkeypatch.setattr("coursescribe.files.os.replace", broken_replace)
    with pytest.raises(OSError):
        atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
