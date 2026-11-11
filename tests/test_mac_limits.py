"""Name and path length limits of the Mac: 255 bytes per name, 1023 bytes per path, a Chinese character is 3 bytes.

The Mac rules run here, on any system, by handing ``jobs`` the Mac's limits; Windows' own rules stay in
test_jobs.py, test_denied_folder.py and test_notes_job.py.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import pytest

from coursescribe import jobs
from coursescribe.jobs import NOTES_SUFFIX, Job, SaveFolderTooLong, job_name, notes_file_for, path_fits
from coursescribe.platform.limits import PathLimits

MAC = PathLimits(path_units=None, component_bytes=255, path_bytes=1023)
WINDOWS = PathLimits(path_units=259, component_bytes=None, path_bytes=None)
NOW = datetime(2026, 11, 1, 19, 30)
# The longest file name a job name can end up in: <name>-整理笔记 (99).md
LONGEST_TAIL = f"{NOTES_SUFFIX} (99).md"


def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(jobs, "LIMITS", MAC)


def test_the_platforms_report_their_own_limits():
    from coursescribe.platform import macos, windows

    assert macos.LIMITS == MAC
    assert windows.LIMITS == WINDOWS


def test_a_long_chinese_name_is_cut_so_the_longest_notes_file_fits_in_255_bytes(mac):
    root = Path("/Users/a/Documents/CourseScribe")
    raw = "讲" * 100
    name = job_name(raw, root)
    assert name == "讲" * 78  # 78 * 3 + the 21 bytes of "-整理笔记 (99).md" = 255
    assert _bytes(name + LONGEST_TAIL) == 255
    assert _bytes(("讲" * 79) + LONGEST_TAIL) == 258  # one more would not fit


def test_names_that_already_fit_are_left_alone_on_the_mac(mac):
    root = Path("/Users/a/Documents/CourseScribe")
    assert job_name("索引" * 30, root) == "索引" * 30  # 60 characters, 180 bytes
    assert job_name("x" * 200, root) == "x" * 80  # the 80-character rule still holds
    assert job_name("😀" * 100, root) == "😀" * 40  # 80 UTF-16 units: 40 emoji, 160 bytes


def test_a_mac_name_is_never_cut_inside_a_character(mac):
    root = Path("/Users/a/Documents/CourseScribe")
    family = "👨‍👩‍👧"  # one symbol, 18 bytes
    assert job_name("讲" * 76 + family + "尾", root) == "讲" * 76  # 6 bytes are left: no half symbol
    assert job_name("讲" * 77 + "é" * 10, root) == "讲" * 77 + "é"  # 3 bytes left: one whole "é"


def test_a_mixed_name_on_the_mac_always_leaves_room_for_every_file_it_makes(mac):
    rng = random.Random(20261112)
    pieces = ["讲", "索", "a", "Z", " ", "-", "😀", "é", "e\u0301", "👨\u200d👩\u200d👧", "１", "。"]
    cut = 0
    for _ in range(400):
        raw = "".join(rng.choice(pieces) for _ in range(rng.randrange(1, 150)))
        root = Path("/Users/" + "u" * rng.randrange(1, 300))
        name = job_name(raw, root)
        for tail in (LONGEST_TAIL, ".txt", ".md", ".wav", " (99)"):
            assert _bytes(name + tail) <= 255, (raw, name)
        assert _bytes(str(root)) + 1 + _bytes(name) + 1 + _bytes(name + LONGEST_TAIL) <= 1023, (raw, name)
        assert raw.startswith(name.rstrip(" .")) or name == "未命名", (raw, name)
        cut += name != raw
    assert cut >= 100, cut


def test_a_save_folder_that_leaves_no_room_for_a_name_is_refused_on_the_mac(mac):
    # Room for the root and two names: 1023 - 2 separators - 21 tail bytes = 1000. Names need at least 60 bytes each.
    assert job_name("讲" * 100, Path("/" + "r" * 879)) == "讲" * 20  # a root of 880 bytes leaves exactly 60
    with pytest.raises(SaveFolderTooLong):
        job_name("讲" * 100, Path("/" + "r" * 880))  # 881 bytes leaves 59


def test_paths_on_the_mac_are_judged_by_bytes_per_name_and_in_total(mac):
    assert path_fits(Path("/a/" + "x" * 255))
    assert not path_fits(Path("/a/" + "x" * 256))
    assert path_fits(Path("/a/" + "讲" * 85))  # 255 bytes
    assert not path_fits(Path("/a/" + "讲" * 85 + "x"))
    deep = "/" + "/".join(["x" * 200] * 5)  # 1005 bytes and no name over 255
    assert path_fits(Path(deep + "/" + "y" * 17))  # 1023
    assert not path_fits(Path(deep + "/" + "y" * 18))  # 1024


def test_the_windows_style_path_limit_does_not_apply_on_the_mac(mac):
    # 300 ASCII characters in short names: too long for Windows' 259, fine for the Mac
    path = Path("/" + "/".join(["abcdefghi"] * 30))
    assert len(str(path)) > 259
    assert path_fits(path)


def test_windows_keeps_judging_by_the_whole_path_only(monkeypatch):
    monkeypatch.setattr(jobs, "LIMITS", WINDOWS)
    assert path_fits(Path("x" * 259))
    assert not path_fits(Path("x" * 260))
    assert path_fits(Path("C:/a/" + "讲" * 120))  # 360 bytes in one name is fine there


def test_a_dropped_proofread_with_a_name_too_long_for_notes_is_not_made_into_notes_on_the_mac(mac):
    short = Path("/a/" + "讲" * 78 + ".md")  # notes name: 234 + 13 + 3 = 250 bytes
    long = Path("/a/" + "讲" * 80 + ".md")  # notes name: 240 + 13 + 3 = 256 bytes
    assert path_fits(notes_file_for(short))
    assert not path_fits(notes_file_for(long))
    numbered = notes_file_for(short).with_name(short.stem + f"{NOTES_SUFFIX} (99).md")
    assert path_fits(numbered)  # 255 bytes exactly


def test_a_real_job_for_a_long_chinese_file_gets_a_name_the_mac_can_store(mac, tmp_path):
    source = tmp_path / "in" / (("讲" * 100) + ".mp3")
    source.parent.mkdir()
    source.write_bytes(b"x")
    root = tmp_path / "out"
    job = Job.new_file(source, root, NOW, "m", "p")
    assert job.folder.name == "讲" * 78
    assert _bytes(notes_file_for(job.md_path).name.replace(".md", " (99).md")) <= 255
    assert job.folder.is_dir()
