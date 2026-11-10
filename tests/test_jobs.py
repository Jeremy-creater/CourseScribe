from __future__ import annotations

import itertools
import logging
import random
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pytest

from coursescribe.files import commit_new_text
from coursescribe.jobs import Job, JobKind, JobStore, SaveFolderTooLong, Stage, job_name
from helpers import hold_without_sharing

NOW = datetime(2026, 11, 1, 19, 30)


def _record(root):
    return Job.new_recording(root, NOW, "large-v3", "model-a")


def test_recording_job_lays_out_the_documented_files(tmp_path):
    job = _record(tmp_path)
    name = "2026-11-01 19-30 录音"
    assert job.kind is JobKind.RECORDING
    assert job.stage is Stage.RECORDING
    assert job.folder == tmp_path / name
    assert job.folder.is_dir()
    assert job.audio_path == job.folder / f"{name}.wav"
    assert job.txt_path == job.folder / f"{name}.txt"
    assert job.md_path == job.folder / f"{name}.md"


def test_two_recordings_in_one_minute_get_separate_folders(tmp_path):
    assert _record(tmp_path).folder != _record(tmp_path).folder


def test_file_job_is_named_after_the_source_and_does_not_copy_it(tmp_path):
    source = tmp_path / "input" / "第一课.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    root = tmp_path / "out"
    job = Job.new_file(source, root, NOW, "large-v3", "model-a")
    assert job.kind is JobKind.FILE
    assert job.stage is Stage.TRANSCRIBING
    assert job.audio_path == source
    assert job.folder == root / "第一课"
    assert job.txt_path == root / "第一课" / "第一课.txt"
    assert job.md_path == root / "第一课" / "第一课.md"
    assert list(job.folder.iterdir()) == []


def test_same_file_twice_gets_a_second_folder_and_keeps_the_first(tmp_path):
    source = tmp_path / "a.mp3"
    source.write_bytes(b"x")
    first = Job.new_file(source, tmp_path / "out", NOW, "m", "p")
    second = Job.new_file(source, tmp_path / "out", NOW, "m", "p")
    assert first.folder.name == "a"
    assert second.folder.name == "a (2)"
    assert second.txt_path.parent == second.folder


def test_store_round_trips_every_field(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _record(tmp_path)
    job.stage = Stage.WAITING_PROOFREAD
    job.message = "校对服务没有启动"
    job.unproofread_chunks = 2
    job.total_chunks = 20
    store.save(job)
    assert JobStore(tmp_path / "state").load_all() == [job]


def test_unfinished_excludes_done_and_failed_and_keeps_creation_order(tmp_path):
    store = JobStore(tmp_path / "state")
    jobs = [_record(tmp_path) for _ in range(5)]
    stages = [Stage.TRANSCRIBING, Stage.DONE, Stage.PROOFREADING, Stage.FAILED, Stage.WAITING_PROOFREAD]
    for job, stage in reversed(list(zip(jobs, stages))):
        job.stage = stage
        store.save(job)
    assert [j.job_id for j in store.unfinished()] == [jobs[0].job_id, jobs[2].job_id, jobs[4].job_id]


def test_a_damaged_state_file_does_not_stop_the_other_jobs(tmp_path):
    store = JobStore(tmp_path / "state")
    good = _record(tmp_path)
    store.save(good)
    (tmp_path / "state" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "state" / "wrong.json").write_text('{"stage": "nonsense"}', encoding="utf-8")
    assert store.load_all() == [good]


def test_a_state_file_held_open_by_another_program_is_still_listed(tmp_path):
    store = JobStore(tmp_path / "state")
    job = _record(tmp_path)
    store.save(job)
    hold_without_sharing(store.folder / f"{job.job_id}.json", 0.2)
    assert store.load_all() == [job]


def test_store_on_a_missing_folder_is_empty(tmp_path):
    assert JobStore(tmp_path / "nothing").load_all() == []


# --- long names (docs/plans/M2-long-names-tests.md) ---

OUT = Path("C:/out")
_ALPHABET = ["a", "讲", "😀", "e", "\u0301", "\u200d", "\ufe0f", "🏽", " ", "."]
_EXTENDERS = {"\u200d", "\ufe0e", "\ufe0f"}


def _u16(text):
    """Length as Windows counts it: most characters 1, emoji 2."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _fits(root):
    """How many characters a long name keeps in ``root``: 80, less when the save folder path is long (N7)."""
    return min(80, (243 - _u16(str(root))) // 2)


def _extends(char):
    return unicodedata.combining(char) > 0 or char in _EXTENDERS or "\U0001f3fb" <= char <= "\U0001f3ff"


def _root(length, base=Path("C:/")):
    """A save folder inside ``base`` whose path is ``length`` units long."""
    probe = base / "r"
    root = base / ("r" * (length - _u16(str(probe)) + 1))
    assert _u16(str(root)) == length
    return root


@pytest.mark.parametrize("raw", ["第一课", "a" * 79, "a" * 80, "讲" * 80, " 开头有空格"])
def test_name_that_fits_is_kept_as_is(raw):
    assert job_name(raw, OUT) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a" * 81, "a" * 80),
        ("讲" * 81, "讲" * 80),
        ("讲" * 300, "讲" * 80),
        ("第" + "x" * 79 + "尾巴", "第" + "x" * 79),
    ],
)
def test_long_name_is_cut_to_80(raw, expected):
    assert job_name(raw, OUT) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a" * 79 + "😀", "a" * 79),
        ("a" * 78 + "😀", "a" * 78 + "😀"),
        ("a" * 78 + "😀" + "b", "a" * 78 + "😀"),
    ],
)
def test_length_is_counted_the_windows_way_and_emoji_are_never_split(raw, expected):
    assert job_name(raw, OUT) == expected


def test_name_with_an_unpaired_surrogate_is_cut_without_error():
    raw = "a" * 40 + "\udc80" + "b" * 44  # how a malformed Windows file name reaches Python
    result = job_name(raw, OUT)
    assert _u16(result) <= 80
    assert raw.startswith(result)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a" * 79 + "e\u0301", "a" * 79),
        ("a" * 78 + "👍🏽", "a" * 78),
        ("a" * 77 + "👨\u200d👩\u200d👧", "a" * 77),
        ("a" * 79 + "❤\ufe0f", "a" * 79),
        ("a" * 79 + "✌\ufe0e", "a" * 79),
    ],
)
def test_cut_never_splits_a_character_from_its_marks(raw, expected):
    result = job_name(raw, OUT)
    assert result == expected
    assert not result.endswith("\u200d")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a" * 79 + " 后面", "a" * 79),
        ("a" * 78 + ". " + "x" * 10, "a" * 78),
        ("第一课 ", "第一课"),
        ("第一课.", "第一课"),
        ("第 1.5 讲", "第 1.5 讲"),
    ],
)
def test_trailing_spaces_and_dots_are_removed(raw, expected):
    assert job_name(raw, OUT) == expected


@pytest.mark.parametrize("raw", ["", "   ", "..", " . . "])
def test_name_left_empty_becomes_untitled(raw):
    assert job_name(raw, OUT) == "未命名"


@pytest.mark.parametrize(("root_length", "expected"), [(83, 80), (84, 79), (85, 79), (120, 61), (203, 20)])
def test_name_limit_shrinks_with_a_long_save_folder(root_length, expected):
    assert len(job_name("讲" * 100, _root(root_length))) == expected


@pytest.mark.parametrize("raw", ["讲" * 100, "第一课"])
def test_save_folder_too_long_for_any_name_is_refused(raw):
    with pytest.raises(SaveFolderTooLong):
        job_name(raw, _root(204))


def _longest_fitting_prefix(raw, limit):
    kept = ""
    for char in raw:
        if _u16(kept + char) > limit:
            break
        kept += char
    return kept


def _check_name(raw, root, limit):
    result = job_name(raw, root)
    why = f"raw={raw!r} result={result!r}"
    cut = _u16(raw) > limit
    assert result and _u16(result) <= limit, why
    assert raw.startswith(result) or result == "未命名", why
    if cut and result != "未命名":
        assert not result.endswith("\u200d"), why
        assert not _extends(raw[len(result)]), why
    assert not result.endswith((" ", ".")), why
    assert job_name(result, root) == result, why
    if raw and _u16(raw) <= limit and not raw.endswith((" ", ".")):
        assert result == raw, why
    if raw and set(raw) <= {"a", "讲", "😀"}:
        assert result == _longest_fitting_prefix(raw, limit), why


def test_name_rules_hold_for_every_short_tail():
    started = time.perf_counter()
    tails = [""]
    for length in range(1, 4):
        tails += ["".join(chars) for chars in itertools.product(_ALPHABET, repeat=length)]
    assert len(tails) == 1111
    for root_length, limit in ((203, 20), (201, 21)):
        root = _root(root_length)
        for p in [0, *range(limit - 7, limit + 1)]:
            for tail in tails:
                _check_name("a" * p + tail, root, limit)
        for raw in ["." * 30, " " * 30, "\u0301" * 30, "e" + "\u0301" * 30, "a" * 10 + "\udc80" + "b" * 20]:
            _check_name(raw, root, limit)
    print(f"name rules checked in {time.perf_counter() - started:.2f} s")


def _source(tmp_path, name, data=b"video"):
    source = tmp_path / "in" / name
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(data)
    return source


def test_long_file_name_job_uses_the_cut_name_and_leaves_the_source_alone(tmp_path):
    source = _source(tmp_path, "讲" * 100 + ".mp4")
    out = tmp_path / "out"
    job = Job.new_file(source, out, NOW, "large-v3", "model-a")
    name = "讲" * _fits(out)
    assert job.folder == out / name
    assert job.txt_path == job.folder / f"{name}.txt"
    assert job.md_path == job.folder / f"{name}.md"
    assert job.audio_path == source
    assert job.folder.is_dir() and list(job.folder.iterdir()) == []
    assert list(source.parent.iterdir()) == [source]
    assert source.read_bytes() == b"video"


def test_two_long_names_alike_at_the_start_get_separate_folders(tmp_path):
    upper = _source(tmp_path, "讲" * 80 + "上.mp3")
    lower = _source(tmp_path, "讲" * 80 + "下.mp3")
    first = Job.new_file(upper, tmp_path / "out", NOW, "m", "p")
    second = Job.new_file(lower, tmp_path / "out", NOW, "m", "p")
    name = "讲" * _fits(tmp_path / "out")
    assert first.folder.name == name
    assert second.folder.name == name + " (2)"
    assert first.txt_path.name == second.txt_path.name == name + ".txt"
    assert first.txt_path.parent != second.txt_path.parent
    assert (first.audio_path, second.audio_path) == (upper, lower)
    assert first.folder.is_dir() and second.folder.is_dir()


@pytest.mark.parametrize("kind", ["file", "recording"])
def test_save_folder_too_long_creates_nothing(tmp_path, kind):
    source = _source(tmp_path, "a.mp3")
    root = _root(204, tmp_path)
    with pytest.raises(SaveFolderTooLong) as caught:
        if kind == "file":
            Job.new_file(source, root, NOW, "m", "p")
        else:
            Job.new_recording(root, NOW, "m", "p")
    assert str(root) in str(caught.value)
    assert isinstance(caught.value, OSError) is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in"]
    assert list(source.parent.iterdir()) == [source]


@pytest.mark.parametrize("letter", ["讲", "😀"])
def test_every_accepted_save_folder_leaves_room_for_the_longest_path(letter):
    for length in range(3, 204):
        root = Path("C:/" + "r" * (length - 3))
        assert _u16(str(root)) == length
        name = job_name(letter * 100, root)
        assert length + 2 * _u16(name) + 16 <= 259, length


def test_longest_path_fits(tmp_path):
    """The worst case: 99th folder, 99th transcript. Written for real, at the length limit."""
    length = max(83, _u16(str(tmp_path)) + 2)
    root = _root(length, tmp_path)
    source = _source(tmp_path, "讲" * 100 + ".mp3")
    name = job_name(source.stem, root)
    n = len(name)
    assert 258 <= length + 2 * n + 16 <= 259  # the name leaves room for " (99)" twice and ".txt"
    (root / name).mkdir(parents=True)
    for number in range(2, 99):
        (root / f"{name} ({number})").mkdir()
    job = Job.new_file(source, root, NOW, "m", "p")
    assert job.folder.name == f"{name} (99)"
    (job.folder / f"{name}.txt").write_text("其他 1", encoding="utf-8")
    for number in range(2, 99):
        (job.folder / f"{name} ({number}).txt").write_text(f"其他 {number}", encoding="utf-8")
    written = commit_new_text(job.txt_path, "内容")
    assert written.name == f"{name} (99).txt"
    assert _u16(str(written)) == length + 2 * n + 16
    assert 258 <= _u16(str(written)) <= 259
    if length == 83:
        assert _u16(str(written)) == 259
    assert written.read_text(encoding="utf-8") == "内容"
    assert not [p for p in job.folder.iterdir() if p.suffix == ".tmp"]


def _folder_contents(folder: Path) -> dict[str, tuple[bytes, int]]:
    if not folder.is_dir():
        return {}
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}


def test_loading_one_job_agrees_with_loading_all_and_logs_a_damaged_record_once(tmp_path, caplog):
    """J1 (12-7): load(id) and load_all() agree on any folder; a damaged record is one warning per read."""
    from test_notes_job import _random_job

    def warnings() -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "coursescribe.jobs"]

    rng = random.Random(20270404)
    damaged_rounds = 0
    with caplog.at_level(logging.WARNING, logger="coursescribe.jobs"):
        for round_ in range(30):
            root = tmp_path / f"r{round_}"
            store = JobStore(root / "state")
            ids = []
            for index in range(rng.randint(0, 5)):
                job = _random_job(rng, root, index)
                store.save(job)
                ids.append(job.job_id)
                for suffix in (".notes.json", ".proofread.json"):
                    if rng.random() < 0.5:
                        (store.folder / f"{job.job_id}{suffix}").write_text('{"reply": "x"}', encoding="utf-8")
            broken = None
            if ids and rng.random() < 0.5:
                broken = rng.choice(ids)
                (store.folder / f"{broken}.json").write_bytes(b"{")
                damaged_rounds += 1
            before = _folder_contents(store.folder)

            caplog.clear()
            everything = {job.job_id: job for job in store.load_all()}
            caplog.clear()
            one_by_one = {}
            for job_id in ids:
                count = len(warnings())
                loaded = store.load(job_id)
                added = warnings()[count:]
                if job_id == broken:
                    assert loaded is None, round_
                    assert len(added) == 1 and f"{job_id}.json" in added[0].getMessage(), round_
                else:
                    assert loaded is not None, round_
                    assert added == [], round_
                    one_by_one[job_id] = loaded
            assert one_by_one == everything, round_
            caplog.clear()
            assert store.load("不存在") is None
            assert warnings() == []
            assert _folder_contents(store.folder) == before, round_
    assert damaged_rounds >= 5

    missing = JobStore(tmp_path / "nowhere" / "state")
    with caplog.at_level(logging.WARNING, logger="coursescribe.jobs"):
        caplog.clear()
        assert missing.load("abc") is None
        assert warnings() == []
    assert not (tmp_path / "nowhere").exists()
