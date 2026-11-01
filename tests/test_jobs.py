from __future__ import annotations

from datetime import datetime

from coursescribe.jobs import Job, JobKind, JobStore, Stage

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


def test_store_on_a_missing_folder_is_empty(tmp_path):
    assert JobStore(tmp_path / "nothing").load_all() == []
