"""Everything real together: generated speech, the real model, real Ollama."""

from __future__ import annotations

import threading
import time

import pytest

from coursescribe.controller import AppController
from coursescribe.jobs import JobStore, Stage
from coursescribe.ollama import OllamaClient, OllamaDown
from coursescribe.platform import create_keep_awake, open_system_audio, prepare_gpu_libraries
from coursescribe.proofread import OllamaProofreader, is_lossless
from coursescribe.settings import Settings, SettingsStore
from coursescribe.text import clean_fillers
from coursescribe.transcribe import WhisperTranscriber, download_whisper_model
from speech import speak_to_wav

pytestmark = pytest.mark.integration

LESSON = (
    "大家好，今天我们来学习数据库索引。索引可以让查询变快，但是写入数据的时候会慢一点。"
    "比如我们用 Python 写一个程序，查询一百万条数据，没有索引的时候要等很久。"
    "所以索引很有用，但也不是越多越好。这节课就讲到这里，谢谢大家。"
)


def _controller(tmp_path):
    try:
        OllamaClient().models()
    except OllamaDown:
        pytest.fail("Ollama is not running. Start Ollama, then run this test again.")
    prepare_gpu_libraries()
    stopping = threading.Event()
    return AppController(
        store=JobStore(tmp_path / "state"),
        settings=SettingsStore(tmp_path / "s.json", Settings(tmp_path / "out", "large-v3", "gemma4:12b")),
        transcriber=WhisperTranscriber(should_stop=stopping.is_set),
        proofreader=OllamaProofreader(),
        open_capture=open_system_audio,
        download_model=download_whisper_model,
        stopping=stopping,
        keep_awake=create_keep_awake(),
    )


def test_local_file_becomes_txt_and_proofread_md(tmp_path):
    source = speak_to_wav(LESSON, tmp_path / "lesson.wav")
    before = source.read_bytes()
    controller = _controller(tmp_path)
    controller.start()
    controller.add_files([source])
    deadline = time.time() + 300
    while controller.state.pending and time.time() < deadline:
        time.sleep(0.2)
    controller.shutdown()

    job = controller.store.load_all()[0]
    assert job.stage is Stage.DONE, job.message
    assert source.read_bytes() == before
    raw = job.txt_path.read_text(encoding="utf-8")
    proofread = job.md_path.read_text(encoding="utf-8")
    assert job.unproofread_chunks == 0
    assert any(mark in proofread for mark in "。，")
    heading, body = proofread.split("\n\n", 1)
    assert heading == f"# {job.md_path.stem}"
    assert is_lossless(clean_fillers(raw), body)
    assert "索引" in body and "#" not in body
    print("\nTXT:", raw, "\nMD:", proofread)


def test_with_auto_notes_on_a_real_lesson_goes_on_to_its_notes_one_model_at_a_time(tmp_path, monkeypatch):
    """12-7 I1: the program's own assembly, from the settings file to the notes, on the real models."""
    import json
    from datetime import date

    from coursescribe.jobs import JobKind
    from coursescribe.models import DEFAULT_PROOFREAD_MODEL, DEFAULT_TRANSCRIBE_MODEL
    from coursescribe.notes import NotesMeta, lecture_date
    from coursescribe.ui.app import build_controller
    from test_notes import header
    from test_notes_job import _loaded_models
    from test_proofread import _ready_notes_model

    client = OllamaClient()
    _ready_notes_model(client)
    if DEFAULT_PROOFREAD_MODEL not in client.models():
        pytest.fail(f"Model {DEFAULT_PROOFREAD_MODEL} is missing. Run: ollama pull {DEFAULT_PROOFREAD_MODEL}")
    prepare_gpu_libraries()
    source = speak_to_wav(LESSON, tmp_path / "lesson.wav")
    home = tmp_path / "家"
    home.mkdir()
    monkeypatch.setenv("COURSESCRIBE_HOME", str(home))
    (home / "settings.json").write_text(
        json.dumps({"format": 2, "output_dir": str(tmp_path / "out"), "transcribe_model": DEFAULT_TRANSCRIBE_MODEL,
                    "proofread_model": DEFAULT_PROOFREAD_MODEL, "auto_notes": True}),
        encoding="utf-8",
    )
    stopping = threading.Event()
    controller = build_controller(home, stopping)
    loaded: list[list[str]] = []
    watching = threading.Event()
    watching.set()

    def watch():
        while watching.is_set():
            loaded.append(_loaded_models())
            time.sleep(0.5)

    kept: dict = {}

    def keep(state):
        if not kept and state.status == "完成":
            job = next(j for j in controller.store.load_all() if j.kind is not JobKind.NOTES)
            kept.update(txt=job.txt_path.read_bytes(), md=job.md_path.read_bytes())

    controller.add_listener(keep)
    watcher = threading.Thread(target=watch, daemon=True)
    started = time.time()
    try:
        watcher.start()
        controller.start()
        controller.add_files([source])
        deadline = time.time() + 900
        while time.time() < deadline:
            jobs = controller.store.load_all()
            if len(jobs) == 2 and all(j.stage in (Stage.DONE, Stage.FAILED, Stage.WAITING_NOTES) for j in jobs) \
                    and controller.state.pending == 0:
                break
            time.sleep(0.5)
    finally:
        watching.clear()
        watcher.join(15)
        controller.shutdown()
    took = time.time() - started

    jobs = controller.store.load_all()
    lesson_job = next(j for j in jobs if j.kind is not JobKind.NOTES)
    notes_job = next(j for j in jobs if j.kind is JobKind.NOTES)
    assert lesson_job.stage is Stage.DONE, lesson_job.message
    assert notes_job.stage is Stage.DONE, notes_job.message
    md = lesson_job.md_path
    notes = md.with_name(f"{md.stem}-整理笔记.md")
    made_on = date.fromtimestamp(notes_job.created_ns / 1e9)
    assert notes.read_text(encoding="utf-8").startswith(header(NotesMeta(md.stem, lecture_date(md.stem, made_on))))
    assert all(len(models) <= 1 for models in loaded), loaded
    deadline = time.time() + 10
    while _loaded_models() and time.time() < deadline:
        time.sleep(0.5)
    assert _loaded_models() == []
    assert (lesson_job.txt_path.read_bytes(), md.read_bytes()) == (kept["txt"], kept["md"])
    print(f"\nremoved quotes: {notes_job.removed_quotes}, removed concepts: {notes_job.removed_concepts}, "
          f"took {took:.0f} s")


def test_dropping_a_proofread_md_makes_its_notes_on_the_real_model(tmp_path, monkeypatch):
    """12-8 I1: a .md dropped on the window becomes notes through the program's own assembly."""
    import json
    from datetime import date

    from coursescribe.jobs import JobKind
    from coursescribe.models import DEFAULT_PROOFREAD_MODEL, DEFAULT_TRANSCRIBE_MODEL
    from coursescribe.notes import NotesMeta, lecture_date
    from coursescribe.ui.app import build_controller
    from test_notes import header
    from test_notes_job import _loaded_models
    from test_proofread import _ready_notes_model

    client = OllamaClient()
    _ready_notes_model(client)
    home = tmp_path / "家"
    home.mkdir()
    monkeypatch.setenv("COURSESCRIBE_HOME", str(home))
    (home / "settings.json").write_text(
        json.dumps({"format": 2, "output_dir": str(tmp_path / "out"), "transcribe_model": DEFAULT_TRANSCRIBE_MODEL,
                    "proofread_model": DEFAULT_PROOFREAD_MODEL}),
        encoding="utf-8",
    )
    md = tmp_path / "数据库索引.md"
    md.write_text(f"# {md.stem}\n\n{LESSON}\n", encoding="utf-8")
    before = md.read_bytes()
    controller = build_controller(home, threading.Event())
    try:
        controller.start()
        controller.add_files([md])
        deadline = time.time() + 600
        while time.time() < deadline:
            jobs = controller.store.load_all()
            if jobs and all(j.stage in (Stage.DONE, Stage.FAILED) for j in jobs) and controller.state.pending == 0:
                break
            time.sleep(0.5)
    finally:
        controller.shutdown()
    job = controller.store.load_all()[0]
    assert job.kind is JobKind.NOTES and job.stage is Stage.DONE, job.message
    notes = md.with_name(f"{md.stem}-整理笔记.md")
    made_on = date.fromtimestamp(job.created_ns / 1e9)
    assert notes.read_text(encoding="utf-8").startswith(header(NotesMeta(md.stem, lecture_date(md.stem, made_on))))
    assert md.read_bytes() == before
    assert "笔记已生成" in controller.state.notice
    deadline = time.time() + 10
    while _loaded_models() and time.time() < deadline:
        time.sleep(0.5)
    assert _loaded_models() == []
    print("\n", controller.state.notice)
