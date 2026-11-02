"""Everything real together: generated speech, the real model, real Ollama."""

from __future__ import annotations

import threading
import time

import pytest

from coursescribe.controller import AppController
from coursescribe.jobs import JobStore, Stage
from coursescribe.ollama import OllamaClient, OllamaDown
from coursescribe.platform import open_system_audio, prepare_gpu_libraries
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
