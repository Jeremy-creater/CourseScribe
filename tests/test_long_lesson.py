"""One hour of generated speech through the whole flow, with timing and memory."""

from __future__ import annotations

import subprocess
import threading
import time

import pytest

from coursescribe.jobs import Job, JobStore, Stage
from coursescribe.pipeline import Pipeline
from coursescribe.platform import prepare_gpu_libraries
from coursescribe.proofread import OllamaProofreader
from coursescribe.text import content_chars
from coursescribe.transcribe import WhisperTranscriber
from coursescribe.wavfile import repair_wav
from no_skip_plugin import require_module
from speech import speak_to_wav

pytestmark = pytest.mark.perf

PASSAGE = (
    "今天我们继续讲数据库的基础知识。首先来看什么是索引，索引就像一本书的目录，可以帮助我们快速找到内容。"
    "如果没有索引，数据库只能一条一条地查找，数据多的时候会非常慢。接下来我们用 Python 写一个小程序，"
    "看一下有索引和没有索引的区别。你会发现查询的时间差了很多倍。但是索引也有代价，每次写入数据都要更新索引，"
    "所以写入会变慢，占用的空间也会变大。那么应该给哪些字段建索引呢？一般来说，经常用来查询和排序的字段才值得建。"
    "另外还要注意，区分度很低的字段，比如性别，建了索引效果也不好。好，这一部分我们先讲到这里，下面看一个实际的例子。"
)


def test_one_hour_lesson_finishes_without_losing_content(tmp_path):
    psutil = require_module("psutil", "run: python -m pip install psutil")
    prepare_gpu_libraries()
    passage = speak_to_wav(PASSAGE, tmp_path / "passage.wav")
    lesson = tmp_path / "lesson.wav"
    seconds = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(passage)],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    repeats = int(3600 / seconds) + 1
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", str(repeats), "-i", str(passage), "-t", "3600", str(lesson)],
        check=True,
    )

    peak = {"rss": 0}
    finished = threading.Event()

    def watch() -> None:
        process = psutil.Process()
        while not finished.wait(1.0):
            peak["rss"] = max(peak["rss"], process.memory_info().rss)

    threading.Thread(target=watch, daemon=True).start()
    store = JobStore(tmp_path / "state")
    from datetime import datetime

    job = Job.new_file(lesson, tmp_path / "out", datetime.now(), "large-v3", "gemma4:12b")
    store.save(job)
    marks: dict[str, float] = {}

    def on_status(job, text, fraction):
        if text.startswith("正在校对"):
            marks.setdefault("proofread_start", time.time())

    started = time.time()
    Pipeline(store, WhisperTranscriber(), OllamaProofreader(), repair_wav, on_status=on_status).run(job)
    ended = time.time()
    finished.set()

    assert job.stage is Stage.DONE, job.message
    raw = job.txt_path.read_text(encoding="utf-8")
    proofread = job.md_path.read_text(encoding="utf-8")
    print(
        f"\ntranscribe {marks['proofread_start'] - started:.0f}s, proofread {ended - marks['proofread_start']:.0f}s, "
        f"peak memory {peak['rss'] / 1e9:.1f} GB, raw {len(content_chars(raw))} chars, "
        f"proofread {len(content_chars(proofread))} chars, paragraphs {proofread.count(chr(10) * 2) + 1}, "
        f"unproofread {job.unproofread_chunks}/{job.total_chunks}"
    )
    assert job.unproofread_chunks <= job.total_chunks * 0.1
    assert len(content_chars(proofread)) > len(content_chars(raw)) * 0.97
    assert peak["rss"] < 6e9
