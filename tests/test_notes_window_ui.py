"""The window for organised notes (12-8), off-screen with real widgets: the two new buttons, files
dropped or chosen, and the file dialog's filters. The controller's side is in test_notes_window.py.

Every case that can reach the notes uses ``notes_screen``, which has the real notes writer talking to
the shared FakeOllama; the plain ``screen`` of test_ui.py would try the real Ollama."""

from __future__ import annotations

import os
import random
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QDir, QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFileDialog

from coursescribe.controller import (
    MEDIA_SUFFIXES,
    NOT_ADDED_TXT,
    NOTES_ALREADY,
    NOTES_ALREADY_WAITING,
    NOTES_NOT_A_FILE,
    AppState,
)
from coursescribe.jobs import JobKind, Stage
from coursescribe.notes_writer import NOTES_BUSY_MESSAGE, NOTES_MODEL_MISSING
from coursescribe.ui.main_window import FILE_DIALOG_TITLE, FILE_FILTER, file_filters
from notes_replies import META  # noqa: F401
from test_controller import _job_records_fail
from test_notes_auto import LESSON_FILE, LessonProofreader, data_problem, source_jobs
from test_notes_job import (  # noqa: F401  (ollama, svc and rigs are fixtures)
    FINISHED_BYTES,
    NAME,
    NOTES_FILE,
    SRC_BYTES,
    gated,
    lesson,
    set_notes_model,
    notes_files,
    notes_jobs,
    ollama,
    rigs,
    svc,
)
from test_notes_writer import BUSY
from test_ui import Screen, _mime, qt_app  # noqa: F401  (qt_app is a fixture)


@pytest.fixture
def notes_screen(tmp_path, qt_app, svc):
    made = Screen(tmp_path, qt_app, proofreader=LessonProofreader(), notes_writer=svc.writer())
    made.rig.controller.start()
    yield made
    made.close()


def drop(screen, paths) -> tuple[bool, bool]:
    """What the window does with real drag events: (entered and accepted, dropped and accepted)."""
    data = _mime(paths)
    enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    screen.window.dragEnterEvent(enter)
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    event.setAccepted(False)
    screen.window.dropEvent(event)
    return enter.isAccepted(), event.isAccepted()


def finish_lesson(screen) -> Path:
    """A file lesson made and proofread; returns the proofread .md the notes button stands for."""
    rig = screen.rig
    rig.controller.add_files([rig.source(LESSON_FILE)])
    screen.idle()
    return source_jobs(rig)[0].md_path


def notice(screen) -> str:
    return screen.window.notice_label.text()


# ---------------------------------------------------------------- W1


def test_the_notes_button_makes_notes_of_the_lesson_just_finished(notes_screen):
    """W1."""
    screen, window = notes_screen, notes_screen.window
    assert not window.notes_button.isVisible() and not window.continue_notes_button.isVisible()
    assert window.notes_button.text() == "生成整理笔记"
    assert window.continue_notes_button.text() == "继续整理笔记"
    assert window.notes_button.toolTip() == ""

    md = finish_lesson(screen)
    assert window.notes_button.isVisible() and window.open_button.isVisible()
    assert window.notes_button.toolTip() == f"整理：{md.name}"
    before = md.read_bytes(), md.stat().st_mtime_ns

    window.notes_button.click()
    screen.wait(lambda: notice(screen).startswith("笔记已生成："))
    screen.idle()

    jobs = notes_jobs(screen.rig.store)
    assert len(jobs) == 1 and jobs[0].source_md == md and jobs[0].stage is Stage.DONE
    assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
    assert notice(screen) == (
        f"笔记已生成：{NOTES_FILE}。删掉了 2 条对不上原文的引述、2 个原文里没有的概念。笔记是初稿，请自己看一遍。"
    )
    assert window.notice_label.isVisible() and window.notice_label.wordWrap()
    assert window.notes_button.isVisible() and not window.continue_notes_button.isVisible()
    assert (md.read_bytes(), md.stat().st_mtime_ns) == before
    assert sorted(p.name for p in md.parent.iterdir()) == sorted([f"{NAME}.txt", f"{NAME}.md", NOTES_FILE])


def test_the_notes_button_asks_for_the_file_when_the_proofread_md_was_renamed(notes_screen):
    """The user checks and renames the .md after the lesson: the button must not say "not found"."""
    screen, window = notes_screen, notes_screen.window
    md = finish_lesson(screen)
    renamed = md.with_name("核对过的课.md")
    md.rename(renamed)
    screen.chosen = [renamed]

    window.notes_button.click()
    screen.wait(lambda: notice(screen).startswith("笔记已生成："))
    screen.idle()

    assert NOTES_NOT_A_FILE.format(name=md.name) not in notice(screen)
    jobs = notes_jobs(screen.rig.store)
    assert len(jobs) == 1 and jobs[0].source_md == renamed and jobs[0].stage is Stage.DONE
    assert (renamed.parent / "核对过的课-整理笔记.md").exists()


# ---------------------------------------------------------------- W2 (the two clicks and the button following last_md)


def test_clicking_twice_makes_one_job_and_the_button_follows_the_newest_lesson(notes_screen, svc):
    """W2, the first two parts."""
    screen, window, rig = notes_screen, notes_screen.window, notes_screen.rig
    md1 = finish_lesson(screen)
    gate = gated(svc)
    window.notes_button.click()
    screen.wait(lambda: len(svc.gens) == 1)
    window.notes_button.click()
    window.notes_button.click()
    assert len(notes_jobs(rig.store)) == 1
    assert rig.controller.state.notice == NOTES_ALREADY.format(name=md1.name)
    gate.set()
    screen.idle()

    rig.controller.add_files([rig.source("乙.mp3")])
    screen.idle()
    md2 = [job for job in source_jobs(rig) if job.md_path != md1][0].md_path
    assert rig.controller.state.last_md == md2
    assert window.notes_button.toolTip() == f"整理：{md2.name}"
    first_notes = (md1.parent / NOTES_FILE).read_bytes(), (md1.parent / NOTES_FILE).stat().st_mtime_ns

    window.notes_button.click()
    screen.wait(lambda: notice(screen).startswith("笔记已生成：乙-整理笔记.md。"))
    screen.idle()
    assert [job.source_md for job in notes_jobs(rig.store)] == [md1, md2]
    assert (md2.parent / "乙-整理笔记.md").read_text(encoding="utf-8").splitlines()[0] == "# 乙"
    assert ((md1.parent / NOTES_FILE).read_bytes(), (md1.parent / NOTES_FILE).stat().st_mtime_ns) == first_notes
    assert notes_files(md1) == [NOTES_FILE]


# ---------------------------------------------------------------- W3


def test_waiting_for_ollama_shows_the_continue_button_that_the_notice_names(notes_screen, svc):
    """W3 (a)."""
    screen, window, rig = notes_screen, notes_screen.window, notes_screen.rig
    md = finish_lesson(screen)
    svc.serve(BUSY, BUSY)
    window.notes_button.click()
    screen.idle()
    assert window.continue_notes_button.isVisible()
    assert notice(screen) == NOTES_BUSY_MESSAGE
    assert f"“{window.continue_notes_button.text()}”" in NOTES_BUSY_MESSAGE
    assert not window.retry_button.isVisible()  # that is "continue proofreading", a different thing
    assert window.notes_button.isVisible()

    window.notes_button.click()
    assert rig.controller.state.notice == NOTES_ALREADY_WAITING.format(name=md.name)
    assert len(notes_jobs(rig.store)) == 1

    svc.serve()
    asked = len(svc.gens)
    window.continue_notes_button.click()
    window.continue_notes_button.click()
    screen.idle()
    assert len(svc.gens) == asked + 1
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE]
    assert notes_files(md) == [NOTES_FILE]
    assert not window.continue_notes_button.isVisible()


# ---------------------------------------------------------------- W4


def _batch(screen):
    rig = screen.rig
    folder = rig.tmp_path
    a = rig.source("a.mp3")
    md = lesson(rig.tmp_path)
    notes = md.parent / "课-整理笔记.md"
    notes.write_bytes(b"x")
    return a, md, md.with_suffix(".txt"), notes, folder / "没有.md"


def _held_screen(tmp_path, qt_app, svc):
    screen = Screen(tmp_path, qt_app, proofreader=LessonProofreader(), notes_writer=svc.writer())
    screen.rig.controller.start()
    return screen


def test_dropped_and_chosen_files_go_each_their_own_way_with_one_notice(tmp_path, qt_app, svc):
    """W4 (a)(b)(c)(e)."""
    results = []
    for how in ("drop", "choose"):
        root = tmp_path / how
        root.mkdir()
        screen = _held_screen(root, qt_app, svc)
        try:
            screen.rig.transcriber.gate = threading.Event()
            gate = gated(svc)
            a, md, txt, notes, missing = _batch(screen)
            batch = [a, md, txt, notes, missing]
            if how == "drop":
                entered, dropped = drop(screen, batch)
                assert entered and dropped
            else:
                screen.chosen = batch
                screen.window.files_button.click()
            assert screen.window.notice_label.isVisible()
            results.append(screen.rig.controller.state.notice)
            lines = screen.rig.controller.state.notice.split("\n")
            assert lines == [
                "没有加入：" + NOT_ADDED_TXT.format(name=txt.name),
                f"没有整理：课-整理笔记.md（这已经是整理笔记了，请选它的校对稿）",
                NOTES_NOT_A_FILE.format(name="没有.md"),
            ]
            assert [job.kind for job in screen.rig.store.load_all()] == [JobKind.FILE, JobKind.NOTES]
            gate.set()
            screen.rig.transcriber.gate.set()
            screen.idle()
            assert (md.parent / NOTES_FILE).read_bytes() == FINISHED_BYTES
        finally:
            screen.close()
    assert results[0] == results[1]  # the same notice by either way in

    (tmp_path / "txt").mkdir()
    only_txt = Screen(tmp_path / "txt", qt_app, notes_writer=svc.writer())
    try:
        only_txt.rig.controller.start()
        only_txt.chosen = [lesson(tmp_path / "txt2").with_suffix(".txt")]
        only_txt.window.files_button.click()
        assert only_txt.window.notice_label.text() == "没有加入：" + NOT_ADDED_TXT.format(name=only_txt.chosen[0].name)
        assert only_txt.rig.store.load_all() == []
        since = len(only_txt.rig.states)
        only_txt.chosen = []
        only_txt.window.files_button.click()
        assert len(only_txt.rig.states) == since
    finally:
        only_txt.close()


# ---------------------------------------------------------------- W6


def test_a_dropped_md_gets_its_notes_but_there_is_no_lesson_to_stand_for(notes_screen):
    """W6 (a)."""
    screen, window = notes_screen, notes_screen.window
    md = lesson(screen.rig.tmp_path)
    drop(screen, [md])
    screen.wait(lambda: notice(screen).startswith("笔记已生成："))
    screen.idle()
    assert not window.notes_button.isVisible()
    assert window.open_button.isVisible()


# ---------------------------------------------------------------- W7


def test_after_a_restart_the_notes_button_is_gone_until_a_lesson_finishes(tmp_path, qt_app, svc):
    """W7 (a)."""
    first = Screen(tmp_path, qt_app, proofreader=LessonProofreader(), notes_writer=svc.writer())
    first.rig.controller.start()
    try:
        md = finish_lesson(first)
        assert first.window.notes_button.isVisible()
    finally:
        first.close()
    second = Screen(tmp_path, qt_app, proofreader=LessonProofreader(), notes_writer=svc.writer())
    second.rig.controller.start()
    try:
        second.idle()
        assert not second.window.notes_button.isVisible()
        assert not second.window.open_button.isVisible()
        drop(second, [md])
        second.wait(lambda: notice(second).startswith("笔记已生成："))
        second.idle()
        assert notes_files(md) == [NOTES_FILE]
    finally:
        second.close()


# ---------------------------------------------------------------- W8


def test_the_notes_button_works_while_the_automatic_notes_are_made(notes_screen, svc):
    """W8."""
    from test_notes_auto import set_auto

    screen, window, rig = notes_screen, notes_screen.window, notes_screen.rig
    set_auto(rig, True)
    gate = gated(svc)
    rig.controller.add_files([rig.source(LESSON_FILE)])
    screen.wait(lambda: len(svc.gens) == 1)
    md = source_jobs(rig)[0].md_path
    assert window.notes_button.isVisible()
    window.notes_button.click()
    assert rig.controller.state.notice == NOTES_ALREADY.format(name=md.name)
    gate.set()
    screen.idle()
    window.notes_button.click()
    screen.wait(lambda: notice(screen).startswith(f"笔记已生成：{NAME}-整理笔记 (2).md。"))
    screen.idle()
    assert len(notes_jobs(rig.store)) == 2
    assert sorted(notes_files(md)) == sorted([NOTES_FILE, f"{NAME}-整理笔记 (2).md"])


# ---------------------------------------------------------------- PW


def test_every_button_is_where_the_state_says(notes_screen):
    """PW."""
    window, tmp = notes_screen.window, notes_screen.rig.tmp_path
    rng = random.Random(20270505)
    mds = [None, tmp / "课甲" / "甲.md", tmp / "课乙" / "乙.md"]
    seen = {"notes only": 0, "continue hidden by pending": 0, "both": 0, "waiting notes only": 0, "changed": 0}
    previous = None
    for _ in range(400):
        state = AppState(
            recording=rng.random() < 0.5,
            pending=rng.choice([0, 0, 1, 2, 5]),
            waiting_notes=rng.choice([0, 0, 1, 3]),
            waiting_proofread=rng.choice([0, 0, 1, 3]),
            failed=rng.choice([0, 0, 1, 3]),
            last_md=rng.choice(mds),
            last_folder=rng.choice([None, tmp / "课甲"]),
            model_needed=rng.choice([None, "large-v3"]),
            downloading=rng.random() < 0.5,
            notice=rng.choice(["", "一句话"]),
        )
        window._show_state(state)
        idle = state.pending == 0
        assert window.notes_button.isVisible() == (state.last_md is not None), state
        assert window.notes_button.toolTip() == (f"整理：{state.last_md.name}" if state.last_md else ""), state
        assert window.continue_notes_button.isVisible() == (state.waiting_notes > 0 and idle), state
        assert window.retry_button.isVisible() == (state.waiting_proofread > 0 and idle), state
        assert window.retry_failed_button.isVisible() == (state.failed > 0 and idle), state
        assert window.dismiss_failed_button.isVisible() == (state.failed > 0 and idle), state
        assert window.download_button.isVisible() == (state.model_needed is not None and not state.downloading and idle), state
        assert window.open_button.isVisible() == (state.last_folder is not None), state
        assert (window.notes_button.text(), window.continue_notes_button.text()) == ("生成整理笔记", "继续整理笔记")
        if state.last_md is not None and state.last_folder is None:
            seen["notes only"] += 1
        if state.waiting_notes > 0 and not idle:
            seen["continue hidden by pending"] += 1
        if state.last_md is not None and state.waiting_notes > 0 and idle:
            seen["both"] += 1
        if state.waiting_notes > 0 and state.waiting_proofread == 0:
            seen["waiting notes only"] += 1
        if previous is not None and state.last_md != previous.last_md and None not in (state.last_md, previous.last_md):
            seen["changed"] += 1
            window._show_state(previous)  # showing another state leaves nothing behind
            assert window.notes_button.toolTip() == f"整理：{previous.last_md.name}"
            window._show_state(state)
            assert window.notes_button.toolTip() == f"整理：{state.last_md.name}"
        previous = state
    floors = {"notes only": 40, "continue hidden by pending": 40, "both": 30, "waiting notes only": 60, "changed": 40}
    for name, floor in floors.items():
        assert seen[name] >= floor, (name, seen)


# ---------------------------------------------------------------- E1, E2


def test_the_file_dialog_offers_media_and_proofread_files_with_both_first(qt_app):
    """E1."""
    dialog = QFileDialog()
    dialog.setNameFilter(FILE_FILTER)
    filters = file_filters()
    assert dialog.nameFilters() == filters and len(filters) == 3
    assert filters[0].startswith("音视频文件和校对稿 (") and filters[0].endswith(" *.md)")
    media = "音视频文件 (" + " ".join(f"*{s}" for s in sorted(MEDIA_SUFFIXES)) + ")"
    assert filters[1] == media
    assert filters[2] == "校对稿 (*.md)"
    assert dialog.selectedNameFilter() == filters[0]
    assert FILE_DIALOG_TITLE == "选择音视频文件或校对稿"
    assert all(f"*{suffix}" in filters[1] for suffix in MEDIA_SUFFIXES)
    source = (Path(__file__).parent.parent / "src" / "coursescribe" / "ui" / "app.py").read_text(encoding="utf-8")
    assert "FILE_FILTER" in source and "FILE_DIALOG_TITLE" in source and '"选择音视频文件"' not in source


def _patterns(entry: str) -> list[str]:
    return entry[entry.index("(") + 1 : -1].split()


def test_what_the_dialog_shows_is_what_add_files_accepts(qt_app):
    """E2 (the filter half)."""
    rng = random.Random(20270504)
    pieces = list("课程数据库索引 .()（）一二三abc")
    suffixes = sorted(MEDIA_SUFFIXES | {".md", ".txt", ".docx", ".mdx", ".m4", ".mp33", ""})
    first, second, third = (_patterns(entry) for entry in file_filters())
    seen = {"fits": 0, "not": 0, "upper": 0}
    for _ in range(300):
        while True:
            stem = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
            if stem.strip(". "):
                break
        suffix = rng.choice(suffixes)
        suffix = "".join(c.upper() if rng.random() < 0.5 else c for c in suffix)
        name = stem + suffix
        ext = Path(name).suffix.lower()
        fits = QDir.match(first, name)
        assert fits == (ext in MEDIA_SUFFIXES or ext == ".md"), name
        assert fits == (QDir.match(second, name) or QDir.match(third, name)), name
        assert QDir.match(third, name) == (ext == ".md"), name
        seen["fits" if fits else "not"] += 1
        seen["upper"] += ext != Path(name).suffix
    assert seen["fits"] >= 100 and seen["not"] >= 60 and seen["upper"] >= 60, seen


# ---------------------------------------------------------------- W3 (b)(c), W4 (d), W5, W6 (b)(c), W7 (b)


def test_a_missing_notes_model_waits_and_continue_uses_the_model_set_now(notes_screen, svc):
    """W3 (b)."""
    screen, window, rig = notes_screen, notes_screen.window, notes_screen.rig
    finish_lesson(screen)
    svc.serve(models=["other"])
    window.notes_button.click()
    screen.idle()
    assert notice(screen) == NOTES_MODEL_MISSING.format(model=rig.controller.settings.notes_model)
    assert window.continue_notes_button.isVisible()
    svc.serve()
    set_notes_model(rig, "model-a")
    window.continue_notes_button.click()
    screen.idle()
    assert [g["model"] for g in svc.gens] == ["model-a"]
    assert [job.stage for job in notes_jobs(rig.store)] == [Stage.DONE]


def test_the_continue_button_waits_for_the_other_jobs_to_finish(notes_screen, svc):
    """W3 (c)."""
    screen, window, rig = notes_screen, notes_screen.window, notes_screen.rig
    finish_lesson(screen)
    svc.serve(BUSY, BUSY)
    window.notes_button.click()
    screen.idle()
    assert window.continue_notes_button.isVisible()
    rig.transcriber.gate = threading.Event()
    rig.controller.add_files([rig.source("乙.mp3")])
    screen.wait(lambda: window.progress_bar.isVisible())
    assert not window.continue_notes_button.isVisible()
    rig.transcriber.gate.set()
    screen.idle()
    assert window.continue_notes_button.isVisible() and rig.controller.state.waiting_notes == 1


def test_a_drop_without_local_files_does_nothing(notes_screen):
    """W4 (d)."""
    from PySide6.QtCore import QMimeData, QUrl

    screen = notes_screen
    since = len(screen.rig.states)
    web = QMimeData()
    web.setUrls([QUrl("http://example.com/a.mp3")])
    text = QMimeData()
    text.setText("hello")
    for data in (web, text):
        event = QDropEvent(QPointF(5, 5), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
        event.setAccepted(False)
        screen.window.dropEvent(event)
        assert not event.isAccepted()
    assert screen.rig.store.load_all() == [] and len(screen.rig.states) == since
    mixed = QMimeData()
    mixed.setUrls([QUrl("http://example.com/a.mp3"), QUrl.fromLocalFile(str(screen.rig.source("本机.mp3")))])
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, mixed, Qt.LeftButton, Qt.NoModifier)
    screen.window.dropEvent(event)
    screen.idle()
    assert [job.kind for job in screen.rig.store.load_all()] == [JobKind.FILE]


def test_a_problem_when_the_button_is_clicked_is_said_and_the_button_stays(notes_screen):
    """W5 (a)(b)."""
    screen, window, rig = notes_screen, notes_screen.window, notes_screen.rig
    md = finish_lesson(screen)
    moved = md.with_name("挪走.md")
    md.rename(moved)
    screen.chosen = []  # the file window opens and the user closes it
    window.notes_button.click()
    assert notes_jobs(rig.store) == [] and window.notes_button.isVisible()
    moved.rename(md)
    _job_records_fail(rig, lambda job: job.kind is JobKind.NOTES)
    window.notes_button.click()
    assert notice(screen) == f"没有整理：{md.name}。" + data_problem(rig)
    assert window.notes_button.isVisible()
    assert "程序出了点意外" not in notice(screen)
    del rig.store.save  # the real one again
    window.notes_button.click()
    screen.wait(lambda: notice(screen).startswith("笔记已生成："))
    screen.idle()
    assert notes_files(md) == [NOTES_FILE]


def test_no_notes_button_while_recording_or_waiting_for_proofreading(notes_screen):
    """W6 (b)(c)."""
    screen, window, rig = notes_screen, notes_screen.window, notes_screen.rig
    rig.controller.toggle_recording()
    screen.wait(lambda: window.record_button.text() == "停止录音")
    assert not window.notes_button.isVisible()
    rig.controller.toggle_recording()
    screen.idle()
    window.notes_button.isVisible()  # the recording finished and was proofread: now there is one
    assert window.notes_button.isVisible()

    rig.proofreader.blocked = "没有启动"
    rig.controller.add_files([rig.source("乙.mp3")])
    screen.idle()
    assert window.retry_button.isVisible() and not window.continue_notes_button.isVisible()


def test_after_a_restart_a_note_waiting_for_ollama_is_tried_again_and_continued(tmp_path, qt_app, svc):
    """W7 (b)."""
    first = Screen(tmp_path, qt_app, proofreader=LessonProofreader(), notes_writer=svc.writer())
    first.rig.controller.start()
    try:
        md = finish_lesson(first)
        svc.serve(BUSY, BUSY)
        first.window.notes_button.click()
        first.idle()
        assert first.window.continue_notes_button.isVisible()
    finally:
        first.close()
    svc.serve(BUSY, BUSY)
    second = Screen(tmp_path, qt_app, proofreader=LessonProofreader(), notes_writer=svc.writer())
    second.rig.controller.start()
    try:
        second.idle()
        assert second.window.continue_notes_button.isVisible()
        assert notice(second) == NOTES_BUSY_MESSAGE
        svc.serve()
        second.window.continue_notes_button.click()
        second.idle()
        assert notes_files(md) == [NOTES_FILE]
        assert [job.stage for job in notes_jobs(second.rig.store)] == [Stage.DONE]
    finally:
        second.close()
