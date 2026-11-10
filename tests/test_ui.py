"""The window, run off-screen with real widgets and the real application logic."""

from __future__ import annotations

import os
import random
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import shiboken6
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from coursescribe.settings import Settings
from coursescribe.ui.main_window import MainWindow
from coursescribe.ui.style import ARROW_IMAGE, COLORS, STYLE_SHEET, apply_style
from coursescribe.ui.settings_dialog import SettingsDialog
from test_controller import Rig


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    apply_style(app)  # what the real program does
    yield app


class Screen:
    def __init__(self, tmp_path, qt_app, notes_rules=None):
        self.qt_app = qt_app
        self.rig = Rig(tmp_path)
        self.chosen: list = []
        self.answer_yes = True
        self.asked: list[str] = []
        self.window = MainWindow(
            self.rig.controller,
            hotkey_label="Ctrl+Alt+R",
            choose_files=lambda: self.chosen,
            confirm=self._confirm,
            open_folder=self._open,
            list_proofread_models=lambda: ["model-a", "model-b"],
            notes_rules=notes_rules,
        )
        self.opened: list = []
        self.window.show()

    def _confirm(self, text: str) -> bool:
        self.asked.append(text)
        return self.answer_yes

    def _open(self, folder):
        self.opened.append(folder)

    def wait(self, condition, seconds=10.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.qt_app.processEvents()
            if condition():
                return
            time.sleep(0.005)
        raise AssertionError(f"Timed out; state: {self.rig.controller.state}")

    def idle(self):
        self.wait(lambda: self.rig.controller.state.pending == 0 and self.window.status_label.text() == "准备好了")

    def close(self):
        """Destroy the window here: Qt aborts if garbage collection does it on a worker thread."""
        self.window.close()
        self.rig.controller._listeners.clear()
        self.qt_app.processEvents()
        shiboken6.delete(self.window)
        self.qt_app.processEvents()


@pytest.fixture
def screen(tmp_path, qt_app):
    made = Screen(tmp_path, qt_app)
    made.rig.controller.start()
    yield made
    made.close()


def test_first_view_has_only_the_main_controls(screen):
    window = screen.window
    assert window.record_button.text() == "开始录音"
    assert window.files_button.isVisible()
    assert window.status_label.text() == "准备好了"
    assert "Ctrl+Alt+R" in window.hotkey_label.text()
    for hidden in (window.retry_button, window.retry_failed_button, window.dismiss_failed_button, window.download_button, window.open_button, window.progress_bar, window.silence_label):
        assert not hidden.isVisible()


def test_record_button_starts_and_stops_and_files_appear(screen):
    screen.window.record_button.click()
    screen.wait(lambda: screen.window.record_button.text() == "停止录音")
    screen.wait(lambda: screen.rig.controller.recording_seconds() >= 0.5)
    screen.window._refresh_recording_time()
    assert screen.window.recording_label.text().startswith("正在录音")
    screen.window.record_button.click()
    screen.wait(lambda: screen.window.record_button.text() == "开始录音")
    screen.idle()
    job = screen.rig.store.load_all()[0]
    assert job.md_path.exists() and job.txt_path.exists() and job.audio_path.exists()
    assert screen.window.open_button.isVisible()
    assert "完成" in screen.window.notice_label.text()


def test_hotkey_signal_toggles_recording(screen):
    screen.window.hotkey_pressed.emit()
    screen.wait(lambda: screen.rig.controller.state.recording)
    screen.window.hotkey_pressed.emit()
    screen.wait(lambda: not screen.rig.controller.state.recording)
    screen.idle()


def test_choose_files_button_queues_the_chosen_files(screen):
    screen.chosen = [screen.rig.source("a.mp3"), screen.rig.source("b.mkv")]
    screen.window.files_button.click()
    screen.idle()
    assert [job.txt_path.stem for job in screen.rig.store.load_all()] == ["a", "b"]


def test_cancelling_the_file_picker_does_nothing(screen):
    screen.chosen = []
    screen.window.files_button.click()
    assert screen.rig.store.load_all() == []


def _mime(paths):
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return data


def test_dragging_files_onto_the_window_starts_them(screen):
    source = screen.rig.source("拖进来.mp4")
    data = _mime([source])
    enter = QDragEnterEvent(screen.window.rect().center(), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(screen.window, enter)
    assert enter.isAccepted()
    drop = QDropEvent(QPointF(screen.window.rect().center()), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(screen.window, drop)
    screen.idle()
    assert screen.rig.store.load_all()[0].md_path.name == "拖进来.md"


def test_dragging_something_without_files_is_refused(screen):
    data = QMimeData()
    data.setText("just text")
    enter = QDragEnterEvent(screen.window.rect().center(), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(screen.window, enter)
    assert not enter.isAccepted()


def test_progress_bar_follows_progress(screen):
    from coursescribe.controller import AppState

    screen.window._show_state(AppState(status="正在校对（第 3/20 段）", fraction=0.15, pending=1))
    assert screen.window.status_label.text() == "正在校对（第 3/20 段）"
    assert screen.window.progress_bar.isVisible()
    assert screen.window.progress_bar.value() == 150
    screen.window._show_state(AppState(status="正在准备转录", fraction=None, pending=1))
    assert screen.window.progress_bar.maximum() == 0  # moving bar: length unknown


def test_blocked_proofreading_shows_the_way_out_and_the_button_finishes_it(screen):
    screen.rig.proofreader.blocked = "校对服务（Ollama）没有启动。"
    screen.rig.controller.add_files([screen.rig.source()])
    screen.idle()
    assert "Ollama" in screen.window.notice_label.text()
    assert screen.window.retry_button.isVisible()
    screen.rig.proofreader.blocked = None
    screen.window.retry_button.click()
    screen.wait(lambda: not screen.window.retry_button.isVisible())
    screen.idle()
    assert screen.rig.store.load_all()[0].md_path.exists()


def test_failed_job_shows_retry_and_ignore_and_retry_finishes_it(screen):
    screen.rig.proofreader.error = RuntimeError("boom")
    screen.rig.controller.add_files([screen.rig.source()])
    screen.wait(lambda: screen.window.retry_failed_button.isVisible())
    assert screen.window.dismiss_failed_button.isVisible()
    screen.rig.proofreader.error = None
    screen.window.retry_failed_button.click()
    screen.wait(lambda: screen.rig.store.load_all()[0].stage.value == "done")
    screen.wait(lambda: not screen.window.retry_failed_button.isVisible())
    assert not screen.window.dismiss_failed_button.isVisible()


def test_ignore_hides_the_failed_job_buttons(screen):
    screen.rig.proofreader.error = RuntimeError("boom")
    screen.rig.controller.add_files([screen.rig.source()])
    screen.wait(lambda: screen.window.dismiss_failed_button.isVisible())
    screen.window.dismiss_failed_button.click()
    screen.wait(lambda: not screen.window.retry_failed_button.isVisible())


def test_download_states_the_size_and_respects_no(screen):
    screen.rig.transcriber.missing.add("large-v3")
    screen.rig.controller.add_files([screen.rig.source()])
    screen.idle()
    assert screen.window.download_button.isVisible()
    assert "3.1 GB" in screen.window.download_button.text()

    screen.answer_yes = False
    screen.window.download_button.click()
    assert "3.1 GB" in screen.asked[0]
    assert screen.rig.downloads == []

    screen.answer_yes = True
    screen.window.download_button.click()
    screen.wait(lambda: screen.rig.downloads == ["large-v3"])
    screen.wait(lambda: not screen.window.download_button.isVisible())


def test_silence_warning_is_shown_and_cleared(screen):
    screen.rig.controller._on_silence(True)
    screen.wait(lambda: screen.window.silence_label.isVisible())
    assert "没有录到声音" in screen.window.silence_label.text()
    screen.rig.controller._on_silence(False)
    screen.wait(lambda: not screen.window.silence_label.isVisible())


def test_open_folder_button_opens_the_last_result(screen):
    screen.rig.controller.add_files([screen.rig.source("课.mp3")])
    screen.idle()
    screen.window.open_button.click()
    assert screen.opened == [screen.rig.tmp_path / "out" / "课"]


def test_closing_the_window_saves_a_running_recording(tmp_path, qt_app):
    made = Screen(tmp_path, qt_app)
    made.rig.controller.start()
    made.window.record_button.click()
    made.wait(lambda: made.rig.controller.recording_seconds() >= 0.25)
    made.close()
    job = made.rig.store.load_all()[0]
    assert job.stage.value == "transcribing"


class TestSettingsDialog:
    def _dialog(self, tmp_path, qt_app):
        current = Settings(tmp_path / "out", "large-v3", "model-a")
        return SettingsDialog(current, ["model-a", "model-b"], choose_folder=lambda: tmp_path / "new")

    def test_shows_current_values(self, tmp_path, qt_app):
        dialog = self._dialog(tmp_path, qt_app)
        assert dialog.transcribe_box.currentText() == "large-v3"
        assert dialog.proofread_box.currentText() == "model-a"
        assert dialog.folder_label.text() == str(tmp_path / "out")

    def test_returns_the_changed_values(self, tmp_path, qt_app):
        dialog = self._dialog(tmp_path, qt_app)
        dialog.transcribe_box.setCurrentText("medium")
        dialog.proofread_box.setCurrentText("model-b")
        dialog.folder_button.click()
        assert dialog.result_settings() == Settings(tmp_path / "new", "medium", "model-b")

    def test_a_current_model_not_in_the_list_still_shows(self, tmp_path, qt_app):
        current = Settings(tmp_path, "large-v3", "my-own-model")
        dialog = SettingsDialog(current, [], choose_folder=lambda: None)
        assert dialog.proofread_box.currentText() == "my-own-model"
        dialog.folder_button.click()
        assert dialog.result_settings().output_dir == tmp_path

    def test_each_transcribe_model_shows_its_size(self, tmp_path, qt_app):
        dialog = self._dialog(tmp_path, qt_app)
        assert dialog.transcribe_box.itemData(0) is not None
        labels = [dialog.transcribe_box.itemText(i) for i in range(dialog.transcribe_box.count())]
        assert "large-v3" in labels
        assert "GB" in dialog.size_label.text()


def test_settings_button_saves_through_the_dialog(screen, tmp_path, monkeypatch):
    from coursescribe.ui import main_window

    class Stub:
        def __init__(self, current, models, parent=None, **links):
            assert models == ["model-a", "model-b"]
            self.current = current

        def exec(self):
            return 1

        def result_settings(self):
            return Settings(tmp_path / "picked", "medium", "model-b")

        def deleteLater(self):
            pass

    monkeypatch.setattr(main_window, "SettingsDialog", Stub)
    screen.window.settings_button.click()
    assert screen.rig.controller.settings.transcribe_model == "medium"
    assert screen.rig.settings.load().output_dir == tmp_path / "picked"


def _middle_colour(widget):
    image = widget.grab().toImage()
    return image.pixelColor(8, image.height() // 2).name()


class TestLook:
    """Light, solid Apple-like look, whatever the system theme."""

    def test_the_window_is_light_even_under_a_dark_system_theme(self, qt_app):
        from PySide6.QtGui import QColor, QPalette

        dark = QPalette()
        dark.setColor(QPalette.Window, QColor("#1e1e1e"))
        dark.setColor(QPalette.WindowText, QColor("#ffffff"))
        qt_app.setPalette(dark)
        apply_style(qt_app)
        assert qt_app.palette().color(QPalette.Window).name() == COLORS["background"]
        assert qt_app.palette().color(QPalette.WindowText).name() == COLORS["text"]

    def test_window_background_is_solid_and_light(self, screen):
        assert _middle_colour(screen.window) == COLORS["background"]

    def test_record_button_is_blue_then_red_while_recording(self, screen):
        assert _middle_colour(screen.window.record_button) == COLORS["accent"]
        screen.window.record_button.click()
        screen.wait(lambda: screen.window.record_button.text() == "停止录音")
        assert _middle_colour(screen.window.record_button) == COLORS["danger"]
        screen.window.record_button.click()
        screen.wait(lambda: screen.window.record_button.text() == "开始录音")
        assert _middle_colour(screen.window.record_button) == COLORS["accent"]
        screen.idle()

    def test_other_buttons_are_quiet_grey(self, screen):
        assert _middle_colour(screen.window.files_button) == COLORS["control"]
        assert _middle_colour(screen.window.settings_button) == COLORS["control"]

    def test_status_sits_on_a_white_card(self, screen):
        assert _middle_colour(screen.window.status_card) == COLORS["surface"]

    def test_combo_box_arrow_image_is_real(self):
        from PySide6.QtGui import QImage

        assert ARROW_IMAGE.as_posix() in STYLE_SHEET
        assert not QImage(str(ARROW_IMAGE)).isNull()

    def test_settings_save_button_is_blue(self, tmp_path, qt_app):
        from PySide6.QtWidgets import QDialogButtonBox

        dialog = SettingsDialog(Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None)
        dialog.show()
        save = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
        assert _middle_colour(save) == COLORS["accent"]


class TestWordList:
    def test_button_creates_the_word_list_and_opens_it(self, tmp_path, qt_app):
        from coursescribe.vocabulary import TEMPLATE

        opened = []
        path = tmp_path / "home" / "词表.txt"
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None,
            word_list=path, open_file=opened.append,
        )
        dialog.word_list_button.click()
        assert opened == [path]
        assert path.read_text(encoding="utf-8") == TEMPLATE

    def test_an_existing_word_list_is_opened_untouched(self, tmp_path, qt_app):
        opened = []
        path = tmp_path / "词表.txt"
        path.write_text("践行\n", encoding="utf-8")
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None,
            word_list=path, open_file=opened.append,
        )
        dialog.word_list_button.click()
        assert opened == [path] and path.read_text(encoding="utf-8") == "践行\n"

    def test_an_existing_gbk_word_list_is_opened_byte_for_byte_untouched(self, tmp_path, qt_app):
        opened = []
        path = tmp_path / "词表.txt"
        path.write_bytes("践行\r\n见行 → 践行\r\n".encode("gbk"))
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None,
            word_list=path, open_file=opened.append,
        )
        dialog.word_list_button.click()
        assert opened == [path]
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before

    def test_a_word_list_that_cannot_be_created_is_explained_and_not_opened(self, tmp_path, qt_app, monkeypatch):
        from coursescribe.vocabulary import WORD_LIST_CREATE_FAILED

        def no_space(descriptor):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("coursescribe.files.os.fsync", no_space)
        opened = []
        path = tmp_path / "home" / "词表.txt"
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None,
            word_list=path, open_file=opened.append,
        )
        dialog.word_list_button.click()
        assert opened == []
        assert dialog.word_note.text() == WORD_LIST_CREATE_FAILED.format(path=path)
        assert not path.exists()
        assert not path.parent.exists() or list(path.parent.iterdir()) == []

    def test_a_folder_where_the_word_list_should_be_is_explained_and_not_opened(self, tmp_path, qt_app):
        from coursescribe.vocabulary import WORD_LIST_UNREADABLE

        opened = []
        path = tmp_path / "词表.txt"
        path.mkdir()
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None,
            word_list=path, open_file=opened.append,
        )
        dialog.word_list_button.click()
        assert opened == []
        assert dialog.word_note.text() == WORD_LIST_UNREADABLE.format(path=path)
        assert path.is_dir() and list(path.iterdir()) == []

    def test_the_note_wraps_and_returns_to_normal_once_the_word_list_opens(self, tmp_path, qt_app, monkeypatch):
        from coursescribe.vocabulary import TEMPLATE, WORD_LIST_CREATE_FAILED

        def no_space(descriptor):
            raise OSError(28, "No space left on device")

        opened = []
        path = tmp_path / "词表.txt"
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None,
            word_list=path, open_file=opened.append,
        )
        usual = dialog.word_note.text()
        assert dialog.word_note.wordWrap()
        monkeypatch.setattr("coursescribe.files.os.fsync", no_space)
        dialog.word_list_button.click()
        assert dialog.word_note.text() == WORD_LIST_CREATE_FAILED.format(path=path) != usual
        monkeypatch.undo()
        dialog.word_list_button.click()
        assert opened == [path]
        assert path.read_text(encoding="utf-8") == TEMPLATE
        assert dialog.word_note.text() == usual



# ---------------------------------------------------------------- notes settings (12-7)


def _items(box) -> list[str]:
    return [box.itemText(i) for i in range(box.count())]


class TestNotesSettings:
    def test_shows_the_current_notes_settings(self, tmp_path, qt_app):
        """D1 (a)-(d)."""
        from coursescribe.models import DEFAULT_NOTES_MODEL
        from coursescribe.ui.settings_dialog import AUTO_NOTES_TEXT, RULES_NOTE

        dialog = SettingsDialog(Settings(tmp_path, "large-v3", "model-a", "model-b", True), ["model-b", "model-a", "model-a"])
        assert dialog.notes_box.currentText() == "model-b"
        assert _items(dialog.notes_box) == _items(dialog.proofread_box) == ["model-a", "model-b"]
        assert dialog.notes_box.isEditable()
        assert dialog.auto_notes_box.isChecked()
        assert dialog.auto_notes_box.text() == AUTO_NOTES_TEXT
        assert dialog.proofread_box.currentText() == "model-a"

        dialog = SettingsDialog(Settings(tmp_path, "large-v3", "model-a", "我的-模型:latest"), ["model-a"])
        assert dialog.notes_box.currentText() == "我的-模型:latest"

        dialog = SettingsDialog(Settings(tmp_path, "large-v3", "model-a", "model-c"), [])
        assert dialog.notes_box.currentText() == "model-c"
        assert dialog.notes_box.isEditable()
        dialog.notes_box.setCurrentText("手打的模型")
        assert dialog.notes_box.currentText() == "手打的模型"

        opened = []
        home = tmp_path / "home"
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"],
            notes_rules=home / "整理规则.txt", open_file=opened.append,
        )
        assert dialog.notes_box.currentText() == DEFAULT_NOTES_MODEL
        assert not dialog.auto_notes_box.isChecked()
        assert dialog.rules_button.text() == "打开整理规则…"
        assert dialog.rules_button.isEnabled()
        assert dialog.rules_note.text() == RULES_NOTE
        assert not home.exists()
        assert opened == []

    @pytest.mark.parametrize(
        "typed, model",
        [("model-a", "model-a"), ("  qwen3:8b  ", "qwen3:8b"), ("", "qwen3.5:9b"), ("   ", "qwen3.5:9b")],
    )
    @pytest.mark.parametrize("checked", [True, False])
    def test_returns_the_changed_notes_settings(self, tmp_path, qt_app, typed, model, checked):
        """D2."""
        from coursescribe.models import DEFAULT_NOTES_MODEL

        assert DEFAULT_NOTES_MODEL == "qwen3.5:9b"
        current = Settings(tmp_path / "out", "small", "model-b", "model-c", not checked)
        dialog = SettingsDialog(current, ["model-a", "model-b"], choose_folder=lambda: None)
        dialog.notes_box.setCurrentText(typed)
        dialog.auto_notes_box.setChecked(checked)
        result = dialog.result_settings()
        assert result.notes_model == model
        assert result.auto_notes is checked

        dialog = SettingsDialog(current, ["model-a", "model-b"], choose_folder=lambda: None)
        dialog.auto_notes_box.setChecked(checked)
        assert dialog.result_settings() == replace(current, auto_notes=checked)
        assert type(dialog.result_settings().auto_notes) is bool

    def test_opening_and_saving_unchanged_gives_the_same_settings(self, tmp_path, qt_app):
        """D3."""
        from coursescribe.models import MODELS

        rng = random.Random(20270406)
        pool = ["qwen3.5:9b", "gemma4:12b", "我的 模型", "x:y"]
        for round_ in range(40):
            folder = tmp_path.joinpath(*(rng.choice(["课程", "资料 2", "甲乙丙", "网课-第3讲"]) for _ in range(rng.randint(1, 3))))
            current = Settings(folder, rng.choice(list(MODELS)), rng.choice(pool), rng.choice(pool), rng.random() < 0.5)
            listed = {name for name in pool if rng.random() < 0.5}
            for name in (current.proofread_model, current.notes_model):
                if rng.random() < 0.3:
                    listed.discard(name)
                else:
                    listed.add(name)
            dialog = SettingsDialog(current, sorted(listed), choose_folder=lambda: None)
            assert dialog.result_settings() == current, (round_, current, listed)

    @pytest.mark.parametrize("case", ["missing", "gbk", "folder", "disk-full", "after-disk-full", "twice", "not-given"])
    def test_the_rules_button(self, tmp_path, qt_app, monkeypatch, case):
        """D4 (a)-(g)."""
        from coursescribe.notes_rules import RULES_CREATE_FAILED, RULES_FOLDER_IN_THE_WAY, SHIPPED_RULES
        from coursescribe.ui.settings_dialog import RULES_NOTE, WORD_NOTE

        def no_space(descriptor):
            raise OSError(28, "No space left on device")

        opened = []
        path = tmp_path / "home" / "整理规则.txt"
        if case == "gbk":
            path.parent.mkdir()
            path.write_bytes("// 说明\r\n每节写案例\r\n".encode("gbk"))
            before = (path.read_bytes(), path.stat().st_mtime_ns)
        elif case == "folder":
            path.mkdir(parents=True)
        if case == "not-given":
            for links in ({"open_file": opened.append}, {"notes_rules": path}):
                dialog = SettingsDialog(Settings(tmp_path, "large-v3", "model-a"), ["model-a"], **links)
                assert not dialog.rules_button.isEnabled()
                assert dialog.rules_note.wordWrap()
                dialog._open_rules()  # reached some other way than the disabled button: still nothing
                assert dialog.rules_note.text() == RULES_NOTE
            assert opened == []
            assert not path.parent.exists()
            return
        dialog = SettingsDialog(
            Settings(tmp_path, "large-v3", "model-a"), ["model-a"], choose_folder=lambda: None,
            word_list=tmp_path / "词表.txt", notes_rules=path, open_file=opened.append,
        )
        assert dialog.rules_note.wordWrap()
        if case in ("disk-full", "after-disk-full"):
            monkeypatch.setattr("coursescribe.files.os.fsync", no_space)
        dialog.rules_button.click()
        if case == "missing":
            assert opened == [path]
            assert path.read_bytes() == SHIPPED_RULES.read_bytes()
            assert dialog.rules_note.text() == RULES_NOTE
        elif case == "gbk":
            assert opened == [path]
            assert (path.read_bytes(), path.stat().st_mtime_ns) == before
            assert dialog.rules_note.text() == RULES_NOTE
        elif case == "folder":
            assert opened == []
            assert dialog.rules_note.text() == RULES_FOLDER_IN_THE_WAY.format(path=path)
            assert path.is_dir() and list(path.iterdir()) == []
        elif case in ("disk-full", "after-disk-full"):
            assert opened == []
            assert dialog.rules_note.text() == RULES_CREATE_FAILED.format(path=path)
            assert not path.exists()
            assert not path.parent.exists() or list(path.parent.iterdir()) == []
            if case == "after-disk-full":
                monkeypatch.undo()
                dialog.rules_button.click()
                assert opened == [path]
                assert path.read_bytes() == SHIPPED_RULES.read_bytes()
                assert dialog.rules_note.text() == RULES_NOTE
        elif case == "twice":
            assert opened == [path]
            first = (path.read_bytes(), path.stat().st_mtime_ns)
            time.sleep(0.02)
            dialog.rules_button.click()
            assert opened == [path, path]
            assert (path.read_bytes(), path.stat().st_mtime_ns) == first
            assert dialog.rules_note.text() == RULES_NOTE
        assert dialog.rules_note.wordWrap()
        assert dialog.word_note.text() == WORD_NOTE
        assert not (tmp_path / "词表.txt").exists()


def _driven(script, code):
    """The real settings window, with exec() doing what a user would and then pressing Save (1) or Cancel (0)."""

    class Driven(SettingsDialog):
        def exec(self):
            script(self)
            return code

    return Driven


def _check_and_pick_b(dialog):
    dialog.auto_notes_box.setChecked(True)
    dialog.notes_box.setCurrentText("model-b")


def test_the_main_window_saves_notes_settings_only_on_save_and_opens_the_rules(tmp_path, qt_app, monkeypatch):
    """D5 (a)-(d)."""
    from coursescribe.notes_rules import SHIPPED_RULES
    from coursescribe.settings import SettingsStore
    from coursescribe.ui import main_window

    rules = tmp_path / "home" / "整理规则.txt"
    made = Screen(tmp_path, qt_app, notes_rules=rules)
    made.rig.controller.start()
    try:
        controller = made.rig.controller
        settings_file = tmp_path / "settings.json"
        first = controller.settings

        monkeypatch.setattr(main_window, "SettingsDialog", _driven(_check_and_pick_b, 0))
        made.window.settings_button.click()
        assert controller.settings == first
        assert not settings_file.exists()
        assert not rules.exists() and made.opened == []

        monkeypatch.setattr(main_window, "SettingsDialog", _driven(lambda dialog: dialog.rules_button.click(), 0))
        made.window.settings_button.click()
        assert made.opened == [rules]
        assert rules.read_bytes() == SHIPPED_RULES.read_bytes()
        assert controller.settings == first
        assert not settings_file.exists()

        monkeypatch.setattr(main_window, "SettingsDialog", _driven(_check_and_pick_b, 1))
        made.window.settings_button.click()
        assert controller.settings.auto_notes is True and controller.settings.notes_model == "model-b"
        loaded = SettingsStore(settings_file, made.rig.settings.defaults).load()
        assert loaded.auto_notes is True and loaded.notes_model == "model-b"
        saved = settings_file.read_bytes()

        seen = []
        monkeypatch.setattr(
            main_window, "SettingsDialog",
            _driven(lambda dialog: seen.append((dialog.auto_notes_box.isChecked(), dialog.notes_box.currentText())), 1),
        )
        made.window.settings_button.click()
        assert seen == [(True, "model-b")]
        assert settings_file.read_bytes() == saved
    finally:
        made.close()


def test_settings_that_cannot_be_saved_are_explained_in_the_window(tmp_path, qt_app, monkeypatch):
    """D6."""
    from coursescribe.controller import SETTINGS_NOT_SAVED
    from coursescribe.ui import main_window
    from test_notes_job import DiskFault

    made = Screen(tmp_path, qt_app, notes_rules=tmp_path / "home" / "整理规则.txt")
    made.rig.controller.start()
    try:
        controller = made.rig.controller
        settings_file = tmp_path / "settings.json"
        controller.update_settings(controller.settings)
        before = settings_file.read_bytes()
        DiskFault(monkeypatch, tmp_path, OSError(28, "No space left on device"))
        monkeypatch.setattr(main_window, "SettingsDialog", _driven(_check_and_pick_b, 1))
        made.window.settings_button.click()
        expected = SETTINGS_NOT_SAVED.format(path=settings_file)
        made.wait(lambda: made.window.notice_label.isVisible())
        assert made.window.notice_label.text() == expected
        assert controller.settings.auto_notes is True
        assert not any("程序出了点意外" in state.notice for state in made.rig.states)
        assert settings_file.read_bytes() == before
    finally:
        made.close()
