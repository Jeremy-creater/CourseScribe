"""The small main window."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from coursescribe.controller import MEDIA_SUFFIXES, AppController, AppState
from coursescribe.settings import Settings
from coursescribe.ui.settings_dialog import SettingsDialog
from coursescribe.ui.style import set_role

_PROGRESS_STEPS = 1000
_SILENCE_TEXT = "已经 30 秒没有录到声音。请确认课程正在播放、没有静音。"
FILE_DIALOG_TITLE = "选择音视频文件或校对稿"


def file_filters() -> list[str]:
    """The kinds of file the "选择文件" window offers; the first one is shown to begin with."""
    media = " ".join(f"*{suffix}" for suffix in sorted(MEDIA_SUFFIXES))
    return [f"音视频文件和校对稿 ({media} *.md)", f"音视频文件 ({media})", "校对稿 (*.md)"]


FILE_FILTER = ";;".join(file_filters())  # the way Qt takes several


def _hidden(widget):
    widget.setVisible(False)
    return widget


def _muted(label: QLabel) -> QLabel:
    label.setProperty("muted", True)
    return label


def _hotkey_hint(label: str, note: str) -> str:
    if note:
        return f"{note}。也可以把文件拖到这里"
    return f"快捷键 {label}，也可以把文件拖到这里" if label else "可以把文件拖到这里"


class MainWindow(QWidget):
    hotkey_pressed = Signal()
    _state_changed = Signal(object)

    def __init__(
        self,
        controller: AppController,
        hotkey_label: str,
        choose_files: Callable[[], Iterable[Path]],
        confirm: Callable[[str], bool],
        open_folder: Callable[[Path], None],
        list_proofread_models: Callable[[], Iterable[str]],
        word_list: Path | None = None,
        notes_rules: Path | None = None,
        on_settings_saved: Callable[[Settings], None] | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self._on_settings_saved = on_settings_saved
        self._choose_files = choose_files
        self._confirm = confirm
        self._open_folder = open_folder
        self._list_proofread_models = list_proofread_models
        self._word_list = word_list
        self._notes_rules = notes_rules
        self.setWindowTitle("CourseScribe")
        self.setObjectName("mainWindow")
        self.setAcceptDrops(True)
        self.setMinimumWidth(420)

        title = QLabel("CourseScribe")
        title.setObjectName("title")
        subtitle = _muted(QLabel("录下网课的声音，或选择音视频文件，自动得到文字稿"))
        subtitle.setWordWrap(True)

        self.record_button = QPushButton("开始录音")
        self.record_button.setMinimumHeight(48)
        set_role(self.record_button, "primary")
        self.record_button.clicked.connect(self.controller.toggle_recording)
        self.files_button = QPushButton("选择文件")
        self.files_button.setMinimumHeight(38)
        self.files_button.clicked.connect(self._pick_files)
        self.settings_button = QPushButton("设置")
        self.settings_button.setMinimumHeight(38)
        self.settings_button.clicked.connect(self._open_settings)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addWidget(self.files_button, 2)
        buttons.addWidget(self.settings_button, 1)

        self.hotkey_label = _muted(QLabel(_hotkey_hint(hotkey_label, "")))
        self.hotkey_label.setAlignment(Qt.AlignCenter)
        self.recording_label = _hidden(QLabel())
        self.recording_label.setObjectName("recording")
        self.silence_label = _hidden(QLabel(_SILENCE_TEXT))
        self.silence_label.setObjectName("silence")
        self.silence_label.setWordWrap(True)
        self.status_label = QLabel("准备好了")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        self.progress_bar = _hidden(QProgressBar())
        self.progress_bar.setTextVisible(False)
        self.notice_label = _hidden(_muted(QLabel()))
        self.notice_label.setWordWrap(True)
        self.notice_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.retry_button = _hidden(QPushButton("继续校对"))
        self.retry_button.clicked.connect(self.controller.retry_waiting)
        self.retry_failed_button = _hidden(QPushButton("重试"))
        self.retry_failed_button.clicked.connect(self.controller.retry_failed)
        self.dismiss_failed_button = _hidden(QPushButton("忽略"))
        self.dismiss_failed_button.clicked.connect(self.controller.dismiss_failed)
        self.download_button = _hidden(QPushButton())
        self.download_button.clicked.connect(self._download_model)
        self.open_button = _hidden(QPushButton("打开文件夹"))
        self.open_button.clicked.connect(self._open_last_folder)
        self.notes_button = _hidden(QPushButton("生成整理笔记"))
        self.notes_button.clicked.connect(self._make_notes)
        self.continue_notes_button = _hidden(QPushButton("继续整理笔记"))
        self.continue_notes_button.clicked.connect(self.controller.continue_notes)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self.retry_button)
        actions.addWidget(self.retry_failed_button)
        actions.addWidget(self.dismiss_failed_button)
        actions.addWidget(self.download_button)
        actions.addWidget(self.open_button)
        actions.addWidget(self.notes_button)
        actions.addWidget(self.continue_notes_button)
        actions.addStretch(1)

        self.status_card = QFrame()
        self.status_card.setObjectName("card")
        card = QVBoxLayout(self.status_card)
        card.setContentsMargins(16, 14, 16, 14)
        card.setSpacing(8)
        card.addWidget(self.recording_label)
        card.addWidget(self.status_label)
        card.addWidget(self.progress_bar)
        card.addWidget(self.notice_label)
        card.addLayout(actions)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(4)
        layout.addWidget(self.record_button)
        layout.addLayout(buttons)
        layout.addWidget(self.silence_label)
        layout.addWidget(self.status_card)
        layout.addStretch(1)
        layout.addWidget(self.hotkey_label)

        self.hotkey_pressed.connect(self.controller.toggle_recording)
        self._state_changed.connect(self._show_state)
        self.controller.add_listener(self._state_changed.emit)  # listeners run on other threads
        self._clock = QTimer(self)
        self._clock.setInterval(500)
        self._clock.timeout.connect(self._refresh_recording_time)
        self._clock.start()
        self._show_state(controller.state)

    def _show_state(self, state: AppState) -> None:
        self.record_button.setText("停止录音" if state.recording else "开始录音")
        set_role(self.record_button, "danger" if state.recording else "primary")
        self.recording_label.setVisible(state.recording)
        self.silence_label.setVisible(state.silence_warning)
        queued = state.pending - 1
        self.status_label.setText(f"{state.status}，后面还有 {queued} 个" if queued > 0 else state.status)
        working = state.pending > 0
        self.progress_bar.setVisible(working)
        if state.fraction is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, _PROGRESS_STEPS)
            self.progress_bar.setValue(int(state.fraction * _PROGRESS_STEPS))
        self.notice_label.setText(state.notice)
        self.notice_label.setVisible(bool(state.notice))
        self.retry_button.setVisible(state.waiting_proofread > 0 and not working)
        self.retry_failed_button.setVisible(state.failed > 0 and not working)
        self.dismiss_failed_button.setVisible(state.failed > 0 and not working)
        needs_download = state.model_needed is not None and not state.downloading and not working
        self.download_button.setVisible(needs_download)
        if needs_download:
            self.download_button.setText(f"下载模型（{self.controller.needed_model_size()}）")
        self.open_button.setVisible(state.last_folder is not None)
        self.notes_button.setVisible(state.last_md is not None)
        self.notes_button.setToolTip(f"整理：{state.last_md.name}" if state.last_md is not None else "")
        self.continue_notes_button.setVisible(state.waiting_notes > 0 and not working)
        self._refresh_recording_time()

    def _refresh_recording_time(self) -> None:
        seconds = int(self.controller.recording_seconds())
        self.recording_label.setText(f"正在录音 {seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}")

    def _pick_files(self) -> None:
        chosen = list(self._choose_files())
        if chosen:
            self.controller.add_files(chosen)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self.controller.settings,
            self._list_proofread_models(),
            parent=self,
            word_list=self._word_list,
            open_file=self._open_folder,  # the system opens a file the same way as a folder
            notes_rules=self._notes_rules,
        )
        if dialog.exec():
            saved = dialog.result_settings()
            self.controller.update_settings(saved)
            if self._on_settings_saved is not None:
                self._on_settings_saved(saved)  # the caller decides what a saved setting changes
        dialog.deleteLater()  # destroyed here, never by garbage collection on a worker thread

    def set_hotkey_hint(self, label: str, note: str = "") -> None:
        """Show the shortcut hint under the buttons; only displays what it is given."""
        self.hotkey_label.setText(_hotkey_hint(label, note))

    def _download_model(self) -> None:
        state = self.controller.state
        question = (
            f"要处理这个任务，需要先从网上下载转录模型 {state.model_needed}"
            f"（{self.controller.needed_model_size()}）。只下载模型，不会上传任何内容。现在下载吗？"
        )
        if self._confirm(question):
            self.controller.download_needed_model()

    def _make_notes(self) -> None:
        md = self.controller.state.last_md
        if md is not None:
            self.controller.make_notes(md)

    def _open_last_folder(self) -> None:
        folder = self.controller.state.last_folder
        if folder is not None:
            self._open_folder(folder)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.controller.add_files(paths)
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        self._clock.stop()
        self.controller.shutdown()
        event.accept()
