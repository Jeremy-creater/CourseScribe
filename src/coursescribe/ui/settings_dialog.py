"""Settings: transcription model, proofreading and notes models, the auto-notes switch, and save folder."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import QEvent, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from coursescribe.models import DEFAULT_NOTES_MODEL, MODELS, model_size_label
from coursescribe.platform import DEFAULT_HOTKEY, HOTKEY_PLATFORM
from coursescribe.notes_rules import NotesRulesUnreadable, ensure_notes_rules
from coursescribe.settings import Settings
from coursescribe.shortcut import InvalidShortcut, Shortcut, parse
from coursescribe.ui.key_capture import candidate_text, is_leaving_press
from coursescribe.ui.style import set_role
from coursescribe.vocabulary import WORD_LIST_CREATE_FAILED, WORD_LIST_UNREADABLE, ensure_word_list

WORD_NOTE = "写常用词，或“听错 → 正确”，校对时自动替换"
AUTO_NOTES_TEXT = "转录完自动生成整理笔记"
RULES_NOTE = "整理笔记时交给模型的规则，改完保存，下次整理时生效"


class KeyRecorder(QLineEdit):
    """Read-only box that turns a key press into candidate shortcut text; it decides nothing itself."""

    captured = Signal(str)

    def __init__(self, platform: str) -> None:
        super().__init__()
        self._platform = platform
        self.setReadOnly(True)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()  # Ctrl+letter is for recording here, not for the window's own shortcuts
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            event.accept()  # holding a key down must not flash the reason
            return
        key, modifiers = event.key(), event.modifiers()
        if is_leaving_press(key, modifiers):
            event.ignore()  # Tab moves on, Esc cancels the dialog, Enter is the dialog's own
            return
        text = candidate_text(key, modifiers, self._platform)
        event.accept()
        if text is not None:  # a modifier alone: keep waiting
            self.captured.emit(text)


class SettingsDialog(QDialog):
    def __init__(
        self,
        current: Settings,
        proofread_models: Iterable[str],
        choose_folder: Callable[[], Path | None] | None = None,
        parent: QWidget | None = None,
        word_list: Path | None = None,
        open_file: Callable[[Path], None] | None = None,
        notes_rules: Path | None = None,
        hotkey_platform: str = HOTKEY_PLATFORM,
        default_hotkey: str = DEFAULT_HOTKEY,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(620)
        self._folder = current.output_dir
        self._choose_folder = choose_folder or self._pick_folder
        self._word_list = word_list
        self._open_file = open_file
        self._notes_rules = notes_rules

        self.transcribe_box = QComboBox()
        for name in MODELS:
            self.transcribe_box.addItem(name, name)
        self.transcribe_box.setCurrentText(current.transcribe_model)
        self.size_label = QLabel()
        self.size_label.setProperty("muted", True)
        self.transcribe_box.currentTextChanged.connect(self._show_size)
        self._show_size(self.transcribe_box.currentText())

        self.proofread_box = QComboBox()
        self.proofread_box.setEditable(True)
        models = sorted(set(proofread_models))  # read once: it may be a generator
        self.proofread_box.addItems(models)
        self.proofread_box.setCurrentText(current.proofread_model)

        self.notes_box = QComboBox()
        self.notes_box.setEditable(True)  # Ollama may be closed: the name can still be typed
        self.notes_box.addItems(models)
        self.notes_box.setCurrentText(current.notes_model)
        self.auto_notes_box = QCheckBox(AUTO_NOTES_TEXT)
        self.auto_notes_box.setChecked(current.auto_notes)

        self.folder_label = QLabel(str(current.output_dir))
        self.folder_label.setWordWrap(True)
        self.folder_button = QPushButton("更改…")
        self.folder_button.clicked.connect(self._change_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(self.folder_button)

        self.word_list_button = QPushButton("打开词表…")
        self.word_list_button.clicked.connect(self._open_word_list)
        self.word_list_button.setEnabled(word_list is not None and open_file is not None)
        self.word_note = QLabel(WORD_NOTE)
        self.word_note.setWordWrap(True)  # a problem with the file is a long sentence
        self.word_note.setProperty("muted", True)
        word_row = QHBoxLayout()
        word_row.addWidget(self.word_note, 1)
        word_row.addWidget(self.word_list_button)

        self.rules_button = QPushButton("打开整理规则…")
        self.rules_button.clicked.connect(self._open_rules)
        self.rules_button.setEnabled(notes_rules is not None and open_file is not None)
        self.rules_note = QLabel(RULES_NOTE)
        self.rules_note.setWordWrap(True)  # a problem with the file is a long sentence
        self.rules_note.setProperty("muted", True)
        rules_row = QHBoxLayout()
        rules_row.addWidget(self.rules_note, 1)
        rules_row.addWidget(self.rules_button)

        self._platform = hotkey_platform
        self._default_shortcut = parse(default_hotkey, hotkey_platform)
        self._shortcut = self._stored_shortcut(current.hotkey)
        self.hotkey_box = KeyRecorder(hotkey_platform)
        self.hotkey_box.captured.connect(self._on_captured)
        self.hotkey_reset_button = QPushButton("恢复默认")
        self.hotkey_reset_button.clicked.connect(self._reset_hotkey)
        self.hotkey_reason = QLabel()
        self.hotkey_reason.setWordWrap(True)
        self.hotkey_reason.setProperty("muted", True)
        hotkey_row = QHBoxLayout()
        hotkey_row.addWidget(self.hotkey_box, 1)
        hotkey_row.addWidget(self.hotkey_reset_button)
        self._show_shortcut()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        set_role(buttons.button(QDialogButtonBox.Ok), "primary")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        note = QLabel("修改后，从下一个任务开始生效。\n快捷键保存后立即生效。")
        note.setProperty("muted", True)

        form = QFormLayout(self)
        form.setContentsMargins(24, 22, 24, 18)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.addRow("转录模型", self.transcribe_box)
        form.addRow("", self.size_label)
        form.addRow("校对模型", self.proofread_box)
        form.addRow("笔记模型", self.notes_box)
        form.addRow("", self.auto_notes_box)
        form.addRow("保存到", folder_row)
        form.addRow("词表", word_row)
        form.addRow("整理规则", rules_row)
        form.addRow("录音快捷键", hotkey_row)
        form.addRow("", self.hotkey_reason)
        form.addRow(note)
        form.addRow(buttons)

    def _stored_shortcut(self, text: str) -> Shortcut:
        try:
            return parse(text, self._platform)
        except InvalidShortcut:
            return self._default_shortcut  # a value that cannot be used here is never shown or returned

    def _show_shortcut(self) -> None:
        self.hotkey_box.setText(self._shortcut.display(self._platform))

    def _on_captured(self, text: str) -> None:
        try:
            self._shortcut = parse(text, self._platform)
        except InvalidShortcut as exc:
            self.hotkey_reason.setText(exc.reason)  # the stored value stays
            return
        self.hotkey_reason.setText("")
        self._show_shortcut()

    def _reset_hotkey(self) -> None:
        self._shortcut = self._default_shortcut
        self.hotkey_reason.setText("")
        self._show_shortcut()

    def _show_size(self, name: str) -> None:
        self.size_label.setText(f"没下载过的模型，用到时会先询问（{model_size_label(name)}）")

    def _pick_folder(self) -> Path | None:
        chosen = QFileDialog.getExistingDirectory(self, "选择保存文件夹", str(self._folder))
        return Path(chosen) if chosen else None

    def _change_folder(self) -> None:
        chosen = self._choose_folder()
        if chosen:
            self._folder = chosen
            self.folder_label.setText(str(chosen))

    def _open_word_list(self) -> None:
        if self._word_list is None or self._open_file is None:
            return
        if self._word_list.is_dir():
            self.word_note.setText(WORD_LIST_UNREADABLE.format(path=self._word_list))
            return
        try:
            ensure_word_list(self._word_list)
        except OSError:
            self.word_note.setText(WORD_LIST_CREATE_FAILED.format(path=self._word_list))
            return
        self.word_note.setText(WORD_NOTE)
        self._open_file(self._word_list)

    def _open_rules(self) -> None:
        if self._notes_rules is None or self._open_file is None:
            return
        try:
            ensure_notes_rules(self._notes_rules)
        except NotesRulesUnreadable as exc:
            self.rules_note.setText(str(exc))
            return
        self.rules_note.setText(RULES_NOTE)
        self._open_file(self._notes_rules)

    def result_settings(self) -> Settings:
        return Settings(
            self._folder,
            self.transcribe_box.currentText(),
            self.proofread_box.currentText().strip(),
            self.notes_box.currentText().strip() or DEFAULT_NOTES_MODEL,  # emptied: back to the default
            self.auto_notes_box.isChecked(),
            self._shortcut.serialize(),
        )
