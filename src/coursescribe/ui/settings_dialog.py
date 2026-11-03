"""Settings: transcription model, proofreading model and save folder."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from coursescribe.models import MODELS, model_size_label
from coursescribe.settings import Settings
from coursescribe.ui.style import set_role
from coursescribe.vocabulary import WORD_LIST_CREATE_FAILED, WORD_LIST_UNREADABLE, ensure_word_list

WORD_NOTE = "写常用词，或“听错 → 正确”，校对时自动替换"


class SettingsDialog(QDialog):
    def __init__(
        self,
        current: Settings,
        proofread_models: Iterable[str],
        choose_folder: Callable[[], Path | None] | None = None,
        parent: QWidget | None = None,
        word_list: Path | None = None,
        open_file: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(620)
        self._folder = current.output_dir
        self._choose_folder = choose_folder or self._pick_folder
        self._word_list = word_list
        self._open_file = open_file

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
        self.proofread_box.addItems(sorted(set(proofread_models)))
        self.proofread_box.setCurrentText(current.proofread_model)

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

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        set_role(buttons.button(QDialogButtonBox.Ok), "primary")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        note = QLabel("修改后，从下一个任务开始生效。")
        note.setProperty("muted", True)

        form = QFormLayout(self)
        form.setContentsMargins(24, 22, 24, 18)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.addRow("转录模型", self.transcribe_box)
        form.addRow("", self.size_label)
        form.addRow("校对模型", self.proofread_box)
        form.addRow("保存到", folder_row)
        form.addRow("词表", word_row)
        form.addRow(note)
        form.addRow(buttons)

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

    def result_settings(self) -> Settings:
        return Settings(self._folder, self.transcribe_box.currentText(), self.proofread_box.currentText().strip())
