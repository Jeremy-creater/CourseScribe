"""Put the application together and open the window."""

from __future__ import annotations

import logging
import logging.handlers
import os
import threading
from pathlib import Path

from PySide6.QtCore import QLockFile, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from coursescribe.controller import MEDIA_SUFFIXES, AppController
from coursescribe.jobs import JobStore
from coursescribe.ollama import OllamaClient, OllamaError
from coursescribe.platform import (
    HOTKEY_LABEL,
    create_global_hotkey,
    open_system_audio,
    prepare_gpu_libraries,
)
from coursescribe.proofread import OllamaProofreader
from coursescribe.settings import SettingsStore, app_home, default_settings
from coursescribe.transcribe import WhisperTranscriber, download_whisper_model
from coursescribe.ui.main_window import MainWindow
from coursescribe.ui.style import apply_style
from coursescribe.vocabulary import FILE_NAME as WORD_LIST_FILE
from coursescribe.vocabulary import load_vocabulary

logger = logging.getLogger(__name__)


def _set_up_logging(home: Path) -> None:
    folder = home / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        folder / "coursescribe.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def _choose_files(parent) -> list[Path]:
    patterns = " ".join(f"*{suffix}" for suffix in sorted(MEDIA_SUFFIXES))
    names, _ = QFileDialog.getOpenFileNames(parent, "选择音视频文件", "", f"音视频文件 ({patterns})")
    return [Path(name) for name in names]


def _list_proofread_models() -> list[str]:
    try:
        return sorted(name for name in OllamaClient(timeout=2.0).models() if name)
    except OllamaError as exc:
        logger.info("Could not list Ollama models: %s", exc)
        return []


def run() -> int:
    home = app_home()
    home.mkdir(parents=True, exist_ok=True)
    _set_up_logging(home)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    prepare_gpu_libraries()

    app = QApplication([])
    apply_style(app)
    lock = QLockFile(str(home / "running.lock"))
    if not lock.tryLock(0):
        QMessageBox.information(None, "CourseScribe", "CourseScribe 已经打开了。")
        return 0

    stopping = threading.Event()
    word_list = home / WORD_LIST_FILE
    controller = AppController(
        store=JobStore(home / "jobs"),
        settings=SettingsStore(home / "settings.json", default_settings()),
        transcriber=WhisperTranscriber(should_stop=stopping.is_set),
        proofreader=OllamaProofreader(vocabulary=lambda: load_vocabulary(word_list)),
        open_capture=open_system_audio,
        download_model=download_whisper_model,
        stopping=stopping,
    )
    window = None

    def confirm(question: str) -> bool:
        answer = QMessageBox.question(window, "下载模型", question)
        return answer == QMessageBox.StandardButton.Yes

    window = MainWindow(
        controller,
        hotkey_label=HOTKEY_LABEL,
        choose_files=lambda: _choose_files(window),
        confirm=confirm,
        open_folder=lambda folder: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))),
        list_proofread_models=_list_proofread_models,
        word_list=word_list,
    )
    hotkey = create_global_hotkey(window.hotkey_pressed.emit)
    if not hotkey.start():
        window.hotkey_label.setText("快捷键被别的程序占用了，请用按钮。也可以把文件拖到这里")
    window.show()
    controller.start()
    try:
        return app.exec()
    finally:
        hotkey.stop()
