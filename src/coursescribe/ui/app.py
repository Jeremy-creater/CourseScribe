"""Put the application together and open the window."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QLockFile, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from coursescribe.controller import AppController, KeepAwake
from coursescribe.jobs import JobStore
from coursescribe.notes_rules import load_notes_rules, rules_path
from coursescribe.notes_writer import OllamaNotesWriter
from coursescribe.ollama import OllamaClient, OllamaError
from coursescribe.platform import (
    HOTKEY_LABEL,
    create_global_hotkey,
    create_keep_awake,
    open_system_audio,
    prepare_gpu_libraries,
)
from coursescribe.proofread import OllamaProofreader
from coursescribe.settings import app_home, open_settings
from coursescribe.transcribe import WhisperTranscriber, download_whisper_model
from coursescribe.ui.main_window import FILE_DIALOG_TITLE, FILE_FILTER, MainWindow
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


def install_error_reporting(notify: Callable[[], None] | None = None) -> None:
    """Errors nobody caught: a packaged program has no console, so log them and tell the user."""

    def report(info) -> None:
        logger.error("Unexpected error", exc_info=info)
        if notify is not None:
            try:
                notify()
            except Exception:
                logger.exception("Could not show the unexpected error")

    def on_window_error(kind, error, trace) -> None:
        report((kind, error, trace))

    def on_thread_error(args) -> None:
        if args.exc_type is not SystemExit:
            report((args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = on_window_error
    threading.excepthook = on_thread_error


def _choose_files(parent) -> list[Path]:
    names, _ = QFileDialog.getOpenFileNames(parent, FILE_DIALOG_TITLE, "", FILE_FILTER)
    return [Path(name) for name in names]


def _list_proofread_models() -> list[str]:
    try:
        return sorted(name for name in OllamaClient(timeout=2.0).models() if name)
    except OllamaError as exc:
        logger.info("Could not list Ollama models: %s", exc)
        return []


def build_controller(home: Path, stopping: threading.Event, keep_awake: KeepAwake | None = None) -> AppController:
    """The application behind the window. ``keep_awake`` is for tests that must not ask the system."""
    word_list = home / WORD_LIST_FILE
    return AppController(
        store=JobStore(home / "jobs"),
        settings=open_settings(home / "settings.json"),
        transcriber=WhisperTranscriber(should_stop=stopping.is_set),
        proofreader=OllamaProofreader(vocabulary=lambda: load_vocabulary(word_list)),
        open_capture=open_system_audio,
        download_model=download_whisper_model,
        stopping=stopping,
        keep_awake=keep_awake if keep_awake is not None else create_keep_awake(),
        notes_writer=OllamaNotesWriter(rules=lambda: load_notes_rules(rules_path())),
    )


def run() -> int:
    home = app_home()
    home.mkdir(parents=True, exist_ok=True)
    _set_up_logging(home)
    install_error_reporting()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    prepare_gpu_libraries()

    app = QApplication([])
    apply_style(app)
    lock = QLockFile(str(home / "running.lock"))
    if not lock.tryLock(0):
        QMessageBox.information(None, "CourseScribe", "CourseScribe 已经打开了。")
        return 0

    stopping = threading.Event()
    controller = build_controller(home, stopping)
    install_error_reporting(controller.report_unexpected_error)
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
        word_list=home / WORD_LIST_FILE,
        notes_rules=rules_path(),
    )
    hotkey = create_global_hotkey(window.hotkey_pressed.emit)
    if not hotkey.start() and HOTKEY_LABEL:  # a system without a shortcut says nothing about one being taken
        window.hotkey_label.setText("快捷键被别的程序占用了，请用按钮。也可以把文件拖到这里")
    window.show()
    controller.start()
    try:
        return app.exec()
    finally:
        hotkey.stop()
