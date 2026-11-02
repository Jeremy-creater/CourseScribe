"""A light, solid look in the style of Apple's own apps, the same on Windows and Mac.

The palette is forced light, so a dark system theme cannot leave dark text on
a dark window. Buttons take their look from a ``role`` property: ``primary``
(blue), ``danger`` (red) or none (quiet grey).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QWidget

COLORS = {
    "background": "#f5f5f7",
    "surface": "#ffffff",
    "border": "#e5e5ea",
    "control": "#e8e8ed",
    "control_pressed": "#d8d8de",
    "text": "#1d1d1f",
    "secondary": "#6e6e73",
    "accent": "#007aff",
    "accent_pressed": "#0062cc",
    "danger": "#ff3b30",
    "danger_pressed": "#d70015",
    "warning_text": "#d70015",
    "warning_fill": "#fff1f0",
}

_C = COLORS
# Shipped next to this file; the packaging spec copies it. Once a combo box is
# styled, Qt draws no arrow unless it is given an image.
ARROW_IMAGE = Path(__file__).with_name("chevron.svg")
STYLE_SHEET = f"""
QWidget {{
    color: {_C["text"]};
    font-size: 14px;
}}
QWidget#mainWindow, QDialog {{
    background: {_C["background"]};
}}
QLabel#title {{
    font-size: 22px;
    font-weight: 600;
}}
QLabel[muted="true"] {{
    color: {_C["secondary"]};
    font-size: 13px;
}}
QLabel#recording {{
    color: {_C["danger"]};
    font-weight: 600;
}}
QLabel#status {{
    font-size: 15px;
    font-weight: 500;
}}
QLabel#silence {{
    color: {_C["warning_text"]};
    background: {_C["warning_fill"]};
    border-radius: 8px;
    padding: 8px 10px;
    font-weight: 600;
}}
QFrame#card {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    border-radius: 12px;
}}
QFrame#card QLabel {{
    background: transparent;
}}
QPushButton {{
    background: {_C["control"]};
    color: {_C["text"]};
    border: none;
    border-radius: 10px;
    padding: 8px 18px;
    min-height: 22px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {_C["control_pressed"]};
}}
QPushButton:pressed {{
    background: {_C["control_pressed"]};
}}
QPushButton:disabled {{
    color: {_C["secondary"]};
}}
QPushButton[role="primary"] {{
    background: {_C["accent"]};
    color: white;
    font-size: 16px;
    font-weight: 600;
}}
QPushButton[role="primary"]:pressed {{
    background: {_C["accent_pressed"]};
}}
QPushButton[role="danger"] {{
    background: {_C["danger"]};
    color: white;
    font-size: 16px;
    font-weight: 600;
}}
QPushButton[role="danger"]:pressed {{
    background: {_C["danger_pressed"]};
}}
QProgressBar {{
    background: {_C["control"]};
    border: none;
    border-radius: 3px;
    max-height: 6px;
    min-height: 6px;
}}
QProgressBar::chunk {{
    background: {_C["accent"]};
    border-radius: 3px;
}}
QComboBox, QLineEdit {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    border-radius: 8px;
    padding: 5px 10px;
    min-height: 22px;
}}
QComboBox:focus, QLineEdit:focus {{
    border: 1px solid {_C["accent"]};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 26px;
    border: none;
}}
QComboBox::down-arrow {{
    image: url("{ARROW_IMAGE.as_posix()}");
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    selection-background-color: {_C["accent"]};
    selection-color: white;
}}
"""


def light_palette() -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.Window: _C["background"],
        QPalette.WindowText: _C["text"],
        QPalette.Base: _C["surface"],
        QPalette.AlternateBase: _C["background"],
        QPalette.Text: _C["text"],
        QPalette.Button: _C["control"],
        QPalette.ButtonText: _C["text"],
        QPalette.Highlight: _C["accent"],
        QPalette.HighlightedText: "#ffffff",
        QPalette.ToolTipBase: _C["surface"],
        QPalette.ToolTipText: _C["text"],
        QPalette.PlaceholderText: _C["secondary"],
        QPalette.Link: _C["accent"],
    }
    for role, colour in roles.items():
        palette.setColor(role, QColor(colour))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(_C["secondary"]))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(_C["secondary"]))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(_C["secondary"]))
    return palette


def apply_style(app: QApplication) -> None:
    app.setStyle("Fusion")  # draws the same on Windows and Mac, and is what the tests run
    app.setPalette(light_palette())
    app.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
    app.setStyleSheet(STYLE_SHEET)


def set_role(widget: QWidget, role: str | None) -> None:
    """Change a button's look; the style sheet only reacts after a re-polish."""
    if widget.property("role") == role:
        return
    widget.setProperty("role", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
