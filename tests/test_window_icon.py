"""The running window carries the app icon itself, not only the .exe resource (taskbar, title bar)."""

from pathlib import Path

from PySide6.QtWidgets import QApplication

from coursescribe.ui.style import app_icon

ROOT = Path(__file__).resolve().parents[1]


def test_app_icon_loads_as_a_large_square_picture() -> None:
    QApplication.instance() or QApplication([])
    icon = app_icon()
    assert not icon.isNull()
    size = icon.actualSize(icon.availableSizes()[-1])
    assert size.width() == size.height() >= 256


def test_the_icon_picture_is_bundled_in_the_package_build() -> None:
    assert "app-icon.png" in (ROOT / "packaging" / "coursescribe.spec").read_text(encoding="utf-8")
