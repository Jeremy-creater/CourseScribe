"""Build the app icons from packaging/icons/source.png.

Run: python packaging/make_icons.py   (needs PySide6; the Mac .icns also needs the iconutil tool)
Outputs, next to the source: icon-mac-1024.png, CourseScribe.icns (Mac only), CourseScribe.ico.

The source art is cropped a little tighter and re-centred on the symbol, then drawn as a rounded
square: with a margin for the Mac (the system does not round .icns icons itself), almost full-bleed
for Windows.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QPainterPath

HERE = Path(__file__).resolve().parent / "icons"
# The symbol's centre in the 1254-pixel source, and how much of the source to keep (symbol ~10% larger).
CENTER = (644, 618)
CROP = 1140
CORNER = 0.225  # corner radius as a share of the icon's side
WINDOWS_SIZES = [16, 24, 32, 48, 64, 128, 256]


def cropped(source: QImage) -> QImage:
    x, y = CENTER[0] - CROP // 2, CENTER[1] - CROP // 2
    return source.copy(x, y, CROP, CROP)


def rounded(art: QImage, side: int, margin: int) -> QImage:
    out = QImage(side, side, QImage.Format_ARGB32)
    out.fill(Qt.transparent)
    inner = side - 2 * margin
    painter = QPainter(out)
    painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    path = QPainterPath()
    path.addRoundedRect(QRectF(margin, margin, inner, inner), inner * CORNER, inner * CORNER)
    painter.setClipPath(path)
    painter.drawImage(QRectF(margin, margin, inner, inner), art)
    painter.end()
    return out


def png_bytes(image: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data)


def write_ico(path: Path, master: QImage) -> None:
    pngs = [(s, png_bytes(master.scaled(s, s, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))) for s in WINDOWS_SIZES]
    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, blobs = b"", b""
    for size, blob in pngs:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(blob), offset)
        blobs += blob
        offset += len(blob)
    path.write_bytes(header + entries + blobs)


def write_icns(path: Path, master: QImage) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "CourseScribe.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                side = base * scale
                scaled = master.scaled(side, side, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                scaled.save(str(iconset / f"icon_{base}x{base}{suffix}.png"), "PNG")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(path)], check=True)


def main() -> None:
    QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    source = QImage(str(HERE / "source.png"))
    if source.isNull():
        raise SystemExit("packaging/icons/source.png is missing or unreadable")
    art = cropped(source).scaled(1024, 1024, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    mac = rounded(art, 1024, 100)
    mac.save(str(HERE / "icon-mac-1024.png"), "PNG")
    write_ico(HERE / "CourseScribe.ico", rounded(art, 1024, 24))
    if shutil.which("iconutil"):
        write_icns(HERE / "CourseScribe.icns", mac)
    else:
        print("iconutil not found: skipped the .icns (build it on a Mac)")


if __name__ == "__main__":
    main()
