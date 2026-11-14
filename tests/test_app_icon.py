"""The app icon files exist, are well formed, and the packaging spec uses them."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "packaging" / "icons"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == PNG_MAGIC
    return struct.unpack(">II", data[16:24])


def test_png_master_is_1024_square():
    assert png_size((ICONS / "icon-mac-1024.png").read_bytes()) == (1024, 1024)


def test_icns_has_the_icns_header_and_matching_length():
    data = (ICONS / "CourseScribe.icns").read_bytes()
    assert data[:4] == b"icns"
    assert struct.unpack(">I", data[4:8])[0] == len(data)


def test_ico_lists_every_standard_size_each_as_a_real_png():
    data = (ICONS / "CourseScribe.ico").read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind) == (0, 1)
    sizes = []
    for i in range(count):
        w, h, _c, _r, _planes, _bpp, length, offset = struct.unpack("<BBBBHHII", data[6 + 16 * i : 22 + 16 * i])
        w, h = w or 256, h or 256
        assert png_size(data[offset : offset + length]) == (w, h)
        sizes.append(w)
    assert sorted(sizes) == [16, 24, 32, 48, 64, 128, 256]


def test_corners_are_transparent_and_center_is_opaque():
    from PySide6.QtGui import QImage

    image = QImage(str(ICONS / "icon-mac-1024.png"))
    assert image.pixelColor(0, 0).alpha() == 0
    assert image.pixelColor(512, 512).alpha() == 255


def test_spec_points_the_exe_and_the_mac_bundle_at_the_icons():
    spec = (ROOT / "packaging" / "coursescribe.spec").read_text(encoding="utf-8")
    assert "CourseScribe.ico" in spec
    assert "CourseScribe.icns" in spec
    for name in ("CourseScribe.ico", "CourseScribe.icns"):
        assert (ICONS / name).exists()


@pytest.mark.parametrize(
    ("name", "limit"), [("CourseScribe.ico", 200_000), ("CourseScribe.icns", 1_500_000), ("icon-mac-1024.png", 600_000)]
)
def test_icons_are_small_enough_to_commit(name, limit):
    assert (ICONS / name).stat().st_size < limit


def test_no_stray_absolute_path_in_the_generator():
    text = (ROOT / "packaging" / "make_icons.py").read_text(encoding="utf-8")
    assert "/Users/" not in text
