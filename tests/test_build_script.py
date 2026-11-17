"""packaging/build.py: which license files go into the bundle, and the Mac app's version."""

from __future__ import annotations

import importlib.util
import plistlib
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("coursescribe_build", ROOT / "packaging" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDist:
    def __init__(self, files: list[str]):
        self.files = [PurePosixPath(f) for f in files]


def targets(build, files: list[str]) -> list[str]:
    return sorted(inner.as_posix() for inner, _ in build._license_files(FakeDist(files)))


def test_license_folder_keeps_texts_but_not_python_bytecode(build):
    # pip compiles .py files in a dist-info licenses folder (PyAV ships AUTHORS.py); the .pyc
    # holds the absolute path of the install, which must not ship.
    files = [
        "av-17.1.0.dist-info/licenses/AUTHORS.py",
        "av-17.1.0.dist-info/licenses/AUTHORS.rst",
        "av-17.1.0.dist-info/licenses/LICENSE.txt",
        "av-17.1.0.dist-info/licenses/__pycache__/AUTHORS.cpython-312.pyc",
        "av-17.1.0.dist-info/METADATA",
    ]
    assert targets(build, files) == ["licenses/AUTHORS.py", "licenses/AUTHORS.rst", "licenses/LICENSE.txt"]


def test_package_level_fallback_skips_bytecode_too(build):
    files = ["onnxruntime/LICENSE", "onnxruntime/__pycache__/NOTICE.cpython-312.pyc", "x-1.0.dist-info/METADATA"]
    assert targets(build, files) == ["LICENSE"]


def test_stamp_version_writes_both_version_keys_and_keeps_the_rest(build, tmp_path):
    plist = tmp_path / "Info.plist"
    plist.write_bytes(plistlib.dumps({"CFBundleName": "CourseScribe", "CFBundleShortVersionString": "0.0.0"}))
    build.stamp_version(plist, "0.1.0")
    data = plistlib.loads(plist.read_bytes())
    assert data == {"CFBundleName": "CourseScribe", "CFBundleShortVersionString": "0.1.0", "CFBundleVersion": "0.1.0"}
