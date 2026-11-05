"""The test groups and repository rules that CLAUDE.md promises."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "coursescribe"
UNIT_FILTER = "not engine and not integration and not gui and not perf"
PLUGIN = ("-p", "no_skip_plugin")


@pytest.fixture
def group_samples(pytester):
    pytester.makepyfile(
        """
        import pytest

        def test_plain(): pass

        @pytest.mark.engine
        def test_engine(): pass

        @pytest.mark.integration
        def test_integration(): pass

        @pytest.mark.gui
        def test_gui(): pass

        @pytest.mark.perf
        def test_perf(): pass
        """
    )
    pytester.makeini(
        "[pytest]\nmarkers =\n  engine: x\n  integration: x\n  gui: x\n  perf: x\n"
    )
    return pytester


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (UNIT_FILTER, "test_plain"),
        ("engine", "test_engine"),
        ("integration", "test_integration"),
        ("gui", "test_gui"),
        ("perf", "test_perf"),
    ],
)
def test_each_group_selects_only_its_own_tests(group_samples, expression, expected):
    result = group_samples.runpytest("-m", expression, *PLUGIN)
    result.assert_outcomes(passed=1, deselected=4)
    verbose = group_samples.runpytest("-m", expression, "-v", *PLUGIN)
    assert re.findall(r"::(test_\w+) PASSED", verbose.stdout.str()) == [expected]


def test_a_skipped_test_is_reported_as_failed(pytester):
    pytester.makepyfile(
        """
        import pytest

        def test_needs_ollama():
            pytest.skip("Ollama is not running")
        """
    )
    result = pytester.runpytest(*PLUGIN)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Ollama is not running*"])


def test_missing_dependency_fails_with_a_hint(pytester):
    pytester.makepyfile(
        """
        from no_skip_plugin import require_module

        def test_needs_module():
            require_module("module_that_does_not_exist", "install it with X")
        """
    )
    result = pytester.runpytest(*PLUGIN)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*module_that_does_not_exist*install it with X*"])


def test_require_module_returns_an_available_module():
    from no_skip_plugin import require_module

    assert require_module("json", "unused").dumps([1]) == "[1]"


def test_package_starts_as_a_module():
    done = subprocess.run(
        [sys.executable, "-m", "coursescribe", "--version"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "CourseScribe 0.1.0"


def _imported_modules(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def _sources_outside(folder: str) -> list[Path]:
    return [p for p in SRC.rglob("*.py") if folder not in p.relative_to(SRC).parts]


def test_only_the_platform_folder_checks_the_operating_system():
    pattern = re.compile(r"sys\.platform|os\.name\b|platform\.system|import platform\b")
    offenders = [p.name for p in _sources_outside("platform") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_platform_factory_is_the_only_entry_with_system_imports():
    system_modules = {"soundcard", "ctypes", "winreg"}
    offenders = [
        p.name for p in _sources_outside("platform") if _imported_modules(p) & system_modules
    ]
    assert offenders == []


def test_only_the_ui_folder_imports_qt():
    offenders = [
        p.name
        for p in _sources_outside("ui")
        if any(name.split(".")[0] == "PySide6" for name in _imported_modules(p))
    ]
    assert offenders == []


def test_pyqt_is_never_used():
    offenders = [
        p.name
        for p in SRC.rglob("*.py")
        if any(name.split(".")[0] == "PyQt6" for name in _imported_modules(p))
    ]
    assert offenders == []



def _names_tempfile(path: Path) -> bool:
    """``import tempfile``, ``from tempfile import …``, or the name as a string (``import_module``)."""
    if any(name.split(".")[0] == "tempfile" for name in _imported_modules(path)):
        return True
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(isinstance(node, ast.Constant) and node.value == "tempfile" for node in ast.walk(tree))


def test_program_code_never_imports_tempfile():
    offenders = [p.name for p in SRC.rglob("*.py") if _names_tempfile(p)]
    assert offenders == [], (
        f"{offenders}: on Windows tempfile retries without end in a folder that refuses new files; "
        "see docs/plans/M2-write-permission-hang-tests.md"
    )

def _tracked_files() -> list[str]:
    done = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=REPO, check=True
    )
    return [name for name in done.stdout.decode("utf-8").split("\0") if name]


def test_repository_tracks_no_media_or_claude_files():
    forbidden = re.compile(r"\.(wav|mp3|m4a|mp4|mkv)$|^\.claude/", re.IGNORECASE)
    assert [name for name in _tracked_files() if forbidden.search(name)] == []


def test_repository_holds_no_personal_absolute_paths():
    pattern = re.compile(r"[A-Za-z]:[\/](?:Users|DevLocal|LLM|Sync)\b|/Users/[a-z]|/home/[a-z]")
    offenders = []
    for name in _tracked_files():
        path = REPO / name
        if path.suffix in {".py", ".md", ".toml", ".txt", ".json", ".spec", ".ps1", ".cfg", ".ini"}:
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(name)
    assert offenders == []


def test_unexpected_errors_in_the_window_reach_the_log_and_the_user(caplog, monkeypatch):
    import logging
    import sys
    import threading

    from coursescribe.ui.app import install_error_reporting

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    told = []
    install_error_reporting(lambda: told.append("told"))
    try:
        raise RuntimeError("from a button")
    except RuntimeError as exc:
        with caplog.at_level(logging.ERROR):
            sys.excepthook(type(exc), exc, exc.__traceback__)
    assert "from a button" in caplog.text
    assert told == ["told"]


def test_unexpected_errors_on_a_background_thread_reach_the_log_and_the_user(caplog, monkeypatch):
    """A packaged program has no console, so the default (printing) shows nothing."""
    import logging
    import sys
    import threading

    from coursescribe.ui.app import install_error_reporting

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    told = []
    install_error_reporting(lambda: told.append("told"))

    def fail():
        raise RuntimeError("from the worker")

    with caplog.at_level(logging.ERROR):
        thread = threading.Thread(target=fail)
        thread.start()
        thread.join()
    assert "from the worker" in caplog.text
    assert told == ["told"]
