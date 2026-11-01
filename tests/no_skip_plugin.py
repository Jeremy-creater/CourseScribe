"""A missing dependency must fail loudly: a skip here would look like a pass."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def require_module(name: str, hint: str) -> ModuleType:
    """Import ``name``, or fail the test with a message saying how to fix it."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.fail(f"Missing dependency {name}: {hint} ({exc})")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
        report.outcome = "failed"
        report.longrepr = f"Skipped tests count as failures in this project: {reason}"
