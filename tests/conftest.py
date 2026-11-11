"""Shared test configuration."""

from __future__ import annotations

from no_skip_plugin import pytest_runtest_makereport  # noqa: F401  (registers the hook)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def windows_path_limits(monkeypatch):
    """The name and path tests were written for Windows' rules; they hold on any system. The Mac's rules have their own tests."""
    from coursescribe import jobs
    from coursescribe.platform.limits import PathLimits

    monkeypatch.setattr(jobs, "LIMITS", PathLimits(path_units=259, component_bytes=None, path_bytes=None))
