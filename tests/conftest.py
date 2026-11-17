"""Shared test configuration."""

from __future__ import annotations

from no_skip_plugin import pytest_runtest_makereport  # noqa: F401  (registers the hook)


import sys  # noqa: E402

import pytest  # noqa: E402

# These files drive a real Mac (Core Audio, Carbon shortcuts, caffeinate). On any other system they
# cannot even be imported, so they are left out of collection there. On a Mac they run with the gui group.
collect_ignore_glob = [] if sys.platform == "darwin" else ["test_mac_*_desktop.py"]


@pytest.fixture(autouse=True)
def windows_path_limits(monkeypatch):
    """The name and path tests were written for Windows' rules; they hold on any system. The Mac's rules have their own tests."""
    from coursescribe import jobs
    from coursescribe.platform.limits import PathLimits

    monkeypatch.setattr(jobs, "LIMITS", PathLimits(path_units=259, component_bytes=None, path_bytes=None))
