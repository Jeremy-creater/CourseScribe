"""The real OneDrive folders this computer knows about, and the default save folder they lead to."""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

_VARIABLES = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
_ACCOUNTS = r"Software\Microsoft\OneDrive\Accounts"
_SCRATCH = r"Software\CourseScribeTest"


def _winreg():
    if sys.platform != "win32":
        pytest.fail("Finding the Mac's synced folders is checked in the Mac phase (see M2-sync-folder-tests.md).")
    import winreg

    return winreg


def _registered_folders(key_path: str = _ACCOUNTS) -> list[str]:
    """Every account's UserFolder, read here independently of the program."""
    winreg = _winreg()
    folders = []
    try:
        accounts = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
    except FileNotFoundError:
        return folders
    with accounts:
        count = winreg.QueryInfoKey(accounts)[0]
        for index in range(count):
            with winreg.OpenKey(accounts, winreg.EnumKey(accounts, index)) as account:
                try:
                    folders.append(winreg.QueryValueEx(account, "UserFolder")[0])
                except FileNotFoundError:
                    pass
    return folders


def _windows():
    _winreg()
    from coursescribe.platform import windows

    return windows


def test_every_onedrive_folder_on_this_computer_is_found():
    windows = _windows()
    roots = windows.cloud_sync_roots()
    for name in _VARIABLES:
        if os.environ.get(name):
            assert os.environ[name] in roots
    for folder in _registered_folders():
        assert folder in roots
    print("OneDrive folders found:", [r for r in roots if r])


def test_onedrive_folder_set_after_login_is_found(tmp_path, monkeypatch):
    windows = _windows()
    monkeypatch.setenv("OneDrive", str(tmp_path / "od"))
    monkeypatch.setenv("OneDriveCommercial", "")
    monkeypatch.delenv("OneDriveConsumer", raising=False)
    assert str(tmp_path / "od") in windows.cloud_sync_roots()


def test_onedrive_folder_known_only_to_the_registry_is_found(monkeypatch):
    windows = _windows()
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    registered = _registered_folders()
    roots = windows.cloud_sync_roots()
    for folder in registered:
        assert folder in roots
    if not registered:
        print("No OneDrive account in the registry here: this test proved nothing.")


@pytest.fixture()
def scratch_accounts(monkeypatch):
    """A throwaway registry key standing in for OneDrive's account list; removed afterwards."""
    winreg = _winreg()
    windows = _windows()
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = _SCRATCH + r"\Accounts"
    monkeypatch.setattr(windows, "_ONEDRIVE_ACCOUNTS", path)

    def remove(key_path: str) -> None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                children = [winreg.EnumKey(key, i) for i in range(winreg.QueryInfoKey(key)[0])]
        except FileNotFoundError:
            return
        for child in children:
            remove(key_path + "\\" + child)
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)

    remove(_SCRATCH)
    yield winreg, path
    remove(_SCRATCH)


def test_registry_without_accounts_finds_nothing(scratch_accounts):
    windows = _windows()
    assert [r for r in windows.cloud_sync_roots() if r] == []


def test_registry_accounts_with_and_without_a_folder(scratch_accounts, tmp_path):
    winreg, path = scratch_accounts
    windows = _windows()
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path + r"\Personal") as key:
        winreg.SetValueEx(key, "UserFolder", 0, winreg.REG_SZ, str(tmp_path / "OneDrive"))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path + r"\Business1") as key:
        winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "not signed in yet")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path + r"\Business2") as key:
        winreg.SetValueEx(key, "UserFolder", 0, winreg.REG_DWORD, 1)
    assert [r for r in windows.cloud_sync_roots() if r] == [str(tmp_path / "OneDrive")]


def test_registry_folder_written_with_variables_is_expanded(scratch_accounts, monkeypatch, tmp_path):
    winreg, path = scratch_accounts
    windows = _windows()
    monkeypatch.setenv("COURSESCRIBE_TEST_DIR", str(tmp_path))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path + r"\Personal") as key:
        winreg.SetValueEx(key, "UserFolder", 0, winreg.REG_EXPAND_SZ, r"%COURSESCRIBE_TEST_DIR%\OneDrive")
    assert [r for r in windows.cloud_sync_roots() if r] == [str(tmp_path / "OneDrive")]


def _parts(path: str) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", os.path.normpath(path)).casefold().split(os.sep))


def test_default_save_folder_on_this_computer(tmp_path):
    winreg = _winreg()
    from coursescribe.settings import open_settings

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
    ) as key:
        documents = winreg.ExpandEnvironmentStrings(winreg.QueryValueEx(key, "Personal")[0])
    onedrive = [os.environ[n] for n in _VARIABLES if os.environ.get(n)] + _registered_folders()
    target = _parts(os.path.join(documents, "CourseScribe"))
    synced = any(target[: len(_parts(root))] == _parts(root) for root in onedrive)
    output_dir = open_settings(tmp_path / "settings.json").load().output_dir
    print("Documents:", documents, "| synced:", synced, "| default save folder:", output_dir)
    expected = Path.home() / "CourseScribe" if synced else Path(documents) / "CourseScribe"
    assert str(output_dir) == str(expected)
