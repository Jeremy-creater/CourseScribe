"""Where files are saved by default, and how saved settings keep the user's own choice."""

from __future__ import annotations

import itertools
import json
import logging
import os
import unicodedata
from pathlib import Path

import pytest

import coursescribe.settings as settings_module
from coursescribe.models import DEFAULT_PROOFREAD_MODEL, DEFAULT_TRANSCRIBE_MODEL
from coursescribe.settings import Settings, SettingsStore, choose_output_dir, default_settings, open_settings

ROOT = Path(__file__).resolve().parents[1]


def _same(result: Path, expected: Path) -> None:
    """Also compares the spelling: on Windows, Path == ignores case."""
    assert result == expected
    assert str(result) == str(expected)


@pytest.fixture()
def home(tmp_path):
    return tmp_path / "Users" / "a"


# ---- choose_output_dir: pure logic --------------------------------------------------


@pytest.mark.parametrize("has_onedrive", [False, True])
def test_documents_not_synced_keeps_documents(home, has_onedrive):
    documents = home / "Documents"
    roots = [str(home / "OneDrive")] if has_onedrive else []
    _same(choose_output_dir(documents, home, roots), documents / "CourseScribe")


@pytest.mark.parametrize(
    "documents_parts, root_parts",
    [
        (("OneDrive", "Documents"), ("OneDrive",)),
        (("OneDrive - 某公司", "文档"), ("OneDrive - 某公司",)),
        (
            ("Library", "Mobile Documents", "com~apple~CloudDocs", "Documents"),
            ("Library", "Mobile Documents", "com~apple~CloudDocs"),
        ),
    ],
)
def test_synced_documents_moves_to_home(home, documents_parts, root_parts):
    documents = home.joinpath(*documents_parts)
    roots = ["", str(home.joinpath(*root_parts))]
    _same(choose_output_dir(documents, home, roots), home / "CourseScribe")


@pytest.mark.parametrize("which", ["documents", "target", "home"])
def test_a_sync_folder_equal_to_documents_target_or_home_counts(home, which):
    documents = home / "Documents"
    root = {"documents": documents, "target": documents / "CourseScribe", "home": home}[which]
    _same(choose_output_dir(documents, home, [str(root)]), home / "CourseScribe")


@pytest.mark.parametrize(
    "documents_parts, root_parts",
    [
        (("OneDrive2", "Documents"), ("OneDrive",)),
        (("OneDrive", "Documents"), ("One",)),
        (("Documents",), ("Documents", "OneDrive")),
        (("Documents",), ("Documents", "CourseScribe-backup")),
    ],
)
def test_names_that_only_look_alike_do_not_count(home, documents_parts, root_parts):
    documents = home.joinpath(*documents_parts)
    _same(choose_output_dir(documents, home, [str(home.joinpath(*root_parts))]), documents / "CourseScribe")


def test_the_same_folder_written_differently_still_counts(home):
    documents = home / "OneDrive" / "Documents"
    for root in (
        str(home / "OneDrive").upper(),
        str(home / "OneDrive") + os.sep,
        str(home / "x" / ".." / "OneDrive"),
    ):
        _same(choose_output_dir(documents, home, [root]), home / "CourseScribe")


@pytest.mark.parametrize("documents_form, root_form", [("NFC", "NFD"), ("NFD", "NFC")])
def test_accented_names_match_in_either_unicode_form(tmp_path, documents_form, root_form):
    def at(form: str) -> Path:
        return tmp_path / unicodedata.normalize(form, "Andrés")

    documents = at(documents_form) / "OneDrive" / "Documents"
    root = str(at(root_form) / "OneDrive")
    _same(choose_output_dir(documents, at("NFC"), [root]), at("NFC") / "CourseScribe")


def test_meaningless_sync_entries_are_ignored(home):
    documents = home / "Documents"
    for roots in ([""], ["   "], ["OneDrive"], ["."], ["", "   ", "OneDrive", ".", str(home / "OneDrive")]):
        _same(choose_output_dir(documents, home, roots), documents / "CourseScribe")


@pytest.mark.parametrize("documents", [None, "", "Documents"])
def test_unknown_or_odd_documents_folder_uses_home(home, documents):
    _same(choose_output_dir(documents, home, []), home / "CourseScribe")


def test_unknown_sync_folders_count_as_synced(home):
    _same(choose_output_dir(home / "Documents", home, None), home / "CourseScribe")


def test_a_whole_drive_synced_covers_everything_on_it(home):
    root = home.anchor or os.sep
    _same(choose_output_dir(home / "Documents", home, [root]), home / "CourseScribe")


def test_every_combination_follows_the_rule(home):
    icloud = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    candidates = [
        str(home / "OneDrive"),
        str(home / "OneDrive").upper(),
        str(icloud) + os.sep,
        str(home / "Documents" / "OneDrive"),
        "",
        "OneDrive",
    ]
    # Which candidates contain each Documents folder, written by hand rather than computed.
    documents_and_hits = [
        (home / "Documents", set()),
        (home / "OneDrive" / "Documents", {0, 1}),
        (home / "OneDrive2" / "Documents", set()),
        (icloud / "Documents", {2}),
        (None, None),
    ]
    subsets = [
        list(chosen)
        for size in range(len(candidates) + 1)
        for chosen in itertools.combinations(range(len(candidates)), size)
    ]
    checked = 0
    for documents, hits in documents_and_hits:
        for chosen in [*subsets, None]:
            roots = None if chosen is None else [candidates[i] for i in chosen]
            synced = documents is None or chosen is None or bool(hits & set(chosen))
            expected = home / "CourseScribe" if synced else documents / "CourseScribe"
            result = choose_output_dir(documents, home, roots)
            assert str(result) == str(expected), (documents, roots)
            if roots:
                reordered = choose_output_dir(documents, home, [r for r in reversed(roots) for _ in range(2)])
                assert str(reordered) == str(expected), (documents, roots)
            checked += 1
    assert checked == 5 * 65


# ---- default_settings: system values handed to the rule -----------------------------


@pytest.fixture()
def system(tmp_path, monkeypatch):
    """The user's home folder points into tmp_path; Documents and sync folders are set per test."""
    user = tmp_path / "home"
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("USERPROFILE", str(user))

    class System:
        home = user
        documents: object = str(user / "OneDrive" / "Documents")
        roots: object = [str(user / "OneDrive")]

    def documents():
        if isinstance(System.documents, Exception):
            raise System.documents
        return System.documents

    def roots():
        if isinstance(System.roots, Exception):
            raise System.roots
        return System.roots

    monkeypatch.setattr(settings_module, "user_documents_dir", documents)
    monkeypatch.setattr(settings_module, "cloud_sync_roots", roots)
    return System


def test_default_folder_follows_the_system(tmp_path, system):
    assert default_settings() == Settings(system.home / "CourseScribe", DEFAULT_TRANSCRIBE_MODEL, DEFAULT_PROOFREAD_MODEL)
    _same(default_settings().output_dir, system.home / "CourseScribe")
    system.roots = []
    _same(default_settings().output_dir, system.home / "OneDrive" / "Documents" / "CourseScribe")
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "where, error",
    [
        ("documents", OSError("known folder lookup failed")),
        ("documents", RuntimeError("unexpected documents")),
        ("roots", RuntimeError("unexpected roots")),
    ],
)
def test_failed_system_lookups_save_to_home_and_are_logged(system, caplog, where, error):
    system.roots = []
    setattr(system, where, error)
    with caplog.at_level(logging.WARNING, logger="coursescribe.settings"):
        settings = default_settings()
    _same(settings.output_dir, system.home / "CourseScribe")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(error) in warnings[0].getMessage()


# ---- SettingsStore: the user's own folder vs. following the default -----------------


def _write(path: Path, data) -> bytes:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path.read_bytes()


@pytest.fixture()
def synced(tmp_path):
    """Documents is now in OneDrive; the old version saved Documents\\CourseScribe."""
    docs = tmp_path / "OneDrive" / "Documents"
    defaults = Settings(tmp_path / "CourseScribe", "large-v3", "m")
    return docs, defaults


@pytest.mark.parametrize("form", ["same", "spelled differently", "missing"])
def test_old_saved_default_follows_the_new_default_and_keeps_models(tmp_path, synced, form):
    docs, defaults = synced
    path = tmp_path / "settings.json"
    data = {"transcribe_model": "small", "proofread_model": "qwen3:8b"}
    if form == "same":
        data["output_dir"] = str(docs / "CourseScribe")
    elif form == "spelled differently":
        data["output_dir"] = str(docs / "CourseScribe").upper() + os.sep
    before = _write(path, data)
    store = SettingsStore(path, defaults, old_default_dir=docs / "CourseScribe")
    loaded = store.load()
    assert loaded == Settings(tmp_path / "CourseScribe", "small", "qwen3:8b")
    _same(loaded.output_dir, tmp_path / "CourseScribe")
    assert path.read_bytes() == before


def test_a_folder_the_user_chose_in_an_old_version_is_kept(tmp_path, synced):
    docs, defaults = synced
    path = tmp_path / "settings.json"
    chosen = tmp_path / "OneDrive" / "课堂录音"
    _write(path, {"output_dir": str(chosen), "transcribe_model": "small", "proofread_model": "m"})
    loaded = SettingsStore(path, defaults, old_default_dir=docs / "CourseScribe").load()
    _same(loaded.output_dir, chosen)


def test_a_folder_chosen_in_this_version_is_kept_even_inside_onedrive(tmp_path, synced):
    docs, defaults = synced
    path = tmp_path / "settings.json"
    store = SettingsStore(path, defaults, old_default_dir=docs / "CourseScribe")
    store.save(Settings(docs / "CourseScribe", "small", "m"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["format"] == 2
    assert saved["output_dir"] == str(docs / "CourseScribe")
    _same(store.load().output_dir, docs / "CourseScribe")


def test_old_default_still_follows_the_default_when_documents_is_unknown(tmp_path, synced):
    """Without Documents the old default cannot be recognised exactly; a CourseScribe folder may be in OneDrive."""
    docs, defaults = synced
    path = tmp_path / "settings.json"
    _write(path, {"output_dir": str(docs / "CourseScribe"), "transcribe_model": "small", "proofread_model": "m"})
    _same(SettingsStore(path, defaults, old_default_dir=None).load().output_dir, defaults.output_dir)
    chosen = tmp_path / "D盘" / "课"
    _write(path, {"output_dir": str(chosen), "transcribe_model": "small", "proofread_model": "m"})
    _same(SettingsStore(path, defaults, old_default_dir=None).load().output_dir, chosen)


def test_a_newer_settings_format_is_not_mistaken_for_an_old_one(tmp_path, synced):
    docs, defaults = synced
    path = tmp_path / "settings.json"
    _write(path, {"format": 3, "output_dir": str(docs / "CourseScribe"), "transcribe_model": "s", "proofread_model": "m"})
    _same(SettingsStore(path, defaults, old_default_dir=docs / "CourseScribe").load().output_dir, docs / "CourseScribe")


def test_saving_only_new_models_does_not_lock_in_the_default_folder(tmp_path):
    path = tmp_path / "settings.json"
    before = Settings(tmp_path / "Documents" / "CourseScribe", "large-v3", "m")
    SettingsStore(path, before).save(Settings(before.output_dir, "small", "m2"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["format"] == 2
    assert "output_dir" not in saved
    after = Settings(tmp_path / "CourseScribe", "large-v3", "m")
    assert SettingsStore(path, after).load() == Settings(tmp_path / "CourseScribe", "small", "m2")


def test_saving_and_loading_keep_choices_and_follow_defaults(tmp_path):
    a = tmp_path / "Documents" / "CourseScribe"
    b = tmp_path / "CourseScribe"
    folders = [a, b, tmp_path / "OneDrive" / "x", tmp_path / "D盘" / "课"]
    model_pairs = [("large-v3", "m"), ("small", "qwen3:8b")]
    for number, (d1, d2, folder, models) in enumerate(itertools.product([a, b], [a, b], folders, model_pairs)):
        path = tmp_path / f"settings-{number}.json"
        SettingsStore(path, Settings(d1, "large-v3", "m")).save(Settings(folder, *models))
        before = path.read_bytes()
        store = SettingsStore(path, Settings(d2, "large-v3", "m"))
        loaded = store.load()
        expected = d2 if folder == d1 else folder
        case = (d1, d2, folder, models)
        assert (loaded.transcribe_model, loaded.proofread_model) == models, case
        assert str(loaded.output_dir) == str(expected), case
        assert store.load() == loaded, case
        assert path.read_bytes() == before, case


@pytest.mark.parametrize(
    "content",
    [
        {"format": 2, "output_dir": 5, "transcribe_model": "small", "proofread_model": "m"},
        {"format": 2, "output_dir": "", "transcribe_model": "small", "proofread_model": "m"},
        {"format": 2, "transcribe_model": "small"},
        [],
        {"format": 2, "output_dir": "CourseScribe", "transcribe_model": "small", "proofread_model": "m"},
        {"output_dir": "", "transcribe_model": "small", "proofread_model": "m"},
        {"format": 2, "transcribe_model": 5, "proofread_model": "m"},
    ],
)
def test_damaged_settings_fall_back_to_defaults_with_a_warning(tmp_path, caplog, content):
    path = tmp_path / "settings.json"
    before = _write(path, content)
    defaults = Settings(tmp_path / "CourseScribe", "large-v3", "m")
    with caplog.at_level(logging.WARNING, logger="coursescribe.settings"):
        assert SettingsStore(path, defaults).load() == defaults
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    assert path.read_bytes() == before


# ---- open_settings: everything put together ------------------------------------------


def test_open_settings_moves_an_old_default_out_of_onedrive(tmp_path, system):
    path = tmp_path / "app" / "settings.json"
    path.parent.mkdir()
    old = system.home / "OneDrive" / "Documents" / "CourseScribe"
    _write(path, {"output_dir": str(old), "transcribe_model": "small", "proofread_model": "m"})
    loaded = open_settings(path).load()
    _same(loaded.output_dir, system.home / "CourseScribe")
    assert loaded.transcribe_model == "small"
    assert not (system.home / "CourseScribe").exists()


@pytest.mark.parametrize("documents", [OSError("known folder lookup failed"), ""])
def test_open_settings_survives_an_unknown_documents_folder(tmp_path, system, caplog, documents):
    path = tmp_path / "settings.json"
    old = system.home / "OneDrive" / "Documents" / "CourseScribe"
    _write(path, {"output_dir": str(old), "transcribe_model": "small", "proofread_model": "m"})
    system.documents = documents
    with caplog.at_level(logging.WARNING, logger="coursescribe.settings"):
        store = open_settings(path)
    assert store.old_default_dir is None
    _same(store.load().output_dir, system.home / "CourseScribe")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Documents" in warnings[0].getMessage()


def test_the_program_opens_settings_through_open_settings():
    source = (ROOT / "src" / "coursescribe" / "ui" / "app.py").read_text(encoding="utf-8")
    assert "open_settings(" in source
    assert "SettingsStore(" not in source


# ---- the Mac's sync folder ------------------------------------------------------------


def test_mac_sync_folder_is_icloud_drive_under_home(tmp_path, monkeypatch):
    from coursescribe.platform import macos

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    roots = macos.cloud_sync_roots()
    assert roots == [str(tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs")]
    assert list(tmp_path.iterdir()) == []
