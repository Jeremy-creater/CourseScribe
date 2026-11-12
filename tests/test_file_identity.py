"""WI-2: the same-file key. Two names for one file must give one key, two files never share one.

The Mac asks the disk (a volume may ignore letter case or not); the Windows key is today's
``normcase(realpath(path))``. Checks that need a real Windows are in ``WindowsOnly`` below
and are collected on Windows only (a skip would count as a failure in this project)."""

from __future__ import annotations

import errno
import os
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from coursescribe.platform.file_identity import mac_file_identity_key as key
from coursescribe.platform.file_identity import windows_file_identity_key

SEED = 20261112
SRC = Path(__file__).resolve().parent.parent / "src" / "coursescribe"


def make(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def blind(folder: Path) -> bool:
    """The real disk's answer, independent of the product: does it ignore letter case here?"""
    probe = make(folder / "case-probe.md")
    upper = folder / "CASE-PROBE.MD"
    return upper.exists() and os.path.samefile(upper, probe)


def fake_stat(monkeypatch, table: dict[str, tuple[int, int]]) -> None:
    """Narrow stand-in for the OS call: fabricated (st_dev, st_ino) for the named paths, the real call otherwise."""
    real = os.stat

    def fake(path, *args, **kwargs):
        found = table.get(os.fspath(path))
        if found is None:
            return real(path, *args, **kwargs)
        dev, ino = found
        return os.stat_result((0o100644, ino, dev, 1, 0, 0, 0, 0, 0, 0))

    monkeypatch.setattr(os, "stat", fake)


# ---------------------------------------------------------------- M1 - M4: existing files


def test_two_spellings_of_one_existing_file_share_a_key(tmp_path):
    a = make(tmp_path / "a.md")
    upper = tmp_path / "A.MD"
    if blind(tmp_path):
        assert key(a) == key(upper)  # the case-blind branch: the same file under another spelling
    else:
        assert not upper.exists()  # the case-sensitive branch: A.MD is a missing file, judged by spelling (M5)
        assert key(a) != key(tmp_path / "b.md")


def test_different_existing_files_never_share_a_key(tmp_path):
    a = make(tmp_path / "a.md")
    b = make(tmp_path / "b.md")
    elsewhere = make(tmp_path / "sub" / "a.md")
    long_name = make(tmp_path / "a.MD.txt")
    accented = make(tmp_path / "résumé.md")
    plain = make(tmp_path / "resume.md")
    files = [a, b, elsewhere, long_name, accented, plain]
    keys = [key(f) for f in files]
    for i, left in enumerate(keys):
        for j, right in enumerate(keys):
            assert (left == right) == (i == j), (files[i], files[j])


def test_a_symlink_and_its_target_share_a_key(tmp_path):
    target = make(tmp_path / "real" / "a.md")
    link = tmp_path / "link.md"
    link.symlink_to(target)
    link2 = tmp_path / "link2.md"
    link2.symlink_to(link)
    folder_link = tmp_path / "folder-link"
    folder_link.symlink_to(target.parent, target_is_directory=True)
    assert key(link) == key(target) == key(link2) == key(folder_link / "a.md")


def test_a_hard_link_is_the_same_file(tmp_path):
    a = make(tmp_path / "a.md")
    b = tmp_path / "b.md"
    os.link(a, b)
    assert key(a) == key(b)


# ---------------------------------------------------------------- M5, M6, M13, M14: files that are not there


def test_a_missing_file_compares_by_spelling(tmp_path):
    gone = tmp_path / "gone.md"
    assert key(gone) == key(tmp_path / "GONE.md")
    assert key(gone) == key(tmp_path / "x" / ".." / "gone.md")
    assert key(gone) != key(tmp_path / "other.md")


@pytest.mark.parametrize(
    "left, right",
    [("é.md", "e\u0301.md"), ("讲.md", "讲.md"), ("\uac00.md", "\u1100\u1161.md")],
)
def test_nfc_and_nfd_spellings_share_a_key_for_missing_and_real_files(tmp_path, left, right):
    assert key(tmp_path / left) == key(tmp_path / right)  # both missing
    made = make(tmp_path / "real" / left)
    assert key(made) == key(tmp_path / "real" / right)  # one real, asked for under the other spelling


def test_the_fallback_is_not_nfkc(tmp_path):
    assert key(tmp_path / "ａ.md") != key(tmp_path / "a.md")  # fullwidth a is another name
    assert key(tmp_path / "ﬁle.md") == key(tmp_path / "file.md")  # casefold itself merges the ligature


def test_missing_files_in_differently_spelled_or_linked_folders_share_a_key(tmp_path):
    # 1. folder letter case
    (tmp_path / "Dir").mkdir()
    left, right = tmp_path / "Dir" / "gone.md", tmp_path / "DIR" / "gone.md"
    # Where the disk says DIR is the same folder the keys agree because it is one file; where DIR is another,
    # missing folder they agree by the casefold rule. Either way: equal.
    assert key(left) == key(right)
    # 2. symlinked folder
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    assert key(tmp_path / "link" / "gone.md") == key(tmp_path / "real" / "gone.md")
    # 3. dangling symlink
    (tmp_path / "dangling.md").symlink_to(tmp_path / "real" / "gone.md")
    assert key(tmp_path / "dangling.md") == key(tmp_path / "real" / "gone.md")
    # 4. folder NFC vs NFD, folder real and folder missing
    (tmp_path / "é").mkdir()
    assert key(tmp_path / "é" / "gone.md") == key(tmp_path / "e\u0301" / "gone.md")
    assert key(tmp_path / "ö" / "gone.md") == key(tmp_path / "o\u0308" / "gone.md")


def test_sharp_s_and_ss_share_a_key_when_missing_on_purpose(tmp_path):
    # DOCUMENTED BEHAVIOUR, not an accident: the spelling fallback uses NFC + casefold, which folds ß to ss.
    # APFS treats ß and ss as different files, so this can say "same file" for two names that could both
    # exist. The effect is small and only reaches files that do not exist (existing files use the disk).
    assert key(tmp_path / "ß.md") == key(tmp_path / "ss.md")
    assert key(tmp_path / "STRASSE.md") == key(tmp_path / "straße.md")
    assert key(tmp_path / "ß.md") != key(tmp_path / "s.md")


# ---------------------------------------------------------------- M7: the equivalence relation, with an oracle


PIECES = ["a", "A", "é", "e\u0301", "讲", "ß", "İ", "ǅ", " ", "-", "1"]


def oracle(p: Path, q: Path) -> bool:
    """What the disk says, written without the product's formula."""
    if p.exists() and q.exists():
        return os.path.samefile(p, q)

    def split(path: Path):
        rest = []
        while not path.exists():
            rest.append(unicodedata.normalize("NFC", path.name).casefold())
            path = path.parent
        return path, tuple(reversed(rest))

    (pa, pr), (qa, qr) = split(p), split(q)
    return os.path.samefile(pa, qa) and pr == qr


def test_the_key_is_an_equivalence_relation_that_agrees_with_the_disk(tmp_path):
    rng = random.Random(SEED)
    paths: list[Path] = []
    for _ in range(40):
        depth = rng.randint(1, 3)
        parts = ["".join(rng.choice(PIECES) for _ in range(rng.randint(1, 12))) for _ in range(depth)]
        path = tmp_path.joinpath(*parts)
        if rng.random() < 0.5:
            if path.exists() or any(q.is_file() for q in path.parents):
                continue
            try:
                make(path)
            except OSError:
                continue
        paths.append(path)
    # the same files named again by the other case, so some pairs really are equal
    paths += [Path(str(p).upper()) for p in list(paths)[:12]] + [Path(str(p)) for p in paths[:8]]
    assert any(p.exists() for p in paths) and any(not p.exists() for p in paths)
    keys = [key(p) for p in paths]
    equal = unequal = missing_pairs = existing_pairs = 0
    for i, p in enumerate(paths):
        assert key(p) == keys[i], (SEED, p)  # reflexive and stable
        assert key(Path(str(p))) == keys[i], (SEED, p)
        for j, q in enumerate(paths):
            same = keys[i] == keys[j]
            assert same == (keys[j] == keys[i]), (SEED, p, q)  # symmetric
            assert same == oracle(p, q), (SEED, p, q)
            equal += same and i != j
            unequal += not same
            if i != j and same:
                existing_pairs += p.exists()
                missing_pairs += not p.exists()
    assert equal >= 1 and unequal >= 100 and missing_pairs >= 1 and existing_pairs >= 1, (equal, unequal, missing_pairs, existing_pairs)
    for i in range(len(paths)):  # transitive over triples that share a key
        group = [j for j in range(len(paths)) if keys[j] == keys[i]]
        assert all(oracle(paths[group[0]], paths[j]) for j in group)


# ---------------------------------------------------------------- M8, M9: it never raises


@pytest.mark.parametrize("error", [PermissionError(errno.EACCES, "denied"), OSError(errno.EIO, "io"), FileNotFoundError(errno.ENOENT, "gone")])
def test_a_failing_stat_falls_back_to_the_spelling_key(tmp_path, monkeypatch, error):
    path = tmp_path / "x.md"
    expected = key(path)  # missing: the spelling key

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(os, "stat", broken)
    assert key(path) == expected
    assert key(tmp_path / "X.MD") == expected


def test_a_file_that_vanishes_between_looking_and_asking_gets_the_spelling_key(tmp_path):
    path = make(tmp_path / "a.md")
    path.unlink()
    assert key(path) == key(tmp_path / "A.MD")


def test_the_odd_paths_each_get_a_key(tmp_path):
    parent_is_file = make(tmp_path / "plain.md")
    odd = [
        tmp_path / ("n" * 300 + ".md"),
        Path(str(tmp_path) + "/" + "/".join("d" * 200 for _ in range(10)) + "/x.md"),
        Path(""),
        Path("a\x00b"),
        parent_is_file / "inside.md",
    ]
    for path in odd:
        assert key(path) == key(path), path
    assert key(odd[0]) != key(odd[1])
    assert key(parent_is_file / "inside.md") == key(parent_is_file / "INSIDE.md")


def test_a_folder_that_cannot_be_searched_still_gives_a_key(tmp_path):
    locked = tmp_path / "locked"
    inside = make(locked / "a.md")
    locked.chmod(0)
    try:
        assert key(inside) == key(locked / "A.MD")  # the disk cannot be asked: judged by spelling
        assert key(inside) != key(tmp_path / "other.md")
    finally:
        locked.chmod(0o700)


# ---------------------------------------------------------------- M10: not cached


def test_the_key_follows_the_disk_each_time_it_is_asked(tmp_path):
    a = make(tmp_path / "a.md")
    first = key(a)
    renamed = tmp_path / "A.md"
    os.rename(a, renamed)
    assert key(renamed) == first  # one file, one key, whatever its name now
    inode = os.stat(renamed).st_ino
    renamed.unlink()
    new = make(tmp_path / "a.md")
    if os.stat(new).st_ino != inode:
        assert key(new) != first  # a different file now: the old identity must not stick to the name
    else:
        assert (key(new) == first) == os.path.samefile(new, new)  # inode reused: the disk says same


# ---------------------------------------------------------------- M11, M12: a case-sensitive disk, simulated


def test_two_files_the_disk_tells_apart_are_two_keys(tmp_path, monkeypatch):
    # Stand-in for a case-sensitive volume (a real one needs a disk image): the OS call is narrowed to
    # fabricated identities for exactly these two names.
    a, upper = str(tmp_path / "a.md"), str(tmp_path / "A.MD")
    fake_stat(monkeypatch, {a: (1, 7), upper: (1, 8)})
    assert key(tmp_path / "a.md") != key(tmp_path / "A.MD")


def test_both_the_device_and_the_inode_count(tmp_path, monkeypatch):
    p = lambda name: str(tmp_path / name)
    fake_stat(monkeypatch, {p("a"): (1, 7), p("b"): (2, 7), p("c"): (1, 8), p("d"): (1, 7), p("e"): (3, 9), p("f"): (3, 9)})
    assert key(tmp_path / "a") != key(tmp_path / "b")  # same inode, other device
    assert key(tmp_path / "a") != key(tmp_path / "c")  # same device, other inode
    assert key(tmp_path / "a") == key(tmp_path / "d")
    assert key(tmp_path / "e") == key(tmp_path / "f")


# ---------------------------------------------------------------- W-1, W-1b, W-2: the Windows key


WINDOWS_ROWS = [
    "C:\\Out\\A.md", "c:/out/a.md", "C:/Out/../Out/A.md", "\\\\host\\share\\A.md", "Z:\\x.md",
    "C:\\" + "d" * 300, "", "C:\\讲\\课.md", "C:\\É.md", "/tmp/Some/File.MD",
]


@pytest.mark.parametrize("row", WINDOWS_ROWS)
def test_the_windows_key_is_the_old_expression(row, tmp_path):
    assert windows_file_identity_key(row) == os.path.normcase(os.path.realpath(row))
    real = make(tmp_path / "Real.MD")
    assert windows_file_identity_key(real) == os.path.normcase(os.path.realpath(real))
    assert windows_file_identity_key(tmp_path / "Missing.MD") == os.path.normcase(os.path.realpath(tmp_path / "Missing.MD"))


def test_the_windows_key_applies_normcase_once_to_the_realpath(tmp_path, monkeypatch):
    # On the Mac normcase does nothing, so a marker stands in for it: a key that drops it would not carry the marker.
    monkeypatch.setattr(os.path, "normcase", lambda s: "N:" + s)
    real = make(tmp_path / "Real.md")
    for path in (real, tmp_path / "Gone.md"):
        assert windows_file_identity_key(path) == "N:" + os.path.realpath(path)


if sys.platform == "win32":

    class TestWindowsOnly:
        """W-2: needs the real Windows path rules and winreg (through windows.py)."""

        def test_windows_uses_the_pure_key_and_folds_case(self):
            from coursescribe.platform import windows  # fails loudly if winreg cannot be imported

            assert windows.file_identity_key is windows_file_identity_key
            for row in WINDOWS_ROWS:
                assert windows.file_identity_key(row) == os.path.normcase(os.path.realpath(row))
            assert windows.file_identity_key("C:\\Out\\A.md") == windows.file_identity_key("c:/out/a.md")
            windows.file_identity_key("Z:\\x.md")


# ---------------------------------------------------------------- P1: the export, and C4: no OS test in the controller


def test_platform_exports_the_key_of_the_running_system(tmp_path):
    from coursescribe import platform as system

    a = make(tmp_path / "a.md")
    if sys.platform == "win32":
        assert system.file_identity_key(a) == windows_file_identity_key(a)
    else:
        from coursescribe.platform import macos

        assert system.file_identity_key is macos.file_identity_key
        assert system.file_identity_key(a) == key(a)
        if blind(tmp_path):
            assert system.file_identity_key(a) == system.file_identity_key(tmp_path / "A.MD")


def test_importing_platform_off_windows_does_not_load_the_windows_module():
    if sys.platform == "win32":
        return
    code = "import sys; import coursescribe.platform.file_identity; print('winreg' in sys.modules)"
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_only_the_platform_folder_tests_the_operating_system():
    controller = (SRC / "controller.py").read_text(encoding="utf-8")
    for banned in (r"sys\.platform", r"os\.name", r"platform\.system", r"normcase"):
        assert not re.search(banned, controller), banned
    for path in SRC.rglob("*.py"):
        if "platform" in path.relative_to(SRC).parts[:1]:
            continue
        assert not re.search(r"sys\.platform|os\.name\b", path.read_text(encoding="utf-8")), path
