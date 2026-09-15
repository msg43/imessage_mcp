from __future__ import annotations

from pathlib import Path

from imsg.paths import is_contained_in, is_same_file, join_under_root, resolve_path


def test_resolve_path_expands_user(monkeypatch: object, tmp_path: Path) -> None:
    import os

    os.environ["HOME"] = str(tmp_path)
    assert resolve_path("~/foo") == (tmp_path / "foo").resolve()


def test_resolve_path_normalizes_dotdot(tmp_path: Path) -> None:
    resolved = resolve_path(tmp_path / "a" / ".." / "b")
    assert resolved == tmp_path / "b"


def test_is_contained_in_simple_child(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "sub" / "file.txt"
    assert is_contained_in(child, root)


def test_is_contained_in_root_itself(tmp_path: Path) -> None:
    assert is_contained_in(tmp_path, tmp_path)


def test_is_contained_in_rejects_sibling(tmp_path: Path) -> None:
    root = tmp_path / "root"
    sibling = tmp_path / "root-but-not-really"
    assert not is_contained_in(sibling, root)


def test_is_contained_in_defeats_dotdot_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    escaping = root / ".." / "outside"
    assert not is_contained_in(escaping, root)


def test_is_contained_in_defeats_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    link = root / "escape_link"
    link.symlink_to(outside)

    # As a string this looks contained ("root/escape_link"); once symlinks
    # are resolved it is not — this is exactly the case SPEC §5.4 calls out.
    assert not is_contained_in(link, root)
    assert not is_contained_in(link / "some_file", root)


def test_join_under_root_relative(tmp_path: Path) -> None:
    assert join_under_root(tmp_path, Path("sub/file.txt")) == tmp_path / "sub" / "file.txt"


def test_join_under_root_absolute_outside_root_stays_absolute(tmp_path: Path) -> None:
    outside = Path("/etc/passwd")
    # pathlib semantics: joining with an absolute path discards the root,
    # which is exactly what lets is_contained_in catch the escape afterward.
    assert join_under_root(tmp_path, outside) == outside
    assert not is_contained_in(join_under_root(tmp_path, outside), tmp_path)


def test_is_same_file_identical_and_dotdot_paths(tmp_path: Path) -> None:
    target = tmp_path / "chat.db"
    target.write_text("")
    assert is_same_file(target, target)
    assert is_same_file(tmp_path / "sub" / ".." / "chat.db", target)


def test_is_same_file_follows_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "chat.db"
    target.write_text("")
    link = tmp_path / "alias.db"
    link.symlink_to(target)
    assert is_same_file(link, target)


def test_is_same_file_sees_through_hard_links(tmp_path: Path) -> None:
    """A hard link resolves to a different string for the same inode —
    exactly the shape of the macOS firmlink alias of `~/Library/Messages`."""
    import os

    target = tmp_path / "chat.db"
    target.write_text("")
    link = tmp_path / "alias.db"
    os.link(target, link)
    assert resolve_path(link) != resolve_path(target)  # resolve() alone would miss it
    assert is_same_file(link, target)


def test_is_same_file_rejects_distinct_files(tmp_path: Path) -> None:
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    a.write_text("")
    b.write_text("")
    assert not is_same_file(a, b)


def test_is_same_file_with_missing_paths(tmp_path: Path) -> None:
    """A missing path can still match by resolved-path equality (the guard
    must refuse a protected path whether or not the file exists), but a
    missing path never matches a different, existing file."""
    existing = tmp_path / "chat.db"
    existing.write_text("")
    missing = tmp_path / "nope.db"
    assert is_same_file(missing, tmp_path / "sub" / ".." / "nope.db")
    assert not is_same_file(missing, existing)
    assert not is_same_file(existing, missing)
