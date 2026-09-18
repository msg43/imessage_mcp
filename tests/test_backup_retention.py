"""The retention rule (`imsg.backup.retention`), proved at its boundaries.

Retention is the only part of `imsg backup` that deletes anything, and
it runs unattended at 04:00, so the tests that matter are the ones that
show it *declining* to delete: a half-written set from an interrupted
run, a staging directory, an operator's own file, a symlink, and — the
one that would hurt most — the newest set.

Fictional personas only (D5).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imsg.backup.retention import (
    DEFAULT_KEEP,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    STAGING_PREFIX,
    apply_retention,
    index_backups,
    plan_retention,
    read_manifest,
)


def make_set(
    backups: Path,
    *,
    stamp: str,
    uid: str = "aaaaaa",
    complete: bool = True,
    manifest: dict[str, object] | None = None,
    write_manifest: bool = True,
) -> Path:
    """A directory shaped like a backup set. `complete=False` / a bespoke
    `manifest` / `write_manifest=False` produce the partial variants."""
    path = backups / f"backup-{stamp}-{uid}"
    path.mkdir(parents=True)
    (path / "postgres.dump").write_bytes(b"not-a-real-dump")
    if not write_manifest:
        return path
    body = manifest if manifest is not None else {
        "format": MANIFEST_FORMAT,
        "complete": complete,
        "created_at": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T04:00:00+00:00",
        "files": [],
    }
    (path / MANIFEST_FILENAME).write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture
def backups(tmp_path: Path) -> Path:
    d = tmp_path / "data_root" / "backups"
    d.mkdir(parents=True)
    return d


def _stamps(n: int) -> list[str]:
    """`n` daily stamps, oldest first."""
    return [f"202609{day:02d}T040000Z" for day in range(1, n + 1)]


# ---------------------------------------------------------------------------
# What retention keeps
# ---------------------------------------------------------------------------


def test_fewer_sets_than_keep_deletes_nothing(backups: Path) -> None:
    for stamp in _stamps(3):
        make_set(backups, stamp=stamp)
    plan = plan_retention(index_backups(backups), keep=DEFAULT_KEEP)
    assert plan.deletes_nothing
    assert len(plan.retained) == 3


def test_exactly_keep_sets_deletes_nothing(backups: Path) -> None:
    """The boundary that matters on night fourteen."""
    for stamp in _stamps(DEFAULT_KEEP):
        make_set(backups, stamp=stamp)
    plan = plan_retention(index_backups(backups), keep=DEFAULT_KEEP)
    assert plan.deletes_nothing
    assert len(plan.retained) == DEFAULT_KEEP


def test_keep_plus_one_deletes_exactly_the_oldest(backups: Path) -> None:
    """Night fifteen: one deletion, and it is the oldest, not an arbitrary one."""
    stamps = _stamps(DEFAULT_KEEP + 1)
    for stamp in stamps:
        make_set(backups, stamp=stamp)
    plan = plan_retention(index_backups(backups), keep=DEFAULT_KEEP)
    assert len(plan.delete) == 1
    assert plan.delete[0].name == f"backup-{stamps[0]}-aaaaaa"
    assert len(plan.retained) == DEFAULT_KEEP


def test_the_newest_set_is_never_a_candidate(backups: Path) -> None:
    stamps = _stamps(20)
    for stamp in stamps:
        make_set(backups, stamp=stamp)
    newest = backups / f"backup-{stamps[-1]}-aaaaaa"
    for keep in (1, 2, DEFAULT_KEEP, 19):
        plan = plan_retention(index_backups(backups), keep=keep)
        assert newest not in plan.delete
        assert plan.retained[0] == newest


def test_keep_zero_is_clamped_so_one_set_always_survives(backups: Path) -> None:
    """A misconfigured `--keep 0` prunes hard but can never leave nothing."""
    stamps = _stamps(3)
    for stamp in stamps:
        make_set(backups, stamp=stamp)
    plan = plan_retention(index_backups(backups), keep=0)
    assert plan.keep == 1
    assert len(plan.retained) == 1
    assert plan.retained[0].name == f"backup-{stamps[-1]}-aaaaaa"
    assert len(plan.delete) == 2


def test_two_sets_in_the_same_second_are_distinct_and_both_counted(backups: Path) -> None:
    """Two runs on one day: the six-hex suffix keeps them apart."""
    make_set(backups, stamp="20260918T040000Z", uid="aaaaaa")
    make_set(backups, stamp="20260918T040000Z", uid="bbbbbb")
    index = index_backups(backups)
    assert len(index.complete) == 2
    assert len({s.path for s in index.complete}) == 2


# ---------------------------------------------------------------------------
# What retention refuses to touch
# ---------------------------------------------------------------------------


def test_a_set_with_no_manifest_is_partial_and_never_deleted(backups: Path) -> None:
    """A run killed between the dump and the manifest."""
    for stamp in _stamps(DEFAULT_KEEP + 2):
        make_set(backups, stamp=stamp)
    orphan = make_set(backups, stamp="20250101T040000Z", write_manifest=False)
    index = index_backups(backups)
    assert orphan in index.partial
    plan = plan_retention(index, keep=DEFAULT_KEEP)
    assert orphan not in plan.delete
    assert apply_retention(plan) and orphan.is_dir()


def test_a_manifest_marked_incomplete_is_partial(backups: Path) -> None:
    path = make_set(backups, stamp="20250101T040000Z", complete=False)
    assert read_manifest(path) is None
    assert path in index_backups(backups).partial


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("malformed json", "{not json at all"),
        ("json but not an object", "[1, 2, 3]"),
        ("unknown format version", json.dumps({"format": 999, "complete": True, "created_at": "x"})),
        ("no created_at", json.dumps({"format": MANIFEST_FORMAT, "complete": True})),
        ("created_at not a string", json.dumps({"format": MANIFEST_FORMAT, "complete": True, "created_at": 7})),
        ("complete is truthy but not true", json.dumps({"format": MANIFEST_FORMAT, "complete": 1, "created_at": "x"})),
    ],
)
def test_an_unprovable_manifest_makes_the_set_partial(backups: Path, label: str, body: str) -> None:
    """Every way a manifest can fail to prove completeness is treated the same."""
    path = backups / "backup-20250101T040000Z-aaaaaa"
    path.mkdir()
    (path / MANIFEST_FILENAME).write_text(body, encoding="utf-8")
    assert read_manifest(path) is None, label
    index = index_backups(backups)
    assert path in index.partial, label
    assert not index.complete, label


def test_staging_directories_are_partial_never_foreign_never_deleted(backups: Path) -> None:
    staging = backups / f"{STAGING_PREFIX}deadbeef"
    staging.mkdir()
    for stamp in _stamps(DEFAULT_KEEP + 3):
        make_set(backups, stamp=stamp)
    index = index_backups(backups)
    assert staging in index.partial
    plan = plan_retention(index, keep=DEFAULT_KEEP)
    assert staging not in plan.delete
    apply_retention(plan)
    assert staging.is_dir()


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",
        "README",
        "backup-2026-09-18",  # right prefix, wrong shape
        "backup-20260918T040000Z",  # missing the uid suffix
        "backup-20260918T040000Z-AAAAAA",  # uppercase hex
        "backup-20260918T040000Z-aaaaaaa",  # seven hex chars
        "snapshot-20260918T040000Z-aaaaaa",
    ],
)
def test_anything_not_matching_the_set_name_is_foreign(backups: Path, name: str) -> None:
    entry = backups / name
    if "." in name or name == "README":
        entry.write_text("operator's own file")
    else:
        entry.mkdir()
        (entry / MANIFEST_FILENAME).write_text(
            json.dumps({"format": MANIFEST_FORMAT, "complete": True, "created_at": "x"})
        )
    index = index_backups(backups)
    assert entry in index.foreign
    assert entry not in [s.path for s in index.complete]
    assert not plan_retention(index, keep=1).delete


def test_a_symlink_that_looks_like_a_set_is_never_followed_or_deleted(
    backups: Path, tmp_path: Path
) -> None:
    """`is_dir()` follows symlinks, so the symlink check has to come first —
    otherwise retention could rmtree a directory anywhere on the volume."""
    elsewhere = tmp_path / "somewhere-important"
    elsewhere.mkdir()
    (elsewhere / MANIFEST_FILENAME).write_text(
        json.dumps({"format": MANIFEST_FORMAT, "complete": True, "created_at": "1999-01-01T00:00:00+00:00"})
    )
    link = backups / "backup-19990101T000000Z-aaaaaa"
    link.symlink_to(elsewhere, target_is_directory=True)
    for stamp in _stamps(DEFAULT_KEEP + 2):
        make_set(backups, stamp=stamp)

    index = index_backups(backups)
    assert link in index.foreign
    plan = plan_retention(index, keep=DEFAULT_KEEP)
    assert link not in plan.delete
    apply_retention(plan)
    assert link.is_symlink()
    assert elsewhere.is_dir()


def test_a_missing_backups_directory_is_an_empty_index_not_an_error(tmp_path: Path) -> None:
    index = index_backups(tmp_path / "never-created")
    assert index.complete == () and index.partial == () and index.foreign == ()


# ---------------------------------------------------------------------------
# apply_retention re-checks before it deletes
# ---------------------------------------------------------------------------


def test_apply_retention_deletes_exactly_the_plan(backups: Path) -> None:
    stamps = _stamps(DEFAULT_KEEP + 3)
    for stamp in stamps:
        make_set(backups, stamp=stamp)
    plan = plan_retention(index_backups(backups), keep=DEFAULT_KEEP)
    deleted = apply_retention(plan)
    assert set(deleted) == set(plan.delete)
    assert len(deleted) == 3
    assert all(not p.exists() for p in deleted)
    assert all(p.is_dir() for p in plan.retained)


def test_apply_retention_skips_a_candidate_that_became_unprovable(backups: Path) -> None:
    """The plan may be minutes old (a dry run the operator then approved).
    A candidate that no longer looks complete is skipped, not removed."""
    stamps = _stamps(DEFAULT_KEEP + 2)
    for stamp in stamps:
        make_set(backups, stamp=stamp)
    plan = plan_retention(index_backups(backups), keep=DEFAULT_KEEP)
    assert len(plan.delete) == 2
    stale = plan.delete[0]
    (stale / MANIFEST_FILENAME).unlink()  # something else half-cleaned it

    deleted = apply_retention(plan)
    assert stale not in deleted
    assert stale.is_dir()
    assert plan.delete[1] not in [p for p in plan.delete if p.exists()]


def test_apply_retention_skips_a_candidate_that_vanished(backups: Path) -> None:
    stamps = _stamps(DEFAULT_KEEP + 1)
    for stamp in stamps:
        make_set(backups, stamp=stamp)
    plan = plan_retention(index_backups(backups), keep=DEFAULT_KEEP)
    import shutil

    shutil.rmtree(plan.delete[0])
    assert apply_retention(plan) == ()
