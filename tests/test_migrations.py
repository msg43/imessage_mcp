"""Pure-logic migration tests (SPEC §7.1) — no database required."""

from __future__ import annotations

from pathlib import Path

import pytest

from imsg.db.migrations import (
    AppliedMigration,
    HashMismatch,
    compute_plan,
    discover_migrations,
)
from imsg.errors import MigrationError

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def test_discover_real_migrations() -> None:
    files = discover_migrations(REAL_MIGRATIONS_DIR)
    # Versions are contiguous from 1 with no gaps or repeats -- asserted
    # as a property so adding a migration does not require editing this
    # test, while a misnumbered or duplicated file still fails loudly.
    versions = [f.version for f in files]
    assert versions == list(range(1, len(files) + 1))
    assert files[0].name == "initial"
    assert files[1].name == "multimodal_vectors"
    assert all(len(f.sha256) == 64 for f in files)


def test_discover_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="not found"):
        discover_migrations(tmp_path / "does_not_exist")


def test_discover_rejects_malformed_filename(tmp_path: Path) -> None:
    (tmp_path / "not_a_migration.sql").write_text("SELECT 1;")
    with pytest.raises(MigrationError, match="pattern"):
        discover_migrations(tmp_path)


def test_discover_rejects_duplicate_version(tmp_path: Path) -> None:
    (tmp_path / "0001_first.sql").write_text("CREATE TABLE a (x int);")
    (tmp_path / "0001_first_again.sql").write_text("CREATE TABLE b (x int);")
    with pytest.raises(MigrationError, match="duplicate"):
        discover_migrations(tmp_path)


def test_discover_sorts_by_version_not_filename_order(tmp_path: Path) -> None:
    (tmp_path / "0010_later.sql").write_text("SELECT 1;")
    (tmp_path / "0002_earlier.sql").write_text("SELECT 1;")
    files = discover_migrations(tmp_path)
    assert [f.version for f in files] == [2, 10]


def test_compute_plan_all_pending_when_nothing_applied() -> None:
    discovered = discover_migrations(REAL_MIGRATIONS_DIR)
    plan = compute_plan(discovered, applied=[])
    # With nothing applied, every discovered migration is pending.
    assert [p.version for p in plan.pending] == [f.version for f in discovered]
    assert plan.applied == ()
    assert plan.is_clean


def test_compute_plan_nothing_pending_when_all_applied_and_matching() -> None:
    discovered = discover_migrations(REAL_MIGRATIONS_DIR)
    applied = [AppliedMigration(version=f.version, sha256=f.sha256) for f in discovered]
    plan = compute_plan(discovered, applied)
    assert plan.pending == ()
    assert plan.is_clean


def test_compute_plan_partial_apply() -> None:
    discovered = discover_migrations(REAL_MIGRATIONS_DIR)
    applied = [AppliedMigration(version=discovered[0].version, sha256=discovered[0].sha256)]
    plan = compute_plan(discovered, applied)
    # Property, not a literal: with only the first migration applied,
    # pending is exactly the remainder, in order. Adding a migration
    # should not require editing this test.
    assert [p.version for p in plan.pending] == [f.version for f in discovered[1:]]
    assert plan.is_clean


def test_compute_plan_detects_hash_drift() -> None:
    discovered = discover_migrations(REAL_MIGRATIONS_DIR)
    tampered = AppliedMigration(version=discovered[0].version, sha256="0" * 64)
    plan = compute_plan(discovered, applied=[tampered])
    assert not plan.is_clean
    assert plan.mismatches == (
        HashMismatch(
            version=discovered[0].version,
            applied_sha256="0" * 64,
            disk_sha256=discovered[0].sha256,
        ),
    )


def test_compute_plan_detects_a_vanished_applied_file() -> None:
    discovered = discover_migrations(REAL_MIGRATIONS_DIR)[:1]  # pretend 0002 doesn't exist on disk
    applied = [
        AppliedMigration(version=1, sha256=discovered[0].sha256),
        AppliedMigration(version=2, sha256="deadbeef" * 8),
    ]
    plan = compute_plan(discovered, applied)
    assert not plan.is_clean
    assert plan.mismatches[0].version == 2
    assert plan.mismatches[0].disk_sha256 == "<file missing>"
