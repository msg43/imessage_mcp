"""`imsg backup` refuses rather than half-running (SPEC §5.3, §14).

A nightly job's refusals are the part that has to work: a backup that
half-succeeds is worse than one that never ran, because it occupies a
retention slot and looks like protection. Every refusal named in the
build brief has a test here — missing destination, unwritable
destination, insufficient space, absent or too-old `pg_dump`, a corrupt
FTS sidecar, and a dump that fails its read-back.

No Postgres server is needed for any of this: the connection is a stub
that answers the two `SHOW`/`SELECT` statements `run_backup` asks it,
and the `pg_restore` verification tests run the real binary against
files on disk (`pg_restore` reads an archive without contacting a
server). The end-to-end path against a live instance is
`test_backup_integration.py`.

Fictional personas only (D5).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import apsw
import pytest

import imsg.backup.pipeline as pipeline_module
import imsg.backup.postgres_dump as dump_module
from imsg.backup.fts_copy import copy_fts_sidecar
from imsg.backup.pipeline import run_backup
from imsg.backup.postgres_dump import (
    check_version_compatibility,
    resolve_pg_dump,
    verify_dump,
)
from imsg.backup.retention import STAGING_PREFIX
from imsg.config.schema import Config
from imsg.errors import BackupError

HAS_PG_RESTORE = shutil.which("pg_restore") is not None
RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


# ---------------------------------------------------------------------------
# A connection stub: `run_backup` asks it two questions and nothing else.
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, answers: dict[str, int]) -> None:
        self._answers = answers
        self._row: tuple[int] | None = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, *args: object) -> None:
        if "server_version_num" in sql:
            self._row = (self._answers["server_version_num"],)
        elif "pg_database_size" in sql:
            self._row = (self._answers["database_size"],)
        else:  # pragma: no cover - the stub is asked nothing else
            raise AssertionError(f"unexpected statement: {sql}")

    def fetchone(self) -> tuple[int] | None:
        return self._row


class FakeConn:
    """Answers `SHOW server_version_num` and `pg_database_size`, nothing more."""

    def __init__(self, *, server_version_num: int = 170009, database_size: int = 1024) -> None:
        self._answers = {
            "server_version_num": server_version_num,
            "database_size": database_size,
        }

    def cursor(self) -> _Cursor:
        return _Cursor(self._answers)


@pytest.fixture
def cfg(config_dict_factory: Any) -> Config:
    return Config.model_validate(config_dict_factory())


@pytest.fixture
def pg_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused-by-these-tests")


@pytest.fixture
def pg_dump_new_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `pg_dump` look present and >= the stub server's major.

    Needed because the version check deliberately runs *before* the
    free-space check — on this build machine $PATH's pg_dump is 16 and
    the stub server reports 17, so without this the space tests would
    pass for the wrong reason.
    """
    monkeypatch.setattr(dump_module.shutil, "which", lambda _n: "/fake/pg_dump")
    monkeypatch.setattr(dump_module, "binary_major_version", lambda _b: 17)
    monkeypatch.setattr(pipeline_module, "binary_major_version", lambda _b: 17)


def _never_dumps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record-and-refuse stand-ins for everything that would actually run.

    Returns a list that stays empty on every refusal path — the point of
    these tests is that the refusal happens *before* anything is dumped.
    """
    calls: list[str] = []

    def _boom(*args: object, **kwargs: object) -> None:
        calls.append("dumped")
        raise AssertionError("pg_dump must not run on a refusal path")

    monkeypatch.setattr(dump_module, "run_pg_dump", _boom)
    return calls


# ---------------------------------------------------------------------------
# Destination
# ---------------------------------------------------------------------------


def test_refuses_when_the_data_root_does_not_exist(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    """The volume is not mounted: refuse, do not create a backups/ on the
    boot disk where the mount point should be."""
    calls = _never_dumps(monkeypatch)
    missing = tmp_path / "not-mounted" / "imsgindex"
    object.__setattr__(cfg.paths, "data_root", missing)
    with pytest.raises(BackupError, match="does not exist"):
        run_backup(conn=FakeConn(), config=cfg, keep=14)
    assert calls == []
    assert not missing.exists()


def test_refuses_when_the_data_root_is_a_file(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    _never_dumps(monkeypatch)
    not_a_dir = tmp_path / "data_root_file"
    not_a_dir.write_text("")
    object.__setattr__(cfg.paths, "data_root", not_a_dir)
    with pytest.raises(BackupError, match="not a directory"):
        run_backup(conn=FakeConn(), config=cfg, keep=14)


@pytest.mark.skipif(RUNNING_AS_ROOT, reason="root ignores the permission bits this test sets")
def test_refuses_when_the_backups_directory_is_unwritable(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    """Probed by writing a real file, not by `os.access` — a read-only
    mount or an ACL passes `os.access(W_OK)` and then fails the first write."""
    _never_dumps(monkeypatch)
    backups = cfg.paths.data_root / "backups"
    backups.mkdir()
    backups.chmod(0o500)
    try:
        with pytest.raises(BackupError, match="not writable"):
            run_backup(conn=FakeConn(), config=cfg, keep=14)
    finally:
        backups.chmod(0o700)


@pytest.mark.skipif(RUNNING_AS_ROOT, reason="root ignores the permission bits this test sets")
def test_refuses_when_the_backups_directory_cannot_be_created(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    _never_dumps(monkeypatch)
    cfg.paths.data_root.chmod(0o500)
    try:
        with pytest.raises(BackupError, match="could not create the backups directory"):
            run_backup(conn=FakeConn(), config=cfg, keep=14)
    finally:
        cfg.paths.data_root.chmod(0o700)


def test_refuses_when_backups_exists_as_a_file(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    _never_dumps(monkeypatch)
    (cfg.paths.data_root / "backups").write_text("someone put a file here")
    with pytest.raises(BackupError, match=r"could not create the backups directory|not a directory"):
        run_backup(conn=FakeConn(), config=cfg, keep=14)


def test_the_writability_probe_leaves_nothing_behind(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None, pg_dump_new_enough: None
) -> None:
    _never_dumps(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_free_space_bytes", lambda _p: 0)
    with pytest.raises(BackupError, match="refusing to back up"):
        run_backup(conn=FakeConn(), config=cfg, keep=14)
    assert list((cfg.paths.data_root / "backups").iterdir()) == []


# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------


def test_refuses_on_insufficient_free_space(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None, pg_dump_new_enough: None
) -> None:
    calls = _never_dumps(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_free_space_bytes", lambda _p: 100)
    with pytest.raises(BackupError, match=r"bytes free .*need >="):
        run_backup(conn=FakeConn(database_size=10_000_000), config=cfg, keep=14)
    assert calls == []


def test_the_space_requirement_includes_the_fts_sidecar(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None, pg_dump_new_enough: None
) -> None:
    """A run that only budgeted for the dump could still fill the volume."""
    _never_dumps(monkeypatch)
    sidecar = cfg.paths.data_root / "fts" / "fts.db"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"x" * 5_000)
    monkeypatch.setattr(pipeline_module, "_free_space_bytes", lambda _p: 3_000)
    with pytest.raises(BackupError, match="plus the 5000-byte FTS sidecar"):
        run_backup(conn=FakeConn(database_size=1_000), config=cfg, keep=14)


# ---------------------------------------------------------------------------
# pg_dump: present, and new enough
# ---------------------------------------------------------------------------


def test_refuses_when_pg_dump_is_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dump_module.shutil, "which", lambda _n: None)
    with pytest.raises(BackupError, match=r"was not found on \$PATH"):
        resolve_pg_dump(None)


def test_refuses_when_an_explicit_pg_dump_path_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(BackupError, match="pg_dump not found at"):
        resolve_pg_dump(tmp_path / "nope" / "pg_dump")


def test_refuses_when_pg_dump_is_older_than_the_server() -> None:
    """The real hazard on the build machine: Homebrew puts postgresql@16's
    pg_dump first on PATH while SPEC §5.3 pins the instance at pg17.
    pg_dump aborts *after* creating its output file, so without this
    pre-check the set would contain a zero-byte 'backup'."""
    with pytest.raises(BackupError, match=r"is PostgreSQL 16 but the instance is PostgreSQL 17"):
        check_version_compatibility(pg_dump_major=16, server_major=17, pg_dump=Path("/x/pg_dump"))


@pytest.mark.parametrize(("dump_major", "server_major"), [(17, 17), (18, 17), (17, 16)])
def test_an_equal_or_newer_pg_dump_is_accepted(dump_major: int, server_major: int) -> None:
    check_version_compatibility(
        pg_dump_major=dump_major, server_major=server_major, pg_dump=Path("/x/pg_dump")
    )


def test_the_version_check_runs_before_any_dump(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    calls = _never_dumps(monkeypatch)
    monkeypatch.setattr(dump_module, "binary_major_version", lambda _b: 16)
    monkeypatch.setattr(pipeline_module, "binary_major_version", lambda _b: 16)
    monkeypatch.setattr(dump_module.shutil, "which", lambda _n: "/fake/pg_dump")
    with pytest.raises(BackupError, match="PostgreSQL 16 but the instance is PostgreSQL 17"):
        run_backup(conn=FakeConn(server_version_num=170009), config=cfg, keep=14)
    assert calls == []
    assert list((cfg.paths.data_root / "backups").iterdir()) == []


# ---------------------------------------------------------------------------
# Verification of what was written
# ---------------------------------------------------------------------------


def test_verify_refuses_a_zero_byte_dump(tmp_path: Path) -> None:
    """Exactly what a version-mismatched pg_dump leaves behind."""
    dump = tmp_path / "postgres.dump"
    dump.write_bytes(b"")
    fake_pg_dump = tmp_path / "pg_dump"
    fake_pg_dump.write_text("")
    (tmp_path / "pg_restore").write_text("")
    with pytest.raises(BackupError, match="zero bytes"):
        verify_dump(dump=dump, pg_dump=fake_pg_dump)


def test_verify_refuses_when_the_dump_file_is_missing(tmp_path: Path) -> None:
    fake_pg_dump = tmp_path / "pg_dump"
    fake_pg_dump.write_text("")
    (tmp_path / "pg_restore").write_text("")
    with pytest.raises(BackupError, match="no dump file at"):
        verify_dump(dump=tmp_path / "absent.dump", pg_dump=fake_pg_dump)


def test_verify_refuses_when_pg_restore_is_not_beside_pg_dump(tmp_path: Path) -> None:
    """An unverifiable dump is not a backup — better to refuse than to
    keep bytes nothing has read back."""
    dump = tmp_path / "postgres.dump"
    dump.write_bytes(b"PGDMP-ish")
    lonely = tmp_path / "bin" / "pg_dump"
    lonely.parent.mkdir()
    lonely.write_text("")
    with pytest.raises(BackupError, match=r"pg_restore.* is not installed next to"):
        verify_dump(dump=dump, pg_dump=lonely)


@pytest.mark.skipif(not HAS_PG_RESTORE, reason="pg_restore not installed")
def test_verify_refuses_a_file_that_is_not_an_archive(tmp_path: Path) -> None:
    dump = tmp_path / "postgres.dump"
    dump.write_bytes(b"this is not a pg_dump archive" * 100)
    pg_dump = Path(shutil.which("pg_dump") or "")
    with pytest.raises(BackupError, match=r"failed its read-back|no readable table of contents"):
        verify_dump(dump=dump, pg_dump=pg_dump)


# ---------------------------------------------------------------------------
# FTS sidecar
# ---------------------------------------------------------------------------


def _make_fts_db(path: Path, *, tables: tuple[str, ...]) -> None:
    conn = apsw.Connection(str(path))
    try:
        for name in tables:
            conn.execute(f"CREATE TABLE {name} (k TEXT)")
    finally:
        conn.close()


def test_an_absent_sidecar_is_recorded_not_a_failure(tmp_path: Path) -> None:
    """A machine that has not reached Phase 3 has no fts/fts.db."""
    result = copy_fts_sidecar(source=tmp_path / "fts.db", dest=tmp_path / "copy.db")
    assert result.present is False
    assert result.path is None
    assert not (tmp_path / "copy.db").exists()


def test_a_sidecar_that_will_not_open_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "fts.db"
    source.write_bytes(b"definitely not sqlite" * 50)
    with pytest.raises(BackupError, match=r"could not be read|will not open"):
        copy_fts_sidecar(source=source, dest=tmp_path / "copy.db")
    assert not (tmp_path / "copy.db").exists()


def test_a_copy_missing_the_sidecar_schema_is_refused(tmp_path: Path) -> None:
    """A valid SQLite file that is not the FTS sidecar: opens cleanly,
    passes integrity_check, and is useless as a backup of the index."""
    source = tmp_path / "fts.db"
    _make_fts_db(source, tables=("something_else",))
    with pytest.raises(BackupError, match="missing expected table"):
        copy_fts_sidecar(source=source, dest=tmp_path / "copy.db")
    assert not (tmp_path / "copy.db").exists()


def test_a_well_formed_sidecar_is_copied_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "fts.db"
    _make_fts_db(source, tables=("meta", "seg_map", "seg_fts", "att_map", "att_fts"))
    result = copy_fts_sidecar(source=source, dest=tmp_path / "copy.db")
    assert result.present and result.path is not None
    assert result.byte_size and result.sha256
    assert not (tmp_path / "copy.db-wal").exists()
    assert not (tmp_path / "copy.db-shm").exists()


def test_a_failed_fts_copy_removes_the_partial_file(tmp_path: Path) -> None:
    source = tmp_path / "fts.db"
    _make_fts_db(source, tables=("meta",))
    dest = tmp_path / "copy.db"
    with pytest.raises(BackupError):
        copy_fts_sidecar(source=source, dest=dest)
    assert not dest.exists()


# ---------------------------------------------------------------------------
# Nothing partial is ever promoted
# ---------------------------------------------------------------------------


def test_a_failure_mid_run_leaves_no_staging_directory(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    """The dump blows up after the staging directory exists. `backups/`
    must come out of it with nothing in it at all — not a set, not a
    staging directory a later run has to reason about."""
    monkeypatch.setattr(dump_module.shutil, "which", lambda _n: "/fake/pg_dump")
    monkeypatch.setattr(dump_module, "binary_major_version", lambda _b: 17)
    monkeypatch.setattr(pipeline_module, "binary_major_version", lambda _b: 17)

    def _explode(**kwargs: object) -> None:
        raise BackupError("pg_dump exited 1: simulated failure")

    monkeypatch.setattr(pipeline_module, "dump_postgres", _explode)
    with pytest.raises(BackupError, match="simulated failure"):
        run_backup(conn=FakeConn(), config=cfg, keep=14)

    backups = cfg.paths.data_root / "backups"
    assert list(backups.iterdir()) == []
    assert not list(backups.glob(f"{STAGING_PREFIX}*"))


def test_an_interrupt_mid_run_also_leaves_no_staging_directory(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, pg_password: None
) -> None:
    """launchd killing the 04:00 job is a `BaseException`, not an
    `Exception` — a bare `except Exception` here would leak a directory
    on exactly the failure most likely to happen in practice."""
    monkeypatch.setattr(dump_module.shutil, "which", lambda _n: "/fake/pg_dump")
    monkeypatch.setattr(dump_module, "binary_major_version", lambda _b: 17)
    monkeypatch.setattr(pipeline_module, "binary_major_version", lambda _b: 17)

    def _interrupt(**kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module, "dump_postgres", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_backup(conn=FakeConn(), config=cfg, keep=14)
    assert list((cfg.paths.data_root / "backups").iterdir()) == []
