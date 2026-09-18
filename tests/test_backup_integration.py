"""`imsg backup` end to end against a live scratch Postgres (SPEC §5.3, §14).

Skips cleanly when no scratch instance is reachable, like every other
`*_integration.py` here. It additionally needs a `pg_dump` whose major
version is at least the scratch server's — which is exactly the
precondition the command itself enforces, so a machine that cannot run
these tests is a machine on which the command would correctly refuse.

The test worth reading is `test_verification_catches_a_truncated_write`:
it performs a real dump and then cuts the file short, which is the
failure a table-of-contents check cannot see (measured 2026-09-18: a
custom-format dump cut in half still lists every TOC entry and exits 0).

Fictional personas only (D5): Jamie Owner, Alice Example, Bob Builder.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from typer.testing import CliRunner

import imsg.backup.postgres_dump as dump_module
import imsg.cli as cli_module
from _export_fixtures import ADMIN_DSN, admin_reachable, create_scratch_db, drop_scratch_db, dsn
from imsg.backup.pipeline import run_backup
from imsg.backup.postgres_dump import binary_major_version
from imsg.backup.retention import MANIFEST_FILENAME, MANIFEST_FORMAT, STAGING_PREFIX, index_backups
from imsg.cli import app
from imsg.config.schema import Config
from imsg.errors import BackupError
from imsg.mount.guard import MountInfo

TEST_DB_NAME = "imsg_index_backup_test"
runner = CliRunner()


def _server_major() -> int | None:
    try:
        conn = psycopg.connect(ADMIN_DSN, connect_timeout=2)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version_num")
            row = cur.fetchone()
        return int(row[0]) // 10000 if row else None
    finally:
        conn.close()


def _find_new_enough_pg_dump(server_major: int) -> Path | None:
    """A `pg_dump` at least as new as the server.

    `$PATH`'s is not assumed to be it: Homebrew commonly puts an older
    major first (16 on the build machine, against a 17 server), which is
    the very mismatch `imsg backup` refuses. Falls back to the versioned
    Homebrew/Postgres.app locations before giving up.
    """
    candidates: list[Path] = []
    on_path = shutil.which("pg_dump")
    if on_path:
        candidates.append(Path(on_path))
    for major in range(server_major, server_major + 4):
        candidates.append(Path(f"/opt/homebrew/opt/postgresql@{major}/bin/pg_dump"))
        candidates.append(Path(f"/usr/local/opt/postgresql@{major}/bin/pg_dump"))
        candidates.append(
            Path(f"/Applications/Postgres.app/Contents/Versions/{major}/bin/pg_dump")
        )
    for candidate in candidates:
        if not candidate.is_file() or not (candidate.parent / "pg_restore").is_file():
            continue
        try:
            if binary_major_version(candidate) >= server_major:
                return candidate
        except BackupError:  # pragma: no cover - a binary that will not report a version
            continue
    return None


SERVER_MAJOR = _server_major()
PG_DUMP = _find_new_enough_pg_dump(SERVER_MAJOR) if SERVER_MAJOR else None

pytestmark = [
    pytest.mark.skipif(
        not admin_reachable(),
        reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
    ),
    pytest.mark.skipif(
        PG_DUMP is None,
        reason=f"no pg_dump >= the scratch server's major ({SERVER_MAJOR}) is installed",
    ),
]


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    # `create_scratch_db` leaves the migrations in an open transaction on a
    # non-autocommit connection, so no other session can see them. Harmless
    # for tests that drive everything through that one connection; fatal
    # here, because `pg_dump` connects as a session of its own and would
    # faithfully dump an empty schema.
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(TEST_DB_NAME)


@pytest.fixture
def cfg(
    db: psycopg.Connection, config_dict_factory: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> Config:
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused-under-trust-auth")
    config = Config.model_validate(config_dict_factory())
    # The config schema pins `database.dsn` to port 5433 (the dedicated
    # instance, non-negotiable #6), so the scratch DSN is substituted after
    # validation rather than by writing an invalid config.
    object.__setattr__(config.database, "dsn", dsn(TEST_DB_NAME))
    return config


@pytest.fixture
def conn(db: psycopg.Connection) -> Iterator[psycopg.Connection]:
    """Depends on `db` so pytest tears this down *first* — dropping the
    scratch database out from under an open session fails with ObjectInUse."""
    c = psycopg.connect(dsn(TEST_DB_NAME), autocommit=True)
    try:
        yield c
    finally:
        c.close()


def _run(config: Config, conn: psycopg.Connection, **kwargs: Any) -> Any:
    return run_backup(conn=conn, config=config, pg_dump_binary=PG_DUMP, **kwargs)


# ---------------------------------------------------------------------------
# The happy path, and what it actually wrote
# ---------------------------------------------------------------------------


def test_a_run_writes_a_verified_complete_set(cfg: Config, conn: psycopg.Connection) -> None:
    report = _run(cfg, conn)
    assert report.set_path is not None and report.set_path.is_dir()
    assert report.dump is not None and report.dump.byte_size > 0

    dump_path = report.set_path / "postgres.dump"
    assert dump_path.is_file()
    # Not "it did not throw": the file on disk is read back by the same
    # binary family that wrote it and every expected table is named in it.
    assert {"message", "chat", "person", "segment"} <= set(report.dump.tables_in_toc)

    manifest = json.loads((report.set_path / MANIFEST_FILENAME).read_text())
    assert manifest["complete"] is True
    assert manifest["format"] == MANIFEST_FORMAT
    assert manifest["files"][0]["sha256"] == report.dump.sha256
    assert manifest["files"][0]["bytes"] == dump_path.stat().st_size
    assert "attachments/" in manifest["out_of_scope"]
    assert "NOT theft or disk failure" in manifest["caveat"]


def test_the_written_dump_is_actually_restorable(cfg: Config, conn: psycopg.Connection) -> None:
    """The strongest available read of what was produced: restore it into a
    fresh database and count the tables that arrived. Reading the code, or
    trusting the exit status, would not establish this."""
    report = _run(cfg, conn)
    assert report.set_path is not None
    restore_db = "imsg_index_backup_restore_test"
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {restore_db}")
            cur.execute(f"CREATE DATABASE {restore_db}")
    finally:
        admin.close()
    try:
        assert PG_DUMP is not None
        proc = subprocess.run(
            [
                str(PG_DUMP.parent / "pg_restore"),
                "--dbname", dsn(restore_db),
                "--no-owner",
                str(report.set_path / "postgres.dump"),
            ],
            capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        check = psycopg.connect(dsn(restore_db), autocommit=True)
        try:
            with check.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name IN "
                    "('message','chat','person','segment','allowlist_person','mcp_audit')"
                )
                row = cur.fetchone()
            assert row is not None and row[0] == 6
        finally:
            check.close()
    finally:
        admin = psycopg.connect(ADMIN_DSN, autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {restore_db}")
        finally:
            admin.close()


def test_a_present_fts_sidecar_is_copied_and_an_absent_one_is_not_an_error(
    cfg: Config, conn: psycopg.Connection
) -> None:
    absent = _run(cfg, conn)
    assert absent.fts is not None and absent.fts.present is False
    assert absent.set_path is not None and not (absent.set_path / "fts.db").exists()

    import apsw

    sidecar = cfg.paths.data_root / "fts" / "fts.db"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sqlite = apsw.Connection(str(sidecar))
    try:
        for name in ("meta", "seg_map", "seg_fts", "att_map", "att_fts"):
            sqlite.execute(f"CREATE TABLE {name} (k TEXT)")
    finally:
        sqlite.close()

    present = _run(cfg, conn)
    assert present.fts is not None and present.fts.present is True
    assert present.set_path is not None and (present.set_path / "fts.db").is_file()
    manifest = json.loads((present.set_path / MANIFEST_FILENAME).read_text())
    assert [f["kind"] for f in manifest["files"]] == ["postgres-dump", "fts5-sidecar"]


# ---------------------------------------------------------------------------
# Verification catches a truncated write
# ---------------------------------------------------------------------------


def test_verification_catches_a_truncated_write(
    cfg: Config, conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real dump, then the file is cut in half — a disk that filled, or a
    process killed mid-write.

    This is the case `pg_restore --list` cannot see: measured 2026-09-18,
    a half-truncated custom-format dump lists all of its TOC entries and
    exits 0, because the table of contents precedes the data blocks. The
    read-back catches it, no set is promoted, and no staging directory
    survives.
    """
    real_run_pg_dump = dump_module.run_pg_dump

    def _truncating(**kwargs: Any) -> None:
        real_run_pg_dump(**kwargs)
        dest: Path = kwargs["dest"]
        size = dest.stat().st_size
        assert size > 0
        with dest.open("r+b") as f:
            f.truncate(size // 2)

    monkeypatch.setattr(dump_module, "run_pg_dump", _truncating)

    with pytest.raises(BackupError, match="failed its read-back and is NOT a usable backup"):
        _run(cfg, conn)

    backups = cfg.paths.data_root / "backups"
    assert list(backups.iterdir()) == []
    assert not list(backups.glob(f"{STAGING_PREFIX}*"))


def test_verification_catches_a_flipped_byte_in_a_data_block(
    cfg: Config, conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silent bit-rot rather than truncation: the compressed block's own
    checksum is what fails.

    **The honest scope of this check.** Truncation is caught
    unconditionally — a short archive always fails the read-back. A single
    flipped byte is caught when it lands inside a compressed data block,
    which is where the overwhelming majority of a real dump's bytes are,
    but a flip inside the archive's uncompressed header or table of
    contents can survive the read-back. Verified while writing this: a
    midpoint flip in a dump of an *empty* schema (nearly all metadata, no
    data blocks) is not detected. So the corpus is populated first and the
    flip is aimed into the data region, which is what makes this test able
    to fail for the reason it claims.
    """
    with conn.cursor() as cur:
        for i in range(400):
            cur.execute(
                "INSERT INTO person (display_name, short_name, needs_review) "
                "VALUES (%s, %s, false)",
                (f"Fictional Person {i}", f"person{i}"),
            )

    real_run_pg_dump = dump_module.run_pg_dump

    def _flipping(**kwargs: Any) -> None:
        real_run_pg_dump(**kwargs)
        dest: Path = kwargs["dest"]
        data = bytearray(dest.read_bytes())
        data[-2000] ^= 0xFF  # inside the trailing data blocks, not the TOC
        dest.write_bytes(bytes(data))

    monkeypatch.setattr(dump_module, "run_pg_dump", _flipping)
    with pytest.raises(BackupError, match="failed its read-back"):
        _run(cfg, conn)
    assert list((cfg.paths.data_root / "backups").iterdir()) == []


def test_a_zero_byte_dump_is_caught(
    cfg: Config, conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a version-mismatched pg_dump leaves behind if the pre-check is
    ever bypassed."""

    def _writes_nothing(**kwargs: Any) -> None:
        Path(kwargs["dest"]).write_bytes(b"")

    monkeypatch.setattr(dump_module, "run_pg_dump", _writes_nothing)
    with pytest.raises(BackupError, match="zero bytes"):
        _run(cfg, conn)
    assert list((cfg.paths.data_root / "backups").iterdir()) == []


# ---------------------------------------------------------------------------
# Idempotency and retention against real sets
# ---------------------------------------------------------------------------


def test_two_runs_in_one_day_produce_two_sets_and_corrupt_neither(
    cfg: Config, conn: psycopg.Connection
) -> None:
    first = _run(cfg, conn)
    second = _run(cfg, conn)
    assert first.set_path != second.set_path
    assert first.set_path is not None and first.set_path.is_dir()
    assert second.set_path is not None and second.set_path.is_dir()

    index = index_backups(cfg.paths.data_root / "backups")
    assert len(index.complete) == 2
    assert index.partial == () and index.foreign == ()
    # Each set's dump still reads back cleanly — neither run wrote into the
    # other's directory.
    for path in (first.set_path, second.set_path):
        manifest = json.loads((path / MANIFEST_FILENAME).read_text())
        assert manifest["complete"] is True
        assert (path / "postgres.dump").stat().st_size == manifest["files"][0]["bytes"]


def test_retention_prunes_to_keep_over_real_sets(
    cfg: Config, conn: psycopg.Connection
) -> None:
    for _ in range(4):
        _run(cfg, conn, keep=2)
    index = index_backups(cfg.paths.data_root / "backups")
    assert len(index.complete) == 2
    assert index.partial == ()


def test_a_corrupt_prior_backup_neither_breaks_the_run_nor_is_deleted(
    cfg: Config, conn: psycopg.Connection
) -> None:
    """Debris from an interrupted 04:00 job is the normal state of
    `backups/` on a machine that has ever lost power mid-run."""
    backups = cfg.paths.data_root / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    half_written = backups / "backup-20200101T040000Z-ffffff"
    half_written.mkdir()
    (half_written / "postgres.dump").write_bytes(b"truncated garbage")
    staging = backups / f"{STAGING_PREFIX}abandoned"
    staging.mkdir()
    (staging / "postgres.dump").write_bytes(b"")

    report = _run(cfg, conn, keep=1)
    assert report.set_path is not None and report.set_path.is_dir()
    assert half_written.is_dir(), "a partial set must never be deleted by retention"
    assert staging.is_dir(), "an abandoned staging dir must never be deleted by retention"
    assert set(report.retention.partial) == {half_written, staging}
    assert report.deleted == ()


# ---------------------------------------------------------------------------
# Through the CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(
    db: psycopg.Connection,
    config_dict_factory: Callable[..., dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Path]:
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused-under-trust-auth")
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="scratch"),
    )
    monkeypatch.setattr(
        cli_module, "connect", lambda database, **kw: psycopg.connect(dsn(TEST_DB_NAME), autocommit=True)
    )
    monkeypatch.setattr(cli_module, "verify_data_directory", lambda conn, data_root: data_root)

    def _write(**overrides: Any) -> Path:
        path = tmp_path / "config.yaml"
        payload = config_dict_factory(**overrides)
        payload["database"]["dsn"] = "postgresql://imsg@127.0.0.1:5433/imsgindex"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    return _write


def _patch_dsn_for_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """`pg_dump` connects itself, so the CLI's config DSN has to be the
    scratch one — patched on the validated Config the command loads."""
    real_load = cli_module.load_config

    def _load(path: Path | None) -> Config:
        config = real_load(path)
        object.__setattr__(config.database, "dsn", dsn(TEST_DB_NAME))
        return config

    monkeypatch.setattr(cli_module, "load_config", _load)


def test_cli_backup_writes_a_set_and_states_what_it_did_not_cover(
    cli_env: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, data_root: Path
) -> None:
    _patch_dsn_for_cli(monkeypatch)
    assert PG_DUMP is not None
    result = runner.invoke(
        app, ["backup", "--config", str(cli_env()), "--pg-dump", str(PG_DUMP)]
    )
    assert result.exit_code == 0, result.output
    assert "backup: wrote" in result.output
    assert "read back in full" in result.output
    # The scope decision is printed, not implied: an operator reading this
    # output must not come away believing the attachment corpus is covered.
    assert "attachments/" in result.output
    assert "NOT theft or disk failure" in result.output
    assert len(index_backups(data_root / "backups").complete) == 1


def test_cli_backup_dry_run_writes_nothing(
    cli_env: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, data_root: Path
) -> None:
    _patch_dsn_for_cli(monkeypatch)
    assert PG_DUMP is not None
    result = runner.invoke(
        app,
        ["backup", "--config", str(cli_env()), "--pg-dump", str(PG_DUMP), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert cli_module.DRY_RUN_MARKER in result.output
    assert "preconditions OK" in result.output
    backups = data_root / "backups"
    assert not backups.exists() or list(backups.iterdir()) == []


def test_cli_backup_refuses_with_one_clean_line_never_a_traceback(
    cli_env: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unattended job's failure has to be legible in a log file."""
    _patch_dsn_for_cli(monkeypatch)
    missing = Path(os.sep) / "nonexistent-pg-dump-for-this-test"
    result = runner.invoke(
        app, ["backup", "--config", str(cli_env()), "--pg-dump", str(missing)]
    )
    assert result.exit_code == 1
    assert result.output.strip().startswith("imsg: ")
    assert "Traceback" not in result.output
