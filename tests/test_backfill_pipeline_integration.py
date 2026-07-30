"""Postgres integration tests for S5a (SPEC §8 S5a) — skips cleanly
when no scratch Postgres is reachable, same pattern as
`tests/test_migrations_integration.py`."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from imsg.backfill.pipeline import MAX_MATERIALIZATION_ATTEMPTS, run_backfill
from imsg.backfill.reconcile import build_reconciliation_report
from imsg.backfill.throttle import RateThrottle
from imsg.db.migrations import PostgresMigrationRunner

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_backfill_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def _dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


ADMIN_DSN = _dsn("postgres")


def _admin_reachable() -> bool:
    try:
        conn = psycopg.connect(ADMIN_DSN, connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


REACHABLE = _admin_reachable()

pytestmark = pytest.mark.skipif(
    not REACHABLE,
    reason=(
        "no reachable scratch Postgres instance "
        f"(tried {TEST_PG_HOST}:{TEST_PG_PORT}) — set IMSG_TEST_PG_HOST/"
        "IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


@pytest.fixture
def scratch_db() -> Iterator[psycopg.Connection]:
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
            cur.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    finally:
        admin.close()

    conn = psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
    try:
        yield conn
    finally:
        conn.close()
        admin = psycopg.connect(ADMIN_DSN, autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        finally:
            admin.close()


def _insert_attachment(conn: psycopg.Connection, *, source_path: str) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key, source_path) "
            "VALUES (%s, %s, %s) RETURNING attachment_id",
            (guid, f"key-{guid}", source_path),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _fetch_state(conn: psycopg.Connection, attachment_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT state FROM attachment WHERE attachment_id = %s", (attachment_id,))
        row = cur.fetchone()
        assert row is not None
        return str(row[0])


@pytest.fixture
def no_sleep_throttle() -> RateThrottle:
    return RateThrottle(10_000, sleep_fn=lambda _: None)


def test_run_backfill_materializes_real_files(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    f1 = attachments_root / "a" / "photo.jpg"
    f1.parent.mkdir(parents=True)
    f1.write_bytes(b"fake jpeg bytes")
    att_id = _insert_attachment(scratch_db, source_path=str(f1))

    report = run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=no_sleep_throttle,
    )

    assert report.materialized == 1
    assert report.errored == 0
    assert _fetch_state(scratch_db, att_id) == "materialized"

    with scratch_db.cursor() as cur:
        cur.execute("SELECT sha256, byte_size, cache_path FROM attachment WHERE attachment_id = %s", (att_id,))
        sha, byte_size, cache_path = cur.fetchone()  # type: ignore[misc]
    assert byte_size == len(b"fake jpeg bytes")
    assert Path(cache_path).is_file()
    assert Path(cache_path).read_bytes() == b"fake jpeg bytes"
    assert cache_path.endswith(f"{sha[:2]}/{sha}")


def test_trial_gate_caps_first_run_at_default_limit(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    for i in range(15):
        f = attachments_root / f"file{i}.bin"
        f.write_bytes(f"content-{i}".encode())
        _insert_attachment(scratch_db, source_path=str(f))

    report = run_backfill(scratch_db, data_root, attachments_root, throttle=no_sleep_throttle)

    assert report.trial_gate_capped is True
    assert report.materialized == 12  # DEFAULT_TRIAL_LIMIT
    assert report.considered == 12

    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM attachment WHERE state = 'materialized'")
        (materialized_count,) = cur.fetchone()  # type: ignore[misc]
    assert materialized_count == 12


def test_yes_full_run_bypasses_trial_gate(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    for i in range(15):
        f = attachments_root / f"file{i}.bin"
        f.write_bytes(f"content-{i}".encode())
        _insert_attachment(scratch_db, source_path=str(f))

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )

    assert report.trial_gate_capped is False
    assert report.materialized == 15


def test_second_run_skips_already_materialized(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    f = attachments_root / "once.bin"
    f.write_bytes(b"data")
    _insert_attachment(scratch_db, source_path=str(f))

    run_backfill(scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle)
    second_report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )

    assert second_report.considered == 0
    assert second_report.materialized == 0


def test_source_path_escaping_attachments_root_is_rejected(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"should never be read")
    data_root = tmp_path / "data_root"

    att_id = _insert_attachment(scratch_db, source_path=str(outside))

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )

    assert report.materialized == 0
    assert report.errored == 1
    assert _fetch_state(scratch_db, att_id) == "error"
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT materialization_last_error FROM attachment WHERE attachment_id = %s", (att_id,)
        )
        (err,) = cur.fetchone()  # type: ignore[misc]
    assert "attachments root" in err


def test_missing_file_backs_off_then_becomes_missing_after_three_attempts(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    ghost = attachments_root / "ghost.bin"  # never actually created
    att_id = _insert_attachment(scratch_db, source_path=str(ghost))

    for attempt in range(1, MAX_MATERIALIZATION_ATTEMPTS + 1):
        report = run_backfill(
            scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
        )
        assert report.considered == 1
        if attempt < MAX_MATERIALIZATION_ATTEMPTS:
            assert report.errored == 1
            # A second run at the same instant must NOT reconsider it — the
            # failure just backed next_attempt_at off into the future.
            immediate_rerun = run_backfill(
                scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
            )
            assert immediate_rerun.considered == 0
            # Force the backoff clock forward so the *next* real attempt is eligible.
            with scratch_db.cursor() as cur:
                cur.execute(
                    "UPDATE attachment SET materialization_next_attempt_at = now() "
                    "WHERE attachment_id = %s",
                    (att_id,),
                )
        else:
            assert report.marked_missing == 1

    assert _fetch_state(scratch_db, att_id) == "missing"
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT materialization_attempts FROM attachment WHERE attachment_id = %s", (att_id,)
        )
        (attempts,) = cur.fetchone()  # type: ignore[misc]
    assert attempts == MAX_MATERIALIZATION_ATTEMPTS


def test_low_disk_space_halts_the_run(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    f = attachments_root / "f.bin"
    f.write_bytes(b"data")
    _insert_attachment(scratch_db, source_path=str(f))

    report = run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=no_sleep_throttle,
        disk_free_fn=lambda _p: 0,  # always "out of space"
    )

    assert report.halted_low_disk_space is True
    assert report.materialized == 0


def test_reconciliation_report_enumerates_gaps(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    ok_file = attachments_root / "ok.bin"
    ok_file.write_bytes(b"ok")
    _insert_attachment(scratch_db, source_path=str(ok_file))
    _insert_attachment(scratch_db, source_path=str(attachments_root / "never.bin"))

    run_backfill(scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle)

    report = build_reconciliation_report(scratch_db)
    assert report.total == 2
    assert report.materialized_and_present == 1
    assert len(report.gaps) == 1
    assert report.gaps[0].state == "error"
    assert 0.0 < report.completeness_ratio < 1.0
