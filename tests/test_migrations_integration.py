"""Live-database integration tests for the migration runner (SPEC §7).

Skips cleanly whenever no Postgres is reachable — the unit suite
(`test_migrations.py`, everything else) never depends on this file, so
`pytest` is green on a machine with no scratch database at all (e.g.
the mini, most of the time). Point this at a real scratch instance via
`IMSG_TEST_PG_HOST` / `IMSG_TEST_PG_PORT` / `IMSG_TEST_PG_USER` env
vars; defaults match the disposable local dev cluster used to build
this repo.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from imsg.db.fingerprint import verify_data_directory
from imsg.db.migrations import PostgresMigrationRunner
from imsg.errors import ClusterFingerprintError

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_migrations_test"

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

    conn = psycopg.connect(_dsn(TEST_DB_NAME))
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
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


def test_apply_pending_creates_every_table_and_hnsw_index(
    scratch_db: psycopg.Connection,
) -> None:
    runner = PostgresMigrationRunner(scratch_db, REAL_MIGRATIONS_DIR)
    applied = runner.apply_pending()
    # Asserted as a property -- applied in ascending version order, with no
    # gaps or repeats -- rather than a literal list that must be edited
    # every time a migration is added.
    versions = [m.version for m in applied]
    assert versions == sorted(versions)
    assert versions == list(range(1, len(versions) + 1))

    plan = runner.plan()
    assert plan.pending == ()
    assert plan.is_clean
    assert [a.version for a in plan.applied] == versions

    with scratch_db.cursor() as cur:
        cur.execute("SELECT indexname FROM pg_indexes WHERE indexname LIKE '%_hnsw'")
        hnsw_indexes = {row[0] for row in cur.fetchall()}
    assert hnsw_indexes == {
        "segment_embedding_hnsw",
        "attachment_chunk_embedding_hnsw",
        "attachment_mm_embedding_hnsw",
    }

    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == 36  # 35 from 0001 + attachment_mm_embedding from 0002


def test_apply_pending_is_idempotent(scratch_db: psycopg.Connection) -> None:
    runner = PostgresMigrationRunner(scratch_db, REAL_MIGRATIONS_DIR)
    # Counted from the directory rather than hardcoded: the property under
    # test is "every pending migration applies exactly once", which must
    # not need editing each time a migration is added.
    expected = len(sorted(REAL_MIGRATIONS_DIR.glob("*.sql")))
    first = runner.apply_pending()
    assert len(first) == expected

    second = runner.apply_pending()
    assert second == []

    plan = runner.plan()
    assert plan.is_clean
    assert plan.pending == ()


def test_verify_after_apply_is_clean(scratch_db: psycopg.Connection) -> None:
    runner = PostgresMigrationRunner(scratch_db, REAL_MIGRATIONS_DIR)
    runner.apply_pending()
    plan = runner.verify()  # raises on any problem
    assert plan.is_clean


def test_verify_data_directory_rejects_a_non_dedicated_cluster(
    scratch_db: psycopg.Connection, tmp_path: Path
) -> None:
    """This scratch cluster is real Postgres, but it is deliberately NOT
    the dedicated imessage-index instance at `$DATA_ROOT/pg17` — proving
    the cluster-fingerprint gate actually rejects a foreign cluster
    (CLAUDE.md non-negotiable #6, SPEC §5.2), using a real connection
    rather than a mock."""
    with pytest.raises(ClusterFingerprintError):
        verify_data_directory(scratch_db, tmp_path / "some_other_data_root")
