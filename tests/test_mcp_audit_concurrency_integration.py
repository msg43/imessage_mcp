"""`PostgresAuditSink` under the public surface's worker threads.

`imsg.mcp.tools.public_server` dispatches each tool call on its own
thread, so `PublicAuthGate` — and through it this sink — writes audit
rows from several threads at once. An audit row that goes missing is a
request served without an audit trail, which SPEC §12 step 4 depends on
not happening, so "it should be fine" is not the standard here: this
runs the real writes against a real `mcp_audit`.

The sink needed no change. It opens a connection per row from its
factory, and `imsg mcp public` passes one that connects afresh each
call, so concurrent writers share nothing. The second test pins that
contract by showing the alternative failing — a factory returning one
shared connection, which looks like a harmless simplification and is
not, because the sink's `with` block closes what the factory gave it.
The failure is therefore immediate rather than a race, which is the
best kind: it cannot lurk.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner
from imsg.mcp.audit import AuditRecord, PostgresAuditSink
from imsg.mcp.errors import AuditWriteError

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_audit_concurrency_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

OWNER_SUB = "300000000000000000003"
THREADS = 6
ROWS_PER_THREAD = 8


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


def _record(tool: str) -> AuditRecord:
    return AuditRecord(
        surface="public",
        subject=OWNER_SUB,
        subject_ok=True,
        tool=tool,
        params_sha256="0" * 64,
        result_count=1,
        latency_ms=7,
        error=None,
    )


def _write_from_many_threads(sink: PostgresAuditSink) -> list[str]:
    failures: list[str] = []
    start = threading.Barrier(THREADS, timeout=30)

    def write(index: int) -> None:
        try:
            start.wait()
            for row in range(ROWS_PER_THREAD):
                sink.record(_record(f"tool-{index}-{row}"))
        except BaseException as exc:  # the failure *is* the finding
            failures.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=write, args=(i,)) for i in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    return failures


def test_a_connection_per_row_audits_every_concurrent_request(
    scratch_db: psycopg.Connection,
) -> None:
    """The wiring `imsg mcp public` actually uses."""
    sink = PostgresAuditSink(lambda: psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True))

    assert _write_from_many_threads(sink) == []

    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT tool) FROM mcp_audit")
        row = cur.fetchone()
    assert row is not None
    assert row[0] == THREADS * ROWS_PER_THREAD, "a lost audit row is a request served unaudited"
    assert row[1] == THREADS * ROWS_PER_THREAD, "every row is the one its writer wrote"


def test_a_shared_connection_factory_cannot_keep_auditing(
    scratch_db: psycopg.Connection,
) -> None:
    """One shared connection writes the first row and then reports `the
    connection is closed` for the rest, because `record` closes what the
    factory hands it. It surfaces as `AuditWriteError`, which the gate
    turns into a 503, so the damage is refused service rather than a
    silent gap in the trail — but it is damage, and this is what pins
    the contract the sink's docstring states."""
    shared = psycopg.connect(_dsn(TEST_DB_NAME))
    try:
        sink = PostgresAuditSink(lambda: shared)
        failures = _write_from_many_threads(sink)
    finally:
        shared.close()

    assert failures, "a shared connection is expected to break under concurrent writes"
    assert all(f.startswith(AuditWriteError.__name__) for f in failures), failures
