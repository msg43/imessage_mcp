"""Postgres integration tests for the S5b enrichment queue's lease/
backoff mechanics (SPEC §8 S5b, D6) — skips cleanly when no scratch
Postgres is reachable, same pattern as
`tests/test_migrations_integration.py`."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner
from imsg.enrich.queue import (
    claim_tasks,
    complete_task,
    enqueue,
    fail_task,
    fail_task_permanently,
    preview_claimable_tasks,
    skip_task,
)

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_enrich_queue_test"

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


def _insert_attachment(conn: psycopg.Connection) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key) VALUES (%s, %s) "
            "RETURNING attachment_id",
            (guid, f"key-{guid}"),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _state(conn: psycopg.Connection, attachment_id: int, kind: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state FROM enrichment WHERE attachment_id = %s AND kind = %s",
            (attachment_id, kind),
        )
        row = cur.fetchone()
        assert row is not None
        return str(row[0])


def test_enqueue_is_idempotent(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr", "caption"))
    enqueue(scratch_db, att_id, ("ocr", "caption"))  # must not duplicate or error
    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM enrichment WHERE attachment_id = %s", (att_id,))
        (count,) = cur.fetchone()  # type: ignore[misc]
    assert count == 2


def test_reenqueue_does_not_reset_a_done_row(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr",))
    complete_task(scratch_db, att_id, "ocr", model="m", model_version=None, text="done text")
    enqueue(scratch_db, att_id, ("ocr",))  # re-routing must not clobber the done row
    assert _state(scratch_db, att_id, "ocr") == "done"


def test_claim_tasks_marks_running_and_returns_them(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr", "caption"))

    claimed = claim_tasks(scratch_db, worker_id="worker-1", limit=10)

    assert {(c.attachment_id, c.kind) for c in claimed} == {(att_id, "ocr"), (att_id, "caption")}
    assert _state(scratch_db, att_id, "ocr") == "running"
    assert _state(scratch_db, att_id, "caption") == "running"


def test_claim_tasks_respects_limit(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr", "caption"))
    claimed = claim_tasks(scratch_db, worker_id="worker-1", limit=1)
    assert len(claimed) == 1


def test_concurrent_claim_skips_locked_rows(scratch_db: psycopg.Connection) -> None:
    """Two separate connections (simulating two workers) must never
    claim the same task — `FOR UPDATE SKIP LOCKED` in action, not just
    the app-level state machine."""
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr",))

    conn_a = psycopg.connect(_dsn(TEST_DB_NAME))
    conn_b = psycopg.connect(_dsn(TEST_DB_NAME))
    try:
        claimed_a = claim_tasks(conn_a, worker_id="worker-a", limit=10)
        assert len(claimed_a) == 1
        # conn_a's transaction is still open (no commit yet) -> its lock holds.
        claimed_b = claim_tasks(conn_b, worker_id="worker-b", limit=10)
        assert claimed_b == []
        conn_a.commit()
    finally:
        conn_a.close()
        conn_b.close()


def test_expired_lease_is_reclaimed(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr",))
    claim_tasks(scratch_db, worker_id="worker-1", limit=10)
    assert _state(scratch_db, att_id, "ocr") == "running"

    # Simulate a worker that died mid-task: back-date the lease.
    with scratch_db.cursor() as cur:
        cur.execute(
            "UPDATE enrichment SET locked_at = now() - interval '1 hour' "
            "WHERE attachment_id = %s AND kind = %s",
            (att_id, "ocr"),
        )

    reclaimed = claim_tasks(scratch_db, worker_id="worker-2", limit=10, lease_seconds=60)
    assert len(reclaimed) == 1
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT locked_by FROM enrichment WHERE attachment_id = %s AND kind = %s",
            (att_id, "ocr"),
        )
        (locked_by,) = cur.fetchone()  # type: ignore[misc]
    assert locked_by == "worker-2"


def test_complete_task_writes_result_and_clears_lock(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("caption",))
    claim_tasks(scratch_db, worker_id="worker-1", limit=10)

    complete_task(
        scratch_db, att_id, "caption",
        model="fake/caption@test", model_version="v1", text="a photo of a cat",
        detail={"confidence": 0.9},
    )

    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT state, model, text, locked_at, locked_by, detail FROM enrichment "
            "WHERE attachment_id = %s AND kind = %s",
            (att_id, "caption"),
        )
        state, model, text, locked_at, locked_by, detail = cur.fetchone()  # type: ignore[misc]
    assert state == "done"
    assert model == "fake/caption@test"
    assert text == "a photo of a cat"
    assert locked_at is None
    assert locked_by is None
    assert detail == {"confidence": 0.9}


def test_fail_task_backs_off_then_becomes_failed_after_max_attempts(
    scratch_db: psycopg.Connection,
) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("transcript",))

    for attempt in range(1, 4):
        claim_tasks(scratch_db, worker_id="worker-1", limit=10)
        permanent = fail_task(
            scratch_db, att_id, "transcript", error=f"boom {attempt}", max_attempts=3
        )
        if attempt < 3:
            assert permanent is False
            assert _state(scratch_db, att_id, "transcript") == "pending"
            # Force the backoff clock forward so the next claim can see it again.
            with scratch_db.cursor() as cur:
                cur.execute(
                    "UPDATE enrichment SET next_attempt_at = now() "
                    "WHERE attachment_id = %s AND kind = %s",
                    (att_id, "transcript"),
                )
        else:
            assert permanent is True
            assert _state(scratch_db, att_id, "transcript") == "failed"


def test_fail_task_permanently_skips_backoff(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr",))
    claim_tasks(scratch_db, worker_id="worker-1", limit=10)

    fail_task_permanently(scratch_db, att_id, "ocr", error="file too large")

    assert _state(scratch_db, att_id, "ocr") == "failed"
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT last_error FROM enrichment WHERE attachment_id = %s AND kind = %s",
            (att_id, "ocr"),
        )
        (err,) = cur.fetchone()  # type: ignore[misc]
    assert "too large" in err


def test_preview_claimable_tasks_reports_counts_by_kind_without_claiming(
    scratch_db: psycopg.Connection,
) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr", "caption"))
    other_att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, other_att_id, ("ocr",))

    preview = preview_claimable_tasks(scratch_db)

    assert preview.total == 3
    assert preview.by_kind == {"ocr": 2, "caption": 1}

    # A read-only preview must never claim a lease — every row is still
    # 'pending', untouched, and a real claim afterward sees them all.
    assert _state(scratch_db, att_id, "ocr") == "pending"
    assert _state(scratch_db, att_id, "caption") == "pending"
    assert _state(scratch_db, other_att_id, "ocr") == "pending"

    claimed = claim_tasks(scratch_db, worker_id="worker-1", limit=10)
    assert len(claimed) == 3


def test_preview_claimable_tasks_includes_expired_leases(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("transcript",))
    claim_tasks(scratch_db, worker_id="worker-1", limit=10)
    assert _state(scratch_db, att_id, "transcript") == "running"

    # Simulate a worker that died mid-task: back-date the lease.
    with scratch_db.cursor() as cur:
        cur.execute(
            "UPDATE enrichment SET locked_at = now() - interval '1 hour' "
            "WHERE attachment_id = %s AND kind = %s",
            (att_id, "transcript"),
        )

    preview = preview_claimable_tasks(scratch_db, lease_seconds=60)
    assert preview.total == 1
    assert preview.by_kind == {"transcript": 1}
    # Still 'running' — the preview does not touch state.
    assert _state(scratch_db, att_id, "transcript") == "running"


def test_preview_claimable_tasks_empty_queue(scratch_db: psycopg.Connection) -> None:
    preview = preview_claimable_tasks(scratch_db)
    assert preview.total == 0
    assert preview.by_kind == {}


def test_skip_task_sets_skipped_state(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr",))
    claim_tasks(scratch_db, worker_id="worker-1", limit=10)

    skip_task(scratch_db, att_id, "ocr", reason="unsupported mime type")

    assert _state(scratch_db, att_id, "ocr") == "skipped"
