"""Live-Postgres integration tests for AT-2 seed completeness
(SPEC §12 AT-2, `imsg.verify.seed`)."""

from __future__ import annotations

import dataclasses
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner
from imsg.verify.seed import (
    build_seed_snapshot,
    snapshot_from_json,
    snapshot_to_json,
    verify_against_reference,
)

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_verify_seed_test"

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


def _insert_chat(conn: psycopg.Connection) -> int:
    guid = f"chat-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') "
            "RETURNING chat_id",
            (guid, f"thread-{guid}"),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_message(
    conn: psycopg.Connection, chat_id: int, *, sent_at: datetime, guid: str | None = None,
    text: str | None = "hi",
) -> str:
    guid = guid or f"msg-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO message (source_guid, message_key, chat_id, is_from_me, sent_at,
                                  service, text_original, text_normalized)
            VALUES (%s, %s, %s, true, %s, 'imessage', %s, %s)
            """,
            (guid, f"key-{guid}", chat_id, sent_at, text, text),
        )
    return guid


def test_build_seed_snapshot_reports_guids_and_diagnostics(scratch_db: psycopg.Connection) -> None:
    chat_id = _insert_chat(scratch_db)
    g1 = _insert_message(scratch_db, chat_id, sent_at=datetime(2023, 6, 1, tzinfo=UTC))
    g2 = _insert_message(scratch_db, chat_id, sent_at=datetime(2024, 1, 1, tzinfo=UTC), text=None)

    snap = build_seed_snapshot(scratch_db, source_label="mini")
    assert snap.guids == {g1, g2}
    assert snap.per_year_counts == {"2023": 1, "2024": 1}
    assert snap.body_decode_null_count == 1
    assert snap.min_sent_at is not None
    assert snap.max_sent_at is not None


def test_snapshot_json_roundtrip(scratch_db: psycopg.Connection) -> None:
    chat_id = _insert_chat(scratch_db)
    _insert_message(scratch_db, chat_id, sent_at=datetime(2023, 6, 1, tzinfo=UTC))
    snap = build_seed_snapshot(scratch_db, source_label="studio")
    back = snapshot_from_json(snapshot_to_json(snap))
    assert back == snap


def test_verify_against_reference_passes_when_everything_present(
    scratch_db: psycopg.Connection,
) -> None:
    chat_id = _insert_chat(scratch_db)
    g1 = _insert_message(scratch_db, chat_id, sent_at=datetime(2023, 6, 1, tzinfo=UTC))
    reference = build_seed_snapshot(scratch_db, source_label="studio")
    assert reference.guids == {g1}

    report = verify_against_reference(scratch_db, reference, local_label="mini")
    assert report.passed is True
    assert report.missing_guids == ()
    assert report.reference_message_count == 1
    assert report.local_message_count == 1


def test_verify_against_reference_fails_on_missing_guid(scratch_db: psycopg.Connection) -> None:
    chat_id = _insert_chat(scratch_db)
    _insert_message(scratch_db, chat_id, sent_at=datetime(2023, 6, 1, tzinfo=UTC))
    reference = build_seed_snapshot(scratch_db, source_label="studio")

    # Simulate a reference that saw one more message this host never got.
    augmented = dataclasses.replace(
        reference, guids=reference.guids | {"phantom-guid-never-extracted"}
    )
    report = verify_against_reference(scratch_db, augmented, local_label="mini")
    assert report.passed is False
    assert report.missing_guids == ("phantom-guid-never-extracted",)
    assert "1 message GUID(s)" in report.reasons[0]


def test_verify_against_reference_accepted_exceptions_excuse_missing_guid(
    scratch_db: psycopg.Connection,
) -> None:
    chat_id = _insert_chat(scratch_db)
    _insert_message(scratch_db, chat_id, sent_at=datetime(2023, 6, 1, tzinfo=UTC))
    reference = build_seed_snapshot(scratch_db, source_label="studio")
    augmented = dataclasses.replace(reference, guids=reference.guids | {"known-corrupted-row"})
    report = verify_against_reference(
        scratch_db, augmented, local_label="mini",
        accepted_exceptions=frozenset({"known-corrupted-row"}),
    )
    assert report.passed is True
    assert report.missing_guids == ()
    assert report.accepted_missing_guids == ("known-corrupted-row",)


def test_verify_against_reference_duplicate_across_sources_is_not_a_gap(
    scratch_db: psycopg.Connection,
) -> None:
    """A message ingested from both sources dedupes on source_guid
    (UNIQUE) — the exact same GUID set on both sides passes cleanly."""
    chat_id = _insert_chat(scratch_db)
    shared_guid = f"shared-{uuid.uuid4()}"
    _insert_message(scratch_db, chat_id, sent_at=datetime(2023, 6, 1, tzinfo=UTC), guid=shared_guid)
    with scratch_db.cursor() as cur:
        cur.execute(
            "INSERT INTO extraction_run (source_name, snapshot_path, snapshot_sha256, status) "
            "VALUES ('mini', 'x', 'y', 'ok') RETURNING run_id"
        )
        run_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute("SELECT message_id FROM message WHERE source_guid = %s", (shared_guid,))
        message_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO message_source (message_id, source_name, source_rowid, extraction_run_id) "
            "VALUES (%s, 'mini', 1, %s), (%s, 'studio-seed', 1, %s)",
            (message_id, run_id, message_id, run_id),
        )
    reference = build_seed_snapshot(scratch_db, source_label="studio")
    report = verify_against_reference(scratch_db, reference, local_label="mini")
    assert report.passed is True
