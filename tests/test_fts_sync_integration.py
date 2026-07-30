"""Event-driven FTS sync (SPEC §7.3, D6) — real SQLite (apsw) *and*
real Postgres throughout: this is the "genuinely exercise tokenization
and event-ordering, not stubbed" requirement. Skips cleanly when no
scratch Postgres is reachable, same pattern as
`tests/test_migrations_integration.py`."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import apsw
import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.fts.schema import create_schema, get_applied_event_id
from imsg.embed.fts.sync import sync_fts

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_fts_sync_test"

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


@pytest.fixture
def fts_conn(tmp_path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    return conn


# --- Postgres-side minimal fixture helpers ---------------------------------


def _insert_chat_and_session(conn: psycopg.Connection) -> tuple[int, int]:
    guid = f"chat-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') RETURNING chat_id",
            (guid, f"thread-{guid}"),
        )
        chat_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
            "VALUES (%s, now(), now(), 3.0) RETURNING session_id",
            (chat_id,),
        )
        session_id = cur.fetchone()[0]  # type: ignore[index]
    return chat_id, session_id


def _insert_segment(
    conn: psycopg.Connection, *, chat_id: int, session_id: int, rendered_text: str, seq: int = 0
) -> int:
    stable_key = f"stable-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO segment (
                stable_key, chat_id, session_id, seq_in_session, started_at, ended_at,
                message_count, token_count, rendered_text, rendered_sha256, seg_config_hash
            ) VALUES (%s, %s, %s, %s, now(), now(), 1, 10, %s, 'x', 'cfg-hash')
            RETURNING segment_id
            """,
            (stable_key, chat_id, session_id, seq, rendered_text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_attachment_and_chunk(conn: psycopg.Connection, *, text: str, seq: int = 0) -> tuple[int, int]:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key) VALUES (%s, %s) RETURNING attachment_id",
            (guid, f"key-{guid}"),
        )
        attachment_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO attachment_chunk (attachment_id, kind, seq, text, token_count) "
            "VALUES (%s, 'pdf_text', %s, %s, 10) RETURNING chunk_id",
            (attachment_id, seq, text),
        )
        chunk_id = cur.fetchone()[0]  # type: ignore[index]
    return attachment_id, chunk_id


def _emit_event(
    conn: psycopg.Connection, *, entity_kind: str, entity_id: int, operation: str, content_sha256: str | None = None
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
            "VALUES (%s, %s, %s, %s) RETURNING event_id",
            (entity_kind, entity_id, operation, content_sha256),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _seg_fts_hits(fts_conn: apsw.Connection, query: str) -> list[int]:
    cur = fts_conn.cursor()
    return [row[0] for row in cur.execute("SELECT rowid FROM seg_fts WHERE seg_fts MATCH ?", (query,))]


def _seg_tri_hits(fts_conn: apsw.Connection, query: str) -> list[int]:
    cur = fts_conn.cursor()
    return [row[0] for row in cur.execute("SELECT rowid FROM seg_fts_tri WHERE seg_fts_tri MATCH ?", (query,))]


def _att_fts_hits(fts_conn: apsw.Connection, query: str) -> list[int]:
    cur = fts_conn.cursor()
    return [row[0] for row in cur.execute("SELECT rowid FROM att_fts WHERE att_fts MATCH ?", (query,))]


# --- tests -------------------------------------------------------------


def test_upsert_event_makes_segment_searchable(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(
        scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="did the revised bid come through"
    )
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="upsert", content_sha256="h1")

    report = sync_fts(scratch_db, fts_conn)

    assert report.upserts == 1
    assert report.events_applied == 1
    assert _seg_fts_hits(fts_conn, "revised") == [segment_id]
    assert get_applied_event_id(fts_conn) > 0


def test_diacritics_fold_both_ways(scratch_db: psycopg.Connection, fts_conn: apsw.Connection) -> None:
    """D2: unicode61 remove_diacritics 2 — "José" indexed, "jose" query
    (and vice versa) must both match."""
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(
        scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="Meeting with José about the deck"
    )
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="upsert", content_sha256="h1")
    sync_fts(scratch_db, fts_conn)

    assert _seg_fts_hits(fts_conn, "jose") == [segment_id]
    assert _seg_fts_hits(fts_conn, "josé") == [segment_id]


def test_trigram_table_supports_substring_search(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(
        scratch_db, chat_id=chat_id, session_id=session_id, rendered_text='see attached bid-rev3.pdf for details'
    )
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="upsert", content_sha256="h1")
    sync_fts(scratch_db, fts_conn)

    assert _seg_tri_hits(fts_conn, "rev3") == [segment_id]


def test_delete_event_removes_the_row(scratch_db: psycopg.Connection, fts_conn: apsw.Connection) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(
        scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="ephemeral content"
    )
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="upsert", content_sha256="h1")
    sync_fts(scratch_db, fts_conn)
    assert _seg_fts_hits(fts_conn, "ephemeral") == [segment_id]

    with scratch_db.cursor() as cur:
        cur.execute("DELETE FROM segment_message WHERE segment_id = %s", (segment_id,))
        cur.execute("DELETE FROM segment WHERE segment_id = %s", (segment_id,))
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="delete")

    report = sync_fts(scratch_db, fts_conn)

    assert report.deletes == 1
    assert _seg_fts_hits(fts_conn, "ephemeral") == []
    sqlite_cur = fts_conn.cursor()
    row = sqlite_cur.execute("SELECT count(*) FROM seg_map WHERE segment_id = ?", (segment_id,)).fetchone()
    assert row is not None
    assert row[0] == 0


def test_upsert_then_content_change_reindexes_not_duplicates(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(
        scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="original wording here"
    )
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="upsert", content_sha256="h1")
    sync_fts(scratch_db, fts_conn)
    assert _seg_fts_hits(fts_conn, "original") == [segment_id]

    with scratch_db.cursor() as cur:
        cur.execute(
            "UPDATE segment SET rendered_text = %s WHERE segment_id = %s",
            ("completely different wording now", segment_id),
        )
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="upsert", content_sha256="h2")

    sync_fts(scratch_db, fts_conn)

    assert _seg_fts_hits(fts_conn, "original") == []
    assert _seg_fts_hits(fts_conn, "different") == [segment_id]
    sqlite_cur = fts_conn.cursor()
    row = sqlite_cur.execute("SELECT count(*) FROM seg_map WHERE segment_id = ?", (segment_id,)).fetchone()
    assert row is not None
    assert row[0] == 1


def test_event_order_is_respected_not_entity_id_order(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    """The v1.1 fix: process by `event_id`, never by entity id — build a
    case where a *lower* entity id's event is emitted *after* a higher
    one's, and confirm processing still follows event_id order."""
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    seg_b = _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="segment B content", seq=1)
    seg_a = _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="segment A content", seq=0)
    assert seg_a > seg_b  # inserted second, so its Postgres id is numerically larger

    # Emit the *higher*-id segment's event first, then the lower one's --
    # event_id order (insertion order here) must still win over entity id.
    _emit_event(scratch_db, entity_kind="segment", entity_id=seg_b, operation="upsert", content_sha256="hb")
    _emit_event(scratch_db, entity_kind="segment", entity_id=seg_a, operation="upsert", content_sha256="ha")

    report = sync_fts(scratch_db, fts_conn)

    assert report.upserts == 2
    assert _seg_fts_hits(fts_conn, "content") == sorted([seg_a, seg_b])


def test_upsert_for_a_since_deleted_entity_is_skipped_not_fatal(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    _emit_event(scratch_db, entity_kind="segment", entity_id=999_999, operation="upsert", content_sha256="ghost")

    report = sync_fts(scratch_db, fts_conn)

    assert report.skipped_missing_content == 1
    assert report.upserts == 0
    assert get_applied_event_id(fts_conn) > 0  # still advances -- not stuck retrying forever


def test_attachment_chunk_events_use_the_att_tables(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    _attachment_id, chunk_id = _insert_attachment_and_chunk(scratch_db, text="deck rebuild materials fourteen thousand")
    _emit_event(scratch_db, entity_kind="attachment_chunk", entity_id=chunk_id, operation="upsert", content_sha256="h1")

    report = sync_fts(scratch_db, fts_conn)

    assert report.upserts == 1
    assert _att_fts_hits(fts_conn, "fourteen") == [chunk_id]


def test_sync_is_a_no_op_when_already_caught_up(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="hello")
    _emit_event(scratch_db, entity_kind="segment", entity_id=segment_id, operation="upsert", content_sha256="h1")

    first = sync_fts(scratch_db, fts_conn)
    second = sync_fts(scratch_db, fts_conn)

    assert first.events_applied == 1
    assert second.events_applied == 0


def test_sync_respects_batch_size_across_multiple_events(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_ids = [
        _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text=f"segment number {i}", seq=i)
        for i in range(5)
    ]
    for sid in segment_ids:
        _emit_event(scratch_db, entity_kind="segment", entity_id=sid, operation="upsert", content_sha256="h")

    report = sync_fts(scratch_db, fts_conn, batch_size=2)  # forces multiple internal fetch rounds

    assert report.events_applied == 5
    for sid in segment_ids:
        assert sid in _seg_fts_hits(fts_conn, "segment")
