"""Full FTS sidecar rebuild (SPEC §7.3/§9.3) — real SQLite (apsw) and
real Postgres. Skips cleanly when no scratch Postgres is reachable."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import apsw
import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.fts.rebuild import rebuild_fts
from imsg.embed.fts.schema import get_applied_event_id
from imsg.embed.fts.sync import sync_fts

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_fts_rebuild_test"

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


def _insert_attachment_and_chunk(conn: psycopg.Connection, *, text: str) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key) VALUES (%s, %s) RETURNING attachment_id",
            (guid, f"key-{guid}"),
        )
        attachment_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO attachment_chunk (attachment_id, kind, seq, text, token_count) "
            "VALUES (%s, 'pdf_text', 0, %s, 10) RETURNING chunk_id",
            (attachment_id, text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _emit_event(conn: psycopg.Connection, *, entity_kind: str, entity_id: int, operation: str, content_sha256: str | None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
            "VALUES (%s, %s, %s, %s)",
            (entity_kind, entity_id, operation, content_sha256),
        )


def test_rebuild_writes_every_segment_and_chunk(scratch_db: psycopg.Connection, tmp_path: Path) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    seg_ids = [
        _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text=f"segment {i}", seq=i)
        for i in range(3)
    ]
    chunk_ids = [_insert_attachment_and_chunk(scratch_db, text=f"chunk text {i}") for i in range(2)]

    fts_path = tmp_path / "fts.db"
    report = rebuild_fts(scratch_db, fts_path)

    assert report.segments_written == 3
    assert report.chunks_written == 2
    assert fts_path.is_file()

    conn = apsw.Connection(str(fts_path))
    cur = conn.cursor()
    seg_map_ids = {row[0] for row in cur.execute("SELECT segment_id FROM seg_map")}
    chunk_map_ids = {row[0] for row in cur.execute("SELECT chunk_id FROM att_map")}
    assert seg_map_ids == set(seg_ids)
    assert chunk_map_ids == set(chunk_ids)

    for i, sid in enumerate(seg_ids):
        hits = [r[0] for r in cur.execute("SELECT rowid FROM seg_fts WHERE seg_fts MATCH ?", (f"segment AND {i}",))]
        assert hits == [sid]


def test_rebuild_captures_watermark_and_sync_resumes_after_it(
    scratch_db: psycopg.Connection, tmp_path: Path
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    seg1 = _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="pre-rebuild content")
    _emit_event(scratch_db, entity_kind="segment", entity_id=seg1, operation="upsert", content_sha256="h1")

    fts_path = tmp_path / "fts.db"
    report = rebuild_fts(scratch_db, fts_path)
    assert report.snapshot_event_id >= 1

    conn = apsw.Connection(str(fts_path))
    assert get_applied_event_id(conn) == report.snapshot_event_id

    # A change *after* the snapshot must NOT already be reflected...
    seg2 = _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="post-rebuild content")
    _emit_event(scratch_db, entity_kind="segment", entity_id=seg2, operation="upsert", content_sha256="h2")
    cur = conn.cursor()
    assert [r[0] for r in cur.execute("SELECT rowid FROM seg_fts WHERE seg_fts MATCH 'post'")] == []

    # ...but a normal sync_fts call picks it up cleanly from the watermark.
    sync_report = sync_fts(scratch_db, conn)
    assert sync_report.events_applied == 1
    assert [r[0] for r in cur.execute("SELECT rowid FROM seg_fts WHERE seg_fts MATCH 'post'")] == [seg2]


def test_rebuild_atomically_replaces_an_existing_sidecar(
    scratch_db: psycopg.Connection, tmp_path: Path
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="first generation")

    fts_path = tmp_path / "fts.db"
    rebuild_fts(scratch_db, fts_path)
    first_bytes = fts_path.read_bytes()
    assert len(first_bytes) > 0

    _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="second generation")
    rebuild_fts(scratch_db, fts_path)  # must overwrite cleanly, not append/corrupt

    conn = apsw.Connection(str(fts_path))
    cur = conn.cursor()
    row = cur.execute("SELECT count(*) FROM seg_map").fetchone()
    assert row is not None
    assert row[0] == 2
    assert not (tmp_path / "fts.db.new").exists()  # no leftover temp file
