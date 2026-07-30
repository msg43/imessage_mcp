"""Shared Postgres helpers for the export-gate test modules.

Fictional personas only (D5): Alice Example, Bob Builder, Carol
Carpenter, Jamie Owner, Acme Construction — never real names.

Each export test module creates its own scratch database (distinct
name) with the real migrations applied, mirroring the pattern in
`tests/test_segment_pipeline_integration.py`.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import psycopg

from imsg.db.migrations import PostgresMigrationRunner

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[0].parent / "migrations"


def dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


ADMIN_DSN = dsn("postgres")


def admin_reachable() -> bool:
    try:
        conn = psycopg.connect(ADMIN_DSN, connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


def create_scratch_db(db_name: str) -> psycopg.Connection:
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
            cur.execute(f"CREATE DATABASE {db_name}")
    finally:
        admin.close()
    conn = psycopg.connect(dsn(db_name))
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
    return conn


def drop_scratch_db(db_name: str) -> None:
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
    finally:
        admin.close()


# --- row helpers -----------------------------------------------------------


def insert_person(
    conn: psycopg.Connection,
    *,
    display_name: str,
    short_name: str,
    is_owner: bool = False,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
            "VALUES (%s, %s, %s, false) RETURNING person_id",
            (display_name, short_name, is_owner),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def allow(
    conn: psycopg.Connection,
    person_id: int,
    *,
    text: bool = True,
    attachments: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO allowlist_person (person_id, text_allowed, attachments_allowed) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (person_id) DO UPDATE SET "
            "text_allowed = EXCLUDED.text_allowed, "
            "attachments_allowed = EXCLUDED.attachments_allowed",
            (person_id, text, attachments),
        )


def insert_chat(
    conn: psycopg.Connection,
    *,
    source_guid: str,
    kind: str = "dm",
    display_name: str | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind, display_name) "
            "VALUES (%s, %s, %s, %s) RETURNING chat_id",
            (source_guid, f"thread-{source_guid}", kind, display_name),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def add_participant(conn: psycopg.Connection, chat_id: int, person_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)",
            (chat_id, person_id),
        )


def add_raw_participant(
    conn: psycopg.Connection,
    chat_id: int,
    raw_value: str,
    *,
    resolve_to_person: int | None = None,
) -> int:
    """A `chat_participant_source` row; optionally resolved to a person
    via a handle. Unresolved (default) models an S3 gap."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO source_handle (raw_value, service) VALUES (%s, 'imessage') "
            "RETURNING source_handle_id",
            (raw_value,),
        )
        row = cur.fetchone()
        assert row is not None
        source_handle_id = int(row[0])
        cur.execute(
            "INSERT INTO chat_participant_source (chat_id, source_handle_id) "
            "VALUES (%s, %s)",
            (chat_id, source_handle_id),
        )
        if resolve_to_person is not None:
            cur.execute(
                "INSERT INTO handle (person_id, kind, normalized_value) "
                "VALUES (%s, 'phone', %s) RETURNING handle_id",
                (resolve_to_person, f"+1555{source_handle_id:07d}"),
            )
            row = cur.fetchone()
            assert row is not None
            cur.execute(
                "INSERT INTO source_handle_resolution (source_handle_id, handle_id) "
                "VALUES (%s, %s)",
                (source_handle_id, int(row[0])),
            )
        return source_handle_id


def insert_message(
    conn: psycopg.Connection,
    *,
    chat_id: int,
    sender_person_id: int | None,
    is_from_me: bool,
    sent_at: datetime,
    text: str | None,
    is_unsent: bool = False,
) -> int:
    guid = f"msg-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO message (
                source_guid, message_key, chat_id, sender_person_id,
                is_from_me, sent_at, service, text_original, text_normalized,
                is_unsent
            ) VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s, %s)
            RETURNING message_id
            """,
            (
                guid,
                f"key-{guid}",
                chat_id,
                sender_person_id,
                is_from_me,
                sent_at,
                text,
                text,
                is_unsent,
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def add_edit_history(
    conn: psycopg.Connection, message_id: int, *, version_idx: int, text: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO message_version (message_id, version_idx, text) "
            "VALUES (%s, %s, %s)",
            (message_id, version_idx, text),
        )
        cur.execute(
            "UPDATE message SET is_edited = true WHERE message_id = %s", (message_id,)
        )


def insert_segment(
    conn: psycopg.Connection,
    *,
    chat_id: int,
    started_at: datetime,
    ended_at: datetime,
    message_ids: list[int],
    stable_key: str | None = None,
) -> int:
    stable_key = stable_key or f"stable-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
            "VALUES (%s, %s, %s, 3.0) RETURNING session_id",
            (chat_id, started_at, ended_at),
        )
        row = cur.fetchone()
        assert row is not None
        session_id = int(row[0])
        cur.execute(
            """
            INSERT INTO segment (
                stable_key, chat_id, session_id, seq_in_session, started_at,
                ended_at, message_count, rendered_text, rendered_sha256,
                seg_config_hash
            ) VALUES (%s, %s, %s, 0, %s, %s, %s, 'local-render', 'x', 'cfg')
            RETURNING segment_id
            """,
            (stable_key, chat_id, session_id, started_at, ended_at, len(message_ids)),
        )
        row = cur.fetchone()
        assert row is not None
        segment_id = int(row[0])
        for message_id in message_ids:
            cur.execute(
                "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
                (segment_id, message_id),
            )
        return segment_id


def insert_attachment(
    conn: psycopg.Connection,
    *,
    filename: str | None,
    mime_type: str | None,
) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, filename, mime_type, state)
            VALUES (%s, %s, %s, %s, 'materialized') RETURNING attachment_id
            """,
            (guid, f"akey-{guid}", filename, mime_type),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def attachment_guid(conn: psycopg.Connection, attachment_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_guid FROM attachment WHERE attachment_id = %s",
            (attachment_id,),
        )
        row = cur.fetchone()
        assert row is not None
        return str(row[0])


def link_attachment(conn: psycopg.Connection, message_id: int, attachment_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO message_attachment (message_id, attachment_id, ordinal) "
            "VALUES (%s, %s, 0)",
            (message_id, attachment_id),
        )
        cur.execute(
            "UPDATE message SET has_attachments = true WHERE message_id = %s",
            (message_id,),
        )


def add_enrichment(
    conn: psycopg.Connection, attachment_id: int, *, kind: str, text: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO enrichment (attachment_id, kind, state, text) "
            "VALUES (%s, %s, 'done', %s)",
            (attachment_id, kind, text),
        )


def add_chunk(
    conn: psycopg.Connection, attachment_id: int, *, kind: str, seq: int, text: str
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment_chunk (attachment_id, kind, seq, text) "
            "VALUES (%s, %s, %s, %s) RETURNING chunk_id",
            (attachment_id, kind, seq, text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def insert_tapback(
    conn: psycopg.Connection,
    *,
    target_message_id: int,
    sender_person_id: int | None,
    is_from_me: bool = False,
    kind: str = "loved",
) -> None:
    guid = f"tap-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute("SELECT source_guid FROM message WHERE message_id = %s", (target_message_id,))
        row = cur.fetchone()
        assert row is not None
        cur.execute(
            """
            INSERT INTO tapback (source_guid, target_source_guid, target_message_id,
                                 sender_person_id, is_from_me, kind)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (guid, str(row[0]), target_message_id, sender_person_id, is_from_me, kind),
        )
