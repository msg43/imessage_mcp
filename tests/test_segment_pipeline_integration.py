"""Postgres integration tests for S4 (SPEC §8 S4) — skips cleanly when
no scratch Postgres is reachable, same pattern as
`tests/test_migrations_integration.py`. Fictional personas only (D5):
Alice Example / Bob Builder, never real names.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import (
    REBUILD_ALL_SENTINEL,
    find_dirty_chats,
    run_segment_for_chat,
)

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_segment_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
PROMPT_BYTES = b"fixed boundary-detection prompt for tests"


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
def config(config_dict_factory: object) -> Config:
    return load_config_dict(config_dict_factory())  # type: ignore[operator]


# --- fixture-data helpers -------------------------------------------------


def _insert_person(
    conn: psycopg.Connection, *, display_name: str, short_name: str, is_owner: bool = False
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


def _insert_chat(
    conn: psycopg.Connection, *, source_guid: str, kind: str = "dm", display_name: str | None = None
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


def _add_participant(conn: psycopg.Connection, chat_id: int, person_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)",
            (chat_id, person_id),
        )


def _insert_message(
    conn: psycopg.Connection,
    *,
    chat_id: int,
    sender_person_id: int,
    is_from_me: bool,
    sent_at: datetime,
    text: str,
) -> int:
    guid = f"msg-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO message (
                source_guid, message_key, chat_id, sender_person_id,
                is_from_me, sent_at, service, text_original, text_normalized
            ) VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s)
            RETURNING message_id
            """,
            (guid, f"key-{guid}", chat_id, sender_person_id, is_from_me, sent_at, text, text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


@pytest.fixture
def dm_chat(scratch_db: psycopg.Connection) -> tuple[int, int, int]:
    """Returns (chat_id, owner_person_id, alice_person_id)."""
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice_id = _insert_person(scratch_db, display_name="Alice Example", short_name="alice")
    chat_id = _insert_chat(scratch_db, source_guid="chat-dm-1")
    _add_participant(scratch_db, chat_id, owner_id)
    _add_participant(scratch_db, chat_id, alice_id)
    scratch_db.commit()
    return chat_id, owner_id, alice_id


_BASE = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


# --- tests -----------------------------------------------------------------


def test_small_session_becomes_one_segment(
    scratch_db: psycopg.Connection, dm_chat: tuple[int, int, int], config: Config
) -> None:
    chat_id, owner_id, alice_id = dm_chat
    for i in range(5):
        sender = owner_id if i % 2 == 0 else alice_id
        _insert_message(
            scratch_db,
            chat_id=chat_id,
            sender_person_id=sender,
            is_from_me=(i % 2 == 0),
            sent_at=_BASE + timedelta(minutes=i),
            text=f"message {i}",
        )
    scratch_db.commit()

    provider = FakeBoundaryProvider(always_fail=True)  # must never be called: 5 <= topical_min_messages
    report = run_segment_for_chat(
        scratch_db, chat_id, config, provider, PROMPT_BYTES, earliest_changed_at=REBUILD_ALL_SENTINEL
    )
    scratch_db.commit()

    assert report.sessions_written == 1
    assert report.segments_written == 1
    assert report.fallback_sessions == 0

    with scratch_db.cursor() as cur:
        cur.execute("SELECT message_count, seg_config_hash FROM segment")
        rows = cur.fetchall()
    assert rows == [(5, rows[0][1])]
    assert len(rows[0][1]) == 64

    with scratch_db.cursor() as cur:
        cur.execute("SELECT operation, content_sha256 FROM search_index_event")
        events = cur.fetchall()
    assert len(events) == 1
    assert events[0][0] == "upsert"
    assert events[0][1] is not None


def test_second_run_with_nothing_changed_is_a_noop(
    scratch_db: psycopg.Connection, dm_chat: tuple[int, int, int], config: Config
) -> None:
    chat_id, owner_id, _alice_id = dm_chat
    for i in range(3):
        _insert_message(
            scratch_db,
            chat_id=chat_id,
            sender_person_id=owner_id,
            is_from_me=True,
            sent_at=_BASE + timedelta(minutes=i),
            text=f"m{i}",
        )
    scratch_db.commit()

    provider = FakeBoundaryProvider()
    run_segment_for_chat(
        scratch_db, chat_id, config, provider, PROMPT_BYTES, earliest_changed_at=REBUILD_ALL_SENTINEL
    )
    scratch_db.commit()

    dirty = find_dirty_chats(scratch_db, index_unsent=config.policy.index_unsent)
    assert chat_id not in dirty


def test_new_message_within_gap_extends_the_tail_session(
    scratch_db: psycopg.Connection, dm_chat: tuple[int, int, int], config: Config
) -> None:
    """The v1.1 incremental-frontier fix: a reply arriving within
    `session_gap_hours` of the last message must join the existing tail
    session (one session, growing), not spawn a second one."""
    chat_id, owner_id, alice_id = dm_chat
    for i in range(3):
        _insert_message(
            scratch_db,
            chat_id=chat_id,
            sender_person_id=owner_id,
            is_from_me=True,
            sent_at=_BASE + timedelta(minutes=i),
            text=f"m{i}",
        )
    scratch_db.commit()

    provider = FakeBoundaryProvider()
    run_segment_for_chat(
        scratch_db, chat_id, config, provider, PROMPT_BYTES, earliest_changed_at=REBUILD_ALL_SENTINEL
    )
    scratch_db.commit()

    with scratch_db.cursor() as cur:
        cur.execute("SELECT session_id, segment_id FROM segment")
        (_old_session_id, old_segment_id) = cur.fetchone()  # type: ignore[misc]

    new_msg_at = _BASE + timedelta(minutes=30)  # well within the 3h default gap
    _insert_message(
        scratch_db,
        chat_id=chat_id,
        sender_person_id=alice_id,
        is_from_me=False,
        sent_at=new_msg_at,
        text="a reply",
    )
    scratch_db.commit()

    dirty = find_dirty_chats(scratch_db, index_unsent=config.policy.index_unsent)
    assert chat_id in dirty

    run_segment_for_chat(
        scratch_db, chat_id, config, provider, PROMPT_BYTES, earliest_changed_at=dirty[chat_id]
    )
    scratch_db.commit()

    with scratch_db.cursor() as cur:
        cur.execute("SELECT session_id FROM session WHERE chat_id = %s", (chat_id,))
        session_ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT message_count FROM segment WHERE chat_id = %s", (chat_id,))
        counts = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT session_id FROM segment WHERE segment_id = %s", (old_segment_id,))
        stale_check = cur.fetchall()

    assert len(session_ids) == 1  # still ONE session, not two
    assert counts == [4]  # extended tail, not a fresh 1-message segment
    assert stale_check == []  # the old segment row is gone (cascaded away)

    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT operation FROM search_index_event WHERE entity_id = %s", (old_segment_id,)
        )
        old_events = [r[0] for r in cur.fetchall()]
    assert "delete" in old_events


def test_edited_message_is_detected_dirty_and_reflected_after_rerun(
    scratch_db: psycopg.Connection, dm_chat: tuple[int, int, int], config: Config
) -> None:
    chat_id, owner_id, _alice_id = dm_chat
    msg_id = _insert_message(
        scratch_db,
        chat_id=chat_id,
        sender_person_id=owner_id,
        is_from_me=True,
        sent_at=_BASE,
        text="original text",
    )
    scratch_db.commit()

    provider = FakeBoundaryProvider()
    run_segment_for_chat(
        scratch_db, chat_id, config, provider, PROMPT_BYTES, earliest_changed_at=REBUILD_ALL_SENTINEL
    )
    scratch_db.commit()

    with scratch_db.cursor() as cur:
        cur.execute(
            "UPDATE message SET text_original = %s, is_edited = true, updated_at = now() "
            "WHERE message_id = %s",
            ("edited text", msg_id),
        )
    scratch_db.commit()

    dirty = find_dirty_chats(scratch_db, index_unsent=config.policy.index_unsent)
    assert chat_id in dirty

    run_segment_for_chat(
        scratch_db, chat_id, config, provider, PROMPT_BYTES, earliest_changed_at=dirty[chat_id]
    )
    scratch_db.commit()

    with scratch_db.cursor() as cur:
        cur.execute("SELECT rendered_text FROM segment WHERE chat_id = %s", (chat_id,))
        (rendered_text,) = cur.fetchone()  # type: ignore[misc]
    assert "edited text" in rendered_text
    assert "original text" not in rendered_text


def test_boundary_provider_splits_a_large_session_into_multiple_segments(
    scratch_db: psycopg.Connection, dm_chat: tuple[int, int, int], config: Config
) -> None:
    chat_id, owner_id, alice_id = dm_chat
    for i in range(20):
        _insert_message(
            scratch_db,
            chat_id=chat_id,
            sender_person_id=owner_id if i % 2 == 0 else alice_id,
            is_from_me=(i % 2 == 0),
            sent_at=_BASE + timedelta(minutes=i),
            text=f"message number {i}",
        )
    scratch_db.commit()

    low_threshold_config = config.model_copy(
        update={
            "segmentation": config.segmentation.model_copy(update={"topical_min_messages": 5})
        }
    )
    provider = FakeBoundaryProvider(messages_per_segment=6)
    report = run_segment_for_chat(
        scratch_db,
        chat_id,
        low_threshold_config,
        provider,
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    assert report.segments_written > 1
    with scratch_db.cursor() as cur:
        cur.execute("SELECT sum(message_count) FROM segment WHERE chat_id = %s", (chat_id,))
        (total,) = cur.fetchone()  # type: ignore[misc]
        cur.execute("SELECT count(*) FROM segment_message sm JOIN segment s USING (segment_id) WHERE s.chat_id = %s", (chat_id,))
        (linked,) = cur.fetchone()  # type: ignore[misc]
    assert total == 20
    assert linked == 20
