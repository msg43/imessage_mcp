"""`get_conversation`'s anchor without an offset is read in the owner's
time zone, not the database session's (QA review 2026-09-24, "a naive
timestamp anchor in `get_conversation` depends on the server's time zone").

`datetime.fromisoformat("2024-01-15T12:00:00")` is naive. It was handed to
Postgres as it was and compared with `timestamptz` columns, so Postgres
read it in the session's `TimeZone` — which nothing in this codebase sets.
The index host's Postgres happened to run in the same zone as
`render.timezone`, so the window was right there by coincidence; on any
other host it was off by the difference. `after`/`before` already read a
bare date in `render.timezone` (`imsg.retrieval.filters`); the anchor now
does the same.

The unit tests need no database. The integration test sets the session
to UTC and then to Asia/Tokyo, with `render.timezone` America/New_York,
and shows the same window both times.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import apsw
import psycopg
import pytest

from imsg import constants
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.fts.schema import create_schema
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.keys import message_key as derive_message_key
from imsg.keys import thread_key as derive_thread_key
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.segments import ThreadResolution, resolve_anchor
from imsg.retrieval.service import RetrievalService

RENDER_ZONE = "America/New_York"

RESOLUTION = ThreadResolution(chat_id=1, thread_key="thread_fictional", default_anchor=None)


# ---------------------------------------------------------------------------
# resolve_anchor, no database (an ISO anchor never touches the connection)
# ---------------------------------------------------------------------------


def test_a_naive_anchor_is_read_in_the_render_time_zone() -> None:
    anchor = resolve_anchor(
        cast("Any", None), RESOLUTION, "2024-01-15T12:00:00", timezone=RENDER_ZONE
    )
    assert anchor.tzinfo is not None
    assert anchor == datetime(2024, 1, 15, 17, 0, tzinfo=UTC)  # noon in New York, winter


def test_a_naive_anchor_follows_daylight_saving_in_the_render_time_zone() -> None:
    anchor = resolve_anchor(
        cast("Any", None), RESOLUTION, "2024-07-15T12:00:00", timezone=RENDER_ZONE
    )
    assert anchor == datetime(2024, 7, 15, 16, 0, tzinfo=UTC)  # noon in New York, summer


def test_an_anchor_with_an_offset_is_left_as_given() -> None:
    anchor = resolve_anchor(
        cast("Any", None), RESOLUTION, "2024-01-15T12:00:00+09:00", timezone=RENDER_ZONE
    )
    assert anchor == datetime(2024, 1, 15, 3, 0, tzinfo=UTC)


def test_a_bare_date_anchor_is_midnight_in_the_render_time_zone() -> None:
    anchor = resolve_anchor(cast("Any", None), RESOLUTION, "2024-01-15", timezone=RENDER_ZONE)
    assert anchor == datetime(2024, 1, 15, tzinfo=ZoneInfo(RENDER_ZONE))


# ---------------------------------------------------------------------------
# The window itself, against Postgres sessions in two other time zones
# ---------------------------------------------------------------------------

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_anchor_timezone_test"
REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def _dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


def _reachable() -> bool:
    try:
        conn = psycopg.connect(_dsn("postgres"), connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


needs_postgres = pytest.mark.skipif(
    not _reachable(),
    reason=(
        f"no reachable scratch Postgres instance (tried {TEST_PG_HOST}:{TEST_PG_PORT}) — "
        "set IMSG_TEST_PG_HOST/IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


@pytest.fixture
def scratch_db() -> Iterator[psycopg.Connection]:
    admin = psycopg.connect(_dsn("postgres"), autocommit=True)
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
        admin = psycopg.connect(_dsn("postgres"), autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        finally:
            admin.close()


class _Cfg:
    class _Retrieval:
        k_fts = 100
        k_vector = 100
        rrf_k = 60
        rerank_top = 20
        default_limit = 10
        hnsw_ef_search = 1000

    class _Render:
        timezone = RENDER_ZONE
        attachment_snippet_chars = 200

    class _Embedding:
        query_instruction = "search"

        class _Multimodal:
            enabled = False

        multimodal = _Multimodal()

    class _Policy:
        index_unsent = False
        index_edit_history = False

    retrieval = _Retrieval()
    render = _Render()
    embedding = _Embedding()
    policy = _Policy()


def _one(cur: psycopg.Cursor[Any]) -> int:
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


@needs_postgres
@pytest.mark.parametrize("session_zone", ["UTC", "Asia/Tokyo"])
def test_the_window_is_the_same_whatever_the_sessions_time_zone(
    scratch_db: psycopg.Connection, tmp_path: Path, session_zone: str
) -> None:
    """Four messages an hour apart from 16:00 UTC (11:00 in New York). An
    anchor of "12:00" with no offset means noon in New York, 17:00 UTC; a
    window of one is the message before it, the one at it, and the one
    after: 16:00, 17:00, 18:00 UTC."""
    first = datetime(2024, 1, 15, 16, 0, tzinfo=UTC)
    with scratch_db.cursor() as cur:
        cur.execute(
            "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
            "VALUES ('Owner Example', 'owner', true, false) RETURNING person_id"
        )
        owner = _one(cur)
        guid = f"chat-{uuid.uuid4()}"
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') "
            "RETURNING chat_id",
            (guid, derive_thread_key(guid)),
        )
        chat_id = _one(cur)
        cur.execute(
            "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)", (chat_id, owner)
        )
        for hour in range(4):
            message_guid = f"msg-{uuid.uuid4()}"
            cur.execute(
                """
                INSERT INTO message (
                    source_guid, message_key, chat_id, sender_person_id, is_from_me,
                    sent_at, service, text_original, text_normalized
                ) VALUES (%s, %s, %s, %s, true, %s, 'imessage', %s, %s)
                """,
                (
                    message_guid,
                    derive_message_key(message_guid),
                    chat_id,
                    owner,
                    first + timedelta(hours=hour),
                    f"note {hour}",
                    f"note {hour}",
                ),
            )
    fts = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(fts)
    scratch_db.execute(f"SET TIME ZONE '{session_zone}'")
    service = RetrievalService(
        pg_conn=scratch_db,
        fts_conn=fts,
        config=_Cfg(),  # type: ignore[arg-type]
        text_provider=FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM),
        reranker=FakeRerankerProvider(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM),
    )

    out = service.get_conversation(
        LOCAL_FULL_ACCESS,
        thread_id=derive_thread_key(guid),
        anchor="2024-01-15T12:00:00",
        window=1,
    )
    fts.close()

    messages = cast("list[dict[str, Any]]", out["messages"])
    sent = [datetime.fromisoformat(str(m["sent_at"])).astimezone(UTC) for m in messages]
    assert sent == [first, first + timedelta(hours=1), first + timedelta(hours=2)]
