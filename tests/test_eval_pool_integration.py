"""Live-Postgres integration tests for the §13.2 pooling workflow
(`imsg.eval.pool`): build a pool from >= 2 backends, render the
worksheet, and import graded results back as `relevance_label` rows.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
import yaml

from imsg.db.migrations import PostgresMigrationRunner
from imsg.eval.io import load_labels, upsert_query
from imsg.eval.models import EvalQuery
from imsg.eval.pool import build_pool, import_pool_worksheet, pool_to_worksheet_yaml
from imsg.keys import thread_key as derive_thread_key

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_eval_pool_test"

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


def _insert_segment(conn: psycopg.Connection, *, rendered_text: str) -> str:
    chat_guid = f"chat-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') RETURNING chat_id",
            (chat_guid, derive_thread_key(chat_guid)),
        )
        chat_id = int(cur.fetchone()[0])  # type: ignore[index]
        started = datetime(2024, 1, 1, tzinfo=UTC)
        cur.execute(
            "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
            "VALUES (%s, %s, %s, 3.0) RETURNING session_id",
            (chat_id, started, started),
        )
        session_id = int(cur.fetchone()[0])  # type: ignore[index]
        stable_key = f"stable-{uuid.uuid4()}"
        cur.execute(
            """
            INSERT INTO segment (stable_key, chat_id, session_id, seq_in_session, started_at,
                                  ended_at, message_count, rendered_text, rendered_sha256, seg_config_hash)
            VALUES (%s, %s, %s, 0, %s, %s, 1, %s, 'x', 'cfg')
            RETURNING segment_id
            """,
            (stable_key, chat_id, session_id, started, started, rendered_text),
        )
        segment_id = int(cur.fetchone()[0])  # type: ignore[index]
        msg_guid = f"msg-{uuid.uuid4()}"
        cur.execute(
            """
            INSERT INTO message (source_guid, message_key, chat_id, is_from_me, sent_at,
                                  service, text_original, text_normalized)
            VALUES (%s, %s, %s, true, %s, 'imessage', %s, %s) RETURNING message_id
            """,
            (msg_guid, f"key-{msg_guid}", chat_id, started, rendered_text, rendered_text),
        )
        message_id = int(cur.fetchone()[0])  # type: ignore[index]
        cur.execute(
            "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
            (segment_id, message_id),
        )
    return stable_key


class _FixedBackend:
    """A trivial `EvalBackend`: always returns the same segment_keys."""

    def __init__(self, segment_keys: list[str]) -> None:
        self._keys = segment_keys

    def search(self, query_text: str, *, k: int) -> list[str]:
        del query_text
        return self._keys[:k]


def test_build_pool_unions_and_dedupes_across_backends(scratch_db: psycopg.Connection) -> None:
    key_a = _insert_segment(scratch_db, rendered_text="the deck rebuild bid")
    key_b = _insert_segment(scratch_db, rendered_text="lunch plans tomorrow")
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="deck bid"))

    backend_default = _FixedBackend([key_a, key_b])
    backend_variant = _FixedBackend([key_b])  # only surfaces key_b

    entries = build_pool(
        scratch_db,
        {"default": backend_default, "no-rerank": backend_variant},
        [EvalQuery(query_id="q001", query_text="deck bid")],
        top_n=20,
        seed=42,
    )
    by_key = {e.segment_key: e for e in entries}
    assert set(by_key) == {key_a, key_b}
    assert by_key[key_a].source_configs == ("default",)
    assert by_key[key_b].source_configs == ("default", "no-rerank")
    assert "deck rebuild bid" in by_key[key_a].text_preview


def test_pool_worksheet_roundtrip_and_import(scratch_db: psycopg.Connection) -> None:
    key_a = _insert_segment(scratch_db, rendered_text="the deck rebuild bid")
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="deck bid"))
    entries = build_pool(
        scratch_db,
        {"default": _FixedBackend([key_a]), "no-rerank": _FixedBackend([key_a])},
        [EvalQuery(query_id="q001", query_text="deck bid")],
        top_n=20,
        seed=1,
    )
    worksheet = pool_to_worksheet_yaml(entries)
    parsed = yaml.safe_load(worksheet)
    assert parsed[0]["grade"] is None

    # Owner grades it.
    parsed[0]["grade"] = 2
    graded = yaml.safe_dump(parsed)

    imported = import_pool_worksheet(scratch_db, graded)
    assert imported == 1

    labels = load_labels(scratch_db, query_id="q001")
    assert len(labels) == 1
    assert labels[0].grade == 2
    assert labels[0].source == "pool_judgment"


def test_import_pool_worksheet_skips_ungraded_entries(scratch_db: psycopg.Connection) -> None:
    key_a = _insert_segment(scratch_db, rendered_text="x")
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="x"))
    entries = build_pool(
        scratch_db, {"default": _FixedBackend([key_a]), "no-rerank": _FixedBackend([key_a])},
        [EvalQuery(query_id="q001", query_text="x")], top_n=20, seed=1,
    )
    worksheet = pool_to_worksheet_yaml(entries)  # every grade still null
    imported = import_pool_worksheet(scratch_db, worksheet)
    assert imported == 0
    assert load_labels(scratch_db, query_id="q001") == []
