"""Postgres integration tests for the S5b enrichment pipeline end to
end (SPEC §8 S5b) — claim -> sniff -> dispatch -> write chunks/events
-> re-render parent segments. Skips cleanly when no scratch Postgres is
reachable, same pattern as `tests/test_migrations_integration.py`.
Fictional personas only (D5).
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from _pdf_fixtures import write_minimal_pdf
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.enrich.pipeline import EnrichmentProviders, process_one_task
from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider
from imsg.enrich.queue import claim_tasks, enqueue
from imsg.enrich.router import kinds_for_mime
from imsg.segment.hashing import compute_seg_config_hash, compute_stable_key

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_enrich_pipeline_test"

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
def config(config_dict_factory: object) -> Config:
    return load_config_dict(config_dict_factory())  # type: ignore[operator]


@pytest.fixture
def providers() -> EnrichmentProviders:
    return EnrichmentProviders(
        ocr=FakeOcrProvider(), caption=FakeCaptionProvider(), transcription=FakeTranscriptionProvider()
    )


_BASE = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


def _insert_person(conn: psycopg.Connection, *, display_name: str, short_name: str, is_owner: bool = False) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
            "VALUES (%s, %s, %s, false) RETURNING person_id",
            (display_name, short_name, is_owner),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_chat(conn: psycopg.Connection, *, source_guid: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') "
            "RETURNING chat_id",
            (source_guid, f"thread-{source_guid}"),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_message(conn: psycopg.Connection, *, chat_id: int, sender_person_id: int, is_from_me: bool, text: str) -> int:
    guid = f"msg-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO message (
                source_guid, message_key, chat_id, sender_person_id,
                is_from_me, sent_at, service, text_original, text_normalized, has_attachments
            ) VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s, true)
            RETURNING message_id
            """,
            (guid, f"key-{guid}", chat_id, sender_person_id, is_from_me, _BASE, text, text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_attachment(conn: psycopg.Connection, *, cache_path: Path, filename: str) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, filename, cache_path, state)
            VALUES (%s, %s, %s, %s, 'materialized')
            RETURNING attachment_id
            """,
            (guid, f"key-{guid}", filename, str(cache_path)),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _link_message_attachment(conn: psycopg.Connection, message_id: int, attachment_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO message_attachment (message_id, attachment_id, ordinal) VALUES (%s, %s, 0)",
            (message_id, attachment_id),
        )


def _seed_minimal_segment(
    conn: psycopg.Connection, *, chat_id: int, message_id: int, chat_source_guid: str, message_guid_row: str
) -> int:
    """Insert a `session` + `segment` (+ `segment_message`) directly,
    bypassing S4's pipeline — S5b's job is re-rendering an *existing*
    segment, not building one."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
            "VALUES (%s, %s, %s, 3.0) RETURNING session_id",
            (chat_id, _BASE, _BASE),
        )
        row = cur.fetchone()
        assert row is not None
        session_id = row[0]

        seg_hash = compute_seg_config_hash(
            session_gap_hours=3.0, topical_min_messages=10, max_messages=50, max_tokens=2000,
            boundary_model="test-model", boundary_prompt_bytes=b"prompt", index_unsent=False,
            index_edit_history=False,
        )
        stable_key = compute_stable_key(
            chat_source_guid=chat_source_guid, first_message_guid=message_guid_row,
            last_message_guid=message_guid_row, seg_config_hash=seg_hash,
        )
        cur.execute(
            """
            INSERT INTO segment (
                stable_key, chat_id, session_id, seq_in_session, started_at, ended_at,
                message_count, token_count, rendered_text, rendered_sha256, seg_config_hash
            ) VALUES (%s, %s, %s, 0, %s, %s, 1, 10, 'placeholder text', 'placeholder-hash', %s)
            RETURNING segment_id
            """,
            (stable_key, chat_id, session_id, _BASE, _BASE, seg_hash),
        )
        row = cur.fetchone()
        assert row is not None
        segment_id = int(row[0])

        cur.execute(
            "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
            (segment_id, message_id),
        )
    return segment_id


def _message_source_guid(conn: psycopg.Connection, message_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT source_guid FROM message WHERE message_id = %s", (message_id,))
        row = cur.fetchone()
        assert row is not None
        return str(row[0])


# --- tests -------------------------------------------------------------


def test_pdf_text_end_to_end_writes_chunks_events_and_refreshes_segment(
    scratch_db: psycopg.Connection, data_root: Path, config: Config, providers: EnrichmentProviders
) -> None:
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    chat_id = _insert_chat(scratch_db, source_guid="chat-1")
    message_id = _insert_message(scratch_db, chat_id=chat_id, sender_person_id=owner_id, is_from_me=True, text="see attached")

    pdf_path = data_root / "bid.pdf"
    write_minimal_pdf(pdf_path, ["Deck rebuild materials cost fourteen thousand dollars total"])
    att_id = _insert_attachment(scratch_db, cache_path=pdf_path, filename="bid.pdf")
    _link_message_attachment(scratch_db, message_id, att_id)

    guid = _message_source_guid(scratch_db, message_id)
    segment_id = _seed_minimal_segment(
        scratch_db, chat_id=chat_id, message_id=message_id, chat_source_guid="chat-1", message_guid_row=guid
    )

    enqueue(scratch_db, att_id, kinds_for_mime("application/pdf"))
    tasks = claim_tasks(scratch_db, worker_id="w1", limit=10)
    assert len(tasks) == 1

    outcome = process_one_task(scratch_db, config, providers, tasks[0])
    assert outcome == "done"

    with scratch_db.cursor() as cur:
        cur.execute("SELECT state, text, model FROM enrichment WHERE attachment_id = %s AND kind = 'pdf_text'", (att_id,))
        state, text, model = cur.fetchone()  # type: ignore[misc]
    assert state == "done"
    assert "fourteen thousand" in text
    assert model == "pdftotext"

    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT seq, text FROM attachment_chunk WHERE attachment_id = %s AND kind = 'pdf_text' ORDER BY seq",
            (att_id,),
        )
        chunks = cur.fetchall()
    assert len(chunks) >= 1
    assert "fourteen thousand" in chunks[0][1]

    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT operation, content_sha256 FROM search_index_event WHERE entity_kind = 'attachment_chunk'"
        )
        chunk_events = cur.fetchall()
    assert len(chunk_events) == len(chunks)
    assert all(op == "upsert" and sha is not None for op, sha in chunk_events)

    with scratch_db.cursor() as cur:
        cur.execute("SELECT rendered_text FROM segment WHERE segment_id = %s", (segment_id,))
        (rendered_text,) = cur.fetchone()  # type: ignore[misc]
    assert rendered_text != "placeholder text"
    assert "fourteen thousand" in rendered_text

    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM search_index_event WHERE entity_kind = 'segment' AND entity_id = %s AND operation = 'upsert'",
            (segment_id,),
        )
        (segment_event_count,) = cur.fetchone()  # type: ignore[misc]
    assert segment_event_count == 1


def test_scanned_pdf_enqueues_ocr_follow_up(
    scratch_db: psycopg.Connection, data_root: Path, config: Config, providers: EnrichmentProviders
) -> None:
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    chat_id = _insert_chat(scratch_db, source_guid="chat-2")
    message_id = _insert_message(scratch_db, chat_id=chat_id, sender_person_id=owner_id, is_from_me=True, text="scan")

    pdf_path = data_root / "scan.pdf"
    write_minimal_pdf(pdf_path, ["x"])  # sparse text -> "scanned"
    att_id = _insert_attachment(scratch_db, cache_path=pdf_path, filename="scan.pdf")
    _link_message_attachment(scratch_db, message_id, att_id)
    guid = _message_source_guid(scratch_db, message_id)
    _seed_minimal_segment(scratch_db, chat_id=chat_id, message_id=message_id, chat_source_guid="chat-2", message_guid_row=guid)

    enqueue(scratch_db, att_id, kinds_for_mime("application/pdf"))
    tasks = claim_tasks(scratch_db, worker_id="w1", limit=10)
    outcome = process_one_task(scratch_db, config, providers, tasks[0])
    assert outcome == "done"

    with scratch_db.cursor() as cur:
        cur.execute("SELECT state FROM enrichment WHERE attachment_id = %s AND kind = 'ocr'", (att_id,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "pending"


def test_image_ocr_and_caption(
    scratch_db: psycopg.Connection, data_root: Path, config: Config, providers: EnrichmentProviders
) -> None:
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    chat_id = _insert_chat(scratch_db, source_guid="chat-3")
    message_id = _insert_message(scratch_db, chat_id=chat_id, sender_person_id=owner_id, is_from_me=True, text="photo")

    image_path = data_root / "photo.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=8x8", "-frames:v", "1", str(image_path)],
        check=True, capture_output=True, timeout=15,
    )
    att_id = _insert_attachment(scratch_db, cache_path=image_path, filename="photo.png")
    _link_message_attachment(scratch_db, message_id, att_id)
    guid = _message_source_guid(scratch_db, message_id)
    _seed_minimal_segment(scratch_db, chat_id=chat_id, message_id=message_id, chat_source_guid="chat-3", message_guid_row=guid)

    enqueue(scratch_db, att_id, kinds_for_mime("image/png"))
    tasks = claim_tasks(scratch_db, worker_id="w1", limit=10)
    assert {t.kind for t in tasks} == {"ocr", "caption"}

    outcomes = {t.kind: process_one_task(scratch_db, config, providers, t) for t in tasks}
    assert outcomes == {"ocr": "done", "caption": "done"}

    with scratch_db.cursor() as cur:
        cur.execute("SELECT kind, model, text FROM enrichment WHERE attachment_id = %s ORDER BY kind", (att_id,))
        rows = cur.fetchall()
    by_kind = {k: (m, t) for k, m, t in rows}
    assert by_kind["ocr"][0] == FakeOcrProvider.model_id
    assert by_kind["caption"][0] == FakeCaptionProvider.model_id


def test_oversized_attachment_fails_permanently_without_retry(
    scratch_db: psycopg.Connection, data_root: Path, config: Config, providers: EnrichmentProviders
) -> None:
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    chat_id = _insert_chat(scratch_db, source_guid="chat-4")
    message_id = _insert_message(scratch_db, chat_id=chat_id, sender_person_id=owner_id, is_from_me=True, text="big")

    pdf_path = data_root / "big.pdf"
    write_minimal_pdf(pdf_path, ["some content"])
    att_id = _insert_attachment(scratch_db, cache_path=pdf_path, filename="big.pdf")
    _link_message_attachment(scratch_db, message_id, att_id)

    tiny_limit_config = config.model_copy(
        update={
            "enrichment": config.enrichment.model_copy(
                update={"limits": config.enrichment.limits.model_copy(update={"max_file_bytes": 1})}
            )
        }
    )

    enqueue(scratch_db, att_id, ("pdf_text",))
    tasks = claim_tasks(scratch_db, worker_id="w1", limit=10)
    outcome = process_one_task(scratch_db, tiny_limit_config, providers, tasks[0])
    assert outcome == "failed"

    with scratch_db.cursor() as cur:
        cur.execute("SELECT state, attempts FROM enrichment WHERE attachment_id = %s AND kind = 'pdf_text'", (att_id,))
        state, attempts = cur.fetchone()  # type: ignore[misc]
    assert state == "failed"
    assert attempts == 1  # no backoff/retry dance for untrusted-boundary violations


def test_mime_kind_mismatch_is_skipped_not_failed(
    scratch_db: psycopg.Connection, data_root: Path, config: Config, providers: EnrichmentProviders
) -> None:
    """A task enqueued for 'ocr' whose *actual* sniffed content isn't a
    PDF or image (SPEC §8 S5b: "unsupported type (skipped)")."""
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    chat_id = _insert_chat(scratch_db, source_guid="chat-5")
    message_id = _insert_message(scratch_db, chat_id=chat_id, sender_person_id=owner_id, is_from_me=True, text="mismatch")

    text_path = data_root / "notes.dat"
    text_path.write_text("just plain text, not an image or pdf at all")
    att_id = _insert_attachment(scratch_db, cache_path=text_path, filename="notes.dat")
    _link_message_attachment(scratch_db, message_id, att_id)

    enqueue(scratch_db, att_id, ("ocr",))
    tasks = claim_tasks(scratch_db, worker_id="w1", limit=10)
    outcome = process_one_task(scratch_db, config, providers, tasks[0])
    assert outcome == "skipped"

    with scratch_db.cursor() as cur:
        cur.execute("SELECT state FROM enrichment WHERE attachment_id = %s AND kind = 'ocr'", (att_id,))
        (state,) = cur.fetchone()  # type: ignore[misc]
    assert state == "skipped"
