"""Enrichment must reach the index. A segment's rendered text quotes an
attachment's derived text — a caption and OCR snippet for an image, a
transcript for audio, the first characters of a PDF or document — and S4
only re-segments a chat when a message's `updated_at` has moved
(`find_dirty_chats`). Enrichment writes to `enrichment` and never to
`message`, so a caption produced *after* a chat was segmented used to
reach `attachment_chunk` and nothing else: the segment, its embedding,
its FTS row and its export document all kept the empty-snippet
placeholder for good.

`complete_task` now marks the chats carrying the attachment, mirroring
`imsg.stages.identity._mark_chats_dirty_for_persons`, which exists for
the same reason. This suite drives the path that was broken: segment a
chat whose image has no caption yet, finish the caption, observe the
chat come back dirty, re-segment, and find the caption in the rendered
text.

OCR escaped the bug only by accident of ordering — it ran before
segmentation on the first corpus — which is why this went unnoticed
until 69,996 captions were queued behind it.

DB-gated, same scratch-instance pattern and fixture-data helpers as
`test_segment_pipeline_integration.py`. Fictional personas only.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import timedelta

import psycopg
import pytest

from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.enrich.queue import RENDERED_KINDS, complete_task, enqueue, skip_task
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import find_dirty_chats, run_segment_for_chat
from test_segment_pipeline_integration import (
    _BASE,
    ADMIN_DSN,
    PROMPT_BYTES,
    REACHABLE,
    REAL_MIGRATIONS_DIR,
    _add_participant,
    _dsn,
    _insert_chat,
    _insert_message,
    _insert_person,
)

TEST_DB_NAME = "imsg_index_enrich_dirty_test"

pytestmark = pytest.mark.skipif(
    not REACHABLE,
    reason="no reachable scratch Postgres instance (see test_segment_pipeline_integration.py)",
)


@pytest.fixture
def config(config_dict_factory: object) -> Config:
    return load_config_dict(config_dict_factory())  # type: ignore[operator]


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


# --- helpers ---------------------------------------------------------------


def _insert_image_attachment(conn: psycopg.Connection, message_id: int) -> int:
    """An image attachment on `message_id`, with its enrichment queued."""
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, mime_type, state)
            VALUES (%s, %s, 'image/jpeg', 'materialized')
            RETURNING attachment_id
            """,
            (guid, f"key-{guid}"),
        )
        row = cur.fetchone()
        assert row is not None
        attachment_id = int(row[0])
        cur.execute(
            "INSERT INTO message_attachment (message_id, attachment_id, ordinal) VALUES (%s, %s, 0)",
            (message_id, attachment_id),
        )
    enqueue(conn, attachment_id, ("ocr", "caption"))
    return attachment_id


def _seed_chat_with_an_image(conn: psycopg.Connection) -> tuple[int, int]:
    """A DM whose single message carries one un-enriched image.
    Returns `(chat_id, attachment_id)`."""
    owner_id = _insert_person(conn, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice_id = _insert_person(conn, display_name="Alice Example", short_name="alice")
    chat_id = _insert_chat(conn, source_guid="chat-dm-image")
    _add_participant(conn, chat_id, owner_id)
    _add_participant(conn, chat_id, alice_id)
    message_id = _insert_message(
        conn,
        chat_id=chat_id,
        sender_person_id=alice_id,
        is_from_me=False,
        sent_at=_BASE,
        text="look at this",
    )
    attachment_id = _insert_image_attachment(conn, message_id)
    conn.commit()
    return chat_id, attachment_id


def _segment(conn: psycopg.Connection, chat_id: int, config: Config) -> None:
    run_segment_for_chat(
        conn,
        chat_id=chat_id,
        config=config,
        boundary_provider=FakeBoundaryProvider(),
        boundary_prompt_bytes=PROMPT_BYTES,
        earliest_changed_at=_BASE - timedelta(days=1),
        latest_changed_at=_BASE + timedelta(days=1),
    )
    conn.commit()


def _dirty_chat_ids(conn: psycopg.Connection) -> set[int]:
    return set(find_dirty_chats(conn, index_unsent=False))


def _rendered(conn: psycopg.Connection, chat_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT rendered_text FROM segment WHERE chat_id = %s", (chat_id,))
        return "\n".join(str(r[0]) for r in cur.fetchall())


def _finish(conn: psycopg.Connection, attachment_id: int, kind: str, text: str | None) -> None:
    complete_task(
        conn,
        attachment_id,
        kind,
        model="fake-model",
        model_version="1",
        text=text,
    )
    conn.commit()


# --- tests -----------------------------------------------------------------


def test_caption_after_segmentation_makes_the_chat_dirty_and_reaches_the_text(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """The path that was broken end to end."""
    chat_id, attachment_id = _seed_chat_with_an_image(scratch_db)
    _segment(scratch_db, chat_id, config)

    before = _rendered(scratch_db, chat_id)
    assert 'caption "—"' in before, "the image starts with no caption"
    assert _dirty_chat_ids(scratch_db) == set(), "nothing pending before the caption"

    _finish(scratch_db, attachment_id, "caption", "two people on a sailboat at dusk")

    assert _dirty_chat_ids(scratch_db) == {chat_id}, "the caption must make the chat dirty"

    _segment(scratch_db, chat_id, config)
    after = _rendered(scratch_db, chat_id)
    assert "two people on a sailboat at dusk" in after
    assert _dirty_chat_ids(scratch_db) == set(), "settled once re-segmented"


def test_empty_caption_does_not_make_the_chat_dirty(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """An OCR or caption pass that legitimately found nothing renders the
    same placeholder, so re-segmenting would burn the stage to produce an
    identical row. 40,877 of the real pending captions are this case."""
    chat_id, attachment_id = _seed_chat_with_an_image(scratch_db)
    _segment(scratch_db, chat_id, config)
    assert _dirty_chat_ids(scratch_db) == set()

    _finish(scratch_db, attachment_id, "ocr", "   ")
    assert _dirty_chat_ids(scratch_db) == set(), "whitespace-only text changes no rendered byte"

    _finish(scratch_db, attachment_id, "caption", None)
    assert _dirty_chat_ids(scratch_db) == set(), "a null caption changes no rendered byte"


def test_a_skipped_task_does_not_make_the_chat_dirty(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    chat_id, attachment_id = _seed_chat_with_an_image(scratch_db)
    _segment(scratch_db, chat_id, config)
    skip_task(scratch_db, attachment_id, "caption", reason="link preview")
    scratch_db.commit()
    assert _dirty_chat_ids(scratch_db) == set()


def test_ocr_text_also_reaches_the_segment(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """Captions are not special — every kind the renderer quotes behaves
    the same way. OCR only avoided the bug by running before segmentation."""
    chat_id, attachment_id = _seed_chat_with_an_image(scratch_db)
    _segment(scratch_db, chat_id, config)

    _finish(scratch_db, attachment_id, "ocr", "BOARDING PASS 14C")
    assert _dirty_chat_ids(scratch_db) == {chat_id}

    _segment(scratch_db, chat_id, config)
    assert "BOARDING PASS 14C" in _rendered(scratch_db, chat_id)


def test_rendered_kinds_covers_every_kind_the_renderer_quotes() -> None:
    """A guard on the constant itself: if a new enrichment kind starts
    rendering and is not added to `RENDERED_KINDS`, its text silently
    stops reaching segments — the original bug, one kind at a time."""
    assert frozenset(
        {"caption", "ocr", "frame_ocr", "transcript", "pdf_text", "doc_text"}
    ) == RENDERED_KINDS
