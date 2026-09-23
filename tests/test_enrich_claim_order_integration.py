"""Claim order and kind selection for the S5b enrichment queue (SPEC §8
S5b): cheap, high-recall kinds are handed out before captions, `kinds`
both filters and orders what a worker claims, and within one kind the
newest attachment goes first. The lease, backoff and SKIP LOCKED
mechanics themselves are covered in `test_enrich_queue_integration.py`.

Skips cleanly when no scratch Postgres is reachable.
"""

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
    enqueue,
    fail_task,
    fail_task_permanently,
    preview_claimable_tasks,
)

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_enrich_claim_order_test"

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


# --------------------------------------------------------------------------
# claim order: cheap kinds first, captions last; `kinds` filters and orders
# --------------------------------------------------------------------------


def _claim_one_at_a_time(conn: psycopg.Connection, **kwargs: object) -> list[tuple[int, str]]:
    order: list[tuple[int, str]] = []
    while True:
        batch = claim_tasks(conn, worker_id="worker-1", limit=1, **kwargs)  # type: ignore[arg-type]
        if not batch:
            return order
        order.extend((t.attachment_id, t.kind) for t in batch)


def test_default_claim_order_puts_cheap_kinds_first_and_captions_last(
    scratch_db: psycopg.Connection,
) -> None:
    """Enqueued caption-first, the queue must still hand out every cheap
    kind before any caption: a caption costs about 14 s of GPU on the
    production host, OCR and text extraction a fraction of a second."""
    photo = _insert_attachment(scratch_db)
    video = _insert_attachment(scratch_db)
    pdf = _insert_attachment(scratch_db)
    card = _insert_attachment(scratch_db)
    voice = _insert_attachment(scratch_db)
    enqueue(scratch_db, photo, ("caption", "ocr"))
    enqueue(scratch_db, video, ("caption", "transcript", "frame_ocr"))
    enqueue(scratch_db, pdf, ("pdf_text",))
    enqueue(scratch_db, card, ("doc_text",))
    enqueue(scratch_db, voice, ("transcript",))

    kinds = [kind for _, kind in _claim_one_at_a_time(scratch_db)]

    assert len(kinds) == 8
    assert kinds[-2:] == ["caption", "caption"]
    assert kinds.index("doc_text") < kinds.index("ocr")
    assert kinds.index("pdf_text") < kinds.index("ocr")
    assert "caption" not in kinds[:6]


def test_one_claim_prefers_a_cheap_kind_over_an_older_caption(
    scratch_db: psycopg.Connection,
) -> None:
    older_photo = _insert_attachment(scratch_db)
    enqueue(scratch_db, older_photo, ("caption",))
    newer_card = _insert_attachment(scratch_db)
    enqueue(scratch_db, newer_card, ("doc_text",))

    claimed = claim_tasks(scratch_db, worker_id="worker-1", limit=1)

    assert [(c.attachment_id, c.kind) for c in claimed] == [(newer_card, "doc_text")]


def test_kinds_filter_claims_only_those_kinds(scratch_db: psycopg.Connection) -> None:
    photo = _insert_attachment(scratch_db)
    enqueue(scratch_db, photo, ("ocr", "caption"))
    pdf = _insert_attachment(scratch_db)
    enqueue(scratch_db, pdf, ("pdf_text",))

    claimed = claim_tasks(scratch_db, worker_id="worker-1", limit=10, kinds=("ocr", "pdf_text"))

    assert {(c.attachment_id, c.kind) for c in claimed} == {(photo, "ocr"), (pdf, "pdf_text")}
    assert _state(scratch_db, photo, "caption") == "pending"


def test_the_order_of_kinds_is_the_claim_order(scratch_db: psycopg.Connection) -> None:
    """`--kinds transcript,ocr` means transcripts first: the owner can put
    voice messages ahead of the OCR backlog without touching the queue."""
    photo = _insert_attachment(scratch_db)
    enqueue(scratch_db, photo, ("ocr",))
    voice = _insert_attachment(scratch_db)
    enqueue(scratch_db, voice, ("transcript",))

    order = _claim_one_at_a_time(scratch_db, kinds=("transcript", "ocr"))

    assert order == [(voice, "transcript"), (photo, "ocr")]


def test_within_a_kind_the_newest_attachment_is_claimed_first(
    scratch_db: psycopg.Connection,
) -> None:
    first = _insert_attachment(scratch_db)
    second = _insert_attachment(scratch_db)
    third = _insert_attachment(scratch_db)
    for att in (first, second, third):
        enqueue(scratch_db, att, ("caption",))

    order = _claim_one_at_a_time(scratch_db)

    assert order == [(third, "caption"), (second, "caption"), (first, "caption")]


def test_a_backed_off_task_is_not_claimed_before_its_time(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr",))
    claim_tasks(scratch_db, worker_id="worker-1", limit=1)
    fail_task(scratch_db, att_id, "ocr", error="transient", max_attempts=5)

    assert claim_tasks(scratch_db, worker_id="worker-1", limit=1) == []


def test_preview_honours_the_kinds_filter(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr", "caption"))

    preview = preview_claimable_tasks(scratch_db, kinds=("ocr",))

    assert preview.total == 1
    assert preview.by_kind == {"ocr": 1}


def test_reset_failed_tasks_is_limited_to_the_selected_kinds(
    scratch_db: psycopg.Connection,
) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr", "caption"))
    claim_tasks(scratch_db, worker_id="worker-1", limit=10)
    fail_task_permanently(scratch_db, att_id, "ocr", error="bad")
    fail_task_permanently(scratch_db, att_id, "caption", error="bad")

    from imsg.enrich.queue import reset_failed_tasks

    reset = reset_failed_tasks(scratch_db, kinds=("ocr",))

    assert reset == 1
    assert _state(scratch_db, att_id, "ocr") == "pending"
    assert _state(scratch_db, att_id, "caption") == "failed"


def test_enqueue_pairs_reports_only_rows_it_inserted(scratch_db: psycopg.Connection) -> None:
    att_id = _insert_attachment(scratch_db)
    enqueue(scratch_db, att_id, ("ocr",))

    from imsg.enrich.queue import enqueue_pairs

    inserted = enqueue_pairs(scratch_db, [(att_id, "ocr"), (att_id, "caption")])

    assert inserted == 1
    assert enqueue_pairs(scratch_db, [(att_id, "ocr"), (att_id, "caption")]) == 0
