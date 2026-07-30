"""Live-Postgres integration tests for AT-3 completion
(SPEC §12 AT-3, `imsg.verify.attachments`) — layered on top of the
already-tested `imsg.backfill.reconcile.build_reconciliation_report`
(`tests/test_backfill_pipeline_integration.py`), so these tests focus
on what this module *adds*: the exception manifest, by-year/by-type
breakdown, CSV rendering, and the stratified integrity sample.
"""

from __future__ import annotations

import csv
import io
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner
from imsg.hashing import sha256_text
from imsg.verify.attachments import (
    EXCEPTION_CATEGORIES,
    build_at3_report,
    report_to_csv,
)

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_verify_at3_test"

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


def _insert_chat_and_message(conn: psycopg.Connection, *, sent_at: datetime) -> int:
    chat_guid = f"chat-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') "
            "RETURNING chat_id",
            (chat_guid, f"thread-{chat_guid}"),
        )
        chat_id = int(cur.fetchone()[0])  # type: ignore[index]
        msg_guid = f"msg-{uuid.uuid4()}"
        cur.execute(
            """
            INSERT INTO message (source_guid, message_key, chat_id, is_from_me, sent_at,
                                  service, text_original, text_normalized)
            VALUES (%s, %s, %s, true, %s, 'imessage', 'x', 'x') RETURNING message_id
            """,
            (msg_guid, f"key-{msg_guid}", chat_id, sent_at),
        )
        message_id = int(cur.fetchone()[0])  # type: ignore[index]
    return message_id


def _insert_attachment(
    conn: psycopg.Connection,
    *,
    message_id: int,
    state: str,
    cache_path: str | None = None,
    sha256: str | None = None,
    mime_type: str | None = "image/png",
) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, state, cache_path, sha256, mime_type)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING attachment_id
            """,
            (guid, f"akey-{guid}", state, cache_path, sha256, mime_type),
        )
        attachment_id = int(cur.fetchone()[0])  # type: ignore[index]
        cur.execute(
            "INSERT INTO message_attachment (message_id, attachment_id, ordinal) VALUES (%s, %s, 0)",
            (message_id, attachment_id),
        )
    return attachment_id


def test_at3_report_classifies_every_gap_into_a_known_category(
    scratch_db: psycopg.Connection, tmp_path: Path
) -> None:
    m1 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 6, 1, tzinfo=UTC))
    m2 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 6, 2, tzinfo=UTC))
    m3 = _insert_chat_and_message(scratch_db, sent_at=datetime(2024, 1, 1, tzinfo=UTC))
    m4 = _insert_chat_and_message(scratch_db, sent_at=datetime(2024, 1, 2, tzinfo=UTC))

    _insert_attachment(scratch_db, message_id=m1, state="dataless")
    _insert_attachment(scratch_db, message_id=m2, state="missing")
    _insert_attachment(scratch_db, message_id=m3, state="error")
    _insert_attachment(scratch_db, message_id=m4, state="materializing")

    report = build_at3_report(scratch_db, integrity_sample_size=10)

    assert report.total == 4
    assert report.materialized_and_present == 0
    categories = {e.category for e in report.exceptions}
    assert categories == {"dataless_retrying", "remote_missing", "error"}
    assert all(e.category in EXCEPTION_CATEGORIES for e in report.exceptions)
    assert report.by_year["2023"] == (0, 2)
    assert report.by_year["2024"] == (0, 2)
    # No materialized files -> nothing to sample -> integrity sample trivially clean.
    assert report.integrity_sample == ()
    assert report.passed is True


def test_at3_report_flags_materialized_but_missing_file_as_error(
    scratch_db: psycopg.Connection,
) -> None:
    m1 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 1, 1, tzinfo=UTC))
    _insert_attachment(
        scratch_db, message_id=m1, state="materialized", cache_path="/no/such/path/at/all.png"
    )
    report = build_at3_report(scratch_db, integrity_sample_size=10)
    assert report.materialized_and_present == 0
    assert len(report.exceptions) == 1
    assert report.exceptions[0].category == "error"


def test_at3_report_integrity_sample_verifies_sha256(
    scratch_db: psycopg.Connection, tmp_path: Path
) -> None:
    m1 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 1, 1, tzinfo=UTC))
    good_file = tmp_path / "good.txt"
    good_file.write_text("hello world")
    good_sha = sha256_text("hello world")
    _insert_attachment(
        scratch_db, message_id=m1, state="materialized", cache_path=str(good_file), sha256=good_sha
    )

    m2 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 1, 2, tzinfo=UTC))
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("tampered content")
    _insert_attachment(
        scratch_db, message_id=m2, state="materialized", cache_path=str(bad_file),
        sha256=good_sha,  # deliberately wrong — simulates a corrupted/mismatched cache file
    )

    report = build_at3_report(scratch_db, integrity_sample_size=10)
    assert report.materialized_and_present == 2
    assert len(report.integrity_sample) == 2
    by_path = {s.cache_path: s for s in report.integrity_sample}
    assert by_path[str(good_file)].ok is True
    assert by_path[str(bad_file)].ok is False
    assert "mismatch" in by_path[str(bad_file)].detail
    assert report.passed is False
    assert any("integrity" in r for r in report.reasons)


def test_at3_report_by_mime_type_breakdown(scratch_db: psycopg.Connection, tmp_path: Path) -> None:
    m1 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 1, 1, tzinfo=UTC))
    f = tmp_path / "a.pdf"
    f.write_text("content")
    _insert_attachment(
        scratch_db, message_id=m1, state="materialized", cache_path=str(f),
        sha256=sha256_text("content"), mime_type="application/pdf",
    )
    m2 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 1, 2, tzinfo=UTC))
    _insert_attachment(scratch_db, message_id=m2, state="dataless", mime_type=None)

    report = build_at3_report(scratch_db, integrity_sample_size=10)
    assert report.by_mime_type["application/pdf"] == (1, 1)
    assert report.by_mime_type["unknown"] == (0, 1)


def test_report_to_csv_contains_all_sections(scratch_db: psycopg.Connection, tmp_path: Path) -> None:
    m1 = _insert_chat_and_message(scratch_db, sent_at=datetime(2023, 1, 1, tzinfo=UTC))
    _insert_attachment(scratch_db, message_id=m1, state="dataless")
    report = build_at3_report(scratch_db, integrity_sample_size=10)

    csv_text = report_to_csv(report)
    rows = list(csv.reader(io.StringIO(csv_text)))
    header_rows = [r for r in rows if r and r[0] == "section"]
    assert header_rows  # overall/by_year/by_mime_type summary section present
    attachment_header = [r for r in rows if r == ["attachment_id", "attachment_key", "state", "category", "reason"]]
    assert attachment_header
    sample_header = [
        r for r in rows
        if r == ["sample_attachment_id", "sample_attachment_key", "cache_path", "ok", "detail"]
    ]
    assert sample_header


def test_at3_report_empty_corpus_passes_trivially(scratch_db: psycopg.Connection) -> None:
    report = build_at3_report(scratch_db, integrity_sample_size=10)
    assert report.total == 0
    assert report.completeness_ratio == 1.0
    assert report.passed is True
