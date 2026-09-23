"""Unit tests for AT-2's `--reference-db` path
(`imsg.verify.seed.build_seed_snapshot_from_chat_db`).

No Postgres and no config file: the whole point of this path is that
the reference comes from a `chat.db`-shaped SQLite file, so the fixture
below *is* the input under test. The schema is written out here rather
than reused from `tests/chatdb_fixture.py` because the columns that
decide inclusion — `associated_message_type`, `item_type`, and the
presence or absence of a `chat_message_join` row — are exactly the ones
that shared builder does not model.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from imsg.cli import app
from imsg.verify.seed import build_seed_snapshot_from_chat_db

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)

_SCHEMA = """
CREATE TABLE message (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT UNIQUE,
    date INTEGER,
    text TEXT,
    attributedBody BLOB,
    item_type INTEGER NOT NULL DEFAULT 0,
    associated_message_type INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
"""


def _apple_ns(dt: datetime) -> int:
    return int((dt.astimezone(UTC) - APPLE_EPOCH).total_seconds() * 1_000_000_000)


def _build_fixture(path: Path) -> None:
    """Seven rows: three the extractor keeps, four it routes elsewhere or
    cannot file."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        rows: list[tuple[int, str, datetime | None, str | None, bytes | None, int, int, bool]] = [
            # (rowid, guid, date, text, attributedBody, item_type, assoc_type, joined)
            (1, "keep-2023", datetime(2023, 6, 1, tzinfo=UTC), "hello", None, 0, 0, True),
            # No text and no attributedBody — the body-decode-null proxy.
            (2, "keep-2024-nobody", datetime(2024, 3, 2, tzinfo=UTC), None, None, 0, 0, True),
            # A reaction: goes to `tapback`, never to `message`.
            (3, "drop-tapback", datetime(2024, 3, 3, tzinfo=UTC), "Liked", None, 0, 2000, True),
            # A sticker: the extractor treats 1000 as a tapback too.
            (4, "drop-sticker", datetime(2024, 3, 4, tzinfo=UTC), None, b"\x01", 0, 1000, True),
            # A system row (someone named the group).
            (5, "drop-system", datetime(2024, 3, 5, tzinfo=UTC), None, None, 2, 0, True),
            # No chat_message_join row at all. Filed since D13 (2026-09-23),
            # so the reference must count it.
            (6, "keep-nochat", datetime(2024, 3, 6, tzinfo=UTC), "orphan", None, 0, 0, False),
            # No chat link and no date: `sent_at` is NOT NULL, so it cannot be filed.
            (7, "drop-nochat-nodate", None, "undated", None, 0, 0, False),
        ]
        for rowid, guid, when, text, body, item_type, assoc, joined in rows:
            conn.execute(
                "INSERT INTO message (ROWID, guid, date, text, attributedBody, item_type, "
                "associated_message_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rowid, guid, None if when is None else _apple_ns(when), text, body, item_type, assoc),
            )
            if joined:
                conn.execute(
                    "INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, ?)", (rowid,)
                )
        # One attachment, on a kept message; one on a dropped one, which must
        # not be counted.
        conn.execute("INSERT INTO message_attachment_join VALUES (1, 100)")
        conn.execute("INSERT INTO message_attachment_join VALUES (5, 101)")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def chat_db(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.db"
    _build_fixture(path)
    return path


def test_guid_set_is_only_what_the_extractor_lands_in_message(chat_db: Path) -> None:
    snap = build_seed_snapshot_from_chat_db(chat_db, source_label="corpus")
    assert snap.guids == {"keep-2023", "keep-2024-nobody", "keep-nochat"}


def test_stickers_are_excluded_like_reactions(chat_db: Path) -> None:
    """assoc type 1000 is the easy one to miss — including it would make
    AT-2 report every sticker as a missing message."""
    snap = build_seed_snapshot_from_chat_db(chat_db, source_label="corpus")
    assert "drop-sticker" not in snap.guids
    assert "drop-tapback" not in snap.guids


def test_system_rows_and_undated_chatless_rows_are_excluded(chat_db: Path) -> None:
    snap = build_seed_snapshot_from_chat_db(chat_db, source_label="corpus")
    assert "drop-system" not in snap.guids
    assert "drop-nochat-nodate" not in snap.guids


def test_chatless_rows_count_because_extraction_files_them(chat_db: Path) -> None:
    """D13 (2026-09-23): a row with no `chat_message_join` is filed into a
    chat by its evidence, or a holding chat, instead of being dropped. A
    reference that still left it out would report every such message the
    index now holds as an extra, and could never catch one that went
    missing."""
    snap = build_seed_snapshot_from_chat_db(chat_db, source_label="corpus")
    assert "keep-nochat" in snap.guids


def test_diagnostics(chat_db: Path) -> None:
    snap = build_seed_snapshot_from_chat_db(chat_db, source_label="corpus")
    assert snap.source_label == "corpus"
    assert snap.per_year_counts == {"2023": 1, "2024": 2}
    assert snap.min_sent_at is not None and snap.min_sent_at.startswith("2023-06-01T00:00:00")
    assert snap.max_sent_at is not None and snap.max_sent_at.startswith("2024-03-06T00:00:00")
    # Only the kept row with neither text nor attributedBody.
    assert snap.body_decode_null_count == 1
    # The attachment on the excluded system row must not be counted.
    assert snap.attachment_join_count == 1


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_seed_snapshot_from_chat_db(tmp_path / "nope.db", source_label="corpus")


def test_non_chat_db_sqlite_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="no 'message' table"):
        build_seed_snapshot_from_chat_db(path, source_label="corpus")


def test_open_is_read_only_and_leaves_no_sidecar_files(chat_db: Path) -> None:
    """CLAUDE.md non-negotiable 1: nothing this system does may write
    next to a chat.db — including the -wal/-shm files SQLite will
    happily create during a plain read."""
    before = sorted(p.name for p in chat_db.parent.iterdir())
    build_seed_snapshot_from_chat_db(chat_db, source_label="corpus")
    assert sorted(p.name for p in chat_db.parent.iterdir()) == before


runner = CliRunner()


def test_cli_exposes_reference_db() -> None:
    result = runner.invoke(app, ["verify-seed", "--help"])
    assert result.exit_code == 0
    assert "--reference-db" in result.output


def test_cli_rejects_reference_and_reference_db_together() -> None:
    result = runner.invoke(
        app, ["verify-seed", "--reference", "/tmp/x.json", "--reference-db", "/tmp/y.db"]
    )
    assert result.exit_code == 2
    assert "exactly one of --export, --reference or --reference-db" in result.output
