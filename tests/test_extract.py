"""Unit tests (link-preview parsing) + a live-Postgres integration
suite for S2 extraction (SPEC §8 S2). The integration suite follows
`test_migrations_integration.py`'s pattern: skip cleanly with no
reachable scratch Postgres, apply the real migrations, then exercise
`run_extract` end to end against a synthetic `chat.db`-shaped snapshot
built with `chatdb_fixture.ChatDbBuilder`. `imsg-dump` itself is never
invoked — `run_imsg_dump_fn` is injected with a fake returning
hand-built `ImsgDumpMessage`s, so these tests exercise the real SQL
merge/upsert logic without requiring a compiled Rust binary.
"""

from __future__ import annotations

import os
import plistlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from chatdb_fixture import (
    ChatDbBuilder,
    FixtureAttachment,
    FixtureChat,
    FixtureHandle,
    FixtureMessage,
)
from imsg.db.migrations import PostgresMigrationRunner
from imsg.stages.extract import (
    ExtractResult,
    parse_link_preview,
    run_extract,
)
from imsg.stages.imsg_dump import EditVersion, ImsgDumpMessage, ImsgDumpRun, TapbackInfo

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_extract_test"

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
def pg_conn() -> Iterator[psycopg.Connection]:
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
    runner = PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR)
    runner.apply_pending()
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


def _dump_message(
    *,
    guid: str,
    rowid: int,
    body_text: str | None = "hello",
    edit_history: tuple[EditVersion, ...] = (),
    tapback: TapbackInfo | None = None,
    reply_to_guid: str | None = None,
    is_unsent: bool = False,
) -> ImsgDumpMessage:
    return ImsgDumpMessage(
        rowid=rowid,
        guid=guid,
        chat_guid=None,
        handle=None,
        is_from_me=False,
        date=None,
        date_edited=None,
        date_retracted=None,
        service="iMessage",
        body_text=body_text,
        edit_history=edit_history,
        is_unsent=is_unsent,
        tapback=tapback,
        attachment_rowids=(),
        reply_to_guid=reply_to_guid,
    )


def _fake_binary(tmp_path: Path) -> Path:
    p = tmp_path / "imsg-dump"
    p.write_text("")
    return p


def test_run_extract_basic_dm_message(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1", style=45))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(_dump_message(guid="msg-1", rowid=1),), stderr_lines=())

    result = run_extract(
        conn=pg_conn,
        source_name="mini",
        snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path),
        run_imsg_dump_fn=fake_run,
    )

    assert isinstance(result, ExtractResult)
    assert result.messages_upserted == 1
    assert result.chats_upserted == 1
    assert result.handles_upserted == 1
    assert result.watermark_before == 0
    assert result.watermark_after == 1
    assert result.bodies_missing == 0

    with pg_conn.cursor() as cur:
        cur.execute("SELECT kind, display_name FROM chat WHERE source_guid = 'chat-1'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "dm"

        cur.execute(
            "SELECT text_original, is_from_me, is_unsent, is_edited, sender_person_id, has_attachments "
            "FROM message WHERE source_guid = 'msg-1'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "hello"
        assert row[1] is False
        assert row[2] is False
        assert row[3] is False
        assert row[4] is None  # S3 has not run — hard requirement 3
        assert row[5] is False

        cur.execute("SELECT raw_value FROM source_handle")
        assert cur.fetchall() == [("+15551234567",)]

        cur.execute("SELECT value FROM sync_state WHERE key = 'watermark.rowid.mini'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "1"


def test_run_extract_dry_run_writes_nothing(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1", style=45))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(_dump_message(guid="msg-1", rowid=1),), stderr_lines=())

    result = run_extract(
        conn=pg_conn,
        source_name="mini",
        snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path),
        run_imsg_dump_fn=fake_run,
        dry_run=True,
    )

    # The preview reports exactly what a real run would have done...
    assert result.dry_run is True
    assert result.messages_upserted == 1
    assert result.chats_upserted == 1
    assert result.watermark_before == 0
    assert result.watermark_after == 1

    # ...but nothing was actually committed to Postgres.
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM chat")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM extraction_run")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM sync_state WHERE key = 'watermark.rowid.mini'")
        assert cur.fetchone() == (0,)

    # A real run afterward is unaffected by the rolled-back preview.
    real_result = run_extract(
        conn=pg_conn,
        source_name="mini",
        snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path),
        run_imsg_dump_fn=fake_run,
    )
    assert real_result.dry_run is False
    assert real_result.messages_upserted == 1
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message")
        assert cur.fetchone() == (1,)


def test_run_extract_is_idempotent(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(_dump_message(guid="msg-1", rowid=1),), stderr_lines=())

    binary = _fake_binary(tmp_path)
    first = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run,
    )
    assert first.messages_upserted == 1

    # Re-run against the *same* snapshot: watermark already at 1, so
    # nothing new is in scope.
    second = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run,
    )
    assert second.messages_upserted == 0
    assert second.watermark_before == 1
    assert second.watermark_after == 1

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM chat")
        assert cur.fetchone() == (1,)


def test_run_extract_incremental_only_processes_new_rows(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    snapshot_path = builder.build(tmp_path / "snapshot1.db")
    binary = _fake_binary(tmp_path)

    def fake_run_1(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        assert since_rowid == 0
        return ImsgDumpRun(messages=(_dump_message(guid="msg-1", rowid=1),), stderr_lines=())

    first = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run_1,
    )
    assert first.watermark_after == 1

    builder2 = ChatDbBuilder()
    chat2 = builder2.add_chat(FixtureChat(guid="chat-1"))
    handle2 = builder2.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder2.link_participant(chat2.guid, handle2.raw_value)
    builder2.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat2.guid, handle_raw_value=handle2.raw_value, rowid=1)
    )
    builder2.add_message(
        FixtureMessage(guid="msg-2", chat_guid=chat2.guid, handle_raw_value=handle2.raw_value, rowid=2)
    )
    snapshot_path_2 = builder2.build(tmp_path / "snapshot2.db")

    calls: list[int] = []

    def fake_run_2(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        calls.append(since_rowid)
        return ImsgDumpRun(messages=(_dump_message(guid="msg-2", rowid=2),), stderr_lines=())

    second = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path_2,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run_2,
    )
    assert calls == [1]  # since_rowid == prior watermark, not 0
    assert second.messages_upserted == 1
    assert second.watermark_before == 1
    assert second.watermark_after == 2

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message")
        assert cur.fetchone() == (2,)


def test_run_extract_edited_message_stores_history_and_latest_text(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(
            guid="msg-1",
            chat_guid=chat.guid,
            handle_raw_value=handle.raw_value,
            date=datetime(2024, 1, 1, tzinfo=UTC),
            date_edited=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
        )
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(
            messages=(
                _dump_message(
                    guid="msg-1",
                    rowid=1,
                    body_text="final version",
                    edit_history=(EditVersion(text="first draft", edited_at="2024-01-01T00:30:00+00:00"),),
                ),
            ),
            stderr_lines=(),
        )

    run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )

    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original, is_edited FROM message WHERE source_guid = 'msg-1'")
        row = cur.fetchone()
        assert row == ("final version", True)

        cur.execute(
            "SELECT version_idx, text FROM message_version mv "
            "JOIN message m ON m.message_id = mv.message_id WHERE m.source_guid = 'msg-1'"
        )
        assert cur.fetchall() == [(0, "first draft")]


def test_run_extract_unsent_message(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(
            guid="msg-1",
            chat_guid=chat.guid,
            handle_raw_value=handle.raw_value,
            date_retracted=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
        )
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        # is_unsent=True here is what makes this authoritative: per the
        # real imsg-dump crate exploration, `is_unsent` comes from
        # typedstream-derived edit status, not the SQL date_retracted
        # column (see extract.py's module docstring "Correction" note).
        return ImsgDumpRun(
            messages=(_dump_message(guid="msg-1", rowid=1, is_unsent=True),), stderr_lines=()
        )

    run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )

    with pg_conn.cursor() as cur:
        cur.execute("SELECT is_unsent FROM message WHERE source_guid = 'msg-1'")
        assert cur.fetchone() == (True,)


def test_run_extract_falls_back_to_sql_unsent_flag_when_no_dump_record(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """When `imsg-dump` never emitted a line for this guid at all (the
    `bodies_missing` case), `is_unsent`/`is_edited` fall back to the SQL
    `date_retracted`/`date_edited` columns rather than defaulting to
    `False` — better a possibly-stale signal than a silently wrong one."""
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(
            guid="msg-1",
            chat_guid=chat.guid,
            handle_raw_value=handle.raw_value,
            date_retracted=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
        )
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(), stderr_lines=())

    result = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )
    assert result.bodies_missing == 1

    with pg_conn.cursor() as cur:
        cur.execute("SELECT is_unsent, text_original FROM message WHERE source_guid = 'msg-1'")
        assert cur.fetchone() == (True, None)


def test_run_extract_tapback_folded_not_a_message(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value, rowid=1)
    )
    builder.add_message(
        FixtureMessage(guid="msg-2-tapback", chat_guid=chat.guid, handle_raw_value=handle.raw_value, rowid=2)
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(
            messages=(
                _dump_message(guid="msg-1", rowid=1),
                _dump_message(
                    guid="msg-2-tapback",
                    rowid=2,
                    body_text=None,
                    tapback=TapbackInfo(kind="loved", target_guid="msg-1"),
                ),
            ),
            stderr_lines=(),
        )

    result = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )
    assert result.messages_upserted == 1
    assert result.tapbacks_upserted == 1

    with pg_conn.cursor() as cur:
        cur.execute("SELECT source_guid FROM message")
        assert cur.fetchall() == [("msg-1",)]

        cur.execute(
            "SELECT kind, target_source_guid, target_message_id IS NOT NULL FROM tapback "
            "WHERE source_guid = 'msg-2-tapback'"
        )
        row = cur.fetchone()
        assert row == ("loved", "msg-1", True)


def test_run_extract_tapback_emoji_kind_and_removed_action(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Per the real `imsg-dump` output (SPEC §7.2's `tapback.kind` comment
    documents "emoji:<char>", but the shim emits a bare "emoji" kind plus a
    separate `emoji` field — combine them). `removed` must come from the
    shim's `action` field, not from the message's own unsent state (a
    tapback removal is a distinct chat.db event, unrelated to the
    *message* being retracted)."""
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value, rowid=1)
    )
    builder.add_message(
        FixtureMessage(
            guid="msg-2-tapback",
            chat_guid=chat.guid,
            handle_raw_value=handle.raw_value,
            rowid=2,
            # The message itself is NOT unsent — only the tapback is removed.
            date_retracted=None,
        )
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(
            messages=(
                _dump_message(guid="msg-1", rowid=1),
                _dump_message(
                    guid="msg-2-tapback",
                    rowid=2,
                    body_text=None,
                    tapback=TapbackInfo(
                        kind="emoji", target_guid="msg-1", emoji="🔥", action="removed"
                    ),
                ),
            ),
            stderr_lines=(),
        )

    run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT kind, removed FROM tapback WHERE source_guid = 'msg-2-tapback'"
        )
        assert cur.fetchone() == ("emoji:🔥", True)


def test_run_extract_system_message_skipped(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(
            guid="msg-system-1",
            chat_guid=chat.guid,
            handle_raw_value=handle.raw_value,
            item_type=1,  # e.g. "member added" per widely-documented chat.db semantics
        )
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(), stderr_lines=())

    result = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )
    assert result.system_messages_skipped == 1
    assert result.messages_upserted == 0

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message")
        assert cur.fetchone() == (0,)


def test_run_extract_attachment_linked_to_message(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    builder.add_attachment(FixtureAttachment(guid="att-1"))
    builder.link_attachment("msg-1", "att-1")
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(_dump_message(guid="msg-1", rowid=1),), stderr_lines=())

    result = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )
    assert result.attachments_upserted == 1

    with pg_conn.cursor() as cur:
        cur.execute("SELECT has_attachments FROM message WHERE source_guid = 'msg-1'")
        assert cur.fetchone() == (True,)

        cur.execute(
            "SELECT a.source_guid, ma.ordinal FROM message_attachment ma "
            "JOIN attachment a ON a.attachment_id = ma.attachment_id "
            "JOIN message m ON m.message_id = ma.message_id WHERE m.source_guid = 'msg-1'"
        )
        assert cur.fetchall() == [("att-1", 0)]


def test_run_extract_group_chat_classified_by_style(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1", style=43, display_name="Deck project"))
    alice = builder.add_handle(FixtureHandle(raw_value="+15551111111"))
    bob = builder.add_handle(FixtureHandle(raw_value="+15552222222"))
    builder.link_participant(chat.guid, alice.raw_value)
    builder.link_participant(chat.guid, bob.raw_value)
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(), stderr_lines=())

    run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )

    with pg_conn.cursor() as cur:
        cur.execute("SELECT kind, display_name FROM chat WHERE source_guid = 'chat-1'")
        assert cur.fetchone() == ("group", "Deck project")


def test_run_extract_body_missing_from_dump_is_counted(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=(), stderr_lines=())  # imsg-dump never emitted this guid

    result = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=fake_run,
    )
    assert result.bodies_missing == 1
    assert result.messages_upserted == 1

    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = 'msg-1'")
        assert cur.fetchone() == (None,)


def test_run_extract_rescans_old_row_edited_after_last_run(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value, rowid=1)
    )
    builder.add_message(
        FixtureMessage(guid="msg-2", chat_guid=chat.guid, handle_raw_value=handle.raw_value, rowid=2)
    )
    snapshot_path_1 = builder.build(tmp_path / "snapshot1.db")
    binary = _fake_binary(tmp_path)

    def fake_run_1(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(
            messages=(
                _dump_message(guid="msg-1", rowid=1, body_text="original text"),
                _dump_message(guid="msg-2", rowid=2),
            ),
            stderr_lines=(),
        )

    first = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path_1,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run_1,
    )
    assert first.watermark_after == 2

    # Now msg-1 (rowid 1, below the watermark of 2) gets edited.
    builder2 = ChatDbBuilder()
    chat2 = builder2.add_chat(FixtureChat(guid="chat-1"))
    handle2 = builder2.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder2.link_participant(chat2.guid, handle2.raw_value)
    builder2.add_message(
        FixtureMessage(
            guid="msg-1", chat_guid=chat2.guid, handle_raw_value=handle2.raw_value, rowid=1,
            date_edited=datetime.now(UTC),
        )
    )
    builder2.add_message(
        FixtureMessage(guid="msg-2", chat_guid=chat2.guid, handle_raw_value=handle2.raw_value, rowid=2)
    )
    snapshot_path_2 = builder2.build(tmp_path / "snapshot2.db")

    dump_since: list[int] = []

    def fake_run_2(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        dump_since.append(since_rowid)
        return ImsgDumpRun(
            messages=(_dump_message(guid="msg-1", rowid=1, body_text="edited text"),),
            stderr_lines=(),
        )

    second = run_extract(
        conn=pg_conn, source_name="mini", snapshot_path=snapshot_path_2,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run_2,
    )
    # since_rowid had to drop below the watermark to cover the dirty old row.
    assert dump_since == [0]
    assert second.messages_upserted == 1

    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original, is_edited FROM message WHERE source_guid = 'msg-1'")
        assert cur.fetchone() == ("edited text", True)


def test_run_extract_failed_run_is_recorded(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def boom(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        raise RuntimeError("simulated imsg-dump crash")

    from imsg.errors import ExtractionError

    with pytest.raises(ExtractionError):
        run_extract(
            conn=pg_conn, source_name="mini", snapshot_path=snapshot_path,
            imsg_dump_binary=_fake_binary(tmp_path), run_imsg_dump_fn=boom,
        )

    with pg_conn.cursor() as cur:
        cur.execute("SELECT status FROM extraction_run WHERE source_name = 'mini'")
        assert cur.fetchall() == [("failed",)]
        # Watermark must not have advanced on a failed run.
        cur.execute("SELECT count(*) FROM sync_state WHERE key = 'watermark.rowid.mini'")
        assert cur.fetchone() == (0,)


# --------------------------------------------------------------------------
# link preview parsing (pure function, no Postgres needed — but this
# module is skipif-gated as a whole per SPEC's "unit tests must pass
# with no DB", so a dedicated no-DB module covers this too; kept here
# as well since it is directly relevant to S2's payload_data handling)
# --------------------------------------------------------------------------


def _nskeyedarchiver_blob(fields: dict[str, str]) -> bytes:
    """Build a minimal NSKeyedArchiver-format binary plist: `$objects[0]`
    is the conventional `$null` sentinel, `$objects[1]` is the root dict
    (`$top.root` -> UID(1)), and each `fields` entry becomes its own
    string object referenced by UID from the root dict — structurally
    analogous to a real `payload_data` link-preview archive."""
    root: dict[str, object] = {}
    objects: list[object] = ["$null", root, *fields.values()]
    for i, key in enumerate(fields, start=2):
        root[key] = plistlib.UID(i)
    data = {
        "$archiver": "NSKeyedArchiver",
        "$version": 100000,
        "$top": {"root": plistlib.UID(1)},
        "$objects": objects,
    }
    return plistlib.dumps(data, fmt=plistlib.FMT_BINARY)


def test_parse_link_preview_extracts_url_and_title() -> None:
    blob = _nskeyedarchiver_blob({"URL": "https://example.com/article", "title": "Example Article"})
    preview = parse_link_preview(blob)
    assert preview is not None
    assert preview.url == "https://example.com/article"
    assert preview.title == "Example Article"


def test_parse_link_preview_returns_none_for_garbage() -> None:
    assert parse_link_preview(b"not a plist at all") is None
    assert parse_link_preview(None) is None
    assert parse_link_preview(b"") is None
