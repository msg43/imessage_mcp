"""Regression suite for S2 state-idempotence (2026-09-17).

Extraction has always been *identity*-idempotent: a second run over the
same corpus produces no duplicate rows, because every upsert keys on a
chat.db GUID. It was not *state*-idempotent. Every `ON CONFLICT DO
UPDATE` fired unconditionally, so re-extracting an unchanged corpus
rewrote each row with its own values, and migration 0003's trigger --
correctly -- moved `updated_at` on every one of those writes.

S4's `find_dirty_chats` keys on exactly that column, so an unchanged
re-extraction proposed discarding and rebuilding the segmentation and
embeddings for effectively the whole corpus. Measured on the target
host the day this suite was written: 662,683 of 673,113 message rows
bumped for 972 genuinely new messages, dragging 8,331 chats' incremental
frontier back to their first message.

The fix is a `WHERE <target>.col IS DISTINCT FROM excluded.col OR ...`
guard on each upsert, so an unchanged row is never written at all and
the trigger never fires. These tests are the proof, and they are written
to be able to FAIL: the corpus is segmented and its identities resolved
before the baseline is taken, so `find_dirty_chats` has real segments to
compare against rather than trivially returning nothing.

Fictional personas only (D5): Alice Example / Bob Builder, and
documentation-range phone numbers.
"""

from __future__ import annotations

import os
import plistlib
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import (
    REBUILD_ALL_SENTINEL,
    find_dirty_chats,
    run_segment_for_chat,
)
from imsg.stages.extract import ExtractResult, run_extract
from imsg.stages.identity import rename_person, run_identity
from imsg.stages.imsg_dump import EditVersion, ImsgDumpMessage, ImsgDumpRun, TapbackInfo

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_extract_noop_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
PROMPT_BYTES = b"fixed boundary-detection prompt for tests"

SOURCE_NAME = "mini"


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
    """Autocommit, as production's `connect()` is. It matters here:
    `now()` is frozen at transaction start, and these tests compare
    `message.updated_at` (written by extraction/identity) against
    `segment.created_at` (written by segmentation). Sharing one long
    transaction would freeze both to the same instant and make the
    comparison meaningless."""
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


# --- the fixture corpus ----------------------------------------------------

_BASE = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)

ALICE_HANDLE = "+15550000001"
BOB_HANDLE = "+15550000002"

# (guid, chat_guid, rowid, handle, minutes_offset)
_ALICE_1 = "msg-alice-1"
_ALICE_2 = "msg-alice-2"
_BOB_1 = "msg-bob-1"
_BOB_2 = "msg-bob-2"
_OWNER_1 = "msg-owner-1"
_TAPBACK_1 = "msg-tapback-1"


def _nskeyedarchiver_blob(fields: dict[str, str]) -> bytes:
    """Minimal NSKeyedArchiver-shaped binary plist (same construction as
    `test_extract.py`'s), so the corpus exercises the `link_preview`
    upsert too."""
    root: dict[str, object] = {}
    objects: list[object] = ["$null", root, *fields.values()]
    for i, key in enumerate(fields, start=2):
        root[key] = plistlib.UID(i)
    return plistlib.dumps(
        {
            "$archiver": "NSKeyedArchiver",
            "$version": 100000,
            "$top": {"root": plistlib.UID(1)},
            "$objects": objects,
        },
        fmt=plistlib.FMT_BINARY,
    )


_LINK_BLOB = _nskeyedarchiver_blob(
    {"URL": "https://example.com/article", "title": "Example Article"}
)


def _build_snapshot(path: Path, *, extra_message: bool = False) -> Path:
    """Two DM chats, five real messages (one of them the owner's), one
    tapback, one attachment, one link preview and one edited message --
    enough to drive every upsert in `imsg.stages.extract` at least once.

    Every ROWID is explicit so a rebuilt snapshot is byte-for-byte
    equivalent in the columns extraction reads: an accidental ROWID
    shuffle would make "the same corpus" a different corpus and quietly
    defeat the whole suite.
    """
    builder = ChatDbBuilder()
    alice_chat = builder.add_chat(FixtureChat(guid="chat-alice", rowid=1, display_name=None))
    bob_chat = builder.add_chat(FixtureChat(guid="chat-bob", rowid=2, display_name=None))
    builder.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    builder.add_handle(FixtureHandle(raw_value=BOB_HANDLE, rowid=2))
    builder.link_participant(alice_chat.guid, ALICE_HANDLE)
    builder.link_participant(bob_chat.guid, BOB_HANDLE)

    builder.add_message(
        FixtureMessage(
            guid=_ALICE_1, chat_guid=alice_chat.guid, handle_raw_value=ALICE_HANDLE,
            rowid=1, date=_BASE,
        )
    )
    builder.add_message(
        FixtureMessage(
            guid=_ALICE_2, chat_guid=alice_chat.guid, handle_raw_value=ALICE_HANDLE,
            rowid=2, date=_BASE + timedelta(minutes=1),
        )
    )
    # The owner's own message: `sender_source_handle_id` is NULL for these,
    # which is the NULL side of the sender comparison in the upsert guard.
    builder.add_message(
        FixtureMessage(
            guid=_OWNER_1, chat_guid=alice_chat.guid, is_from_me=True,
            rowid=3, date=_BASE + timedelta(minutes=2),
        )
    )
    builder.add_message(
        FixtureMessage(
            guid=_BOB_1, chat_guid=bob_chat.guid, handle_raw_value=BOB_HANDLE,
            rowid=4, date=_BASE + timedelta(minutes=3), payload_data=_LINK_BLOB,
        )
    )
    builder.add_message(
        FixtureMessage(
            guid=_BOB_2, chat_guid=bob_chat.guid, handle_raw_value=BOB_HANDLE,
            rowid=5, date=_BASE + timedelta(minutes=4),
            date_edited=_BASE + timedelta(minutes=5),
        )
    )
    builder.add_message(
        FixtureMessage(
            guid=_TAPBACK_1, chat_guid=alice_chat.guid, handle_raw_value=BOB_HANDLE,
            rowid=6, date=_BASE + timedelta(minutes=6),
        )
    )
    builder.add_attachment(FixtureAttachment(guid="att-1", rowid=1))
    builder.link_attachment(_ALICE_2, "att-1")

    if extra_message:
        builder.add_message(
            FixtureMessage(
                guid="msg-bob-3", chat_guid=bob_chat.guid, handle_raw_value=BOB_HANDLE,
                rowid=7, date=_BASE + timedelta(minutes=7),
            )
        )
    return builder.build(path)


def _dump_message(
    *,
    guid: str,
    rowid: int,
    body_text: str | None,
    edit_history: tuple[EditVersion, ...] = (),
    tapback: TapbackInfo | None = None,
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
        is_unsent=False,
        tapback=tapback,
        attachment_rowids=(),
        reply_to_guid=None,
    )


def _dump_run(
    *, alice_1_body: str = "morning", extra_message: bool = False
) -> ImsgDumpRun:
    messages = [
        _dump_message(guid=_ALICE_1, rowid=1, body_text=alice_1_body),
        _dump_message(guid=_ALICE_2, rowid=2, body_text="here is a photo"),
        _dump_message(guid=_OWNER_1, rowid=3, body_text="thanks"),
        _dump_message(guid=_BOB_1, rowid=4, body_text="see https://example.com/article"),
        _dump_message(
            guid=_BOB_2, rowid=5, body_text="corrected text",
            edit_history=(EditVersion(text="original text", edited_at=None),),
        ),
        _dump_message(
            guid=_TAPBACK_1, rowid=6, body_text=None,
            tapback=TapbackInfo(target_guid=_ALICE_1, kind="loved", emoji=None, action="added"),
        ),
    ]
    if extra_message:
        messages.append(_dump_message(guid="msg-bob-3", rowid=7, body_text="one more thing"))
    return ImsgDumpRun(messages=tuple(messages), stderr_lines=())


def _fake_binary(tmp_path: Path) -> Path:
    p = tmp_path / "imsg-dump"
    p.write_text("")
    return p


def _extract(
    conn: psycopg.Connection,
    tmp_path: Path,
    snapshot_path: Path,
    dump: ImsgDumpRun,
) -> ExtractResult:
    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return dump

    return run_extract(
        conn=conn,
        source_name=SOURCE_NAME,
        snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path),
        run_imsg_dump_fn=fake_run,
    )


def _reset_watermark(conn: psycopg.Connection) -> None:
    """Put the source's rowid watermark back to zero, which is what makes
    the next run a full re-extraction rather than an incremental one.
    This is the shape of the run that exposed the defect: after a seed
    merge the watermark no longer covered the corpus, so every message
    came back into scope even though none of them had changed."""
    conn.execute("DELETE FROM sync_state WHERE key = %s", (f"watermark.rowid.{SOURCE_NAME}",))


# --- row-version bookkeeping ----------------------------------------------

# Every table `imsg.stages.extract` writes, with the columns that identify
# a row. `xmin` (the inserting transaction id of the live tuple) changes on
# ANY update, including one that writes a column its own value, which makes
# it a stronger probe than `updated_at` -- and the only probe available for
# the tables that carry no `updated_at` column at all.
_TRACKED_TABLES: dict[str, tuple[str, ...]] = {
    "chat": ("source_guid",),
    "source_handle": ("raw_value", "service"),
    "chat_participant_source": ("chat_id", "source_handle_id"),
    "message": ("source_guid",),
    "message_source": ("source_name", "source_rowid"),
    "message_version": ("message_id", "version_idx"),
    "message_attachment": ("message_id", "attachment_id"),
    "tapback": ("source_guid",),
    "link_preview": ("message_id", "url"),
    "attachment": ("source_guid",),
    "attachment_source": ("source_name", "source_rowid"),
}


def _row_versions(conn: psycopg.Connection) -> dict[tuple[str, tuple[object, ...]], str]:
    """`{(table, identifying-values): xmin}` for every row extraction owns."""
    versions: dict[tuple[str, tuple[object, ...]], str] = {}
    with conn.cursor() as cur:
        for table, key_columns in _TRACKED_TABLES.items():
            keys = ", ".join(key_columns)
            cur.execute(f"SELECT {keys}, xmin::text FROM {table}")
            for row in cur.fetchall():
                versions[(table, tuple(row[:-1]))] = str(row[-1])
    return versions


def _changed_rows(
    before: dict[tuple[str, tuple[object, ...]], str],
    after: dict[tuple[str, tuple[object, ...]], str],
) -> set[tuple[str, tuple[object, ...]]]:
    """Rows present in both snapshots whose tuple was rewritten."""
    return {key for key, version in before.items() if after.get(key, version) != version}


def _message_updated_at(conn: psycopg.Connection) -> dict[str, datetime]:
    with conn.cursor() as cur:
        cur.execute("SELECT source_guid, updated_at FROM message")
        return {str(guid): stamp for guid, stamp in cur.fetchall()}


def _chat_ids_by_guid(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT source_guid, chat_id FROM chat")
        return {str(guid): int(chat_id) for guid, chat_id in cur.fetchall()}


# --- the settled, fully-segmented baseline --------------------------------


def _fake_contacts_importer(default_region: str) -> list[object]:
    """No Contacts access in tests: every handle becomes an auto-created
    stub person, which is all S4 needs to render."""
    return []


def _settle(conn: psycopg.Connection, config: Config) -> None:
    """Resolve identities and segment every chat, leaving the corpus with
    no segmentation work pending. Without this the dirty-chat assertions
    below could not fail for the right reason: `find_dirty_chats` would
    return nothing because there is nothing segmented to compare against,
    not because nothing changed."""
    run_identity(conn=conn, config=config, contacts_importer=_fake_contacts_importer)  # type: ignore[arg-type]
    for chat_id in _chat_ids_by_guid(conn).values():
        run_segment_for_chat(
            conn,
            chat_id,
            config,
            FakeBoundaryProvider(),
            PROMPT_BYTES,
            earliest_changed_at=REBUILD_ALL_SENTINEL,
        )
    assert find_dirty_chats(conn, index_unsent=config.policy.index_unsent) == {}, (
        "the baseline must start settled, or every assertion below is vacuous"
    )


@pytest.fixture
def settled_corpus(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> Path:
    """Extract the corpus once, resolve identities, segment everything.
    Returns the snapshot path, which the caller re-extracts."""
    snapshot_path = _build_snapshot(tmp_path / "snapshot.db")
    result = _extract(pg_conn, tmp_path, snapshot_path, _dump_run())
    assert result.messages_upserted == 5, "5 real messages; the tapback is folded, not a message"
    assert result.tapbacks_upserted == 1
    assert result.attachments_upserted == 1
    assert result.link_previews_upserted == 1
    _settle(pg_conn, config)
    return snapshot_path


# --- tests ----------------------------------------------------------------


def test_reextracting_an_unchanged_corpus_rewrites_no_row(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, settled_corpus: Path
) -> None:
    """The defect, stated as an assertion. Before the fix every message,
    attachment, chat, handle, tapback, link preview and edit version was
    rewritten, `updated_at` moved on all of them, and every chat came
    back dirty."""
    before_versions = _row_versions(pg_conn)
    before_updated_at = _message_updated_at(pg_conn)
    assert before_updated_at, "no messages captured — the probe would pass vacuously"

    _reset_watermark(pg_conn)
    result = _extract(pg_conn, tmp_path, settled_corpus, _dump_run())

    # The whole corpus really was back in scope: this is a full
    # re-extraction, not a run that quietly did nothing.
    assert result.watermark_before == 0
    assert result.messages_upserted == 5

    after_updated_at = _message_updated_at(pg_conn)
    moved = {
        guid: (before_updated_at[guid], stamp)
        for guid, stamp in after_updated_at.items()
        if before_updated_at[guid] != stamp
    }
    assert moved == {}, f"updated_at moved on unchanged messages: {moved}"

    rewritten = _changed_rows(before_versions, _row_versions(pg_conn))
    # `message_source` is the one deliberate exception: it records the
    # extraction run that last observed each row, so a new run id is a real
    # change every time. It carries no `updated_at` and nothing downstream
    # watches it, so it drags no re-segmentation with it.
    assert {table for table, _ in rewritten} <= {"message_source"}, (
        f"rows rewritten by an unchanged re-extraction: {sorted(rewritten)}"
    )

    assert find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent) == {}


def test_reextraction_reports_every_row_as_unchanged(
    pg_conn: psycopg.Connection, tmp_path: Path, settled_corpus: Path
) -> None:
    """The operator-facing half: a re-extraction that changed nothing has
    to *say* it changed nothing. A bare `messages_upserted=5` cannot tell
    a no-op apart from five rewritten rows."""
    _reset_watermark(pg_conn)
    result = _extract(pg_conn, tmp_path, settled_corpus, _dump_run())

    assert result.message_upserts.inserted == 0
    assert result.message_upserts.updated == 0
    assert result.message_upserts.unchanged == 5
    assert result.message_upserts.total == result.messages_upserted

    assert result.attachment_upserts.unchanged == 1
    assert result.attachment_upserts.updated == 0
    assert result.chat_upserts.unchanged == 2
    assert result.chat_upserts.updated == 0
    assert result.handle_upserts.unchanged == 2
    assert result.handle_upserts.updated == 0


def test_first_extraction_reports_every_row_as_inserted(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    snapshot_path = _build_snapshot(tmp_path / "snapshot.db")
    result = _extract(pg_conn, tmp_path, snapshot_path, _dump_run())

    assert result.message_upserts.inserted == 5
    assert result.message_upserts.updated == 0
    assert result.message_upserts.unchanged == 0
    assert result.chat_upserts.inserted == 2
    assert result.handle_upserts.inserted == 2
    assert result.attachment_upserts.inserted == 1


def test_one_changed_body_moves_one_row_and_dirties_one_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, settled_corpus: Path
) -> None:
    """The other half of the invariant: skipping unchanged rows must not
    skip changed ones. An edited body still moves exactly its own row and
    dirties exactly its own chat."""
    before_versions = _row_versions(pg_conn)
    before_updated_at = _message_updated_at(pg_conn)
    chat_ids = _chat_ids_by_guid(pg_conn)

    _reset_watermark(pg_conn)
    result = _extract(
        pg_conn, tmp_path, settled_corpus, _dump_run(alice_1_body="morning, corrected")
    )

    assert result.message_upserts.updated == 1
    assert result.message_upserts.unchanged == 4
    assert result.message_upserts.inserted == 0

    after_updated_at = _message_updated_at(pg_conn)
    moved = {guid for guid, stamp in after_updated_at.items() if before_updated_at[guid] != stamp}
    assert moved == {_ALICE_1}

    rewritten = _changed_rows(before_versions, _row_versions(pg_conn))
    assert {key for table, key in rewritten if table == "message"} == {(_ALICE_1,)}

    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = %s", (_ALICE_1,))
        assert cur.fetchone() == ("morning, corrected",)

    dirty = find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent)
    assert set(dirty) == {chat_ids["chat-alice"]}


def test_a_null_to_value_transition_counts_as_a_change(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, settled_corpus: Path
) -> None:
    """`IS DISTINCT FROM` is the reason a NULL-sided change is caught at
    all -- plain `<>` would evaluate to NULL and skip the update, leaving
    a message that gained a body looking unchanged forever. Proven here
    in the direction that is still a change, with NULL-to-NULL as the
    control.

    The opposite direction is asserted here too, and it changed on
    2026-09-17: value -> NULL is no longer a change, because a NULL body
    from `imsg-dump` is not "this message has no text", it is "this
    source could not produce the text" -- and several witnesses of the
    same conversation feed this index. See
    `test_extract_poorer_source_never_overwrites_integration.py` for the
    invariant and the production damage that motivated it. The two rules
    are independent: this suite is about not writing a row whose content
    is unchanged, that one about not writing a column this source cannot
    see.
    """
    chat_ids = _chat_ids_by_guid(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = %s", (_TAPBACK_1,))
        assert cur.fetchone() is None, "the tapback row is folded, not a message"

    # A source that returns no body for the owner's message no longer blanks
    # the body an earlier source stored -- and, being no change at all, does
    # not move the row either.
    dump = _dump_run()
    without_owner_body = ImsgDumpRun(
        messages=tuple(
            replace(m, body_text=None) if m.guid == _OWNER_1 else m for m in dump.messages
        ),
        stderr_lines=(),
    )
    before_updated_at = _message_updated_at(pg_conn)
    _reset_watermark(pg_conn)
    result = _extract(pg_conn, tmp_path, settled_corpus, without_owner_body)
    assert result.message_upserts.updated == 0
    assert _message_updated_at(pg_conn) == before_updated_at
    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = %s", (_OWNER_1,))
        assert cur.fetchone() == ("thanks",)

    # A row that arrives with no body at all, and later gains one. This is the
    # NULL-sided comparison `IS DISTINCT FROM` exists for: `<>` against a
    # stored NULL evaluates to NULL, so the update would be skipped and the
    # message would stay bodyless forever.
    grown = _build_snapshot(tmp_path / "snapshot_grown.db", extra_message=True)
    bodyless = ImsgDumpRun(
        messages=tuple(
            replace(m, body_text=None) if m.guid == "msg-bob-3" else m
            for m in _dump_run(extra_message=True).messages
        ),
        stderr_lines=(),
    )
    result = _extract(pg_conn, tmp_path, grown, bodyless)
    assert result.message_upserts.inserted == 1
    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = %s", ("msg-bob-3",))
        assert cur.fetchone() == (None,)

    # NULL -> NULL is the control: it must NOT count as a change.
    before_updated_at = _message_updated_at(pg_conn)
    _reset_watermark(pg_conn)
    result = _extract(pg_conn, tmp_path, grown, bodyless)
    assert result.message_upserts.updated == 0
    assert result.message_upserts.unchanged == 6
    assert _message_updated_at(pg_conn) == before_updated_at

    # And NULL -> value, which must.
    before_updated_at = _message_updated_at(pg_conn)
    _reset_watermark(pg_conn)
    result = _extract(pg_conn, tmp_path, grown, _dump_run(extra_message=True))
    assert result.message_upserts.updated == 1
    after_updated_at = _message_updated_at(pg_conn)
    assert {g for g, s in after_updated_at.items() if before_updated_at[g] != s} == {"msg-bob-3"}
    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = %s", ("msg-bob-3",))
        assert cur.fetchone() == ("one more thing",)

    run_identity(conn=pg_conn, config=config, contacts_importer=_fake_contacts_importer)  # type: ignore[arg-type]
    assert set(find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent)) == {
        chat_ids["chat-bob"]
    }


def test_a_new_message_in_an_existing_chat_marks_that_chat_dirty(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, settled_corpus: Path
) -> None:
    """A real incremental run -- watermark left alone, one genuinely new
    message. The new row must reach the index; only its chat is dirty.

    Identity runs between extraction and the dirty check because that is
    the production order (`imsg.stages.sync`: extract -> identity ->
    segment) and because `find_dirty_chats` only counts a message as
    unsegmented once `sender_person_id` is resolved. Skipping S3 here
    would leave the new message invisible to S4 and the test would
    "pass" for the wrong reason."""
    chat_ids = _chat_ids_by_guid(pg_conn)
    before_updated_at = _message_updated_at(pg_conn)

    snapshot_path = _build_snapshot(tmp_path / "snapshot2.db", extra_message=True)
    result = _extract(pg_conn, tmp_path, snapshot_path, _dump_run(extra_message=True))

    assert result.message_upserts.inserted == 1
    assert result.message_upserts.updated == 0
    assert result.message_upserts.unchanged == 0, "the watermark kept the settled rows out of scope"

    run_identity(conn=pg_conn, config=config, contacts_importer=_fake_contacts_importer)  # type: ignore[arg-type]

    after_updated_at = _message_updated_at(pg_conn)
    assert set(after_updated_at) - set(before_updated_at) == {"msg-bob-3"}
    # S3's sender backfill is itself guarded (`WHERE sender_person_id IS
    # NULL`), so it touches only the new row and the settled ones stay put.
    assert all(before_updated_at[g] == s for g, s in after_updated_at.items() if g in before_updated_at)

    dirty = find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent)
    assert set(dirty) == {chat_ids["chat-bob"]}


def test_identity_curation_still_bumps_every_message_in_the_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, settled_corpus: Path
) -> None:
    """The bump S3 performs deliberately is the mechanism that gets a
    renamed person back into the index, and skipping unchanged rows in S2
    must not weaken it. Run straight after a no-op re-extraction, so a
    regression that stopped the rename from marking anything would show
    up as an empty dirty set rather than being masked by S2's churn."""
    _reset_watermark(pg_conn)
    _extract(pg_conn, tmp_path, settled_corpus, _dump_run())
    assert find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent) == {}

    chat_ids = _chat_ids_by_guid(pg_conn)
    before_updated_at = _message_updated_at(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.person_id
            FROM person p
            JOIN handle h ON h.person_id = p.person_id
            WHERE h.normalized_value = %s
            """,
            (ALICE_HANDLE,),
        )
        row = cur.fetchone()
    assert row is not None
    alice_person_id = int(row[0])

    marked = rename_person(
        pg_conn,
        person_id=alice_person_id,
        display_name="Alice Example",
        short_name=f"alice-{uuid.uuid4().hex[:6]}",
    )
    assert marked == frozenset({chat_ids["chat-alice"]})

    after_updated_at = _message_updated_at(pg_conn)
    bumped = {g for g, s in after_updated_at.items() if before_updated_at[g] != s}
    assert bumped == {_ALICE_1, _ALICE_2, _OWNER_1}, "every message in Alice's chat, not just hers"

    dirty = find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent)
    assert set(dirty) == {chat_ids["chat-alice"]}
    assert dirty[chat_ids["chat-alice"]] == _BASE, "dirty from the chat's first message"
