"""Regression suite for S2's merge policy (2026-09-17).

INVARIANT UNDER TEST: a source that carries no evidence for a column
must never overwrite a value another source supplied — extraction
writes a column only where the incoming snapshot positively asserts it,
so absence of evidence is never recorded as evidence of absence.

Several witnesses of the same conversations feed this index (each Mac's
own `chat.db`, plus a recovered seed) and they do not carry the same
information. Until this date every `ON CONFLICT DO UPDATE` assigned
each column from the incoming row unconditionally, so the *last* source
to run won every column — including the columns it simply could not
see. Measured on the target host after one host re-extracted its own
`chat.db`, against a pre-run dump of the same index: 14,282 messages
lost `has_attachments`, 2,203 lost `is_edited` and its `date_edited`,
263 lost `is_unsent`, 3,125 chats lost `display_name`, and 4,775
attachments lost `source_path`.

Every test below extracts a RICH snapshot first and a POOR one second.
The two describe the same conversation; the poor one is a witness that
never received some of it. Nothing the rich source established may be
lost, and — the other half — a genuine correction must still land, which
the `*_still_lands` tests assert.

Since owner decision D12 (2026-09-23) "a genuine correction" means one
the live run carries: the poor witness here is `mini`, this machine's own
database, so it is extracted as `MergeMode.LIVE` -- the only mode that
may replace a non-empty value, which makes "it never overwrites with an
absence" the stronger claim. A seed carrying the same corrections only
fills, and a body changes only for a strictly newer edit; the full D12
suite is `test_extract_merges_only_add_integration.py`.

Fictional personas only (D5): Alice Example / Bob Builder, and
documentation-range phone numbers.
"""

from __future__ import annotations

import os
import plistlib
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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
from imsg.stages.extract import ExtractResult, MergeMode, run_extract
from imsg.stages.identity import run_identity
from imsg.stages.imsg_dump import EditVersion, ImsgDumpMessage, ImsgDumpRun, TapbackInfo

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_extract_merge_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
PROMPT_BYTES = b"fixed boundary-detection prompt for tests"

RICH_SOURCE = "studio"
POOR_SOURCE = "mini"


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


# --- the two witnesses -----------------------------------------------------

_BASE = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)

ALICE_HANDLE = "+15550000001"
BOB_HANDLE = "+15550000002"

CHAT_GUID = "chat-weekend"
GROUP_NAME = "Weekend Plans"

_PHOTO = "msg-photo"          # attachment join row: present in rich, absent in poor
_DOC = "msg-doc"              # attachment join row in both; metadata only in rich
_EDITED = "msg-edited"        # edit history + date_edited only in rich
_VERSIONED = "msg-versioned"  # edit history in both; the poor one cannot decode the text
_UNSENT = "msg-unsent"        # retraction only in rich
_PLAIN = "msg-plain"          # body only in rich (poor's shim returns no record at all)
_LINK = "msg-link"            # link preview: full in rich, url-only in poor
_FROM_BOB = "msg-from-bob"    # sender handle only in rich
_TAPBACK = "msg-tapback"      # tapback on _PLAIN: removed + dated in rich only

ATT_PHOTO = "att-photo"
ATT_DOC = "att-doc"


def _nskeyedarchiver_blob(fields: dict[str, str]) -> bytes:
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


_RICH_LINK_BLOB = _nskeyedarchiver_blob(
    {"URL": "https://example.com/trail", "title": "Trail Map", "siteName": "Example"}
)
_POOR_LINK_BLOB = _nskeyedarchiver_blob({"URL": "https://example.com/trail"})


def _messages(*, rich: bool) -> list[FixtureMessage]:
    """The same six conversational rows, one tapback, twice over. Every
    ROWID is explicit: the two snapshots must agree on them, or "the
    same conversation" would not be the same rows."""
    return [
        FixtureMessage(
            guid=_PHOTO, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=1, date=_BASE,
        ),
        FixtureMessage(
            guid=_DOC, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=2, date=_BASE + timedelta(minutes=1),
        ),
        FixtureMessage(
            guid=_EDITED, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=3, date=_BASE + timedelta(minutes=2),
            # The plain SQL half of "this was edited". A witness that never
            # received the edit has NULL here, exactly as it has NULL for a
            # message that was never edited at all.
            date_edited=(_BASE + timedelta(minutes=3)) if rich else None,
        ),
        FixtureMessage(
            guid=_VERSIONED, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=4, date=_BASE + timedelta(minutes=4),
            date_edited=_BASE + timedelta(minutes=5),
        ),
        FixtureMessage(
            guid=_UNSENT, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=5, date=_BASE + timedelta(minutes=6),
            date_retracted=(_BASE + timedelta(minutes=7)) if rich else None,
        ),
        FixtureMessage(
            guid=_PLAIN, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=6, date=_BASE + timedelta(minutes=8),
        ),
        FixtureMessage(
            guid=_LINK, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=7, date=_BASE + timedelta(minutes=9),
            payload_data=_RICH_LINK_BLOB if rich else _POOR_LINK_BLOB,
        ),
        FixtureMessage(
            # The poor witness has the row but not the `handle` reference:
            # `message.handle_id` is NULL, which is what a merged or partially
            # recovered corpus looks like, and is indistinguishable from the
            # owner's own message except for `is_from_me`.
            guid=_FROM_BOB, chat_guid=CHAT_GUID,
            handle_raw_value=BOB_HANDLE if rich else None,
            rowid=8, date=_BASE + timedelta(minutes=10),
        ),
        FixtureMessage(
            # A tapback row. The poor witness has no `date` for it at all,
            # which `chat.db` does carry on some rows and which the tapback
            # path (unlike the message path) tolerates rather than raising.
            guid=_TAPBACK, chat_guid=CHAT_GUID, handle_raw_value=BOB_HANDLE,
            rowid=9, date=(_BASE + timedelta(minutes=11)) if rich else None,
        ),
    ]


def _build_rich(path: Path) -> Path:
    builder = ChatDbBuilder()
    builder.add_chat(
        FixtureChat(guid=CHAT_GUID, rowid=1, style=43, display_name=GROUP_NAME,
                    service_name="iMessage")
    )
    builder.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    builder.add_handle(FixtureHandle(raw_value=BOB_HANDLE, rowid=2))
    builder.link_participant(CHAT_GUID, ALICE_HANDLE)
    builder.link_participant(CHAT_GUID, BOB_HANDLE)
    for message in _messages(rich=True):
        builder.add_message(message)
    builder.add_attachment(FixtureAttachment(guid=ATT_PHOTO, rowid=1))
    builder.add_attachment(
        FixtureAttachment(
            guid=ATT_DOC, rowid=2, filename="trip.pdf",
            source_path="/attachments/aa/bb/trip.pdf",
            uti="com.adobe.pdf", mime_type="application/pdf", byte_size=98765,
            is_sticker=True,
        )
    )
    builder.link_attachment(_PHOTO, ATT_PHOTO)
    builder.link_attachment(_DOC, ATT_DOC)
    return builder.build(path)


def _build_poor(path: Path) -> Path:
    """The same conversation as seen by a witness that never received
    the group's name, the attachment linkage, the edit, the retraction,
    or the attachment metadata — and whose `chat.style` is a value this
    build does not recognize, so `kind` can only be guessed."""
    builder = ChatDbBuilder()
    builder.add_chat(
        FixtureChat(guid=CHAT_GUID, rowid=1, style=0, display_name=None, service_name=None)
    )
    builder.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    builder.add_handle(FixtureHandle(raw_value=BOB_HANDLE, rowid=2))
    # Bob is in this witness's `handle` table, but not in its
    # `chat_handle_join` (so the chat looks like a DM to the participant-count
    # heuristic) and not on `_FROM_BOB`, whose `message.handle_id` is NULL.
    builder.link_participant(CHAT_GUID, ALICE_HANDLE)
    for message in _messages(rich=False):
        builder.add_message(message)
    builder.add_attachment(
        FixtureAttachment(
            guid=ATT_DOC, rowid=2, filename=None, source_path=None, uti=None,
            mime_type=None, byte_size=None, is_sticker=False,
        )
    )
    # No `message_attachment_join` row for _PHOTO at all — the join table is
    # what `has_attachments` is read off, and this witness never synced it.
    builder.link_attachment(_DOC, ATT_DOC)
    return builder.build(path)


def _dump_message(
    *,
    guid: str,
    rowid: int,
    body_text: str | None,
    edit_history: tuple[EditVersion, ...] = (),
    is_unsent: bool = False,
    tapback: TapbackInfo | None = None,
) -> ImsgDumpMessage:
    return ImsgDumpMessage(
        rowid=rowid, guid=guid, chat_guid=None, handle=None, is_from_me=False,
        date=None, date_edited=None, date_retracted=None, service="iMessage",
        body_text=body_text, edit_history=edit_history, is_unsent=is_unsent,
        tapback=tapback, attachment_rowids=(), reply_to_guid=None,
    )


def _rich_dump() -> ImsgDumpRun:
    return ImsgDumpRun(
        messages=(
            _dump_message(guid=_PHOTO, rowid=1, body_text="look at this"),
            _dump_message(guid=_DOC, rowid=2, body_text="the itinerary"),
            _dump_message(
                guid=_EDITED, rowid=3, body_text="saturday at ten",
                edit_history=(EditVersion(text="saturday at nine", edited_at=None),),
            ),
            _dump_message(
                guid=_VERSIONED, rowid=4, body_text="bring boots",
                edit_history=(EditVersion(text="bring shoes", edited_at="2024-06-01T09:05:00+00:00"),),
            ),
            _dump_message(guid=_UNSENT, rowid=5, body_text="never mind", is_unsent=True),
            _dump_message(guid=_PLAIN, rowid=6, body_text="sounds good"),
            _dump_message(guid=_LINK, rowid=7, body_text="see https://example.com/trail"),
            _dump_message(guid=_FROM_BOB, rowid=8, body_text="count me in"),
            _dump_message(
                guid=_TAPBACK, rowid=9, body_text=None,
                tapback=TapbackInfo(
                    target_guid=_PLAIN, kind="loved", emoji=None, action="removed"
                ),
            ),
        ),
        stderr_lines=(),
    )


def _poor_dump() -> ImsgDumpRun:
    """Same rows, minus everything this witness cannot decode. `_PLAIN`
    is absent from the output entirely — the `bodies_missing` case,
    where the shim returned no record for a guid the SQL half did
    return."""
    return ImsgDumpRun(
        messages=(
            _dump_message(guid=_PHOTO, rowid=1, body_text="look at this"),
            _dump_message(guid=_DOC, rowid=2, body_text="the itinerary"),
            _dump_message(guid=_EDITED, rowid=3, body_text="saturday at ten"),
            _dump_message(
                guid=_VERSIONED, rowid=4, body_text="bring boots",
                # The version is still reported, but its text did not decode.
                edit_history=(EditVersion(text=None, edited_at=None),),
            ),
            _dump_message(guid=_UNSENT, rowid=5, body_text="never mind", is_unsent=False),
            _dump_message(guid=_LINK, rowid=7, body_text="see https://example.com/trail"),
            _dump_message(guid=_FROM_BOB, rowid=8, body_text="count me in"),
            _dump_message(
                guid=_TAPBACK, rowid=9, body_text=None,
                tapback=TapbackInfo(
                    target_guid=_PLAIN, kind="loved", emoji=None, action="added"
                ),
            ),
        ),
        stderr_lines=(),
    )


def _fake_binary(tmp_path: Path) -> Path:
    p = tmp_path / "imsg-dump"
    p.write_text("")
    return p


def _extract(
    conn: psycopg.Connection,
    tmp_path: Path,
    snapshot_path: Path,
    dump: ImsgDumpRun,
    source_name: str,
) -> ExtractResult:
    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return dump

    return run_extract(
        conn=conn,
        source_name=source_name,
        snapshot_path=snapshot_path,
        imsg_dump_binary=_fake_binary(tmp_path),
        run_imsg_dump_fn=fake_run,
        # `mini` is this machine's own database; every other name is a seed.
        merge_mode=MergeMode.LIVE if source_name == POOR_SOURCE else MergeMode.SEED,
    )


# --- probes ----------------------------------------------------------------


def _message(conn: psycopg.Connection, guid: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sender_source_handle_id, is_from_me, sent_at, service, text_original,
                   text_normalized, is_unsent, is_edited, date_edited, has_attachments,
                   updated_at,
                   EXISTS (SELECT 1 FROM message_attachment ma
                            WHERE ma.message_id = message.message_id) AS linked
            FROM message WHERE source_guid = %s
            """,
            (guid,),
        )
        row = cur.fetchone()
    assert row is not None, f"no message row for {guid!r}"
    keys = (
        "sender_source_handle_id", "is_from_me", "sent_at", "service", "text_original",
        "text_normalized", "is_unsent", "is_edited", "date_edited", "has_attachments",
        "updated_at", "linked",
    )
    return dict(zip(keys, row, strict=True))


def _one(conn: psycopg.Connection, sql: str, *params: Any) -> tuple[Any, ...]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    assert row is not None
    return tuple(row)


def _chat_row(conn: psycopg.Connection) -> tuple[Any, ...]:
    return _one(
        conn, "SELECT kind::text, display_name, service::text FROM chat WHERE source_guid = %s",
        CHAT_GUID,
    )


def _attachment_row(conn: psycopg.Connection, guid: str) -> tuple[Any, ...]:
    return _one(
        conn,
        "SELECT filename, source_path, uti, mime_type, byte_size, is_sticker, state::text "
        "FROM attachment WHERE source_guid = %s",
        guid,
    )


def _fake_contacts_importer(default_region: str) -> list[object]:
    return []


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def rich_then_poor(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> dict[str, Any]:
    """Extract the rich witness, snapshot what it established, then
    extract the poor one. Returns the rich-side state so each test can
    assert against it rather than against a hand-copied literal."""
    rich_path = _build_rich(tmp_path / "rich.db")
    _extract(pg_conn, tmp_path, rich_path, _rich_dump(), RICH_SOURCE)

    before: dict[str, Any] = {
        "chat": _chat_row(pg_conn),
        "att_photo": _attachment_row(pg_conn, ATT_PHOTO),
        "att_doc": _attachment_row(pg_conn, ATT_DOC),
        "messages": {
            guid: _message(pg_conn, guid)
            for guid in (_PHOTO, _DOC, _EDITED, _VERSIONED, _UNSENT, _PLAIN, _LINK, _FROM_BOB)
        },
        "version": _one(
            pg_conn,
            "SELECT text, edited_at FROM message_version mv JOIN message m USING (message_id) "
            "WHERE m.source_guid = %s AND mv.version_idx = 0",
            _VERSIONED,
        ),
        "tapback": _one(
            pg_conn,
            "SELECT removed, acted_at, target_message_id FROM tapback WHERE source_guid = %s",
            _TAPBACK,
        ),
        "link_preview": _one(
            pg_conn,
            "SELECT title, summary, site_name FROM link_preview lp JOIN message m USING (message_id) "
            "WHERE m.source_guid = %s",
            _LINK,
        ),
    }
    # Guard: the rich extraction really did establish the things the poor one
    # is about to be tested against. Without this every assertion below could
    # pass by both sides being empty.
    assert before["chat"] == ("group", GROUP_NAME, "imessage")
    assert before["messages"][_PHOTO]["has_attachments"] is True
    assert before["messages"][_EDITED]["is_edited"] is True
    assert before["messages"][_UNSENT]["is_unsent"] is True
    assert before["messages"][_PLAIN]["text_original"] == "sounds good"
    assert before["messages"][_FROM_BOB]["sender_source_handle_id"] is not None
    assert before["att_doc"][0] == "trip.pdf"
    assert before["version"][0] == "bring shoes"
    assert before["tapback"][0] is True
    assert before["link_preview"] == ("Trail Map", None, "Example")

    poor_path = _build_poor(tmp_path / "poor.db")
    result = _extract(pg_conn, tmp_path, poor_path, _poor_dump(), POOR_SOURCE)
    return {"before": before, "result": result, "poor_path": poor_path}


# --- the instances ---------------------------------------------------------


def test_a_source_without_attachment_join_rows_keeps_has_attachments(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """Instance 1, and the one measured in production: 14,282 messages
    flipped to `has_attachments = false` while keeping their
    `message_attachment` rows, because the second witness's
    `message_attachment_join` was never synced. Rendering reads the
    real attachment rows; the segmentation boundary prompt reads this
    column, so segmenting would feed degraded input into new
    embeddings."""
    row = _message(pg_conn, _PHOTO)
    assert row["has_attachments"] is True
    assert row["linked"] is True, "the join row is what makes the flip detectable at all"


def test_a_source_with_no_dump_record_keeps_the_body(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """Instance 2: when the shim returns no record for a guid,
    `_upsert_message` used to write NULL over `text_original` /
    `text_normalized` that an earlier run stored, count it in
    `bodies_missing`, log it, and carry on. The count and the log are
    right; the write was not."""
    result: ExtractResult = rich_then_poor["result"]
    assert result.bodies_missing == 1, "the boundary anomaly must still be reported"

    row = _message(pg_conn, _PLAIN)
    assert row["text_original"] == "sounds good"
    assert row["text_normalized"] is not None


def test_a_source_that_never_saw_the_edit_keeps_is_edited_and_date_edited(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """A witness without the edit reports `edit_history=[]` and a NULL
    `date_edited` — the same two values it reports for a message that
    was never edited. Editedness does not un-happen."""
    before = rich_then_poor["before"]["messages"][_EDITED]
    row = _message(pg_conn, _EDITED)
    assert row["is_edited"] is True
    assert row["date_edited"] == before["date_edited"]


def test_a_source_that_never_saw_the_retraction_keeps_is_unsent(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """`is_unsent` is derived from a typedstream blob a witness may not
    have; its `false` means "I see no retraction", never "this was not
    retracted". Unsending is terminal."""
    assert _message(pg_conn, _UNSENT)["is_unsent"] is True


def test_a_source_without_the_handle_row_keeps_the_sender(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """`sender_source_handle_id` is the only column S3 joins on to
    resolve a sender. A snapshot whose `message.handle_id` is NULL
    would otherwise detach the message from its sender permanently."""
    before = rich_then_poor["before"]["messages"][_FROM_BOB]
    row = _message(pg_conn, _FROM_BOB)
    assert row["sender_source_handle_id"] == before["sender_source_handle_id"]
    assert row["is_from_me"] is False


def test_a_source_without_the_group_name_keeps_it(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """3,125 chats lost their `display_name` this way in production. A
    device that joined late, or never received the rename event, has
    NULL here."""
    assert _chat_row(pg_conn)[1] == GROUP_NAME


def test_an_unrecognized_chat_style_does_not_retype_the_chat(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """`kind` falls back to a participant-count guess when `chat.style`
    is a value this build does not recognize. A guess is enough to
    insert a new row with and not enough to overwrite one: the poor
    witness links a single participant, so its guess is `dm` for a chat
    the rich witness typed `group` off a real `style`."""
    assert _chat_row(pg_conn)[0] == "group"


def test_an_unknown_service_does_not_overwrite_a_known_one(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """`unknown` is not a service; it is this module's marker for "the
    snapshot did not say"."""
    assert _chat_row(pg_conn)[2] == "imessage"


def test_a_source_without_attachment_metadata_keeps_it(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """Every chat.db-derived attachment column is nullable at the
    source, so a NULL is exactly what a snapshot missing the row's
    metadata reports. 4,775 attachments lost `source_path` in
    production — all but 33 of them already materialized, so the loss
    was of provenance rather than of retrievability, which is precisely
    the kind of damage nothing downstream would have surfaced."""
    assert _attachment_row(pg_conn, ATT_DOC) == rich_then_poor["before"]["att_doc"]


def test_a_version_the_shim_could_not_decode_does_not_blank_it(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """`message_version.text` is NOT NULL, so an undecodable version
    used to land as the empty string — and an empty string written over
    a version another source decoded is the same loss as a blanked
    body, minus the NULL that would have made it visible."""
    row = _one(
        pg_conn,
        "SELECT text, edited_at FROM message_version mv JOIN message m USING (message_id) "
        "WHERE m.source_guid = %s AND mv.version_idx = 0",
        _VERSIONED,
    )
    assert row == rich_then_poor["before"]["version"]


def test_a_tapback_keeps_its_removal_and_its_timestamp(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """`ImsgDumpMessage.action` defaults to "added" when the shim omits
    it, so `removed = false` is also what "the shim said nothing" looks
    like; and `acted_at` comes from a `chat.db` `date` that this path —
    unlike the message path — tolerates as NULL."""
    assert _one(
        pg_conn,
        "SELECT removed, acted_at, target_message_id FROM tapback WHERE source_guid = %s",
        _TAPBACK,
    ) == rich_then_poor["before"]["tapback"]


def test_a_link_preview_without_a_title_keeps_it(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """`parse_link_preview` is a best-effort walk of an NSKeyedArchiver
    blob: a key it does not find comes back NULL whether the preview had
    no title or this source's `payload_data` was shaped differently."""
    assert _one(
        pg_conn,
        "SELECT title, summary, site_name FROM link_preview lp JOIN message m USING (message_id) "
        "WHERE m.source_guid = %s",
        _LINK,
    ) == rich_then_poor["before"]["link_preview"]


def test_the_poorer_run_writes_no_message_row_at_all(
    pg_conn: psycopg.Connection, rich_then_poor: dict[str, Any]
) -> None:
    """The operator-facing half, and the reason this defect cost 17
    hours of recomputed embeddings rather than a quiet wrong value: a
    source with nothing new to say must report every row unchanged and
    move no `updated_at`, so S4 proposes no re-segmentation.

    Before the fix this run rewrote nine message rows and bumped all of
    them."""
    result: ExtractResult = rich_then_poor["result"]
    assert result.message_upserts.updated == 0
    assert result.message_upserts.inserted == 0
    assert result.message_upserts.unchanged == result.messages_upserted
    assert result.chat_upserts.updated == 0
    assert result.attachment_upserts.updated == 0

    before = rich_then_poor["before"]["messages"]
    moved = {
        guid: _message(pg_conn, guid)["updated_at"]
        for guid in before
        if _message(pg_conn, guid)["updated_at"] != before[guid]["updated_at"]
    }
    assert moved == {}


def test_the_poorer_run_dirties_no_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    """The same property stated where it is actually paid for. The
    corpus is segmented and its identities resolved between the two
    extractions, so `find_dirty_chats` has real segments to compare
    against and could genuinely return something."""
    rich_path = _build_rich(tmp_path / "rich.db")
    _extract(pg_conn, tmp_path, rich_path, _rich_dump(), RICH_SOURCE)

    run_identity(conn=pg_conn, config=config, contacts_importer=_fake_contacts_importer)  # type: ignore[arg-type]
    with pg_conn.cursor() as cur:
        cur.execute("SELECT chat_id FROM chat")
        chat_ids = [int(r[0]) for r in cur.fetchall()]
    for chat_id in chat_ids:
        run_segment_for_chat(
            pg_conn, chat_id, config, FakeBoundaryProvider(), PROMPT_BYTES,
            earliest_changed_at=REBUILD_ALL_SENTINEL,
        )
    assert find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent) == {}, (
        "the baseline must start settled, or the assertion below is vacuous"
    )

    poor_path = _build_poor(tmp_path / "poor.db")
    _extract(pg_conn, tmp_path, poor_path, _poor_dump(), POOR_SOURCE)

    assert find_dirty_chats(pg_conn, index_unsent=config.policy.index_unsent) == {}


# --- the other half: a genuine correction still flows ----------------------


def _corrected_dump() -> ImsgDumpRun:
    """The poor witness, but carrying real news: an edited body, a
    message that has since been unsent, and a tapback that has since
    been taken back."""
    messages = []
    for message in _poor_dump().messages:
        if message.guid == _PLAIN:
            continue
        if message.guid == _PHOTO:
            messages.append(_dump_message(guid=_PHOTO, rowid=1, body_text="look at THIS"))
        elif message.guid == _DOC:
            messages.append(_dump_message(guid=_DOC, rowid=2, body_text="the itinerary",
                                          is_unsent=True))
        elif message.guid == _TAPBACK:
            messages.append(
                _dump_message(
                    guid=_TAPBACK, rowid=9, body_text=None,
                    tapback=TapbackInfo(
                        target_guid=_PLAIN, kind="loved", emoji=None, action="removed"
                    ),
                )
            )
        else:
            messages.append(message)
    return ImsgDumpRun(messages=tuple(messages), stderr_lines=())


_PHOTO_EDITED_AT = _BASE + timedelta(minutes=30)


def _build_corrected(path: Path) -> Path:
    """The poor snapshot plus the corrections a real second witness can
    legitimately carry: the group was renamed, the attachment moved and
    was renamed, and `_PHOTO` was edited -- which `chat.db` records as a
    `date_edited`, the edit time D12 compares."""
    builder = ChatDbBuilder()
    builder.add_chat(
        FixtureChat(guid=CHAT_GUID, rowid=1, style=43, display_name="Weekend Plans v2",
                    service_name="SMS")
    )
    builder.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    builder.add_handle(FixtureHandle(raw_value=BOB_HANDLE, rowid=2))
    # Bob is in this witness's `handle` table, but not in its
    # `chat_handle_join` (so the chat looks like a DM to the participant-count
    # heuristic) and not on `_FROM_BOB`, whose `message.handle_id` is NULL.
    builder.link_participant(CHAT_GUID, ALICE_HANDLE)
    for message in _messages(rich=False):
        if message.guid == _PHOTO:
            message = replace(message, date_edited=_PHOTO_EDITED_AT)
        builder.add_message(message)
    builder.add_attachment(
        FixtureAttachment(
            guid=ATT_DOC, rowid=2, filename="trip-final.pdf",
            source_path="/attachments/cc/dd/trip-final.pdf",
            uti="com.adobe.pdf", mime_type="application/pdf", byte_size=112233,
            is_sticker=True,
        )
    )
    builder.link_attachment(_DOC, ATT_DOC)
    return builder.build(path)


def test_a_genuine_correction_from_the_live_run_still_lands(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The invariant blocks only *absence*. Everything the live run
    positively asserts — an edited body, a retraction that really did
    happen since, a renamed chat, a service it now knows, an attachment
    that moved, a tapback that was taken back — must still land, or the
    fix would have traded one silent corruption for another."""
    rich_path = _build_rich(tmp_path / "rich.db")
    _extract(pg_conn, tmp_path, rich_path, _rich_dump(), RICH_SOURCE)

    corrected_path = _build_corrected(tmp_path / "corrected.db")
    result = _extract(pg_conn, tmp_path, corrected_path, _corrected_dump(), POOR_SOURCE)

    assert _message(pg_conn, _PHOTO)["text_original"] == "look at THIS"
    assert _message(pg_conn, _DOC)["is_unsent"] is True
    assert _chat_row(pg_conn) == ("group", "Weekend Plans v2", "sms")
    assert _attachment_row(pg_conn, ATT_DOC)[:6] == (
        "trip-final.pdf", "/attachments/cc/dd/trip-final.pdf", "com.adobe.pdf",
        "application/pdf", 112233, True,
    )
    assert _one(
        pg_conn, "SELECT removed FROM tapback WHERE source_guid = %s", _TAPBACK
    ) == (True,)

    # And the corrections are counted as corrections, not as no-ops: the
    # edit as a newer edit, the retraction as a fill, the rename and the
    # move as replacements -- which only the live run may make.
    assert (result.message_upserts.newer_edit, result.message_upserts.filled) == (1, 1)
    assert result.message_upserts.replaced == 0
    assert result.chat_upserts.replaced == 1
    assert result.attachment_upserts.replaced == 1


def test_the_same_corrections_from_a_seed_only_fill(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """D12: a seed carrying the same snapshot adds what the index lacks
    -- the retraction, and the body a strictly newer edit brought, from
    any source -- and replaces nothing: the rename, the other service and
    the other attachment metadata are refused."""
    rich_path = _build_rich(tmp_path / "rich.db")
    _extract(pg_conn, tmp_path, rich_path, _rich_dump(), RICH_SOURCE)
    chat_before = _chat_row(pg_conn)
    attachment_before = _attachment_row(pg_conn, ATT_DOC)

    corrected_path = _build_corrected(tmp_path / "corrected.db")
    result = _extract(pg_conn, tmp_path, corrected_path, _corrected_dump(), "recovered-copy")

    assert _message(pg_conn, _PHOTO)["text_original"] == "look at THIS"
    assert _message(pg_conn, _DOC)["is_unsent"] is True
    assert _chat_row(pg_conn) == chat_before
    assert _attachment_row(pg_conn, ATT_DOC) == attachment_before
    assert (result.message_upserts.newer_edit, result.message_upserts.filled) == (1, 1)
    for table, counts in result.table_counts().items():
        assert counts.replaced == 0, table


def test_a_missing_attachment_path_that_later_arrives_still_reopens_materialization(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The one conditional assignment in `_upsert_attachment` — a
    `missing` row that had no path and now has one goes back to
    `dataless` — depends on `source_path` actually being written. It
    still is, because a path is evidence."""
    builder = ChatDbBuilder()
    builder.add_chat(FixtureChat(guid=CHAT_GUID, rowid=1, style=45))
    builder.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    builder.link_participant(CHAT_GUID, ALICE_HANDLE)
    builder.add_message(
        FixtureMessage(guid=_DOC, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
                       rowid=1, date=_BASE)
    )
    builder.add_attachment(FixtureAttachment(guid=ATT_DOC, rowid=1, source_path=None))
    builder.link_attachment(_DOC, ATT_DOC)
    pathless = builder.build(tmp_path / "pathless.db")
    dump = ImsgDumpRun(
        messages=(_dump_message(guid=_DOC, rowid=1, body_text="the itinerary"),),
        stderr_lines=(),
    )
    _extract(pg_conn, tmp_path, pathless, dump, RICH_SOURCE)
    assert _attachment_row(pg_conn, ATT_DOC)[6] == "missing"

    builder = ChatDbBuilder()
    builder.add_chat(FixtureChat(guid=CHAT_GUID, rowid=1, style=45))
    builder.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    builder.link_participant(CHAT_GUID, ALICE_HANDLE)
    builder.add_message(
        FixtureMessage(guid=_DOC, chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
                       rowid=1, date=_BASE)
    )
    builder.add_attachment(
        FixtureAttachment(guid=ATT_DOC, rowid=1, source_path="/attachments/ee/ff/trip.pdf")
    )
    builder.link_attachment(_DOC, ATT_DOC)
    with_path = builder.build(tmp_path / "with_path.db")
    _extract(pg_conn, tmp_path, with_path, dump, POOR_SOURCE)

    row = _attachment_row(pg_conn, ATT_DOC)
    assert row[1] == "/attachments/ee/ff/trip.pdf"
    assert row[6] == "dataless"
