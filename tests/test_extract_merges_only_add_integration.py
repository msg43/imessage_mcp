"""Regression suite for owner decision D12 (2026-09-23): corpus merges only add.

"Always want to merge and maintain the fullest corpus of chat.db and
attachments, not overwrite and lose rows."

Several witnesses feed one index: this machine's own `chat.db`, another
Mac's, an older merged corpus, recovery candidates. Until this rule each
column's merge policy (`imsg.stages.extract.Merge`) decided only whether
an incoming value counted as evidence, and under `PRESENT` any non-NULL
value did -- an empty string and a 0 included. Whichever source ran last
won. A dry run on the production host of a recovery candidate built on
an older corpus would have put the older values back over 526 messages,
32,971 attachment rows and 365 chats (one chat name blanked to `''`),
and the command printed only the message count.

The rule under test:

  1. A seed (`MergeMode.SEED`: anything but this machine's own live
     database) inserts rows and fills empty values. It never replaces a
     non-empty value -- `sent_at` and `is_from_me` included.
  2. An empty string, a 0 size, or an `unknown` service is never
     evidence, from any source.
  3. The live run (`MergeMode.LIVE`) still applies genuine changes: a
     renamed chat, an attachment the Messages app moved.
  4. A message body changes only when it is empty or a strictly newer
     edit (`date_edited`) brings the new text, from any source. The body
     it replaces is kept in `message_version`, and a stored version is
     never replaced.
  5. Every table the run writes reports inserted / filled / newer edit /
     replaced / unchanged, so a seed can be checked against "inserts and
     fills only".
  6. A fill that changes what a segment renders still marks the chat for
     re-segmentation -- including fills of tables that have no
     `updated_at` of their own (chat, attachment, tapback, edit history).

Each test extracts one witness and then a second one derived from it with
`dataclasses.replace`, changing only the fields under test, so nothing
else can be the reason a row moved.

Fictional personas only (D5): Alice Example / Bob Builder, and
documentation-range phone numbers.
"""

from __future__ import annotations

import json
import os
import plistlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
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
TEST_DB_NAME = "imsg_index_extract_only_add_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
PROMPT_BYTES = b"fixed boundary-detection prompt for tests"

LIVE_SOURCE = "mini"
"""The live run's source name. Seeds each take a fresh name, as the CLI
requires, so each seed's ROWID watermark starts at zero."""


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


# --- one witness of two group chats -----------------------------------------

_BASE = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
_EDIT_1 = _BASE + timedelta(hours=1)
_EDIT_2 = _BASE + timedelta(hours=2)

ALICE = "+15550000001"
BOB = "+15550000002"

CHAT_NAMED = "chat-weekend"
CHAT_UNNAMED = "chat-bookclub"

MSG_TEXT = "msg-text"      # the body edits and seeds disagree about
MSG_PHOTO = "msg-photo"    # carries ATT_PHOTO when `photo_linked`
MSG_BARE = "msg-bare"      # carries ATT_BARE; its body is what `bare_body` says
MSG_CLUB = "msg-club"      # the only message in the unnamed group; has a link preview
MSG_REACT = "msg-react"    # Bob's tapback on MSG_TEXT, present when `reaction`

ATT_PHOTO = "att-photo"
ATT_BARE = "att-bare"


@dataclass(frozen=True)
class Att:
    """The chat.db-derived columns of one attachment. Frozen (the
    fixture's own `FixtureAttachment` is not), so it can sit in `World`."""

    filename: str | None
    source_path: str | None
    uti: str | None
    mime_type: str | None
    byte_size: int | None
    is_sticker: bool = False


_PHOTO = Att(
    filename="IMG_0001.jpeg",
    source_path="~/Library/Messages/Attachments/aa/01/IMG_0001.jpeg",
    uti="public.jpeg",
    mime_type="image/jpeg",
    byte_size=48213,
)
_BARE = Att(
    filename=None,
    source_path="~/Library/Messages/Attachments/bb/02/notes",
    uti=None,
    mime_type=None,
    byte_size=None,
)


@dataclass(frozen=True)
class World:
    """Everything one witness says. Defaults are the index's first view;
    each test derives the second witness with `replace()`."""

    named_group_name: str | None = "Weekend Plans"
    named_group_service: str | None = "iMessage"
    unnamed_group_name: str | None = None
    unnamed_group_service: str | None = "iMessage"

    text_body: str | None = "saturday at ten"
    text_edited_at: datetime | None = None
    text_history: tuple[EditVersion, ...] = ()
    text_sent_at: datetime = _BASE
    text_from_me: bool = False
    text_unsent: bool = False

    bare_body: str | None = None

    photo_linked: bool = True
    photo_also_on_bare: bool = False
    """A second attachment link on MSG_BARE, which already has one: a new
    link that does not change `has_attachments`."""
    photo: Att = _PHOTO
    bare: Att = _BARE

    reaction: bool = False


def _link_blob(url: str, title: str) -> bytes:
    root: dict[str, object] = {}
    fields = {"URL": url, "title": title}
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


def _fixture_attachment(guid: str, rowid: int, att: Att) -> FixtureAttachment:
    return FixtureAttachment(
        guid=guid, rowid=rowid, filename=att.filename, source_path=att.source_path,
        uti=att.uti, mime_type=att.mime_type, byte_size=att.byte_size,
        is_sticker=att.is_sticker,
    )


def _build(world: World, path: Path) -> Path:
    """Every ROWID explicit, so two witnesses agree on which row is
    which and differ only in what `World` says."""
    builder = ChatDbBuilder()
    builder.add_chat(
        FixtureChat(guid=CHAT_NAMED, rowid=1, style=43, display_name=world.named_group_name,
                    service_name=world.named_group_service)
    )
    builder.add_chat(
        FixtureChat(guid=CHAT_UNNAMED, rowid=2, style=43, display_name=world.unnamed_group_name,
                    service_name=world.unnamed_group_service)
    )
    builder.add_handle(FixtureHandle(raw_value=ALICE, rowid=1))
    builder.add_handle(FixtureHandle(raw_value=BOB, rowid=2))
    for chat in (CHAT_NAMED, CHAT_UNNAMED):
        builder.link_participant(chat, ALICE)
        builder.link_participant(chat, BOB)

    builder.add_message(
        FixtureMessage(
            guid=MSG_TEXT, chat_guid=CHAT_NAMED, rowid=1, is_from_me=world.text_from_me,
            handle_raw_value=None if world.text_from_me else ALICE,
            date=world.text_sent_at, date_edited=world.text_edited_at,
        )
    )
    builder.add_message(
        FixtureMessage(guid=MSG_PHOTO, chat_guid=CHAT_NAMED, handle_raw_value=ALICE, rowid=2,
                       date=_BASE + timedelta(minutes=1))
    )
    builder.add_message(
        FixtureMessage(guid=MSG_BARE, chat_guid=CHAT_NAMED, handle_raw_value=BOB, rowid=3,
                       date=_BASE + timedelta(minutes=2))
    )
    builder.add_message(
        FixtureMessage(guid=MSG_CLUB, chat_guid=CHAT_UNNAMED, handle_raw_value=BOB, rowid=4,
                       date=_BASE + timedelta(minutes=3),
                       payload_data=_link_blob("https://example.com/club", "Reading List"))
    )
    if world.reaction:
        builder.add_message(
            FixtureMessage(guid=MSG_REACT, chat_guid=CHAT_NAMED, handle_raw_value=BOB, rowid=5,
                           date=_BASE + timedelta(minutes=4))
        )

    builder.add_attachment(_fixture_attachment(ATT_PHOTO, 1, world.photo))
    builder.add_attachment(_fixture_attachment(ATT_BARE, 2, world.bare))
    if world.photo_linked:
        builder.link_attachment(MSG_PHOTO, ATT_PHOTO)
    builder.link_attachment(MSG_BARE, ATT_BARE)
    if world.photo_also_on_bare:
        builder.link_attachment(MSG_BARE, ATT_PHOTO)
    return builder.build(path)


def _dump_message(
    guid: str,
    rowid: int,
    body_text: str | None,
    *,
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


def _dump(world: World) -> ImsgDumpRun:
    messages = [
        _dump_message(MSG_TEXT, 1, world.text_body, edit_history=world.text_history,
                      is_unsent=world.text_unsent),
        _dump_message(MSG_PHOTO, 2, "look at this"),
        _dump_message(MSG_BARE, 3, world.bare_body),
        _dump_message(MSG_CLUB, 4, "chapter three is up https://example.com/club"),
    ]
    if world.reaction:
        messages.append(
            _dump_message(
                MSG_REACT, 5, None,
                tapback=TapbackInfo(target_guid=MSG_TEXT, kind="loved", emoji=None, action="added"),
            )
        )
    return ImsgDumpRun(messages=tuple(messages), stderr_lines=())


_RUNS = iter(range(1, 10_000))


def _extract(
    conn: psycopg.Connection,
    tmp_path: Path,
    world: World,
    *,
    source: str,
    mode: MergeMode | None,
) -> ExtractResult:
    """`mode=None` leaves `merge_mode` unpassed, which is how the
    fail-safe default is exercised."""
    snapshot = _build(world, tmp_path / f"{source}-{next(_RUNS)}.db")
    binary = tmp_path / "imsg-dump"
    binary.write_text("")
    extra: dict[str, Any] = {} if mode is None else {"merge_mode": mode}
    return run_extract(
        conn=conn,
        source_name=source,
        snapshot_path=snapshot,
        imsg_dump_binary=binary,
        run_imsg_dump_fn=lambda _binary, _snapshot, _since: _dump(world),
        **extra,
    )


def _establish(conn: psycopg.Connection, tmp_path: Path, world: World) -> ExtractResult:
    """The index's first view, from the live run."""
    return _extract(conn, tmp_path, world, source=LIVE_SOURCE, mode=MergeMode.LIVE)


def _live_again(conn: psycopg.Connection, tmp_path: Path, world: World) -> ExtractResult:
    """A second live run over the whole corpus. The watermark is reset so
    every row is back in scope, as it would be for any row the live
    database rescans."""
    conn.execute("DELETE FROM sync_state WHERE key = %s", (f"watermark.rowid.{LIVE_SOURCE}",))
    return _extract(conn, tmp_path, world, source=LIVE_SOURCE, mode=MergeMode.LIVE)


def _seed(conn: psycopg.Connection, tmp_path: Path, world: World, name: str) -> ExtractResult:
    return _extract(conn, tmp_path, world, source=name, mode=MergeMode.SEED)


# --- probes -----------------------------------------------------------------


def _row(conn: psycopg.Connection, sql: str, *params: Any) -> tuple[Any, ...]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    assert row is not None, sql
    return tuple(row)


def _chat(conn: psycopg.Connection, guid: str) -> tuple[Any, ...]:
    return _row(
        conn, "SELECT kind::text, display_name, service::text FROM chat WHERE source_guid = %s", guid
    )


def _attachment(conn: psycopg.Connection, guid: str) -> Att:
    filename, source_path, uti, mime_type, byte_size, is_sticker = _row(
        conn,
        "SELECT filename, source_path, uti, mime_type, byte_size, is_sticker "
        "FROM attachment WHERE source_guid = %s",
        guid,
    )
    return Att(filename, source_path, uti, mime_type, byte_size, is_sticker)


def _message(conn: psycopg.Connection, guid: str) -> dict[str, Any]:
    keys = (
        "text_original", "text_normalized", "date_edited", "is_edited", "is_unsent",
        "has_attachments", "sent_at", "is_from_me", "updated_at",
    )
    values = _row(
        conn, f"SELECT {', '.join(keys)} FROM message WHERE source_guid = %s", guid
    )
    return dict(zip(keys, values, strict=True))


def _versions(conn: psycopg.Connection, guid: str) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT mv.version_idx, mv.text FROM message_version mv "
            "JOIN message m USING (message_id) WHERE m.source_guid = %s ORDER BY mv.version_idx",
            (guid,),
        )
        return [(int(idx), str(text)) for idx, text in cur.fetchall()]


def _chat_ids(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT source_guid, chat_id FROM chat")
        return {str(guid): int(chat_id) for guid, chat_id in cur.fetchall()}


def _fake_contacts_importer(default_region: str) -> list[object]:
    return []


def _settle(conn: psycopg.Connection, config: Config) -> None:
    """Resolve identities and segment every chat, so `find_dirty_chats`
    has real segments to compare against. Without this the dirty-chat
    assertions could pass because nothing was ever segmented."""
    run_identity(conn=conn, config=config, contacts_importer=_fake_contacts_importer)  # type: ignore[arg-type]
    for chat_id in _chat_ids(conn).values():
        run_segment_for_chat(
            conn, chat_id, config, FakeBoundaryProvider(), PROMPT_BYTES,
            earliest_changed_at=REBUILD_ALL_SENTINEL,
        )
    assert find_dirty_chats(conn, index_unsent=config.policy.index_unsent) == {}, (
        "the baseline must start settled, or every dirty-chat assertion is vacuous"
    )


def _dirty(conn: psycopg.Connection, config: Config) -> set[int]:
    return set(find_dirty_chats(conn, index_unsent=config.policy.index_unsent))


def _assert_no_replacement(result: ExtractResult) -> None:
    """The seed contract, stated over every table the run reports."""
    for table, counts in result.table_counts().items():
        assert counts.replaced == 0, f"{table}: a seed replaced a non-empty value"


# --- 1. a seed never replaces a non-empty value ------------------------------


def test_a_seed_with_an_older_body_does_not_replace_a_newer_one(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The recovery candidate is built on an older corpus. Where the index
    holds a later edit, the candidate's earlier text must not come back
    over it -- whether the candidate saw the message before any edit or
    at an earlier edit."""
    stored = World(
        text_body="saturday at ten", text_edited_at=_EDIT_2,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),),
    )
    _establish(pg_conn, tmp_path, stored)
    before = _message(pg_conn, MSG_TEXT)

    before_any_edit = replace(stored, text_body="saturday at nine", text_edited_at=None,
                              text_history=())
    result = _seed(pg_conn, tmp_path, before_any_edit, "seed-older")
    at_an_earlier_edit = replace(stored, text_body="saturday at 9:30", text_edited_at=_EDIT_1)
    result_2 = _seed(pg_conn, tmp_path, at_an_earlier_edit, "seed-earlier-edit")

    assert _message(pg_conn, MSG_TEXT) == before
    for r in (result, result_2):
        assert r.message_upserts.newer_edit == 0
        _assert_no_replacement(r)


def test_a_seed_body_that_differs_only_in_form_does_not_replace_it(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Most bodies the production dry run would have rewritten differed
    only in form: a curly apostrophe against a straight one, Unicode
    normalization, whitespace. Neither side is newer, so neither wins."""
    stored = World(text_body="it\N{RIGHT SINGLE QUOTATION MARK}s at ten")
    _establish(pg_conn, tmp_path, stored)
    before = _message(pg_conn, MSG_TEXT)

    _seed(pg_conn, tmp_path, replace(stored, text_body="it's at ten"), "seed-form")

    assert _message(pg_conn, MSG_TEXT) == before


def test_a_seed_with_an_empty_display_name_does_not_blank_it(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The dry run would have blanked one chat's name to `''`."""
    _establish(pg_conn, tmp_path, World())

    result = _seed(pg_conn, tmp_path, replace(World(), named_group_name=""), "seed-blank-name")

    assert _chat(pg_conn, CHAT_NAMED) == ("group", "Weekend Plans", "imessage")
    assert result.chat_upserts.updated == 0


def test_a_seed_with_a_zero_byte_size_does_not_replace_the_size(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """`chat.db` reports 0 for a transfer that never completed; the dry run
    would have zeroed 29 sizes."""
    _establish(pg_conn, tmp_path, World())

    _seed(pg_conn, tmp_path, replace(World(), photo=replace(_PHOTO, byte_size=0)), "seed-zero")

    assert _attachment(pg_conn, ATT_PHOTO).byte_size == 48213


def test_a_seed_does_not_replace_attachment_metadata_or_a_chat_service(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Another Mac's paths are its own (`/Users/<name>/...` against `~/...`),
    and its filename, type and service can differ for the same GUID.
    None of that is newer than what the index holds."""
    _establish(pg_conn, tmp_path, World())
    other_mac = replace(
        World(),
        named_group_service="SMS",
        photo=Att(
            filename="IMG_0001 2.jpeg",
            source_path="/Users/example/Library/Messages/Attachments/aa/01/IMG_0001.jpeg",
            uti="public.heic",
            mime_type="image/heic",
            byte_size=51000,
        ),
    )

    result = _seed(pg_conn, tmp_path, other_mac, "seed-other-mac")

    assert _attachment(pg_conn, ATT_PHOTO) == _PHOTO
    assert _chat(pg_conn, CHAT_NAMED)[2] == "imessage"
    assert result.attachment_upserts.updated == 0
    assert result.chat_upserts.updated == 0
    _assert_no_replacement(result)


def test_a_seed_does_not_move_sent_at_or_flip_is_from_me(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """`sent_at` and `is_from_me` are asserted by every source, so they
    used to follow whichever ran last. The dry run moved four `sent_at`
    values by under a second."""
    _establish(pg_conn, tmp_path, World())
    before = _message(pg_conn, MSG_TEXT)

    shifted = replace(World(), text_sent_at=_BASE + timedelta(milliseconds=400), text_from_me=True)
    result = _seed(pg_conn, tmp_path, shifted, "seed-shifted")

    after = _message(pg_conn, MSG_TEXT)
    assert (after["sent_at"], after["is_from_me"]) == (before["sent_at"], False)
    assert after["updated_at"] == before["updated_at"]
    assert result.message_upserts.updated == 0


def test_a_caller_that_does_not_name_a_mode_gets_the_seed_rule(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Fail safe: `run_extract` without `merge_mode` must not be able to
    replace anything."""
    _establish(pg_conn, tmp_path, World())

    result = _extract(
        pg_conn, tmp_path, replace(World(), named_group_name="Renamed Elsewhere"),
        source="seed-unnamed-mode", mode=None,
    )

    assert _chat(pg_conn, CHAT_NAMED)[1] == "Weekend Plans"
    assert result.merge_mode is MergeMode.SEED


# --- 2. a seed fills what the index lacks ------------------------------------


def test_a_seed_fills_values_the_index_lacks_and_counts_them_as_fills(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Stored NULL (or the `unknown` service marker) plus a seed value is
    a fill, the one kind of change a seed makes to an existing row."""
    stored = replace(World(), unnamed_group_service=None)
    _establish(pg_conn, tmp_path, stored)
    assert _chat(pg_conn, CHAT_UNNAMED) == ("group", None, "unknown")

    richer = replace(
        stored,
        unnamed_group_name="Book Club",
        unnamed_group_service="iMessage",
        bare_body="found the notes",
        bare=replace(_BARE, filename="notes.pdf", uti="com.adobe.pdf",
                     mime_type="application/pdf", byte_size=2048),
    )
    result = _seed(pg_conn, tmp_path, richer, "seed-richer")

    assert _chat(pg_conn, CHAT_UNNAMED) == ("group", "Book Club", "imessage")
    assert _attachment(pg_conn, ATT_BARE) == replace(
        _BARE, filename="notes.pdf", uti="com.adobe.pdf", mime_type="application/pdf",
        byte_size=2048,
    )
    assert _message(pg_conn, MSG_BARE)["text_original"] == "found the notes"

    assert (result.chat_upserts.filled, result.chat_upserts.unchanged) == (1, 1)
    assert (result.attachment_upserts.filled, result.attachment_upserts.unchanged) == (1, 1)
    assert (result.message_upserts.filled, result.message_upserts.unchanged) == (1, 3)
    _assert_no_replacement(result)


def test_a_seed_true_fills_a_stored_false_flag(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """A `POSITIVE` flag's `false` is what a source says when it cannot
    see the thing, so a seed's `true` adds evidence: a fill."""
    stored = replace(World(), photo_linked=False)
    _establish(pg_conn, tmp_path, stored)
    assert _message(pg_conn, MSG_PHOTO)["has_attachments"] is False

    seed = replace(stored, photo_linked=True, text_unsent=True,
                   bare=replace(_BARE, is_sticker=True))
    result = _seed(pg_conn, tmp_path, seed, "seed-flags")

    assert _message(pg_conn, MSG_PHOTO)["has_attachments"] is True
    assert _message(pg_conn, MSG_TEXT)["is_unsent"] is True
    assert _attachment(pg_conn, ATT_BARE).is_sticker is True
    assert result.message_upserts.filled == 2
    assert result.attachment_upserts.filled == 1
    assert result.attachment_upserts.inserted == 1, "the photo was never linked before"
    _assert_no_replacement(result)


# --- 3. the live run still applies genuine changes ---------------------------


def test_the_live_run_applies_a_rename_and_a_moved_attachment(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    _establish(pg_conn, tmp_path, World())

    live = replace(
        World(),
        named_group_name="Weekend Plans (October)",
        photo=replace(_PHOTO, source_path="~/Library/Messages/Attachments/cc/03/IMG_0001.jpeg"),
    )
    result = _live_again(pg_conn, tmp_path, live)

    assert _chat(pg_conn, CHAT_NAMED)[1] == "Weekend Plans (October)"
    assert _attachment(pg_conn, ATT_PHOTO).source_path == (
        "~/Library/Messages/Attachments/cc/03/IMG_0001.jpeg"
    )
    assert result.merge_mode is MergeMode.LIVE
    assert result.chat_upserts.replaced == 1
    assert result.attachment_upserts.replaced == 1


def test_the_live_run_never_blanks_a_value_with_an_empty_one(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Rule 2 holds for the live run too: `''` and 0 say nothing."""
    _establish(pg_conn, tmp_path, World())

    empties = replace(
        World(),
        named_group_name="",
        photo=Att(filename="", source_path="", uti="", mime_type="", byte_size=0),
    )
    result = _live_again(pg_conn, tmp_path, empties)

    assert _chat(pg_conn, CHAT_NAMED)[1] == "Weekend Plans"
    assert _attachment(pg_conn, ATT_PHOTO) == _PHOTO
    assert result.chat_upserts.updated == 0
    assert result.attachment_upserts.updated == 0


def test_the_live_run_blanks_no_body_with_an_empty_string(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """A retracted message comes back from the decoder with its text gone.
    The index keeps the text; `is_unsent` records the retraction."""
    _establish(pg_conn, tmp_path, World())

    _live_again(pg_conn, tmp_path, replace(World(), text_body="", text_edited_at=_EDIT_1,
                                           text_unsent=True))

    row = _message(pg_conn, MSG_TEXT)
    assert row["text_original"] == "saturday at ten"
    assert row["is_unsent"] is True


# --- 4. edits: a strictly newer edit wins, and history only grows -----------


def test_a_strictly_newer_edit_replaces_the_body_even_from_a_seed(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """D12 rule 4: text follows edit recency from any source. The replaced
    body is still in the index, as the version the edit displaced."""
    _establish(pg_conn, tmp_path, World(text_body="saturday at nine"))

    edited = World(
        text_body="saturday at ten", text_edited_at=_EDIT_1,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),),
    )
    result = _seed(pg_conn, tmp_path, edited, "seed-newer-edit")

    row = _message(pg_conn, MSG_TEXT)
    assert (row["text_original"], row["date_edited"], row["is_edited"]) == (
        "saturday at ten", _EDIT_1, True,
    )
    assert _versions(pg_conn, MSG_TEXT) == [(0, "saturday at nine")]
    assert result.message_upserts.newer_edit == 1
    _assert_no_replacement(result)


def test_the_body_a_newer_edit_displaces_is_kept_when_the_source_lacks_that_history(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Append, not replace: the text a newer edit displaces stays in
    `message_version` even when the newer source's own edit history does
    not carry it."""
    _establish(
        pg_conn, tmp_path,
        World(text_body="saturday at ten", text_edited_at=_EDIT_1,
              text_history=(EditVersion(text="saturday at nine", edited_at=None),)),
    )

    newer = World(text_body="sunday instead", text_edited_at=_EDIT_2, text_history=())
    result = _seed(pg_conn, tmp_path, newer, "seed-no-history")

    assert _message(pg_conn, MSG_TEXT)["text_original"] == "sunday instead"
    assert _versions(pg_conn, MSG_TEXT) == [(0, "saturday at nine"), (1, "saturday at ten")]
    assert result.bodies_kept_as_history == 1


def test_a_same_time_edit_never_replaces_the_body_even_from_the_live_run(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    stored = World(
        text_body="saturday at ten", text_edited_at=_EDIT_1,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),),
    )
    _establish(pg_conn, tmp_path, stored)
    before = _message(pg_conn, MSG_TEXT)

    _live_again(pg_conn, tmp_path, replace(stored, text_body="saturday at 10"))

    assert _message(pg_conn, MSG_TEXT) == before


def test_an_older_edit_never_replaces_the_body_even_from_the_live_run(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    stored = World(
        text_body="sunday instead", text_edited_at=_EDIT_2,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),
                      EditVersion(text="saturday at ten", edited_at=None)),
    )
    _establish(pg_conn, tmp_path, stored)
    before = _message(pg_conn, MSG_TEXT)

    older = replace(stored, text_body="saturday at ten", text_edited_at=_EDIT_1,
                    text_history=(EditVersion(text="saturday at nine", edited_at=None),))
    _live_again(pg_conn, tmp_path, older)

    assert _message(pg_conn, MSG_TEXT) == before


def test_a_newer_edit_without_its_text_does_not_lock_that_text_out(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """If the live run learns of an edit but cannot decode the new body,
    the edit time must not move ahead of the body the index holds: a
    later source carrying that edit's text would then look same-time and
    be refused for good."""
    stored = World(
        text_body="saturday at ten", text_edited_at=_EDIT_1,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),),
    )
    _establish(pg_conn, tmp_path, stored)

    _live_again(pg_conn, tmp_path, replace(stored, text_body=None, text_edited_at=_EDIT_2))
    row = _message(pg_conn, MSG_TEXT)
    assert (row["text_original"], row["date_edited"]) == ("saturday at ten", _EDIT_1)

    with_text = replace(
        stored, text_body="sunday instead", text_edited_at=_EDIT_2,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),
                      EditVersion(text="saturday at ten", edited_at=None)),
    )
    _seed(pg_conn, tmp_path, with_text, "seed-carries-the-edit")
    row = _message(pg_conn, MSG_TEXT)
    assert (row["text_original"], row["date_edited"]) == ("sunday instead", _EDIT_2)


def test_edit_history_only_grows_even_from_the_live_run(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """A stored prior version never changes: a different text at the same
    position is a different decode, not a correction. An undecodable
    version (stored `''`) is still filled when a source decodes it."""
    stored = World(
        text_body="sunday instead", text_edited_at=_EDIT_2,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),
                      EditVersion(text=None, edited_at=None)),
    )
    _establish(pg_conn, tmp_path, stored)
    assert _versions(pg_conn, MSG_TEXT) == [(0, "saturday at nine"), (1, "")]

    other_decode = replace(
        stored,
        text_history=(EditVersion(text="saturday at 9", edited_at=None),
                      EditVersion(text="saturday at ten", edited_at=None)),
    )
    result = _live_again(pg_conn, tmp_path, other_decode)

    assert _versions(pg_conn, MSG_TEXT) == [(0, "saturday at nine"), (1, "saturday at ten")]
    assert result.message_version_upserts.filled == 1
    assert result.message_version_upserts.replaced == 0


# --- 5. every table is counted, and fills are told apart ---------------------


def test_every_table_reports_inserts_then_nothing_on_a_replay(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    world = World(
        reaction=True, text_edited_at=_EDIT_1,
        text_history=(EditVersion(text="saturday at nine", edited_at=None),),
    )
    first = _establish(pg_conn, tmp_path, world)

    counts = first.table_counts()
    assert set(counts) >= {
        "chat", "source_handle", "attachment", "message", "tapback", "message_version",
        "link_preview", "message_attachment", "chat_participant_source", "message_source",
        "attachment_source",
    }
    assert {t: c.inserted for t, c in counts.items()} == {
        "chat": 2, "source_handle": 2, "attachment": 2, "message": 4, "tapback": 1,
        "message_version": 1, "link_preview": 1, "message_attachment": 2,
        "chat_participant_source": 4, "message_source": 4, "attachment_source": 2,
    }

    replay = _seed(pg_conn, tmp_path, world, "seed-replay")
    for table, c in replay.table_counts().items():
        if table in ("message_source", "attachment_source"):
            # A new source name records its own provenance rows.
            continue
        assert (c.inserted, c.updated) == (0, 0), table


def test_the_run_row_records_the_mode_and_every_table(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Durable, not just printed: a real seed run can be audited after the
    terminal is gone."""
    live = _establish(pg_conn, tmp_path, World())
    seed = _seed(pg_conn, tmp_path, replace(World(), unnamed_group_name="Book Club"), "seed-a")

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT run_id, merge_mode, upsert_counts FROM extraction_run ORDER BY run_id"
        )
        rows = {int(r[0]): (r[1], r[2]) for r in cur.fetchall()}
    assert rows[live.run_id][0] == "live"
    mode, counts = rows[seed.run_id]
    assert mode == "seed"
    counts = counts if isinstance(counts, dict) else json.loads(counts)
    assert counts["chat"]["filled"] == 1
    assert counts["chat"]["replaced"] == 0
    assert set(counts) == set(seed.table_counts())


# --- 6. downstream invalidation ----------------------------------------------


@pytest.mark.parametrize(
    ("change", "dirty_chat"),
    [
        pytest.param({"bare_body": "found the notes"}, CHAT_NAMED, id="message-body-fill"),
        pytest.param(
            {"bare": replace(_BARE, filename="notes.pdf", mime_type="application/pdf")},
            CHAT_NAMED,
            id="rendered-attachment-metadata-fill",
        ),
        pytest.param({"unnamed_group_name": "Book Club"}, CHAT_UNNAMED, id="chat-name-fill"),
        pytest.param({"reaction": True}, CHAT_NAMED, id="new-tapback-on-a-segmented-message"),
        pytest.param({"photo_also_on_bare": True}, CHAT_NAMED, id="new-link-on-a-message-with-one"),
    ],
)
def test_a_fill_that_changes_what_a_segment_renders_marks_its_chat(
    pg_conn: psycopg.Connection,
    tmp_path: Path,
    config: Config,
    change: dict[str, Any],
    dirty_chat: str,
) -> None:
    """S4 re-segments a chat only when one of its messages' `updated_at`
    moves. A fill on the message row does that by itself; a fill on a
    chat, attachment, link or tapback -- tables the segment renders
    through the message -- must move it too, or the segment keeps
    showing the value the fill replaced."""
    _establish(pg_conn, tmp_path, World())
    _settle(pg_conn, config)

    result = _seed(pg_conn, tmp_path, replace(World(), **change), "seed-fill")

    assert _dirty(pg_conn, config) == {_chat_ids(pg_conn)[dirty_chat]}
    _assert_no_replacement(result)


def test_a_live_rename_marks_the_whole_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    """Every segment header carries a group's name, and `chat` has no
    `updated_at` of its own, so a rename has to move its messages'."""
    _establish(pg_conn, tmp_path, World())
    _settle(pg_conn, config)

    _live_again(pg_conn, tmp_path, replace(World(), named_group_name="Weekend Plans (October)"))

    assert _dirty(pg_conn, config) == {_chat_ids(pg_conn)[CHAT_NAMED]}


def test_an_edit_history_fill_marks_its_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    stored = World(
        text_body="saturday at ten", text_edited_at=_EDIT_1,
        text_history=(EditVersion(text=None, edited_at=None),),
    )
    _establish(pg_conn, tmp_path, stored)
    _settle(pg_conn, config)

    decoded = replace(stored, text_history=(EditVersion(text="saturday at nine", edited_at=None),))
    _seed(pg_conn, tmp_path, decoded, "seed-decoded-history")

    assert _versions(pg_conn, MSG_TEXT) == [(0, "saturday at nine")]
    assert _dirty(pg_conn, config) == {_chat_ids(pg_conn)[CHAT_NAMED]}


def test_a_seed_that_brings_nothing_new_marks_no_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    """The recovery candidate's shape: older bodies, other paths and
    services, empties, sub-second `sent_at` differences. All of it is
    refused, so none of it may cost a re-segmentation either."""
    _establish(pg_conn, tmp_path, World())
    _settle(pg_conn, config)

    older = replace(
        World(),
        named_group_name="",
        named_group_service="SMS",
        text_body="saturday at nine",
        text_sent_at=_BASE + timedelta(milliseconds=400),
        photo=Att(filename="IMG_0001 2.jpeg",
                  source_path="/Users/example/Library/Messages/Attachments/aa/01/IMG_0001.jpeg",
                  uti="public.heic", mime_type="image/heic", byte_size=0),
    )
    result = _seed(pg_conn, tmp_path, older, "seed-older")

    assert _dirty(pg_conn, config) == set()
    assert result.messages_marked_for_resegmentation == 0
    for table, counts in result.table_counts().items():
        assert counts.updated == 0, table


def test_a_fill_the_segments_do_not_render_marks_no_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    """`uti` and `byte_size` are not rendered; filling them is worth
    keeping but not worth re-segmenting for."""
    _establish(pg_conn, tmp_path, World())
    _settle(pg_conn, config)

    result = _seed(
        pg_conn, tmp_path, replace(World(), bare=replace(_BARE, uti="com.adobe.pdf", byte_size=2048)),
        "seed-unrendered",
    )

    assert result.attachment_upserts.filled == 1
    assert _dirty(pg_conn, config) == set()
