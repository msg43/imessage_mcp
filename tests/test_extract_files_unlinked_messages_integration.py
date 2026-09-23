"""Owner decision D13 (2026-09-23): every message `chat.db` holds goes into
the corpus, including the rows it keeps with no `chat_message_join` link.

"I'd rather keep all the messages in the active corpus." Extraction used to
log such a row (`extract.message_without_chat`) and drop it. The rules under
test (`imsg.stages.unlinked_filing`):

  1. Apple's "Recently Deleted" (`chat_recoverable_message_join`) names the
     chat: file it there and keep the delete date (`deleted_at`).
  2. A 1:1-style `ck_chat_id` whose handle is the sender, or the owner sent
     it: that 1:1 chat, created under Apple's own GUID when the index lacks
     it.
  3. A group-style id that exactly one indexed group carries: that group.
  4. Otherwise a holding chat per lost group id.
  5. No evidence: a holding chat per sender (and one for the owner).

A message moves only toward stronger evidence and never out of a real chat.
Rows below a source's ROWID watermark are re-read once, when they still need
filing.

Every test here also runs against the code from before D13 (core 14ee55c,
migrations through 0006) and fails there: the old extractor never files an
unlinked row, so each test fails at its first "the message is in this chat"
check, before it reads a column migration 0007 adds.

Fictional personas only (D5): Alice Example, Bob Builder, Carol Carpenter,
Dave Drywall, and documentation-range phone numbers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import find_dirty_chats, run_segment, run_segment_for_chat
from imsg.stages.extract import ExtractResult, MergeMode, run_extract
from imsg.stages.identity import ContactRecord, run_identity
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun, TapbackInfo

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_unlinked_filing_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
PROMPT_BYTES = b"fixed boundary-detection prompt for tests"


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


pytestmark = pytest.mark.skipif(
    not _admin_reachable(),
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


# --- the cast ------------------------------------------------------------------

ALICE = "+15550000001"
BOB = "+15550000002"
CAROL = "+15550000003"
DAVE = "+15550000004"

CHAT_ALICE = f"iMessage;-;{ALICE}"
CHAT_DAVE = f"iMessage;-;{DAVE}"
CHAT_CLUB = "iMessage;+;chat-club"
CHAT_TWIN_A = "iMessage;+;chat-twin-a"
CHAT_TWIN_B = "iMessage;+;chat-twin-b"
CHAT_LATE = "iMessage;+;chat-late"
CAROL_SMS = f"SMS;-;{CAROL}"
"""A 1:1 chat no snapshot has: created from a `ck_chat_id`."""

GROUP_CLUB = "group-id-club"
GROUP_TWIN = "group-id-twin"
"""Carried by two group chats: ambiguous, so it matches neither."""
GROUP_GONE = "group-id-gone"
"""Carried by no chat anywhere."""
GROUP_DAVE = "group-id-dave"
"""Carried only by a 1:1 chat, which is not a group."""
GROUP_LATE = "group-id-late"
"""Carried by a group only a later source shows."""

HOLD_BOB = "unfiled:sender:" + BOB
HOLD_OWNER = "unfiled:owner"
HOLD_TWIN = "unfiled:lost-group:" + GROUP_TWIN
HOLD_GONE = "unfiled:lost-group:" + GROUP_GONE
HOLD_DAVE = "unfiled:lost-group:" + GROUP_DAVE
HOLD_LATE = "unfiled:lost-group:" + GROUP_LATE

_T0 = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
_DELETED_AT = datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)

CHATS: tuple[FixtureChat, ...] = (
    FixtureChat(guid=CHAT_ALICE, rowid=1, style=45),
    FixtureChat(guid=CHAT_CLUB, rowid=2, style=43, display_name="Book Club", group_id=GROUP_CLUB),
    FixtureChat(guid=CHAT_TWIN_A, rowid=3, style=43, group_id=GROUP_TWIN),
    FixtureChat(guid=CHAT_TWIN_B, rowid=4, style=43, original_group_id=GROUP_TWIN),
    FixtureChat(guid=CHAT_DAVE, rowid=5, style=45, group_id=GROUP_DAVE),
)
PARTICIPANTS: tuple[tuple[str, str], ...] = (
    (CHAT_ALICE, ALICE),
    (CHAT_CLUB, ALICE),
    (CHAT_CLUB, BOB),
    (CHAT_TWIN_A, ALICE),
    (CHAT_TWIN_B, BOB),
    (CHAT_DAVE, DAVE),
)


@dataclass(frozen=True)
class Row:
    """One `message` row of a fixture snapshot."""

    guid: str
    rowid: int
    chat: str | None = None
    """The chat `chat_message_join` links it to; None for no link."""
    sender: str | None = None
    """The sender's handle; None for the owner's own message."""
    ck: str | None = None
    recoverable: str | None = None
    deleted: datetime | None = None
    minute: int = 0
    dated: bool = True
    item_type: int = 0
    tapback_on: str | None = None
    """This row is a reaction to that message (the decoder says so)."""


# The world every rule is exercised in, one row per case.
LINKED = Row("msg-linked", 1, chat=CHAT_ALICE, sender=ALICE, minute=0)
DELETED = Row("msg-deleted", 2, sender=ALICE, recoverable=CHAT_ALICE, deleted=_DELETED_AT, minute=1)
CK_ALICE = Row("msg-ck-alice", 3, sender=ALICE, ck=f"iMessage;-;{ALICE}", minute=2)
CK_CAROL = Row("msg-ck-carol", 4, ck=CAROL_SMS, minute=3)
CK_WRONG = Row("msg-ck-wrong-sender", 5, sender=BOB, ck=f"iMessage;-;{ALICE}", minute=4)
CLUB = Row("msg-club", 6, sender=BOB, ck=GROUP_CLUB, minute=5)
TWIN = Row("msg-twin", 7, sender=ALICE, ck=GROUP_TWIN, minute=6)
GONE_A = Row("msg-gone-alice", 8, sender=ALICE, ck=GROUP_GONE, minute=7)
GONE_B = Row("msg-gone-bob", 9, sender=BOB, ck=GROUP_GONE, minute=8)
DAVE_ID = Row("msg-dave-group-id", 10, sender=DAVE, ck=GROUP_DAVE, minute=9)
BARE_BOB = Row("msg-bare-bob", 11, sender=BOB, minute=10)
BARE_ME = Row("msg-bare-owner", 12, minute=11)
UNDATED = Row("msg-undated", 13, sender=BOB, dated=False)
SYSTEM = Row("msg-system", 14, sender=BOB, item_type=2, minute=12)
REACTION = Row("msg-reaction", 15, sender=BOB, tapback_on=LINKED.guid, minute=13)

WORLD: tuple[Row, ...] = (
    LINKED, DELETED, CK_ALICE, CK_CAROL, CK_WRONG, CLUB, TWIN, GONE_A, GONE_B,
    DAVE_ID, BARE_BOB, BARE_ME, UNDATED, SYSTEM, REACTION,
)


def _body(guid: str) -> str:
    return f"body of {guid}"


def _build(path: Path, chats: Sequence[FixtureChat], rows: Sequence[Row]) -> Path:
    builder = ChatDbBuilder()
    for chat in chats:
        builder.add_chat(replace(chat))
    for i, raw in enumerate(sorted({ALICE, BOB, CAROL, DAVE}), start=1):
        builder.add_handle(FixtureHandle(raw_value=raw, rowid=i))
    chat_guids = {chat.guid for chat in chats}
    for chat_guid, raw in PARTICIPANTS:
        if chat_guid in chat_guids:
            builder.link_participant(chat_guid, raw)
    for row in rows:
        builder.add_message(
            FixtureMessage(
                guid=row.guid,
                chat_guid=row.chat,
                rowid=row.rowid,
                is_from_me=row.sender is None,
                handle_raw_value=row.sender,
                date=_T0 + timedelta(minutes=row.minute) if row.dated else None,
                item_type=row.item_type,
                ck_chat_id=row.ck,
                associated_message_type=2000 if row.tapback_on else 0,
                recoverable_chat_guid=row.recoverable,
                delete_date=row.deleted,
            )
        )
    return builder.build(path)


def _dump(rows: Sequence[Row], since_rowid: int) -> ImsgDumpRun:
    """What `imsg-dump --since-rowid` would print: every row above the
    cursor, decoded."""
    return ImsgDumpRun(
        messages=tuple(
            ImsgDumpMessage(
                rowid=row.rowid, guid=row.guid, chat_guid=None, handle=None,
                is_from_me=row.sender is None, date=None, date_edited=None,
                date_retracted=None, service="iMessage",
                body_text=None if row.tapback_on else _body(row.guid),
                edit_history=(), is_unsent=False,
                tapback=(
                    TapbackInfo(kind="loved", target_guid=row.tapback_on, emoji=None, action="added")
                    if row.tapback_on
                    else None
                ),
                attachment_rowids=(), reply_to_guid=None,
            )
            for row in rows
            if row.rowid > since_rowid
        ),
        stderr_lines=(),
    )


_RUNS = iter(range(1, 10_000))


@dataclass
class Run:
    result: ExtractResult
    dump_cursors: list[int] = field(default_factory=list)
    """The `--since-rowid` value the run handed the decoder."""


def _extract(
    conn: psycopg.Connection,
    tmp_path: Path,
    rows: Sequence[Row],
    *,
    source: str,
    mode: MergeMode = MergeMode.SEED,
    chats: Sequence[FixtureChat] = CHATS,
) -> Run:
    snapshot = _build(tmp_path / f"{source}-{next(_RUNS)}.db", chats, rows)
    binary = tmp_path / "imsg-dump"
    binary.write_text("")
    cursors: list[int] = []

    def fake_dump(_binary: Path, _snapshot: Path, since_rowid: int) -> ImsgDumpRun:
        cursors.append(since_rowid)
        return _dump(rows, since_rowid)

    result = run_extract(
        conn=conn,
        source_name=source,
        snapshot_path=snapshot,
        imsg_dump_binary=binary,
        run_imsg_dump_fn=fake_dump,
        merge_mode=mode,
    )
    return Run(result, cursors)


# --- probes (the first ones read only columns that predate D13) -----------------


def _chat_of(conn: psycopg.Connection, guid: str) -> str | None:
    """The `chat.source_guid` the message is filed in, or None when the
    index does not have the message."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.source_guid FROM message m JOIN chat c USING (chat_id) "
            "WHERE m.source_guid = %s",
            (guid,),
        )
        row = cur.fetchone()
    return None if row is None else str(row[0])


def _evidence_of(conn: psycopg.Connection, guid: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT chat_evidence FROM message WHERE source_guid = %s", (guid,))
        row = cur.fetchone()
    assert row is not None, guid
    return str(row[0])


def _deleted_at(conn: psycopg.Connection, guid: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute("SELECT deleted_at FROM message WHERE source_guid = %s", (guid,))
        row = cur.fetchone()
    assert row is not None, guid
    value: datetime | None = row[0]
    return value


def _chat_row(conn: psycopg.Connection, guid: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind::text, display_name, service::text, unfiled_key FROM chat "
            "WHERE source_guid = %s",
            (guid,),
        )
        row = cur.fetchone()
    assert row is not None, guid
    return dict(zip(("kind", "display_name", "service", "unfiled_key"), row, strict=True))


def _raw_participants(conn: psycopg.Connection, chat_guid: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sh.raw_value FROM chat_participant_source cps
            JOIN chat c USING (chat_id)
            JOIN source_handle sh USING (source_handle_id)
            WHERE c.source_guid = %s
            """,
            (chat_guid,),
        )
        return {str(r[0]) for r in cur.fetchall()}


def _count(conn: psycopg.Connection, sql: str, *params: Any) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _no_contacts(default_region: str) -> list[ContactRecord]:
    return []


def _identify(conn: psycopg.Connection, config: Config) -> None:
    run_identity(conn=conn, config=config, contacts_importer=_no_contacts)


def _segment_texts(conn: psycopg.Connection, chat_guid: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.rendered_text FROM segment s JOIN chat c USING (chat_id) "
            "WHERE c.source_guid = %s ORDER BY s.started_at",
            (chat_guid,),
        )
        return [str(r[0]) for r in cur.fetchall()]


def _segment_chat_of(conn: psycopg.Connection, guid: str) -> list[str]:
    """Which chats' segments hold the message (at most one, by schema)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.source_guid FROM segment_message sm
            JOIN segment s USING (segment_id)
            JOIN chat c ON c.chat_id = s.chat_id
            JOIN message m ON m.message_id = sm.message_id
            WHERE m.source_guid = %s
            """,
            (guid,),
        )
        return [str(r[0]) for r in cur.fetchall()]


def _assert_no_replacement(result: ExtractResult) -> None:
    for table, counts in result.table_counts().items():
        assert counts.replaced == 0, f"{table}: a seed replaced a non-empty value"


# --- 1. each rule files the message where its evidence says -------------------


@pytest.mark.parametrize(
    ("row", "chat", "evidence"),
    [
        pytest.param(LINKED, CHAT_ALICE, "chat_message_join", id="chat-link"),
        pytest.param(DELETED, CHAT_ALICE, "recoverable_join", id="recently-deleted"),
        pytest.param(CK_ALICE, CHAT_ALICE, "ck_1to1", id="1to1-id-matches-sender"),
        pytest.param(CK_CAROL, CAROL_SMS, "ck_1to1", id="1to1-id-owner-sent-chat-created"),
        pytest.param(CK_WRONG, HOLD_BOB, "holding_sender", id="1to1-id-names-someone-else"),
        pytest.param(CLUB, CHAT_CLUB, "ck_group_match", id="group-id-one-indexed-group"),
        pytest.param(TWIN, HOLD_TWIN, "holding_lost_group", id="group-id-two-chats"),
        pytest.param(GONE_A, HOLD_GONE, "holding_lost_group", id="group-id-no-chat"),
        pytest.param(DAVE_ID, HOLD_DAVE, "holding_lost_group", id="group-id-on-a-1to1-chat"),
        pytest.param(BARE_BOB, HOLD_BOB, "holding_sender", id="no-evidence-incoming"),
        pytest.param(BARE_ME, HOLD_OWNER, "holding_sender", id="no-evidence-owner"),
    ],
)
def test_each_rule_files_the_message_where_its_evidence_says(
    pg_conn: psycopg.Connection, tmp_path: Path, row: Row, chat: str, evidence: str
) -> None:
    _extract(pg_conn, tmp_path, WORLD, source="mini", mode=MergeMode.LIVE)

    assert _chat_of(pg_conn, row.guid) == chat
    assert _evidence_of(pg_conn, row.guid) == evidence


def test_rows_that_cannot_or_need_not_be_filed_keep_their_old_handling(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """System rows are skipped and reactions fold into `tapback`, linked
    or not. An undated row cannot be filed (`sent_at` is NOT NULL): it is
    counted, and the run does not fail on it."""
    run = _extract(pg_conn, tmp_path, WORLD, source="mini", mode=MergeMode.LIVE)

    assert _chat_of(pg_conn, CLUB.guid) == CHAT_CLUB  # the run filed unlinked rows at all
    assert _chat_of(pg_conn, SYSTEM.guid) is None
    assert _chat_of(pg_conn, UNDATED.guid) is None
    assert _count(pg_conn, "SELECT count(*) FROM tapback WHERE source_guid = %s", REACTION.guid) == 1
    assert run.result.system_messages_skipped == 1
    assert run.result.unlinked.skipped_without_date == 1


def test_holding_chats_are_marked_named_and_hold_their_senders(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    _extract(pg_conn, tmp_path, WORLD, source="mini", mode=MergeMode.LIVE)

    assert _chat_of(pg_conn, GONE_B.guid) == HOLD_GONE  # one holding chat per lost group
    assert _chat_row(pg_conn, HOLD_GONE) == {
        "kind": "group",
        "display_name": "Unfiled: group no longer in Messages",
        "service": "unknown",
        "unfiled_key": "lost-group:" + GROUP_GONE,
    }
    assert _raw_participants(pg_conn, HOLD_GONE) == {ALICE, BOB}

    assert _chat_row(pg_conn, HOLD_BOB)["unfiled_key"] == "sender:" + BOB
    assert _chat_row(pg_conn, HOLD_BOB)["kind"] == "dm"
    assert _raw_participants(pg_conn, HOLD_BOB) == {BOB}

    assert _chat_row(pg_conn, HOLD_OWNER)["unfiled_key"] == "owner"
    assert _raw_participants(pg_conn, HOLD_OWNER) == set()

    # Real chats are never marked.
    assert _count(
        pg_conn, "SELECT count(*) FROM chat WHERE unfiled_key IS NOT NULL "
        "AND source_guid NOT LIKE 'unfiled:%%'"
    ) == 0


def test_a_1to1_chat_the_index_lacks_is_created_under_apples_guid(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    _extract(pg_conn, tmp_path, WORLD, source="mini", mode=MergeMode.LIVE)

    assert _chat_of(pg_conn, CK_CAROL.guid) == CAROL_SMS
    assert _chat_row(pg_conn, CAROL_SMS) == {
        "kind": "dm", "display_name": None, "service": "sms", "unfiled_key": None,
    }
    assert _raw_participants(pg_conn, CAROL_SMS) == {CAROL}
    # An existing 1:1 chat gets no participant from a ck id.
    assert _raw_participants(pg_conn, CHAT_ALICE) == {ALICE}


def test_a_deleted_message_keeps_its_delete_date(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    run = _extract(pg_conn, tmp_path, WORLD, source="mini", mode=MergeMode.LIVE)

    assert _chat_of(pg_conn, DELETED.guid) == CHAT_ALICE
    assert _deleted_at(pg_conn, DELETED.guid) == _DELETED_AT
    assert _deleted_at(pg_conn, LINKED.guid) is None
    assert run.result.unlinked.in_recently_deleted == 1


def test_the_run_reports_how_each_message_was_filed(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    run = _extract(pg_conn, tmp_path, WORLD, source="mini", mode=MergeMode.LIVE)
    u = run.result.unlinked

    assert _chat_of(pg_conn, TWIN.guid) == HOLD_TWIN
    assert (u.recoverable_join, u.ck_1to1, u.ck_group_match) == (1, 2, 1)
    assert (u.holding_lost_group, u.holding_sender) == (4, 3)
    assert (u.chats_created, u.holding_chats_created) == (1, 5)
    assert (u.rescanned, u.moved_from_holding, u.evidence_raised) == (0, 0, 0)
    assert run.result.chat_group_ids.inserted == 4


def test_a_dry_run_reports_the_filing_and_writes_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """`imsg extract --dry-run` is how the counts are checked before a seed
    runs for real: the same numbers, and no chat or message left behind."""
    snapshot = _build(tmp_path / "dry.db", CHATS, WORLD)
    binary = tmp_path / "imsg-dump"
    binary.write_text("")
    dry = run_extract(
        conn=pg_conn, source_name="seed-dry", snapshot_path=snapshot, imsg_dump_binary=binary,
        run_imsg_dump_fn=lambda _b, _s, since: _dump(WORLD, since), dry_run=True,
    )

    assert dry.dry_run
    assert (dry.unlinked.holding_lost_group, dry.unlinked.holding_chats_created) == (4, 5)
    assert _count(pg_conn, "SELECT count(*) FROM message") == 0
    assert _count(pg_conn, "SELECT count(*) FROM chat") == 0

    real = _extract(pg_conn, tmp_path, WORLD, source="seed-dry")
    assert real.result.unlinked == dry.unlinked


# --- 2. rows below the watermark are re-read once ------------------------------


def test_rows_below_the_watermark_are_rescanned_once_and_then_left_alone(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Every unlinked row sits below its source's watermark: a seed's
    watermark reached the end of its file on its first run, before D13.
    The next run must reach back for them, and the one after must not.

    The earlier run is recorded as it would be: a watermark at the end of
    the file and a successful run row. (Without the run row the extractor
    treats the source as new and re-reads every row anyway.)"""
    pg_conn.execute(
        "INSERT INTO sync_state (key, value) VALUES ('watermark.rowid.seed-old', %s)",
        (str(max(row.rowid for row in WORLD)),),
    )
    pg_conn.execute(
        "INSERT INTO extraction_run (source_name, snapshot_path, snapshot_sha256, status) "
        "VALUES ('seed-old', 'earlier.db', 'earlier', 'ok')"
    )

    first = _extract(pg_conn, tmp_path, WORLD, source="seed-old")

    assert _chat_of(pg_conn, GONE_A.guid) == HOLD_GONE
    assert _chat_of(pg_conn, DELETED.guid) == CHAT_ALICE
    # The linked row, the undated row, the system row and the reaction are
    # not candidates; the eleven fileable unlinked rows are.
    assert first.result.unlinked.rescanned == 11
    assert first.dump_cursors == [DELETED.rowid - 1]

    second = _extract(pg_conn, tmp_path, WORLD, source="seed-old")

    assert second.result.unlinked.rescanned == 0
    assert second.dump_cursors == [max(row.rowid for row in WORLD)]
    assert second.result.message_upserts.total == 0


def test_the_rescan_fills_a_delete_date_the_index_lacks(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """A message the index filed from its chat link, later moved to
    "Recently Deleted" on the live Mac: its ROWID is long below the
    watermark, and the run still labels it."""
    live = replace(LINKED, chat=CHAT_ALICE)
    _extract(pg_conn, tmp_path, (live,), source="mini", mode=MergeMode.LIVE)
    assert _chat_of(pg_conn, LINKED.guid) == CHAT_ALICE

    deleted_later = replace(LINKED, chat=None, recoverable=CHAT_ALICE, deleted=_DELETED_AT)
    run = _extract(pg_conn, tmp_path, (deleted_later,), source="mini", mode=MergeMode.LIVE)

    assert run.result.unlinked.rescanned == 1
    assert _deleted_at(pg_conn, LINKED.guid) == _DELETED_AT
    assert _chat_of(pg_conn, LINKED.guid) == CHAT_ALICE
    assert _evidence_of(pg_conn, LINKED.guid) == "chat_message_join"


def test_a_delete_date_is_never_cleared(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    deleted = replace(LINKED, chat=None, recoverable=CHAT_ALICE, deleted=_DELETED_AT)
    _extract(pg_conn, tmp_path, (deleted,), source="seed-a")
    assert _chat_of(pg_conn, LINKED.guid) == CHAT_ALICE

    pg_conn.execute("DELETE FROM sync_state WHERE key = 'watermark.rowid.mini'")
    _extract(pg_conn, tmp_path, (LINKED,), source="mini", mode=MergeMode.LIVE)

    assert _deleted_at(pg_conn, LINKED.guid) == _DELETED_AT


# --- 3. a message moves only toward stronger evidence --------------------------


def test_a_message_never_moves_from_one_real_chat_to_another(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Two witnesses disagree about a message's chat. Whichever real chat
    it was filed in first keeps it: a group id (~3% misfiling) does not
    pull a linked message away, and a chat link does not pull a message
    away from a group its id named."""
    linked_first = Row("msg-linked-first", 1, chat=CHAT_ALICE, sender=ALICE)
    guessed_first = Row("msg-guessed-first", 2, sender=BOB, ck=GROUP_CLUB, minute=1)
    _extract(pg_conn, tmp_path, (linked_first, guessed_first), source="seed-a")
    assert _chat_of(pg_conn, guessed_first.guid) == CHAT_CLUB

    run = _extract(
        pg_conn, tmp_path,
        (
            replace(linked_first, chat=None, ck=GROUP_CLUB),
            replace(guessed_first, chat=CHAT_ALICE, ck=None),
        ),
        source="seed-b",
    )

    assert _chat_of(pg_conn, linked_first.guid) == CHAT_ALICE
    assert _evidence_of(pg_conn, linked_first.guid) == "chat_message_join"
    assert _chat_of(pg_conn, guessed_first.guid) == CHAT_CLUB
    assert _evidence_of(pg_conn, guessed_first.guid) == "ck_group_match"
    assert (run.result.unlinked.moved_from_holding, run.result.unlinked.evidence_raised) == (0, 0)


def test_a_later_source_that_links_the_message_moves_it_out_of_its_holding_chat(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    unfiled = Row("msg-later-linked", 1, sender=BOB)
    _extract(pg_conn, tmp_path, (unfiled,), source="seed-a")
    assert _chat_of(pg_conn, unfiled.guid) == HOLD_BOB

    run = _extract(pg_conn, tmp_path, (replace(unfiled, chat=CHAT_CLUB),), source="seed-b")

    assert _chat_of(pg_conn, unfiled.guid) == CHAT_CLUB
    assert _evidence_of(pg_conn, unfiled.guid) == "chat_message_join"
    assert run.result.unlinked.moved_from_holding == 1
    _assert_no_replacement(run.result)  # a move is filing, not a merge replacement


def test_a_group_a_later_source_shows_collects_its_lost_messages(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The group id is recorded from the source that has the chat
    (`chat_group_id`); re-running the source that has the message then
    finds exactly one indexed group and moves the message into it."""
    lost = Row("msg-late-group", 1, sender=ALICE, ck=GROUP_LATE)
    _extract(pg_conn, tmp_path, (lost,), source="seed-a")
    assert _chat_of(pg_conn, lost.guid) == HOLD_LATE

    late_chat = FixtureChat(guid=CHAT_LATE, rowid=9, style=43, group_id=GROUP_LATE)
    _extract(pg_conn, tmp_path, (), source="seed-b", chats=(*CHATS, late_chat))
    rerun = _extract(pg_conn, tmp_path, (lost,), source="seed-a")

    assert _chat_of(pg_conn, lost.guid) == CHAT_LATE
    assert _evidence_of(pg_conn, lost.guid) == "ck_group_match"
    assert rerun.result.unlinked.rescanned == 1
    assert rerun.result.unlinked.moved_from_holding == 1


def test_stronger_evidence_for_the_same_chat_raises_the_recorded_evidence(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    guessed = Row("msg-confirmed", 1, sender=BOB, ck=GROUP_CLUB)
    _extract(pg_conn, tmp_path, (guessed,), source="seed-a")
    assert _chat_of(pg_conn, guessed.guid) == CHAT_CLUB

    run = _extract(pg_conn, tmp_path, (replace(guessed, chat=CHAT_CLUB),), source="seed-b")

    assert _chat_of(pg_conn, guessed.guid) == CHAT_CLUB
    assert _evidence_of(pg_conn, guessed.guid) == "chat_message_join"
    assert run.result.unlinked.evidence_raised == 1


def test_a_sender_holding_chat_gives_way_to_a_lost_group_holding_chat(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """Weakest to strongest holds between holding chats too, so the order
    the sources run in does not decide which holding chat a message ends
    up in."""
    bare = Row("msg-bare-then-group", 1, sender=BOB)
    _extract(pg_conn, tmp_path, (bare,), source="seed-a")
    assert _chat_of(pg_conn, bare.guid) == HOLD_BOB

    _extract(pg_conn, tmp_path, (replace(bare, ck=GROUP_GONE),), source="seed-b")

    assert _chat_of(pg_conn, bare.guid) == HOLD_GONE
    assert _evidence_of(pg_conn, bare.guid) == "holding_lost_group"


# --- 4. segmentation --------------------------------------------------------------


def test_filed_rows_segment_and_render_their_labels(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    _extract(pg_conn, tmp_path, WORLD, source="mini", mode=MergeMode.LIVE)
    assert _chat_of(pg_conn, DELETED.guid) == CHAT_ALICE
    _identify(pg_conn, config)
    run_segment(pg_conn, config, FakeBoundaryProvider(), PROMPT_BYTES)

    [alice_text] = _segment_texts(pg_conn, CHAT_ALICE)
    assert f"[deleted] {_body(DELETED.guid)}" in alice_text
    assert f"[deleted] {_body(LINKED.guid)}" not in alice_text
    [held] = _segment_texts(pg_conn, HOLD_BOB)
    assert held.splitlines()[0].endswith("(Unfiled: no chat recorded)")
    assert _body(BARE_BOB.guid) in held
    [lost] = _segment_texts(pg_conn, HOLD_GONE)
    assert '(group "Unfiled: group no longer in Messages")' in lost.splitlines()[0]
    assert find_dirty_chats(pg_conn, index_unsent=False) == {}


@pytest.mark.parametrize("only_the_destination", [False, True])
def test_a_moved_message_leaves_its_old_segment(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, only_the_destination: bool
) -> None:
    """`segment_message` allows one segment per message. The chat the
    message joins is re-segmented without tripping over the holding
    chat's segment -- also when it runs first, or alone -- and the
    holding chat is rebuilt without it."""
    moving = Row("msg-moving", 1, sender=BOB)
    staying = Row("msg-staying", 2, sender=BOB, minute=1)
    _extract(pg_conn, tmp_path, (moving, staying), source="seed-a")
    assert _chat_of(pg_conn, moving.guid) == HOLD_BOB
    _identify(pg_conn, config)
    run_segment(pg_conn, config, FakeBoundaryProvider(), PROMPT_BYTES)
    assert _segment_chat_of(pg_conn, moving.guid) == [HOLD_BOB]

    _extract(pg_conn, tmp_path, (replace(moving, chat=CHAT_CLUB),), source="seed-b")
    _identify(pg_conn, config)
    dirty = find_dirty_chats(pg_conn, index_unsent=False)
    chat_ids = {guid: cid for cid, guid in _chat_guids(pg_conn).items()}
    assert set(dirty) == {chat_ids[CHAT_CLUB], chat_ids[HOLD_BOB]}

    if only_the_destination:
        span = dirty[chat_ids[CHAT_CLUB]]
        run_segment_for_chat(
            pg_conn, chat_ids[CHAT_CLUB], config, FakeBoundaryProvider(), PROMPT_BYTES,
            earliest_changed_at=span.earliest_changed_at,
            latest_changed_at=span.latest_changed_at,
        )
        assert _segment_chat_of(pg_conn, moving.guid) == [CHAT_CLUB]
    run_segment(pg_conn, config, FakeBoundaryProvider(), PROMPT_BYTES)

    assert _segment_chat_of(pg_conn, moving.guid) == [CHAT_CLUB]
    assert _segment_chat_of(pg_conn, staying.guid) == [HOLD_BOB]
    [held] = _segment_texts(pg_conn, HOLD_BOB)
    assert _body(moving.guid) not in held
    assert find_dirty_chats(pg_conn, index_unsent=False) == {}


def _chat_guids(conn: psycopg.Connection) -> dict[int, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT chat_id, source_guid FROM chat")
        return {int(cid): str(guid) for cid, guid in cur.fetchall()}
