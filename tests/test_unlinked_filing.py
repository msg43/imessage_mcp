"""Unit tests for D13's filing rules (`imsg.stages.unlinked_filing`), the
`[deleted]` and holding-chat rendering, and the snapshot reader on a
`chat.db` that predates the columns the rules read. No database.

Fictional personas only (D5).
"""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import apsw
import pytest

from imsg.segment.models import MessageForSegmentation, SegmentDraft
from imsg.segment.render import render_segment
from imsg.stages.extract import SnapshotReader
from imsg.stages.unlinked_filing import (
    EVIDENCE_ORDER,
    ChatChoice,
    ChatEvidence,
    GroupDirectory,
    IndexedFiling,
    UnlinkedMessage,
    choose_chat,
    refile_wanted,
    rescan_wanted,
)

ALICE = "+15550000001"
BOB = "+15550000002"
MIGRATION_0007 = Path(__file__).resolve().parents[1] / "migrations" / "0007_unlinked_messages.sql"


def _message(**overrides: object) -> UnlinkedMessage:
    fields: dict[str, object] = {
        "rowid": 1, "is_from_me": False, "sender": ALICE,
        "ck_chat_id": None, "recoverable_chat_guid": None,
    }
    fields.update(overrides)
    return UnlinkedMessage(**fields)  # type: ignore[arg-type]


def _groups(*entries: tuple[str, str, str | None]) -> GroupDirectory:
    directory = GroupDirectory()
    for group_id, chat_guid, kind in entries:
        directory.add(group_id, chat_guid, kind)
    return directory


# --- the rules, in order ------------------------------------------------------


def test_recently_deleted_wins_over_every_other_signal() -> None:
    choice = choose_chat(
        _message(recoverable_chat_guid="iMessage;+;chat-a", ck_chat_id="group-x"),
        _groups(("group-x", "iMessage;+;chat-b", "group")),
    )
    assert (choice.evidence, choice.chat_guid) == (ChatEvidence.RECOVERABLE_JOIN, "iMessage;+;chat-a")
    assert not choice.may_create_chat


@pytest.mark.parametrize(
    ("sender", "is_from_me", "expected"),
    [
        pytest.param(ALICE, False, ChatEvidence.CK_1TO1, id="sender-is-the-handle"),
        pytest.param(None, True, ChatEvidence.CK_1TO1, id="owner-sent-it"),
        pytest.param(BOB, False, ChatEvidence.HOLDING_SENDER, id="someone-else-sent-it"),
    ],
)
def test_a_1to1_id_counts_only_when_the_sender_agrees(
    sender: str | None, is_from_me: bool, expected: ChatEvidence
) -> None:
    ck = f"SMS;-;{ALICE}"
    choice = choose_chat(_message(sender=sender, is_from_me=is_from_me, ck_chat_id=ck), GroupDirectory())
    assert choice.evidence is expected
    if expected is ChatEvidence.CK_1TO1:
        assert (choice.chat_guid, choice.kind, choice.ck_service, choice.ck_handle) == (
            ck, "dm", "SMS", ALICE,
        )
    else:
        assert choice.chat_guid == f"unfiled:sender:{BOB}"


def test_a_group_id_matches_only_one_group() -> None:
    one = _groups(("group-x", "iMessage;+;chat-x", "group"))
    two = _groups(("group-x", "iMessage;+;chat-x", "group"), ("group-x", "iMessage;+;chat-y", "group"))
    dm_only = _groups(("group-x", f"iMessage;-;{BOB}", "dm"))
    msg = _message(ck_chat_id="group-x")

    assert choose_chat(msg, one) == ChatChoice(ChatEvidence.CK_GROUP_MATCH, "iMessage;+;chat-x")
    for directory in (two, dm_only, GroupDirectory()):
        held = choose_chat(msg, directory)
        assert (held.evidence, held.chat_guid, held.kind, held.unfiled_key) == (
            ChatEvidence.HOLDING_LOST_GROUP, "unfiled:lost-group:group-x", "group", "lost-group:group-x",
        )


def test_the_same_chat_seen_twice_is_still_one_match() -> None:
    """A snapshot's own chat and the index's record of it are one chat."""
    directory = _groups(("group-x", "iMessage;+;chat-x", None), ("group-x", "iMessage;+;chat-x", "group"))
    assert directory.sole_group("group-x") == "iMessage;+;chat-x"


@pytest.mark.parametrize(
    ("sender", "is_from_me", "key", "name"),
    [
        pytest.param(BOB, False, f"sender:{BOB}", "Unfiled: no chat recorded", id="incoming"),
        pytest.param(None, True, "owner", "Unfiled: sent, no chat recorded", id="owner"),
        pytest.param(None, False, "unknown-sender", "Unfiled: no chat or sender recorded", id="no-sender"),
    ],
)
def test_no_evidence_files_by_sender(sender: str | None, is_from_me: bool, key: str, name: str) -> None:
    choice = choose_chat(_message(sender=sender, is_from_me=is_from_me), GroupDirectory())
    assert (choice.evidence, choice.unfiled_key, choice.chat_guid, choice.display_name, choice.kind) == (
        ChatEvidence.HOLDING_SENDER, key, f"unfiled:{key}", name, "dm",
    )


# --- moves -------------------------------------------------------------------------


def _filed(evidence: ChatEvidence | None, *, chat: str = "iMessage;+;chat-x", holding: bool = False,
           deleted: bool = False) -> IndexedFiling:
    return IndexedFiling(chat_id=1, chat_guid=chat, evidence=evidence, in_holding_chat=holding,
                         has_deleted_at=deleted)


def test_evidence_order_is_weakest_first_and_matches_the_migration() -> None:
    assert [e.rank for e in ChatEvidence] == list(range(len(ChatEvidence)))
    assert ChatEvidence.CHAT_MESSAGE_JOIN.rank == max(e.rank for e in ChatEvidence)
    assert ChatEvidence.HOLDING_SENDER.rank == 0
    allowed = re.search(r"CHECK \(chat_evidence IN \((.*?)\)\)", MIGRATION_0007.read_text(), re.S)
    assert allowed is not None
    assert set(re.findall(r"'([a-z0-9_]+)'", allowed.group(1))) == set(EVIDENCE_ORDER)


def test_a_real_chat_never_gives_up_its_message() -> None:
    real = _filed(ChatEvidence.CK_GROUP_MATCH, chat="iMessage;+;chat-x")
    elsewhere = ChatChoice(ChatEvidence.CHAT_MESSAGE_JOIN, "iMessage;+;chat-y")
    same_chat = ChatChoice(ChatEvidence.CHAT_MESSAGE_JOIN, "iMessage;+;chat-x")
    assert not refile_wanted(elsewhere, real)
    assert refile_wanted(same_chat, real)  # only the recorded evidence rises


def test_a_holding_chat_gives_way_to_anything_stronger_and_nothing_weaker() -> None:
    lost = _filed(ChatEvidence.HOLDING_LOST_GROUP, chat="unfiled:lost-group:x", holding=True)
    assert refile_wanted(ChatChoice(ChatEvidence.CK_GROUP_MATCH, "iMessage;+;chat-x"), lost)
    weaker = choose_chat(_message(), GroupDirectory())
    assert not refile_wanted(weaker, lost)


def test_an_unknown_stored_evidence_is_never_moved() -> None:
    assert not refile_wanted(ChatChoice(ChatEvidence.CHAT_MESSAGE_JOIN, "x"), _filed(None, holding=True))


def test_the_rescan_selects_only_rows_with_work_left() -> None:
    choice = ChatChoice(ChatEvidence.RECOVERABLE_JOIN, "iMessage;+;chat-x")
    when = datetime(2025, 1, 1, tzinfo=UTC)
    assert rescan_wanted(choice, None, deleted_at=None, is_tapback=False)
    assert not rescan_wanted(choice, None, deleted_at=None, is_tapback=True)
    linked = _filed(ChatEvidence.CHAT_MESSAGE_JOIN)
    assert rescan_wanted(choice, linked, deleted_at=when, is_tapback=False)
    labelled = _filed(ChatEvidence.CHAT_MESSAGE_JOIN, deleted=True)
    assert not rescan_wanted(choice, labelled, deleted_at=when, is_tapback=False)
    assert not rescan_wanted(choice, linked, deleted_at=None, is_tapback=False)


# --- rendering ---------------------------------------------------------------------


def _draft(*, deleted: bool) -> SegmentDraft:
    sent = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
    message = MessageForSegmentation(
        message_id=1, source_guid="m1", chat_id=1, sent_at=sent, is_from_me=False,
        sender_short_name="alice", text="see you at nine", is_unsent=False, is_edited=False,
        has_attachments=False, is_deleted=deleted,
    )
    return SegmentDraft(session_started_at=sent, seq_in_session=0, messages=(message,))


def _render(draft: SegmentDraft, *, kind: str = "dm", name: str | None = None, unfiled: bool = False) -> str:
    return render_segment(
        draft, participants=("Alice Example",), chat_kind=kind, chat_display_name=name,
        timezone="UTC", attachment_snippet_chars=200, unfiled=unfiled,
    )


def test_a_deleted_message_renders_labelled_and_nothing_else_changes() -> None:
    assert _render(_draft(deleted=True)).splitlines()[-1] == "[09:00] alice: [deleted] see you at nine"
    assert _render(_draft(deleted=False)).splitlines()[-1] == "[09:00] alice: see you at nine"


def test_a_holding_chat_says_so_in_its_header_and_a_real_chat_renders_as_before() -> None:
    assert _render(_draft(deleted=False), name="Unfiled: no chat recorded", unfiled=True).splitlines()[0] == (
        "Chat: Alice Example (Unfiled: no chat recorded)"
    )
    assert _render(_draft(deleted=False), name="A DM someone named").splitlines()[0] == "Chat: Alice Example"


# --- a chat.db without the columns the rules read -------------------------------


def test_the_reader_treats_missing_columns_and_tables_as_no_evidence(tmp_path: Path) -> None:
    """An older or hand-built `chat.db` has no `ck_chat_id`, no group ids
    and no "Recently Deleted" table. It extracts as before: those read as
    NULL, and an unlinked row is still found."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, style INTEGER,
                           display_name TEXT, service_name TEXT);
        CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, handle_id INTEGER,
                              is_from_me INTEGER, date INTEGER, date_edited INTEGER,
                              date_retracted INTEGER, service TEXT, thread_originator_guid TEXT,
                              item_type INTEGER, payload_data BLOB);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO chat VALUES (1, 'iMessage;-;+15550000001', 45, NULL, 'iMessage');
        INSERT INTO message VALUES (1, 'linked', 0, 1, 700000000000000000, NULL, NULL,
                                    'iMessage', NULL, 0, NULL);
        INSERT INTO message VALUES (2, 'unlinked', 0, 1, 700000060000000000, NULL, NULL,
                                    'iMessage', NULL, 0, NULL);
        INSERT INTO chat_message_join VALUES (1, 1);
        """
    )
    conn.commit()
    conn.close()

    reader = SnapshotReader(
        apsw.Connection(
            f"file:{path}?mode=ro&immutable=1",
            flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
        )
    )
    [chat] = reader.fetch_chats()
    assert chat.group_ids == ()
    linked, unlinked = reader.fetch_target_messages(0, 0)
    assert (linked.chat_rowid, unlinked.chat_rowid) == (1, None)
    assert (unlinked.ck_chat_id, unlinked.recoverable_chat_rowid, unlinked.deleted_at) == (None, None, None)
    assert [m.guid for m in reader.fetch_unlinked_messages(2)] == ["unlinked"]
