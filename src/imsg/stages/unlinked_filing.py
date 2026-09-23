"""Filing the messages `chat.db` keeps with no chat link (owner decision
D13, 2026-09-23: recall over purity).

A `chat.db` message is normally tied to its conversation by a
`chat_message_join` row. Some rows have none, and S2 used to skip them
(`extract.message_without_chat`). The 2026-09-23 survey counted 2,543
distinct such messages across the four sources the index reads that were
in no chat of the index at all. Every one of them now goes into the
corpus. The owner's words are in `docs/DECISIONS.md` D13: "I'd rather
keep all the messages in the active corpus".

**How a chat is chosen**, in this order (`choose_chat`). The evidence is
recorded on the message (`message.chat_evidence`, migration 0007):

1. `recoverable_join` -- Apple's "Recently Deleted" (`chat_recoverable_
   message_join`) names the chat. The message also gets `deleted_at`, and
   renders as `[deleted]`.
2. `ck_1to1` -- `message.ck_chat_id` has the 1:1 form `<service>;-;<handle>`
   and the handle is the sender, or the owner sent the message. The
   survey measured this rule on the mini's linked messages: it named the
   chat the message is actually linked to 489,267 times out of 489,267.
   The chat is filed under Apple's own GUID (the id itself), and created
   when the index does not have it.
3. `ck_group_match` -- a group-style `ck_chat_id` (no `;-;`) that exactly
   one indexed chat carries as its `group_id` or `original_group_id`, and
   that chat is a group. Filed automatically, accepting the measured
   misfiling rate: on the mini a group-style id named a different chat
   than the linked one for 2.8% of linked group messages.
4. `holding_lost_group` -- a group-style id no single indexed group
   carries. One holding chat per id.
5. `holding_sender` -- no chat evidence. One holding chat per sender
   handle, one for the owner's own sent messages, and one for incoming
   rows with no sender at all.

A 1:1-style id whose handle is neither the sender nor the owner
contradicts the row's own sender, so it counts as no evidence (rule 5).

**Holding chats** are ordinary `chat` rows with `unfiled_key` set. They
segment and search like any chat under `full` scope, render with a
header that says they are unfiled, and are denied outright under
`allowlist` scope and export (`imsg.export.eligibility`).

**A message moves only toward stronger evidence, and never out of a
real chat.** `ChatEvidence` is ordered weakest first. A message in a
holding chat moves when a later run (or a later source) brings evidence
that ranks higher: a real chat, or a lost-group holding chat for one
that had only a sender. A message in a real chat never changes chat; if
stronger evidence names the same chat, only the recorded evidence rises.
`refile_wanted` is the rule in Python and `refile_message` is the same
rule in SQL, and the two are kept in step on purpose: the rescan below
selects exactly the rows the SQL would change.

**Rescan.** These rows sit below every ROWID watermark: a seed's watermark
already reached the end of its file, and the live source's passed them
long ago. So every run also reads the snapshot's unlinked rows at or
below the watermark (one indexed anti-join; 0.24 s on a 699,673-message
snapshot, measured 2026-09-23) and selects the ones that still need
work: not in the index at all, in a holding chat that stronger evidence
can now move, or deleted in this snapshot while the index has no delete
date. Nothing else is re-read. When nothing is selected, the run costs
what it cost before plus that one query; when something is, `imsg-dump`
decodes from the lowest selected ROWID once (8.2-8.5 s for 700,000 rows
on the Studio, measured 2026-09-23), and the next run selects nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any

import psycopg

from imsg.keys import thread_key

CHAT_DB_TAPBACK_TYPES_SQL = (
    "COALESCE(m.associated_message_type, 0) = 1000 "
    "OR COALESCE(m.associated_message_type, 0) BETWEEN 2000 AND 3999"
)
"""Which `chat.db` rows the extractor routes to `tapback` instead of
`message`, restated in SQL for the places that must decide before the
typedstream decoder has run: the rescan's candidate query here, and
AT-2's reference built from a bare `chat.db` (`imsg.verify.seed`).

S2 itself decides from `imsg-dump`'s decoded record, not from this
clause. The `2000..3999` half is the documented reaction range; the
`1000` half (stickers) is included on evidence: in the 2026-08-20 merged
corpus all 17 `associated_message_type = 1000` rows are in Postgres's
`tapback` table and absent from `message`."""

ONE_TO_ONE_MARK = ";-;"
"""What a 1:1 chat GUID, and a 1:1-style `ck_chat_id`, carry between the
service and the handle: `iMessage;-;+15550000001`. A group id has no
such mark (it is a bare UUID-like string)."""

HOLDING_GUID_PREFIX = "unfiled:"
"""A holding chat's `source_guid` is this prefix plus its `unfiled_key`.
Apple's chat GUIDs start with a service name and a semicolon
(`iMessage;`, `SMS;`, `RCS;`, `any;`), so the two can never collide."""

LOST_GROUP_KEY_PREFIX = "lost-group:"
SENDER_KEY_PREFIX = "sender:"
OWNER_KEY = "owner"
UNKNOWN_SENDER_KEY = "unknown-sender"


class ChatEvidence(StrEnum):
    """How a message's chat was chosen (`message.chat_evidence`),
    **weakest first**: the declaration order is the rank a move compares
    (`rank`). The values are the ones migration 0007's CHECK allows."""

    HOLDING_SENDER = auto()
    HOLDING_LOST_GROUP = auto()
    CK_GROUP_MATCH = auto()
    CK_1TO1 = "ck_1to1"
    RECOVERABLE_JOIN = auto()
    CHAT_MESSAGE_JOIN = auto()
    """`chat.db` links the message to the chat. Every message filed
    before migration 0007 carries this value."""

    @property
    def rank(self) -> int:
        return EVIDENCE_ORDER.index(self.value)

    @property
    def files_into_holding_chat(self) -> bool:
        return self in (ChatEvidence.HOLDING_SENDER, ChatEvidence.HOLDING_LOST_GROUP)


EVIDENCE_ORDER: tuple[str, ...] = tuple(evidence.value for evidence in ChatEvidence)
"""Weakest first. Bound into `refile_message`'s SQL as an array, so the
database compares ranks with exactly this order."""


def parse_evidence(value: str | None) -> ChatEvidence | None:
    """The stored value as a `ChatEvidence`, or None for a value this
    build does not know (a later migration's). An unknown value ranks
    above everything, so a message carrying it is never moved."""
    if value is None:
        return None
    try:
        return ChatEvidence(value)
    except ValueError:
        return None


def is_group_id(ck_chat_id: str | None) -> bool:
    """A group-style `ck_chat_id`: set, and without the 1:1 mark."""
    return bool(ck_chat_id) and ONE_TO_ONE_MARK not in (ck_chat_id or "")


def holding_chat_guid(unfiled_key: str) -> str:
    return HOLDING_GUID_PREFIX + unfiled_key


@dataclass(frozen=True, slots=True)
class UnlinkedMessage:
    """What one snapshot says about a message it links to no chat."""

    rowid: int
    is_from_me: bool
    sender: str | None
    """The raw handle (`handle.id`) of an incoming message's sender. None
    for the owner's own messages, and for an incoming row whose handle
    the snapshot does not have."""
    ck_chat_id: str | None
    recoverable_chat_guid: str | None
    """The GUID of the chat `chat_recoverable_message_join` names, when
    that chat is in the same snapshot."""


@dataclass(frozen=True, slots=True)
class ChatChoice:
    """Where one unlinked message goes, and why."""

    evidence: ChatEvidence
    chat_guid: str
    """`chat.source_guid` of the target: Apple's own GUID for a real
    chat, `unfiled:<key>` for a holding chat."""
    kind: str | None = None
    """`dm` or `group` when this choice may create its chat (a 1:1 chat
    the index lacks, or a holding chat); None when the chat is known to
    exist already (named by the snapshot or found in the index)."""
    unfiled_key: str | None = None
    display_name: str | None = None
    ck_service: str | None = None
    """The service prefix of a 1:1 id (`iMessage`, `SMS`, `RCS`), for the
    service of a chat this choice creates."""
    ck_handle: str | None = None
    """The handle of a 1:1 id: the other participant of a chat this
    choice creates."""

    @property
    def may_create_chat(self) -> bool:
        return self.kind is not None


HOLDING_DISPLAY_NAMES: Mapping[str, str] = {
    LOST_GROUP_KEY_PREFIX: "Unfiled: group no longer in Messages",
    SENDER_KEY_PREFIX: "Unfiled: no chat recorded",
    OWNER_KEY: "Unfiled: sent, no chat recorded",
    UNKNOWN_SENDER_KEY: "Unfiled: no chat or sender recorded",
}
"""What a holding chat is called. Rendered in its segment header, so it
names no handle: the header lists the participants by name already."""


def _holding(evidence: ChatEvidence, key: str, name_key: str, kind: str) -> ChatChoice:
    return ChatChoice(
        evidence=evidence,
        chat_guid=holding_chat_guid(key),
        kind=kind,
        unfiled_key=key,
        display_name=HOLDING_DISPLAY_NAMES[name_key],
    )


@dataclass(slots=True)
class GroupDirectory:
    """Which chats carry each group id: every indexed chat's recorded ids
    (`chat_group_id`) plus the snapshot's own chats, by chat GUID."""

    _kinds_by_group: dict[str, dict[str, str | None]] = field(default_factory=dict)

    def add(self, group_id: str | None, chat_guid: str, kind: str | None) -> None:
        if not group_id:
            return
        kinds = self._kinds_by_group.setdefault(group_id, {})
        if kinds.get(chat_guid) is None:
            kinds[chat_guid] = kind

    def sole_group(self, group_id: str) -> str | None:
        """The GUID of the one chat carrying `group_id`, if there is
        exactly one and it is a group; otherwise None. Two chats with the
        same id is ambiguity, and ambiguity files into a holding chat."""
        chats = self._kinds_by_group.get(group_id, {})
        if len(chats) != 1:
            return None
        ((chat_guid, kind),) = chats.items()
        return chat_guid if kind == "group" else None


def choose_chat(message: UnlinkedMessage, groups: GroupDirectory) -> ChatChoice:
    """The D13 filing rule for one message with no chat link (module
    docstring, rules 1-5, in order)."""
    if message.recoverable_chat_guid is not None:
        return ChatChoice(ChatEvidence.RECOVERABLE_JOIN, message.recoverable_chat_guid)

    ck = message.ck_chat_id or None
    if ck is not None and ONE_TO_ONE_MARK in ck:
        handle = ck.split(";")[-1]
        if handle and (message.is_from_me or message.sender == handle):
            return ChatChoice(
                ChatEvidence.CK_1TO1,
                ck,
                kind="dm",
                ck_service=ck.split(";", 1)[0] or None,
                ck_handle=handle,
            )
        # A 1:1 id naming someone other than the sender contradicts the row:
        # no evidence, so the sender's holding chat below.
    elif ck is not None:
        matched = groups.sole_group(ck)
        if matched is not None:
            return ChatChoice(ChatEvidence.CK_GROUP_MATCH, matched)
        return _holding(
            ChatEvidence.HOLDING_LOST_GROUP, LOST_GROUP_KEY_PREFIX + ck, LOST_GROUP_KEY_PREFIX, "group"
        )

    if message.is_from_me:
        return _holding(ChatEvidence.HOLDING_SENDER, OWNER_KEY, OWNER_KEY, "dm")
    if message.sender:
        return _holding(
            ChatEvidence.HOLDING_SENDER,
            SENDER_KEY_PREFIX + message.sender,
            SENDER_KEY_PREFIX,
            "dm",
        )
    return _holding(ChatEvidence.HOLDING_SENDER, UNKNOWN_SENDER_KEY, UNKNOWN_SENDER_KEY, "dm")


# --------------------------------------------------------------------------
# where the index has a message filed, and whether a run must touch it
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexedFiling:
    """Where the index has one message filed today."""

    chat_id: int
    chat_guid: str
    evidence: ChatEvidence | None
    """None: a value this build does not know. Never moved."""
    in_holding_chat: bool
    has_deleted_at: bool


def refile_wanted(choice: ChatChoice, filed: IndexedFiling) -> bool:
    """Whether `choice` moves the message or raises its evidence: the
    evidence ranks strictly higher than what the index recorded, and
    either names the same chat (only the evidence changes) or the message
    sits in a holding chat (it moves). `refile_message` applies exactly
    this in SQL."""
    if filed.evidence is None or choice.evidence.rank <= filed.evidence.rank:
        return False
    return choice.chat_guid == filed.chat_guid or filed.in_holding_chat


def rescan_wanted(
    choice: ChatChoice,
    filed: IndexedFiling | None,
    *,
    deleted_at: datetime | None,
    is_tapback: bool,
) -> bool:
    """Whether a run must re-read a row below its watermark: the index
    lacks it (as a message, or as the tapback the decoder turned it
    into), a refile would change it, or the snapshot has a delete date
    the index lacks. Each of these is something the run then does, so a
    selected row is not selected again by the next run."""
    if filed is None:
        return not is_tapback
    if refile_wanted(choice, filed):
        return True
    return deleted_at is not None and not filed.has_deleted_at


def fetch_indexed_filings(
    cur: psycopg.Cursor[Any], guids: Iterable[str]
) -> tuple[dict[str, IndexedFiling], set[str]]:
    """`({message guid: where it is filed}, {guids the index holds as a
    tapback})` for the given `chat.db` message GUIDs."""
    wanted = sorted(set(guids))
    if not wanted:
        return {}, set()
    cur.execute(
        """
        SELECT m.source_guid, c.chat_id, c.source_guid, m.chat_evidence,
               c.unfiled_key IS NOT NULL, m.deleted_at IS NOT NULL
        FROM message m
        JOIN chat c ON c.chat_id = m.chat_id
        WHERE m.source_guid = ANY(%(guids)s::text[])
        """,
        {"guids": wanted},
    )
    filings = {
        str(guid): IndexedFiling(
            chat_id=int(chat_id),
            chat_guid=str(chat_guid),
            evidence=parse_evidence(evidence),
            in_holding_chat=bool(in_holding),
            has_deleted_at=bool(has_deleted),
        )
        for guid, chat_id, chat_guid, evidence, in_holding, has_deleted in cur.fetchall()
    }
    cur.execute(
        "SELECT source_guid FROM tapback WHERE source_guid = ANY(%(guids)s::text[])",
        {"guids": wanted},
    )
    return filings, {str(row[0]) for row in cur.fetchall()}


def fetch_weakly_filed(cur: psycopg.Cursor[Any]) -> dict[str, ChatEvidence | None]:
    """Every message the index filed on evidence weaker than a
    `chat_message_join` link, by GUID. These are the only messages a run
    may refile, so a run checks this map instead of issuing a refile for
    every message it reads. Small by construction: the rows D13 filed.

    The predicate is a literal, not a parameter, so it matches migration
    0007's partial index (`message_weak_evidence_idx`) word for word."""
    cur.execute(
        "SELECT source_guid, chat_evidence FROM message "
        "WHERE chat_evidence <> 'chat_message_join'"
    )
    return {str(guid): parse_evidence(evidence) for guid, evidence in cur.fetchall()}


def fetch_group_chats(
    cur: psycopg.Cursor[Any], group_ids: Iterable[str], directory: GroupDirectory
) -> None:
    """Add every indexed chat that carries one of `group_ids` to
    `directory`."""
    wanted = sorted({g for g in group_ids if g})
    if not wanted:
        return
    cur.execute(
        """
        SELECT g.group_id, c.source_guid, c.kind::text
        FROM chat_group_id g
        JOIN chat c ON c.chat_id = g.chat_id
        WHERE g.group_id = ANY(%(ids)s::text[])
        """,
        {"ids": wanted},
    )
    for group_id, chat_guid, kind in cur.fetchall():
        directory.add(str(group_id), str(chat_guid), str(kind))


# --------------------------------------------------------------------------
# writes (inside S2's transaction)
# --------------------------------------------------------------------------


def record_group_ids(cur: psycopg.Cursor[Any], pairs: Iterable[tuple[str, int]]) -> tuple[int, int]:
    """Insert-or-ignore `(group_id, chat_id)` pairs into `chat_group_id`,
    in one statement. Returns `(inserted, already present)`."""
    unique = sorted({(g, c) for g, c in pairs if g})
    if not unique:
        return 0, 0
    cur.execute(
        """
        INSERT INTO chat_group_id (group_id, chat_id)
        SELECT * FROM unnest(%(group_ids)s::text[], %(chat_ids)s::bigint[])
        ON CONFLICT DO NOTHING
        """,
        {"group_ids": [g for g, _ in unique], "chat_ids": [c for _, c in unique]},
    )
    inserted = max(cur.rowcount, 0)
    return inserted, len(unique) - inserted


def ensure_chat(
    cur: psycopg.Cursor[Any], choice: ChatChoice, *, service: str | None
) -> tuple[int, bool]:
    """`(chat_id, created)` for the chat `choice` names.

    A choice that may create its chat inserts it when the index lacks the
    GUID and otherwise leaves the stored row exactly as it is (merges only
    add, D12: a real source's own chat row fills its fields later through
    the ordinary chat upsert). A choice that may not create raises when
    the chat is missing, because it names a chat this run's snapshot or
    the index was read to contain."""
    if choice.may_create_chat:
        cur.execute(
            """
            WITH created AS (
                INSERT INTO chat (source_guid, thread_key, kind, display_name, service, unfiled_key)
                VALUES (%(guid)s, %(thread_key)s, %(kind)s::chat_kind, %(display_name)s,
                        COALESCE(%(service)s::service_kind, 'unknown'::service_kind),
                        %(unfiled_key)s)
                ON CONFLICT (source_guid) DO NOTHING
                RETURNING chat_id
            )
            SELECT chat_id, true FROM created
            UNION ALL
            SELECT chat_id, false FROM chat
             WHERE source_guid = %(guid)s AND NOT EXISTS (SELECT 1 FROM created)
            """,
            {
                "guid": choice.chat_guid,
                "thread_key": thread_key(choice.chat_guid),
                "kind": choice.kind,
                "display_name": choice.display_name,
                "service": service,
                "unfiled_key": choice.unfiled_key,
            },
        )
    else:
        cur.execute(
            "SELECT chat_id, false FROM chat WHERE source_guid = %(guid)s",
            {"guid": choice.chat_guid},
        )
    row = cur.fetchone()
    if row is None:
        raise LookupError(
            f"chat {choice.chat_guid!r} named by {choice.evidence.value} evidence is not in the index"
        )
    return int(row[0]), bool(row[1])


class RefileOutcome(StrEnum):
    NONE = auto()
    MOVED = auto()
    """Out of a holding chat, into a chat with stronger evidence."""
    EVIDENCE_RAISED = auto()
    """Same chat; only the recorded evidence rose."""


def refile_message(
    cur: psycopg.Cursor[Any], *, message_id: int, chat_id: int, evidence: ChatEvidence
) -> RefileOutcome:
    """`refile_wanted`, applied in SQL to a message already in the index.

    The row changes only when the new evidence ranks strictly higher than
    the stored one, and then only if it names the same chat or the
    message sits in a holding chat. A message in a real chat never moves
    to another chat. An UPDATE moves `updated_at` (migration 0003), so
    the chat it lands in is re-segmented, and S4 finds the segment it
    left (`imsg.segment.pipeline.find_dirty_chats`)."""
    cur.execute(
        """
        WITH before AS (SELECT chat_id FROM message WHERE message_id = %(message_id)s)
        UPDATE message m
        SET chat_id = %(chat_id)s, chat_evidence = %(evidence)s
        FROM before
        WHERE m.message_id = %(message_id)s
          AND coalesce(array_position(%(order)s::text[], m.chat_evidence),
                       cardinality(%(order)s::text[]) + 1)
              < array_position(%(order)s::text[], %(evidence)s::text)
          AND (m.chat_id = %(chat_id)s
               OR EXISTS (SELECT 1 FROM chat c
                           WHERE c.chat_id = m.chat_id AND c.unfiled_key IS NOT NULL))
        RETURNING before.chat_id
        """,
        {
            "message_id": message_id,
            "chat_id": chat_id,
            "evidence": evidence.value,
            "order": list(EVIDENCE_ORDER),
        },
    )
    row = cur.fetchone()
    if row is None:
        return RefileOutcome.NONE
    return RefileOutcome.MOVED if int(row[0]) != chat_id else RefileOutcome.EVIDENCE_RAISED


__all__ = [
    "CHAT_DB_TAPBACK_TYPES_SQL",
    "EVIDENCE_ORDER",
    "HOLDING_DISPLAY_NAMES",
    "HOLDING_GUID_PREFIX",
    "ONE_TO_ONE_MARK",
    "ChatChoice",
    "ChatEvidence",
    "GroupDirectory",
    "IndexedFiling",
    "RefileOutcome",
    "UnlinkedMessage",
    "choose_chat",
    "ensure_chat",
    "fetch_group_chats",
    "fetch_indexed_filings",
    "fetch_weakly_filed",
    "holding_chat_guid",
    "is_group_id",
    "parse_evidence",
    "record_group_ids",
    "refile_message",
    "refile_wanted",
    "rescan_wanted",
]
