"""S2 — Extract chats/messages/attachments from a snapshot (SPEC §8 S2).

Reads a completed S1 snapshot (never the live `chat.db` — S2 never
touches `paths.live_chat_db` at all, only the snapshot path S1 handed
back) and upserts into the Postgres schema from migration 0001. Two
sources feed each run, per SPEC §8 S2:

  (a) Plain SQL over the snapshot for everything that is already a
      first-class column in `chat.db`: chat/handle/attachment metadata,
      join tables, `sent_at`/`date_edited` timestamps, and
      `reply_to_guid` (`message.thread_originator_guid`).
  (b) `tools/imsg-dump` (`imsg.stages.imsg_dump`) for the parts that
      genuinely do require the crate's typedstream decoder: the
      current message body, prior edit-version text, tapback (reaction)
      classification, and — see below — `is_unsent`.

  **Correction after building `tools/imsg-dump` against the real
  `imessage-database` crate (this module's original draft assumed
  otherwise)**: `is_unsent` is *not* a plain SQL column. The crate's
  author grepped its entire source for any "retract"-shaped column or
  accessor and found none; it derives unsent status from
  `edited_parts`/`EditStatus::Unsent`, which requires the same
  typedstream decode as edit history. `message.date_retracted` is
  therefore **not treated as authoritative** — this module still reads
  whatever value is at that SQL column position (kept for the rescan
  watermark clause below, and as a cross-check, logged on mismatch),
  but `dump_msg.is_unsent` from `imsg-dump` is what actually populates
  `message.is_unsent` when a dump record is available. Likewise
  `is_edited` prefers `bool(dump_msg.edit_history)` over the SQL
  `date_edited IS NOT NULL` check. The rescan clause in
  `SnapshotReader.fetch_target_messages` still queries SQL
  `date_edited`/`date_retracted` as the cheapest available signal for
  "might be dirty" — a false negative there (a real chat.db where
  these columns don't carry the values SPEC §8 S2 assumed) would only
  under-select rescans, not corrupt data that *is* selected, since the
  per-row `is_unsent`/`is_edited` truth still comes from the shim.

Raw handles never leave this module's staging tables (`source_handle`,
`chat_participant_source`, `message.sender_source_handle_id`,
`tapback.sender_source_handle_id`) — hard requirement 3 (CLAUDE.md):
identity resolution is S3's job, and S2 does none of it. Every message
row this stage writes leaves `sender_person_id` NULL; S3 backfills it.

**Why this function does not take a `Config`** (unlike most stage
entry points): the "load config once, pass the object down" convention
exists so a *pipeline* only reads `config.yaml` once — the orchestrator
(S7) is where that read happens and where `config.sync.sources` is
resolved into a concrete `(source_name, snapshot_path)` pair per
source. By the time S2 runs, there is nothing left in `Config` it
needs: policy flags (`policy.index_unsent`/`index_edit_history`) are
explicitly S4's concern, not S2's (D1 — "indexing flags default false
and are honored downstream, not here"), and everything else S2 touches
(snapshot path, source name, the `imsg-dump` binary) is already a
concrete value by the time this stage is invoked. Taking an unused
`Config` parameter here just to look conventional would be worse than
explaining why it is absent.

**Tapback folding (D6)**: a chat.db tapback is a real row in its
`message` table, but this project's own schema keeps `tapback` entirely
separate from `message` — "folded metadata; never standalone
documents" (SPEC §7.2). So a target row whose `imsg-dump` record has a
non-null `tapback` field is written to the `tapback` table only; no
`message` row is created for it at all.

**Idempotence is about state, not just identity (2026-09-17)**: every
upsert here keys on a chat.db GUID, so a second run over the same
corpus has always produced no duplicate rows. That is identity
idempotence, and it is not enough. Until this date each `ON CONFLICT DO
UPDATE` fired unconditionally, rewriting every in-scope row with its
own values; migration 0003's trigger then moved `updated_at` on all of
them, exactly as it is supposed to for a real UPDATE. S4's
`find_dirty_chats` keys on that column, so an unchanged re-extraction
proposed discarding and rebuilding the segmentation and embeddings for
effectively the whole corpus (measured on the target host: 662,683 of
673,113 message rows bumped for 972 genuinely new messages, dragging
8,331 chats' incremental frontier back to their first message).

Each upsert below therefore carries an explicit
`WHERE <target>.col IS DISTINCT FROM excluded.col OR ...` guard, and a
row whose content is unchanged is not written at all. Two rules keep
that honest:

  * **The compared columns are exactly the assigned columns.** Compare
    a column the SET does not write and the row updates on every run
    forever (the value never converges), which is the original bug made
    permanent. Write a column the WHERE does not compare and a genuine
    change to it alone is silently dropped.
  * **`IS DISTINCT FROM`, never `<>`.** Half these columns are
    nullable; `<>` against NULL evaluates to NULL, so a message that
    gained a body — or lost one — would read as unchanged for good.

Which columns those are, and why the others are left first-write-wins,
is documented on `_upsert_message` and `_upsert_attachment`.

**Merges only add (owner decision D12, 2026-09-23)**: several databases
feed this index, and only this machine's own live `chat.db` may replace
a value the index already holds. Any other run (`MergeMode.SEED`: a
`--snapshot` seed, another Mac's database, a recovery candidate) inserts
rows and fills empty values, nothing more. A message body changes only
when a strictly newer edit brings it, from any source, and the body it
replaces is kept as edit history. An empty string or a 0 is never
evidence. Every table's inserts, fills, newer edits and replacements
are counted (`ExtractResult.table_counts`). The rule, and why, is in the
merge-policy section below.

**Messages with no chat link (owner decision D13, 2026-09-23)**: a
`chat.db` row with no `chat_message_join` row used to be logged
(`extract.message_without_chat`) and dropped; 2,543 such messages across
the four sources the index reads were in no chat at all. Every one is now
filed -- into the chat Apple's "Recently Deleted" names (with
`deleted_at`), the 1:1 or group chat its `ck_chat_id` names, or a holding
chat -- and records how (`message.chat_evidence`). They sit below every
ROWID watermark, so each run also re-reads the snapshot's unlinked rows
that still need filing. The rules, the rescan and its measured cost are
in `imsg.stages.unlinked_filing`.

**System/group-action messages**: chat.db also carries non-conversational
rows (member added/removed, group name changed, ...) via a nonzero
`item_type` column. Migration 0001 has no dedicated table for these —
unlike tapbacks, which got one — so there is nowhere schema-correct to
put them as a queryable entity. This build's reading: skip them (do not
insert a `message` row for "Alice added Bob to the conversation"),
count them in `ExtractResult.system_messages_skipped` for visibility
rather than silently discarding them without a trace. Flagged in the
build report as a spec gap worth a real decision if group-metadata
visibility in retrieval ever matters.
"""

from __future__ import annotations

import json
import plistlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import apsw
import psycopg
import structlog

from imsg.backfill.classify import NO_SOURCE_PATH_ERROR
from imsg.backfill.locations import record_recorded_path
from imsg.errors import ExtractionError
from imsg.hashing import sha256_file
from imsg.keys import attachment_key, message_key, thread_key
from imsg.paths import is_same_file
from imsg.sqlite_readonly import open_readonly_immutable, wal_frame_bytes, wal_sidecar_path
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun, run_imsg_dump
from imsg.stages.unlinked_filing import (
    CHAT_DB_TAPBACK_TYPES_SQL,
    ChatChoice,
    ChatEvidence,
    GroupDirectory,
    RefileOutcome,
    UnlinkedMessage,
    choose_chat,
    ensure_chat,
    fetch_group_chats,
    fetch_indexed_filings,
    fetch_weakly_filed,
    is_group_id,
    record_group_ids,
    refile_message,
    refile_wanted,
    rescan_wanted,
)
from imsg.textnorm import normalize_text, strip_nul

logger = structlog.get_logger(__name__)

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
"""chat.db timestamps (modern macOS/iOS, post-typedstream-era) are
nanoseconds since this epoch. Older chat.db versions used seconds
instead of nanoseconds; this build targets macOS 15+ (SPEC §5.1) where
the nanosecond form is what `imessage-exporter`/`imessage-database`
themselves assume, so no dual-scale heuristic is implemented here —
flagged as an assumption to verify against a real `chat.db` at Phase 1."""

OBJECT_REPLACEMENT_CHAR = "￼"
"""U+FFFC — the placeholder chat.db stores as `message.text` for an
attachment-only message with no caption. Verbatim in `text_original`
(lossless storage), stripped from the text handed to `normalize_text`
for the indexed copy (it carries no search value)."""

# chat.db's `chat.style` column (widely documented across independent
# iMessage-forensics tooling, not verified against a real chat.db in
# this environment — flagged in the build report).
CHAT_STYLE_GROUP = 43
CHAT_STYLE_DM = 45

_KNOWN_SERVICES = {"imessage": "imessage", "sms": "sms", "rcs": "rcs"}

# SQLite caps host parameters per statement (SQLITE_MAX_VARIABLE_NUMBER:
# 999 on builds before 3.32, 32766 after). Any `IN (...)` whose
# placeholder count is sized by caller data will therefore fail on a real
# corpus — a full extract passes every target message rowid at once, so
# this is not an edge case but the normal path. 900 sits under the
# conservative floor and costs nothing: these are indexed lookups.
#
# INVARIANT: never build a placeholder list directly from a caller-sized
# sequence. Route it through `_sql_var_chunks` and accumulate.
_SQL_VAR_CHUNK = 900


def _sql_var_chunks(values: Sequence[int], size: int = _SQL_VAR_CHUNK) -> Iterator[list[int]]:
    """Yield `values` in chunks that fit SQLite's host-parameter limit."""
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _normalize_service(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    return _KNOWN_SERVICES.get(raw.strip().lower(), "unknown")


def _service_evidence(raw: str | None) -> str | None:
    """`_normalize_service`, but returning None where it would return
    `"unknown"`.

    `unknown` is not a service; it is this module's marker for "the
    snapshot did not say". Written as a value it overwrites a service
    another source knew, which is the merge-policy defect this module
    documents under `Merge` — so the evidence-bearing callers take this
    form instead, and `unknown` survives only as the value a first
    INSERT records."""
    normalized = _normalize_service(raw)
    return None if normalized == "unknown" else normalized


def _apple_ns_to_datetime(value: int | None) -> datetime | None:
    if value is None or value == 0:
        return None
    return APPLE_EPOCH + timedelta(seconds=value / 1_000_000_000)


def _datetime_to_apple_ns(value: datetime) -> int:
    delta = value.astimezone(UTC) - APPLE_EPOCH
    return int(delta.total_seconds() * 1_000_000_000)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("extract.unparseable_timestamp", value=value)
        return None


# --------------------------------------------------------------------------
# snapshot row shapes
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatRow:
    rowid: int
    guid: str
    style: int | None
    display_name: str | None
    service_name: str | None
    participant_count: int
    group_ids: tuple[str, ...] = ()
    """`chat.group_id` and `chat.original_group_id`, where set: what a
    lost group's `message.ck_chat_id` is matched against (D13,
    `imsg.stages.unlinked_filing`)."""


@dataclass(frozen=True, slots=True)
class HandleRow:
    rowid: int
    raw_value: str
    service: str | None


@dataclass(frozen=True, slots=True)
class MessageRow:
    rowid: int
    guid: str
    handle_rowid: int | None
    is_from_me: bool
    date: datetime | None
    date_edited: datetime | None
    date_retracted: datetime | None
    service: str | None
    reply_to_guid: str | None
    item_type: int
    payload_data: bytes | None
    chat_rowid: int | None
    """The chat `chat_message_join` links the message to (the lowest chat
    ROWID when there are several), or None when no link names a chat in
    this snapshot: then `imsg.stages.unlinked_filing` chooses one."""
    ck_chat_id: str | None = None
    recoverable_chat_rowid: int | None = None
    """The chat Apple's "Recently Deleted" (`chat_recoverable_message_join`)
    names, when that chat is in this snapshot."""
    deleted_at: datetime | None = None
    """`chat_recoverable_message_join.delete_date`: set when the message is
    in "Recently Deleted" in this snapshot."""


@dataclass(frozen=True, slots=True)
class AttachmentRow:
    rowid: int
    guid: str
    source_path: str | None
    filename: str | None
    uti: str | None
    mime_type: str | None
    byte_size: int | None
    is_sticker: bool


# --------------------------------------------------------------------------
# snapshot reader
# --------------------------------------------------------------------------

OpenSnapshotFn = Callable[[str], "apsw.Connection"]


def _default_open_snapshot(path: str) -> apsw.Connection:
    """Open the snapshot (an S1 output, or a `--snapshot` seed) strictly
    read-only and without leaving `-wal`/`-shm` sidecars next to it.

    Uses `imsg.sqlite_readonly`'s `mode=ro&immutable=1` open, which is
    what keeps a read from writing anything beside a chat.db-shaped file
    (observed 2026-09-14: the plain `SQLITE_OPEN_READONLY` open this used
    to be moved a seed corpus's `-shm` mtime on every run). `immutable=1`
    also makes SQLite ignore the file's write-ahead log, so it is only
    correct for a file with no uncheckpointed frames — true of every S1
    output (the backup API writes the whole database into the main file)
    but not guaranteed for a seed someone copied together with its
    `-wal`. Refusing such a file, naming the one-line fix, is the
    fail-closed choice: an open that succeeded would silently skip every
    message committed to that log, and AT-2 (`imsg verify-seed
    --reference-db`) would not catch it, because it reads the reference
    the same way.
    """
    pending = wal_frame_bytes(path)
    if pending:
        raise ExtractionError(
            f"refusing to read '{path}': its write-ahead log '{wal_sidecar_path(path)}' holds "
            f"{pending} bytes of frames, and the immutable read-only open S2 uses (so that it "
            f"never writes next to a database) cannot see rows committed there — they would "
            f"be skipped silently. Fold the log into the file first, on this copy only, never "
            f"on the live chat.db: sqlite3 '{path}' 'PRAGMA wal_checkpoint(TRUNCATE)'"
        )
    return open_readonly_immutable(path)


class SnapshotReader:
    """Read-only SQL access to one S1 snapshot file.

    A thin wrapper, not an ORM: each method runs one query and returns
    plain dataclasses. Kept separate from the Postgres-upsert half of
    this module so the two can be tested/reasoned about independently.
    """

    def __init__(self, conn: apsw.Connection) -> None:
        self._conn = conn
        # Columns and a table that every modern `chat.db` has, and that an
        # older or hand-built database may lack: each is read as NULL where
        # it is missing, so a file without them extracts exactly as before.
        self._chat_columns = self._columns("chat")
        self._message_columns = self._columns("message")
        self._has_recoverable_join = bool(
            list(
                self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'chat_recoverable_message_join'"
                )
            )
        )

    def _columns(self, table: str) -> frozenset[str]:
        return frozenset(str(row[1]) for row in self._conn.execute(f"PRAGMA table_info({table})"))

    def _message_column(self, name: str) -> str:
        return f"m.{name}" if name in self._message_columns else "NULL"

    def fetch_max_message_rowid(self) -> int:
        row = next(self._conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message"), None)
        return int(row[0]) if row else 0

    def fetch_chats(self) -> list[ChatRow]:
        group_id = "c.group_id" if "group_id" in self._chat_columns else "NULL"
        original_group_id = (
            "c.original_group_id" if "original_group_id" in self._chat_columns else "NULL"
        )
        rows = list(
            self._conn.execute(
                f"""
                SELECT c.ROWID, c.guid, c.style, c.display_name, c.service_name,
                       (SELECT COUNT(*) FROM chat_handle_join chj WHERE chj.chat_id = c.ROWID),
                       {group_id}, {original_group_id}
                FROM chat c
                """
            )
        )
        return [
            ChatRow(
                rowid=r[0],
                guid=r[1],
                style=r[2],
                display_name=r[3],
                service_name=r[4],
                participant_count=r[5],
                group_ids=tuple(dict.fromkeys(str(g) for g in (r[6], r[7]) if g)),
            )
            for r in rows
        ]

    def _message_select(self) -> str:
        """The columns every message query returns, in `_to_message_row`'s
        order. The chat link is read as the lowest-ROWID chat that exists in
        this snapshot, which makes a message joined to several chats land
        in the same one on every run (and matches `imsg-dump`, which orders
        the same way); a link to a chat row the file lacks counts as no
        link, and the D13 rules file the message instead."""
        if self._has_recoverable_join:
            recoverable_chat = (
                "(SELECT r.chat_id FROM chat_recoverable_message_join r "
                "JOIN chat rc ON rc.ROWID = r.chat_id "
                "WHERE r.message_id = m.ROWID ORDER BY r.chat_id LIMIT 1)"
            )
            deleted_at = (
                "(SELECT MAX(r.delete_date) FROM chat_recoverable_message_join r "
                "WHERE r.message_id = m.ROWID)"
            )
        else:
            recoverable_chat = deleted_at = "NULL"
        return f"""
            SELECT m.ROWID, m.guid, m.handle_id, m.is_from_me, m.date, m.date_edited,
                   m.date_retracted, m.service, m.thread_originator_guid,
                   COALESCE(m.item_type, 0), m.payload_data,
                   (SELECT cmj.chat_id FROM chat_message_join cmj
                    JOIN chat jc ON jc.ROWID = cmj.chat_id
                    WHERE cmj.message_id = m.ROWID ORDER BY cmj.chat_id LIMIT 1),
                   {self._message_column("ck_chat_id")},
                   {recoverable_chat},
                   {deleted_at}
            FROM message m
        """

    @staticmethod
    def _to_message_row(r: Sequence[Any]) -> MessageRow:
        return MessageRow(
            rowid=r[0],
            guid=r[1],
            handle_rowid=r[2],
            is_from_me=bool(r[3]),
            date=_apple_ns_to_datetime(r[4]),
            date_edited=_apple_ns_to_datetime(r[5]),
            date_retracted=_apple_ns_to_datetime(r[6]),
            service=r[7],
            reply_to_guid=r[8],
            item_type=r[9],
            payload_data=r[10],
            chat_rowid=r[11],
            ck_chat_id=r[12] or None,
            recoverable_chat_rowid=r[13],
            deleted_at=_apple_ns_to_datetime(r[14]),
        )

    def fetch_handles(self) -> list[HandleRow]:
        rows = list(self._conn.execute("SELECT ROWID, id, service FROM handle"))
        return [HandleRow(rowid=r[0], raw_value=r[1], service=r[2]) for r in rows]

    def fetch_chat_handle_joins(self) -> list[tuple[int, int]]:
        return [
            (r[0], r[1]) for r in self._conn.execute("SELECT chat_id, handle_id FROM chat_handle_join")
        ]

    def fetch_target_messages(self, watermark: int, last_run_start_ns: int) -> list[MessageRow]:
        """Every message row in scope for this run (SPEC §8 S2):
        `ROWID > watermark` (new), plus any row whose `date_edited` or
        `date_retracted` moved past `last_run_start_ns` (a rescan of an
        older row that was edited/retracted since the last successful
        run, regardless of its own ROWID)."""
        query = (
            self._message_select()
            + """
            WHERE m.ROWID > ?
               OR COALESCE(m.date_edited, 0) > ?
               OR COALESCE(m.date_retracted, 0) > ?
            ORDER BY m.ROWID
        """
        )
        rows = self._conn.execute(query, (watermark, last_run_start_ns, last_run_start_ns))
        return [self._to_message_row(r) for r in rows]

    def fetch_unlinked_messages(self, max_rowid: int) -> list[MessageRow]:
        """Every row at or below `max_rowid` that is a candidate for D13's
        rescan: a message (not a tapback by its SQL type, not a system
        row), dated, with a GUID, and linked to no chat in this snapshot.
        The run decides which of them need work
        (`imsg.stages.unlinked_filing.rescan_wanted`).

        A row with no date is left out: `message.sent_at` is NOT NULL, so
        it cannot be filed, and selecting it would re-read it forever."""
        tapback_filter = (
            f"AND NOT ({CHAT_DB_TAPBACK_TYPES_SQL})"
            if "associated_message_type" in self._message_columns
            else ""
        )
        query = (
            self._message_select()
            + f"""
            WHERE m.ROWID <= ?
              AND m.guid IS NOT NULL
              AND COALESCE(m.item_type, 0) = 0
              AND COALESCE(m.date, 0) <> 0
              {tapback_filter}
              AND NOT EXISTS (
                  SELECT 1 FROM chat_message_join j JOIN chat jc ON jc.ROWID = j.chat_id
                  WHERE j.message_id = m.ROWID
              )
            ORDER BY m.ROWID
        """
        )
        return [self._to_message_row(r) for r in self._conn.execute(query, (max_rowid,))]

    def fetch_attachments_for_messages(self, message_rowids: list[int]) -> tuple[
        list[AttachmentRow], dict[int, list[int]]
    ]:
        """Returns `(attachment rows, {message_rowid: [attachment_rowid, ...]})`
        for the given message rowids, ordered by attachment ROWID
        (chat.db's `message_attachment_join` is not documented to carry
        an explicit ordinal, so this is our own deterministic ordering
        — flagged as an assumption)."""
        if not message_rowids:
            return [], {}
        joins: list[tuple[int, int]] = []
        for chunk in _sql_var_chunks(message_rowids):
            placeholders = ",".join("?" for _ in chunk)
            joins.extend(
                self._conn.execute(
                    f"SELECT message_id, attachment_id FROM message_attachment_join "
                    f"WHERE message_id IN ({placeholders})",
                    chunk,
                )
            )
        # Ordering is applied here rather than per-chunk: an ORDER BY inside
        # each chunk only sorts that chunk, so the concatenation would not be
        # globally ordered. This preserves the documented contract.
        joins.sort(key=lambda r: (r[0], r[1]))

        by_message: dict[int, list[int]] = {}
        attachment_ids: list[int] = []
        for message_id, attachment_id in joins:
            by_message.setdefault(message_id, []).append(attachment_id)
            attachment_ids.append(attachment_id)

        if not attachment_ids:
            return [], by_message

        # Dedupe BEFORE chunking. `IN (...)` is a per-row predicate, so a
        # repeated id in one list never duplicated output — but running the
        # query once per chunk does, whenever the two references land in
        # different chunks. The real chat.db has 98 attachments joined to more
        # than one message, so this was over-counting `attachments_upserted`
        # and upserting each shared attachment twice per run. Harmless to the
        # data (both upserts are ON CONFLICT idempotent), wrong in the counts.
        rows: list[tuple[Any, ...]] = []
        for chunk in _sql_var_chunks(sorted(set(attachment_ids))):
            att_placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                self._conn.execute(
                    f"SELECT ROWID, guid, filename, transfer_name, uti, mime_type, "
                    f"total_bytes, COALESCE(is_sticker, 0) FROM attachment "
                    f"WHERE ROWID IN ({att_placeholders})",
                    chunk,
                )
            )
        attachments = [
            AttachmentRow(
                rowid=r[0],
                guid=r[1],
                source_path=r[2],
                filename=r[3],
                uti=r[4],
                mime_type=r[5],
                byte_size=r[6],
                is_sticker=bool(r[7]),
            )
            for r in rows
        ]
        return attachments, by_message


# --------------------------------------------------------------------------
# link previews (best-effort NSKeyedArchiver decode of `payload_data`)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinkPreview:
    url: str
    title: str | None
    summary: str | None
    site_name: str | None


_LINK_PREVIEW_KEYS = {
    "url": "url",
    "URL": "url",
    "title": "title",
    "Title": "title",
    "summary": "summary",
    "Summary": "summary",
    "siteName": "site_name",
    "SiteName": "site_name",
    "site_name": "site_name",
}


def _resolve_nskeyedarchiver_uid(obj: Any, objects: list[Any], *, _depth: int = 0) -> Any:
    """Recursively resolve `plistlib.UID` references in an
    NSKeyedArchiver-format plist's `$objects` array into plain
    Python values.

    This is a **best-effort, generic** resolver, not a faithful
    `NSKeyedUnarchiver`: it does not reconstruct real classes, it just
    walks `UID` -> `$objects[uid]` and turns `$null` into `None` so
    plain dict/string/number values underneath become directly
    searchable. `payload_data` (SPEC §8 S2 "link previews parsed from
    `payload_data` plists") was not available to verify this against a
    real sample in this environment — see the module-level note in the
    build report. Depth-bounded against cyclic object graphs.
    """
    if _depth > 20:
        return None
    if isinstance(obj, plistlib.UID):
        if 0 <= obj.data < len(objects):
            return _resolve_nskeyedarchiver_uid(objects[obj.data], objects, _depth=_depth + 1)
        return None
    if isinstance(obj, dict):
        if obj == {"$class": None} or obj.get("$classname") == "$null":
            return None
        return {
            k: _resolve_nskeyedarchiver_uid(v, objects, _depth=_depth + 1)
            for k, v in obj.items()
            if k not in ("$class",)
        }
    if isinstance(obj, list):
        return [_resolve_nskeyedarchiver_uid(v, objects, _depth=_depth + 1) for v in obj]
    if obj == "$null":
        return None
    return obj


def _search_for_link_preview(resolved: Any, *, _depth: int = 0) -> dict[str, str] | None:
    """Depth-first search of a resolved NSKeyedArchiver tree for the
    first dict that looks like link-preview metadata (has a URL-shaped
    string under any of the recognized key spellings)."""
    if _depth > 20:
        return None
    if isinstance(resolved, dict):
        found: dict[str, str] = {}
        for raw_key, mapped_key in _LINK_PREVIEW_KEYS.items():
            value = resolved.get(raw_key)
            if isinstance(value, str) and value:
                found.setdefault(mapped_key, value)
        if "url" in found and (
            found["url"].startswith("http://") or found["url"].startswith("https://")
        ):
            return found
        for value in resolved.values():
            nested = _search_for_link_preview(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    elif isinstance(resolved, list):
        for item in resolved:
            nested = _search_for_link_preview(item, _depth=_depth + 1)
            if nested is not None:
                return nested
    return None


def parse_link_preview(payload_data: bytes | None) -> LinkPreview | None:
    """Best-effort extraction of a link preview from a message's
    `payload_data` blob. Returns `None` for anything that is not a
    parseable NSKeyedArchiver-format binary plist containing a
    recognizable URL field — never raises, per S2's "degrade and
    continue" philosophy for attachment/payload edge cases (SPEC §8
    S2)."""
    if not payload_data:
        return None
    try:
        top = plistlib.loads(payload_data)
    except (plistlib.InvalidFileException, ValueError, TypeError):
        return None
    if not isinstance(top, dict) or "$objects" not in top or "$top" not in top:
        return None
    objects = top.get("$objects")
    if not isinstance(objects, list):
        return None
    root_ref = top["$top"].get("root") if isinstance(top["$top"], dict) else None
    if root_ref is None:
        return None
    resolved = _resolve_nskeyedarchiver_uid(root_ref, objects)
    found = _search_for_link_preview(resolved)
    if found is None:
        return None
    return LinkPreview(
        url=found["url"],
        title=found.get("title"),
        summary=found.get("summary"),
        site_name=found.get("site_name"),
    )


# --------------------------------------------------------------------------
# which merge rule a run applies (D12)
# --------------------------------------------------------------------------


class MergeMode(StrEnum):
    """Which rule a run applies to a row the index already holds.

    Owner decision D12 (2026-09-23): "Always want to merge and maintain
    the fullest corpus of chat.db and attachments, not overwrite and
    lose rows." Several witnesses feed one index -- this machine's own
    `chat.db`, another Mac's, an older merged corpus, recovery
    candidates -- and until this rule whichever ran last won every
    column it carried. Each column's `Merge` rule says what counts as
    evidence; this says whose evidence may replace a value the index
    already holds.
    """

    LIVE = "live"
    """S1's copy of this machine's own `paths.live_chat_db`. The one
    source whose non-empty value may replace a different non-empty one:
    a renamed chat, an attachment the Messages app moved."""

    SEED = "seed"
    """Any other database: a `--snapshot` seed, another Mac's `chat.db`,
    a recovery candidate. Inserts rows and fills empty values; never
    replaces a non-empty one. `run_extract`'s default, so a caller that
    cannot vouch for its snapshot gets the rule that loses nothing."""


def merge_mode_for_source(source_chat_db: Path | None, live_chat_db: Path) -> MergeMode:
    """`LIVE` only when `source_chat_db` -- the database S1 copied -- is
    this machine's own `paths.live_chat_db`, compared by resolved path
    and then inode (`imsg.paths.is_same_file`, the test the seed guard in
    `imsg.cli` uses). `SEED` for anything else: no S1 source at all (a
    `--snapshot` seed), a configured source that points at another
    Mac's database, or a comparison that fails.

    Being listed in `sync.sources` is deliberately not the test:
    `config.example.yaml` shows a transferred Studio snapshot configured
    there, and that witness must only add."""
    if source_chat_db is None:
        return MergeMode.SEED
    try:
        same = is_same_file(source_chat_db, live_chat_db)
    except (OSError, RuntimeError, ValueError):
        return MergeMode.SEED
    return MergeMode.LIVE if same else MergeMode.SEED


# --------------------------------------------------------------------------
# extraction result + main entry point
# --------------------------------------------------------------------------


class UpsertOutcome(StrEnum):
    """What one upsert statement actually did to its row. The values are
    `UpsertCounts`' field names."""

    INSERTED = "inserted"
    FILLED = "filled"
    """Updated, and every value it changed was empty before (see
    `Empty`: NULL, `''`, a size of 0, the `unknown` service marker, or
    `false` on a `POSITIVE` flag). Apart from a newer edit, the only
    update a seed makes."""
    NEWER_EDIT = "newer_edit"
    """Updated because a strictly newer edit replaced a message body
    (D12: text follows edit recency, from any source), plus any fills.
    The body it replaced is kept in `message_version`."""
    REPLACED = "replaced"
    """Updated, and at least one non-empty value was replaced by a
    different one. Only the live run does this; a seed never can."""
    UNCHANGED = "unchanged"
    """The row already held exactly these values, so no UPDATE was
    performed at all — no new row version, and migration 0003's trigger
    never fired, so `updated_at` did not move."""


@dataclass(frozen=True, slots=True)
class UpsertCounts:
    """How one table's rows landed.

    `total` is the number of rows the loop processed, which is what the
    `*_upserted` fields below have always reported — the split is the
    part an operator needs to trust a re-extraction. "Upserted: 662,683"
    reads identically whether the run corrected 662,683 rows or
    rewrote them with their own values; `unchanged=662,683` says
    plainly that the run wrote nothing, moved no `updated_at`, and
    therefore proposes no re-segmentation downstream.

    `updated` is split three ways (D12) so a seed can be checked against
    "inserts and fills only": `filled` changed only empty values,
    `newer_edit` took a strictly newer edit's body, and `replaced`
    overwrote a non-empty value -- which only the live run may do.
    """

    inserted: int = 0
    filled: int = 0
    newer_edit: int = 0
    replaced: int = 0
    unchanged: int = 0

    @property
    def updated(self) -> int:
        """Every existing row this run rewrote."""
        return self.filled + self.newer_edit + self.replaced

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.unchanged

    def as_dict(self) -> dict[str, int]:
        return {outcome.value: getattr(self, outcome.value) for outcome in UpsertOutcome}


@dataclass(slots=True)
class _UpsertTally:
    """Mutable accumulator for `UpsertCounts` (which is frozen, as every
    field of `ExtractResult` is)."""

    inserted: int = 0
    filled: int = 0
    newer_edit: int = 0
    replaced: int = 0
    unchanged: int = 0

    def record(self, outcome: UpsertOutcome) -> None:
        setattr(self, outcome.value, getattr(self, outcome.value) + 1)

    def freeze(self) -> UpsertCounts:
        return UpsertCounts(
            inserted=self.inserted,
            filled=self.filled,
            newer_edit=self.newer_edit,
            replaced=self.replaced,
            unchanged=self.unchanged,
        )


@dataclass(frozen=True, slots=True)
class UnlinkedCounts:
    """What a run did with messages its snapshot links to no chat (owner
    decision D13; `imsg.stages.unlinked_filing`)."""

    rescanned: int = 0
    """Rows at or below the watermark this run read again because they
    still needed work: missing from the index, movable out of a holding
    chat, or deleted without a delete date in the index. Zero on a run
    with nothing left to file."""
    recoverable_join: int = 0
    ck_1to1: int = 0
    ck_group_match: int = 0
    holding_lost_group: int = 0
    holding_sender: int = 0
    """Messages this run placed by each rule: inserted, or moved out of a
    holding chat. A message already where the rule would put it is not
    counted."""
    moved_from_holding: int = 0
    """Messages moved out of a holding chat into a chat with stronger
    evidence (by any rule, a chat link included)."""
    evidence_raised: int = 0
    """Messages left in the same chat whose recorded evidence rose."""
    in_recently_deleted: int = 0
    """Messages this run read that its snapshot has in Apple's "Recently
    Deleted" (a delete date). Their `deleted_at` is filled where the index
    has none (the `message` table's `filled` count includes it)."""
    chats_created: int = 0
    """1:1 chats created under Apple's own GUID from a `ck_chat_id`."""
    holding_chats_created: int = 0
    skipped_without_date: int = 0
    """Unlinked rows with no `date`, which cannot be filed
    (`message.sent_at` is NOT NULL). Logged, not raised: before D13 these
    rows were skipped with every other unlinked row."""

    def placed(self, evidence: ChatEvidence) -> int:
        return int(getattr(self, evidence.value, 0))


@dataclass(slots=True)
class _UnlinkedTally:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def freeze(self) -> UnlinkedCounts:
        return UnlinkedCounts(**self.counts)


@dataclass(frozen=True, slots=True)
class ExtractResult:
    run_id: int
    watermark_before: int
    watermark_after: int
    chats_upserted: int
    handles_upserted: int
    messages_upserted: int
    tapbacks_upserted: int
    system_messages_skipped: int
    attachments_upserted: int
    link_previews_upserted: int
    bodies_missing: int
    """Target rows whose guid never appeared in `imsg-dump`'s output at
    all (a boundary anomaly, distinct from the shim's own per-row
    null-body degrade, which *does* appear)."""
    dump_stderr_line_count: int
    chat_upserts: UpsertCounts = field(default_factory=UpsertCounts)
    handle_upserts: UpsertCounts = field(default_factory=UpsertCounts)
    message_upserts: UpsertCounts = field(default_factory=UpsertCounts)
    attachment_upserts: UpsertCounts = field(default_factory=UpsertCounts)
    """Inserted / filled / newer edit / replaced / left untouched.
    `*.total` equals the matching `*_upserted` count above, which keeps
    its original meaning ("rows this run processed") — the breakdown is
    additive, so nothing reading the old fields changes behaviour."""
    tapback_upserts: UpsertCounts = field(default_factory=UpsertCounts)
    message_version_upserts: UpsertCounts = field(default_factory=UpsertCounts)
    link_preview_upserts: UpsertCounts = field(default_factory=UpsertCounts)
    """The same split for the other three tables with a guarded upsert
    (D12 rule 5: every table the run writes is reported; the 2026-09-22
    dry run printed message counts only and hid its chat and attachment
    rewrites)."""
    message_attachment_links: UpsertCounts = field(default_factory=UpsertCounts)
    chat_participant_links: UpsertCounts = field(default_factory=UpsertCounts)
    """Join rows: inserted, or already present (`unchanged`). They are
    insert-or-ignore, so they are never updated."""
    message_source_rows: UpsertCounts = field(default_factory=UpsertCounts)
    attachment_source_rows: UpsertCounts = field(default_factory=UpsertCounts)
    """Provenance: which row of which source is which message or
    attachment. `unchanged` means this source had already recorded that
    row. For `message_source` the run id that last read the row still
    moves to this run -- bookkeeping, not corpus content (see
    `_upsert_message_source`). `replaced` on `attachment_source` means a
    source row now resolves to a different attachment."""
    attachment_location_rows: UpsertCounts = field(default_factory=UpsertCounts)
    """The path this source's chat.db recorded for each attachment it read
    (`imsg.backfill.locations.record_recorded_path`): inserted, or already
    recorded (`unchanged`). Never updated, so a path is never lost."""
    merge_mode: MergeMode = MergeMode.SEED
    """The rule this run applied to rows the index already held."""
    messages_marked_for_resegmentation: int = 0
    """Messages whose `updated_at` this run moved because something a
    segment renders through them changed in another table: a chat's
    name or kind, an attachment's name or type, a new attachment link,
    a tapback, an edit-history version. (Messages whose own row changed
    move by themselves and are not counted here.)"""
    bodies_kept_as_history: int = 0
    """Bodies a strictly newer edit replaced that the index did not
    already hold as an edit-history version, appended to
    `message_version` so the replacement lost no text."""
    tapback_targets_resolved: int = 0
    """Tapbacks whose target message had not been extracted yet when
    they were, attached to it at the end of this run."""
    chat_group_ids: UpsertCounts = field(default_factory=UpsertCounts)
    """`(group id, chat)` pairs recorded in `chat_group_id`: inserted, or
    already present (`unchanged`). Insert-only."""
    unlinked: UnlinkedCounts = field(default_factory=UnlinkedCounts)
    """Messages with no chat link: how this run filed them (D13)."""
    dry_run: bool = False
    """True when this result came from `run_extract(dry_run=True)`
    (SPEC §8: "takes --dry-run where writes leave the machine"): every
    count above is accurate (the real extraction logic ran end to
    end, including the read-only `imsg-dump` subprocess call), but the
    transaction that produced them was rolled back — nothing was
    actually written to Postgres, and `run_id` refers to an
    `extraction_run` row that existed only for the rolled-back
    transaction's duration."""

    def table_counts(self) -> dict[str, UpsertCounts]:
        """Every table this run writes, by its Postgres name, in the
        order the run writes them. What `imsg extract` prints and what
        `extraction_run.upsert_counts` records."""
        return {
            "chat": self.chat_upserts,
            "chat_group_id": self.chat_group_ids,
            "source_handle": self.handle_upserts,
            "chat_participant_source": self.chat_participant_links,
            "attachment": self.attachment_upserts,
            "attachment_source": self.attachment_source_rows,
            "attachment_location": self.attachment_location_rows,
            "message": self.message_upserts,
            "message_source": self.message_source_rows,
            "message_version": self.message_version_upserts,
            "message_attachment": self.message_attachment_links,
            "link_preview": self.link_preview_upserts,
            "tapback": self.tapback_upserts,
        }


RunImsgDumpFn = Callable[[Path, Path, int], ImsgDumpRun]


def _default_run_imsg_dump(binary_path: Path, snapshot_path: Path, since_rowid: int) -> ImsgDumpRun:
    return run_imsg_dump(binary_path=binary_path, snapshot_path=snapshot_path, since_rowid=since_rowid)


def _watermark_key(source_name: str) -> str:
    return f"watermark.rowid.{source_name}"


def _fetch_watermark(cur: psycopg.Cursor[Any], source_name: str) -> int:
    cur.execute("SELECT value FROM sync_state WHERE key = %s", (_watermark_key(source_name),))
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _fetch_last_successful_run_start(cur: psycopg.Cursor[Any], source_name: str) -> datetime:
    cur.execute(
        "SELECT started_at FROM extraction_run WHERE source_name = %s AND status = 'ok' "
        "ORDER BY started_at DESC LIMIT 1",
        (source_name,),
    )
    row = cur.fetchone()
    if row is None:
        return datetime(1970, 1, 1, tzinfo=UTC)
    started_at: datetime = row[0]
    return started_at


class _DryRunRollback(Exception):
    """Internal sentinel: forces the outer `with conn.transaction():`
    block `_do_extract_dry_run` opens to ROLLBACK — a psycopg3 ROLLBACK
    undoes nested SAVEPOINTs too, so `run_extract(dry_run=True)`
    genuinely writes nothing to Postgres while still running the real
    extraction logic (accurate counts, including the read-only
    `imsg-dump` subprocess call). Caught immediately below; never
    allowed to escape this module."""

    def __init__(self, result: ExtractResult) -> None:
        self.result = result


def _do_extract_dry_run(
    *,
    conn: psycopg.Connection,
    reader: SnapshotReader,
    source_name: str,
    snapshot_path: Path,
    snapshot_sha256: str,
    watermark_before: int,
    last_run_start: datetime,
    snapshot_max_rowid: int,
    imsg_dump_binary: Path,
    run_imsg_dump_fn: RunImsgDumpFn,
    merge_mode: MergeMode,
) -> ExtractResult:
    """SPEC §8 dry-run for S2: `_begin_extraction_run` and `_do_extract`
    each already open their own `conn.transaction()`; calling both from
    inside one outer `with conn.transaction():` turns those into
    SAVEPOINTs instead of independent top-level transactions, so
    raising `_DryRunRollback` after they both complete rolls back
    everything at once. If `_do_extract` raises a real error instead,
    that (not `_DryRunRollback`) propagates out of the `with` block —
    the transaction still rolls back (any exception does that), but
    this function deliberately does not call `_fail_extraction_run`
    itself: nothing was ever going to be persisted either way, so
    there is nothing to mark failed.
    """
    try:
        with conn.transaction():
            run_id = _begin_extraction_run(
                conn,
                source_name=source_name,
                snapshot_path=snapshot_path,
                snapshot_sha256=snapshot_sha256,
                rowid_before=watermark_before,
                merge_mode=merge_mode,
            )
            result = _do_extract(
                conn=conn,
                reader=reader,
                source_name=source_name,
                snapshot_path=snapshot_path,
                watermark_before=watermark_before,
                last_run_start=last_run_start,
                snapshot_max_rowid=snapshot_max_rowid,
                run_id=run_id,
                imsg_dump_binary=imsg_dump_binary,
                run_imsg_dump_fn=run_imsg_dump_fn,
                merge_mode=merge_mode,
            )
            raise _DryRunRollback(result)
    except _DryRunRollback as sentinel:
        return replace(sentinel.result, dry_run=True)


def run_extract(
    *,
    conn: psycopg.Connection,
    source_name: str,
    snapshot_path: Path,
    snapshot_sha256: str | None = None,
    imsg_dump_binary: Path,
    open_snapshot: OpenSnapshotFn = _default_open_snapshot,
    run_imsg_dump_fn: RunImsgDumpFn = _default_run_imsg_dump,
    dry_run: bool = False,
    merge_mode: MergeMode = MergeMode.SEED,
) -> ExtractResult:
    """Extract one snapshot into Postgres (SPEC §8 S2).

    `conn` must already be open (this module never owns connection
    lifecycle); `snapshot_path` is an S1 output, never the live
    `chat.db`. Raises `ExtractionError` for boundary failures (the
    snapshot does not open as SQLite, `imsg-dump` fails to run); a
    single message's decode failure inside `imsg-dump` degrades to a
    null body there and is *not* an `ExtractionError` here.

    `merge_mode` is the D12 rule for rows the index already holds.
    Only a caller that knows this snapshot is S1's copy of this
    machine's own live database passes `MergeMode.LIVE`
    (`merge_mode_for_source` decides it); the default is `SEED`, which
    inserts and fills and never replaces a non-empty value.

    `dry_run=True` (SPEC §8: "takes --dry-run where writes leave the
    machine") runs the entire real extraction body — including the
    `imsg-dump` subprocess call, which is read-only — inside one outer
    transaction, then forces a ROLLBACK before returning (see
    `_do_extract_dry_run`), so every count in the returned
    `ExtractResult` is accurate but nothing is actually written to
    Postgres.
    """
    if snapshot_sha256 is None:
        snapshot_sha256 = sha256_file(snapshot_path)

    try:
        snapshot = open_snapshot(str(snapshot_path))
    except apsw.Error as exc:
        raise ExtractionError(f"snapshot at '{snapshot_path}' will not open as SQLite: {exc}") from exc

    try:
        reader = SnapshotReader(snapshot)

        with conn.cursor() as cur:
            watermark_before = _fetch_watermark(cur, source_name)
            last_run_start = _fetch_last_successful_run_start(cur, source_name)
            snapshot_max_rowid = reader.fetch_max_message_rowid()

        # NOTE: no transaction management here on purpose. `connect()` sets
        # autocommit=True (see db/connection.py), so the reads above leave the
        # connection idle and each `conn.transaction()` below is a genuine
        # top-level transaction. This function must NOT call rollback/commit
        # itself — it does not own the connection's lifecycle, and a caller may
        # legitimately hold state on it.

        if dry_run:
            return _do_extract_dry_run(
                conn=conn,
                reader=reader,
                source_name=source_name,
                snapshot_path=snapshot_path,
                snapshot_sha256=snapshot_sha256,
                watermark_before=watermark_before,
                last_run_start=last_run_start,
                snapshot_max_rowid=snapshot_max_rowid,
                imsg_dump_binary=imsg_dump_binary,
                run_imsg_dump_fn=run_imsg_dump_fn,
                merge_mode=merge_mode,
            )

        run_id = _begin_extraction_run(
            conn,
            source_name=source_name,
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha256,
            rowid_before=watermark_before,
            merge_mode=merge_mode,
        )

        try:
            result = _do_extract(
                conn=conn,
                reader=reader,
                source_name=source_name,
                snapshot_path=snapshot_path,
                watermark_before=watermark_before,
                last_run_start=last_run_start,
                snapshot_max_rowid=snapshot_max_rowid,
                run_id=run_id,
                imsg_dump_binary=imsg_dump_binary,
                run_imsg_dump_fn=run_imsg_dump_fn,
                merge_mode=merge_mode,
            )
        except Exception as exc:
            _fail_extraction_run(conn, run_id)
            if isinstance(exc, ExtractionError):
                raise
            raise ExtractionError(f"extraction run {run_id} for source '{source_name}' failed: {exc}") from exc

        return result
    finally:
        snapshot.close()


def _begin_extraction_run(
    conn: psycopg.Connection,
    *,
    source_name: str,
    snapshot_path: Path,
    snapshot_sha256: str,
    rowid_before: int,
    merge_mode: MergeMode,
) -> int:
    # The mode is recorded when the run starts, so a run that fails still
    # says which rule it was applying (migration 0006).
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO extraction_run
                (source_name, snapshot_path, snapshot_sha256, rowid_before, status, merge_mode)
            VALUES (%s, %s, %s, %s, 'running', %s)
            RETURNING run_id
            """,
            (source_name, str(snapshot_path), snapshot_sha256, rowid_before, merge_mode.value),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _fail_extraction_run(conn: psycopg.Connection, run_id: int) -> None:
    # clock_timestamp() for the same reason as the success path in
    # _do_extract: this helper opens its own short transaction today, but the
    # stamp must mean "wall clock at failure" whatever transaction it lands in.
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "UPDATE extraction_run SET status = 'failed', finished_at = clock_timestamp() "
            "WHERE run_id = %s",
            (run_id,),
        )


def _do_extract(
    *,
    conn: psycopg.Connection,
    reader: SnapshotReader,
    source_name: str,
    snapshot_path: Path,
    watermark_before: int,
    last_run_start: datetime,
    snapshot_max_rowid: int,
    run_id: int,
    imsg_dump_binary: Path,
    run_imsg_dump_fn: RunImsgDumpFn,
    merge_mode: MergeMode,
) -> ExtractResult:
    chats = reader.fetch_chats()
    handles = reader.fetch_handles()
    chat_handle_joins = reader.fetch_chat_handle_joins()

    last_run_start_ns = _datetime_to_apple_ns(last_run_start)
    # D13: the rows this snapshot links to no chat, including the ones below
    # the watermark that still need filing, are chosen a chat *before* the
    # decoder runs, so `imsg-dump` reaches back only as far as they need.
    unlinked_plan = _plan_unlinked_filing(
        conn,
        reader,
        chats=chats,
        handles=handles,
        targets=reader.fetch_target_messages(watermark_before, last_run_start_ns),
        watermark=watermark_before,
    )
    target_messages = unlinked_plan.targets
    target_rowids = [m.rowid for m in target_messages]

    dump_since_rowid = watermark_before
    if target_rowids:
        dump_since_rowid = min(watermark_before, min(target_rowids) - 1)

    dump_run = run_imsg_dump_fn(imsg_dump_binary, snapshot_path, dump_since_rowid)
    dump_by_guid: dict[str, ImsgDumpMessage] = {m.guid: m for m in dump_run.messages}

    attachments, attachments_by_message = reader.fetch_attachments_for_messages(target_rowids)

    with conn.transaction(), conn.cursor() as cur:
        # What a segment renders through a message but stores elsewhere.
        # Changing it moves no `message.updated_at` by itself, so it is
        # collected here and marked once, at the end of the run
        # (`_mark_for_resegmentation`).
        rerender = _Rerender()

        chat_id_by_rowid: dict[int, int] = {}
        chat_tally = _UpsertTally()
        for chat in chats:
            chat_row = _upsert_chat(cur, chat, merge_mode)
            chat_id_by_rowid[chat.rowid] = chat_row.row_id
            chat_tally.record(chat_row.outcome)
            if chat_row.renders_changed:
                rerender.chat_ids.add(chat_row.row_id)

        group_ids_inserted, group_ids_present = record_group_ids(
            cur,
            ((group_id, chat_id_by_rowid[chat.rowid]) for chat in chats for group_id in chat.group_ids),
        )

        handle_id_by_rowid: dict[int, int] = {}
        handle_tally = _UpsertTally()
        for handle in handles:
            handle_id_by_rowid[handle.rowid], outcome = _upsert_source_handle(cur, handle)
            handle_tally.record(outcome)

        participant_tally = _UpsertTally()
        for chat_rowid, handle_rowid in chat_handle_joins:
            chat_id = chat_id_by_rowid.get(chat_rowid)
            source_handle_id = handle_id_by_rowid.get(handle_rowid)
            if chat_id is not None and source_handle_id is not None:
                cur.execute(
                    "INSERT INTO chat_participant_source (chat_id, source_handle_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (chat_id, source_handle_id),
                )
                participant_tally.record(_inserted_or_unchanged(cur))

        attachment_id_by_rowid: dict[int, int] = {}
        attachment_tally = _UpsertTally()
        attachment_source_tally = _UpsertTally()
        attachment_location_tally = _UpsertTally()
        for att in attachments:
            att_row = _upsert_attachment(cur, att, merge_mode)
            attachment_id_by_rowid[att.rowid] = att_row.row_id
            attachment_tally.record(att_row.outcome)
            if att_row.renders_changed:
                rerender.attachment_ids.add(att_row.row_id)
            attachment_source_tally.record(
                _upsert_attachment_source(cur, source_name, att.rowid, att_row.row_id)
            )
            # Every source's recorded path is kept, under the source's name
            # (D12, D13): `attachment.source_path` holds only one, and the
            # attachment fetcher tries the others.
            if att.source_path:
                attachment_location_tally.record(
                    UpsertOutcome.INSERTED
                    if record_recorded_path(
                        cur, attachment_id=att_row.row_id, source_name=source_name,
                        path=att.source_path,
                    )
                    else UpsertOutcome.UNCHANGED
                )

        message_tally = _UpsertTally()
        message_source_tally = _UpsertTally()
        version_tally = _UpsertTally()
        link_tally = _UpsertTally()
        link_preview_tally = _UpsertTally()
        tapback_tally = _UpsertTally()
        tapbacks_upserted = 0
        system_messages_skipped = 0
        link_previews_upserted = 0
        bodies_missing = 0
        bodies_kept_as_history = 0

        # D13: where the index already has each unlinked target, and every
        # message it filed on weaker evidence than a chat link (the only
        # ones a run may refile).
        unlinked = _UnlinkedTally()
        unlinked.add("rescanned", unlinked_plan.rescanned)
        filings, _ = fetch_indexed_filings(
            cur, (m.guid for m in target_messages if m.rowid in unlinked_plan.choices)
        )
        weakly_filed = fetch_weakly_filed(cur)
        resolver = _ChatResolver(
            cur,
            chat_ids={chat.guid: chat_id_by_rowid[chat.rowid] for chat in chats},
            handles=handles,
            handle_id_by_rowid=handle_id_by_rowid,
            chat_tally=chat_tally,
            participant_tally=participant_tally,
            handle_tally=handle_tally,
            unlinked=unlinked,
        )

        for msg in target_messages:
            dump_msg = dump_by_guid.get(msg.guid)
            if dump_msg is None:
                bodies_missing += 1
                logger.warning("extract.body_missing_from_dump", guid=msg.guid, rowid=msg.rowid)

            if dump_msg is not None and dump_msg.tapback is not None:
                tapback_row, target_message_id = _upsert_tapback(
                    cur, msg, dump_msg, handle_id_by_rowid, merge_mode
                )
                tapback_tally.record(tapback_row.outcome)
                tapbacks_upserted += 1
                if target_message_id is not None and (
                    tapback_row.outcome is UpsertOutcome.INSERTED or tapback_row.renders_changed
                ):
                    rerender.message_ids.add(target_message_id)
                continue

            if msg.item_type != 0:
                system_messages_skipped += 1
                continue

            # Which chat, and on what evidence (D13). A chat link wins; a row
            # without one is filed by `imsg.stages.unlinked_filing`'s rules.
            # A message the index already has keeps its chat (`chat_id` is
            # insert-only) unless `refile_message` below moves it out of a
            # holding chat, so a chat is only resolved -- and a 1:1 or
            # holding chat only created -- where the message will land in it.
            linked_chat_id = (
                chat_id_by_rowid.get(msg.chat_rowid) if msg.chat_rowid is not None else None
            )
            choice = unlinked_plan.choices.get(msg.rowid)
            refile = False
            if linked_chat_id is not None:
                chat_id, evidence = linked_chat_id, ChatEvidence.CHAT_MESSAGE_JOIN
                stored_evidence = weakly_filed.get(msg.guid, evidence)
                refile = stored_evidence is not None and evidence.rank > stored_evidence.rank
            elif choice is None or msg.date is None:
                # `message.sent_at` is NOT NULL: a row with no date cannot be
                # filed. Counted and logged, as every unlinked row used to be.
                unlinked.add("skipped_without_date")
                logger.warning("extract.unlinked_message_without_date", guid=msg.guid, rowid=msg.rowid)
                continue
            else:
                evidence = choice.evidence
                filing = filings.get(msg.guid)
                if filing is None:
                    chat_id = resolver.chat_for(msg, choice)
                elif refile_wanted(choice, filing):
                    chat_id, refile = resolver.chat_for(msg, choice), True
                else:
                    chat_id = filing.chat_id
            if msg.deleted_at is not None:
                unlinked.add("in_recently_deleted")

            message_row = _upsert_message(
                cur, msg, dump_msg, chat_id=chat_id, chat_evidence=evidence,
                handle_id_by_rowid=handle_id_by_rowid,
                has_attachments=bool(attachments_by_message.get(msg.rowid)),
                merge_mode=merge_mode,
            )
            message_id = message_row.row_id
            message_tally.record(message_row.outcome)
            # A message inserted by this run is unsegmented, so everything
            # below that changes how it renders needs no mark of its own.
            existed = message_row.outcome is not UpsertOutcome.INSERTED
            if not existed and evidence is not ChatEvidence.CHAT_MESSAGE_JOIN:
                unlinked.add(evidence.value)
            elif existed and refile:
                refiled = refile_message(
                    cur, message_id=message_id, chat_id=chat_id, evidence=evidence
                )
                if refiled is RefileOutcome.MOVED:
                    unlinked.add("moved_from_holding")
                    if evidence is not ChatEvidence.CHAT_MESSAGE_JOIN:
                        unlinked.add(evidence.value)
                elif refiled is RefileOutcome.EVIDENCE_RAISED:
                    unlinked.add("evidence_raised")

            message_source_tally.record(
                _upsert_message_source(cur, message_id, source_name, msg.rowid, run_id)
            )

            if dump_msg is not None:
                for idx, version in enumerate(dump_msg.edit_history):
                    version_row = _upsert_message_version(cur, message_id, idx, version, merge_mode)
                    version_tally.record(version_row.outcome)
                    if existed and version_row.outcome is not UpsertOutcome.UNCHANGED:
                        rerender.message_ids.add(message_id)

            # After this message's own versions, so a body its source
            # already lists as a prior version is not added twice.
            if message_row.displaced_body is not None and _keep_displaced_body(
                cur, message_id, message_row.displaced_body, message_row.displaced_edited_at
            ):
                bodies_kept_as_history += 1

            for ordinal, att_rowid in enumerate(attachments_by_message.get(msg.rowid, [])):
                att_id = attachment_id_by_rowid.get(att_rowid)
                if att_id is not None:
                    cur.execute(
                        "INSERT INTO message_attachment (message_id, attachment_id, ordinal) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (message_id, att_id, ordinal),
                    )
                    link_outcome = _inserted_or_unchanged(cur)
                    link_tally.record(link_outcome)
                    if existed and link_outcome is UpsertOutcome.INSERTED:
                        rerender.message_ids.add(message_id)

            preview = parse_link_preview(msg.payload_data)
            if preview is not None:
                link_preview_tally.record(_upsert_link_preview(cur, message_id, preview, merge_mode))
                link_previews_upserted += 1

        resolved_targets = _backfill_tapback_targets(cur)
        rerender.message_ids.update(resolved_targets)
        messages_marked = _mark_for_resegmentation(cur, rerender)

        cur.execute(
            "INSERT INTO sync_state (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (_watermark_key(source_name), str(snapshot_max_rowid)),
        )

        result = ExtractResult(
            run_id=run_id,
            watermark_before=watermark_before,
            watermark_after=snapshot_max_rowid,
            chats_upserted=chat_tally.freeze().total,
            handles_upserted=handle_tally.freeze().total,
            messages_upserted=message_tally.freeze().total,
            tapbacks_upserted=tapbacks_upserted,
            system_messages_skipped=system_messages_skipped,
            attachments_upserted=attachment_tally.freeze().total,
            link_previews_upserted=link_previews_upserted,
            bodies_missing=bodies_missing,
            dump_stderr_line_count=len(dump_run.stderr_lines),
            chat_upserts=chat_tally.freeze(),
            handle_upserts=handle_tally.freeze(),
            message_upserts=message_tally.freeze(),
            attachment_upserts=attachment_tally.freeze(),
            tapback_upserts=tapback_tally.freeze(),
            message_version_upserts=version_tally.freeze(),
            link_preview_upserts=link_preview_tally.freeze(),
            message_attachment_links=link_tally.freeze(),
            chat_participant_links=participant_tally.freeze(),
            message_source_rows=message_source_tally.freeze(),
            attachment_source_rows=attachment_source_tally.freeze(),
            attachment_location_rows=attachment_location_tally.freeze(),
            merge_mode=merge_mode,
            messages_marked_for_resegmentation=messages_marked,
            bodies_kept_as_history=bodies_kept_as_history,
            tapback_targets_resolved=len(resolved_targets),
            chat_group_ids=UpsertCounts(inserted=group_ids_inserted, unchanged=group_ids_present),
            unlinked=unlinked.freeze(),
        )

        # clock_timestamp(), not now(): now() is frozen at transaction start,
        # and this UPDATE runs inside the transaction opened above, before the
        # whole upsert loop. With now() the row recorded when the loop *began*
        # writing, so every duration read from extraction_run excluded the
        # write phase entirely (2026-09-14: a seed run that worked for about
        # 2m35s recorded 11.9s). started_at is unaffected — its DEFAULT fires
        # in _begin_extraction_run's own short transaction.
        #
        # `messages_upserted` keeps its original meaning — rows this run
        # processed — so anything already reading the column is unaffected.
        # The three columns beside it (migration 0005) are what make that
        # number legible after the fact: without them a run row saying
        # "662,683" cannot be told apart from a run that rewrote 662,683
        # rows, which is the ambiguity that let the state-idempotence
        # defect sit unnoticed. `messages_updated` counts every rewrite,
        # fills included; `upsert_counts` (migration 0006) carries the
        # split for every table, so a real seed run can be checked against
        # "inserts and fills only" after the terminal is gone.
        message_counts = result.message_upserts
        cur.execute(
            """
            UPDATE extraction_run
            SET status = 'ok', finished_at = clock_timestamp(), rowid_after = %s,
                messages_upserted = %s, messages_inserted = %s,
                messages_updated = %s, messages_unchanged = %s,
                upsert_counts = %s::jsonb
            WHERE run_id = %s
            """,
            (
                snapshot_max_rowid,
                message_counts.total,
                message_counts.inserted,
                message_counts.updated,
                message_counts.unchanged,
                json.dumps(
                    {table: counts.as_dict() for table, counts in result.table_counts().items()}
                ),
                run_id,
            ),
        )

    return result


# --------------------------------------------------------------------------
# messages with no chat link (owner decision D13)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _UnlinkedPlan:
    targets: list[MessageRow]
    """This run's targets, in ROWID order: the usual ones plus every row
    below the watermark the rescan selected."""
    choices: dict[int, ChatChoice]
    """Where each target with no chat link goes, by ROWID."""
    rescanned: int


def _chat_kind_for_style(style: int | None) -> str | None:
    if style == CHAT_STYLE_GROUP:
        return "group"
    if style == CHAT_STYLE_DM:
        return "dm"
    return None


def _plan_unlinked_filing(
    conn: psycopg.Connection,
    reader: SnapshotReader,
    *,
    chats: Sequence[ChatRow],
    handles: Sequence[HandleRow],
    targets: Sequence[MessageRow],
    watermark: int,
) -> _UnlinkedPlan:
    """Choose a chat for every target this snapshot links to no chat, and
    add the rows below the watermark that still need filing
    (`imsg.stages.unlinked_filing`, "Rescan"). Reads only: the snapshot's
    unlinked rows, and the index's filings of them and of the group ids
    they carry. Runs before `imsg-dump`, so the decoder reaches back only
    to the lowest ROWID that has work.

    A group id is matched against the chats this snapshot has and every
    chat the index recorded one for (`chat_group_id`), so a group that
    another source showed still counts, and two chats carrying one id are
    ambiguity, not a match."""
    in_scope = {m.rowid for m in targets}
    candidates = [m for m in reader.fetch_unlinked_messages(watermark) if m.rowid not in in_scope]
    unlinked = [m for m in targets if m.chat_rowid is None] + candidates
    if not unlinked:
        return _UnlinkedPlan(targets=list(targets), choices={}, rescanned=0)

    chat_guid_by_rowid = {chat.rowid: chat.guid for chat in chats}
    sender_by_rowid = {handle.rowid: handle.raw_value for handle in handles}
    groups = GroupDirectory()
    for chat in chats:
        for group_id in chat.group_ids:
            groups.add(group_id, chat.guid, _chat_kind_for_style(chat.style))
    with conn.cursor() as cur:
        fetch_group_chats(
            cur, (m.ck_chat_id for m in unlinked if m.ck_chat_id and is_group_id(m.ck_chat_id)), groups
        )
        filings, tapbacks = fetch_indexed_filings(cur, (m.guid for m in candidates))

    choices: dict[int, ChatChoice] = {}
    for m in unlinked:
        sender = None
        if not m.is_from_me and m.handle_rowid is not None:
            sender = sender_by_rowid.get(m.handle_rowid)
        recoverable = (
            chat_guid_by_rowid.get(m.recoverable_chat_rowid)
            if m.recoverable_chat_rowid is not None
            else None
        )
        choices[m.rowid] = choose_chat(
            UnlinkedMessage(
                rowid=m.rowid,
                is_from_me=m.is_from_me,
                sender=sender,
                ck_chat_id=m.ck_chat_id,
                recoverable_chat_guid=recoverable,
            ),
            groups,
        )

    selected = [
        m
        for m in candidates
        if rescan_wanted(
            choices[m.rowid], filings.get(m.guid), deleted_at=m.deleted_at, is_tapback=m.guid in tapbacks
        )
    ]
    kept = in_scope | {m.rowid for m in selected}
    return _UnlinkedPlan(
        targets=sorted([*targets, *selected], key=lambda m: m.rowid),
        choices={rowid: choice for rowid, choice in choices.items() if rowid in kept},
        rescanned=len(selected),
    )


class _ChatResolver:
    """The chat a D13 choice names, as a `chat_id`, inside S2's
    transaction. A chat this snapshot has was upserted already; one only
    the index has is looked up; a 1:1 chat or a holding chat the index
    lacks is created, with the participants it is known to have:

    * a holding chat: every incoming sender filed into it;
    * a 1:1 chat created from a `ck_chat_id`: the handle the id names
      (the sender, or for the owner's own message a source handle this
      snapshot has for it, created if it has none).

    A 1:1 chat that already existed gets no participant from here: its
    own source's `chat_handle_join` supplied them, and a differently
    written copy of the same handle would only add a stub person."""

    def __init__(
        self,
        cur: psycopg.Cursor[Any],
        *,
        chat_ids: dict[str, int],
        handles: Sequence[HandleRow],
        handle_id_by_rowid: dict[int, int],
        chat_tally: _UpsertTally,
        participant_tally: _UpsertTally,
        handle_tally: _UpsertTally,
        unlinked: _UnlinkedTally,
    ) -> None:
        self._cur = cur
        self._chat_ids = dict(chat_ids)
        self._handle_id_by_rowid = handle_id_by_rowid
        self._chat_tally = chat_tally
        self._participant_tally = participant_tally
        self._handle_tally = handle_tally
        self._unlinked = unlinked
        self._handle_ids_by_raw: dict[str, dict[str, int]] = {}
        for handle in handles:
            source_handle_id = handle_id_by_rowid.get(handle.rowid)
            if source_handle_id is not None:
                self._handle_ids_by_raw.setdefault(handle.raw_value, {})[
                    _normalize_service(handle.service)
                ] = source_handle_id

    def chat_for(self, msg: MessageRow, choice: ChatChoice) -> int:
        cached = self._chat_ids.get(choice.chat_guid)
        created = False
        if cached is not None:
            chat_id = cached
        else:
            chat_id, created = ensure_chat(
                self._cur, choice, service=_service_evidence(choice.ck_service)
            )
            self._chat_ids[choice.chat_guid] = chat_id
            if created:
                # Reported with the snapshot's own chats too (`table=chat`):
                # D12 rule 5, every table the run writes shows its inserts.
                self._chat_tally.record(UpsertOutcome.INSERTED)
                self._unlinked.add(
                    "holding_chats_created" if choice.unfiled_key is not None else "chats_created"
                )
        participant = self._participant(msg, choice, created=created)
        if participant is not None:
            self._cur.execute(
                "INSERT INTO chat_participant_source (chat_id, source_handle_id) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (chat_id, participant),
            )
            self._participant_tally.record(_inserted_or_unchanged(self._cur))
        return chat_id

    def _participant(self, msg: MessageRow, choice: ChatChoice, *, created: bool) -> int | None:
        sender = None
        if not msg.is_from_me and msg.handle_rowid is not None:
            sender = self._handle_id_by_rowid.get(msg.handle_rowid)
        if choice.unfiled_key is not None:
            return sender
        if not created or not choice.ck_handle:
            return None
        if sender is not None:
            # `ck_1to1` requires the sender to be the id's handle.
            return sender
        return self._source_handle(choice.ck_handle, choice.ck_service)

    def _source_handle(self, raw_value: str, raw_service: str | None) -> int:
        known = self._handle_ids_by_raw.get(raw_value, {})
        service = _normalize_service(raw_service)
        if service in known:
            return known[service]
        if known:
            return known[min(known)]
        source_handle_id, outcome = _upsert_source_handle(
            self._cur, HandleRow(rowid=0, raw_value=raw_value, service=raw_service)
        )
        self._handle_tally.record(outcome)
        self._handle_ids_by_raw.setdefault(raw_value, {})[service] = source_handle_id
        return source_handle_id


# --------------------------------------------------------------------------
# merge policy: what each source is allowed to overwrite
# --------------------------------------------------------------------------
#
# INVARIANT (2026-09-17): a source that carries no evidence for a column
# must never overwrite a value another source supplied -- extraction
# writes a column only where the incoming snapshot positively asserts
# it, so absence of evidence is never recorded as evidence of absence.
#
# Several witnesses of the same conversations feed this index (each
# Mac's own `chat.db`, plus a recovered seed), and they do not carry
# the same information. Until this date every `ON CONFLICT DO UPDATE`
# assigned each column from the incoming row unconditionally, so the
# *last* source to run won every column, including the columns it
# simply could not see. Measured on the target host after the mini
# re-extracted its own `chat.db` (compared against a pre-run dump of
# the same index): 14,282 messages lost `has_attachments`, 2,203 lost
# `is_edited` and its `date_edited`, 263 lost `is_unsent`, 3,125 chats
# lost `display_name`, and 4,775 attachments lost `source_path` --
# every one of them because the incoming snapshot was silent, not
# because anything about those messages had changed.
#
# INVARIANT (D12, owner decision 2026-09-23): merges only add. A run that
# is not this machine's own live database (`MergeMode.SEED`) inserts rows
# and fills empty values; it never replaces a non-empty value. Only the
# live run may, and a message body changes only when it is empty or a
# strictly newer edit brings the new text -- from any source. An empty
# string, a size of 0 or the `unknown` service marker is evidence from
# nobody. The 2026-09-17 rule above stopped NULL from overwriting, but
# `''` and 0 still counted as values, so the last source to run still
# won: a dry run on the target host of a recovery candidate built on an
# older corpus would have put the older values back over 526 messages,
# 32,971 attachment rows and 365 chats -- one group name blanked to `''`
# -- while `imsg extract` printed only the message counts.
#
# The rule is per column and declared once, in the `_Column` tables
# below. Each column answers two questions: which of its values say
# nothing (`Empty`), and who may change a stored value (`Merge`, bounded
# by the run's `MergeMode`).


class Empty(StrEnum):
    """Which values of a column say nothing (D12: an empty string or a 0
    is not evidence, from any source).

    Applied twice, the same way: to what a source sends (is it evidence
    at all?) and to what the index holds (is there anything to lose, or
    only something to fill?). `_is_empty` and `_empty_sql` are the one
    definition of each kind, in Python and in SQL."""

    NULL = "null"
    """Only NULL: every non-NULL value means something. Timestamps
    (`chat.db`'s 0 date is already NULL by the time it gets here),
    surrogate ids (they start at 1), enums without a marker value, and
    the two `ASSERTED` columns, where `false` is a real answer."""

    BLANK = "blank"
    """NULL or `''`. Names, paths, types and bodies: an empty string
    names nothing that NULL does not. `chat.db` stores `''` as the name
    of a group nobody named."""

    NON_POSITIVE = "non_positive"
    """NULL, 0 or less. Sizes: `chat.db` records 0 for a transfer that
    never completed. The price is that a genuinely empty file's size can
    be filled in by another source's; nothing is ever lost to it."""

    UNKNOWN = "unknown"
    """NULL or `'unknown'`, the `service_kind` value this module records
    when `chat.db` names no service it recognizes."""

    FALSE = "false"
    """NULL or `false`: the empty value of every `POSITIVE` flag, whose
    `false` is what a source reports when it cannot see the thing."""


def _is_empty(kind: Empty, value: Any) -> bool:
    if value is None:
        return True
    if kind is Empty.BLANK:
        return bool(value == "")
    if kind is Empty.NON_POSITIVE:
        return bool(value <= 0)
    if kind is Empty.UNKNOWN:
        return bool(value == "unknown")
    if kind is Empty.FALSE:
        return not value
    return False


def _empty_sql(kind: Empty, ref: str) -> str:
    """`_is_empty`, as a SQL boolean over the column reference `ref`.
    Never NULL itself, so it can be negated safely."""
    if kind is Empty.BLANK:
        return f"({ref} IS NULL OR {ref} = '')"
    if kind is Empty.NON_POSITIVE:
        return f"({ref} IS NULL OR {ref} <= 0)"
    if kind is Empty.UNKNOWN:
        return f"({ref} IS NULL OR {ref} = 'unknown')"
    if kind is Empty.FALSE:
        return f"({ref} IS NULL OR NOT {ref})"
    return f"({ref} IS NULL)"


class Merge(StrEnum):
    """How one column of an already-present row may be rewritten.

    Every rule is bounded by the run's `MergeMode` and by `Empty`: an
    empty incoming value is never written over a stored one, and only a
    `LIVE` run may replace a non-empty value (for `ASSERTED`, `PRESENT`
    and `POSITIVE` columns). A `SEED` run fills."""

    ASSERTED = "asserted"
    """Every source that has the row carries this column, so a `LIVE`
    run writes whatever it says -- `false` included. A `SEED` run never
    moves it: the column is NOT NULL, so there is nothing to fill.
    `message.sent_at` and `is_from_me` are the two."""

    PRESENT = "present"
    """An empty value (the column's `Empty`) means "this source has
    nothing here". A non-empty value replaces the stored one in a `LIVE`
    run (a renamed chat, a moved attachment) and fills an empty one in
    any run."""

    POSITIVE = "positive"
    """A boolean naming the presence of something (`has_attachments`,
    `is_edited`, `is_unsent`, `is_sticker`, `removed`). Only `true`
    carries evidence: `false` is what a source reports both when the
    thing is genuinely absent and when the source cannot see it -- a
    `chat.db` whose `message_attachment_join` rows never synced, or a
    row whose typedstream blob is gone, reports exactly the same
    `false` as a plain text message. All five also name facts that do
    not un-happen: a message that was edited, retracted or given an
    attachment stays that way, so `true -> false` is never a
    correction being blocked, and `false -> true` is a fill that any
    source may make."""

    BODY = "body"
    """A message body and its normalized copy, decided together on the
    table's `_BodyRule.text` column so the two can never come from
    different sources. Filled when the stored body is empty; replaced
    only by a strictly newer edit (`_BodyRule.edited_at`), from any
    source, `LIVE` or `SEED`. A same-or-older edit never replaces a
    body, and neither does an empty one."""

    EDIT_TIME = "edit_time"
    """The edit time a body goes with. It moves forward only with a body
    a newer edit brought, or when the body is unchanged -- never ahead
    of a body the index does not hold, which would make the source that
    does carry that edit's text look same-time and be refused."""

    HISTORY = "history"
    """A prior version of an edited message. History only grows: an
    empty value is filled by any source, and a non-empty one is never
    replaced, the live run included -- a different text at the same
    position is a different decode, not a correction."""

    INSERT_ONLY = "insert_only"
    """First write wins: recorded on insert as provenance, never
    updated. Keys and derived keys, plus the columns documented on
    `_upsert_message` as deliberately frozen."""


@dataclass(frozen=True, slots=True)
class _Column:
    """One column of a guarded upsert, and how it merges."""

    name: str
    merge: Merge
    empty: Empty = Empty.NULL
    cast: str = ""
    """Postgres cast appended to the VALUES placeholder, e.g.
    `::service_kind`. Only needed where the parameter is a bare Python
    string feeding an enum column."""
    insert_default: str | None = None
    """SQL fragment supplying the INSERT value when the parameter is
    NULL, for a column that the schema declares NOT NULL (a NULL there
    means "no evidence", but the row still has to be insertable).
    Unused on the UPDATE path, which keeps the stored value instead."""
    renders: bool = True
    """Whether a segment shows this column. Consulted for the tables a
    segment renders through a message (chat, attachment, tapback,
    edit history): when one of their rendered columns changes, the
    messages that show it are marked for re-segmentation, because no
    `message` row moves by itself (see `_mark_for_resegmentation`)."""

    def __post_init__(self) -> None:
        if (self.merge is Merge.POSITIVE) != (self.empty is Empty.FALSE):
            raise ValueError(
                f"column {self.name!r}: POSITIVE flags, and only they, have Empty.FALSE"
            )


@dataclass(frozen=True, slots=True)
class _BodyRule:
    """The columns `Merge.BODY` and `Merge.EDIT_TIME` are decided on."""

    text: str
    """Whose emptiness and value decide the body: `text_original`."""
    edited_at: str
    """The edit time the body goes with: `date_edited`."""


_LIVE_PARAM = "run__live"
_LIVE = f"%({_LIVE_PARAM})s::boolean"


def _known_param(column: str) -> str:
    return f"{column}__known"


def _known(column: str) -> str:
    """True when the incoming value of `column` is evidence."""
    return f"%({_known_param(column)})s::boolean"


def _evidence_flags(columns: Sequence[_Column], values: dict[str, Any]) -> dict[str, Any]:
    """Derive one `<column>__known` boolean per assigned column from the
    policy declared on the column, not from anything the call site
    remembers to pass: a value is evidence unless its `Empty` kind says
    it is not. Computed from the parameters rather than from `EXCLUDED`,
    because `EXCLUDED` already carries a column's `insert_default` in
    place of a NULL, and a default is not evidence."""
    return {
        _known_param(column.name): not _is_empty(column.empty, values[column.name])
        for column in columns
        if column.merge is not Merge.INSERT_ONLY
    }


def _newer_edit_sql(table: str, rule: _BodyRule) -> str:
    stored = f"{table}.{rule.edited_at}"
    return (
        f"({_known(rule.edited_at)} AND "
        f"({stored} IS NULL OR EXCLUDED.{rule.edited_at} > {stored}))"
    )


def _body_write_sql(table: str, rule: _BodyRule) -> str:
    stored_empty = _empty_sql(Empty.BLANK, f"{table}.{rule.text}")
    return f"({_known(rule.text)} AND ({stored_empty} OR {_newer_edit_sql(table, rule)}))"


def _edit_time_write_sql(table: str, rule: _BodyRule) -> str:
    same_body = f"EXCLUDED.{rule.text} IS NOT DISTINCT FROM {table}.{rule.text}"
    return f"({_newer_edit_sql(table, rule)} AND ({_body_write_sql(table, rule)} OR {same_body}))"


def _write_sql(table: str, column: _Column, body: _BodyRule | None) -> str | None:
    """When a conflicting row's `column` takes the incoming value: the
    whole merge rule for one column, as one SQL boolean over the stored
    row (`<table>.col`), the incoming row (`EXCLUDED.col`) and the
    parameters `_execute_upsert` binds. None for a column that is never
    updated."""
    stored_empty = _empty_sql(column.empty, f"{table}.{column.name}")
    if column.merge is Merge.INSERT_ONLY:
        return None
    if column.merge is Merge.ASSERTED:
        return f"({_LIVE} OR {stored_empty})"
    if column.merge in (Merge.PRESENT, Merge.POSITIVE):
        return f"({_known(column.name)} AND ({_LIVE} OR {stored_empty}))"
    if column.merge is Merge.HISTORY:
        return f"({_known(column.name)} AND {stored_empty})"
    if body is None:
        raise ValueError(f"{table}.{column.name}: {column.merge.value} needs a _BodyRule")
    if column.merge is Merge.BODY:
        return _body_write_sql(table, body)
    return _edit_time_write_sql(table, body)


def _set_clause(table: str, column: _Column, body: _BodyRule | None) -> str | None:
    """The `SET` fragment for one column, or None when the column is
    never updated."""
    write = _write_sql(table, column, body)
    if write is None:
        return None
    return (
        f"{column.name} = CASE WHEN {write} "
        f"THEN EXCLUDED.{column.name} ELSE {table}.{column.name} END"
    )


def _differs_clause(table: str, column: _Column, body: _BodyRule | None) -> str | None:
    """The `WHERE` disjunct for one column: the row is rewritten only
    when this column would actually change.

    The compared columns are exactly the assigned columns, and the
    comparison is `IS DISTINCT FROM`, never `<>` -- both rules are the
    2026-09-17 state-idempotence fix and both still hold here, with the
    column's write rule ANDed in so a value the rule refuses is not
    merely left unwritten but also cannot drag the row into an UPDATE
    (which would move `updated_at` and re-segment the chat for nothing).
    """
    write = _write_sql(table, column, body)
    if write is None:
        return None
    return f"({write} AND {table}.{column.name} IS DISTINCT FROM EXCLUDED.{column.name})"


def _replaced_sql(column: _Column) -> str:
    """True when the statement replaced a non-empty stored value of
    `column`; `o` is the row before the statement, `u` the row after."""
    before = f"o.{column.name}"
    return (
        f"({before} IS DISTINCT FROM u.{column.name} "
        f"AND NOT {_empty_sql(column.empty, before)})"
    )


def _any_sql(parts: Iterable[str]) -> str:
    joined = " OR ".join(parts)
    return f"({joined})" if joined else "false"


def _build_upsert_sql(
    *,
    table: str,
    conflict: Sequence[str],
    columns: Sequence[_Column],
    returning: str,
    body: _BodyRule | None = None,
    extra_set: str = "",
) -> str:
    """Render one guarded, provenance-respecting upsert statement.

    Built once per table at import time, so the SQL below is a constant
    a reader can print, not a string assembled per row. The shape, and
    what each returned column means, is documented on
    `_read_upsert_row`; `returning` names the column that comes back as
    the row id (the table's own key, or the parent message for the
    tables keyed by one).
    """
    assigned = [column for column in columns if column.merge is not Merge.INSERT_ONLY]
    set_clauses = [clause for clause in (_set_clause(table, c, body) for c in assigned) if clause]
    where_clauses = [
        clause for clause in (_differs_clause(table, c, body) for c in assigned) if clause
    ]
    if extra_set:
        set_clauses.append(extra_set)
    assert set_clauses, f"{table}: nothing to update — use ON CONFLICT DO NOTHING instead"

    insert_columns = ", ".join(column.name for column in columns)
    insert_values = ", ".join(_insert_value_sql(column) for column in columns)
    conflict_columns = ", ".join(conflict)
    tracked = ", ".join(column.name for column in assigned)
    match = " AND ".join(f"{col} = %({col})s" for col in conflict)

    edit_rules = (Merge.BODY, Merge.EDIT_TIME)
    replaced = _any_sql(_replaced_sql(c) for c in assigned if c.merge not in edit_rules)
    newer_edit = _any_sql(_replaced_sql(c) for c in assigned if c.merge in edit_rules)
    renders = _any_sql(
        f"o.{c.name} IS DISTINCT FROM u.{c.name}" for c in assigned if c.renders
    )

    displaced_after, displaced_unchanged = "", ""
    if body is not None:
        text_replaced = (
            f"NOT u.inserted AND o.{body.text} IS DISTINCT FROM u.{body.text} "
            f"AND NOT {_empty_sql(Empty.BLANK, 'o.' + body.text)}"
        )
        displaced_after = (
            f",\n       CASE WHEN {text_replaced} THEN o.{body.text} END"
            f",\n       CASE WHEN {text_replaced} THEN o.{body.edited_at} END"
        )
        displaced_unchanged = ", NULL::text, NULL::timestamptz"

    return (
        f"WITH old AS (\n"
        f"    SELECT {returning}, {tracked} FROM {table} WHERE {match}\n"
        f"),\n"
        f"upserted AS (\n"
        f"    INSERT INTO {table} ({insert_columns})\n"
        f"    VALUES ({insert_values})\n"
        f"    ON CONFLICT ({conflict_columns}) DO UPDATE SET\n        "
        + ",\n        ".join(set_clauses)
        + "\n    WHERE "
        + "\n       OR ".join(where_clauses)
        + f"\n    RETURNING {returning}, (xmax = 0) AS inserted, {tracked}\n"
        f")\n"
        f"SELECT u.{returning}, u.inserted, true,\n"
        f"       NOT u.inserted AND {replaced},\n"
        f"       NOT u.inserted AND {newer_edit},\n"
        f"       NOT u.inserted AND {renders}{displaced_after}\n"
        f"FROM upserted u LEFT JOIN old o ON true\n"
        f"UNION ALL\n"
        f"SELECT o.{returning}, false, false, false, false, false{displaced_unchanged}\n"
        f"FROM old o WHERE NOT EXISTS (SELECT 1 FROM upserted)"
    )


def _insert_value_sql(column: _Column) -> str:
    placeholder = f"%({column.name})s{column.cast}"
    if column.insert_default is None:
        return placeholder
    return f"COALESCE({placeholder}, {column.insert_default})"


@dataclass(frozen=True, slots=True)
class _UpsertRow:
    """What one guarded upsert did, read back by `_read_upsert_row`."""

    row_id: int
    outcome: UpsertOutcome
    renders_changed: bool
    """An existing row changed a column a segment renders
    (`_Column.renders`). Always false for an inserted row."""
    displaced_body: str | None = None
    displaced_edited_at: datetime | None = None
    """For the message upsert only: the non-empty body this statement
    replaced, and the edit time it went with. `_keep_displaced_body`
    keeps it as edit history."""


def _execute_upsert(
    cur: psycopg.Cursor[Any],
    sql: str,
    columns: Sequence[_Column],
    values: dict[str, Any],
    merge_mode: MergeMode,
) -> _UpsertRow:
    """Run one built statement, adding the evidence flags the policy
    implies and the run's mode. Call sites pass values only; they never
    hand-write a flag, which is what keeps the rule in one place."""
    cur.execute(
        sql,
        {
            **values,
            **_evidence_flags(columns, values),
            _LIVE_PARAM: merge_mode is MergeMode.LIVE,
        },
    )
    return _read_upsert_row(cur)


# --------------------------------------------------------------------------
# Postgres upsert helpers
# --------------------------------------------------------------------------


def _read_upsert_row(cur: psycopg.Cursor[Any]) -> _UpsertRow:
    """Read the single row every guarded upsert below returns.

    The statements are all shaped the same way:

        WITH old AS (SELECT <pk>, <assigned columns> FROM <table> WHERE <conflict key>),
        upserted AS (
            INSERT ... ON CONFLICT (...) DO UPDATE SET ... WHERE <differs>
            RETURNING <pk>, (xmax = 0) AS inserted, <assigned columns>
        )
        SELECT <pk>, inserted, true, <replaced>, <newer edit>, <renders changed> [, <displaced body>]
          FROM upserted u LEFT JOIN old o ON true
        UNION ALL
        SELECT <pk>, false, false, false, false, false [, NULL, NULL]
          FROM old o WHERE NOT EXISTS (SELECT 1 FROM upserted)

    Every sub-statement runs against the same snapshot, so `old` is the
    row as it was before this statement, whatever the upsert did to it.
    Comparing it with the `RETURNING` row is what tells a fill (every
    changed value was empty before) from a replacement, per column,
    using each column's own `Empty` kind.

    The trailing branch exists because a WHERE-guarded `DO UPDATE` that
    skips the row returns nothing — and the caller still needs the row
    id. It reads `old`, which cannot see the CTE's own insert; that is
    fine, because when the CTE inserts, the `NOT EXISTS` is false and
    the branch contributes nothing. Exactly one row comes back either
    way.

    `xmax = 0` distinguishes the inserted row from the updated one: an
    `INSERT ... ON CONFLICT DO UPDATE` leaves `xmax` set to the updating
    transaction on the update path and zero on the insert path. All of
    this is read-only bookkeeping — a misclassification would mis-report
    a count, never write a wrong row — and it is asserted directly by
    `tests/test_extract_unchanged_rows_stay_untouched_integration.py`
    and `tests/test_extract_merges_only_add_integration.py`.
    """
    row = cur.fetchone()
    assert row is not None, "guarded upsert returned no row — the fallback branch is wrong"
    row_id, inserted, written = int(row[0]), bool(row[1]), bool(row[2])
    replaced, newer_edit, renders_changed = bool(row[3]), bool(row[4]), bool(row[5])
    if not written:
        outcome = UpsertOutcome.UNCHANGED
    elif inserted:
        outcome = UpsertOutcome.INSERTED
    elif replaced:
        outcome = UpsertOutcome.REPLACED
    elif newer_edit:
        outcome = UpsertOutcome.NEWER_EDIT
    else:
        outcome = UpsertOutcome.FILLED
    displaced = tuple(row[6:8]) if len(row) > 6 else (None, None)
    return _UpsertRow(
        row_id=row_id,
        outcome=outcome,
        renders_changed=renders_changed,
        displaced_body=displaced[0],
        displaced_edited_at=displaced[1],
    )


def _inserted_or_unchanged(cur: psycopg.Cursor[Any]) -> UpsertOutcome:
    """For an `ON CONFLICT DO NOTHING` insert just executed on `cur`."""
    return UpsertOutcome.INSERTED if cur.rowcount == 1 else UpsertOutcome.UNCHANGED


_CHAT_COLUMNS: tuple[_Column, ...] = (
    _Column("source_guid", Merge.INSERT_ONLY),
    _Column("thread_key", Merge.INSERT_ONLY),
    # `kind` is read off `chat.style`, a plain column — but only the two
    # documented values mean anything, and an unrecognized one falls back to a
    # participant-count guess. A guess is not evidence: a source whose
    # `chat_handle_join` rows are incomplete would guess `dm` for a group and
    # overwrite a correctly-typed row. So the parameter carries the value only
    # when `style` said it, and the heuristic supplies the INSERT default.
    # Empty: NULL only -- `dm` and `group` are both real answers. Rendered: a
    # group's segment header says so.
    _Column(
        "kind", Merge.PRESENT, cast="::chat_kind",
        insert_default="%(kind_fallback)s::chat_kind",
    ),
    # A device that joined a named group late, or never received the rename
    # event, has NULL here — indistinguishable from a group whose name was
    # cleared. Measured cost of getting this wrong on the target host: 3,125
    # chats lost their `display_name` to a silent re-extraction.
    # Empty: NULL or ''. `chat.db` stores '' for a group nobody named, and a
    # '' from another source is what the 2026-09-22 dry run would have written
    # over a real name. Rendered in every segment header of a group.
    _Column("display_name", Merge.PRESENT, empty=Empty.BLANK),
    # `_service_evidence` returns NULL for the `unknown` bucket, which is the
    # absence marker itself: `chat.service_name` was NULL, or held a string
    # this build does not recognize. `unknown` is still what a first insert
    # records, so it is also what a later source may fill.
    # Empty: NULL or 'unknown'. Not rendered.
    _Column(
        "service", Merge.PRESENT, empty=Empty.UNKNOWN, cast="::service_kind",
        insert_default="'unknown'::service_kind", renders=False,
    ),
)

_CHAT_UPSERT_SQL = _build_upsert_sql(
    table="chat", conflict=("source_guid",), columns=_CHAT_COLUMNS, returning="chat_id"
)


def _upsert_chat(cur: psycopg.Cursor[Any], chat: ChatRow, merge_mode: MergeMode) -> _UpsertRow:
    kind: str | None
    if chat.style == CHAT_STYLE_GROUP:
        kind = "group"
    elif chat.style == CHAT_STYLE_DM:
        kind = "dm"
    else:
        # Unexpected/unknown `style` value: fall back to a participant-count
        # heuristic rather than guessing a magic number wrong (logged so an
        # unexpected chat.db `style` value is visible, not silent). The
        # fallback is good enough to insert a new row with and not good enough
        # to overwrite an existing one, which is what `kind`'s NULL says.
        kind = None
        logger.warning(
            "extract.unexpected_chat_style",
            guid=chat.guid,
            style=chat.style,
            fallback_kind="group" if chat.participant_count > 1 else "dm",
        )

    # `thread_key` is a pure function of `source_guid` (the conflict key),
    # so it can never differ and is neither assigned nor compared.
    # `created_at` is first-write provenance and is likewise left alone.
    return _execute_upsert(
        cur,
        _CHAT_UPSERT_SQL,
        _CHAT_COLUMNS,
        {
            "source_guid": chat.guid,
            "thread_key": thread_key(chat.guid),
            "kind": kind,
            "kind_fallback": "group" if chat.participant_count > 1 else "dm",
            "display_name": chat.display_name,
            "service": _service_evidence(chat.service_name),
        },
        merge_mode,
    )


def _upsert_source_handle(
    cur: psycopg.Cursor[Any], handle: HandleRow
) -> tuple[int, UpsertOutcome]:
    """`(raw_value, service)` IS the whole row apart from the surrogate
    key, so there is nothing an existing row could need corrected —
    hence `DO NOTHING` rather than a guarded `DO UPDATE`. The previous
    `DO UPDATE SET raw_value = EXCLUDED.raw_value` was a no-op write
    performed purely so `RETURNING` would fire; the `UNION ALL` branch
    is what supplies the id now, at no write cost."""
    cur.execute(
        """
        WITH upserted AS (
            INSERT INTO source_handle (raw_value, service)
            VALUES (%(raw_value)s, %(service)s::service_kind)
            ON CONFLICT (raw_value, service) DO NOTHING
            RETURNING source_handle_id
        )
        SELECT source_handle_id, true FROM upserted
        UNION ALL
        SELECT source_handle_id, false FROM source_handle
         WHERE raw_value = %(raw_value)s AND service = %(service)s::service_kind
           AND NOT EXISTS (SELECT 1 FROM upserted)
        """,
        {"raw_value": handle.raw_value, "service": _normalize_service(handle.service)},
    )
    row = cur.fetchone()
    assert row is not None, "handle upsert returned no row — the fallback branch is wrong"
    return int(row[0]), UpsertOutcome.INSERTED if row[1] else UpsertOutcome.UNCHANGED


_ATTACHMENT_COLUMNS: tuple[_Column, ...] = (
    _Column("source_guid", Merge.INSERT_ONLY),
    _Column("attachment_key", Merge.INSERT_ONLY),
    # Empty for the four text columns: NULL or ''. A name, path or type that
    # is the empty string names nothing. Of the four, a segment shows the
    # name and the kind read off the MIME type
    # (`imsg.segment.pipeline._fetch_attachments`); the path and the UTI are
    # not rendered, so changing them re-segments nothing.
    _Column("filename", Merge.PRESENT, empty=Empty.BLANK),
    _Column("source_path", Merge.PRESENT, empty=Empty.BLANK, renders=False),
    _Column("uti", Merge.PRESENT, empty=Empty.BLANK, renders=False),
    _Column("mime_type", Merge.PRESENT, empty=Empty.BLANK),
    # Empty: NULL, 0 or less. `chat.db` records 0 for a transfer that never
    # completed; the 2026-09-22 dry run would have zeroed 29 sizes this way.
    _Column("byte_size", Merge.PRESENT, empty=Empty.NON_POSITIVE, renders=False),
    _Column("is_sticker", Merge.POSITIVE, empty=Empty.FALSE, renders=False),
    # S5a owns both of these from the first insert onwards; the one
    # exception is spelled out in `_ATTACHMENT_STATE_SET` below.
    _Column("state", Merge.INSERT_ONLY, cast="::materialization_state"),
    _Column("materialization_last_error", Merge.INSERT_ONLY),
)

# "Had no path and is given one" is exactly a fill of `source_path`, which
# every run may make; an empty incoming path is not one.
_PATH_ARRIVES = (
    f"attachment.state = 'missing' AND {_empty_sql(Empty.BLANK, 'attachment.source_path')}\n"
    f"             AND {_known('source_path')}"
)

_ATTACHMENT_STATE_SET = f"""state = CASE
        WHEN {_PATH_ARRIVES}
        THEN 'dataless'::materialization_state
        ELSE attachment.state
    END,
    materialization_last_error = CASE
        WHEN {_PATH_ARRIVES}
        THEN NULL
        ELSE attachment.materialization_last_error
    END"""

_ATTACHMENT_UPSERT_SQL = _build_upsert_sql(
    table="attachment",
    conflict=("source_guid",),
    columns=_ATTACHMENT_COLUMNS,
    returning="attachment_id",
    extra_set=_ATTACHMENT_STATE_SET,
)


def _upsert_attachment(
    cur: psycopg.Cursor[Any], att: AttachmentRow, merge_mode: MergeMode
) -> _UpsertRow:
    """`state` is decided here, once, from whether chat.db recorded a
    path at all: with one the row starts `dataless` (S5a's "pending");
    with none there is nothing on disk to read, so it starts `missing`
    with `NO_SOURCE_PATH_ERROR` — calling it `dataless` would promise a
    retry that can never happen. On re-extraction the state is left
    alone (S5a owns it from then on), with one exception: a `missing`
    row that had no path and now has one goes back to `dataless`, so
    a transfer that completed since the last run gets its first
    attempt without anyone hand-editing the row.

    The columns compared below are the chat.db-derived facts this stage
    owns: `filename`, `source_path`, `uti`, `mime_type`, `byte_size`,
    `is_sticker`. Everything else on the row belongs to S5a/S5b
    (`cache_path`, `sha256`, `materialization_attempts`,
    `materialization_next_attempt_at`, the enrichment ladder) and is
    never written here, so it is never compared either.

    `state`/`materialization_last_error` are the one conditional
    assignment, and they need no disjunct of their own: the condition
    that moves them (`missing` with no path, now given one) is a fill
    of `source_path`, which the `source_path` comparison already
    catches. Anything that guard would let through, this one lets
    through identically.

    The explicit `updated_at = now()` is gone: migration 0003's trigger
    assigns it on every real UPDATE, and spelling it out here invited
    the reading that the bump is this statement's choice. It is not —
    it is the schema's, and the fix is to not perform the UPDATE.

    Every chat.db-derived column here is `PRESENT` or `POSITIVE` (see
    `Merge`): each is nullable in `chat.db` itself, so a NULL is exactly
    what a snapshot missing the row's metadata reports, and `is_sticker`
    arrives through a `COALESCE(is_sticker, 0)` that turns a missing
    flag into `false`. On the target host a silent re-extraction blanked
    `source_path` on 4,775 already-materialized attachments this way.
    `source_path` differing *between* hosts is normal — it is a
    per-device path — and under D12 only this machine's own live run may
    replace it; another host's path fills an empty one and nothing
    more.
    """
    initial_state = "dataless" if att.source_path is not None else "missing"
    initial_error = None if att.source_path is not None else NO_SOURCE_PATH_ERROR
    return _execute_upsert(
        cur,
        _ATTACHMENT_UPSERT_SQL,
        _ATTACHMENT_COLUMNS,
        {
            "source_guid": att.guid,
            "attachment_key": attachment_key(att.guid),
            "filename": att.filename,
            "source_path": att.source_path,
            "uti": att.uti,
            "mime_type": att.mime_type,
            "byte_size": att.byte_size,
            "is_sticker": att.is_sticker,
            "state": initial_state,
            "materialization_last_error": initial_error,
        },
        merge_mode,
    )


def _upsert_attachment_source(
    cur: psycopg.Cursor[Any], source_name: str, source_rowid: int, attachment_id: int
) -> UpsertOutcome:
    """Not evidence-gated, and does not need to be: `attachment_id` is a
    row id this same transaction just resolved, so it is never absent.
    The statement records "this source's row N is that attachment", which
    only the source itself can assert. A source row that now resolves to
    a different attachment is reported as `REPLACED`."""
    cur.execute(
        """
        INSERT INTO attachment_source (attachment_id, source_name, source_rowid)
        VALUES (%s, %s, %s)
        ON CONFLICT (source_name, source_rowid) DO UPDATE SET attachment_id = EXCLUDED.attachment_id
        WHERE attachment_source.attachment_id IS DISTINCT FROM EXCLUDED.attachment_id
        RETURNING (xmax = 0)
        """,
        (attachment_id, source_name, source_rowid),
    )
    row = cur.fetchone()
    if row is None:
        return UpsertOutcome.UNCHANGED
    return UpsertOutcome.INSERTED if row[0] else UpsertOutcome.REPLACED


_MESSAGE_COLUMNS: tuple[_Column, ...] = (
    _Column("source_guid", Merge.INSERT_ONLY),
    _Column("message_key", Merge.INSERT_ONLY),
    _Column("chat_id", Merge.INSERT_ONLY),
    # NULL here has two causes that this stage cannot tell apart from the
    # row alone: the message is the owner's (there is no handle to point
    # at), or the snapshot's `handle` table is missing the row its
    # `message.handle_id` refers to. The second is a real possibility for a
    # recovered or merged corpus, and writing its NULL would detach the
    # message from the only column S3 joins on to resolve a sender.
    # `PRESENT` costs one thing: a message that later flips to
    # `is_from_me` keeps the handle id it already had. Nothing renders
    # from it in that case — `is_from_me` is checked first — and no
    # observed `chat.db` flips that column.
    # Empty: NULL only -- a surrogate id is never 0.
    _Column("sender_source_handle_id", Merge.PRESENT),
    # Plain `chat.db` columns, NOT NULL on the way in (`sent_at` is
    # checked above and raises), so every source asserts them -- and only
    # the live run may move them. The 2026-09-22 dry run moved four
    # `sent_at` values by under a second.
    # Empty: NULL only; `is_from_me = false` is a real answer.
    _Column("is_from_me", Merge.ASSERTED),
    _Column("sent_at", Merge.ASSERTED),
    _Column("service", Merge.INSERT_ONLY, cast="::service_kind"),
    # NULL means the shim returned no record for this guid at all, or
    # returned one whose typedstream body it could not decode. Neither is
    # "this message has no text": an attachment-only message comes back as
    # the object-replacement character, not NULL. Before this policy a run
    # whose shim was silent wrote NULL over a body an earlier run had
    # stored, counted it in `bodies_missing`, logged it, and kept going.
    # Empty: NULL or ''. A retracted message comes back with its text gone;
    # the index keeps the text and `is_unsent` records the retraction. The
    # pair is decided on `text_original` (`_MESSAGE_BODY`), so
    # `text_normalized` is always the normalized form of the stored body --
    # including `''` for an attachment-only one.
    _Column("text_original", Merge.BODY, empty=Empty.BLANK),
    _Column("text_normalized", Merge.BODY, empty=Empty.BLANK),
    # `POSITIVE`: false is what a source reports both for "not retracted /
    # not edited / no attachments" and for "I cannot see whether it was",
    # and all three name facts that do not un-happen. False -> true (a
    # message really is unsent later) still flows; true -> false was never
    # a correction.
    _Column("is_unsent", Merge.POSITIVE, empty=Empty.FALSE),
    _Column("is_edited", Merge.POSITIVE, empty=Empty.FALSE),
    # The timestamp that goes with the body; a source without the edit
    # carries NULL, and an edit's timestamp only ever moves forward.
    # Empty: NULL only.
    _Column("date_edited", Merge.EDIT_TIME),
    _Column("reply_to_guid", Merge.INSERT_ONLY),
    _Column("has_attachments", Merge.POSITIVE, empty=Empty.FALSE),
    # How the chat was chosen (D13), recorded with `chat_id` on insert and,
    # like it, never assigned here: a later run changes the pair only through
    # `imsg.stages.unlinked_filing.refile_message`, toward stronger evidence.
    _Column("chat_evidence", Merge.INSERT_ONLY),
    # Apple's "Recently Deleted" delete date (D13). Rendered: the message
    # shows as "[deleted]". Empty: NULL only, and NULL is never evidence, so
    # no run ever clears it -- a message restored on one device keeps the
    # label another device's delete gave it. A seed fills it; only the live
    # run may replace one date with another.
    _Column("deleted_at", Merge.PRESENT),
)

_MESSAGE_BODY = _BodyRule(text="text_original", edited_at="date_edited")

_MESSAGE_UPSERT_SQL = _build_upsert_sql(
    table="message",
    conflict=("source_guid",),
    columns=_MESSAGE_COLUMNS,
    returning="message_id",
    body=_MESSAGE_BODY,
)


def _upsert_message(
    cur: psycopg.Cursor[Any],
    msg: MessageRow,
    dump_msg: ImsgDumpMessage | None,
    *,
    chat_id: int,
    chat_evidence: ChatEvidence = ChatEvidence.CHAT_MESSAGE_JOIN,
    handle_id_by_rowid: dict[int, int],
    has_attachments: bool,
    merge_mode: MergeMode,
) -> _UpsertRow:
    """**Which columns invalidate downstream work.** The `ON CONFLICT`
    clause assigns and compares exactly the columns whose value can
    change what S4 renders, what S6 embeds, or whether a message is
    indexed at all — read off what `imsg.segment.pipeline` actually
    selects (`sent_at`, `is_from_me`, `text_original`, `is_unsent`,
    `is_edited`, `has_attachments`, `sender_person_id`) plus what S3
    derives `sender_person_id` from:

      * `text_original` / `text_normalized` — the rendered body and its
        indexed copy. An edit is the whole reason the rescan clause in
        `fetch_target_messages` exists.
      * `is_unsent` — gates whether the message is segmented at all
        (`policy.index_unsent`, D1).
      * `is_edited` / `date_edited` — edit-history rendering, and the
        rescan signal itself.
      * `has_attachments` — rendered as an attachment snippet, and a
        retrieval filter (`imsg.retrieval.filters`).
      * `sent_at` — session-gap boundaries and every time-window filter.
      * `is_from_me` — renders as the literal `owner` rather than a
        short name.
      * `sender_source_handle_id` — S2's raw provenance and the only
        thing S3 joins on to backfill `sender_person_id`, which is the
        rendered sender. Stale here means a message attributed to the
        wrong person forever.
      * `deleted_at` — Apple's "Recently Deleted" date (D13); the
        message renders as `[deleted]`.

    A fill is an UPDATE like any other, so migration 0003's trigger
    moves `updated_at` and S4 re-segments the chat. What a segment
    renders through this message but stores in another table (the
    chat's name, attachments, tapbacks, edit history) is marked
    separately, by `_mark_for_resegmentation`.

    **Deliberately left first-write-wins**, and therefore neither
    assigned nor compared:

      * `chat_id` (and `chat_evidence`, which says how it was chosen).
        It affects rendering more than anything on the list, and it is
        still excluded here, because moving a message between chats is
        a filing decision, not a column merge. Until 2026-09-23 the
        incoming value came from `chat_message_join ... LIMIT 1` with no
        `ORDER BY`, so a message in more than one chat could flap
        between them. The link is now read deterministically (lowest
        chat ROWID), and the one move a later run may make is D13's:
        out of a holding chat, toward stronger evidence, never from one
        real chat to another (`imsg.stages.unlinked_filing.
        refile_message`, which `_do_extract` calls after this upsert).
      * `service` and `reply_to_guid`. Nothing downstream reads either
        column — not segmentation, not retrieval, not export, not the
        MCP surface — so a stale value cannot affect rendering or
        retrieval. They are recorded on insert as provenance.
      * `message_key` is a pure function of `source_guid`, the conflict
        key, so it cannot differ.
      * `sender_person_id` is S3's to write and S2's never (hard
        requirement 3); `created_at` is first-write provenance.

    **And which of the assigned columns a source may write.** Being
    allowed to invalidate downstream work is not the same as having
    something to say; `_MESSAGE_COLUMNS` carries the second decision,
    one `Merge` value per column, and the reasoning for each is on the
    declaration. Only `is_from_me` and `sent_at` are `ASSERTED` — they
    are plain `chat.db` columns present on every row of every source, so
    no source can be silent about them, and only the live run moves
    them. The body follows edit recency from any source (`Merge.BODY`).
    Everything else this statement writes is something a witness of the
    same conversation can genuinely lack.
    """
    sender_source_handle_id = None
    if not msg.is_from_me and msg.handle_rowid is not None:
        sender_source_handle_id = handle_id_by_rowid.get(msg.handle_rowid)

    body_text = dump_msg.body_text if dump_msg is not None else None
    if body_text is not None:
        # Strip NUL before it reaches ANY text column, including the verbatim
        # `text_original` — psycopg rejects the whole statement otherwise
        # (see textnorm.strip_nul). Real chat.db bodies contain NUL.
        body_text = strip_nul(body_text)
    text_for_index = None
    if body_text is not None:
        text_for_index = normalize_text(body_text.replace(OBJECT_REPLACEMENT_CHAR, ""))

    # `dump_msg.is_unsent` (typedstream-derived, via imsg-dump) is authoritative —
    # see the module docstring's "Correction" note: the SQL `date_retracted`
    # column is not a reliable signal in the real crate's own findings. The
    # SQL-derived value is still computed as a cross-check, logged on mismatch,
    # and used as the fallback when no dump record is available at all.
    sql_is_unsent = msg.date_retracted is not None
    sql_is_edited = msg.date_edited is not None
    if dump_msg is not None:
        is_unsent = dump_msg.is_unsent
        is_edited = bool(dump_msg.edit_history) or sql_is_edited
        if dump_msg.is_unsent != sql_is_unsent:
            logger.warning(
                "extract.is_unsent_mismatch",
                guid=msg.guid,
                sql=sql_is_unsent,
                imsg_dump=dump_msg.is_unsent,
            )
    else:
        is_unsent = sql_is_unsent
        is_edited = sql_is_edited

    reply_to_guid = msg.reply_to_guid or (dump_msg.reply_to_guid if dump_msg is not None else None)

    if msg.date is None:
        raise ExtractionError(f"message guid={msg.guid!r} rowid={msg.rowid} has no 'date' — chat.db invariant violated")

    return _execute_upsert(
        cur,
        _MESSAGE_UPSERT_SQL,
        _MESSAGE_COLUMNS,
        {
            "source_guid": msg.guid,
            "message_key": message_key(msg.guid),
            "chat_id": chat_id,
            "sender_source_handle_id": sender_source_handle_id,
            "is_from_me": msg.is_from_me,
            "sent_at": msg.date,
            "service": _normalize_service(msg.service),
            "text_original": body_text,
            "text_normalized": text_for_index,
            "is_unsent": is_unsent,
            "is_edited": is_edited,
            "date_edited": msg.date_edited,
            "reply_to_guid": reply_to_guid,
            "has_attachments": has_attachments,
            "chat_evidence": chat_evidence.value,
            "deleted_at": msg.deleted_at,
        },
        merge_mode,
    )


def _upsert_message_source(
    cur: psycopg.Cursor[Any], message_id: int, source_name: str, source_rowid: int, run_id: int
) -> UpsertOutcome:
    """Deliberately NOT guarded by an `IS DISTINCT FROM` clause, unlike
    every other upsert in this module. `extraction_run_id` means "the
    run that last observed this row", so a new run id is a genuine
    change by construction and the guard could never skip anything —
    it would be a comparison that can only ever be true.

    This is therefore the one write a no-op re-extraction still
    performs, once per in-scope message. It is affordable because
    `message_source` carries no `updated_at` column and nothing
    downstream watches it, so it drags no re-segmentation behind it:
    the cost is table churn, not 17 hours of recomputed embeddings.

    Reported as `INSERTED` for a row this source had not recorded and
    `UNCHANGED` for one it had: the mapping itself (`message_id`) is
    never rewritten, only the bookkeeping run id."""
    cur.execute(
        """
        INSERT INTO message_source (message_id, source_name, source_rowid, extraction_run_id)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_name, source_rowid) DO UPDATE SET extraction_run_id = EXCLUDED.extraction_run_id
        RETURNING (xmax = 0)
        """,
        (message_id, source_name, source_rowid, run_id),
    )
    row = cur.fetchone()
    return UpsertOutcome.INSERTED if row is not None and row[0] else UpsertOutcome.UNCHANGED


_MESSAGE_VERSION_COLUMNS: tuple[_Column, ...] = (
    _Column("message_id", Merge.INSERT_ONLY),
    _Column("version_idx", Merge.INSERT_ONLY),
    # A version the shim could not decode arrives as None, which the
    # column's NOT NULL forces to `''` on insert — and `''` written over a
    # version text another source decoded is the same loss as a blanked
    # body. The parameter therefore carries None for "could not decode"
    # and the empty string is only ever an INSERT default.
    # Empty: NULL or '' -- the stored '' is that undecodable placeholder,
    # which any source that decodes the version fills.
    _Column("text", Merge.HISTORY, empty=Empty.BLANK, insert_default="''"),
    # Empty: NULL only.
    _Column("edited_at", Merge.HISTORY),
)

_MESSAGE_VERSION_UPSERT_SQL = _build_upsert_sql(
    table="message_version",
    conflict=("message_id", "version_idx"),
    columns=_MESSAGE_VERSION_COLUMNS,
    returning="message_id",
)


def _upsert_message_version(
    cur: psycopg.Cursor[Any], message_id: int, version_idx: int, version: Any, merge_mode: MergeMode
) -> _UpsertRow:
    return _execute_upsert(
        cur,
        _MESSAGE_VERSION_UPSERT_SQL,
        _MESSAGE_VERSION_COLUMNS,
        {
            "message_id": message_id,
            "version_idx": version_idx,
            "text": None if version.text is None else strip_nul(version.text),
            "edited_at": _parse_iso(version.edited_at),
        },
        merge_mode,
    )


def _keep_displaced_body(
    cur: psycopg.Cursor[Any], message_id: int, body: str, edited_at: datetime | None
) -> bool:
    """Append a body a newer edit replaced to the message's edit history,
    unless the history already holds that exact text (normally it does:
    a source that carries an edit also lists the versions before it).
    This is what makes a newer edit an append rather than a replacement
    even when the newer source's own history lacks the displaced text.
    Appended after the highest existing position, where the most recent
    prior version belongs. Returns whether a row was added."""
    cur.execute(
        """
        INSERT INTO message_version (message_id, version_idx, text, edited_at)
        SELECT %(message_id)s,
               COALESCE((SELECT MAX(version_idx) FROM message_version
                          WHERE message_id = %(message_id)s), -1) + 1,
               %(text)s, %(edited_at)s
        WHERE NOT EXISTS (
            SELECT 1 FROM message_version WHERE message_id = %(message_id)s AND text = %(text)s
        )
        """,
        {"message_id": message_id, "text": body, "edited_at": edited_at},
    )
    return cur.rowcount == 1


_TAPBACK_COLUMNS: tuple[_Column, ...] = (
    _Column("source_guid", Merge.INSERT_ONLY),
    _Column("target_source_guid", Merge.INSERT_ONLY),
    # NULL means the target message is not in this database *yet* — it may
    # be outside the run's scope, or simply later in the same run.
    # `_backfill_tapback_targets` resolves those at the end of the
    # transaction; writing the NULL first would blank an id an earlier run
    # had already resolved, only for the backfill to re-derive it, so this
    # is both a correctness and a churn fix.
    # Empty: NULL only -- a message id is never 0.
    _Column("target_message_id", Merge.PRESENT),
    _Column("sender_source_handle_id", Merge.INSERT_ONLY),
    _Column("is_from_me", Merge.INSERT_ONLY),
    _Column("kind", Merge.INSERT_ONLY),
    # `ImsgDumpMessage.action` defaults to "added" when the shim omits it,
    # so `removed=false` is also what "the shim said nothing" looks like.
    # Un-reacting is a separate event that only ever sets this true.
    _Column("removed", Merge.POSITIVE, empty=Empty.FALSE),
    # `msg.date` — NULL for a tapback row whose `chat.db` date is missing.
    # (Unlike `_upsert_message`, this path has no NOT NULL check to raise
    # on, because `tapback.acted_at` is nullable.)
    # Empty: NULL only (`chat.db`'s 0 date is already NULL here).
    _Column("acted_at", Merge.PRESENT),
)

_TAPBACK_UPSERT_SQL = _build_upsert_sql(
    table="tapback", conflict=("source_guid",), columns=_TAPBACK_COLUMNS, returning="tapback_id"
)


def _upsert_tapback(
    cur: psycopg.Cursor[Any],
    msg: MessageRow,
    dump_msg: ImsgDumpMessage,
    handle_id_by_rowid: dict[int, int],
    merge_mode: MergeMode,
) -> tuple[_UpsertRow, int | None]:
    """Returns the upsert's row and the target message's id, if that
    message is in the index already: a segment renders the tapback on
    it, so a new or changed tapback marks it for re-segmentation."""
    assert dump_msg.tapback is not None
    sender_source_handle_id = None
    if not msg.is_from_me and msg.handle_rowid is not None:
        sender_source_handle_id = handle_id_by_rowid.get(msg.handle_rowid)

    cur.execute("SELECT message_id FROM message WHERE source_guid = %s", (dump_msg.tapback.target_guid,))
    target_row = cur.fetchone()
    target_message_id = int(target_row[0]) if target_row else None

    # SPEC §7.2's `tapback.kind` comment documents the emoji case as
    # "emoji:<char>", not the bare "emoji" `imsg-dump` emits — combine them
    # here. `removed` comes from the shim's own `action` field
    # ("added"/"removed": chat.db models un-reacting as a second tapback
    # event targeting the same message, not a mutation of the first) —
    # NOT from `msg.date_retracted`, which is about the *message* being
    # unsent, an unrelated concept.
    kind = dump_msg.tapback.kind
    if kind == "emoji" and dump_msg.tapback.emoji:
        kind = f"emoji:{dump_msg.tapback.emoji}"
    removed = dump_msg.tapback.action == "removed"

    row = _execute_upsert(
        cur,
        _TAPBACK_UPSERT_SQL,
        _TAPBACK_COLUMNS,
        {
            "source_guid": msg.guid,
            "target_source_guid": dump_msg.tapback.target_guid,
            "target_message_id": target_message_id,
            "sender_source_handle_id": sender_source_handle_id,
            "is_from_me": msg.is_from_me,
            "kind": kind,
            "removed": removed,
            "acted_at": msg.date,
        },
        merge_mode,
    )
    return row, target_message_id


def _backfill_tapback_targets(cur: psycopg.Cursor[Any]) -> list[int]:
    """A tapback can arrive before the message it targets (out-of-order
    extraction, or the target is outside this run's scope). Resolve any
    still-unresolved `target_message_id` whose target has since landed
    (SPEC §7.2: "backfilled when target is present"). Returns the target
    ids it attached tapbacks to, which now render them."""
    cur.execute(
        """
        UPDATE tapback t
        SET target_message_id = m.message_id
        FROM message m
        WHERE t.target_message_id IS NULL AND m.source_guid = t.target_source_guid
        RETURNING t.target_message_id
        """
    )
    return [int(row[0]) for row in cur.fetchall()]


_LINK_PREVIEW_COLUMNS: tuple[_Column, ...] = (
    _Column("message_id", Merge.INSERT_ONLY),
    _Column("url", Merge.INSERT_ONLY),
    # `parse_link_preview` is a best-effort walk of an NSKeyedArchiver
    # blob: a key it does not find comes back NULL, whether the preview
    # genuinely had no title or this source's `payload_data` was truncated
    # or shaped differently. Same policy as everywhere else.
    # Empty: NULL or '' (the parser already drops empty strings). Not
    # rendered: segments do not show link previews.
    _Column("title", Merge.PRESENT, empty=Empty.BLANK, renders=False),
    _Column("summary", Merge.PRESENT, empty=Empty.BLANK, renders=False),
    _Column("site_name", Merge.PRESENT, empty=Empty.BLANK, renders=False),
)

_LINK_PREVIEW_UPSERT_SQL = _build_upsert_sql(
    table="link_preview",
    conflict=("message_id", "url"),
    columns=_LINK_PREVIEW_COLUMNS,
    returning="message_id",
)


def _upsert_link_preview(
    cur: psycopg.Cursor[Any], message_id: int, preview: LinkPreview, merge_mode: MergeMode
) -> UpsertOutcome:
    return _execute_upsert(
        cur,
        _LINK_PREVIEW_UPSERT_SQL,
        _LINK_PREVIEW_COLUMNS,
        {
            "message_id": message_id,
            "url": preview.url,
            "title": preview.title,
            "summary": preview.summary,
            "site_name": preview.site_name,
        },
        merge_mode,
    ).outcome


# --------------------------------------------------------------------------
# downstream invalidation for what a segment renders from other tables
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Rerender:
    """What changed this run that a segment renders through a message
    but stores in another table.

    INVARIANT: every write here that changes what a segment renders must
    move `message.updated_at` for the messages that render it, because
    `imsg.segment.pipeline.find_dirty_chats` watches nothing else. A
    write to the `message` row itself does that through migration
    0003's trigger; these tables have no such path:

      * `chat` (`kind`, `display_name`) -- every segment header of the
        chat, so every message in it (the same whole-chat mark
        `imsg.stages.identity._mark_chats_dirty_for_persons` makes for a
        renamed person);
      * `attachment` (`filename`, `mime_type`) -- every message linked
        to it;
      * `message_attachment` -- a new link on an existing message;
      * `tapback` -- the message it targets;
      * `message_version` -- the message it is a version of.

    Collected during the run and marked once at the end, when every
    link the run creates exists."""

    chat_ids: set[int] = field(default_factory=set)
    attachment_ids: set[int] = field(default_factory=set)
    message_ids: set[int] = field(default_factory=set)


def _mark_for_resegmentation(cur: psycopg.Cursor[Any], rerender: _Rerender) -> int:
    """Move `updated_at` (migration 0003's trigger sets it on any UPDATE)
    for every message `rerender` names, and return how many moved.
    Messages this transaction already wrote carry `updated_at = now()`
    and are skipped, so a message is marked once however many of its
    parts changed, and messages inserted by this run are left alone."""
    if not (rerender.chat_ids or rerender.attachment_ids or rerender.message_ids):
        return 0
    cur.execute(
        """
        UPDATE message SET updated_at = now()
        WHERE (chat_id = ANY(%(chat_ids)s::bigint[])
               OR message_id = ANY(%(message_ids)s::bigint[])
               OR message_id IN (SELECT ma.message_id FROM message_attachment ma
                                  WHERE ma.attachment_id = ANY(%(attachment_ids)s::bigint[])))
          AND updated_at < now()
        """,
        {
            "chat_ids": sorted(rerender.chat_ids),
            "message_ids": sorted(rerender.message_ids),
            "attachment_ids": sorted(rerender.attachment_ids),
        },
    )
    return cur.rowcount


__all__ = [
    "APPLE_EPOCH",
    "AttachmentRow",
    "ChatRow",
    "Empty",
    "ExtractResult",
    "HandleRow",
    "LinkPreview",
    "Merge",
    "MergeMode",
    "MessageRow",
    "SnapshotReader",
    "UnlinkedCounts",
    "UpsertCounts",
    "UpsertOutcome",
    "merge_mode_for_source",
    "parse_link_preview",
    "run_extract",
]
