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

import plistlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import apsw
import psycopg
import structlog

from imsg.backfill.classify import NO_SOURCE_PATH_ERROR
from imsg.errors import ExtractionError
from imsg.hashing import sha256_file
from imsg.keys import attachment_key, message_key, thread_key
from imsg.sqlite_readonly import open_readonly_immutable, wal_frame_bytes, wal_sidecar_path
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun, run_imsg_dump
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

    def fetch_max_message_rowid(self) -> int:
        row = next(self._conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message"), None)
        return int(row[0]) if row else 0

    def fetch_chats(self) -> list[ChatRow]:
        rows = list(
            self._conn.execute(
                """
                SELECT c.ROWID, c.guid, c.style, c.display_name, c.service_name,
                       (SELECT COUNT(*) FROM chat_handle_join chj WHERE chj.chat_id = c.ROWID)
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
            )
            for r in rows
        ]

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
        query = """
            SELECT m.ROWID, m.guid, m.handle_id, m.is_from_me, m.date, m.date_edited,
                   m.date_retracted, m.service, m.thread_originator_guid,
                   COALESCE(m.item_type, 0), m.payload_data,
                   (SELECT cmj.chat_id FROM chat_message_join cmj
                    WHERE cmj.message_id = m.ROWID LIMIT 1)
            FROM message m
            WHERE m.ROWID > ?
               OR COALESCE(m.date_edited, 0) > ?
               OR COALESCE(m.date_retracted, 0) > ?
            ORDER BY m.ROWID
        """
        rows = list(self._conn.execute(query, (watermark, last_run_start_ns, last_run_start_ns)))
        return [
            MessageRow(
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
            )
            for r in rows
        ]

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
# extraction result + main entry point
# --------------------------------------------------------------------------


class UpsertOutcome(StrEnum):
    """What one upsert statement actually did to its row."""

    INSERTED = "inserted"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    """The row already held exactly these values, so no UPDATE was
    performed at all — no new row version, and migration 0003's trigger
    never fired, so `updated_at` did not move."""


@dataclass(frozen=True, slots=True)
class UpsertCounts:
    """How an upsert loop landed, split three ways.

    `total` is the number of rows the loop processed, which is what the
    `*_upserted` fields below have always reported — the split is the
    part an operator needs to trust a re-extraction. "Upserted: 662,683"
    reads identically whether the run corrected 662,683 rows or
    rewrote them with their own values; `unchanged=662,683` says
    plainly that the run wrote nothing, moved no `updated_at`, and
    therefore proposes no re-segmentation downstream.
    """

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.unchanged


@dataclass(slots=True)
class _UpsertTally:
    """Mutable accumulator for `UpsertCounts` (which is frozen, as every
    field of `ExtractResult` is)."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0

    def record(self, outcome: UpsertOutcome) -> None:
        if outcome is UpsertOutcome.INSERTED:
            self.inserted += 1
        elif outcome is UpsertOutcome.UPDATED:
            self.updated += 1
        else:
            self.unchanged += 1

    def freeze(self) -> UpsertCounts:
        return UpsertCounts(
            inserted=self.inserted, updated=self.updated, unchanged=self.unchanged
        )


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
    """Inserted / genuinely updated / left untouched, for the four
    entities whose upserts return a row id. `*.total` equals the
    matching `*_upserted` count above, which keeps its original meaning
    ("rows this run processed") — the breakdown is additive, so nothing
    reading the old fields changes behaviour.

    Tapbacks, link previews, message versions and the join tables are
    deliberately not split: they have no `RETURNING` clause to hang the
    classification on, and none of them carries an `updated_at` column,
    so a spurious write there cost table churn but never dragged the
    segmentation frontier."""
    dry_run: bool = False
    """True when this result came from `run_extract(dry_run=True)`
    (SPEC §8: "takes --dry-run where writes leave the machine"): every
    count above is accurate (the real extraction logic ran end to
    end, including the read-only `imsg-dump` subprocess call), but the
    transaction that produced them was rolled back — nothing was
    actually written to Postgres, and `run_id` refers to an
    `extraction_run` row that existed only for the rolled-back
    transaction's duration."""


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
) -> ExtractResult:
    """Extract one snapshot into Postgres (SPEC §8 S2).

    `conn` must already be open (this module never owns connection
    lifecycle); `snapshot_path` is an S1 output, never the live
    `chat.db`. Raises `ExtractionError` for boundary failures (the
    snapshot does not open as SQLite, `imsg-dump` fails to run); a
    single message's decode failure inside `imsg-dump` degrades to a
    null body there and is *not* an `ExtractionError` here.

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
            )

        run_id = _begin_extraction_run(
            conn,
            source_name=source_name,
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha256,
            rowid_before=watermark_before,
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
) -> int:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO extraction_run (source_name, snapshot_path, snapshot_sha256, rowid_before, status)
            VALUES (%s, %s, %s, %s, 'running')
            RETURNING run_id
            """,
            (source_name, str(snapshot_path), snapshot_sha256, rowid_before),
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
) -> ExtractResult:
    chats = reader.fetch_chats()
    handles = reader.fetch_handles()
    chat_handle_joins = reader.fetch_chat_handle_joins()

    last_run_start_ns = _datetime_to_apple_ns(last_run_start)
    target_messages = reader.fetch_target_messages(watermark_before, last_run_start_ns)
    target_rowids = [m.rowid for m in target_messages]

    dump_since_rowid = watermark_before
    if target_rowids:
        dump_since_rowid = min(watermark_before, min(target_rowids) - 1)

    dump_run = run_imsg_dump_fn(imsg_dump_binary, snapshot_path, dump_since_rowid)
    dump_by_guid: dict[str, ImsgDumpMessage] = {m.guid: m for m in dump_run.messages}

    attachments, attachments_by_message = reader.fetch_attachments_for_messages(target_rowids)

    with conn.transaction(), conn.cursor() as cur:
        chat_id_by_rowid: dict[int, int] = {}
        chat_tally = _UpsertTally()
        for chat in chats:
            chat_id_by_rowid[chat.rowid], outcome = _upsert_chat(cur, chat)
            chat_tally.record(outcome)

        handle_id_by_rowid: dict[int, int] = {}
        handle_tally = _UpsertTally()
        for handle in handles:
            handle_id_by_rowid[handle.rowid], outcome = _upsert_source_handle(cur, handle)
            handle_tally.record(outcome)

        for chat_rowid, handle_rowid in chat_handle_joins:
            chat_id = chat_id_by_rowid.get(chat_rowid)
            source_handle_id = handle_id_by_rowid.get(handle_rowid)
            if chat_id is not None and source_handle_id is not None:
                cur.execute(
                    "INSERT INTO chat_participant_source (chat_id, source_handle_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (chat_id, source_handle_id),
                )

        attachment_id_by_rowid: dict[int, int] = {}
        attachment_tally = _UpsertTally()
        for att in attachments:
            attachment_id_by_rowid[att.rowid], outcome = _upsert_attachment(cur, att)
            attachment_tally.record(outcome)
            _upsert_attachment_source(cur, source_name, att.rowid, attachment_id_by_rowid[att.rowid])

        message_tally = _UpsertTally()
        tapbacks_upserted = 0
        system_messages_skipped = 0
        link_previews_upserted = 0
        bodies_missing = 0

        for msg in target_messages:
            dump_msg = dump_by_guid.get(msg.guid)
            if dump_msg is None:
                bodies_missing += 1
                logger.warning("extract.body_missing_from_dump", guid=msg.guid, rowid=msg.rowid)

            if dump_msg is not None and dump_msg.tapback is not None:
                _upsert_tapback(cur, msg, dump_msg, handle_id_by_rowid)
                tapbacks_upserted += 1
                continue

            if msg.item_type != 0:
                system_messages_skipped += 1
                continue

            chat_id = chat_id_by_rowid.get(msg.chat_rowid) if msg.chat_rowid is not None else None
            if chat_id is None:
                logger.warning("extract.message_without_chat", guid=msg.guid, rowid=msg.rowid)
                continue

            message_id, outcome = _upsert_message(
                cur, msg, dump_msg, chat_id=chat_id, handle_id_by_rowid=handle_id_by_rowid,
                has_attachments=bool(attachments_by_message.get(msg.rowid)),
            )
            message_tally.record(outcome)

            _upsert_message_source(cur, message_id, source_name, msg.rowid, run_id)

            if dump_msg is not None:
                for idx, version in enumerate(dump_msg.edit_history):
                    _upsert_message_version(cur, message_id, idx, version)

            for ordinal, att_rowid in enumerate(attachments_by_message.get(msg.rowid, [])):
                att_id = attachment_id_by_rowid.get(att_rowid)
                if att_id is not None:
                    cur.execute(
                        "INSERT INTO message_attachment (message_id, attachment_id, ordinal) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (message_id, att_id, ordinal),
                    )

            preview = parse_link_preview(msg.payload_data)
            if preview is not None:
                _upsert_link_preview(cur, message_id, preview)
                link_previews_upserted += 1

        _backfill_tapback_targets(cur)

        cur.execute(
            "INSERT INTO sync_state (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (_watermark_key(source_name), str(snapshot_max_rowid)),
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
        # defect sit unnoticed.
        cur.execute(
            """
            UPDATE extraction_run
            SET status = 'ok', finished_at = clock_timestamp(), rowid_after = %s,
                messages_upserted = %s, messages_inserted = %s,
                messages_updated = %s, messages_unchanged = %s
            WHERE run_id = %s
            """,
            (
                snapshot_max_rowid,
                message_tally.inserted + message_tally.updated + message_tally.unchanged,
                message_tally.inserted,
                message_tally.updated,
                message_tally.unchanged,
                run_id,
            ),
        )

    message_upserts = message_tally.freeze()
    attachment_upserts = attachment_tally.freeze()
    chat_upserts = chat_tally.freeze()
    handle_upserts = handle_tally.freeze()
    return ExtractResult(
        run_id=run_id,
        watermark_before=watermark_before,
        watermark_after=snapshot_max_rowid,
        chats_upserted=chat_upserts.total,
        handles_upserted=handle_upserts.total,
        messages_upserted=message_upserts.total,
        tapbacks_upserted=tapbacks_upserted,
        system_messages_skipped=system_messages_skipped,
        attachments_upserted=attachment_upserts.total,
        link_previews_upserted=link_previews_upserted,
        bodies_missing=bodies_missing,
        dump_stderr_line_count=len(dump_run.stderr_lines),
        chat_upserts=chat_upserts,
        handle_upserts=handle_upserts,
        message_upserts=message_upserts,
        attachment_upserts=attachment_upserts,
    )


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
# The rule is per column and declared once, in the `_Column` tables
# below, rather than spelled out per column inside each statement. Each
# column answers one question: *can this source's "empty" be told apart
# from this source having nothing to say?*


class Merge(StrEnum):
    """How one column of an already-present row may be rewritten."""

    ASSERTED = "asserted"
    """Every source that has the row at all carries this column, so
    whatever it says -- NULL and false included -- is a positive
    assertion and overwrites. `message.sent_at` is the type case: it is
    a plain `chat.db` column, present on every row of every source."""

    PRESENT = "present"
    """NULL means "this source has nothing here", which no source can
    distinguish from "this source says this is empty". A NULL therefore
    leaves the stored value alone; any non-NULL value still overwrites,
    so a genuine correction (an edited body, a renamed chat) flows
    through untouched."""

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
    correction being blocked."""

    INSERT_ONLY = "insert_only"
    """First write wins: recorded on insert as provenance, never
    updated. Keys and derived keys, plus the columns documented on
    `_upsert_message` as deliberately frozen."""


@dataclass(frozen=True, slots=True)
class _Column:
    """One column of a guarded upsert, and how it merges."""

    name: str
    merge: Merge
    cast: str = ""
    """Postgres cast appended to the VALUES placeholder, e.g.
    `::service_kind`. Only needed where the parameter is a bare Python
    string feeding an enum column."""
    insert_default: str | None = None
    """SQL fragment supplying the INSERT value when the parameter is
    NULL, for a `PRESENT` column that the schema declares NOT NULL (a
    NULL there means "no evidence", but the row still has to be
    insertable). Unused on the UPDATE path, which keeps the stored
    value instead."""


def _known_param(column: str) -> str:
    return f"{column}__known"


def _evidence_flags(columns: Sequence[_Column], values: dict[str, Any]) -> dict[str, Any]:
    """Derive one `<column>__known` boolean per evidence-bearing column,
    from the policy declared on the column rather than from anything the
    call site remembers to pass. This is the whole rule, mechanically
    applied: `PRESENT` is known when the value is not NULL, `POSITIVE`
    when it is true."""
    flags: dict[str, Any] = {}
    for column in columns:
        if column.merge is Merge.PRESENT:
            flags[_known_param(column.name)] = values[column.name] is not None
        elif column.merge is Merge.POSITIVE:
            flags[_known_param(column.name)] = bool(values[column.name])
    return flags


def _insert_value_sql(column: _Column) -> str:
    placeholder = f"%({column.name})s{column.cast}"
    if column.insert_default is None:
        return placeholder
    return f"COALESCE({placeholder}, {column.insert_default})"


def _set_clause(table: str, column: _Column) -> str | None:
    """The `SET` fragment for one column, or None when the column is
    never updated."""
    if column.merge is Merge.INSERT_ONLY:
        return None
    if column.merge is Merge.ASSERTED:
        return f"{column.name} = EXCLUDED.{column.name}"
    known = f"%({_known_param(column.name)})s::boolean"
    return (
        f"{column.name} = CASE WHEN {known} "
        f"THEN EXCLUDED.{column.name} ELSE {table}.{column.name} END"
    )


def _differs_clause(table: str, column: _Column) -> str | None:
    """The `WHERE` disjunct for one column: the row is rewritten only
    when this column would actually change.

    The compared columns are exactly the assigned columns, and the
    comparison is `IS DISTINCT FROM`, never `<>` -- both rules are the
    2026-09-17 state-idempotence fix and both still hold here, with the
    evidence flag ANDed in so a column the source cannot see is not
    merely left unwritten but also cannot drag the row into an UPDATE
    (which would move `updated_at` and re-segment the chat for nothing).
    """
    if column.merge is Merge.INSERT_ONLY:
        return None
    differs = f"{table}.{column.name} IS DISTINCT FROM EXCLUDED.{column.name}"
    if column.merge is Merge.ASSERTED:
        return differs
    return f"(%({_known_param(column.name)})s::boolean AND {differs})"


def _build_upsert_sql(
    *,
    table: str,
    conflict: Sequence[str],
    columns: Sequence[_Column],
    returning: str | None = None,
    extra_set: str = "",
    extra_where: str = "",
) -> str:
    """Render one guarded, provenance-respecting upsert statement.

    Built once per table at import time, so the SQL below is a constant
    a reader can print, not a string assembled per row. `returning` is
    the primary-key column for the three upserts whose caller needs the
    row id back (chat, attachment, message); the shape it produces and
    why it has a `UNION ALL` branch is documented on `_read_upsert_row`.
    """
    set_clauses = [c for c in (_set_clause(table, col) for col in columns) if c]
    where_clauses = [c for c in (_differs_clause(table, col) for col in columns) if c]
    if extra_set:
        set_clauses.append(extra_set)
    if extra_where:
        where_clauses.append(extra_where)
    assert set_clauses, f"{table}: nothing to update — use ON CONFLICT DO NOTHING instead"

    insert_columns = ", ".join(col.name for col in columns)
    insert_values = ", ".join(_insert_value_sql(col) for col in columns)
    conflict_columns = ", ".join(conflict)
    body = (
        f"INSERT INTO {table} ({insert_columns})\n"
        f"VALUES ({insert_values})\n"
        f"ON CONFLICT ({conflict_columns}) DO UPDATE SET\n    "
        + ",\n    ".join(set_clauses)
        + "\nWHERE "
        + "\n   OR ".join(where_clauses)
    )
    if returning is None:
        return body

    match = " AND ".join(f"{col} = %({col})s" for col in conflict)
    return (
        "WITH upserted AS (\n"
        f"{body}\n"
        f"RETURNING {returning}, (xmax = 0) AS inserted\n"
        ")\n"
        f"SELECT {returning}, inserted, true FROM upserted\n"
        "UNION ALL\n"
        f"SELECT {returning}, false, false FROM {table}\n"
        f" WHERE {match} AND NOT EXISTS (SELECT 1 FROM upserted)"
    )


def _execute_upsert(
    cur: psycopg.Cursor[Any], sql: str, columns: Sequence[_Column], values: dict[str, Any]
) -> None:
    """Run one built statement, adding the evidence flags the policy
    implies. Call sites pass values only; they never hand-write a flag,
    which is what keeps the rule in one place."""
    cur.execute(sql, {**values, **_evidence_flags(columns, values)})


# --------------------------------------------------------------------------
# Postgres upsert helpers
# --------------------------------------------------------------------------


def _read_upsert_row(cur: psycopg.Cursor[Any]) -> tuple[int, UpsertOutcome]:
    """Read the single row every guarded upsert below returns.

    The statements are all shaped the same way:

        WITH upserted AS (
            INSERT ... ON CONFLICT (...) DO UPDATE SET ... WHERE <differs>
            RETURNING <pk>, (xmax = 0) AS inserted
        )
        SELECT <pk>, inserted, true  FROM upserted
        UNION ALL
        SELECT <pk>, false,    false FROM <table>
         WHERE <conflict key> = ... AND NOT EXISTS (SELECT 1 FROM upserted)

    The trailing branch exists because a WHERE-guarded `DO UPDATE` that
    skips the row returns nothing — and the caller still needs the row
    id. It reads the table as of statement start, which cannot see the
    CTE's own insert; that is fine, because when the CTE inserts, the
    `NOT EXISTS` is false and the branch contributes nothing. Exactly
    one row comes back either way.

    `xmax = 0` distinguishes the inserted row from the updated one: an
    `INSERT ... ON CONFLICT DO UPDATE` leaves `xmax` set to the updating
    transaction on the update path and zero on the insert path. It is
    read-only bookkeeping — a misclassification would mis-report a
    count, never write a wrong row — and it is asserted directly by
    `tests/test_extract_unchanged_rows_stay_untouched_integration.py`.
    """
    row = cur.fetchone()
    assert row is not None, "guarded upsert returned no row — the fallback branch is wrong"
    row_id, inserted, written = int(row[0]), bool(row[1]), bool(row[2])
    if not written:
        return row_id, UpsertOutcome.UNCHANGED
    return row_id, UpsertOutcome.INSERTED if inserted else UpsertOutcome.UPDATED


_CHAT_COLUMNS: tuple[_Column, ...] = (
    _Column("source_guid", Merge.INSERT_ONLY),
    _Column("thread_key", Merge.INSERT_ONLY),
    # `kind` is read off `chat.style`, a plain column — but only the two
    # documented values mean anything, and an unrecognized one falls back to a
    # participant-count guess. A guess is not evidence: a source whose
    # `chat_handle_join` rows are incomplete would guess `dm` for a group and
    # overwrite a correctly-typed row. So the parameter carries the value only
    # when `style` said it, and the heuristic supplies the INSERT default.
    _Column(
        "kind", Merge.PRESENT, cast="::chat_kind",
        insert_default="%(kind_fallback)s::chat_kind",
    ),
    # A device that joined a named group late, or never received the rename
    # event, has NULL here — indistinguishable from a group whose name was
    # cleared. Measured cost of getting this wrong on the target host: 3,125
    # chats lost their `display_name` to a silent re-extraction.
    _Column("display_name", Merge.PRESENT),
    # `_service_evidence` returns NULL for the `unknown` bucket, which is the
    # absence marker itself: `chat.service_name` was NULL, or held a string
    # this build does not recognize. `unknown` is still what a first insert
    # records.
    _Column(
        "service", Merge.PRESENT, cast="::service_kind",
        insert_default="'unknown'::service_kind",
    ),
)

_CHAT_UPSERT_SQL = _build_upsert_sql(
    table="chat", conflict=("source_guid",), columns=_CHAT_COLUMNS, returning="chat_id"
)


def _upsert_chat(cur: psycopg.Cursor[Any], chat: ChatRow) -> tuple[int, UpsertOutcome]:
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
    _execute_upsert(
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
    )
    return _read_upsert_row(cur)


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
        SELECT source_handle_id, true, true FROM upserted
        UNION ALL
        SELECT source_handle_id, false, false FROM source_handle
         WHERE raw_value = %(raw_value)s AND service = %(service)s::service_kind
           AND NOT EXISTS (SELECT 1 FROM upserted)
        """,
        {"raw_value": handle.raw_value, "service": _normalize_service(handle.service)},
    )
    return _read_upsert_row(cur)


_ATTACHMENT_COLUMNS: tuple[_Column, ...] = (
    _Column("source_guid", Merge.INSERT_ONLY),
    _Column("attachment_key", Merge.INSERT_ONLY),
    _Column("filename", Merge.PRESENT),
    _Column("source_path", Merge.PRESENT),
    _Column("uti", Merge.PRESENT),
    _Column("mime_type", Merge.PRESENT),
    _Column("byte_size", Merge.PRESENT),
    _Column("is_sticker", Merge.POSITIVE),
    # S5a owns both of these from the first insert onwards; the one
    # exception is spelled out in `_ATTACHMENT_STATE_SET` below.
    _Column("state", Merge.INSERT_ONLY, cast="::materialization_state"),
    _Column("materialization_last_error", Merge.INSERT_ONLY),
)

_ATTACHMENT_STATE_SET = """state = CASE
        WHEN attachment.state = 'missing' AND attachment.source_path IS NULL
             AND EXCLUDED.source_path IS NOT NULL
        THEN 'dataless'::materialization_state
        ELSE attachment.state
    END,
    materialization_last_error = CASE
        WHEN attachment.state = 'missing' AND attachment.source_path IS NULL
             AND EXCLUDED.source_path IS NOT NULL
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
    cur: psycopg.Cursor[Any], att: AttachmentRow
) -> tuple[int, UpsertOutcome]:
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
    that moves them (`missing` with no path, now given one) requires
    `source_path` to have changed, which the `source_path` comparison
    already catches. Anything that guard would let through, this one
    lets through identically.

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
    `source_path` differing *between* hosts is normal and still flows —
    it is a per-device path — because a non-NULL value is evidence.
    """
    initial_state = "dataless" if att.source_path is not None else "missing"
    initial_error = None if att.source_path is not None else NO_SOURCE_PATH_ERROR
    _execute_upsert(
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
    )
    return _read_upsert_row(cur)


def _upsert_attachment_source(
    cur: psycopg.Cursor[Any], source_name: str, source_rowid: int, attachment_id: int
) -> None:
    """Not evidence-gated, and does not need to be: `attachment_id` is a
    row id this same transaction just resolved, so it is never absent.
    The statement records "this source's row N is that attachment", which
    only the source itself can assert."""
    cur.execute(
        """
        INSERT INTO attachment_source (attachment_id, source_name, source_rowid)
        VALUES (%s, %s, %s)
        ON CONFLICT (source_name, source_rowid) DO UPDATE SET attachment_id = EXCLUDED.attachment_id
        WHERE attachment_source.attachment_id IS DISTINCT FROM EXCLUDED.attachment_id
        """,
        (attachment_id, source_name, source_rowid),
    )


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
    _Column("sender_source_handle_id", Merge.PRESENT),
    # Plain `chat.db` columns, NOT NULL on the way in (`sent_at` is
    # checked above and raises), so every source asserts them.
    _Column("is_from_me", Merge.ASSERTED),
    _Column("sent_at", Merge.ASSERTED),
    _Column("service", Merge.INSERT_ONLY, cast="::service_kind"),
    # NULL means the shim returned no record for this guid at all, or
    # returned one whose typedstream body it could not decode. Neither is
    # "this message has no text": an attachment-only message comes back as
    # the object-replacement character, not NULL. Before this policy a run
    # whose shim was silent wrote NULL over a body an earlier run had
    # stored, counted it in `bodies_missing`, logged it, and kept going.
    _Column("text_original", Merge.PRESENT),
    _Column("text_normalized", Merge.PRESENT),
    # `POSITIVE`: false is what a source reports both for "not retracted /
    # not edited / no attachments" and for "I cannot see whether it was",
    # and all three name facts that do not un-happen. False -> true (a
    # message really is unsent later) still flows; true -> false was never
    # a correction.
    _Column("is_unsent", Merge.POSITIVE),
    _Column("is_edited", Merge.POSITIVE),
    # The timestamp that goes with `is_edited`; a source without the edit
    # carries NULL, and an edit's timestamp only ever moves forward.
    _Column("date_edited", Merge.PRESENT),
    _Column("reply_to_guid", Merge.INSERT_ONLY),
    _Column("has_attachments", Merge.POSITIVE),
)

_MESSAGE_UPSERT_SQL = _build_upsert_sql(
    table="message", conflict=("source_guid",), columns=_MESSAGE_COLUMNS, returning="message_id"
)


def _upsert_message(
    cur: psycopg.Cursor[Any],
    msg: MessageRow,
    dump_msg: ImsgDumpMessage | None,
    *,
    chat_id: int,
    handle_id_by_rowid: dict[int, int],
    has_attachments: bool,
) -> tuple[int, UpsertOutcome]:
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

    **Deliberately left first-write-wins**, and therefore neither
    assigned nor compared:

      * `chat_id`. It affects rendering more than anything on the list,
        and it is still excluded, for a stronger reason than
        irrelevance: the incoming value is not trustworthy.
        `fetch_target_messages` derives it from
        `chat_message_join ... LIMIT 1` with no `ORDER BY`, so for a
        message that belongs to more than one chat SQLite may return
        either. Reassigning it per run would flap the row between chats
        on every extraction — permanently dirty, and orphaning the
        `segment_message` rows that place it. Correcting a genuinely
        wrong chat association needs a deterministic choice first;
        that is a separate change, not a side effect of this one.
      * `service` and `reply_to_guid`. Nothing downstream reads either
        column — not segmentation, not retrieval, not export, not the
        MCP surface — so a stale value cannot affect rendering or
        retrieval. They are recorded on insert as provenance.
      * `message_key` is a pure function of `source_guid`, the conflict
        key, so it cannot differ.
      * `sender_person_id` is S3's to write and S2's never (hard
        requirement 3); `created_at` is first-write provenance.

    **And which of the assigned columns a silent source may write.**
    Being allowed to invalidate downstream work is not the same as
    having something to say; `_MESSAGE_COLUMNS` below carries the
    second decision, one `Merge` value per column, and the reasoning
    for each is on the declaration. Only `is_from_me` and `sent_at` are
    `ASSERTED` — they are plain `chat.db` columns present on every row
    of every source, so no source can be silent about them. Everything
    else this statement writes is something a witness of the same
    conversation can genuinely lack.
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

    _execute_upsert(
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
        },
    )
    return _read_upsert_row(cur)


def _upsert_message_source(
    cur: psycopg.Cursor[Any], message_id: int, source_name: str, source_rowid: int, run_id: int
) -> None:
    """Deliberately NOT guarded by an `IS DISTINCT FROM` clause, unlike
    every other upsert in this module. `extraction_run_id` means "the
    run that last observed this row", so a new run id is a genuine
    change by construction and the guard could never skip anything —
    it would be a comparison that can only ever be true.

    This is therefore the one write a no-op re-extraction still
    performs, once per in-scope message. It is affordable because
    `message_source` carries no `updated_at` column and nothing
    downstream watches it, so it drags no re-segmentation behind it:
    the cost is table churn, not 17 hours of recomputed embeddings."""
    cur.execute(
        """
        INSERT INTO message_source (message_id, source_name, source_rowid, extraction_run_id)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_name, source_rowid) DO UPDATE SET extraction_run_id = EXCLUDED.extraction_run_id
        """,
        (message_id, source_name, source_rowid, run_id),
    )


_MESSAGE_VERSION_COLUMNS: tuple[_Column, ...] = (
    _Column("message_id", Merge.INSERT_ONLY),
    _Column("version_idx", Merge.INSERT_ONLY),
    # A version the shim could not decode arrives as None, which the
    # column's NOT NULL forces to `''` on insert — and `''` written over a
    # version text another source decoded is the same loss as a blanked
    # body. The parameter therefore carries None for "could not decode"
    # and the empty string is only ever an INSERT default.
    _Column("text", Merge.PRESENT, insert_default="''"),
    _Column("edited_at", Merge.PRESENT),
)

_MESSAGE_VERSION_UPSERT_SQL = _build_upsert_sql(
    table="message_version",
    conflict=("message_id", "version_idx"),
    columns=_MESSAGE_VERSION_COLUMNS,
)


def _upsert_message_version(
    cur: psycopg.Cursor[Any], message_id: int, version_idx: int, version: Any
) -> None:
    _execute_upsert(
        cur,
        _MESSAGE_VERSION_UPSERT_SQL,
        _MESSAGE_VERSION_COLUMNS,
        {
            "message_id": message_id,
            "version_idx": version_idx,
            "text": None if version.text is None else strip_nul(version.text),
            "edited_at": _parse_iso(version.edited_at),
        },
    )


_TAPBACK_COLUMNS: tuple[_Column, ...] = (
    _Column("source_guid", Merge.INSERT_ONLY),
    _Column("target_source_guid", Merge.INSERT_ONLY),
    # NULL means the target message is not in this database *yet* — it may
    # be outside the run's scope, or simply later in the same run.
    # `_backfill_tapback_targets` resolves those at the end of the
    # transaction; writing the NULL first would blank an id an earlier run
    # had already resolved, only for the backfill to re-derive it, so this
    # is both a correctness and a churn fix.
    _Column("target_message_id", Merge.PRESENT),
    _Column("sender_source_handle_id", Merge.INSERT_ONLY),
    _Column("is_from_me", Merge.INSERT_ONLY),
    _Column("kind", Merge.INSERT_ONLY),
    # `ImsgDumpMessage.action` defaults to "added" when the shim omits it,
    # so `removed=false` is also what "the shim said nothing" looks like.
    # Un-reacting is a separate event that only ever sets this true.
    _Column("removed", Merge.POSITIVE),
    # `msg.date` — NULL for a tapback row whose `chat.db` date is missing.
    # (Unlike `_upsert_message`, this path has no NOT NULL check to raise
    # on, because `tapback.acted_at` is nullable.)
    _Column("acted_at", Merge.PRESENT),
)

_TAPBACK_UPSERT_SQL = _build_upsert_sql(
    table="tapback", conflict=("source_guid",), columns=_TAPBACK_COLUMNS
)


def _upsert_tapback(
    cur: psycopg.Cursor[Any],
    msg: MessageRow,
    dump_msg: ImsgDumpMessage,
    handle_id_by_rowid: dict[int, int],
) -> None:
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

    _execute_upsert(
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
    )


def _backfill_tapback_targets(cur: psycopg.Cursor[Any]) -> None:
    """A tapback can arrive before the message it targets (out-of-order
    extraction, or the target is outside this run's scope). Resolve any
    still-unresolved `target_message_id` whose target has since landed
    (SPEC §7.2: "backfilled when target is present")."""
    cur.execute(
        """
        UPDATE tapback t
        SET target_message_id = m.message_id
        FROM message m
        WHERE t.target_message_id IS NULL AND m.source_guid = t.target_source_guid
        """
    )


_LINK_PREVIEW_COLUMNS: tuple[_Column, ...] = (
    _Column("message_id", Merge.INSERT_ONLY),
    _Column("url", Merge.INSERT_ONLY),
    # `parse_link_preview` is a best-effort walk of an NSKeyedArchiver
    # blob: a key it does not find comes back NULL, whether the preview
    # genuinely had no title or this source's `payload_data` was truncated
    # or shaped differently. Same policy as everywhere else.
    _Column("title", Merge.PRESENT),
    _Column("summary", Merge.PRESENT),
    _Column("site_name", Merge.PRESENT),
)

_LINK_PREVIEW_UPSERT_SQL = _build_upsert_sql(
    table="link_preview", conflict=("message_id", "url"), columns=_LINK_PREVIEW_COLUMNS
)


def _upsert_link_preview(cur: psycopg.Cursor[Any], message_id: int, preview: LinkPreview) -> None:
    _execute_upsert(
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
    )


__all__ = [
    "APPLE_EPOCH",
    "AttachmentRow",
    "ChatRow",
    "ExtractResult",
    "HandleRow",
    "LinkPreview",
    "MessageRow",
    "SnapshotReader",
    "UpsertCounts",
    "UpsertOutcome",
    "parse_link_preview",
    "run_extract",
]
