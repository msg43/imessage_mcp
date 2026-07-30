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
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import apsw
import psycopg
import structlog

from imsg.errors import ExtractionError
from imsg.hashing import sha256_file
from imsg.keys import attachment_key, message_key, thread_key
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun, run_imsg_dump
from imsg.textnorm import normalize_text

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


def _normalize_service(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    return _KNOWN_SERVICES.get(raw.strip().lower(), "unknown")


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
    return apsw.Connection(path, flags=apsw.SQLITE_OPEN_READONLY)


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
        placeholders = ",".join("?" for _ in message_rowids)
        joins = list(
            self._conn.execute(
                f"SELECT message_id, attachment_id FROM message_attachment_join "
                f"WHERE message_id IN ({placeholders}) ORDER BY message_id, attachment_id",
                message_rowids,
            )
        )
        by_message: dict[int, list[int]] = {}
        attachment_ids: list[int] = []
        for message_id, attachment_id in joins:
            by_message.setdefault(message_id, []).append(attachment_id)
            attachment_ids.append(attachment_id)

        if not attachment_ids:
            return [], by_message

        att_placeholders = ",".join("?" for _ in attachment_ids)
        rows = list(
            self._conn.execute(
                f"SELECT ROWID, guid, filename, transfer_name, uti, mime_type, "
                f"total_bytes, COALESCE(is_sticker, 0) FROM attachment "
                f"WHERE ROWID IN ({att_placeholders})",
                attachment_ids,
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
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "UPDATE extraction_run SET status = 'failed', finished_at = now() WHERE run_id = %s",
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
        chats_upserted = 0
        for chat in chats:
            chat_id_by_rowid[chat.rowid] = _upsert_chat(cur, chat)
            chats_upserted += 1

        handle_id_by_rowid: dict[int, int] = {}
        handles_upserted = 0
        for handle in handles:
            handle_id_by_rowid[handle.rowid] = _upsert_source_handle(cur, handle)
            handles_upserted += 1

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
        attachments_upserted = 0
        for att in attachments:
            attachment_id_by_rowid[att.rowid] = _upsert_attachment(cur, att)
            _upsert_attachment_source(cur, source_name, att.rowid, attachment_id_by_rowid[att.rowid])
            attachments_upserted += 1

        messages_upserted = 0
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

            message_id = _upsert_message(
                cur, msg, dump_msg, chat_id=chat_id, handle_id_by_rowid=handle_id_by_rowid,
                has_attachments=bool(attachments_by_message.get(msg.rowid)),
            )
            messages_upserted += 1

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

        cur.execute(
            """
            UPDATE extraction_run
            SET status = 'ok', finished_at = now(), rowid_after = %s, messages_upserted = %s
            WHERE run_id = %s
            """,
            (snapshot_max_rowid, messages_upserted, run_id),
        )

    return ExtractResult(
        run_id=run_id,
        watermark_before=watermark_before,
        watermark_after=snapshot_max_rowid,
        chats_upserted=chats_upserted,
        handles_upserted=handles_upserted,
        messages_upserted=messages_upserted,
        tapbacks_upserted=tapbacks_upserted,
        system_messages_skipped=system_messages_skipped,
        attachments_upserted=attachments_upserted,
        link_previews_upserted=link_previews_upserted,
        bodies_missing=bodies_missing,
        dump_stderr_line_count=len(dump_run.stderr_lines),
    )


# --------------------------------------------------------------------------
# Postgres upsert helpers
# --------------------------------------------------------------------------


def _upsert_chat(cur: psycopg.Cursor[Any], chat: ChatRow) -> int:
    if chat.style == CHAT_STYLE_GROUP:
        kind = "group"
    elif chat.style == CHAT_STYLE_DM:
        kind = "dm"
    else:
        # Unexpected/unknown `style` value: fall back to a participant-count
        # heuristic rather than guessing a magic number wrong (logged so an
        # unexpected chat.db `style` value is visible, not silent).
        kind = "group" if chat.participant_count > 1 else "dm"
        logger.warning(
            "extract.unexpected_chat_style", guid=chat.guid, style=chat.style, fallback_kind=kind
        )

    cur.execute(
        """
        INSERT INTO chat (source_guid, thread_key, kind, display_name, service)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_guid) DO UPDATE SET
            kind = EXCLUDED.kind, display_name = EXCLUDED.display_name, service = EXCLUDED.service
        RETURNING chat_id
        """,
        (chat.guid, thread_key(chat.guid), kind, chat.display_name, _normalize_service(chat.service_name)),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _upsert_source_handle(cur: psycopg.Cursor[Any], handle: HandleRow) -> int:
    cur.execute(
        """
        INSERT INTO source_handle (raw_value, service)
        VALUES (%s, %s)
        ON CONFLICT (raw_value, service) DO UPDATE SET raw_value = EXCLUDED.raw_value
        RETURNING source_handle_id
        """,
        (handle.raw_value, _normalize_service(handle.service)),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _upsert_attachment(cur: psycopg.Cursor[Any], att: AttachmentRow) -> int:
    cur.execute(
        """
        INSERT INTO attachment (
            source_guid, attachment_key, filename, source_path, uti, mime_type, byte_size, is_sticker
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_guid) DO UPDATE SET
            filename = EXCLUDED.filename, source_path = EXCLUDED.source_path,
            uti = EXCLUDED.uti, mime_type = EXCLUDED.mime_type, byte_size = EXCLUDED.byte_size,
            is_sticker = EXCLUDED.is_sticker, updated_at = now()
        RETURNING attachment_id
        """,
        (
            att.guid,
            attachment_key(att.guid),
            att.filename,
            att.source_path,
            att.uti,
            att.mime_type,
            att.byte_size,
            att.is_sticker,
        ),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _upsert_attachment_source(
    cur: psycopg.Cursor[Any], source_name: str, source_rowid: int, attachment_id: int
) -> None:
    cur.execute(
        """
        INSERT INTO attachment_source (attachment_id, source_name, source_rowid)
        VALUES (%s, %s, %s)
        ON CONFLICT (source_name, source_rowid) DO UPDATE SET attachment_id = EXCLUDED.attachment_id
        """,
        (attachment_id, source_name, source_rowid),
    )


def _upsert_message(
    cur: psycopg.Cursor[Any],
    msg: MessageRow,
    dump_msg: ImsgDumpMessage | None,
    *,
    chat_id: int,
    handle_id_by_rowid: dict[int, int],
    has_attachments: bool,
) -> int:
    sender_source_handle_id = None
    if not msg.is_from_me and msg.handle_rowid is not None:
        sender_source_handle_id = handle_id_by_rowid.get(msg.handle_rowid)

    body_text = dump_msg.body_text if dump_msg is not None else None
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

    cur.execute(
        """
        INSERT INTO message (
            source_guid, message_key, chat_id, sender_source_handle_id, is_from_me, sent_at,
            service, text_original, text_normalized, is_unsent, is_edited, date_edited,
            reply_to_guid, has_attachments
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_guid) DO UPDATE SET
            text_original = EXCLUDED.text_original, text_normalized = EXCLUDED.text_normalized,
            is_unsent = EXCLUDED.is_unsent, is_edited = EXCLUDED.is_edited,
            date_edited = EXCLUDED.date_edited, has_attachments = EXCLUDED.has_attachments,
            updated_at = now()
        RETURNING message_id
        """,
        (
            msg.guid,
            message_key(msg.guid),
            chat_id,
            sender_source_handle_id,
            msg.is_from_me,
            msg.date,
            _normalize_service(msg.service),
            body_text,
            text_for_index,
            is_unsent,
            is_edited,
            msg.date_edited,
            reply_to_guid,
            has_attachments,
        ),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _upsert_message_source(
    cur: psycopg.Cursor[Any], message_id: int, source_name: str, source_rowid: int, run_id: int
) -> None:
    cur.execute(
        """
        INSERT INTO message_source (message_id, source_name, source_rowid, extraction_run_id)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_name, source_rowid) DO UPDATE SET extraction_run_id = EXCLUDED.extraction_run_id
        """,
        (message_id, source_name, source_rowid, run_id),
    )


def _upsert_message_version(
    cur: psycopg.Cursor[Any], message_id: int, version_idx: int, version: Any
) -> None:
    edited_at = _parse_iso(version.edited_at)
    cur.execute(
        """
        INSERT INTO message_version (message_id, version_idx, text, edited_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (message_id, version_idx) DO UPDATE SET
            text = EXCLUDED.text, edited_at = EXCLUDED.edited_at
        """,
        (message_id, version_idx, version.text or "", edited_at),
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

    cur.execute(
        """
        INSERT INTO tapback (
            source_guid, target_source_guid, target_message_id, sender_source_handle_id,
            is_from_me, kind, removed, acted_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_guid) DO UPDATE SET
            target_message_id = EXCLUDED.target_message_id, removed = EXCLUDED.removed,
            acted_at = EXCLUDED.acted_at
        """,
        (
            msg.guid,
            dump_msg.tapback.target_guid,
            target_message_id,
            sender_source_handle_id,
            msg.is_from_me,
            kind,
            removed,
            msg.date,
        ),
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


def _upsert_link_preview(cur: psycopg.Cursor[Any], message_id: int, preview: LinkPreview) -> None:
    cur.execute(
        """
        INSERT INTO link_preview (message_id, url, title, summary, site_name)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (message_id, url) DO UPDATE SET
            title = EXCLUDED.title, summary = EXCLUDED.summary, site_name = EXCLUDED.site_name
        """,
        (message_id, preview.url, preview.title, preview.summary, preview.site_name),
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
    "parse_link_preview",
    "run_extract",
]
