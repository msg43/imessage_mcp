"""Event-driven FTS sync (SPEC §7.3, D6): consumes `search_index_event`
in `event_id` order and advances `meta.applied_event_id` only after a
committed SQLite transaction — **not** a max-entity-id watermark. v1.0
used a watermark and it silently missed deletes, edits, and
re-segmentation replacements landing at a lower id than something
already indexed; the outbox fixes that because every mutation gets its
own strictly-increasing event, processed in order, regardless of which
entity id it touches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from imsg.embed.fts.schema import assert_schema_current, get_applied_event_id, set_meta
from imsg.textnorm import normalize_text

if TYPE_CHECKING:
    import apsw
    import psycopg

DEFAULT_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class SyncEvent:
    event_id: int
    entity_kind: str  # 'segment' | 'attachment_chunk'
    entity_id: int
    operation: str  # 'upsert' | 'delete'


@dataclass
class SyncReport:
    events_applied: int = 0
    upserts: int = 0
    deletes: int = 0
    skipped_missing_content: int = 0
    """Upsert events whose entity no longer exists in Postgres by the
    time this ran — e.g. a segment deleted after the upsert event was
    enqueued but before a later delete event was, if one ever lands.
    Not an error: processing continues; the row is simply absent from
    the sidecar, same end state as if the delete had already run."""
    notes: list[str] = field(default_factory=list)


def _fetch_events_after(
    pg_conn: psycopg.Connection, after_event_id: int, limit: int
) -> list[SyncEvent]:
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, entity_kind, entity_id, operation FROM search_index_event "
            "WHERE event_id > %s ORDER BY event_id LIMIT %s",
            (after_event_id, limit),
        )
        return [SyncEvent(*row) for row in cur.fetchall()]


def _fetch_segment_content(pg_conn: psycopg.Connection, segment_id: int) -> tuple[str, str] | None:
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT stable_key, rendered_text FROM segment WHERE segment_id = %s", (segment_id,)
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else None


def _fetch_chunk_content(pg_conn: psycopg.Connection, chunk_id: int) -> tuple[int, str] | None:
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, text FROM attachment_chunk WHERE chunk_id = %s", (chunk_id,)
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else None


# --- SQLite-side row mutation, shared with `imsg.embed.fts.rebuild` --------


def remove_segment_row(fts_conn: apsw.Connection, segment_id: int) -> None:
    cur = fts_conn.cursor()
    row = cur.execute(
        "SELECT fts_rowid FROM seg_map WHERE segment_id = ?", (segment_id,)
    ).fetchone()
    if row is None:
        return
    fts_rowid = row[0]
    cur.execute("DELETE FROM seg_fts WHERE rowid = ?", (fts_rowid,))
    cur.execute("DELETE FROM seg_fts_tri WHERE rowid = ?", (fts_rowid,))
    cur.execute("DELETE FROM seg_map WHERE fts_rowid = ?", (fts_rowid,))


def upsert_segment_row(
    fts_conn: apsw.Connection, segment_id: int, stable_key: str, rendered_text: str
) -> None:
    remove_segment_row(fts_conn, segment_id)  # contentless tables: no UPDATE, delete+insert
    normalized = normalize_text(rendered_text)
    cur = fts_conn.cursor()
    cur.execute(
        "INSERT INTO seg_map (fts_rowid, segment_id, stable_key) VALUES (?, ?, ?)",
        (segment_id, segment_id, stable_key),
    )
    cur.execute("INSERT INTO seg_fts (rowid, text) VALUES (?, ?)", (segment_id, normalized))
    cur.execute("INSERT INTO seg_fts_tri (rowid, text) VALUES (?, ?)", (segment_id, normalized))


def remove_chunk_row(fts_conn: apsw.Connection, chunk_id: int) -> None:
    cur = fts_conn.cursor()
    row = cur.execute("SELECT fts_rowid FROM att_map WHERE chunk_id = ?", (chunk_id,)).fetchone()
    if row is None:
        return
    fts_rowid = row[0]
    cur.execute("DELETE FROM att_fts WHERE rowid = ?", (fts_rowid,))
    cur.execute("DELETE FROM att_fts_tri WHERE rowid = ?", (fts_rowid,))
    cur.execute("DELETE FROM att_map WHERE fts_rowid = ?", (fts_rowid,))


def upsert_chunk_row(
    fts_conn: apsw.Connection, chunk_id: int, attachment_id: int, text: str
) -> None:
    remove_chunk_row(fts_conn, chunk_id)
    normalized = normalize_text(text)
    cur = fts_conn.cursor()
    cur.execute(
        "INSERT INTO att_map (fts_rowid, chunk_id, attachment_id) VALUES (?, ?, ?)",
        (chunk_id, chunk_id, attachment_id),
    )
    cur.execute("INSERT INTO att_fts (rowid, text) VALUES (?, ?)", (chunk_id, normalized))
    cur.execute("INSERT INTO att_fts_tri (rowid, text) VALUES (?, ?)", (chunk_id, normalized))


# --- the sync loop -----------------------------------------------------


def _apply_one_event(pg_conn: psycopg.Connection, fts_conn: apsw.Connection, event: SyncEvent) -> str:
    """Applies one event inside its own committed SQLite transaction.
    Returns `'upsert' | 'delete' | 'skipped'`."""
    with fts_conn:
        if event.operation == "delete":
            if event.entity_kind == "segment":
                remove_segment_row(fts_conn, event.entity_id)
            else:
                remove_chunk_row(fts_conn, event.entity_id)
            outcome = "delete"
        else:
            if event.entity_kind == "segment":
                content = _fetch_segment_content(pg_conn, event.entity_id)
                if content is None:
                    outcome = "skipped"
                else:
                    stable_key, rendered_text = content
                    upsert_segment_row(fts_conn, event.entity_id, stable_key, rendered_text)
                    outcome = "upsert"
            else:
                chunk_content = _fetch_chunk_content(pg_conn, event.entity_id)
                if chunk_content is None:
                    outcome = "skipped"
                else:
                    attachment_id, text = chunk_content
                    upsert_chunk_row(fts_conn, event.entity_id, attachment_id, text)
                    outcome = "upsert"
        set_meta(fts_conn, "applied_event_id", str(event.event_id))
    return outcome


def sync_fts(
    pg_conn: psycopg.Connection, fts_conn: apsw.Connection, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> SyncReport:
    """Drain every unapplied `search_index_event` in order. Safe to
    call repeatedly (a no-op once caught up) and safe to interrupt at
    any point — `applied_event_id` only ever reflects fully-committed
    SQLite state (SPEC §7.3)."""
    assert_schema_current(fts_conn)
    report = SyncReport()
    while True:
        after = get_applied_event_id(fts_conn)
        events = _fetch_events_after(pg_conn, after, batch_size)
        if not events:
            break
        for event in events:
            outcome = _apply_one_event(pg_conn, fts_conn, event)
            report.events_applied += 1
            if outcome == "upsert":
                report.upserts += 1
            elif outcome == "delete":
                report.deletes += 1
            else:
                report.skipped_missing_content += 1
                report.notes.append(
                    f"event {event.event_id}: {event.entity_kind} {event.entity_id} "
                    f"no longer exists in Postgres — skipped"
                )
    return report


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "SyncEvent",
    "SyncReport",
    "remove_chunk_row",
    "remove_segment_row",
    "sync_fts",
    "upsert_chunk_row",
    "upsert_segment_row",
]
