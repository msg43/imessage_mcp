"""Full FTS sidecar rebuild from Postgres (SPEC §7.3/§9.3): captures
the current max `search_index_event.event_id` as a watermark *before*
reading any content, streams every current segment/attachment_chunk
into a fresh `fts.db.new`, verifies row counts before ever touching
the live file, and atomically renames it into place. A later
`imsg.embed.fts.sync.sync_fts` call resumes cleanly from the captured
watermark, self-healing anything that changed during the rebuild.

**Simplification vs. SPEC's literal "repeatable-read snapshot" wording**:
this reads the watermark and the content under Postgres's default READ
COMMITTED isolation (one snapshot per statement, not one for the whole
rebuild) rather than opening an explicit REPEATABLE READ transaction —
avoids coupling this function to the caller's own transaction/autocommit
setup, which varies across call sites. The residual risk (a row
committed between the watermark read and the content scan reads
"content that arrived one event too early") is bounded and self-heals
on the very next `sync_fts` call, which starts strictly after the
captured watermark. Worth tightening at Phase 3+ if it ever matters in
practice; flagged here rather than silently narrowed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import apsw

from imsg.embed.fts.schema import create_schema, set_meta
from imsg.embed.fts.sync import upsert_chunk_row, upsert_segment_row
from imsg.errors import FtsSidecarError

if TYPE_CHECKING:
    import psycopg


@dataclass(frozen=True, slots=True)
class RebuildReport:
    segments_written: int
    chunks_written: int
    snapshot_event_id: int


def _iter_segments(pg_conn: psycopg.Connection) -> Iterator[tuple[int, str, str]]:
    with pg_conn.cursor() as cur:
        cur.execute("SELECT segment_id, stable_key, rendered_text FROM segment ORDER BY segment_id")
        yield from cur.fetchall()


def _iter_chunks(pg_conn: psycopg.Connection) -> Iterator[tuple[int, int, str]]:
    with pg_conn.cursor() as cur:
        cur.execute("SELECT chunk_id, attachment_id, text FROM attachment_chunk ORDER BY chunk_id")
        yield from cur.fetchall()


def rebuild_fts(pg_conn: psycopg.Connection, fts_db_path: Path) -> RebuildReport:
    fts_db_path.parent.mkdir(parents=True, exist_ok=True)
    new_path = fts_db_path.with_name(fts_db_path.name + ".new")
    if new_path.exists():
        new_path.unlink()

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COALESCE(max(event_id), 0) FROM search_index_event")
        row = cur.fetchone()
        snapshot_event_id = int(row[0]) if row else 0

    new_conn = apsw.Connection(str(new_path))
    try:
        create_schema(new_conn)

        segments_written = 0
        for segment_id, stable_key, rendered_text in _iter_segments(pg_conn):
            with new_conn:
                upsert_segment_row(new_conn, segment_id, stable_key, rendered_text)
            segments_written += 1

        chunks_written = 0
        for chunk_id, attachment_id, text in _iter_chunks(pg_conn):
            with new_conn:
                upsert_chunk_row(new_conn, chunk_id, attachment_id, text)
            chunks_written += 1

        with new_conn:
            set_meta(new_conn, "applied_event_id", str(snapshot_event_id))

        new_cur = new_conn.cursor()
        seg_count_row = new_cur.execute("SELECT count(*) FROM seg_map").fetchone()
        att_count_row = new_cur.execute("SELECT count(*) FROM att_map").fetchone()
        assert seg_count_row is not None
        assert att_count_row is not None
        seg_count = seg_count_row[0]
        att_count = att_count_row[0]
        if seg_count != segments_written or att_count != chunks_written:
            raise FtsSidecarError(
                f"rebuild verification failed: wrote {segments_written} segment(s) / "
                f"{chunks_written} chunk(s) but seg_map/att_map hold "
                f"{seg_count}/{att_count} — refusing to promote 'fts.db.new'"
            )
    finally:
        new_conn.close()

    os.replace(new_path, fts_db_path)
    return RebuildReport(
        segments_written=segments_written,
        chunks_written=chunks_written,
        snapshot_event_id=snapshot_event_id,
    )


__all__ = ["RebuildReport", "rebuild_fts"]
