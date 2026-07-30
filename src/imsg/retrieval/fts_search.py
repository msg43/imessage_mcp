"""Channel A (SPEC §9.4 steps 3-5): FTS5/BM25 (and trigram/emoji)
candidate generation, authorized against Postgres.

FTS5 lives in a SQLite sidecar that knows nothing about people/date/
attachment filters or public scope (`imsg.embed.fts.schema`'s
docstring: "never the source of truth"), so filtering happens in two
phases per query, exactly as SPEC §9.4 step 5 describes:

1. Fetch up to `scan_cap` raw candidates from SQLite, ranked by FTS5's
   own `rank` column (bm25, both for the ordinary unicode61 tables and
   — same underlying mechanism — the trigram tables).
2. Authorize/filter that whole batch in one Postgres round trip
   (`imsg.retrieval.filters.compile_predicate`), then take the first
   `k` eligible entries in original rank order.

This is a **single-shot overfetch**, not the incrementally-growing
batches SPEC §9.4 step 5's prose literally suggests ("fetch ... continue
until k eligible ... or a scan cap") — flagged as a simplification: the
observable behavior (authorize a pool larger than `k` before truncating,
so a selective filter cannot silently starve the result set — the exact
v1.0 bug D6 calls out) is identical, and a single round trip per channel
is both simpler to reason about and cheaper against a local SQLite file
than adaptive re-querying would be at this corpus scale.

`scan_cap` itself is an internal implementation constant
(`SCAN_CAP_MULTIPLIER * k`) — SPEC §9.4 step 5 says "a configured scan
cap" but §6's config schema has no such key; flagged in the build
report as a spec/config gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.retrieval.filters import CompiledPredicate
from imsg.retrieval.query import (
    AnalyzedQuery,
    bm25_match_expression,
    like_pattern,
    trigram_match_expression,
)

if TYPE_CHECKING:
    import apsw
    import psycopg

SCAN_CAP_MULTIPLIER = 10
MIN_SCAN_CAP = 200


@dataclass(frozen=True, slots=True)
class ChannelResult:
    """One channel's contribution to the hybrid merge: eligible
    `segment_id`s in best-first rank order, plus whether the scan cap
    (or the corpus itself) yielded fewer than the requested `k` (SPEC
    §9.4 step 5's "emit a metric")."""

    segment_ids: tuple[int, ...]
    scan_cap_reached: bool


def _scan_cap(k: int) -> int:
    return max(k * SCAN_CAP_MULTIPLIER, MIN_SCAN_CAP)


# --------------------------------------------------------------------------
# SQLite side: raw ranked candidates
# --------------------------------------------------------------------------


def _fts_match_query(query: AnalyzedQuery) -> str | None:
    """The FTS5 `MATCH` argument for `query`'s mode, or `None` for the
    emoji mode (which never queries the FTS5 tables at all — SPEC
    §7.3: emoji queries route to a `LIKE` scan on `message.
    text_original`/`attachment_chunk.text` in Postgres instead)."""
    if query.mode == "bm25":
        return bm25_match_expression(query.phrase)
    if query.mode == "trigram":
        return trigram_match_expression(query.phrase)
    return None


def _raw_segment_candidates(
    fts_conn: apsw.Connection, query: AnalyzedQuery, limit: int
) -> list[int]:
    match = _fts_match_query(query)
    if match is None:
        return []
    table = "seg_fts_tri" if query.mode == "trigram" else "seg_fts"
    rows = fts_conn.execute(
        f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
        (match, limit),
    )
    # `seg_map.fts_rowid` is defined equal to `segment.segment_id` (SPEC
    # §7.3's schema docstring), so the FTS rowid *is* the segment id —
    # no join back through seg_map is needed to recover it.
    return [int(r[0]) for r in rows]


def _raw_chunk_candidates(
    fts_conn: apsw.Connection, query: AnalyzedQuery, limit: int
) -> list[int]:
    match = _fts_match_query(query)
    if match is None:
        return []
    table = "att_fts_tri" if query.mode == "trigram" else "att_fts"
    rows = fts_conn.execute(
        f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
        (match, limit),
    )
    return [int(r[0]) for r in rows]


# --------------------------------------------------------------------------
# Postgres side: authorize (+ map attachment_chunk hits to parent segments)
# --------------------------------------------------------------------------


def _authorize_segment_ids(
    pg_conn: psycopg.Connection, segment_ids: list[int], predicate: CompiledPredicate
) -> set[int]:
    if not segment_ids:
        return set()
    with pg_conn.transaction(), pg_conn.cursor() as cur:
        cur.execute(
            f"SELECT s.segment_id FROM segment s "
            f"WHERE s.segment_id = ANY(%(ids)s) AND ({predicate.sql})",
            {"ids": segment_ids, **predicate.params},
        )
        return {int(r[0]) for r in cur.fetchall()}


def _map_chunks_to_segments_authorized(
    pg_conn: psycopg.Connection, chunk_ids_in_rank_order: list[int], predicate: CompiledPredicate
) -> list[int]:
    """Map attachment-chunk FTS hits through `attachment_chunk` ->
    `message_attachment` -> `segment_message` to their *current* parent
    segment, authorize, and keep only the best (earliest-ranked)
    occurrence per segment (SPEC §9.4 step 4: "a many-page attachment
    cannot dominate by repetition"). Returns segment ids in best-rank
    order."""
    if not chunk_ids_in_rank_order:
        return []
    ranks = list(range(len(chunk_ids_in_rank_order)))
    with pg_conn.transaction(), pg_conn.cursor() as cur:
        cur.execute(
            f"""
            WITH candidate (chunk_id, rnk) AS (
                SELECT * FROM unnest(%(ids)s::bigint[], %(ranks)s::int[])
            )
            SELECT c.rnk, s.segment_id
            FROM candidate c
            JOIN attachment_chunk ac ON ac.chunk_id = c.chunk_id
            JOIN message_attachment ma ON ma.attachment_id = ac.attachment_id
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE {predicate.sql}
            """,
            {"ids": chunk_ids_in_rank_order, "ranks": ranks, **predicate.params},
        )
        rows = cur.fetchall()
    return _best_rank_per_segment(rows)


def _best_rank_per_segment(rows: list[tuple[int, int]]) -> list[int]:
    best: dict[int, int] = {}
    for rnk, segment_id in rows:
        if segment_id not in best or rnk < best[segment_id]:
            best[segment_id] = rnk
    return [sid for sid, _ in sorted(best.items(), key=lambda kv: kv[1])]


# --------------------------------------------------------------------------
# Emoji path: bounded LIKE scan directly on Postgres (SPEC §7.3)
# --------------------------------------------------------------------------


def _emoji_segment_candidates(
    pg_conn: psycopg.Connection, phrase: str, predicate: CompiledPredicate, k: int
) -> list[int]:
    pattern = like_pattern(phrase)
    with pg_conn.transaction(), pg_conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (s.segment_id) s.segment_id, m.sent_at
            FROM message m
            JOIN segment_message sm ON sm.message_id = m.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE m.text_original LIKE %(pattern)s ESCAPE '\\' AND ({predicate.sql})
            ORDER BY s.segment_id, m.sent_at
            """,
            {"pattern": pattern, **predicate.params},
        )
        rows = cur.fetchall()
    rows.sort(key=lambda r: r[1], reverse=True)  # most recent segment first
    return [int(r[0]) for r in rows[:k]]


def _emoji_chunk_candidates(
    pg_conn: psycopg.Connection, phrase: str, predicate: CompiledPredicate, k: int
) -> list[int]:
    pattern = like_pattern(phrase)
    with pg_conn.transaction(), pg_conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (s.segment_id) s.segment_id
            FROM attachment_chunk ac
            JOIN message_attachment ma ON ma.attachment_id = ac.attachment_id
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE ac.text LIKE %(pattern)s ESCAPE '\\' AND ({predicate.sql})
            ORDER BY s.segment_id
            LIMIT %(k)s
            """,
            {"pattern": pattern, "k": k, **predicate.params},
        )
        rows = cur.fetchall()
    return [int(r[0]) for r in rows]


# --------------------------------------------------------------------------
# public entry points — channel A1 (segment) / A2 (attachment chunk)
# --------------------------------------------------------------------------


def search_segment_fts(
    fts_conn: apsw.Connection,
    pg_conn: psycopg.Connection,
    query: AnalyzedQuery,
    predicate: CompiledPredicate,
    k: int,
) -> ChannelResult:
    """Channel A1: segment BM25/trigram/emoji search, authorized."""
    if query.mode == "emoji":
        ids = _emoji_segment_candidates(pg_conn, query.phrase, predicate, k)
        # The emoji path is already predicate-bounded and LIMIT-ed at the
        # SQL layer, so there is no separate "scan cap" concept to report.
        return ChannelResult(segment_ids=tuple(ids), scan_cap_reached=False)

    cap = _scan_cap(k)
    raw = _raw_segment_candidates(fts_conn, query, cap)
    eligible = _authorize_segment_ids(pg_conn, raw, predicate)
    ordered = [sid for sid in raw if sid in eligible][:k]
    scan_cap_reached = len(ordered) < k and len(raw) >= cap
    return ChannelResult(segment_ids=tuple(ordered), scan_cap_reached=scan_cap_reached)


def search_attachment_chunk_fts(
    fts_conn: apsw.Connection,
    pg_conn: psycopg.Connection,
    query: AnalyzedQuery,
    predicate: CompiledPredicate,
    k: int,
) -> ChannelResult:
    """Channel A2: attachment-chunk BM25/trigram/emoji search, mapped
    to parent segments and authorized."""
    if query.mode == "emoji":
        ids = _emoji_chunk_candidates(pg_conn, query.phrase, predicate, k)
        return ChannelResult(segment_ids=tuple(ids), scan_cap_reached=False)

    cap = _scan_cap(k)
    raw = _raw_chunk_candidates(fts_conn, query, cap)
    mapped = _map_chunks_to_segments_authorized(pg_conn, raw, predicate)
    ordered = mapped[:k]
    scan_cap_reached = len(ordered) < k and len(raw) >= cap
    return ChannelResult(segment_ids=tuple(ordered), scan_cap_reached=scan_cap_reached)


__all__ = [
    "MIN_SCAN_CAP",
    "SCAN_CAP_MULTIPLIER",
    "ChannelResult",
    "search_attachment_chunk_fts",
    "search_segment_fts",
]
