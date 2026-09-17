"""Channels B (primary text vector) and C (secondary multimodal
vector) — SPEC §9.4 step 3/5, §9.5.

The authorization/filter predicate is embedded directly in each
candidate query's `WHERE` clause (unlike the FTS channels, which
authorize a fetched batch afterward — SQLite has no access to the
Postgres predicate at all). SPEC §9.4 step 5: "Vector SQL carries the
predicate in the candidate query with `SET LOCAL hnsw.iterative_scan =
'strict_order'`; if selectivity or the cap yields fewer than `k`,
return the smaller set and emit a metric." Each function here runs
inside its own transaction (`SET LOCAL` only applies for the remainder
of the current transaction) so nothing leaks into whatever the caller
does next on the same long-lived connection.

**How much of the index each search looks at (2026-09-17).**
`hnsw.ef_search` is "the size of the dynamic candidate list for search
(40 by default) ... a higher value provides better recall at the cost of
speed" (pgvector 0.8.6 README §Query Options), and it is also the size of
each batch an iterative scan reads. It is set per search from
`retrieval.hnsw_ef_search`, next to the iterative-scan setting and in the
same transaction. The two interact: with `iterative_scan` on, pgvector
"will automatically scan more of the index until enough results are found
(or it reaches `hnsw.max_scan_tuples`)" — 20,000 by default, and "this is
approximate and does not affect the initial scan" (README §Iterative Index
Scans). So `ef_search` sets what the first pass considers and what each
later pass adds, while `max_scan_tuples` bounds the total; the iterative
scan rescues a filtered query that would otherwise return fewer than `k`
rows (README §Filtering), it does not make a small `ef_search` as accurate
as a large one.

Measured on the live index, 20 fictional queries, warm pages, recall@100
against exact search — `ef_search` 40 / 100 / 200 / 400 / 1000: 0.944,
0.969, 0.989, 0.997, 1.000 for the segment channel and 0.792, 0.887,
0.945, 0.985, 0.998 for the multimodal one, worst single query 0.79 and
0.38 at 40 against 1.00 and 0.98 at 1000. Per-channel p95 over the same
queries: 10.6 and 10.1 ms at 40 against 45.5 and 27.3 ms at 1000. The
default is 1000 (pgvector's maximum): the whole-query budget is 2 s and
the reranker spends most of it, so recall is the better use of 50 ms.

**Simplification, flagged**: pgvector does not surface an explicit
"iterative scan hit its cap" signal over plain SQL. This module treats
"returned fewer than the requested `k`" as the metric SPEC §9.4 step 5
asks for ("selectivity **or** the cap" — the spec itself groups both
causes under one flag), rather than trying to distinguish "the corpus
genuinely has fewer than k matches" from "the scan cap was hit". See
the build report for the same caveat repeated once, not per-channel.

**The collapsing channels stop reading early (2026-09-16).** Channels
B2 and C overfetch `max(5k, 50)` item-level rows and collapse them to
the `k` nearest segments. Measured on the real corpus (20 fictional
queries, `k` = 100), channel C's 500 rows held 316-465 distinct
segments, and the 100th distinct segment had appeared by row 115
(median) / 157 (max): reading all 500 did three to four times the index
and heap work the answer needs, which is what a query pays in page reads
when the index is not cached. The query is unchanged, but its rows are
now pulled through a server-side cursor in batches, and reading stops
once no unread row can change the collapsed result — see
:func:`collapse_is_settled`. The result is identical to collapsing the
whole overfetch: on the real corpus, 40 of 40 fictional queries returned
the same segments in the same order, reading 192 rows instead of 500
(channel C per call, repeated queries: median 20.0 -> 8.9 ms, p95 157 ->
46 ms).
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from imsg.embed.vector_codec import vector_literal
from imsg.retrieval.filters import CompiledPredicate

if TYPE_CHECKING:
    import psycopg

_SET_ITERATIVE_SCAN = "SET LOCAL hnsw.iterative_scan = 'strict_order'"

_SET_EF_SEARCH = "SELECT set_config('hnsw.ef_search', %(ef_search)s, true)"
"""`SET LOCAL hnsw.ef_search`, written as `set_config(..., is_local =>
true)` because `SET` takes no query parameters. pgvector's own range is
1..1000 (rejected at 1001 by the live instance, pgvector 0.8.6,
2026-09-17); `imsg.config.schema` bounds the config field the same way."""

_KEEP_THE_INDEX = "SET LOCAL enable_seqscan = off"
"""Keep the planner on the HNSW index — pgvector's own remedy ("You can
encourage the planner to use an index for a query with: BEGIN; SET LOCAL
enable_seqscan = off; ...", README §"Why isn't my index being used?",
pgvector 0.8.6).

It is needed because the planner's two estimates are not comparable here
(all numbers measured on the live index, 2026-09-17, `EXPLAIN`):

- The sequential alternative is undercosted. pgvector says so directly:
  "The planner doesn't consider out-of-line storage in cost estimates,
  which can make a serial scan look cheaper" (same README) — and the
  vectors this channel sorts are exactly that, 642 MiB of TOAST against a
  25 MiB main fork. Its estimate is a flat 8,827; running it touched
  538,197 buffers and took 240-260 ms, against 4,032 buffers and 10-18 ms
  for the index scan at `ef_search` 400.
- The index estimate is not monotonic in `ef_search`. pgvector's
  `hnswcostestimate` bounds layer-0 tuples by `HnswGetLayerM(m, 0) *
  hnsw_ef_search` while its selectivity term carries `1 + log(ef_search)`
  in the denominator (src/hnsw.c, v0.8.6), so the estimate climbs with
  `ef_search` until the other term takes over and it drops back: 3,252 at
  40, 8,558 at 160, 13,287 at 280, then 3,719 at 285 and 3,765 at 1,000.
  Between those two curves — `ef_search` 170 through 284 here — the seq
  scan wins the comparison, and the same query that took 9 ms at 160 took
  245 ms at 200.

Without this, the search's latency would depend on where a tuned
`ef_search` happened to fall against a cost curve, and on a corpus size
that moves the seq-scan estimate. A penalty, not a prohibition: with no
usable index the planner still scans, so the channel degrades instead of
failing."""


def _apply_scan_settings(cur: Any, ef_search: int | None) -> None:
    """The GUCs every vector channel runs under, for the remainder of the
    caller's transaction. `ef_search` of `None` leaves the server's own
    value (40 by default) alone."""
    cur.execute(_SET_ITERATIVE_SCAN)
    cur.execute(_KEEP_THE_INDEX)
    if ef_search is not None:
        cur.execute(_SET_EF_SEARCH, {"ef_search": str(int(ef_search))})

_PLAN_CURSOR_LIKE_A_QUERY = "SET LOCAL cursor_tuple_fraction = 1.0"
"""PostgreSQL plans a `DECLARE ... CURSOR` for fast start — for the first
`cursor_tuple_fraction` (default 0.1) of its rows — so the cursor can get a
different plan, and different rows, from the same SELECT run plainly.
Observed on a small scratch table (2026-09-16, `auto_explain`): the plain
query sorted exactly over hash joins while the cursor at 0.1 chose the
approximate HNSW index scan, and an integration test that compared the two
failed intermittently. At 1.0 the cursor is planned exactly like the
plain query; the executor still produces rows only as they are fetched,
so reading stops early all the same."""

COLLAPSE_FETCH_ROWS = 64
"""Rows per round trip while a collapsing channel streams its overfetch.
The 100th distinct segment arrived by row 157 at worst on the measured
query set, so a `k` = 100 search usually settles within three fetches."""

DISTANCE_ORDER_TOLERANCE = 1e-3
"""How much closer a later row may be than an earlier one. With
`hnsw.iterative_scan = strict_order` the index returns rows in order of
its own distance — pgvector 0.8.6's `halfvec_cosine_ops` for HNSW orders
by `halfvec_negative_inner_product` over `l2_norm`-normalized half-
precision vectors — while the `<=>` the query selects is the cosine
distance recomputed from the stored vectors, so the two orders differ by
rounding. Measured on the real corpus (20 queries, 500 rows each): a
later row was at most 3.7e-5 closer than an earlier one, up to 13
positions out of place. The early stop keeps a margin 27 times that; a
query whose rows were further out of order than this could collapse
differently from reading every row."""


@dataclass(frozen=True, slots=True)
class VectorChannelResult:
    segment_ids: tuple[int, ...]
    scan_cap_reached: bool


def _result(rows: list[tuple[int, ...]], k: int) -> VectorChannelResult:
    ids = [int(r[0]) for r in rows]
    return VectorChannelResult(segment_ids=tuple(ids), scan_cap_reached=len(ids) < k)


def search_segment_vector(
    conn: psycopg.Connection,
    query_vector: list[float],
    predicate: CompiledPredicate,
    k: int,
    *,
    ef_search: int | None = None,
) -> VectorChannelResult:
    """Channel B1: primary text vector search over `segment_embedding`."""
    qv = vector_literal(query_vector)
    with conn.transaction(), conn.cursor() as cur:
        _apply_scan_settings(cur, ef_search)
        cur.execute(
            f"""
            SELECT s.segment_id
            FROM segment_embedding se
            JOIN segment s ON s.segment_id = se.segment_id
            WHERE {predicate.sql}
            ORDER BY se.vec <=> %(qv)s::halfvec
            LIMIT %(k)s
            """,
            {"qv": qv, "k": k, **predicate.params},
        )
        rows = cur.fetchall()
    return _result(rows, k)


def search_attachment_chunk_vector(
    conn: psycopg.Connection,
    query_vector: list[float],
    predicate: CompiledPredicate,
    k: int,
    *,
    ef_search: int | None = None,
) -> VectorChannelResult:
    """Channel B2: primary text vector search over
    `attachment_chunk_embedding`, mapped to parent segments and
    deduplicated to the best (nearest) occurrence per segment (SPEC
    §9.4 step 4)."""
    qv = vector_literal(query_vector)
    # Overfetch chunk-level hits before collapsing to one row per segment,
    # for the same reason `imsg.retrieval.fts_search` overfetches: the
    # nearest chunk for a given segment may not be within the first `k`
    # chunk-level rows once duplicates from other segments are removed.
    chunk_limit = max(k * 5, 50)
    return _collapsed_nearest_segments(
        conn,
        f"""
        SELECT s.segment_id, ace.vec <=> %(qv)s::halfvec AS distance
        FROM attachment_chunk_embedding ace
        JOIN attachment_chunk ac ON ac.chunk_id = ace.chunk_id
        JOIN message_attachment ma ON ma.attachment_id = ac.attachment_id
        JOIN segment_message sm ON sm.message_id = ma.message_id
        JOIN segment s ON s.segment_id = sm.segment_id
        WHERE {predicate.sql}
        ORDER BY ace.vec <=> %(qv)s::halfvec
        LIMIT %(row_limit)s
        """,
        {"qv": qv, "row_limit": chunk_limit, **predicate.params},
        k,
        ef_search=ef_search,
    )


def search_multimodal_vector(
    conn: psycopg.Connection,
    query_vector: list[float],
    predicate: CompiledPredicate,
    k: int,
    *,
    ef_search: int | None = None,
) -> VectorChannelResult:
    """Channel C: secondary multimodal vector search over
    `attachment_mm_embedding` (SPEC §9.5, D3a) — active whenever the
    caller supplies a query vector (query-side text embedded through
    the multimodal provider's text tower); the caller is responsible
    for honoring `embedding.multimodal.enabled` (this function has no
    config access and always searches if asked to)."""
    qv = vector_literal(query_vector)
    attachment_limit = max(k * 5, 50)
    return _collapsed_nearest_segments(
        conn,
        f"""
        SELECT s.segment_id, mm.vec <=> %(qv)s::halfvec AS distance
        FROM attachment_mm_embedding mm
        JOIN message_attachment ma ON ma.attachment_id = mm.attachment_id
        JOIN segment_message sm ON sm.message_id = ma.message_id
        JOIN segment s ON s.segment_id = sm.segment_id
        WHERE {predicate.sql}
        ORDER BY mm.vec <=> %(qv)s::halfvec
        LIMIT %(row_limit)s
        """,
        {"qv": qv, "row_limit": attachment_limit, **predicate.params},
        k,
        ef_search=ef_search,
    )


def collapse_is_settled(
    best: dict[int, float],
    farthest_read: float,
    k: int,
    *,
    tolerance: float = DISTANCE_ORDER_TOLERANCE,
) -> bool:
    """Whether reading more rows could still change the `k` segments
    :func:`_best_distance_per_segment` keeps (or their order).

    `best` holds each segment's smallest distance among the rows read so
    far and `farthest_read` the largest distance read. Rows arrive in
    distance order to within `tolerance`, so every unread row is at least
    `farthest_read - tolerance` away; once that exceeds the `k`-th best
    distance, no unread row can enter the top `k`, improve a segment in
    it, or tie with one."""
    if len(best) < k:
        return False
    kth_best = heapq.nsmallest(k, best.values())[-1]
    return farthest_read - tolerance > kth_best


def _collapsed_nearest_segments(
    conn: psycopg.Connection,
    sql: str,
    params: dict[str, Any],
    k: int,
    *,
    ef_search: int | None = None,
) -> VectorChannelResult:
    """Run an item-level `(segment_id, distance)` query ordered by
    distance and collapse it to the `k` nearest segments, reading only as
    many rows as the answer needs (module docstring)."""
    rows: list[tuple[int, float]] = []
    best: dict[int, float] = {}
    farthest = float("-inf")
    with conn.transaction():
        with conn.cursor() as cur:
            _apply_scan_settings(cur, ef_search)
            cur.execute(_PLAN_CURSOR_LIKE_A_QUERY)
        with conn.cursor(name="imsg_vector_overfetch") as cur:
            cur.execute(sql, params)
            while True:
                batch = cur.fetchmany(COLLAPSE_FETCH_ROWS)
                for segment_id, distance in batch:
                    sid, dist = int(segment_id), float(distance)
                    rows.append((sid, dist))
                    if sid not in best or dist < best[sid]:
                        best[sid] = dist
                    farthest = max(farthest, dist)
                if len(batch) < COLLAPSE_FETCH_ROWS or collapse_is_settled(best, farthest, k):
                    break
    return _best_distance_per_segment(rows, k)


def _best_distance_per_segment(
    rows: Sequence[tuple[int, float]], k: int
) -> VectorChannelResult:
    best: dict[int, float] = {}
    for segment_id, distance in rows:
        if segment_id not in best or distance < best[segment_id]:
            best[segment_id] = distance
    ordered = sorted(best.items(), key=lambda kv: kv[1])[:k]
    ids = [sid for sid, _ in ordered]
    return VectorChannelResult(segment_ids=tuple(ids), scan_cap_reached=len(ids) < k)


__all__ = [
    "COLLAPSE_FETCH_ROWS",
    "DISTANCE_ORDER_TOLERANCE",
    "VectorChannelResult",
    "collapse_is_settled",
    "search_attachment_chunk_vector",
    "search_multimodal_vector",
    "search_segment_vector",
]
