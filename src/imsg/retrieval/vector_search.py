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

**Simplification, flagged**: pgvector does not surface an explicit
"iterative scan hit its cap" signal over plain SQL. This module treats
"returned fewer than the requested `k`" as the metric SPEC §9.4 step 5
asks for ("selectivity **or** the cap" — the spec itself groups both
causes under one flag), rather than trying to distinguish "the corpus
genuinely has fewer than k matches" from "the scan cap was hit". See
the build report for the same caveat repeated once, not per-channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.embed.vector_codec import vector_literal
from imsg.retrieval.filters import CompiledPredicate

if TYPE_CHECKING:
    import psycopg

_SET_ITERATIVE_SCAN = "SET LOCAL hnsw.iterative_scan = 'strict_order'"


@dataclass(frozen=True, slots=True)
class VectorChannelResult:
    segment_ids: tuple[int, ...]
    scan_cap_reached: bool


def _result(rows: list[tuple[int, ...]], k: int) -> VectorChannelResult:
    ids = [int(r[0]) for r in rows]
    return VectorChannelResult(segment_ids=tuple(ids), scan_cap_reached=len(ids) < k)


def search_segment_vector(
    conn: psycopg.Connection, query_vector: list[float], predicate: CompiledPredicate, k: int
) -> VectorChannelResult:
    """Channel B1: primary text vector search over `segment_embedding`."""
    qv = vector_literal(query_vector)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(_SET_ITERATIVE_SCAN)
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
    conn: psycopg.Connection, query_vector: list[float], predicate: CompiledPredicate, k: int
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
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(_SET_ITERATIVE_SCAN)
        cur.execute(
            f"""
            SELECT s.segment_id, ace.vec <=> %(qv)s::halfvec AS distance
            FROM attachment_chunk_embedding ace
            JOIN attachment_chunk ac ON ac.chunk_id = ace.chunk_id
            JOIN message_attachment ma ON ma.attachment_id = ac.attachment_id
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE {predicate.sql}
            ORDER BY ace.vec <=> %(qv)s::halfvec
            LIMIT %(chunk_limit)s
            """,
            {"qv": qv, "chunk_limit": chunk_limit, **predicate.params},
        )
        rows = cur.fetchall()
    return _best_distance_per_segment(rows, k)


def search_multimodal_vector(
    conn: psycopg.Connection, query_vector: list[float], predicate: CompiledPredicate, k: int
) -> VectorChannelResult:
    """Channel C: secondary multimodal vector search over
    `attachment_mm_embedding` (SPEC §9.5, D3a) — active whenever the
    caller supplies a query vector (query-side text embedded through
    the multimodal provider's text tower); the caller is responsible
    for honoring `embedding.multimodal.enabled` (this function has no
    config access and always searches if asked to)."""
    qv = vector_literal(query_vector)
    attachment_limit = max(k * 5, 50)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(_SET_ITERATIVE_SCAN)
        cur.execute(
            f"""
            SELECT s.segment_id, mm.vec <=> %(qv)s::halfvec AS distance
            FROM attachment_mm_embedding mm
            JOIN message_attachment ma ON ma.attachment_id = mm.attachment_id
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE {predicate.sql}
            ORDER BY mm.vec <=> %(qv)s::halfvec
            LIMIT %(attachment_limit)s
            """,
            {"qv": qv, "attachment_limit": attachment_limit, **predicate.params},
        )
        rows = cur.fetchall()
    return _best_distance_per_segment(rows, k)


def _best_distance_per_segment(rows: list[tuple[int, float]], k: int) -> VectorChannelResult:
    best: dict[int, float] = {}
    for segment_id, distance in rows:
        if segment_id not in best or distance < best[segment_id]:
            best[segment_id] = distance
    ordered = sorted(best.items(), key=lambda kv: kv[1])[:k]
    ids = [sid for sid, _ in ordered]
    return VectorChannelResult(segment_ids=tuple(ids), scan_cap_reached=len(ids) < k)


__all__ = [
    "VectorChannelResult",
    "search_attachment_chunk_vector",
    "search_multimodal_vector",
    "search_segment_vector",
]
