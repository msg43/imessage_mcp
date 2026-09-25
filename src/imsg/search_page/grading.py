"""Grading mode: grade every candidate of one search, for the eval harness.

The reranker evaluation plan of 2026-09-24 needs, for each real query, the
whole candidate list a search produced and a grade for every candidate in
its top 20, given without seeing where each candidate ranked. This module
is that workflow's storage (migration 0011):

- `create_candidate_list` stores one search: the query text and the
  filters as the owner set them, how the list was ranked, and the fused
  list itself, the segment hits in reciprocal-rank-fusion order down to
  position 30 (the order `rerank_candidates` gives the reranker), each with
  its anchor (the GUID of the segment's first message, as SPEC §13.1 and
  `imsg.eval.io.label_segment_by_key` define it), its segment key and a
  copy of its text. Positions 1-20 get a random display order, and so do
  21-30 (shown after the first 20 when the owner asks for ten more).
- `write_grade` records a grade as an ordinary `relevance_label` row,
  source `pool_judgment` (SPEC §13.2's "judge the pool 0/1/2"): 0 not
  relevant, 1 relevant, 2 exactly what was wanted.
- `load_graded_lists` and `fused_order_metrics` read it all back for
  offline scoring: a reranker reorders a list's stored candidates, and
  `imsg.eval.metrics` scores that order against the same grades.

Which eval query a list belongs to: a search without filters uses the
page's own query for that text (a curated query with the same text, else
`adhoc:<text>`), so its grades join the Relevant and Not relevant labels,
`imsg eval run` and the AT-4 check. A filtered search gets its own query,
`adhoc:<text> [<filters>]`, with no retrieval target: `imsg eval run`
cannot apply filters, so it must not score these grades against an
unfiltered search. They stay with their candidate list for offline use.

Unindexed messages (the page's recent-message channel) are left out: they
are in no segment, so neither a reranker nor the eval harness can use
them.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from imsg.eval.io import upsert_query
from imsg.eval.metrics import QueryMetrics, compute_query_metrics
from imsg.eval.models import EvalQuery
from imsg.search_page.labels import ensure_query_id

if TYPE_CHECKING:
    import psycopg

    from imsg.search_page.search import Hit, SearchResult

GRADE_SOURCE = "pool_judgment"
GRADED_POSITIONS = 20
EXTRA_POSITIONS = 10
MAX_POSITIONS = GRADED_POSITIONS + EXTRA_POSITIONS
GRADES = (0, 1, 2)
GRADE_LABELS = {0: "Not relevant", 1: "Relevant", 2: "Exactly what I wanted"}
FILTERED_QUERY_NOTE = (
    "created by the search page's grading mode for a filtered search; filters: {filters}. "
    "No retrieval target: imsg eval run cannot apply the filters."
)


def describe_filters(filters: Mapping[str, Any]) -> str:
    """`people alice, bob; from 2023-04-01; to 2023-04-30; with attachments`."""
    parts: list[str] = []
    people = [str(p) for p in filters.get("people") or []]
    if people:
        parts.append("people " + ", ".join(people))
    if filters.get("from"):
        parts.append(f"from {filters['from']}")
    if filters.get("to"):
        parts.append(f"to {filters['to']}")
    attachments = filters.get("attachments")
    if attachments == "with":
        parts.append("with attachments")
    elif attachments == "without":
        parts.append("without attachments")
    return "; ".join(parts)


def grading_query_id(pg: psycopg.Connection, query_text: str, filters: Mapping[str, Any]) -> str:
    text = " ".join(query_text.split())
    described = describe_filters(filters)
    if not described:
        return ensure_query_id(pg, text)
    query_id = f"adhoc:{text} [{described}]"
    with pg.cursor() as cur:
        cur.execute("SELECT 1 FROM eval_query WHERE query_id = %s", (query_id,))
        if cur.fetchone() is not None:
            return query_id
    upsert_query(
        pg,
        EvalQuery(
            query_id=query_id,
            query_text=text,
            notes=FILTERED_QUERY_NOTE.format(filters=described),
            targets=(),
        ),
    )
    return query_id


def fused_candidates(result: SearchResult, limit: int = MAX_POSITIONS) -> list[Hit]:
    """The segment hits in reciprocal-rank-fusion order: best fused score
    first, newer first on a tie (the order `rerank_candidates` uses)."""
    segments = [h for h in result.hits.values() if h.segment_id is not None]
    segments.sort(key=lambda h: (-h.score, -h.at.timestamp()))
    return segments[:limit]


def display_order(count: int, seed: int) -> list[int]:
    """`order[i]` is where fused position `i + 1` is shown: positions 1-20
    are shuffled among the first 20 places, 21-30 among the next ten."""
    rng = random.Random(seed)
    first = list(range(1, min(count, GRADED_POSITIONS) + 1))
    rest = list(range(GRADED_POSITIONS + 1, count + 1))
    rng.shuffle(first)
    rng.shuffle(rest)
    return first + rest


def create_candidate_list(
    pg: psycopg.Connection,
    result: SearchResult,
    *,
    query_text: str,
    filters: Mapping[str, Any],
    ranking: Mapping[str, Any],
    seed: int,
) -> int:
    """Store the search's fused list (up to 30 segment hits) and return the
    new list's id. One transaction."""
    hits = fused_candidates(result)
    segment_ids = [int(h.segment_id) for h in hits if h.segment_id is not None]
    with pg.transaction():
        query_id = grading_query_id(pg, query_text, filters)
        with pg.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (sm.segment_id) sm.segment_id, m.source_guid
                FROM segment_message sm JOIN message m ON m.message_id = sm.message_id
                WHERE sm.segment_id = ANY(%(ids)s::bigint[])
                ORDER BY sm.segment_id, m.sent_at, m.message_id
                """,
                {"ids": segment_ids},
            )
            anchors = {int(sid): str(guid) for sid, guid in cur.fetchall()}
            cur.execute(
                "SELECT segment_id, stable_key, rendered_text FROM segment "
                "WHERE segment_id = ANY(%(ids)s::bigint[])",
                {"ids": segment_ids},
            )
            segments = {int(sid): (str(key), str(text)) for sid, key, text in cur.fetchall()}
            kept = [h for h in hits if h.segment_id in anchors and h.segment_id in segments]
            cur.execute(
                """
                INSERT INTO eval_candidate_list (query_id, query_text, filters, ranking, shuffle_seed)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s) RETURNING list_id
                """,
                (
                    query_id,
                    " ".join(query_text.split()),
                    json.dumps(dict(filters), sort_keys=True),
                    json.dumps(dict(ranking), sort_keys=True),
                    seed,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            list_id = int(row[0])
            order = display_order(len(kept), seed)
            for position, (hit, shown) in enumerate(zip(kept, order, strict=True), start=1):
                segment_id = int(hit.segment_id or 0)
                key, text = segments[segment_id]
                cur.execute(
                    """
                    INSERT INTO eval_candidate (list_id, fused_rank, anchor_guid, segment_key,
                        fused_score, channel_ranks, segment_text, shown_order)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    """,
                    (
                        list_id,
                        position,
                        anchors[segment_id],
                        key,
                        hit.score,
                        json.dumps(dict(hit.ranks), sort_keys=True),
                        text,
                        shown,
                    ),
                )
    return list_id


@dataclass(frozen=True, slots=True)
class GradedCandidate:
    fused_rank: int
    shown_order: int
    anchor_guid: str
    segment_key: str
    fused_score: float
    channel_ranks: dict[str, int]
    segment_text: str
    grade: int | None
    grade_source: str | None
    current_segment_id: int | None
    """The segment that holds the anchor message now (`None` if none)."""


@dataclass(frozen=True, slots=True)
class GradedList:
    list_id: int
    query_id: str
    query_text: str
    filters: dict[str, Any]
    ranking: dict[str, Any]
    shuffle_seed: int
    created_at: datetime
    candidates: tuple[GradedCandidate, ...]
    """In reciprocal-rank-fusion order."""

    def graded(self, positions: int = GRADED_POSITIONS) -> int:
        return sum(1 for c in self.candidates if c.fused_rank <= positions and c.grade is not None)

    def in_view(self, positions: int) -> list[GradedCandidate]:
        """The candidates the grading view shows, in their display order."""
        return sorted(
            (c for c in self.candidates if c.shown_order <= positions), key=lambda c: c.shown_order
        )


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _load(pg: psycopg.Connection, where: str, params: dict[str, object]) -> list[GradedList]:
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT list_id, query_id, query_text, filters, ranking, shuffle_seed, created_at
            FROM eval_candidate_list WHERE {where} ORDER BY created_at DESC, list_id DESC
            """,
            params,
        )
        lists = cur.fetchall()
        if not lists:
            return []
        ids = [int(r[0]) for r in lists]
        cur.execute(
            """
            SELECT c.list_id, c.fused_rank, c.shown_order, c.anchor_guid, c.segment_key,
                   c.fused_score, c.channel_ranks, c.segment_text, rl.grade, rl.source,
                   sm.segment_id
            FROM eval_candidate c
            JOIN eval_candidate_list l ON l.list_id = c.list_id
            LEFT JOIN relevance_label rl
                   ON rl.query_id = l.query_id AND rl.anchor_guid = c.anchor_guid
            LEFT JOIN message m ON m.source_guid = c.anchor_guid
            LEFT JOIN segment_message sm ON sm.message_id = m.message_id
            WHERE c.list_id = ANY(%(ids)s::bigint[])
            ORDER BY c.list_id, c.fused_rank
            """,
            {"ids": ids},
        )
        by_list: dict[int, list[GradedCandidate]] = {}
        for row in cur.fetchall():
            by_list.setdefault(int(row[0]), []).append(
                GradedCandidate(
                    fused_rank=int(row[1]),
                    shown_order=int(row[2]),
                    anchor_guid=str(row[3]),
                    segment_key=str(row[4]),
                    fused_score=float(row[5]),
                    channel_ranks={str(k): int(v) for k, v in _as_dict(row[6]).items()},
                    segment_text=str(row[7]),
                    grade=int(row[8]) if row[8] is not None else None,
                    grade_source=str(row[9]) if row[9] is not None else None,
                    current_segment_id=int(row[10]) if row[10] is not None else None,
                )
            )
    return [
        GradedList(
            list_id=int(list_id),
            query_id=str(query_id),
            query_text=str(query_text),
            filters=_as_dict(filters),
            ranking=_as_dict(ranking),
            shuffle_seed=int(seed),
            created_at=created,
            candidates=tuple(by_list.get(int(list_id), [])),
        )
        for list_id, query_id, query_text, filters, ranking, seed, created in lists
    ]


def load_graded_list(pg: psycopg.Connection, list_id: int) -> GradedList | None:
    found = _load(pg, "list_id = %(id)s", {"id": list_id})
    return found[0] if found else None


def load_graded_lists(pg: psycopg.Connection) -> list[GradedList]:
    """Every stored list, newest first, each candidate with its current
    grade: what an offline reranker comparison reads."""
    return _load(pg, "TRUE", {})


def fused_order_metrics(graded: GradedList, *, k: int = 10) -> QueryMetrics:
    """The eval harness's metrics for the list's own fused order (the
    "no reranker" row of the evaluation plan). A reranker is scored the
    same way on its reordering of `graded.candidates`."""
    return reordered_metrics(graded, [c.anchor_guid for c in graded.candidates], k=k)


def reordered_metrics(graded: GradedList, anchors: Sequence[str], *, k: int = 10) -> QueryMetrics:
    grades = {c.anchor_guid: c.grade for c in graded.candidates if c.grade is not None}
    return compute_query_metrics(list(anchors), grades, k)


class UnknownCandidateError(ValueError):
    """The grade names a list or candidate that does not exist."""


@dataclass(frozen=True, slots=True)
class GradeOutcome:
    grade: int | None
    graded: int
    graded_extra: int
    total: int


def write_grade(
    pg: psycopg.Connection, *, list_id: int, anchor_guid: str, grade: int | None
) -> GradeOutcome:
    """Set (0, 1 or 2) or clear (`None`) the grade of one candidate. A
    clear removes only a grade this mode wrote; a label from another
    source is replaced by a new grade but never deleted from here."""
    if grade is not None and grade not in GRADES:
        raise ValueError("grade must be 0, 1, 2 or None")
    with pg.transaction(), pg.cursor() as cur:
        cur.execute(
            "SELECT l.query_id FROM eval_candidate c JOIN eval_candidate_list l "
            "ON l.list_id = c.list_id WHERE c.list_id = %s AND c.anchor_guid = %s",
            (list_id, anchor_guid),
        )
        row = cur.fetchone()
        if row is None:
            raise UnknownCandidateError("no such candidate in this list")
        query_id = str(row[0])
        if grade is None:
            cur.execute(
                "DELETE FROM relevance_label WHERE query_id = %s AND anchor_guid = %s "
                "AND source = %s",
                (query_id, anchor_guid, GRADE_SOURCE),
            )
        else:
            cur.execute(
                """
                    INSERT INTO relevance_label (query_id, anchor_guid, grade, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (query_id, anchor_guid) DO UPDATE SET
                        grade = EXCLUDED.grade, source = EXCLUDED.source
                    """,
                (query_id, anchor_guid, grade, GRADE_SOURCE),
            )
    graded = load_graded_list(pg, list_id)
    assert graded is not None
    return GradeOutcome(
        grade=grade,
        graded=graded.graded(GRADED_POSITIONS),
        graded_extra=graded.graded(MAX_POSITIONS),
        total=len(graded.candidates),
    )


__all__ = [
    "EXTRA_POSITIONS",
    "GRADED_POSITIONS",
    "GRADES",
    "GRADE_LABELS",
    "GRADE_SOURCE",
    "MAX_POSITIONS",
    "GradeOutcome",
    "GradedCandidate",
    "GradedList",
    "UnknownCandidateError",
    "create_candidate_list",
    "describe_filters",
    "display_order",
    "fused_candidates",
    "fused_order_metrics",
    "grading_query_id",
    "load_graded_list",
    "load_graded_lists",
    "reordered_metrics",
    "write_grade",
]
