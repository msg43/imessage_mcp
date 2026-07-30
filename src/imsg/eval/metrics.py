"""Graded-relevance retrieval metrics (SPEC §13.3): nDCG@k, pooled
Recall@k, reciprocal rank / MRR, success@k, and judged coverage.

Every function here is pure — a ranked list of ids plus a `{id: grade}`
label map in, a number out — precisely so it can be tested against
hand-computed fixtures independent of the database, the retrieval
service, or any config (see `tests/test_eval_metrics.py`). Retrieval
quality claims elsewhere in this codebase must route through these
functions rather than re-deriving a metric inline: "A silently wrong
metric is worse than no metric, because every later retrieval decision
routes through it" (build brief).

**Unjudged handling (SPEC §13.2/§13.3, explicit by design):** an id
with no entry in the `grades` mapping is *unjudged*, not negative.
- In `ndcg_at_k`, an unjudged id contributes gain 0 at its rank
  position (it neither helps nor is excluded — it still occupies a
  slot and discounts whatever ranks below it). The ideal ranking is
  built from every *known* grade for the query, not just the ones the
  ranked list happened to retrieve, so a relevant item the retriever
  missed entirely still depresses the score.
- In `recall_at_k_pooled`, the denominator is the count of
  known-relevant (`grade >= 1`) ids **in the label set**, not the
  corpus — SPEC §13.3: "Recall's denominator is known relevant items
  in the judged pool and is labeled `pooled_recall`, not corpus-wide
  recall." A query with zero relevant labels returns `None` (no
  meaningful denominator), not 0 or 1.
- A query with **zero labels at all** returns `None` from `ndcg_at_k`
  (no ideal ranking exists to divide by) — callers exclude it from an
  aggregate mean rather than scoring it 0, which would penalize
  queries nobody has judged yet rather than the retriever.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imsg.eval.models import QueryRunResult

RELEVANT_GRADE_THRESHOLD = 1
"""grade >= this counts as "relevant" for recall/MRR/success@k (grade
0 = not relevant, 1 = relevant, 2 = highly relevant — SPEC §13.1)."""


def _dcg(gains: Sequence[float]) -> float:
    """Sum of gain[i] / log2(rank+1) for 1-indexed rank — `gains` is
    already in rank order, `gains[0]` at rank 1."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int) -> float | None:
    """Graded nDCG@k. Gain = 2**grade - 1 (standard exponential gain,
    so a highly-relevant hit (grade 2, gain 3) outweighs three
    merely-relevant hits (grade 1, gain 1) at the same rank); discount
    = log2(rank+1), rank 1-indexed.

    Returns `None` when `grades` is empty (no labels for this query at
    all — see module docstring).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not grades:
        return None
    actual_gains = [float(2 ** grades.get(doc_id, 0) - 1) for doc_id in ranked_ids[:k]]
    dcg = _dcg(actual_gains)
    ideal_gains = [float(2**g - 1) for g in sorted(grades.values(), reverse=True)[:k]]
    idcg = _dcg(ideal_gains)
    if idcg == 0:
        # every known label for this query is grade 0 — no positive
        # ideal DCG exists to divide by.
        return None
    return dcg / idcg


def recall_at_k_pooled(
    ranked_ids: Sequence[str], grades: Mapping[str, int], k: int
) -> float | None:
    """Pooled Recall@k (SPEC §13.3, D6): fraction of known-relevant
    (`grade >= RELEVANT_GRADE_THRESHOLD`) labeled ids that appear in
    the top-k of `ranked_ids`. The denominator is the labeled/pooled
    relevant set, not the corpus — this is `pooled_recall`, explicitly
    not corpus-wide recall (SPEC §13.3).

    Returns `None` when there are zero relevant labels for this query
    (no meaningful denominator).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = {
        doc_id for doc_id, grade in grades.items() if grade >= RELEVANT_GRADE_THRESHOLD
    }
    if not relevant_ids:
        return None
    window = set(ranked_ids[:k])
    hits = len(relevant_ids & window)
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """1/rank of the first relevant (`grade >= RELEVANT_GRADE_THRESHOLD`)
    id within the top-k of `ranked_ids`; 0.0 if none. The mean of this
    across queries is MRR (SPEC §13.3)."""
    if k <= 0:
        raise ValueError("k must be positive")
    for i, doc_id in enumerate(ranked_ids[:k]):
        if grades.get(doc_id, 0) >= RELEVANT_GRADE_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


def success_at_k(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int) -> bool:
    """True iff at least one relevant (`grade >= RELEVANT_GRADE_THRESHOLD`)
    id appears in the top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    window = ranked_ids[:k]
    return any(grades.get(doc_id, 0) >= RELEVANT_GRADE_THRESHOLD for doc_id in window)


def judged_coverage(
    ranked_ids: Sequence[str], grades: Mapping[str, int], k: int
) -> tuple[int, int]:
    """`(judged_count, window_size)` — how many of the top-k results
    carry *any* label (SPEC §13.3: "report pool/judgment coverage with
    every metric"). `window_size` can be less than `k` when the
    backend returned fewer than k results."""
    if k <= 0:
        raise ValueError("k must be positive")
    window = ranked_ids[:k]
    judged = sum(1 for doc_id in window if doc_id in grades)
    return judged, len(window)


# ---------------------------------------------------------------------------
# Aggregation across a whole run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """The five per-query numbers this module produces, bundled — a
    convenience for callers that want to compute all of them from one
    `(ranked_ids, grades, k)` triple in one call (`runner.py` does)."""

    ndcg_at_k: float | None
    recall_at_k: float | None
    reciprocal_rank: float
    success_at_k: bool
    judged: int
    window_size: int


def compute_query_metrics(
    ranked_ids: Sequence[str], grades: Mapping[str, int], k: int
) -> QueryMetrics:
    judged, window_size = judged_coverage(ranked_ids, grades, k)
    return QueryMetrics(
        ndcg_at_k=ndcg_at_k(ranked_ids, grades, k),
        recall_at_k=recall_at_k_pooled(ranked_ids, grades, k),
        reciprocal_rank=reciprocal_rank(ranked_ids, grades, k),
        success_at_k=success_at_k(ranked_ids, grades, k),
        judged=judged,
        window_size=window_size,
    )


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Run-level rollup (SPEC §13.3: "aggregate nDCG@10, Recall@10,
    MRR, success@10, judged coverage")."""

    ndcg_at_k_mean: float | None
    recall_at_k_mean: float | None
    mrr: float
    success_at_k_rate: float
    judged_coverage: float
    scored_query_count: int
    """Queries that contributed to `ndcg_at_k_mean` (had >= 1 label)."""
    total_query_count: int


def aggregate_run(per_query: Sequence[QueryRunResult]) -> AggregateMetrics:
    """Roll up a run's per-query results. MRR and success@k are always
    means over *every* query (an unjudged query legitimately scores 0
    on both — there was nothing to find). nDCG/recall means are over
    only the queries that had a score (excluding `None`s per the
    module docstring) — `scored_query_count` reports how many that was
    so a run heavy on unjudged queries doesn't quietly look better than
    it is.
    """
    total = len(per_query)
    ndcg_values = [q.ndcg_at_k for q in per_query if q.ndcg_at_k is not None]
    recall_values = [q.recall_at_k for q in per_query if q.recall_at_k is not None]
    judged_sum = sum(q.judged_at_k for q in per_query)
    window_sum = sum(len(q.ranked_segment_keys) for q in per_query)
    return AggregateMetrics(
        ndcg_at_k_mean=(sum(ndcg_values) / len(ndcg_values)) if ndcg_values else None,
        recall_at_k_mean=(sum(recall_values) / len(recall_values)) if recall_values else None,
        mrr=(sum(q.reciprocal_rank for q in per_query) / total) if total else 0.0,
        success_at_k_rate=(sum(1 for q in per_query if q.success_at_k) / total) if total else 0.0,
        judged_coverage=(judged_sum / window_sum) if window_sum else 0.0,
        scored_query_count=len(ndcg_values),
        total_query_count=total,
    )


__all__ = [
    "RELEVANT_GRADE_THRESHOLD",
    "AggregateMetrics",
    "QueryMetrics",
    "aggregate_run",
    "compute_query_metrics",
    "judged_coverage",
    "ndcg_at_k",
    "recall_at_k_pooled",
    "reciprocal_rank",
    "success_at_k",
]
