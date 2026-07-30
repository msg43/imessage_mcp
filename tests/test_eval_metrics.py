"""Known-answer fixtures for the eval metrics (SPEC §13.3).

Every expected value below is computed independently in the test body
(a small explicit loop, not a call into `imsg.eval.metrics`) so this
is a genuine check, not a tautology — "a hand-computed nDCG/MRR
fixture where you can verify the number by inspection" (build brief).
"""

from __future__ import annotations

import math

import pytest

from imsg.eval.metrics import (
    QueryMetrics,
    aggregate_run,
    compute_query_metrics,
    judged_coverage,
    ndcg_at_k,
    recall_at_k_pooled,
    reciprocal_rank,
    success_at_k,
)
from imsg.eval.models import QueryRunResult


def _manual_dcg(gains: list[float]) -> float:
    """Independent re-implementation of the DCG sum, rank 1-indexed."""
    total = 0.0
    for i, g in enumerate(gains):
        rank = i + 1
        total += g / math.log2(rank + 1)
    return total


# ==========================================================================
# ndcg_at_k
# ==========================================================================


def test_ndcg_perfect_ranking_is_1() -> None:
    # Two labels, best-first: DCG == IDCG by construction.
    ranked = ["a", "b"]
    grades = {"a": 2, "b": 1}
    result = ndcg_at_k(ranked, grades, k=10)
    assert result == pytest.approx(1.0)


def test_ndcg_hand_computed_mixed_case() -> None:
    """ranked = [a, b, c, d]; grades = {a: 2, c: 1} (b, d unjudged).

    Raw gain (2**grade - 1) at each rank, then `_manual_dcg` applies
    the log2(rank+1) discount itself:

    DCG@4 raw gains by rank: [a:3, b:0, c:1, d:0]
      rank1 a grade2: gain=2**2-1=3, discount=log2(2)=1  -> 3/1   = 3.0
      rank2 b unjudged: gain=0                             -> 0/log2(3) = 0.0
      rank3 c grade1: gain=2**1-1=1, discount=log2(4)=2   -> 1/2   = 0.5
      rank4 d unjudged: gain=0                             -> 0/log2(5) = 0.0
      DCG = 3.5

    IDCG@4 (ideal = known grades sorted desc = [2, 1] — only 2 labels
    exist for this query, so the ideal ranking is just those two):
      rank1 grade2: raw gain=3, discount=log2(2)=1        -> 3.0
      rank2 grade1: raw gain=1, discount=log2(3)          -> 1/log2(3)
      IDCG = 3 + 1/log2(3)

    nDCG = 3.5 / (3 + 1/log2(3))
    """
    ranked = ["a", "b", "c", "d"]
    grades = {"a": 2, "c": 1}

    expected_dcg = _manual_dcg([3.0, 0.0, 1.0, 0.0])  # raw gains, not pre-discounted
    expected_idcg = _manual_dcg([3.0, 1.0])
    expected = expected_dcg / expected_idcg
    assert expected_dcg == pytest.approx(3.5)
    assert expected_idcg == pytest.approx(3 + 1 / math.log2(3))

    result = ndcg_at_k(ranked, grades, k=4)
    assert result == pytest.approx(expected)
    assert result == pytest.approx(0.96394043, abs=1e-6)


def test_ndcg_no_labels_returns_none() -> None:
    assert ndcg_at_k(["a", "b"], {}, k=10) is None


def test_ndcg_all_zero_grades_returns_none() -> None:
    # Labels exist but every one is grade 0 — no positive ideal DCG.
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=10) is None


def test_ndcg_truncates_ideal_and_actual_at_k() -> None:
    # 3 labels but k=2: both DCG and IDCG only look at the top 2 slots.
    ranked = ["c", "a", "b"]  # c is unjudged and ranked first
    grades = {"a": 2, "b": 2, "d": 1}  # d never retrieved at all
    # DCG@2: rank1 c unjudged -> 0; rank2 a grade2 -> gain3/log2(3)
    expected_dcg = _manual_dcg([0.0, 3.0])
    # IDCG@2: best 2 known grades = [2, 2] (a, b) -> 3/1 + 3/log2(3)
    expected_idcg = _manual_dcg([3.0, 3.0])
    result = ndcg_at_k(ranked, grades, k=2)
    assert result == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k(["a"], {"a": 1}, k=0)


# ==========================================================================
# recall_at_k_pooled
# ==========================================================================


def test_recall_pooled_hand_computed() -> None:
    # relevant (grade>=1) labeled set = {a, c, e} (e never retrieved).
    grades = {"a": 2, "c": 1, "e": 1, "z": 0}
    ranked = ["a", "b", "c", "d"]
    # top-4 contains a, c -> 2 of 3 relevant labels.
    assert recall_at_k_pooled(ranked, grades, k=4) == pytest.approx(2 / 3)
    # top-1 contains only a -> 1 of 3.
    assert recall_at_k_pooled(ranked, grades, k=1) == pytest.approx(1 / 3)


def test_recall_pooled_no_relevant_labels_returns_none() -> None:
    assert recall_at_k_pooled(["a", "b"], {"a": 0, "b": 0}, k=10) is None
    assert recall_at_k_pooled(["a", "b"], {}, k=10) is None


def test_recall_pooled_perfect_when_all_relevant_retrieved() -> None:
    grades = {"a": 1, "b": 2}
    assert recall_at_k_pooled(["x", "a", "b"], grades, k=3) == pytest.approx(1.0)


# ==========================================================================
# reciprocal_rank / MRR
# ==========================================================================


def test_reciprocal_rank_hand_computed() -> None:
    # b unjudged (grade 0 implicit), a grade2, c grade1 -> first
    # relevant hit is "a" at rank 2 -> RR = 1/2.
    ranked = ["b", "a", "c"]
    grades = {"a": 2, "c": 1}
    assert reciprocal_rank(ranked, grades, k=3) == pytest.approx(0.5)


def test_reciprocal_rank_first_position_is_1() -> None:
    assert reciprocal_rank(["a", "b"], {"a": 1}, k=10) == pytest.approx(1.0)


def test_reciprocal_rank_no_relevant_hit_is_zero() -> None:
    assert reciprocal_rank(["a", "b"], {"a": 0, "b": 0}, k=10) == 0.0
    assert reciprocal_rank(["a", "b"], {}, k=10) == 0.0


def test_reciprocal_rank_respects_k_cutoff() -> None:
    # the only relevant doc is at rank 3, but k=2 -> not found within window.
    assert reciprocal_rank(["x", "y", "a"], {"a": 2}, k=2) == 0.0
    assert reciprocal_rank(["x", "y", "a"], {"a": 2}, k=3) == pytest.approx(1 / 3)


def test_mean_reciprocal_rank_across_queries() -> None:
    # Query 1: hit at rank 1 -> RR=1.0. Query 2: hit at rank 2 -> RR=0.5.
    # Query 3: no hit -> RR=0.0. MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5.
    rr1 = reciprocal_rank(["a"], {"a": 1}, k=10)
    rr2 = reciprocal_rank(["x", "a"], {"a": 1}, k=10)
    rr3 = reciprocal_rank(["x", "y"], {"a": 1}, k=10)
    assert (rr1 + rr2 + rr3) / 3 == pytest.approx(0.5)


# ==========================================================================
# success_at_k
# ==========================================================================


def test_success_at_k_true_and_false() -> None:
    assert success_at_k(["a"], {"a": 1}, k=10) is True
    assert success_at_k(["a"], {"a": 0}, k=10) is False
    assert success_at_k(["a"], {}, k=10) is False


def test_success_at_k_respects_cutoff() -> None:
    assert success_at_k(["x", "a"], {"a": 2}, k=1) is False
    assert success_at_k(["x", "a"], {"a": 2}, k=2) is True


# ==========================================================================
# judged_coverage
# ==========================================================================


def test_judged_coverage_hand_computed() -> None:
    ranked = ["a", "b", "c", "d"]
    grades = {"a": 2, "c": 0}  # both judged (grade 0 still counts as judged)
    judged, window = judged_coverage(ranked, grades, k=4)
    assert (judged, window) == (2, 4)


def test_judged_coverage_window_smaller_than_k() -> None:
    # backend returned only 2 results even though k=10
    judged, window = judged_coverage(["a", "b"], {"a": 1}, k=10)
    assert (judged, window) == (1, 2)


# ==========================================================================
# compute_query_metrics — bundles all five from one call
# ==========================================================================


def test_compute_query_metrics_matches_individual_calls() -> None:
    ranked = ["a", "b", "c", "d"]
    grades = {"a": 2, "c": 1}
    bundled = compute_query_metrics(ranked, grades, k=4)
    assert bundled == QueryMetrics(
        ndcg_at_k=ndcg_at_k(ranked, grades, 4),
        recall_at_k=recall_at_k_pooled(ranked, grades, 4),
        reciprocal_rank=reciprocal_rank(ranked, grades, 4),
        success_at_k=success_at_k(ranked, grades, 4),
        judged=2,
        window_size=4,
    )


# ==========================================================================
# aggregate_run
# ==========================================================================


def _qrr(
    *,
    query_id: str,
    ranked: tuple[str, ...],
    grades: dict[str, int],
    k: int = 10,
) -> QueryRunResult:
    m = compute_query_metrics(list(ranked), grades, k)
    return QueryRunResult(
        query_id=query_id,
        query_text=query_id,
        ranked_segment_keys=ranked,
        resolved_grades=grades,
        unresolved_label_count=0,
        ndcg_at_k=m.ndcg_at_k,
        recall_at_k=m.recall_at_k,
        reciprocal_rank=m.reciprocal_rank,
        success_at_k=m.success_at_k,
        judged_at_k=m.judged,
    )


def test_aggregate_run_hand_computed() -> None:
    # q1: perfect single-hit ranking -> ndcg=1.0, recall=1.0, rr=1.0, success=True
    q1 = _qrr(query_id="q1", ranked=("a",), grades={"a": 2})
    # q2: no labels at all -> ndcg=None, recall=None, rr=0.0, success=False
    q2 = _qrr(query_id="q2", ranked=("x", "y"), grades={})
    # q3: relevant doc present but ranked second -> rr=0.5
    q3 = _qrr(query_id="q3", ranked=("x", "a"), grades={"a": 1})

    agg = aggregate_run([q1, q2, q3])

    # ndcg mean over scored queries only (q1, q3) — q3's ndcg:
    #   DCG@2 = 0/log2(2) + 1/log2(3); IDCG@2 = 1/log2(2) = 1
    q3_ndcg = (0.0 + 1.0 / math.log2(3)) / 1.0
    assert agg.ndcg_at_k_mean == pytest.approx((1.0 + q3_ndcg) / 2)
    assert agg.scored_query_count == 2
    assert agg.total_query_count == 3

    # recall mean over scored queries (q1: 1.0, q3: 1.0) — q2 excluded (no relevant labels)
    assert agg.recall_at_k_mean == pytest.approx(1.0)

    # MRR is a mean over ALL queries, including the unjudged one (rr=0):
    # (1.0 + 0.0 + 0.5) / 3
    assert agg.mrr == pytest.approx((1.0 + 0.0 + 0.5) / 3)

    # success rate over all 3 queries: q1 True, q2 False, q3 True -> 2/3
    assert agg.success_at_k_rate == pytest.approx(2 / 3)

    # judged coverage: judged results / total results across all queries.
    # q1: 1/1 judged; q2: 0/2 judged; q3: 1/2 judged -> (1+0+1)/(1+2+2) = 2/5
    assert agg.judged_coverage == pytest.approx(2 / 5)


def test_aggregate_run_empty_list() -> None:
    agg = aggregate_run([])
    assert agg.ndcg_at_k_mean is None
    assert agg.recall_at_k_mean is None
    assert agg.mrr == 0.0
    assert agg.success_at_k_rate == 0.0
    assert agg.judged_coverage == 0.0
    assert agg.total_query_count == 0
