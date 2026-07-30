"""Unit tests for Reciprocal Rank Fusion (SPEC §9.4 step 6) — no
database required."""

from __future__ import annotations

from imsg.retrieval.fuse import reciprocal_rank_fusion


def test_single_list_scores_by_rank() -> None:
    fused = reciprocal_rank_fusion({"segment_fts": [10, 20, 30]}, rrf_k=60)
    assert [r.segment_id for r in fused] == [10, 20, 30]
    assert fused[0].score == 1 / 61
    assert fused[1].score == 1 / 62
    assert fused[2].score == 1 / 63


def test_agreement_across_lists_boosts_score() -> None:
    # segment 1 is #1 in both lists; segment 2 is #1 in only one.
    fused = reciprocal_rank_fusion(
        {"segment_fts": [1, 2], "segment_vector": [1, 3]}, rrf_k=60
    )
    by_id = {r.segment_id: r for r in fused}
    assert by_id[1].score == 2 * (1 / 61)
    assert by_id[2].score == 1 / 62
    assert by_id[3].score == 1 / 62
    # Ranked highest first; segment 1 (present in both lists) wins.
    assert fused[0].segment_id == 1


def test_missing_list_contributes_nothing() -> None:
    # SPEC §9.4 step 6: "Missing or disabled lists contribute nothing."
    with_list = reciprocal_rank_fusion({"a": [1]}, rrf_k=60)
    without_list = reciprocal_rank_fusion({"a": [1], "b": []}, rrf_k=60)
    assert with_list[0].score == without_list[0].score


def test_per_list_ranks_are_recorded() -> None:
    fused = reciprocal_rank_fusion({"a": [5, 6], "b": [6]}, rrf_k=60)
    by_id = {r.segment_id: r for r in fused}
    assert by_id[5].per_list_ranks == {"a": 1}
    assert by_id[6].per_list_ranks == {"a": 2, "b": 1}


def test_ties_break_by_segment_id_ascending() -> None:
    fused = reciprocal_rank_fusion({"a": [2], "b": [1]}, rrf_k=60)
    assert fused[0].score == fused[1].score
    assert [r.segment_id for r in fused] == [1, 2]


def test_empty_lists_produce_no_results() -> None:
    assert reciprocal_rank_fusion({}, rrf_k=60) == []
    assert reciprocal_rank_fusion({"a": []}, rrf_k=60) == []


def test_larger_rrf_k_flattens_the_score_gap_between_ranks() -> None:
    small_k = reciprocal_rank_fusion({"a": [1, 2]}, rrf_k=1)
    large_k = reciprocal_rank_fusion({"a": [1, 2]}, rrf_k=1000)
    small_gap = small_k[0].score - small_k[1].score
    large_gap = large_k[0].score - large_k[1].score
    assert small_gap > large_gap
