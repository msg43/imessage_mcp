"""Reciprocal Rank Fusion (SPEC §9.4 step 6):

    score = sum(1 / (rrf_k + rank_list(segment))) over every list the
    segment appears in

"Missing or disabled lists contribute nothing; raw BM25/cosine scores
are never compared across corpora." `rank_list` is 1-indexed rank
*within that list* (best = 1); a segment absent from a list contributes
0 for that list, not a penalty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusedResult:
    segment_id: int
    score: float
    per_list_ranks: Mapping[str, int]
    """1-indexed rank in each list the segment appeared in — carried
    through for `candidate_lists` diagnostics / debugging, not used by
    the fusion math itself beyond what already went into `score`."""


def reciprocal_rank_fusion(
    lists: Mapping[str, Sequence[int]], *, rrf_k: int
) -> list[FusedResult]:
    """`lists` maps a channel name (e.g. `'segment_fts'`) to its
    ranked `segment_id`s, best first. A channel that produced no
    results (disabled, or genuinely empty) should simply be omitted or
    passed as an empty sequence — both are equivalent here.

    Returns every segment that appeared in at least one list, sorted
    by fused score descending (ties broken by `segment_id` ascending,
    for deterministic output — RRF scores collide often on short
    lists)."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for list_name, segment_ids in lists.items():
        for position, segment_id in enumerate(segment_ids, start=1):
            scores[segment_id] = scores.get(segment_id, 0.0) + 1.0 / (rrf_k + position)
            ranks.setdefault(segment_id, {})[list_name] = position

    fused = [
        FusedResult(segment_id=sid, score=score, per_list_ranks=ranks.get(sid, {}))
        for sid, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r.score, r.segment_id))
    return fused


__all__ = ["FusedResult", "reciprocal_rank_fusion"]
