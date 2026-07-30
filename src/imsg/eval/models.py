"""Domain objects for the eval harness (SPEC §13). Pure dataclasses —
`metrics`/`diff` operate on these without a database; `io`/`runner` are
the modules that build them from Postgres or a retrieval backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Canonical store rows (SPEC §13.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One row of `eval_query`. `targets` names which retrieval targets
    this query is meant to be run against (`local`, `gemini`) — SPEC
    §13.1's `queries.yaml` shape."""

    query_id: str
    query_text: str
    notes: str | None = None
    targets: tuple[str, ...] = ("local",)


@dataclass(frozen=True, slots=True)
class RelevanceLabel:
    """One row of `relevance_label`. **Anchored on a message GUID, not
    a segment id** (SPEC §13.1, judgment call): segment ids churn on
    re-segmentation; the anchor is permanent. `source` records how the
    grade was produced — `mark_relevant` (local MCP tool, not this
    build's scope), `manual` (`imsg eval label`), or `pool_judgment`
    (the §13.2 pooling workflow)."""

    query_id: str
    anchor_guid: str
    grade: int  # 0=not relevant, 1=relevant, 2=highly relevant
    source: str  # 'mark_relevant' | 'manual' | 'pool_judgment'
    added_at: datetime | None = None


# ---------------------------------------------------------------------------
# Run artifacts (SPEC §13.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryRunResult:
    """One query's outcome within one `imsg eval run`."""

    query_id: str
    query_text: str
    ranked_segment_keys: tuple[str, ...]
    """Top-k segment_keys returned by the backend, in rank order."""
    resolved_grades: dict[str, int]
    """segment_key -> grade, for every label that resolved to a
    *currently existing* segment for this query (SPEC §13.1: "an anchor
    resolves to whichever current segment contains that message")."""
    unresolved_label_count: int
    """Labels for this query whose anchor message no longer belongs to
    any segment (deleted, or the message/segment vanished) — reported,
    not silently dropped."""
    ndcg_at_k: float | None
    """None when the query has zero graded labels — there is no ideal
    ranking to compare against, so the query is excluded from the
    nDCG aggregate rather than scored as 0 (SPEC §13.3: "explicit
    unjudged handling")."""
    recall_at_k: float | None
    """Pooled recall (SPEC §13.3, D6): denominator is known-relevant
    items *in the judged pool* for this query, not corpus-wide. None
    when there are zero relevant (grade >= 1) labels for this query."""
    reciprocal_rank: float
    success_at_k: bool
    judged_at_k: int
    """How many of the top-k results carry a label (any grade)."""


@dataclass(frozen=True, slots=True)
class AT4Check:
    """SPEC §12 AT-4's minimums, evaluated against whatever query/label
    set this run's `target` covers — not against the whole corpus, so a
    run over only a handful of debug queries correctly reports FAIL."""

    query_count: int
    pooled_judgment_count: int
    queries_with_a_positive: int
    has_any_positive: bool
    has_any_negative: bool
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """`eval/runs/<ISO-date>-<target>-<config-sha>.json` (SPEC §13.3)."""

    run_id: str
    target: str  # 'local' | 'gemini'
    config_sha256: str
    k: int
    created_at: datetime
    per_query: tuple[QueryRunResult, ...]
    ndcg_at_k_mean: float | None
    recall_at_k_mean: float | None
    mrr: float
    success_at_k_rate: float
    judged_coverage: float
    at4: AT4Check


__all__ = [
    "AT4Check",
    "EvalQuery",
    "EvalRunResult",
    "QueryRunResult",
    "RelevanceLabel",
]
