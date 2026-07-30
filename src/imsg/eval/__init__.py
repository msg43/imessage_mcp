"""The eval harness (SPEC §13): query set + labels, a runner that
targets the local retrieval service (or, later, the Gemini Enterprise
ingestion leg), graded-relevance metrics, a per-run diff table, and the
pooling workflow that grows ground truth through use (§13.2).

D4's segmentation thresholds — and every other tuned retrieval knob —
are frozen until this harness produces a baseline (AT-4, SPEC §12).
Retrieval-quality claims in this codebase route through here; nothing
downstream should hand-roll a metric.
"""

from imsg.eval.diff import RunDiff, diff_runs, format_diff_markdown
from imsg.eval.metrics import (
    AggregateMetrics,
    QueryMetrics,
    aggregate_run,
    compute_query_metrics,
    judged_coverage,
    ndcg_at_k,
    recall_at_k_pooled,
    reciprocal_rank,
    success_at_k,
)
from imsg.eval.models import AT4Check, EvalQuery, EvalRunResult, QueryRunResult, RelevanceLabel

__all__ = [
    "AT4Check",
    "AggregateMetrics",
    "EvalQuery",
    "EvalRunResult",
    "QueryMetrics",
    "QueryRunResult",
    "RelevanceLabel",
    "RunDiff",
    "aggregate_run",
    "compute_query_metrics",
    "diff_runs",
    "format_diff_markdown",
    "judged_coverage",
    "ndcg_at_k",
    "recall_at_k_pooled",
    "reciprocal_rank",
    "success_at_k",
]
