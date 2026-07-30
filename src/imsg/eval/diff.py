"""`imsg eval diff <run-a> <run-b>` (SPEC §13.3): "This is the artifact
every retrieval change must include (repo rule: baseline before
tuning)." Per-query and aggregate deltas between two `EvalRunResult`s,
rendered as a markdown table.

Pure and DB-free: operates entirely on two already-loaded
`EvalRunResult` objects (`imsg.eval.io.load_run_json` reads them off
disk), so this module has no database or config dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from imsg.eval.metrics import aggregate_run
from imsg.eval.models import EvalRunResult


@dataclass(frozen=True, slots=True)
class QueryDelta:
    query_id: str
    query_text: str
    ndcg_a: float | None
    ndcg_b: float | None
    ndcg_delta: float | None
    recall_a: float | None
    recall_b: float | None
    recall_delta: float | None
    rr_a: float
    rr_b: float
    rr_delta: float


@dataclass(frozen=True, slots=True)
class RunDiff:
    run_a_id: str
    run_b_id: str
    per_query: tuple[QueryDelta, ...]
    ndcg_mean_a: float | None
    ndcg_mean_b: float | None
    recall_mean_a: float | None
    recall_mean_b: float | None
    mrr_a: float
    mrr_b: float
    success_rate_a: float
    success_rate_b: float
    judged_coverage_a: float
    judged_coverage_b: float
    """Only queries present in *both* runs are compared — a run over a
    different query subset is not directly comparable and is reported
    separately rather than silently ignored."""
    queries_only_in_a: tuple[str, ...]
    queries_only_in_b: tuple[str, ...]


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def diff_runs(run_a: EvalRunResult, run_b: EvalRunResult) -> RunDiff:
    by_id_a = {q.query_id: q for q in run_a.per_query}
    by_id_b = {q.query_id: q for q in run_b.per_query}
    common = sorted(set(by_id_a) & set(by_id_b))

    per_query = []
    for qid in common:
        qa, qb = by_id_a[qid], by_id_b[qid]
        per_query.append(
            QueryDelta(
                query_id=qid,
                query_text=qa.query_text,
                ndcg_a=qa.ndcg_at_k,
                ndcg_b=qb.ndcg_at_k,
                ndcg_delta=_sub(qa.ndcg_at_k, qb.ndcg_at_k),
                recall_a=qa.recall_at_k,
                recall_b=qb.recall_at_k,
                recall_delta=_sub(qa.recall_at_k, qb.recall_at_k),
                rr_a=qa.reciprocal_rank,
                rr_b=qb.reciprocal_rank,
                rr_delta=qb.reciprocal_rank - qa.reciprocal_rank,
            )
        )

    agg_a = aggregate_run([by_id_a[qid] for qid in common]) if common else aggregate_run([])
    agg_b = aggregate_run([by_id_b[qid] for qid in common]) if common else aggregate_run([])

    return RunDiff(
        run_a_id=run_a.run_id,
        run_b_id=run_b.run_id,
        per_query=tuple(per_query),
        ndcg_mean_a=agg_a.ndcg_at_k_mean,
        ndcg_mean_b=agg_b.ndcg_at_k_mean,
        recall_mean_a=agg_a.recall_at_k_mean,
        recall_mean_b=agg_b.recall_at_k_mean,
        mrr_a=agg_a.mrr,
        mrr_b=agg_b.mrr,
        success_rate_a=agg_a.success_at_k_rate,
        success_rate_b=agg_b.success_at_k_rate,
        judged_coverage_a=agg_a.judged_coverage,
        judged_coverage_b=agg_b.judged_coverage,
        queries_only_in_a=tuple(sorted(set(by_id_a) - set(by_id_b))),
        queries_only_in_b=tuple(sorted(set(by_id_b) - set(by_id_a))),
    )


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def format_diff_markdown(diff: RunDiff) -> str:
    """Markdown table: aggregate summary first, then per-query deltas.
    `run_a` is the baseline, `run_b` the candidate — deltas are `b - a`
    (positive = candidate improved)."""
    lines = [
        f"# Eval diff: `{diff.run_a_id}` -> `{diff.run_b_id}`",
        "",
        "## Aggregate",
        "",
        "| metric | a (baseline) | b (candidate) | delta |",
        "|---|---|---|---|",
        f"| nDCG@k (mean) | {_fmt(diff.ndcg_mean_a)} | {_fmt(diff.ndcg_mean_b)} | "
        f"{_fmt_delta(_sub(diff.ndcg_mean_a, diff.ndcg_mean_b))} |",
        f"| Recall@k (pooled, mean) | {_fmt(diff.recall_mean_a)} | {_fmt(diff.recall_mean_b)} | "
        f"{_fmt_delta(_sub(diff.recall_mean_a, diff.recall_mean_b))} |",
        f"| MRR | {_fmt(diff.mrr_a)} | {_fmt(diff.mrr_b)} | "
        f"{_fmt_delta(diff.mrr_b - diff.mrr_a)} |",
        f"| success@k | {_fmt(diff.success_rate_a)} | {_fmt(diff.success_rate_b)} | "
        f"{_fmt_delta(diff.success_rate_b - diff.success_rate_a)} |",
        f"| judged coverage | {_fmt(diff.judged_coverage_a)} | {_fmt(diff.judged_coverage_b)} | "
        f"{_fmt_delta(diff.judged_coverage_b - diff.judged_coverage_a)} |",
        "",
    ]
    if diff.queries_only_in_a or diff.queries_only_in_b:
        lines.append(
            f"_{len(diff.queries_only_in_a)} quer(y/ies) only in a, "
            f"{len(diff.queries_only_in_b)} only in b — excluded from the "
            f"comparison above (only common queries are diffed)._"
        )
        lines.append("")

    lines += [
        "## Per-query",
        "",
        "| query_id | nDCG a | nDCG b | Δ | Recall a | Recall b | Δ | RR a | RR b | Δ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in diff.per_query:
        lines.append(
            f"| {d.query_id} | {_fmt(d.ndcg_a)} | {_fmt(d.ndcg_b)} | {_fmt_delta(d.ndcg_delta)} | "
            f"{_fmt(d.recall_a)} | {_fmt(d.recall_b)} | {_fmt_delta(d.recall_delta)} | "
            f"{d.rr_a:.3f} | {d.rr_b:.3f} | {_fmt_delta(d.rr_delta)} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["QueryDelta", "RunDiff", "diff_runs", "format_diff_markdown"]
