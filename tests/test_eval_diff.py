"""`imsg.eval.diff` — the `imsg eval diff` artifact (SPEC §13.3: "the
artifact every retrieval change must include")."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from imsg.eval.diff import diff_runs, format_diff_markdown
from imsg.eval.models import AT4Check, EvalRunResult, QueryRunResult

_AT4 = AT4Check(
    query_count=2, pooled_judgment_count=2, queries_with_a_positive=2,
    has_any_positive=True, has_any_negative=False, passed=False, reasons=("x",),
)


def _run(run_id: str, per_query: tuple[QueryRunResult, ...]) -> EvalRunResult:
    return EvalRunResult(
        run_id=run_id,
        target="local",
        config_sha256="a" * 64,
        k=10,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        per_query=per_query,
        ndcg_at_k_mean=None,
        recall_at_k_mean=None,
        mrr=0.0,
        success_at_k_rate=0.0,
        judged_coverage=0.0,
        at4=_AT4,
    )


def _qrr(query_id: str, *, ndcg: float | None, rr: float) -> QueryRunResult:
    return QueryRunResult(
        query_id=query_id,
        query_text=query_id,
        ranked_segment_keys=("seg",),
        resolved_grades={"seg": 1},
        unresolved_label_count=0,
        ndcg_at_k=ndcg,
        recall_at_k=1.0 if ndcg is not None else None,
        reciprocal_rank=rr,
        success_at_k=rr > 0,
        judged_at_k=1,
    )


def test_diff_runs_computes_per_query_deltas() -> None:
    run_a = _run("a", (_qrr("q1", ndcg=0.5, rr=0.5), _qrr("q2", ndcg=0.2, rr=0.0)))
    run_b = _run("b", (_qrr("q1", ndcg=0.9, rr=1.0), _qrr("q2", ndcg=0.2, rr=0.0)))

    diff = diff_runs(run_a, run_b)
    by_id = {d.query_id: d for d in diff.per_query}
    assert by_id["q1"].ndcg_delta == pytest.approx(0.4)
    assert by_id["q1"].rr_delta == pytest.approx(0.5)
    assert by_id["q2"].ndcg_delta == pytest.approx(0.0)
    assert diff.queries_only_in_a == ()
    assert diff.queries_only_in_b == ()


def test_diff_runs_reports_queries_only_on_one_side() -> None:
    run_a = _run("a", (_qrr("q1", ndcg=0.5, rr=0.5),))
    run_b = _run("b", (_qrr("q2", ndcg=0.5, rr=0.5),))
    diff = diff_runs(run_a, run_b)
    assert diff.per_query == ()
    assert diff.queries_only_in_a == ("q1",)
    assert diff.queries_only_in_b == ("q2",)


def test_format_diff_markdown_contains_run_ids_and_table_rows() -> None:
    run_a = _run("run-a-id", (_qrr("q1", ndcg=0.5, rr=0.5),))
    run_b = _run("run-b-id", (_qrr("q1", ndcg=0.9, rr=1.0),))
    diff = diff_runs(run_a, run_b)
    markdown = format_diff_markdown(diff)
    assert "run-a-id" in markdown
    assert "run-b-id" in markdown
    assert "q1" in markdown
    assert "+0.400" in markdown  # ndcg delta for q1


def test_format_diff_markdown_handles_none_metrics_gracefully() -> None:
    run_a = _run("a", (_qrr("q1", ndcg=None, rr=0.0),))
    run_b = _run("b", (_qrr("q1", ndcg=None, rr=0.0),))
    diff = diff_runs(run_a, run_b)
    markdown = format_diff_markdown(diff)
    assert "—" in markdown  # em-dash placeholder for a None value
