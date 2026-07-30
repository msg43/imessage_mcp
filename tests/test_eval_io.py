"""`imsg.eval.io` — YAML interchange and the run-JSON artifact (SPEC
§13.1/§13.3). No database needed for these; `label_segment_by_key` and
the Postgres load/upsert helpers are covered by
`tests/test_eval_runner_integration.py` (real Postgres, same skip
pattern as the rest of the DB-backed suite).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from imsg.eval.io import (
    config_projection_sha256,
    labels_from_yaml,
    labels_to_yaml,
    make_run_id,
    queries_from_yaml,
    queries_to_yaml,
    run_filename,
    run_result_from_json,
    run_result_to_json,
)
from imsg.eval.models import AT4Check, EvalQuery, EvalRunResult, QueryRunResult, RelevanceLabel


def test_queries_yaml_roundtrip() -> None:
    queries = [
        EvalQuery(query_id="q001", query_text="deck bid", notes="from Alice", targets=("local", "gemini")),
        EvalQuery(query_id="q002", query_text="lunch plans"),
    ]
    text = queries_to_yaml(queries)
    back = queries_from_yaml(text)
    assert back == queries


def test_queries_from_yaml_defaults_targets_to_local() -> None:
    text = "- id: q001\n  query: hello\n"
    back = queries_from_yaml(text)
    assert back == [EvalQuery(query_id="q001", query_text="hello", targets=("local",))]


def test_labels_yaml_roundtrip() -> None:
    labels = [
        RelevanceLabel(query_id="q001", anchor_guid="ABC-123", grade=2, source="pool_judgment",
                        added_at=datetime(2026, 7, 30, 9, 0, tzinfo=UTC)),
        RelevanceLabel(query_id="q001", anchor_guid="DEF-456", grade=0, source="manual"),
    ]
    text = labels_to_yaml(labels)
    back = labels_from_yaml(text)
    assert back == labels


def test_run_filename_shape() -> None:
    name = run_filename(run_id="2026-07-30", target="local", config_sha256="a" * 64)
    assert name == f"2026-07-30-local-{'a' * 12}.json"


def test_make_run_id_is_iso_date() -> None:
    run_id = make_run_id(datetime(2026, 7, 30, 12, 0, tzinfo=UTC))
    assert run_id == "2026-07-30"


def test_config_projection_sha256_is_stable_and_sensitive_to_input() -> None:
    a = config_projection_sha256(target="local", k=10, extra={"variant": "default"})
    b = config_projection_sha256(target="local", k=10, extra={"variant": "default"})
    c = config_projection_sha256(target="local", k=10, extra={"variant": "no-rerank"})
    assert a == b
    assert a != c


def test_run_result_json_roundtrip() -> None:
    per_query = (
        QueryRunResult(
            query_id="q001",
            query_text="deck bid",
            ranked_segment_keys=("seg-a", "seg-b"),
            resolved_grades={"seg-a": 2},
            unresolved_label_count=1,
            ndcg_at_k=0.9,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            success_at_k=True,
            judged_at_k=1,
        ),
    )
    at4 = AT4Check(
        query_count=1,
        pooled_judgment_count=1,
        queries_with_a_positive=1,
        has_any_positive=True,
        has_any_negative=False,
        passed=False,
        reasons=("only 1 queries recorded (need >= 30)",),
    )
    result = EvalRunResult(
        run_id="2026-07-30",
        target="local",
        config_sha256="a" * 64,
        k=10,
        created_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
        per_query=per_query,
        ndcg_at_k_mean=0.9,
        recall_at_k_mean=1.0,
        mrr=1.0,
        success_at_k_rate=1.0,
        judged_coverage=0.5,
        at4=at4,
    )
    text = run_result_to_json(result)
    back = run_result_from_json(text)
    assert back == result
    assert not math.isnan(back.mrr)
