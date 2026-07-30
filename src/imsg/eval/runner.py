"""`imsg eval run --target local|gemini [--k 10]` (SPEC §13.3).

Backend-agnostic: `run_eval` takes any `imsg.eval.backend.EvalBackend`
(local retrieval service or Gemini search client) and produces one
`EvalRunResult` — per-query ranked `segment_key`s, resolved grades,
judged status, and metrics, plus the run's `AT4Check` against the
minimums in SPEC §12 AT-4. Writing the JSON artifact to
`eval/runs/<...>.json` is the CLI's job (`imsg.cli`), via
`imsg.eval.io.run_result_to_json`, so this module stays DB-and-backend
facing only and easy to call from a test without touching the
filesystem.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from imsg.eval import io
from imsg.eval.metrics import aggregate_run, compute_query_metrics
from imsg.eval.models import AT4Check, EvalRunResult, QueryRunResult

if TYPE_CHECKING:
    import psycopg

    from imsg.eval.backend import EvalBackend

AT4_MIN_QUERIES = 30
AT4_MIN_POOLED_JUDGMENTS = 100
AT4_MIN_QUERIES_WITH_A_POSITIVE = 25


def _resolve_labels_for_query(
    conn: psycopg.Connection, query_id: str
) -> tuple[dict[str, int], int]:
    """`{segment_key: grade}` for every label of `query_id` whose
    anchor message currently belongs to a segment, plus the count that
    don't (SPEC §13.1: "an anchor resolves to whichever current
    segment contains that message" — a message that has since been
    deleted, or dropped from any segment, cannot resolve). If two
    anchors happen to resolve to the same segment (a merge, or two
    pooled messages inside one segment), the higher grade wins —
    conservative in the direction of "don't under-credit a segment a
    judge did mark relevant."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rl.grade, s.stable_key
            FROM relevance_label rl
            LEFT JOIN message m ON m.source_guid = rl.anchor_guid
            LEFT JOIN segment_message sm ON sm.message_id = m.message_id
            LEFT JOIN segment s ON s.segment_id = sm.segment_id
            WHERE rl.query_id = %s
            """,
            (query_id,),
        )
        rows = cur.fetchall()

    grades: dict[str, int] = {}
    unresolved = 0
    for grade, stable_key in rows:
        if stable_key is None:
            unresolved += 1
            continue
        key = str(stable_key)
        if key not in grades or int(grade) > grades[key]:
            grades[key] = int(grade)
    return grades, unresolved


def compute_at4_check(conn: psycopg.Connection, *, target: str) -> AT4Check:
    """SPEC §12 AT-4's minimums, evaluated over whichever queries carry
    `target` in `eval_query.targets` and every label recorded against
    them — regardless of whether those labels currently resolve to a
    live segment (an AT-4 judgment doesn't stop counting just because
    the corpus re-segmented since it was made)."""
    queries = io.load_queries(conn, target=target)
    query_ids = [q.query_id for q in queries]

    labels: list[tuple[str, int]] = []
    if query_ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT query_id, grade FROM relevance_label WHERE query_id = ANY(%s)",
                (query_ids,),
            )
            labels = [(str(qid), int(grade)) for qid, grade in cur.fetchall()]

    positives_by_query: dict[str, int] = {}
    has_positive = False
    has_negative = False
    for qid, grade in labels:
        if grade >= 1:
            positives_by_query[qid] = positives_by_query.get(qid, 0) + 1
            has_positive = True
        else:
            has_negative = True

    query_count = len(query_ids)
    pooled_judgment_count = len(labels)
    queries_with_positive = len(positives_by_query)

    reasons: list[str] = []
    if query_count < AT4_MIN_QUERIES:
        reasons.append(f"{query_count} queries recorded (need >= {AT4_MIN_QUERIES})")
    if pooled_judgment_count < AT4_MIN_POOLED_JUDGMENTS:
        reasons.append(
            f"{pooled_judgment_count} pooled graded judgments (need >= "
            f"{AT4_MIN_POOLED_JUDGMENTS})"
        )
    if queries_with_positive < AT4_MIN_QUERIES_WITH_A_POSITIVE:
        reasons.append(
            f"{queries_with_positive} queries have >= 1 positive label (need >= "
            f"{AT4_MIN_QUERIES_WITH_A_POSITIVE})"
        )
    if not has_positive:
        reasons.append("no positive (grade >= 1) judgments recorded at all")
    if not has_negative:
        reasons.append("no negative (grade = 0) judgments recorded at all")

    return AT4Check(
        query_count=query_count,
        pooled_judgment_count=pooled_judgment_count,
        queries_with_a_positive=queries_with_positive,
        has_any_positive=has_positive,
        has_any_negative=has_negative,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def run_eval(
    conn: psycopg.Connection,
    backend: EvalBackend,
    *,
    target: str,
    config_sha256: str,
    k: int = 10,
    created_at: datetime | None = None,
) -> EvalRunResult:
    """Run every `eval_query` whose `targets` include `target` through
    `backend`, score each against its resolved labels, and roll up the
    run (SPEC §13.3). `config_sha256` identifies *this* run's
    configuration for the output filename and for `eval diff` — pass
    `imsg.eval.io.config_projection_sha256(...)` or a `Config`-derived
    hash from the caller; this function doesn't compute it itself
    because what varies between config variants (segmentation
    thresholds vs. reranker on/off vs. dual-vector on/off) is a CLI/
    caller concern, not a runner concern.
    """
    created = created_at or datetime.now(UTC)
    queries = io.load_queries(conn, target=target)

    per_query: list[QueryRunResult] = []
    for q in queries:
        ranked = tuple(backend.search(q.query_text, k=k))
        grades, unresolved = _resolve_labels_for_query(conn, q.query_id)
        m = compute_query_metrics(list(ranked), grades, k)
        per_query.append(
            QueryRunResult(
                query_id=q.query_id,
                query_text=q.query_text,
                ranked_segment_keys=ranked,
                resolved_grades=grades,
                unresolved_label_count=unresolved,
                ndcg_at_k=m.ndcg_at_k,
                recall_at_k=m.recall_at_k,
                reciprocal_rank=m.reciprocal_rank,
                success_at_k=m.success_at_k,
                judged_at_k=m.judged,
            )
        )

    agg = aggregate_run(per_query)
    at4 = compute_at4_check(conn, target=target)

    return EvalRunResult(
        run_id=io.make_run_id(created),
        target=target,
        config_sha256=config_sha256,
        k=k,
        created_at=created,
        per_query=tuple(per_query),
        ndcg_at_k_mean=agg.ndcg_at_k_mean,
        recall_at_k_mean=agg.recall_at_k_mean,
        mrr=agg.mrr,
        success_at_k_rate=agg.success_at_k_rate,
        judged_coverage=agg.judged_coverage,
        at4=at4,
    )


__all__ = [
    "AT4_MIN_POOLED_JUDGMENTS",
    "AT4_MIN_QUERIES",
    "AT4_MIN_QUERIES_WITH_A_POSITIVE",
    "compute_at4_check",
    "run_eval",
]
