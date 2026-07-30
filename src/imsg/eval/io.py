"""Canonical store (Postgres `eval_query`/`relevance_label`) <-> YAML
interchange (SPEC §13.1, D6).

"Canonical queries and labels live in Postgres ... the local MCP tool
upserts rows rather than doing concurrent YAML file appends. YAML is
the import/export format for review and versioned private snapshots."
This module is that boundary: `load_queries`/`load_labels` read the
canonical Postgres rows; `queries_to_yaml`/`labels_to_yaml` and their
`*_from_yaml` counterparts convert to/from the exact `queries.yaml`/
`labels.yaml` shapes documented in SPEC §13.1; `upsert_query`/
`upsert_label` are the single write path every producer (the
`mark_relevant` MCP tool — not this build's scope — and the
`imsg eval label`/`import-*` CLI commands, which are) should share.

Also writes/reads the run-JSON artifact
(`eval/runs/<ISO-date>-<target>-<config-sha>.json`, SPEC §13.3).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import yaml

from imsg.eval.models import AT4Check, EvalQuery, EvalRunResult, QueryRunResult, RelevanceLabel
from imsg.hashing import sha256_text

if TYPE_CHECKING:
    import psycopg

# ---------------------------------------------------------------------------
# Postgres <-> in-memory rows
# ---------------------------------------------------------------------------


def load_queries(conn: psycopg.Connection, *, target: str | None = None) -> list[EvalQuery]:
    """All `eval_query` rows, optionally restricted to those whose
    `targets` array contains `target` (SPEC §13.1: `targets: [local,
    gemini]`)."""
    sql = "SELECT query_id, query_text, notes, targets FROM eval_query"
    params: tuple[object, ...] = ()
    if target is not None:
        sql += " WHERE %s = ANY(targets)"
        params = (target,)
    sql += " ORDER BY query_id"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        EvalQuery(query_id=qid, query_text=text, notes=notes, targets=tuple(targets))
        for qid, text, notes, targets in rows
    ]


def upsert_query(conn: psycopg.Connection, query: EvalQuery) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_query (query_id, query_text, notes, targets)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (query_id) DO UPDATE SET
                query_text = EXCLUDED.query_text,
                notes = EXCLUDED.notes,
                targets = EXCLUDED.targets
            """,
            (query.query_id, query.query_text, query.notes, list(query.targets)),
        )


def load_labels(conn: psycopg.Connection, *, query_id: str | None = None) -> list[RelevanceLabel]:
    sql = "SELECT query_id, anchor_guid, grade, source, added_at FROM relevance_label"
    params: tuple[object, ...] = ()
    if query_id is not None:
        sql += " WHERE query_id = %s"
        params = (query_id,)
    sql += " ORDER BY query_id, anchor_guid"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        RelevanceLabel(query_id=qid, anchor_guid=guid, grade=grade, source=source, added_at=added_at)
        for qid, guid, grade, source, added_at in rows
    ]


def upsert_label(conn: psycopg.Connection, label: RelevanceLabel) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO relevance_label (query_id, anchor_guid, grade, source)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (query_id, anchor_guid) DO UPDATE SET
                grade = EXCLUDED.grade,
                source = EXCLUDED.source
            """,
            (label.query_id, label.anchor_guid, label.grade, label.source),
        )


def label_segment_by_key(
    conn: psycopg.Connection,
    *,
    query_id: str,
    segment_key: str,
    grade: int,
    source: str = "manual",
    query_text_if_new: str | None = None,
) -> RelevanceLabel:
    """The CLI/local-tool write path (SPEC §10.2 `mark_relevant`,
    §13.2 "the local `mark_relevant` tool ... and `imsg eval label` CLI
    upsert database judgments"): resolve `segment_key` to its anchor —
    the segment's *first* message GUID, by `sent_at` — and upsert the
    label anchored there, not on the segment id (SPEC §13.1: segment
    ids churn on re-segmentation, the anchor does not).

    If `query_id` doesn't exist yet and `query_text_if_new` is given,
    creates it first (mirrors `mark_relevant`'s `'adhoc:<text>'`
    query-id convenience, SPEC §10.2).
    """
    if not (0 <= grade <= 2):
        raise ValueError(f"grade must be 0-2, got {grade}")

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM eval_query WHERE query_id = %s", (query_id,))
        exists = cur.fetchone() is not None
    if not exists:
        if query_text_if_new is None:
            raise ValueError(
                f"eval_query '{query_id}' does not exist and no query_text_if_new was "
                f"given to create it"
            )
        upsert_query(conn, EvalQuery(query_id=query_id, query_text=query_text_if_new))

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.source_guid
            FROM segment s
            JOIN segment_message sm ON sm.segment_id = s.segment_id
            JOIN message m ON m.message_id = sm.message_id
            WHERE s.stable_key = %s
            ORDER BY m.sent_at, m.message_id
            LIMIT 1
            """,
            (segment_key,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"segment_key '{segment_key}' does not resolve to any segment with at "
            f"least one message — cannot derive an anchor GUID"
        )
    anchor_guid = str(row[0])
    label = RelevanceLabel(query_id=query_id, anchor_guid=anchor_guid, grade=grade, source=source)
    upsert_label(conn, label)
    return label


# ---------------------------------------------------------------------------
# YAML interchange (SPEC §13.1)
# ---------------------------------------------------------------------------


def queries_to_yaml(queries: list[EvalQuery]) -> str:
    docs = [
        {
            "id": q.query_id,
            "query": q.query_text,
            **({"notes": q.notes} if q.notes else {}),
            "targets": list(q.targets),
        }
        for q in queries
    ]
    return yaml.safe_dump(docs, sort_keys=False, allow_unicode=True)


def queries_from_yaml(text: str) -> list[EvalQuery]:
    data = yaml.safe_load(text) or []
    if not isinstance(data, list):
        raise ValueError("queries.yaml must be a top-level YAML list")
    out = []
    for entry in data:
        out.append(
            EvalQuery(
                query_id=str(entry["id"]),
                query_text=str(entry["query"]),
                notes=entry.get("notes"),
                targets=tuple(entry.get("targets", ["local"])),
            )
        )
    return out


def labels_to_yaml(labels: list[RelevanceLabel]) -> str:
    docs = [
        {
            "query_id": lbl.query_id,
            "anchor_guid": lbl.anchor_guid,
            "grade": lbl.grade,
            "source": lbl.source,
            **({"added_at": lbl.added_at.isoformat()} if lbl.added_at else {}),
        }
        for lbl in labels
    ]
    return yaml.safe_dump(docs, sort_keys=False, allow_unicode=True)


def labels_from_yaml(text: str) -> list[RelevanceLabel]:
    data = yaml.safe_load(text) or []
    if not isinstance(data, list):
        raise ValueError("labels.yaml must be a top-level YAML list")
    out = []
    for entry in data:
        added_at = entry.get("added_at")
        out.append(
            RelevanceLabel(
                query_id=str(entry["query_id"]),
                anchor_guid=str(entry["anchor_guid"]),
                grade=int(entry["grade"]),
                source=str(entry.get("source", "manual")),
                added_at=datetime.fromisoformat(added_at) if added_at else None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Run-JSON artifact (SPEC §13.3)
# ---------------------------------------------------------------------------


def run_filename(*, run_id: str, target: str, config_sha256: str) -> str:
    return f"{run_id}-{target}-{config_sha256[:12]}.json"


def make_run_id(created_at: datetime | None = None) -> str:
    return (created_at or datetime.now(UTC)).strftime("%Y-%m-%d")


def _query_result_to_dict(q: QueryRunResult) -> dict[str, Any]:
    d = asdict(q)
    d["ranked_segment_keys"] = list(q.ranked_segment_keys)
    return d


def _query_result_from_dict(d: dict[str, Any]) -> QueryRunResult:
    return QueryRunResult(
        query_id=d["query_id"],
        query_text=d["query_text"],
        ranked_segment_keys=tuple(d["ranked_segment_keys"]),
        resolved_grades={str(k): int(v) for k, v in d["resolved_grades"].items()},
        unresolved_label_count=d["unresolved_label_count"],
        ndcg_at_k=d["ndcg_at_k"],
        recall_at_k=d["recall_at_k"],
        reciprocal_rank=d["reciprocal_rank"],
        success_at_k=d["success_at_k"],
        judged_at_k=d["judged_at_k"],
    )


def run_result_to_json(result: EvalRunResult) -> str:
    payload = {
        "run_id": result.run_id,
        "target": result.target,
        "config_sha256": result.config_sha256,
        "k": result.k,
        "created_at": result.created_at.isoformat(),
        "per_query": [_query_result_to_dict(q) for q in result.per_query],
        "aggregate": {
            "ndcg_at_k_mean": result.ndcg_at_k_mean,
            "recall_at_k_mean": result.recall_at_k_mean,
            "mrr": result.mrr,
            "success_at_k_rate": result.success_at_k_rate,
            "judged_coverage": result.judged_coverage,
        },
        "at4": asdict(result.at4),
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def run_result_from_json(text: str) -> EvalRunResult:
    payload = json.loads(text)
    at4_d = payload["at4"]
    return EvalRunResult(
        run_id=payload["run_id"],
        target=payload["target"],
        config_sha256=payload["config_sha256"],
        k=payload["k"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        per_query=tuple(_query_result_from_dict(q) for q in payload["per_query"]),
        ndcg_at_k_mean=payload["aggregate"]["ndcg_at_k_mean"],
        recall_at_k_mean=payload["aggregate"]["recall_at_k_mean"],
        mrr=payload["aggregate"]["mrr"],
        success_at_k_rate=payload["aggregate"]["success_at_k_rate"],
        judged_coverage=payload["aggregate"]["judged_coverage"],
        at4=AT4Check(
            query_count=at4_d["query_count"],
            pooled_judgment_count=at4_d["pooled_judgment_count"],
            queries_with_a_positive=at4_d["queries_with_a_positive"],
            has_any_positive=at4_d["has_any_positive"],
            has_any_negative=at4_d["has_any_negative"],
            passed=at4_d["passed"],
            reasons=tuple(at4_d["reasons"]),
        ),
    )


def config_projection_sha256(*, target: str, k: int, extra: dict[str, Any] | None = None) -> str:
    """Hash of whatever distinguishes one eval *configuration* from
    another (SPEC §13.3's `<config-sha>` filename component) — callers
    pass in whatever varies for the comparison at hand (segmentation
    thresholds live in the loaded `Config`'s own values; reranker
    on/off and dual-vector on/off are runner-level toggles not present
    in `config.yaml` — see `imsg.eval.runner`), so this stays a plain
    dict-hash rather than coupling to `Config`'s shape.
    """
    payload = {"target": target, "k": k, **(extra or {})}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


__all__ = [
    "config_projection_sha256",
    "label_segment_by_key",
    "labels_from_yaml",
    "labels_to_yaml",
    "load_labels",
    "load_queries",
    "make_run_id",
    "queries_from_yaml",
    "queries_to_yaml",
    "run_filename",
    "run_result_from_json",
    "run_result_to_json",
    "upsert_label",
    "upsert_query",
]
