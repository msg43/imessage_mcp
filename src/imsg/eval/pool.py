"""§13.2 pooling workflow: "run at least two materially different
retrieval configurations, pool their top-20 unique results per query,
randomize presentation, and judge the pool 0/1/2." This module builds
that pool and an un-graded judgment worksheet; `import_pool_worksheet`
reads it back once the owner has filled in grades, upserting
`relevance_label` rows with `source='pool_judgment'` via
`imsg.eval.io.label_segment_by_key` — the same anchor-on-message-GUID
write path `imsg eval label` and (eventually) the local `mark_relevant`
tool use.

"Ground truth accumulates through use" (§13.2) — this is the mechanism
that makes accumulation more than one grade at a time: comparing two
configs surfaces disagreements, and judging the union is cheaper than
judging either config's list twice.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from imsg.eval.io import label_segment_by_key

if TYPE_CHECKING:
    import psycopg

    from imsg.eval.backend import EvalBackend
    from imsg.eval.models import EvalQuery

DEFAULT_POOL_TOP_N = 20


@dataclass(frozen=True, slots=True)
class PoolEntry:
    query_id: str
    query_text: str
    segment_key: str
    text_preview: str
    source_configs: tuple[str, ...]
    """Which named backend(s) in the `build_pool` call surfaced this
    candidate — visible on the worksheet purely as context for the
    judge/reviewer, never fed back into scoring."""


def _segment_preview(conn: psycopg.Connection, segment_key: str, *, chars: int = 300) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT rendered_text FROM segment WHERE stable_key = %s", (segment_key,))
        row = cur.fetchone()
    if row is None:
        return "(segment no longer exists)"
    text = str(row[0])
    return text if len(text) <= chars else text[:chars] + "…"


def build_pool(
    conn: psycopg.Connection,
    backends: Mapping[str, EvalBackend],
    queries: list[EvalQuery],
    *,
    top_n: int = DEFAULT_POOL_TOP_N,
    seed: int | None = None,
) -> list[PoolEntry]:
    """Union of every backend's top-`top_n` results per query, deduped
    by `segment_key`, in an order randomized independently per query
    (SPEC §13.2: "randomize presentation" — so a judge can't infer
    which config ranked a candidate higher from worksheet position).
    `seed` makes the randomization reproducible for tests; omit it for
    a real judging session.
    """
    rng = random.Random(seed)
    entries: list[PoolEntry] = []
    for q in queries:
        by_key: dict[str, set[str]] = {}
        for config_name, backend in backends.items():
            for segment_key in backend.search(q.query_text, k=top_n):
                by_key.setdefault(segment_key, set()).add(config_name)
        keys = list(by_key)
        rng.shuffle(keys)
        for segment_key in keys:
            entries.append(
                PoolEntry(
                    query_id=q.query_id,
                    query_text=q.query_text,
                    segment_key=segment_key,
                    text_preview=_segment_preview(conn, segment_key),
                    source_configs=tuple(sorted(by_key[segment_key])),
                )
            )
    return entries


def pool_to_worksheet_yaml(entries: list[PoolEntry]) -> str:
    """An un-graded judgment worksheet: every entry carries `grade:
    null` for the owner to fill in with 0/1/2. Round-trips through
    `import_pool_worksheet`."""
    docs = [
        {
            "query_id": e.query_id,
            "query_text": e.query_text,
            "segment_key": e.segment_key,
            "text_preview": e.text_preview,
            "source_configs": list(e.source_configs),
            "grade": None,
        }
        for e in entries
    ]
    return yaml.safe_dump(docs, sort_keys=False, allow_unicode=True)


def import_pool_worksheet(conn: psycopg.Connection, text: str) -> int:
    """Import a (possibly partially) graded worksheet: every entry
    whose `grade` is 0, 1, or 2 becomes a `relevance_label` row
    (`source='pool_judgment'`); entries still `grade: null` are skipped
    rather than erroring, so a judging session can be imported
    incrementally without re-grading everything first. Returns the
    number of labels actually imported.
    """
    data = yaml.safe_load(text) or []
    if not isinstance(data, list):
        raise ValueError("pool worksheet must be a top-level YAML list")
    imported = 0
    for entry in data:
        grade = entry.get("grade")
        if grade is None:
            continue
        label_segment_by_key(
            conn,
            query_id=str(entry["query_id"]),
            segment_key=str(entry["segment_key"]),
            grade=int(grade),
            source="pool_judgment",
        )
        imported += 1
    return imported


__all__ = [
    "DEFAULT_POOL_TOP_N",
    "PoolEntry",
    "build_pool",
    "import_pool_worksheet",
    "pool_to_worksheet_yaml",
]
