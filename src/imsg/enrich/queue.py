"""Enrichment queue lease/backoff (SPEC §8 S5b, D6 additions): workers
claim with `SELECT ... FOR UPDATE SKIP LOCKED`, hold a lease
(`locked_at`/`locked_by`), and an expired lease returns to the
claimable pool automatically — the claim query itself re-checks lease
expiry, so no separate reaper process is needed.

Claim order: kinds are claimed in a priority order —
`imsg.enrich.router.DEFAULT_CLAIM_ORDER` (cheap kinds first, captions
last) unless the caller names its own — and within one kind the newest
attachment (highest `attachment_id`) goes first, so a months-long
caption backlog captions recent photos before old ones and a new
attachment never waits behind it. `next_attempt_at` is a gate, not an
order: a backed-off task is simply not claimable until its time comes.
Migration 0009's `enrichment_claim_idx` serves the one-kind-at-a-time
query without a sort.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from imsg.enrich.router import DEFAULT_CLAIM_ORDER

if TYPE_CHECKING:
    import psycopg

DEFAULT_LEASE_SECONDS = 1800  # matches enrichment.limits.task_timeout_seconds's default
_BACKOFF_BASE_SECONDS = 60
_ENQUEUE_BATCH = 5000


@dataclass(frozen=True, slots=True)
class EnrichmentTask:
    attachment_id: int
    kind: str
    attempts: int


@dataclass(frozen=True, slots=True)
class EnrichPreviewReport:
    """Read-only preview of what `claim_tasks` would claim right now
    (SPEC §8: dry-run for S5b), broken down by `kind`."""

    total: int
    by_kind: dict[str, int]


def enqueue(conn: psycopg.Connection, attachment_id: int, kinds: tuple[str, ...]) -> None:
    """Idempotent enqueue: a `(attachment_id, kind)` pair already
    present (in any state) is left untouched — routing an
    already-processed attachment again must never silently reset a
    `done` row (SPEC §8 S5b: reprocessing only via explicit
    `imsg enrich --rerun`)."""
    if not kinds:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO enrichment (attachment_id, kind) VALUES (%s, %s) "
            "ON CONFLICT (attachment_id, kind) DO NOTHING",
            [(attachment_id, kind) for kind in kinds],
        )


def enqueue_pairs(conn: psycopg.Connection, pairs: Iterable[tuple[int, str]]) -> int:
    """`enqueue` for many attachments at once, returning how many rows it
    actually inserted — an `(attachment_id, kind)` already queued, in any
    state, is left untouched and not counted."""
    return sum(enqueue_pairs_by_kind(conn, pairs).values())


def enqueue_pairs_by_kind(
    conn: psycopg.Connection, pairs: Iterable[tuple[int, str]]
) -> dict[str, int]:
    """`enqueue_pairs`, counting the inserted rows per kind."""
    rows = list(pairs)
    inserted: dict[str, int] = {}
    with conn.cursor() as cur:
        for start in range(0, len(rows), _ENQUEUE_BATCH):
            batch = rows[start : start + _ENQUEUE_BATCH]
            cur.execute(
                """
                INSERT INTO enrichment (attachment_id, kind)
                SELECT a, k::enrichment_kind FROM unnest(%s::bigint[], %s::text[]) AS t(a, k)
                ON CONFLICT (attachment_id, kind) DO NOTHING
                RETURNING kind::text
                """,
                ([a for a, _ in batch], [k for _, k in batch]),
            )
            for (kind,) in cur.fetchall():
                inserted[str(kind)] = inserted.get(str(kind), 0) + 1
    return inserted


def _claim_one_kind(
    conn: psycopg.Connection, kind: str, *, worker_id: str, limit: int, lease_seconds: int
) -> list[EnrichmentTask]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH candidates AS (
                SELECT attachment_id, kind
                FROM enrichment
                WHERE kind = %(kind)s::enrichment_kind
                  AND state IN ('pending', 'running')
                  AND next_attempt_at <= now()
                  AND (
                    state = 'pending'
                    OR locked_at < now() - (%(lease_seconds)s || ' seconds')::interval
                  )
                ORDER BY attachment_id DESC
                LIMIT %(limit)s
                FOR UPDATE SKIP LOCKED
            )
            UPDATE enrichment e
            SET state = 'running', locked_at = now(), locked_by = %(worker_id)s
            FROM candidates c
            WHERE e.attachment_id = c.attachment_id AND e.kind = c.kind
            RETURNING e.attachment_id, e.kind, e.attempts
            """,
            {"kind": kind, "lease_seconds": lease_seconds, "limit": limit, "worker_id": worker_id},
        )
        rows = cur.fetchall()
    tasks = [EnrichmentTask(attachment_id=a, kind=k, attempts=att) for a, k, att in rows]
    return sorted(tasks, key=lambda t: -t.attachment_id)


def claim_tasks(
    conn: psycopg.Connection,
    *,
    worker_id: str,
    limit: int,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    kinds: Sequence[str] | None = None,
) -> list[EnrichmentTask]:
    """Claim up to `limit` runnable tasks: pending tasks whose backoff
    has elapsed, plus `running` tasks whose lease has expired (a worker
    that died mid-task). `FOR UPDATE SKIP LOCKED` lets multiple workers
    claim concurrently without blocking on each other.

    `kinds` is both a filter and the claim order: every claimable task of
    the first kind before any of the second, and so on. `None` means all
    kinds in `DEFAULT_CLAIM_ORDER`. Within a kind, newest attachment
    first."""
    order = tuple(kinds) if kinds is not None else DEFAULT_CLAIM_ORDER
    claimed: list[EnrichmentTask] = []
    for kind in order:
        remaining = limit - len(claimed)
        if remaining <= 0:
            break
        claimed.extend(
            _claim_one_kind(
                conn, kind, worker_id=worker_id, limit=remaining, lease_seconds=lease_seconds
            )
        )
    return claimed


def preview_claimable_tasks(
    conn: psycopg.Connection,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    kinds: Sequence[str] | None = None,
) -> EnrichPreviewReport:
    """Read-only preview of what `claim_tasks` would claim right now,
    broken down by `kind` (SPEC §8: "takes --dry-run where writes
    leave the machine", applied to S5b).

    Runs the exact same selection WHERE clause `claim_tasks` uses —
    pending tasks whose backoff has elapsed, plus `running` tasks
    whose lease has expired — but with no `FOR UPDATE SKIP LOCKED` and
    no state mutation, so it never claims a lease (claiming is itself
    a side effect other workers would observe, and a dry run must not
    cause one).

    This is the honest limit of "dry-run" for S5b: a full per-task
    preview isn't meaningful, because a task's actual output (OCR
    text, a caption, a transcript) can only be known by running the
    real model/subprocess against it — ffmpeg/pdftotext/Vision/whisper
    have no side-effect-free preview mode. So dry-run for this stage
    reports only "N tasks are claimable, broken down by kind," and
    touches nothing else.
    """
    order = tuple(kinds) if kinds is not None else DEFAULT_CLAIM_ORDER
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kind::text, count(*)
            FROM enrichment
            WHERE kind::text = ANY(%(kinds)s)
              AND next_attempt_at <= now()
              AND (
                state = 'pending'
                OR (state = 'running'
                    AND locked_at < now() - (%(lease_seconds)s || ' seconds')::interval)
              )
            GROUP BY kind
            """,
            {"lease_seconds": lease_seconds, "kinds": list(order)},
        )
        rows = cur.fetchall()
    counts = {str(kind): int(count) for kind, count in rows}
    # In claim order, so the preview reads the way the run will go.
    by_kind = {kind: counts[kind] for kind in order if kind in counts}
    return EnrichPreviewReport(total=sum(by_kind.values()), by_kind=by_kind)


def reset_failed_tasks(conn: psycopg.Connection, *, kinds: Sequence[str] | None = None) -> int:
    """`imsg enrich --retry-failed`: every `failed` task of the given kinds
    (all kinds when `None`) back to `pending`, attempts 0, eligible now.
    Returns how many rows it reset."""
    order = tuple(kinds) if kinds is not None else DEFAULT_CLAIM_ORDER
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE enrichment SET state = 'pending', attempts = 0, "
            "next_attempt_at = now(), last_error = NULL "
            "WHERE state = 'failed' AND kind::text = ANY(%s)",
            (list(order),),
        )
        return max(cur.rowcount, 0)


def missing_enrichment_kinds(conn: psycopg.Connection, kinds: Sequence[str]) -> list[str]:
    """The kinds in `kinds` that this database's `enrichment_kind` type
    does not have. Non-empty means a migration has not been applied —
    `doc_text` arrives in 0009 — and a claim for that kind would fail on
    the enum cast, so a caller checks this first and says so plainly."""
    with conn.cursor() as cur:
        cur.execute("SELECT unnest(enum_range(NULL::enrichment_kind))::text")
        present = {str(row[0]) for row in cur.fetchall()}
    return [kind for kind in kinds if kind not in present]


def complete_task(
    conn: psycopg.Connection,
    attachment_id: int,
    kind: str,
    *,
    model: str,
    model_version: str | None,
    text: str | None,
    detail: dict[str, object] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE enrichment
            SET state = 'done', model = %s, model_version = %s, text = %s,
                detail = %s, last_error = NULL, locked_at = NULL, locked_by = NULL,
                updated_at = now()
            WHERE attachment_id = %s AND kind = %s
            """,
            (
                model,
                model_version,
                text,
                json.dumps(detail) if detail is not None else None,
                attachment_id,
                kind,
            ),
        )


def fail_task(
    conn: psycopg.Connection, attachment_id: int, kind: str, *, error: str, max_attempts: int
) -> bool:
    """A transient failure (subprocess crash, model OOM, etc.) —
    backs off and retries until `max_attempts` is reached. Returns
    True if this failure was the one that exhausted the budget
    (state -> `failed`, permanent until `--retry-failed`)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT attempts FROM enrichment WHERE attachment_id = %s AND kind = %s",
            (attachment_id, kind),
        )
        row = cur.fetchone()
        attempts = (row[0] if row else 0) + 1
        permanent = attempts >= max_attempts
        if permanent:
            cur.execute(
                """
                UPDATE enrichment
                SET state = 'failed', attempts = %s, last_error = %s,
                    locked_at = NULL, locked_by = NULL, updated_at = now()
                WHERE attachment_id = %s AND kind = %s
                """,
                (attempts, error[:2000], attachment_id, kind),
            )
        else:
            backoff = timedelta(seconds=_BACKOFF_BASE_SECONDS * (2 ** (attempts - 1)))
            cur.execute(
                """
                UPDATE enrichment
                SET state = 'pending', attempts = %s, next_attempt_at = %s,
                    last_error = %s, locked_at = NULL, locked_by = NULL, updated_at = now()
                WHERE attachment_id = %s AND kind = %s
                """,
                (attempts, datetime.now(UTC) + backoff, error[:2000], attachment_id, kind),
            )
    return permanent


def fail_task_permanently(
    conn: psycopg.Connection, attachment_id: int, kind: str, *, error: str
) -> None:
    """Untrusted-attachment-boundary violations (SPEC §8 S5b, D6) are
    permanent by nature — a file that's too big will always be too big
    — so skip the retry/backoff dance entirely and go straight to
    `failed`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT attempts FROM enrichment WHERE attachment_id = %s AND kind = %s",
            (attachment_id, kind),
        )
        row = cur.fetchone()
        attempts = (row[0] if row else 0) + 1
        cur.execute(
            """
            UPDATE enrichment
            SET state = 'failed', attempts = %s, last_error = %s,
                locked_at = NULL, locked_by = NULL, updated_at = now()
            WHERE attachment_id = %s AND kind = %s
            """,
            (attempts, error[:2000], attachment_id, kind),
        )


def skip_task(conn: psycopg.Connection, attachment_id: int, kind: str, *, reason: str) -> None:
    """Unsupported type (SPEC §8 S5b failure modes) — not a failure,
    just nothing this pipeline knows how to do with it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE enrichment
            SET state = 'skipped', last_error = %s, locked_at = NULL, locked_by = NULL,
                updated_at = now()
            WHERE attachment_id = %s AND kind = %s
            """,
            (reason[:2000], attachment_id, kind),
        )


def release_task(
    conn: psycopg.Connection, attachment_id: int, kind: str, *, worker_id: str
) -> bool:
    """Hand a claimed task back unfinished: `pending` again, lease cleared,
    its attempt not counted — for a worker interrupted while processing
    it (`imsg.enrich.worker`). Only a task still `running` under this
    worker's lease is touched, so a lease that expired and was claimed by
    someone else is left alone. Returns whether a row was released."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE enrichment
            SET state = 'pending', locked_at = NULL, locked_by = NULL, updated_at = now()
            WHERE attachment_id = %s AND kind = %s AND state = 'running' AND locked_by = %s
            """,
            (attachment_id, kind, worker_id),
        )
        return cur.rowcount == 1


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "EnrichPreviewReport",
    "EnrichmentTask",
    "claim_tasks",
    "complete_task",
    "enqueue",
    "enqueue_pairs",
    "enqueue_pairs_by_kind",
    "fail_task",
    "fail_task_permanently",
    "missing_enrichment_kinds",
    "preview_claimable_tasks",
    "release_task",
    "reset_failed_tasks",
    "skip_task",
]
