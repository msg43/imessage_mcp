"""Enrichment queue lease/backoff (SPEC §8 S5b, D6 additions): workers
claim with `SELECT ... FOR UPDATE SKIP LOCKED`, hold a lease
(`locked_at`/`locked_by`), and an expired lease returns to the
claimable pool automatically — the claim query itself re-checks lease
expiry, so no separate reaper process is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg

DEFAULT_LEASE_SECONDS = 1800  # matches enrichment.limits.task_timeout_seconds's default
_BACKOFF_BASE_SECONDS = 60


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


def claim_tasks(
    conn: psycopg.Connection,
    *,
    worker_id: str,
    limit: int,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> list[EnrichmentTask]:
    """Claim up to `limit` runnable tasks: pending tasks whose backoff
    has elapsed, plus `running` tasks whose lease has expired (a worker
    that died mid-task). `FOR UPDATE SKIP LOCKED` lets multiple workers
    claim concurrently without blocking on each other."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH candidates AS (
                SELECT attachment_id, kind
                FROM enrichment
                WHERE next_attempt_at <= now()
                  AND (
                    state = 'pending'
                    OR (state = 'running'
                        AND locked_at < now() - (%(lease_seconds)s || ' seconds')::interval)
                  )
                ORDER BY next_attempt_at
                LIMIT %(limit)s
                FOR UPDATE SKIP LOCKED
            )
            UPDATE enrichment e
            SET state = 'running', locked_at = now(), locked_by = %(worker_id)s
            FROM candidates c
            WHERE e.attachment_id = c.attachment_id AND e.kind = c.kind
            RETURNING e.attachment_id, e.kind, e.attempts
            """,
            {"lease_seconds": lease_seconds, "limit": limit, "worker_id": worker_id},
        )
        rows = cur.fetchall()
    return [EnrichmentTask(attachment_id=a, kind=k, attempts=att) for a, k, att in rows]


def preview_claimable_tasks(
    conn: psycopg.Connection,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
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
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kind, count(*)
            FROM enrichment
            WHERE next_attempt_at <= now()
              AND (
                state = 'pending'
                OR (state = 'running'
                    AND locked_at < now() - (%(lease_seconds)s || ' seconds')::interval)
              )
            GROUP BY kind
            """,
            {"lease_seconds": lease_seconds},
        )
        rows = cur.fetchall()
    by_kind = {kind: int(count) for kind, count in rows}
    return EnrichPreviewReport(total=sum(by_kind.values()), by_kind=by_kind)


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


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "EnrichPreviewReport",
    "EnrichmentTask",
    "claim_tasks",
    "complete_task",
    "enqueue",
    "fail_task",
    "fail_task_permanently",
    "preview_claimable_tasks",
    "skip_task",
]
