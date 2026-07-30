"""S5a Postgres + filesystem orchestration (SPEC §8 S5a): queries
candidate `attachment` rows, applies the first-run trial gate, throttles
materialization, checks free space periodically, and drives the
`materialization_state` state machine (`dataless` -> `materializing` ->
`materialized` / `missing` / `error`).

Takes an already-open `psycopg.Connection`, never owns its lifecycle.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.backfill.dataless import is_dataless
from imsg.backfill.materialize import materialize_attachment
from imsg.backfill.throttle import RateThrottle
from imsg.paths import is_contained_in, resolve_path

if TYPE_CHECKING:
    import psycopg

DEFAULT_RATE_PER_MINUTE = 60.0
DEFAULT_TRIAL_LIMIT = 12
"""SPEC §8 S5a: "Trial gate: first run refuses to process more than 12
files without --yes-full-run" (architecture §5.8: "test on a dozen
before ten years")."""
DEFAULT_FREE_SPACE_CHECK_INTERVAL = 100
DEFAULT_MIN_FREE_BYTES = 50 * 1024**3  # 50 GB
MAX_MATERIALIZATION_ATTEMPTS = 3
"""SPEC §8 S5a: "state 'missing' after 3 attempts across >= 2 runs".
Each failed attempt pushes `materialization_next_attempt_at` into the
future, and a run only considers rows whose next-attempt time has
already passed — so attempt 2 can never happen in the same run as
attempt 1, which is what makes "across >= 2 runs" fall out for free
rather than needing separate run-boundary bookkeeping."""
_BACKOFF_BASE_MINUTES = 30

DiskFreeFn = Callable[[Path], int]


def _default_disk_free(path: Path) -> int:
    """`shutil.disk_usage` requires an existing path; `data_root` may not
    exist yet on a first run before anything has been written under it,
    so walk up to the nearest existing ancestor first (same fallback
    `imsg.diagnostics.disk_free_bytes` uses)."""
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:  # reached the filesystem root without finding anything
            return 0
        candidate = parent
    return shutil.disk_usage(candidate).free


@dataclass(frozen=True, slots=True)
class BackfillCandidate:
    attachment_id: int
    attachment_key: str
    source_path: str


@dataclass
class BackfillRunReport:
    considered: int = 0
    materialized: int = 0
    errored: int = 0
    marked_missing: int = 0
    detected_dataless: int = 0
    detected_already_local: int = 0
    trial_gate_capped: bool = False
    halted_low_disk_space: bool = False
    notes: list[str] = field(default_factory=list)
    dry_run: bool = False
    """True when this report came from `run_backfill(dry_run=True)`
    (SPEC §8: "takes --dry-run where writes leave the machine").
    `considered`/`detected_dataless`/`detected_already_local`/
    `trial_gate_capped` are accurate (the same read-only candidate
    selection and dataless classification ran); `materialized`/
    `errored`/`marked_missing` are always 0 — see `notes` — because
    materialization outcome (success/failure) can only be known by
    attempting it, which a dry run never does."""


def _count_pending(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM attachment WHERE state IN ('dataless', 'materializing', 'error') "
            "AND source_path IS NOT NULL AND materialization_next_attempt_at <= now()"
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def _fetch_candidates(conn: psycopg.Connection, *, limit: int | None) -> list[BackfillCandidate]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT attachment_id, attachment_key, source_path
            FROM attachment
            WHERE state IN ('dataless', 'materializing', 'error')
              AND source_path IS NOT NULL
              AND materialization_next_attempt_at <= now()
            ORDER BY attachment_id
            """
        )
        rows = cur.fetchall()
    candidates = [BackfillCandidate(*row) for row in rows]
    return candidates[:limit] if limit is not None else candidates


def _has_ever_materialized_anything(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM attachment WHERE state = 'materialized' LIMIT 1")
        return cur.fetchone() is not None


def _mark_materializing(conn: psycopg.Connection, attachment_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE attachment SET state = 'materializing', updated_at = now() "
            "WHERE attachment_id = %s",
            (attachment_id,),
        )
    conn.commit()


def _mark_materialized(
    conn: psycopg.Connection,
    attachment_id: int,
    *,
    sha256: str,
    byte_size: int,
    cache_path: Path,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE attachment
            SET state = 'materialized', sha256 = %s, byte_size = %s, cache_path = %s,
                materialization_last_error = NULL, updated_at = now()
            WHERE attachment_id = %s
            """,
            (sha256, byte_size, str(cache_path), attachment_id),
        )
    conn.commit()


def _mark_failure(conn: psycopg.Connection, attachment_id: int, error: str) -> bool:
    """Returns True if this attempt pushed the row to `missing`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT materialization_attempts FROM attachment WHERE attachment_id = %s",
            (attachment_id,),
        )
        row = cur.fetchone()
        attempts = (row[0] if row else 0) + 1
        became_missing = attempts >= MAX_MATERIALIZATION_ATTEMPTS
        next_state = "missing" if became_missing else "error"
        backoff = timedelta(minutes=_BACKOFF_BASE_MINUTES * (2 ** (attempts - 1)))
        cur.execute(
            """
            UPDATE attachment
            SET state = %s, materialization_attempts = %s,
                materialization_next_attempt_at = %s,
                materialization_last_error = %s, updated_at = now()
            WHERE attachment_id = %s
            """,
            (next_state, attempts, datetime.now(UTC) + backoff, error[:2000], attachment_id),
        )
    conn.commit()
    return became_missing


def run_backfill(
    conn: psycopg.Connection,
    data_root: Path,
    attachments_root: Path,
    *,
    rate_per_minute: float = DEFAULT_RATE_PER_MINUTE,
    yes_full_run: bool = False,
    trial_limit: int = DEFAULT_TRIAL_LIMIT,
    free_space_check_interval: int = DEFAULT_FREE_SPACE_CHECK_INTERVAL,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    throttle: RateThrottle | None = None,
    disk_free_fn: DiskFreeFn | None = None,
    dry_run: bool = False,
) -> BackfillRunReport:
    """Run one backfill pass. `attachments_root` is the live
    `~/Library/Messages/Attachments` directory — every candidate's
    `source_path` must resolve underneath it (defense in depth: it
    already comes from our own DB, populated by S2, but path
    containment goes through `imsg.paths` everywhere per convention,
    never a trusted-by-construction shortcut).

    `dry_run=True` (SPEC §8: "takes --dry-run where writes leave the
    machine") still does the read-only candidate selection
    (`_fetch_candidates`), trial-gate accounting, and `is_dataless`
    classification, but never calls `throttle.wait()`,
    `_mark_materializing`, `materialize_attachment`, or
    `_mark_materialized`/`_mark_failure` — no filesystem copy and no
    Postgres write happens. See `BackfillRunReport.dry_run`'s docstring
    for why `materialized`/`errored`/`marked_missing` stay 0.
    """
    disk_free_fn = disk_free_fn or _default_disk_free
    throttle = throttle or RateThrottle(rate_per_minute)

    is_first_run = not _has_ever_materialized_anything(conn)
    trial_gate_active = is_first_run and not yes_full_run
    limit = trial_limit if trial_gate_active else None

    total_pending = _count_pending(conn)
    candidates = _fetch_candidates(conn, limit=limit)

    report = BackfillRunReport(considered=len(candidates), dry_run=dry_run)
    if trial_gate_active and total_pending > trial_limit:
        report.trial_gate_capped = True
        report.notes.append(
            f"first run: capped at {trial_limit} of {total_pending} pending "
            f"attachments — pass yes_full_run=True to process the rest"
        )

    resolved_attachments_root = resolve_path(attachments_root)
    resolved_data_root = resolve_path(data_root)

    if dry_run:
        for i, candidate in enumerate(candidates, start=1):
            if i == 1 or i % free_space_check_interval == 0:
                free = disk_free_fn(resolved_data_root)
                if free < min_free_bytes:
                    report.halted_low_disk_space = True
                    report.notes.append(
                        f"dry run: a real run would halt after {i - 1} file(s): "
                        f"free space {free} bytes < minimum {min_free_bytes} bytes"
                    )
                    break

            source_path = resolve_path(candidate.source_path)
            if not is_contained_in(source_path, resolved_attachments_root):
                # Never stat/read a path outside the trusted attachments
                # root, even for read-only classification — the same
                # defense-in-depth boundary the real path enforces
                # before ever calling is_dataless() on it.
                continue

            if is_dataless(source_path):
                report.detected_dataless += 1
            else:
                report.detected_already_local += 1

        report.notes.append(
            "dry run — materialized/errored/marked_missing are always 0: whether "
            "materialization would succeed or fail can't be known without "
            "attempting it"
        )
        return report

    for i, candidate in enumerate(candidates, start=1):
        if i == 1 or i % free_space_check_interval == 0:
            free = disk_free_fn(resolved_data_root)
            if free < min_free_bytes:
                report.halted_low_disk_space = True
                report.notes.append(
                    f"halted after {i - 1} file(s): free space {free} bytes < "
                    f"minimum {min_free_bytes} bytes"
                )
                break

        source_path = resolve_path(candidate.source_path)
        if not is_contained_in(source_path, resolved_attachments_root):
            _mark_failure(
                conn,
                candidate.attachment_id,
                f"source_path '{candidate.source_path}' does not resolve under the "
                f"Messages attachments root ('{resolved_attachments_root}') — refusing to read it",
            )
            report.errored += 1
            continue

        if is_dataless(source_path):
            report.detected_dataless += 1
        else:
            report.detected_already_local += 1

        throttle.wait()
        _mark_materializing(conn, candidate.attachment_id)
        try:
            result = materialize_attachment(source_path, resolved_data_root)
        except OSError as exc:
            became_missing = _mark_failure(conn, candidate.attachment_id, str(exc))
            if became_missing:
                report.marked_missing += 1
            else:
                report.errored += 1
            continue

        _mark_materialized(
            conn,
            candidate.attachment_id,
            sha256=result.sha256,
            byte_size=result.byte_size,
            cache_path=result.cache_path,
        )
        report.materialized += 1

    return report


__all__ = [
    "DEFAULT_FREE_SPACE_CHECK_INTERVAL",
    "DEFAULT_MIN_FREE_BYTES",
    "DEFAULT_RATE_PER_MINUTE",
    "DEFAULT_TRIAL_LIMIT",
    "MAX_MATERIALIZATION_ATTEMPTS",
    "BackfillCandidate",
    "BackfillRunReport",
    "run_backfill",
]
