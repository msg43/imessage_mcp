"""S5a Postgres + filesystem orchestration (SPEC §8 S5a): queries
candidate `attachment` rows, applies the first-run trial gate, throttles
materialization, checks free space periodically, and drives the
`materialization_state` state machine (`dataless` -> `materializing` ->
`materialized` / `missing` / `error` / `unsupported`).

Takes an already-open `psycopg.Connection`, never owns its lifecycle.

Two kinds of write happen in a run, and the report keeps them apart:

- **attempts** — a candidate row is read and copied (`materialized`),
  or the read fails: transiently (`error`, then `missing` once the
  retry budget is spent) or deterministically (`unsupported`, no retry
  schedule — see `imsg.backfill.classify`);
- **reclassifications** — rows whose *current* state is provably wrong
  without reading anything: a NULL `source_path` can only ever be
  `missing`, and an `error`/`missing` row whose path fails the
  containment check can only ever be `unsupported`. Both are applied
  at the start of every run so an index built before those states
  existed heals on its next pass, with no manual UPDATE.

After those, when the caller passes `locations`, the **location phase**
(`imsg.backfill.fetch`, owner decision D13) tries every other known copy
of each attachment still not materialized — `missing` and `unsupported`
included — best first, verified before it reaches the cache. Its report
is `BackfillRunReport.locations`.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.backfill.classify import (
    NO_SOURCE_PATH_ERROR,
    UnsupportedReason,
    classify_os_error,
    classify_out_of_root,
    format_unsupported_error,
)
from imsg.backfill.dataless import is_dataless
from imsg.backfill.fetch import LocationFetchReport, LocationFetchSettings, fetch_from_locations
from imsg.backfill.materialize import materialize_attachment
from imsg.backfill.throttle import RateThrottle
from imsg.background_gate import StopCheck, StopReason
from imsg.enrich.planner import enqueue_for_materialized
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
    marked_unsupported: int = 0
    """Candidates whose attempt ended in `unsupported`: refused by the
    containment check before any read, or read and failed with a
    deterministic errno (`imsg.backfill.classify`)."""
    reclassified_unsupported: int = 0
    """`error`/`missing` rows moved to `unsupported` by the pre-pass,
    without being read — their `source_path` fails containment."""
    reclassified_missing_no_source: int = 0
    """Not-yet-materialized rows with a NULL `source_path` moved to
    `missing` by the pre-pass — there is nothing on disk to retry."""
    retry_reset: int = 0
    """Rows `retry_failed=True` put back on the ladder (attempts 0,
    eligible now) before candidates were selected."""
    detected_dataless: int = 0
    detected_already_local: int = 0
    enrichment_enqueued: int = 0
    """S5b tasks queued for the rows this run materialized, from their own
    source path or (location phase) from another copy: the router's kinds
    for each file's content-sniffed MIME type
    (`imsg.enrich.planner.enqueue_for_materialized`)."""
    enrichment_unroutable: int = 0
    """Materialized rows whose sniffed type has no enrichment route."""
    enrichment_plan_errors: int = 0
    """Materialized rows whose file could not be sniffed; the
    materialization stands, and `imsg enrich --plan` retries them."""
    trial_gate_capped: bool = False
    halted_low_disk_space: bool = False
    stopped: StopReason | None = None
    """Set when the run stopped between files because heavy background
    work was paused or the host was short of memory
    (`imsg.background_gate`); the rest are tried on the next run."""
    notes: list[str] = field(default_factory=list)
    dry_run: bool = False
    """True when this report came from `run_backfill(dry_run=True)`
    (SPEC §8: "takes --dry-run where writes leave the machine").
    `considered`/`detected_dataless`/`detected_already_local`/
    `trial_gate_capped`, the two `reclassified_*` counts, `retry_reset`
    and `marked_unsupported` are accurate as "would happen" numbers —
    every one of them is decided by the database row and the path
    alone, never by attempting a read; `materialized`/`errored`/
    `marked_missing` are always 0 — see `notes` — because a read's
    outcome can only be known by attempting it, which a dry run never
    does."""
    locations: LocationFetchReport | None = None
    """The location phase (`imsg.backfill.fetch`), when it ran. Its
    materializations are counted there, not in `materialized`, which
    stays "materialized from the row's own `source_path`"."""


def _fetch_candidates(
    conn: psycopg.Connection, *, include_failed_for_retry: bool = False
) -> list[BackfillCandidate]:
    """Every row a run would attempt, in `attachment_id` order.

    `include_failed_for_retry` exists for the dry run only: it selects
    what the candidate set *would be* after `_reset_failed_for_retry`
    has run (backed-off `error` rows and given-up `missing` rows become
    eligible immediately) without applying that reset. A real run
    applies the reset first and then uses the plain query — the two
    agree by construction, because the reset leaves no `missing` row
    with a source path and no row with a future next-attempt time."""
    if include_failed_for_retry:
        where = (
            "state IN ('dataless', 'materializing', 'error', 'missing') "
            "AND source_path IS NOT NULL"
        )
    else:
        where = (
            "state IN ('dataless', 'materializing', 'error') "
            "AND source_path IS NOT NULL AND materialization_next_attempt_at <= now()"
        )
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT attachment_id, attachment_key, source_path FROM attachment "
            f"WHERE {where} ORDER BY attachment_id"
        )
        rows = cur.fetchall()
    return [BackfillCandidate(*row) for row in rows]


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
    """A *transient* failure: one more rung on the retry ladder. Returns
    True if this attempt pushed the row to `missing`. Deterministic
    failures never come here — see `_mark_unsupported`."""
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


def _mark_unsupported(
    conn: psycopg.Connection, attachment_id: int, reason: UnsupportedReason, detail: str
) -> None:
    """Terminal, and deliberately off the retry ladder: no backoff is
    scheduled (`materialization_next_attempt_at` is left at "now" so
    nothing ever shows as "retrying"), and `materialization_attempts`
    is not touched — the counter measures rungs on a ladder this row is
    not on. Only a manual UPDATE moves a row out of `unsupported`;
    `_reset_failed_for_retry` deliberately skips it, because the reason
    is a property of the row and would recur on retry."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE attachment
            SET state = 'unsupported', materialization_next_attempt_at = now(),
                materialization_last_error = %s, updated_at = now()
            WHERE attachment_id = %s
            """,
            (format_unsupported_error(reason, detail)[:2000], attachment_id),
        )
    conn.commit()


def _refusal_reason(
    source_path: str, resolved_attachments_root: Path
) -> tuple[Path, UnsupportedReason | None]:
    """The one containment code path for both the pre-pass and the
    candidate loop. Returns the resolved path and, when it does NOT lie
    under the attachments root, the reason class — the caller must then
    refuse to read it. `None` means contained."""
    resolved = resolve_path(source_path)
    if is_contained_in(resolved, resolved_attachments_root):
        return resolved, None
    return resolved, classify_out_of_root(resolved)


def _out_of_root_detail(
    source_path: str, resolved: Path, resolved_attachments_root: Path, reason: UnsupportedReason
) -> str:
    return (
        f"source_path '{source_path}' resolves to '{resolved}', which does not resolve under "
        f"the Messages attachments root ('{resolved_attachments_root}') — refusing to read it "
        f"({reason.description})"
    )


def _reset_failed_for_retry(conn: psycopg.Connection, *, dry_run: bool) -> int:
    """`imsg backfill-attachments --retry-failed`, the S5a counterpart of
    `imsg enrich --retry-failed`: put every failed row that still has
    something to read back at the bottom of the ladder, eligible now.

    Covers both failure outcomes of an attempt — `error` (backed off)
    and `missing` (given up after `MAX_MATERIALIZATION_ATTEMPTS`) — so a
    corrected classifier can be applied to all of them in one run
    rather than waiting out backoffs or hand-editing `missing` rows.
    `missing` rows go to `error` so the ordinary candidate query sees
    them; `materialization_last_error` is kept until the retry
    overwrites it (a run that halts before reaching the row still
    shows why it failed last time). `unsupported` and NULL-path rows
    are never reset: nothing about them changes on retry."""
    where = "state IN ('error', 'missing') AND source_path IS NOT NULL"
    with conn.cursor() as cur:
        if dry_run:
            cur.execute(f"SELECT count(*) FROM attachment WHERE {where}")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        cur.execute(
            f"UPDATE attachment SET state = 'error', materialization_attempts = 0, "
            f"materialization_next_attempt_at = now(), updated_at = now() WHERE {where}"
        )
        reset = cur.rowcount
    conn.commit()
    return reset


def _reclassify_no_source_path(conn: psycopg.Connection, *, dry_run: bool) -> tuple[int, set[int]]:
    """A row with no `source_path` has no placeholder to read, so
    `dataless` ("not yet materialized, will retry") is false for it: it
    is `missing`. S2 now inserts such rows as `missing` directly
    (`imsg.stages.extract`); this heals rows inserted before it did.
    Returns `(count, attachment_ids)`."""
    where = "source_path IS NULL AND state IN ('dataless', 'materializing', 'error')"
    with conn.cursor() as cur:
        cur.execute(f"SELECT attachment_id FROM attachment WHERE {where}")
        ids = {int(row[0]) for row in cur.fetchall()}
        if ids and not dry_run:
            cur.execute(
                f"UPDATE attachment SET state = 'missing', materialization_last_error = %s, "
                f"materialization_next_attempt_at = now(), updated_at = now() WHERE {where}",
                (NO_SOURCE_PATH_ERROR,),
            )
    if not dry_run:
        conn.commit()
    return len(ids), ids


def _reclassify_out_of_root(
    conn: psycopg.Connection, resolved_attachments_root: Path, *, dry_run: bool
) -> tuple[int, set[int]]:
    """`error`/`missing` rows whose `source_path` fails the containment
    check were never going to succeed on retry — the refusal is a
    property of the path. Move them to `unsupported` with the reason
    class, without reading anything (nothing outside the root is ever
    read, in this pass or any other). Rows still `dataless` are left to
    the candidate loop, which refuses them the same way when their turn
    comes. Returns `(count, attachment_ids)`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, source_path FROM attachment "
            "WHERE state IN ('error', 'missing') AND source_path IS NOT NULL "
            "ORDER BY attachment_id"
        )
        rows = cur.fetchall()
    reclassified: set[int] = set()
    for attachment_id, source_path in rows:
        resolved, reason = _refusal_reason(source_path, resolved_attachments_root)
        if reason is None:
            continue
        reclassified.add(int(attachment_id))
        if not dry_run:
            _mark_unsupported(
                conn,
                int(attachment_id),
                reason,
                _out_of_root_detail(source_path, resolved, resolved_attachments_root, reason),
            )
    return len(reclassified), reclassified


def _enqueue_enrichment(
    conn: psycopg.Connection,
    attachment_id: int,
    cache_path: Path,
    data_root: Path,
    report: BackfillRunReport,
) -> None:
    """The row just became `materialized`: queue its S5b enrichment now,
    so the worker has something to claim (SPEC §8 S5b)."""
    outcome = enqueue_for_materialized(conn, attachment_id, cache_path, data_root=data_root)
    report.enrichment_enqueued += len(outcome.enqueued)
    report.enrichment_unroutable += int(outcome.unroutable)
    report.enrichment_plan_errors += int(outcome.error is not None)


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
    retry_failed: bool = False,
    locations: LocationFetchSettings | None = None,
    stop_check: StopCheck | None = None,
) -> BackfillRunReport:
    """Run one backfill pass. `attachments_root` is the live
    `~/Library/Messages/Attachments` directory — every candidate's
    `source_path` must resolve underneath it (defense in depth: it
    already comes from our own DB, populated by S2, but path
    containment goes through `imsg.paths` everywhere per convention,
    never a trusted-by-construction shortcut).

    Order of operations, so the printed report and the database agree
    row for row: (1) `retry_failed` reset, (2) the two reclassification
    pre-passes (module docstring), (3) candidate selection + trial
    gate, (4) the attempt loop.

    `dry_run=True` (SPEC §8: "takes --dry-run where writes leave the
    machine") performs steps 1-3 as read-only counts (the pre-passes
    report what they *would* reclassify, and candidates are selected as
    they would be after those writes) and, in step 4, still classifies
    each candidate (`is_dataless`, containment) but never calls
    `throttle.wait()`, `_mark_materializing`, `materialize_attachment`,
    `_mark_materialized`/`_mark_failure`/`_mark_unsupported` — no
    filesystem copy and no Postgres write happens. See
    `BackfillRunReport.dry_run`'s docstring for which counts stay 0.

    `locations` adds step (5), the location phase (module docstring). It
    runs after the attempt loop, so an attachment's own path is always
    tried first; it shares the loop's throttle, free-space floor and, on a
    first run, what is left of the trial gate's allowance.

    `stop_check` (`imsg.background_gate`) is asked before each file of a
    real run, and by the location phase before each copy it tries; on a
    reason the run stops there with `report.stopped` set, and the
    location phase does not start.
    """
    disk_free_fn = disk_free_fn or _default_disk_free
    throttle = throttle or RateThrottle(rate_per_minute)

    report = BackfillRunReport(dry_run=dry_run)
    resolved_attachments_root = resolve_path(attachments_root)
    resolved_data_root = resolve_path(data_root)

    if retry_failed:
        report.retry_reset = _reset_failed_for_retry(conn, dry_run=dry_run)

    report.reclassified_missing_no_source, healed_no_source = _reclassify_no_source_path(
        conn, dry_run=dry_run
    )
    report.reclassified_unsupported, healed_out_of_root = _reclassify_out_of_root(
        conn, resolved_attachments_root, dry_run=dry_run
    )
    # In a real run the pre-passes have already been written, so the
    # candidate query cannot return those rows; in a dry run it can,
    # and they are excluded here so `considered` means the same thing
    # in both modes.
    already_reclassified = (healed_no_source | healed_out_of_root) if dry_run else set()

    is_first_run = not _has_ever_materialized_anything(conn)
    trial_gate_active = is_first_run and not yes_full_run

    all_candidates = [
        c
        for c in _fetch_candidates(conn, include_failed_for_retry=dry_run and retry_failed)
        if c.attachment_id not in already_reclassified
    ]
    candidates = all_candidates[:trial_limit] if trial_gate_active else all_candidates
    report.considered = len(candidates)
    if trial_gate_active and len(all_candidates) > trial_limit:
        report.trial_gate_capped = True
        report.notes.append(
            f"first run: capped at {trial_limit} of {len(all_candidates)} pending "
            f"attachments — pass yes_full_run=True to process the rest"
        )

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

            source_path, refusal = _refusal_reason(candidate.source_path, resolved_attachments_root)
            if refusal is not None:
                # Never stat/read a path outside the trusted attachments
                # root, even for read-only classification — the same
                # defense-in-depth boundary the real path enforces
                # before ever calling is_dataless() on it. The outcome
                # IS known without reading, so it is counted.
                report.marked_unsupported += 1
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
        if locations is not None and not report.halted_low_disk_space:
            report.locations = fetch_from_locations(conn, resolved_data_root, locations, dry_run=True)
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
        stop = stop_check() if stop_check is not None else None
        if stop is not None:
            report.stopped = stop
            report.notes.append(f"stopped after {i - 1} file(s): {stop.line()}")
            break

        source_path, refusal = _refusal_reason(candidate.source_path, resolved_attachments_root)
        if refusal is not None:
            _mark_unsupported(
                conn,
                candidate.attachment_id,
                refusal,
                _out_of_root_detail(
                    candidate.source_path, source_path, resolved_attachments_root, refusal
                ),
            )
            report.marked_unsupported += 1
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
            deterministic = classify_os_error(exc)
            if deterministic is not None:
                _mark_unsupported(conn, candidate.attachment_id, deterministic, str(exc))
                report.marked_unsupported += 1
            elif _mark_failure(conn, candidate.attachment_id, str(exc)):
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
        _enqueue_enrichment(conn, candidate.attachment_id, result.cache_path, resolved_data_root, report)

    if locations is not None and not report.halted_low_disk_space and report.stopped is None:
        budget = max(0, trial_limit - report.considered) if trial_gate_active else None
        report.locations = fetch_from_locations(
            conn,
            resolved_data_root,
            locations,
            throttle=throttle,
            budget=budget,
            disk_free_fn=disk_free_fn,
            min_free_bytes=min_free_bytes,
            free_space_check_interval=free_space_check_interval,
            read_this_run={c.attachment_id: c.source_path for c in candidates},
            stop_check=stop_check,
        )
        if report.locations.halted_low_disk_space:
            report.halted_low_disk_space = True
        if report.locations.stopped is not None:
            report.stopped = report.locations.stopped
        if report.locations.budget_capped:
            report.trial_gate_capped = True
        report.enrichment_enqueued += report.locations.enrichment_enqueued
        report.enrichment_unroutable += report.locations.enrichment_unroutable
        report.enrichment_plan_errors += report.locations.enrichment_plan_errors

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
