"""Audit logging for the MCP surfaces — `mcp_audit` (SPEC §7.2, §10.4, §14).

Invariants enforced here, not merely documented:

- **Bodies are never logged.** The sink API physically cannot receive
  message content: it takes an :class:`AuditRecord`, whose only
  params-shaped field is a sha256 hex digest produced by
  :func:`hash_params`.
- **Error strings come from a closed set.** Free-text errors (which could
  carry exception text, paths, SQL, or corpus content) are replaced with
  ``INTERNAL`` before storage.
- **Raw rejected subjects are recorded** (SPEC §10.4 item 3) so AT-1
  step 4 can prove rejections happened — but they are length-capped so an
  upstream anomaly cannot bloat rows.

AT-1's standing invariant (SPEC §12 step 4) is expressed here once, as
:func:`accepted_foreign_subjects` (in-memory) and
:data:`ACCEPTED_FOREIGN_SQL` (Postgres), so the probe and ops checks
cannot drift from each other.

**Two kinds of row (2026-09-24).** `mcp_audit` holds one row per request
that carried a token the gate judged, and per tool call. Requests turned
away before any token was judged — no `Authorization` header, a malformed
one, a client already throttled, the tokeninfo breaker open — are counted
in memory (:class:`RejectionTally`) and written as one row per code per
interval to `mcp_audit_rollup` (:class:`RejectionTallyWriter`). Internet
scanners send exactly those requests, and a row per request meant a
Postgres connection and a write on the event loop for each of them, with
no limit (QA review 2026-09-24). Nothing about a subject is lost: such a
request has none.

**Retention** (:func:`prune_audit`, `imsg mcp audit-prune`): detailed rows
older than `mcp.audit_retention_days` are rolled up into one
`mcp_audit_rollup` row per UTC day and outcome, then deleted — except
accepted public rows, which are kept whole, because AT-1's standing check
(`ACCEPTED_FOREIGN_SQL`) is asserted over the table's entire history and
a pruned row could be the one that proves a breach. There are only as many
of those as the owner's own tool calls. Aggregate rows are never deleted;
interval rows older than the window are merged into one row per day.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

import psycopg

from imsg.db.pool import PoolExhaustedError
from imsg.mcp.errors import AuditWriteError

logger = logging.getLogger(__name__)

# Stable machine codes permitted in mcp_audit.error. RATE_LIMITED and
# INTERNAL overlap SPEC §10.1's tool-error codes; UNAUTHORIZED and
# UNAVAILABLE are HTTP-boundary rejections that occur before any tool
# runs and therefore have no §10.1 equivalent. WARMING_UP and
# WARM_UP_FAILED are the local surface's answers while its models load
# in the background (imsg.retrieval.errors) — fixed codes like the rest,
# kept distinct so a warm-up shows in the table as what it is, not as
# INTERNAL.
ALLOWED_ERROR_CODES: frozenset[str] = frozenset(
    {
        "INVALID_ARGUMENT",
        "PERSON_NOT_FOUND",
        "PERSON_AMBIGUOUS",
        "DATE_RANGE_INVALID",
        "NOT_FOUND",
        "NOT_ENRICHED",
        "SCOPE_DENIED",
        "RATE_LIMITED",
        "INTERNAL",
        "UNAUTHORIZED",
        "UNAVAILABLE",
        "WARMING_UP",
        "WARM_UP_FAILED",
    }
)

_MAX_SUBJECT_CHARS = 256


def sanitize_error_code(code: str | None) -> str | None:
    """Collapse anything outside the closed error-code set to ``INTERNAL``.

    This is what keeps exception text (which can embed paths, SQL, or
    quoted corpus content) out of the audit table.
    """
    if code is None:
        return None
    return code if code in ALLOWED_ERROR_CODES else "INTERNAL"


def truncate_subject(subject: str | None) -> str | None:
    if subject is None:
        return None
    return subject[:_MAX_SUBJECT_CHARS]


def hash_params(params: Mapping[str, object] | None) -> str | None:
    """Canonical sha256 of tool params — the only params-derived value stored.

    Canonical form: JSON with sorted keys, compact separators, UTF-8. Two
    semantically identical param dicts always hash identically, so audit
    rows are joinable across requests without ever storing the params.
    """
    if params is None:
        return None
    canonical = json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One row of `mcp_audit`. Construct via the gate, not by hand."""

    surface: str  # 'local' | 'public'
    subject: str | None
    subject_ok: bool
    tool: str | None
    params_sha256: str | None
    result_count: int | None
    latency_ms: int | None
    error: str | None

    def sanitized(self) -> AuditRecord:
        return replace(
            self,
            subject=truncate_subject(self.subject),
            error=sanitize_error_code(self.error),
        )


class AuditSink(Protocol):
    """Where audit rows go. Implementations must raise AuditWriteError on failure."""

    def record(self, rec: AuditRecord) -> None: ...


class AuditReader(Protocol):
    """Read-back interface the AT-1 probe consumes (SPEC §12 steps 2-4)."""

    def snapshot(self) -> Sequence[AuditRecord]: ...


UNAUTHENTICATED_SOURCE = "unauthenticated"
"""`mcp_audit_rollup.source` for counts the public server kept in memory:
requests turned away before any token was judged (:class:`RejectionTally`)."""

RETENTION_SOURCE = "retention"
"""`mcp_audit_rollup.source` for detailed `mcp_audit` rows rolled up by
:func:`prune_audit` once they were older than the retention window."""


@dataclass(frozen=True, slots=True)
class AggregateRecord:
    """One row of `mcp_audit_rollup`: how many requests ended one way —
    the same `surface`/`subject_ok`/`tool`/`error` columns `mcp_audit`
    has — over one period. Never carries a subject: the unauthenticated
    counts have none, and retention drops rejected subjects on purpose."""

    period_start: datetime
    period_end: datetime
    source: str  # UNAUTHENTICATED_SOURCE | RETENTION_SOURCE
    surface: str  # 'local' | 'public'
    subject_ok: bool
    tool: str | None
    error: str | None
    request_count: int

    def sanitized(self) -> AggregateRecord:
        return replace(self, error=sanitize_error_code(self.error))


class AggregateAuditSink(Protocol):
    """Where aggregate rows go. Must write all of `records` or none, and
    raise AuditWriteError on failure."""

    def record_aggregates(self, records: Sequence[AggregateRecord]) -> None: ...


class MemoryAuditSink:
    """Thread-safe in-memory sink for tests and the synthetic AT-1 probe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []
        self._aggregates: list[AggregateRecord] = []

    def record(self, rec: AuditRecord) -> None:
        with self._lock:
            self._records.append(rec.sanitized())

    def snapshot(self) -> Sequence[AuditRecord]:
        with self._lock:
            return tuple(self._records)

    def record_aggregates(self, records: Sequence[AggregateRecord]) -> None:
        with self._lock:
            self._aggregates.extend(r.sanitized() for r in records)

    def aggregate_snapshot(self) -> Sequence[AggregateRecord]:
        with self._lock:
            return tuple(self._aggregates)


def accepted_foreign_subjects(
    records: Sequence[AuditRecord], owner_subject: str
) -> tuple[AuditRecord, ...]:
    """AT-1 step 4, in memory: accepted public rows whose subject is not the owner.

    This must be empty — permanently, not just during the test window.
    """
    return tuple(
        r
        for r in records
        if r.surface == "public" and r.subject_ok and r.subject != owner_subject
    )


# AT-1 step 4, verbatim shape from SPEC §12: must return 0 rows.
ACCEPTED_FOREIGN_SQL = (
    "SELECT count(*) FROM mcp_audit "
    "WHERE surface = 'public' AND subject_ok AND subject <> %(owner_subject)s"
)

_INSERT_SQL = (
    "INSERT INTO mcp_audit "
    "(surface, subject, subject_ok, tool, params_sha256, result_count, latency_ms, error) "
    "VALUES (%(surface)s, %(subject)s, %(subject_ok)s, %(tool)s, %(params_sha256)s, "
    "%(result_count)s, %(latency_ms)s, %(error)s)"
)

_SNAPSHOT_SQL = (
    "SELECT surface, subject, subject_ok, tool, params_sha256, result_count, "
    "latency_ms, error FROM mcp_audit ORDER BY audit_id"
)

_INSERT_AGGREGATE_SQL = (
    "INSERT INTO mcp_audit_rollup "
    "(period_start, period_end, source, surface, subject_ok, tool, error, request_count) "
    "VALUES (%(period_start)s, %(period_end)s, %(source)s, %(surface)s, %(subject_ok)s, "
    "%(tool)s, %(error)s, %(request_count)s)"
)


class PostgresAuditSink:
    """Writes audit rows to the dedicated instance's `mcp_audit` table.

    Takes a connection *factory* rather than holding a connection so the
    caller controls pooling/lifecycle; every write failure surfaces as
    AuditWriteError, which the gate converts into a 503 denial — an
    unauditable request is never served (SPEC §12 step 4 depends on the
    audit trail being complete).

    **The factory's return value is used as a `with` block that yields
    the connection, and leaving the block must be safe.** Two factories
    qualify. `lambda: connect(cfg.database, autocommit=True)` returns a
    fresh `psycopg.Connection`, which the block closes on exit — a
    connection per row, 8.2-10.1 ms each on the index host (QA review
    2026-09-24). `pool.lease` (`imsg.db.pool.ResourcePool`) lends an
    open connection and takes it back on exit — what `imsg mcp public`
    passes since 2026-09-24. A factory returning one *shared* connection
    does not qualify: the block closes it after the first row, and every
    row after answers `the connection is closed`. The factory is called
    from several threads at once, because the public surface writes
    rows on worker threads (`imsg.mcp.tools.public_server`); both
    qualifying factories are safe for that.
    """

    def __init__(
        self,
        connection_factory: Callable[[], AbstractContextManager[psycopg.Connection]],
    ) -> None:
        self._connection_factory = connection_factory

    def record(self, rec: AuditRecord) -> None:
        clean = rec.sanitized()
        try:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(
                    _INSERT_SQL,
                    {
                        "surface": clean.surface,
                        "subject": clean.subject,
                        "subject_ok": clean.subject_ok,
                        "tool": clean.tool,
                        "params_sha256": clean.params_sha256,
                        "result_count": clean.result_count,
                        "latency_ms": clean.latency_ms,
                        "error": clean.error,
                    },
                )
                conn.commit()
        except (psycopg.Error, PoolExhaustedError) as exc:
            # Deliberately does not interpolate exc into the message:
            # driver errors can quote SQL and parameter values.
            raise AuditWriteError("failed to write mcp_audit row") from exc

    def record_aggregates(self, records: Sequence[AggregateRecord]) -> None:
        """Write every record in one transaction, or none."""
        if not records:
            return
        rows = [r.sanitized() for r in records]
        try:
            with self._connection_factory() as conn, conn.transaction(), conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        _INSERT_AGGREGATE_SQL,
                        {
                            "period_start": row.period_start,
                            "period_end": row.period_end,
                            "source": row.source,
                            "surface": row.surface,
                            "subject_ok": row.subject_ok,
                            "tool": row.tool,
                            "error": row.error,
                            "request_count": row.request_count,
                        },
                    )
        except (psycopg.Error, PoolExhaustedError) as exc:
            raise AuditWriteError("failed to write mcp_audit_rollup rows") from exc

    def snapshot(self) -> Sequence[AuditRecord]:
        try:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(_SNAPSHOT_SQL)
                rows = cur.fetchall()
        except (psycopg.Error, PoolExhaustedError) as exc:
            raise AuditWriteError("failed to read mcp_audit") from exc
        return tuple(
            AuditRecord(
                surface=row[0],
                subject=row[1],
                subject_ok=row[2],
                tool=row[3],
                params_sha256=row[4],
                result_count=row[5],
                latency_ms=row[6],
                error=row[7],
            )
            for row in rows
        )

    def count_accepted_foreign(self, owner_subject: str) -> int:
        """AT-1 step 4 against the live table. MUST return 0."""
        try:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(ACCEPTED_FOREIGN_SQL, {"owner_subject": owner_subject})
                row = cur.fetchone()
        except (psycopg.Error, PoolExhaustedError) as exc:
            raise AuditWriteError("failed to query mcp_audit") from exc
        return int(row[0]) if row is not None else 0


# ---------------------------------------------------------------------------
# Requests turned away before any token was judged: counted, not logged
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RejectionCounts:
    """What :class:`RejectionTally` counted over one period."""

    period_start: datetime
    period_end: datetime
    counts: Mapping[str, int]


class RejectionTally:
    """Counts, in memory, the public requests turned away before any token
    was judged, by machine code: no `Authorization` header or a malformed
    one, a bad `Host`/`Origin` or a duplicated header, a client already
    throttled, the global failure budget spent, the tokeninfo breaker
    open. :class:`RejectionTallyWriter` turns each period's counts into
    one `mcp_audit_rollup` row per code.

    Adding is a dictionary update under a lock — no I/O — so the event
    loop can do it for a flood of requests without slowing down. Thread-
    safe. Counts not yet written when the process dies are lost (at most
    one writer interval's worth); none of them could have carried a
    subject."""

    def __init__(self, *, now: Callable[[], datetime] = _utc_now) -> None:
        self._now = now
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._since = now()

    def add(self, code: str) -> None:
        clean = sanitize_error_code(code) or "INTERNAL"
        with self._lock:
            self._counts[clean] = self._counts.get(clean, 0) + 1

    def pending(self) -> dict[str, int]:
        """The counts not yet taken by a writer."""
        with self._lock:
            return dict(self._counts)

    def take(self) -> RejectionCounts:
        """Everything counted since the last take, and start a new period."""
        with self._lock:
            now = self._now()
            taken = RejectionCounts(
                period_start=self._since, period_end=now, counts=self._counts
            )
            self._since, self._counts = now, {}
        return taken

    def put_back(self, taken: RejectionCounts) -> None:
        """Return counts a writer could not store, so the next write
        includes them and its period starts where theirs did."""
        with self._lock:
            self._since = min(self._since, taken.period_start)
            for code, count in taken.counts.items():
                self._counts[code] = self._counts.get(code, 0) + count


DEFAULT_REJECTION_WRITE_INTERVAL_SECONDS = 60


class RejectionTallyWriter:
    """Writes a :class:`RejectionTally`'s counts to `mcp_audit_rollup`
    every `interval_seconds`, on a daemon thread of its own, and once more
    when stopped.

    A failed write keeps the counts for the next attempt and says so on
    the log; it never raises into the thread or blocks a request. The
    rows it writes are about requests that were refused anyway, so a
    database that is down costs a gap in the counts, not a request served
    unaudited — which is why this is allowed to fail softly where
    `PostgresAuditSink.record` on the accept path is not (D7.2)."""

    def __init__(
        self,
        tally: RejectionTally,
        sink: AggregateAuditSink,
        *,
        interval_seconds: float = DEFAULT_REJECTION_WRITE_INTERVAL_SECONDS,
        surface: str = "public",
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("the write interval must be positive")
        self._tally = tally
        self._sink = sink
        self._interval = interval_seconds
        self._surface = surface
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failing = False

    def write_now(self) -> int:
        """Write what has been counted since the last write; return the
        number of rows written (0 when nothing was counted, or the write
        failed and the counts were kept)."""
        taken = self._tally.take()
        if not taken.counts:
            return 0
        records = [
            AggregateRecord(
                period_start=taken.period_start,
                period_end=taken.period_end,
                source=UNAUTHENTICATED_SOURCE,
                surface=self._surface,
                subject_ok=False,
                tool=None,
                error=code,
                request_count=count,
            )
            for code, count in sorted(taken.counts.items())
        ]
        try:
            self._sink.record_aggregates(records)
        except AuditWriteError:
            self._tally.put_back(taken)
            if not self._failing:
                logger.warning(
                    "mcp.rejection_counts_not_written",
                    extra={"requests": sum(taken.counts.values())},
                )
            self._failing = True
            return 0
        self._failing = False
        return len(records)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.write_now()
            except Exception:  # a writer that dies stops counting reaching the table
                logger.exception("mcp.rejection_count_writer_error")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="imsg-rejection-counts", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and write whatever is still counted."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 10)
            self._thread = None
        self.write_now()


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

DEFAULT_RETENTION_DAYS = 90
"""SPEC §14's retention for logs, applied to the detailed audit rows."""


@dataclass(frozen=True, slots=True)
class AuditPruneReport:
    """What one :func:`prune_audit` run did (or, dry, would do)."""

    cutoff: datetime
    detail_rows_rolled_up: int
    """`mcp_audit` rows older than the cutoff, counted into daily
    `mcp_audit_rollup` rows and deleted."""
    daily_rows_written: int
    """`retention` rows written for them."""
    accepted_public_rows_kept: int
    """Accepted public rows older than the cutoff, kept whole on purpose
    (module docstring: AT-1's standing check reads the whole history)."""
    interval_rows_merged: int
    """Unauthenticated interval rows older than the cutoff, merged into
    one row per day."""
    merged_rows_written: int
    dry_run: bool


_COUNT_KEPT_SQL = (
    "SELECT count(*) FROM mcp_audit "
    "WHERE ts < %(cutoff)s AND surface = 'public' AND subject_ok"
)

_ROLL_UP_DETAIL_SQL = """
WITH moved AS (
    DELETE FROM mcp_audit
    WHERE ts < %(cutoff)s
      AND NOT (surface = 'public' AND subject_ok)
    RETURNING ts, surface, subject_ok, tool, error
), by_day AS (
    SELECT date_trunc('day', ts AT TIME ZONE 'UTC') AS day_utc,
           surface, subject_ok, tool, error, count(*) AS n
    FROM moved
    GROUP BY 1, 2, 3, 4, 5
)
INSERT INTO mcp_audit_rollup
    (period_start, period_end, source, surface, subject_ok, tool, error, request_count)
SELECT day_utc AT TIME ZONE 'UTC', (day_utc + interval '1 day') AT TIME ZONE 'UTC',
       'retention', surface, subject_ok, tool, error, n
FROM by_day
RETURNING request_count
"""

_COUNT_INTERVALS_SQL = (
    "SELECT count(*) FROM mcp_audit_rollup "
    "WHERE source = 'unauthenticated' AND period_end <= %(cutoff)s "
    "AND period_end - period_start < interval '1 day'"
)

_MERGE_INTERVALS_SQL = """
WITH merged AS (
    DELETE FROM mcp_audit_rollup
    WHERE source = 'unauthenticated'
      AND period_end <= %(cutoff)s
      AND period_end - period_start < interval '1 day'
    RETURNING period_start, surface, subject_ok, tool, error, request_count
), by_day AS (
    SELECT date_trunc('day', period_start AT TIME ZONE 'UTC') AS day_utc,
           surface, subject_ok, tool, error, sum(request_count) AS n
    FROM merged
    GROUP BY 1, 2, 3, 4, 5
)
INSERT INTO mcp_audit_rollup
    (period_start, period_end, source, surface, subject_ok, tool, error, request_count)
SELECT day_utc AT TIME ZONE 'UTC', (day_utc + interval '1 day') AT TIME ZONE 'UTC',
       'unauthenticated', surface, subject_ok, tool, error, n
FROM by_day
RETURNING request_count
"""


def retention_cutoff(now: datetime, days: int) -> datetime:
    """UTC midnight `days` days before `now`: rows before it are pruned.
    A whole-day boundary, so one run rolls up whole days and a day never
    ends up split across two runs' rows."""
    if days < 1:
        raise ValueError("retention must keep at least one day")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    utc = now.astimezone(UTC)
    midnight = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=days)


def prune_audit(
    conn: psycopg.Connection, *, cutoff: datetime, dry_run: bool = False
) -> AuditPruneReport:
    """Roll `mcp_audit` rows older than `cutoff` into one
    `mcp_audit_rollup` row per UTC day and outcome and delete them —
    except accepted public rows, which stay whole — and merge
    unauthenticated interval rows older than `cutoff` into one row per
    day. One transaction; `dry_run` runs it and rolls it back, so the
    counts are exactly what a real run would do.

    Safe to run while a server writes: new rows are newer than any
    cutoff, and two concurrent runs cannot count a row twice (the second
    run's DELETE waits for the first and then finds the rows gone)."""
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    params = {"cutoff": cutoff}

    def count(cur: psycopg.Cursor[tuple[int]], sql: str) -> int:
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row is not None else 0

    with conn.transaction():
        with conn.cursor() as cur:
            kept = count(cur, _COUNT_KEPT_SQL)
            intervals = count(cur, _COUNT_INTERVALS_SQL)
            cur.execute(_ROLL_UP_DETAIL_SQL, params)
            daily = [int(r[0]) for r in cur.fetchall()]
            cur.execute(_MERGE_INTERVALS_SQL, params)
            merged = [int(r[0]) for r in cur.fetchall()]
        report = AuditPruneReport(
            cutoff=cutoff,
            detail_rows_rolled_up=sum(daily),
            daily_rows_written=len(daily),
            accepted_public_rows_kept=kept,
            interval_rows_merged=intervals,
            merged_rows_written=len(merged),
            dry_run=dry_run,
        )
        if dry_run:
            # Swallowed by the transaction block: everything above is
            # undone, and the counts are what a real run would have done.
            raise psycopg.Rollback()
    return report


__all__ = [
    "ACCEPTED_FOREIGN_SQL",
    "ALLOWED_ERROR_CODES",
    "DEFAULT_REJECTION_WRITE_INTERVAL_SECONDS",
    "DEFAULT_RETENTION_DAYS",
    "RETENTION_SOURCE",
    "UNAUTHENTICATED_SOURCE",
    "AggregateAuditSink",
    "AggregateRecord",
    "AuditPruneReport",
    "AuditReader",
    "AuditRecord",
    "AuditSink",
    "MemoryAuditSink",
    "PostgresAuditSink",
    "RejectionCounts",
    "RejectionTally",
    "RejectionTallyWriter",
    "accepted_foreign_subjects",
    "hash_params",
    "prune_audit",
    "retention_cutoff",
    "sanitize_error_code",
    "truncate_subject",
]
