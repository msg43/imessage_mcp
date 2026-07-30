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
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import psycopg

from imsg.mcp.errors import AuditWriteError

# Stable machine codes permitted in mcp_audit.error. RATE_LIMITED and
# INTERNAL overlap SPEC §10.1's tool-error codes; UNAUTHORIZED and
# UNAVAILABLE are HTTP-boundary rejections that occur before any tool
# runs and therefore have no §10.1 equivalent.
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


class MemoryAuditSink:
    """Thread-safe in-memory sink for tests and the synthetic AT-1 probe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []

    def record(self, rec: AuditRecord) -> None:
        with self._lock:
            self._records.append(rec.sanitized())

    def snapshot(self) -> Sequence[AuditRecord]:
        with self._lock:
            return tuple(self._records)


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


class PostgresAuditSink:
    """Writes audit rows to the dedicated instance's `mcp_audit` table.

    Takes a connection *factory* rather than holding a connection so the
    caller controls pooling/lifecycle; every write failure surfaces as
    AuditWriteError, which the gate converts into a 503 denial — an
    unauditable request is never served (SPEC §12 step 4 depends on the
    audit trail being complete).
    """

    def __init__(self, connection_factory: Callable[[], psycopg.Connection]) -> None:
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
        except psycopg.Error as exc:
            # Deliberately does not interpolate exc into the message:
            # driver errors can quote SQL and parameter values.
            raise AuditWriteError("failed to write mcp_audit row") from exc

    def snapshot(self) -> Sequence[AuditRecord]:
        try:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(_SNAPSHOT_SQL)
                rows = cur.fetchall()
        except psycopg.Error as exc:
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
        except psycopg.Error as exc:
            raise AuditWriteError("failed to query mcp_audit") from exc
        return int(row[0]) if row is not None else 0


__all__ = [
    "ACCEPTED_FOREIGN_SQL",
    "ALLOWED_ERROR_CODES",
    "AuditReader",
    "AuditRecord",
    "AuditSink",
    "MemoryAuditSink",
    "PostgresAuditSink",
    "accepted_foreign_subjects",
    "hash_params",
    "sanitize_error_code",
    "truncate_subject",
]
