"""Local-surface tool dispatch (SPEC §10.1, §10.3): wraps one tool
call in the SPEC §10.1 error model (a stable machine-code first line,
never a raw traceback) and writes one `mcp_audit` row with
`surface='local'` (SPEC §10.3: "Audit rows written with
`surface='local'`").

This is the local surface's analogue of
`imsg.mcp.auth.PublicAuthGate.dispatch`, reusing only the audit half of
that pattern — the local surface has no OAuth subject to validate (it
is reached over SSH on the tailnet instead, SPEC §10.3), so there is
no gate to run, only the same `imsg.mcp.audit.AuditSink`/
`AuditRecord`/`hash_params` the public gate itself writes through
(imported, never modified — see that package's own docstring for why
`audit.py` is off-limits to edit this wave).

**Audit-write-failure handling differs from the public gate on
purpose**: `PublicAuthGate.dispatch` withholds a computed payload and
answers 503 when the accept-path audit row cannot be written (SPEC
§10.4's confidentiality-driven "an unauditable request is denied"
rule, load-bearing for AT-1). The local surface's access boundary is
SSH + tailnet reachability, not the audit trail, so this module's
judgment call is to log the audit-write failure and still return the
successful payload rather than fail an otherwise-successful local
request over a logging problem.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any

import structlog

from imsg.mcp.audit import AuditRecord, AuditSink, hash_params
from imsg.mcp.errors import AuditWriteError
from imsg.retrieval.errors import RetrievalError

logger = structlog.get_logger(__name__)

LOCAL_SUBJECT = "local"
"""`mcp_audit.subject` for every local-surface row (SPEC §7.2 column
comment: "raw OAuth sub on public; 'local' on local")."""


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Either `payload` (success) or `error_code`/`error_message` (a
    SPEC §10.1 tool error) — never both. `error_code` is one of the
    closed SPEC §10.1 machine codes."""

    payload: dict[str, Any] | None
    error_code: str | None
    error_message: str | None

    @property
    def is_error(self) -> bool:
        return self.error_code is not None


def _result_count(payload: dict[str, Any]) -> int | None:
    for key in ("results", "messages", "people", "texts"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _record(
    audit: AuditSink,
    *,
    tool: str,
    params: Mapping[str, Any] | None,
    result_count: int | None,
    latency_ms: int,
    error: str | None,
) -> None:
    record = AuditRecord(
        surface="local",
        subject=LOCAL_SUBJECT,
        subject_ok=True,
        tool=tool,
        params_sha256=hash_params(dict(params) if params is not None else None),
        result_count=result_count,
        latency_ms=latency_ms,
        error=error,
    )
    try:
        audit.record(record)
    except AuditWriteError:
        logger.warning("mcp.local_audit_write_failed", tool=tool, exc_info=True)


def call_tool(
    audit: AuditSink,
    *,
    tool: str,
    params: Mapping[str, Any] | None,
    handler: Callable[[], dict[str, Any]],
) -> ToolCallResult:
    """Run `handler` (a zero-argument closure over the already-parsed
    params — the caller decides how those get there), catching
    `RetrievalError` into the SPEC §10.1 error model and anything else
    into `INTERNAL`. Always writes exactly one audit row."""
    started = monotonic()
    try:
        payload = handler()
    except RetrievalError as exc:
        latency_ms = int((monotonic() - started) * 1000)
        _record(audit, tool=tool, params=params, result_count=None, latency_ms=latency_ms, error=exc.code)
        return ToolCallResult(payload=None, error_code=exc.code, error_message=str(exc))
    except Exception:
        latency_ms = int((monotonic() - started) * 1000)
        logger.error("mcp.tool_internal_error", tool=tool, exc_info=True)
        _record(
            audit, tool=tool, params=params, result_count=None, latency_ms=latency_ms, error="INTERNAL"
        )
        return ToolCallResult(
            payload=None, error_code="INTERNAL", error_message="internal error"
        )

    latency_ms = int((monotonic() - started) * 1000)
    _record(
        audit,
        tool=tool,
        params=params,
        result_count=_result_count(payload),
        latency_ms=latency_ms,
        error=None,
    )
    return ToolCallResult(payload=payload, error_code=None, error_message=None)


__all__ = ["LOCAL_SUBJECT", "ToolCallResult", "call_tool"]
