"""Unit tests for the local-surface tool-call dispatch (SPEC §10.1,
§10.3) — no database required, uses `imsg.mcp.audit.MemoryAuditSink`
(the same in-memory sink the AT-1 synthetic probe's own tests use)."""

from __future__ import annotations

from imsg.mcp.audit import MemoryAuditSink
from imsg.mcp.errors import AuditWriteError
from imsg.mcp.tools.dispatch import LOCAL_SUBJECT, call_tool
from imsg.retrieval.errors import NotFoundError, PersonAmbiguousError, PersonCandidate


def test_success_writes_an_allowed_local_audit_row() -> None:
    audit = MemoryAuditSink()
    result = call_tool(
        audit, tool="search_messages", params={"query": "hi"}, handler=lambda: {"results": [1, 2]}
    )
    assert not result.is_error
    assert result.payload == {"results": [1, 2]}

    rows = audit.snapshot()
    assert len(rows) == 1
    row = rows[0]
    assert row.surface == "local"
    assert row.subject == LOCAL_SUBJECT
    assert row.subject_ok is True
    assert row.tool == "search_messages"
    assert row.result_count == 2  # len(payload["results"])
    assert row.error is None


def test_retrieval_error_maps_to_its_own_code_and_is_audited() -> None:
    audit = MemoryAuditSink()

    def handler() -> dict[str, object]:
        raise NotFoundError("no such thing")

    result = call_tool(audit, tool="get_attachment_text", params={}, handler=handler)
    assert result.is_error
    assert result.error_code == "NOT_FOUND"
    assert result.error_message == "no such thing"

    rows = audit.snapshot()
    assert rows[0].error == "NOT_FOUND"
    assert rows[0].subject_ok is True  # local surface: always "authorized", never denied


def test_person_ambiguous_carries_its_own_code() -> None:
    audit = MemoryAuditSink()

    def handler() -> dict[str, object]:
        raise PersonAmbiguousError(
            "ali", (PersonCandidate(short_name="alice", display_name="Alice Example"),)
        )

    result = call_tool(audit, tool="search_messages", params={}, handler=handler)
    assert result.error_code == "PERSON_AMBIGUOUS"


def test_unexpected_exception_becomes_internal_never_leaks_details() -> None:
    audit = MemoryAuditSink()

    def handler() -> dict[str, object]:
        raise RuntimeError("some internal detail with a /path/or/sql LEAK")

    result = call_tool(audit, tool="search_messages", params={}, handler=handler)
    assert result.error_code == "INTERNAL"
    assert result.error_message is not None
    assert "LEAK" not in result.error_message
    assert "/path/or/sql" not in result.error_message

    rows = audit.snapshot()
    assert rows[0].error == "INTERNAL"


def test_params_are_never_stored_raw_only_a_hash() -> None:
    audit = MemoryAuditSink()
    call_tool(
        audit,
        tool="search_messages",
        params={"query": "very private corpus content"},
        handler=lambda: {"results": []},
    )
    row = audit.snapshot()[0]
    assert row.params_sha256 is not None
    assert "very private corpus content" not in row.params_sha256


def test_audit_write_failure_does_not_fail_a_successful_local_call() -> None:
    """Judgment call documented in `imsg.mcp.tools.dispatch`: unlike the
    public gate (which withholds the payload on an audit-write
    failure), the local surface's access boundary is SSH + tailnet
    reachability, not the audit trail — an audit-write failure is
    logged, not fatal, on the local surface."""

    class _FailingAudit:
        def record(self, rec: object) -> None:
            raise AuditWriteError("disk full")

    result = call_tool(
        _FailingAudit(), tool="search_messages", params={}, handler=lambda: {"results": []}
    )
    assert not result.is_error
    assert result.payload == {"results": []}


def test_result_count_falls_back_to_none_for_shapes_without_a_list() -> None:
    audit = MemoryAuditSink()
    call_tool(
        audit, tool="check_permissions", params={}, handler=lambda: {"mount_ok": True}
    )
    row = audit.snapshot()[0]
    assert row.result_count is None
