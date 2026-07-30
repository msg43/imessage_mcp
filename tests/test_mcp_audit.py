"""Audit-sink tests: params hashing, closed error codes, AT-1's standing SQL.

All values fictional (D5). The Postgres sink is exercised against a fake
connection — no network, no live database.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import psycopg
import pytest

from imsg.mcp.audit import (
    ACCEPTED_FOREIGN_SQL,
    AuditRecord,
    MemoryAuditSink,
    PostgresAuditSink,
    accepted_foreign_subjects,
    hash_params,
    sanitize_error_code,
    truncate_subject,
)
from imsg.mcp.errors import AuditWriteError

OWNER_SUB = "100000000000000000001"


def record(**overrides: Any) -> AuditRecord:
    base: dict[str, Any] = {
        "surface": "public",
        "subject": OWNER_SUB,
        "subject_ok": True,
        "tool": "search_messages",
        "params_sha256": "ab" * 32,
        "result_count": 1,
        "latency_ms": 5,
        "error": None,
    }
    base.update(overrides)
    return AuditRecord(**base)


# ---------------------------------------------------------------------------
# hash_params — the only params-derived value that may be stored
# ---------------------------------------------------------------------------


def test_hash_params_is_order_independent() -> None:
    assert hash_params({"a": 1, "b": 2}) == hash_params({"b": 2, "a": 1})


def test_hash_params_distinguishes_values() -> None:
    assert hash_params({"a": 1}) != hash_params({"a": 2})


def test_hash_params_none_is_none() -> None:
    assert hash_params(None) is None


def test_hash_params_never_returns_plaintext() -> None:
    digest = hash_params({"query": "very private fictional text"})
    assert digest is not None
    assert "private" not in digest
    assert len(digest) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# Closed error-code set — free text (exception messages, paths) never lands
# ---------------------------------------------------------------------------


def test_sanitize_keeps_known_codes() -> None:
    assert sanitize_error_code("RATE_LIMITED") == "RATE_LIMITED"
    assert sanitize_error_code("UNAUTHORIZED") == "UNAUTHORIZED"
    assert sanitize_error_code(None) is None


@pytest.mark.parametrize(
    "junk",
    [
        "Traceback (most recent call last): ...",
        "/fictional/path/leaked.txt",
        "SELECT * FROM message",
        "rate_limited",  # case matters — the set is closed, not fuzzy
    ],
)
def test_sanitize_collapses_unknown_text_to_internal(junk: str) -> None:
    assert sanitize_error_code(junk) == "INTERNAL"


def test_memory_sink_sanitizes_on_write() -> None:
    sink = MemoryAuditSink()
    sink.record(record(error="some free text with /a/path", subject="s" * 1000))
    (row,) = sink.snapshot()
    assert row.error == "INTERNAL"
    assert row.subject is not None and len(row.subject) == 256


def test_truncate_subject_caps_length() -> None:
    assert truncate_subject(None) is None
    assert truncate_subject("abc") == "abc"
    long = "9" * 1000
    truncated = truncate_subject(long)
    assert truncated is not None and len(truncated) == 256


# ---------------------------------------------------------------------------
# AT-1 standing invariant helpers
# ---------------------------------------------------------------------------


def test_accepted_foreign_subjects_catches_the_breach_shape() -> None:
    rows = [
        record(),  # owner accepted — fine
        record(subject_ok=False, subject="300000000000000000003"),  # rejected — fine
        record(surface="local", subject="local"),  # local surface — out of scope
        record(subject="300000000000000000003"),  # accepted foreign — THE breach
    ]
    breaches = accepted_foreign_subjects(rows, OWNER_SUB)
    assert len(breaches) == 1
    assert breaches[0].subject == "300000000000000000003"


def test_accepted_foreign_sql_matches_spec_shape() -> None:
    """AT-1 step 4's query, kept in one place so probe and ops cannot drift."""
    assert "surface = 'public'" in ACCEPTED_FOREIGN_SQL
    assert "subject_ok" in ACCEPTED_FOREIGN_SQL
    assert "subject <> %(owner_subject)s" in ACCEPTED_FOREIGN_SQL


# ---------------------------------------------------------------------------
# Postgres sink — parameterized insert, sanitized values, closed failure mode
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        if self.fail:
            raise psycopg.OperationalError("fictional connection lost")
        self.executed.append((sql, params))

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeConnection:
    def __init__(self, fail: bool = False) -> None:
        self.cursor_obj = FakeCursor(fail=fail)
        self.committed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_postgres_sink_inserts_sanitized_parameterized_row() -> None:
    conn = FakeConnection()
    sink = PostgresAuditSink(cast(Callable[[], psycopg.Connection], lambda: conn))
    sink.record(record(error="free text that must not land"))
    (sql, params) = conn.cursor_obj.executed[0]
    assert sql.startswith("INSERT INTO mcp_audit")
    assert "%(subject)s" in sql  # parameterized, never interpolated
    assert params["error"] == "INTERNAL"
    assert params["subject"] == OWNER_SUB
    assert conn.committed


def test_postgres_sink_wraps_driver_errors_without_leaking_them() -> None:
    conn = FakeConnection(fail=True)
    sink = PostgresAuditSink(cast(Callable[[], psycopg.Connection], lambda: conn))
    with pytest.raises(AuditWriteError) as excinfo:
        sink.record(record())
    # The wrapped message must not carry driver text (which can quote SQL
    # and parameter values, i.e. subjects).
    assert "fictional connection lost" not in str(excinfo.value)
