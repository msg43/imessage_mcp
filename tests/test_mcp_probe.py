"""AT-1 synthetic probe tests (SPEC §12): two-sided by construction.

The one property that matters most here: a probe run against a server
that is completely broken — rejects everyone, or was never wired to the
introspector — must come back INVALID, never PASS. "The other subject
saw nothing" is not evidence when the owner saw nothing too.

All identities and tokens fictional (D5).
"""

from __future__ import annotations

from collections.abc import Sequence

from imsg.mcp.audit import AuditRecord, MemoryAuditSink
from imsg.mcp.auth import PublicAuthGate, TokenIntrospection
from imsg.mcp.errors import IntrospectionUnavailableError, TokenInvalidError
from imsg.mcp.probe import PROBE_TOOL, ProbeVerdict, run_auth_probe

OWNER_SUB = "100000000000000000001"
FOREIGN_SUB = "200000000000000000002"
CLIENT_ID = "000000000000-fictional.apps.example"
OWNER_TOKEN = "fictional-owner-token"
FOREIGN_TOKEN = "fictional-foreign-token"


def claims(sub: str, *, aud: str = CLIENT_ID, expires_in: int = 3600) -> TokenIntrospection:
    return TokenIntrospection(
        subject=sub,
        audience=aud,
        authorized_party=None,
        scopes=frozenset({"openid", "email", "profile"}),
        expires_in_seconds=expires_in,
    )


class StubIntrospector:
    def __init__(self, outcomes: dict[str, TokenIntrospection | Exception]) -> None:
        self.outcomes = outcomes

    def introspect(self, token: str) -> TokenIntrospection:
        outcome = self.outcomes.get(token)
        if outcome is None:
            raise TokenInvalidError("unknown token")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_gate(
    outcomes: dict[str, TokenIntrospection | Exception],
    sink: MemoryAuditSink,
) -> PublicAuthGate:
    return PublicAuthGate(
        client_id=CLIENT_ID,
        owner_subject=OWNER_SUB,
        introspector=StubIntrospector(outcomes),
        audit=sink,
    )


def run(
    gate: PublicAuthGate,
    sink: MemoryAuditSink,
    *,
    owner_token: str = OWNER_TOKEN,
    foreign_token: str = FOREIGN_TOKEN,
) -> tuple[ProbeVerdict, tuple[str, ...]]:
    report = run_auth_probe(
        gate,
        sink,
        owner_token=owner_token,
        foreign_token=foreign_token,
        owner_subject=OWNER_SUB,
    )
    return report.verdict, report.reasons


# ---------------------------------------------------------------------------
# PASS: healthy gate — owner in, foreign out, audit complete
# ---------------------------------------------------------------------------


def test_probe_passes_on_a_healthy_gate() -> None:
    sink = MemoryAuditSink()
    gate = make_gate(
        {OWNER_TOKEN: claims(OWNER_SUB), FOREIGN_TOKEN: claims(FOREIGN_SUB)}, sink
    )
    verdict, reasons = run(gate, sink)
    assert verdict is ProbeVerdict.PASS
    assert reasons == ()
    # Both sides visible in the audit trail: one accepted owner row, one rejection.
    rows = sink.snapshot()
    assert any(r.subject_ok and r.subject == OWNER_SUB and r.tool == PROBE_TOOL for r in rows)
    assert any(not r.subject_ok and r.subject == FOREIGN_SUB for r in rows)


# ---------------------------------------------------------------------------
# INVALID: an empty result must never count as isolation
# ---------------------------------------------------------------------------


def test_probe_is_invalid_when_the_whole_server_rejects_everyone() -> None:
    """The trap AT-1 exists to close: a dead server rejects the non-owner too."""
    sink = MemoryAuditSink()
    gate = make_gate(
        {
            OWNER_TOKEN: TokenInvalidError("broken"),
            FOREIGN_TOKEN: TokenInvalidError("broken"),
        },
        sink,
    )
    verdict, reasons = run(gate, sink)
    assert verdict is ProbeVerdict.INVALID
    assert any("positive control failed" in r for r in reasons)


def test_probe_is_invalid_when_introspection_is_down() -> None:
    """503s are fail-closed but prove nothing about isolation."""
    sink = MemoryAuditSink()
    gate = make_gate(
        {
            OWNER_TOKEN: IntrospectionUnavailableError("down"),
            FOREIGN_TOKEN: IntrospectionUnavailableError("down"),
        },
        sink,
    )
    verdict, _ = run(gate, sink)
    assert verdict is ProbeVerdict.INVALID


def test_probe_is_invalid_when_tokens_are_missing_or_identical() -> None:
    sink = MemoryAuditSink()
    gate = make_gate({OWNER_TOKEN: claims(OWNER_SUB)}, sink)
    assert run(gate, sink, owner_token="")[0] is ProbeVerdict.INVALID
    assert run(gate, sink, foreign_token="")[0] is ProbeVerdict.INVALID
    assert (
        run(gate, sink, owner_token="same-tok", foreign_token="same-tok")[0]
        is ProbeVerdict.INVALID
    )


def test_probe_is_invalid_when_foreign_rejection_is_429_not_401() -> None:
    """A rate-limited foreign request is not a subject rejection."""
    sink = MemoryAuditSink()
    gate = PublicAuthGate(
        client_id=CLIENT_ID,
        owner_subject=OWNER_SUB,
        introspector=StubIntrospector(
            {OWNER_TOKEN: claims(OWNER_SUB), FOREIGN_TOKEN: claims(FOREIGN_SUB)}
        ),
        audit=sink,
        failure_budget_per_minute=1,
    )
    # Exhaust the failure budget so the foreign probe request answers 429.
    gate.authorize("Bearer some-other-garbage")
    verdict, reasons = run(gate, sink)
    assert verdict is ProbeVerdict.INVALID
    assert any("expected 401" in r for r in reasons)


class DroppingSink(MemoryAuditSink):
    """A sink that silently loses rejection rows — audit incompleteness."""

    def record(self, rec: AuditRecord) -> None:
        if rec.subject_ok:
            super().record(rec)


def test_probe_is_invalid_when_rejections_leave_no_audit_trail() -> None:
    """AT-1 step 3: an apparently empty answer without an audit rejection is not proof."""
    sink = DroppingSink()
    gate = make_gate(
        {OWNER_TOKEN: claims(OWNER_SUB), FOREIGN_TOKEN: claims(FOREIGN_SUB)}, sink
    )
    verdict, reasons = run(gate, sink)
    assert verdict is ProbeVerdict.INVALID
    assert any("no rejection row" in r for r in reasons)


# ---------------------------------------------------------------------------
# FAIL: any foreign subject getting through outranks everything
# ---------------------------------------------------------------------------


def test_probe_fails_when_platform_presents_shared_credentials() -> None:
    """The AT-1 rejection scenario (D6): distinct users arrive with the SAME
    subject — the platform cannot isolate users, so the foreign request is
    indistinguishable from the owner and gets through. The gate is doing
    exactly what it was told; the *design* is broken, and the probe must
    say FAIL (not tuning, per D6: launch scope becomes allowlist)."""
    sink = MemoryAuditSink()
    gate = make_gate(
        {
            OWNER_TOKEN: claims(OWNER_SUB),
            FOREIGN_TOKEN: claims(OWNER_SUB),  # same sub for a different human
        },
        sink,
    )
    verdict, reasons = run(gate, sink)
    assert verdict is ProbeVerdict.FAIL
    assert any("foreign token reached the tool handler" in r for r in reasons)


def test_probe_fails_on_historical_accepted_foreign_rows() -> None:
    """AT-1 step 4 is permanent: a breach row from ANY time fails the probe,
    even when live isolation currently holds."""
    sink = MemoryAuditSink()
    sink.record(
        AuditRecord(
            surface="public",
            subject="300000000000000000003",
            subject_ok=True,
            tool="search_messages",
            params_sha256=None,
            result_count=3,
            latency_ms=10,
            error=None,
        )
    )
    gate = make_gate(
        {OWNER_TOKEN: claims(OWNER_SUB), FOREIGN_TOKEN: claims(FOREIGN_SUB)}, sink
    )
    verdict, reasons = run(gate, sink)
    assert verdict is ProbeVerdict.FAIL
    assert any("audit invariant violated" in r for r in reasons)


def test_probe_fail_outranks_invalid() -> None:
    """A breach must never be masked by a broken positive control."""
    sink = MemoryAuditSink()
    gate = make_gate(
        {
            OWNER_TOKEN: TokenInvalidError("owner side broken"),
            FOREIGN_TOKEN: claims(OWNER_SUB),  # and foreign gets through
        },
        sink,
    )
    verdict, reasons = run(gate, sink)
    assert verdict is ProbeVerdict.FAIL
    assert any("foreign token" in r for r in reasons)
    assert any("positive control failed" in r for r in reasons)  # still reported


def test_probe_never_writes_tokens_into_audit(
) -> None:
    sink = MemoryAuditSink()
    gate = make_gate(
        {OWNER_TOKEN: claims(OWNER_SUB), FOREIGN_TOKEN: claims(FOREIGN_SUB)}, sink
    )
    run(gate, sink)
    rows: Sequence[AuditRecord] = sink.snapshot()
    for row in rows:
        for field in (row.subject, row.tool, row.params_sha256, row.error):
            if field is not None:
                assert OWNER_TOKEN not in field
                assert FOREIGN_TOKEN not in field
