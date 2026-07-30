"""AT-1 synthetic auth probe — the two-sided isolation test (SPEC §12 AT-1).

Two-sided by design, because a one-sided probe is worthless: "the other
subject got nothing" is equally consistent with *working isolation* and
with *the whole server being broken* (or the probe pointing at the wrong
endpoint). This probe therefore cannot pass on emptiness:

- **Positive control first** (AT-1 step 2): the owner's token must be
  ACCEPTED, the handler must actually run, the sentinel payload must come
  back, and an accepted audit row for the owner must exist. Any failure
  here makes the run INVALID — plumbing broken, *not* passed.
- **Isolation** (AT-1 step 3): the foreign token must be REJECTED with a
  401, and the handler must not have executed. A 503 (introspection
  down) is fail-closed but proves nothing → INVALID, not PASS.
- **Audit assertions** (AT-1 step 4): the rejection must be visible as a
  `subject_ok = false` row, and the standing invariant — zero accepted
  rows with a non-owner subject, over the *entire* log, not just the
  test window — must hold.

Verdict ordering: any evidence of a foreign subject getting through
(handler ran, request accepted, or an accepted-foreign audit row
anywhere) is FAIL and outranks INVALID — a breach is never masked by a
broken positive control.

This module performs step 0's server-side half with synthetic data only;
the full AT-1 run (real GE OAuth exchange, distinct human accounts,
protocol capture) is a Phase 6 operation recorded in the private ops
ledger, not here.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass

from imsg.mcp.audit import AuditReader, accepted_foreign_subjects
from imsg.mcp.auth import AuthorizedRequest, PublicAuthGate, ToolOutcome

PROBE_TOOL = "auth_probe_echo"


class ProbeVerdict(enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Outcome plus the specific evidence for it. `reasons` is empty on PASS."""

    verdict: ProbeVerdict
    reasons: tuple[str, ...]


def run_auth_probe(
    gate: PublicAuthGate,
    audit: AuditReader,
    *,
    owner_token: str,
    foreign_token: str,
    owner_subject: str,
) -> ProbeReport:
    """Run the two-sided synthetic probe against a live gate.

    `audit` must be the reader side of the same sink the gate writes to.
    Tokens are used in-memory only and never logged (AT-1 step 5: tokens
    are never written to disk).
    """
    invalid: list[str] = []
    breaches: list[str] = []

    if not owner_token or not foreign_token:
        return ProbeReport(
            ProbeVerdict.INVALID, ("probe misconfigured: a token is empty",)
        )
    if owner_token == foreign_token:
        return ProbeReport(
            ProbeVerdict.INVALID,
            ("probe misconfigured: owner and foreign tokens are identical",),
        )

    sentinel = f"PROBE-{uuid.uuid4()}"

    # --- Positive control (owner) — AT-1 step 2 ---------------------------
    owner_handler_ran = False

    def owner_handler(request: AuthorizedRequest) -> ToolOutcome[str]:
        nonlocal owner_handler_ran
        owner_handler_ran = True
        if request.subject != owner_subject:
            # The gate handed the handler a context whose subject is not
            # the pinned owner: that is a breach, not plumbing.
            breaches.append(
                "positive control ran with a non-owner subject in its context"
            )
        return ToolOutcome(payload=sentinel, result_count=1)

    owner_result = gate.dispatch(
        f"Bearer {owner_token}",
        tool=PROBE_TOOL,
        params={"sentinel": sentinel},
        handler=owner_handler,
    )
    if owner_result.rejection is not None:
        invalid.append(
            "positive control failed: owner token was rejected "
            f"(status {owner_result.rejection.status}) — an empty result "
            "cannot count as isolation (AT-1: invalid, not passed)"
        )
    if not owner_handler_ran:
        invalid.append("positive control failed: owner handler never executed")
    elif owner_result.rejection is None and owner_result.payload != sentinel:
        invalid.append("positive control failed: sentinel payload did not round-trip")

    # --- Isolation (foreign subject) — AT-1 step 3 ------------------------
    foreign_handler_ran = False

    def foreign_handler(request: AuthorizedRequest) -> ToolOutcome[str]:
        nonlocal foreign_handler_ran
        foreign_handler_ran = True
        return ToolOutcome(payload=sentinel, result_count=1)

    foreign_result = gate.dispatch(
        f"Bearer {foreign_token}",
        tool=PROBE_TOOL,
        params={"sentinel": sentinel},
        handler=foreign_handler,
    )
    if foreign_handler_ran:
        breaches.append("isolation breach: foreign token reached the tool handler")
    if foreign_result.rejection is None:
        breaches.append("isolation breach: foreign token was accepted")
    elif foreign_result.rejection.status == 503:
        invalid.append(
            "isolation not provable: foreign request answered 503 "
            "(introspection unavailable) — fail-closed, but proves nothing"
        )
    elif foreign_result.rejection.status != 401:
        invalid.append(
            "isolation not provable: foreign request was rejected with "
            f"status {foreign_result.rejection.status}, expected 401"
        )

    # --- Audit assertions — AT-1 step 4 -----------------------------------
    records = audit.snapshot()

    foreign_breach_rows = accepted_foreign_subjects(records, owner_subject)
    if foreign_breach_rows:
        breaches.append(
            "audit invariant violated: accepted public request(s) with a "
            f"non-owner subject ({len(foreign_breach_rows)} row(s))"
        )

    owner_accepted = [
        r
        for r in records
        if r.surface == "public"
        and r.subject_ok
        and r.subject == owner_subject
        and r.tool == PROBE_TOOL
    ]
    if not owner_accepted:
        invalid.append(
            "audit incomplete: no accepted audit row for the owner's probe "
            "request — the positive control is unverifiable"
        )

    rejected_rows = [
        r for r in records if r.surface == "public" and not r.subject_ok
    ]
    if foreign_result.rejection is not None and not rejected_rows:
        invalid.append(
            "audit incomplete: foreign request was rejected but no rejection "
            "row exists — an apparently empty answer without an audit "
            "rejection is not proof (AT-1 step 3)"
        )

    # --- Verdict ordering: breach > invalid > pass ------------------------
    if breaches:
        return ProbeReport(ProbeVerdict.FAIL, tuple(breaches + invalid))
    if invalid:
        return ProbeReport(ProbeVerdict.INVALID, tuple(invalid))
    return ProbeReport(ProbeVerdict.PASS, ())


__all__ = ["PROBE_TOOL", "ProbeReport", "ProbeVerdict", "run_auth_probe"]
