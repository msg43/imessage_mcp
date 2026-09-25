"""Operator-facing plumbing for `imsg mcp public --probe` (SPEC §12 AT-1).

`imsg.mcp.probe.run_auth_probe` is the test itself and is fully tested.
This module is everything between an operator at a terminal and that
function: where the two bearer tokens come from, what is refused before
anything can reach the network, and how a verdict is rendered so that
the decision it gates — *does a personal corpus become reachable from
the internet* — cannot be misread.

--------------------------------------------------------------------
Token handling
--------------------------------------------------------------------

AT-1 step 5 says the two tokens are "obtained via browser OAuth, never
written to disk". The naive command line —
``imsg mcp public --probe --owner-token ya29.…`` — violates that twice
over: the token lands in the shell's history file, and `argv` is
world-readable through `ps -ww` for the whole life of the process, on a
host that by design has a public tunnel attached to it.

So this command accepts **references, not values**, in exactly the form
the rest of the project already uses for secrets
(`imsg.config.secrets.SecretRef`): ``keychain:<item>``, ``env:<VAR>`` or
``file:<absolute path>``.
A literal is rejected by `SecretRef.parse` itself, which is the point of
reusing that type rather than inventing a parallel convention — the
refusal is structural, not a rule an operator is trusted to follow. What
appears in `ps` and in history is the *name of a Keychain item*; the
token is resolved in-process at the point of use, is never logged, never
printed, never written to disk, and appears in no exception message here
(a refusal names the reference, never what it resolved to).

The Keychain is the recommended side of that choice. An environment
variable is visible to anything that can read the process environment
and tends to get exported in a shell rc file; a Keychain item can be
created without the value ever entering a command line, because
``security add-generic-password`` prompts for it when ``-w`` is given no
argument. :data:`TOKEN_SETUP_HINT` is what the refusal prints.

--------------------------------------------------------------------
Reaching the network
--------------------------------------------------------------------

A real probe run *must* introspect both tokens against Google — that is
what makes it a test of the live gate rather than a test of a mock. The
guarantee this module provides is narrower and checkable: **no network
call is reachable until every precondition has been satisfied.**
:func:`check_probe_preconditions` is a pure function over config and two
reference strings, it runs to completion before the audit sink or the
auth gate is constructed, and the CLI builds neither on any refusal
path. Missing tokens, malformed tokens, identical tokens and missing
OAuth configuration therefore all fail with the
`GoogleTokeninfoIntrospector` never instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.config.secrets import SecretRef
from imsg.errors import ImsgError, SecretResolutionError
from imsg.mcp.probe import ProbeReport, ProbeVerdict

if TYPE_CHECKING:
    from imsg.config.schema import Config

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INVALID = 2
EXIT_CONFIG = 78
"""`EX_CONFIG`, the same code the mount gate uses (SPEC §5.4). A
configuration refusal is deliberately *not* folded into `EXIT_INVALID`:
"the probe ran and proved nothing" and "the probe never ran" lead to
different operator actions, and a wrapper script that treats them alike
would report a green AT-1 for a command that never opened a socket."""

MIN_TOKEN_LENGTH = 20
"""A heuristic floor, and labelled as one. Google OAuth access tokens
run to a few hundred characters; this only has to catch the values a
hurried operator actually stores — a placeholder, `changeme`, an
accidentally-empty Keychain item. It is not a format check, because
access tokens are opaque by specification and anything stricter would
start rejecting valid ones the day Google changes their shape."""

TOKEN_SETUP_HINT = (
    "Store each token in the Keychain without it entering your shell history:\n"
    "    security add-generic-password -a \"$USER\" -s imsgindex-at1-owner -w\n"
    "    security add-generic-password -a \"$USER\" -s imsgindex-at1-nonowner -w\n"
    "(-w with no value makes 'security' prompt for it.) Then:\n"
    "    imsg mcp public --probe \\\n"
    "        --owner-token-ref keychain:imsgindex-at1-owner \\\n"
    "        --foreign-token-ref keychain:imsgindex-at1-nonowner\n"
    "Delete both items afterwards: 'security delete-generic-password -s <item>'."
)


class ProbeConfigurationError(ImsgError):
    """A precondition for running AT-1 is missing or malformed.

    Raised only from :func:`check_probe_preconditions`, i.e. strictly
    before any gate, audit sink or introspector exists — so a run that
    fails this way cannot have contacted anything.
    """


@dataclass(frozen=True, slots=True)
class ProbeTokens:
    """Two resolved bearer tokens and the references they came from.

    `__repr__` is overridden: a dataclass repr of this would put both
    tokens into any traceback that happened to include a frame holding
    one, which is precisely the leak the reference convention exists to
    prevent.
    """

    owner: str
    foreign: str
    owner_ref: str
    foreign_ref: str

    def __repr__(self) -> str:
        return f"ProbeTokens(owner_ref={self.owner_ref!r}, foreign_ref={self.foreign_ref!r})"


def _parse_ref(raw: str | None, *, flag: str, role: str) -> SecretRef:
    if raw is None or not raw.strip():
        raise ProbeConfigurationError(
            f"AT-1 needs a {role} bearer token and none was given: pass {flag} "
            f"with a 'keychain:<item>', 'env:<VAR>' or 'file:<absolute path>' "
            f"reference.\n\n{TOKEN_SETUP_HINT}"
        )
    try:
        return SecretRef.parse(raw)
    except ValueError as exc:
        raise ProbeConfigurationError(
            f"{flag} must be a secret *reference*, not a token: {exc}\n\n"
            f"A token passed on the command line lands in your shell history and "
            f"is visible to every process on this host via 'ps -ww'.\n\n"
            f"{TOKEN_SETUP_HINT}"
        ) from exc


def _resolve_ref(ref: SecretRef, *, flag: str, role: str) -> str:
    try:
        return ref.resolve()
    except SecretResolutionError as exc:
        raise ProbeConfigurationError(
            f"the {role} token reference {ref.raw!r} ({flag}) could not be "
            f"resolved: {exc}\n\n{TOKEN_SETUP_HINT}"
        ) from exc


def validate_token_shape(token: str, *, ref: str, role: str) -> None:
    """Reject values that cannot be a bearer token, without echoing them.

    Every message names the *reference* and describes the defect; none
    of them contains any part of the resolved value, because a refusal
    printed to a terminal is the one place a carefully-protected secret
    would end up in a scrollback buffer.
    """
    if not token:
        raise ProbeConfigurationError(
            f"the {role} token reference {ref!r} resolved to an empty value — "
            f"the Keychain item or environment variable exists but holds nothing"
        )
    if token != token.strip():
        raise ProbeConfigurationError(
            f"the {role} token from {ref!r} has leading or trailing whitespace. "
            f"This is almost always a trailing newline from a file redirect; "
            f"store the token with no surrounding whitespace."
        )
    # Before the generic whitespace check, which would otherwise shadow it:
    # "you stored the whole header" is a more actionable diagnosis than
    # "there is a space in this", and it is the commoner mistake.
    if token.lower().startswith("bearer "):
        raise ProbeConfigurationError(
            f"the {role} token from {ref!r} starts with 'Bearer ' — store the "
            f"token alone; this command adds the scheme itself."
        )
    if any(c.isspace() for c in token):
        raise ProbeConfigurationError(
            f"the {role} token from {ref!r} contains whitespace, which no bearer "
            f"token does. If you stored a whole 'Authorization: Bearer …' header, "
            f"store only the token itself."
        )
    if not token.isprintable() or not token.isascii():
        raise ProbeConfigurationError(
            f"the {role} token from {ref!r} contains non-ASCII or non-printable "
            f"characters, which no bearer token does — the stored value is "
            f"probably binary or wrongly encoded."
        )
    if len(token) < MIN_TOKEN_LENGTH:
        raise ProbeConfigurationError(
            f"the {role} token from {ref!r} is {len(token)} characters, shorter "
            f"than any real OAuth access token (floor: {MIN_TOKEN_LENGTH}). This is "
            f"a placeholder, not a token."
        )


def check_probe_preconditions(
    config: Config, *, owner_token_ref: str | None, foreign_token_ref: str | None
) -> ProbeTokens:
    """Every refusal, before anything is constructed that could reach out.

    Order matters and is deliberate: the two token references are parsed
    (cheap, and catches the command-line-literal mistake before a
    Keychain prompt appears), then the OAuth configuration is checked,
    then the tokens are actually resolved and shape-checked. Returns the
    resolved pair; raises :class:`ProbeConfigurationError` otherwise.
    """
    owner_ref = _parse_ref(owner_token_ref, flag="--owner-token-ref", role="owner")
    foreign_ref = _parse_ref(foreign_token_ref, flag="--foreign-token-ref", role="non-owner")

    if owner_ref.raw == foreign_ref.raw:
        raise ProbeConfigurationError(
            f"--owner-token-ref and --foreign-token-ref are the same reference "
            f"({owner_ref.raw!r}). AT-1 is two-sided by design: one token must "
            f"belong to the owner and one to a different account, or the test "
            f"proves nothing."
        )

    oauth = config.mcp.public.oauth
    if oauth.owner_subject is None:
        raise ProbeConfigurationError(
            "mcp.public.oauth.owner_subject is not configured — AT-1 cannot check "
            "isolation without knowing which numeric Google 'sub' is the owner "
            "(SPEC §10.4; the public server refuses to start without it either)"
        )
    if oauth.client_id is None:
        raise ProbeConfigurationError(
            "mcp.public.oauth.client_id is not configured — the audience check "
            "requires it (SPEC §10.4)"
        )

    owner_token = _resolve_ref(owner_ref, flag="--owner-token-ref", role="owner")
    foreign_token = _resolve_ref(foreign_ref, flag="--foreign-token-ref", role="non-owner")
    validate_token_shape(owner_token, ref=owner_ref.raw, role="owner")
    validate_token_shape(foreign_token, ref=foreign_ref.raw, role="non-owner")

    if owner_token == foreign_token:
        raise ProbeConfigurationError(
            f"{owner_ref.raw!r} and {foreign_ref.raw!r} resolve to the same token. "
            f"AT-1 step 0 requires the owner and the non-owner to present distinct "
            f"'sub' values; one credential used twice fails the auth design itself, "
            f"not the probe."
        )

    return ProbeTokens(
        owner=owner_token,
        foreign=foreign_token,
        owner_ref=owner_ref.raw,
        foreign_ref=foreign_ref.raw,
    )


VERDICT_EXIT_CODES = {
    ProbeVerdict.PASS: EXIT_PASS,
    ProbeVerdict.FAIL: EXIT_FAIL,
    ProbeVerdict.INVALID: EXIT_INVALID,
}

_HEADLINE = {
    ProbeVerdict.PASS: "AT-1 PROBE: PASS",
    ProbeVerdict.FAIL: "AT-1 PROBE: FAIL",
    ProbeVerdict.INVALID: "AT-1 PROBE: INVALID",
}

_MEANING = {
    ProbeVerdict.PASS: (
        "The owner was accepted and the handler ran with the owner's own subject; "
        "the non-owner was rejected 401 and never reached a handler; the audit log "
        "shows the accepted owner row, the rejection, and zero accepted non-owner "
        "rows over its entire history."
    ),
    ProbeVerdict.FAIL: (
        "A non-owner subject got through, or the audit log contains an accepted "
        "non-owner request. This is a breach, not a tuning problem."
    ),
    ProbeVerdict.INVALID: (
        "The probe did not prove isolation. An empty result from the non-owner side "
        "is equally consistent with working isolation and with broken plumbing, so "
        "this is NOT a pass — nothing was demonstrated."
    ),
}

_NEXT_ACTION = {
    ProbeVerdict.PASS: (
        "Next (SPEC §12 AT-1 / D6): 'scope: full' is permitted. Record the ops "
        "approval in $DATA_ROOT/ops/auth-tests/ (date, scope, protocol version, "
        "auth config sha, verdict) before exposing the corpus. AT-1 must be re-run "
        "after ANY auth, scope, hostname or protocol change."
    ),
    ProbeVerdict.FAIL: (
        "Next (SPEC §12 AT-1 / D6): set 'scope: allowlist'. Moving to 'full' now "
        "requires a FRESH owner decision — a failed probe is not a tuning problem. "
        "Do not expose the corpus. Record this run in $DATA_ROOT/ops/auth-tests/."
    ),
    ProbeVerdict.INVALID: (
        "Next (SPEC §12 AT-1 / D6): treat this exactly as a probe weakness — set "
        "'scope: allowlist' and do not expose the corpus. Fix the plumbing named "
        "above and re-run. Record this run in $DATA_ROOT/ops/auth-tests/."
    ),
}

_RULE = "=" * 72


def format_probe_report(report: ProbeReport, *, scope: str) -> list[str]:
    """Render a verdict an operator cannot misread.

    The headline is on its own line between rules, the reasons are
    enumerated, and the next action is stated in terms of the D6
    condition rather than left as an inference — because the question
    this gates is whether a personal corpus becomes reachable from the
    internet, and "the output looked mostly fine" is not an acceptable
    way to answer it.
    """
    lines = [
        _RULE,
        _HEADLINE[report.verdict],
        _RULE,
        _MEANING[report.verdict],
        "",
    ]
    if report.reasons:
        lines.append(f"Evidence ({len(report.reasons)}):")
        lines.extend(f"  {i}. {reason}" for i, reason in enumerate(report.reasons, start=1))
        lines.append("")
    lines.append(f"Configured public scope at the time of this run: {scope}")
    lines.append("")
    lines.append(_NEXT_ACTION[report.verdict])
    lines.append(_RULE)
    return lines


def verdict_exit_code(verdict: ProbeVerdict) -> int:
    return VERDICT_EXIT_CODES[verdict]


__all__ = [
    "EXIT_CONFIG",
    "EXIT_FAIL",
    "EXIT_INVALID",
    "EXIT_PASS",
    "MIN_TOKEN_LENGTH",
    "TOKEN_SETUP_HINT",
    "ProbeConfigurationError",
    "ProbeTokens",
    "check_probe_preconditions",
    "format_probe_report",
    "validate_token_shape",
    "verdict_exit_code",
]
