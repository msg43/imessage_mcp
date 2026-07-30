"""Transport-layer obligations for the public MCP surface (SPEC §10.4).

`imsg.mcp.auth`'s own module docstring is explicit about the boundary
between what it owns and what it hands off: "Transport contract (the
parts that live outside this module, SPEC §10.4): the HTTP layer MUST
validate `Host` and any present `Origin` against config before dispatch
(403 on mismatch), reject duplicate Authorization headers rather than
joining them, attach `RESPONSE_CACHE_HEADERS` to every tool response,
and route every request through `PublicAuthGate.dispatch`." This module
implements the first two of those four as small, pure functions —
independently unit-testable without an ASGI server, a socket, or the
`mcp` SDK — so `imsg.mcp.tools.public_server` (the actual transport
wiring) can stay a thin, obviously-correct composition of them.

**Why raw ASGI header pairs, not a folded mapping**: Starlette's own
`Headers.get()` (and most HTTP libraries' header maps) silently return
only the *first* value when a header name repeats. That is exactly the
ambiguity a duplicate-header check exists to catch — a client and an
intermediary (or two code paths inside the same server) disagreeing
about *which* occurrence is authoritative is a request-smuggling-
adjacent primitive, not a protocol accident. `find_duplicate_header`
therefore takes the raw `(name, value)` pairs straight off the ASGI
`scope["headers"]` list, before anything has folded them.
"""

from __future__ import annotations

from collections.abc import Sequence

RawHeaders = Sequence[tuple[bytes, bytes]]
"""The shape of `scope["headers"]` in the ASGI spec: a list of raw
`(name, value)` byte pairs, lowercase name by ASGI convention but not
guaranteed here — this module lowercases defensively rather than
trusting that."""

GUARDED_HEADER_NAMES: frozenset[str] = frozenset({"authorization", "host", "origin"})
"""Headers whose presence more than once must be rejected outright,
never resolved by picking the first or the last occurrence.
`Authorization` is named explicitly by `imsg.mcp.auth`'s docstring;
`Host` and `Origin` are guarded for the identical reason — the whole
point of validating them is to close a DNS-rebinding/smuggling class of
ambiguity, and a duplicate is exactly that ambiguity, not a detail the
check can safely ignore."""


def find_duplicate_header(
    raw_headers: RawHeaders, names: frozenset[str] = GUARDED_HEADER_NAMES
) -> str | None:
    """Return the lowercase name of the first guarded header that
    appears more than once in `raw_headers`, or `None` if none does.

    Undecodable header bytes (not valid latin-1, which practically
    never happens for HTTP header field names) are treated as an
    unrecognized name rather than raising — this function's only job
    is to *find a problem*, never to become one itself on malformed
    input; downstream validation will reject the request anyway once
    a required header comes back empty/missing.
    """
    seen: set[str] = set()
    for raw_name, _ in raw_headers:
        try:
            name = raw_name.decode("latin-1").lower()
        except UnicodeDecodeError:  # pragma: no cover - ASCII-range bytes always decode
            continue
        if name not in names:
            continue
        if name in seen:
            return name
        seen.add(name)
    return None


def validate_host(host: str | None, allowed_hosts: Sequence[str]) -> bool:
    """SPEC §10.4: "Validate `Host` ... against `allowed_hosts`."

    Exact, case-insensitive match against `mcp.public.allowed_hosts`
    (SPEC §6) only — no wildcard/suffix matching. `allowed_hosts` is a
    short, owner-curated list of exact `host[:port]` values; wildcard
    support would widen precisely the DNS-rebinding surface this check
    exists to close. A missing `Host` header is always rejected: every
    HTTP/1.1+ request carries one, so its absence is itself an anomaly
    worth refusing rather than a case to special-case as "allow."
    """
    if not host:
        return False
    lowered = host.lower()
    return any(lowered == allowed.lower() for allowed in allowed_hosts)


def validate_origin(origin: str | None, allowed_origins: Sequence[str]) -> bool:
    """SPEC §10.4: "any present Origin" — Origin is legitimately absent
    on non-browser, server-to-server requests (Gemini Enterprise's own
    calls chief among them), so absence passes. A *present* Origin must
    match `mcp.public.allowed_origins` exactly; unlike `Host`, Origin
    values are compared case-sensitively on the path/host components in
    general, but in practice every configured origin here is a bare
    `scheme://host[:port]` — an exact string match is the correct and
    simplest rule (MCP spec MUST, DNS-rebinding defense, D6).
    """
    if origin is None:
        return True
    return origin in allowed_origins


__all__ = [
    "GUARDED_HEADER_NAMES",
    "RawHeaders",
    "find_duplicate_header",
    "validate_host",
    "validate_origin",
]
