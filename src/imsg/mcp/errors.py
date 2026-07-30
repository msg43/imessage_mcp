"""Exceptions raised by the public MCP security boundary.

Internal signal types (`TokenInvalidError`, `IntrospectionUnavailableError`)
deliberately do NOT derive from :class:`imsg.errors.ImsgError`: they are
control-flow signals between the introspector and the gate, never
operator-facing messages, and they must never carry a token, a subject,
or any response body in their message.
"""

from __future__ import annotations

from imsg.errors import ImsgError


class PublicSurfaceStartupError(ImsgError):
    """The public MCP surface refused to start (SPEC §10.4, hard requirement 4).

    Raised when the pinned ``owner_subject`` is missing, unresolvable, or
    not a plausible numeric Google ``sub``, or when the OAuth client id is
    missing. Startup refusal is the fail-closed alternative to ever
    running an allow-all surface.
    """


class TokenInvalidError(Exception):
    """The presented token is verifiably unacceptable (HTTP 401).

    Covers: Google tokeninfo rejected it (4xx), it is expired, or the
    introspection response lacks the claims required to prove identity.
    The exception message must never include the token or any claim value.
    """


class IntrospectionUnavailableError(Exception):
    """Token validity could not be established (HTTP 503, fail closed).

    Covers: network failure reaching tokeninfo, upstream 5xx, oversized or
    structurally malformed responses. SPEC §10.4 item 4: never allow on
    introspection failure. The message must never include the token.
    """


class AuditWriteError(ImsgError):
    """An audit row could not be written.

    The audit log is load-bearing for AT-1 (SPEC §12): an unauditable
    request is denied (503) rather than served, on both the accept and
    reject paths.
    """


__all__ = [
    "AuditWriteError",
    "IntrospectionUnavailableError",
    "PublicSurfaceStartupError",
    "TokenInvalidError",
]
