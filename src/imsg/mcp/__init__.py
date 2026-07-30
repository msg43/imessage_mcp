"""The public MCP surface's security boundary (SPEC §10.4; hard requirement 4).

This package is the ONLY access control on the public surface. The
transport layer (built separately) consumes it through exactly one path:

    gate = build_public_gate(config.mcp.public, audit=PostgresAuditSink(...))
    ...
    result = gate.dispatch(authorization_header, tool=..., params=..., handler=...)

`dispatch` validates the bearer token via live Google tokeninfo
introspection (opaque tokens — never local signature verification),
pins the subject to the configured owner, rate limits, and writes the
`mcp_audit` row, all before/around the handler. There is no config flag,
constructor parameter, or code path that skips any step; startup refuses
when the pinned subject is unresolvable.

The transport additionally owes (SPEC §10.4, outside this package):
Host/Origin validation against config (403 on mismatch), rejection of
duplicate Authorization headers, protocol-revision negotiation, and
attaching `RESPONSE_CACHE_HEADERS` to every tool response.

AT-1's synthetic two-sided auth probe lives in `imsg.mcp.probe`.
"""

from imsg.mcp.audit import (
    ACCEPTED_FOREIGN_SQL,
    AuditReader,
    AuditRecord,
    AuditSink,
    MemoryAuditSink,
    PostgresAuditSink,
    accepted_foreign_subjects,
    hash_params,
)
from imsg.mcp.auth import (
    GOOGLE_TOKENINFO_URL,
    RESPONSE_CACHE_HEADERS,
    AuthorizedRequest,
    DispatchResult,
    GoogleTokeninfoIntrospector,
    PublicAuthGate,
    Rejection,
    TokenIntrospection,
    TokenIntrospector,
    ToolOutcome,
    build_public_gate,
    parse_tokeninfo_response,
    resource_metadata_url_for,
)
from imsg.mcp.errors import (
    AuditWriteError,
    IntrospectionUnavailableError,
    PublicSurfaceStartupError,
    TokenInvalidError,
)
from imsg.mcp.probe import PROBE_TOOL, ProbeReport, ProbeVerdict, run_auth_probe
from imsg.mcp.ratelimit import SlidingWindowLimiter

__all__ = [
    "ACCEPTED_FOREIGN_SQL",
    "GOOGLE_TOKENINFO_URL",
    "PROBE_TOOL",
    "RESPONSE_CACHE_HEADERS",
    "AuditReader",
    "AuditRecord",
    "AuditSink",
    "AuditWriteError",
    "AuthorizedRequest",
    "DispatchResult",
    "GoogleTokeninfoIntrospector",
    "IntrospectionUnavailableError",
    "MemoryAuditSink",
    "PostgresAuditSink",
    "ProbeReport",
    "ProbeVerdict",
    "PublicAuthGate",
    "PublicSurfaceStartupError",
    "Rejection",
    "SlidingWindowLimiter",
    "TokenIntrospection",
    "TokenIntrospector",
    "TokenInvalidError",
    "ToolOutcome",
    "accepted_foreign_subjects",
    "build_public_gate",
    "hash_params",
    "parse_tokeninfo_response",
    "resource_metadata_url_for",
    "run_auth_probe",
]
