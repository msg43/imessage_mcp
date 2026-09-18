"""The public MCP surface (SPEC §10.4): StreamableHTTP, behind
`imsg.mcp.auth.PublicAuthGate`, registering exactly
`imsg.mcp.tools.schemas.PUBLIC_TOOL_DEFINITIONS` (SPEC §10.2's public
subset — no `include_handles`, no `find_similar_attachments`, no
`mark_relevant`, no `check_permissions`).

**Two gates, one `PublicAuthGate`, by design.** `imsg.mcp.auth`'s own
module docstring splits the public surface's obligations in two: the
gate module owns subject validation, rate limiting, the tokeninfo
circuit breaker, and the audit trail; "the parts that live outside this
module" — Host/Origin validation, duplicate-header rejection, cache
headers, and "route every request through `gate.dispatch`" — are
this module's job. Concretely, that becomes two layers wrapping the
same `PublicAuthGate` instance:

1. :class:`TransportGuardASGIApp` — a plain ASGI middleware in front of
   *everything*, including MCP protocol bookkeeping that never reaches a
   tool handler at all (`initialize`, `tools/list`, `ping`). It rejects
   malformed transport framing (duplicate Authorization/Host/Origin
   headers, an invalid Host, a present-and-invalid Origin) and calls
   :meth:`~imsg.mcp.auth.PublicAuthGate.authorize` on *every* HTTP
   request before the MCP session manager ever sees it — hard
   requirement 4 says the validation middleware is "unconditionally in
   the request path of the public transport," not "only for tool
   calls." It is also the only place an HTTP-level 401/403/429/503 with
   a correct status code and `WWW-Authenticate` challenge can be
   produced: by the time a JSON-RPC message reaches a low-level
   `Server` callback, the outer HTTP response has already committed to
   a 200 envelope for that exchange.
2. :meth:`PublicMcpServer.on_call_tool` — the *only* place a tool
   handler function is ever invoked, and it is invoked exclusively
   through :meth:`~imsg.mcp.auth.PublicAuthGate.dispatch`, per that
   method's own docstring: "there is no other supported path to a tool
   handler." This is what produces the per-tool `mcp_audit` row SPEC
   §10.4 requires, and what turns a `RetrievalError` into SPEC §10.1
   tool-error content instead of a stack trace.

Both layers share the same gate, so the tool-call-time re-validation in
layer 2 is a cache hit against layer 1's check, not a second network
round trip to Google — see `imsg.mcp.auth`'s verdict cache. Tests in
`tests/test_mcp_public_server.py` assert directly that no path from an
unauthenticated or malformed request reaches a tool handler.

**Serving before the models are warm.** The models load in the
background (`imsg.retrieval.background_warm_up`), started by
`imsg mcp public` at process start; the transport binds and answers
without waiting for them. Only the four retrieval tools are gated on the
warm-up: such a call waits for it — with `anyio.sleep`, so the event
loop keeps answering everything else — for at most
`TOOL_CALL_WARM_UP_WAIT_SECONDS`, then gets a retryable `WARMING_UP`
tool error carrying an estimate; a warm-up that failed turns every
retrieval call into `WARM_UP_FAILED` naming the cause, so a process that
came back up with half its models loaded says so instead of serving.
Both are ordinary tool errors (`isError: true` with the machine code on
the first line), and both keep their own code in `mcp_audit` rather than
collapsing to `INTERNAL` — `imsg.mcp.audit.ALLOWED_ERROR_CODES` lists
them. Restarts are the normal case here, not the exception: the launchd
agent is `KeepAlive`, so every crash is followed by another cold load.

That the warm-up wait sits *outside* `gate.dispatch` — the one thing in
this module that is ordered differently from the security layers above —
is deliberate and costs nothing: `TransportGuardASGIApp` has already
authorized this very HTTP request before the session manager could route
it here, so no unauthenticated caller can reach the wait, let alone park
a request on it. Putting the wait inside `dispatch` would mean blocking
the event-loop thread (dispatch is synchronous), which is the one thing
a bounded wait must not do.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
import jsonschema
import mcp.types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from imsg.mcp.auth import (
    RESPONSE_CACHE_HEADERS,
    AuthorizedRequest,
    PublicAuthGate,
    Rejection,
    ToolOutcome,
)
from imsg.mcp.errors import PublicSurfaceStartupError
from imsg.mcp.tools import handlers
from imsg.mcp.tools.schemas import (
    PUBLIC_TOOL_DEFINITIONS,
    PUBLIC_TOOL_DEFINITIONS_BY_NAME,
    ToolDefinition,
)
from imsg.mcp.transport import RawHeaders, find_duplicate_header, validate_host, validate_origin
from imsg.retrieval.access import AccessContext
from imsg.retrieval.access import Scope as AccessScope
from imsg.retrieval.errors import RetrievalError

if TYPE_CHECKING:
    from starlette.routing import Route

    from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpStatus
    from imsg.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

SERVER_NAME = "imsg-public"
DEFAULT_STREAMABLE_HTTP_PATH = "/mcp"

TOOL_CALL_WARM_UP_WAIT_SECONDS = 20.0
"""How long a retrieval call that arrives during warm-up waits for the
models before answering `WARMING_UP`.

Much shorter than the local surface's 90 s
(`imsg.mcp.tools.local_server.TOOL_CALL_WARM_UP_WAIT_SECONDS`), because
a public call is an HTTP request held open across a Cloudflare tunnel
rather than a stdio exchange with no clock on it:

- **A documented hard ceiling at 125 s.** Cloudflare's Proxy Read Timeout
  — how long the edge waits for the origin's response before returning
  524 — is 125 s, and is configurable only on Enterprise zones
  (developers.cloudflare.com/fundamentals/reference/connection-limits/,
  read 2026-09-18). No `cloudflared` `originRequest` setting moves it:
  `connectTimeout`/`tlsTimeout` bound connection setup and
  `keepAliveTimeout`/`tcpKeepAlive` bound idle connections; none of them
  is a response-read timeout. Past 125 s the client gets an HTML error
  page instead of JSON-RPC, i.e. a transport failure rather than a tool
  error it can act on — strictly worse than answering early.
- **An undocumented client timeout.** Whether Gemini Enterprise gives up
  sooner than the edge does is not published: none of the custom-MCP,
  workflow-builder, or quota pages names a per-tool-call timeout (read
  2026-09-18). With no number to design against, the bound has to be
  small enough that no plausible one binds first.
- **Room left for the answer.** 20 s of waiting plus a warm query
  (2.1-2.3 s on the M2 Ultra host, 2026-09-17) leaves about 100 s of the
  edge's budget unused, so a call that *does* catch the end of a warm-up
  still returns inside it comfortably.

Waiting at all is what makes a request arriving in the last seconds of a
warm-up succeed instead of bouncing; waiting longer than this trades
that small win for the risk of no structured answer at all."""

WARM_UP_POLL_SECONDS = 0.1
"""How often a waiting tool call re-checks the warm-up."""

_RETRIEVAL_HANDLERS: dict[
    str, Callable[[RetrievalService, AccessContext, dict[str, Any]], dict[str, Any]]
] = {
    "search_messages": handlers.search_messages,
    "get_conversation": handlers.get_conversation,
    "list_people": handlers.list_people,
    "get_attachment_text": handlers.get_attachment_text,
}


def _to_mcp_tool(definition: ToolDefinition) -> types.Tool:
    return types.Tool(
        name=definition.name,
        description=definition.description,
        input_schema=definition.input_schema,
        annotations=types.ToolAnnotations(
            read_only_hint=definition.annotations["readOnlyHint"],
            destructive_hint=definition.annotations["destructiveHint"],
            idempotent_hint=definition.annotations["idempotentHint"],
            open_world_hint=definition.annotations["openWorldHint"],
        ),
    )


def _invalid_argument(name: str, arguments: dict[str, Any]) -> str | None:
    """The SPEC §10.1 `INVALID_ARGUMENT` message for this call, or `None`
    if the tool exists and its arguments satisfy its hand-authored JSON
    Schema (`imsg.mcp.tools.schemas`).

    Called twice per request on purpose: once by `on_call_tool`, so a
    malformed call is answered at once instead of waiting out a warm-up
    it could never use, and once inside `_run_tool`, which stays correct
    on its own rather than trusting a caller to have checked. Validating
    a handful of keys against a small schema costs far less than either
    a wrong answer or the wait it skips."""
    definition = PUBLIC_TOOL_DEFINITIONS_BY_NAME.get(name)
    if definition is None:
        return f"unknown tool {name!r}"
    try:
        jsonschema.validate(arguments, definition.input_schema)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def _result_count(payload: dict[str, Any] | None) -> int | None:
    if payload is None:
        return None
    for key in ("results", "messages", "people", "texts"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _authorization_header(context: ServerRequestContext[None]) -> str | None:
    """Read the (already duplicate-checked — see `TransportGuardASGIApp`)
    `Authorization` header off the transport's per-message request
    object. `context.request` is `RequestT`-generic in the SDK (`Any` in
    practice for a `Server[None]`); this stays defensive rather than
    asserting a concrete Starlette `Request` type, since a non-HTTP
    transport (never used for the public surface, but not ruled out by
    the type system) would carry `None` here."""
    request: Any = context.request
    if request is None:
        return None
    request_headers = getattr(request, "headers", None)
    if request_headers is None:
        return None
    value = request_headers.get("authorization")
    return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class _ToolCallOutcome:
    """What one public tool call produced — always constructed, never
    raised, by the time it reaches `PublicAuthGate.dispatch`'s handler
    contract (that contract treats a raised exception as `INTERNAL` and
    re-raises it; a `RetrievalError` is an expected, closed-set SPEC
    §10.1 outcome, not an internal failure, so it is caught here and
    carried as data). Mirrors `imsg.mcp.tools.dispatch.ToolCallResult`
    (the local surface's analogous shape) but is a distinct type: the
    local surface's `call_tool` can raise/catch freely since it owns
    its own control flow, while this type is specifically what
    `gate.dispatch`'s `handler: Callable[[AuthorizedRequest],
    ToolOutcome[T]]` parameter requires as `T`."""

    payload: dict[str, Any] | None
    error_code: str | None
    error_message: str | None


@dataclass(slots=True)
class PublicMcpServer:
    """Owns the retrieval service, the auth gate, and the effective
    scope (SPEC §10.3a) — exposes the two callbacks
    `mcp.server.lowlevel.Server` needs, mirroring
    `imsg.mcp.tools.local_server.LocalMcpServer`'s shape but gated by
    `PublicAuthGate.dispatch` instead of the local surface's plain
    audit-only `call_tool`.
    """

    service: RetrievalService
    gate: PublicAuthGate
    scope: AccessScope
    """`config.mcp.public.scope` (SPEC §10.3a/§10.4) — `full` or
    `allowlist`, REQUIRED with no default at the config layer (D6).
    Every `AccessContext` this server ever constructs uses exactly this
    value; there is no per-request or per-tool override."""

    warm_up: BackgroundWarmUp
    """The background model load every retrieval tool is gated on.
    Required, with no default: a `PublicMcpServer` built without one
    would answer its first query on cold models, which is the gap this
    field exists to close."""

    warm_up_wait_seconds: float = TOOL_CALL_WARM_UP_WAIT_SECONDS

    async def on_list_tools(
        self,
        context: ServerRequestContext[None],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        del context, params  # no pagination; listing tools needs no auth beyond the transport gate
        return types.ListToolsResult(tools=[_to_mcp_tool(d) for d in PUBLIC_TOOL_DEFINITIONS])

    async def _settled_warm_up(self) -> WarmUpStatus:
        """The warm-up's status once it is ready or failed, or after this
        call has waited `warm_up_wait_seconds` for it. Polls with
        `anyio.sleep` rather than blocking on the warm-up's own
        condition, so the event loop keeps serving other requests
        meanwhile and a client that hangs up stops the wait at once."""
        deadline = anyio.current_time() + self.warm_up_wait_seconds
        while True:
            status = self.warm_up.status()
            left = deadline - anyio.current_time()
            if status.settled or left <= 0:
                return status
            await anyio.sleep(min(WARM_UP_POLL_SECONDS, left))

    def _run_tool(
        self,
        authorized: AuthorizedRequest,
        name: str,
        arguments: dict[str, Any],
        *,
        warm_up: WarmUpStatus | None = None,
    ) -> _ToolCallOutcome:
        violation = _invalid_argument(name, arguments)
        if violation is not None:
            return _ToolCallOutcome(
                payload=None, error_code="INVALID_ARGUMENT", error_message=violation
            )

        # SPEC §10.3a: the single place the effective AccessContext is
        # built for the public surface. `authorized.subject` is carried
        # for audit purposes only — the *scope*, fixed at server
        # construction, decides what is visible, never the subject
        # (there is only ever one subject that can reach this point:
        # the pinned owner).
        context = AccessContext(surface="public", scope=self.scope, subject=authorized.subject)
        tool_fn = _RETRIEVAL_HANDLERS[name]
        try:
            if warm_up is not None:
                # `WARMING_UP` / `WARM_UP_FAILED` are `RetrievalError`s, so
                # they take the same route out as any other SPEC §10.1
                # outcome: their own machine code in the tool error and in
                # the audit row, never `INTERNAL`.
                warm_up.raise_unless_ready(wait_bound_seconds=self.warm_up_wait_seconds)
            payload = tool_fn(self.service, context, arguments)
        except RetrievalError as exc:
            return _ToolCallOutcome(payload=None, error_code=exc.code, error_message=str(exc))
        return _ToolCallOutcome(payload=payload, error_code=None, error_message=None)

    async def on_call_tool(
        self, context: ServerRequestContext[None], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        name = params.name
        arguments = dict(params.arguments or {})
        authorization = _authorization_header(context)

        # A call that can never run — unknown tool, arguments that fail the
        # schema — is answered without waiting: no amount of warming makes
        # it valid. Everything else needs the models, so it waits for them
        # here, off the event loop, before the (synchronous) dispatch.
        warm_up: WarmUpStatus | None = None
        if _invalid_argument(name, arguments) is None:
            warm_up = await self._settled_warm_up()

        def handler(authorized: AuthorizedRequest) -> ToolOutcome[_ToolCallOutcome]:
            outcome = self._run_tool(authorized, name, arguments, warm_up=warm_up)
            return ToolOutcome(
                payload=outcome,
                result_count=_result_count(outcome.payload),
                error=outcome.error_code,
            )

        try:
            result = self.gate.dispatch(
                authorization, tool=name, params=arguments, handler=handler
            )
        except Exception:
            # `gate.dispatch` already wrote an INTERNAL audit row and
            # re-raised (its own docstring: "the transport maps it to
            # §10.1 INTERNAL — never a stack trace") — this is that
            # mapping. Nothing here is safe to include in the message:
            # the whole point of SPEC §10.1 is that public errors never
            # carry filesystem paths, SQL, or exception text.
            logger.error("mcp.public_tool_internal_error", extra={"tool": name}, exc_info=True)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="INTERNAL\ninternal error")],
                is_error=True,
            )

        if result.rejection is not None:
            # Rare TOCTOU: `TransportGuardASGIApp` already authorized
            # this HTTP request before the MCP dispatcher ever saw it.
            # This branch fires only if the verdict changed in the
            # narrow window between that check and this one (e.g. the
            # rate limit or the tokeninfo cache boundary was crossed
            # mid-request). The outer HTTP response for this JSON-RPC
            # exchange already committed to a 200 envelope by this
            # point in the SDK's request lifecycle, so a tool-error
            # result naming the rejection is the best available signal
            # — never a silent success, never the corpus payload.
            text = f"{result.rejection.code}\nauthorization could not be confirmed for this request"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], is_error=True
            )

        outcome = result.payload
        assert outcome is not None  # dispatch() pairs payload=None only with a rejection
        if outcome.error_code is not None:
            text = f"{outcome.error_code}\n{outcome.error_message}"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], is_error=True
            )

        payload = outcome.payload or {}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, default=str))],
            structured_content=payload,
            is_error=False,
        )

    def build_server(self) -> Server[None]:
        return Server(name=SERVER_NAME, on_list_tools=self.on_list_tools, on_call_tool=self.on_call_tool)


# ---------------------------------------------------------------------------
# Transport: duplicate-header rejection, Host/Origin validation, the
# blanket bearer-auth check, and response cache headers (SPEC §10.4 —
# "the parts that live outside" imsg.mcp.auth).
# ---------------------------------------------------------------------------


def _cache_header_pairs(extra: Mapping[str, str] | None = None) -> list[tuple[bytes, bytes]]:
    merged: dict[str, str] = {**RESPONSE_CACHE_HEADERS, **(extra or {})}
    return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in merged.items()]


def _with_cache_headers(send: Send) -> Send:
    """Wrap an ASGI `send` so `RESPONSE_CACHE_HEADERS` is attached to
    every response `inner` produces — SPEC §10.4: "attach
    RESPONSE_CACHE_HEADERS to every tool response." Applied uniformly to
    every response this app answers (not only literal tool-call
    responses): the transport layer cannot cheaply distinguish a
    `tools/call` response from `initialize`/`tools/list` without
    parsing the JSON-RPC body it is streaming, and stamping "never
    cache, corpus text may be present" on protocol bookkeeping too is
    harmless — the failure mode this guards against (a shared cache
    serving one subject's corpus text to another) only ever runs in one
    direction (over-applying `no-store` costs nothing)."""

    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.extend(_cache_header_pairs())
            message = {**message, "headers": headers}
        await send(message)

    return wrapped


class TransportGuardASGIApp:
    """Coarse, always-on gate in front of the whole public MCP app —
    see the module docstring for why this exists alongside (not
    instead of) `PublicMcpServer.on_call_tool`'s `gate.dispatch` calls.

    `unauthenticated_paths` is deliberately narrow — by default empty,
    meaning literally every HTTP request requires a valid bearer token
    (hard requirement 4: "there is no unauthenticated path"). The one
    principled exception is RFC 9728 protected-resource metadata
    (SPEC §10.4: "Publish RFC 9728 protected-resource metadata..."): a
    client cannot present a token to fetch the document that tells it
    *how* to obtain one, so that single well-known path is exempted
    from the bearer check — never from Host/Origin/duplicate-header
    validation, which cost the caller nothing and apply uniformly.
    """

    def __init__(
        self,
        inner: ASGIApp,
        *,
        gate: PublicAuthGate,
        allowed_hosts: Sequence[str],
        allowed_origins: Sequence[str],
        unauthenticated_paths: frozenset[str] = frozenset(),
    ) -> None:
        self._inner = inner
        self._gate = gate
        self._allowed_hosts = tuple(allowed_hosts)
        self._allowed_origins = tuple(allowed_origins)
        self._unauthenticated_paths = unauthenticated_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # Lifespan (StreamableHTTPSessionManager.run()'s startup/
            # shutdown) and anything else non-HTTP passes straight
            # through: Host/Origin/bearer checks are meaningless outside
            # an HTTP request, and swallowing the lifespan event here
            # would mean the session manager's task group never starts.
            await self._inner(scope, receive, send)
            return

        raw_headers: RawHeaders = scope.get("headers", [])

        duplicate = find_duplicate_header(raw_headers)
        if duplicate is not None:
            await self._respond(send, 400, f"UNAUTHORIZED\nduplicate {duplicate!r} header")
            return

        headers = Headers(raw=list(raw_headers))
        if not validate_host(headers.get("host"), self._allowed_hosts):
            await self._respond(send, 403, "UNAUTHORIZED\ninvalid Host header")
            return
        if not validate_origin(headers.get("origin"), self._allowed_origins):
            await self._respond(send, 403, "UNAUTHORIZED\ninvalid Origin header")
            return

        if scope["path"] not in self._unauthenticated_paths:
            verdict = self._gate.authorize(headers.get("authorization"))
            if isinstance(verdict, Rejection):
                extra = (
                    {"WWW-Authenticate": verdict.www_authenticate}
                    if verdict.www_authenticate
                    else None
                )
                await self._respond(send, verdict.status, verdict.code, extra_headers=extra)
                return

        await self._inner(scope, receive, _with_cache_headers(send))

    async def _respond(
        self,
        send: Send,
        status: int,
        message: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        headers = [(b"content-type", b"text/plain; charset=utf-8"), *_cache_header_pairs(extra_headers)]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": message.encode("utf-8")})


GOOGLE_OAUTH_ISSUER = "https://accounts.google.com"
"""RFC 9728 `authorization_servers` entry — where a client discovers
Google's *own* OAuth metadata (authorization/token endpoints, JWKS,
...). Distinct from `imsg.mcp.auth.GOOGLE_TOKENINFO_URL`, which this
server calls directly to introspect a presented token; a client never
talks to that endpoint itself. Google is the fixed IdP for this build
(SPEC §10.4 judgment call, provisional until AT-1)."""

PUBLIC_OAUTH_SCOPES: tuple[str, ...] = ("openid", "email", "profile")
"""Matches the GE data-store OAuth config's scopes (SPEC §10.4: "scopes
`openid email profile`")."""

WELL_KNOWN_METADATA_PATH = "/.well-known/oauth-protected-resource"
"""RFC 9728 §3.1's fixed path — must match
`imsg.mcp.auth.resource_metadata_url_for`'s own construction exactly,
since that is the URL this server puts in the `WWW-Authenticate`
challenge on every 401."""


def _resource_metadata_routes(external_url: str) -> list[Route]:
    """Serve RFC 9728 protected-resource metadata at exactly
    `WELL_KNOWN_METADATA_PATH` (SPEC §10.4: "Publish RFC 9728
    protected-resource metadata and include its URL in the
    `WWW-Authenticate` challenge...").

    Deliberately does **not** use the SDK's own
    `create_protected_resource_routes`/`build_resource_metadata_url`
    convenience — those insert the resource's own URL *path* after the
    well-known prefix (`/.well-known/oauth-protected-resource/mcp` for
    an `external_url` ending in `/mcp`), which is a legitimate RFC 9728
    §3.1 form but does not match
    `imsg.mcp.auth.resource_metadata_url_for` (unmodifiable this wave),
    which always builds the bare form
    (`/.well-known/oauth-protected-resource`, no suffix) for the
    `WWW-Authenticate` challenge on every 401. Registering at a
    different path than the one advertised in the challenge would make
    the challenge lie, so this builds the same
    `ProtectedResourceMetadata` document the SDK would, and registers
    it at the bare path instead — RFC 9728 permits either form; only
    one of them can match the challenge this server actually sends.
    Imported lazily so callers that never touch this (e.g.
    `parse_bind_address`-only tooling, or a bare app for unit tests)
    don't pay for `mcp.server.auth`'s import graph.
    """
    from mcp.server.auth.handlers.metadata import ProtectedResourceMetadataHandler
    from mcp.server.auth.routes import cors_middleware
    from mcp.shared.auth import ProtectedResourceMetadata
    from starlette.routing import Route

    metadata = ProtectedResourceMetadata(
        resource=external_url,  # pydantic coerces str -> AnyHttpUrl
        authorization_servers=[GOOGLE_OAUTH_ISSUER],
        scopes_supported=list(PUBLIC_OAUTH_SCOPES),
        resource_name="imessage-index",
    )
    handler = ProtectedResourceMetadataHandler(metadata)
    return [
        Route(
            WELL_KNOWN_METADATA_PATH,
            endpoint=cors_middleware(handler.handle, ["GET", "OPTIONS"]),
            methods=["GET", "OPTIONS"],
        )
    ]


def build_public_asgi_app(
    public: PublicMcpServer,
    *,
    allowed_hosts: Sequence[str],
    allowed_origins: Sequence[str],
    external_url: str | None = None,
    streamable_http_path: str = DEFAULT_STREAMABLE_HTTP_PATH,
) -> ASGIApp:
    """Assemble the full public-surface ASGI app: the `mcp` SDK's own
    StreamableHTTP session manager (protocol_versions negotiation,
    session tracking, SSE where the negotiated revision permits — SPEC
    §10.4), wrapped by :class:`TransportGuardASGIApp`.

    The SDK's own DNS-rebinding middleware (`transport_security`) is
    deliberately disabled here (`enable_dns_rebinding_protection=
    False`) rather than configured: `TransportGuardASGIApp` performs
    that check itself, once, for every request regardless of which
    internal protocol-version path the SDK routes it through — running
    both would mean two independently-configured Host/Origin checks
    that could drift apart, and the SDK's own version does not cover
    duplicate-header rejection at all (SPEC §10.4's own list of
    transport obligations this build owns).

    `external_url` (`config.mcp.public.external_url`) is optional here
    only so tests/tooling can build a bare app without it; the real
    `imsg mcp public` CLI entry point always has one (config validation
    requires it whenever `mcp.public.enabled` is true) and always
    passes it, which is what makes the RFC 9728 metadata endpoint
    present on the real server.
    """
    low_level = public.build_server()
    custom_routes = _resource_metadata_routes(external_url) if external_url else None
    inner = low_level.streamable_http_app(
        streamable_http_path=streamable_http_path,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        custom_starlette_routes=custom_routes,
    )
    unauthenticated_paths = frozenset({WELL_KNOWN_METADATA_PATH}) if external_url else frozenset()
    return TransportGuardASGIApp(
        inner,
        gate=public.gate,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        unauthenticated_paths=unauthenticated_paths,
    )


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def parse_bind_address(bind: str) -> tuple[str, int]:
    """Parse `mcp.public.bind` ("host:port") and refuse anything but a
    loopback host (SPEC §10.4: "Server binds `127.0.0.1:8700`;
    cloudflared publishes `https://<host>/mcp`" — cloudflared, not this
    process, is the only thing that should ever face a public
    interface). Raises `PublicSurfaceStartupError` — the same
    fail-to-start class `imsg.mcp.auth.build_public_gate` raises for
    every other public-surface misconfiguration, so a CLI entry point
    only needs to catch one exception type.
    """
    host, _, port_str = bind.rpartition(":")
    if not host or not port_str.isdigit():
        raise PublicSurfaceStartupError(
            f"mcp.public.bind must look like 'host:port', got {bind!r}"
        )
    if host not in _LOOPBACK_HOSTS:
        raise PublicSurfaceStartupError(
            f"mcp.public.bind must be a loopback address, got {host!r} (SPEC §10.4: "
            f"'Server binds 127.0.0.1:8700' — cloudflared is the only process that "
            f"should ever face a public interface, never this one directly)"
        )
    return host, int(port_str)


__all__ = [
    "DEFAULT_STREAMABLE_HTTP_PATH",
    "GOOGLE_OAUTH_ISSUER",
    "PUBLIC_OAUTH_SCOPES",
    "SERVER_NAME",
    "TOOL_CALL_WARM_UP_WAIT_SECONDS",
    "WELL_KNOWN_METADATA_PATH",
    "PublicMcpServer",
    "TransportGuardASGIApp",
    "build_public_asgi_app",
    "parse_bind_address",
]
