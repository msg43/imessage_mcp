"""Tests for the public MCP surface transport (SPEC §10.4).

The property most worth proving, per this build's own test mandate: no
code path reaches a tool handler without passing the gate — including
malformed requests, wrong Host/Origin, duplicate headers, and unknown
tools. Exercised at two layers, matching `imsg.mcp.tools.public_server`'s
own two-gate design:

- `TransportGuardASGIApp` in isolation, against a trivial recording
  `inner` ASGI app (no real MCP session manager involved) — proves the
  coarse, always-on HTTP-level gate blocks *before* anything downstream
  ever runs.
- `PublicMcpServer._run_tool`/`on_call_tool` directly, against a fake
  `RetrievalService` and a real `PublicAuthGate` (stub introspector,
  same pattern as `tests/test_mcp_auth.py`) — proves the tool-handler
  gate (`PublicAuthGate.dispatch`) is the only path to a handler, that
  scope (SPEC §10.3a) is applied centrally, and that `RetrievalError`
  maps to SPEC §10.1 tool-error content rather than a stack trace.

No network, no real Postgres, no real MCP client — this is exactly the
"stub the transport boundary" testing the build task calls for.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from types import SimpleNamespace
from typing import Any, cast

import mcp.types as types
import pytest
from starlette.types import Message, Receive, Scope, Send

from imsg.mcp.audit import MemoryAuditSink
from imsg.mcp.auth import PublicAuthGate, TokenIntrospection
from imsg.mcp.errors import PublicSurfaceStartupError, TokenInvalidError
from imsg.mcp.tools.public_server import (
    WELL_KNOWN_METADATA_PATH,
    PublicMcpServer,
    TransportGuardASGIApp,
    build_public_asgi_app,
    parse_bind_address,
)
from imsg.retrieval.access import AccessContext
from imsg.retrieval.errors import NotFoundError, PersonAmbiguousError, PersonCandidate
from imsg.retrieval.service import SearchMessagesResult

OWNER_SUB = "300000000000000000003"
OTHER_SUB = "400000000000000000004"
CLIENT_ID = "111111111111-fictional.apps.example"
OWNER_TOKEN = "fictional-public-owner-token"
OTHER_TOKEN = "fictional-public-foreign-token"

ALLOWED_HOST = "mcp.fictional.example"
ALLOWED_ORIGIN = "https://vertexaisearch.fictional.example"


def run[T](awaitable: Awaitable[T]) -> T:
    # `ASGIApp.__call__` is typed `Awaitable[None]` (the ASGI protocol's own
    # shape), not `Coroutine` — but every awaitable actually produced here
    # is a plain coroutine, which is all `asyncio.run` needs at runtime.
    return asyncio.run(cast("Coroutine[Any, Any, T]", awaitable))


def bearer(token: str) -> str:
    return f"Bearer {token}"


class StubIntrospector:
    """Same shape as `tests/test_mcp_auth.py`'s — kept file-local rather
    than shared, matching this repo's existing per-file fixture style."""

    def __init__(self) -> None:
        self.outcomes: dict[str, TokenIntrospection | Exception] = {
            OWNER_TOKEN: TokenIntrospection(
                subject=OWNER_SUB,
                audience=CLIENT_ID,
                authorized_party=None,
                scopes=frozenset({"openid"}),
                expires_in_seconds=3600,
            ),
            OTHER_TOKEN: TokenIntrospection(
                subject=OTHER_SUB,
                audience=CLIENT_ID,
                authorized_party=None,
                scopes=frozenset({"openid"}),
                expires_in_seconds=3600,
            ),
        }
        self.calls: list[str] = []

    def introspect(self, token: str) -> TokenIntrospection:
        self.calls.append(token)
        outcome = self.outcomes.get(token)
        if outcome is None:
            raise TokenInvalidError("unknown token")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_gate(**overrides: Any) -> tuple[PublicAuthGate, MemoryAuditSink]:
    audit = MemoryAuditSink()
    gate = PublicAuthGate(
        client_id=CLIENT_ID,
        owner_subject=OWNER_SUB,
        introspector=StubIntrospector(),
        audit=audit,
        **overrides,
    )
    return gate, audit


# ---------------------------------------------------------------------------
# Minimal ASGI test harness — no real socket, no httpx dependency.
# ---------------------------------------------------------------------------


def headers(*pairs: tuple[str, str]) -> list[tuple[bytes, bytes]]:
    return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in pairs]


async def send_request(
    app: Callable[[Scope, Receive, Send], Awaitable[None]],
    *,
    request_headers: list[tuple[bytes, bytes]],
    method: str = "POST",
    body: bytes = b"{}",
) -> tuple[int | None, list[tuple[bytes, bytes]], bytes]:
    scope: Scope = {
        "type": "http",
        "method": method,
        "path": "/mcp",
        "raw_path": b"/mcp",
        "headers": request_headers,
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "scheme": "http",
        "http_version": "1.1",
    }
    consumed = False

    async def receive() -> Message:
        nonlocal consumed
        if consumed:
            return {"type": "http.disconnect"}
        consumed = True
        return {"type": "http.request", "body": body, "more_body": False}

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    await app(scope, receive, send)

    status: int | None = None
    response_headers: list[tuple[bytes, bytes]] = []
    response_body = b""
    for message in messages:
        if message["type"] == "http.response.start":
            status = message["status"]
            response_headers = list(message.get("headers", []))
        elif message["type"] == "http.response.body":
            response_body += message.get("body", b"")
    return status, response_headers, response_body


class RecordingInnerApp:
    """Stands in for the real MCP session manager — records whether (and
    with what scope type) it was ever reached, and nothing more."""

    def __init__(self) -> None:
        self.called = False
        self.last_scope_type: str | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        del receive
        self.called = True
        self.last_scope_type = scope["type"]
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})


def guarded(
    gate: PublicAuthGate,
) -> tuple[TransportGuardASGIApp, RecordingInnerApp]:
    inner = RecordingInnerApp()
    app = TransportGuardASGIApp(
        inner, gate=gate, allowed_hosts=[ALLOWED_HOST], allowed_origins=[ALLOWED_ORIGIN]
    )
    return app, inner


# ---------------------------------------------------------------------------
# TransportGuardASGIApp — the coarse, always-on gate
# ---------------------------------------------------------------------------


def test_duplicate_authorization_header_is_rejected_before_inner_runs() -> None:
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(
            app,
            request_headers=headers(
                ("host", ALLOWED_HOST),
                ("authorization", bearer(OWNER_TOKEN)),
                ("authorization", bearer(OWNER_TOKEN)),
            ),
        )
    )
    assert status == 400
    assert inner.called is False


def test_duplicate_host_header_is_rejected_before_inner_runs() -> None:
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(
            app,
            request_headers=headers(
                ("host", ALLOWED_HOST),
                ("host", "evil.fictional.example"),
                ("authorization", bearer(OWNER_TOKEN)),
            ),
        )
    )
    assert status == 400
    assert inner.called is False


def test_invalid_host_is_rejected_before_inner_runs() -> None:
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(
            app,
            request_headers=headers(
                ("host", "evil.fictional.example"), ("authorization", bearer(OWNER_TOKEN))
            ),
        )
    )
    assert status == 403
    assert inner.called is False


def test_missing_host_is_rejected_before_inner_runs() -> None:
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(app, request_headers=headers(("authorization", bearer(OWNER_TOKEN))))
    )
    assert status == 403
    assert inner.called is False


def test_invalid_origin_is_rejected_before_inner_runs() -> None:
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(
            app,
            request_headers=headers(
                ("host", ALLOWED_HOST),
                ("origin", "https://evil.fictional.example"),
                ("authorization", bearer(OWNER_TOKEN)),
            ),
        )
    )
    assert status == 403
    assert inner.called is False


def test_absent_origin_is_fine_server_to_server_calls_carry_none() -> None:
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(
            app, request_headers=headers(("host", ALLOWED_HOST), ("authorization", bearer(OWNER_TOKEN)))
        )
    )
    assert status == 200
    assert inner.called is True


def test_missing_authorization_is_401_before_inner_runs() -> None:
    gate, audit = make_gate()
    app, inner = guarded(gate)
    status, response_headers, _ = run(
        send_request(app, request_headers=headers(("host", ALLOWED_HOST)))
    )
    assert status == 401
    assert inner.called is False
    header_map = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in response_headers}
    assert "www-authenticate" in header_map
    rows = audit.snapshot()
    assert len(rows) == 1 and rows[0].subject_ok is False


def test_foreign_subject_is_401_before_inner_runs_and_is_audited() -> None:
    gate, audit = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(
            app,
            request_headers=headers(
                ("host", ALLOWED_HOST), ("authorization", bearer(OTHER_TOKEN))
            ),
        )
    )
    assert status == 401
    assert inner.called is False
    rows = audit.snapshot()
    assert any(r.subject == OTHER_SUB and not r.subject_ok for r in rows)


def test_rate_limited_subject_gets_429_before_inner_runs() -> None:
    gate, _ = make_gate(rate_limit_per_minute=1)
    app, inner = guarded(gate)
    ok_status, _, _ = run(
        send_request(
            app, request_headers=headers(("host", ALLOWED_HOST), ("authorization", bearer(OWNER_TOKEN)))
        )
    )
    assert ok_status == 200
    assert inner.called is True
    inner.called = False
    limited_status, _, _ = run(
        send_request(
            app, request_headers=headers(("host", ALLOWED_HOST), ("authorization", bearer(OWNER_TOKEN)))
        )
    )
    assert limited_status == 429
    assert inner.called is False


def test_valid_owner_token_reaches_inner_and_gets_cache_headers() -> None:
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, response_headers, _ = run(
        send_request(
            app, request_headers=headers(("host", ALLOWED_HOST), ("authorization", bearer(OWNER_TOKEN)))
        )
    )
    assert status == 200
    assert inner.called is True
    header_map = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in response_headers}
    assert header_map["cache-control"] == "private, no-store"
    assert header_map["vary"] == "Authorization"


def test_rejection_responses_also_carry_no_store_cache_headers() -> None:
    gate, _ = make_gate()
    app, _inner = guarded(gate)
    status, response_headers, _ = run(
        send_request(app, request_headers=headers(("host", ALLOWED_HOST)))
    )
    assert status == 401
    header_map = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in response_headers}
    assert header_map["cache-control"] == "private, no-store"


def test_non_http_scope_passes_through_untouched() -> None:
    """Lifespan events must reach `inner` unexamined — swallowing one
    here would mean the real session manager's task group never
    starts (every subsequent request would fail)."""
    gate, _ = make_gate()
    app, inner = guarded(gate)

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        del message

    run(app({"type": "lifespan"}, receive, send))
    assert inner.called is True
    assert inner.last_scope_type == "lifespan"


def test_bearer_scheme_case_and_whitespace_still_goes_through_the_gate() -> None:
    """Not a transport-layer bypass — `PublicAuthGate.authorize` itself
    already covers this (see `tests/test_mcp_auth.py`); this just
    confirms the ASGI wrapper doesn't do any of its own parsing that
    could disagree with the gate's."""
    gate, _ = make_gate()
    app, inner = guarded(gate)
    status, _, _ = run(
        send_request(
            app,
            request_headers=headers(("host", ALLOWED_HOST), ("authorization", "Bearer  two words")),
        )
    )
    assert status == 401
    assert inner.called is False


# ---------------------------------------------------------------------------
# PublicMcpServer — the tool-handler gate (gate.dispatch)
# ---------------------------------------------------------------------------


class FakeRetrievalService:
    """Duck-typed stand-in for `imsg.retrieval.service.RetrievalService`
    — records every call's `AccessContext` (so scope enforcement is
    directly assertable) and can be told to raise a `RetrievalError`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, AccessContext, dict[str, Any]]] = []
        self.raise_error: Exception | None = None

    def search_messages(self, context: AccessContext, **kwargs: Any) -> SearchMessagesResult:
        self.calls.append(("search_messages", context, kwargs))
        if self.raise_error is not None:
            raise self.raise_error
        return SearchMessagesResult(results=[], candidate_lists={}, scan_cap_reached=False)

    def get_conversation(self, context: AccessContext, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_conversation", context, kwargs))
        if self.raise_error is not None:
            raise self.raise_error
        return {"thread_key": "thread_fictional", "messages": []}

    def list_people(self, context: AccessContext, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_people", context, kwargs))
        if self.raise_error is not None:
            raise self.raise_error
        return {"people": []}

    def get_attachment_text(self, context: AccessContext, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_attachment_text", context, kwargs))
        if self.raise_error is not None:
            raise self.raise_error
        return {
            "attachment_key": "att_fictional",
            "filename": None,
            "mime_type": None,
            "texts": [],
            "untrusted_content": True,
        }


def make_server(
    *, scope: str = "allowlist", gate: PublicAuthGate | None = None
) -> tuple[PublicMcpServer, FakeRetrievalService, PublicAuthGate]:
    g = gate if gate is not None else make_gate()[0]
    service = FakeRetrievalService()
    server = PublicMcpServer(service=service, gate=g, scope=scope)  # type: ignore[arg-type]
    return server, service, g


def fake_context(authorization: str | None) -> Any:
    """A minimal duck-typed stand-in for `ServerRequestContext[None]` —
    only `.request.headers.get("authorization")` is ever read by
    `PublicMcpServer`."""
    request_headers: dict[str, str] = {}
    if authorization is not None:
        request_headers["authorization"] = authorization
    return SimpleNamespace(request=SimpleNamespace(headers=request_headers))


def call_params(name: str, arguments: dict[str, Any]) -> types.CallToolRequestParams:
    return types.CallToolRequestParams(name=name, arguments=arguments)


def test_unknown_tool_is_invalid_argument_and_never_reaches_a_handler() -> None:
    server, service, _ = make_server()
    result = run(
        server.on_call_tool(fake_context(bearer(OWNER_TOKEN)), call_params("run_sql", {}))
    )
    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert text.startswith("INVALID_ARGUMENT")
    assert service.calls == []


def test_check_permissions_is_not_a_public_tool() -> None:
    """SPEC §10.2: check_permissions is local-surface only."""
    server, service, _ = make_server()
    result = run(
        server.on_call_tool(fake_context(bearer(OWNER_TOKEN)), call_params("check_permissions", {}))
    )
    assert result.is_error is True
    assert service.calls == []


def test_schema_violation_is_invalid_argument_and_never_reaches_the_service() -> None:
    server, service, _ = make_server()
    result = run(
        server.on_call_tool(
            fake_context(bearer(OWNER_TOKEN)),
            call_params("search_messages", {"query": ""}),  # violates minLength: 1
        )
    )
    assert result.is_error is True
    assert result.content[0].text.startswith("INVALID_ARGUMENT")  # type: ignore[union-attr]
    assert service.calls == []


def test_list_people_include_handles_is_rejected_by_schema_not_silently_dropped() -> None:
    """SPEC §10.2: the public registration omits `include_handles`
    entirely (`additionalProperties: false`) — a client sending it
    anyway is a schema violation, not a silently-ignored field."""
    server, service, _ = make_server()
    result = run(
        server.on_call_tool(
            fake_context(bearer(OWNER_TOKEN)),
            call_params("list_people", {"include_handles": True}),
        )
    )
    assert result.is_error is True
    assert service.calls == []


def test_missing_or_wrong_subject_never_reaches_a_handler() -> None:
    server, service, _ = make_server()
    for auth in (None, bearer(OTHER_TOKEN), bearer("garbage")):
        result = run(
            server.on_call_tool(fake_context(auth), call_params("search_messages", {"query": "hi"}))
        )
        assert result.is_error is True
        assert service.calls == []


def test_successful_call_reaches_the_service_with_the_configured_scope() -> None:
    server, service, _ = make_server(scope="allowlist")
    result = run(
        server.on_call_tool(
            fake_context(bearer(OWNER_TOKEN)), call_params("search_messages", {"query": "fictional query"})
        )
    )
    assert result.is_error is False
    assert len(service.calls) == 1
    tool_name, context, kwargs = service.calls[0]
    assert tool_name == "search_messages"
    assert context.surface == "public"
    assert context.scope == "allowlist"
    assert context.subject == OWNER_SUB
    assert kwargs["query"] == "fictional query"


def test_full_scope_is_threaded_through_too() -> None:
    server, service, _ = make_server(scope="full")
    run(server.on_call_tool(fake_context(bearer(OWNER_TOKEN)), call_params("list_people", {})))
    _, context, _ = service.calls[0]
    assert context.scope == "full"


def test_retrieval_error_becomes_its_own_spec_code_not_internal() -> None:
    server, service, _ = make_server()
    service.raise_error = NotFoundError("no such thread")
    result = run(
        server.on_call_tool(
            fake_context(bearer(OWNER_TOKEN)), call_params("get_conversation", {"thread_id": "t_x"})
        )
    )
    assert result.is_error is True
    assert result.content[0].text.startswith("NOT_FOUND")  # type: ignore[union-attr]


def test_person_ambiguous_error_is_preserved() -> None:
    server, service, _ = make_server()
    service.raise_error = PersonAmbiguousError(
        "ali", (PersonCandidate(short_name="alice", display_name="Alice Example"),)
    )
    result = run(
        server.on_call_tool(fake_context(bearer(OWNER_TOKEN)), call_params("search_messages", {"query": "ali"}))
    )
    assert result.content[0].text.startswith("PERSON_AMBIGUOUS")  # type: ignore[union-attr]


def test_unexpected_exception_becomes_internal_never_leaks_details() -> None:
    server, service, _ = make_server()
    service.raise_error = RuntimeError("path=/fictional/should/never/leak LEAK")
    result = run(
        server.on_call_tool(fake_context(bearer(OWNER_TOKEN)), call_params("search_messages", {"query": "x"}))
    )
    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert text.startswith("INTERNAL")
    assert "LEAK" not in text
    assert "/fictional/should/never/leak" not in text


def test_every_public_call_is_audited_with_the_tool_name() -> None:
    gate, audit = make_gate()
    server, _, _ = make_server(gate=gate)
    run(server.on_call_tool(fake_context(bearer(OWNER_TOKEN)), call_params("list_people", {})))
    rows = audit.snapshot()
    assert any(r.tool == "list_people" and r.subject_ok is True for r in rows)


def test_on_list_tools_returns_exactly_the_public_four() -> None:
    server, _, _ = make_server()
    result = run(server.on_list_tools(fake_context(None), None))
    names = {t.name for t in result.tools}
    assert names == {"search_messages", "get_conversation", "list_people", "get_attachment_text"}


# ---------------------------------------------------------------------------
# build_public_asgi_app — assembly smoke test
# ---------------------------------------------------------------------------


def test_build_public_asgi_app_wraps_the_transport_guard() -> None:
    gate, _ = make_gate()
    server, _, _ = make_server(gate=gate)
    app = build_public_asgi_app(
        server, allowed_hosts=[ALLOWED_HOST], allowed_origins=[ALLOWED_ORIGIN]
    )
    assert isinstance(app, TransportGuardASGIApp)


def test_build_public_asgi_app_rejects_unauthenticated_requests_end_to_end() -> None:
    gate, _ = make_gate()
    server, _, _ = make_server(gate=gate)
    app = build_public_asgi_app(
        server, allowed_hosts=[ALLOWED_HOST], allowed_origins=[ALLOWED_ORIGIN]
    )
    status, _, _ = run(send_request(app, request_headers=headers(("host", ALLOWED_HOST))))
    assert status == 401


@pytest.mark.parametrize(
    "pairs",
    [
        [("host", "evil.fictional.example"), ("authorization", bearer(OWNER_TOKEN))],
        [
            ("host", ALLOWED_HOST),
            ("origin", "https://evil.fictional.example"),
            ("authorization", bearer(OWNER_TOKEN)),
        ],
    ],
)
def test_build_public_asgi_app_rejects_bad_transport_headers_end_to_end(
    pairs: list[tuple[str, str]],
) -> None:
    gate, _ = make_gate()
    server, _, _ = make_server(gate=gate)
    app = build_public_asgi_app(
        server, allowed_hosts=[ALLOWED_HOST], allowed_origins=[ALLOWED_ORIGIN]
    )
    status, _, _ = run(send_request(app, request_headers=headers(*pairs)))
    assert status == 403


# ---------------------------------------------------------------------------
# RFC 9728 protected-resource metadata — the one unauthenticated path
# ---------------------------------------------------------------------------

EXTERNAL_URL = "https://mcp.fictional.example/mcp"


def test_well_known_metadata_is_reachable_without_a_token() -> None:
    """A client cannot present a bearer token to fetch the document that
    tells it how to obtain one — RFC 9728 metadata must be the one
    exception to "no unauthenticated path" (SPEC §10.4)."""
    gate, _ = make_gate()
    server, _, _ = make_server(gate=gate)
    app = build_public_asgi_app(
        server,
        allowed_hosts=[ALLOWED_HOST],
        allowed_origins=[ALLOWED_ORIGIN],
        external_url=EXTERNAL_URL,
    )
    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/.well-known/oauth-protected-resource",
        "raw_path": b"/.well-known/oauth-protected-resource",
        "headers": headers(("host", ALLOWED_HOST)),
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "scheme": "http",
        "http_version": "1.1",
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    run(app(scope, receive, send))

    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    assert status == 200
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    import json

    payload = json.loads(body)
    assert payload["resource"].rstrip("/") == EXTERNAL_URL.rstrip("/")
    assert payload["authorization_servers"] == ["https://accounts.google.com"]


def test_well_known_metadata_still_enforces_host_validation() -> None:
    """Exempt from the *bearer* check only — Host/Origin/duplicate-header
    validation costs the caller nothing and still applies uniformly."""
    gate, _ = make_gate()
    server, _, _ = make_server(gate=gate)
    app = build_public_asgi_app(
        server,
        allowed_hosts=[ALLOWED_HOST],
        allowed_origins=[ALLOWED_ORIGIN],
        external_url=EXTERNAL_URL,
    )
    status, _, _ = run(
        send_request(
            app,
            method="GET",
            request_headers=headers(("host", "evil.fictional.example")),
        )
    )
    assert status == 403


def test_other_paths_still_require_a_token_when_metadata_is_configured() -> None:
    gate, _ = make_gate()
    server, _, _ = make_server(gate=gate)
    app = build_public_asgi_app(
        server,
        allowed_hosts=[ALLOWED_HOST],
        allowed_origins=[ALLOWED_ORIGIN],
        external_url=EXTERNAL_URL,
    )
    status, _, _ = run(send_request(app, request_headers=headers(("host", ALLOWED_HOST))))
    assert status == 401


def test_www_authenticate_challenge_points_at_the_metadata_route_actually_served() -> None:
    """The subtle bit: the SDK's own metadata-route helper would compute
    a *different* well-known path (with the resource's URL path
    appended) than `imsg.mcp.auth.resource_metadata_url_for` puts in
    the challenge. This proves the two agree — a client that follows
    the challenge URL must land on a route that actually exists."""
    from imsg.mcp.auth import resource_metadata_url_for

    metadata_url = resource_metadata_url_for(EXTERNAL_URL)
    gate, _ = make_gate(resource_metadata_url=metadata_url)
    server, _, _ = make_server(gate=gate)
    app = build_public_asgi_app(
        server,
        allowed_hosts=[ALLOWED_HOST],
        allowed_origins=[ALLOWED_ORIGIN],
        external_url=EXTERNAL_URL,
    )

    status, response_headers, _ = run(
        send_request(app, request_headers=headers(("host", ALLOWED_HOST)))
    )
    assert status == 401
    header_map = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in response_headers}
    challenge = header_map["www-authenticate"]
    assert f'resource_metadata="{metadata_url}"' in challenge

    # The path segment of that exact URL must be the one this app serves.
    from urllib.parse import urlsplit

    served_path = urlsplit(metadata_url).path
    assert served_path == WELL_KNOWN_METADATA_PATH

    # And fetching that exact path (no token) must actually succeed.
    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": served_path,
        "raw_path": served_path.encode(),
        "headers": headers(("host", ALLOWED_HOST)),
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "scheme": "http",
        "http_version": "1.1",
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    run(app(scope, receive, send))
    metadata_status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    assert metadata_status == 200


# ---------------------------------------------------------------------------
# parse_bind_address — loopback-only (SPEC §10.4)
# ---------------------------------------------------------------------------


def test_parse_bind_address_accepts_the_default() -> None:
    assert parse_bind_address("127.0.0.1:8700") == ("127.0.0.1", 8700)


@pytest.mark.parametrize("host", ["localhost", "::1", "[::1]"])
def test_parse_bind_address_accepts_other_loopback_forms(host: str) -> None:
    assert parse_bind_address(f"{host}:8700") == (host, 8700)


@pytest.mark.parametrize(
    "bind",
    [
        "0.0.0.0:8700",
        "mcp.fictional.example:8700",
        "8700",
        "127.0.0.1",
        "127.0.0.1:not-a-port",
        "",
    ],
)
def test_parse_bind_address_refuses_non_loopback_or_malformed(bind: str) -> None:
    with pytest.raises(PublicSurfaceStartupError):
        parse_bind_address(bind)
