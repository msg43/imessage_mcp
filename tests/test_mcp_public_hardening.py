"""The public surface's fixes from the QA review of 2026-09-24.

Each test here was run against the code before the fix and failed there;
the module-level note on each group says how. Where it could, a test
drives the transport (`TransportGuardASGIApp`) or `PublicMcpServer` the
way a real request would, rather than a new helper, so the failure on the
old code is the behaviour itself and not a missing import.

1. A token-less request costs no audit row and no database work: it is
   counted in memory (`imsg.mcp.audit.RejectionTally`).
2. A client that keeps getting rejected is throttled before any tokeninfo
   call or audit write — per client address, so one source cannot spend
   the budget that decides whether the owner's next token is checked.
3. A tool call costs one rate-limit event, not two.
4. The metadata document names no software and sends one Cache-Control;
   uvicorn sends no `server` header and keeps connections 60 s.
5. A token check that needs Google, and an audit write, run off the event
   loop.
6. The tool result still carries its payload twice (text and structured),
   because nothing shows the hosted client can do without the text copy.

Every value is fictional (D5): subjects, client ids, hostnames, and
addresses from the documentation ranges (RFC 5737, RFC 3849).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.request
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from types import SimpleNamespace
from typing import Any, cast

import anyio
import mcp.types as types
import pytest
from starlette.types import Message, Receive, Scope, Send

from imsg.mcp.audit import AuditRecord, MemoryAuditSink
from imsg.mcp.auth import PublicAuthGate, TokenIntrospection
from imsg.mcp.errors import TokenInvalidError
from imsg.mcp.tools.public_server import (
    WELL_KNOWN_METADATA_PATH,
    PublicMcpServer,
    TransportGuardASGIApp,
    build_public_asgi_app,
)
from imsg.retrieval.access import AccessContext
from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpPhase
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import SearchMessagesResult

OWNER_SUB = "500000000000000000005"
OTHER_SUB = "600000000000000000006"
CLIENT_ID = "222222222222-fictional.apps.example"
OWNER_TOKEN = "fictional-hardening-owner-token"
OTHER_TOKEN = "fictional-hardening-foreign-token"

ALLOWED_HOST = "mcp.fictional.example"
ALLOWED_ORIGIN = "https://assistant.fictional.example"
EXTERNAL_URL = "https://mcp.fictional.example/mcp"

SCANNER = "203.0.113.7"  # RFC 5737 documentation range
OWNER_CLIENT = "198.51.100.20"  # RFC 5737 documentation range


def run[T](awaitable: Awaitable[T]) -> T:
    return asyncio.run(cast("Coroutine[Any, Any, T]", awaitable))


def bearer(token: str) -> str:
    return f"Bearer {token}"


def claims(sub: str) -> TokenIntrospection:
    return TokenIntrospection(
        subject=sub,
        audience=CLIENT_ID,
        authorized_party=None,
        scopes=frozenset({"openid"}),
        expires_in_seconds=3600,
    )


class StubIntrospector:
    """Owner and foreign tokens are known; every other token is invalid.
    Records every call, from which thread, and can be held on an event."""

    def __init__(self) -> None:
        self.outcomes: dict[str, TokenIntrospection | Exception] = {
            OWNER_TOKEN: claims(OWNER_SUB),
            OTHER_TOKEN: claims(OTHER_SUB),
        }
        self.calls: list[str] = []
        self.threads: list[str] = []
        self.hold: threading.Event | None = None
        self.entered = threading.Event()

    def introspect(self, token: str) -> TokenIntrospection:
        self.calls.append(token)
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=30), "the introspection was never released"
        outcome = self.outcomes.get(token)
        if outcome is None:
            raise TokenInvalidError("unknown token")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingSink(MemoryAuditSink):
    """A memory sink that also records the thread of every write, and can
    hold a write on an event."""

    def __init__(self) -> None:
        super().__init__()
        self.write_threads: list[str] = []
        self.hold: threading.Event | None = None
        self.entered = threading.Event()

    def record(self, rec: AuditRecord) -> None:
        self.write_threads.append(threading.current_thread().name)
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=30), "the audit write was never released"
        super().record(rec)


def make_gate(**overrides: Any) -> tuple[PublicAuthGate, StubIntrospector, RecordingSink]:
    introspector = StubIntrospector()
    sink = RecordingSink()
    gate = PublicAuthGate(
        client_id=CLIENT_ID,
        owner_subject=OWNER_SUB,
        introspector=introspector,
        audit=sink,
        **overrides,
    )
    return gate, introspector, sink


def headers(*pairs: tuple[str, str]) -> list[tuple[bytes, bytes]]:
    return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in pairs]


def http_scope(
    *,
    request_headers: list[tuple[bytes, bytes]],
    client: str = SCANNER,
    path: str = "/mcp",
    method: str = "POST",
) -> Scope:
    """What uvicorn hands the app once it has resolved the proxy's
    `X-Forwarded-For` into `client` (`public_uvicorn_options`)."""
    return {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "headers": request_headers,
        "query_string": b"",
        "server": ("127.0.0.1", 8700),
        "client": (client, 0),
        "scheme": "http",
        "http_version": "1.1",
    }


async def send_request(
    app: Callable[[Scope, Receive, Send], Awaitable[None]],
    scope: Scope,
    *,
    body: bytes = b"{}",
) -> tuple[int | None, list[tuple[bytes, bytes]], bytes]:
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
    """Stands in for the MCP session manager."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        del receive
        self.calls += 1
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})


def guarded(gate: PublicAuthGate, inner: Any = None) -> tuple[TransportGuardASGIApp, Any]:
    inner_app = RecordingInnerApp() if inner is None else inner
    app = TransportGuardASGIApp(
        inner_app, gate=gate, allowed_hosts=[ALLOWED_HOST], allowed_origins=[ALLOWED_ORIGIN]
    )
    return app, inner_app


def request_with(
    authorization: str | None, *, client: str = SCANNER
) -> Scope:
    pairs: list[tuple[str, str]] = [("host", ALLOWED_HOST)]
    if authorization is not None:
        pairs.append(("authorization", authorization))
    return http_scope(request_headers=headers(*pairs), client=client)


# ===========================================================================
# 1. Token-less requests are counted, not written row by row
#    (old code: one mcp_audit row per request, 1,000 rows for 1,000 probes)
# ===========================================================================


def test_a_flood_of_token_less_requests_writes_no_audit_rows() -> None:
    gate, introspector, sink = make_gate()
    app, inner = guarded(gate)

    async def flood() -> list[int | None]:
        statuses = []
        for _ in range(1000):
            status, _, _ = await send_request(app, request_with(None))
            statuses.append(status)
        return statuses

    statuses = run(flood())

    assert sink.snapshot() == ()  # no per-request rows: nothing to judge, nothing to record
    assert sink.write_threads == []  # and no write attempted at all
    assert introspector.calls == []
    assert inner.calls == 0
    # Every refusal is still counted, by code (imsg.mcp.audit.RejectionTally):
    # ten 401s, then the address is throttled (ten failures a minute) and the
    # rest are 429s.
    counted = gate.rejection_tally.pending()
    assert sum(counted.values()) == 1000
    assert counted == {"UNAUTHORIZED": 10, "RATE_LIMITED": 990}
    assert statuses[:10] == [401] * 10 and set(statuses[10:]) == {429}


def test_a_request_whose_token_was_judged_still_gets_its_own_row() -> None:
    """The aggregate is for requests with nothing to judge. A foreign
    subject's rejection is what AT-1 step 4 reads, and it keeps its row."""
    gate, _, sink = make_gate()
    app, _ = guarded(gate)

    status, _, _ = run(send_request(app, request_with(bearer(OTHER_TOKEN))))

    assert status == 401
    (row,) = sink.snapshot()
    assert (row.subject, row.subject_ok, row.error) == (OTHER_SUB, False, "UNAUTHORIZED")


def test_transport_refusals_before_the_token_are_counted_too() -> None:
    """A wrong Host, a wrong Origin or a duplicated header is refused
    before the token is read; each still counts."""
    gate, _, sink = make_gate()
    app, _ = guarded(gate)
    wrong_host = http_scope(request_headers=headers(("host", "evil.fictional.example")))
    duplicated = http_scope(
        request_headers=headers(
            ("host", ALLOWED_HOST), ("authorization", "Bearer a"), ("authorization", "Bearer b")
        )
    )

    assert run(send_request(app, wrong_host))[0] == 403
    assert run(send_request(app, duplicated))[0] == 400

    assert sink.snapshot() == ()
    assert gate.rejection_tally.pending() == {"UNAUTHORIZED": 2}


# ===========================================================================
# 2. The per-client throttle (old code: no per-client limit, so a single
#    address kept every bad token going to Google and to the audit table)
# ===========================================================================


def test_a_client_over_its_failure_budget_is_refused_before_any_token_check() -> None:
    """The shipped budget: ten rejections a minute per address."""
    gate, introspector, sink = make_gate()
    app, _ = guarded(gate)

    statuses = [
        run(send_request(app, request_with(bearer(f"guess-{i}"))))[0] for i in range(25)
    ]

    assert statuses[:10] == [401] * 10
    assert statuses[10:] == [429] * 15
    assert len(introspector.calls) == 10  # no tokeninfo call once throttled
    assert len(sink.snapshot()) == 10  # and no audit row: counted instead
    assert gate.rejection_tally.pending() == {"RATE_LIMITED": 15}


def test_one_client_being_throttled_does_not_throttle_another() -> None:
    gate, introspector, _ = make_gate()
    app, _ = guarded(gate)
    for i in range(20):
        run(send_request(app, request_with(bearer(f"guess-{i}"), client=SCANNER)))
    assert run(send_request(app, request_with(bearer("guess-x"), client=SCANNER)))[0] == 429

    status, _, _ = run(
        send_request(app, request_with(bearer("other-guess"), client=OWNER_CLIENT))
    )

    assert status == 401  # judged on its merits, not refused for the scanner's sake
    assert introspector.calls[-1] == "other-guess"


def test_a_session_already_working_is_never_throttled_whoever_shares_its_address() -> None:
    """The owner's token, once checked, keeps working even if a scanner
    behind the same address (a shared egress) gets that address throttled."""
    gate, _, _ = make_gate()
    app, _ = guarded(gate)
    assert run(send_request(app, request_with(bearer(OWNER_TOKEN), client=SCANNER)))[0] == 200
    for i in range(20):
        run(send_request(app, request_with(bearer(f"guess-{i}"), client=SCANNER)))
    assert run(send_request(app, request_with(bearer("guess-x"), client=SCANNER)))[0] == 429

    assert run(send_request(app, request_with(bearer(OWNER_TOKEN), client=SCANNER)))[0] == 200


def test_requests_with_no_identifiable_client_share_the_global_budget() -> None:
    """No forwarded address (a direct local connection, or a proxy that did
    not say): one shared bucket, with the global budget as its limit, rather
    than each request counting as a fresh client."""
    gate, introspector, _ = make_gate(failure_budget_per_minute=5)
    app, _ = guarded(gate)

    statuses = [
        run(send_request(app, request_with(bearer(f"local-{i}"), client="127.0.0.1")))[0]
        for i in range(8)
    ]

    assert statuses == [401] * 5 + [429] * 3
    assert len(introspector.calls) == 5


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_requests_refused_while_throttled_do_not_extend_the_throttle() -> None:
    """A throttled client that keeps sending is let back in a window after
    its last real failures — the refusals themselves are not failures — so
    a client retrying through a bad minute is not kept out indefinitely."""
    from imsg.mcp.auth import Admission, Rejection

    clock = FakeClock()
    gate, introspector, _ = make_gate(clock=clock)
    for i in range(10):
        gate.admit(bearer(f"guess-{i}"), client=OWNER_CLIENT, io_allowed=True)
    for _ in range(11):  # keeps retrying every 5 s for 55 s
        clock.advance(5)
        verdict = gate.admit(bearer(OWNER_TOKEN), client=OWNER_CLIENT, io_allowed=True)
        assert isinstance(verdict, Rejection) and verdict.status == 429
    clock.advance(6)  # the ten failures are now more than 60 s old

    verdict = gate.admit(bearer(OWNER_TOKEN), client=OWNER_CLIENT, io_allowed=True)

    assert isinstance(verdict, Admission)
    assert introspector.calls[-1] == OWNER_TOKEN


def test_a_tokeninfo_outage_does_not_throttle_the_owners_address() -> None:
    """While Google cannot be reached every uncached token is answered 503.
    Those are not the client's failures: when Google is back, the owner's
    next token is checked and admitted, however many 503s came first."""
    from imsg.mcp.auth import Admission, Rejection
    from imsg.mcp.errors import IntrospectionUnavailableError

    clock = FakeClock()
    gate, introspector, _ = make_gate(clock=clock)
    introspector.outcomes[OWNER_TOKEN] = IntrospectionUnavailableError("Google unreachable")
    for _ in range(15):
        verdict = gate.admit(bearer(OWNER_TOKEN), client=OWNER_CLIENT, io_allowed=True)
        assert isinstance(verdict, Rejection) and verdict.status == 503
        clock.advance(2)
    introspector.outcomes[OWNER_TOKEN] = claims(OWNER_SUB)
    clock.advance(6)  # past the breaker's cool-down

    verdict = gate.admit(bearer(OWNER_TOKEN), client=OWNER_CLIENT, io_allowed=True)

    assert isinstance(verdict, Admission)
    assert gate.rejection_tally.pending() == {"UNAVAILABLE": 15}


# ===========================================================================
# 3. Bad tokens cannot lock the owner out (old code: 60 failures a minute,
#    from anywhere, and the owner's next token got 429 without being checked)
# ===========================================================================


def test_sixty_one_bad_tokens_from_one_address_do_not_lock_out_the_owners_new_token() -> None:
    """The QA review's verification, verbatim: 61 distinct bad tokens from
    one address, then one valid token from another; the valid token must be
    introspected and admitted."""
    gate, introspector, _ = make_gate()  # the shipped defaults
    app, inner = guarded(gate)
    for i in range(61):
        run(send_request(app, request_with(bearer(f"random-{i}"), client=SCANNER)))

    status, _, _ = run(
        send_request(app, request_with(bearer(OWNER_TOKEN), client=OWNER_CLIENT))
    )

    assert status == 200
    assert introspector.calls[-1] == OWNER_TOKEN
    assert inner.calls == 1


def test_the_global_budget_still_bounds_tokeninfo_calls_across_many_addresses() -> None:
    """D7.3's amplification guard survives: spread over many addresses,
    failures still stop reaching Google once the shared budget is spent."""
    gate, introspector, _ = make_gate(failure_budget_per_minute=10)
    app, _ = guarded(gate)
    for i in range(30):
        client = f"192.0.2.{i + 1}"  # RFC 5737 documentation range
        run(send_request(app, request_with(bearer(f"random-{i}"), client=client)))

    assert len(introspector.calls) == 10


# ===========================================================================
# 4. One rate-limit event per tool call (old code: the transport and
#    dispatch both charged, so a limit of 2 admitted one tool call)
# ===========================================================================


class _FakeService:
    def __init__(self) -> None:
        self.calls = 0

    def list_people(self, context: AccessContext, **kwargs: Any) -> dict[str, Any]:
        del context, kwargs
        self.calls += 1
        return {"people": []}

    def search_messages(self, context: AccessContext, **kwargs: Any) -> SearchMessagesResult:
        del context, kwargs
        self.calls += 1
        return SearchMessagesResult(
            results=[
                {
                    "segment_key": "seg_fictional",
                    "text": "the ferry leaves at 7:40 — bring the \U0001f6b2",
                    "score": 0.5,
                    "untrusted_content": True,
                }
            ],
            candidate_lists={"segment_fts": 1},
            scan_cap_reached=False,
        )


@pytest.fixture
def warm() -> Iterator[BackgroundWarmUp]:
    thread = ModelThread(name="test-hardening-warm")
    warm_up = BackgroundWarmUp([], model_thread=thread, log=lambda line: None)
    warm_up.start()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    yield warm_up
    thread.close()


class ToolCallingInnerApp:
    """Stands in for the MCP SDK between the transport and the tool: what
    reaches `on_call_tool` is a context whose `request` was built from the
    very scope the transport saw — which is what the SDK does
    (`mcp.server.streamable_http`: `Request(scope, receive)`)."""

    def __init__(self, server: PublicMcpServer, name: str, arguments: dict[str, Any]) -> None:
        self.server = server
        self.name = name
        self.arguments = arguments
        self.results: list[types.CallToolResult] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        del receive
        header_map = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        context = SimpleNamespace(request=SimpleNamespace(headers=header_map, scope=scope))
        result = await self.server.on_call_tool(
            cast("Any", context),
            types.CallToolRequestParams(name=self.name, arguments=self.arguments),
        )
        self.results.append(result)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def test_a_tool_call_through_the_transport_costs_one_rate_limit_event(
    warm: BackgroundWarmUp,
) -> None:
    gate, _, sink = make_gate(rate_limit_per_minute=2)
    service = _FakeService()
    server = PublicMcpServer(
        service=cast("Any", service), gate=gate, scope=cast("Any", "full"), warm_up=warm
    )
    inner = ToolCallingInnerApp(server, "list_people", {"limit": 5})
    app, _ = guarded(gate, inner)

    statuses = [
        run(send_request(app, request_with(bearer(OWNER_TOKEN), client=OWNER_CLIENT)))[0]
        for _ in range(3)
    ]

    assert statuses == [200, 200, 429]
    assert [r.is_error for r in inner.results] == [False, False]
    assert service.calls == 2
    assert [r.error for r in sink.snapshot()] == [None, None, "RATE_LIMITED"]


def test_a_rate_limit_receipt_pays_for_one_tool_call_only() -> None:
    """Two tool calls in one HTTP request (a JSON-RPC batch) pay twice: the
    transport's receipt covers the first."""
    from imsg.mcp.auth import Admission, AuthorizedRequest, ToolOutcome

    gate, _, _ = make_gate(rate_limit_per_minute=2)
    admission = gate.admit(bearer(OWNER_TOKEN), client=OWNER_CLIENT, io_allowed=True)
    assert isinstance(admission, Admission)

    def handler(request: AuthorizedRequest) -> ToolOutcome[str]:
        return ToolOutcome(payload=request.subject)

    first = gate.dispatch(
        bearer(OWNER_TOKEN), tool="t", params={}, handler=handler, charge=admission.charge
    )
    second = gate.dispatch(
        bearer(OWNER_TOKEN), tool="t", params={}, handler=handler, charge=admission.charge
    )
    third = gate.dispatch(
        bearer(OWNER_TOKEN), tool="t", params={}, handler=handler, charge=admission.charge
    )

    assert first.rejection is None  # paid by the transport
    assert second.rejection is None  # charged: the second event of two
    assert third.rejection is not None and third.rejection.code == "RATE_LIMITED"


def test_a_receipt_is_only_good_for_the_token_and_gate_that_issued_it() -> None:
    from imsg.mcp.auth import Admission, AuthorizedRequest, ToolOutcome

    gate, introspector, _ = make_gate(rate_limit_per_minute=1)
    introspector.outcomes["second-owner-token"] = claims(OWNER_SUB)
    other_gate, _, _ = make_gate(rate_limit_per_minute=1)
    admission = gate.admit(bearer(OWNER_TOKEN), client=OWNER_CLIENT, io_allowed=True)
    assert isinstance(admission, Admission)

    def handler(request: AuthorizedRequest) -> ToolOutcome[str]:
        return ToolOutcome(payload=request.subject)

    elsewhere = other_gate.dispatch(
        bearer(OWNER_TOKEN), tool="t", params={}, handler=handler, charge=admission.charge
    )
    assert elsewhere.rejection is None  # other_gate charged its own limiter
    other_token = gate.dispatch(
        bearer("second-owner-token"), tool="t", params={}, handler=handler, charge=admission.charge
    )
    assert other_token.rejection is not None  # not paid: limit 1 already used by admit
    assert other_token.rejection.code == "RATE_LIMITED"


def test_dispatch_without_a_receipt_still_charges() -> None:
    """The AT-1 probe and every other direct caller keep paying their way:
    leaving the receipt out can only over-count."""
    from imsg.mcp.auth import AuthorizedRequest, ToolOutcome

    gate, _, _ = make_gate(rate_limit_per_minute=1)

    def handler(request: AuthorizedRequest) -> ToolOutcome[str]:
        return ToolOutcome(payload=request.subject)

    assert gate.dispatch(bearer(OWNER_TOKEN), tool="t", params={}, handler=handler).rejection is None
    limited = gate.dispatch(bearer(OWNER_TOKEN), tool="t", params={}, handler=handler)
    assert limited.rejection is not None and limited.rejection.code == "RATE_LIMITED"


# ===========================================================================
# 5. The unauthenticated metadata document and uvicorn's headers
#    (old code: `resource_name: imessage-index`, two Cache-Control headers,
#    `server: uvicorn`, keep-alive 5 s)
# ===========================================================================


def _metadata_response() -> tuple[int | None, list[tuple[bytes, bytes]], bytes]:
    gate, _, _ = make_gate()
    server = PublicMcpServer(
        service=cast("Any", _FakeService()),
        gate=gate,
        scope=cast("Any", "full"),
        warm_up=cast("Any", None),
    )
    app = build_public_asgi_app(
        server,
        allowed_hosts=[ALLOWED_HOST],
        allowed_origins=[ALLOWED_ORIGIN],
        external_url=EXTERNAL_URL,
    )
    scope = http_scope(
        request_headers=headers(("host", ALLOWED_HOST)),
        path=WELL_KNOWN_METADATA_PATH,
        method="GET",
    )
    return run(send_request(app, scope, body=b""))


def test_the_metadata_document_sends_exactly_one_cache_control() -> None:
    status, response_headers, _ = _metadata_response()
    assert status == 200
    cache_controls = [v for k, v in response_headers if k.lower() == b"cache-control"]
    assert cache_controls == [b"private, no-store"]


def test_the_metadata_document_does_not_name_the_software() -> None:
    status, _, body = _metadata_response()
    assert status == 200
    document = json.loads(body)
    assert "resource_name" not in document
    assert "imessage" not in body.decode().lower()
    assert "imsg" not in body.decode().lower()
    # What a client needs from it is all still there.
    assert document["resource"].rstrip("/") == EXTERNAL_URL.rstrip("/")
    assert document["authorization_servers"] == ["https://accounts.google.com"]


class _NoDatabase:
    """What `connect` returns in the CLI tests below: nothing in them
    reaches a database (uvicorn is replaced, or serves only the metadata
    document), so any query is a test bug and fails loudly."""

    closed = False
    broken = False

    def __init__(self) -> None:
        import psycopg

        self.info = SimpleNamespace(transaction_status=psycopg.pq.TransactionStatus.IDLE)

    def cursor(self) -> Any:
        raise AssertionError("no database in this test")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def mocked_public_cli(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A config with `mcp.public` enabled and the mount and database checks
    replaced, as `tests/test_cli.py` does for `imsg mcp public`."""
    import imsg.cli as cli_module
    import imsg.config.schema as schema_module
    from imsg.mount.guard import MountInfo

    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", "300000000000000000009")
    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")
    live_chat_db = messages_dir / "chat.db"
    live_chat_db.write_text("")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  data_root: {data_root}
  live_chat_db: {live_chat_db}
database:
  dsn: postgresql://imsg@127.0.0.1:5433/imsgindex
  password: env:IMSG_TEST_PG_PASSWORD
sync:
  interval_seconds: 900
  sources:
    - name: mini
      chat_db: {live_chat_db}
embedding:
  revision: deadbeef
  query_instruction: "test instruction"
  multimodal:
    revision: cafef00d
retrieval:
  reranker_revision: f00dcafe
models:
  backend: fake
mcp:
  public:
    enabled: true
    external_url: {EXTERNAL_URL}
    allowed_origins: [{ALLOWED_ORIGIN}]
    allowed_hosts: [{ALLOWED_HOST}]
    scope: allowlist
    oauth:
      client_id: fictional-client-id.apps.example
      owner_subject: env:IMSG_TEST_OWNER_SUBJECT
export:
  gcp_project: example-project
  gcs_bucket: example-bucket
  data_store_id: example-datastore
"""
    )
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda root: MountInfo(mount_point=root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _NoDatabase())
    monkeypatch.setattr(cli_module, "verify_data_directory", lambda conn, root: root)
    return config_path


def _captured_uvicorn_options(monkeypatch: pytest.MonkeyPatch, config_path: Any) -> dict[str, Any]:
    """Run `imsg mcp public` with `uvicorn.run` replaced, and return the
    keyword arguments it was given."""
    import uvicorn
    from typer.testing import CliRunner

    from imsg.cli import app as cli_app

    captured: dict[str, Any] = {}

    def fake_run(app_arg: Any, **kw: Any) -> None:
        captured["app"] = app_arg
        captured.update(kw)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = CliRunner().invoke(cli_app, ["mcp", "public", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    return captured


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_imsg_mcp_public_runs_uvicorn_without_a_server_header_and_with_a_long_keep_alive(
    mocked_public_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = _captured_uvicorn_options(monkeypatch, mocked_public_cli)
    assert options.get("server_header") is False
    assert options.get("timeout_keep_alive") == 60
    assert options.get("proxy_headers") is True
    assert options.get("forwarded_allow_ips") == ["127.0.0.1", "::1"]


def test_a_real_uvicorn_run_with_those_options_sends_no_server_header(
    mocked_public_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header is added by uvicorn's protocol layer, not the app, so
    only a real server shows it: run one on a free loopback port with the
    options `imsg mcp public` passes, and read a real response."""
    import uvicorn

    options = _captured_uvicorn_options(monkeypatch, mocked_public_cli)
    app = options.pop("app")
    port = _free_port()
    options.update({"host": "127.0.0.1", "port": port, "log_level": "warning", "lifespan": "off"})
    server = uvicorn.Server(uvicorn.Config(app, **options))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn never started"
            time.sleep(0.02)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{WELL_KNOWN_METADATA_PATH}",
            headers={"Host": "mcp.fictional.example"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response_headers = response.headers
            status = response.status
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert status == 200
    assert response_headers.get("server") is None
    assert response_headers.get_all("cache-control") == ["private, no-store"]


# ===========================================================================
# 6. Token checks and audit writes run off the event loop (old code: the
#    guard called the gate inline, so a slow tokeninfo answer or audit write
#    stopped every other request)
# ===========================================================================

HOLD_SECONDS = 1.0


def test_a_slow_tokeninfo_answer_does_not_stop_other_requests() -> None:
    """While one request waits on Google, a request the gate can answer from
    memory is answered. Released from a timer thread, so on the old code
    (loop blocked) the test finishes with the wrong order instead of
    hanging."""
    gate, introspector, _ = make_gate()
    app, _ = guarded(gate)
    # The owner's token is checked once, so its verdict is cached...
    assert run(send_request(app, request_with(bearer(OWNER_TOKEN), client=OWNER_CLIENT)))[0] == 200
    # ...and the next unknown token will hang in tokeninfo.
    introspector.hold = threading.Event()
    introspector.entered.clear()
    timer = threading.Timer(HOLD_SECONDS, introspector.hold.set)
    finished: list[str] = []

    async def slow() -> None:
        status, _, _ = await send_request(app, request_with(bearer("slow-new-token")))
        finished.append(f"slow:{status}")

    async def fast() -> None:
        status, _, _ = await send_request(
            app, request_with(bearer(OWNER_TOKEN), client=OWNER_CLIENT)
        )
        finished.append(f"fast:{status}")

    async def main() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(slow)
            for _ in range(200):
                if introspector.entered.is_set():
                    break
                await anyio.sleep(0.01)
            await fast()

    timer.start()
    try:
        anyio.run(main)
    finally:
        introspector.hold.set()
        timer.cancel()
        timer.join()

    assert finished == ["fast:200", "slow:401"]
    assert introspector.threads[-1] != threading.main_thread().name


def test_a_slow_audit_write_does_not_stop_other_requests() -> None:
    gate, _, sink = make_gate()
    app, _ = guarded(gate)
    assert run(send_request(app, request_with(bearer(OWNER_TOKEN), client=OWNER_CLIENT)))[0] == 200
    sink.hold = threading.Event()
    sink.entered.clear()
    timer = threading.Timer(HOLD_SECONDS, sink.hold.set)
    finished: list[str] = []

    async def slow() -> None:
        # A foreign subject's rejection is written row by row — the slow part.
        status, _, _ = await send_request(app, request_with(bearer(OTHER_TOKEN)))
        finished.append(f"slow:{status}")

    async def fast() -> None:
        status, _, _ = await send_request(
            app, request_with(bearer(OWNER_TOKEN), client=OWNER_CLIENT)
        )
        finished.append(f"fast:{status}")

    async def main() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(slow)
            for _ in range(200):
                if sink.entered.is_set():
                    break
                await anyio.sleep(0.01)
            await fast()

    timer.start()
    try:
        anyio.run(main)
    finally:
        sink.hold.set()
        timer.cancel()
        timer.join()

    assert finished == ["fast:200", "slow:401"]
    assert sink.write_threads[-1] != threading.main_thread().name


def test_token_checks_that_need_a_thread_are_bounded() -> None:
    """At most `MAX_CONCURRENT_AUTH_CHECKS` token checks wait on Google at
    once; the rest queue for a thread rather than piling up. (Read with a
    default because the code before this change had no such bound: it ran
    every check on the event loop, one at a time.)"""
    from imsg.mcp.tools import public_server

    bound = getattr(public_server, "MAX_CONCURRENT_AUTH_CHECKS", 4)
    gate, introspector, _ = make_gate()
    app, _ = guarded(gate)
    inside = 0
    high_water = 0
    lock = threading.Lock()
    real = introspector.introspect

    def counting(token: str) -> TokenIntrospection:
        nonlocal inside, high_water
        with lock:
            inside += 1
            high_water = max(high_water, inside)
        try:
            time.sleep(0.05)
            return real(token)
        finally:
            with lock:
                inside -= 1

    introspector.introspect = counting  # type: ignore[method-assign]

    async def main() -> None:
        async with anyio.create_task_group() as group:
            for i in range(bound * 4):
                client = f"192.0.2.{i + 1}"  # one address each: nobody is throttled
                group.start_soon(send_request, app, request_with(bearer(f"parallel-{i}"), client=client))

    anyio.run(main)

    assert len(introspector.calls) == bound * 4
    assert 1 < high_water <= bound


# ===========================================================================
# 7. The payload still goes out twice, on purpose
# ===========================================================================


def test_a_tool_result_carries_structured_content_and_the_same_json_as_text(
    warm: BackgroundWarmUp,
) -> None:
    """The MCP specification: a tool returning structured content SHOULD
    also return it serialized in a text block, for clients that read only
    `content`. Gemini Enterprise's documentation does not say which it
    reads, and at least one client breaks without the text block, so both
    stay — byte for byte what they were (payload compatibility)."""
    gate, _, _ = make_gate()
    server = PublicMcpServer(
        service=cast("Any", _FakeService()), gate=gate, scope=cast("Any", "full"), warm_up=warm
    )
    context = SimpleNamespace(request=SimpleNamespace(headers={"authorization": bearer(OWNER_TOKEN)}))

    result = run(
        server.on_call_tool(
            cast("Any", context),
            types.CallToolRequestParams(name="search_messages", arguments={"query": "ferry"}),
        )
    )

    assert result.is_error is False
    assert result.structured_content is not None
    (block,) = result.content
    assert isinstance(block, types.TextContent)
    assert block.text == json.dumps(result.structured_content, default=str)
    assert json.loads(block.text) == result.structured_content
