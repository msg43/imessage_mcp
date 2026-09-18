"""The public MCP server answers before its models are warm.

`imsg mcp public` used to build its providers and then serve with them
cold: the first request after every restart — and launchd restarts this
agent on every crash, `KeepAlive` — paid the whole model load inside the
request. The local surface already had the remedy
(`imsg.retrieval.background_warm_up`); this is the same mechanism on the
public surface, with the public surface's own differences respected:

- the wait a request spends on the warm-up is bounded well inside the
  time a hosted client (and the tunnel in front of it) will hold an HTTP
  request open, and ends in a retryable `WARMING_UP` tool error;
- the audit row keeps `WARMING_UP`/`WARM_UP_FAILED` rather than
  collapsing to `INTERNAL`;
- **the auth gate does not move.** Every test below that touches the
  warm-up also asserts the gate still rejects — an unauthenticated
  request must never reach a handler, a warm-up wait, or a tool error
  that says anything about the server's state.

Warm-ups here are fake steps held on an event, so "during warm-up" is a
state the test controls rather than a race.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import mcp.types as types
import pytest
from starlette.types import Message, Receive, Scope, Send

from imsg.errors import ProviderUnavailableError
from imsg.mcp.audit import ALLOWED_ERROR_CODES, MemoryAuditSink
from imsg.mcp.auth import PublicAuthGate, TokenIntrospection
from imsg.mcp.errors import TokenInvalidError
from imsg.mcp.tools.local_server import (
    TOOL_CALL_WARM_UP_WAIT_SECONDS as LOCAL_WARM_UP_WAIT_SECONDS,
)
from imsg.mcp.tools.public_server import (
    TOOL_CALL_WARM_UP_WAIT_SECONDS,
    PublicMcpServer,
    TransportGuardASGIApp,
)
from imsg.mcp.warm_up_readiness import (
    WarmUpReadinessFile,
    read_warm_up_readiness,
    readiness_path,
)
from imsg.retrieval.access import AccessContext
from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpPhase, WarmUpStep
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import SearchMessagesResult

OWNER_SUB = "300000000000000000003"
OTHER_SUB = "400000000000000000004"
CLIENT_ID = "111111111111-fictional.apps.example"
OWNER_TOKEN = "fictional-public-owner-token"
OTHER_TOKEN = "fictional-public-foreign-token"

ALLOWED_HOST = "mcp.fictional.example"
ALLOWED_ORIGIN = "https://vertexaisearch.fictional.example"


def run[T](awaitable: Awaitable[T]) -> T:
    return asyncio.run(cast("Coroutine[Any, Any, T]", awaitable))


def bearer(token: str) -> str:
    return f"Bearer {token}"


class StubIntrospector:
    """Same shape as `tests/test_mcp_public_server.py`'s — file-local, per
    this repo's per-file fixture style."""

    def __init__(self) -> None:
        self.outcomes: dict[str, TokenIntrospection] = {
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

    def introspect(self, token: str) -> TokenIntrospection:
        outcome = self.outcomes.get(token)
        if outcome is None:
            raise TokenInvalidError("unknown token")
        return outcome


class FakeRetrievalService:
    """Duck-typed `RetrievalService` that records that it ran at all —
    "the service never ran" is the assertion most of these tests turn on."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, AccessContext]] = []

    def search_messages(self, context: AccessContext, **kwargs: Any) -> SearchMessagesResult:
        self.calls.append(("search_messages", context))
        return SearchMessagesResult(results=[], candidate_lists={}, scan_cap_reached=False)

    def list_people(self, context: AccessContext, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_people", context))
        return {"people": []}


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-public-server")
    yield thread
    thread.close()


class Harness:
    """A `PublicMcpServer` whose single warm-up step is held on an event."""

    def __init__(
        self,
        model_thread: ModelThread,
        *,
        wait_seconds: float,
        failure: Exception | None = None,
        held: bool = True,
        scope: str = "allowlist",
    ) -> None:
        self.release = threading.Event()
        if not held:
            self.release.set()
        self.log: list[str] = []
        self.published: list[str] = []

        def step() -> None:
            assert self.release.wait(timeout=30), "the warm-up was never released"
            if failure is not None:
                raise failure

        self.warm_up = BackgroundWarmUp(
            [WarmUpStep("reranker", 20.0, step)],
            model_thread=model_thread,
            log=self.log.append,
            on_status=lambda status: self.published.append(status.phase.value),
        )
        self.audit = MemoryAuditSink()
        self.gate = PublicAuthGate(
            client_id=CLIENT_ID,
            owner_subject=OWNER_SUB,
            introspector=StubIntrospector(),
            audit=self.audit,
        )
        self.service = FakeRetrievalService()
        self.server = PublicMcpServer(
            service=cast(Any, self.service),
            gate=self.gate,
            scope=cast(Any, scope),
            warm_up=self.warm_up,
            warm_up_wait_seconds=wait_seconds,
        )

    async def call(
        self, name: str, arguments: dict[str, Any], *, authorization: str | None = None
    ) -> types.CallToolResult:
        header = bearer(OWNER_TOKEN) if authorization is None else authorization
        return await self.server.on_call_tool(
            cast(Any, fake_context(header)),
            types.CallToolRequestParams(name=name, arguments=arguments),
        )


def fake_context(authorization: str | None) -> Any:
    request_headers: dict[str, str] = {}
    if authorization is not None:
        request_headers["authorization"] = authorization
    return SimpleNamespace(request=SimpleNamespace(headers=request_headers))


def text_of(result: types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


def headers(*pairs: tuple[str, str]) -> list[tuple[bytes, bytes]]:
    return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in pairs]


class RecordingInnerApp:
    """Stands in for the MCP session manager: records whether the guard
    ever let a request through."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        del receive
        self.called = True
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})


async def send_request(
    app: Callable[[Scope, Receive, Send], Awaitable[None]],
    *,
    request_headers: list[tuple[bytes, bytes]],
    path: str = "/mcp",
) -> int | None:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
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
        return {"type": "http.request", "body": b"{}", "more_body": False}

    status: int | None = None

    async def send(message: Message) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]

    await app(scope, receive, send)
    return status


def guarded(harness: Harness) -> tuple[TransportGuardASGIApp, RecordingInnerApp]:
    inner = RecordingInnerApp()
    app = TransportGuardASGIApp(
        inner, gate=harness.gate, allowed_hosts=[ALLOWED_HOST], allowed_origins=[ALLOWED_ORIGIN]
    )
    return app, inner


# ---------------------------------------------------------------------------
# The bound itself
# ---------------------------------------------------------------------------


def test_the_public_wait_bound_is_far_shorter_than_the_local_surfaces() -> None:
    """The local surface holds a stdio call for 90 s because nothing in
    that path times out. A public request is held open across a
    Cloudflare tunnel whose edge gives up at 125 s, so the bound has to
    leave the query itself room inside that — and the hosted client's own
    timeout is undocumented, which argues for shorter still."""
    assert 0 < TOOL_CALL_WARM_UP_WAIT_SECONDS < LOCAL_WARM_UP_WAIT_SECONDS
    assert TOOL_CALL_WARM_UP_WAIT_SECONDS <= 30


# ---------------------------------------------------------------------------
# The transport answers before the models are ready
# ---------------------------------------------------------------------------


def test_tools_list_is_answered_while_the_models_are_still_loading(
    model_thread: ModelThread,
) -> None:
    harness = Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()

    async def main() -> types.ListToolsResult:
        with anyio.fail_after(1.0):
            return await harness.server.on_list_tools(cast(Any, None), None)

    assert len(anyio.run(main).tools) == 4
    assert harness.warm_up.status().phase is WarmUpPhase.WARMING
    harness.release.set()


def test_the_transport_guard_serves_authenticated_requests_during_warm_up(
    model_thread: ModelThread,
) -> None:
    """Nothing in the request path is gated on the warm-up — only the
    retrieval tools are."""
    harness = Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()
    app, inner = guarded(harness)

    began = time.monotonic()
    status = run(
        send_request(
            app,
            request_headers=headers(
                ("host", ALLOWED_HOST), ("authorization", bearer(OWNER_TOKEN))
            ),
        )
    )
    assert status == 200
    assert inner.called is True
    assert time.monotonic() - began < 5.0
    assert harness.warm_up.status().phase is WarmUpPhase.WARMING
    harness.release.set()


def test_the_event_loop_keeps_answering_while_a_tool_call_waits(
    model_thread: ModelThread,
) -> None:
    harness = Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()
    finished: list[str] = []

    async def waiting_call() -> None:
        await harness.call("search_messages", {"query": "kite festival"})
        finished.append("search_messages")

    async def main() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(waiting_call)
            await anyio.sleep(0.05)
            with anyio.fail_after(1.0):
                await harness.server.on_list_tools(cast(Any, None), None)
            finished.append("tools/list")
            harness.release.set()

    anyio.run(main)
    assert finished == ["tools/list", "search_messages"]


# ---------------------------------------------------------------------------
# A request that arrives during warm-up
# ---------------------------------------------------------------------------


def test_a_tool_call_during_warm_up_waits_for_it_then_succeeds(
    model_thread: ModelThread,
) -> None:
    harness = Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()
    timer = threading.Timer(0.3, harness.release.set)
    timer.start()

    began = time.monotonic()
    result = anyio.run(harness.call, "search_messages", {"query": "kite festival"})
    waited = time.monotonic() - began
    timer.join()

    assert not result.is_error, text_of(result)
    assert [name for name, _ in harness.service.calls] == ["search_messages"]
    assert 0.3 <= waited < 10.0
    (row,) = harness.audit.snapshot()
    assert (row.tool, row.error, row.subject_ok) == ("search_messages", None, True)


def test_a_tool_call_that_outwaits_the_bound_gets_a_retryable_warming_up_error(
    model_thread: ModelThread,
) -> None:
    harness = Harness(model_thread, wait_seconds=0.2)
    harness.warm_up.start()

    began = time.monotonic()
    result = anyio.run(harness.call, "search_messages", {"query": "kite festival"})
    waited = time.monotonic() - began

    assert result.is_error
    code, message = text_of(result).split("\n", 1)
    assert code == "WARMING_UP"
    assert "Retry this call" in message
    assert "about " in message and " s remaining" in message
    assert "Traceback" not in message
    assert 0.2 <= waited < 5.0
    assert harness.service.calls == []  # the service never ran

    # The audit row keeps its own code — a retryable warm-up must not read
    # as an internal failure in the table AT-1 is judged on.
    (row,) = harness.audit.snapshot()
    assert row.error == "WARMING_UP"
    assert row.error in ALLOWED_ERROR_CODES
    assert row.subject == OWNER_SUB and row.subject_ok is True

    harness.release.set()
    assert harness.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    retried = anyio.run(harness.call, "search_messages", {"query": "kite festival"})
    assert not retried.is_error, text_of(retried)


def test_arguments_that_fail_the_schema_are_answered_without_waiting(
    model_thread: ModelThread,
) -> None:
    harness = Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()

    began = time.monotonic()
    result = anyio.run(harness.call, "search_messages", {"limit": 5})
    assert time.monotonic() - began < 5.0
    assert result.is_error
    assert text_of(result).startswith("INVALID_ARGUMENT\n")
    harness.release.set()


def test_an_unknown_tool_is_answered_without_waiting(model_thread: ModelThread) -> None:
    harness = Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()

    began = time.monotonic()
    result = anyio.run(harness.call, "run_sql", {})
    assert time.monotonic() - began < 5.0
    assert result.is_error
    assert text_of(result).startswith("INVALID_ARGUMENT\n")
    harness.release.set()


# ---------------------------------------------------------------------------
# A warm-up that failed
# ---------------------------------------------------------------------------


def test_a_failed_warm_up_is_reported_on_every_subsequent_request(
    model_thread: ModelThread,
) -> None:
    harness = Harness(
        model_thread,
        wait_seconds=30.0,
        failure=ProviderUnavailableError("reranker weights not found"),
        held=False,
    )
    harness.warm_up.start()
    assert harness.warm_up.wait(timeout=5).phase is WarmUpPhase.FAILED

    for name, arguments in [
        ("search_messages", {"query": "kite festival"}),
        ("list_people", {}),
        ("search_messages", {"query": "deck rebuild"}),
    ]:
        began = time.monotonic()
        result = anyio.run(harness.call, name, arguments)
        assert time.monotonic() - began < 5.0  # a settled failure is not waited on
        assert result.is_error
        code, message = text_of(result).split("\n", 1)
        assert code == "WARM_UP_FAILED"
        assert "(reranker: reranker weights not found)" in message
        assert "restart the MCP server" in message

    assert harness.service.calls == []  # nothing served on half-loaded models
    assert [row.error for row in harness.audit.snapshot()] == ["WARM_UP_FAILED"] * 3
    assert sum("FAILED" in line for line in harness.log) == 1  # logged once, not per call


# ---------------------------------------------------------------------------
# The gate does not move (the one property that must not regress)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_headers", "expected"),
    [
        (headers(("host", ALLOWED_HOST)), 401),
        (headers(("host", ALLOWED_HOST), ("authorization", bearer(OTHER_TOKEN))), 401),
        (headers(("host", ALLOWED_HOST), ("authorization", "Bearer nonsense")), 401),
        (headers(("host", "evil.fictional.example"), ("authorization", bearer(OWNER_TOKEN))), 403),
        (
            headers(
                ("host", ALLOWED_HOST),
                ("origin", "https://evil.fictional.example"),
                ("authorization", bearer(OWNER_TOKEN)),
            ),
            403,
        ),
        (
            headers(
                ("host", ALLOWED_HOST),
                ("authorization", bearer(OWNER_TOKEN)),
                ("authorization", bearer(OWNER_TOKEN)),
            ),
            400,
        ),
    ],
)
def test_the_gate_still_rejects_during_warm_up(
    model_thread: ModelThread, request_headers: list[tuple[bytes, bytes]], expected: int
) -> None:
    """Warming up changes nothing about who gets in. The rejection is the
    same status, at the same layer, before the session manager (and so
    before any warm-up wait) ever runs."""
    harness = Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()
    app, inner = guarded(harness)

    began = time.monotonic()
    status = run(send_request(app, request_headers=request_headers))

    assert status == expected
    assert inner.called is False
    assert time.monotonic() - began < 5.0  # rejected outright, never parked on the warm-up
    assert harness.service.calls == []
    assert harness.warm_up.status().phase is WarmUpPhase.WARMING
    harness.release.set()


def test_the_gate_still_rejects_a_failed_warm_up_server(model_thread: ModelThread) -> None:
    """A server whose models failed still refuses strangers first — it
    does not leak `WARM_UP_FAILED` (or anything else about itself) to an
    unauthenticated caller."""
    harness = Harness(
        model_thread,
        wait_seconds=1.0,
        failure=ProviderUnavailableError("reranker weights not found"),
        held=False,
    )
    harness.warm_up.start()
    assert harness.warm_up.wait(timeout=5).phase is WarmUpPhase.FAILED
    app, inner = guarded(harness)

    assert run(send_request(app, request_headers=headers(("host", ALLOWED_HOST)))) == 401
    assert inner.called is False


def test_a_foreign_subject_is_rejected_at_the_dispatcher_during_warm_up_too(
    model_thread: ModelThread,
) -> None:
    """Defence in depth: even reaching `on_call_tool` directly (which the
    transport guard makes unreachable without a token) a foreign subject
    gets a rejection, not a warm-up answer and not a payload."""
    harness = Harness(model_thread, wait_seconds=0.2)
    harness.warm_up.start()

    for authorization in [bearer(OTHER_TOKEN), "Bearer nonsense", ""]:
        result = anyio.run(
            lambda auth=authorization: harness.call(  # type: ignore[misc]
                "search_messages", {"query": "kite festival"}, authorization=auth
            )
        )
        assert result.is_error
        assert text_of(result).startswith("UNAUTHORIZED\n")
        assert harness.service.calls == []

    assert all(row.subject_ok is False for row in harness.audit.snapshot())
    harness.release.set()


# ---------------------------------------------------------------------------
# Readiness an operator can reach without an unauthenticated endpoint
# ---------------------------------------------------------------------------


def test_the_warm_up_publishes_every_phase_it_passes_through(
    model_thread: ModelThread,
) -> None:
    harness = Harness(model_thread, wait_seconds=1.0, held=False)
    harness.warm_up.start()
    assert harness.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    assert harness.published[0] == "warming"
    assert harness.published[-1] == "ready"


def test_a_readiness_file_round_trips_the_warm_ups_phase(tmp_path: Path) -> None:
    thread = ModelThread(name="test-readiness")
    try:
        publisher = WarmUpReadinessFile(readiness_path(tmp_path))
        warm_up = BackgroundWarmUp(
            [WarmUpStep("reranker", 20.0, lambda: None)],
            model_thread=thread,
            log=lambda line: None,
            on_status=publisher.publish,
        )
        warm_up.start()
        assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    finally:
        thread.close()

    report = read_warm_up_readiness(tmp_path)
    assert report.state == "ready"
    assert report.pid == os.getpid()
    assert report.failure is None


def test_a_readiness_file_whose_process_is_gone_reads_as_not_running(tmp_path: Path) -> None:
    thread = ModelThread(name="test-readiness-dead")
    try:
        publisher = WarmUpReadinessFile(readiness_path(tmp_path), pid=_a_pid_that_is_not_running())
        warm_up = BackgroundWarmUp(
            [WarmUpStep("reranker", 20.0, lambda: None)],
            model_thread=thread,
            log=lambda line: None,
            on_status=publisher.publish,
        )
        warm_up.start()
        assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    finally:
        thread.close()

    report = read_warm_up_readiness(tmp_path)
    assert report.state == "not_running"
    assert "ready" in report.detail


def test_no_readiness_file_reads_as_not_running_rather_than_ready(tmp_path: Path) -> None:
    report = read_warm_up_readiness(tmp_path)
    assert report.state == "not_running"
    assert report.pid is None


def test_an_unreadable_readiness_file_is_not_reported_as_ready(tmp_path: Path) -> None:
    path = readiness_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    report = read_warm_up_readiness(tmp_path)
    assert report.state == "unreadable"


def test_a_failed_warm_up_is_published_with_its_cause(tmp_path: Path) -> None:
    thread = ModelThread(name="test-readiness-failed")

    def boom() -> None:
        raise ProviderUnavailableError("reranker weights not found")

    try:
        publisher = WarmUpReadinessFile(readiness_path(tmp_path))
        warm_up = BackgroundWarmUp(
            [WarmUpStep("reranker", 20.0, boom)],
            model_thread=thread,
            log=lambda line: None,
            on_status=publisher.publish,
        )
        warm_up.start()
        assert warm_up.wait(timeout=5).phase is WarmUpPhase.FAILED
    finally:
        thread.close()

    report = read_warm_up_readiness(tmp_path)
    assert report.state == "failed"
    assert report.failure is not None
    assert "reranker weights not found" in report.failure


def test_a_publisher_that_cannot_write_never_breaks_the_warm_up(tmp_path: Path) -> None:
    """The readiness file is an operator convenience; the warm-up is
    load-bearing. A read-only volume must cost the second, not the first."""
    thread = ModelThread(name="test-readiness-unwritable")
    unwritable = tmp_path / "not-a-directory"
    unwritable.write_text("", encoding="utf-8")
    try:
        publisher = WarmUpReadinessFile(readiness_path(unwritable))
        warm_up = BackgroundWarmUp(
            [WarmUpStep("reranker", 20.0, lambda: None)],
            model_thread=thread,
            log=lambda line: None,
            on_status=publisher.publish,
        )
        warm_up.start()
        assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    finally:
        thread.close()


def _a_pid_that_is_not_running() -> int:
    """A pid nothing holds. Walks down from a high number rather than
    picking one, so a busy machine cannot make this flaky."""
    for candidate in range(4_000_000, 3_900_000, -1):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except OSError:
            continue
    raise AssertionError("no free pid found")


def test_a_warming_estimate_ticks_down_between_writes(tmp_path: Path) -> None:
    """The file is written only when the phase or step changes, so a long
    step would otherwise leave 'about 91 s remaining' saying 91 s a minute
    later. The reader ages it against the timestamp in the file."""
    path = readiness_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = datetime.now(UTC) - timedelta(seconds=40)
    path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "phase": "warming",
                "loading": "multimodal text tower",
                "steps_done": 1,
                "steps_total": 4,
                "elapsed_seconds": 6.0,
                "seconds_remaining": 91,
                "failure": None,
                "updated_at": written.isoformat(timespec="seconds"),
            }
        ),
        encoding="utf-8",
    )
    report = read_warm_up_readiness(tmp_path)
    assert report.state == "warming"
    remaining = int(report.detail.split("about ", 1)[1].split(" s remaining")[0])
    assert 45 <= remaining <= 55, report.detail


def test_an_aged_estimate_never_goes_negative(tmp_path: Path) -> None:
    path = readiness_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = datetime.now(UTC) - timedelta(hours=3)
    path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "phase": "warming",
                "loading": "reranker",
                "steps_done": 2,
                "steps_total": 4,
                "elapsed_seconds": 6.0,
                "seconds_remaining": 20,
                "failure": None,
                "updated_at": written.isoformat(timespec="seconds"),
            }
        ),
        encoding="utf-8",
    )
    assert "about 1 s remaining" in read_warm_up_readiness(tmp_path).detail
