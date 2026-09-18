"""The public MCP surface serves concurrent requests.

`PublicMcpServer.on_call_tool` is `async`, but everything it does after
the warm-up wait — `PublicAuthGate.dispatch`, and through it the whole
retrieval query: FTS, vector search, embedding, reranking — is
synchronous. Run directly in the coroutine, that work occupies the
uvicorn event loop for the entire query (0.96-1.87 s measured against
the live corpus, 2026-09-18), during which the process cannot make
progress on anything else: no second tool call, no `tools/list`, no
`ping`, and no other request's warm-up wait can tick.

The local surface never showed this — stdio, one client, one call at a
time — but the public surface exists to serve a hosted assistant over
StreamableHTTP, where concurrent tool calls are ordinary.

**Why the blocked call is released by a timer thread rather than by the
test's own coroutine.** When the dispatch runs on the event loop,
nothing else on that loop runs, so `anyio.fail_after` cannot fire and a
coroutine cannot release anything: a test written that way would hang
instead of failing. Every hold here is released from an OS thread, so
the failing case finishes and reports an ordering that is wrong rather
than no result at all.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Iterator
from types import SimpleNamespace
from typing import Any, cast

import anyio
import mcp.types as types
import pytest

from imsg.mcp.audit import MemoryAuditSink
from imsg.mcp.auth import PublicAuthGate, TokenIntrospection
from imsg.mcp.errors import TokenInvalidError
from imsg.mcp.tools.public_server import MAX_CONCURRENT_TOOL_CALLS, PublicMcpServer
from imsg.retrieval.access import AccessContext
from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpStep
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import SearchMessagesResult

OWNER_SUB = "300000000000000000003"
CLIENT_ID = "111111111111-fictional.apps.example"
OWNER_TOKEN = "fictional-public-owner-token"

HOLD_SECONDS = 1.0
"""How long the first tool call is held inside the retrieval service.

Long enough that a second request answered only *after* it is
unmistakably ordered after it, short enough that the failing case costs
about a second."""


class StubIntrospector:
    """Same shape as `tests/test_mcp_public_server.py`'s — file-local, per
    this repo's per-file fixture style."""

    def introspect(self, token: str) -> TokenIntrospection:
        if token != OWNER_TOKEN:
            raise TokenInvalidError("unknown token")
        return TokenIntrospection(
            subject=OWNER_SUB,
            audience=CLIENT_ID,
            authorized_party=None,
            scopes=frozenset({"openid"}),
            expires_in_seconds=3600,
        )


class HoldingRetrievalService:
    """Duck-typed `RetrievalService` whose `search_messages` blocks on an
    event — the stand-in for a real query's seconds of synchronous FTS,
    vector search, embedding and reranking. `list_people` returns at
    once, so a second call's progress is measured against the first
    call's hold rather than against another hold."""

    def __init__(self) -> None:
        self.entered_search = threading.Event()
        self.release_search = threading.Event()
        self.list_people_calls = 0

    def search_messages(self, context: AccessContext, **kwargs: Any) -> SearchMessagesResult:
        del context, kwargs
        self.entered_search.set()
        assert self.release_search.wait(timeout=HOLD_SECONDS + 30), "the query was never released"
        return SearchMessagesResult(results=[], candidate_lists={}, scan_cap_reached=False)

    def list_people(self, context: AccessContext, **kwargs: Any) -> dict[str, Any]:
        del context, kwargs
        self.list_people_calls += 1
        return {"people": []}


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-public-concurrency")
    yield thread
    thread.close()


class CountingRetrievalService:
    """Records how many calls are inside the service at the same moment —
    the observable that says whether the concurrency bound is real."""

    def __init__(self, dwell_seconds: float = 0.05) -> None:
        self._lock = threading.Lock()
        self._inside = 0
        self.high_water = 0
        self.calls = 0
        self._dwell = dwell_seconds

    def list_people(self, context: AccessContext, **kwargs: Any) -> dict[str, Any]:
        del context, kwargs
        with self._lock:
            self._inside += 1
            self.calls += 1
            self.high_water = max(self.high_water, self._inside)
        try:
            time.sleep(self._dwell)
        finally:
            with self._lock:
                self._inside -= 1
        return {"people": []}


class Harness:
    """A `PublicMcpServer` whose warm-up has already settled, so the only
    thing a tool call waits on is the retrieval service itself."""

    def __init__(
        self,
        model_thread: ModelThread,
        *,
        service: Any = None,
        rate_limit_per_minute: int = 60,
    ) -> None:
        self.service = HoldingRetrievalService() if service is None else service
        self.warm_up = BackgroundWarmUp(
            [WarmUpStep("reranker", 0.0, lambda: None)],
            model_thread=model_thread,
            log=lambda _line: None,
        )
        self.warm_up.start()
        assert self.warm_up.wait(timeout=30.0).settled
        self.audit = MemoryAuditSink()
        self.gate = PublicAuthGate(
            client_id=CLIENT_ID,
            owner_subject=OWNER_SUB,
            introspector=StubIntrospector(),
            audit=self.audit,
            rate_limit_per_minute=rate_limit_per_minute,
        )
        self.server = PublicMcpServer(
            service=cast("Any", self.service),
            gate=self.gate,
            scope=cast("Any", "allowlist"),
            warm_up=self.warm_up,
        )

    def call(self, name: str, arguments: dict[str, Any]) -> Awaitable[types.CallToolResult]:
        context = SimpleNamespace(
            request=SimpleNamespace(headers={"authorization": f"Bearer {OWNER_TOKEN}"})
        )
        return self.server.on_call_tool(
            cast("Any", context), types.CallToolRequestParams(name=name, arguments=arguments)
        )


def test_a_query_in_flight_does_not_stop_the_server_answering(
    model_thread: ModelThread,
) -> None:
    """The whole point: while one `search_messages` is running, a second
    tool call and a `tools/list` both complete. Ordering is the
    assertion — `search_messages` finishes last because its hold is
    released on a timer, and everything that can only run on a free
    event loop finishes before it."""
    harness = Harness(model_thread)
    finished: list[str] = []
    timer = threading.Timer(HOLD_SECONDS, harness.service.release_search.set)
    timer.start()

    async def held_call() -> None:
        result = await harness.call("search_messages", {"query": "kite festival"})
        assert result.is_error is False
        finished.append("search_messages")

    async def main() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(held_call)
            # Let `held_call` reach the service before measuring anything
            # against it; on a free loop this returns in one tick.
            for _ in range(200):
                if harness.service.entered_search.is_set():
                    break
                await anyio.sleep(0.01)
            assert harness.service.entered_search.is_set(), "the held query never started"

            listed = await harness.server.on_list_tools(cast("Any", None), None)
            assert len(listed.tools) == 4
            finished.append("tools/list")

            second = await harness.call("list_people", {"limit": 5})
            assert second.is_error is False
            finished.append("list_people")

    try:
        anyio.run(main)
    finally:
        harness.service.release_search.set()
        timer.cancel()
        timer.join()

    assert finished == ["tools/list", "list_people", "search_messages"]
    assert harness.service.list_people_calls == 1


def test_the_concurrency_bound_is_a_small_number_this_module_chose() -> None:
    """Not anyio's shared default of 40, and not so large that a queued
    request outlives the tunnel's 125 s ceiling at ~2 s a query."""
    assert 1 < MAX_CONCURRENT_TOOL_CALLS <= 8
    default_bound = PublicMcpServer.__dataclass_fields__["max_concurrent_tool_calls"].default
    assert default_bound == MAX_CONCURRENT_TOOL_CALLS


def test_no_more_tool_calls_run_at_once_than_the_bound_allows(
    model_thread: ModelThread,
) -> None:
    """`MAX_CONCURRENT_TOOL_CALLS` is what keeps this surface off anyio's
    process-wide default thread limiter, so it has to be the number that
    actually binds."""
    bound = 3
    counting = CountingRetrievalService()
    harness = Harness(model_thread, service=counting)
    harness.server.max_concurrent_tool_calls = bound

    async def one() -> None:
        await harness.call("list_people", {"limit": 5})

    async def main() -> None:
        async with anyio.create_task_group() as group:
            for _ in range(bound * 4):
                group.start_soon(one)

    anyio.run(main)

    assert counting.calls == bound * 4
    assert counting.high_water <= bound
    # And the bound is really being reached, so `<=` above is not passing
    # for the trivial reason that nothing ever overlapped.
    assert counting.high_water > 1


def test_the_per_subject_rate_limit_is_not_bypassable_by_concurrency(
    model_thread: ModelThread,
) -> None:
    """`PublicAuthGate` now runs on several threads at once. Its rate
    limiter is the one piece of gate state where a lost update would be
    a security regression rather than a glitch: a check-then-record race
    would let more calls through than the configured limit. Every call
    is audited either way — an unaudited request is never served."""
    limit = 5
    attempts = 20
    counting = CountingRetrievalService(dwell_seconds=0.01)
    harness = Harness(model_thread, service=counting, rate_limit_per_minute=limit)

    outcomes: list[str] = []

    async def one() -> None:
        result = await harness.call("list_people", {"limit": 5})
        outcomes.append("error" if result.is_error else "ok")

    async def main() -> None:
        async with anyio.create_task_group() as group:
            for _ in range(attempts):
                group.start_soon(one)

    anyio.run(main)

    assert outcomes.count("ok") == limit
    assert outcomes.count("error") == attempts - limit
    assert counting.calls == limit, "a rejected call must never reach the service"
    records = harness.audit.snapshot()
    assert len(records) == attempts, "every attempt is audited, allowed or not"
    assert sum(1 for r in records if r.error == "RATE_LIMITED") == attempts - limit
    assert {r.subject for r in records} == {OWNER_SUB}
