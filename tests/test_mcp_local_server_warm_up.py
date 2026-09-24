"""The local MCP server answers before its models are warm.

`imsg mcp local` used to warm every model before answering anything, so
an MCP client waiting on `initialize` gave up first. Now the handshake is
answered at once, the warm-up runs in the background, and tool calls are
gated on it: a call that arrives during warm-up waits (off the event
loop) up to a bound, then gets a `WARMING_UP` tool error with an
estimate; a warm-up that failed is reported by every call.

Warm-ups here are fake steps held on an event, so "during warm-up" is a
state the test controls rather than a race. The last test runs the real
stdio transport in a child process.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anyio
import mcp.types as types
import pytest

from imsg.errors import ProviderUnavailableError
from imsg.mcp.audit import MemoryAuditSink
from imsg.mcp.tools import local_server
from imsg.mcp.tools.local_server import TOOL_CALL_WARM_UP_WAIT_SECONDS, LocalMcpServer
from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpPhase, WarmUpStep
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import SearchMessagesResult

HELPER = Path(__file__).with_name("_run_local_mcp_server_with_held_warm_up.py")


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-local-server")
    yield thread
    thread.close()


class _FakeRetrievalService:
    """Duck-typed `RetrievalService` that records what ran, and when."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def search_messages(self, context: Any, **kwargs: Any) -> SearchMessagesResult:
        self.events.append("search_messages")
        return SearchMessagesResult(
            results=[], candidate_lists={"segment_fts": 0}, scan_cap_reached=False
        )

    def list_people(self, context: Any, **kwargs: Any) -> dict[str, Any]:
        self.events.append("list_people")
        return {"people": []}


class _Harness:
    def __init__(
        self,
        model_thread: ModelThread,
        *,
        wait_seconds: float,
        failure: Exception | None = None,
        held: bool = True,
    ) -> None:
        self.events: list[str] = []
        self.release = threading.Event()
        if not held:
            self.release.set()
        self.log: list[str] = []

        def step() -> None:
            assert self.release.wait(timeout=30), "the warm-up was never released"
            if failure is not None:
                raise failure
            self.events.append("warm-up finished")

        self.warm_up = BackgroundWarmUp(
            [WarmUpStep("reranker", 20.0, step)], model_thread=model_thread, log=self.log.append
        )
        self.audit = MemoryAuditSink()
        self.server = LocalMcpServer(
            service=cast(Any, _FakeRetrievalService(self.events)),
            audit=self.audit,
            config=cast(Any, None),
            conn=cast(Any, None),
            warm_up=self.warm_up,
            warm_up_wait_seconds=wait_seconds,
        )

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        params = types.CallToolRequestParams(name=name, arguments=arguments)
        return await self.server.on_call_tool(cast(Any, None), params)


def _text(result: types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


def _wait_until(condition: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def _release_after(event: threading.Event, seconds: float) -> threading.Timer:
    timer = threading.Timer(seconds, event.set)
    timer.start()
    return timer


def test_the_default_wait_stays_inside_claude_codes_two_minute_foreground_window() -> None:
    assert 0 < TOOL_CALL_WARM_UP_WAIT_SECONDS < 120


def test_tools_list_does_not_wait_for_the_warm_up(model_thread: ModelThread) -> None:
    harness = _Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()

    async def main() -> types.ListToolsResult:
        with anyio.fail_after(1.0):
            return await harness.server.on_list_tools(cast(Any, None), None)

    assert len(anyio.run(main).tools) == 5
    assert harness.warm_up.status().phase is WarmUpPhase.WARMING
    harness.release.set()


def test_a_tool_call_during_warm_up_waits_for_it_then_succeeds(model_thread: ModelThread) -> None:
    harness = _Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()
    timer = _release_after(harness.release, 0.3)

    began = time.monotonic()
    result = anyio.run(harness.call, "search_messages", {"query": "kite festival"})
    waited = time.monotonic() - began
    timer.join()

    assert not result.is_error, _text(result)
    assert result.structured_content is not None
    assert result.structured_content["results"] == []
    assert harness.events == ["warm-up finished", "search_messages"]
    assert 0.3 <= waited < 10.0
    (row,) = harness.audit.snapshot()
    assert (row.tool, row.error) == ("search_messages", None)
    assert row.latency_ms >= 300  # the audited latency includes the wait


def test_a_tool_call_that_outwaits_the_bound_gets_warming_up_with_an_estimate(
    model_thread: ModelThread,
) -> None:
    harness = _Harness(model_thread, wait_seconds=0.2)
    harness.warm_up.start()

    began = time.monotonic()
    result = anyio.run(harness.call, "search_messages", {"query": "kite festival"})
    waited = time.monotonic() - began

    assert result.is_error
    code, message = _text(result).split("\n", 1)
    assert code == "WARMING_UP"
    assert message.startswith("the index is warming up (loading the reranker): about ")
    remaining = int(message.split("about ", 1)[1].split(" s remaining")[0])
    assert 1 <= remaining <= 20
    assert "each call waits up to 0.2 s" in message
    assert "Traceback" not in message
    assert 0.2 <= waited < 5.0
    assert harness.events == []  # the service never ran
    (row,) = harness.audit.snapshot()
    assert row.error == "WARMING_UP"

    harness.release.set()
    assert harness.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    retried = anyio.run(harness.call, "search_messages", {"query": "kite festival"})
    assert not retried.is_error


def test_the_event_loop_keeps_answering_while_a_call_waits(model_thread: ModelThread) -> None:
    harness = _Harness(model_thread, wait_seconds=30.0)
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


def test_a_cancelled_call_stops_waiting_at_once(model_thread: ModelThread) -> None:
    harness = _Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()

    async def main() -> None:
        with anyio.move_on_after(0.2):
            await harness.call("search_messages", {"query": "kite festival"})

    began = time.monotonic()
    anyio.run(main)
    assert time.monotonic() - began < 5.0
    assert harness.events == []
    harness.release.set()


def test_arguments_that_fail_the_schema_are_rejected_without_waiting(
    model_thread: ModelThread,
) -> None:
    harness = _Harness(model_thread, wait_seconds=30.0)
    harness.warm_up.start()

    began = time.monotonic()
    result = anyio.run(harness.call, "search_messages", {"limit": 5})
    assert time.monotonic() - began < 5.0
    assert result.is_error
    assert _text(result).startswith("INVALID_ARGUMENT\n")
    harness.release.set()


def test_a_failed_warm_up_is_reported_on_every_tool_call(model_thread: ModelThread) -> None:
    harness = _Harness(
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
        code, message = _text(result).split("\n", 1)
        assert code == "WARM_UP_FAILED"
        assert "(reranker: reranker weights not found)" in message
        assert "restart the MCP server" in message

    assert harness.events == []  # nothing served on half-loaded models
    assert [row.error for row in harness.audit.snapshot()] == ["WARM_UP_FAILED"] * 3
    assert sum("FAILED" in line for line in harness.log) == 1  # logged once, not per call


def _read_json_line(lines: queue.Queue[bytes | None], stray: list[bytes], timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        try:
            line = lines.get(timeout=max(deadline - time.monotonic(), 0.01))
        except queue.Empty:
            raise AssertionError(f"the server sent no message within {timeout} s") from None
        assert line is not None, "the server closed stdout"
        try:
            return json.loads(line)
        except ValueError:
            stray.append(line)


# --------------------------------------------------------------------------
# check_permissions: diagnostics, answered while the models are loading
# --------------------------------------------------------------------------


@pytest.fixture
def stub_check_permissions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """`check_permissions` reads the mount, the OS and Postgres; here it
    only records that it ran."""
    ran: list[str] = []

    def fake(*, config: Any, conn: Any) -> dict[str, Any]:
        ran.append("check_permissions")
        return {"mount_ok": True, "pg_ok": True}

    monkeypatch.setattr(local_server.handlers, "check_permissions", fake)
    return ran


def _permissions(harness: _Harness) -> dict[str, Any]:
    async def main() -> types.CallToolResult:
        with anyio.fail_after(2.0):  # far below the wait bound: it must not wait
            return await harness.call("check_permissions", {})

    result = anyio.run(main)
    assert not result.is_error, _text(result)
    assert result.structured_content is not None
    return dict(result.structured_content)


def test_check_permissions_answers_during_warm_up_and_says_what_is_loading(
    model_thread: ModelThread, stub_check_permissions: list[str]
) -> None:
    """The question "what is this server doing?" must be answerable while
    it is doing it — so this call neither waits for the warm-up nor is
    refused by it."""
    harness = _Harness(model_thread, wait_seconds=300.0)
    harness.warm_up.start()
    _wait_until(lambda: harness.warm_up.status().loading == "reranker")

    payload = _permissions(harness)

    assert payload["mount_ok"] is True and stub_check_permissions == ["check_permissions"]
    warm_up = payload["warm_up"]
    assert warm_up["state"] == "warming"
    assert warm_up["loading"] == "reranker"
    assert (warm_up["steps_done"], warm_up["steps_total"]) == (0, 1)
    assert isinstance(warm_up["seconds_remaining"], int) and warm_up["seconds_remaining"] >= 1
    assert warm_up["failure"] is None
    assert harness.warm_up.status().phase is WarmUpPhase.WARMING  # still loading
    harness.release.set()


def test_check_permissions_reports_not_started_ready_and_failed(
    model_thread: ModelThread, stub_check_permissions: list[str]
) -> None:
    harness = _Harness(model_thread, wait_seconds=300.0)
    not_started = _permissions(harness)["warm_up"]
    assert (not_started["state"], not_started["loading"]) == ("not_started", None)
    assert not_started["steps_done"] == 0

    harness.warm_up.start()
    harness.release.set()
    assert harness.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    ready = _permissions(harness)["warm_up"]
    assert ready == {
        "state": "ready",
        "loading": None,
        "steps_done": 1,
        "steps_total": 1,
        "seconds_remaining": None,
        "failure": None,
        "memory_detail": None,
    }

    failing = _Harness(
        model_thread,
        wait_seconds=300.0,
        failure=ProviderUnavailableError("weights missing from the data volume"),
        held=False,
    )
    failing.warm_up.start()
    assert failing.warm_up.wait(timeout=5).phase is WarmUpPhase.FAILED
    failed = _permissions(failing)["warm_up"]
    assert failed["state"] == "failed" and failed["loading"] is None
    assert failed["failure"] is not None
    assert "weights missing from the data volume" in failed["failure"]
    assert failed["seconds_remaining"] is None

    # ... while every retrieval tool still refuses, naming the same cause.
    async def search() -> types.CallToolResult:
        return await failing.call("search_messages", {"query": "kite festival"})

    refused = anyio.run(search)
    assert refused.is_error and _text(refused).startswith("WARM_UP_FAILED\n")


def test_check_permissions_is_audited_like_any_other_call(
    model_thread: ModelThread, stub_check_permissions: list[str]
) -> None:
    harness = _Harness(model_thread, wait_seconds=300.0)
    harness.warm_up.start()
    _permissions(harness)
    records = harness.audit.snapshot()
    assert [r.tool for r in records] == ["check_permissions"]
    assert records[0].error is None and records[0].surface == "local"
    harness.release.set()


def test_over_real_stdio_the_handshake_is_answered_while_the_warm_up_is_held(
    tmp_path: Path,
) -> None:
    release_file = tmp_path / "release"
    stderr_path = tmp_path / "stderr.log"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    with stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            [sys.executable, str(HELPER), str(release_file), "0.3"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
        )
    assert proc.stdin is not None and proc.stdout is not None
    lines: queue.Queue[bytes | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    stray: list[bytes] = []

    def send(message: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(message) + "\n").encode())
        proc.stdin.flush()

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "warm-up-test", "version": "0"},
                },
            }
        )
        initialized = _read_json_line(lines, stray, timeout=20)
        assert initialized["id"] == 1 and "result" in initialized
        assert not release_file.exists()  # answered while the warm-up is still held

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listed = _read_json_line(lines, stray, timeout=10)
        assert listed["id"] == 2 and len(listed["result"]["tools"]) == 5

        call = {"name": "search_messages", "arguments": {"query": "kite festival"}}
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": call})
        warming = _read_json_line(lines, stray, timeout=10)
        assert warming["id"] == 3 and warming["result"]["isError"] is True
        assert warming["result"]["content"][0]["text"].startswith("WARMING_UP\n")

        release_file.write_text("")
        for request_id in range(4, 50):  # the child notices the file within a poll or two
            send({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": call})
            answered = _read_json_line(lines, stray, timeout=10)
            assert answered["id"] == request_id
            if not answered["result"].get("isError"):
                break
            assert answered["result"]["content"][0]["text"].startswith("WARMING_UP\n")
        assert answered["result"]["structuredContent"]["results"] == []

        proc.stdin.close()
        assert proc.wait(timeout=20) == 0
        while (rest := lines.get(timeout=5)) is not None:
            stray.append(rest)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert stray == []  # nothing but JSON-RPC frames ever reached stdout
    log = stderr_path.read_text()
    assert "stray print from a loading model" in log
    assert "stray structlog line from a loading model" in log
    assert "stray write to fd 1 from a loading model" in log
    assert "warm-up started: 1 step in the background (text embedder)" in log
    assert "text embedder ready in " in log
    assert "warm-up done: 1 step ready in " in log


def test_over_real_stdio_a_lazy_server_loads_nothing_until_asked(tmp_path: Path) -> None:
    """The real `run_local_server` in a child process, as `imsg mcp local`
    runs by default: the handshake and `tools/list` load nothing; the first
    search starts the (held) load and is answered once it finishes."""
    release_file = tmp_path / "release"
    stderr_path = tmp_path / "stderr.log"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    with stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            [sys.executable, str(HELPER), str(release_file), "0.3", "600", "--lazy"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
        )
    assert proc.stdin is not None and proc.stdout is not None
    lines: queue.Queue[bytes | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    stray: list[bytes] = []

    def send(message: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(message) + "\n").encode())
        proc.stdin.flush()

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "lazy-test", "version": "0"},
                },
            }
        )
        assert _read_json_line(lines, stray, timeout=20)["id"] == 1
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert _read_json_line(lines, stray, timeout=10)["id"] == 2
        time.sleep(0.5)
        assert "warm-up started" not in stderr_path.read_text()  # nothing loaded yet

        call = {"name": "search_messages", "arguments": {"query": "kite festival"}}
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": call})
        first = _read_json_line(lines, stray, timeout=10)
        assert first["result"]["content"][0]["text"].startswith("WARMING_UP\n")
        assert "warm-up started: 1 step in the background (text embedder)" in (
            stderr_path.read_text()
        )

        release_file.write_text("")
        for request_id in range(4, 50):
            send({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": call})
            answered = _read_json_line(lines, stray, timeout=10)
            if not answered["result"].get("isError"):
                break
        assert answered["result"]["structuredContent"]["results"] == []
        proc.stdin.close()
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    assert stray == []
