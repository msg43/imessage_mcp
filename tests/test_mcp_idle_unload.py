"""Idle MCP servers give their models back, and disconnected ones exit.

Each `imsg mcp local` process holds its own copy of the retrieval models,
and before this change it held them until it exited — so idle client
sessions, each with its own server, exhausted the index host's memory.
These tests pin the two fixes:

- `imsg.retrieval.idle_unload.IdleModelUnloader` drops the models after
  an idle period, never while a retrieval call is in flight, and the next
  call reloads them through the same warm-up path (and the same wait) as
  the first load. Time is a fake clock moved by hand, and the watchdog is
  replaced by calling `unload_if_idle()` directly.
- A server whose stdin closes (the SSH client went away) exits, even
  while its warm-up is held and a call is waiting on it, and even with
  the idle watchdog running.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import types as pytypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anyio
import mcp.types as types
import pytest
from pydantic import ValidationError

from _mlx_fakes import FakeRuntime, FakeTokenizer
from imsg.config.schema import McpLocalConfig, McpPublicConfig
from imsg.embed.mlx_text import MlxTextEmbeddingProvider
from imsg.mcp.audit import MemoryAuditSink
from imsg.mcp.tools.local_server import LocalMcpServer, warm_up_report
from imsg.mcp.warm_up_readiness import WarmUpReadinessFile, read_warm_up_readiness
from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpPhase, WarmUpStep
from imsg.retrieval.idle_unload import IdleModelUnloader, release_freed_memory
from imsg.retrieval.mlx_reranker import MlxRerankerProvider
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import RetrievalService, SearchMessagesResult

HELPER = Path(__file__).with_name("_run_local_mcp_server_with_held_warm_up.py")
IDLE = 600.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeModels:
    """What the warm-up loads and the unloader drops, with a record of
    both, in the order the model thread ran them."""

    def __init__(self) -> None:
        self.loaded = False
        self.events: list[str] = []
        self.load_gate = threading.Event()
        self.load_gate.set()
        self.unload_gate = threading.Event()
        self.unload_gate.set()

    def load(self) -> None:
        assert self.load_gate.wait(timeout=10), "the load was never released"
        self.loaded = True
        self.events.append("load")

    def unload(self) -> None:
        assert self.unload_gate.wait(timeout=10), "the unload was never released"
        self.loaded = False
        self.events.append("unload")


class FakeService:
    def __init__(self, models: FakeModels) -> None:
        self.models = models
        self.calls = 0

    def search_messages(self, context: Any, **kwargs: Any) -> SearchMessagesResult:
        assert self.models.loaded, "a query ran on unloaded models"
        self.calls += 1
        return SearchMessagesResult(results=[], candidate_lists={}, scan_cap_reached=False)


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-idle-unload")
    yield thread
    thread.close()


class Harness:
    def __init__(self, model_thread: ModelThread, *, idle_seconds: float = IDLE) -> None:
        self.clock = FakeClock()
        self.models = FakeModels()
        self.log: list[str] = []
        self.published: list[str] = []
        self.model_thread = model_thread
        self.warm_up = BackgroundWarmUp(
            [WarmUpStep("text embedder", 13.0, self.models.load)],
            model_thread=model_thread,
            log=self.log.append,
            on_status=lambda status: self.published.append(status.phase.value),
        )
        self.unloader = IdleModelUnloader(
            warm_up=self.warm_up,
            model_thread=model_thread,
            unload=self.models.unload,
            idle_seconds=idle_seconds,
            log=self.log.append,
            clock=self.clock,
        )
        self.service = FakeService(self.models)
        self.server = LocalMcpServer(
            service=cast(Any, self.service),
            audit=MemoryAuditSink(),
            config=cast(Any, None),
            conn=cast(Any, None),
            warm_up=self.warm_up,
            warm_up_wait_seconds=10.0,
            idle_unloader=self.unloader,
        )

    def warm(self) -> None:
        self.warm_up.start()
        assert self.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
        # The first check that sees the models ready starts the idle period.
        assert self.unloader.unload_if_idle() is False

    def drain(self) -> None:
        """Wait until the model thread has run everything queued so far."""
        self.model_thread.run(lambda: None)

    async def search(self) -> types.CallToolResult:
        params = types.CallToolRequestParams(name="search_messages", arguments={"query": "kites"})
        return await self.server.on_call_tool(cast(Any, None), params)


# --------------------------------------------------------------------------
# the idle timer
# --------------------------------------------------------------------------


def test_models_unload_once_the_idle_period_passes(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.warm()

    h.clock.advance(IDLE - 1)
    assert h.unloader.unload_if_idle() is False
    assert h.models.loaded

    h.clock.advance(1)
    assert h.unloader.unload_if_idle() is True
    h.drain()
    assert not h.models.loaded
    assert h.warm_up.status().phase is WarmUpPhase.UNLOADED
    assert h.models.events == ["load", "unload"]
    assert any(line.startswith("idle for 600 s with no tool call") for line in h.log)
    assert any(line.startswith("models unloaded in ") for line in h.log)
    assert h.published[-1] == "unloaded"  # the readiness file says so too

    # Already unloaded: nothing more to do, however long it stays idle.
    h.clock.advance(10 * IDLE)
    assert h.unloader.unload_if_idle() is False
    assert h.unloader.unloads == 1


def test_the_idle_period_counts_from_when_the_models_became_ready(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread)
    h.clock.advance(5 * IDLE)  # a very long warm-up, with no call meanwhile
    h.warm()
    h.clock.advance(IDLE - 1)
    assert h.unloader.unload_if_idle() is False
    h.clock.advance(1)
    assert h.unloader.unload_if_idle() is True


def test_every_call_restarts_the_idle_period(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.warm()
    for _ in range(3):
        h.clock.advance(IDLE - 1)
        assert not anyio.run(h.search).is_error
        assert h.unloader.unload_if_idle() is False
    h.clock.advance(IDLE)
    assert h.unloader.unload_if_idle() is True


def test_nothing_unloads_while_a_call_is_in_flight(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.warm()
    with h.unloader.call():
        assert h.unloader.in_flight == 1
        h.clock.advance(100 * IDLE)
        assert h.unloader.unload_if_idle() is False
        assert h.models.loaded
    # The call's end is activity too: idle counts from there.
    assert h.unloader.unload_if_idle() is False
    h.clock.advance(IDLE)
    assert h.unloader.unload_if_idle() is True


def test_a_slow_call_on_the_event_loop_holds_off_the_watchdog(model_thread: ModelThread) -> None:
    """The real server's shape: the call is in flight on the event loop
    while the watchdog checks from its own thread."""
    h = Harness(model_thread)
    h.warm()
    entered = threading.Event()
    leave = threading.Event()

    def slow_search(context: Any, **kwargs: Any) -> SearchMessagesResult:
        entered.set()
        assert leave.wait(timeout=10)
        assert h.models.loaded
        return SearchMessagesResult(results=[], candidate_lists={}, scan_cap_reached=False)

    h.service.search_messages = slow_search  # type: ignore[method-assign]
    verdicts: list[bool] = []

    def watchdog() -> None:
        assert entered.wait(timeout=10)
        h.clock.advance(100 * IDLE)
        verdicts.append(h.unloader.unload_if_idle())
        leave.set()

    checker = threading.Thread(target=watchdog)
    checker.start()
    assert not anyio.run(h.search).is_error
    checker.join(timeout=10)
    assert verdicts == [False]
    assert h.models.loaded


def test_a_failed_or_unfinished_warm_up_is_never_unloaded(model_thread: ModelThread) -> None:
    clock = FakeClock()
    release = threading.Event()
    unloads: list[str] = []

    def held() -> None:
        assert release.wait(timeout=10)
        raise RuntimeError("weights missing")

    warm_up = BackgroundWarmUp(
        [WarmUpStep("reranker", 10.0, held)], model_thread=model_thread, log=lambda _: None
    )
    unloader = IdleModelUnloader(
        warm_up=warm_up,
        model_thread=model_thread,
        unload=lambda: unloads.append("unload"),
        idle_seconds=IDLE,
        log=lambda _: None,
        clock=clock,
    )
    warm_up.start()
    for _ in range(3):  # still warming
        clock.advance(10 * IDLE)
        assert unloader.unload_if_idle() is False
    release.set()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.FAILED
    for _ in range(3):
        clock.advance(10 * IDLE)
        assert unloader.unload_if_idle() is False
    assert unloads == []
    assert warm_up.status().phase is WarmUpPhase.FAILED


def test_zero_turns_unloading_off(model_thread: ModelThread) -> None:
    h = Harness(model_thread, idle_seconds=0)
    h.warm()
    h.clock.advance(1_000_000)
    assert h.unloader.unload_if_idle() is False
    assert not h.unloader.enabled
    h.unloader.start_watchdog()  # a no-op when off: no thread to leak
    assert not any(t.name == "imsg-idle-unload" for t in threading.enumerate())


def test_the_watchdog_unloads_on_its_own(model_thread: ModelThread) -> None:
    clock = FakeClock()
    models = FakeModels()
    warm_up = BackgroundWarmUp(
        [WarmUpStep("text embedder", 1.0, models.load)],
        model_thread=model_thread,
        log=lambda _: None,
    )
    unloader = IdleModelUnloader(
        warm_up=warm_up,
        model_thread=model_thread,
        unload=models.unload,
        idle_seconds=10,  # checked every second
        log=lambda _: None,
        clock=clock,
    )
    warm_up.start()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    unloader.start_watchdog()
    try:
        deadline = time.monotonic() + 10
        while unloader.unloads == 0:
            assert time.monotonic() < deadline, "the watchdog never unloaded"
            clock.advance(5)
            time.sleep(0.2)
        model_thread.run(lambda: None)
        assert not models.loaded
    finally:
        unloader.stop()


# --------------------------------------------------------------------------
# reloading on the next call
# --------------------------------------------------------------------------


def test_the_next_call_reloads_the_models_and_is_answered(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.warm()
    h.clock.advance(IDLE)
    assert h.unloader.unload_if_idle() is True
    h.drain()

    result = anyio.run(h.search)

    assert not result.is_error
    assert h.service.calls == 1
    assert h.models.events == ["load", "unload", "load"]
    assert h.warm_up.status().phase is WarmUpPhase.READY
    assert any(line.startswith("reload started: 1 step") for line in h.log)


def test_a_call_that_arrives_during_the_unload_waits_for_the_reload(
    model_thread: ModelThread,
) -> None:
    """The unload and the reload both run on the model thread, in the
    order they were queued, so a call arriving mid-unload never sees
    half-dropped models."""
    h = Harness(model_thread)
    h.warm()
    h.models.unload_gate.clear()  # the unload will sit on the model thread
    h.clock.advance(IDLE)
    assert h.unloader.unload_if_idle() is True

    threading.Timer(0.3, h.models.unload_gate.set).start()
    result = anyio.run(h.search)

    assert not result.is_error
    assert h.models.events == ["load", "unload", "load"]


def test_a_reload_that_outlasts_the_wait_answers_warming_up_like_a_cold_start(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread)
    h.warm()
    h.clock.advance(IDLE)
    assert h.unloader.unload_if_idle() is True
    h.drain()
    h.models.load_gate.clear()  # the reload will take longer than the wait
    h.server.warm_up_wait_seconds = 0.2

    result = anyio.run(h.search)
    block = result.content[0]
    assert result.is_error and isinstance(block, types.TextContent)
    assert block.text.startswith("WARMING_UP\n")
    assert h.unloader.in_flight == 0  # the call let go when it answered

    h.models.load_gate.set()
    assert h.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    assert not anyio.run(h.search).is_error


def test_a_cancelled_call_does_not_stay_registered(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.warm()
    h.clock.advance(IDLE)
    assert h.unloader.unload_if_idle() is True
    h.drain()
    h.models.load_gate.clear()  # the reload never finishes while the call waits

    async def main() -> None:
        with anyio.move_on_after(0.2):
            await h.search()

    anyio.run(main)
    assert h.unloader.in_flight == 0
    h.models.load_gate.set()


def test_check_permissions_reports_the_unloaded_state(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.warm()
    h.clock.advance(IDLE)
    h.unloader.unload_if_idle()
    report = warm_up_report(h.warm_up.status())
    assert report["state"] == "unloaded"
    assert report["seconds_remaining"] == 13


def test_the_public_readiness_file_describes_an_unloaded_server(
    model_thread: ModelThread, tmp_path: Path
) -> None:
    h = Harness(model_thread)
    readiness = WarmUpReadinessFile(tmp_path / "run" / "mcp-public-warm-up.json")
    h.warm_up = BackgroundWarmUp(
        [WarmUpStep("reranker", 1.0, lambda: None)],
        model_thread=model_thread,
        log=lambda _: None,
        on_status=readiness.publish,
    )
    h.warm_up.start()
    assert h.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    assert h.warm_up.mark_unloaded() is True
    report = read_warm_up_readiness(tmp_path)
    assert report.state == "unloaded"
    assert "reloads them" in report.detail


# --------------------------------------------------------------------------
# what an unload actually drops
# --------------------------------------------------------------------------


def test_the_service_unloads_every_provider_that_can(model_thread: ModelThread) -> None:
    dropped: list[str] = []

    class Unloadable:
        def __init__(self, name: str) -> None:
            self.name = name

        def unload(self) -> None:
            dropped.append(self.name)

    service = RetrievalService(
        pg_conn=cast(Any, None),
        fts_conn=cast(Any, None),
        config=cast(Any, None),
        text_provider=cast(Any, Unloadable("text")),
        reranker=cast(Any, Unloadable("reranker")),
        multimodal_provider=cast(Any, object()),  # a fake with nothing to drop
    )
    service.unload_models()
    assert dropped == ["text", "reranker"]


def test_mlx_providers_drop_their_weights_and_load_them_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(tokenizer=FakeTokenizer(eos_suffix=(7,), pad_token_id=9)).install(
        monkeypatch
    )
    embedder = MlxTextEmbeddingProvider("org/embed", "rev1", 3)
    embedder.embed_query("a", instruction="find")
    assert embedder.is_loaded
    embedder.unload()
    assert not embedder.is_loaded
    embedder.embed_query("a", instruction="find")
    assert embedder.is_loaded
    assert len(runtime.load_calls) == 2

    reranker = MlxRerankerProvider("org/rerank", "rev2")
    reranker.load()
    reranker.unload()
    assert not reranker.is_loaded
    reranker.unload()  # idempotent


def test_release_freed_memory_empties_the_mlx_and_torch_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    mx = pytypes.ModuleType("mlx.core")
    mx.clear_cache = lambda: calls.append("mlx.clear_cache")  # type: ignore[attr-defined]
    mx.get_active_memory = lambda: 2**30  # type: ignore[attr-defined]
    mx.get_cache_memory = lambda: 0  # type: ignore[attr-defined]
    torch = pytypes.ModuleType("torch")
    torch.mps = pytypes.SimpleNamespace(  # type: ignore[attr-defined]
        empty_cache=lambda: calls.append("torch.mps.empty_cache")
    )
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "torch", torch)

    detail = release_freed_memory()

    assert calls == ["mlx.clear_cache", "torch.mps.empty_cache"]
    assert detail == "MLX now holds 1.00 GiB active, 0.00 GiB cached"


def test_release_freed_memory_imports_no_runtime_that_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert release_freed_memory() == ""
    assert "mlx.core" not in sys.modules and "torch" not in sys.modules


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_config_defaults_unload_local_servers_after_ten_minutes_and_not_public_ones() -> None:
    assert McpLocalConfig().idle_unload_seconds == 600
    assert McpPublicConfig(scope="allowlist").idle_unload_seconds == 0


@pytest.mark.parametrize("value", [0, 60, 3600])
def test_config_accepts_zero_or_at_least_a_minute(value: int) -> None:
    assert McpLocalConfig(idle_unload_seconds=value).idle_unload_seconds == value
    public = McpPublicConfig(scope="allowlist", idle_unload_seconds=value)
    assert public.idle_unload_seconds == value


@pytest.mark.parametrize("value", [-1, 1, 59])
def test_config_rejects_negative_and_sub_minute_periods(value: int) -> None:
    with pytest.raises(ValidationError, match="idle_unload_seconds"):
        McpLocalConfig(idle_unload_seconds=value)
    with pytest.raises(ValidationError, match="idle_unload_seconds"):
        McpPublicConfig(scope="allowlist", idle_unload_seconds=value)


# --------------------------------------------------------------------------
# stdin EOF: a server whose client is gone exits
# --------------------------------------------------------------------------


def _start_child(tmp_path: Path, *extra: str) -> subprocess.Popen[bytes]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    return subprocess.Popen(
        [sys.executable, str(HELPER), str(tmp_path / "never-released"), "30", *extra],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _send(proc: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(message) + "\n").encode())
    proc.stdin.flush()


def _initialize(proc: subprocess.Popen[bytes]) -> None:
    assert proc.stdout is not None
    _send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "eof-test", "version": "0"},
            },
        },
    )
    answer = json.loads(proc.stdout.readline())
    assert answer["id"] == 1 and "result" in answer
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})


def _exits_after_stdin_closes(proc: subprocess.Popen[bytes]) -> float:
    assert proc.stdin is not None
    started = time.monotonic()
    proc.stdin.close()
    try:
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    return time.monotonic() - started


@pytest.mark.parametrize("idle_args", [(), ("600",)], ids=["no-unloader", "with-watchdog"])
def test_stdin_eof_during_a_held_warm_up_exits_the_server(
    tmp_path: Path, idle_args: tuple[str, ...]
) -> None:
    """The warm-up is stuck loading (the model thread is busy and never
    finishes) when the client goes away: the process still exits."""
    proc = _start_child(tmp_path, *idle_args)
    _initialize(proc)
    assert _exits_after_stdin_closes(proc) < 10


@pytest.mark.parametrize("idle_args", [(), ("600",)], ids=["no-unloader", "with-watchdog"])
def test_stdin_eof_while_a_call_waits_for_the_models_exits_the_server(
    tmp_path: Path, idle_args: tuple[str, ...]
) -> None:
    """A call is waiting (for up to 30 s here) on a warm-up that never
    finishes when the client goes away: the process does not sit out the
    wait."""
    proc = _start_child(tmp_path, *idle_args)
    _initialize(proc)
    call = {"name": "search_messages", "arguments": {"query": "kite festival"}}
    _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": call})
    time.sleep(0.5)  # the call is now waiting on the held warm-up
    assert proc.poll() is None
    assert _exits_after_stdin_closes(proc) < 10
