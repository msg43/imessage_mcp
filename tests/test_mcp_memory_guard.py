"""MCP servers load their models only when the host has room, and give
them back when the host is short.

- Every load — at startup and after an idle unload — asks the memory
  admission check first (`imsg.retrieval.background_warm_up`). Refused,
  nothing loads and a retrieval call is answered at once with the
  retryable `WARMING_UP` code and a "host memory busy" message.
- The public surface says so only behind its auth gate, and in numbers
  only.
- Under critical memory pressure a loaded server unloads before its idle
  timer, never under a call in flight (`imsg.retrieval.idle_unload`); the
  public server can be told never to, and reloads by itself once the host
  has room again.

Time is a fake clock moved by hand; the watchdog's checks are called
directly except where the thread itself is the point.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import mcp.types as types
import pytest

from imsg.host_memory import GIB, HostMemory, PressureLevel
from imsg.mcp.audit import MemoryAuditSink
from imsg.mcp.auth import PublicAuthGate, TokenIntrospection
from imsg.mcp.errors import TokenInvalidError
from imsg.mcp.tools.local_server import LocalMcpServer, warm_up_report
from imsg.mcp.tools.public_server import PublicMcpServer
from imsg.mcp.warm_up_readiness import WarmUpReadinessFile, read_warm_up_readiness
from imsg.memory_admission import AdmissionDecision, decide
from imsg.retrieval.background_warm_up import (
    REFUSAL_LOG_INTERVAL_SECONDS,
    BackgroundWarmUp,
    WarmUpPhase,
    WarmUpStep,
)
from imsg.retrieval.idle_unload import IdleModelUnloader, PressureRelease
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import SearchMessagesResult

IDLE = 600.0
RETRY = 15.0
COOLDOWN = 300.0

OWNER_SUB = "300000000000000000003"
CLIENT_ID = "111111111111-fictional.apps.example"
OWNER_TOKEN = "fictional-owner-token"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def host(available_gib: float, pressure: PressureLevel = PressureLevel.NORMAL) -> HostMemory:
    return HostMemory(64 * GIB, int(available_gib * GIB), 0, 0, pressure)


class FakeAdmission:
    """Decides from `available_gib` and `pressure`, which a test changes;
    counts every check and every release."""

    def __init__(self, available_gib: float = 40.0) -> None:
        self.available_gib = available_gib
        self.pressure = PressureLevel.NORMAL
        self.checks = 0
        self.releases = 0
        self.reason_suffix = ""

    def check(self) -> AdmissionDecision:
        self.checks += 1
        decision = decide(
            host(self.available_gib, self.pressure),
            what="the local MCP server's models",
            required_bytes=int(17.2 * GIB),
            reserve_bytes=8 * GIB,
            max_pressure=PressureLevel.NORMAL,
        )
        if self.reason_suffix and not decision.admitted:
            return AdmissionDecision(
                admitted=False,
                reason=decision.reason + self.reason_suffix,
                required_bytes=decision.required_bytes,
                reserve_bytes=decision.reserve_bytes,
                memory=decision.memory,
            )
        return decision

    def release(self) -> None:
        self.releases += 1


class FakeModels:
    def __init__(self) -> None:
        self.loaded = False
        self.events: list[str] = []

    def load(self) -> None:
        self.loaded = True
        self.events.append("load")

    def unload(self) -> None:
        self.loaded = False
        self.events.append("unload")


class FakeService:
    def __init__(self, models: FakeModels) -> None:
        self.models = models
        self.calls = 0
        self.entered = threading.Event()
        self.leave = threading.Event()
        self.leave.set()

    def search_messages(self, context: Any, **kwargs: Any) -> SearchMessagesResult:
        assert self.models.loaded, "a query ran on unloaded models"
        self.calls += 1
        self.entered.set()
        assert self.leave.wait(timeout=10)
        return SearchMessagesResult(results=[], candidate_lists={}, scan_cap_reached=False)


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-memory-guard")
    yield thread
    thread.close()


class Harness:
    def __init__(
        self,
        model_thread: ModelThread,
        *,
        available_gib: float = 40.0,
        release_at: PressureLevel | None = PressureLevel.CRITICAL,
        rewarm: bool = False,
        public: bool = False,
        tmp_path: Path | None = None,
    ) -> None:
        self.clock = FakeClock()
        self.models = FakeModels()
        self.admission = FakeAdmission(available_gib)
        self.pressure = PressureLevel.NORMAL
        self.log: list[str] = []
        self.model_thread = model_thread
        readiness = (
            WarmUpReadinessFile(tmp_path / "run" / "mcp-public-warm-up.json")
            if tmp_path is not None
            else None
        )
        self.warm_up = BackgroundWarmUp(
            [WarmUpStep("text embedder", 13.0, self.models.load)],
            model_thread=model_thread,
            log=self.log.append,
            clock=self.clock,
            admission=self.admission.check,
            admission_retry_seconds=RETRY,
            on_status=readiness.publish if readiness is not None else None,
        )
        self.unloader = IdleModelUnloader(
            warm_up=self.warm_up,
            model_thread=model_thread,
            unload=self.models.unload,
            idle_seconds=0 if public else IDLE,
            log=self.log.append,
            clock=self.clock,
            pressure_release=PressureRelease(
                read_pressure=lambda: self.pressure,
                release_at=release_at,
                check_seconds=5.0,
                rewarm=rewarm,
                rewarm_cooldown_seconds=COOLDOWN,
            ),
            after_unload=self.admission.release,
        )
        self.service = FakeService(self.models)
        self.audit = MemoryAuditSink()
        self.gate: PublicAuthGate | None = None
        if public:
            self.gate = PublicAuthGate(
                client_id=CLIENT_ID,
                owner_subject=OWNER_SUB,
                introspector=_OwnerOnly(),
                audit=self.audit,
            )
            self.server: LocalMcpServer | PublicMcpServer = PublicMcpServer(
                service=cast(Any, self.service),
                gate=self.gate,
                scope="allowlist",
                warm_up=self.warm_up,
                warm_up_wait_seconds=5.0,
                idle_unloader=self.unloader,
            )
        else:
            self.server = LocalMcpServer(
                service=cast(Any, self.service),
                audit=self.audit,
                config=cast(Any, None),
                conn=cast(Any, None),
                warm_up=self.warm_up,
                warm_up_wait_seconds=5.0,
                idle_unloader=self.unloader,
            )

    def settle(self) -> WarmUpPhase:
        self.model_thread.run(lambda: None)
        return self.warm_up.status().phase

    def start(self) -> WarmUpPhase:
        self.warm_up.start()
        return self.settle()

    async def search(self, authorization: str | None = f"Bearer {OWNER_TOKEN}") -> types.CallToolResult:
        params = types.CallToolRequestParams(name="search_messages", arguments={"query": "kites"})
        context = SimpleNamespace(
            request=SimpleNamespace(
                headers={} if authorization is None else {"authorization": authorization}
            )
        )
        return await self.server.on_call_tool(cast(Any, context), params)


class _OwnerOnly:
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


def text_of(result: types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


# --------------------------------------------------------------------------
# startup: a load the host has no room for does not start
# --------------------------------------------------------------------------


def test_a_refused_startup_loads_nothing_and_calls_answer_host_memory_busy(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread, available_gib=20.0)  # 17.2 + 8 does not fit
    assert h.start() is WarmUpPhase.MEMORY_BUSY
    assert h.models.events == []  # no step ran

    started = time.monotonic()
    result = anyio.run(h.search)
    assert time.monotonic() - started < 2.0  # answered at once, not after the 5 s wait
    assert result.is_error
    text = text_of(result)
    assert text.startswith("WARMING_UP\nhost memory busy")
    assert "20.0 GiB available" in text and "17.2 GiB plus a reserve of 8.0 GiB" in text
    assert h.service.calls == 0
    assert [row.error for row in h.audit.snapshot()] == ["WARMING_UP"]
    assert any(line.startswith("not loading the models — host memory busy") for line in h.log)


def test_a_refused_load_is_tried_again_only_after_the_retry_interval(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread, available_gib=20.0)
    h.start()
    assert h.admission.checks == 1

    h.clock.advance(RETRY - 1)
    assert text_of(anyio.run(h.search)).startswith("WARMING_UP\nhost memory busy")
    assert h.admission.checks == 1  # too soon: the answer has not changed

    h.clock.advance(1)
    h.admission.available_gib = 40.0  # the host has room now
    result = anyio.run(h.search)
    assert not result.is_error
    assert h.admission.checks == 2
    assert h.models.events == ["load"]
    assert any(line.startswith("host memory is available again") for line in h.log)


def test_refusals_in_a_row_are_logged_once_per_interval(model_thread: ModelThread) -> None:
    h = Harness(model_thread, available_gib=20.0)
    h.start()
    # Every try inside one log interval, each refused for the same reason.
    for _ in range(int(REFUSAL_LOG_INTERVAL_SECONDS // RETRY) - 1):
        h.clock.advance(RETRY)
        h.start()
    assert h.admission.checks == int(REFUSAL_LOG_INTERVAL_SECONDS // RETRY)
    refusals = [line for line in h.log if line.startswith("not loading the models")]
    assert len(refusals) == 1
    h.clock.advance(REFUSAL_LOG_INTERVAL_SECONDS)
    h.start()
    assert len([line for line in h.log if line.startswith("not loading the models")]) == 2


def test_a_refusal_is_logged_again_when_its_cause_changes_and_at_least_once_a_minute(
    model_thread: ModelThread,
) -> None:
    """2026-09-25: a refused public server tried again every 15 s but
    logged a refusal only every 300 s, so its log read as if it rechecked
    every ten minutes. Now a changed cause is logged at once, and an
    unchanged one at least once a minute."""
    h = Harness(model_thread, available_gib=20.0, public=True, rewarm=True)
    h.start()

    def refusals() -> list[str]:
        return [line for line in h.log if line.startswith("not loading the models")]

    assert len(refusals()) == 1
    h.clock.advance(RETRY)
    h.start()
    assert len(refusals()) == 1  # the same wait: not said again yet

    h.admission.pressure = PressureLevel.WARN  # now pressure, not free memory
    h.clock.advance(RETRY)
    h.start()
    assert len(refusals()) == 2
    assert "warn memory pressure" in refusals()[-1]

    for _ in range(3):
        h.clock.advance(RETRY)
        h.start()
    assert len(refusals()) == 2
    h.clock.advance(RETRY)  # a minute since the last line
    h.start()
    assert len(refusals()) == 3


def test_an_admission_check_that_raises_refuses_the_load(model_thread: ModelThread) -> None:
    def broken() -> AdmissionDecision:
        raise RuntimeError("probe exploded")

    loads: list[str] = []
    warm_up = BackgroundWarmUp(
        [WarmUpStep("reranker", 1.0, lambda: loads.append("load"))],
        model_thread=model_thread,
        log=lambda _: None,
        admission=broken,
    )
    warm_up.start()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.MEMORY_BUSY
    assert loads == []
    assert "probe exploded" in (warm_up.status().detail or "")


def test_check_permissions_says_why_the_models_are_not_loaded(model_thread: ModelThread) -> None:
    h = Harness(model_thread, available_gib=20.0)
    h.start()
    report = warm_up_report(h.warm_up.status())
    assert report["state"] == "memory_busy"
    assert report["memory_detail"].startswith("host memory busy: 20.0 GiB available")
    assert report["seconds_remaining"] is None


def test_the_readiness_file_reports_a_refused_load(model_thread: ModelThread, tmp_path: Path) -> None:
    h = Harness(model_thread, available_gib=20.0, public=True, rewarm=True, tmp_path=tmp_path)
    h.start()
    report = read_warm_up_readiness(tmp_path)
    assert report.state == "memory_busy"
    assert "host did not have the memory" in report.detail
    assert "20.0 GiB available" in report.detail


# --------------------------------------------------------------------------
# reload after an idle unload
# --------------------------------------------------------------------------


def test_a_reload_after_an_idle_unload_waits_for_memory(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    assert h.start() is WarmUpPhase.READY
    assert h.unloader.unload_if_idle() is False  # starts the idle period
    h.clock.advance(IDLE)
    assert h.unloader.unload_if_idle() is True
    assert h.settle() is WarmUpPhase.UNLOADED
    assert h.admission.releases == 1  # the reservation goes with the models

    h.admission.available_gib = 10.0  # meanwhile something else took the memory
    result = anyio.run(h.search)
    assert text_of(result).startswith("WARMING_UP\nhost memory busy")
    assert h.models.events == ["load", "unload"]  # not loaded again
    assert h.warm_up.status().phase is WarmUpPhase.MEMORY_BUSY


# --------------------------------------------------------------------------
# the public surface: behind the gate, numbers only
# --------------------------------------------------------------------------


def test_the_public_surface_says_host_memory_busy_only_behind_its_auth_gate(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread, available_gib=20.0, public=True, rewarm=True)
    h.admission.reason_suffix = " (vm_stat at /usr/bin/vm_stat)"
    h.start()

    stranger = anyio.run(lambda: h.search("Bearer not-the-owner"))
    assert stranger.is_error
    assert text_of(stranger).startswith("UNAUTHORIZED")
    assert "memory" not in text_of(stranger)

    anonymous = anyio.run(lambda: h.search(None))
    assert anonymous.is_error and "memory" not in text_of(anonymous)

    owner = anyio.run(h.search)
    text = text_of(owner)
    assert text.startswith("WARMING_UP\nhost memory busy")
    assert "20.0 GiB available" in text
    assert "/usr/bin" not in text  # no text of a measurement, only numbers
    assert h.service.calls == 0


# --------------------------------------------------------------------------
# emergency release under memory pressure
# --------------------------------------------------------------------------


def test_critical_pressure_unloads_a_local_server_before_its_idle_timer(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread)
    h.start()
    assert h.unloader.release_if_under_pressure() is False  # normal pressure: keep them

    h.pressure = PressureLevel.CRITICAL
    assert h.unloader.release_if_under_pressure() is True  # no idle time has passed
    assert h.settle() is WarmUpPhase.UNLOADED
    assert h.models.events == ["load", "unload"]
    assert h.admission.releases == 1
    assert h.unloader.pressure_releases == 1
    assert any("critical memory pressure: unloading the models now" in line for line in h.log)

    # A call while the host is still critical is refused, not reloaded.
    h.admission.pressure = PressureLevel.CRITICAL
    assert text_of(anyio.run(h.search)).startswith("WARMING_UP\nhost memory busy")
    assert h.models.events == ["load", "unload"]


def test_warn_pressure_leaves_the_models_at_the_default_release_level(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread)
    h.start()
    h.pressure = PressureLevel.WARN
    assert h.unloader.release_if_under_pressure() is False
    assert h.models.loaded


def test_critical_pressure_waits_for_the_call_in_flight(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.start()
    h.pressure = PressureLevel.CRITICAL
    h.service.leave.clear()
    verdicts: list[bool] = []

    def watchdog() -> None:
        assert h.service.entered.wait(timeout=10)
        verdicts.append(h.unloader.release_if_under_pressure())
        h.service.leave.set()

    checker = threading.Thread(target=watchdog)
    checker.start()
    assert not anyio.run(h.search).is_error  # answered on the models it started with
    checker.join(timeout=10)
    assert verdicts == [False]
    assert h.unloader.release_if_under_pressure() is True  # the next check, call finished
    assert h.settle() is WarmUpPhase.UNLOADED


def test_a_public_server_set_to_never_keeps_its_models(model_thread: ModelThread) -> None:
    h = Harness(model_thread, public=True, release_at=None, rewarm=True)
    h.start()
    h.pressure = PressureLevel.CRITICAL
    assert h.unloader.release_if_under_pressure() is False
    assert h.models.loaded


def test_the_public_server_reloads_by_itself_after_the_cooldown(model_thread: ModelThread) -> None:
    h = Harness(model_thread, public=True, rewarm=True)
    h.start()
    h.pressure = PressureLevel.CRITICAL
    assert h.unloader.release_if_under_pressure() is True
    assert h.settle() is WarmUpPhase.UNLOADED

    h.pressure = PressureLevel.NORMAL
    h.clock.advance(COOLDOWN - 1)
    assert h.unloader.rewarm_if_due() is False
    h.clock.advance(1)
    assert h.unloader.rewarm_if_due() is True
    assert h.settle() is WarmUpPhase.READY
    assert h.models.events == ["load", "unload", "load"]


def test_the_public_server_retries_a_refused_startup_by_itself(model_thread: ModelThread) -> None:
    h = Harness(model_thread, available_gib=20.0, public=True, rewarm=True)
    assert h.start() is WarmUpPhase.MEMORY_BUSY
    h.admission.available_gib = 40.0
    h.clock.advance(RETRY)
    assert h.unloader.rewarm_if_due() is True
    assert h.settle() is WarmUpPhase.READY


def test_a_local_server_does_not_reload_by_itself(model_thread: ModelThread) -> None:
    h = Harness(model_thread, available_gib=20.0)
    h.start()
    h.admission.available_gib = 40.0
    h.clock.advance(10 * RETRY)
    assert h.unloader.rewarm_if_due() is False
    assert h.warm_up.status().phase is WarmUpPhase.MEMORY_BUSY  # waits for a call


def test_an_idle_unloaded_public_server_is_not_reloaded_by_itself(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread, public=True, rewarm=True)
    h.start()
    assert h.warm_up.mark_unloaded()  # as the idle timer would
    h.clock.advance(10 * COOLDOWN)
    assert h.unloader.rewarm_if_due() is False


def test_the_watchdog_runs_for_memory_pressure_alone(model_thread: ModelThread) -> None:
    h = Harness(model_thread, public=True, rewarm=False)  # idle unloading off
    h.start()
    assert h.unloader.watching and not h.unloader.enabled
    h.pressure = PressureLevel.CRITICAL
    h.unloader._pressure_release = PressureRelease(  # check every second, not every 5 s
        read_pressure=lambda: h.pressure, release_at=PressureLevel.CRITICAL, check_seconds=1.0
    )
    h.unloader.start_watchdog()
    try:
        deadline = time.monotonic() + 10
        while h.unloader.pressure_releases == 0:
            assert time.monotonic() < deadline, "the watchdog never released the models"
            time.sleep(0.1)
        assert h.settle() is WarmUpPhase.UNLOADED
    finally:
        h.unloader.stop()


# --------------------------------------------------------------------------
# the local server loads nothing until its first retrieval call
# --------------------------------------------------------------------------


def test_a_local_server_started_with_no_call_loads_nothing(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    assert isinstance(h.server, LocalMcpServer) and h.server.warm_at_start is False
    h.server.begin_serving()  # what run_local_server does once stdio is up
    try:
        time.sleep(0.2)
        assert h.settle() is WarmUpPhase.NOT_STARTED
        assert h.admission.checks == 0  # not even measured
        assert h.models.events == []
        assert h.unloader.watching  # the watchdog runs all the same
    finally:
        h.unloader.stop()


def test_the_first_call_starts_an_admitted_load_and_is_answered(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    h.server.begin_serving()
    try:
        result = anyio.run(h.search)
        assert not result.is_error
        assert h.admission.checks == 1
        assert h.models.events == ["load"]
        assert h.service.calls == 1
        assert any(line.startswith("memory admitted:") for line in h.log)
    finally:
        h.unloader.stop()


def test_the_first_call_is_answered_host_memory_busy_when_the_host_is_short(
    model_thread: ModelThread,
) -> None:
    h = Harness(model_thread, available_gib=20.0)
    h.server.begin_serving()
    try:
        assert text_of(anyio.run(h.search)).startswith("WARMING_UP\nhost memory busy")
        assert h.models.events == []
    finally:
        h.unloader.stop()


def test_warm_at_start_restores_loading_before_any_call(model_thread: ModelThread) -> None:
    h = Harness(model_thread)
    assert isinstance(h.server, LocalMcpServer)
    h.server.warm_at_start = True
    h.server.begin_serving()
    try:
        assert h.warm_up.wait(timeout=5).phase is WarmUpPhase.READY
        assert h.models.events == ["load"]
        assert h.service.calls == 0
    finally:
        h.unloader.stop()
