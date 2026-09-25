"""Live MCP servers load before background work (`imsg.memory_admission`,
"Live servers first"; `imsg.live_server_notice`; `imsg.background_gate`).

**The incident these reproduce (2026-09-25).** The public MCP server was
restarted with new code while a scheduled `imsg sync` was segmenting.
The sync had been admitted for its 21.0 GiB segmentation model about
nine minutes earlier and held 11.1 GiB, so admission counted 9.9 GiB of
"promised" memory against the server: 29.4 GiB available less 9.9 was
short of the 17.2 GiB it needs plus the 8.0 GiB reserve, though 29.4
alone covered it. The server stayed unloaded, and Gemini had no working
search for about 12 minutes, until the sync was stopped by hand.

The first tests run that scenario for real: `imsg mcp public` in this
process, and a background job in another process that holds a
reservation, exactly as sync's segmentation step does, and checks
between units of work. Real processes, because a reservation, a notice
and a lock only mean something across processes. The rest pin the
policy's details on a fake clock.

Modules and names this change adds are imported inside the tests that
need them, so the tests that drive only the old interfaces run, and fail
for what they check, against a build without the change.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from typer.testing import CliRunner

import imsg.cli as cli_module
from imsg.background_gate import EXIT_DEFERRED_MEMORY, BackgroundGate, DeferralKind
from imsg.cli import app
from imsg.config.schema import MemoryConfig
from imsg.host_memory import GIB, HostMemory, PressureLevel
from imsg.memory_admission import MemoryAdmission, ModelRole, Reservation, ReservationBook
from imsg.mount.guard import MountInfo
from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpPhase, WarmUpStep
from imsg.retrieval.idle_unload import IdleModelUnloader, PressureRelease
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import RetrievalService
from test_cli import _FakePgConn, _write_config, _write_public_enabled_config

runner = CliRunner()

PUBLIC_FOOTPRINT = int(17.2 * GIB)
RESERVE = 8 * GIB
SEGMENT_RESERVED = int(21.0 * GIB)
SEGMENT_HELD = int(11.1 * GIB)  # 9.9 GiB of the reservation not taken up
INCIDENT_AVAILABLE_GIB = 29.4
RETRY = 15.0
YIELD = 60.0
COOLDOWN = 300.0


def host_memory() -> ModuleType:
    return importlib.import_module("imsg.host_memory")


def _child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}


class FixedProbe:
    """`available_gib` available at `pressure`; counts full readings (one
    per admission check)."""

    def __init__(self, available_gib: float, pressure: PressureLevel = PressureLevel.NORMAL) -> None:
        self.available_gib = available_gib
        self.level = pressure
        self.reads: list[float] = []

    def read(self) -> HostMemory:
        self.reads.append(time.monotonic())
        return HostMemory(64 * GIB, int(self.available_gib * GIB), 0, 0, self.level)

    def pressure(self) -> PressureLevel:
        return self.level


# --------------------------------------------------------------------------
# a background job in its own process, as sync's segmentation step runs
# --------------------------------------------------------------------------

_BACKGROUND_JOB = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from imsg.background_gate import BackgroundGate
    from imsg.config.schema import MemoryConfig
    from imsg.host_memory import GIB, HostMemory, PressureLevel
    from imsg.memory_admission import MemoryAdmission, ModelRole, ReservationBook

    class Ample:
        def read(self):
            return HostMemory(64 * GIB, 60 * GIB, 0, 0, PressureLevel.NORMAL)
        def pressure(self):
            return PressureLevel.NORMAL

    data_root, mode, out, stop = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])

    def say(line):
        with out.open("a") as handle:
            handle.write(line + "\\n")

    probe = Ample()
    gate = BackgroundGate(
        data_root=data_root,
        host_pause_file=None,
        memory=MemoryConfig(admission_wait_seconds=0),
        probe=probe,
        log=lambda line: None,
    )
    admission = MemoryAdmission(
        probe=probe,
        book=ReservationBook(data_root),
        role=ModelRole.SEGMENT,
        required_bytes=int(21.0 * GIB),
        reserve_bytes=8 * GIB,
        max_pressure=PressureLevel.NORMAL,
        command="imsg sync",
    )
    reason = gate.admit(admission)
    if reason is not None:
        say(reason.line())
        sys.exit(reason.exit_code)
    say("admitted")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not stop.exists():
        time.sleep(0.05)  # one unit of work: one chat's segmentation
        if mode == "units":  # the check sync makes between chats
            reason = gate.between_units()
            if reason is not None:
                admission.release()  # sync drops its models and reservation
                say(f"stopped_at {time.time()}")
                say(reason.line())
                sys.exit(reason.exit_code)
    admission.release()
    say("ran to the end")
    sys.exit(3)
    """
)


class BackgroundJob:
    """`imsg sync`'s segmentation step in its own process: admitted for
    21.0 GiB, then units of work; `mode="units"` checks between them,
    `mode="never"` never does (a unit that does not end, eval, an older
    build) and holds its reservation until `stop()`."""

    def __init__(self, data_root: Path, tmp_path: Path, mode: str) -> None:
        self.out = tmp_path / f"job-{mode}.log"
        self.stop_file = tmp_path / f"job-{mode}.stop"
        self.process = subprocess.Popen(
            [sys.executable, "-c", _BACKGROUND_JOB, str(data_root), mode, str(self.out), str(self.stop_file)],
            env=_child_env(),
        )

    @property
    def pid(self) -> int:
        return self.process.pid

    def lines(self) -> list[str]:
        return self.out.read_text().splitlines() if self.out.exists() else []

    def wait_for(self, predicate: Callable[[list[str]], bool], timeout: float = 30.0) -> list[str]:
        deadline = time.monotonic() + timeout
        while not predicate(self.lines()):
            if time.monotonic() > deadline:
                raise AssertionError(f"background job said {self.lines()!r}")
            time.sleep(0.02)
        return self.lines()

    def stop(self) -> None:
        self.stop_file.write_text("x")

    def exit_code(self, timeout: float) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=30)


@pytest.fixture
def jobs(data_root: Path, tmp_path: Path) -> Iterator[Callable[[str], BackgroundJob]]:
    started: list[BackgroundJob] = []

    def start(mode: str) -> BackgroundJob:
        job = BackgroundJob(data_root, tmp_path, mode)
        started.append(job)
        return job

    yield start
    for job in started:
        job.kill()


# --------------------------------------------------------------------------
# `imsg mcp public` in this process
# --------------------------------------------------------------------------


@pytest.fixture
def public_config(data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A public-server config (fake backend) with the mount and the
    database mocked, as in `tests/test_cli_memory_aware.py`."""
    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", "300000000000000000009")
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    config_path = tmp_path / "config.yaml"
    _write_public_enabled_config(config_path, data_root, messages_dir)
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakePgConn())
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    return config_path


def set_memory(config: Path, **values: float) -> None:
    lines = "".join(f"  {key}: {value}\n" for key, value in values.items())
    config.write_text(config.read_text() + f"memory:\n{lines}")


class PublicRun:
    """Runs `imsg mcp public`; `during(public)` runs while it serves (in
    place of uvicorn), with the warm-up and the watchdog live."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import uvicorn

        self.loads: list[float] = []
        self.publics: list[Any] = []
        real_build = cli_module.build_public_asgi_app

        def build(public: Any, **kwargs: Any) -> Any:
            self.publics.append(public)
            return real_build(public, **kwargs)

        monkeypatch.setattr(cli_module, "build_public_asgi_app", build)
        monkeypatch.setattr(
            RetrievalService,
            "warm_up_steps",
            lambda service: [WarmUpStep("query models", 1.0, lambda: self.loads.append(1.0))],
        )
        self._uvicorn = uvicorn
        self._monkeypatch = monkeypatch

    def run(self, config: Path, during: Callable[[Any], None]) -> Any:
        self._monkeypatch.setattr(self._uvicorn, "run", lambda app_arg, **kw: during(self.publics[0]))
        return runner.invoke(app, ["mcp", "public", "--config", str(config)])


def wait_for_phase(public: Any, phases: set[WarmUpPhase], timeout: float) -> WarmUpPhase:
    deadline = time.monotonic() + timeout
    phase: WarmUpPhase = public.warm_up.status().phase
    while phase not in phases and time.monotonic() < deadline:
        time.sleep(0.02)
        phase = public.warm_up.status().phase
    return phase


def test_the_public_server_loads_though_a_background_job_holds_a_reservation(
    public_config: Path,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs: Callable[[str], BackgroundJob],
) -> None:
    """The 2026-09-25 scenario. A sync-like job holds a 21.0 GiB
    reservation it has barely taken up when the public server starts on a
    host with 29.4 GiB available. The server's first check is refused for
    that promise; the job stops at its next unit and drops its
    reservation; the server loads at its next check, one retry interval
    later. Before the fix the job never stopped and the server never
    loaded."""
    set_memory(public_config, admission_retry_seconds=0.2, pressure_check_seconds=0.1)
    job = jobs("units")
    job.wait_for(lambda lines: "admitted" in lines)
    probe = FixedProbe(INCIDENT_AVAILABLE_GIB)
    monkeypatch.setattr(host_memory(), "default_probe", lambda: probe)
    run = PublicRun(monkeypatch)
    seen: dict[str, Any] = {}

    def during(public: Any) -> None:
        seen["phase"] = wait_for_phase(public, {WarmUpPhase.READY, WarmUpPhase.FAILED}, timeout=15)
        seen["ready_at"] = time.time()
        seen["job_exit"] = job.exit_code(timeout=5)

    result = run.run(public_config, during)

    assert result.exit_code == 0, result.output
    assert seen["phase"] is WarmUpPhase.READY, (
        f"the public server did not load while a background job held a reservation "
        f"(phase {seen['phase']}); stderr: {result.stderr}"
    )
    assert run.loads == [1.0]
    assert seen["job_exit"] == EXIT_DEFERRED_MEMORY
    [stopped_at] = [float(line.split()[1]) for line in job.lines() if line.startswith("stopped_at")]
    [stopped] = [line for line in job.lines() if line.startswith("deferred:")]
    assert stopped.startswith(
        f"deferred: memory — a live MCP server is waiting to load its models "
        f"(imsg mcp public (pid {os.getpid()}) since "
    )
    assert "not loading the models — host memory busy: 29.4 GiB available" in result.stderr
    assert f"imsg sync pid {job.pid}, background: 21.0 GiB" in result.stderr
    assert "host memory is available again: 29.4 GiB available" in result.stderr
    # Loaded at the first try after the job stopped: one retry interval
    # (0.2 s here), with room for a busy test machine's scheduling.
    assert seen["ready_at"] - stopped_at < 0.2 + 1.8, (seen["ready_at"], stopped_at, probe.reads)


def test_no_background_load_starts_while_the_public_server_waits_for_memory(
    public_config: Path,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs: Callable[[str], BackgroundJob],
) -> None:
    """The host really is short (10 GiB), so the public server waits. A
    background job that starts meanwhile, on a probe that says the host
    has room for it, is refused and defers with exit 75 instead of
    loading."""
    set_memory(public_config, admission_retry_seconds=0.2, pressure_check_seconds=0.1)
    monkeypatch.setattr(host_memory(), "default_probe", lambda: FixedProbe(10.0))
    run = PublicRun(monkeypatch)
    seen: dict[str, Any] = {}

    def during(public: Any) -> None:
        seen["phase"] = wait_for_phase(public, {WarmUpPhase.MEMORY_BUSY}, timeout=10)
        job = jobs("units")
        seen["job"] = job
        seen["job_exit"] = job.exit_code(timeout=15)
        seen["reservations"] = [r.pid for r in ReservationBook(data_root).reservations()]

    result = run.run(public_config, during)

    assert result.exit_code == 0, result.output
    assert seen["phase"] is WarmUpPhase.MEMORY_BUSY
    job = seen["job"]
    assert seen["job_exit"] == EXIT_DEFERRED_MEMORY, (
        f"a background job loaded while the public server waited: {job.lines()!r}"
    )
    [line] = job.lines()
    assert line.startswith(
        f"deferred: memory — a live MCP server is waiting to load its models "
        f"(imsg mcp public (pid {os.getpid()}) since "
    )
    assert job.pid not in seen["reservations"]
    assert run.loads == []


def test_a_refused_public_server_tries_again_every_retry_interval(
    public_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While its models are not loaded, the public server re-checks every
    `memory.admission_retry_seconds`, on time, even when its watchdog's
    regular pass (`memory.pressure_check_seconds`) is slower. Scaled down:
    0.2 s tries against a 2 s pass."""
    set_memory(public_config, admission_retry_seconds=0.2, pressure_check_seconds=2.0)
    probe = FixedProbe(10.0)
    monkeypatch.setattr(host_memory(), "default_probe", lambda: probe)
    run = PublicRun(monkeypatch)

    def during(public: Any) -> None:
        wait_for_phase(public, {WarmUpPhase.MEMORY_BUSY}, timeout=10)
        time.sleep(1.5)

    result = run.run(public_config, during)

    assert result.exit_code == 0, result.output
    gaps = [later - earlier for earlier, later in itertools.pairwise(probe.reads)]
    assert len(probe.reads) >= 5, f"{len(probe.reads)} checks in 1.5 s: {gaps}"
    assert max(gaps) < 0.6, gaps
    assert "the load is tried again every 0.2 s" in result.stderr


# --------------------------------------------------------------------------
# imsg status and imsg background status
# --------------------------------------------------------------------------

_CHILD_POSTS_AND_PUBLISHES = textwrap.dedent(
    """
    import pathlib, sys, time
    from imsg.host_memory import GIB, HostMemory, PressureLevel
    from imsg.live_server_notice import LiveServerNotice
    from imsg.mcp.warm_up_readiness import WarmUpReadinessFile, readiness_path
    from imsg.memory_admission import OtherReservation, Reservation, decide
    from imsg.retrieval.background_warm_up import WarmUpPhase, WarmUpStatus

    data_root, ready = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    LiveServerNotice(data_root, role="public_server", command="imsg mcp public").post()
    sync = OtherReservation(
        Reservation(4242, "segment", int(21.0 * GIB), "2026-09-25T20:17:00+00:00", "imsg sync", False),
        int(11.1 * GIB),
    )
    decision = decide(
        HostMemory(64 * GIB, int(29.4 * GIB), 0, 0, PressureLevel.NORMAL),
        what="the public MCP server's models",
        required_bytes=int(17.2 * GIB),
        reserve_bytes=8 * GIB,
        max_pressure=PressureLevel.NORMAL,
        promised_to_others_bytes=sync.promised_bytes,
        counted=[sync],
    )
    WarmUpReadinessFile(readiness_path(data_root)).publish(
        WarmUpStatus(WarmUpPhase.MEMORY_BUSY, None, 0, 4, 0.0, 12.0, None, admission=decision)
    )
    ready.write_text("x")
    time.sleep(60)
    """
)


@pytest.fixture
def status_config(data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, data_root, messages_dir)
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakePgConn())
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    return config_path


def test_status_shows_the_waiting_server_and_what_it_waits_on(
    status_config: Path, data_root: Path, tmp_path: Path
) -> None:
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_POSTS_AND_PUBLISHES, str(data_root), str(ready)],
        env=_child_env(),
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert time.monotonic() < deadline, "the child never posted its notice"
            time.sleep(0.02)

        result = runner.invoke(app, ["status", "--config", str(status_config), "--json"])
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        [waiting] = report["live_servers_waiting"]
        assert waiting.startswith(f"imsg mcp public (pid {child.pid}) since ")
        assert report["mcp_public_warm_up"] == "memory_busy"
        assert report["mcp_public_waiting_on"] == [
            "imsg sync pid 4242 (background, segment): admitted for 21.0 GiB, holds 11.1 GiB, "
            "9.9 GiB not taken up yet, counted against this load"
        ]
        document = json.loads((data_root / "run" / "mcp-public-warm-up.json").read_text())
        [row] = document["waiting_on"]
        assert row["promised_bytes"] == int(21.0 * GIB) - int(11.1 * GIB)
        assert set(row) == {
            "pid", "command", "role", "live", "reserved_bytes", "held_bytes",
            "promised_bytes", "counted", "admitted_at",
        }

        text = runner.invoke(app, ["status", "--config", str(status_config)]).output
        assert "live_servers_waiting:\n  imsg mcp public (pid" in text
        assert "mcp_public_waiting_on:\n  imsg sync pid 4242 (background, segment)" in text

        background = runner.invoke(app, ["background", "status", "--config", str(status_config)])
        assert background.exit_code == 0, background.output
        assert (
            f"background: giving way to a live MCP server waiting for memory: imsg mcp public "
            f"(pid {child.pid})"
        ) in background.output
    finally:
        child.kill()
        child.wait(timeout=30)

    # The server is gone: nothing is waiting any more.
    report = json.loads(runner.invoke(app, ["status", "--config", str(status_config), "--json"]).output)
    assert report["live_servers_waiting"] == []


# --------------------------------------------------------------------------
# the policy on a fake clock
# --------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeModels:
    def __init__(self) -> None:
        self.events: list[str] = []

    def load(self) -> None:
        self.events.append("load")

    def unload(self) -> None:
        self.events.append("unload")


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-live-priority")
    yield thread
    thread.close()


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A running process to own a reservation: another process's
    reservation counts only while that process runs."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield process
    process.kill()
    process.wait(timeout=30)


class LiveHarness:
    """A live server's warm-up, watchdog and admission on a fake clock,
    wired as `imsg mcp public` (or, `public=False`, `imsg mcp local`) wires
    them, over a real reservation book and real notices."""

    def __init__(
        self,
        data_root: Path,
        model_thread: ModelThread,
        *,
        available_gib: float,
        public: bool = True,
        yield_seconds: float = YIELD,
        retry_seconds: float = RETRY,
        check_seconds: float = 5.0,
    ) -> None:
        from imsg.live_server_notice import LiveServerNotice
        from imsg.memory_admission import LiveServerAdmission

        self.clock = FakeClock()
        self.probe = FixedProbe(available_gib)
        self.pressure = PressureLevel.NORMAL
        self.footprints: dict[int, int] = {}
        self.book = ReservationBook(data_root, footprint=self.footprints.get)
        self.log: list[str] = []
        self.models = FakeModels()
        self.model_thread = model_thread
        role = ModelRole.PUBLIC_SERVER if public else ModelRole.LOCAL_SERVER
        command = "imsg mcp public" if public else "imsg mcp local"
        self.admission = LiveServerAdmission(
            MemoryAdmission(
                probe=self.probe,
                book=self.book,
                role=role,
                required_bytes=PUBLIC_FOOTPRINT,
                reserve_bytes=RESERVE,
                max_pressure=PressureLevel.NORMAL,
                command=command,
            ),
            LiveServerNotice(data_root, role=role.value, command=command, clock=self.clock),
            background_yield_seconds=yield_seconds,
            clock=self.clock,
            log=self.log.append,
        )
        self.warm_up = BackgroundWarmUp(
            [WarmUpStep("query models", 1.0, self.models.load)],
            model_thread=model_thread,
            log=self.log.append,
            clock=self.clock,
            admission=self.admission.check,
            admission_retry_seconds=retry_seconds,
            retries_by_itself=public,
        )
        self.unloader = IdleModelUnloader(
            warm_up=self.warm_up,
            model_thread=model_thread,
            unload=self.models.unload,
            idle_seconds=0 if public else 600,
            log=self.log.append,
            clock=self.clock,
            pressure_release=PressureRelease(
                read_pressure=lambda: self.pressure,
                release_at=PressureLevel.CRITICAL,
                check_seconds=check_seconds,
                rewarm=public,
                rewarm_cooldown_seconds=COOLDOWN,
            ),
            after_unload=self.admission.release,
            after_pressure_unload=self.admission.release_for_reload if public else None,
            on_watchdog_pass=self.admission.tick,
        )

    def settle(self) -> WarmUpPhase:
        self.model_thread.run(lambda: None)
        return self.warm_up.status().phase

    def start(self) -> WarmUpPhase:
        self.warm_up.start()
        return self.settle()

    def watchdog_pass(self) -> float:
        """Sleep as the watchdog would, then make one pass; returns how
        long it slept."""
        wait = self.unloader.next_check_in()
        self.clock.advance(wait)
        self.unloader.check_once()
        self.settle()
        return wait

    def run_watchdog_for(self, seconds: float) -> None:
        end = self.clock.now + seconds
        while self.clock.now + self.unloader.next_check_in() <= end + 1e-9:
            self.watchdog_pass()

    def refusals(self) -> list[str]:
        return [line for line in self.log if line.startswith("not loading the models")]


def background_reservation(
    h: LiveHarness, pid: int, *, role: str = "segment", reserved: int = SEGMENT_RESERVED,
    held: int = SEGMENT_HELD, live: bool = False, command: str = "imsg sync",
) -> None:
    h.book.write(Reservation(pid, role, reserved, "2026-09-25T20:17:00+00:00", command, live))
    h.footprints[pid] = held


def background_gate(data_root: Path, log: list[str] | None = None) -> BackgroundGate:
    return BackgroundGate(
        data_root=data_root,
        host_pause_file=None,
        memory=MemoryConfig(admission_wait_seconds=0),
        probe=FixedProbe(500.0),
        log=(log if log is not None else []).append,
    )


def embed_admission(data_root: Path, **kwargs: Any) -> MemoryAdmission:
    return MemoryAdmission(
        probe=FixedProbe(500.0),
        book=ReservationBook(data_root),
        role=kwargs.pop("role", ModelRole.EMBED),
        required_bytes=int(24.9 * GIB),
        reserve_bytes=RESERVE,
        max_pressure=PressureLevel.NORMAL,
        command=kwargs.pop("command", "imsg embed"),
        **kwargs,
    )


def notices(data_root: Path) -> list[str]:
    from imsg.live_server_notice import waiting_live_servers

    return [server.command for server in waiting_live_servers(data_root)]


def test_background_promises_count_for_the_first_minute_then_give_way(
    data_root: Path, model_thread: ModelThread, sleeper: subprocess.Popen[bytes]
) -> None:
    """A job that never stops (a long unit, eval, an older build) holds
    the server off for `memory.background_yield_seconds` at most. The
    server then loads on the memory really available, and keeps its
    notice up until the job has stopped, so the job cannot grow into the
    memory it was promised."""
    h = LiveHarness(data_root, model_thread, available_gib=INCIDENT_AVAILABLE_GIB)
    background_reservation(h, sleeper.pid)

    assert h.start() is WarmUpPhase.MEMORY_BUSY
    [refusal] = h.refusals()
    assert "9.9 GiB of it promised to other model processes still loading" in refusal
    assert f"(imsg sync pid {sleeper.pid}, background: 9.9 GiB)" in refusal
    assert "background work has been asked to give way" in refusal
    assert "for the first 60 s of the wait" in refusal
    assert notices(data_root) == ["imsg mcp public"]
    # What the public surface may say stays numbers only: no pid, no command.
    status = h.warm_up.status()
    assert status.admission is not None
    summary = status.admission.public_summary()
    assert "29.4 GiB available" in summary
    assert "pid" not in summary and "imsg" not in summary

    h.run_watchdog_for(YIELD - 1)
    assert h.warm_up.status().phase is WarmUpPhase.MEMORY_BUSY
    assert h.models.events == []

    h.run_watchdog_for(RETRY)
    assert h.warm_up.status().phase is WarmUpPhase.READY
    assert h.models.events == ["load"]
    assert any(
        line.startswith("host memory is available again")
        and f"9.9 GiB promised to background work not counted (imsg sync pid {sleeper.pid}" in line
        for line in h.log
    )
    # Loaded past the job's promise: the notice stays up while the job runs,
    # so no background load starts and the job stops at its next unit.
    assert notices(data_root) == ["imsg mcp public"]
    refused = embed_admission(data_root).check()
    assert not refused.admitted
    assert refused.reason.startswith("a live MCP server is waiting to load its models")
    h.run_watchdog_for(30)
    assert notices(data_root) == ["imsg mcp public"]

    h.book.remove(sleeper.pid)  # the job stopped and dropped its reservation
    h.watchdog_pass()
    assert notices(data_root) == []
    assert any("the background jobs this server loaded past have stopped" in line for line in h.log)
    admitted = embed_admission(data_root)
    assert admitted.check().admitted
    admitted.release()


def test_a_job_that_gives_way_lets_the_server_load_at_its_next_check(
    data_root: Path, model_thread: ModelThread, sleeper: subprocess.Popen[bytes]
) -> None:
    """The normal case: nothing is set aside, because the job stops first."""
    h = LiveHarness(data_root, model_thread, available_gib=INCIDENT_AVAILABLE_GIB)
    background_reservation(h, sleeper.pid)
    assert h.start() is WarmUpPhase.MEMORY_BUSY
    gate = background_gate(data_root)
    gate._models_admitted = True  # a command that holds models, like sync's segmentation
    stop = gate.between_units()
    assert stop is not None and stop.kind is DeferralKind.MEMORY
    assert stop.exit_code == EXIT_DEFERRED_MEMORY
    h.book.remove(sleeper.pid)  # what sync does on that stop

    started = h.clock.now
    while h.warm_up.status().phase is not WarmUpPhase.READY:
        h.watchdog_pass()
        assert h.clock.now - started <= RETRY, "not loaded at the next try"
    assert h.clock.now - started == RETRY
    assert notices(data_root) == []  # nothing was loaded past


def test_real_memory_the_reserve_and_pressure_still_count(
    data_root: Path, model_thread: ModelThread, sleeper: subprocess.Popen[bytes]
) -> None:
    """Background promises give way; the host's real memory never does."""
    h = LiveHarness(data_root, model_thread, available_gib=20.0)  # < 17.2 + 8 with nothing promised
    background_reservation(h, sleeper.pid)
    h.start()
    h.run_watchdog_for(10 * YIELD)
    assert h.warm_up.status().phase is WarmUpPhase.MEMORY_BUSY
    assert h.models.events == []
    status = h.warm_up.status()
    assert status.admission is not None
    assert status.admission.reason.startswith("host memory busy: 20.0 GiB available")
    assert "promised to background work not counted" in status.admission.reason

    h.probe.available_gib = 500.0
    h.probe.level = PressureLevel.WARN
    h.run_watchdog_for(RETRY)
    status = h.warm_up.status()
    assert status.phase is WarmUpPhase.MEMORY_BUSY
    assert status.admission is not None and "warn memory pressure" in status.admission.reason

    h.probe.level = PressureLevel.NORMAL
    h.run_watchdog_for(RETRY)
    assert h.warm_up.status().phase is WarmUpPhase.READY


def test_other_live_servers_promises_always_count(
    data_root: Path, model_thread: ModelThread, sleeper: subprocess.Popen[bytes]
) -> None:
    """Five local servers started in the same second are still admitted
    one at a time: live servers give way to no one but each other."""
    h = LiveHarness(data_root, model_thread, available_gib=INCIDENT_AVAILABLE_GIB)
    background_reservation(
        h, sleeper.pid, role="local_server", reserved=PUBLIC_FOOTPRINT, held=0, live=True,
        command="imsg mcp local",
    )
    h.start()
    h.run_watchdog_for(10 * YIELD)
    status = h.warm_up.status()
    assert status.phase is WarmUpPhase.MEMORY_BUSY
    assert status.admission is not None
    assert f"imsg mcp local pid {sleeper.pid}, live server: 17.2 GiB" in status.admission.reason


def test_a_dead_processs_reservation_never_blocks_the_live_server(
    data_root: Path, model_thread: ModelThread
) -> None:
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=30)
    h = LiveHarness(data_root, model_thread, available_gib=INCIDENT_AVAILABLE_GIB)
    h.book.write(Reservation(gone.pid, "segment", 40 * GIB, "2026-09-25T20:17:00+00:00", "imsg sync"))
    assert h.start() is WarmUpPhase.READY
    assert [r.pid for r in h.book.reservations()] == [os.getpid()]
    assert notices(data_root) == []


_CHILD_POSTS = """
import pathlib, sys, time
from imsg.live_server_notice import LiveServerNotice
LiveServerNotice(pathlib.Path(sys.argv[1]), role="public_server", command="imsg mcp public").post()
pathlib.Path(sys.argv[2]).write_text("x")
time.sleep(60)
"""


def test_a_dead_live_servers_notice_never_blocks_background_work(
    data_root: Path, tmp_path: Path
) -> None:
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_POSTS, str(data_root), str(ready)], env=_child_env()
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert not embed_admission(data_root).check().admitted
    finally:
        child.kill()
        child.wait(timeout=30)
    gate = background_gate(data_root)
    admission = embed_admission(data_root)
    assert gate.admit(admission) is None
    assert gate.between_units() is None
    admission.release()


def test_background_work_that_holds_models_stops_between_units_for_a_waiting_live_server(
    data_root: Path,
) -> None:
    from imsg.live_server_notice import LiveServerNotice

    gate = background_gate(data_root)
    admission = embed_admission(data_root)
    assert gate.admit(admission) is None
    assert gate.between_units() is None

    notice = LiveServerNotice(data_root, role="local_server", command="imsg mcp local")
    notice.post()
    try:
        stop = gate.between_units()
        assert stop is not None and stop.exit_code == EXIT_DEFERRED_MEMORY
        assert stop.line().startswith(
            f"deferred: memory — a live MCP server is waiting to load its models "
            f"(imsg mcp local (pid {os.getpid()}) since "
        )
        assert stop.line().endswith("background work gives way, stopping after this unit of work")
    finally:
        notice.withdraw()
        admission.release()
    assert gate.between_units() is None


def test_work_that_loads_no_models_does_not_stop_for_a_live_server(data_root: Path) -> None:
    """Attachment backfill never passes admission: it holds no models to
    give back, so a waiting server gains nothing from stopping it."""
    from imsg.live_server_notice import LiveServerNotice

    gate = background_gate(data_root)
    notice = LiveServerNotice(data_root, role="public_server", command="imsg mcp public")
    notice.post()
    try:
        assert gate.between_units() is None
    finally:
        notice.withdraw()


def test_unreadable_notices_stop_background_work(data_root: Path) -> None:
    def unreadable() -> list[Any]:
        raise PermissionError("denied")

    gate = BackgroundGate(
        data_root=data_root,
        host_pause_file=None,
        memory=MemoryConfig(admission_wait_seconds=0),
        probe=FixedProbe(500.0),
        log=lambda line: None,
        live_servers_waiting=unreadable,
    )
    admission = embed_admission(data_root)
    assert gate.admit(admission) is None
    stop = gate.between_units()
    assert stop is not None and stop.kind is DeferralKind.MEMORY
    assert "could not be read (PermissionError: denied)" in stop.detail
    admission.release()
    refusal = embed_admission(data_root, live_servers_waiting=unreadable).check()
    assert not refusal.admitted and "PermissionError: denied" in refusal.reason


def test_eval_is_background_work_though_it_loads_the_query_models(
    data_root: Path, config_dict_factory: Any
) -> None:
    from imsg.config.loader import load_config_dict
    from imsg.live_server_notice import LiveServerNotice

    cfg = load_config_dict(config_dict_factory())
    evaluation = MemoryAdmission.for_role(
        cfg, ModelRole.LOCAL_SERVER, command="eval run", probe=FixedProbe(500.0), live=False
    )
    local = MemoryAdmission.for_role(
        cfg, ModelRole.LOCAL_SERVER, command="imsg mcp local", probe=FixedProbe(500.0)
    )
    assert (evaluation.live, local.live) == (False, True)
    notice = LiveServerNotice(data_root, role="public_server", command="imsg mcp public")
    notice.post()
    try:
        assert not evaluation.check().admitted
        assert local.check().admitted  # a live server is never refused for another's notice
    finally:
        notice.withdraw()
        local.release()
    assert evaluation.check().admitted
    document = json.loads(
        (data_root / "run" / "memory-reservations" / f"{os.getpid()}.json").read_text()
    )
    assert document["live"] is False
    evaluation.release()


def test_a_reservation_written_before_the_live_flag_goes_by_its_role(data_root: Path) -> None:
    directory = data_root / "run" / "memory-reservations"
    directory.mkdir(parents=True)
    for pid, role in ((101, "local_server"), (102, "embed")):
        (directory / f"{pid}.json").write_text(
            json.dumps(
                {"pid": pid, "role": role, "reserved_bytes": GIB, "admitted_at": "x", "command": "y"}
            )
        )
    by_pid = {r.pid: r for r in ReservationBook(data_root).reservations()}
    assert (by_pid[101].is_live, by_pid[102].is_live) == (True, False)


# --------------------------------------------------------------------------
# emergency release and its cooldown
# --------------------------------------------------------------------------


def test_a_pressure_release_still_unloads_at_once_and_reloads_after_the_cooldown(
    data_root: Path, model_thread: ModelThread
) -> None:
    """Unchanged: critical pressure unloads the public server at once, and
    it reloads by itself only after `memory.public_rewarm_cooldown_seconds`.
    New: during the cooldown its notice holds background loads back."""
    h = LiveHarness(data_root, model_thread, available_gib=100.0)
    assert h.start() is WarmUpPhase.READY
    assert notices(data_root) == []

    h.pressure = PressureLevel.CRITICAL
    assert h.unloader.release_if_under_pressure() is True
    assert h.settle() is WarmUpPhase.UNLOADED
    assert h.models.events == ["load", "unload"]
    assert [r.pid for r in h.book.reservations()] == []  # the reservation went with the models
    assert notices(data_root) == ["imsg mcp public"]
    assert not embed_admission(data_root).check().admitted

    h.pressure = PressureLevel.NORMAL
    h.clock.advance(COOLDOWN - 1)
    assert h.unloader.rewarm_if_due() is False
    assert h.unloader.next_check_in() == pytest.approx(1.0)
    assert h.watchdog_pass() == pytest.approx(1.0)
    assert h.settle() is WarmUpPhase.READY
    assert h.models.events == ["load", "unload", "load"]
    assert notices(data_root) == []
    admission = embed_admission(data_root)
    assert admission.check().admitted
    admission.release()


def test_a_local_servers_pressure_release_holds_nothing_back(
    data_root: Path, model_thread: ModelThread
) -> None:
    """A local server does not reload by itself, so it posts nothing when
    it releases its models under pressure."""
    h = LiveHarness(data_root, model_thread, available_gib=100.0, public=False)
    assert h.start() is WarmUpPhase.READY
    h.pressure = PressureLevel.CRITICAL
    assert h.unloader.release_if_under_pressure() is True
    assert h.settle() is WarmUpPhase.UNLOADED
    assert notices(data_root) == []


# --------------------------------------------------------------------------
# how often a waiting server tries, and what it says
# --------------------------------------------------------------------------


def test_a_refused_public_server_tries_every_15_seconds_for_as_long_as_it_waits(
    data_root: Path, model_thread: ModelThread
) -> None:
    """Default numbers: tries every 15 s (`memory.admission_retry_seconds`),
    watchdog pass every 5 s. Ten minutes of waiting is 40 more tries."""
    h = LiveHarness(data_root, model_thread, available_gib=10.0)
    h.start()
    before = len(h.probe.reads)
    started = h.clock.now
    tries: list[float] = []
    while h.clock.now - started < 600:
        h.watchdog_pass()
        if len(h.probe.reads) > before + len(tries):
            tries.append(h.clock.now)
    gaps = {round(later - earlier, 6) for earlier, later in itertools.pairwise(tries)}
    assert len(tries) == 40
    assert gaps == {RETRY}
    assert notices(data_root) == ["imsg mcp public"]  # never lapses while it tries


def test_the_watchdog_wakes_when_a_try_is_due_not_at_its_next_regular_pass(
    data_root: Path, model_thread: ModelThread
) -> None:
    h = LiveHarness(data_root, model_thread, available_gib=10.0, check_seconds=20.0)
    h.start()
    assert h.unloader.next_check_in() == RETRY
    h.clock.advance(10)
    assert h.unloader.next_check_in() == pytest.approx(RETRY - 10)
    h.clock.advance(RETRY - 10)
    assert h.unloader.next_check_in() > 0  # never a busy loop


def test_a_waiting_server_logs_what_it_waits_on_when_that_changes_and_at_least_once_a_minute(
    data_root: Path, model_thread: ModelThread, sleeper: subprocess.Popen[bytes]
) -> None:
    h = LiveHarness(data_root, model_thread, available_gib=20.0)
    background_reservation(h, sleeper.pid)
    h.start()
    assert len(h.refusals()) == 1
    assert h.refusals()[0].endswith("the load is tried again every 15 s")

    h.run_watchdog_for(45)  # three more tries, waiting on the same job
    assert len(h.probe.reads) == 4
    assert len(h.refusals()) == 1

    h.book.remove(sleeper.pid)  # the job stopped; the host is still short
    h.run_watchdog_for(RETRY)
    assert len(h.refusals()) == 2
    assert "promised" not in h.refusals()[1].split("host memory busy:")[1].split(";")[0]

    h.run_watchdog_for(60)
    assert len(h.refusals()) == 3  # the same wait, said again a minute later


def test_a_local_servers_notice_lapses_when_its_client_stops_asking(
    data_root: Path, model_thread: ModelThread
) -> None:
    from imsg.memory_admission import NOTICE_LAPSE_SECONDS

    h = LiveHarness(data_root, model_thread, available_gib=10.0, public=False)
    assert h.start() is WarmUpPhase.MEMORY_BUSY  # a retrieval call started it
    assert notices(data_root) == ["imsg mcp local"]
    h.clock.advance(NOTICE_LAPSE_SECONDS - 1)
    h.unloader.check_once()
    assert notices(data_root) == ["imsg mcp local"]
    h.clock.advance(1)
    h.unloader.check_once()
    assert notices(data_root) == []
    assert any("no load has been tried for 120 s" in line for line in h.log)

    assert h.start() is WarmUpPhase.MEMORY_BUSY  # the client asks again
    assert notices(data_root) == ["imsg mcp local"]


def test_a_notice_that_cannot_be_posted_changes_no_decision(
    data_root: Path, model_thread: ModelThread, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.live_server_notice import LiveServerNotice

    def broken(self: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(LiveServerNotice, "post", broken)
    h = LiveHarness(data_root, model_thread, available_gib=10.0)
    assert h.start() is WarmUpPhase.MEMORY_BUSY
    h.run_watchdog_for(3 * RETRY)
    failures = [line for line in h.log if line.startswith("could not post the notice")]
    assert len(failures) == 1 and "read-only file system" in failures[0]
    h.probe.available_gib = 100.0
    h.run_watchdog_for(RETRY)
    assert h.warm_up.status().phase is WarmUpPhase.READY


def test_the_warm_up_file_says_what_the_public_server_waits_on(
    data_root: Path, model_thread: ModelThread, sleeper: subprocess.Popen[bytes]
) -> None:
    from imsg.mcp.warm_up_readiness import (
        WarmUpReadinessFile,
        read_warm_up_readiness,
        readiness_path,
    )

    h = LiveHarness(data_root, model_thread, available_gib=20.0)
    h.warm_up._on_status = WarmUpReadinessFile(readiness_path(data_root)).publish
    background_reservation(h, sleeper.pid)
    h.start()
    document = json.loads(readiness_path(data_root).read_text())
    assert document["phase"] == "memory_busy"
    assert document["memory_retry_seconds"] == RETRY
    [row] = document["waiting_on"]
    assert (row["pid"], row["command"], row["live"], row["counted"]) == (
        sleeper.pid, "imsg sync", False, True,
    )
    report = read_warm_up_readiness(data_root)
    assert report.detail.endswith("the server retries by itself every 15 s")
    assert report.waiting_on == (
        f"imsg sync pid {sleeper.pid} (background, segment): admitted for 21.0 GiB, holds "
        f"11.1 GiB, 9.9 GiB not taken up yet, counted against this load",
    )

    h.run_watchdog_for(YIELD)  # the job had its minute: its promise no longer counts
    [row] = json.loads(readiness_path(data_root).read_text())["waiting_on"]
    assert row["counted"] is False
    assert read_warm_up_readiness(data_root).waiting_on[0].endswith(
        "not counted: background work gives way"
    )
