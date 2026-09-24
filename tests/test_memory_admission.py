"""A model set loads only when the host has room for it
(`imsg.memory_admission`).

The rule is exercised on fixed numbers; the reservations are exercised
with real processes, because the failure they exist to prevent — several
servers started in the same second, each seeing the same free memory —
only exists across processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types as pytypes
from pathlib import Path
from typing import Any

import pytest

from conftest import ConfigDictFactory
from imsg import mlx_runtime
from imsg.config.loader import load_config_dict
from imsg.host_memory import GIB, HostMemory, HostMemoryUnavailableError, PressureLevel
from imsg.memory_admission import (
    MemoryAdmission,
    ModelRole,
    Reservation,
    ReservationBook,
    configure_mlx_memory_limit,
    decide,
    footprint_bytes,
    mlx_memory_limit_bytes,
)

FOOTPRINT = int(17.2 * GIB)
RESERVE = 8 * GIB


def memory(available_gib: float, pressure: PressureLevel = PressureLevel.NORMAL) -> HostMemory:
    return HostMemory(
        total_bytes=64 * GIB,
        free_bytes=int(available_gib * GIB),
        inactive_bytes=0,
        speculative_bytes=0,
        pressure=pressure,
    )


class FakeProbe:
    def __init__(self, *readings: HostMemory | Exception) -> None:
        self.readings = list(readings)
        self.reads = 0

    def read(self) -> HostMemory:
        self.reads += 1
        reading = self.readings[min(self.reads, len(self.readings)) - 1]
        if isinstance(reading, Exception):
            raise reading
        return reading

    def pressure(self) -> PressureLevel:
        reading = self.read()
        return reading.pressure


def admission(probe: FakeProbe, data_root: Path, **kwargs: Any) -> MemoryAdmission:
    footprint = {101: 0}
    return MemoryAdmission(
        probe=probe,
        book=kwargs.pop("book", ReservationBook(data_root, footprint=footprint.get)),
        role=ModelRole.LOCAL_SERVER,
        required_bytes=kwargs.pop("required_bytes", FOOTPRINT),
        reserve_bytes=kwargs.pop("reserve_bytes", RESERVE),
        max_pressure=kwargs.pop("max_pressure", PressureLevel.NORMAL),
        command="imsg mcp local",
    )


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def test_a_load_that_fits_with_the_reserve_is_admitted() -> None:
    decision = decide(
        memory(26.0),
        what="the local MCP server's models",
        required_bytes=FOOTPRINT,
        reserve_bytes=RESERVE,
        max_pressure=PressureLevel.NORMAL,
    )
    assert decision.admitted
    assert "17.2 GiB plus a reserve of 8.0 GiB" in decision.reason


def test_a_load_that_does_not_fit_is_refused_naming_the_numbers() -> None:
    decision = decide(
        memory(25.0),
        what="the local MCP server's models",
        required_bytes=FOOTPRINT,
        reserve_bytes=RESERVE,
        max_pressure=PressureLevel.NORMAL,
    )
    assert not decision.admitted
    assert decision.reason.startswith("host memory busy: 25.0 GiB available")
    assert "need 17.2 GiB plus a reserve of 8.0 GiB" in decision.reason


def test_warn_pressure_refuses_a_load_that_would_fit() -> None:
    decision = decide(
        memory(60.0, PressureLevel.WARN),
        what="the embedding models",
        required_bytes=FOOTPRINT,
        reserve_bytes=RESERVE,
        max_pressure=PressureLevel.NORMAL,
    )
    assert not decision.admitted
    assert "warn memory pressure" in decision.reason


@pytest.mark.parametrize("max_pressure", [PressureLevel.WARN, PressureLevel.CRITICAL])
def test_critical_pressure_refuses_every_load_whatever_the_setting(
    max_pressure: PressureLevel,
) -> None:
    decision = decide(
        memory(60.0, PressureLevel.CRITICAL),
        what="the embedding models",
        required_bytes=FOOTPRINT,
        reserve_bytes=RESERVE,
        max_pressure=max_pressure,
    )
    assert not decision.admitted


def test_memory_promised_to_other_loading_processes_does_not_count_as_free() -> None:
    decision = decide(
        memory(40.0),
        what="the local MCP server's models",
        required_bytes=FOOTPRINT,
        reserve_bytes=RESERVE,
        max_pressure=PressureLevel.NORMAL,
        promised_to_others_bytes=15 * GIB,
    )
    assert not decision.admitted
    assert "15.0 GiB of it promised to other model processes" in decision.reason


def test_the_public_summary_is_numbers_only(data_root: Path) -> None:
    probe = FakeProbe(HostMemoryUnavailableError("vm_stat could not run: /usr/bin/vm_stat"))
    decision = admission(probe, data_root).check()
    assert not decision.admitted
    assert "/usr/bin" in decision.reason
    assert "/" not in decision.public_summary()


# --------------------------------------------------------------------------
# failing closed
# --------------------------------------------------------------------------


def test_a_host_that_cannot_be_measured_is_refused(data_root: Path) -> None:
    probe = FakeProbe(HostMemoryUnavailableError("sysctl hw.memsize: Operation not permitted"))
    decision = admission(probe, data_root).check()
    assert not decision.admitted
    assert decision.reason.startswith("host memory could not be measured")
    assert not ReservationBook(data_root).reservations()


def test_any_unexpected_error_in_the_check_is_a_refusal(data_root: Path) -> None:
    decision = admission(FakeProbe(RuntimeError("something odd")), data_root).check()
    assert not decision.admitted
    assert "RuntimeError: something odd" in decision.reason


# --------------------------------------------------------------------------
# reservations
# --------------------------------------------------------------------------


def test_an_admitted_load_leaves_a_reservation_that_release_removes(data_root: Path) -> None:
    gate = admission(FakeProbe(memory(40.0)), data_root)
    assert gate.check().admitted
    [reservation] = ReservationBook(data_root).reservations()
    assert (reservation.pid, reservation.role, reservation.reserved_bytes) == (
        os.getpid(),
        "local_server",
        FOOTPRINT,
    )
    gate.release()
    gate.release()  # idempotent
    assert ReservationBook(data_root).reservations() == []


def test_another_processs_unfulfilled_reservation_is_subtracted(data_root: Path) -> None:
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        footprints = {sleeper.pid: 7 * GIB}  # it has taken up 7 of its 17.2 GiB so far
        book = ReservationBook(data_root, footprint=footprints.get)
        book.write(Reservation(sleeper.pid, "embed", FOOTPRINT, "2026-09-24T00:00:00+00:00", "x"))
        with book.locked():
            assert book.promised_to_others(os.getpid()) == FOOTPRINT - 7 * GIB
        # 33 GiB free, 10.2 GiB of it promised: 22.8 < 25.2 needed.
        decision = admission(FakeProbe(memory(33.0)), data_root, book=book).check()
        assert not decision.admitted
        assert "promised to other model processes" in decision.reason
    finally:
        sleeper.kill()
        sleeper.wait(timeout=30)


def test_a_dead_processs_reservation_is_swept_not_counted(data_root: Path) -> None:
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=30)
    book = ReservationBook(data_root)
    book.write(Reservation(gone.pid, "embed", 40 * GIB, "2026-09-24T00:00:00+00:00", "x"))
    assert admission(FakeProbe(memory(26.0)), data_root, book=book).check().admitted
    assert [r.pid for r in book.reservations()] == [os.getpid()]


_CHILD = """
import pathlib, sys
from imsg.host_memory import GIB, HostMemory, PressureLevel
from imsg.memory_admission import MemoryAdmission, ModelRole, ReservationBook

class Probe:
    def read(self):
        return HostMemory(64 * GIB, 50 * GIB, 0, 0, PressureLevel.NORMAL)
    def pressure(self):
        return PressureLevel.NORMAL

data_root, go, release = (pathlib.Path(p) for p in sys.argv[1:4])
check = MemoryAdmission(
    probe=Probe(),
    book=ReservationBook(data_root, footprint=lambda pid: 0),
    role=ModelRole.LOCAL_SERVER,
    required_bytes=int(17.2 * GIB),
    reserve_bytes=8 * GIB,
    max_pressure=PressureLevel.NORMAL,
    command="imsg mcp local",
)
import time
while not go.exists():
    time.sleep(0.001)
print("admitted" if check.check().admitted else "refused", flush=True)
while not release.exists():
    time.sleep(0.01)
"""


def test_five_servers_started_in_the_same_second_are_admitted_one_at_a_time(
    data_root: Path, tmp_path: Path
) -> None:
    """50 GiB free, 17.2 GiB each plus an 8 GiB reserve: two fit (50 - 17.2
    = 32.8 >= 25.2; 50 - 34.4 = 15.6 < 25.2). Without the lock and the
    reservations, all five would read 50 GiB free and all five would load."""
    go, release = tmp_path / "go", tmp_path / "release"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(data_root), str(go), str(release)],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(5)
    ]
    try:
        time.sleep(1.0)  # every child imported and is spinning on `go`
        go.write_text("x")
        answers = []
        for child in children:
            assert child.stdout is not None
            answers.append(child.stdout.readline().strip())
        assert sorted(answers) == ["admitted"] * 2 + ["refused"] * 3
        assert len(ReservationBook(data_root).reservations()) == 2
    finally:
        release.write_text("x")
        for child in children:
            child.wait(timeout=30)


# --------------------------------------------------------------------------
# per-role numbers, and the MLX memory limit
# --------------------------------------------------------------------------


def test_every_role_has_a_footprint_and_an_mlx_limit(config_dict_factory: ConfigDictFactory) -> None:
    cfg = load_config_dict(config_dict_factory())
    footprints = {role: footprint_bytes(cfg.memory, role) for role in ModelRole}
    limits = {role: mlx_memory_limit_bytes(cfg.memory, role) for role in ModelRole}
    assert footprints == {
        ModelRole.PUBLIC_SERVER: int(17.2 * GIB),
        ModelRole.LOCAL_SERVER: int(17.2 * GIB),
        ModelRole.EMBED: int(24.9 * GIB),
        ModelRole.ENRICH: int(24.8 * GIB),
        ModelRole.SEGMENT: int(21.0 * GIB),
    }
    assert limits == {
        ModelRole.PUBLIC_SERVER: 12 * GIB,
        ModelRole.LOCAL_SERVER: 12 * GIB,
        ModelRole.EMBED: 14 * GIB,
        ModelRole.ENRICH: 31 * GIB,
        ModelRole.SEGMENT: 24 * GIB,
    }
    assert cfg.memory.reserve_bytes == 8 * GIB


def _fake_mlx(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, int]]) -> None:
    mx = pytypes.ModuleType("mlx.core")

    def set_cache_limit(limit: int) -> int:
        calls.append(("set_cache_limit", limit))
        return 64 * GIB

    def set_memory_limit(limit: int) -> int:
        calls.append(("set_memory_limit", limit))
        return 96 * GIB

    mx.set_cache_limit = set_cache_limit  # type: ignore[attr-defined]
    mx.set_memory_limit = set_memory_limit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx.core", mx)


def test_a_roles_limit_is_applied_at_once_when_mlx_is_loaded(
    config_dict_factory: ConfigDictFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    _fake_mlx(monkeypatch, calls)
    cfg = load_config_dict(config_dict_factory(**{"models.backend": "real"}))
    assert configure_mlx_memory_limit(cfg, ModelRole.PUBLIC_SERVER) == 12 * GIB
    assert calls == [("set_memory_limit", 12 * GIB)]
    assert mlx_runtime.process_memory_limit() == 12 * GIB


def test_a_roles_limit_is_applied_when_the_first_provider_loads(
    config_dict_factory: ConfigDictFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    cfg = load_config_dict(config_dict_factory(**{"models.backend": "real"}))
    assert configure_mlx_memory_limit(cfg, ModelRole.ENRICH) == 31 * GIB
    assert "mlx.core" not in sys.modules  # recording it imported nothing

    calls: list[tuple[str, int]] = []
    _fake_mlx(monkeypatch, calls)
    mlx_runtime.bound_buffer_cache(4 * GIB)  # what every MLX provider calls at load
    assert calls == [("set_cache_limit", 4 * GIB), ("set_memory_limit", 31 * GIB)]


def test_the_fake_backend_and_a_zero_limit_leave_mlx_alone(
    config_dict_factory: ConfigDictFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    _fake_mlx(monkeypatch, calls)
    fake = load_config_dict(config_dict_factory())
    assert configure_mlx_memory_limit(fake, ModelRole.EMBED) is None
    zero = load_config_dict(
        config_dict_factory(
            **{"models.backend": "real", "memory": {"mlx_memory_limits": {"embed_bytes": 0}}}
        )
    )
    assert configure_mlx_memory_limit(zero, ModelRole.EMBED) is None
    mlx_runtime.bound_buffer_cache(8 * GIB)
    assert calls == [("set_cache_limit", 8 * GIB)]


def test_the_reservation_file_is_plain_json(data_root: Path) -> None:
    assert admission(FakeProbe(memory(40.0)), data_root).check().admitted
    [path] = (data_root / "run" / "memory-reservations").glob("*.json")
    document = json.loads(path.read_text())
    assert document["pid"] == os.getpid()
    assert document["reserved_bytes"] == FOOTPRINT
