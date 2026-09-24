"""What heavy background commands check, and when (`imsg.background_gate`):
the pause switch before they start, memory admission before their models
load (a bounded, logged wait, then `deferred: memory`), and pause or
memory pressure between units of work."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ConfigDictFactory
from imsg.background_gate import (
    EXIT_DEFERRED_MEMORY,
    EXIT_DEFERRED_PAUSED,
    BackgroundGate,
    DeferralKind,
)
from imsg.background_pause import write_pause
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.host_memory import GIB, HostMemory, HostMemoryUnavailableError, PressureLevel
from imsg.memory_admission import MemoryAdmission, ModelRole, ReservationBook


class ScriptedProbe:
    """Readings in order; the last one repeats."""

    def __init__(
        self,
        available_gib: list[float] | None = None,
        pressures: list[PressureLevel | Exception] | None = None,
    ) -> None:
        self.available = available_gib or [512.0]
        self.pressures = pressures or [PressureLevel.NORMAL]
        self.reads = 0
        self.pressure_reads = 0

    def read(self) -> HostMemory:
        self.reads += 1
        gib = self.available[min(self.reads, len(self.available)) - 1]
        return HostMemory(64 * GIB, int(gib * GIB), 0, 0, PressureLevel.NORMAL)

    def pressure(self) -> PressureLevel:
        self.pressure_reads += 1
        reading = self.pressures[min(self.pressure_reads, len(self.pressures)) - 1]
        if isinstance(reading, Exception):
            raise reading
        return reading


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def cfg(config_dict_factory: ConfigDictFactory) -> Config:
    return load_config_dict(config_dict_factory())


def make_gate(
    cfg: Config, probe: ScriptedProbe, time: FakeTime, log: list[str], **memory: object
) -> BackgroundGate:
    return BackgroundGate(
        data_root=cfg.paths.data_root,
        host_pause_file=cfg.background.host_pause_file,
        memory=cfg.memory.model_copy(update=memory),
        probe=probe,
        log=log.append,
        sleep=time.sleep,
        clock=time.clock,
    )


def embed_admission(cfg: Config, probe: ScriptedProbe) -> MemoryAdmission:
    return MemoryAdmission.for_role(cfg, ModelRole.EMBED, command="imsg embed", probe=probe)


def test_the_deferral_exit_codes_are_distinct_from_success_and_failure() -> None:
    assert EXIT_DEFERRED_MEMORY == 75  # sysexits EX_TEMPFAIL
    assert EXIT_DEFERRED_PAUSED == 76
    assert len({0, 1, 2, EXIT_DEFERRED_MEMORY, EXIT_DEFERRED_PAUSED}) == 5


def test_a_pause_is_the_first_answer(cfg: Config) -> None:
    log: list[str] = []
    gate = make_gate(cfg, ScriptedProbe(), FakeTime(), log)
    assert gate.paused() is None
    write_pause(cfg.paths.data_root, reason="library import", until=None)
    reason = gate.paused()
    assert reason is not None
    assert reason.kind is DeferralKind.PAUSED and reason.exit_code == EXIT_DEFERRED_PAUSED
    assert "library import" in reason.line()
    assert reason.line().startswith("deferred: paused — ")


def test_admission_waits_logging_each_refusal_then_defers_for_memory(cfg: Config) -> None:
    time, log = FakeTime(), []
    probe = ScriptedProbe(available_gib=[10.0])  # never enough for 24.9 + 8
    gate = make_gate(cfg, probe, time, log, admission_wait_seconds=90.0, admission_poll_seconds=30.0)

    reason = gate.admit(embed_admission(cfg, probe))

    assert reason is not None
    assert reason.kind is DeferralKind.MEMORY and reason.exit_code == EXIT_DEFERRED_MEMORY
    assert reason.detail.startswith("host memory busy: 10.0 GiB available")
    assert time.sleeps == [30.0, 30.0, 30.0]
    waits = [line for line in log if line.startswith("waiting for memory")]
    assert [line.split(":")[0] for line in waits] == [
        "waiting for memory (0 of 90 s)",
        "waiting for memory (30 of 90 s)",
        "waiting for memory (60 of 90 s)",
    ]
    assert not ReservationBook(cfg.paths.data_root).reservations()


def test_admission_goes_ahead_as_soon_as_memory_frees(cfg: Config) -> None:
    time, log = FakeTime(), []
    probe = ScriptedProbe(available_gib=[10.0, 10.0, 40.0])
    gate = make_gate(cfg, probe, time, log, admission_wait_seconds=600.0, admission_poll_seconds=30.0)
    admission = embed_admission(cfg, probe)

    assert gate.admit(admission) is None
    assert time.sleeps == [30.0, 30.0]
    assert log[-1].startswith("memory admitted: 40.0 GiB available")
    assert [r.role for r in ReservationBook(cfg.paths.data_root).reservations()] == ["embed"]
    admission.release()


def test_a_pause_set_while_waiting_for_memory_wins(cfg: Config) -> None:
    time, log = FakeTime(), []

    class PausedDuringTheFirstCheck(ScriptedProbe):
        def read(self) -> HostMemory:
            if self.reads == 0:
                write_pause(cfg.paths.data_root, reason="import starting", until=None)
            return super().read()

    probe = PausedDuringTheFirstCheck(available_gib=[10.0])
    gate = make_gate(cfg, probe, time, log, admission_wait_seconds=600.0, admission_poll_seconds=30.0)
    reason = gate.admit(embed_admission(cfg, probe))
    assert reason is not None and reason.kind is DeferralKind.PAUSED
    assert time.sleeps == [30.0]  # stopped at the first re-check, not after ten minutes


def test_a_zero_wait_defers_at_once(cfg: Config) -> None:
    time, log = FakeTime(), []
    probe = ScriptedProbe(available_gib=[10.0])
    gate = make_gate(cfg, probe, time, log, admission_wait_seconds=0.0)
    reason = gate.admit(embed_admission(cfg, probe))
    assert reason is not None and reason.kind is DeferralKind.MEMORY
    assert time.sleeps == []


# --------------------------------------------------------------------------
# between units of work
# --------------------------------------------------------------------------


def test_normal_pressure_carries_on(cfg: Config) -> None:
    time = FakeTime()
    gate = make_gate(cfg, ScriptedProbe(pressures=[PressureLevel.NORMAL]), time, [])
    assert gate.between_units() is None
    assert time.sleeps == []


def test_critical_pressure_stops_at_once(cfg: Config) -> None:
    time = FakeTime()
    gate = make_gate(cfg, ScriptedProbe(pressures=[PressureLevel.CRITICAL]), time, [])
    reason = gate.between_units()
    assert reason is not None and reason.kind is DeferralKind.MEMORY
    assert "critical memory pressure" in reason.detail
    assert time.sleeps == []


def test_warn_pressure_stops_only_if_it_is_still_there_after_the_confirmation(cfg: Config) -> None:
    time, log = FakeTime(), []
    gate = make_gate(cfg, ScriptedProbe(pressures=[PressureLevel.WARN, PressureLevel.WARN]), time, log)
    reason = gate.between_units()
    assert reason is not None and reason.kind is DeferralKind.MEMORY
    assert time.sleeps == [10.0]  # memory.warn_confirm_seconds
    assert "warn memory pressure, and warn 10 s later" in reason.detail


def test_a_warn_blip_does_not_stop_the_work(cfg: Config) -> None:
    time, log = FakeTime(), []
    gate = make_gate(cfg, ScriptedProbe(pressures=[PressureLevel.WARN, PressureLevel.NORMAL]), time, log)
    assert gate.between_units() is None
    assert time.sleeps == [10.0]
    assert log[-1] == "memory pressure is normal again; carrying on"


def test_stopping_only_at_critical_ignores_warn(cfg: Config) -> None:
    time = FakeTime()
    gate = make_gate(
        cfg, ScriptedProbe(pressures=[PressureLevel.WARN]), time, [], background_stop_at="critical"
    )
    assert gate.between_units() is None
    assert time.sleeps == []


def test_pressure_that_cannot_be_read_stops_the_work(cfg: Config) -> None:
    gate = make_gate(
        cfg, ScriptedProbe(pressures=[HostMemoryUnavailableError("sysctl failed")]), FakeTime(), []
    )
    reason = gate.between_units()
    assert reason is not None and reason.kind is DeferralKind.MEMORY
    assert "could not be read" in reason.detail


def test_a_pause_stops_the_work_between_units(cfg: Config, tmp_path: Path) -> None:
    flag = cfg.background.host_pause_file
    assert flag is not None
    gate = make_gate(cfg, ScriptedProbe(), FakeTime(), [])
    assert gate.between_units() is None
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("reason=photo import\n")
    reason = gate.between_units()
    assert reason is not None and reason.kind is DeferralKind.PAUSED
    assert "photo import" in reason.detail
