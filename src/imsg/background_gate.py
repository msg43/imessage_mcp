"""What a heavy background command checks, and when.

Heavy background commands are segmentation, embedding, the enrichment
worker, attachment backfill and `imsg sync`'s segmentation and embedding
steps. Each asks this gate three things:

1. **Before it starts: is background work paused?**
   (`imsg.background_pause`.) If so it does no heavy work and exits
   `EXIT_DEFERRED_PAUSED`.
2. **Before its models load: does the host have the memory?**
   (`imsg.memory_admission`.) Asked after the command holds the
   host-wide heavy-model lock (`imsg.heavy_lock`), so memory freed by
   the lock's previous holder counts. The answer is also no while a live
   MCP server is waiting to load its models (`imsg.live_server_notice`):
   they load first. A refusal is re-checked every
   `memory.admission_poll_seconds`, each wait logged, for up to
   `memory.admission_wait_seconds`; still refused, the command releases
   the lock and exits `EXIT_DEFERRED_MEMORY`.
3. **Between units of work** (one enrichment task, one embedding batch,
   one chat's segmentation, one attachment copied): paused again, a live
   MCP server waiting to load its models (asked only by a command that
   has loaded models: attachment backfill holds none to give back), or
   the kernel reporting `memory.background_stop_at` pressure (warn by
   default) or worse? If so the command stops after the unit it is on:
   that unit is finished and committed (an enrichment task's lease is
   released by completing it), nothing new is claimed, the models, the
   memory reservation and the heavy lock are released, and the command
   exits with the matching code. A waiting live server and critical
   pressure stop it at once; a warn reading must still be there
   `memory.warn_confirm_seconds` later, because on the production host
   warn came and went in single samples (2 of 133 in the 2026-09-17
   overlap run). The next scheduled run carries on from the queue and
   the dirty flags, so a stop loses no work.

The MCP servers never ask the pause question: they keep answering, and
their own memory checks are in `imsg.retrieval.background_warm_up` and
`imsg.retrieval.idle_unload`.

**Exit codes.** `EXIT_DEFERRED_MEMORY` is 75, sysexits' `EX_TEMPFAIL`
("try again later"), which is what it means. `EXIT_DEFERRED_PAUSED` is
76, the next number: this project's own code for "deferred because
background work is paused", not sysexits' meaning for 76. Both differ
from 0 (did its work) and 1 (failed), so a LaunchAgent's last exit
status says which it was.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from imsg import host_memory
from imsg.background_pause import PauseState, read_pause_state
from imsg.errors import ImsgError
from imsg.host_memory import HostMemoryProbe, PressureLevel
from imsg.live_server_notice import WaitingLiveServer, waiting_live_servers

if TYPE_CHECKING:
    from imsg.config.schema import Config, MemoryConfig
    from imsg.memory_admission import MemoryAdmission

EXIT_DEFERRED_MEMORY = 75
EXIT_DEFERRED_PAUSED = 76


class DeferralKind(StrEnum):
    PAUSED = "paused"
    MEMORY = "memory"


@dataclass(frozen=True, slots=True)
class StopReason:
    kind: DeferralKind
    detail: str

    @property
    def exit_code(self) -> int:
        return EXIT_DEFERRED_PAUSED if self.kind is DeferralKind.PAUSED else EXIT_DEFERRED_MEMORY

    def line(self) -> str:
        return f"deferred: {self.kind.value} — {self.detail}"


StopCheck = Callable[[], StopReason | None]
"""Asked between units of work; a reason means stop after this unit."""


class BackgroundWorkDeferred(ImsgError):
    """Heavy background work did not run, or stopped between units, for
    `reason`. `partial` carries what was done before the stop, when a
    pipeline raises this mid-run (`imsg.segment.pipeline.run_segment`)."""

    def __init__(self, reason: StopReason, *, partial: object = None) -> None:
        self.reason = reason
        self.partial = partial
        super().__init__(reason.line())


def pause_reason(state: PauseState) -> StopReason | None:
    if not state.paused:
        return None
    return StopReason(
        DeferralKind.PAUSED,
        f"heavy background work is {state.describe()} — `imsg background resume` "
        f"(or removing the host pause file) lets it run",
    )


class BackgroundGate:
    """The three checks in the module docstring, for one command."""

    def __init__(
        self,
        *,
        data_root: Path,
        host_pause_file: Path | None,
        memory: MemoryConfig,
        probe: HostMemoryProbe,
        log: Callable[[str], None],
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        live_servers_waiting: Callable[[], Sequence[WaitingLiveServer]] | None = None,
    ) -> None:
        self._data_root = data_root
        self._host_pause_file = host_pause_file
        self._memory = memory
        self._probe = probe
        self._log = log
        self._sleep = sleep
        self._clock = clock
        self._live_servers_waiting = (
            live_servers_waiting
            if live_servers_waiting is not None
            else lambda: waiting_live_servers(data_root)
        )
        self._models_admitted = False

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        log: Callable[[str], None],
        probe: HostMemoryProbe | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> BackgroundGate:
        return cls(
            data_root=cfg.paths.data_root,
            host_pause_file=cfg.background.host_pause_file,
            memory=cfg.memory,
            probe=probe if probe is not None else host_memory.default_probe(),
            log=log,
            sleep=sleep,
        )

    @property
    def probe(self) -> HostMemoryProbe:
        return self._probe

    def pause_state(self) -> PauseState:
        return read_pause_state(self._data_root, host_pause_file=self._host_pause_file)

    def paused(self) -> StopReason | None:
        """Check 1."""
        return pause_reason(self.pause_state())

    def admit(self, admission: MemoryAdmission) -> StopReason | None:
        """Check 2: `None` once admitted (the reservation is then held
        until `admission.release()`), else the reason to defer. While
        refused it re-checks every `memory.admission_poll_seconds`, logging
        each refusal, for up to `memory.admission_wait_seconds`. The pause
        switch is asked first and between checks too, because the heavy
        lock this runs under can take a long time to come free, and a pause
        set meanwhile should win over a wait for memory."""
        paused = self.paused()
        if paused is not None:
            return paused
        max_wait = self._memory.admission_wait_seconds
        poll = self._memory.admission_poll_seconds
        started = self._clock()
        decision = admission.check()
        while not decision.admitted:
            waited = self._clock() - started
            if waited >= max_wait:
                return StopReason(DeferralKind.MEMORY, decision.reason)
            self._log(f"waiting for memory ({waited:.0f} of {max_wait:.0f} s): {decision.reason}")
            self._sleep(min(poll, max_wait - waited))
            paused = self.paused()
            if paused is not None:
                return paused
            decision = admission.check()
        self._models_admitted = True
        self._log(f"memory admitted: {decision.reason}")
        return None

    def _pressure(self) -> tuple[PressureLevel | None, str | None]:
        """The pressure level, or `None` and why it could not be read."""
        try:
            return self._probe.pressure(), None
        except Exception as exc:  # unreadable: the caller stops, never guesses
            return None, f"{type(exc).__name__}: {exc}"

    def live_server_waiting(self) -> StopReason | None:
        """Part of check 3: a live MCP server waiting to load its models
        stops background work that holds models. A notice directory that
        cannot be read stops it too, as an unreadable pressure level does:
        the command never guesses that the host can spare its memory."""
        try:
            waiting = self._live_servers_waiting()
        except Exception as exc:
            return StopReason(
                DeferralKind.MEMORY,
                f"whether a live MCP server is waiting for memory could not be read "
                f"({type(exc).__name__}: {exc})",
            )
        if not waiting:
            return None
        who = "; ".join(server.describe() for server in waiting)
        return StopReason(
            DeferralKind.MEMORY,
            f"a live MCP server is waiting to load its models ({who}); background work "
            f"gives way, stopping after this unit of work",
        )

    def between_units(self) -> StopReason | None:
        """Check 3."""
        paused = self.paused()
        if paused is not None:
            return paused
        if self._models_admitted:
            waiting = self.live_server_waiting()
            if waiting is not None:
                return waiting
        stop_at = PressureLevel(self._memory.background_stop_at)
        level, error = self._pressure()
        if level is None:
            return StopReason(DeferralKind.MEMORY, f"memory pressure could not be read ({error})")
        if not level.at_least(stop_at):
            return None
        confirm = self._memory.warn_confirm_seconds
        if level is PressureLevel.CRITICAL or confirm <= 0:
            return StopReason(DeferralKind.MEMORY, f"the kernel reports {level.value} memory pressure")
        self._log(
            f"the kernel reports {level.value} memory pressure; checking again in "
            f"{confirm:g} s before stopping"
        )
        self._sleep(confirm)
        again, error = self._pressure()
        if again is None:
            return StopReason(DeferralKind.MEMORY, f"memory pressure could not be read ({error})")
        if again.at_least(stop_at):
            return StopReason(
                DeferralKind.MEMORY,
                f"the kernel reported {level.value} memory pressure, and "
                f"{again.value} {confirm:g} s later",
            )
        self._log(f"memory pressure is {again.value} again; carrying on")
        return None


__all__ = [
    "EXIT_DEFERRED_MEMORY",
    "EXIT_DEFERRED_PAUSED",
    "BackgroundGate",
    "BackgroundWorkDeferred",
    "DeferralKind",
    "StopCheck",
    "StopReason",
    "pause_reason",
]
