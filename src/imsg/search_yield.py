"""Segmentation and embedding wait while a search is in flight, as the
enrichment worker already does (2026-09-26).

**Why.** `imsg sync`'s segmentation and embedding share the GPU with the
MCP servers. Measured on an M2 Ultra (2026-09-25): one process embedding
in the sync's batch shape beside the public server made its reranking
2.3x slower with the bf16 reranker and 3.6x slower with the mxfp8 one it
replaced; two such processes made it 6.4x and 10.4x slower. The
enrichment worker has waited between tasks while a search is in flight
since 2026-09-17 (`imsg.db.enrichment_yield_locks`). These steps now use
the same gate, the same marker and the same settings.

**Where they wait.** Before a unit of GPU work, never inside one:

- segmentation: before each call to the boundary model (one window of one
  session of one chat), in `imsg sync`, `imsg segment` and
  `imsg segment --rebuild` (:class:`YieldingBoundaryProvider`);
- embedding: before each text batch and each image or video, where the
  pause switch and memory pressure are already checked
  (:meth:`SearchYield.stop_check`), in `imsg sync` and `imsg embed`.

**What counts as a search.** Anything holding the query-in-flight marker:
a public or local MCP search for its whole length (model calls and the
database work between them), the servers' warm-up, and each embedding or
rerank the search page's model API computes.

**Bounded.** One wait lasts at most `enrichment.yield_max_pause_seconds`
(300 s by default); then the unit runs anyway, so a stream of searches
cannot starve background work. `enrichment.yield_poll_interval_seconds`
sets how soon a waiting step notices the search has finished, and
`enrichment.yield_to_queries` turns the whole mechanism off, on the
servers' side and on every background step's. The names are the
enrichment worker's because it had them first.

**The pause switch and a waiting live server still win.** While it
waits, a step also asks whether heavy background work has been paused
or a live MCP server is waiting for memory (`imsg.background_gate`), and
stops waiting as soon as either is true. Embedding then asks its usual
stop check again, as it does after every wait (the enrichment worker's
order). Segmentation's stop check stays between chats: a chat is one
transaction.

**It never slows a search.** A search marks itself with a try-lock that
never waits, and the step probes with one (`QueryYieldGate`).

**It does not preempt.** A search that starts while a batch or a model
call is running shares the GPU with it until that unit ends. What this
stops is the next unit starting during a search. Measured on the M2
Ultra (2026-09-26, CHANGELOG): beside a sync-shaped embedding load, 20
back-to-back reranks went from 2.6x slower to 1.0x; reranks 3 s apart
stayed 2.4-2.5x slower, because each met a batch already running.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from imsg.db.enrichment_yield_locks import QueryYieldGate, YieldReport

if TYPE_CHECKING:
    import psycopg

    from imsg.background_gate import StopCheck, StopReason
    from imsg.config.schema import Config
    from imsg.segment.boundaries import BoundaryProvider
    from imsg.segment.models import MessageForSegmentation


class YieldStep(StrEnum):
    """A heavy step that waits for searches; the value names its log
    events (``segment.yielding``, ``embed.resumed``) and its yielding key
    in `imsg status`."""

    SEGMENT = "segment"
    EMBED = "embed"

    @property
    def label(self) -> str:
        return "segmentation" if self is YieldStep.SEGMENT else "embedding"


@dataclass
class YieldTally:
    """What one step's waits added up to over one command."""

    waits: int = 0
    seconds: float = 0.0
    gave_up: int = 0
    """Waits that reached `enrichment.yield_max_pause_seconds` with a
    search still in flight, after which the unit ran anyway."""

    def add(self, report: YieldReport) -> None:
        if not report.paused:
            return
        self.waits += 1
        self.seconds += report.waited_seconds
        self.gave_up += int(report.gave_up)

    def describe(self, max_pause_seconds: float) -> str | None:
        """`yielded to in-flight searches N time(s), X.Xs total`, or
        `None` when the step never waited."""
        if not self.waits:
            return None
        line = f"yielded to in-flight searches {self.waits} time(s), {self.seconds:.1f}s total"
        if self.gave_up:
            line += (
                f"; went ahead after the {max_pause_seconds:g} s limit {self.gave_up} time(s) "
                f"(enrichment.yield_max_pause_seconds)"
            )
        return line


class WaitGate(Protocol):
    def wait_until_clear(self, *, interrupt: Callable[[], bool] | None = None) -> YieldReport: ...


class SearchYield:
    """One step's waiting for searches within one command: the gate on the
    command's own connection, what to ask while waiting, and a tally the
    command reports when it ends."""

    def __init__(
        self,
        gate: WaitGate,
        *,
        tally: YieldTally | None = None,
        interrupt: Callable[[], bool] | None = None,
    ) -> None:
        self._gate = gate
        self._interrupt = interrupt
        self.tally = tally if tally is not None else YieldTally()

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        conn: psycopg.Connection,
        step: YieldStep,
        *,
        tally: YieldTally | None = None,
        interrupt: Callable[[], bool] | None = None,
    ) -> SearchYield:
        """The gate enrichment uses, with enrichment's settings, publishing
        `step`'s yielding key on `conn` (the command's own connection, the
        one its step writes on) while it waits."""
        gate = QueryYieldGate(
            conn,
            step=step.value,
            enabled=cfg.enrichment.yield_to_queries,
            poll_interval_seconds=cfg.enrichment.yield_poll_interval_seconds,
            max_pause_seconds=cfg.enrichment.yield_max_pause_seconds,
        )
        return cls(gate, tally=tally, interrupt=interrupt)

    def wait(self) -> YieldReport:
        """Wait, bounded, while a search is in flight; one round trip and
        no wait when none is."""
        report = self._gate.wait_until_clear(interrupt=self._interrupt)
        self.tally.add(report)
        return report

    def stop_check(self, check: StopCheck | None) -> StopCheck:
        """`check` (asked between units: paused, memory, a waiting live
        server) with the wait for searches after it: a reason to stop is
        returned at once, without waiting; otherwise the step waits while
        a search is in flight, and after any wait `check` is asked again,
        because the wait can be long. The enrichment worker's order
        (`imsg.enrich.worker`)."""

        def between_units() -> StopReason | None:
            if check is not None:
                reason = check()
                if reason is not None:
                    return reason
            waited = self.wait()
            if waited.paused and check is not None:
                return check()
            return None

        return between_units


class YieldingBoundaryProvider:
    """The boundary model, waiting before each call while a search is in
    flight. Every call the segmentation pipeline makes goes through
    `detect_boundaries`, including its one retry after a malformed
    answer, so every model call waits."""

    def __init__(self, inner: BoundaryProvider, search_yield: SearchYield) -> None:
        self.inner = inner
        self.search_yield = search_yield
        self.model_id = inner.model_id

    def detect_boundaries(self, window: Sequence[MessageForSegmentation]) -> list[int]:
        self.search_yield.wait()
        return self.inner.detect_boundaries(window)

    def unload(self) -> None:
        unload = getattr(self.inner, "unload", None)
        if callable(unload):
            unload()


__all__ = [
    "SearchYield",
    "WaitGate",
    "YieldStep",
    "YieldTally",
    "YieldingBoundaryProvider",
]
