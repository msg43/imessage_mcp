"""The enrichment worker loop (SPEC §8 S5b): claim one task, process it,
repeat — and stop cleanly between two tasks.

Moved here from `imsg enrich` (2026-09-24) when stopping between tasks
was added, so that behaviour is tested against a real queue.

**One task at a time.** A task is claimed with a lease
(`imsg.enrich.queue`) and processed to the end, which completes, fails
or skips it and so clears its lease, before the next one is claimed. So
between two tasks the worker holds no lease, and that is where it:

1. asks the stop check (`imsg.background_gate`): background work paused,
   or the host short of memory, and it stops here, holding nothing;
2. yields to in-flight queries (`imsg.db.enrichment_yield_locks`, D10.3:
   enrichment waits for searches, never the reverse);
3. asks the stop check again, because the yield can wait a while.

**Interrupted mid-task.** A task whose processing raises something the
pipeline does not handle — Ctrl-C, launchd's SIGTERM surfacing as an
exception, a bug — has its lease released before the exception goes on
(`release_task`: back to `pending`, its attempt not counted). It would
otherwise sit `running` until the lease expired, 30 minutes by default.

**Work directories left behind.** Each task works in its own directory
under `<data_root>/artifacts/enrich-work` (`imsg.enrich.pipeline`). A
worker killed mid-task leaves its directory behind, so each run first
removes those whose process is gone (`sweep_stale_work_dirs`).

**Handing the heavy-model lock to a waiting command (2026-09-24, QA
review).** One 5,000-task run held the host-wide lock from 22:36 to
05:04, and the scheduled `imsg sync` waited behind it the whole time, so
new messages were not segmented, embedded or searchable for hours. Right
before each claim the worker now calls `lock_handoff`
(`HeavyLockHandoff`): when another command is waiting for the lock, it
unloads the enrichment models, drops the memory reservation, releases
the lock, waits for the lock to come back, and passes memory admission
again (pause switch included) before claiming the next task. A waiting
sync therefore waits for the task in hand, not for the run. The two model
sets are never loaded under the lock together: the lock is released only
after the enrichment models are dropped, and taken back before any of
them load again.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import structlog

from imsg.background_gate import StopCheck, StopReason
from imsg.enrich.pipeline import EnrichmentProviders, process_one_task, sweep_stale_work_dirs
from imsg.enrich.queue import EnrichmentTask, claim_tasks, release_task

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config
    from imsg.db.enrichment_yield_locks import YieldReport
    from imsg.heavy_lock import HeavyModelLock

logger = structlog.get_logger(__name__)

LockHandoff = Callable[[], StopReason | None]
"""Asked right before each claim: hand the heavy-model lock to a command
waiting for it, if one is, and get it back. A reason means the worker must
stop instead (paused, or refused memory, while the lock was away)."""


@dataclass
class HeavyLockHandoff:
    """`LockHandoff` for a worker holding the host-wide heavy-model lock
    (`imsg.heavy_lock`, created with `yields_to_waiters=True`).

    Each step is a callable so the order is visible here and testable:
    `unload_models` drops the enrichment models and returns their memory,
    `release_reservation` removes this process's memory reservation, then
    the lock is released and taken back (waiting as long as the other
    command runs), and `readmit` runs memory admission again, returning a
    reason to stop if the host is paused or short of memory."""

    lock: HeavyModelLock
    unload_models: Callable[[], None]
    release_reservation: Callable[[], None]
    readmit: Callable[[], StopReason | None]
    log: Callable[[str], None]
    clock: Callable[[], float] = time.monotonic
    handoffs: int = 0
    seconds_away: float = 0.0

    def __call__(self) -> StopReason | None:
        if not self.lock.waiter_present():
            return None
        self.log(
            "another model-heavy command is waiting for the host-wide lock; "
            "unloading the enrichment models and handing the lock over"
        )
        self.unload_models()
        self.release_reservation()
        self.lock.release()
        started = self.clock()
        self.lock.acquire(wait=True)
        away = self.clock() - started
        self.handoffs += 1
        self.seconds_away += away
        self.log(f"took the host-wide lock back after {away:.1f} s")
        return self.readmit()


class YieldGate(Protocol):
    def wait_until_clear(self) -> YieldReport: ...


ClaimFn = Callable[..., list[EnrichmentTask]]
ProcessFn = Callable[["psycopg.Connection", "Config", EnrichmentProviders, EnrichmentTask], str]


@dataclass
class WorkerReport:
    processed: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    yielded_seconds: float = 0.0
    yield_pauses: int = 0
    stopped: StopReason | None = None
    """Why the worker stopped before its limit or an empty queue."""


def run_enrich_worker(
    conn: psycopg.Connection,
    config: Config,
    providers: EnrichmentProviders,
    *,
    worker_id: str,
    claim_order: Sequence[str],
    limit: int,
    yield_gate: YieldGate,
    stop_check: StopCheck | None = None,
    claim: ClaimFn = claim_tasks,
    process: ProcessFn = process_one_task,
    lock_handoff: LockHandoff | None = None,
) -> WorkerReport:
    """Process up to `limit` tasks (module docstring). `claim` and
    `process` are the queue's and the pipeline's own functions unless a
    caller substitutes them. `lock_handoff` is asked right before each
    claim, holding no lease."""
    report = WorkerReport()
    swept = sweep_stale_work_dirs(config.paths.data_root)
    if swept:
        logger.info("enrich.stale_work_dirs_removed", count=swept)

    def stop_for(reason: StopReason | None) -> bool:
        if reason is not None:
            report.stopped = reason
            logger.info("enrich.stopped", processed=report.processed, reason=reason.line())
        return reason is not None

    def should_stop() -> bool:
        return stop_check is not None and stop_for(stop_check())

    while report.processed < limit:
        if should_stop():
            break
        waited = yield_gate.wait_until_clear()
        report.yielded_seconds += waited.waited_seconds
        report.yield_pauses += int(waited.paused)
        if waited.paused and should_stop():
            break
        if lock_handoff is not None and stop_for(lock_handoff()):
            break
        tasks = claim(conn, worker_id=worker_id, limit=1, kinds=claim_order)
        if not tasks:
            break
        for task in tasks:
            try:
                outcome = process(conn, config, providers, task)
            except BaseException:
                with contextlib.suppress(Exception):
                    release_task(conn, task.attachment_id, task.kind, worker_id=worker_id)
                raise
            report.outcomes[outcome] = report.outcomes.get(outcome, 0) + 1
            report.processed += 1
    return report


__all__ = ["HeavyLockHandoff", "LockHandoff", "WorkerReport", "YieldGate", "run_enrich_worker"]
