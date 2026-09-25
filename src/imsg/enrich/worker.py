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
"""

from __future__ import annotations

import contextlib
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

logger = structlog.get_logger(__name__)


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
) -> WorkerReport:
    """Process up to `limit` tasks (module docstring). `claim` and
    `process` are the queue's and the pipeline's own functions unless a
    caller substitutes them."""
    report = WorkerReport()
    swept = sweep_stale_work_dirs(config.paths.data_root)
    if swept:
        logger.info("enrich.stale_work_dirs_removed", count=swept)

    def should_stop() -> bool:
        if stop_check is None:
            return False
        reason = stop_check()
        if reason is not None:
            report.stopped = reason
            logger.info("enrich.stopped", processed=report.processed, reason=reason.line())
        return reason is not None

    while report.processed < limit:
        if should_stop():
            break
        waited = yield_gate.wait_until_clear()
        report.yielded_seconds += waited.waited_seconds
        report.yield_pauses += int(waited.paused)
        if waited.paused and should_stop():
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


__all__ = ["WorkerReport", "YieldGate", "run_enrich_worker"]
