"""The enrichment worker hands the host-wide heavy-model lock to a waiting
command between tasks (`imsg.enrich.worker`; QA review 2026-09-24). No
database: the worker's claim and process steps are stand-ins here, and
`tests/test_cli_memory_aware.py` drives the same path through
`imsg enrich` with a real second process waiting for the lock."""

from __future__ import annotations

from typing import Any

import pytest

from conftest import ConfigDictFactory
from imsg.background_gate import DeferralKind, StopReason
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.enrichment_yield_locks import YieldReport
from imsg.enrich.queue import EnrichmentTask
from imsg.enrich.worker import HeavyLockHandoff, run_enrich_worker

PAUSED = StopReason(DeferralKind.PAUSED, "heavy background work is paused: test")


@pytest.fixture
def config(config_dict_factory: ConfigDictFactory) -> Config:
    return load_config_dict(config_dict_factory())


class NeverYields:
    def wait_until_clear(self) -> YieldReport:
        return YieldReport(paused=False, waited_seconds=0.0)


def test_the_handoff_is_asked_right_before_every_claim(config: Config) -> None:
    events: list[str] = []
    served = iter(range(1, 100))

    def claim(conn: Any, **kwargs: Any) -> list[EnrichmentTask]:
        events.append("claim")
        return [EnrichmentTask(attachment_id=next(served), kind="ocr", attempts=0)]

    def process(conn: Any, cfg: Any, providers: Any, task: EnrichmentTask) -> str:
        events.append("process")
        return "done"

    def handoff() -> StopReason | None:
        events.append("handoff")
        return None

    report = run_enrich_worker(
        None,  # type: ignore[arg-type]
        config,
        providers=None,  # type: ignore[arg-type]
        worker_id="w",
        claim_order=("ocr",),
        limit=3,
        yield_gate=NeverYields(),
        claim=claim,
        process=process,
        lock_handoff=handoff,
    )
    assert report.processed == 3
    assert events == ["handoff", "claim", "process"] * 3


def test_a_stop_reason_from_the_handoff_stops_the_worker_before_it_claims(config: Config) -> None:
    """Paused (or refused memory) while the lock was away: the worker
    stops holding no lease, and claims nothing more."""
    answers = iter([None, PAUSED])
    claims: list[int] = []

    def claim(conn: Any, **kwargs: Any) -> list[EnrichmentTask]:
        claims.append(1)
        return [EnrichmentTask(attachment_id=len(claims), kind="ocr", attempts=0)]

    report = run_enrich_worker(
        None,  # type: ignore[arg-type]
        config,
        providers=None,  # type: ignore[arg-type]
        worker_id="w",
        claim_order=("ocr",),
        limit=10,
        yield_gate=NeverYields(),
        claim=claim,
        process=lambda *args: "done",
        lock_handoff=lambda: next(answers),
    )
    assert report.processed == 1
    assert len(claims) == 1
    assert report.stopped == PAUSED


class FakeLock:
    def __init__(self, events: list[str], waiting: bool) -> None:
        self.events = events
        self.waiting = waiting

    def waiter_present(self) -> bool:
        self.events.append("waiter_present")
        return self.waiting

    def release(self) -> None:
        self.events.append("release lock")

    def acquire(self, *, wait: bool | None = None) -> None:
        self.events.append(f"acquire lock (wait={wait})")


def _handoff(events: list[str], *, waiting: bool, readmit: StopReason | None = None) -> HeavyLockHandoff:
    return HeavyLockHandoff(
        lock=FakeLock(events, waiting),  # type: ignore[arg-type]
        unload_models=lambda: events.append("unload models"),
        release_reservation=lambda: events.append("release reservation"),
        readmit=lambda: events.append("readmit") or readmit,  # type: ignore[func-returns-value]
        log=lambda line: None,
    )


def test_with_nobody_waiting_the_handoff_does_nothing() -> None:
    events: list[str] = []
    handoff = _handoff(events, waiting=False)
    assert handoff() is None
    assert events == ["waiter_present"]
    assert handoff.handoffs == 0


def test_the_models_and_the_reservation_go_before_the_lock_and_admission_comes_after() -> None:
    """The order is the safety property: the lock is released only once
    the enrichment models are unloaded and their reservation dropped, so
    the waiting command never loads its models beside them; and memory
    admission (with the pause switch) runs again before any enrichment
    model can load."""
    events: list[str] = []
    handoff = _handoff(events, waiting=True)
    assert handoff() is None
    assert events == [
        "waiter_present",
        "unload models",
        "release reservation",
        "release lock",
        "acquire lock (wait=True)",
        "readmit",
    ]
    assert handoff.handoffs == 1


def test_a_refusal_after_taking_the_lock_back_is_returned() -> None:
    events: list[str] = []
    assert _handoff(events, waiting=True, readmit=PAUSED)() == PAUSED

