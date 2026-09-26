"""`imsg.search_yield` — segmentation and embedding wait while a search is
in flight, as the enrichment worker does, against a scripted connection.

What is pinned here is the policy: the order of the stop check and the
wait, what a wait costs when nobody is searching, the bound, that a pause
or a waiting live server ends a wait at once, and that every boundary-model
call waits. `tests/test_search_yield_integration.py` runs the same against
a real Postgres through `imsg sync`.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog

from imsg.background_gate import DeferralKind, StopReason
from imsg.config.loader import load_config_dict
from imsg.db.enrichment_yield_locks import (
    EMBED_YIELDING_LOCK_KEY,
    ENRICHMENT_PAUSED_LOCK_KEY,
    QUERY_IN_FLIGHT_LOCK_KEY,
    SEGMENT_YIELDING_LOCK_KEY,
    EnrichmentYieldGate,
    QueryYieldGate,
    YieldReport,
)
from imsg.search_yield import SearchYield, YieldingBoundaryProvider, YieldStep, YieldTally
from imsg.segment.boundaries import FakeBoundaryProvider
from test_enrichment_yield_locks import FakeClock, FakeConn, _sqls

PAUSED = StopReason(DeferralKind.PAUSED, "heavy background work is paused: test")
PRESSURE = StopReason(DeferralKind.MEMORY, "the kernel reports critical memory pressure")
PROBE = ["SELECT pg_try_advisory_lock(%s)", "SELECT pg_advisory_unlock(%s)"]


def _gate(conn: FakeConn, clock: FakeClock, step: str = "embed", **kw: Any) -> QueryYieldGate:
    options: dict[str, Any] = {
        "step": step,
        "poll_interval_seconds": 0.25,
        "max_pause_seconds": 10.0,
        "monotonic": clock.monotonic,
        "sleep": clock.sleep,
    }
    options.update(kw)
    return QueryYieldGate(conn, **options)  # type: ignore[arg-type]


class Checks:
    """A stop check that answers from a script, then keeps the last answer."""

    def __init__(self, *answers: StopReason | None) -> None:
        self.answers = list(answers) or [None]
        self.asked = 0

    def __call__(self) -> StopReason | None:
        self.asked += 1
        return self.answers[min(self.asked, len(self.answers)) - 1]


# --------------------------------------------------------------------------
# nothing in flight: the overnight case costs one round trip
# --------------------------------------------------------------------------


def test_nothing_in_flight_costs_one_probe_no_sleep_and_one_stop_check() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [True]
    search_yield = SearchYield(_gate(conn, clock))
    check = Checks(None)

    assert search_yield.stop_check(check)() is None

    assert check.asked == 1
    assert _sqls(conn) == PROBE
    assert clock.slept == []
    assert search_yield.tally == YieldTally()


def test_a_disabled_yield_asks_the_database_nothing() -> None:
    """`enrichment.yield_to_queries: false` covers segmentation and
    embedding too: no probe at all, so the step runs exactly as before."""
    conn, clock = FakeConn(), FakeClock()
    search_yield = SearchYield(_gate(conn, clock, enabled=False))
    assert search_yield.stop_check(Checks(None))() is None
    assert search_yield.wait().paused is False
    assert conn.statements == []


def test_a_stop_reason_is_returned_before_any_wait() -> None:
    """Paused, short of memory or a live server waiting: the step stops
    after the unit it finished, without first waiting for a search."""
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False] * 10
    search_yield = SearchYield(_gate(conn, clock))

    assert search_yield.stop_check(Checks(PRESSURE))() == PRESSURE
    assert conn.statements == []
    assert clock.slept == []


# --------------------------------------------------------------------------
# a search in flight
# --------------------------------------------------------------------------


def test_the_next_batch_waits_while_a_search_is_in_flight_then_goes_on() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False, False, True]  # busy, busy, then clear
    search_yield = SearchYield(_gate(conn, clock))
    check = Checks(None)

    assert search_yield.stop_check(check)() is None

    assert clock.slept == [0.25, 0.25]
    assert check.asked == 2  # once before the wait, once after it
    assert search_yield.tally == YieldTally(waits=1, seconds=pytest.approx(0.5))


def test_a_stop_that_arrives_during_the_wait_is_honoured_after_it() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False, True]
    search_yield = SearchYield(_gate(conn, clock))
    assert search_yield.stop_check(Checks(None, PAUSED))() == PAUSED


def test_a_step_publishes_its_own_yielding_key_while_it_waits() -> None:
    """What `imsg status` reports as `segment_yielding_now` and
    `embed_yielding_now`; the enrichment worker's key is untouched."""
    for step, key in (("segment", SEGMENT_YIELDING_LOCK_KEY), ("embed", EMBED_YIELDING_LOCK_KEY)):
        conn, clock = FakeConn(), FakeClock()
        conn.probe_results = [False, True]
        _gate(conn, clock, step=step).wait_until_clear()
        sqls = _sqls(conn)
        take = sqls.index("SELECT pg_try_advisory_lock_shared(%s)")
        drop = sqls.index("SELECT pg_advisory_unlock_shared(%s)")
        assert take < drop
        assert conn.statements[take][1] == (key,)
        assert conn.statements[drop][1] == (key,)
        assert conn.statements[0][1] == (QUERY_IN_FLIGHT_LOCK_KEY,)
    assert len({ENRICHMENT_PAUSED_LOCK_KEY, SEGMENT_YIELDING_LOCK_KEY, EMBED_YIELDING_LOCK_KEY}) == 3


def test_each_step_logs_its_waits_under_its_own_name() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False, True]
    with structlog.testing.capture_logs() as logs:
        _gate(conn, clock, step="segment").wait_until_clear()
    assert [entry["event"] for entry in logs] == ["segment.yielding", "segment.resumed"]
    assert logs[1]["waited_seconds"] == 0.25


def test_the_enrichment_gate_keeps_its_name_key_and_events() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False, True]
    gate = EnrichmentYieldGate(conn, monotonic=clock.monotonic, sleep=clock.sleep)  # type: ignore[arg-type]
    with structlog.testing.capture_logs() as logs:
        gate.wait_until_clear()
    assert [entry["event"] for entry in logs] == ["enrich.yielding", "enrich.resumed"]
    assert (ENRICHMENT_PAUSED_LOCK_KEY,) in [params for _sql, params in conn.statements]


def test_an_unknown_step_is_refused() -> None:
    with pytest.raises(ValueError, match="step"):
        QueryYieldGate(FakeConn(), step="caption")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# bounded: background work cannot starve
# --------------------------------------------------------------------------


def test_the_wait_is_bounded_and_the_unit_then_runs() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False] * 100
    search_yield = SearchYield(_gate(conn, clock, max_pause_seconds=1.0))
    check = Checks(None)

    assert search_yield.stop_check(check)() is None  # the batch runs anyway

    assert sum(clock.slept) == pytest.approx(1.0)
    assert check.asked == 2
    assert search_yield.tally.gave_up == 1
    assert search_yield.tally.describe(1.0) == (
        "yielded to in-flight searches 1 time(s), 1.0s total; went ahead after the 1 s "
        "limit 1 time(s) (enrichment.yield_max_pause_seconds)"
    )


def test_every_wait_is_bounded_separately() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False] * 100
    search_yield = SearchYield(_gate(conn, clock, max_pause_seconds=0.5))
    for _ in range(3):
        search_yield.wait()
    assert search_yield.tally == YieldTally(waits=3, seconds=pytest.approx(1.5), gave_up=3)


# --------------------------------------------------------------------------
# the pause switch and a waiting live server still win
# --------------------------------------------------------------------------


def test_a_pause_set_during_the_wait_ends_it_at_once() -> None:
    """Without this a paused host would wait behind a stream of searches
    for up to the bound, where it used to stop after one unit."""
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False] * 100
    asked: list[int] = []

    def paused_after_two_polls() -> bool:
        asked.append(1)
        return len(asked) >= 3

    search_yield = SearchYield(_gate(conn, clock), interrupt=paused_after_two_polls)
    check = Checks(None, PAUSED)

    assert search_yield.stop_check(check)() == PAUSED

    assert clock.slept == [0.25, 0.25]
    assert search_yield.tally == YieldTally(waits=1, seconds=pytest.approx(0.5))
    assert _sqls(conn)[-1] == "SELECT pg_advisory_unlock_shared(%s)"  # no longer "yielding"


def test_the_interrupt_is_not_asked_when_nothing_is_in_flight() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [True]

    def must_not_be_asked() -> bool:
        raise AssertionError("asked while nothing was in flight")

    assert SearchYield(_gate(conn, clock), interrupt=must_not_be_asked).wait().paused is False


# --------------------------------------------------------------------------
# segmentation: every boundary-model call waits
# --------------------------------------------------------------------------


class ScriptedGate:
    def __init__(self, *reports: YieldReport) -> None:
        self.reports = list(reports)
        self.calls = 0

    def wait_until_clear(self, *, interrupt: Any = None) -> YieldReport:
        self.calls += 1
        return self.reports.pop(0) if self.reports else YieldReport(paused=False, waited_seconds=0.0)


class RecordingBoundaryProvider(FakeBoundaryProvider):
    def __init__(self, gate: ScriptedGate) -> None:
        super().__init__(messages_per_segment=2)
        self.gate = gate
        self.gate_calls_at_each_call: list[int] = []
        self.unloaded = False

    def detect_boundaries(self, window: Any) -> list[int]:
        self.gate_calls_at_each_call.append(self.gate.calls)
        return super().detect_boundaries(window)

    def unload(self) -> None:
        self.unloaded = True


def test_the_boundary_model_waits_before_every_call() -> None:
    gate = ScriptedGate(YieldReport(paused=True, waited_seconds=2.0))
    inner = RecordingBoundaryProvider(gate)
    search_yield = SearchYield(gate)
    provider = YieldingBoundaryProvider(inner, search_yield)

    window = [object()] * 6
    provider.detect_boundaries(window)  # type: ignore[arg-type]
    provider.detect_boundaries(window)  # type: ignore[arg-type]

    assert inner.gate_calls_at_each_call == [1, 2]  # each call came after its own wait
    assert search_yield.tally == YieldTally(waits=1, seconds=2.0)
    assert provider.model_id == inner.model_id
    provider.unload()
    assert inner.unloaded is True


def test_a_boundary_provider_with_no_unload_is_left_alone() -> None:
    provider = YieldingBoundaryProvider(FakeBoundaryProvider(), SearchYield(ScriptedGate()))
    provider.unload()


# --------------------------------------------------------------------------
# settings: the enrichment worker's
# --------------------------------------------------------------------------


def test_the_settings_are_the_enrichment_workers(config_dict_factory: Any) -> None:
    raw = config_dict_factory()
    raw["enrichment"] = {
        **raw.get("enrichment", {}),
        "yield_to_queries": True,
        "yield_poll_interval_seconds": 0.5,
        "yield_max_pause_seconds": 2.0,
    }
    cfg = load_config_dict(raw)
    conn = FakeConn()
    conn.probe_results = [False] * 100
    slept: list[float] = []
    search_yield = SearchYield.from_config(cfg, conn, YieldStep.SEGMENT)  # type: ignore[arg-type]
    gate = search_yield._gate
    assert isinstance(gate, QueryYieldGate)
    assert gate.step == "segment"
    assert gate.enabled is True
    assert gate.max_pause_seconds == 2.0
    gate._sleep = slept.append  # type: ignore[attr-defined]
    gate._monotonic = lambda: sum(slept)  # type: ignore[attr-defined]
    assert search_yield.wait().gave_up is True
    assert slept == [0.5, 0.5, 0.5, 0.5]

    raw["enrichment"]["yield_to_queries"] = False
    off = SearchYield.from_config(load_config_dict(raw), FakeConn(), YieldStep.EMBED)  # type: ignore[arg-type]
    assert off._gate.enabled is False  # type: ignore[attr-defined]


def test_the_step_labels() -> None:
    assert YieldStep.SEGMENT.label == "segmentation"
    assert YieldStep.EMBED.label == "embedding"
    assert YieldTally().describe(300.0) is None
