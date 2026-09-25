"""Unit tests for the pieces behind the public surface's refusal handling
(QA review 2026-09-24): the per-client failure throttle
(`imsg.mcp.ratelimit.ClientFailureThrottle`), how a request's client is
identified (`imsg.mcp.tools.public_server.client_key`), and the in-memory
count of refusals and its writer (`imsg.mcp.audit.RejectionTally`,
`RejectionTallyWriter`). The end-to-end behaviour is in
`tests/test_mcp_public_hardening.py`.

Addresses are from the documentation ranges (RFC 5737, RFC 3849).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from imsg.mcp.audit import (
    UNAUTHENTICATED_SOURCE,
    AggregateRecord,
    MemoryAuditSink,
    RejectionTally,
    RejectionTallyWriter,
)
from imsg.mcp.errors import AuditWriteError
from imsg.mcp.ratelimit import ClientFailureThrottle
from imsg.mcp.tools.public_server import client_key


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# ClientFailureThrottle
# ---------------------------------------------------------------------------


def test_a_client_is_throttled_at_its_limit_and_not_before() -> None:
    throttle = ClientFailureThrottle(per_client_limit=3, shared_limit=100, clock=FakeClock())
    for _ in range(2):
        throttle.note_failure("203.0.113.7")
    assert throttle.is_throttled("203.0.113.7") is False
    throttle.note_failure("203.0.113.7")
    assert throttle.is_throttled("203.0.113.7") is True
    assert throttle.is_throttled("203.0.113.8") is False


def test_a_client_that_stops_is_let_back_in_a_window_after_its_last_failures() -> None:
    clock = FakeClock()
    throttle = ClientFailureThrottle(per_client_limit=2, shared_limit=100, clock=clock)
    throttle.note_failure("203.0.113.7")
    clock.advance(30)
    throttle.note_failure("203.0.113.7")
    assert throttle.is_throttled("203.0.113.7")
    clock.advance(31)  # the first failure has left the window
    assert throttle.is_throttled("203.0.113.7") is False


def test_a_client_whose_failures_keep_coming_stays_throttled() -> None:
    """While new failures keep arriving inside the window, the client stays
    throttled; only the most recent `limit` are kept to decide it."""
    clock = FakeClock()
    throttle = ClientFailureThrottle(per_client_limit=2, shared_limit=100, clock=clock)
    for _ in range(12):
        throttle.note_failure("203.0.113.7")
        clock.advance(10)
    assert throttle.is_throttled("203.0.113.7")


def test_unidentified_requests_share_one_bucket_with_its_own_limit() -> None:
    throttle = ClientFailureThrottle(per_client_limit=1, shared_limit=3, clock=FakeClock())
    throttle.note_failure(None)
    throttle.note_failure(None)
    assert throttle.is_throttled(None) is False
    throttle.note_failure(None)
    assert throttle.is_throttled(None) is True


def test_the_throttle_keeps_a_bounded_number_of_clients() -> None:
    """The keys are addresses an attacker chooses, so memory has to be
    bounded: the least recently rejected client is dropped first."""
    throttle = ClientFailureThrottle(
        per_client_limit=5, shared_limit=5, max_clients=100, clock=FakeClock()
    )
    for i in range(1000):
        throttle.note_failure(f"client-{i}")
    assert throttle.tracked_clients() == 100
    assert throttle.is_throttled("client-999") is False  # one failure, kept
    # Each client's own record is bounded too: a million failures keep
    # only the most recent `limit` timestamps.
    for _ in range(10_000):
        throttle.note_failure("client-999")
    assert throttle.is_throttled("client-999")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"per_client_limit": 0, "shared_limit": 1}, "limits"),
        ({"per_client_limit": 1, "shared_limit": 0}, "limits"),
        ({"per_client_limit": 1, "shared_limit": 1, "window_seconds": 0}, "window"),
        ({"per_client_limit": 1, "shared_limit": 1, "max_clients": 0}, "max_clients"),
    ],
)
def test_the_throttle_refuses_nonsense_settings(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClientFailureThrottle(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# client_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (("203.0.113.7", 443), "203.0.113.7"),
        # uvicorn resolved nothing: the peer is still the local proxy.
        (("127.0.0.1", 50000), None),
        (("::1", 50000), None),
        # One IPv6 host is normally given a /64; rotating inside it buys nothing.
        (("2001:db8:1:2:3:4:5:6", 0), "2001:db8:1:2::/64"),
        (("2001:db8:1:2:ffff:ffff:ffff:ffff", 0), "2001:db8:1:2::/64"),
        # An IPv4 client written the IPv6 way is still that IPv4 client.
        (("::ffff:203.0.113.7", 0), "203.0.113.7"),
        # Not an address at all (a test harness, a Unix socket).
        (("testclient", 0), None),
        (("", 0), None),
    ],
)
def test_client_key(client: tuple[str, int], expected: str | None) -> None:
    assert client_key({"type": "http", "client": client}) == expected


def test_client_key_without_a_client() -> None:
    assert client_key({"type": "http"}) is None
    assert client_key({"type": "http", "client": None}) is None


# ---------------------------------------------------------------------------
# RejectionTally and RejectionTallyWriter
# ---------------------------------------------------------------------------


class StepClock:
    """A wall clock that moves one second per reading."""

    def __init__(self) -> None:
        self.now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def test_the_tally_counts_by_code_and_collapses_unknown_codes() -> None:
    tally = RejectionTally()
    for _ in range(3):
        tally.add("UNAUTHORIZED")
    tally.add("RATE_LIMITED")
    tally.add("free text that must never be stored")
    assert tally.pending() == {"UNAUTHORIZED": 3, "RATE_LIMITED": 1, "INTERNAL": 1}


def test_the_tally_is_safe_from_many_threads() -> None:
    tally = RejectionTally()

    def add_many() -> None:
        for _ in range(5000):
            tally.add("UNAUTHORIZED")

    threads = [threading.Thread(target=add_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert tally.pending() == {"UNAUTHORIZED": 40_000}


def test_the_writer_writes_one_row_per_code_for_the_period() -> None:
    clock = StepClock()
    tally = RejectionTally(now=clock)
    sink = MemoryAuditSink()
    writer = RejectionTallyWriter(tally, sink, interval_seconds=60)
    for _ in range(31):
        tally.add("UNAUTHORIZED")
    for _ in range(7):
        tally.add("RATE_LIMITED")

    assert writer.write_now() == 2

    rows = sink.aggregate_snapshot()
    assert [(r.error, r.request_count) for r in rows] == [("RATE_LIMITED", 7), ("UNAUTHORIZED", 31)]
    assert all(r.source == UNAUTHENTICATED_SOURCE for r in rows)
    assert all(r.surface == "public" and r.subject_ok is False and r.tool is None for r in rows)
    assert rows[0].period_start < rows[0].period_end
    assert sink.snapshot() == ()  # nothing in the per-request table
    # Nothing counted since: nothing written.
    assert writer.write_now() == 0
    assert len(sink.aggregate_snapshot()) == 2


class FlakySink:
    """Fails its first `failures` writes."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.written: list[AggregateRecord] = []

    def record_aggregates(self, records: Sequence[AggregateRecord]) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise AuditWriteError("store down")
        self.written.extend(records)


def test_a_failed_write_keeps_the_counts_and_the_period_start_for_the_next() -> None:
    clock = StepClock()
    tally = RejectionTally(now=clock)
    sink = FlakySink(failures=1)
    writer = RejectionTallyWriter(tally, sink, interval_seconds=60)
    for _ in range(5):
        tally.add("UNAUTHORIZED")
    first_start = clock.now

    assert writer.write_now() == 0  # failed; kept
    tally.add("UNAUTHORIZED")
    assert writer.write_now() == 1

    (row,) = sink.written
    assert row.request_count == 6
    assert row.period_start <= first_start


def test_the_writer_thread_writes_on_its_interval_and_once_more_on_stop() -> None:
    tally = RejectionTally()
    sink = MemoryAuditSink()
    writer = RejectionTallyWriter(tally, sink, interval_seconds=0.05)
    writer.start()
    try:
        tally.add("UNAUTHORIZED")
        deadline = time.monotonic() + 5
        while not sink.aggregate_snapshot():
            assert time.monotonic() < deadline, "the writer never wrote"
            time.sleep(0.01)
        tally.add("RATE_LIMITED")
    finally:
        writer.stop()
    codes = [r.error for r in sink.aggregate_snapshot()]
    assert codes.count("UNAUTHORIZED") == 1
    assert codes.count("RATE_LIMITED") == 1
    assert tally.pending() == {}


def test_the_writer_refuses_a_non_positive_interval() -> None:
    with pytest.raises(ValueError):
        RejectionTallyWriter(RejectionTally(), MemoryAuditSink(), interval_seconds=0)
