"""`imsg.db.enrichment_yield_locks` — enrichment stands aside for an
in-flight query, never the reverse (D10.3's ratified remedy).

Two layers, deliberately:

- unit tests against a scripted fake connection, which pin the *policy*
  (what is probed, when, how long it waits, what it costs when idle);
- integration tests against a real scratch Postgres, which pin the
  *mechanism* — above all that a killed MCP server cannot wedge
  enrichment paused, which is a property of session-level advisory locks
  and cannot be asserted against a fake at all.

The integration layer skips cleanly with no scratch instance, the same
pattern as `tests/test_migrations_integration.py`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from imsg.db.enrichment_yield_locks import (
    ENRICHMENT_PAUSED_LOCK_KEY,
    QUERY_IN_FLIGHT_LOCK_KEY,
    EnrichmentYieldGate,
    QueryInFlightMarker,
    read_yield_state,
)

# --------------------------------------------------------------------------
# unit: a scripted connection
# --------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._conn.statements.append((sql.strip(), params))
        if self._conn.raise_on_execute is not None:
            raise self._conn.raise_on_execute

    def fetchone(self) -> tuple[Any, ...] | None:
        sql, _params = self._conn.statements[-1]
        if "pg_try_advisory_lock_shared" in sql:
            return (self._conn.shared_lock_succeeds,)
        if "pg_try_advisory_lock" in sql:
            # The exclusive probe: it succeeds exactly when nobody holds
            # the shared lock, which is what "no query in flight" means.
            return (self._conn.probe_results.pop(0) if self._conn.probe_results else True,)
        return (None,)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._conn.locks_rows


class FakeConn:
    def __init__(self) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.probe_results: list[bool] = []
        self.shared_lock_succeeds = True
        self.locks_rows: list[tuple[Any, ...]] = []
        self.raise_on_execute: Exception | None = None
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def _sqls(conn: FakeConn) -> list[str]:
    return [sql for sql, _ in conn.statements]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _gate(conn: FakeConn, clock: FakeClock, **kw: Any) -> EnrichmentYieldGate:
    options: dict[str, Any] = {
        "poll_interval_seconds": 0.25,
        "max_pause_seconds": 10.0,
        "monotonic": clock.monotonic,
        "sleep": clock.sleep,
    }
    options.update(kw)
    return EnrichmentYieldGate(conn, **options)


# --- the no-query case, called out explicitly ------------------------------


def test_no_query_running_costs_one_probe_and_no_sleep() -> None:
    """"Pausing costs nothing when no query is running" — the overnight
    common case, since nobody is searching at 03:00. One round trip to a
    local Postgres, no sleep, no paused marker published."""
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [True]

    report = _gate(conn, clock).wait_until_clear()

    assert report.paused is False
    assert report.waited_seconds == 0.0
    assert report.gave_up is False
    assert clock.slept == []
    assert _sqls(conn) == [
        "SELECT pg_try_advisory_lock(%s)",
        "SELECT pg_advisory_unlock(%s)",
    ]


def test_the_idle_probe_releases_the_exclusive_lock_immediately() -> None:
    """The worker must not hold the exclusive lock while it works: an MCP
    server's own acquisition is a try, so holding it would silently cost
    the server its marker for the whole batch."""
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [True]
    _gate(conn, clock).wait_until_clear()
    assert _sqls(conn)[-1] == "SELECT pg_advisory_unlock(%s)"
    assert conn.statements[0][1] == (QUERY_IN_FLIGHT_LOCK_KEY,)


def test_disabled_gate_asks_the_database_nothing() -> None:
    conn, clock = FakeConn(), FakeClock()
    report = _gate(conn, clock, enabled=False).wait_until_clear()
    assert report.paused is False
    assert conn.statements == []


# --- the query-in-flight case ---------------------------------------------


def test_a_query_in_flight_pauses_until_it_finishes() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False, False, False, True]  # busy, busy, busy, then clear

    report = _gate(conn, clock).wait_until_clear()

    assert report.paused is True
    assert report.gave_up is False
    assert report.waited_seconds == pytest.approx(0.75)
    assert clock.slept == [0.25, 0.25, 0.25]


def test_a_pausing_worker_publishes_that_it_is_yielding() -> None:
    """What `imsg status` reports as `enrichment_yielding_now`."""
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False, True]

    _gate(conn, clock).wait_until_clear()

    sqls = _sqls(conn)
    take = sqls.index("SELECT pg_try_advisory_lock_shared(%s)")
    drop = sqls.index("SELECT pg_advisory_unlock_shared(%s)")
    assert take < drop
    assert conn.statements[take][1] == (ENRICHMENT_PAUSED_LOCK_KEY,)
    assert conn.statements[drop][1] == (ENRICHMENT_PAUSED_LOCK_KEY,)


def test_the_paused_marker_is_dropped_even_when_the_wait_times_out() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False] * 200

    report = _gate(conn, clock, max_pause_seconds=1.0).wait_until_clear()

    assert report.gave_up is True
    assert report.paused is True
    assert _sqls(conn)[-1] == "SELECT pg_advisory_unlock_shared(%s)"


def test_the_wait_is_bounded_so_the_queue_still_drains() -> None:
    """Not the crash backstop — that is the session lock dying with the
    server — but the "someone is searching continuously" one."""
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False] * 500

    report = _gate(conn, clock, max_pause_seconds=2.0, poll_interval_seconds=0.5).wait_until_clear()

    assert report.gave_up is True
    assert report.waited_seconds == pytest.approx(2.0)
    assert sum(clock.slept) == pytest.approx(2.0)


def test_a_zero_bound_means_check_once_and_proceed() -> None:
    conn, clock = FakeConn(), FakeClock()
    conn.probe_results = [False]
    report = _gate(conn, clock, max_pause_seconds=0.0).wait_until_clear()
    assert report.gave_up is True
    assert clock.slept == []


@pytest.mark.parametrize(("field", "value"), [("poll_interval_seconds", 0.0), ("max_pause_seconds", -1.0)])
def test_nonsense_timings_are_rejected_at_construction(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        EnrichmentYieldGate(FakeConn(), **{field: value})  # type: ignore[arg-type]


# --- the query side's marker ----------------------------------------------


def test_the_marker_is_reentrant_and_takes_the_lock_once() -> None:
    conn = FakeConn()
    marker = QueryInFlightMarker(lambda: conn)  # type: ignore[arg-type]
    with marker:
        assert marker.is_marking is True
        with marker:
            assert marker.is_marking is True
        assert marker.is_marking is True  # the inner exit must not release
    assert marker.is_marking is False
    assert _sqls(conn) == [
        "SELECT pg_try_advisory_lock_shared(%s)",
        "SELECT pg_advisory_unlock_shared(%s)",
    ]


def test_a_marker_that_loses_the_race_does_not_delay_the_query() -> None:
    """The worker's probe holds the exclusive lock for microseconds. If a
    query lands in that window the marker simply does not mark — making
    the query WAIT would be the inversion this module exists to
    prevent."""
    conn = FakeConn()
    conn.shared_lock_succeeds = False
    marker = QueryInFlightMarker(lambda: conn)  # type: ignore[arg-type]
    with marker:
        assert marker.is_marking is False
    assert "pg_advisory_unlock_shared" not in " ".join(_sqls(conn))


def test_a_marker_whose_database_is_unreachable_degrades_to_a_no_op() -> None:
    """A query must never fail because the thing that makes *enrichment*
    considerate could not talk to Postgres."""

    def _explode() -> Any:
        raise psycopg.OperationalError("connection refused")

    marker = QueryInFlightMarker(_explode)
    with marker:
        assert marker.is_marking is False
    with marker:  # and it does not keep retrying on every query
        assert marker.is_marking is False


def test_the_marker_survives_concurrent_requests_on_several_threads() -> None:
    """A public server answers on a thread pool. An unguarded counter
    could miss a decrement and leave the lock held for the life of the
    process — enrichment wedged paused, which is the one outcome this
    design exists to rule out."""
    import threading

    conn = FakeConn()
    marker = QueryInFlightMarker(lambda: conn)  # type: ignore[arg-type]
    barrier = threading.Barrier(8)

    def one_request() -> None:
        barrier.wait()
        for _ in range(50):
            with marker:
                pass

    threads = [threading.Thread(target=one_request) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert marker.is_marking is False
    takes = _sqls(conn).count("SELECT pg_try_advisory_lock_shared(%s)")
    drops = _sqls(conn).count("SELECT pg_advisory_unlock_shared(%s)")
    assert takes == drops


def test_a_disabled_marker_opens_no_connection() -> None:
    opened = []

    def _connect() -> Any:
        opened.append(1)
        return FakeConn()

    marker = QueryInFlightMarker(_connect, enabled=False)
    with marker:
        pass
    assert opened == []
    assert marker.enabled is False


# --------------------------------------------------------------------------
# integration: a real Postgres, where the crash case is actually testable
# --------------------------------------------------------------------------

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_yield_locks_test"


def _dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


ADMIN_DSN = _dsn("postgres")


def _admin_reachable() -> bool:
    try:
        conn = psycopg.connect(ADMIN_DSN, connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


REACHABLE = _admin_reachable()

integration = pytest.mark.skipif(
    not REACHABLE,
    reason=(
        f"no reachable scratch Postgres instance (tried {TEST_PG_HOST}:{TEST_PG_PORT}) — "
        "set IMSG_TEST_PG_HOST/IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


@pytest.fixture
def live_db() -> Iterator[str]:
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        admin.close()
    yield _dsn(TEST_DB_NAME)
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
    finally:
        admin.close()


@pytest.fixture
def worker_conn(live_db: str) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(live_db, autocommit=True)
    yield conn
    conn.close()


def _no_wait_gate(conn: psycopg.Connection, **kw: Any) -> EnrichmentYieldGate:
    options: dict[str, Any] = {"poll_interval_seconds": 0.01, "max_pause_seconds": 2.0}
    options.update(kw)
    return EnrichmentYieldGate(conn, **options)


@integration
def test_live_no_query_running_returns_at_once(worker_conn: psycopg.Connection) -> None:
    report = _no_wait_gate(worker_conn).wait_until_clear()
    assert report.paused is False
    assert report.gave_up is False


@integration
def test_live_a_marked_query_pauses_the_worker(
    live_db: str, worker_conn: psycopg.Connection
) -> None:
    marker = QueryInFlightMarker(lambda: psycopg.connect(live_db, autocommit=True))
    try:
        with marker:
            assert marker.is_marking is True
            report = _no_wait_gate(worker_conn, max_pause_seconds=0.2).wait_until_clear()
            assert report.paused is True
            assert report.gave_up is True  # it never cleared, so the bound fired
        # Released: the very next check is clear.
        assert _no_wait_gate(worker_conn).wait_until_clear().paused is False
    finally:
        marker.close()


@integration
def test_live_a_crashed_mcp_server_cannot_wedge_enrichment_paused(
    live_db: str, worker_conn: psycopg.Connection
) -> None:
    """THE failure mode this design is chosen for. A session-level
    advisory lock is released by the server when the session ends, however
    it ends — so a killed, crashed or jetsammed MCP server releases its
    marker with no timeout to tune and no stale file to reap. A lock file
    or a status row would leave enrichment paused until someone noticed.

    `close()` here without releasing first stands in for the crash: the
    connection simply goes away while the lock is held.
    """
    victim = psycopg.connect(live_db, autocommit=True)
    victim.execute("SELECT pg_advisory_lock_shared(%s)", (QUERY_IN_FLIGHT_LOCK_KEY,))
    assert _no_wait_gate(worker_conn, max_pause_seconds=0.2).wait_until_clear().paused is True

    victim.close()  # the crash: no unlock, the session is just gone

    report = _no_wait_gate(worker_conn).wait_until_clear()
    assert report.paused is False, "the dead server's lock outlived its session"
    assert report.waited_seconds == 0.0


@integration
def test_live_status_sees_both_markers_without_taking_a_lock(
    live_db: str, worker_conn: psycopg.Connection
) -> None:
    """`imsg status` reads `pg_locks`; this asserts the key layout
    (classid = the high 32 bits, objid = the low 32) against a real server
    rather than trusting the documentation."""
    reader = psycopg.connect(live_db, autocommit=True)
    holder = psycopg.connect(live_db, autocommit=True)
    try:
        assert read_yield_state(reader) == read_yield_state(reader)
        idle = read_yield_state(reader)
        assert idle.query_in_flight is False
        assert idle.enrichment_paused is False
        assert idle.reason is None

        holder.execute("SELECT pg_advisory_lock_shared(%s)", (QUERY_IN_FLIGHT_LOCK_KEY,))
        assert read_yield_state(reader).query_in_flight is True

        holder.execute("SELECT pg_advisory_lock_shared(%s)", (ENRICHMENT_PAUSED_LOCK_KEY,))
        busy = read_yield_state(reader)
        assert busy.query_in_flight is True
        assert busy.enrichment_paused is True

        # Reading must not itself take the lock — a worker probing right
        # after a status call must still find the system clear once the
        # holder lets go.
        holder.close()
        cleared = read_yield_state(reader)
        assert cleared.query_in_flight is False
        assert cleared.enrichment_paused is False
    finally:
        reader.close()
        if not holder.closed:
            holder.close()


@integration
def test_live_the_worker_probe_never_blocks_a_second_query(
    live_db: str, worker_conn: psycopg.Connection
) -> None:
    """Several queries can be in flight at once — the marker is a shared
    lock — and the worker's exclusive probe is a `try`, so it never queues
    behind them."""
    first = QueryInFlightMarker(lambda: psycopg.connect(live_db, autocommit=True))
    second = QueryInFlightMarker(lambda: psycopg.connect(live_db, autocommit=True))
    try:
        with first, second:
            assert first.is_marking and second.is_marking
            assert _no_wait_gate(worker_conn, max_pause_seconds=0.1).wait_until_clear().paused
        assert _no_wait_gate(worker_conn).wait_until_clear().paused is False
    finally:
        first.close()
        second.close()
