"""Enrichment yields to in-flight queries, never the reverse (D10.3's
ratified remedy, preference 3).

Why this exists
---------------

The nightly enrichment window overlaps the always-on MCP server, and the
two compete for one 20-core GPU. Measured on the production host
(2026-09-17): with a *single* copy of the 35B resident — so no swap, and
memory pressure never critical — query p95 still trebled while enrichment
ran. That residue is GPU contention, and the ratified answer is to degrade
the invisible job rather than the visible one: enrichment is restartable
batch work with a queue and a lease, the MCP server is the externally
visible surface with a 2.0 s latency budget.

The asymmetry is deliberate and total. A query never waits for
enrichment, not even for the length of a lock acquisition (see
:class:`QueryInFlightMarker`). Enrichment waits, bounded, and only
between units of work.

The signal, and why Postgres advisory locks
-------------------------------------------

The two sides are separate processes (`…mcp-public` / `…mcp-local` and
`…enrich`), so the signal has to cross a process boundary. They already
share one thing that is always up whenever either of them can run: the
dedicated Postgres instance. Its **session-level advisory locks** have
exactly the property this needs — *the server releases them when the
session ends, however it ends*. A killed, crashed, or OOM-jetsammed MCP
server cannot leave enrichment paused, because its lock dies with its
connection, with no timeout to tune and no stale-marker file to reap.
That is the failure mode a lock file or a status row would get wrong.

Two keys, each held *shared* by whoever is publishing a state, and probed
*exclusively* by whoever is asking:

``QUERY_IN_FLIGHT_LOCK_KEY``
    Held by an MCP server for the span of a query (and of its warm-up,
    which is the same GPU work). Any number of concurrent queries can
    hold it at once, which is why it is a shared lock.

``ENRICHMENT_PAUSED_LOCK_KEY``
    Held by an enrichment worker while it is parked waiting. Purely
    observability: it is what lets `imsg status` say enrichment is
    *currently yielding* rather than merely that it would.

`imsg status` reads both from ``pg_locks`` and takes no lock itself, so
looking at the system cannot perturb it.

What this does not do
---------------------

It does not preempt. MLX generation cannot be interrupted mid-kernel, and
a half-captioned attachment is not a thing the queue can represent, so the
worker checks the gate *between* tasks and a query that arrives one
millisecond after a task starts waits out that task. The relief is
therefore coarse — it stops the *next* several units of work, not the one
already running — which is what a batch job sharing a GPU can honestly
offer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Self

import structlog

from imsg import constants

if TYPE_CHECKING:
    from collections.abc import Callable

    import psycopg

logger = structlog.get_logger(__name__)

_LOCK_NAMESPACE = 0x696D7367
"""``'imsg'`` as ASCII, the high 32 bits of every key below — so these
locks cannot collide with another application's advisory keys in a
database that somehow ends up shared, and so a key seen in ``pg_locks``
is identifiable at a glance."""

QUERY_IN_FLIGHT_LOCK_KEY = (_LOCK_NAMESPACE << 32) | 1
ENRICHMENT_PAUSED_LOCK_KEY = (_LOCK_NAMESPACE << 32) | 2

_PG_LOCKS_CLASSID = _LOCK_NAMESPACE
"""How Postgres splits a single-``bigint`` advisory key across
``pg_locks``: ``classid`` is the high 32 bits, ``objid`` the low 32, and
``objsubid`` is 1 (2 for the two-integer form). Documented in "Advisory
Locks" / ``pg_locks``; asserted against a live server by the integration
tests rather than trusted."""

DEFAULT_POLL_INTERVAL_SECONDS = constants.ENRICHMENT_YIELD_POLL_INTERVAL_SECONDS
DEFAULT_MAX_PAUSE_SECONDS = constants.ENRICHMENT_YIELD_MAX_PAUSE_SECONDS
"""Defaults for the two knobs, defined in `imsg.constants` so
`imsg.config.schema` can default to them without importing `imsg.db`
(which imports the schema). See there for what each is for."""


# --------------------------------------------------------------------------
# the query side: publish "a query is in flight"
# --------------------------------------------------------------------------


class QueryInFlightMarker:
    """Publishes, for as long as it is entered, that this process is
    running a query.

    Opens and keeps **its own** connection (from ``connect``), because the
    retrieval service's connection is busy answering the very query this
    marks, and because a session-level lock has to live on a session that
    lasts as long as the marker does.

    Re-entrant by count, and safe to enter from several threads: a public
    server answers requests on a thread pool while model calls run on one
    dedicated thread (`imsg.retrieval.model_thread`), so the counter is
    guarded. An unguarded counter could miss a decrement and leave the
    lock held for the life of the process — precisely the "enrichment
    wedged paused" failure this design exists to rule out.

    **Never blocks.** The lock is taken with ``pg_try_advisory_lock_shared``
    and a failure is simply logged and ignored — the only thing that can
    hold the conflicting exclusive lock is an enrichment worker's probe,
    which holds it for microseconds, and making a query wait on enrichment
    is the exact inversion this module exists to prevent. The cost of
    losing that race is that one query goes unmarked and the worker starts
    one more unit of work.

    A database that cannot be reached is likewise not a reason to fail a
    query: the marker degrades to a no-op and says so once.
    """

    def __init__(
        self,
        connect: Callable[[], psycopg.Connection],
        *,
        enabled: bool = True,
        lock_key: int = QUERY_IN_FLIGHT_LOCK_KEY,
    ) -> None:
        self._connect = connect
        self._enabled = enabled
        self._lock_key = lock_key
        self._conn: psycopg.Connection | None = None
        self._depth = 0
        self._held = False
        self._degraded = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_marking(self) -> bool:
        """Whether the lock is held right now — what a test asserts on."""
        return self._held

    def __enter__(self) -> Self:
        with self._lock:
            self._depth += 1
            if self._depth == 1:
                self._acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        with self._lock:
            self._depth -= 1
            if self._depth <= 0:
                self._depth = 0
                self._release()

    def close(self) -> None:
        """Release and drop the connection. The lock would die with the
        session anyway; this just makes shutdown tidy."""
        with self._lock:
            self._depth = 0
            self._release()
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # a connection that is already gone is fine
                logger.debug("query_marker.close_failed")

    def _connection(self) -> psycopg.Connection | None:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _acquire(self) -> None:
        if not self._enabled or self._degraded:
            return
        try:
            conn = self._connection()
            assert conn is not None
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock_shared(%s)", (self._lock_key,))
                row = cur.fetchone()
            self._held = bool(row and row[0])
        except Exception as exc:
            # Marking is an optimisation for someone else's throughput; a
            # query must never fail or slow down because of it.
            self._degraded = True
            self._held = False
            self._conn = None
            logger.warning(
                "query_marker.disabled",
                error=f"{type(exc).__name__}: {exc}",
                note="enrichment will not see this process's queries and so will not yield",
            )

    def _release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            conn = self._conn
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock_shared(%s)", (self._lock_key,))
        except Exception as exc:
            # The lock dies with the session, so a failure here leaks
            # nothing beyond this process's lifetime.
            logger.warning("query_marker.unlock_failed", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# the enrichment side: wait for the query side to be idle
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class YieldReport:
    """What one gate check did — logged per unit of work, and summed into
    `imsg enrich`'s closing line."""

    paused: bool
    waited_seconds: float
    gave_up: bool = False
    """True when `max_pause_seconds` elapsed with a query still in flight
    and the worker proceeded anyway."""


class EnrichmentYieldGate:
    """Blocks an enrichment worker between units of work while a query is
    in flight.

    Takes the worker's own connection: a session-level lock has to live on
    a session that lasts as long as the worker, this is that session, and
    a worker that dies takes its "I am paused" marker with it.

    Disabled (`enabled=False`), every check returns immediately — the
    escape hatch for an operator who would rather have throughput.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        *,
        enabled: bool = True,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        max_pause_seconds: float = DEFAULT_MAX_PAUSE_SECONDS,
        query_lock_key: int = QUERY_IN_FLIGHT_LOCK_KEY,
        paused_lock_key: int = ENRICHMENT_PAUSED_LOCK_KEY,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError(f"poll_interval_seconds must be > 0, got {poll_interval_seconds}")
        if max_pause_seconds < 0:
            raise ValueError(f"max_pause_seconds must be >= 0, got {max_pause_seconds}")
        self._conn = conn
        self._enabled = enabled
        self._poll_interval = poll_interval_seconds
        self._max_pause = max_pause_seconds
        self._query_lock_key = query_lock_key
        self._paused_lock_key = paused_lock_key
        self._monotonic = monotonic
        self._sleep = sleep

    @property
    def enabled(self) -> bool:
        return self._enabled

    def wait_until_clear(self) -> YieldReport:
        """Return once no query is in flight, or once `max_pause_seconds`
        has elapsed.

        The fast path — the overwhelmingly common one, because nobody is
        searching at 03:00 — is a single round trip to a local Postgres
        over a unix socket and no sleep at all. That is what "pausing
        costs nothing when no query is running" means concretely.

        A database error here propagates rather than being swallowed, and
        that is deliberate: this runs on the worker's own connection, the
        same one the queue claims and completes tasks on, so a connection
        that cannot answer this cannot run the worker either. Degrading to
        "assume idle" would only move the failure one line later, with a
        misleading message. (The *query* side's marker degrades instead —
        see :class:`QueryInFlightMarker` — because there a database
        problem would otherwise cost a user their search.)
        """
        if not self._enabled:
            return YieldReport(paused=False, waited_seconds=0.0)
        if self._query_side_is_idle():
            return YieldReport(paused=False, waited_seconds=0.0)

        started = self._monotonic()
        logger.info("enrich.yielding", note="a query is in flight; pausing between tasks")
        with self._published_as_paused():
            while True:
                waited = self._monotonic() - started
                if waited >= self._max_pause:
                    logger.warning(
                        "enrich.yield_timed_out",
                        waited_seconds=round(waited, 2),
                        max_pause_seconds=self._max_pause,
                        note="proceeding anyway so the queue still makes progress",
                    )
                    return YieldReport(paused=True, waited_seconds=waited, gave_up=True)
                self._sleep(self._poll_interval)
                if self._query_side_is_idle():
                    waited = self._monotonic() - started
                    logger.info("enrich.resumed", waited_seconds=round(waited, 2))
                    return YieldReport(paused=True, waited_seconds=waited)

    def _query_side_is_idle(self) -> bool:
        """Whether nobody holds the query-in-flight lock.

        Probed by *taking* the conflicting exclusive lock and letting it go
        again immediately: `pg_try_advisory_lock` succeeds only when no
        shared holder exists, and never waits. Holding it for longer would
        be the inversion this module forbids — an MCP server's own
        acquisition is a try, so it would simply skip its marker, but the
        worker has no business making that happen.
        """
        with self._conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (self._query_lock_key,))
            row = cur.fetchone()
            acquired = bool(row and row[0])
            if acquired:
                cur.execute("SELECT pg_advisory_unlock(%s)", (self._query_lock_key,))
        return acquired

    def _published_as_paused(self) -> _PausedMarker:
        return _PausedMarker(self._conn, self._paused_lock_key)


class _PausedMarker:
    """Holds the "an enrichment worker is currently yielding" lock for the
    length of a pause, so `imsg status` can report it. Shared, so two
    workers pausing at once both show."""

    def __init__(self, conn: psycopg.Connection, lock_key: int) -> None:
        self._conn = conn
        self._lock_key = lock_key
        self._held = False

    def __enter__(self) -> Self:
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock_shared(%s)", (self._lock_key,))
                row = cur.fetchone()
            self._held = bool(row and row[0])
        except Exception as exc:  # observability must not break the worker
            logger.warning("enrich.paused_marker_failed", error=f"{type(exc).__name__}: {exc}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._held:
            return
        self._held = False
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock_shared(%s)", (self._lock_key,))
        except Exception as exc_:
            logger.warning("enrich.paused_unmark_failed", error=f"{type(exc_).__name__}: {exc_}")


# --------------------------------------------------------------------------
# observability: what `imsg status` reports
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class YieldState:
    """Who is holding what, right now, in this database."""

    query_in_flight: bool
    enrichment_paused: bool
    reason: str | None = None
    """Set instead of the flags when the locks could not be read."""


_PG_LOCKS_QUERY = """
    SELECT objid, count(*)
    FROM pg_locks
    WHERE locktype = 'advisory'
      AND granted
      AND database = (SELECT oid FROM pg_database WHERE datname = current_database())
      AND classid = %s
      AND objid = ANY(%s)
    GROUP BY objid
"""


def read_yield_state(conn: psycopg.Connection) -> YieldState:
    """Read both markers without taking a lock — a status command must be
    able to observe the system without joining the contention it is
    reporting on."""
    query_objid = QUERY_IN_FLIGHT_LOCK_KEY & 0xFFFFFFFF
    paused_objid = ENRICHMENT_PAUSED_LOCK_KEY & 0xFFFFFFFF
    try:
        with conn.cursor() as cur:
            cur.execute(_PG_LOCKS_QUERY, (_PG_LOCKS_CLASSID, [query_objid, paused_objid]))
            held = {int(objid) for objid, _count in cur.fetchall()}
    except Exception as exc:
        return YieldState(
            query_in_flight=False,
            enrichment_paused=False,
            reason=f"advisory locks not read: {type(exc).__name__}: {exc}",
        )
    return YieldState(
        query_in_flight=query_objid in held,
        enrichment_paused=paused_objid in held,
    )


__all__ = [
    "DEFAULT_MAX_PAUSE_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "ENRICHMENT_PAUSED_LOCK_KEY",
    "QUERY_IN_FLIGHT_LOCK_KEY",
    "EnrichmentYieldGate",
    "QueryInFlightMarker",
    "YieldReport",
    "YieldState",
    "read_yield_state",
]
