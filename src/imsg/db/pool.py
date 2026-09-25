"""A small, bounded pool of reusable connections, for a process that
answers more than one request at a time.

`imsg mcp public` opened a new Postgres connection for every audit row
(8.2-10.1 ms to connect on the index host, QA review 2026-09-24) and ran
every query on the one connection it opened at start, which is part of
why it answered one request at a time. A pool fixes both: an audit write
borrows an open connection instead of opening one, and each tool call —
and each candidate channel inside a search — gets a connection of its own.

The pool never owns what a connection is used for. A caller leases one,
uses it exactly as it would a connection it had opened (DB code still
takes an already-open connection and never owns its lifecycle), and the
lease hands it back. A connection that comes back unusable — closed,
broken, a transaction left open — is closed and dropped, never handed to
the next caller.

Generic over the connection type, because the search needs two kinds: a
`psycopg.Connection` for Postgres and an `apsw.Connection` for the FTS5
sidecar (:func:`postgres_pool`, and `imsg.retrieval.connections` for the
sidecar).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from imsg.errors import ImsgError

if TYPE_CHECKING:
    import psycopg

DEFAULT_WAIT_SECONDS = 30.0


class PoolExhaustedError(ImsgError):
    """No connection came free within the pool's wait — every one is in
    use and the pool is at its maximum size."""


class PoolClosedError(ImsgError):
    """A lease was asked of a pool that has been closed."""


class ResourcePool[T]:
    """Up to `max_size` open connections, handed out one caller at a time.

    `lease()` waits (up to `wait_seconds`) for a free connection;
    `lease_if_free()` never waits and yields `None` instead, for work that
    has somewhere else to run (a search channel falls back to the
    caller's own connection). Idle connections are reused newest first,
    so a quiet server keeps its few warm connections rather than cycling
    through all of them. Connections open lazily, on demand. Thread-safe.
    """

    def __init__(
        self,
        open_connection: Callable[[], T],
        *,
        close_connection: Callable[[T], None],
        reusable: Callable[[T], bool],
        max_size: int,
        name: str,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
    ) -> None:
        if max_size < 1:
            raise ValueError("a pool needs room for at least one connection")
        self._open = open_connection
        self._close_one = close_connection
        self._reusable = reusable
        self._max_size = max_size
        self._name = name
        self._wait_seconds = wait_seconds
        self._condition = threading.Condition()
        self._idle: list[T] = []
        self._size = 0  # connections that exist: idle + leased + being opened
        self._leased = 0
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_size(self) -> int:
        return self._max_size

    def in_use(self) -> int:
        """How many connections are leased right now."""
        with self._condition:
            return self._leased

    def size(self) -> int:
        """How many connections exist right now, idle or leased."""
        with self._condition:
            return self._size

    def _take(self, *, wait: bool) -> T | None:
        deadline = time.monotonic() + self._wait_seconds
        with self._condition:
            while True:
                if self._closed:
                    raise PoolClosedError(f"connection pool {self._name!r} is closed")
                if self._idle:
                    self._leased += 1
                    return self._idle.pop()
                if self._size < self._max_size:
                    self._size += 1
                    self._leased += 1
                    break  # open one, outside the lock
                if not wait:
                    return None
                left = deadline - time.monotonic()
                if left <= 0:
                    raise PoolExhaustedError(
                        f"no connection in pool {self._name!r} came free within "
                        f"{self._wait_seconds:g} s (all {self._max_size} in use)"
                    )
                self._condition.wait(left)
        try:
            return self._open()
        except BaseException:
            with self._condition:
                self._size -= 1
                self._leased -= 1
                self._condition.notify()
            raise

    def _give_back(self, conn: T) -> None:
        keep = False
        try:
            keep = self._reusable(conn)
        except Exception:
            keep = False
        with self._condition:
            self._leased -= 1
            if keep and not self._closed:
                self._idle.append(conn)
                self._condition.notify()
                return
            self._size -= 1
            self._condition.notify()
        self._discard(conn)

    def _discard(self, conn: T) -> None:
        # A connection that will not close cleanly is gone to us either way.
        with contextlib.suppress(Exception):
            self._close_one(conn)

    @contextmanager
    def lease(self) -> Iterator[T]:
        """A connection for the length of the `with` block, waiting up to
        `wait_seconds` for one to come free."""
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    @contextmanager
    def lease_if_free(self) -> Iterator[T | None]:
        """A connection if one is free or can be opened now, else `None`."""
        conn = self.acquire_if_free()
        try:
            yield conn
        finally:
            if conn is not None:
                self.release(conn)

    def acquire(self) -> T:
        """A connection, waiting up to `wait_seconds`; the caller must
        :meth:`release` it. Prefer :meth:`lease` — this pair exists for a
        connection taken on one thread and handed back on another (a
        search channel's worker)."""
        conn = self._take(wait=True)
        assert conn is not None  # _take(wait=True) returns or raises
        return conn

    def acquire_if_free(self) -> T | None:
        """A connection if one is free or can be opened now, else `None`,
        without waiting; the caller must :meth:`release` it."""
        return self._take(wait=False)

    def release(self, conn: T) -> None:
        """Hand back a connection from :meth:`acquire`/`acquire_if_free`."""
        self._give_back(conn)

    def close(self) -> None:
        """Close every idle connection now and every leased one as it
        comes back; any later lease raises :class:`PoolClosedError`."""
        with self._condition:
            self._closed = True
            idle, self._idle = self._idle, []
            self._size -= len(idle)
            self._condition.notify_all()
        for conn in idle:
            self._discard(conn)


def postgres_connection_is_reusable(conn: psycopg.Connection) -> bool:
    """Open, not broken, and idle — no transaction left open, no query
    still running. Every connection this codebase opens is autocommit
    (`imsg.db.connection.connect`), so a connection that is not idle was
    left mid-transaction by an error, and the next caller must not
    inherit it."""
    import psycopg

    if conn.closed or conn.broken:
        return False
    return conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def postgres_pool(
    open_connection: Callable[[], psycopg.Connection],
    *,
    max_size: int,
    name: str,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
) -> ResourcePool[psycopg.Connection]:
    """A pool of Postgres connections, each opened by `open_connection`
    (which should run the same checks as the process's first connection,
    e.g. the cluster fingerprint)."""
    return ResourcePool(
        open_connection,
        close_connection=lambda conn: conn.close(),
        reusable=postgres_connection_is_reusable,
        max_size=max_size,
        name=name,
        wait_seconds=wait_seconds,
    )


__all__ = [
    "DEFAULT_WAIT_SECONDS",
    "PoolClosedError",
    "PoolExhaustedError",
    "ResourcePool",
    "postgres_connection_is_reusable",
    "postgres_pool",
]
