"""Connections for a retrieval service that answers several calls at once
and runs a search's candidate channels side by side.

A search has five candidate channels (SPEC §9.4): segment FTS and
attachment FTS (the SQLite sidecar, authorized against Postgres), and the
segment, attachment and multimodal vector searches (Postgres). They are
independent of each other — each takes the query and the compiled filter
and returns ranked segment ids — so they can run at the same time, and
the two FTS channels can run while the query is being embedded. They
cannot share a connection to do so: neither driver tolerates two threads
on one connection (`imsg.retrieval.service` has the two driver errors).
So a service built with a :class:`RetrievalConnections` gives each call
its own Postgres connection, and each channel its own connections and a
worker thread whenever the pools have one free; a channel that finds
none runs on the call's own connection instead, after the others, so a
busy pool slows a search down and never deadlocks it.

Ownership follows the codebase's rule: the process that builds this
(`imsg mcp public`) opens and closes it; retrieval code only borrows.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import apsw

from imsg.db.pool import DEFAULT_WAIT_SECONDS, ResourcePool

if TYPE_CHECKING:
    import psycopg

CHANNEL_THREAD_PREFIX = "imsg-channel"


def fts_connection_is_reusable(conn: apsw.Connection) -> bool:
    """Not left inside a transaction. The channels only read, so this is
    a guard against the unexpected, not a routine case."""
    return not conn.in_transaction


def fts_pool(
    open_connection: Callable[[], apsw.Connection],
    *,
    max_size: int,
    name: str = "fts",
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
) -> ResourcePool[apsw.Connection]:
    """A pool of connections to the FTS5 sidecar."""
    return ResourcePool(
        open_connection,
        close_connection=lambda conn: conn.close(),
        reusable=fts_connection_is_reusable,
        max_size=max_size,
        name=name,
        wait_seconds=wait_seconds,
    )


def open_fts_reader(path: Path) -> apsw.Connection:
    """Another connection to the sidecar the server already opened and
    checked (`imsg mcp public` runs `create_schema` and
    `assert_schema_current` on its first one), opened the same way —
    read-write flags, no per-connection settings — so a channel on it
    behaves exactly as it would on the first."""
    return apsw.Connection(str(path))


class RetrievalConnections:
    """A Postgres pool, an FTS-sidecar pool, and the worker threads search
    channels run on. Thread-safe; close once, after the last call."""

    def __init__(
        self,
        *,
        pg: ResourcePool[psycopg.Connection],
        fts: ResourcePool[apsw.Connection],
        channel_threads: int | None = None,
    ) -> None:
        self.pg = pg
        self.fts = fts
        # A channel only runs on a worker once it holds a Postgres
        # connection, so there is never more channel work than the
        # Postgres pool's size: that many threads never leaves a leased
        # connection waiting for one.
        workers = channel_threads if channel_threads is not None else pg.max_size
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=CHANNEL_THREAD_PREFIX
        )

    def submit[R](self, work: Callable[[], R]) -> Future[R]:
        return self._executor.submit(work)

    def close(self) -> None:
        """Wait for running channels, then close every pooled connection."""
        self._executor.shutdown(wait=True)
        self.pg.close()
        self.fts.close()


__all__ = [
    "CHANNEL_THREAD_PREFIX",
    "RetrievalConnections",
    "fts_connection_is_reusable",
    "fts_pool",
    "open_fts_reader",
]
