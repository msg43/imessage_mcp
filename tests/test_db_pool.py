"""`imsg.db.pool.ResourcePool`: bounded, reuses what is healthy, drops what
is not, never deadlocks a caller that can go elsewhere. Connections here
are plain objects; `tests/test_retrieval_concurrent_channels_integration.py`
runs the pool against real Postgres and SQLite."""

from __future__ import annotations

import threading
import time

import pytest

from imsg.db.pool import PoolClosedError, PoolExhaustedError, ResourcePool


class Conn:
    opened = 0

    def __init__(self) -> None:
        Conn.opened += 1
        self.id = Conn.opened
        self.healthy = True
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_pool(max_size: int = 2, wait_seconds: float = 0.2) -> ResourcePool[Conn]:
    return ResourcePool(
        Conn,
        close_connection=lambda conn: conn.close(),
        reusable=lambda conn: conn.healthy and not conn.closed,
        max_size=max_size,
        name="test",
        wait_seconds=wait_seconds,
    )


def test_connections_open_lazily_and_are_reused() -> None:
    pool = make_pool()
    assert pool.size() == 0
    with pool.lease() as first:
        pass
    with pool.lease() as second:
        pass
    assert first is second
    assert pool.size() == 1


def test_the_pool_never_exceeds_its_size_and_a_lease_waits_then_gives_up() -> None:
    pool = make_pool(max_size=2, wait_seconds=0.2)
    a = pool.acquire()
    b = pool.acquire()
    assert a is not b
    assert pool.acquire_if_free() is None  # full: never waits
    started = time.monotonic()
    with pytest.raises(PoolExhaustedError):
        pool.acquire()
    assert time.monotonic() - started >= 0.2
    pool.release(a)
    assert pool.acquire() is a
    pool.release(a)
    pool.release(b)


def test_a_waiting_lease_gets_the_connection_handed_back() -> None:
    pool = make_pool(max_size=1, wait_seconds=5)
    held = pool.acquire()
    got: list[Conn] = []

    def wait_for_it() -> None:
        with pool.lease() as conn:
            got.append(conn)

    waiter = threading.Thread(target=wait_for_it)
    waiter.start()
    time.sleep(0.1)
    assert got == []
    pool.release(held)
    waiter.join(timeout=5)
    assert got == [held]


def test_an_unusable_connection_is_closed_and_never_handed_out_again() -> None:
    pool = make_pool()
    with pool.lease() as conn:
        conn.healthy = False
    assert conn.closed
    assert pool.size() == 0
    with pool.lease() as fresh:
        assert fresh is not conn


def test_a_connection_that_fails_to_open_frees_its_place() -> None:
    attempts = 0

    def flaky() -> Conn:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("database unreachable")
        return Conn()

    pool: ResourcePool[Conn] = ResourcePool(
        flaky,
        close_connection=lambda conn: conn.close(),
        reusable=lambda conn: True,
        max_size=1,
        name="flaky",
    )
    with pytest.raises(OSError):
        pool.acquire()
    assert pool.size() == 0 and pool.in_use() == 0
    with pool.lease() as conn:
        assert isinstance(conn, Conn)


def test_closing_closes_idle_connections_now_and_leased_ones_on_return() -> None:
    pool = make_pool()
    idle = pool.acquire()
    leased = pool.acquire()
    pool.release(idle)
    pool.close()
    assert idle.closed
    assert not leased.closed
    pool.release(leased)
    assert leased.closed
    with pytest.raises(PoolClosedError):
        pool.acquire()


def test_a_pool_needs_room_for_one() -> None:
    with pytest.raises(ValueError):
        make_pool(max_size=0)
