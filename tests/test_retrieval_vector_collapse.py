"""`imsg.retrieval.vector_search`'s collapsing channels (B2, C) read their
overfetch through a server-side cursor and stop once no unread row can
change the collapsed result. These tests pin that the early stop returns
exactly what collapsing every row returns — including when rows arrive
slightly out of distance order, as pgvector's rows do — and that it
really does stop early. No database: a fake connection serves the rows
(the scratch-Postgres check lives in `test_retrieval_integration.py`)."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import psycopg
import pytest

from imsg.retrieval import vector_search
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.filters import SearchFilters, compile_predicate
from imsg.retrieval.vector_search import (
    COLLAPSE_FETCH_ROWS,
    DISTANCE_ORDER_TOLERANCE,
    _best_distance_per_segment,
    _collapsed_nearest_segments,
    collapse_is_settled,
    search_attachment_chunk_vector,
    search_multimodal_vector,
)


class _FakeCursor:
    def __init__(self, conn: _FakeConn, name: str | None) -> None:
        self._conn = conn
        self.name = name
        self._position = 0

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        assert self._conn.in_transaction, "every statement runs inside the channel's transaction"
        self._conn.statements.append((self.name, sql, params))

    def fetchmany(self, size: int) -> list[tuple[int, float]]:
        assert self.name is not None, "rows are streamed through a named (server-side) cursor"
        self._conn.fetch_sizes.append(size)
        batch = self._conn.rows[self._position : self._position + size]
        self._position += len(batch)
        self._conn.rows_served += len(batch)
        return list(batch)


class _FakeConn:
    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self.rows = rows
        self.in_transaction = False
        self.statements: list[tuple[str | None, str, dict[str, Any] | None]] = []
        self.fetch_sizes: list[int] = []
        self.rows_served = 0

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.in_transaction = True
        try:
            yield
        finally:
            self.in_transaction = False

    def cursor(self, name: str | None = None) -> _FakeCursor:
        return _FakeCursor(self, name)


def _collapse(conn: _FakeConn, k: int) -> vector_search.VectorChannelResult:
    return _collapsed_nearest_segments(
        cast(psycopg.Connection, conn), "SELECT ...", {"row_limit": len(conn.rows)}, k
    )


def _noisy_rows(rng: random.Random, n: int, segments: int, noise: float) -> list[tuple[int, float]]:
    """`n` rows in true distance order, each carrying rounding-like noise
    of at most `noise`, drawn from `segments` segments with repeats."""
    true = sorted(rng.uniform(0.2, 0.9) for _ in range(n))
    return [(rng.randrange(segments), t + rng.uniform(-noise, noise)) for t in true]


# --- collapse_is_settled ---------------------------------------------------


def test_not_settled_before_k_distinct_segments() -> None:
    assert not collapse_is_settled({1: 0.1, 2: 0.2}, 5.0, 3)


def test_settled_only_once_the_farthest_row_clears_the_kth_best_by_the_tolerance() -> None:
    best = {1: 0.10, 2: 0.20, 3: 0.30}
    assert not collapse_is_settled(best, 0.30, 3, tolerance=0.01)
    assert not collapse_is_settled(best, 0.31, 3, tolerance=0.01)  # a tie is still possible
    assert collapse_is_settled(best, 0.3101, 3, tolerance=0.01)
    assert collapse_is_settled(best, 0.25, 2, tolerance=0.01)  # k-th best is 0.20 for k=2


# --- the streamed collapse -------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_early_stop_returns_exactly_the_full_collapse(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.choice([50, 200, 500])
    segments = rng.choice([30, 120, 400])
    k = rng.choice([5, 20, 100])
    # rows out of order by up to 90 % of the tolerance, as pgvector's are (by far less)
    rows = _noisy_rows(rng, n, segments, noise=0.45 * DISTANCE_ORDER_TOLERANCE)
    conn = _FakeConn(rows)
    assert _collapse(conn, k) == _best_distance_per_segment(rows, k)


def test_out_of_order_rows_keep_the_full_collapse_order() -> None:
    # Segment 7's first row is farther than segment 3's later row: the
    # collapse orders by best distance, not by first appearance.
    rows = [(7, 0.5004), (3, 0.5003), (3, 0.5001), (9, 0.6), (8, 0.7), *[(9, 0.8)] * 200]
    conn = _FakeConn(rows)
    result = _collapse(conn, 2)
    assert result.segment_ids == (3, 7) == _best_distance_per_segment(rows, 2).segment_ids


def test_stops_reading_once_settled() -> None:
    rng = random.Random(3)
    rows = [(i, 0.3 + i * 0.001) for i in range(20)]  # 20 distinct segments first
    rows += [(rng.randrange(20), 0.5 + i * 0.001) for i in range(480)]
    conn = _FakeConn(rows)
    result = _collapse(conn, 10)
    assert result == _best_distance_per_segment(rows, 10)
    assert conn.rows_served == COLLAPSE_FETCH_ROWS  # one fetch was enough
    assert conn.rows_served < len(rows)


def test_reads_everything_when_fewer_than_k_segments_exist() -> None:
    rows = [(i % 4, 0.1 + i * 0.001) for i in range(150)]
    conn = _FakeConn(rows)
    result = _collapse(conn, 10)
    assert result == _best_distance_per_segment(rows, 10)
    assert result.scan_cap_reached is True
    assert conn.rows_served == len(rows)


def test_sets_the_scan_and_cursor_planning_inside_the_same_transaction_before_streaming() -> None:
    conn = _FakeConn([(1, 0.1)])
    predicate = compile_predicate(SearchFilters(), LOCAL_FULL_ACCESS)
    search_multimodal_vector(cast(psycopg.Connection, conn), [0.1, 0.2], predicate, 5)
    (scan_name, scan_sql, _), (plan_name, plan_sql, _), (cursor_name, sql, params) = (
        conn.statements
    )
    assert scan_name is None and "hnsw.iterative_scan = 'strict_order'" in scan_sql
    # planned like the plain query, so the cursor cannot switch plans (and rows)
    assert plan_name is None and "cursor_tuple_fraction = 1.0" in plan_sql
    assert cursor_name is not None and "attachment_mm_embedding" in sql
    assert params is not None and params["row_limit"] == 50  # max(5 * 5, 50)
    assert conn.fetch_sizes == [COLLAPSE_FETCH_ROWS]

    conn = _FakeConn([(1, 0.1)])
    search_attachment_chunk_vector(cast(psycopg.Connection, conn), [0.1, 0.2], predicate, 20)
    (_, _, _), (_, _, _), (_, chunk_sql, chunk_params) = conn.statements
    assert "attachment_chunk_embedding" in chunk_sql
    assert chunk_params is not None and chunk_params["row_limit"] == 100
