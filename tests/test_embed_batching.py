"""Length-sorted, token-budgeted batching for the text embedders:
`imsg.embed.batching.plan_batches` (pure) and the S6 pipeline's
`plan_pending_batches` / `_embed_segments` / `_embed_chunks` driven by
the fake provider through a recording stand-in for the Postgres
connection — no model, no database. What matters: every row is
embedded exactly once, keyed by its id, in batches that respect both
bounds, and the rows written do not depend on the order the pending
rows arrived in."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from imsg.embed.batching import DEFAULT_MAX_BATCH_TOKENS, padded_tokens, plan_batches
from imsg.embed.pipeline import (
    CHUNK_UPSERT_SQL,
    SEGMENT_UPSERT_SQL,
    PendingRow,
    _embed_chunks,
    _embed_segments,
    plan_pending_batches,
)
from imsg.embed.provider import FakeTextEmbeddingProvider
from imsg.embed.vector_codec import vector_literal
from imsg.errors import EmbeddingError
from imsg.hashing import sha256_text
from imsg.textnorm import normalize_text
from imsg.tokens import estimate_tokens

# --- plan_batches -----------------------------------------------------------


def _check_invariants(
    lengths: list[int], batches: list[list[int]], *, max_batch_size: int, max_batch_tokens: int
) -> None:
    seen = sorted(i for batch in batches for i in batch)
    assert seen == list(range(len(lengths))), "every index exactly once"
    widths = []
    for batch in batches:
        assert batch, "no empty batches"
        assert len(batch) <= max_batch_size
        rows = [lengths[i] for i in batch]
        assert rows == sorted(rows, reverse=True), "longest row first within a batch"
        width = rows[0]
        assert len(batch) * width <= max_batch_tokens or len(batch) == 1, (
            "over budget only when a single row is itself larger than the budget"
        )
        widths.append(width)
    assert widths == sorted(widths, reverse=True), "longest batch first"


def test_plan_batches_respects_both_bounds_on_a_skewed_distribution() -> None:
    rng = random.Random(7)
    lengths = [max(1, round(rng.lognormvariate(4.5, 1.1))) for _ in range(500)]
    for max_batch_size, budget in [(32, 8192), (8, 100_000), (4096, 2048), (1, 16)]:
        batches = plan_batches(lengths, max_batch_size=max_batch_size, max_batch_tokens=budget)
        _check_invariants(lengths, batches, max_batch_size=max_batch_size, max_batch_tokens=budget)


def test_plan_batches_packs_greedily_longest_first() -> None:
    lengths = [10, 300, 20, 100, 100, 5]
    batches = plan_batches(lengths, max_batch_size=32, max_batch_tokens=400)
    # 300 alone (2 x 300 > 400); 100, 100, 20, 10 (4 x 100 = 400); 5 alone
    assert batches == [[1], [3, 4, 2, 0], [5]]
    assert padded_tokens(lengths, batches) == 300 + 400 + 5


def test_plan_batches_row_cap_binds_before_the_budget() -> None:
    lengths = [3] * 10
    assert plan_batches(lengths, max_batch_size=4, max_batch_tokens=10_000) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


def test_plan_batches_oversize_row_is_embedded_alone() -> None:
    lengths = [9000, 50, 50, 8193]
    batches = plan_batches(lengths, max_batch_size=32, max_batch_tokens=8192)
    assert batches == [[0], [3], [1, 2]]


def test_plan_batches_stable_for_equal_lengths_and_empty_input() -> None:
    assert plan_batches([], max_batch_size=8, max_batch_tokens=100) == []
    assert plan_batches([7, 7, 7], max_batch_size=2, max_batch_tokens=100) == [[0, 1], [2]]


def test_plan_batches_is_order_independent_up_to_ties() -> None:
    rng = random.Random(11)
    lengths = rng.sample(range(1, 2000), 300)  # distinct, so the plan is unique
    batches = plan_batches(lengths, max_batch_size=16, max_batch_tokens=4096)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    shuffled = [lengths[i] for i in order]
    shuffled_batches = plan_batches(shuffled, max_batch_size=16, max_batch_tokens=4096)
    as_lengths = [[lengths[i] for i in batch] for batch in batches]
    assert [[shuffled[i] for i in batch] for batch in shuffled_batches] == as_lengths


@pytest.mark.parametrize(
    ("lengths", "kwargs"),
    [
        ([1], {"max_batch_size": 0, "max_batch_tokens": 10}),
        ([1], {"max_batch_size": 1, "max_batch_tokens": 0}),
        ([0], {"max_batch_size": 1, "max_batch_tokens": 10}),
    ],
)
def test_plan_batches_rejects_nonsense(lengths: list[int], kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        plan_batches(lengths, **kwargs)


def test_default_budget_is_the_measured_one() -> None:
    # scripts/bench_text_embedding.py, 2026-09-15: padded throughput is flat
    # from 2 rows upward, so the budget is the smallest that keeps padding
    # under 10% (see imsg.embed.batching.DEFAULT_MAX_BATCH_TOKENS).
    assert DEFAULT_MAX_BATCH_TOKENS == 2048


def test_padded_tokens_counts_rows_times_width() -> None:
    assert padded_tokens([5, 3, 9], [[2, 0], [1]]) == 2 * 9 + 3
    assert padded_tokens([5, 3, 9], []) == 0


# --- the pipeline, on a fake connection ------------------------------------


class _RecordingCursor:
    def __init__(self, sink: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._sink = sink

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._sink.append((sql, params))


class _RecordingConnection:
    """Just enough of `psycopg.Connection` for `_embed_rows`: every
    `execute` is recorded under the transaction it ran in."""

    def __init__(self) -> None:
        self.transactions: list[list[tuple[str, tuple[Any, ...]]]] = []
        self._open: list[tuple[str, tuple[Any, ...]]] | None = None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._open = []
        try:
            yield
        finally:
            self.transactions.append(self._open)
            self._open = None

    @contextmanager
    def cursor(self) -> Iterator[_RecordingCursor]:
        assert self._open is not None, "writes happen inside a transaction"
        yield _RecordingCursor(self._open)


class _SpyProvider(FakeTextEmbeddingProvider):
    """The fake provider, recording every `embed_documents` batch."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__(dim)
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return super().embed_documents(texts)


def _pending(count: int, seed: int) -> list[PendingRow]:
    """Pending rows of wildly different lengths, ids not in length order."""
    rng = random.Random(seed)
    rows: list[PendingRow] = []
    for row_id in range(1, count + 1):
        words = max(1, round(rng.lognormvariate(3.5, 1.2)))
        text = " ".join(f"w{row_id}x{k}" for k in range(words)) + "  \u00a0trailing"
        rows.append((row_id, text, sha256_text(text)))
    return rows


def _written(conn: _RecordingConnection) -> dict[int, tuple[str, str]]:
    """`{id: (text_sha256, vector literal)}` across every transaction."""
    out: dict[int, tuple[str, str]] = {}
    for transaction in conn.transactions:
        for _sql, params in transaction:
            row_id, _model, _dim, text_hash, literal = params
            assert row_id not in out, f"row {row_id} written twice"
            out[row_id] = (text_hash, literal)
    return out


def test_plan_pending_batches_normalizes_text_and_keys_rows_by_id() -> None:
    pending = _pending(40, seed=1)
    batches = plan_pending_batches(pending, batch_size=8, max_batch_tokens=200)
    flat = [row for batch in batches for row in batch]
    assert sorted(row_id for row_id, _, _ in flat) == list(range(1, 41))
    by_id = {row_id: (text, digest) for row_id, text, digest in pending}
    for row_id, text, digest in flat:
        assert text == normalize_text(by_id[row_id][0])
        assert digest == by_id[row_id][1]  # the raw-text hash travels untouched
    lengths = [estimate_tokens(text) for _, text, _ in flat]
    index_batches: list[list[int]] = []
    cursor = 0
    for batch in batches:
        index_batches.append(list(range(cursor, cursor + len(batch))))
        cursor += len(batch)
    _check_invariants(lengths, index_batches, max_batch_size=8, max_batch_tokens=200)


def test_embed_segments_writes_every_row_once_in_one_transaction_per_batch() -> None:
    pending = _pending(50, seed=2)
    conn = _RecordingConnection()
    provider = _SpyProvider()
    written = _embed_segments(conn, provider, pending, 8, 300)  # type: ignore[arg-type]
    assert written == 50
    assert len(conn.transactions) == len(provider.batches) > 1
    assert [len(t) for t in conn.transactions] == [len(b) for b in provider.batches]
    assert all(sql == SEGMENT_UPSERT_SQL for t in conn.transactions for sql, _ in t)
    rows = _written(conn)
    assert sorted(rows) == list(range(1, 51))
    for row_id, text, digest in pending:
        [expected] = provider.embed_documents([normalize_text(text)])
        assert rows[row_id][0] == digest
        assert rows[row_id][1] == vector_literal(expected)


def test_embed_segments_result_does_not_depend_on_pending_order() -> None:
    pending = _pending(60, seed=3)
    shuffled = list(pending)
    random.Random(4).shuffle(shuffled)

    first, second = _RecordingConnection(), _RecordingConnection()
    _embed_segments(first, _SpyProvider(), pending, 16, 400)  # type: ignore[arg-type]
    _embed_segments(second, _SpyProvider(), shuffled, 16, 400)  # type: ignore[arg-type]
    assert _written(first) == _written(second)


def test_embed_segments_batches_respect_the_budget_and_isolate_an_oversize_row() -> None:
    pending = _pending(30, seed=5)
    huge = " ".join(f"z{k}" for k in range(3000))  # ~ thousands of estimated tokens
    pending.append((999, huge, sha256_text(huge)))
    provider = _SpyProvider()
    conn = _RecordingConnection()
    _embed_segments(conn, provider, pending, 32, 512)  # type: ignore[arg-type]

    assert provider.batches[0] == [normalize_text(huge)], "the oversize row goes first and alone"
    for batch in provider.batches[1:]:
        lengths = [estimate_tokens(text) for text in batch]
        assert len(batch) * max(lengths) <= 512 or len(batch) == 1
        assert len(batch) <= 32
    assert 999 in _written(conn)


def test_embed_chunks_uses_the_chunk_table_with_the_same_batching() -> None:
    pending = _pending(20, seed=6)
    conn = _RecordingConnection()
    provider = _SpyProvider()
    assert _embed_chunks(conn, provider, pending, 4, 1_000_000) == 20  # type: ignore[arg-type]
    assert all(sql == CHUNK_UPSERT_SQL for t in conn.transactions for sql, _ in t)
    assert [len(b) for b in provider.batches] == [4, 4, 4, 4, 4]
    assert sorted(_written(conn)) == list(range(1, 21))


def test_embed_segments_rejects_a_provider_that_returns_the_wrong_count() -> None:
    class _ShortProvider(FakeTextEmbeddingProvider):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents(texts)[:-1]

    conn = _RecordingConnection()
    with pytest.raises(EmbeddingError, match=r"returned 2 vectors for a batch of 3 segments"):
        _embed_segments(conn, _ShortProvider(8), _pending(3, seed=8), 32, 1_000_000)  # type: ignore[arg-type]
    assert conn.transactions == []  # nothing committed for the failing batch


def test_embed_segments_rejects_a_wrong_dimension_before_writing() -> None:
    class _WideProvider(FakeTextEmbeddingProvider):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[*v, 0.0] for v in super().embed_documents(texts)]

    conn = _RecordingConnection()
    with pytest.raises(EmbeddingError, match=r"returned a 9-dim vector for segment"):
        _embed_segments(conn, _WideProvider(8), _pending(2, seed=9), 32, 1_000_000)  # type: ignore[arg-type]
