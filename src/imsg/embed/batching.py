"""Length-sorted, token-budgeted batch planning for the text embedders
(SPEC §8 S6).

A right-padded batch costs the model ``rows x longest_row`` tokens, not
``sum(rows)``: every row is padded to the longest one, and the padding
positions are computed (and attended over) like real tokens. Batching
segments in ``segment_id`` order therefore pays for whatever the
longest segment in each arbitrary group of 32 happens to be — measured
on the real corpus (2026-09-15, `scripts/bench_text_embedding.py`) that
inflated 17.3M real tokens to 68.3M padded ones, a 3.9x tax on top of
the model's actual work.

:func:`plan_batches` removes that tax without touching the results:
it groups indices so that each batch holds rows of similar length,
and bounds every batch by a *padded* token budget as well as a row
count. Because the pipeline keys every result by its id and commits
per batch, the order in which rows are embedded changes nothing
observable — it only changes how much padding the model computes.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MAX_BATCH_TOKENS = 2048
"""Default padded-token budget of one batch (`rows x longest row`).
Shared by the pipeline (`embedding.max_batch_tokens`, in
`imsg.tokens.estimate_tokens` units) and the MLX provider (in its own
tokenizer's counts).

Measured 2026-09-15 with `scripts/bench_text_embedding.py` on an M2
Ultra (Qwen3-Embedding-8B, mxfp8, 256 synthetic texts drawn from the
real corpus's length distribution — median 96, p99 805 tokens):
padded throughput is flat at ~900-930 tokens/s for every batch shape
from 2 rows up to 32k padded tokens, so a bigger batch buys nothing
and every padded token is a lost real one. Real tokens/s by budget:
1k 861, 2k 845, 4k 790, 8k 686, 16k 561, 32k 429; fixed batches of 32
in id order (the behaviour before this module) 178. 2k is where
padded throughput has reached ~99% of its ceiling (911 vs 923) while
padding is still under 10%; 1k is within noise of it with twice the
transactions. Peak active memory at 2k is 8.6 GiB including the
weights."""


def plan_batches(
    lengths: Sequence[int], *, max_batch_size: int, max_batch_tokens: int
) -> list[list[int]]:
    """Group the indices of ``lengths`` into batches for a right-padded
    encoder.

    Rows are taken longest-first (a stable sort, so equal lengths keep
    their input order). Each batch is filled greedily while both bounds
    hold: at most ``max_batch_size`` rows, and a *padded* cost of
    ``rows x longest row in the batch`` no greater than
    ``max_batch_tokens``. Because the first row of a batch is its
    longest, the padded width is fixed the moment a batch opens, so
    the greedy fill never has to re-check earlier rows.

    A single row longer than ``max_batch_tokens`` is emitted alone: the
    budget bounds *padding*, and a row that is all real tokens has none
    to trim — the caller's ``max_length`` truncation is what bounds a
    single row. Every index appears in exactly one batch; the longest
    batch comes first, so an out-of-memory failure surfaces on the
    first forward pass, not hours in.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    if max_batch_tokens < 1:
        raise ValueError(f"max_batch_tokens must be >= 1, got {max_batch_tokens}")
    for index, length in enumerate(lengths):
        if length < 1:
            raise ValueError(f"row {index} has length {length}; every row needs >= 1 token")

    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for index in order:
        if current:
            fits_rows = len(current) < max_batch_size
            fits_tokens = (len(current) + 1) * width <= max_batch_tokens
            if fits_rows and fits_tokens:
                current.append(index)
                continue
            batches.append(current)
        current = [index]
        width = lengths[index]
    if current:
        batches.append(current)
    return batches


def padded_tokens(lengths: Sequence[int], batches: Sequence[Sequence[int]]) -> int:
    """The tokens a right-padded encoder computes for ``batches`` —
    ``sum(rows x longest row)`` — versus ``sum(lengths)`` of real tokens.
    A diagnostic for logs and the benchmark, not a decision input."""
    return sum(len(batch) * max(lengths[i] for i in batch) for batch in batches if batch)


__all__ = ["DEFAULT_MAX_BATCH_TOKENS", "padded_tokens", "plan_batches"]
