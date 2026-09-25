"""Qwen3-Reranker via MLX — the real :class:`~imsg.retrieval.reranker.
RerankerProvider` (SPEC §4.1, §9.4 step 7: "Rerank top ``rerank_top``
with Qwen3-Reranker-0.6B on (query, rendered_text) pairs"). Any
Qwen3-Reranker size loads through it: the 8B has a separate ``lm_head``,
the 0.6B ties it to the input embedding
(:func:`imsg.mlx_runtime.lm_head_logits` reads either).

The scoring recipe is the model card's Transformers reference, verified
against the published card and tokenizer files (not from memory):

- one chat-formatted prompt per (query, document) pair —
  ``RERANKER_PREFIX`` (system turn asking for a "yes"/"no" judgement and
  the opening of the user turn) + ``"<Instruct>: …\\n<Query>: …\\n
  <Document>: …"`` + ``RERANKER_SUFFIX`` (closing the user turn and
  opening an assistant turn with an empty ``<think>`` block);
- the pair body is truncated so prefix + body + suffix fits
  ``max_length``, exactly as the reference computes its budget;
- the score is the softmax over the *last-position* logits of the
  ``"yes"`` and ``"no"`` tokens, read as P(yes) — the reference's
  ``log_softmax([no, yes])[1].exp()``.

Only the base transformer's last-token hidden state is projected
through the LM head (never the full sequence), so the vocabulary-sized
logits are ``batch x 1 x vocab`` rather than ``batch x seq x vocab``.

**Batching (2026-09-16).** A right-padded batch costs ``rows x longest
row``, and the rerank pool mixes a few long segments with many short
ones, so fixed groups of 8 in pool order paid for the longest pair in
every group: one warm query on the real corpus spent 63.75 s in
:meth:`MlxRerankerProvider.score` for a 50-document pool. Pairs are now
tokenized first and packed by :func:`imsg.embed.batching.plan_batches`
— longest first, bounded by rows *and* by padded tokens, an oversize
pair alone — the embedder's planner, applied to each assembled row's
real length. Scores are written back by index, so the order they are
returned in is the input order whatever the batches were.

**Document cap.** ``doc_max_tokens`` (``retrieval.
rerank_doc_max_tokens``) bounds the *document* part of the pair only:
the instruction and query are never cut, and the chat suffix is appended
after every truncation, so the position the yes/no logits are read from
is always the row's last real token.

**Shared prefix (2026-09-25).** Every row of one query starts with the
same tokens: the chat prefix, the instruction, the query and
``"<Document>:"`` — 69 of a pair's 301 tokens on average on the
2026-09-24 benchmark pools. With ``reuse_prefix`` (the default,
``retrieval.rerank_reuse_prefix``), :meth:`MlxRerankerProvider.score`
runs those tokens through the model once per query, keeps each layer's
keys and values for them (``mlx_lm``'s ``KVCache``), and then runs each
batch on the rest of its rows, which attend to a copy of that cache
instead of recomputing it (:meth:`MlxRerankerProvider.plan`). Under the
causal mask a token depends only on the tokens before it, so each row's
last-token logits are the same function of the same tokens either way.
The arithmetic is grouped differently, so the scores differ by rounding.
Measured 2026-09-25 on the 45 benchmark pools of 20, against scoring each
row alone and whole: in bf16, P(yes) moved by up to 0.061, where batching
whole rows (the path before) moved it by up to 0.070; with the same
weights in float32, by at most 1e-5, with every pair in the same order
(``tests/test_mlx_reranker_real.py``). ``reuse_prefix=False`` runs every
row whole, which is also what a pool of one document does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from imsg.embed.batching import plan_batches
from imsg.mlx_runtime import (
    DEFAULT_CACHE_LIMIT_BYTES,
    MlxRuntimeError,
    base_transformer_hidden_states,
    bound_buffer_cache,
    float_rows,
    format_model_id,
    gather_last_token_states,
    import_mlx_core,
    import_mlx_lm_cache,
    lm_head_logits,
    load_model_and_tokenizer,
    right_pad,
)

RERANKER_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
RERANKER_PREFIX = f"<|im_start|>system\n{RERANKER_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PAIR_TEMPLATE = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
"""The model card's default instruction, used when none is configured."""
YES_TOKEN = "yes"
NO_TOKEN = "no"

DEFAULT_RERANK_BATCH_SIZE = 32
"""Most pairs in one forward pass. The padded-token budget below is the
bound that matters; this only stops a pool of very short pairs from
becoming one enormous batch."""

DEFAULT_RERANK_MAX_BATCH_TOKENS = 8192
"""Most padded tokens (``rows x longest row``) in one forward pass;
``retrieval.rerank_max_batch_tokens`` sets it through the factory.

8,192 since 2026-09-25, for Qwen3-Reranker-0.6B. Measured 2026-09-24 on
an M2 Ultra, 20 synthetic documents of 40-700 tokens cut to 256, 42
timed pools: p95 0.682 s at 1,024 against 0.619 s at 8,192 for the mxfp8
build, and 0.552 s against 0.448 s for the bf16 build. A 1,024-token
budget holds about three ~300-token pairs, so a pool of 20 took about
seven passes; at 8,192 it takes one or two.

The 8B was fastest at 1,024 (below); pass 1,024 when running it.

Measured 2026-09-16 on an M2 Ultra (the pinned mxfp8 conversion of
Qwen3-Reranker-8B, synthetic word-salad pairs, fixed seeds): padded
throughput is flat at ~700-770 tokens/s from 2 rows up, with a few tens
of milliseconds of fixed cost per forward pass (one 84-token pair:
160 ms; two: 285 ms; eight: 893 ms), so the budget only trades padding
against pass count. A 50-pair pool of
13,393 real tokens (longest 845): fixed batches of 8 in pool order
36.3 s (27,784 padded); length-sorted under 1,024 / 2,048 / 4,096 /
8,192 / 16,384 padded tokens 18.9 / 19.8-19.9 / 20.7-21.6 / 22.1-25.0
/ 22.1-31.2 s, with identical scores (largest difference 4.4e-6, same
order). For pools of 10-20 pairs with documents capped at 128-512
tokens, 1,024 was the fastest or within 2 % of it in every case; 256
(one pair per pass) was up to 15 % slower."""

DEFAULT_RERANK_REUSE_PREFIX = True
"""Whether :meth:`MlxRerankerProvider.score` reads the tokens every row of
a query shares once (module docstring, "Shared prefix");
``retrieval.rerank_reuse_prefix`` sets it through the factory. Measured
2026-09-24 on an M2 Ultra with the bf16 build at 8,192 tokens per batch,
same pools as above: p95 0.448 s reading every row whole, 0.395 s
reading the shared prefix once."""


def format_reranker_pair(instruction: str, query: str, document: str) -> str:
    """The user-turn body for one (query, document) pair."""
    return PAIR_TEMPLATE.format(instruction=instruction, query=query, document=document)


def yes_probability(*, no_logit: float, yes_logit: float) -> float:
    """Softmax over the two logits, read as P(yes). Computed in a
    numerically stable form (shift by the max), so extreme logits give
    0.0/1.0 rather than overflow."""
    shift = max(no_logit, yes_logit)
    e_yes = math.exp(yes_logit - shift)
    e_no = math.exp(no_logit - shift)
    return e_yes / (e_yes + e_no)


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    """How many leading tokens ``a`` and ``b`` share."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def shared_prefix_length(rows: Sequence[Sequence[int]]) -> int:
    """How many leading tokens every row has in common, stopping one short
    of the shortest row. The yes/no logits are read at a row's last token,
    so every row keeps at least that token for its own pass."""
    if not rows:
        return 0
    first = rows[0]
    shared = min(common_prefix_length(first, row) for row in rows)
    return max(0, min(shared, min(len(row) for row in rows) - 1))


@dataclass(frozen=True, slots=True)
class RerankPlan:
    """The forward passes :meth:`MlxRerankerProvider.score` makes for one
    query's rows: first the ``shared_prefix`` leading tokens every row has,
    once (0: no such pass, every row is read whole), then one pass per
    entry of ``batches`` over those rows' remaining tokens, right-padded.
    ``row_lengths`` are the whole rows' lengths, in input order."""

    shared_prefix: int
    batches: tuple[tuple[int, ...], ...]
    row_lengths: tuple[int, ...]

    @property
    def passes(self) -> int:
        """Forward passes, the prefix pass included."""
        return len(self.batches) + (1 if self.shared_prefix else 0)

    @property
    def padded_tokens(self) -> int:
        """Token positions the passes compute, padding included: the prefix
        once, plus each batch's rows times its longest remaining row."""
        rest = [n - self.shared_prefix for n in self.row_lengths]
        return self.shared_prefix + sum(
            len(batch) * max(rest[i] for i in batch) for batch in self.batches
        )


def _single_token_id(tokenizer: Any, token: str) -> int:
    """``convert_tokens_to_ids`` for a token that must exist as one vocab
    entry (``"yes"``/``"no"`` do in every Qwen3 tokenizer)."""
    unk = getattr(tokenizer, "unk_token_id", None)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise MlxRuntimeError(f"tokenizer has no single-token id for {token!r}")
    if unk is not None and token_id == unk:
        raise MlxRuntimeError(f"tokenizer maps {token!r} to its unknown-token id")
    return token_id


class MlxRerankerProvider:
    """Any Qwen3-Reranker size (0.6B pinned, 8B retained) through ``mlx_lm``.

    Constructor arguments are plain values (the CLI reads
    ``retrieval.reranker_model`` / ``retrieval.reranker_revision`` /
    ``retrieval.rerank_doc_max_tokens`` and passes them here); weights
    load lazily on the first :meth:`score` (or explicitly via
    :meth:`load`). ``model_repo`` may be a Hub repo id or a local
    directory holding an MLX-layout checkpoint (``mlx_lm.load`` accepts
    both); for a local directory the factory passes ``revision=None`` and
    a ``model_id`` of ``<data-root-relative dir>@<upstream sha>`` so
    provenance is recorded without the absolute path — by default
    ``model_id`` is ``<model_repo>@<revision or main>``.

    ``batch_size`` caps the pairs of one forward pass and
    ``max_batch_tokens`` its padded tokens; ``max_length`` bounds a whole
    row (prefix + body + suffix) and ``doc_max_tokens`` (``None``: no cap
    beyond ``max_length``) the document part of the body.
    ``reuse_prefix`` reads the tokens all of a query's rows share once
    per query (module docstring, "Shared prefix"); the batch bounds then
    apply to what is left of each row. ``cache_limit_bytes`` bounds MLX's
    process-wide buffer cache when the weights load, keeping a tighter
    bound already in place (:func:`imsg.mlx_runtime.bound_buffer_cache`;
    ``None`` leaves it).
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        *,
        instruction: str | None = None,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        max_length: int = 8192,
        max_batch_tokens: int = DEFAULT_RERANK_MAX_BATCH_TOKENS,
        doc_max_tokens: int | None = None,
        reuse_prefix: bool = DEFAULT_RERANK_REUSE_PREFIX,
        cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT_BYTES,
        model_id: str | None = None,
    ) -> None:
        if not model_repo:
            raise ValueError("model_repo must be a non-empty repo id or local path")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        if max_batch_tokens < 1:
            raise ValueError(f"max_batch_tokens must be >= 1, got {max_batch_tokens}")
        if cache_limit_bytes is not None and cache_limit_bytes < 0:
            raise ValueError(f"cache_limit_bytes must be >= 0 or None, got {cache_limit_bytes}")
        if model_id is not None and not model_id.strip():
            raise ValueError("model_id, when given, must be a non-empty string")
        self.model_id = model_id or format_model_id(model_repo, revision)
        self.instruction = instruction or DEFAULT_RERANK_INSTRUCTION
        self._model_repo = model_repo
        self._revision = revision
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_batch_tokens = max_batch_tokens
        self._doc_max_tokens = _checked_doc_max_tokens(doc_max_tokens)
        self._reuse_prefix = bool(reuse_prefix)
        self._cache_limit_bytes = cache_limit_bytes
        self._model: Any = None
        self._tokenizer: Any = None
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        self._yes_id = -1
        self._no_id = -1
        self._pad_id = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def body_token_budget(self) -> int:
        """Tokens available to the pair body once the chat prefix/suffix
        are accounted for (the reference's ``max_length - len(prefix) -
        len(suffix)``). Only meaningful after :meth:`load`."""
        return self._max_length - len(self._prefix_ids) - len(self._suffix_ids)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_batch_tokens(self) -> int:
        return self._max_batch_tokens

    @property
    def reuse_prefix(self) -> bool:
        return self._reuse_prefix

    @property
    def doc_max_tokens(self) -> int | None:
        """The document-token cap (``None``: only ``max_length`` bounds a
        pair). Settable, so one loaded model can be measured under several
        caps; it takes effect on the next :meth:`score`."""
        return self._doc_max_tokens

    @doc_max_tokens.setter
    def doc_max_tokens(self, value: int | None) -> None:
        self._doc_max_tokens = _checked_doc_max_tokens(value)

    def load(self) -> None:
        """Load weights + tokenizer and resolve the fixed prompt tokens
        (idempotent)."""
        if self._model is not None:
            return
        model, tokenizer = load_model_and_tokenizer(self._model_repo, self._revision)
        prefix_ids = [int(t) for t in tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)]
        suffix_ids = [int(t) for t in tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)]
        budget = self._max_length - len(prefix_ids) - len(suffix_ids)
        if budget < 1:
            raise MlxRuntimeError(
                f"max_length {self._max_length} is smaller than the reranker's fixed chat "
                f"prefix+suffix ({len(prefix_ids) + len(suffix_ids)} tokens)"
            )
        yes_id = _single_token_id(tokenizer, YES_TOKEN)
        no_id = _single_token_id(tokenizer, NO_TOKEN)
        if yes_id == no_id:
            raise MlxRuntimeError("tokenizer maps 'yes' and 'no' to the same id")
        pad = getattr(tokenizer, "pad_token_id", None)
        self._pad_id = pad if isinstance(pad, int) and not isinstance(pad, bool) else suffix_ids[-1]
        self._prefix_ids = prefix_ids
        self._suffix_ids = suffix_ids
        self._yes_id = yes_id
        self._no_id = no_id
        self._model = model
        self._tokenizer = tokenizer
        if self._cache_limit_bytes is not None:
            bound_buffer_cache(self._cache_limit_bytes)

    def unload(self) -> None:
        """Drop the weights and tokenizer (idempotent); the next call that
        needs them loads them again. Dropping the references is all this
        does — the caller returns the freed memory to the system
        (`imsg.retrieval.idle_unload.release_freed_memory`), because MLX
        keeps freed buffers in its own cache until told otherwise."""
        self._model = None
        self._tokenizer = None

    def score(self, query: str, documents: list[str]) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        rows = self.token_rows(query, docs)
        plan = self.plan(rows)
        shared = plan.shared_prefix
        prefix_cache = self._read_prefix(rows[0][:shared]) if shared else None
        scores: list[float | None] = [None] * len(rows)
        for group in plan.batches:
            if prefix_cache is None:
                batch_scores = self.score_token_rows([rows[i] for i in group])
            else:
                batch_scores = self._score_after_prefix(
                    prefix_cache, [rows[i][shared:] for i in group]
                )
            for index, value in zip(group, batch_scores, strict=True):
                scores[index] = value
        out = [s for s in scores if s is not None]
        if len(out) != len(docs):
            raise MlxRuntimeError(
                f"scored {len(out)} documents out of {len(docs)} (internal batching bug)"
            )
        return out

    def plan(self, rows: Sequence[Sequence[int]]) -> RerankPlan:
        """The passes :meth:`score` makes for ``rows`` (as
        :meth:`token_rows` builds them). With ``reuse_prefix`` and two or
        more rows, the tokens they all share are read in a pass of their
        own; a single row has nothing to share it with and is read whole.
        The remaining tokens of each row are packed by
        :func:`imsg.embed.batching.plan_batches` under the batch bounds:
        longest first, at most ``batch_size`` rows and ``max_batch_tokens``
        padded tokens per pass, a row over the token bound alone."""
        lengths = tuple(len(row) for row in rows)
        shared = shared_prefix_length(rows) if self._reuse_prefix and len(rows) > 1 else 0
        batches = plan_batches(
            [n - shared for n in lengths],
            max_batch_size=self._batch_size,
            max_batch_tokens=self._max_batch_tokens,
        )
        return RerankPlan(shared, tuple(tuple(batch) for batch in batches), lengths)

    def token_rows(self, query: str, documents: Sequence[str]) -> list[list[int]]:
        """The model input for each (query, document) pair, in order:
        ``prefix + body + suffix`` with the body truncated to the budget
        ``max_length`` leaves and its document part to ``doc_max_tokens``.

        The body is tokenized whole, as the reference does, so without a
        document cap every row is exactly the reference's. The document
        part is whatever follows the tokens the body shares with the same
        pair rendered around an empty document; for the Qwen3 tokenizer
        that boundary is the token after ``"<Document>:"``. A tokenizer
        that merged across it would only move the boundary earlier, so
        the cap would count a few of the pair's own tokens as document —
        a row is never longer than the cap allows."""
        self.load()
        cap = self._doc_max_tokens
        head_ids = (
            self._encode(format_reranker_pair(self.instruction, query, ""))
            if cap is not None
            else []
        )
        rows: list[list[int]] = []
        for document in documents:
            body_ids = self._encode(format_reranker_pair(self.instruction, query, document))
            keep = self.body_token_budget
            if cap is not None:
                keep = min(keep, common_prefix_length(head_ids, body_ids) + cap)
            rows.append([*self._prefix_ids, *body_ids[:keep], *self._suffix_ids])
        return rows

    def score_token_rows(self, rows: Sequence[Sequence[int]]) -> list[float]:
        """P(yes) for each already-assembled row (see :meth:`token_rows`),
        in order, from one right-padded forward pass over the whole rows —
        the model card's computation, and what :meth:`score` runs without
        ``reuse_prefix``. This runs exactly the batch it is given."""
        self.load()
        mx = import_mlx_core()
        padded, lengths = right_pad(rows, self._pad_id)
        hidden = base_transformer_hidden_states(self._model, mx.array(padded))
        return self._yes_probabilities(mx, hidden, lengths)

    def _read_prefix(self, prefix: Sequence[int]) -> list[tuple[Any, Any]]:
        """Run ``prefix`` through the model once and return each layer's
        keys and values for it, computed, as ``(keys, values)`` of shape
        ``(1, kv_heads, len(prefix), head_dim)``."""
        mx = import_mlx_core()
        caches = import_mlx_lm_cache().make_prompt_cache(self._model)
        base_transformer_hidden_states(self._model, mx.array([list(prefix)]), cache=caches)
        state = [tuple(cache.state) for cache in caches]
        mx.eval([array for pair in state for array in pair])
        return [(keys, values) for keys, values in state]

    def _score_after_prefix(
        self, prefix_cache: Sequence[tuple[Any, Any]], rests: Sequence[Sequence[int]]
    ) -> list[float]:
        """P(yes) for rows whose shared prefix :meth:`_read_prefix` has
        read, from one right-padded forward pass over the rest of each row
        (``rests``, in order). Every row of the batch gets its own copy of
        the prefix's keys and values, so it continues from the prefix's
        last position exactly as if it had been read whole; right padding
        follows each row's real tokens, which never attend to it."""
        mx = import_mlx_core()
        caches = import_mlx_lm_cache().make_prompt_cache(self._model)
        if len(caches) != len(prefix_cache):
            raise MlxRuntimeError(
                f"the model has {len(caches)} layer caches but the prefix was read into "
                f"{len(prefix_cache)}"
            )
        rows = len(rests)
        for cache, (keys, values) in zip(caches, prefix_cache, strict=True):
            cache.state = (mx.repeat(keys, rows, axis=0), mx.repeat(values, rows, axis=0))
        padded, lengths = right_pad(rests, self._pad_id)
        hidden = base_transformer_hidden_states(self._model, mx.array(padded), cache=caches)
        return self._yes_probabilities(mx, hidden, lengths)

    def _yes_probabilities(self, mx: Any, hidden: Any, lengths: Sequence[int]) -> list[float]:
        """P(yes) per row from its last real token's hidden state."""
        last = gather_last_token_states(mx, hidden, lengths)  # (batch, 1, hidden)
        logits = lm_head_logits(self._model, last)  # (batch, 1, vocab)
        pair = mx.take(logits, mx.array([self._no_id, self._yes_id]), axis=-1)  # (batch, 1, 2)
        out: list[float] = []
        for row in float_rows(mx, pair):
            if len(row) != 2:
                raise MlxRuntimeError(f"expected [no, yes] logits per document, got {len(row)}")
            out.append(yes_probability(no_logit=row[0], yes_logit=row[1]))
        if len(out) != len(lengths):
            raise MlxRuntimeError(f"scored {len(out)} rows out of {len(lengths)} in one batch")
        return out

    def _encode(self, text: str) -> list[int]:
        return [int(t) for t in self._tokenizer.encode(text, add_special_tokens=False)]


def _checked_doc_max_tokens(value: int | None) -> int | None:
    if value is not None and (isinstance(value, bool) or value < 1):
        raise ValueError(f"doc_max_tokens must be >= 1 or None, got {value}")
    return value


__all__ = [
    "DEFAULT_RERANK_BATCH_SIZE",
    "DEFAULT_RERANK_INSTRUCTION",
    "DEFAULT_RERANK_MAX_BATCH_TOKENS",
    "DEFAULT_RERANK_REUSE_PREFIX",
    "NO_TOKEN",
    "PAIR_TEMPLATE",
    "RERANKER_PREFIX",
    "RERANKER_SUFFIX",
    "RERANKER_SYSTEM_PROMPT",
    "YES_TOKEN",
    "MlxRerankerProvider",
    "RerankPlan",
    "common_prefix_length",
    "format_reranker_pair",
    "shared_prefix_length",
    "yes_probability",
]
