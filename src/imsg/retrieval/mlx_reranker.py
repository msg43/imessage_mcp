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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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

DEFAULT_RERANK_MAX_BATCH_TOKENS = 1024
"""Most padded tokens (``rows x longest row``) in one forward pass.

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
    ``cache_limit_bytes`` bounds MLX's process-wide buffer cache when the
    weights load, keeping a tighter bound already in place
    (:func:`imsg.mlx_runtime.bound_buffer_cache`; ``None`` leaves it).
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

    def score(self, query: str, documents: list[str]) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        rows = self.token_rows(query, docs)
        plan = plan_batches(
            [len(row) for row in rows],
            max_batch_size=self._batch_size,
            max_batch_tokens=self._max_batch_tokens,
        )
        scores: list[float | None] = [None] * len(rows)
        for group in plan:
            batch_scores = self.score_token_rows([rows[i] for i in group])
            for index, value in zip(group, batch_scores, strict=True):
                scores[index] = value
        out = [s for s in scores if s is not None]
        if len(out) != len(docs):
            raise MlxRuntimeError(
                f"scored {len(out)} documents out of {len(docs)} (internal batching bug)"
            )
        return out

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
        in order, from one right-padded forward pass. :meth:`score` plans
        the batches; this runs exactly the batch it is given."""
        self.load()
        mx = import_mlx_core()
        padded, lengths = right_pad(rows, self._pad_id)
        hidden = base_transformer_hidden_states(self._model, mx.array(padded))
        last = gather_last_token_states(mx, hidden, lengths)  # (batch, 1, hidden)
        logits = lm_head_logits(self._model, last)  # (batch, 1, vocab)
        pair = mx.take(logits, mx.array([self._no_id, self._yes_id]), axis=-1)  # (batch, 1, 2)
        out: list[float] = []
        for row in float_rows(mx, pair):
            if len(row) != 2:
                raise MlxRuntimeError(f"expected [no, yes] logits per document, got {len(row)}")
            out.append(yes_probability(no_logit=row[0], yes_logit=row[1]))
        if len(out) != len(rows):
            raise MlxRuntimeError(f"scored {len(out)} rows out of {len(rows)} in one batch")
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
    "NO_TOKEN",
    "PAIR_TEMPLATE",
    "RERANKER_PREFIX",
    "RERANKER_SUFFIX",
    "RERANKER_SYSTEM_PROMPT",
    "YES_TOKEN",
    "MlxRerankerProvider",
    "common_prefix_length",
    "format_reranker_pair",
    "yes_probability",
]
