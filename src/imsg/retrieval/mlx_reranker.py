"""Qwen3-Reranker via MLX — the real :class:`~imsg.retrieval.reranker.
RerankerProvider` (SPEC §4.1, §9.4 step 7: "Rerank top ``rerank_top``
with Qwen3-Reranker-8B on (query, rendered_text) pairs").

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
"""

from __future__ import annotations

import math
from typing import Any

from imsg.mlx_runtime import (
    MlxRuntimeError,
    base_transformer_hidden_states,
    batched,
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
    """Qwen3-Reranker-8B (or any Qwen3-Reranker size) through ``mlx_lm``.

    Constructor arguments are plain values (the CLI reads
    ``retrieval.reranker_model`` / ``retrieval.reranker_revision`` and
    passes them here); weights load lazily on the first :meth:`score`
    (or explicitly via :meth:`load`).
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        *,
        instruction: str | None = None,
        batch_size: int = 8,
        max_length: int = 8192,
    ) -> None:
        if not model_repo:
            raise ValueError("model_repo must be a non-empty repo id or local path")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        self.model_id = format_model_id(model_repo, revision)
        self.instruction = instruction or DEFAULT_RERANK_INSTRUCTION
        self._model_repo = model_repo
        self._revision = revision
        self._batch_size = batch_size
        self._max_length = max_length
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

    def score(self, query: str, documents: list[str]) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        self.load()
        mx = import_mlx_core()
        scores: list[float] = []
        for batch in batched(docs, self._batch_size):
            scores.extend(self._score_batch(mx, query, batch))
        if len(scores) != len(docs):
            raise MlxRuntimeError(
                f"scored {len(scores)} documents out of {len(docs)} (internal batching bug)"
            )
        return scores

    def _encode_pair(self, query: str, document: str) -> list[int]:
        body = format_reranker_pair(self.instruction, query, document)
        body_ids = [int(t) for t in self._tokenizer.encode(body, add_special_tokens=False)]
        return [*self._prefix_ids, *body_ids[: self.body_token_budget], *self._suffix_ids]

    def _score_batch(self, mx: Any, query: str, documents: list[str]) -> list[float]:
        rows = [self._encode_pair(query, doc) for doc in documents]
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
        return out


__all__ = [
    "DEFAULT_RERANK_INSTRUCTION",
    "NO_TOKEN",
    "PAIR_TEMPLATE",
    "RERANKER_PREFIX",
    "RERANKER_SUFFIX",
    "RERANKER_SYSTEM_PROMPT",
    "YES_TOKEN",
    "MlxRerankerProvider",
    "format_reranker_pair",
    "yes_probability",
]
