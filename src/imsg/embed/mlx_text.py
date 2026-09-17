"""Qwen3-Embedding via MLX — the real :class:`~imsg.embed.provider.
TextEmbeddingProvider` (SPEC §4.1, §6 ``embedding.*``, §8 S6; D3
ratified local-only).

What this reproduces from the published model (verified against the
model card, the reference Transformers/Sentence-Transformers configs and
the tokenizer files, not from memory):

- **Token layout.** The tokenizer's post-processor appends a single EOS
  special token (``<|endoftext|>``, id 151643, for ``Qwen3-Embedding-*``)
  to every sequence, which is what every reference path (Transformers,
  Sentence-Transformers, vLLM, TEI) feeds the model. The suffix is
  derived here at load time from what the tokenizer itself adds
  (``encode("", add_special_tokens=True)``), so a conversion that ships
  the same tokenizer files gets exactly the same ids; a tokenizer with
  no such template falls back to its ``eos_token_id``. Text is
  truncated so the whole row fits ``max_length`` *including* the suffix.
- **Pooling.** Last-token pooling (``1_Pooling/config.json``:
  ``pooling_mode_lasttoken``) over the base transformer's last-layer
  hidden states — never the LM head — read at each sequence's last real
  token (see ``imsg.mlx_runtime`` on right padding).
- **Output.** Matryoshka truncation to the first ``dim`` components
  (the card: "user-defined output dimensions ranging from 32 to 4096",
  applied as ``output[:, :dim]`` in the reference code) followed by L2
  normalisation (``2_Normalize``). ``dim`` is pinned to
  ``PRIMARY_EMBEDDING_DIM`` (2048) by config validation; the provider
  refuses a model whose hidden width is smaller than ``dim``.
- **Instruction asymmetry.** Queries are embedded as
  ``"Instruct: {instruction}\\nQuery: {text}"``; documents bare.

Weights are loaded lazily on first use (or via :meth:`load`) so
constructing the provider — which the CLI does at startup — costs
nothing, and importing this module never requires ``mlx``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from imsg.embed.batching import DEFAULT_MAX_BATCH_TOKENS, plan_batches
from imsg.errors import EmbeddingError
from imsg.mlx_runtime import (
    DEFAULT_CACHE_LIMIT_BYTES,
    base_transformer_hidden_states,
    bound_buffer_cache,
    float_rows,
    format_model_id,
    gather_last_token_states,
    hidden_size_of,
    import_mlx_core,
    load_model_and_tokenizer,
    right_pad,
)

QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"
"""Qwen3-Embedding's instruction-aware query format."""

EMBEDDING_MODEL_CONFIG: dict[str, Any] = {"tie_word_embeddings": True}
"""Config override handed to ``mlx_lm.load``. The pinned mlx-embeddings
conversion (``mlx-community/Qwen3-Embedding-8B-mxfp8``) ships no
``lm_head.*`` tensor at all — read from its safetensors headers on
2026-09-14 (651 tensors; its ``model.safetensors.index.json`` does not
describe the shards) — while its ``config.json`` says
``tie_word_embeddings: false``. ``mlx_lm``'s Qwen3 ``Model`` allocates an
``lm_head`` whenever the config says untied, and its strict weight load
then fails with ``Missing 1 parameters: lm_head.weight``. This provider
never calls the head (it reads the base transformer's hidden states), so
it declares the head tied: ``mlx_lm`` then allocates none and every
tensor the checkpoint does ship is loaded. A conversion that *does* ship
a head still loads — ``mlx_lm``'s ``sanitize`` drops ``lm_head.weight``
for a tied model — and the reranker, which needs the head, does not use
this override."""


def format_query_text(instruction: str, text: str) -> str:
    """Render a query the way Qwen3-Embedding expects at query time only
    (documents are embedded bare — the instruction-asymmetric scheme)."""
    return QUERY_TEMPLATE.format(instruction=instruction, text=text)


def eos_suffix_for(tokenizer: Any) -> list[int]:
    """The special-token suffix the tokenizer itself appends to a bare
    sequence (its post-processor template), falling back to
    ``eos_token_id`` when it appends nothing."""
    added = [int(t) for t in tokenizer.encode("", add_special_tokens=True)]
    if added:
        return added
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and not isinstance(eos, bool):
        return [eos]
    raise EmbeddingError(
        "tokenizer neither appends special tokens nor defines eos_token_id; cannot build "
        "Qwen3-Embedding's `<text><eos>` input layout"
    )


def pad_token_id_for(tokenizer: Any, fallback: int) -> int:
    """``pad_token_id`` when the tokenizer defines one, else ``fallback``.
    The value never influences a real token under right padding (see
    ``imsg.mlx_runtime``), so any in-vocabulary id is correct."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad, int) and not isinstance(pad, bool):
        return pad
    return fallback


def tokenize_for_embedding(
    tokenizer: Any, text: str, *, max_length: int, eos_suffix: Sequence[int]
) -> list[int]:
    """``encode(text)`` without special tokens, truncated so that text plus
    the EOS suffix fits ``max_length``, then the suffix appended."""
    budget = max_length - len(eos_suffix)
    if budget < 1:
        raise EmbeddingError(
            f"max_length {max_length} leaves no room for text next to the "
            f"{len(eos_suffix)}-token EOS suffix"
        )
    ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    return [*ids[:budget], *eos_suffix]


def matryoshka_normalize(vector: Sequence[float], dim: int) -> list[float]:
    """Keep the first ``dim`` components (MRL truncation) and L2-normalise
    them. Raises :class:`EmbeddingError` if the vector is too short or has
    no finite, non-zero norm — a NaN/zero vector must never reach
    pgvector silently."""
    if dim < 1:
        raise EmbeddingError(f"embedding dim must be >= 1, got {dim}")
    if len(vector) < dim:
        raise EmbeddingError(
            f"hidden state has {len(vector)} components, fewer than the requested dim {dim}"
        )
    head = [float(v) for v in vector[:dim]]
    norm = math.sqrt(math.fsum(v * v for v in head))
    if not math.isfinite(norm) or norm == 0.0:
        raise EmbeddingError("cannot L2-normalise a zero or non-finite embedding vector")
    return [v / norm for v in head]


class MlxTextEmbeddingProvider:
    """Qwen3-Embedding-8B (or any Qwen3-Embedding size) through ``mlx_lm``.

    ``model_repo`` must be an MLX-layout checkpoint (``mlx-community/*``
    conversions or the output of ``mlx_lm.convert``); the raw
    ``Qwen/Qwen3-Embedding-8B`` repo ships its weights without the
    ``model.`` prefix ``mlx_lm`` expects and does not load.
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        dim: int,
        *,
        batch_size: int = 32,
        max_length: int = 8192,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
        cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT_BYTES,
        model_id: str | None = None,
    ) -> None:
        """``batch_size`` caps the rows of one forward pass and
        ``max_batch_tokens`` its padded tokens (``rows x longest row``);
        :func:`imsg.embed.batching.plan_batches` packs every
        ``embed_documents`` call under both, longest rows first, and the
        results come back in input order. ``cache_limit_bytes`` bounds
        MLX's process-wide buffer cache once the weights load, keeping a
        tighter bound already in place (``imsg.mlx_runtime.
        bound_buffer_cache``; ``None`` leaves the cache alone).
        ``model_id`` overrides the recorded ``'<repo>@<revision>'`` — a
        local conversion passes its data-root-relative directory plus the
        upstream sha."""
        if not model_repo:
            raise ValueError("model_repo must be a non-empty repo id or local path")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_length < 2:
            raise ValueError(f"max_length must be >= 2 (one text token plus EOS), got {max_length}")
        if max_batch_tokens < 1:
            raise ValueError(f"max_batch_tokens must be >= 1, got {max_batch_tokens}")
        if cache_limit_bytes is not None and cache_limit_bytes < 0:
            raise ValueError(f"cache_limit_bytes must be >= 0 or None, got {cache_limit_bytes}")
        self.model_id = model_id or format_model_id(model_repo, revision)
        self.dim = dim
        self._model_repo = model_repo
        self._revision = revision
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_batch_tokens = max_batch_tokens
        self._cache_limit_bytes = cache_limit_bytes
        self._model: Any = None
        self._tokenizer: Any = None
        self._eos_suffix: list[int] = []
        self._pad_id = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load weights + tokenizer now (idempotent). Called implicitly by
        the first embed; exposed so a caller can pay the cost — and hit
        any load error — at startup rather than mid-pipeline."""
        if self._model is not None:
            return
        model, tokenizer = load_model_and_tokenizer(
            self._model_repo, self._revision, model_config=EMBEDDING_MODEL_CONFIG
        )
        hidden = hidden_size_of(model)
        if hidden < self.dim:
            raise EmbeddingError(
                f"{self.model_id} has hidden size {hidden}, smaller than the requested "
                f"embedding dim {self.dim}; Matryoshka truncation cannot widen a vector"
            )
        suffix = eos_suffix_for(tokenizer)
        if len(suffix) >= self._max_length:
            raise EmbeddingError(
                f"max_length {self._max_length} leaves no room for text next to the "
                f"{len(suffix)}-token EOS suffix"
            )
        self._eos_suffix = suffix
        self._pad_id = pad_token_id_for(tokenizer, suffix[-1])
        self._model = model
        self._tokenizer = tokenizer
        if self._cache_limit_bytes is not None:
            bound_buffer_cache(self._cache_limit_bytes)

    @property
    def tokenizer(self) -> Any:
        """The loaded model's tokenizer (loads the weights on first use)."""
        self.load()
        return self._tokenizer

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_batch_tokens(self) -> int:
        return self._max_batch_tokens

    def token_length(self, text: str) -> int:
        """How many tokens ``text`` occupies in a batch row — after
        truncation to ``max_length``, EOS suffix included — i.e. the
        length :func:`imsg.embed.batching.plan_batches` budgets."""
        self.load()
        return len(self._tokenize(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        return self._embed([format_query_text(instruction, text)])[0]

    def _tokenize(self, text: str) -> list[int]:
        return tokenize_for_embedding(
            self._tokenizer, text, max_length=self._max_length, eos_suffix=self._eos_suffix
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.load()
        mx = import_mlx_core()
        rows = [self._tokenize(text) for text in texts]
        plan = plan_batches(
            [len(row) for row in rows],
            max_batch_size=self._batch_size,
            max_batch_tokens=self._max_batch_tokens,
        )
        vectors: list[list[float] | None] = [None] * len(texts)
        for group in plan:
            for index, vector in zip(
                group, self._embed_rows(mx, [rows[i] for i in group]), strict=True
            ):
                vectors[index] = vector
        out = [v for v in vectors if v is not None]
        if len(out) != len(texts):
            raise EmbeddingError(
                f"embedded {len(out)} vectors for {len(texts)} texts (internal batching bug)"
            )
        return out

    def _embed_rows(self, mx: Any, rows: list[list[int]]) -> list[list[float]]:
        padded, lengths = right_pad(rows, self._pad_id)
        hidden = base_transformer_hidden_states(self._model, mx.array(padded))
        pooled = float_rows(mx, gather_last_token_states(mx, hidden, lengths))
        out: list[list[float]] = []
        for vector in pooled:
            normalized = matryoshka_normalize(vector, self.dim)
            if len(normalized) != self.dim:
                raise EmbeddingError(
                    f"produced a {len(normalized)}-dim vector, expected {self.dim}"
                )
            out.append(normalized)
        return out


__all__ = [
    "DEFAULT_CACHE_LIMIT_BYTES",
    "DEFAULT_MAX_BATCH_TOKENS",
    "QUERY_TEMPLATE",
    "MlxTextEmbeddingProvider",
    "eos_suffix_for",
    "format_query_text",
    "matryoshka_normalize",
    "pad_token_id_for",
    "tokenize_for_embedding",
]
