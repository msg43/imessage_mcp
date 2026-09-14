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

from imsg.errors import EmbeddingError
from imsg.mlx_runtime import (
    base_transformer_hidden_states,
    batched,
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
    ) -> None:
        if not model_repo:
            raise ValueError("model_repo must be a non-empty repo id or local path")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_length < 2:
            raise ValueError(f"max_length must be >= 2 (one text token plus EOS), got {max_length}")
        self.model_id = format_model_id(model_repo, revision)
        self.dim = dim
        self._model_repo = model_repo
        self._revision = revision
        self._batch_size = batch_size
        self._max_length = max_length
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
        model, tokenizer = load_model_and_tokenizer(self._model_repo, self._revision)
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        return self._embed([format_query_text(instruction, text)])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.load()
        mx = import_mlx_core()
        vectors: list[list[float]] = []
        for batch in batched(texts, self._batch_size):
            vectors.extend(self._embed_batch(mx, batch))
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedded {len(vectors)} vectors for {len(texts)} texts (internal batching bug)"
            )
        return vectors

    def _embed_batch(self, mx: Any, texts: list[str]) -> list[list[float]]:
        rows = [
            tokenize_for_embedding(
                self._tokenizer, text, max_length=self._max_length, eos_suffix=self._eos_suffix
            )
            for text in texts
        ]
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
    "QUERY_TEMPLATE",
    "MlxTextEmbeddingProvider",
    "eos_suffix_for",
    "format_query_text",
    "matryoshka_normalize",
    "pad_token_id_for",
    "tokenize_for_embedding",
]
