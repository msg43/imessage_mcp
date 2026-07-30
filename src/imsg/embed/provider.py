"""Embedding provider abstraction (SPEC §6 `embedding.*`, §8 S6, §4.1).

Two roles:

- `TextEmbeddingProvider` — the primary text embedder (Qwen3-Embedding-
  8B at Phase 3/5, D3 ratified local-only). Instruction-asymmetric
  (Qwen3's own scheme): documents are embedded bare; a query is
  embedded with `embedding.query_instruction` prefixed *at query time
  only* — a retrieval-time concern, but the method split lives on the
  provider so the real backend implements the asymmetry correctly from
  day one rather than bolting it on later.
- `MultimodalEmbeddingProvider` — the secondary vector (PE-Core-G14-448,
  D3a ratified), embedding raw attachment bytes (images, sampled video
  keyframes) into the same visual space §9.4 channel C searches.

No hosted API implementation exists or will be added here (D3/D3a:
local-only for the full corpus — see SPEC §4.1's egress table). No
model weights exist in this build environment either way.
`FakeTextEmbeddingProvider` / `FakeMultimodalEmbeddingProvider` are
deterministic, dependency-free stand-ins every test in this build uses:
they produce vectors of the exact configured dimension, L2-normalized,
so cosine-distance math and pgvector's `halfvec` CHECK constraints
behave exactly as they would against the real models. The real
MLX-backed implementations are Phase 3/5 work and drop in behind these
same Protocols — nothing above this layer (the pipeline, retrieval)
should need to change.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol


class TextEmbeddingProvider(Protocol):
    model_id: str
    """e.g. `'Qwen/Qwen3-Embedding-8B@<revision>'` — recorded verbatim
    into `segment_embedding.model` / `attachment_chunk_embedding.model`."""
    dim: int
    """Must equal `imsg.constants.PRIMARY_EMBEDDING_DIM` (2048)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document-side text bare (no instruction prefix) —
        called on already-`imsg.textnorm.normalize_text`-normalized
        text. Returns one L2-normalized vector per input, same order,
        each exactly `self.dim` long."""
        ...

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        """Embed a query, prefixed per Qwen3's instruction-asymmetric
        scheme (retrieval-service concern; included here so the real
        provider gets the asymmetry right from the start)."""
        ...


class MultimodalEmbeddingProvider(Protocol):
    model_id: str
    """e.g. `'facebook/PE-Core-G14-448@<revision>'`."""
    dim: int
    """Must equal `imsg.constants.MULTIMODAL_EMBEDDING_DIM` (1280)."""

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        """One L2-normalized vector per input image path, same order."""
        ...


def _deterministic_vector(seed: str, dim: int) -> list[float]:
    """A deterministic, seed-derived unit vector of length `dim`. Not
    remotely a real embedding — exercises the pipeline's idempotency
    (`text_sha256`/`media_sha256` skip-if-unchanged), shape, and
    dimension contracts without model weights. The same seed always
    yields the same vector, which is exactly what the idempotency
    tests need: re-embedding unchanged content must be a no-op.
    """
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        for i in range(0, len(digest), 8):
            if len(values) >= dim:
                break
            chunk = digest[i : i + 8]
            as_int = int.from_bytes(chunk, "big")
            values.append((as_int / 2**64) * 2 - 1)  # in [-1, 1)
        counter += 1
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:  # astronomically unlikely, but keep the contract airtight
        values[0] = 1.0
        norm = 1.0
    return [v / norm for v in values]


class FakeTextEmbeddingProvider:
    model_id = "fake/text-embedder@test"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_deterministic_vector(f"doc:{t}", self.dim) for t in texts]

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        return _deterministic_vector(f"query:{instruction}:{text}", self.dim)


class FakeMultimodalEmbeddingProvider:
    model_id = "fake/multimodal-embedder@test"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [_deterministic_vector(f"img:{p}", self.dim) for p in image_paths]


__all__ = [
    "FakeMultimodalEmbeddingProvider",
    "FakeTextEmbeddingProvider",
    "MultimodalEmbeddingProvider",
    "TextEmbeddingProvider",
]
