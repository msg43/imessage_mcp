"""Reranker provider abstraction (SPEC §4.1, §9.4 step 7):
"Rerank top `rerank_top` (50) with Qwen3-Reranker-8B on (query,
rendered_text) pairs."

Same shape as `imsg.embed.provider`'s embedding Protocols (this build's
established pattern): a `Protocol` the real MLX-backed model drops in
behind at Phase 3/5, plus a deterministic `Fake*` every test in this
build uses. No reranker code existed anywhere in the codebase before
this build (the indexing agents' scope was embedding + FTS, not
reranking), so this module — unlike `imsg.embed.provider` — is
authored fresh here, not extended.
"""

from __future__ import annotations

import re
from typing import Protocol


class RerankerProvider(Protocol):
    model_id: str
    """e.g. `'Qwen/Qwen3-Reranker-8B@<revision>'`."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """One relevance score per document, same order as `documents`.
        Higher is more relevant; scores are only meaningful relative to
        each other within one call (not calibrated across calls or
        comparable to RRF scores — SPEC §9.4 step 6's "raw ... scores
        are never compared across corpora" applies here too: the
        reranker replaces RRF ordering for the reranked pool, it does
        not blend with it)."""
        ...


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


class FakeRerankerProvider:
    """Deterministic stand-in: a Jaccard-similarity lexical-overlap
    score between the query's tokens and each document's tokens. Not a
    real reranker, but — unlike a pure hash-based fake — it actually
    orders more-relevant-looking documents higher, which is what
    exercises the reranking *plumbing* (score -> reorder -> truncate to
    `limit`) meaningfully in tests without model weights. The real
    Qwen3-Reranker-8B backend is Phase 3/5 work and drops in behind the
    same Protocol."""

    model_id = "fake/reranker@test"

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return [0.0] * len(documents)
        scores = []
        for doc in documents:
            doc_tokens = _tokens(doc)
            if not doc_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            scores.append(overlap / union if union else 0.0)
        return scores


__all__ = ["FakeRerankerProvider", "RerankerProvider"]
