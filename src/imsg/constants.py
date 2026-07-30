"""Single-source-of-truth constants shared between config validation, the
migration DDL, and the DDL lint.

These numbers are asserted against the actual migration SQL text in
``tests/test_ddl_lint.py`` — if migration 0001/0002 ever changes a
dimension, that test fails until this module is updated too, so config
validation and the schema cannot silently drift apart.

pgvector dimension caps (empirically verified against pgvector 0.8.6,
2026-07-30, against a live scratch instance — not merely the docs):

- The ``vector``/``halfvec`` column *types* allow up to 16,000 dims —
  ``CREATE TABLE`` with a wide column succeeds either way.
- An HNSW or IVFFlat *index* on ``vector`` caps at 2,000 dims.
- An HNSW or IVFFlat *index* on ``halfvec`` caps at 4,000 dims.

A column can therefore be perfectly legal DDL whose ANN index can never
be created — the error surfaces at ``CREATE INDEX``, not at column
definition, so an oversight here degrades silently to sequential scan
rather than failing loudly. That exact bug (a ``halfvec(4096)`` column
paired with an HNSW index) was the blocker this spec revision fixed;
see ``scripts/lint_ddl.py``.
"""

from __future__ import annotations

# --- Model-pinned embedding dimensions actually used by this build ---

PRIMARY_EMBEDDING_DIM = 2048
"""Qwen3-Embedding-8B MRL output dim, as hardcoded in migration 0001's
``segment_embedding``/``attachment_chunk_embedding`` CHECK constraints."""

MULTIMODAL_EMBEDDING_DIM = 1280
"""PE-Core-G14-448 output dim, as hardcoded in migration 0002's
``attachment_mm_embedding`` CHECK constraint."""

# --- pgvector index/type caps (empirically verified, pgvector 0.8.6) ---

VECTOR_TYPE_MAX_DIM = 16_000
"""Max dimension the ``vector`` column type itself will accept."""

HALFVEC_TYPE_MAX_DIM = 16_000
"""Max dimension the ``halfvec`` column type itself will accept."""

VECTOR_INDEX_MAX_DIM = 2_000
"""Max dimension for an HNSW or IVFFlat index on a ``vector`` column."""

HALFVEC_INDEX_MAX_DIM = 4_000
"""Max dimension for an HNSW or IVFFlat index on a ``halfvec`` column."""

__all__ = [
    "HALFVEC_INDEX_MAX_DIM",
    "HALFVEC_TYPE_MAX_DIM",
    "MULTIMODAL_EMBEDDING_DIM",
    "PRIMARY_EMBEDDING_DIM",
    "VECTOR_INDEX_MAX_DIM",
    "VECTOR_TYPE_MAX_DIM",
]
