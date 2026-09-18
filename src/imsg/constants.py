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

# --- Model pins: the config-schema defaults for repo + revision ---
#
# These mirror ``models/manifest.lock.yaml`` — the lock is the source of
# truth (SPEC: "a build MUST NOT silently advance a model ... because
# 'latest' changed"), and ``tests/test_provider_factory.py`` asserts
# these constants equal the lock so the two cannot drift apart. Update
# both in the same commit (``imsg models verify --write`` refreshes the
# lock; this module is edited by hand).

TEXT_EMBEDDING_MODEL_REPO = "mlx-community/Qwen3-Embedding-8B-mxfp8"
TEXT_EMBEDDING_MODEL_REVISION = "51c773b7464b630a6c67b4f75dbd796b658d6236"

RERANKER_MODEL = "models/qwen3-reranker-0.6b-mxfp8-e61197ed"
"""The reranker is pinned as a LOCAL CONVERSION (``models/manifest.lock.yaml``
entry ``qwen3-reranker-0.6b``, ``source: local_conversion``, the active
entry for the ``reranker`` role): this is its ``output_dir``, a directory
relative to ``paths.data_root`` that the recorded ``mlx_lm.convert``
command produces from the upstream repo. ``imsg.providers.factory`` reads a
``retrieval.reranker_model`` value as a local directory when
``<data_root>/<value>`` exists and as a Hugging Face repo id
(``owner/name``) otherwise. Qwen3-Reranker-0.6B replaced the 8B on
2026-09-17 for search's latency budget; the 8B conversion stays in the lock
as ``status: retained`` (not active) for a quality comparison."""
RERANKER_MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
"""For a local conversion, the UPSTREAM commit it was converted from
(``Qwen/Qwen3-Reranker-0.6B``); recorded in the provider's ``model_id`` as
``<output_dir>@<sha>`` so provenance survives the move off the Hub."""

BOUNDARY_MODEL_REPO = "mlx-community/Qwen3.5-35B-A3B-4bit"
BOUNDARY_MODEL_REVISION = "1e20fd8d42056f870933bf98ca6211024744f7ec"

CAPTION_MODEL_REPO = BOUNDARY_MODEL_REPO
"""SPEC §4.1: one local VLM serves both S4 boundary detection and S5b
captioning — same weights, loaded by different runtimes."""
CAPTION_MODEL_REVISION = BOUNDARY_MODEL_REVISION

TRANSCRIPTION_MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
TRANSCRIPTION_MODEL_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"

MULTIMODAL_EMBEDDING_MODEL_REPO = "timm/PE-Core-bigG-14-448"
MULTIMODAL_EMBEDDING_MODEL_REVISION = "17aa0c25addfa14198fa2ff73d845a22d433432e"
"""timm's open_clip-layout remap of Meta's ``facebook/PE-Core-G14-448``
(``a6046680086f67d1f24d4b465a240de0578dfc0b``, Apache-2.0) — the repo
``imsg.embed.pe_core_multimodal`` actually downloads from. A revision is
a commit of exactly one repo, so the pin names the mirror: the provider
maps the canonical id to this mirror, and pinned the other way round it
asked the mirror for the canonical repo's sha, which the mirror cannot
resolve (found 2026-09-14 by resolving both ids against the Hugging
Face API; the canonical repo holds only Meta's own ``.pt`` layout)."""

ENRICHMENT_YIELD_POLL_INTERVAL_SECONDS = 0.25
"""How often an enrichment worker paused behind an in-flight query
re-checks (`imsg.db.enrichment_yield_locks`). Reached only while a query
is actually running — the idle check is one round trip and no sleep — so
it sets how promptly the worker resumes, not what yielding costs."""

ENRICHMENT_YIELD_MAX_PAUSE_SECONDS = 300.0
"""How long that worker waits for one unit of work before proceeding
anyway. Not the crash backstop (a killed MCP server's advisory lock dies
with its database session): the "someone is searching continuously and the
queue still has to drain overnight" backstop, five minutes out of a
six-hour window.

Here rather than in `imsg.db.enrichment_yield_locks` because
`imsg.config.schema` defaults to both values and cannot import from
`imsg.db` — `imsg.db.connection` imports the schema."""

__all__ = [
    "BOUNDARY_MODEL_REPO",
    "BOUNDARY_MODEL_REVISION",
    "CAPTION_MODEL_REPO",
    "CAPTION_MODEL_REVISION",
    "ENRICHMENT_YIELD_MAX_PAUSE_SECONDS",
    "ENRICHMENT_YIELD_POLL_INTERVAL_SECONDS",
    "HALFVEC_INDEX_MAX_DIM",
    "HALFVEC_TYPE_MAX_DIM",
    "MULTIMODAL_EMBEDDING_DIM",
    "MULTIMODAL_EMBEDDING_MODEL_REPO",
    "MULTIMODAL_EMBEDDING_MODEL_REVISION",
    "PRIMARY_EMBEDDING_DIM",
    "RERANKER_MODEL",
    "RERANKER_MODEL_REVISION",
    "TEXT_EMBEDDING_MODEL_REPO",
    "TEXT_EMBEDDING_MODEL_REVISION",
    "TRANSCRIPTION_MODEL_REPO",
    "TRANSCRIPTION_MODEL_REVISION",
    "VECTOR_INDEX_MAX_DIM",
    "VECTOR_TYPE_MAX_DIM",
]
