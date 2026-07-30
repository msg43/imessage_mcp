"""S6 Postgres embedding pipeline (SPEC §8 S6): finds segments and
attachment chunks lacking embeddings or with a stale `text_sha256`,
embeds them via the configured `TextEmbeddingProvider` in batches, and
writes `segment_embedding` / `attachment_chunk_embedding` rows —
per-batch transactions, so a mid-run failure (model load, OOM) leaves
whatever already committed intact rather than a half-written batch
(SPEC failure mode: "model load failure -> abort run, nothing
partial"). Also embeds images/video-keyframes into
`attachment_mm_embedding` when `embedding.multimodal.enabled` (D3a).

Text fed to the provider is always `imsg.textnorm.normalize_text`-
normalized first (SPEC §9.2: normalization applies "before FTS
insertion and embedding"); `text_sha256`/`media_sha256` are computed
over the *raw* (pre-normalization) content, which is simpler and
sufficient — any raw change that could possibly change the normalized
output also changes the raw hash, so staleness detection never misses
a real change; it can only occasionally re-embed unchanged content
where the raw bytes moved but normalization would have collapsed the
difference (harmless, not a correctness issue).

FTS sidecar sync is a separate concern
(`imsg.embed.fts.sync.sync_fts`) — segment/attachment_chunk content
already carries its own `search_index_event` rows from S4/S5b at the
point it's created or changed; this module only computes vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imsg import constants
from imsg.embed.provider import MultimodalEmbeddingProvider, TextEmbeddingProvider
from imsg.embed.vector_codec import vector_literal
from imsg.errors import EmbeddingError
from imsg.hashing import sha256_text
from imsg.textnorm import normalize_text

if TYPE_CHECKING:
    import psycopg

DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class EmbedRunReport:
    segments_embedded: int = 0
    chunks_embedded: int = 0
    attachments_embedded: int = 0
    attachments_skipped_no_frames: int = 0
    """Video attachments whose `frame_ocr` enrichment is done but whose
    persisted keyframe files (`imsg.enrich.pipeline.
    frames_dir_for_attachment`) are missing — e.g. the 30-day artifact
    GC already reclaimed them (SPEC §5.3). Not an error: the caption/
    OCR text already made it into the primary text index; only the
    secondary multimodal vector for that attachment is skipped."""


def _pending_segments(conn: psycopg.Connection) -> list[tuple[int, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.segment_id, s.rendered_text, se.text_sha256 "
            "FROM segment s LEFT JOIN segment_embedding se ON se.segment_id = s.segment_id "
            "ORDER BY s.segment_id"
        )
        rows = cur.fetchall()
    pending = []
    for segment_id, text, existing_hash in rows:
        h = sha256_text(text)
        if h != existing_hash:
            pending.append((segment_id, text, h))
    return pending


def _pending_chunks(conn: psycopg.Connection) -> list[tuple[int, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.chunk_id, c.text, ce.text_sha256 "
            "FROM attachment_chunk c "
            "LEFT JOIN attachment_chunk_embedding ce ON ce.chunk_id = c.chunk_id "
            "ORDER BY c.chunk_id"
        )
        rows = cur.fetchall()
    pending = []
    for chunk_id, text, existing_hash in rows:
        h = sha256_text(text)
        if h != existing_hash:
            pending.append((chunk_id, text, h))
    return pending


def _embed_segments(
    conn: psycopg.Connection, provider: TextEmbeddingProvider, pending: list[tuple[int, str, str]], batch_size: int
) -> int:
    written = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        vectors = provider.embed_documents([normalize_text(text) for _, text, _ in batch])
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"provider returned {len(vectors)} vectors for a batch of {len(batch)} segments"
            )
        with conn.transaction(), conn.cursor() as cur:
            for (segment_id, _, text_hash), vec in zip(batch, vectors, strict=True):
                if len(vec) != provider.dim:
                    raise EmbeddingError(
                        f"provider {provider.model_id!r} returned a {len(vec)}-dim vector "
                        f"for segment {segment_id}, expected {provider.dim}"
                    )
                cur.execute(
                    """
                        INSERT INTO segment_embedding (segment_id, model, dim, text_sha256, vec)
                        VALUES (%s, %s, %s, %s, %s::halfvec)
                        ON CONFLICT (segment_id) DO UPDATE SET
                            model = EXCLUDED.model, dim = EXCLUDED.dim,
                            text_sha256 = EXCLUDED.text_sha256, vec = EXCLUDED.vec,
                            embedded_at = now()
                        """,
                    (segment_id, provider.model_id, provider.dim, text_hash, vector_literal(vec)),
                )
                written += 1
    return written


def _embed_chunks(
    conn: psycopg.Connection, provider: TextEmbeddingProvider, pending: list[tuple[int, str, str]], batch_size: int
) -> int:
    written = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        vectors = provider.embed_documents([normalize_text(text) for _, text, _ in batch])
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"provider returned {len(vectors)} vectors for a batch of {len(batch)} chunks"
            )
        with conn.transaction(), conn.cursor() as cur:
            for (chunk_id, _, text_hash), vec in zip(batch, vectors, strict=True):
                if len(vec) != provider.dim:
                    raise EmbeddingError(
                        f"provider {provider.model_id!r} returned a {len(vec)}-dim vector "
                        f"for chunk {chunk_id}, expected {provider.dim}"
                    )
                cur.execute(
                    """
                    INSERT INTO attachment_chunk_embedding (chunk_id, model, dim, text_sha256, vec)
                    VALUES (%s, %s, %s, %s, %s::halfvec)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        model = EXCLUDED.model, dim = EXCLUDED.dim,
                        text_sha256 = EXCLUDED.text_sha256, vec = EXCLUDED.vec,
                        embedded_at = now()
                    """,
                    (chunk_id, provider.model_id, provider.dim, text_hash, vector_literal(vec)),
                )
                written += 1
    return written


# --------------------------------------------------------------------------
# secondary multimodal vector (D3a) — images directly, videos via their
# persisted keyframes (imsg.enrich.pipeline.frames_dir_for_attachment)
# --------------------------------------------------------------------------


def _pending_multimodal_images(conn: psycopg.Connection) -> list[tuple[int, str, str]]:
    """(attachment_id, cache_path, media_sha256) — `media_sha256` is
    just the file's own content hash (`attachment.sha256`, from S5a)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attachment_id, a.cache_path, a.sha256, mm.media_sha256
            FROM attachment a
            LEFT JOIN attachment_mm_embedding mm ON mm.attachment_id = a.attachment_id
            WHERE a.state = 'materialized' AND a.mime_type LIKE 'image/%'
              AND a.cache_path IS NOT NULL AND a.sha256 IS NOT NULL
            """
        )
        rows = cur.fetchall()
    return [
        (attachment_id, cache_path, sha)
        for attachment_id, cache_path, sha, existing in rows
        if sha != existing
    ]


def _frame_paths_from_detail(detail_dict: dict[str, object]) -> list[str]:
    """Extract `frames[*].path` strings from an S5b `frame_ocr`
    enrichment's `detail` jsonb (`imsg.enrich.pipeline._sample_and_run`
    is what writes these), sorted for a stable content-hash input."""
    frames_raw = detail_dict.get("frames")
    if not isinstance(frames_raw, list):
        return []
    paths: list[str] = []
    for frame in frames_raw:
        if isinstance(frame, dict):
            path = frame.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
    return sorted(paths)


def _pending_multimodal_videos(conn: psycopg.Connection) -> list[tuple[int, dict[str, object]]]:
    """(attachment_id, frame_ocr detail jsonb) for videos whose
    `frame_ocr` enrichment is done — the frame paths live in that
    detail (SPEC §7.4: "videos via sampled keyframes ... per-frame
    vectors mean-pooled then re-normalized")."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attachment_id, e.detail, mm.media_sha256
            FROM attachment a
            JOIN enrichment e ON e.attachment_id = a.attachment_id
                AND e.kind = 'frame_ocr' AND e.state = 'done'
            LEFT JOIN attachment_mm_embedding mm ON mm.attachment_id = a.attachment_id
            WHERE a.state = 'materialized' AND a.mime_type LIKE 'video/%'
            """
        )
        rows = cur.fetchall()

    pending: list[tuple[int, dict[str, object]]] = []
    for attachment_id, detail, existing_hash in rows:
        detail_dict: dict[str, object] = json.loads(detail) if isinstance(detail, str) else (detail or {})
        paths = _frame_paths_from_detail(detail_dict)
        if not paths:
            continue
        media_hash = sha256_text("\n".join(paths))
        if media_hash != existing_hash:
            pending.append((attachment_id, detail_dict))
    return pending


def _mean_pool_and_normalize(vectors: list[list[float]]) -> list[float]:
    import math

    dim = len(vectors[0])
    summed = [0.0] * dim
    for vec in vectors:
        for i, v in enumerate(vec):
            summed[i] += v
    mean = [v / len(vectors) for v in summed]
    norm = math.sqrt(sum(v * v for v in mean))
    if norm == 0:
        return mean
    return [v / norm for v in mean]


def _upsert_mm_embedding(
    conn: psycopg.Connection, attachment_id: int, provider: MultimodalEmbeddingProvider, media_sha256: str, vec: list[float]
) -> None:
    if len(vec) != provider.dim:
        raise EmbeddingError(
            f"multimodal provider {provider.model_id!r} returned a {len(vec)}-dim vector "
            f"for attachment {attachment_id}, expected {provider.dim}"
        )
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment_mm_embedding (attachment_id, model, dim, media_sha256, vec)
            VALUES (%s, %s, %s, %s, %s::halfvec)
            ON CONFLICT (attachment_id) DO UPDATE SET
                model = EXCLUDED.model, dim = EXCLUDED.dim,
                media_sha256 = EXCLUDED.media_sha256, vec = EXCLUDED.vec, embedded_at = now()
            """,
            (attachment_id, provider.model_id, provider.dim, media_sha256, vector_literal(vec)),
        )


def _embed_multimodal(
    conn: psycopg.Connection, provider: MultimodalEmbeddingProvider
) -> tuple[int, int]:
    written = 0
    skipped_no_frames = 0

    for attachment_id, cache_path, media_sha256 in _pending_multimodal_images(conn):
        vectors = provider.embed_images([Path(cache_path)])
        if len(vectors) != 1:
            raise EmbeddingError(
                f"multimodal provider returned {len(vectors)} vectors for 1 image "
                f"(attachment {attachment_id})"
            )
        _upsert_mm_embedding(conn, attachment_id, provider, media_sha256, vectors[0])
        written += 1

    for attachment_id, detail in _pending_multimodal_videos(conn):
        frame_paths = [Path(p) for p in _frame_paths_from_detail(detail)]
        existing_paths = [p for p in frame_paths if p.is_file()]
        if not existing_paths:
            skipped_no_frames += 1
            continue
        media_hash = sha256_text("\n".join(str(p) for p in frame_paths))
        vectors = provider.embed_images(existing_paths)
        if len(vectors) != len(existing_paths):
            raise EmbeddingError(
                f"multimodal provider returned {len(vectors)} vectors for "
                f"{len(existing_paths)} frames (attachment {attachment_id})"
            )
        pooled = _mean_pool_and_normalize(vectors)
        _upsert_mm_embedding(conn, attachment_id, provider, media_hash, pooled)
        written += 1

    return written, skipped_no_frames


def run_embed(
    conn: psycopg.Connection,
    text_provider: TextEmbeddingProvider,
    *,
    multimodal_provider: MultimodalEmbeddingProvider | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbedRunReport:
    """One full embedding pass: every segment/chunk lacking an
    up-to-date embedding, then (if `multimodal_provider` is given)
    every image/video attachment lacking an up-to-date
    `attachment_mm_embedding` (D3a)."""
    if text_provider.dim != constants.PRIMARY_EMBEDDING_DIM:
        raise EmbeddingError(
            f"text_provider.dim ({text_provider.dim}) does not match the primary "
            f"embedding dimension migration 0001 requires ({constants.PRIMARY_EMBEDDING_DIM}) "
            f"— refusing to embed anything with it"
        )
    if multimodal_provider is not None and multimodal_provider.dim != constants.MULTIMODAL_EMBEDDING_DIM:
        raise EmbeddingError(
            f"multimodal_provider.dim ({multimodal_provider.dim}) does not match the "
            f"multimodal embedding dimension migration 0002 requires "
            f"({constants.MULTIMODAL_EMBEDDING_DIM}) — refusing to embed anything with it"
        )

    segments_written = _embed_segments(conn, text_provider, _pending_segments(conn), batch_size)
    chunks_written = _embed_chunks(conn, text_provider, _pending_chunks(conn), batch_size)

    attachments_written = 0
    skipped_no_frames = 0
    if multimodal_provider is not None:
        attachments_written, skipped_no_frames = _embed_multimodal(conn, multimodal_provider)

    return EmbedRunReport(
        segments_embedded=segments_written,
        chunks_embedded=chunks_written,
        attachments_embedded=attachments_written,
        attachments_skipped_no_frames=skipped_no_frames,
    )


__all__ = ["DEFAULT_BATCH_SIZE", "EmbedRunReport", "run_embed"]
