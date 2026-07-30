"""Retrieval backends the eval runner can target (SPEC §13.3: "Target
`local`: in-process retrieval service ... Target `gemini`: Discovery
Engine `servingConfigs.search` API against the ingested store").

Both backends implement the same tiny `EvalBackend` Protocol so
`imsg.eval.runner.run_eval` never branches on which one it was given —
the same pattern as `imsg.export.transport.ExportTransport`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import psycopg

    from imsg.retrieval.access import AccessContext
    from imsg.retrieval.service import RetrievalService


class EvalBackend(Protocol):
    def search(self, query_text: str, *, k: int) -> Sequence[str]:
        """Ranked `segment_key`s for `query_text`, best first, at most
        `k` of them."""
        ...


@dataclass
class LocalEvalBackend:
    """Target `local` (SPEC §13.3): wraps an already-constructed
    `RetrievalService` — segmentation thresholds, reranker on/off, and
    dual-vector on/off are all properties of *how that service was
    built* (its `Config` and provider choices), not of this backend, so
    comparing config variants is just constructing two services and two
    `LocalEvalBackend`s (see `imsg.cli`'s `eval run` for the on/off
    toggles this build wires up)."""

    service: RetrievalService
    context: AccessContext

    def search(self, query_text: str, *, k: int) -> Sequence[str]:
        result = self.service.search_messages(self.context, query=query_text, limit=k)
        return [str(r["segment_key"]) for r in result.results]


class PassthroughReranker:
    """Implements `imsg.retrieval.reranker.RerankerProvider` as a
    no-op: assigns strictly descending scores in input order, so
    `RetrievalService.search_messages`'s `sorted(..., key=lambda t:
    -score)` reproduces the RRF-fused order unchanged. This is how the
    eval CLI realizes a "reranker off" config variant (SPEC §13.3:
    "reranker on/off ... the variants exist to be compared") without
    touching `imsg.retrieval.service` — the reranking *stage* still
    runs, it just doesn't change anything.
    """

    model_id = "eval/passthrough-no-rerank@n/a"

    def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        n = len(documents)
        return [float(n - i) for i in range(n)]


class GeminiSearchClient(Protocol):
    """The thin boundary to Discovery Engine's search API — mirrors
    `imsg.export.transport.ExportTransport`'s shape so a fake
    implementation can drive `GeminiEvalBackend`'s doc-id-resolution
    logic in tests without any network access."""

    def search(self, query_text: str, *, page_size: int) -> Sequence[str]:
        """Ranked Discovery Engine `Document.id`s for `query_text`."""
        ...


@dataclass
class GeminiEvalBackend:
    """Target `gemini` (SPEC §13.3, judgment call #7): "API-level search
    scores the ingested leg, and a documented manual omnibar protocol
    ... covers the last mile for Phase 8." This backend is that
    API-level half — it maps returned Discovery Engine document ids
    back to `segment_key`s via `export_document` (SPEC §13.3:
    "mapping returned doc ids -> `export_document.segment_id` ->
    `stable_key`; attachment-chunk hits fold to their parent segment
    before scoring") and de-duplicates, since a segment document and
    one of its attachment-chunk documents can both appear in a result
    page and must count once.

    `export_document.segment_id` already holds the *parent* segment id
    for both `kind='segment'` and `kind='attachment_chunk'` rows (see
    `imsg.export.planner._upsert_export_document` / `PlannedUpsert`),
    so no `kind`-specific join is needed here — folding is just "look
    up `segment_id`, then `stable_key`."

    Only overfetches Google itself to absorb the fold-to-parent
    deduplication; it never asks Google for more than a small constant
    multiple of `k`. `client` is real-API-shaped but unverified against
    the live API in this build (Phase 8 is when target=gemini is first
    exercised for real) — see `imsg.eval.gemini_client` for the real
    implementation behind this Protocol.
    """

    conn: psycopg.Connection
    client: GeminiSearchClient
    overfetch_multiplier: int = 3

    def search(self, query_text: str, *, k: int) -> Sequence[str]:
        doc_ids = self.client.search(query_text, page_size=k * self.overfetch_multiplier)
        segment_keys: list[str] = []
        seen: set[str] = set()
        for doc_id in doc_ids:
            key = self._resolve_document_to_segment_key(doc_id)
            if key is not None and key not in seen:
                seen.add(key)
                segment_keys.append(key)
            if len(segment_keys) >= k:
                break
        return segment_keys

    def _resolve_document_to_segment_key(self, document_id: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.stable_key
                FROM export_document ed
                JOIN segment s ON s.segment_id = ed.segment_id
                WHERE ed.document_id = %s AND ed.state = 'pushed'
                """,
                (document_id,),
            )
            row = cur.fetchone()
        return str(row[0]) if row is not None else None


__all__ = [
    "EvalBackend",
    "GeminiEvalBackend",
    "GeminiSearchClient",
    "LocalEvalBackend",
    "PassthroughReranker",
]
