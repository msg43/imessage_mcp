"""The hybrid query flow (SPEC §9.4) and the retrieval-service façade
behind `search_messages` / `get_conversation` / `list_people` /
`get_attachment_text` (SPEC §10.2). Pure domain logic — this module
never imports anything from `mcp` or knows about MCP wire types; see
`imsg.mcp.tools` for that adapter layer.

Takes the whole `Config` object (this codebase's established
convention — `imsg.segment.pipeline`/`imsg.enrich.pipeline` do the
same) rather than a narrower bespoke settings type, and an
already-open `psycopg.Connection` / `apsw.Connection` pair it never
owns the lifecycle of, matching every other DB-touching module in this
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.embed.provider import MultimodalEmbeddingProvider, TextEmbeddingProvider
from imsg.retrieval import directory, fts_search, segments, vector_search
from imsg.retrieval.access import AccessContext, segment_eligibility_predicate
from imsg.retrieval.errors import InvalidArgumentError, NotFoundError
from imsg.retrieval.filters import compile_predicate, resolve_filters
from imsg.retrieval.fuse import reciprocal_rank_fusion
from imsg.retrieval.query import analyze_query
from imsg.retrieval.reranker import RerankerProvider

if TYPE_CHECKING:
    import apsw
    import psycopg

    from imsg.config.schema import Config

MAX_QUERY_CHARS = 1000
MAX_SEARCH_LIMIT = 50
MAX_CONVERSATION_WINDOW = 200
MAX_LIST_PEOPLE_LIMIT = 500


@dataclass(frozen=True, slots=True)
class SearchMessagesResult:
    results: list[dict[str, object]]
    candidate_lists: dict[str, int]
    scan_cap_reached: bool


class RetrievalService:
    """One instance per MCP server process — holds the (long-lived)
    connections and model providers every tool call needs. Every
    method's first positional argument is a non-optional
    `AccessContext` (SPEC §10.3a)."""

    def __init__(
        self,
        *,
        pg_conn: psycopg.Connection,
        fts_conn: apsw.Connection,
        config: Config,
        text_provider: TextEmbeddingProvider,
        reranker: RerankerProvider,
        multimodal_provider: MultimodalEmbeddingProvider | None = None,
    ) -> None:
        self._pg = pg_conn
        self._fts = fts_conn
        self._config = config
        self._text_provider = text_provider
        self._reranker = reranker
        self._multimodal_provider = multimodal_provider

    # -- search_messages ----------------------------------------------------

    def search_messages(
        self,
        context: AccessContext,
        *,
        query: str,
        people: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        has_attachment: bool | None = None,
        limit: int | None = None,
    ) -> SearchMessagesResult:
        if not query or not query.strip():
            raise InvalidArgumentError("'query' must not be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise InvalidArgumentError(f"'query' exceeds {MAX_QUERY_CHARS} characters")

        effective_limit = limit if limit is not None else self._config.retrieval.default_limit
        if not (1 <= effective_limit <= MAX_SEARCH_LIMIT):
            raise InvalidArgumentError(f"'limit' must be between 1 and {MAX_SEARCH_LIMIT}")

        filters = resolve_filters(
            self._pg,
            context,
            people=people,
            after=after,
            before=before,
            has_attachment=has_attachment,
            timezone=self._config.render.timezone,
        )
        predicate = compile_predicate(filters, context)
        analyzed = analyze_query(query)

        k_fts = self._config.retrieval.k_fts
        k_vector = self._config.retrieval.k_vector

        seg_fts = fts_search.search_segment_fts(self._fts, self._pg, analyzed, predicate, k_fts)
        att_fts = fts_search.search_attachment_chunk_fts(
            self._fts, self._pg, analyzed, predicate, k_fts
        )

        query_vec = self._text_provider.embed_query(
            analyzed.phrase, instruction=self._config.embedding.query_instruction
        )
        seg_vec = vector_search.search_segment_vector(self._pg, query_vec, predicate, k_vector)
        att_vec = vector_search.search_attachment_chunk_vector(
            self._pg, query_vec, predicate, k_vector
        )

        mm_channel = None
        if self._config.embedding.multimodal.enabled and self._multimodal_provider is not None:
            mm_query_vec = self._multimodal_provider.embed_text(analyzed.phrase)
            mm_channel = vector_search.search_multimodal_vector(
                self._pg, mm_query_vec, predicate, k_vector
            )

        lists: dict[str, tuple[int, ...]] = {
            "segment_fts": seg_fts.segment_ids,
            "attachment_fts": att_fts.segment_ids,
            "segment_vector": seg_vec.segment_ids,
            "attachment_vector": att_vec.segment_ids,
        }
        if mm_channel is not None:
            lists["multimodal_vector"] = mm_channel.segment_ids

        fused = reciprocal_rank_fusion(lists, rrf_k=self._config.retrieval.rrf_k)

        rerank_top = self._config.retrieval.rerank_top
        pool = fused[:rerank_top]
        summaries = segments.fetch_segment_summaries(self._pg, [r.segment_id for r in pool])
        # A fused id can vanish between fusion and this fetch (concurrent
        # delete/re-segmentation) — drop rather than crash; RRF already
        # ranked the survivors correctly relative to each other.
        pool = [r for r in pool if r.segment_id in summaries]

        reranked: list[tuple[int, float]] = []
        if pool:
            documents = [summaries[r.segment_id].text for r in pool]
            scores = self._reranker.score(analyzed.phrase, documents)
            reranked = sorted(
                ((r.segment_id, s) for r, s in zip(pool, scores, strict=True)),
                key=lambda t: -t[1],
            )

        results: list[dict[str, object]] = []
        for segment_id, score in reranked[:effective_limit]:
            summary = summaries[segment_id]
            results.append(
                {
                    "segment_key": summary.segment_key,
                    "thread_key": summary.thread_key,
                    "chat": {
                        "thread_key": summary.thread_key,
                        "kind": summary.chat_kind,
                        "display_name": summary.chat_display_name,
                    },
                    "people": list(summary.people),
                    "started_at": summary.started_at.isoformat(),
                    "ended_at": summary.ended_at.isoformat(),
                    "message_count": summary.message_count,
                    "has_attachments": summary.has_attachments,
                    "score": score,
                    "text": summary.text,
                    "untrusted_content": True,
                }
            )

        candidate_lists = {
            "segment_fts": len(seg_fts.segment_ids),
            "attachment_fts": len(att_fts.segment_ids),
            "segment_vector": len(seg_vec.segment_ids),
            "attachment_vector": len(att_vec.segment_ids),
            "multimodal_vector": len(mm_channel.segment_ids) if mm_channel is not None else 0,
        }
        scan_cap_reached = any(
            (
                seg_fts.scan_cap_reached,
                att_fts.scan_cap_reached,
                seg_vec.scan_cap_reached,
                att_vec.scan_cap_reached,
                mm_channel.scan_cap_reached if mm_channel is not None else False,
            )
        )

        return SearchMessagesResult(
            results=results, candidate_lists=candidate_lists, scan_cap_reached=scan_cap_reached
        )

    # -- get_conversation -----------------------------------------------

    def get_conversation(
        self,
        context: AccessContext,
        *,
        thread_id: str,
        anchor: str | None = None,
        window: int = 20,
    ) -> dict[str, object]:
        if not (1 <= window <= MAX_CONVERSATION_WINDOW):
            raise InvalidArgumentError(f"'window' must be between 1 and {MAX_CONVERSATION_WINDOW}")

        resolution = segments.resolve_thread(self._pg, thread_id)
        self._assert_chat_authorized(context, resolution.chat_id)
        anchor_dt = segments.resolve_anchor(self._pg, resolution, anchor)
        messages = segments.fetch_conversation_window(
            self._pg,
            resolution,
            anchor_dt,
            window,
            index_unsent=self._config.policy.index_unsent,
            include_edit_history=self._config.policy.index_edit_history,
            timezone=self._config.render.timezone,
            attachment_snippet_chars=self._config.render.attachment_snippet_chars,
        )
        return {"thread_key": resolution.thread_key, "messages": messages}

    def _assert_chat_authorized(self, context: AccessContext, chat_id: int) -> None:
        predicate_sql = segment_eligibility_predicate(context, chat_id_expr="%(chat_id)s")
        if predicate_sql == "TRUE":
            return
        with self._pg.cursor() as cur:
            cur.execute(f"SELECT ({predicate_sql})", {"chat_id": chat_id})
            row = cur.fetchone()
        if row is None or not row[0]:
            raise NotFoundError("no thread found for the given identifier")

    # -- list_people ------------------------------------------------------

    def list_people(
        self,
        context: AccessContext,
        *,
        query: str | None = None,
        limit: int = 100,
        include_handles: bool = False,
    ) -> dict[str, object]:
        if not (1 <= limit <= MAX_LIST_PEOPLE_LIMIT):
            raise InvalidArgumentError(f"'limit' must be between 1 and {MAX_LIST_PEOPLE_LIMIT}")
        listings = directory.list_people(
            self._pg, context, query=query, limit=limit, include_handles=include_handles
        )
        people: list[dict[str, object]] = []
        for p in listings:
            entry: dict[str, object] = {
                "short_name": p.short_name,
                "display_name": p.display_name,
                "organization": p.organization,
                "message_count": p.message_count,
                "first_message": p.first_message,
                "last_message": p.last_message,
            }
            if include_handles:
                entry["handles"] = list(p.handles or ())
            people.append(entry)
        return {"people": people}

    # -- get_attachment_text ------------------------------------------------

    def get_attachment_text(
        self, context: AccessContext, *, attachment_key: str
    ) -> dict[str, object]:
        if not attachment_key or not (16 <= len(attachment_key) <= 128):
            raise InvalidArgumentError(
                "'attachment_key' must be between 16 and 128 characters"
            )
        result = directory.get_attachment_text(self._pg, context, attachment_key)
        return {
            "attachment_key": result.attachment_key,
            "filename": result.filename,
            "mime_type": result.mime_type,
            "texts": [dict(t) for t in result.texts],
            "untrusted_content": True,
        }


__all__ = ["RetrievalService", "SearchMessagesResult"]
