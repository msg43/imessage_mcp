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

**One call at a time per service.** Both connections are single
objects shared by every method, and neither driver tolerates two
threads using one connection at once — this is a measured property of
the drivers, not a precaution:

- `psycopg.Connection` (3.3.4) serializes individual operations behind
  its own lock, but `Connection.transaction()` — which every query here
  goes through — keeps one *connection-global* savepoint stack. Two
  threads of different durations therefore exit out of order and both
  raise `OutOfOrderTransactionNesting`, leaving the connection
  mid-transaction and unusable for every later request. Before that
  point they also share one transaction, so the `SET LOCAL
  hnsw.ef_search` / `enable_seqscan` settings
  `imsg.retrieval.vector_search` relies on leak between queries: a probe
  that set 400 in one thread read back 40, the other thread's value,
  which is a silently wrong recall rather than an error.
- `apsw.Connection` (3.53.4) refuses outright, with
  `ThreadingViolationError: Cursor couldn't run because the Connection
  is busy in another thread`.

So every public method below holds `_connections` for its whole
duration. A lock rather than a connection pool because the ceiling on
any parallelism here is `imsg.retrieval.model_thread`, which is single
by construction (MLX gives each OS thread its own default GPU stream):
the embedding and reranking stages, ~0.62 s p50 of a 0.96-1.87 s query
on the production host (2026-09-17/18), are serialized whatever the
database does. Pooling would overlap the remainder — worth revisiting
with a measured per-stage split, and it would need a pool for the
SQLite side too, since that connection is equally exclusive.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.db.enrichment_yield_locks import QueryInFlightMarker
from imsg.db.prewarm import prewarm_query_path
from imsg.embed.provider import MultimodalEmbeddingProvider, TextEmbeddingProvider
from imsg.retrieval import directory, fts_search, segments, vector_search
from imsg.retrieval.access import AccessContext, resolve_request_scope
from imsg.retrieval.background_warm_up import WarmUpStep
from imsg.retrieval.errors import InvalidArgumentError, NotFoundError
from imsg.retrieval.filters import compile_predicate, resolve_filters
from imsg.retrieval.fuse import reciprocal_rank_fusion
from imsg.retrieval.query import analyze_query
from imsg.retrieval.reranker import RerankerProvider

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    import apsw
    import psycopg

    from imsg.config.schema import Config
    from imsg.retrieval.model_thread import ModelThread

WARM_UP_QUERY = "warm-up query"
WARM_UP_DOCUMENT = "warm-up document"
"""The throwaway inputs :meth:`RetrievalService.warm_up` runs through each
model — fixed, generic, never a real query."""

TEXT_EMBEDDER_STEP = "text embedder"
MULTIMODAL_STEP = "multimodal text tower"
RERANKER_STEP = "reranker"
BUFFER_POOL_STEP = "database buffer pool"

ESTIMATED_WARM_UP_SECONDS: dict[str, float] = {
    TEXT_EMBEDDER_STEP: 13.0,
    MULTIMODAL_STEP: 51.0,
    RERANKER_STEP: 10.0,
    BUFFER_POOL_STEP: 30.0,
}
"""How long each warm-up step is expected to take, used only for the
seconds-remaining estimate a tool call gets while the models load
(`imsg.retrieval.background_warm_up`). Each is the slowest time seen for
that step on the M2 Ultra host, rounded up, so the estimate errs long.

Measured through `imsg mcp local` on 2026-09-17, four cold starts, with
the data volume's array carrying another job's 216-2,177 MB/s throughout:
text embedder 4.8-12.5 s, PE-Core text tower 33.8-50.2 s, reranker (the
0.58 GiB 0.6B conversion) 2.7-3.2 s, buffer pool 0.1 s with the pages
already resident — 14.3 s when it has to read all 2,296 MiB
(`imsg.db.prewarm`). Whole warm-up: 41.8-65.4 s. The reranker step took
51.3-154.8 s while the 7.9 GiB 8B conversion was pinned, which is what
made the old estimate 155 s."""

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
    `AccessContext` (SPEC §10.3a), which the method turns into the
    request's `imsg.retrieval.access.RequestScope` before reading
    anything."""

    def __init__(
        self,
        *,
        pg_conn: psycopg.Connection,
        fts_conn: apsw.Connection,
        config: Config,
        text_provider: TextEmbeddingProvider,
        reranker: RerankerProvider,
        multimodal_provider: MultimodalEmbeddingProvider | None = None,
        model_thread: ModelThread | None = None,
        query_marker: QueryInFlightMarker | None = None,
    ) -> None:
        """`model_thread`, when given, is where every model call runs —
        the warm-up and every query alike (`imsg.retrieval.model_thread`
        explains why a server that warms up in the background needs
        that). Without one, models run on the calling thread.

        `query_marker`, when given, publishes "a query is in flight" for
        the span of every search and every model call, so the nightly
        enrichment worker pauses between its units of work
        (`imsg.db.enrichment_yield_locks`). It is re-entrant, so the
        overlapping `search_messages` and `_run_model` scopes below take
        it once. Nothing about a query depends on it: the marker never
        blocks and never raises."""
        self._pg = pg_conn
        self._fts = fts_conn
        self._config = config
        self._text_provider = text_provider
        self._reranker = reranker
        self._multimodal_provider = multimodal_provider
        self._model_thread = model_thread
        self._query_marker = query_marker
        self._connections = threading.RLock()

    def _exclusive(self) -> AbstractContextManager[object]:
        """Hold the connections for the caller's whole scope — the module
        docstring has the two driver errors this prevents.

        Re-entrant because `get_conversation` reaches the database both
        directly and through `_assert_chat_authorized`; making that a
        deadlock would be a worse failure than the one being fixed. It is
        held across `_run_model` too, so a query keeps the connections
        while it waits on the model thread: that is the cost of the lock,
        and it is bounded by the model thread already being the serial
        stage."""
        return self._connections

    def _marking(self) -> AbstractContextManager[object]:
        """Publish "a query is in flight" for this scope, when a marker
        was wired; otherwise do nothing at all."""
        if self._query_marker is None:
            return contextlib.nullcontext()
        return self._query_marker

    def _run_model[T](self, call: Callable[[], T]) -> T:
        """The one way this class invokes a model provider."""
        with self._marking():
            if self._model_thread is None:
                return call()
            return self._model_thread.run(call)

    # -- warm-up ------------------------------------------------------------

    def warm_up_steps(self) -> tuple[WarmUpStep, ...]:
        """One step per model provider, each loading it and running one
        throwaway input through it — the query embedder, the multimodal
        text tower when channel C is enabled, and the reranker — so a
        server's first real query does not pay for loading weights and
        compiling kernels (loading alone took 4-5 s for the text embedder,
        4-7 s for the reranker and 26-32 s for PE-Core on an M2 Ultra,
        2026-09-16), and one last step that pulls the query path's
        relations into the database's buffer pool
        (`imsg.db.prewarm.prewarm_query_path`), so a server started after a
        reboot does not pay 300-900 ms per vector search for cold index
        pages either. A provider that cannot load raises from its step; a
        pool that cannot be prewarmed is reported, not raised (an unwarmed
        pool is slow, not wrong).

        The models come first because nothing can be answered without
        them, and because both they and the prewarm read from the same
        volume."""
        instruction = self._config.embedding.query_instruction

        def text_embedder() -> None:
            self._run_model(
                lambda: self._text_provider.embed_query(WARM_UP_QUERY, instruction=instruction)
            )

        def reranker() -> None:
            self._run_model(lambda: self._reranker.score(WARM_UP_QUERY, [WARM_UP_DOCUMENT]))

        estimate = ESTIMATED_WARM_UP_SECONDS
        steps = [WarmUpStep(TEXT_EMBEDDER_STEP, estimate[TEXT_EMBEDDER_STEP], text_embedder)]
        multimodal = self._multimodal_provider
        if self._config.embedding.multimodal.enabled and multimodal is not None:

            def multimodal_text_tower() -> None:
                self._run_model(lambda: multimodal.embed_text(WARM_UP_QUERY))

            steps.append(
                WarmUpStep(MULTIMODAL_STEP, estimate[MULTIMODAL_STEP], multimodal_text_tower)
            )
        steps.append(WarmUpStep(RERANKER_STEP, estimate[RERANKER_STEP], reranker))

        def buffer_pool() -> str:
            with self._exclusive():
                return prewarm_query_path(self._pg).summary

        steps.append(WarmUpStep(BUFFER_POOL_STEP, estimate[BUFFER_POOL_STEP], buffer_pool))
        return tuple(steps)

    def unload_models(self) -> None:
        """Drop every model provider's weights, on the calling thread —
        the idle unloader runs this on the model thread
        (`imsg.retrieval.idle_unload`). A provider with no `unload` (the
        fake backend's) holds no weights and is skipped. The next model
        call loads each one again: every real provider loads lazily, and
        the server re-runs `warm_up_steps` before it answers anyway."""
        for provider in (self._text_provider, self._multimodal_provider, self._reranker):
            unload = getattr(provider, "unload", None)
            if callable(unload):
                unload()

    def warm_up(self) -> float:
        """Run every :meth:`warm_up_steps` step now, on this thread's
        behalf, and return the seconds it took. The local MCP server runs
        the same steps in the background instead
        (`imsg.retrieval.background_warm_up`)."""
        started = time.perf_counter()
        for step in self.warm_up_steps():
            step.run()
        return time.perf_counter() - started

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
        """SPEC §9.4's hybrid search. The whole request — model calls and
        the database work between them — runs inside the query-in-flight
        marker, so an enrichment worker that checks between tasks sees the
        gap between this search's embedding and its rerank as busy rather
        than idle."""
        with self._marking(), self._exclusive():
            return self._search_messages(
                context,
                query=query,
                people=people,
                after=after,
                before=before,
                has_attachment=has_attachment,
                limit=limit,
            )

    def _search_messages(
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

        scope = resolve_request_scope(self._pg, context)
        filters = resolve_filters(
            self._pg,
            scope,
            people=people,
            after=after,
            before=before,
            has_attachment=has_attachment,
            timezone=self._config.render.timezone,
        )
        predicate = compile_predicate(filters, scope)
        analyzed = analyze_query(query)
        if scope.admits_nothing:
            # Nothing is eligible (an empty allowlist): every channel would
            # return nothing, so skip the model calls and the scans. SPEC
            # §10.1: an empty result is a success.
            return SearchMessagesResult(
                results=[],
                candidate_lists=dict.fromkeys(
                    (
                        "segment_fts",
                        "attachment_fts",
                        "segment_vector",
                        "attachment_vector",
                        "multimodal_vector",
                    ),
                    0,
                ),
                scan_cap_reached=False,
            )

        k_fts = self._config.retrieval.k_fts
        k_vector = self._config.retrieval.k_vector
        ef_search = self._config.retrieval.hnsw_ef_search

        seg_fts = fts_search.search_segment_fts(self._fts, self._pg, analyzed, predicate, k_fts)
        att_fts = fts_search.search_attachment_chunk_fts(
            self._fts, self._pg, analyzed, predicate, k_fts
        )

        instruction = self._config.embedding.query_instruction
        query_vec = self._run_model(
            lambda: self._text_provider.embed_query(analyzed.phrase, instruction=instruction)
        )
        seg_vec = vector_search.search_segment_vector(
            self._pg, query_vec, predicate, k_vector, ef_search=ef_search
        )
        att_vec = vector_search.search_attachment_chunk_vector(
            self._pg, query_vec, predicate, k_vector, ef_search=ef_search
        )

        mm_channel = None
        multimodal = self._multimodal_provider
        if self._config.embedding.multimodal.enabled and multimodal is not None:
            mm_query_vec = self._run_model(lambda: multimodal.embed_text(analyzed.phrase))
            mm_channel = vector_search.search_multimodal_vector(
                self._pg, mm_query_vec, predicate, k_vector, ef_search=ef_search
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

        # The pool is never smaller than the caller's `limit`: SPEC §9.4
        # step 8 returns the top `limit` of the reranked pool, so a
        # `rerank_top` tuned below `limit` for latency must not silently
        # return fewer results. A caller asking for more results pays for
        # reranking them.
        rerank_top = max(self._config.retrieval.rerank_top, effective_limit)
        pool = fused[:rerank_top]
        summaries = segments.fetch_segment_summaries(self._pg, [r.segment_id for r in pool])
        # A fused id can vanish between fusion and this fetch (concurrent
        # delete/re-segmentation) — drop rather than crash; RRF already
        # ranked the survivors correctly relative to each other.
        pool = [r for r in pool if r.segment_id in summaries]
        texts = {r.segment_id: summaries[r.segment_id].text for r in pool}
        if scope.withholds_ineligible_content:
            # The stored text carries every attachment's snippet and
            # whatever `policy.*` allowed at segmentation time; under this
            # scope both the reranker and the caller see only the
            # re-rendered, gated text (imsg.retrieval.access).
            texts = segments.render_segment_texts(
                self._pg,
                scope,
                [r.segment_id for r in pool],
                timezone=self._config.render.timezone,
                attachment_snippet_chars=self._config.render.attachment_snippet_chars,
            )
            pool = [r for r in pool if r.segment_id in texts]

        reranked: list[tuple[int, float]] = []
        if pool:
            documents = [texts[r.segment_id] for r in pool]
            scores = self._run_model(lambda: self._reranker.score(analyzed.phrase, documents))
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
                    "text": texts[segment_id],
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

        with self._exclusive():
            resolution = segments.resolve_thread(self._pg, thread_id)
            scope = resolve_request_scope(self._pg, context, among_chat_ids=[resolution.chat_id])
            if not scope.admits_chat(resolution.chat_id):
                # The same error, word for word, as an identifier that matches
                # nothing (segments.THREAD_NOT_FOUND_MESSAGE).
                raise NotFoundError(segments.THREAD_NOT_FOUND_MESSAGE)
            anchor_dt = segments.resolve_anchor(self._pg, resolution, anchor)
            messages = segments.fetch_conversation_window(
                self._pg,
                resolution,
                anchor_dt,
                window,
                scope=scope,
                index_unsent=self._config.policy.index_unsent,
                include_edit_history=self._config.policy.index_edit_history,
                timezone=self._config.render.timezone,
                attachment_snippet_chars=self._config.render.attachment_snippet_chars,
            )
        return {"thread_key": resolution.thread_key, "messages": messages}

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
        if include_handles and context.surface != "local":
            # SPEC §10.2: "never raw handles on the public surface" — the
            # public schema omits the property; this holds even if a
            # caller reaches the service some other way.
            raise InvalidArgumentError("'include_handles' is available on the local surface only")
        with self._exclusive():
            scope = resolve_request_scope(self._pg, context)
            listings = directory.list_people(
                self._pg, scope, query=query, limit=limit, include_handles=include_handles
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
        with self._exclusive():
            result = directory.get_attachment_text(self._pg, context, attachment_key)
        return {
            "attachment_key": result.attachment_key,
            "filename": result.filename,
            "mime_type": result.mime_type,
            "texts": [dict(t) for t in result.texts],
            "untrusted_content": True,
        }


__all__ = [
    "BUFFER_POOL_STEP",
    "ESTIMATED_WARM_UP_SECONDS",
    "MULTIMODAL_STEP",
    "RERANKER_STEP",
    "TEXT_EMBEDDER_STEP",
    "WARM_UP_DOCUMENT",
    "WARM_UP_QUERY",
    "RetrievalService",
    "SearchMessagesResult",
]
