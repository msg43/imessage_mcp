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

**Connections: shared and locked, or pooled.** A service is built one
of two ways, and every method behaves the same either way — only what
runs at the same time differs.

*Shared* (`pg_conn`/`fts_conn` only — `imsg mcp local`, the bench and
eval scripts, most tests): both connections are single objects shared
by every method, and neither driver tolerates two threads using one
connection at once — a measured property of the drivers, not a
precaution:

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

So every public method holds a lock over the shared pair for its whole
duration, and the channels of a search run one after another, as they
always did.

*Pooled* (`connections=` a `imsg.retrieval.connections.
RetrievalConnections` — `imsg mcp public` since 2026-09-24): each call
borrows a Postgres connection of its own, so calls no longer queue
behind each other's model time (a `get_conversation` does not wait for a
search's reranker), and a search's five candidate channels each borrow
their own connections and run side by side on worker threads — the two
FTS channels while the query is embedded, the vector channels together
after it (QA review 2026-09-24). Results are the same as the shared
build's, byte for byte (`tests/test_retrieval_concurrent_channels_
integration.py`): every channel is its own transaction on either build,
fusion takes the lists in a fixed order whatever order they finish in,
and a channel that finds no free connection runs on the call's own,
after the others.

What pooling cannot overlap is the models: every model call runs on the
single `imsg.retrieval.model_thread` (MLX gives each OS thread its own
default GPU stream). On the production host that is most of a search:
`scripts/bench_retrieval_latency.py --configs new:20:256` (2026-09-18,
20 queries x 2 passes against the real index, warm) measured rerank
611 ms, query embedding 54 ms and PE-Core text 20 ms, against 74 ms for
every Postgres and SQLite stage together, of a 767 ms call. The channels
are the part of those 74 ms that can run at once.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import wait as wait_for_futures
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    from imsg.retrieval.connections import RetrievalConnections
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


class _Pending[R]:
    """One candidate channel's result, however it ran: already computed
    (shared build), running on a worker (`future`), or deferred to run on
    the call's own connection when asked for (no connection was free)."""

    __slots__ = ("future", "has_value", "needs_fts", "value", "work")

    def __init__(
        self,
        work: Callable[[psycopg.Connection, apsw.Connection | None], R],
        *,
        needs_fts: bool,
    ) -> None:
        self.work = work
        self.needs_fts = needs_fts
        self.future: Future[R] | None = None
        self.has_value = False
        self.value: R | None = None


class _ChannelRun:
    """Runs one search's candidate channels (module docstring).

    Shared build (`pools is None`): each channel runs the moment it is
    submitted, on the call's own connections — exactly the old sequential
    order. Pooled build: a channel that can borrow a Postgres connection
    (and a sidecar connection, for FTS) without waiting runs on a worker
    thread; one that cannot is deferred and runs on the call's own
    Postgres connection when its result is asked for. Borrowing never
    waits, so a call that already holds a connection can never wait for
    another held by a call waiting for its own — the pool cannot
    deadlock a search.

    :meth:`finish` waits for every worker, so none is still running when
    the call returns or raises."""

    def __init__(
        self,
        pools: RetrievalConnections | None,
        pg: psycopg.Connection,
        fts: apsw.Connection | None,
    ) -> None:
        self._pools = pools
        self._pg = pg
        self._fts = fts
        self._outstanding: list[Future[Any]] = []

    def submit_pg[R](self, work: Callable[[psycopg.Connection], R]) -> _Pending[R]:
        return self._submit(lambda pg, _fts: work(pg), needs_fts=False)

    def submit_fts[R](
        self, work: Callable[[psycopg.Connection, apsw.Connection], R]
    ) -> _Pending[R]:
        def with_sidecar(pg: psycopg.Connection, fts: apsw.Connection | None) -> R:
            assert fts is not None  # needs_fts=True always supplies one
            return work(pg, fts)

        return self._submit(with_sidecar, needs_fts=True)

    def _submit[R](
        self,
        work: Callable[[psycopg.Connection, apsw.Connection | None], R],
        *,
        needs_fts: bool,
    ) -> _Pending[R]:
        pending = _Pending(work, needs_fts=needs_fts)
        pools = self._pools
        if pools is None:
            pending.value = work(self._pg, self._fts)
            pending.has_value = True
            return pending

        pg = pools.pg.acquire_if_free()
        if pg is None:
            return pending
        fts: apsw.Connection | None = None
        if needs_fts:
            fts = pools.fts.acquire_if_free()
            if fts is None:
                pools.pg.release(pg)
                return pending

        def run(pg: psycopg.Connection = pg, fts: apsw.Connection | None = fts) -> R:
            try:
                return work(pg, fts)
            finally:
                pools.pg.release(pg)
                if fts is not None:
                    pools.fts.release(fts)

        try:
            future = pools.submit(run)
        except RuntimeError:  # the executor is shutting down: run it here instead
            pools.pg.release(pg)
            if fts is not None:
                pools.fts.release(fts)
            return pending
        self._outstanding.append(future)
        pending.future = future
        return pending

    def result[R](self, pending: _Pending[R]) -> R:
        if pending.has_value:
            return pending.value  # type: ignore[return-value]  # set with has_value
        if pending.future is not None:
            return pending.future.result()
        if not pending.needs_fts:
            return pending.work(self._pg, None)
        if self._fts is not None:
            return pending.work(self._pg, self._fts)
        assert self._pools is not None  # the pooled build has no call-owned sidecar
        with self._pools.fts.lease() as fts:
            return pending.work(self._pg, fts)

    def finish(self) -> None:
        wait_for_futures(self._outstanding)


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
        connections: RetrievalConnections | None = None,
    ) -> None:
        """`connections`, when given, is where every call borrows its
        connections, and `pg_conn`/`fts_conn` go unused by calls (module
        docstring: shared or pooled). The caller opened both and closes
        both.

        `model_thread`, when given, is where every model call runs —
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
        self._pools = connections
        self._shared_lock = threading.RLock()

    def _exclusive(self) -> AbstractContextManager[object]:
        """Hold the shared connections for the caller's whole scope — the
        module docstring has the two driver errors this prevents.

        Re-entrant so that a method reaching the database twice cannot
        deadlock itself. It is held across `_run_model` too, so a query
        keeps the connections while it waits on the model thread: that is
        the cost of the lock, and it is bounded by the model thread
        already being the serial stage. Only the shared build takes it."""
        return self._shared_lock

    @contextlib.contextmanager
    def _call_connection(self) -> Iterator[psycopg.Connection]:
        """The Postgres connection one call uses for its own steps: the
        shared one under the lock, or one borrowed from the pool for the
        length of the call."""
        if self._pools is None:
            with self._exclusive():
                yield self._pg
        else:
            with self._pools.pg.lease() as pg:
                yield pg

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
            with self._call_connection() as pg:
                return prewarm_query_path(pg).summary

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
        with self._marking(), self._call_connection() as pg:
            return self._search_messages(
                pg,
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
        pg: psycopg.Connection,
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

        scope = resolve_request_scope(pg, context)
        filters = resolve_filters(
            pg,
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

        # The five candidate channels (module docstring: shared or
        # pooled). Submitted in the old sequential order, collected in
        # it, and fused from a dict built in it — so neither which
        # connection a channel ran on nor the order channels finish in
        # can change a result.
        channels = _ChannelRun(self._pools, pg, self._fts if self._pools is None else None)
        try:
            seg_fts_run = channels.submit_fts(
                lambda cpg, fts: fts_search.search_segment_fts(fts, cpg, analyzed, predicate, k_fts)
            )
            att_fts_run = channels.submit_fts(
                lambda cpg, fts: fts_search.search_attachment_chunk_fts(
                    fts, cpg, analyzed, predicate, k_fts
                )
            )

            instruction = self._config.embedding.query_instruction
            query_vec = self._run_model(
                lambda: self._text_provider.embed_query(analyzed.phrase, instruction=instruction)
            )
            seg_vec_run = channels.submit_pg(
                lambda cpg: vector_search.search_segment_vector(
                    cpg, query_vec, predicate, k_vector, ef_search=ef_search
                )
            )
            att_vec_run = channels.submit_pg(
                lambda cpg: vector_search.search_attachment_chunk_vector(
                    cpg, query_vec, predicate, k_vector, ef_search=ef_search
                )
            )

            mm_run = None
            multimodal = self._multimodal_provider
            if self._config.embedding.multimodal.enabled and multimodal is not None:
                mm_query_vec = self._run_model(lambda: multimodal.embed_text(analyzed.phrase))
                mm_run = channels.submit_pg(
                    lambda cpg: vector_search.search_multimodal_vector(
                        cpg, mm_query_vec, predicate, k_vector, ef_search=ef_search
                    )
                )

            seg_fts = channels.result(seg_fts_run)
            att_fts = channels.result(att_fts_run)
            seg_vec = channels.result(seg_vec_run)
            att_vec = channels.result(att_vec_run)
            mm_channel = channels.result(mm_run) if mm_run is not None else None
        finally:
            channels.finish()

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
        summaries = segments.fetch_segment_summaries(pg, [r.segment_id for r in pool])
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
                pg,
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

        with self._call_connection() as pg:
            resolution = segments.resolve_thread(pg, thread_id)
            scope = resolve_request_scope(pg, context, among_chat_ids=[resolution.chat_id])
            if not scope.admits_chat(resolution.chat_id):
                # The same error, word for word, as an identifier that matches
                # nothing (segments.THREAD_NOT_FOUND_MESSAGE).
                raise NotFoundError(segments.THREAD_NOT_FOUND_MESSAGE)
            # A naive anchor means the owner's local time, as `after` and
            # `before` do (`imsg.retrieval.filters`), never the database
            # session's time zone.
            anchor_dt = segments.resolve_anchor(
                pg, resolution, anchor, timezone=self._config.render.timezone
            )
            messages = segments.fetch_conversation_window(
                pg,
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
        with self._call_connection() as pg:
            scope = resolve_request_scope(pg, context)
            listings = directory.list_people(
                pg, scope, query=query, limit=limit, include_handles=include_handles
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
        with self._call_connection() as pg:
            result = directory.get_attachment_text(pg, context, attachment_key)
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
