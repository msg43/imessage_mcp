"""`RetrievalService.warm_up`: one throwaway input through every model
provider and then the database's buffer pool pulled in behind them, so a
server's first real query pays for neither loading weights nor reading
index pages from disk."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from imsg.db.prewarm import PREWARM_UNAVAILABLE, PrewarmReport
from imsg.errors import ImsgError
from imsg.retrieval import service as service_module
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import (
    BUFFER_POOL_STEP,
    ESTIMATED_WARM_UP_SECONDS,
    MULTIMODAL_STEP,
    RERANKER_STEP,
    TEXT_EMBEDDER_STEP,
    WARM_UP_DOCUMENT,
    WARM_UP_QUERY,
    RetrievalService,
)


class _NoDatabase:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"warm_up must not touch a connection (asked for {name!r})")


class _PoolCursor:
    """Answers only the `pg_prewarm` availability probe, with "no" — so the
    prewarm step runs for real against this connection and reports that the
    extension is absent, without a database."""

    def __init__(self, conn: _BufferPool) -> None:
        self._conn = conn

    def __enter__(self) -> _PoolCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        assert "proname = 'pg_prewarm'" in sql, f"unexpected statement during warm-up: {sql}"
        self._conn.probes += 1

    def fetchone(self) -> tuple[Any, ...]:
        return (False,)


class _BufferPool:
    def __init__(self) -> None:
        self.probes = 0

    def cursor(self) -> _PoolCursor:
        return _PoolCursor(self)


class _Text:
    model_id = "fake/text@warm"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.threads: list[threading.Thread] = []
        self.on_call: Callable[[str], None] | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("warm_up embeds a query, not documents")

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        self.calls.append((text, instruction))
        self.threads.append(threading.current_thread())
        if self.on_call is not None:
            self.on_call(text)
        return [0.5] * 4


class _Multimodal:
    model_id = "fake/mm@warm"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.threads: list[threading.Thread] = []

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        raise AssertionError("warm_up never embeds images")

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        self.threads.append(threading.current_thread())
        return [0.5] * 4


class _Reranker:
    model_id = "fake/reranker@warm"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.threads: list[threading.Thread] = []
        self.error = error

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        self.threads.append(threading.current_thread())
        if self.error is not None:
            raise self.error
        return [0.5] * len(documents)


def _config(*, multimodal: bool) -> Any:
    return SimpleNamespace(
        embedding=SimpleNamespace(
            query_instruction="find the conversation",
            multimodal=SimpleNamespace(enabled=multimodal),
        ),
        retrieval=SimpleNamespace(
            default_limit=5, k_fts=10, k_vector=10, rrf_k=60, rerank_top=5, hnsw_ef_search=1000
        ),
        render=SimpleNamespace(timezone="America/Los_Angeles"),
    )


def _service(
    multimodal_enabled: bool,
    reranker: _Reranker,
    text: _Text,
    mm: _Multimodal | None,
    model_thread: ModelThread | None = None,
    pg_conn: Any = None,
    query_marker: Any = None,
) -> RetrievalService:
    return RetrievalService(
        pg_conn=cast(Any, pg_conn if pg_conn is not None else _BufferPool()),
        fts_conn=cast(Any, _NoDatabase()),
        config=_config(multimodal=multimodal_enabled),
        text_provider=text,
        reranker=reranker,
        multimodal_provider=mm,
        model_thread=model_thread,
        query_marker=query_marker,
    )


def test_warm_up_runs_one_input_through_every_provider() -> None:
    text, mm, reranker = _Text(), _Multimodal(), _Reranker()
    seconds = _service(True, reranker, text, mm).warm_up()
    assert seconds >= 0.0
    assert text.calls == [(WARM_UP_QUERY, "find the conversation")]
    assert mm.calls == [WARM_UP_QUERY]
    assert reranker.calls == [(WARM_UP_QUERY, [WARM_UP_DOCUMENT])]


@pytest.mark.parametrize("enabled, provider", [(False, _Multimodal()), (True, None)])
def test_warm_up_skips_a_disabled_or_absent_multimodal_channel(
    enabled: bool, provider: _Multimodal | None
) -> None:
    text, reranker = _Text(), _Reranker()
    _service(enabled, reranker, text, provider).warm_up()
    assert len(text.calls) == 1 and len(reranker.calls) == 1
    if provider is not None:
        assert provider.calls == []


def test_warm_up_surfaces_a_provider_that_cannot_load() -> None:
    reranker = _Reranker(error=ImsgError("weights missing"))
    with pytest.raises(ImsgError, match="weights missing"):
        _service(True, reranker, _Text(), _Multimodal()).warm_up()


def test_warm_up_steps_name_each_model_and_its_estimated_duration() -> None:
    steps = _service(True, _Reranker(), _Text(), _Multimodal()).warm_up_steps()
    assert [(step.name, step.estimated_seconds) for step in steps] == [
        (TEXT_EMBEDDER_STEP, ESTIMATED_WARM_UP_SECONDS[TEXT_EMBEDDER_STEP]),
        (MULTIMODAL_STEP, ESTIMATED_WARM_UP_SECONDS[MULTIMODAL_STEP]),
        (RERANKER_STEP, ESTIMATED_WARM_UP_SECONDS[RERANKER_STEP]),
        (BUFFER_POOL_STEP, ESTIMATED_WARM_UP_SECONDS[BUFFER_POOL_STEP]),
    ]
    text_only = _service(False, _Reranker(), _Text(), _Multimodal()).warm_up_steps()
    assert [step.name for step in text_only] == [
        TEXT_EMBEDDER_STEP,
        RERANKER_STEP,
        BUFFER_POOL_STEP,
    ]


def test_the_buffer_pool_is_warmed_last_on_the_service_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The models come first (nothing is answerable without them); the pool
    is pulled in behind them, on the service's own connection, and the step
    reports what it moved so the server's log says so."""
    pool = _BufferPool()
    calls: list[Any] = []
    report = PrewarmReport(
        relations=17, blocks=294_000, bytes_prewarmed=294_000 * 8192, seconds=16.2
    )

    def fake_prewarm(conn: Any) -> PrewarmReport:
        calls.append(conn)
        return report

    monkeypatch.setattr(service_module, "prewarm_query_path", fake_prewarm)
    service = _service(True, _Reranker(), _Text(), _Multimodal(), pg_conn=pool)
    steps = service.warm_up_steps()
    assert steps[-1].name == BUFFER_POOL_STEP
    assert steps[-1].run() == report.summary
    assert calls == [pool]


def test_a_buffer_pool_that_cannot_be_warmed_does_not_fail_the_warm_up() -> None:
    """A cold pool is slow, not wrong: the step reports the cause and the
    server still serves (contrast a model that cannot load, above)."""
    pool = _BufferPool()
    service = _service(True, _Reranker(), _Text(), _Multimodal(), pg_conn=pool)
    assert service.warm_up() >= 0.0
    assert pool.probes == 1
    assert service.warm_up_steps()[-1].run() == PREWARM_UNAVAILABLE


def test_with_a_model_thread_every_model_call_runs_there() -> None:
    """The warm-up and each query's three model calls all run on the
    service's model thread, whichever thread asks — MLX cannot evaluate,
    on one thread, work that was set up on another
    (imsg.retrieval.model_thread)."""
    text, mm, reranker = _Text(), _Multimodal(), _Reranker()
    model_thread = ModelThread(name="test-service-models")
    try:
        service = _service(True, reranker, text, mm, model_thread=model_thread)
        service.warm_up()
        _search_without_a_database(service)
    finally:
        model_thread.close()

    assert len(text.threads) == len(mm.threads) == len(reranker.threads) == 2
    assert set(text.threads + mm.threads + reranker.threads) == {model_thread.thread}


def test_without_a_model_thread_models_run_on_the_calling_thread() -> None:
    text, mm, reranker = _Text(), _Multimodal(), _Reranker()
    service = _service(True, reranker, text, mm)
    service.warm_up()
    _search_without_a_database(service)
    assert set(text.threads + mm.threads + reranker.threads) == {threading.current_thread()}


def _search_without_a_database(service: RetrievalService) -> None:
    """Run `search_messages` with every database-facing helper stubbed, so
    only its model calls are real (fakes)."""
    channel = SimpleNamespace(segment_ids=(7,), scan_cap_reached=False)
    summary = SimpleNamespace(
        segment_key="seg_fictional",
        thread_key="thread_fictional",
        chat_kind="direct",
        chat_display_name=None,
        people=("alice",),
        started_at=datetime(2026, 1, 2, tzinfo=UTC),
        ended_at=datetime(2026, 1, 2, tzinfo=UTC),
        message_count=3,
        has_attachments=False,
        text="[09:00] alice: kite festival on saturday?",
    )
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(service_module, "resolve_filters", lambda *a, **k: None)
        patch.setattr(service_module, "compile_predicate", lambda *a, **k: None)
        for name in ("search_segment_fts", "search_attachment_chunk_fts"):
            patch.setattr(service_module.fts_search, name, lambda *a, **k: channel)
        for name in (
            "search_segment_vector",
            "search_attachment_chunk_vector",
            "search_multimodal_vector",
        ):
            patch.setattr(service_module.vector_search, name, lambda *a, **k: channel)
        patch.setattr(
            service_module.segments, "fetch_segment_summaries", lambda *a, **k: {7: summary}
        )
        result = service.search_messages(LOCAL_FULL_ACCESS, query="kite festival")
    finally:
        patch.undo()
    assert [r["segment_key"] for r in result.results] == ["seg_fictional"]


# --------------------------------------------------------------------------
# the query-in-flight marker (D10.3): enrichment has to be able to see that
# the GPU is busy on someone's behalf
# --------------------------------------------------------------------------


class _RecordingMarker:
    """Stands in for `imsg.db.enrichment_yield_locks.QueryInFlightMarker`,
    recording when the service considers a query to be in flight."""

    def __init__(self) -> None:
        self.depth = 0
        self.entries = 0
        self.max_depth = 0
        self.model_calls_while_marked = 0

    def __enter__(self) -> _RecordingMarker:
        self.depth += 1
        self.entries += 1
        self.max_depth = max(self.max_depth, self.depth)
        return self

    def __exit__(self, *exc: object) -> None:
        self.depth -= 1


def test_every_warm_up_model_call_runs_inside_the_query_marker() -> None:
    """Warm-up is the same GPU work a query does, so an enrichment worker
    must see it as busy — otherwise it starts a batch against a server
    that is still loading weights."""
    marker = _RecordingMarker()
    text, mm, reranker = _Text(), _Multimodal(), _Reranker()

    def _note(_: object) -> None:
        assert marker.depth > 0, "a model ran with no query-in-flight marker held"
        marker.model_calls_while_marked += 1

    text.on_call = _note
    _service(True, reranker, text, mm, query_marker=marker).warm_up()

    assert marker.model_calls_while_marked == 1
    assert marker.depth == 0  # balanced: every enter had its exit
    assert marker.entries >= 3  # embedder, multimodal tower, reranker


def test_a_service_without_a_marker_still_warms_up() -> None:
    """The marker is optional wiring; nothing about answering queries may
    depend on it."""
    text, mm, reranker = _Text(), _Multimodal(), _Reranker()
    _service(True, reranker, text, mm, query_marker=None).warm_up()
    assert text.calls == [(WARM_UP_QUERY, "find the conversation")]
