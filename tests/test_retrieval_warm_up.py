"""`RetrievalService.warm_up`: one throwaway input through every model
provider, no database, so a server's first real query is not the one that
loads the weights."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from imsg.errors import ImsgError
from imsg.retrieval import service as service_module
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import (
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


class _Text:
    model_id = "fake/text@warm"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.threads: list[threading.Thread] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("warm_up embeds a query, not documents")

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        self.calls.append((text, instruction))
        self.threads.append(threading.current_thread())
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
        retrieval=SimpleNamespace(default_limit=5, k_fts=10, k_vector=10, rrf_k=60, rerank_top=5),
        render=SimpleNamespace(timezone="America/Los_Angeles"),
    )


def _service(
    multimodal_enabled: bool,
    reranker: _Reranker,
    text: _Text,
    mm: _Multimodal | None,
    model_thread: ModelThread | None = None,
) -> RetrievalService:
    return RetrievalService(
        pg_conn=cast(Any, _NoDatabase()),
        fts_conn=cast(Any, _NoDatabase()),
        config=_config(multimodal=multimodal_enabled),
        text_provider=text,
        reranker=reranker,
        multimodal_provider=mm,
        model_thread=model_thread,
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
    ]
    text_only = _service(False, _Reranker(), _Text(), _Multimodal()).warm_up_steps()
    assert [step.name for step in text_only] == [TEXT_EMBEDDER_STEP, RERANKER_STEP]


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
