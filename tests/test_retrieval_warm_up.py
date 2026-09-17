"""`RetrievalService.warm_up`: one throwaway input through every model
provider, no database, so a server's first real query is not the one that
loads the weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from imsg.errors import ImsgError
from imsg.retrieval.service import WARM_UP_DOCUMENT, WARM_UP_QUERY, RetrievalService


class _NoDatabase:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"warm_up must not touch a connection (asked for {name!r})")


class _Text:
    model_id = "fake/text@warm"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("warm_up embeds a query, not documents")

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        self.calls.append((text, instruction))
        return [0.5] * 4


class _Multimodal:
    model_id = "fake/mm@warm"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        raise AssertionError("warm_up never embeds images")

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.5] * 4


class _Reranker:
    model_id = "fake/reranker@warm"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.error = error

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self.error is not None:
            raise self.error
        return [0.5] * len(documents)


def _config(*, multimodal: bool) -> Any:
    class _Multi:
        enabled = multimodal

    class _Embedding:
        query_instruction = "find the conversation"
        multimodal = _Multi()

    class _Cfg:
        embedding = _Embedding()

    return _Cfg()


def _service(multimodal_enabled: bool, reranker: _Reranker, text: _Text, mm: _Multimodal | None) -> RetrievalService:
    return RetrievalService(
        pg_conn=cast(Any, _NoDatabase()),
        fts_conn=cast(Any, _NoDatabase()),
        config=_config(multimodal=multimodal_enabled),
        text_provider=text,
        reranker=reranker,
        multimodal_provider=mm,
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
