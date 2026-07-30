"""Fake embedding providers (SPEC §4.1, §6, §8 S6) — deterministic
stand-ins; no model weights in this build environment."""

from __future__ import annotations

import math
from pathlib import Path

from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def test_embed_documents_returns_correct_dim_and_normalized() -> None:
    provider = FakeTextEmbeddingProvider(dim=2048)
    vecs = provider.embed_documents(["hello world", "goodbye world"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == 2048
        assert abs(_norm(v) - 1.0) < 1e-9


def test_embed_documents_is_deterministic() -> None:
    provider = FakeTextEmbeddingProvider(dim=64)
    assert provider.embed_documents(["same text"]) == provider.embed_documents(["same text"])


def test_embed_documents_differs_by_content() -> None:
    provider = FakeTextEmbeddingProvider(dim=64)
    a = provider.embed_documents(["text a"])[0]
    b = provider.embed_documents(["text b"])[0]
    assert a != b


def test_embed_query_differs_from_embed_documents_for_same_text() -> None:
    """Instruction-asymmetric scheme (Qwen3): a query and a document
    with identical text must NOT collapse to the same vector."""
    provider = FakeTextEmbeddingProvider(dim=64)
    doc_vec = provider.embed_documents(["find the deck bid"])[0]
    query_vec = provider.embed_query("find the deck bid", instruction="retrieve relevant segments")
    assert doc_vec != query_vec


def test_embed_query_differs_by_instruction() -> None:
    provider = FakeTextEmbeddingProvider(dim=64)
    a = provider.embed_query("same text", instruction="instruction A")
    b = provider.embed_query("same text", instruction="instruction B")
    assert a != b


def test_multimodal_embed_images_returns_correct_dim_and_normalized(tmp_path: Path) -> None:
    provider = FakeMultimodalEmbeddingProvider(dim=1280)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake jpeg bytes")
    vecs = provider.embed_images([img])
    assert len(vecs) == 1
    assert len(vecs[0]) == 1280
    assert abs(_norm(vecs[0]) - 1.0) < 1e-9


def test_multimodal_embed_images_deterministic_by_path(tmp_path: Path) -> None:
    provider = FakeMultimodalEmbeddingProvider(dim=32)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"data")
    assert provider.embed_images([img]) == provider.embed_images([img])
