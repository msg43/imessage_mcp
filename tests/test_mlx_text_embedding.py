"""`imsg.embed.mlx_text.MlxTextEmbeddingProvider` — Qwen3-Embedding via
MLX, exercised against the dependency-free fakes in `_mlx_fakes`: the
pooling / truncation / normalisation arithmetic, the EOS + padding token
layout, instruction asymmetry, batching, and every failure path."""

from __future__ import annotations

import math

import pytest

from _mlx_fakes import (
    FakeModel,
    FakeRuntime,
    FakeTokenizer,
    fake_hidden_state,
    prefix_sums,
    uninstall_runtime,
)
from imsg.embed.mlx_text import (
    QUERY_TEMPLATE,
    MlxTextEmbeddingProvider,
    eos_suffix_for,
    format_query_text,
    matryoshka_normalize,
    pad_token_id_for,
    tokenize_for_embedding,
)
from imsg.errors import EmbeddingError, ImsgError
from imsg.mlx_runtime import MlxRuntimeError, MlxRuntimeUnavailableError

VOCAB = {"a": 1, "b": 2, "c": 3, "d": 4}
EOS = 7
PAD = 9


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _unit(vec: list[float]) -> list[float]:
    n = _norm(vec)
    return [v / n for v in vec]


def _tokenizer(**overrides: object) -> FakeTokenizer:
    kwargs: dict[str, object] = {"vocab": VOCAB, "eos_suffix": (EOS,), "pad_token_id": PAD}
    kwargs.update(overrides)
    return FakeTokenizer(**kwargs)  # type: ignore[arg-type]


def _provider(**overrides: object) -> MlxTextEmbeddingProvider:
    kwargs: dict[str, object] = {"dim": 3}
    kwargs.update(overrides)
    return MlxTextEmbeddingProvider("org/embed", "rev1", **kwargs)  # type: ignore[arg-type]


def _expected_vector(row: list[int], dim: int, hidden_size: int = 3) -> list[float]:
    """The fake transformer's state at the row's last token, truncated and normalised."""
    last = len(row) - 1
    return _unit(fake_hidden_state(prefix_sums(row)[last], last, hidden_size)[:dim])


# --- identity / construction ---------------------------------------------


def test_model_id_format_and_dim() -> None:
    provider = MlxTextEmbeddingProvider("org/embed", "rev1", 2048)
    assert provider.model_id == "org/embed@rev1"
    assert provider.dim == 2048
    assert MlxTextEmbeddingProvider("org/embed", None, 8).model_id == "org/embed@main"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_repo": "", "revision": None, "dim": 8},
        {"model_repo": "org/embed", "revision": None, "dim": 0},
        {"model_repo": "org/embed", "revision": None, "dim": 8, "batch_size": 0},
        {"model_repo": "org/embed", "revision": None, "dim": 8, "max_length": 1},
    ],
)
def test_constructor_rejects_invalid_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MlxTextEmbeddingProvider(**kwargs)  # type: ignore[arg-type]


def test_construction_does_not_import_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    uninstall_runtime(monkeypatch)
    provider = _provider()
    assert provider.is_loaded is False


# --- pure helpers ---------------------------------------------------------


def test_format_query_text_matches_qwen3_scheme() -> None:
    assert QUERY_TEMPLATE == "Instruct: {instruction}\nQuery: {text}"
    assert (
        format_query_text("Find the message that answers the question", "where is it")
        == "Instruct: Find the message that answers the question\nQuery: where is it"
    )


def test_matryoshka_normalize_truncates_first_then_normalizes() -> None:
    assert matryoshka_normalize([3.0, 4.0, 100.0, 100.0], 2) == pytest.approx([0.6, 0.8])
    assert matryoshka_normalize([1.0, 2.0, 2.0, 9.0], 3) == pytest.approx([1 / 3, 2 / 3, 2 / 3])
    full = matryoshka_normalize([1.0, 2.0, 2.0], 3)
    assert _norm(full) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("vector", "dim"),
    [
        ([1.0, 2.0], 3),  # narrower than dim
        ([0.0, 0.0, 0.0], 2),  # zero norm
        ([float("nan"), 1.0], 2),  # non-finite
        ([1.0, 2.0], 0),  # nonsensical dim
    ],
)
def test_matryoshka_normalize_rejects_unusable_vectors(vector: list[float], dim: int) -> None:
    with pytest.raises(EmbeddingError):
        matryoshka_normalize(vector, dim)


def test_eos_suffix_comes_from_tokenizer_template_else_eos_token_id() -> None:
    assert eos_suffix_for(_tokenizer(eos_suffix=(7,))) == [7]
    assert eos_suffix_for(_tokenizer(eos_suffix=(7, 8))) == [7, 8]
    assert eos_suffix_for(_tokenizer(eos_suffix=(), eos_token_id=9)) == [9]
    with pytest.raises(EmbeddingError):
        eos_suffix_for(_tokenizer(eos_suffix=(), eos_token_id=None))


def test_pad_token_id_falls_back_when_undefined() -> None:
    assert pad_token_id_for(_tokenizer(pad_token_id=4), fallback=7) == 4
    assert pad_token_id_for(_tokenizer(pad_token_id=None), fallback=7) == 7


def test_tokenize_for_embedding_truncates_text_then_appends_suffix() -> None:
    tokenizer = _tokenizer()
    assert tokenize_for_embedding(tokenizer, "a b c", max_length=8, eos_suffix=[EOS]) == [
        1,
        2,
        3,
        EOS,
    ]
    assert tokenize_for_embedding(tokenizer, "a b c", max_length=3, eos_suffix=[EOS]) == [
        1,
        2,
        EOS,
    ]
    assert tokenize_for_embedding(tokenizer, "", max_length=3, eos_suffix=[EOS]) == [EOS]
    assert tokenizer.encode_calls[-1] == ("", False)
    with pytest.raises(EmbeddingError):
        tokenize_for_embedding(tokenizer, "a", max_length=1, eos_suffix=[EOS])


# --- end-to-end against the fake runtime ---------------------------------


def test_embed_documents_pools_last_real_token_under_right_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(model=FakeModel(hidden_size=3), tokenizer=_tokenizer()).install(
        monkeypatch
    )
    provider = _provider(dim=3)
    vectors = provider.embed_documents(["a b c", "a"])

    # The batch is right-padded with the tokenizer's pad id; EOS closes each row.
    assert runtime.model.model.calls == [[[1, 2, 3, EOS], [1, EOS, PAD, PAD]]]
    # Row 0 pools position 3 (prefix sum 13), row 1 pools position 1 (prefix sum 8)
    # — not the padded position 3, whose fake state would be [8+2*9, ...].
    assert vectors[0] == pytest.approx(_unit([13.0, 16.0, 19.0]))
    assert vectors[1] == pytest.approx(_unit([8.0, 9.0, 10.0]))
    assert vectors[0] == pytest.approx(_expected_vector([1, 2, 3, EOS], 3))
    assert vectors[1] == pytest.approx(_expected_vector([1, EOS], 3))
    assert runtime.model.head.calls == 0  # base transformer only, never the LM head


def test_embeddings_are_truncated_to_dim_and_unit_length(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(model=FakeModel(hidden_size=5), tokenizer=_tokenizer()).install(monkeypatch)
    provider = _provider(dim=2)
    [vector] = provider.embed_documents(["a b"])
    # last position 2 (prefix sum 1+2+EOS = 10): fake state [10, 12, 14, 16, 18] -> [10, 12]
    assert len(vector) == 2
    assert vector == pytest.approx(_unit([10.0, 12.0]))
    assert _norm(vector) == pytest.approx(1.0)


def test_embed_query_applies_instruction_and_documents_stay_bare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(tokenizer=_tokenizer()).install(monkeypatch)
    provider = _provider()
    query_vector = provider.embed_query("hello", instruction="Retrieve the chat")
    provider.embed_documents(["hello"])
    texts = [text for text, _ in runtime.tokenizer.encode_calls if text]
    assert "Instruct: Retrieve the chat\nQuery: hello" in texts
    assert "hello" in texts
    assert all(add_special is False for text, add_special in runtime.tokenizer.encode_calls if text)
    assert len(query_vector) == 3
    assert query_vector != provider.embed_documents(["hello"])[0]


def test_batches_by_batch_size_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(tokenizer=_tokenizer()).install(monkeypatch)
    provider = _provider(batch_size=2)
    texts = ["a", "a b", "a b c", "b", "c d"]
    vectors = provider.embed_documents(texts)
    assert [len(call) for call in runtime.model.model.calls] == [2, 2, 1]
    assert len(vectors) == 5
    for text, vector in zip(texts, vectors, strict=True):
        assert vector == pytest.approx(provider.embed_documents([text])[0])


def test_truncation_keeps_eos_as_the_last_token(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(tokenizer=_tokenizer()).install(monkeypatch)
    provider = _provider(max_length=3)
    provider.embed_documents(["a b c d"])
    assert runtime.model.model.calls == [[[1, 2, EOS]]]


def test_loads_once_with_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(tokenizer=_tokenizer()).install(monkeypatch)
    provider = _provider()
    provider.embed_documents(["a"])
    provider.embed_query("b", instruction="x")
    provider.load()
    assert runtime.load_calls == [
        {"path": "org/embed", "tokenizer_config": None, "revision": "rev1"}
    ]
    assert provider.is_loaded is True


def test_empty_input_returns_empty_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime().install(monkeypatch)
    provider = _provider()
    assert provider.embed_documents([]) == []
    assert runtime.load_calls == []
    assert provider.is_loaded is False


def test_rejects_model_narrower_than_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(model=FakeModel(hidden_size=2), tokenizer=_tokenizer()).install(monkeypatch)
    provider = _provider(dim=4)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.embed_documents(["a"])
    assert "hidden size 2" in str(excinfo.value)
    assert "dim 4" in str(excinfo.value)


def test_rejects_max_length_that_leaves_no_room_for_text(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(tokenizer=_tokenizer(eos_suffix=(7, 8))).install(monkeypatch)
    provider = _provider(max_length=2)
    with pytest.raises(EmbeddingError):
        provider.embed_documents(["a"])


def test_missing_runtime_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    uninstall_runtime(monkeypatch)
    provider = _provider()
    with pytest.raises(MlxRuntimeUnavailableError) as excinfo:
        provider.embed_documents(["a"])
    assert isinstance(excinfo.value, ImsgError)
    assert "mlx" in str(excinfo.value)


def test_load_failure_surfaces_as_mlx_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(load_error=ValueError("Model type embedding not supported")).install(monkeypatch)
    provider = _provider()
    with pytest.raises(MlxRuntimeError) as excinfo:
        provider.embed_query("a", instruction="x")
    assert "org/embed@rev1" in str(excinfo.value)
