"""`imsg.retrieval.mlx_reranker.MlxRerankerProvider` — Qwen3-Reranker via
MLX against the `_mlx_fakes` runtime: the model-card prompt format, the
yes/no probability math, truncation budget, batching, and every failure
path."""

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
from imsg.errors import ImsgError
from imsg.mlx_runtime import MlxRuntimeError, MlxRuntimeUnavailableError
from imsg.retrieval.mlx_reranker import (
    DEFAULT_RERANK_INSTRUCTION,
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
    MlxRerankerProvider,
    format_reranker_pair,
    yes_probability,
)

YES_WEIGHT = 1e-3
NO_WEIGHT = 2.5e-4
HIDDEN = 3


def _model(**overrides: object) -> FakeModel:
    kwargs: dict[str, object] = {
        "hidden_size": HIDDEN,
        "head_weights": {FakeTokenizer.YES_ID: YES_WEIGHT, FakeTokenizer.NO_ID: NO_WEIGHT},
    }
    kwargs.update(overrides)
    return FakeModel(**kwargs)  # type: ignore[arg-type]


def _provider(**overrides: object) -> MlxRerankerProvider:
    return MlxRerankerProvider("org/rerank", "rev2", **overrides)  # type: ignore[arg-type]


def _expected_score(row: list[int]) -> float:
    """P(yes) the fake model yields for one token row: the LM head reads
    the fake state at the last real position and scales it by the yes/no
    weights."""
    last = len(row) - 1
    total = sum(fake_hidden_state(prefix_sums(row)[last], last, HIDDEN))
    return yes_probability(no_logit=NO_WEIGHT * total, yes_logit=YES_WEIGHT * total)


def _expected_row(tokenizer: FakeTokenizer, instruction: str, query: str, doc: str) -> list[int]:
    return (
        tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
        + tokenizer.encode(format_reranker_pair(instruction, query, doc), add_special_tokens=False)
        + tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    )


# --- identity / construction ---------------------------------------------


def test_model_id_and_default_instruction() -> None:
    provider = MlxRerankerProvider("org/rerank", "rev2")
    assert provider.model_id == "org/rerank@rev2"
    assert provider.instruction == DEFAULT_RERANK_INSTRUCTION
    assert MlxRerankerProvider("org/rerank", None).model_id == "org/rerank@main"
    assert MlxRerankerProvider("org/rerank", None, instruction="Custom").instruction == "Custom"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_repo": "", "revision": None},
        {"model_repo": "org/rerank", "revision": None, "batch_size": 0},
        {"model_repo": "org/rerank", "revision": None, "max_length": 0},
    ],
)
def test_constructor_rejects_invalid_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MlxRerankerProvider(**kwargs)  # type: ignore[arg-type]


# --- pure helpers ---------------------------------------------------------


def test_prompt_pieces_match_the_model_card() -> None:
    assert RERANKER_PREFIX == (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        'the Query and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n'
    )
    assert RERANKER_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert (
        format_reranker_pair("Find the passage", "capital of x", "The capital is y.")
        == "<Instruct>: Find the passage\n<Query>: capital of x\n<Document>: The capital is y."
    )


def test_yes_probability_is_a_two_way_softmax() -> None:
    assert yes_probability(no_logit=0.0, yes_logit=0.0) == pytest.approx(0.5)
    assert yes_probability(no_logit=0.0, yes_logit=math.log(3)) == pytest.approx(0.75)
    assert yes_probability(no_logit=math.log(3), yes_logit=0.0) == pytest.approx(0.25)
    assert yes_probability(no_logit=-1000.0, yes_logit=1000.0) == pytest.approx(1.0)
    assert yes_probability(no_logit=1000.0, yes_logit=-1000.0) == pytest.approx(0.0)
    a, b = 1.3, -0.4
    assert yes_probability(no_logit=a, yes_logit=b) + yes_probability(
        no_logit=b, yes_logit=a
    ) == pytest.approx(1.0)


# --- end-to-end against the fake runtime ---------------------------------


def test_score_returns_p_yes_per_document_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider()
    docs = ["short", "a much longer document body here", "mid length"]
    scores = provider.score("the query", docs)

    tokenizer = runtime.tokenizer
    rows = [_expected_row(tokenizer, DEFAULT_RERANK_INSTRUCTION, "the query", d) for d in docs]
    assert scores == pytest.approx([_expected_score(row) for row in rows])
    assert all(0.0 < s < 1.0 for s in scores)
    # One right-padded batch: each row is prefix + body + suffix, padded with pad_token_id.
    width = max(len(r) for r in rows)
    padded = [[*r, *([tokenizer.pad_token_id] * (width - len(r)))] for r in rows]
    assert runtime.model.model.calls == [padded]
    assert runtime.model.head.calls == 1


def test_batches_by_batch_size_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider(batch_size=2)
    docs = ["one", "two words", "three words here", "four", "five words in total"]
    scores = provider.score("q", docs)
    assert [len(call) for call in runtime.model.model.calls] == [2, 2, 1]
    for doc, score in zip(docs, scores, strict=True):
        assert score == pytest.approx(provider.score("q", [doc])[0])


def test_custom_instruction_is_rendered_into_the_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider(instruction="Judge topical fit")
    provider.score("q", ["d"])
    assert ("<Instruct>: Judge topical fit\n<Query>: q\n<Document>: d", False) in (
        runtime.tokenizer.encode_calls
    )


def test_body_is_truncated_to_the_budget_and_suffix_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    tokenizer = runtime.tokenizer
    prefix = tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    provider = _provider(max_length=len(prefix) + len(suffix) + 2)
    provider.score("q", ["one two three four five"])
    assert provider.body_token_budget == 2
    body = tokenizer.encode(
        format_reranker_pair(DEFAULT_RERANK_INSTRUCTION, "q", "one two three four five"),
        add_special_tokens=False,
    )
    assert runtime.model.model.calls == [[[*prefix, *body[:2], *suffix]]]


def test_rejects_max_length_smaller_than_the_fixed_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider(max_length=1)
    with pytest.raises(MlxRuntimeError) as excinfo:
        provider.score("q", ["d"])
    assert "prefix+suffix" in str(excinfo.value)


def test_uses_tied_embeddings_when_the_model_has_no_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(model=_model(tie_word_embeddings=True)).install(monkeypatch)
    provider = _provider()
    [score] = provider.score("q", ["d"])
    row = _expected_row(runtime.tokenizer, DEFAULT_RERANK_INSTRUCTION, "q", "d")
    assert score == pytest.approx(_expected_score(row))


@pytest.mark.parametrize("special_ids", [{"yes": None}, {"no": None}, {"yes": 6}])
def test_rejects_tokenizer_without_distinct_yes_no_tokens(
    monkeypatch: pytest.MonkeyPatch, special_ids: dict[str, int | None]
) -> None:
    FakeRuntime(model=_model(), tokenizer=FakeTokenizer(special_ids=special_ids)).install(
        monkeypatch
    )
    provider = _provider()
    with pytest.raises(MlxRuntimeError):
        provider.score("q", ["d"])


def test_rejects_yes_token_that_is_the_unknown_token(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(model=_model(), tokenizer=FakeTokenizer(unk_token_id=FakeTokenizer.YES_ID)).install(
        monkeypatch
    )
    with pytest.raises(MlxRuntimeError):
        _provider().score("q", ["d"])


def test_loads_once_with_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider()
    provider.score("q", ["d"])
    provider.score("q", ["e"])
    provider.load()
    assert runtime.load_calls == [
        {"path": "org/rerank", "tokenizer_config": None, "revision": "rev2"}
    ]
    assert provider.is_loaded is True


def test_empty_documents_return_empty_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider()
    assert provider.score("q", []) == []
    assert runtime.load_calls == []


def test_missing_runtime_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    uninstall_runtime(monkeypatch)
    provider = _provider()
    with pytest.raises(MlxRuntimeUnavailableError) as excinfo:
        provider.score("q", ["d"])
    assert isinstance(excinfo.value, ImsgError)
    assert "mlx" in str(excinfo.value)


def test_load_failure_surfaces_as_mlx_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(load_error=FileNotFoundError("No safetensors found")).install(monkeypatch)
    with pytest.raises(MlxRuntimeError) as excinfo:
        _provider().score("q", ["d"])
    assert "org/rerank@rev2" in str(excinfo.value)
