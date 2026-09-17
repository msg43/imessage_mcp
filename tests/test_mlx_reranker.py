"""`imsg.retrieval.mlx_reranker.MlxRerankerProvider` — Qwen3-Reranker via
MLX against the `_mlx_fakes` runtime: the model-card prompt format, the
yes/no probability math, truncation budget and document cap, planned
(length-sorted, token-budgeted) batching, the buffer-cache bound, and
every failure path."""

from __future__ import annotations

import math
from pathlib import Path

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
from imsg.mlx_runtime import (
    DEFAULT_CACHE_LIMIT_BYTES,
    MlxRuntimeError,
    MlxRuntimeUnavailableError,
)
from imsg.retrieval.mlx_reranker import (
    DEFAULT_RERANK_INSTRUCTION,
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
    MlxRerankerProvider,
    common_prefix_length,
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
        {"model_repo": "org/rerank", "revision": None, "max_batch_tokens": 0},
        {"model_repo": "org/rerank", "revision": None, "doc_max_tokens": 0},
        {"model_repo": "org/rerank", "revision": None, "doc_max_tokens": True},
        {"model_repo": "org/rerank", "revision": None, "cache_limit_bytes": -1},
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
    # One right-padded batch, longest row first: each row is prefix + body +
    # suffix, padded with pad_token_id — and the scores still come back in
    # input order.
    width = max(len(r) for r in rows)
    padded = [[*r, *([tokenizer.pad_token_id] * (width - len(r)))] for r in rows]
    assert runtime.model.model.calls == [[padded[1], padded[2], padded[0]]]
    assert runtime.model.head.calls == 1


def test_batches_by_batch_size_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider(batch_size=2)
    docs = ["one", "two words", "three words here", "four", "five words in total"]
    scores = provider.score("q", docs)
    assert [len(call) for call in runtime.model.model.calls] == [2, 2, 1]
    rows = provider.token_rows("q", docs)
    # Longest first: the first forward pass holds the two longest pairs
    # ("five words in total", then "three words here"), the longer unpadded.
    first = runtime.model.model.calls[0]
    assert first[0] == rows[4]
    assert first[1][: len(rows[2])] == rows[2]
    for doc, score in zip(docs, scores, strict=True):
        assert score == pytest.approx(provider.score("q", [doc])[0])


def _real_lengths(call: list[list[int]], pad_id: int, suffix: list[int]) -> list[int]:
    """Each padded row's real length: through the end of its suffix."""
    out = []
    for row in call:
        n = len(row)
        while row[n - len(suffix) : n] != suffix:
            assert row[n - 1] == pad_id, "only padding may follow the suffix"
            n -= 1
        out.append(n)
    return out


def test_scores_do_not_depend_on_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider(batch_size=3, max_batch_tokens=200)
    docs = [
        "alpha",
        "bravo charlie delta echo foxtrot golf hotel",
        "india juliet",
        "kilo lima mike november",
        "oscar",
        "papa quebec romeo sierra tango uniform victor whiskey",
        "xray yankee zulu",
    ]
    forward = provider.score("the query", docs)
    reverse = provider.score("the query", list(reversed(docs)))
    assert list(reversed(reverse)) == pytest.approx(forward)
    shuffled_order = [3, 0, 6, 2, 5, 1, 4]
    shuffled = provider.score("the query", [docs[i] for i in shuffled_order])
    for position, index in enumerate(shuffled_order):
        assert shuffled[position] == pytest.approx(forward[index])
    alone = [provider.score("the query", [doc])[0] for doc in docs]
    assert forward == pytest.approx(alone)


def test_batches_respect_the_padded_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    budget = 120
    provider = _provider(batch_size=32, max_batch_tokens=budget)
    docs = [" ".join(["word"] * n) for n in (1, 30, 4, 12, 2, 25, 7, 3, 18, 9)]
    rows = provider.token_rows("q", docs)
    assert max(len(r) for r in rows) <= budget  # nothing here is oversize on its own
    provider.score("q", docs)
    calls = runtime.model.model.calls
    assert len(calls) > 1
    scored = 0
    for call in calls:
        width = len(call[0])
        assert all(len(row) == width for row in call)
        assert len(call) * width <= budget
        scored += len(call)
    assert scored == len(docs)
    widths = [len(call[0]) for call in calls]
    assert widths == sorted(widths, reverse=True)  # the longest batch comes first


def test_an_oversize_pair_is_scored_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    tokenizer = runtime.tokenizer
    provider = _provider(batch_size=32, max_batch_tokens=60)
    long_doc = " ".join(["word"] * 80)
    docs = ["one", long_doc, "two words", "three words here"]
    scores = provider.score("q", docs)
    rows = provider.token_rows("q", docs)
    assert len(rows[1]) > 60
    calls = runtime.model.model.calls
    assert calls[0] == [rows[1]]  # alone, unpadded, first
    assert all(rows[1] not in call for call in calls[1:])
    assert scores[1] == pytest.approx(
        _expected_score(_expected_row(tokenizer, DEFAULT_RERANK_INSTRUCTION, "q", long_doc))
    )


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


def test_document_longer_than_max_length_scores_from_its_last_real_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real corpus has a segment of ~8,600 tokens against a
    ``max_length`` of 8,192: its body is cut, the suffix is still
    appended, and the logits are read at the suffix's last token — also
    when it shares a padded batch with shorter pairs."""
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    tokenizer = runtime.tokenizer
    prefix = tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    body_budget = 20
    max_length = len(prefix) + len(suffix) + body_budget
    provider = _provider(max_length=max_length, batch_size=32, max_batch_tokens=10_000)
    long_doc = " ".join(f"w{i}" for i in range(200))
    docs = ["one", long_doc, "two words"]
    scores = provider.score("the query", docs)

    body = tokenizer.encode(
        format_reranker_pair(DEFAULT_RERANK_INSTRUCTION, "the query", long_doc),
        add_special_tokens=False,
    )
    assert len(body) > body_budget
    expected_long = [*prefix, *body[:body_budget], *suffix]
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None
    [call] = runtime.model.model.calls  # one padded batch holds all three
    assert call[0] == expected_long and len(call[0]) == max_length
    assert _real_lengths(call, pad_id, suffix)[0] == max_length
    assert scores[1] == pytest.approx(_expected_score(expected_long))
    for index in (0, 2):
        row = _expected_row(tokenizer, DEFAULT_RERANK_INSTRUCTION, "the query", docs[index])
        assert scores[index] == pytest.approx(_expected_score(row))
        # these rows are padded in the batch, and reading a padding slot
        # instead of the suffix's last token would give a different score
        assert len(row) < max_length
        padded = [*row, *([pad_id] * (max_length - len(row)))]
        assert scores[index] != pytest.approx(_expected_score(padded))


def test_no_document_cap_keeps_the_reference_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider()
    docs = ["one two three", "a b c d e f g h i j k l m n o p"]
    assert provider.doc_max_tokens is None
    assert provider.token_rows("q", docs) == [
        _expected_row(runtime.tokenizer, DEFAULT_RERANK_INSTRUCTION, "q", d) for d in docs
    ]


def test_document_cap_cuts_only_the_document_and_keeps_the_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    tokenizer = runtime.tokenizer
    prefix = tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    provider = _provider(doc_max_tokens=3)
    instruction_query = "a long instruction-free query with many words in it"
    docs = ["d1 d22 d333 d4444 d55555 d666666", "tiny"]
    rows = provider.token_rows(instruction_query, docs)

    head = tokenizer.encode(
        format_reranker_pair(DEFAULT_RERANK_INSTRUCTION, instruction_query, ""),
        add_special_tokens=False,
    )
    long_doc_ids = tokenizer.encode(docs[0], add_special_tokens=False)
    assert rows[0] == [*prefix, *head, *long_doc_ids[:3], *suffix]
    # the whole query and instruction survive; only the document was cut
    assert rows[0][len(prefix) : len(prefix) + len(head)] == head
    # a document already under the cap is untouched
    assert rows[1] == _expected_row(tokenizer, DEFAULT_RERANK_INSTRUCTION, instruction_query, "tiny")
    assert all(row[-len(suffix) :] == suffix for row in rows)

    scores = provider.score(instruction_query, docs)
    assert scores[0] == pytest.approx(_expected_score(rows[0]))


def test_both_caps_together_keep_the_suffix_and_the_tighter_cap_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    tokenizer = runtime.tokenizer
    prefix = tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    doc = " ".join(f"t{i}" for i in range(40))
    head = tokenizer.encode(format_reranker_pair(DEFAULT_RERANK_INSTRUCTION, "q", ""), False)
    body = tokenizer.encode(format_reranker_pair(DEFAULT_RERANK_INSTRUCTION, "q", doc), False)

    # max_length is the tighter cap: the body is cut at the budget.
    tight_length = len(prefix) + len(suffix) + len(head) + 5
    provider = _provider(max_length=tight_length, doc_max_tokens=30)
    [row] = provider.token_rows("q", [doc])
    assert row == [*prefix, *body[: len(head) + 5], *suffix]

    # the document cap is the tighter one.
    provider = _provider(doc_max_tokens=7)
    [row] = provider.token_rows("q", [doc])
    assert row == [*prefix, *body[: len(head) + 7], *suffix]
    assert row[-len(suffix) :] == suffix


def test_document_cap_is_settable_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(model=_model()).install(monkeypatch)
    provider = _provider()
    doc = " ".join(f"t{i}" for i in range(12))
    uncapped = provider.token_rows("q", [doc])[0]
    provider.doc_max_tokens = 2
    assert provider.doc_max_tokens == 2
    capped = provider.token_rows("q", [doc])[0]
    assert len(uncapped) - len(capped) == 10
    provider.doc_max_tokens = None
    assert provider.token_rows("q", [doc])[0] == uncapped
    for bad in (0, -3, True):
        with pytest.raises(ValueError, match="doc_max_tokens"):
            provider.doc_max_tokens = bad


def test_common_prefix_length() -> None:
    assert common_prefix_length([], [1, 2]) == 0
    assert common_prefix_length([1, 2, 3], [1, 2, 3, 4]) == 3
    assert common_prefix_length([1, 2, 9], [1, 2, 3, 4]) == 2
    assert common_prefix_length([5], [1]) == 0


def test_bounds_the_mlx_buffer_cache_when_the_weights_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime(model=_model()).install(monkeypatch)
    mx = __import__("sys").modules["mlx.core"]
    provider = _provider()
    assert mx.cache_limit_calls == []
    provider.score("q", ["d"])
    provider.load()
    assert mx.cache_limit_calls == [DEFAULT_CACHE_LIMIT_BYTES]


def test_cache_bound_already_set_by_the_embedder_is_left_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime(model=_model()).install(monkeypatch)
    mx = __import__("sys").modules["mlx.core"]
    mx.set_cache_limit(2 * 2**30)  # another provider in this process bound it tighter
    _provider().load()
    assert mx.cache_limit_calls[-1] == 2 * 2**30


def test_cache_limit_none_leaves_the_runtime_default(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(model=_model()).install(monkeypatch)
    _provider(cache_limit_bytes=None).load()
    assert __import__("sys").modules["mlx.core"].cache_limit_calls == []


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
        {"path": "org/rerank", "tokenizer_config": None, "model_config": None, "revision": "rev2"}
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


# --- a local conversion: directory, no revision, relative model_id ---------


def test_model_id_override_records_the_relative_dir_not_the_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = FakeRuntime(model=_model()).install(monkeypatch)
    directory = tmp_path / "models" / "example-reranker-mxfp8"
    directory.mkdir(parents=True)
    provider = MlxRerankerProvider(str(directory), None, model_id="models/example-reranker-mxfp8@rev9")
    assert provider.model_id == "models/example-reranker-mxfp8@rev9"
    provider.score("q", ["d"])
    # Loaded from the directory, with no revision to pin.
    assert runtime.load_calls == [
        {"path": str(directory), "tokenizer_config": None, "model_config": None, "revision": None}
    ]
    with pytest.raises(ValueError, match="model_id"):
        MlxRerankerProvider("org/rerank", None, model_id="   ")
