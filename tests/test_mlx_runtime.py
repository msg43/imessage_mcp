"""`imsg.mlx_runtime`: lazy import, revision pinning across mlx_lm
versions, right padding and the last-real-token gather the real
providers share. No MLX runtime is installed here; see `_mlx_fakes`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from _mlx_fakes import (
    FAKE_RUNTIME_DEFAULT_CACHE_LIMIT,
    FakeArray,
    FakeModel,
    FakeRuntime,
    make_mx_module,
    uninstall_runtime,
)
from imsg.errors import ImsgError
from imsg.mlx_runtime import (
    DEFAULT_CACHE_LIMIT_BYTES,
    MlxRuntimeError,
    MlxRuntimeUnavailableError,
    base_transformer_hidden_states,
    batched,
    bound_buffer_cache,
    float_rows,
    format_model_id,
    gather_last_token_states,
    hidden_size_of,
    import_mlx_core,
    import_mlx_lm,
    lm_head_logits,
    load_model_and_tokenizer,
    right_pad,
)

# --- lazy import ----------------------------------------------------------


def test_missing_runtime_raises_clear_imsg_error(monkeypatch: pytest.MonkeyPatch) -> None:
    uninstall_runtime(monkeypatch)
    with pytest.raises(MlxRuntimeUnavailableError) as excinfo:
        import_mlx_lm()
    assert isinstance(excinfo.value, ImsgError)
    assert "mlx_lm" in str(excinfo.value)
    assert "Fake*" in str(excinfo.value)
    with pytest.raises(MlxRuntimeUnavailableError):
        import_mlx_core()


def test_installed_fake_runtime_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime().install(monkeypatch)
    assert hasattr(import_mlx_lm(), "load")
    assert hasattr(import_mlx_core(), "take_along_axis")


# --- the process-wide buffer-cache bound -----------------------------------


def test_bound_buffer_cache_applies_the_bound_over_the_runtime_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime().install(monkeypatch)
    mx = import_mlx_core()
    assert FAKE_RUNTIME_DEFAULT_CACHE_LIMIT > DEFAULT_CACHE_LIMIT_BYTES
    bound_buffer_cache(DEFAULT_CACHE_LIMIT_BYTES)
    assert mx.cache_limit_calls == [DEFAULT_CACHE_LIMIT_BYTES]
    bound_buffer_cache(DEFAULT_CACHE_LIMIT_BYTES)  # a second provider, same bound
    assert mx.cache_limit_calls == [DEFAULT_CACHE_LIMIT_BYTES, DEFAULT_CACHE_LIMIT_BYTES]


def test_bound_buffer_cache_never_loosens_a_tighter_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime().install(monkeypatch)
    mx = import_mlx_core()
    bound_buffer_cache(2**30)
    bound_buffer_cache(DEFAULT_CACHE_LIMIT_BYTES)
    assert mx.cache_limit_calls == [2**30, DEFAULT_CACHE_LIMIT_BYTES, 2**30]
    bound_buffer_cache(0)  # caching disabled is tighter still
    bound_buffer_cache(2**30)
    assert mx.cache_limit_calls[-1] == 0


def test_bound_buffer_cache_tolerates_a_runtime_without_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime().install(monkeypatch)
    monkeypatch.delattr(import_mlx_core(), "set_cache_limit")
    bound_buffer_cache(DEFAULT_CACHE_LIMIT_BYTES)  # no error
    with pytest.raises(ValueError):
        bound_buffer_cache(-1)


# --- model id / loading ---------------------------------------------------


def test_format_model_id() -> None:
    assert format_model_id("org/model", "abc123") == "org/model@abc123"
    assert format_model_id("org/model", None) == "org/model@main"
    assert format_model_id("org/model", "") == "org/model@main"


def test_load_passes_revision_when_load_supports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(load_supports_revision=True).install(monkeypatch)
    model, tokenizer = load_model_and_tokenizer("org/model", "abc123")
    assert model is runtime.model
    assert tokenizer is runtime.tokenizer
    assert runtime.load_calls == [
        {"path": "org/model", "tokenizer_config": None, "model_config": None, "revision": "abc123"}
    ]
    assert runtime.get_model_path_calls == []
    assert runtime.snapshot_calls == []


def test_load_resolves_pinned_snapshot_via_get_model_path_on_older_mlx_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(load_supports_revision=False).install(monkeypatch)
    load_model_and_tokenizer("org/model", "abc123")
    assert runtime.get_model_path_calls == [("org/model", "abc123")]
    assert runtime.snapshot_calls == []
    assert runtime.load_calls[0]["path"] == "/fake/snapshots/org/model/abc123"
    assert runtime.load_calls[0]["revision"] is None


def test_load_falls_back_to_huggingface_hub_snapshot_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(load_supports_revision=False, expose_get_model_path=False).install(
        monkeypatch
    )
    load_model_and_tokenizer("org/model", "abc123")
    assert len(runtime.snapshot_calls) == 1
    call = runtime.snapshot_calls[0]
    assert call["repo_id"] == "org/model"
    assert call["revision"] == "abc123"
    assert "*.safetensors" in call["allow_patterns"]
    assert runtime.load_calls[0]["path"] == "/fake/hub/org/model/abc123"


def test_load_without_revision_never_resolves_a_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(load_supports_revision=False).install(monkeypatch)
    load_model_and_tokenizer("org/model", None)
    assert runtime.load_calls[0]["path"] == "org/model"
    assert runtime.get_model_path_calls == []
    assert runtime.snapshot_calls == []


def test_load_forwards_tokenizer_config(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime().install(monkeypatch)
    load_model_and_tokenizer("org/model", None, tokenizer_config={"trust_remote_code": False})
    assert runtime.load_calls[0]["tokenizer_config"] == {"trust_remote_code": False}


def test_load_forwards_model_config_only_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime().install(monkeypatch)
    load_model_and_tokenizer("org/model", None, model_config={"tie_word_embeddings": True})
    assert runtime.load_calls[0]["model_config"] == {"tie_word_embeddings": True}
    load_model_and_tokenizer("org/model", None, model_config={})
    assert runtime.load_calls[1]["model_config"] is None


def test_load_failure_is_wrapped_as_mlx_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(load_error=FileNotFoundError("No safetensors found")).install(monkeypatch)
    with pytest.raises(MlxRuntimeError) as excinfo:
        load_model_and_tokenizer("org/model", "abc123")
    message = str(excinfo.value)
    assert "org/model@abc123" in message
    assert "FileNotFoundError" in message
    assert "MLX-layout checkpoint" in message
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_missing_runtime_propagates_unavailable_error_from_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninstall_runtime(monkeypatch)
    with pytest.raises(MlxRuntimeUnavailableError):
        load_model_and_tokenizer("org/model", "abc123")


# --- padding / gathering --------------------------------------------------


def test_right_pad_pads_to_the_right_and_reports_lengths() -> None:
    padded, lengths = right_pad([[1, 2, 3], [4]], pad_id=0)
    assert padded == [[1, 2, 3], [4, 0, 0]]
    assert lengths == [3, 1]


def test_right_pad_rejects_empty_batch_and_empty_rows() -> None:
    with pytest.raises(MlxRuntimeError):
        right_pad([], pad_id=0)
    with pytest.raises(MlxRuntimeError):
        right_pad([[1], []], pad_id=0)


def test_gather_last_token_states_reads_each_rows_last_real_token() -> None:
    mx = make_mx_module()
    hidden = FakeArray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],  # row 0: 3 real tokens
            [[4.0, 40.0], [0.0, 0.0], [0.0, 0.0]],  # row 1: 1 real token + 2 pads
        ]
    )
    gathered = gather_last_token_states(mx, hidden, [3, 1])
    assert gathered.tolist() == [[[3.0, 30.0]], [[4.0, 40.0]]]
    assert float_rows(mx, gathered) == [[3.0, 30.0], [4.0, 40.0]]


@pytest.mark.parametrize(
    ("hidden", "lengths"),
    [
        (FakeArray([[1.0, 2.0], [3.0, 4.0]]), [2, 2]),  # not 3-D
        (FakeArray([[[1.0]], [[2.0]]]), [1]),  # batch/lengths mismatch
        (FakeArray([[[1.0]], [[2.0]]]), [1, 2]),  # length beyond the padded width
        (FakeArray([[[1.0]], [[2.0]]]), [1, 0]),  # a row with no real tokens
    ],
)
def test_gather_last_token_states_rejects_inconsistent_input(
    hidden: FakeArray, lengths: list[int]
) -> None:
    with pytest.raises(MlxRuntimeError):
        gather_last_token_states(make_mx_module(), hidden, lengths)


def test_float_rows_requires_batch_one_features_shape() -> None:
    mx = make_mx_module()
    assert float_rows(mx, FakeArray([[[1, 2]], [[3, 4]]])) == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(MlxRuntimeError):
        float_rows(mx, FakeArray([[1.0, 2.0]]))
    with pytest.raises(MlxRuntimeError):
        float_rows(mx, FakeArray([[[1.0], [2.0]]]))


# --- model plumbing -------------------------------------------------------


def test_base_transformer_hidden_states_runs_the_inner_model_only() -> None:
    model = FakeModel(hidden_size=2)
    out = base_transformer_hidden_states(model, FakeArray([[1, 2]]))
    assert out.shape == (1, 2, 2)
    assert model.model.calls == [[[1, 2]]]
    assert model.head.calls == 0  # the LM head is never touched


def test_base_transformer_hidden_states_rejects_unknown_model_classes() -> None:
    with pytest.raises(MlxRuntimeError):
        base_transformer_hidden_states(SimpleNamespace(args=None), FakeArray([[1]]))


def test_lm_head_logits_uses_lm_head_or_tied_embeddings() -> None:
    states = FakeArray([[[1.0, 2.0]]])
    untied = FakeModel(hidden_size=2, vocab_size=3, head_weights={1: 2.0})
    assert lm_head_logits(untied, states).tolist() == [[[0.0, 6.0, 0.0]]]
    tied = FakeModel(hidden_size=2, vocab_size=3, head_weights={1: 2.0}, tie_word_embeddings=True)
    assert not hasattr(tied, "lm_head")
    assert lm_head_logits(tied, states).tolist() == [[[0.0, 6.0, 0.0]]]
    broken = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False))
    with pytest.raises(MlxRuntimeError):
        lm_head_logits(broken, states)


def test_hidden_size_of() -> None:
    assert hidden_size_of(FakeModel(hidden_size=7)) == 7
    with pytest.raises(MlxRuntimeError):
        hidden_size_of(SimpleNamespace())


def test_batched_chunks_in_order() -> None:
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(batched([], 3)) == []
    with pytest.raises(ValueError):
        list(batched([1], 0))
