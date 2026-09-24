"""`imsg.shared_vlm_runtime` — one loaded copy of a vision-language model
serving both the captioner and boundary detection (D10.3 defect 1).

Driven through fake `mlx_vlm` / `mlx.core` / `huggingface_hub` modules in
`sys.modules`: no weights, no network. What these tests pin down is the
*sharing contract* — how many times the weights are loaded, and that the
sharing is something a caller passes in rather than a module-level cache
that would be invisible at the call site.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _mlx_fakes import make_mx_module
from _model_runtime_stubs import install_hub_stub
from imsg.enrich.mlx_vlm_caption import MlxVlmCaptionProvider
from imsg.enrich.model_runtime import ModelRuntimeUnavailableError
from imsg.segment.mlx_boundaries import MlxBoundaryProvider
from imsg.segment.models import MessageForSegmentation
from imsg.shared_vlm_runtime import (
    DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES,
    SharedVlmRuntime,
    vlm_model_id,
)

REPO = "example-org/vision-language-model-4bit"
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
CAPTION_PROMPT = "Describe the image in one paragraph.\n"
BOUNDARY_TEMPLATE = 'Split the conversation into topics.\nReturn {"boundaries": [...]}.\n'


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


@dataclass
class VlmRuntimeStub:
    """Records what `mlx_vlm` was asked to do. `load_calls` is the number
    that matters: the whole point of a shared runtime is that it is 1."""

    load_calls: list[str] = field(default_factory=list)
    config_calls: list[str] = field(default_factory=list)
    template_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    generate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    stream_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    stream_chunks: list[str] = field(default_factory=lambda: ['{"boundaries"', ": [3]}"])
    caption_text: str = "A red bicycle against a wall."
    load_exception: Exception | None = None
    models_built: list[Any] = field(default_factory=list)


@pytest.fixture
def vlm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> VlmRuntimeStub:
    stub = VlmRuntimeStub()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    install_hub_stub(monkeypatch, snapshot)
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", make_mx_module())

    def load(path: str) -> tuple[Any, Any]:
        stub.load_calls.append(path)
        if stub.load_exception is not None:
            raise stub.load_exception
        model = SimpleNamespace(name=f"model-{len(stub.load_calls)}")
        stub.models_built.append(model)
        return model, SimpleNamespace(name="processor")

    def load_config(path: str) -> dict[str, Any]:
        stub.config_calls.append(path)
        return {"model_type": "example_vl"}

    def apply_chat_template(*args: Any, **kwargs: Any) -> str:
        stub.template_calls.append((args, kwargs))
        return f"<chat>{args[2]}</chat>"

    def generate(*args: Any, **kwargs: Any) -> Any:
        stub.generate_calls.append((args, kwargs))
        return SimpleNamespace(text=stub.caption_text)

    def stream_generate(*args: Any, **kwargs: Any) -> Any:
        stub.stream_calls.append((args, kwargs))
        return iter([SimpleNamespace(text=chunk) for chunk in stub.stream_chunks])

    package = types.ModuleType("mlx_vlm")
    package.load = load  # type: ignore[attr-defined]
    package.generate = generate  # type: ignore[attr-defined]
    package.stream_generate = stream_generate  # type: ignore[attr-defined]
    utils = types.ModuleType("mlx_vlm.utils")
    utils.load_config = load_config  # type: ignore[attr-defined]
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = apply_chat_template  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    return stub


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"not really a jpeg")
    return path


_BASE = datetime(2024, 3, 2, 9, 0, tzinfo=UTC)


def _window(n: int) -> list[MessageForSegmentation]:
    return [
        MessageForSegmentation(
            message_id=i,
            source_guid=f"guid-{i}",
            chat_id=1,
            sent_at=_BASE + timedelta(minutes=i),
            is_from_me=(i % 2 == 0),
            sender_short_name="owner" if i % 2 == 0 else "alice",
            text=f"message {i}",
            is_unsent=False,
            is_edited=False,
            has_attachments=False,
        )
        for i in range(n)
    ]


def _caption_provider(runtime: SharedVlmRuntime | None, **kw: Any) -> MlxVlmCaptionProvider:
    return MlxVlmCaptionProvider(REPO, REVISION, CAPTION_PROMPT, shared_runtime=runtime, **kw)


def _boundary_provider(runtime: SharedVlmRuntime | None, **kw: Any) -> MlxBoundaryProvider:
    return MlxBoundaryProvider(REPO, REVISION, BOUNDARY_TEMPLATE, shared_runtime=runtime, **kw)


# --------------------------------------------------------------------------
# the sharing contract
# --------------------------------------------------------------------------


def test_one_runtime_loads_the_weights_once_for_both_roles(
    vlm: VlmRuntimeStub, image: Path
) -> None:
    """The defect D10.3 names: captioning and boundary detection are
    pinned to the same checkpoint, and loading it twice cost 18 GiB."""
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    runtime = SharedVlmRuntime()
    caption = _caption_provider(runtime)
    boundary = _boundary_provider(runtime)

    assert boundary.detect_boundaries(_window(8)) == [3]
    assert caption.caption(image) == "A red bicycle against a wall."

    assert len(vlm.load_calls) == 1
    assert runtime.loaded_model_ids == (vlm_model_id(REPO, REVISION),)
    # Not merely "loaded once" — both roles ran against the *same* object.
    assert len(vlm.models_built) == 1
    assert vlm.stream_calls[0][0][0] is vlm.models_built[0]
    assert vlm.generate_calls[0][0][0] is vlm.models_built[0]


def test_two_private_runtimes_load_the_weights_twice(
    vlm: VlmRuntimeStub, image: Path
) -> None:
    """The before picture, and proof the fixture would notice a second
    load: providers that share nothing each load their own copy."""
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    caption = _caption_provider(None)
    boundary = _boundary_provider(SharedVlmRuntime())

    boundary.detect_boundaries(_window(8))
    caption.caption(image)

    assert len(vlm.load_calls) == 2
    assert len(vlm.models_built) == 2


def test_the_sharing_is_an_argument_not_a_module_level_cache(
    vlm: VlmRuntimeStub, image: Path
) -> None:
    """Two independently-constructed runtimes must not silently share:
    the object passed in is the whole mechanism, so a test (or an
    operator reading the call site) can see what is shared with what."""
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    first, second = SharedVlmRuntime(), SharedVlmRuntime()
    _caption_provider(first).caption(image)
    _caption_provider(second).caption(image)

    assert len(vlm.load_calls) == 2
    assert first.loaded_model_ids == second.loaded_model_ids == (vlm_model_id(REPO, REVISION),)


def test_different_pins_are_different_entries(vlm: VlmRuntimeStub, image: Path) -> None:
    """Sharing is keyed on (repo, revision) — an operator who points the
    two roles at different revisions gets two loads, correctly."""
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    runtime = SharedVlmRuntime()
    _caption_provider(runtime).caption(image)
    MlxVlmCaptionProvider(
        REPO, OTHER_REVISION, CAPTION_PROMPT, shared_runtime=runtime
    ).caption(image)

    assert len(vlm.load_calls) == 2
    assert runtime.loaded_model_ids == (
        vlm_model_id(REPO, REVISION),
        vlm_model_id(REPO, OTHER_REVISION),
    )


def test_a_caption_provider_given_no_runtime_still_works_and_exposes_its_own(
    vlm: VlmRuntimeStub, image: Path
) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    provider = _caption_provider(None)
    assert provider.caption(image) == "A red bicycle against a wall."
    assert provider.shared_runtime.loaded_model_ids == (vlm_model_id(REPO, REVISION),)


def test_acquire_is_idempotent_per_pin(vlm: VlmRuntimeStub) -> None:
    runtime = SharedVlmRuntime()
    assert not runtime.is_loaded(REPO, REVISION)
    first = runtime.acquire(REPO, REVISION)
    second = runtime.acquire(REPO, REVISION)
    assert first is second
    assert runtime.is_loaded(REPO, REVISION)
    assert len(vlm.load_calls) == 1


def test_a_load_failure_is_an_environment_error_not_a_per_task_one(
    vlm: VlmRuntimeStub,
) -> None:
    """`ModelRuntimeUnavailableError` aborts the run; an
    `EnrichmentError` would burn every queued task's retry budget one
    task at a time (see `imsg.enrich.model_runtime`)."""
    vlm.load_exception = RuntimeError("no such file")
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        SharedVlmRuntime().acquire(REPO, REVISION)
    assert vlm_model_id(REPO, REVISION) in str(excinfo.value)


# --------------------------------------------------------------------------
# the buffer-cache bound (D10.2)
# --------------------------------------------------------------------------


def test_loading_bounds_the_process_wide_buffer_cache(vlm: VlmRuntimeStub) -> None:
    """The enrichment providers called nothing, so their limit sat at the
    runtime default (60.8 GiB on the production host) and one process was
    seen holding 37.25 GiB of pooled freed buffers."""
    runtime = SharedVlmRuntime(cache_limit_bytes=3 * 2**30)
    runtime.acquire(REPO, REVISION)
    assert sys.modules["mlx.core"].cache_limit_calls == [3 * 2**30]  # type: ignore[union-attr]


def test_the_default_bound_is_the_enrichment_one(vlm: VlmRuntimeStub) -> None:
    SharedVlmRuntime().acquire(REPO, REVISION)
    assert sys.modules["mlx.core"].cache_limit_calls == [  # type: ignore[union-attr]
        DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES
    ]


def test_cache_limit_none_leaves_the_runtime_alone(vlm: VlmRuntimeStub) -> None:
    SharedVlmRuntime(cache_limit_bytes=None).acquire(REPO, REVISION)
    assert sys.modules["mlx.core"].cache_limit_calls == []  # type: ignore[union-attr]


def test_a_negative_bound_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="cache_limit_bytes"):
        SharedVlmRuntime(cache_limit_bytes=-1)


# --------------------------------------------------------------------------
# boundary detection through the shared model
# --------------------------------------------------------------------------


def test_boundary_generation_through_the_vlm_is_text_only_and_greedy(
    vlm: VlmRuntimeStub,
) -> None:
    boundary = _boundary_provider(SharedVlmRuntime())
    assert boundary.detect_boundaries(_window(8)) == [3]

    (_args, kwargs) = vlm.stream_calls[0]
    assert kwargs["image"] is None  # a text-only turn through a VLM
    assert kwargs["temperature"] == 0.0  # the counterpart of make_sampler(temp=0)
    assert kwargs["max_tokens"] == 512

    (template_args, template_kwargs) = vlm.template_calls[0]
    assert template_kwargs["num_images"] == 0
    assert template_kwargs["enable_thinking"] is False
    assert BOUNDARY_TEMPLATE.rstrip() in template_args[2]


def test_a_shared_boundary_provider_never_touches_mlx_lm(
    vlm: VlmRuntimeStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saving is only real if the `mlx_lm` loader is not reached at
    all — so make importing it fail and require the run to succeed."""
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    boundary = _boundary_provider(SharedVlmRuntime())
    assert boundary.detect_boundaries(_window(8)) == [3]
    assert boundary.shares_vlm_weights is True
    assert boundary.is_loaded is True


def test_an_unshared_boundary_provider_reports_that_it_does_not_share() -> None:
    assert _boundary_provider(None).shares_vlm_weights is False


def test_a_window_too_short_to_split_asks_no_model(vlm: VlmRuntimeStub) -> None:
    """No index can satisfy 0 < i < len(window), so nothing loads."""
    assert _boundary_provider(SharedVlmRuntime()).detect_boundaries(_window(1)) == []
    assert vlm.load_calls == []
