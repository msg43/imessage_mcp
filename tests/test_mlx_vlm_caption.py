"""mlx-vlm captioning provider (`imsg.enrich.mlx_vlm_caption`), driven
through fake `mlx_vlm` modules stood in `sys.modules` — no weights, no
network — plus the shipped `prompts/caption.txt`."""

from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _model_runtime_stubs import block_module, install_hub_stub
from imsg.enrich.mlx_vlm_caption import (
    CAPTION_TEMPERATURE,
    DEFAULT_CAPTION_PROMPT_PATH,
    MlxVlmCaptionProvider,
    normalize_caption,
)
from imsg.enrich.model_runtime import ModelRuntimeUnavailableError
from imsg.enrich.provider import CaptionProvider
from imsg.errors import ConfigError, EnrichmentError
from imsg.hashing import sha256_text

REPO = "example-org/vision-language-model-4bit"
PROMPT = "Describe the image in one paragraph.\n"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class VlmStub:
    output: Any = field(default_factory=lambda: SimpleNamespace(text="A red bicycle."))
    load_exception: Exception | None = None
    generate_exception: Exception | None = None
    load_calls: list[str] = field(default_factory=list)
    config_calls: list[str] = field(default_factory=list)
    template_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    generate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    model: Any = field(default_factory=lambda: SimpleNamespace(name="model"))
    processor: Any = field(default_factory=lambda: SimpleNamespace(name="processor"))
    config: dict[str, Any] = field(default_factory=lambda: {"model_type": "example_vl"})


@pytest.fixture
def vlm(monkeypatch: pytest.MonkeyPatch) -> VlmStub:
    stub = VlmStub()

    def load(path: str) -> tuple[Any, Any]:
        stub.load_calls.append(path)
        if stub.load_exception is not None:
            raise stub.load_exception
        return stub.model, stub.processor

    def load_config(path: str) -> dict[str, Any]:
        stub.config_calls.append(path)
        return stub.config

    def apply_chat_template(*args: Any, **kwargs: Any) -> str:
        stub.template_calls.append((args, kwargs))
        return f"<chat>{args[2]}</chat>"

    def generate(*args: Any, **kwargs: Any) -> Any:
        stub.generate_calls.append((args, kwargs))
        if stub.generate_exception is not None:
            raise stub.generate_exception
        return stub.output

    package = types.ModuleType("mlx_vlm")
    package.load = load  # type: ignore[attr-defined]
    package.generate = generate  # type: ignore[attr-defined]
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


# --------------------------------------------------------------------------
# construction, identity, prompt provenance
# --------------------------------------------------------------------------


def test_model_id_is_repo_at_revision_with_main_as_the_default() -> None:
    assert MlxVlmCaptionProvider(REPO, None, PROMPT).model_id == f"{REPO}@main"
    assert MlxVlmCaptionProvider(REPO, "abc123", PROMPT).model_id == f"{REPO}@abc123"


def test_prompt_sha256_hashes_the_prompt_verbatim() -> None:
    provider = MlxVlmCaptionProvider(REPO, None, PROMPT)
    assert provider.prompt == PROMPT
    assert provider.prompt_sha256 == sha256_text(PROMPT)
    assert provider.prompt_sha256 == hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    assert MlxVlmCaptionProvider(REPO, None, PROMPT.strip()).prompt_sha256 != provider.prompt_sha256


def test_shipped_caption_prompt_exists_at_the_boundary_prompt_convention() -> None:
    assert Path("prompts/caption.txt") == DEFAULT_CAPTION_PROMPT_PATH
    shipped = REPO_ROOT / DEFAULT_CAPTION_PROMPT_PATH
    text = shipped.read_text(encoding="utf-8")
    assert text.strip()
    provider = MlxVlmCaptionProvider(REPO, None, text)
    assert provider.prompt_sha256 == sha256_text(text)


def test_blank_prompt_is_rejected_at_construction() -> None:
    with pytest.raises(ConfigError):
        MlxVlmCaptionProvider(REPO, None, "   \n")


def test_non_positive_max_tokens_is_rejected_at_construction() -> None:
    with pytest.raises(ConfigError):
        MlxVlmCaptionProvider(REPO, None, PROMPT, max_tokens=0)


def test_construction_never_imports_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    block_module(monkeypatch, "mlx_vlm")
    assert MlxVlmCaptionProvider(REPO, None, PROMPT).max_tokens == 256


# --------------------------------------------------------------------------
# caption normalisation (pure)
# --------------------------------------------------------------------------


def test_normalize_caption_collapses_to_one_stripped_paragraph() -> None:
    raw = "  A photo of\n\na red  bicycle\tleaning on a wall.\n"
    assert normalize_caption(raw) == "A photo of a red bicycle leaning on a wall."


def test_normalize_caption_removes_thinking_blocks() -> None:
    raw = "<think>\nLet me look closely.\n</think>\n\nA screenshot of a calendar."
    assert normalize_caption(raw) == "A screenshot of a calendar."
    assert normalize_caption("<think>ran out of tokens mid-thought") == ""


# --------------------------------------------------------------------------
# the provider against the fake runtime
# --------------------------------------------------------------------------


def test_caption_returns_the_normalised_model_output(vlm: VlmStub, image: Path) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    vlm.output = SimpleNamespace(text="  A red bicycle\nleaning on a wall.  ")
    assert MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image) == (
        "A red bicycle leaning on a wall."
    )


def test_caption_accepts_a_plain_string_from_older_runtimes(vlm: VlmStub, image: Path) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    vlm.output = "A plain string caption."
    assert MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image) == "A plain string caption."


def test_caption_passes_deterministic_decode_parameters(vlm: VlmStub, image: Path) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    MlxVlmCaptionProvider(REPO, None, PROMPT, max_tokens=64).caption(image)
    ((args, kwargs),) = vlm.generate_calls
    assert args == (vlm.model, vlm.processor, f"<chat>{PROMPT}</chat>")
    assert kwargs == {
        "image": [str(image)],
        "max_tokens": 64,
        "temperature": CAPTION_TEMPERATURE,
        "verbose": False,
    }
    assert CAPTION_TEMPERATURE == 0.0


def test_caption_defaults_to_256_max_tokens(vlm: VlmStub, image: Path) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image)
    ((_, kwargs),) = vlm.generate_calls
    assert kwargs["max_tokens"] == 256


def test_caption_formats_the_fixed_prompt_through_the_chat_template(
    vlm: VlmStub, image: Path
) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image)
    ((args, kwargs),) = vlm.template_calls
    assert args == (vlm.processor, vlm.config, PROMPT)
    # Thinking is switched off explicitly: the pinned Qwen3.5 template
    # honours `enable_thinking` (rendered 2026-09-14), and relying on
    # mlx_vlm's per-model default would let a library upgrade silently
    # spend the caption's token budget on a <think> block.
    assert kwargs == {"num_images": 1, "enable_thinking": False}


def test_model_is_loaded_once_and_reused_across_captions(vlm: VlmStub, image: Path) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    provider = MlxVlmCaptionProvider(REPO, None, PROMPT)
    provider.caption(image)
    provider.caption(image)
    assert vlm.load_calls == [REPO]  # unpinned: the runtime resolves `main` itself
    assert vlm.config_calls == [REPO]
    assert len(vlm.generate_calls) == 2


def test_pinned_revision_loads_from_the_resolved_snapshot(
    monkeypatch: pytest.MonkeyPatch, vlm: VlmStub, image: Path, tmp_path: Path
) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    hub = install_hub_stub(monkeypatch, tmp_path / "snapshot")
    MlxVlmCaptionProvider(REPO, "abc123", PROMPT).caption(image)
    assert hub.calls == [{"repo_id": REPO, "revision": "abc123"}]
    assert vlm.load_calls == [str(tmp_path / "snapshot")]
    assert vlm.config_calls == [str(tmp_path / "snapshot")]


def test_empty_output_is_an_enrichment_error(vlm: VlmStub, image: Path) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    provider = MlxVlmCaptionProvider(REPO, None, PROMPT)
    for output in ("", "   \n", SimpleNamespace(text="<think>only thoughts</think>")):
        vlm.output = output
        with pytest.raises(EnrichmentError) as excinfo:
            provider.caption(image)
        assert "empty caption" in str(excinfo.value)


def test_textless_output_object_is_an_enrichment_error(vlm: VlmStub, image: Path) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    vlm.output = SimpleNamespace(text=None)
    with pytest.raises(EnrichmentError):
        MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image)


def test_generate_failure_is_wrapped_as_enrichment_error(
    monkeypatch: pytest.MonkeyPatch, vlm: VlmStub, image: Path, tmp_path: Path
) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    install_hub_stub(monkeypatch, tmp_path)
    vlm.generate_exception = ValueError("could not decode image")
    with pytest.raises(EnrichmentError) as excinfo:
        MlxVlmCaptionProvider(REPO, "abc123", PROMPT).caption(image)
    assert "could not decode image" in str(excinfo.value)
    assert f"{REPO}@abc123" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_load_failure_is_runtime_unavailable_not_a_task_failure(vlm: VlmStub, image: Path) -> None:
    vlm.load_exception = OSError("weights missing")
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image)
    assert "weights missing" in str(excinfo.value)
    assert vlm.generate_calls == []


def test_missing_image_is_an_enrichment_error_without_touching_the_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    block_module(monkeypatch, "mlx_vlm")
    with pytest.raises(EnrichmentError):
        MlxVlmCaptionProvider(REPO, None, PROMPT).caption(tmp_path / "missing.jpg")


def test_missing_runtime_is_a_clear_error_naming_the_package(
    monkeypatch: pytest.MonkeyPatch, image: Path
) -> None:
    block_module(monkeypatch, "mlx_vlm")
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image)
    assert "mlx-vlm" in str(excinfo.value)
    assert not isinstance(excinfo.value, EnrichmentError)


def test_satisfies_the_caption_provider_protocol() -> None:
    provider: CaptionProvider = MlxVlmCaptionProvider(REPO, None, PROMPT)
    assert provider.model_id == f"{REPO}@main"


# --------------------------------------------------------------------------
# HEIC: the iPhone's default photo format. `mlx_vlm` opens the image with
# PIL, and PIL decodes HEIC only after pillow-heif registers its opener.
# Nothing in the enrichment process did, so every HEIC caption failed.
# --------------------------------------------------------------------------


def test_caption_registers_the_heif_opener_before_the_model_reads_the_image(
    vlm: VlmStub, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    events: list[str] = []
    heif = types.ModuleType("pillow_heif")

    def register_heif_opener(**kwargs: Any) -> None:  # pillow-heif 1.7.0's signature
        events.append("registered")

    heif.register_heif_opener = register_heif_opener  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pillow_heif", heif)
    real_generate = sys.modules["mlx_vlm"].generate  # type: ignore[attr-defined]

    def generate(*args: Any, **kwargs: Any) -> Any:
        events.append("generate")
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(sys.modules["mlx_vlm"], "generate", generate)
    provider = MlxVlmCaptionProvider(REPO, None, PROMPT)

    provider.caption(image)
    provider.caption(image)

    assert events[0] == "registered"
    assert events.count("generate") == 2


def test_a_real_heic_photo_decodes_by_the_time_the_model_sees_it(
    vlm: VlmStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    pil_image = pytest.importorskip("PIL.Image")
    heic = tmp_path / "photo"  # content-addressed cache names carry no extension
    pillow_heif.from_pillow(pil_image.new("RGB", (16, 16), "red")).save(heic, quality=50)
    decoded: list[tuple[int, int]] = []

    def generate(*args: Any, **kwargs: Any) -> Any:
        # What mlx_vlm.utils.load_image does with a path.
        with pil_image.open(kwargs["image"][0]) as img:
            img.load()
            decoded.append(img.size)
        return SimpleNamespace(text="A red square.")

    monkeypatch.setattr(sys.modules["mlx_vlm"], "generate", generate)

    assert MlxVlmCaptionProvider(REPO, None, PROMPT).caption(heic) == "A red square."
    assert decoded == [(16, 16)]


def test_caption_still_works_without_pillow_heif(
    vlm: VlmStub, image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JPEG needs no HEIF support; a missing pillow-heif costs HEICs
    alone, and each of those fails as its own task with PIL's error."""
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    assert MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image) == "A red bicycle."


# --------------------------------------------------------------------------
# Image size: a caption's cost follows the pixels the model is shown.
# Measured 2026-09-23 on an M2 Ultra with the pinned Qwen3.5-35B-A3B: a
# 4032x3024 photo took 37.6 s to caption, the same photo scaled to
# 1920x1440 took 5.7 s, and the 1440x1920 image the production host's
# 14.2 s figure was measured on took 5.2 s. Photos are scaled to fit
# `max_image_side` before the model sees them.
# --------------------------------------------------------------------------


def _photo(path: Path, size: tuple[int, int]) -> Path:
    pil_image = pytest.importorskip("PIL.Image")
    pil_image.new("RGB", size, "blue").save(path, "JPEG", quality=80)
    return path


def _recording_generate(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """A fake `mlx_vlm.generate` that opens the image it is given while
    it still exists, the way the real one does."""
    pil_image = pytest.importorskip("PIL.Image")
    seen: list[tuple[int, int]] = []

    def generate(*args: Any, **kwargs: Any) -> Any:
        with pil_image.open(kwargs["image"][0]) as img:
            seen.append(img.size)
        return SimpleNamespace(text="A blue rectangle.")

    monkeypatch.setattr(sys.modules["mlx_vlm"], "generate", generate)
    return seen


def test_a_full_resolution_photo_is_scaled_to_fit_before_captioning(
    vlm: VlmStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording_generate(monkeypatch)
    photo = _photo(tmp_path / "photo", (4032, 3024))

    MlxVlmCaptionProvider(REPO, None, PROMPT).caption(photo)

    assert seen == [(1920, 1440)]
    assert photo.stat().st_size > 0  # the attachment itself is never rewritten


def test_the_default_bound_is_the_size_the_caption_cost_was_measured_at() -> None:
    from imsg.enrich.mlx_vlm_caption import DEFAULT_CAPTION_MAX_IMAGE_SIDE

    assert DEFAULT_CAPTION_MAX_IMAGE_SIDE == 1920
    assert MlxVlmCaptionProvider(REPO, None, PROMPT).max_image_side == 1920


def test_an_image_that_already_fits_is_passed_through_untouched(
    vlm: VlmStub, tmp_path: Path
) -> None:
    small = _photo(tmp_path / "small", (1200, 630))
    MlxVlmCaptionProvider(REPO, None, PROMPT).caption(small)
    ((_, kwargs),) = vlm.generate_calls
    assert kwargs["image"] == [str(small)]


def test_no_bound_keeps_full_resolution(
    vlm: VlmStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording_generate(monkeypatch)
    photo = _photo(tmp_path / "photo", (4032, 3024))

    MlxVlmCaptionProvider(REPO, None, PROMPT, max_image_side=None).caption(photo)

    assert seen == [(4032, 3024)]


def test_orientation_is_applied_before_scaling(
    vlm: VlmStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A portrait iPhone photo is stored landscape with an EXIF rotation.
    The model loader applies that rotation to what it opens, so a scaled
    copy must carry it already applied."""
    pil_image = pytest.importorskip("PIL.Image")
    seen = _recording_generate(monkeypatch)
    photo = tmp_path / "portrait"
    img = pil_image.new("RGB", (4032, 3024), "blue")
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    img.save(photo, "JPEG", exif=exif)

    MlxVlmCaptionProvider(REPO, None, PROMPT).caption(photo)

    assert seen == [(1440, 1920)]


def test_an_unreadable_image_goes_to_the_model_unchanged(vlm: VlmStub, image: Path) -> None:
    """PIL cannot open it; the model's own loader reports that, as a
    per-task failure, exactly as before."""
    pytest.importorskip("PIL", reason="needs the `models` extra (`uv sync --extra models`)")
    pytest.importorskip("mlx", reason="needs the `models` extra (`uv sync --extra models`)")
    MlxVlmCaptionProvider(REPO, None, PROMPT).caption(image)
    ((_, kwargs),) = vlm.generate_calls
    assert kwargs["image"] == [str(image)]


def test_a_non_positive_bound_is_rejected_at_construction() -> None:
    with pytest.raises(ConfigError):
        MlxVlmCaptionProvider(REPO, None, PROMPT, max_image_side=0)
