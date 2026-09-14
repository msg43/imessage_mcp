"""PE-Core multimodal provider (`imsg.embed.pe_core_multimodal`),
exercised entirely against `sys.modules` fakes for torch / open_clip /
huggingface_hub / PIL — no model weights, no network, and none of those
packages installed in this build environment. The fakes implement only
the exact backend API names the provider calls; anything else raising
`AttributeError` is the point."""

from __future__ import annotations

import contextlib
import json
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from imsg import constants
from imsg.embed.pe_core_multimodal import (
    OPEN_CLIP_CONFIG_FILENAME,
    OPEN_CLIP_MIRRORS,
    OPEN_CLIP_PYTORCH_BIN_FILENAME,
    OPEN_CLIP_SAFETENSORS_FILENAME,
    ImageEmbeddingError,
    PeCoreMultimodalEmbeddingProvider,
    PeCoreRuntimeError,
    UnreadableImageError,
    resolve_weights_repo,
)
from imsg.errors import EmbeddingError, ImsgError

CANONICAL = "facebook/PE-Core-G14-448"
MIRROR = "timm/PE-Core-bigG-14-448"
REVISION = "deadbeefcafe"


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _unit(vec: list[float]) -> list[float]:
    n = _norm(vec)
    return [v / n for v in vec]


def _close(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) < 1e-9 for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------
# fakes — only the API surface the provider actually touches
# --------------------------------------------------------------------------


class FakeTensor:
    """`.float().cpu().tolist()` — the exact chain `_rows` uses."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[list[float]]:
        return [list(r) for r in self.rows]


class FakeImageTensor:
    def __init__(self, tag: str) -> None:
        self.tag = tag


class FakeBatch:
    def __init__(self, items: list[FakeImageTensor]) -> None:
        self.items = items
        self.device: str | None = None

    def to(self, device: str) -> FakeBatch:
        self.device = device
        return self


class FakeTokens:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.device: str | None = None

    def to(self, device: str) -> FakeTokens:
        self.device = device
        return self


class FakeImage:
    def __init__(self, path: Path, mode: str = "L") -> None:
        self.path = path
        self.mode = mode

    def __enter__(self) -> FakeImage:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def convert(self, mode: str) -> FakeImage:
        return FakeImage(self.path, mode)


@dataclass
class FakeWorld:
    """Knobs the fakes read, and a record of every backend call."""

    root: Path
    dim: int = 8
    config_embed_dim: int | None = None
    """What the fetched `open_clip_config.json` claims; `dim` when None."""
    text_dim: int | None = None
    """Width the fake text tower actually produces; `dim` when None."""
    mps_available: bool = True
    cuda_available: bool = False
    has_safetensors: bool = True
    has_bin: bool = True
    unreadable: set[str] = field(default_factory=set)
    """File names whose `Image.open` fails."""
    tower_fails_for: set[str] = field(default_factory=set)
    """Image tags whose presence makes the image forward pass raise."""
    fail_batches_larger_than: int | None = None
    zero_vector_for: set[str] = field(default_factory=set)
    build_error: Exception | None = None
    hub_error: Exception | None = None

    image_calls: list[list[str]] = field(default_factory=list)
    image_kwargs: list[dict[str, Any]] = field(default_factory=list)
    text_calls: list[str] = field(default_factory=list)
    text_kwargs: list[dict[str, Any]] = field(default_factory=list)
    tokenizer_calls: list[list[str]] = field(default_factory=list)
    build_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    tokenizer_build_calls: list[str] = field(default_factory=list)
    hub_download_calls: list[dict[str, Any]] = field(default_factory=list)
    snapshot_calls: list[dict[str, Any]] = field(default_factory=list)
    exif_calls: int = 0
    eval_calls: int = 0

    def vector_for(self, tag: str) -> list[float]:
        """Deterministic, distinct per tag, deliberately NOT unit-norm."""
        if tag in self.zero_vector_for:
            return [0.0] * self.dim
        seed = [float(ord(ch)) for ch in tag]
        return [seed[i % len(seed)] + i for i in range(self.dim)]

    def text_vector(self, text: str) -> list[float]:
        width = self.text_dim if self.text_dim is not None else self.dim
        seed = [float(ord(ch)) + 1.0 for ch in text]
        return [seed[i % len(seed)] * 2 + i for i in range(width)]

    def snapshot_dir(self, repo_id: str, revision: str | None) -> Path:
        return self.root / repo_id.replace("/", "--") / (revision or "main")


class FakeModel:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def eval(self) -> FakeModel:
        self.world.eval_calls += 1
        return self

    def encode_image(self, batch: FakeBatch, normalize: bool = False) -> FakeTensor:
        tags = [t.tag for t in batch.items]
        self.world.image_calls.append(tags)
        self.world.image_kwargs.append({"normalize": normalize, "device": batch.device})
        limit = self.world.fail_batches_larger_than
        if limit is not None and len(tags) > limit:
            raise RuntimeError("MPS backend out of memory")
        if any(t in self.world.tower_fails_for for t in tags):
            raise RuntimeError("image tower failure")
        return FakeTensor([self.world.vector_for(t) for t in tags])

    def encode_text(self, tokens: FakeTokens, normalize: bool = False) -> FakeTensor:
        text = tokens.texts[0]
        self.world.text_calls.append(text)
        self.world.text_kwargs.append({"normalize": normalize, "device": tokens.device})
        return FakeTensor([self.world.text_vector(text)])


class FakeTokenizer:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def __call__(self, texts: list[str], context_length: int | None = None) -> FakeTokens:
        self.world.tokenizer_calls.append(list(texts))
        return FakeTokens(list(texts))


def _install_fakes(world: FakeWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")

    def mps_is_available() -> bool:
        return world.mps_available

    def cuda_is_available() -> bool:
        return world.cuda_available

    def stack(items: list[FakeImageTensor]) -> FakeBatch:
        return FakeBatch(list(items))

    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=mps_is_available))
    torch.cuda = types.SimpleNamespace(is_available=cuda_is_available)
    torch.inference_mode = contextlib.nullcontext
    torch.stack = stack

    open_clip = types.ModuleType("open_clip")

    def preprocess(image: FakeImage) -> FakeImageTensor:
        if image.mode != "RGB":
            raise AssertionError(f"preprocess expects an RGB image, got mode {image.mode!r}")
        return FakeImageTensor(image.path.name)

    def create_model_and_transforms(model_name: str, **kwargs: Any) -> tuple[Any, Any, Any]:
        world.build_calls.append((model_name, kwargs))
        if world.build_error is not None:
            raise world.build_error
        return FakeModel(world), None, preprocess

    def get_tokenizer(model_name: str) -> FakeTokenizer:
        world.tokenizer_build_calls.append(model_name)
        return FakeTokenizer(world)

    open_clip.create_model_and_transforms = create_model_and_transforms
    open_clip.get_tokenizer = get_tokenizer

    hub = types.ModuleType("huggingface_hub")

    def hf_hub_download(
        repo_id: str, filename: str, revision: str | None = None, cache_dir: str | None = None
    ) -> str:
        world.hub_download_calls.append(
            {"repo_id": repo_id, "filename": filename, "revision": revision, "cache_dir": cache_dir}
        )
        if world.hub_error is not None:
            raise world.hub_error
        directory = world.snapshot_dir(repo_id, revision)
        directory.mkdir(parents=True, exist_ok=True)
        embed_dim = world.config_embed_dim if world.config_embed_dim is not None else world.dim
        config = {
            "model_cfg": {"embed_dim": embed_dim, "custom_text": True},
            "preprocess_cfg": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        }
        target = directory / filename
        target.write_text(json.dumps(config), encoding="utf-8")
        return str(target)

    def snapshot_download(
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_patterns: list[str] | None = None,
    ) -> str:
        patterns = list(allow_patterns or [])
        world.snapshot_calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "allow_patterns": patterns,
            }
        )
        directory = world.snapshot_dir(repo_id, revision)
        directory.mkdir(parents=True, exist_ok=True)
        if OPEN_CLIP_SAFETENSORS_FILENAME in patterns and world.has_safetensors:
            (directory / OPEN_CLIP_SAFETENSORS_FILENAME).write_bytes(b"")
        if OPEN_CLIP_PYTORCH_BIN_FILENAME in patterns and world.has_bin:
            (directory / OPEN_CLIP_PYTORCH_BIN_FILENAME).write_bytes(b"")
        return str(directory)

    hub.hf_hub_download = hf_hub_download
    hub.snapshot_download = snapshot_download

    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image_ops = types.ModuleType("PIL.ImageOps")

    def image_open(path: Path) -> FakeImage:
        path = Path(path)
        if path.name in world.unreadable or not path.is_file():
            raise OSError(f"cannot identify image file {path}")
        return FakeImage(path)

    def exif_transpose(image: FakeImage) -> FakeImage:
        world.exif_calls += 1
        return image

    pil_image.open = image_open
    pil_image_ops.exif_transpose = exif_transpose
    pil.Image = pil_image
    pil.ImageOps = pil_image_ops

    for name, module in {
        "torch": torch,
        "open_clip": open_clip,
        "huggingface_hub": hub,
        "PIL": pil,
        "PIL.Image": pil_image,
        "PIL.ImageOps": pil_image_ops,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    w = FakeWorld(root=tmp_path / "hub")
    _install_fakes(w, monkeypatch)
    return w


def _provider(world: FakeWorld, **overrides: Any) -> PeCoreMultimodalEmbeddingProvider:
    kwargs: dict[str, Any] = {"cache_dir": world.root / "cache"}
    kwargs.update(overrides)
    return PeCoreMultimodalEmbeddingProvider(CANONICAL, REVISION, world.dim, **kwargs)


def _images(tmp_path: Path, names: list[str]) -> list[Path]:
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"not really pixels")
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# identity, mapping, error hierarchy — no runtime needed
# --------------------------------------------------------------------------


def test_model_id_is_repo_at_revision() -> None:
    provider = PeCoreMultimodalEmbeddingProvider(CANONICAL, "abc123", 1280)
    assert provider.model_id == "facebook/PE-Core-G14-448@abc123"
    assert provider.dim == 1280


def test_model_id_defaults_revision_to_main() -> None:
    provider = PeCoreMultimodalEmbeddingProvider(CANONICAL, None, 1280)
    assert provider.model_id == "facebook/PE-Core-G14-448@main"


def test_dim_matches_the_multimodal_constant_for_the_pinned_model() -> None:
    """The pinned G14 model is 1280-wide (open_clip config `embed_dim`);
    the pipeline separately requires `dim == MULTIMODAL_EMBEDDING_DIM`."""
    provider = PeCoreMultimodalEmbeddingProvider(
        CANONICAL, None, constants.MULTIMODAL_EMBEDDING_DIM
    )
    assert provider.dim == 1280


def test_canonical_meta_id_resolves_to_open_clip_mirror() -> None:
    assert resolve_weights_repo(CANONICAL) == MIRROR
    assert OPEN_CLIP_MIRRORS[CANONICAL] == MIRROR


def test_open_clip_layout_repo_passes_through_unchanged() -> None:
    assert resolve_weights_repo(MIRROR) == MIRROR
    assert resolve_weights_repo("example-org/PE-Core-fork") == "example-org/PE-Core-fork"


def test_constructor_rejects_non_positive_dim_and_batch_size() -> None:
    with pytest.raises(EmbeddingError, match="dim"):
        PeCoreMultimodalEmbeddingProvider(CANONICAL, None, 0)
    with pytest.raises(EmbeddingError, match="batch_size"):
        PeCoreMultimodalEmbeddingProvider(CANONICAL, None, 8, batch_size=0)


def test_error_hierarchy_is_imsg_rooted() -> None:
    assert issubclass(PeCoreRuntimeError, EmbeddingError)
    assert issubclass(ImageEmbeddingError, EmbeddingError)
    assert issubclass(UnreadableImageError, ImageEmbeddingError)
    assert issubclass(EmbeddingError, ImsgError)


def test_construction_does_not_touch_the_runtime(world: FakeWorld) -> None:
    _provider(world)
    assert world.build_calls == []
    assert world.hub_download_calls == []


# --------------------------------------------------------------------------
# lazy import failures
# --------------------------------------------------------------------------


def test_missing_torch_raises_clear_runtime_error(
    world: FakeWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import of a None entry raises ImportError
    provider = _provider(world)
    with pytest.raises(PeCoreRuntimeError, match=r"'torch'.*models") as excinfo:
        provider.embed_text("a dog")
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_missing_open_clip_names_the_package(
    world: FakeWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "open_clip", None)
    with pytest.raises(PeCoreRuntimeError, match="'open_clip'"):
        _provider(world).embed_text("a dog")


def test_missing_pil_names_the_package(world: FakeWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "PIL.Image", None)
    with pytest.raises(PeCoreRuntimeError, match=r"'PIL\.Image'"):
        _provider(world).embed_text("a dog")


# --------------------------------------------------------------------------
# loading: device rules, revision pinning, dim checks, run-once
# --------------------------------------------------------------------------


def test_mps_unavailable_without_fallback_refuses_to_load(world: FakeWorld) -> None:
    world.mps_available = False
    provider = _provider(world)
    with pytest.raises(PeCoreRuntimeError, match=r"'mps' is not available.*allow_cpu_fallback"):
        provider.embed_text("a dog")
    assert world.snapshot_calls == [], "must fail before downloading any weights"
    assert world.build_calls == []


def test_mps_unavailable_with_fallback_runs_on_cpu(world: FakeWorld) -> None:
    world.mps_available = False
    provider = _provider(world, allow_cpu_fallback=True)
    provider.embed_text("a dog")
    assert world.build_calls[0][1]["device"] == "cpu"
    assert world.text_kwargs[-1]["device"] == "cpu"


def test_available_device_is_used_verbatim(world: FakeWorld) -> None:
    provider = _provider(world, device="mps")
    provider.embed_text("a dog")
    assert world.build_calls[0][1]["device"] == "mps"
    assert world.text_kwargs[-1]["device"] == "mps"


def test_cpu_device_never_needs_fallback(world: FakeWorld) -> None:
    world.mps_available = False
    provider = _provider(world, device="cpu")
    provider.embed_text("a dog")
    assert world.build_calls[0][1]["device"] == "cpu"


def test_cuda_device_is_checked_against_cuda_availability(world: FakeWorld) -> None:
    world.cuda_available = False
    with pytest.raises(PeCoreRuntimeError, match="'cuda' is not available"):
        _provider(world, device="cuda").embed_text("a dog")
    world.cuda_available = True
    _provider(world, device="cuda:0").embed_text("a dog")
    assert world.build_calls[-1][1]["device"] == "cuda:0"


def test_unknown_device_kind_is_rejected(world: FakeWorld) -> None:
    with pytest.raises(PeCoreRuntimeError, match="unsupported device 'tpu'"):
        _provider(world, device="tpu").embed_text("a dog")


def test_weights_are_fetched_from_the_mirror_at_the_pinned_revision(world: FakeWorld) -> None:
    provider = _provider(world)
    provider.embed_text("a dog")

    config_call = world.hub_download_calls[0]
    assert config_call["repo_id"] == MIRROR
    assert config_call["filename"] == OPEN_CLIP_CONFIG_FILENAME
    assert config_call["revision"] == REVISION
    assert config_call["cache_dir"] == str(world.root / "cache")

    snapshot_call = world.snapshot_calls[0]
    assert snapshot_call["repo_id"] == MIRROR
    assert snapshot_call["revision"] == REVISION
    assert snapshot_call["cache_dir"] == str(world.root / "cache")
    assert OPEN_CLIP_SAFETENSORS_FILENAME in snapshot_call["allow_patterns"]
    assert OPEN_CLIP_PYTORCH_BIN_FILENAME not in snapshot_call["allow_patterns"]

    model_name, kwargs = world.build_calls[0]
    assert model_name == f"local-dir:{world.snapshot_dir(MIRROR, REVISION)}"
    assert kwargs["require_pretrained"] is True
    assert world.tokenizer_build_calls == [model_name]
    assert world.eval_calls == 1
    # the recorded model id is what the operator configured, verbatim
    assert provider.model_id == f"{CANONICAL}@{REVISION}"
    assert provider.weights_repo == MIRROR


def test_open_clip_layout_repo_is_fetched_directly(world: FakeWorld) -> None:
    provider = PeCoreMultimodalEmbeddingProvider(MIRROR, None, world.dim, cache_dir=world.root)
    provider.embed_text("a dog")
    assert world.hub_download_calls[0]["repo_id"] == MIRROR
    assert world.hub_download_calls[0]["revision"] is None
    assert provider.model_id == f"{MIRROR}@main"


def test_config_dim_mismatch_refuses_before_downloading_weights(world: FakeWorld) -> None:
    world.config_embed_dim = 1024
    provider = _provider(world)
    with pytest.raises(PeCoreRuntimeError, match=r"embed_dim 1024.*dim 8"):
        provider.embed_text("a dog")
    assert world.snapshot_calls == []
    assert world.build_calls == []


def test_loaded_model_width_mismatch_is_caught_by_the_load_probe(world: FakeWorld) -> None:
    world.text_dim = 4  # config says 8, the loaded towers actually produce 4
    provider = _provider(world)
    with pytest.raises(PeCoreRuntimeError, match=r"4-wide text embedding, expected dim 8"):
        provider.embed_text("a dog")


def test_snapshot_without_safetensors_falls_back_to_the_bin_checkpoint(world: FakeWorld) -> None:
    world.has_safetensors = False
    _provider(world).embed_text("a dog")
    assert len(world.snapshot_calls) == 2
    assert OPEN_CLIP_PYTORCH_BIN_FILENAME not in world.snapshot_calls[0]["allow_patterns"]
    assert OPEN_CLIP_PYTORCH_BIN_FILENAME in world.snapshot_calls[1]["allow_patterns"]


def test_snapshot_with_no_checkpoint_at_all_is_refused(world: FakeWorld) -> None:
    world.has_safetensors = False
    world.has_bin = False
    with pytest.raises(
        PeCoreRuntimeError, match=r"neither .* nor .* — not an open_clip weights repo"
    ):
        _provider(world).embed_text("a dog")
    assert world.build_calls == []


def test_hub_failure_is_wrapped_and_names_the_pinned_repo(world: FakeWorld) -> None:
    world.hub_error = ConnectionError("no route to host")
    with pytest.raises(PeCoreRuntimeError, match=rf"{MIRROR}@{REVISION}.*ConnectionError"):
        _provider(world).embed_text("a dog")


def test_open_clip_build_failure_is_wrapped(world: FakeWorld) -> None:
    world.build_error = RuntimeError("Required pretrained weights could not be loaded")
    with pytest.raises(PeCoreRuntimeError, match=r"open_clip could not build.*Required pretrained"):
        _provider(world).embed_text("a dog")


def test_load_runs_once_across_calls(world: FakeWorld, tmp_path: Path) -> None:
    provider = _provider(world)
    provider.embed_text("a dog")
    provider.embed_text("a cat")
    provider.embed_images(_images(tmp_path, ["one.png"]))
    assert len(world.build_calls) == 1
    assert len(world.snapshot_calls) == 1
    assert len(world.hub_download_calls) == 1


def test_failed_load_is_retried_on_the_next_call(world: FakeWorld) -> None:
    world.mps_available = False
    provider = _provider(world)
    with pytest.raises(PeCoreRuntimeError):
        provider.embed_text("a dog")
    world.mps_available = True
    provider.embed_text("a dog")
    assert len(world.build_calls) == 1


# --------------------------------------------------------------------------
# embed_images
# --------------------------------------------------------------------------


def test_embed_images_returns_unit_vectors_in_input_order_batched(
    world: FakeWorld, tmp_path: Path
) -> None:
    names = ["alpha.png", "bravo.jpg", "charlie.png", "delta.heic", "echo.png"]
    paths = _images(tmp_path, names)
    provider = _provider(world, batch_size=2)

    vectors = provider.embed_images(paths)

    assert world.image_calls == [
        ["alpha.png", "bravo.jpg"],
        ["charlie.png", "delta.heic"],
        ["echo.png"],
    ]
    assert len(vectors) == 5
    for name, vec in zip(names, vectors, strict=True):
        assert len(vec) == world.dim
        assert abs(_norm(vec) - 1.0) < 1e-9
        assert _close(vec, _unit(world.vector_for(name)))
        assert all(isinstance(v, float) for v in vec)
    assert all(kw["normalize"] is True for kw in world.image_kwargs)
    assert all(kw["device"] == "mps" for kw in world.image_kwargs)


def test_embed_images_normalizes_even_when_the_backend_did_not(
    world: FakeWorld, tmp_path: Path
) -> None:
    """The fake tower ignores `normalize=True` and returns raw vectors;
    the provider's own L2 pass is what guarantees the contract."""
    (path,) = _images(tmp_path, ["raw.png"])
    raw = world.vector_for("raw.png")
    assert abs(_norm(raw) - 1.0) > 0.5
    (vec,) = _provider(world).embed_images([path])
    assert abs(_norm(vec) - 1.0) < 1e-9


def test_embed_images_empty_list_returns_empty_without_loading(world: FakeWorld) -> None:
    assert _provider(world).embed_images([]) == []
    assert world.build_calls == []


def test_images_are_exif_transposed_and_converted_to_rgb(world: FakeWorld, tmp_path: Path) -> None:
    paths = _images(tmp_path, ["gray.png", "cmyk.jpg"])
    _provider(world).embed_images(paths)
    # the fake preprocess raises unless it is handed an RGB image
    assert world.image_calls == [["gray.png", "cmyk.jpg"]]
    assert world.exif_calls == 2


def test_unreadable_image_raises_naming_the_path(world: FakeWorld, tmp_path: Path) -> None:
    paths = _images(tmp_path, ["good1.png", "broken.png", "good2.png"])
    world.unreadable = {"broken.png"}
    provider = _provider(world)

    with pytest.raises(UnreadableImageError) as excinfo:
        provider.embed_images(paths)

    assert excinfo.value.path == paths[1]
    assert str(paths[1]) in str(excinfo.value)
    assert "cannot identify image file" in excinfo.value.reason
    assert isinstance(excinfo.value, ImageEmbeddingError)
    # the bad file never reached a forward pass, and the good ones still work
    assert world.image_calls == []
    assert len(provider.embed_images([paths[0], paths[2]])) == 2


def test_missing_image_file_is_unreadable(world: FakeWorld, tmp_path: Path) -> None:
    missing = tmp_path / "never-written.png"
    with pytest.raises(UnreadableImageError) as excinfo:
        _provider(world).embed_images([missing])
    assert excinfo.value.path == missing


def test_batch_tower_failure_retries_individually_and_names_the_bad_item(
    world: FakeWorld, tmp_path: Path
) -> None:
    paths = _images(tmp_path, ["a.png", "b.png", "c.png", "d.png"])
    world.tower_fails_for = {"c.png"}
    provider = _provider(world, batch_size=4)

    with pytest.raises(ImageEmbeddingError) as excinfo:
        provider.embed_images(paths)

    assert excinfo.value.path == paths[2]
    assert "image tower failure" in excinfo.value.reason
    assert not isinstance(excinfo.value, UnreadableImageError)
    # the whole batch failed, then items were retried one by one until the culprit
    assert world.image_calls == [
        ["a.png", "b.png", "c.png", "d.png"],
        ["a.png"],
        ["b.png"],
        ["c.png"],
    ]


def test_transient_batch_failure_recovers_through_individual_retry(
    world: FakeWorld, tmp_path: Path
) -> None:
    paths = _images(tmp_path, ["a.png", "b.png", "c.png"])
    world.fail_batches_larger_than = 1  # any multi-image forward pass "runs out of memory"
    provider = _provider(world, batch_size=3)

    vectors = provider.embed_images(paths)

    assert world.image_calls == [["a.png", "b.png", "c.png"], ["a.png"], ["b.png"], ["c.png"]]
    assert [
        _close(v, _unit(world.vector_for(p.name))) for v, p in zip(vectors, paths, strict=True)
    ] == [
        True,
        True,
        True,
    ]


def test_single_image_tower_failure_names_the_path(world: FakeWorld, tmp_path: Path) -> None:
    (path,) = _images(tmp_path, ["solo.png"])
    world.tower_fails_for = {"solo.png"}
    with pytest.raises(ImageEmbeddingError) as excinfo:
        _provider(world).embed_images([path])
    assert excinfo.value.path == path
    assert world.image_calls == [["solo.png"]]


def test_zero_vector_from_the_image_tower_is_an_image_error(
    world: FakeWorld, tmp_path: Path
) -> None:
    paths = _images(tmp_path, ["fine.png", "blank.png"])
    world.zero_vector_for = {"blank.png"}
    with pytest.raises(ImageEmbeddingError, match="zero or non-finite") as excinfo:
        _provider(world).embed_images(paths)
    assert excinfo.value.path == paths[1]


# --------------------------------------------------------------------------
# embed_text
# --------------------------------------------------------------------------


def test_embed_text_uses_the_text_tower_and_normalizes(world: FakeWorld) -> None:
    provider = _provider(world)
    vec = provider.embed_text("a photo of a deck under construction")

    assert world.tokenizer_calls[-1] == ["a photo of a deck under construction"]
    assert world.text_calls[-1] == "a photo of a deck under construction"
    assert world.text_kwargs[-1] == {"normalize": True, "device": "mps"}
    assert len(vec) == world.dim
    assert abs(_norm(vec) - 1.0) < 1e-9
    assert _close(vec, _unit(world.text_vector("a photo of a deck under construction")))


def test_embed_text_has_no_instruction_prefix(world: FakeWorld) -> None:
    _provider(world).embed_text("kitchen tiles")
    assert world.tokenizer_calls[-1] == ["kitchen tiles"]


def test_embed_text_failure_is_an_embedding_error(world: FakeWorld) -> None:
    provider = _provider(world)
    provider.embed_text("warm up")  # load succeeds (the probe passes)

    def boom(_tokens: FakeTokens, normalize: bool = False) -> FakeTensor:
        raise RuntimeError("text tower failure")

    model = provider._runtime.model  # poking the loaded fake on purpose
    model.encode_text = boom
    with pytest.raises(EmbeddingError, match=r"text tower failed.*text tower failure"):
        provider.embed_text("after")


def test_provider_satisfies_the_protocol_surface(world: FakeWorld, tmp_path: Path) -> None:
    """The exact attribute/method set `imsg.embed.pipeline` and the
    retrieval service use, on a real (loaded) instance."""
    provider = _provider(world)
    assert isinstance(provider.model_id, str)
    assert isinstance(provider.dim, int)
    assert callable(provider.embed_images)
    assert callable(provider.embed_text)
    (vec,) = provider.embed_images(_images(tmp_path, ["p.png"]))
    assert len(vec) == provider.dim
    assert len(provider.embed_text("p")) == provider.dim
