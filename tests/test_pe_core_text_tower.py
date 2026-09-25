"""The PE-Core text-tower checkpoint (`imsg.embed.pe_core_text_tower`) and
the provider's text-only load path.

The identity tests use the REAL torch, open_clip and safetensors (the
`models` extra) on a tiny two-tower `CustomTextCLIP` built from an
open_clip config, so they exercise open_clip's own text-tower code, not a
stand-in. They skip when the extra is not installed; the provenance and
config checks at the top need none of it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from imsg.embed.pe_core_text_tower import (
    TEXT_TOWER_CONFIG_FILENAME,
    TEXT_TOWER_FORMAT,
    TEXT_TOWER_PROVENANCE_FILENAME,
    TEXT_TOWER_WEIGHTS_FILENAME,
    UPSTREAM_WEIGHTS_FILENAME,
    TextTowerCheckpointError,
    TextTowerProvenance,
    read_provenance,
    read_text_config,
)
from imsg.errors import EmbeddingError

REPO = "example-org/tiny-clip"
REVISION = "5" * 40
OTHER_REVISION = "6" * 40
DIM = 16
TEXTS = (
    "a photo of a kite over a harbour",
    "",
    "Café crème — naïve",
    "東京タワー",
    "the quick brown fox jumps over the lazy dog " * 6,  # past the 12-token context
)

TINY_CONFIG: dict[str, Any] = {
    "model_cfg": {
        "embed_dim": DIM,
        "vision_cfg": {"image_size": 32, "layers": 1, "width": 32, "patch_size": 16, "head_width": 16},
        # vocab_size must cover open_clip's SimpleTokenizer ids (49408).
        "text_cfg": {"context_length": 12, "vocab_size": 49408, "width": 32, "heads": 2, "layers": 2},
        "custom_text": True,
    },
    "preprocess_cfg": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "interpolation": "bilinear",
        "resize_mode": "squash",
    },
}


def _bits(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}d", *vector)


# --------------------------------------------------------------------------
# provenance and config checks — no torch needed
# --------------------------------------------------------------------------


def _provenance(**overrides: Any) -> TextTowerProvenance:
    fields: dict[str, Any] = {
        "upstream_repo": REPO,
        "upstream_revision": REVISION,
        "upstream_weights_file": UPSTREAM_WEIGHTS_FILENAME,
        "upstream_weights_sha256": "a" * 64,
        "tensor_count": 3,
        "parameter_count": 42,
        "non_persistent_buffers": ("attn_mask",),
    }
    fields.update(overrides)
    return TextTowerProvenance(**fields)


def test_provenance_json_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    provenance = _provenance()
    text = provenance.to_json()
    assert text == _provenance().to_json()
    payload = json.loads(text)
    assert list(payload) == sorted(payload), "keys are written sorted, so bytes are stable"
    assert payload["format"] == TEXT_TOWER_FORMAT
    (tmp_path / TEXT_TOWER_PROVENANCE_FILENAME).write_text(text, encoding="utf-8")
    assert read_provenance(tmp_path) == provenance


def test_read_provenance_refuses_a_directory_without_one(tmp_path: Path) -> None:
    with pytest.raises(TextTowerCheckpointError, match="not a PE-Core text-tower checkpoint"):
        read_provenance(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.update(format="something-else/1"), "provenance file"),
        (lambda p: p.pop("upstream_revision"), "incomplete"),
        (lambda p: p.update(tensor_count=-1), "incomplete"),
        (lambda p: p.update(non_persistent_buffers="attn_mask"), "incomplete"),
    ],
)
def test_read_provenance_refuses_malformed_files(tmp_path: Path, mutate: Any, match: str) -> None:
    payload = json.loads(_provenance().to_json())
    mutate(payload)
    (tmp_path / TEXT_TOWER_PROVENANCE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TextTowerCheckpointError, match=match):
        read_provenance(tmp_path)


def test_text_config_is_read_exactly_as_custom_text_clip_passes_it(tmp_path: Path) -> None:
    path = tmp_path / TEXT_TOWER_CONFIG_FILENAME
    path.write_text(json.dumps(TINY_CONFIG), encoding="utf-8")
    embed_dim, text_cfg, quick_gelu = read_text_config(path)
    assert (embed_dim, quick_gelu) == (DIM, False)
    assert text_cfg == TINY_CONFIG["model_cfg"]["text_cfg"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"custom_text": False}, "custom-text"),
        ({"text_cfg": {"hf_model_name": "example/bert", "width": 8}}, "Hugging Face text encoder"),
        ({"embed_dim": "16"}, "integer model_cfg.embed_dim"),
    ],
)
def test_text_config_refuses_models_this_loader_cannot_build(
    tmp_path: Path, change: dict[str, Any], match: str
) -> None:
    config = json.loads(json.dumps(TINY_CONFIG))
    config["model_cfg"].update(change)
    path = tmp_path / TEXT_TOWER_CONFIG_FILENAME
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(TextTowerCheckpointError, match=match):
        read_text_config(path)


def test_checkpoint_errors_are_embedding_errors() -> None:
    """The provider's callers already treat `EmbeddingError` as a load
    failure (SPEC §8 S6: abort, nothing partial)."""
    assert issubclass(TextTowerCheckpointError, EmbeddingError)


# --------------------------------------------------------------------------
# real torch + open_clip on a tiny model
# --------------------------------------------------------------------------


@pytest.fixture
def runtime() -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    open_clip = pytest.importorskip("open_clip")
    open_clip_model = pytest.importorskip("open_clip.model")
    safetensors = pytest.importorskip("safetensors")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    return {
        "torch": torch,
        "open_clip": open_clip,
        "open_clip_model": open_clip_model,
        "safetensors_torch": safetensors_torch,
        "safe_open": safetensors.safe_open,
    }


@pytest.fixture
def snapshot(tmp_path: Path, runtime: dict[str, Any]) -> Path:
    """An open_clip `local-dir:` snapshot of a randomly initialised tiny
    two-tower model: `open_clip_config.json` + `open_clip_model.safetensors`,
    the layout of the pinned PE-Core mirror."""
    torch = runtime["torch"]
    torch.manual_seed(1234)
    model_cfg = {k: v for k, v in TINY_CONFIG["model_cfg"].items() if k != "custom_text"}
    model = runtime["open_clip_model"].CustomTextCLIP(**model_cfg)
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / TEXT_TOWER_CONFIG_FILENAME).write_text(json.dumps(TINY_CONFIG), encoding="utf-8")
    runtime["safetensors_torch"].save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(directory / UPSTREAM_WEIGHTS_FILENAME),
    )
    return directory


def _convert(snapshot: Path, out: Path, runtime: dict[str, Any]) -> TextTowerProvenance:
    from imsg.embed.pe_core_text_tower import write_text_tower_checkpoint

    return write_text_tower_checkpoint(
        snapshot,
        out,
        upstream_repo=REPO,
        upstream_revision=REVISION,
        torch=runtime["torch"],
        open_clip_model=runtime["open_clip_model"],
        safetensors_torch=runtime["safetensors_torch"],
        safe_open=runtime["safe_open"],
    )


def _load(text_dir: Path, runtime: dict[str, Any], *, revision: str = REVISION, dim: int = DIM) -> Any:
    from imsg.embed.pe_core_text_tower import load_text_tower

    return load_text_tower(
        text_dir,
        device="cpu",
        weights_repo=REPO,
        revision=revision,
        dim=dim,
        torch=runtime["torch"],
        open_clip=runtime["open_clip"],
        open_clip_model=runtime["open_clip_model"],
        safetensors_torch=runtime["safetensors_torch"],
    )


def _reference(snapshot: Path, runtime: dict[str, Any]) -> tuple[Any, Any]:
    """The full-model path the provider has always used: open_clip builds
    both towers and loads the whole checkpoint."""
    open_clip = runtime["open_clip"]
    model, _, _ = open_clip.create_model_and_transforms(
        f"local-dir:{snapshot}", device="cpu", require_pretrained=True
    )
    model.eval()
    return model, open_clip.get_tokenizer(f"local-dir:{snapshot}")


def _encode(model: Any, tokenizer: Any, text: str, torch: Any) -> list[float]:
    with torch.inference_mode():
        features = model.encode_text(tokenizer([text]), normalize=True)
    return [float(v) for v in features.float().cpu().tolist()[0]]


def test_text_tower_checkpoint_gives_bit_identical_vectors_and_tensors(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    torch = runtime["torch"]
    out = tmp_path / "models" / "tiny-text"
    provenance = _convert(snapshot, out, runtime)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [TEXT_TOWER_CONFIG_FILENAME, TEXT_TOWER_PROVENANCE_FILENAME, TEXT_TOWER_WEIGHTS_FILENAME]
    )
    assert provenance.non_persistent_buffers == ("attn_mask",)

    reference, ref_tokenizer = _reference(snapshot, runtime)
    candidate, tokenizer = _load(out, runtime)
    assert candidate.visual is None
    for text in TEXTS:
        assert _bits(_encode(candidate, tokenizer, text, torch)) == _bits(
            _encode(reference, ref_tokenizer, text, torch)
        ), text

    def every_tensor(module: Any) -> dict[str, Any]:
        found = {f"p:{n}": t for n, t in module.named_parameters()}
        found.update({f"b:{n}": t for n, t in module.named_buffers()})
        return found

    ref_tensors, cand_tensors = every_tensor(reference.text), every_tensor(candidate.text)
    assert set(ref_tensors) == set(cand_tensors)
    assert "b:attn_mask" in cand_tensors
    for name, tensor in ref_tensors.items():
        other = cand_tensors[name]
        assert other.dtype == tensor.dtype and other.shape == tensor.shape, name
        assert not other.is_meta, name
        assert torch.equal(other, tensor), name


def test_conversion_reads_only_text_tensors_and_is_byte_reproducible(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    from imsg.providers.manifest import artifact_digest

    first = _convert(snapshot, tmp_path / "a" / "tiny-text", runtime)
    second = _convert(snapshot, tmp_path / "b" / "tiny-text", runtime)
    assert first == second
    assert artifact_digest(tmp_path / "a" / "tiny-text").sha256 == artifact_digest(
        tmp_path / "b" / "tiny-text"
    ).sha256
    with runtime["safe_open"](
        str(tmp_path / "a" / "tiny-text" / TEXT_TOWER_WEIGHTS_FILENAME), framework="pt"
    ) as handle:
        keys = set(handle.keys())
    assert not any(k.startswith(("visual.", "text.")) or k == "logit_scale" for k in keys)
    assert first.tensor_count == len(keys) - len(first.non_persistent_buffers)
    # The copied config is the upstream file, byte for byte.
    assert (tmp_path / "a" / "tiny-text" / TEXT_TOWER_CONFIG_FILENAME).read_bytes() == (
        snapshot / TEXT_TOWER_CONFIG_FILENAME
    ).read_bytes()


def test_conversion_refuses_to_overwrite_and_leaves_nothing_on_failure(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    out = tmp_path / "models" / "tiny-text"
    out.mkdir(parents=True)
    with pytest.raises(TextTowerCheckpointError, match="already exists"):
        _convert(snapshot, out, runtime)

    (snapshot / UPSTREAM_WEIGHTS_FILENAME).unlink()
    fresh = tmp_path / "models" / "fresh"
    with pytest.raises(TextTowerCheckpointError, match="safetensors checkpoint only"):
        _convert(snapshot, fresh, runtime)
    assert not fresh.exists()
    assert [p.name for p in (tmp_path / "models").iterdir()] == ["tiny-text"]


def test_loader_refuses_a_checkpoint_cut_from_another_revision(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    with pytest.raises(TextTowerCheckpointError, match="would not share a space"):
        _load(out, runtime, revision=OTHER_REVISION)


def test_loader_refuses_a_width_other_than_the_configured_dim(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    with pytest.raises(TextTowerCheckpointError, match="embed_dim 16"):
        _load(out, runtime, dim=DIM * 2)


def _rewrite_weights(text_dir: Path, runtime: dict[str, Any], edit: Any) -> None:
    path = text_dir / TEXT_TOWER_WEIGHTS_FILENAME
    tensors = runtime["safetensors_torch"].load_file(str(path))
    edit(tensors)
    runtime["safetensors_torch"].save_file(tensors, str(path), metadata={"format": "pt"})


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda t: t.pop("ln_final.weight"), "does not match the text tower"),
        (lambda t: t.update(extra=t["ln_final.bias"].clone()), "does not match the text tower"),
        (lambda t: t.pop("attn_mask"), "lacks non-persistent buffers"),
    ],
)
def test_loader_refuses_tensors_that_do_not_cover_the_tower_exactly(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any], edit: Any, match: str
) -> None:
    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    _rewrite_weights(out, runtime, edit)
    with pytest.raises(TextTowerCheckpointError, match=match):
        _load(out, runtime)


def test_loader_refuses_a_provenance_that_misnames_the_non_persistent_buffers(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    path = out / TEXT_TOWER_PROVENANCE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["non_persistent_buffers"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TextTowerCheckpointError, match="non-persistent buffers"):
        _load(out, runtime)


# --------------------------------------------------------------------------
# the provider: which path loads, and what it computes
# --------------------------------------------------------------------------


def _provider(snapshot: Path, text_dir: Path | None, *, forbid_snapshot: bool = False) -> Any:
    from imsg.embed.pe_core_multimodal import PeCoreMultimodalEmbeddingProvider

    provider = PeCoreMultimodalEmbeddingProvider(
        REPO, REVISION, DIM, device="cpu", text_tower_dir=text_dir
    )

    def fetch_snapshot(hub: Any) -> Path:
        if forbid_snapshot:
            raise AssertionError("the whole-model snapshot was fetched")
        return snapshot

    provider._fetch_snapshot = fetch_snapshot  # type: ignore[method-assign,assignment]
    return provider


def test_provider_text_call_loads_only_the_checkpoint_and_matches_the_full_model(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    full = _provider(snapshot, None)
    tower = _provider(snapshot, out, forbid_snapshot=True)
    for text in TEXTS:
        assert _bits(tower.embed_text(text)) == _bits(full.embed_text(text)), text
    assert tower.loaded_source == "text_tower_checkpoint"
    assert full.loaded_source == "full_model"
    assert tower.model_id == full.model_id, "same vectors, same provenance"


def test_provider_reloads_from_the_checkpoint_after_an_unload(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    """The emergency release (D10.5) unloads the models and the public
    server reloads them: that reload is the one this checkpoint shortens."""
    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    tower = _provider(snapshot, out, forbid_snapshot=True)
    before = tower.embed_text(TEXTS[0])
    tower.unload()
    assert tower.loaded_source is None
    assert _bits(tower.embed_text(TEXTS[0])) == _bits(before)
    assert tower.loaded_source == "text_tower_checkpoint"


def test_provider_image_call_after_a_text_only_load_rebuilds_the_full_model(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    from PIL import Image

    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    tower = _provider(snapshot, out)
    text_before = tower.embed_text(TEXTS[0])
    image_path = tmp_path / "probe.png"
    Image.new("RGB", (40, 40), (10, 200, 30)).save(image_path)
    [image_vector] = tower.embed_images([image_path])
    assert len(image_vector) == DIM
    assert tower.loaded_source == "full_model"
    assert _bits(tower.embed_text(TEXTS[0])) == _bits(text_before)


def test_provider_refuses_a_mismatched_checkpoint_instead_of_embedding(
    tmp_path: Path, snapshot: Path, runtime: dict[str, Any]
) -> None:
    from imsg.embed.pe_core_multimodal import PeCoreMultimodalEmbeddingProvider

    out = tmp_path / "tiny-text"
    _convert(snapshot, out, runtime)
    provider = PeCoreMultimodalEmbeddingProvider(
        REPO, OTHER_REVISION, DIM, device="cpu", text_tower_dir=out
    )
    with pytest.raises(TextTowerCheckpointError):
        provider.embed_text("anything")
    assert not provider.is_loaded
