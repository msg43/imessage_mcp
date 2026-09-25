"""The PE-Core text tower saved as its own checkpoint, and the loader that
builds the query side's runtime from it without the vision tower.

Why this exists
---------------

The MCP servers use PE-Core only to embed short query texts (SPEC §9.4
channel C). Loading that through open_clip builds the whole
`CustomTextCLIP` first: both towers are allocated and randomly
initialised on the CPU, moved to the device, and then overwritten by the
9.01 GiB checkpoint, after which the vision tower is dropped
(`imsg.embed.pe_core_multimodal._drop_unused_towers`). Timed on the
development host (M2 Ultra, warm page cache, 2026-09-24) the load took
22.9 s, of which 13.3 s was randomly initialising the vision tower and
7.0 s the text tower; the production host logged 102.8 s for the same
step after a restart. None of that work survives into the result.

This module removes it in two steps:

1. `write_text_tower_checkpoint` copies the 293 `text.*` tensors
   (537,233,920 fp32 parameters, 2.0 GiB) out of the pinned open_clip
   checkpoint, byte for byte, into `open_clip_text_tower.safetensors`,
   next to a byte-for-byte copy of `open_clip_config.json` and a small
   provenance file. The output is a `local_conversion` entry in
   `models/manifest.lock.yaml`, digest-checked by `imsg models verify`.
2. `load_text_tower` builds the text tower alone, on torch's `meta`
   device (no memory, no initialisation), then assigns every tensor from
   that file and moves the tower to the device. Nothing is initialised
   and then overwritten, and the vision tower is never built.

Every tensor comes from the file
--------------------------------

The persistent tensors are the upstream bytes. The tower also has
*non-persistent* buffers (`attn_mask`, the causal mask), which open_clip
computes in the constructor and never writes to a checkpoint. A tower
built on `meta` has no values for them, so the conversion stores them
too, computed by open_clip's own constructor, and lists their names in
the provenance file. The loader assigns them and then refuses to return a
tower that still holds any `meta` tensor. So every tensor the loaded
tower computes with was either copied from the upstream checkpoint or
computed by open_clip's constructor, exactly as on the full-model path,
and `scripts/verify_pe_core_text_tower.py` proves the resulting query
vectors are bit-identical to the full-model path's.

Fail-closed rules: the provenance must name the same weights repo and
revision the provider is configured with (a text tower from another
revision would embed into a different space from the stored image
vectors, silently); the tensor set must match the tower open_clip builds
exactly (`load_state_dict(strict=True)`); the embedding width must equal
the configured `dim`. Each violation raises `TextTowerCheckpointError`.

Like `imsg.embed.pe_core_multimodal`, every heavy package is imported
lazily by the caller and passed in, so importing this module needs
neither torch nor open_clip.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imsg.errors import EmbeddingError

TEXT_TOWER_FORMAT = "imsg-pe-core-text-tower/1"
TEXT_TOWER_WEIGHTS_FILENAME = "open_clip_text_tower.safetensors"
TEXT_TOWER_PROVENANCE_FILENAME = "imsg_text_tower.json"
TEXT_TOWER_CONFIG_FILENAME = "open_clip_config.json"
"""The same name open_clip's `local-dir:` reader looks for, so
`open_clip.get_tokenizer("local-dir:<dir>")` builds the identical
tokenizer from the copied config."""

UPSTREAM_WEIGHTS_FILENAME = "open_clip_model.safetensors"
TEXT_KEY_PREFIX = "text."
"""`CustomTextCLIP` stores the text tower under `.text`, so its tensors
are the checkpoint keys with this prefix."""

_SAFETENSORS_METADATA = {"format": "pt"}
"""Exactly one metadata entry. safetensors serialises the metadata map
in hash order, so more than one key could make two conversions of the
same input differ byte for byte; everything else goes in the provenance
JSON, which is written with sorted keys."""

_HASH_CHUNK_BYTES = 8 * 2**20


class TextTowerCheckpointError(EmbeddingError):
    """The text-tower checkpoint is missing a file, malformed, from a
    different model revision, or does not match the tower open_clip
    builds. The provider must not embed with it."""


@dataclass(frozen=True, slots=True)
class TextTowerProvenance:
    """`imsg_text_tower.json`: what the checkpoint was cut from, and which
    of its tensors are non-persistent buffers. Deterministic by design —
    no dates, hosts or package versions (those belong to the manifest
    entry's `tool`), so two conversions of the same input produce the same
    bytes and the same `artifact_sha256`."""

    upstream_repo: str
    upstream_revision: str
    upstream_weights_file: str
    upstream_weights_sha256: str
    tensor_count: int
    parameter_count: int
    non_persistent_buffers: tuple[str, ...]

    def to_json(self) -> str:
        payload = {
            "format": TEXT_TOWER_FORMAT,
            "tower": "text",
            "key_prefix_removed": TEXT_KEY_PREFIX,
            "upstream_repo": self.upstream_repo,
            "upstream_revision": self.upstream_revision,
            "upstream_weights_file": self.upstream_weights_file,
            "upstream_weights_sha256": self.upstream_weights_sha256,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "non_persistent_buffers": list(self.non_persistent_buffers),
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def read_provenance(text_dir: Path) -> TextTowerProvenance:
    """The provenance file of a text-tower directory, validated for shape.
    Whether it matches the provider's pin is `load_text_tower`'s check."""
    path = text_dir / TEXT_TOWER_PROVENANCE_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TextTowerCheckpointError(
            f"{text_dir} has no {TEXT_TOWER_PROVENANCE_FILENAME} — not a PE-Core text-tower "
            f"checkpoint (produce one with scripts/convert_pe_core_text_tower.py)"
        ) from exc
    except (OSError, ValueError) as exc:
        raise TextTowerCheckpointError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != TEXT_TOWER_FORMAT:
        raise TextTowerCheckpointError(
            f"{path} is not a {TEXT_TOWER_FORMAT!r} provenance file "
            f"(format {payload.get('format') if isinstance(payload, dict) else None!r})"
        )
    try:
        buffers = payload["non_persistent_buffers"]
        if not isinstance(buffers, list) or not all(isinstance(b, str) for b in buffers):
            raise TypeError("non_persistent_buffers must be a list of strings")
        provenance = TextTowerProvenance(
            upstream_repo=_require_str(payload, "upstream_repo"),
            upstream_revision=_require_str(payload, "upstream_revision"),
            upstream_weights_file=_require_str(payload, "upstream_weights_file"),
            upstream_weights_sha256=_require_str(payload, "upstream_weights_sha256"),
            tensor_count=_require_int(payload, "tensor_count"),
            parameter_count=_require_int(payload, "parameter_count"),
            non_persistent_buffers=tuple(buffers),
        )
    except (KeyError, TypeError) as exc:
        raise TextTowerCheckpointError(f"{path} is incomplete: {exc}") from exc
    return provenance


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{key} must be a non-negative integer")
    return value


def read_text_config(config_path: Path) -> tuple[int, dict[str, Any], bool]:
    """`(embed_dim, text_cfg, quick_gelu)` from an `open_clip_config.json`
    — exactly the three values `CustomTextCLIP.__init__` passes to
    open_clip's `_build_text_tower` (with `cast_dtype=None` at fp32, the
    precision the provider has always loaded at). Only open_clip's own
    `TextTransformer` is supported: a Hugging Face text encoder
    (`hf_model_name`) builds differently and is refused rather than
    guessed at."""
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TextTowerCheckpointError(f"{config_path} is not a readable open_clip config: {exc}") from exc
    model_cfg = payload.get("model_cfg") if isinstance(payload, dict) else None
    if not isinstance(model_cfg, dict):
        raise TextTowerCheckpointError(f"{config_path} has no model_cfg — not an open_clip config")
    embed_dim = model_cfg.get("embed_dim")
    text_cfg = model_cfg.get("text_cfg")
    if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or not isinstance(text_cfg, dict):
        raise TextTowerCheckpointError(
            f"{config_path} needs an integer model_cfg.embed_dim and a model_cfg.text_cfg mapping"
        )
    if not model_cfg.get("custom_text", False):
        raise TextTowerCheckpointError(
            f"{config_path} is not a custom-text (two-tower `CustomTextCLIP`) model: its text "
            f"weights are not stored under {TEXT_KEY_PREFIX!r}"
        )
    if text_cfg.get("hf_model_name"):
        raise TextTowerCheckpointError(
            f"{config_path} uses a Hugging Face text encoder ({text_cfg['hf_model_name']!r}); "
            f"only open_clip's own TextTransformer is supported"
        )
    return embed_dim, dict(text_cfg), bool(model_cfg.get("quick_gelu", False))


def build_text_tower(
    open_clip_model: Any, *, embed_dim: int, text_cfg: Mapping[str, Any], quick_gelu: bool
) -> Any:
    """open_clip's own text-tower builder, the function `CustomTextCLIP`
    calls for its `.text` attribute, with the arguments it passes at fp32.
    A private name in open_clip 3.x (`open_clip.model._build_text_tower`),
    so its absence is reported as a runtime mismatch rather than an
    AttributeError."""
    builder = getattr(open_clip_model, "_build_text_tower", None)
    if not callable(builder):
        raise TextTowerCheckpointError(
            "open_clip.model has no _build_text_tower — this open_clip version cannot build a "
            "text tower on its own; the manifest entry's min_runtime names the tested version"
        )
    return builder(embed_dim, dict(text_cfg), quick_gelu, None)


def non_persistent_buffer_names(tower: Any) -> tuple[str, ...]:
    """Buffers the module holds but does not put in its state dict — here,
    open_clip's causal `attn_mask`. Public API only: every buffer minus
    the state dict's keys."""
    persistent = set(tower.state_dict(keep_vars=True).keys())
    return tuple(sorted(name for name, _ in tower.named_buffers() if name not in persistent))


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_tower_checkpoint(
    snapshot_dir: Path,
    out_dir: Path,
    *,
    upstream_repo: str,
    upstream_revision: str,
    torch: Any,
    open_clip_model: Any,
    safetensors_torch: Any,
    safe_open: Any,
) -> TextTowerProvenance:
    """Cut the text tower out of an open_clip snapshot into `out_dir`.

    `snapshot_dir` holds `open_clip_config.json` and
    `open_clip_model.safetensors` at `upstream_repo@upstream_revision`.
    The `text.*` tensors are read one at a time with `safe_open` (the 7
    GiB vision half is never read) and written unchanged, with the prefix
    removed so the names are the tower's own state-dict keys. The tower is
    also built once on the CPU, only to check the tensor set against what
    open_clip constructs (`strict=True`) and to take its non-persistent
    buffers.

    `out_dir` must not exist; the result is written to a temporary
    sibling and renamed into place, so a failed run leaves nothing a
    loader could mistake for a checkpoint.
    """
    if out_dir.exists():
        raise TextTowerCheckpointError(f"{out_dir} already exists; refusing to overwrite it")
    config_path = snapshot_dir / TEXT_TOWER_CONFIG_FILENAME
    weights_path = snapshot_dir / UPSTREAM_WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise TextTowerCheckpointError(
            f"{snapshot_dir} has no {UPSTREAM_WEIGHTS_FILENAME}; the conversion reads the "
            f"safetensors checkpoint only"
        )
    embed_dim, text_cfg, quick_gelu = read_text_config(config_path)

    persistent: dict[str, Any] = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            if key.startswith(TEXT_KEY_PREFIX):
                persistent[key.removeprefix(TEXT_KEY_PREFIX)] = handle.get_tensor(key)
    if not persistent:
        raise TextTowerCheckpointError(f"{weights_path} holds no {TEXT_KEY_PREFIX!r} tensors")

    tower = build_text_tower(
        open_clip_model, embed_dim=embed_dim, text_cfg=text_cfg, quick_gelu=quick_gelu
    )
    try:
        tower.load_state_dict(persistent, strict=True)
    except RuntimeError as exc:
        raise TextTowerCheckpointError(
            f"the {TEXT_KEY_PREFIX!r} tensors of {weights_path} do not match the text tower "
            f"open_clip builds from {config_path}: {exc}"
        ) from exc
    buffer_names = non_persistent_buffer_names(tower)
    all_buffers = dict(tower.named_buffers())
    tensors: dict[str, Any] = dict(persistent)
    for name in buffer_names:
        if name in tensors:
            raise TextTowerCheckpointError(f"non-persistent buffer {name!r} collides with a weight")
        tensors[name] = all_buffers[name].detach().to("cpu").contiguous()

    provenance = TextTowerProvenance(
        upstream_repo=upstream_repo,
        upstream_revision=upstream_revision,
        upstream_weights_file=UPSTREAM_WEIGHTS_FILENAME,
        upstream_weights_sha256=sha256_of_file(weights_path),
        tensor_count=len(persistent),
        parameter_count=sum(int(t.numel()) for t in persistent.values()),
        non_persistent_buffers=buffer_names,
    )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        safetensors_torch.save_file(
            tensors, str(staging / TEXT_TOWER_WEIGHTS_FILENAME), metadata=dict(_SAFETENSORS_METADATA)
        )
        shutil.copyfile(config_path, staging / TEXT_TOWER_CONFIG_FILENAME)
        (staging / TEXT_TOWER_PROVENANCE_FILENAME).write_text(provenance.to_json(), encoding="utf-8")
        os.chmod(staging, 0o755)
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return provenance


class TextTowerModel:
    """What `CustomTextCLIP.encode_text` computes, for a runtime that holds
    only the text tower: `F.normalize(self.text(tokens), dim=-1)` when
    `normalize`. Same call, same tensors, so the same bits. There is no
    vision tower; `encode_image` raising is the point — the provider
    rebuilds the full model before any image call."""

    visual: None = None

    def __init__(self, text: Any, functional: Any) -> None:
        self.text = text
        self._functional = functional

    def eval(self) -> TextTowerModel:
        self.text.eval()
        return self

    def encode_text(self, tokens: Any, normalize: bool = False) -> Any:
        features = self.text(tokens)
        return self._functional.normalize(features, dim=-1) if normalize else features

    def encode_image(self, images: Any, normalize: bool = False) -> Any:
        raise TextTowerCheckpointError(
            "this runtime holds only the PE-Core text tower; images need the full model"
        )


def load_text_tower(
    text_dir: Path,
    *,
    device: str,
    weights_repo: str,
    revision: str | None,
    dim: int,
    torch: Any,
    open_clip: Any,
    open_clip_model: Any,
    safetensors_torch: Any,
) -> tuple[TextTowerModel, Any]:
    """`(model, tokenizer)` for query embedding, built from a text-tower
    checkpoint without building or reading the vision tower.

    Raises `TextTowerCheckpointError` when the checkpoint was cut from a
    different repo or revision than `weights_repo@revision`, when its
    width is not `dim`, or when its tensors do not exactly cover the tower
    open_clip builds."""
    provenance = read_provenance(text_dir)
    if provenance.upstream_repo != weights_repo or provenance.upstream_revision != revision:
        raise TextTowerCheckpointError(
            f"{text_dir} was cut from {provenance.upstream_repo}@{provenance.upstream_revision}, "
            f"but the multimodal provider is pinned to {weights_repo}@{revision or 'main'}; its "
            f"query vectors would not share a space with the stored image vectors"
        )
    embed_dim, text_cfg, quick_gelu = read_text_config(text_dir / TEXT_TOWER_CONFIG_FILENAME)
    if embed_dim != dim:
        raise TextTowerCheckpointError(
            f"{text_dir} declares embed_dim {embed_dim}, but the provider is configured with dim {dim}"
        )
    weights_path = text_dir / TEXT_TOWER_WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise TextTowerCheckpointError(f"{text_dir} has no {TEXT_TOWER_WEIGHTS_FILENAME}")
    try:
        tensors = safetensors_torch.load_file(str(weights_path), device="cpu")
    except Exception as exc:
        raise TextTowerCheckpointError(
            f"could not read {weights_path}: {type(exc).__name__}: {exc}"
        ) from exc

    with torch.device("meta"):
        tower = build_text_tower(
            open_clip_model, embed_dim=embed_dim, text_cfg=text_cfg, quick_gelu=quick_gelu
        )
    expected_buffers = non_persistent_buffer_names(tower)
    if tuple(sorted(provenance.non_persistent_buffers)) != expected_buffers:
        raise TextTowerCheckpointError(
            f"{text_dir} stores non-persistent buffers {list(provenance.non_persistent_buffers)}, "
            f"but open_clip's text tower has {list(expected_buffers)}"
        )
    missing_buffers = [name for name in expected_buffers if name not in tensors]
    if missing_buffers:
        raise TextTowerCheckpointError(f"{weights_path} lacks non-persistent buffers {missing_buffers}")
    persistent = {k: v for k, v in tensors.items() if k not in set(expected_buffers)}
    try:
        tower.load_state_dict(persistent, strict=True, assign=True)
    except RuntimeError as exc:
        raise TextTowerCheckpointError(
            f"{weights_path} does not match the text tower open_clip builds: {exc}"
        ) from exc
    for name in expected_buffers:
        owner_path, _, attribute = name.rpartition(".")
        owner = tower.get_submodule(owner_path) if owner_path else tower
        owner.register_buffer(attribute, tensors[name], persistent=False)
    left_on_meta = [
        name
        for name, tensor in [*tower.named_parameters(), *tower.named_buffers()]
        if tensor.is_meta
    ]
    if left_on_meta:
        raise TextTowerCheckpointError(
            f"{weights_path} left {len(left_on_meta)} tensor(s) without values "
            f"(first: {left_on_meta[0]!r}); refusing to embed with them"
        )
    tower = tower.to(device)
    model = TextTowerModel(tower, torch.nn.functional).eval()
    try:
        tokenizer = open_clip.get_tokenizer(f"local-dir:{text_dir}")
    except Exception as exc:
        raise TextTowerCheckpointError(
            f"open_clip could not build the tokenizer from {text_dir}: {type(exc).__name__}: {exc}"
        ) from exc
    return model, tokenizer


__all__ = [
    "TEXT_KEY_PREFIX",
    "TEXT_TOWER_CONFIG_FILENAME",
    "TEXT_TOWER_FORMAT",
    "TEXT_TOWER_PROVENANCE_FILENAME",
    "TEXT_TOWER_WEIGHTS_FILENAME",
    "UPSTREAM_WEIGHTS_FILENAME",
    "TextTowerCheckpointError",
    "TextTowerModel",
    "TextTowerProvenance",
    "build_text_tower",
    "load_text_tower",
    "non_persistent_buffer_names",
    "read_provenance",
    "read_text_config",
    "sha256_of_file",
    "write_text_tower_checkpoint",
]
