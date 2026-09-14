"""Real `MultimodalEmbeddingProvider` for Meta's Perception Encoder
(PE-Core, D3a) on Apple Silicon: the secondary vector that
`attachment_mm_embedding` stores (SPEC §7.4) and §9.4 channel C
searches. Drops in behind `imsg.embed.provider.
MultimodalEmbeddingProvider` exactly where
`FakeMultimodalEmbeddingProvider` sits in tests.

Backend: `open_clip_torch` on PyTorch (`mps`). Chosen by checking, not
assuming — every lookup below was run on 2026-09-14:

- `facebook/PE-Core-G14-448` on the Hugging Face Hub (commit
  `a6046680086f67d1f24d4b465a240de0578dfc0b`, `library_name:
  perception-encoder`, Apache-2.0) holds a single checkpoint,
  `PE-Core-G14-448.pt`, in Meta's own layout. The only loader for that
  layout is Meta's `perception_models` package
  (`core.vision_encoder.pe.CLIP.from_config("PE-Core-G14-448")`),
  which is GitHub-only — `https://pypi.org/pypi/perception_models/json`
  (and the `perception-models` / `perception-encoder` spellings)
  return 404 — and whose `setup.py` declares "FAIR Noncommercial
  Research License" for the *code* (the weights themselves are
  Apache-2.0). Rejected: not installable from an index, and that
  licence must not enter a public repo's dependency set.
- `timm` (1.0.29 on PyPI, 2026-08-28) registers the *vision tower
  only*: `vit_pe_core_gigantic_patch14_448` (pretrained tag `.fb`, in
  `timm/models/eva.py` since v1.0.16). No text tower, so `embed_text`
  is impossible on timm alone. Used indirectly, through open_clip.
- `open_clip_torch` (3.3.0 on PyPI, 2026-02-27; requires
  `timm>=1.0.17`) registers `PE-Core-bigG-14-448` with pretrained tag
  `meta` in `src/open_clip/pretrained.py` — present in every release
  tag from v3.0.0 through v3.3.0, with the source comment "original at
  facebook/PE-Core-G14-448/PE-Core-G14-448.pt". Its model config
  (`model_configs/PE-Core-bigG-14-448.json`): `embed_dim` 1280, vision
  tower `timm_model_name: vit_pe_core_gigantic_patch14_448` with
  `timm_pool: map` at 448 px, text tower width 1280 / 24 layers / 20
  heads, `context_length` 72, `vocab_size` 49408 (the CLIP BPE
  `SimpleTokenizer`), `custom_text: true` (model class
  `CustomTextCLIP`). Preprocessing per its `_pecfg` helper: mean/std
  0.5, bilinear, `resize_mode: squash`. The weights come from
  `timm/PE-Core-bigG-14-448` (commit
  `17aa0c25addfa14198fa2ff73d845a22d433432e`, 2025-07-24,
  `library_name: open_clip`, Apache-2.0), whose model card describes
  itself as "an OpenCLIP (image + text) remaped version of the
  original" and ships `open_clip_config.json` (model + preprocess
  config) and `open_clip_model.safetensors`. Chosen.

Revision pinning: open_clip 3.3.0's `create_model` never forwards a
Hub `revision` (`download_pretrained_from_hf` accepts one, but neither
the `hf-hub:` branch nor the registry-tag branch passes it). So this
module fetches the snapshot itself with
`huggingface_hub.snapshot_download(revision=...)` and hands the
directory to open_clip's `local-dir:` schema, which reads
`open_clip_config.json` and the checkpoint next to it. `revision`
therefore pins a commit of the repo the weights are *fetched from*: for
the canonical `facebook/PE-Core-G14-448` id that is the
`timm/PE-Core-bigG-14-448` mirror (`OPEN_CLIP_MIRRORS`), so the
`embedding.multimodal.revision` value must be one of the mirror's
commits. `model_id` records what the operator configured, verbatim.

Backend API names this module relies on (all read from the v3.3.0
sources, never guessed):
`open_clip.create_model_and_transforms(model_name, device=,
require_pretrained=)` -> `(model, train_preprocess, eval_preprocess)`;
`open_clip.get_tokenizer(model_name)` -> callable
`(texts) -> LongTensor[n, context_length]`;
`model.encode_image(x, normalize=)`, `model.encode_text(t, normalize=)`,
`model.eval()`; `huggingface_hub.hf_hub_download(repo_id=, filename=,
revision=, cache_dir=)` and `huggingface_hub.snapshot_download(repo_id=,
revision=, cache_dir=, allow_patterns=)`;
`torch.backends.mps.is_available()`, `torch.cuda.is_available()`,
`torch.inference_mode()`, `torch.stack(...)`, `Tensor.to/float/cpu/
tolist`; `PIL.Image.open`, `PIL.ImageOps.exif_transpose`,
`Image.convert("RGB")`; `pillow_heif.register_heif_opener()` (1.7.0:
`(**kwargs) -> None`), optional — see `_register_heif_opener`.

Everything heavy is imported lazily inside `_load()` — via
`importlib.import_module`, so mypy strict (with `warn_unused_ignores`)
passes whether or not the `models` extra is installed — and importing
this module never needs torch. Every test in this build stubs the
runtime in `sys.modules`.
"""

from __future__ import annotations

import importlib
import json
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from imsg.errors import EmbeddingError, ImageEmbeddingError, UnreadableImageError

logger = structlog.get_logger(__name__)

MODELS_EXTRA_HINT = "install the `models` extra (`uv sync --extra models`)"

OPEN_CLIP_MIRRORS: dict[str, str] = {
    # Meta's canonical repo id -> the open_clip-layout mirror that
    # open_clip's own registry downloads from (`src/open_clip/
    # pretrained.py`, v3.0.0+; each entry there carries an "original
    # at facebook/..." comment naming the left-hand side).
    "facebook/PE-Core-G14-448": "timm/PE-Core-bigG-14-448",
    "facebook/PE-Core-L14-336": "timm/PE-Core-L-14-336",
    "facebook/PE-Core-B16-224": "timm/PE-Core-B-16",
    "facebook/PE-Core-S16-384": "timm/PE-Core-S-16-384",
    "facebook/PE-Core-T16-384": "timm/PE-Core-T-16-384",
}

OPEN_CLIP_CONFIG_FILENAME = "open_clip_config.json"
OPEN_CLIP_SAFETENSORS_FILENAME = "open_clip_model.safetensors"
OPEN_CLIP_PYTORCH_BIN_FILENAME = "open_clip_pytorch_model.bin"

_SNAPSHOT_PATTERNS: tuple[str, ...] = ("*.json", "*.txt", OPEN_CLIP_SAFETENSORS_FILENAME)
"""Config + tokenizer files + the safetensors weights. The multi-GB
`.bin` duplicate of the same weights is only fetched when a revision
has no safetensors file at all (see `_fetch_snapshot`)."""


class PeCoreRuntimeError(EmbeddingError):
    """The PE-Core runtime could not be brought up: a package from the
    `models` extra is missing, the requested device is unavailable (and
    CPU fallback was not allowed), the pinned weights could not be
    fetched or built, or the model's embedding width is not the
    configured `dim`. Abort-the-run territory (SPEC §8 S6: "model load
    failure -> abort run, nothing partial")."""


# `ImageEmbeddingError` / `UnreadableImageError` — the per-item failures
# this provider raises — are defined in `imsg.errors` (so the pipeline can
# catch them without importing this module) and re-exported here.


def resolve_weights_repo(model_repo: str) -> str:
    """The Hub repo the weights are actually fetched from: the
    open_clip-layout mirror for a canonical Meta id, otherwise
    `model_repo` itself (assumed to already be in open_clip layout,
    e.g. `timm/PE-Core-bigG-14-448` or a private fork of it)."""
    return OPEN_CLIP_MIRRORS.get(model_repo, model_repo)


def _import_runtime(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise PeCoreRuntimeError(
            f"the PE-Core multimodal runtime is not installed: could not import "
            f"{module!r} ({exc}) — {MODELS_EXTRA_HINT}"
        ) from exc


def _register_heif_opener() -> bool:
    """Teach PIL to decode HEIC/HEIF — the iPhone camera's default format,
    so a large share of image attachments — through `pillow-heif`
    (pinned in the `models` extra) when it is importable. Without it
    `Image.open` raises `UnidentifiedImageError` for every HEIC and each
    one fails as `UnreadableImageError`. Returns whether the opener was
    registered; the call is idempotent and cheap, so it runs on every
    load."""
    try:
        pillow_heif = importlib.import_module("pillow_heif")
    except ImportError:
        logger.warning(
            "pe_core.heif_unsupported",
            hint="install pillow-heif (`uv sync --extra models`) to embed HEIC attachments",
        )
        return False
    try:
        pillow_heif.register_heif_opener()
    except Exception as exc:  # a broken libheif build: HEICs fail per item, not the run
        logger.warning("pe_core.heif_registration_failed", error=f"{type(exc).__name__}: {exc}")
        return False
    return True


def _config_embed_dim(config_path: Path) -> int:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PeCoreRuntimeError(
            f"{config_path} is not a readable {OPEN_CLIP_CONFIG_FILENAME}: {exc}"
        ) from exc
    model_cfg = payload.get("model_cfg", payload) if isinstance(payload, dict) else None
    embed_dim = model_cfg.get("embed_dim") if isinstance(model_cfg, dict) else None
    if not isinstance(embed_dim, int) or isinstance(embed_dim, bool):
        raise PeCoreRuntimeError(
            f"{config_path} has no integer model_cfg.embed_dim — not an open_clip model config"
        )
    return embed_dim


def _rows(features: Any) -> list[list[float]]:
    """A `[n, dim]` tensor (any dtype, any device) as plain Python
    floats: fp32 first so half-precision outputs round-trip exactly
    through `tolist`, then off the accelerator."""
    raw = features.float().cpu().tolist()
    return [[float(v) for v in row] for row in raw]


def _l2_normalize(vec: Sequence[float]) -> list[float] | None:
    """Unit-normalise in Python regardless of what the backend did
    (`normalize=True` on the towers is asked for too; re-normalising a
    unit vector is a no-op). `None` for a zero or non-finite vector,
    which must never be stored as if it were an embedding."""
    norm = math.sqrt(math.fsum(v * v for v in vec))
    if norm == 0.0 or not math.isfinite(norm):
        return None
    return [v / norm for v in vec]


@dataclass(frozen=True, slots=True)
class _Runtime:
    torch: Any
    model: Any
    preprocess: Any
    tokenizer: Any
    pil_image: Any
    pil_image_ops: Any
    device: str


class PeCoreMultimodalEmbeddingProvider:
    """`MultimodalEmbeddingProvider` backed by open_clip's PE-Core port.

    `model_repo` is the Hub id the operator configured
    (`embedding.multimodal.model`, canonically
    `facebook/PE-Core-G14-448`); `revision` pins a commit of the repo
    the weights are fetched from (`resolve_weights_repo`); `dim` must
    equal the model's `embed_dim` (checked from the config before any
    weights download, and again on the loaded model). The pipeline
    separately requires `dim == imsg.constants.MULTIMODAL_EMBEDDING_DIM`
    (migration 0002's CHECK constraint) — that is its check, not this
    class's, so the class stays usable for the smaller PE-Core sizes.

    `device` is `mps` by default; when it is unavailable the load
    fails unless `allow_cpu_fallback=True`, in which case the towers
    run on CPU (correct, much slower). `batch_size` caps how many
    images share one forward pass. `cache_dir` is where
    `huggingface_hub` keeps the snapshot (its default cache when
    `None`).

    Nothing is loaded until the first `embed_images`/`embed_text`
    call; `_load()` runs once per instance (thread-safe).
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        dim: int,
        *,
        device: str = "mps",
        batch_size: int = 16,
        allow_cpu_fallback: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        if dim <= 0:
            raise EmbeddingError(f"dim must be a positive integer, got {dim}")
        if batch_size <= 0:
            raise EmbeddingError(f"batch_size must be a positive integer, got {batch_size}")
        self.model_id = f"{model_repo}@{revision or 'main'}"
        self.dim = dim
        self.model_repo = model_repo
        self.revision = revision
        self.weights_repo = resolve_weights_repo(model_repo)
        self._device = device
        self._batch_size = batch_size
        self._allow_cpu_fallback = allow_cpu_fallback
        self._cache_dir = cache_dir
        self._runtime: _Runtime | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _load(self) -> _Runtime:
        if self._runtime is not None:
            return self._runtime
        with self._load_lock:
            if self._runtime is None:
                self._runtime = self._bring_up()
            return self._runtime

    def _bring_up(self) -> _Runtime:
        torch = _import_runtime("torch")
        open_clip = _import_runtime("open_clip")
        hf_hub = _import_runtime("huggingface_hub")
        pil_image = _import_runtime("PIL.Image")
        pil_image_ops = _import_runtime("PIL.ImageOps")
        _register_heif_opener()

        # Fail on the cheap checks (device, config width) before
        # pulling several GB of weights.
        device = self._resolve_device(torch)
        snapshot = self._fetch_snapshot(hf_hub)

        model_name = f"local-dir:{snapshot}"
        try:
            # `require_pretrained=True` makes a snapshot without a
            # loadable checkpoint raise instead of open_clip's default
            # of *warning* and handing back a randomly initialised
            # model — silent garbage vectors, the worst failure mode.
            model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(
                model_name, device=device, require_pretrained=True
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise PeCoreRuntimeError(
                f"open_clip could not build {self.model_id} from {snapshot}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        model.eval()

        runtime = _Runtime(
            torch=torch,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            pil_image=pil_image,
            pil_image_ops=pil_image_ops,
            device=device,
        )
        # One cheap forward through the text tower proves the *loaded*
        # model's width, not just what its config claimed.
        self._embed_text_with(runtime, "a photo")
        logger.info(
            "pe_core.loaded",
            model_id=self.model_id,
            weights_repo=self.weights_repo,
            device=device,
            dim=self.dim,
        )
        return runtime

    def _resolve_device(self, torch: Any) -> str:
        requested = self._device
        kind = requested.split(":", 1)[0]
        if kind == "cpu":
            return requested
        if kind == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            available = mps_backend is not None and bool(mps_backend.is_available())
        elif kind == "cuda":
            available = bool(torch.cuda.is_available())
        else:
            raise PeCoreRuntimeError(
                f"unsupported device {requested!r}: expected 'mps', 'cuda', or 'cpu'"
            )
        if available:
            return requested
        if self._allow_cpu_fallback:
            logger.warning("pe_core.device_fallback", requested=requested, using="cpu")
            return "cpu"
        raise PeCoreRuntimeError(
            f"device {requested!r} is not available on this host; pass "
            f"allow_cpu_fallback=True to run the PE-Core towers on CPU instead (much slower)"
        )

    def _fetch_snapshot(self, hf_hub: Any) -> Path:
        cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
        pinned = f"{self.weights_repo}@{self.revision or 'main'}"
        try:
            config_path = Path(
                str(
                    hf_hub.hf_hub_download(
                        repo_id=self.weights_repo,
                        filename=OPEN_CLIP_CONFIG_FILENAME,
                        revision=self.revision,
                        cache_dir=cache_dir,
                    )
                )
            )
        except Exception as exc:
            raise PeCoreRuntimeError(
                f"could not fetch {OPEN_CLIP_CONFIG_FILENAME} for {pinned} "
                f"(resolved from {self.model_repo}): {type(exc).__name__}: {exc}"
            ) from exc
        embed_dim = _config_embed_dim(config_path)
        if embed_dim != self.dim:
            raise PeCoreRuntimeError(
                f"{pinned} declares embed_dim {embed_dim} but this provider was configured "
                f"with dim {self.dim} — refusing to load weights whose vectors cannot fill "
                f"attachment_mm_embedding"
            )

        def _snapshot(patterns: Sequence[str]) -> Path:
            logger.info(
                "pe_core.fetching_weights",
                weights_repo=self.weights_repo,
                revision=self.revision or "main",
            )
            try:
                return Path(
                    str(
                        hf_hub.snapshot_download(
                            repo_id=self.weights_repo,
                            revision=self.revision,
                            cache_dir=cache_dir,
                            allow_patterns=list(patterns),
                        )
                    )
                )
            except Exception as exc:
                raise PeCoreRuntimeError(
                    f"could not fetch weights for {pinned}: {type(exc).__name__}: {exc}"
                ) from exc

        snapshot = _snapshot(_SNAPSHOT_PATTERNS)
        if not (snapshot / OPEN_CLIP_SAFETENSORS_FILENAME).is_file():
            snapshot = _snapshot([*_SNAPSHOT_PATTERNS, OPEN_CLIP_PYTORCH_BIN_FILENAME])
        if not any(
            (snapshot / name).is_file()
            for name in (OPEN_CLIP_SAFETENSORS_FILENAME, OPEN_CLIP_PYTORCH_BIN_FILENAME)
        ):
            raise PeCoreRuntimeError(
                f"{pinned} has neither {OPEN_CLIP_SAFETENSORS_FILENAME} nor "
                f"{OPEN_CLIP_PYTORCH_BIN_FILENAME} — not an open_clip weights repo"
            )
        return snapshot

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            return []
        runtime = self._load()
        out: list[list[float]] = []
        for start in range(0, len(image_paths), self._batch_size):
            out.extend(
                self._embed_image_batch(runtime, image_paths[start : start + self._batch_size])
            )
        return out

    def _embed_image_batch(self, runtime: _Runtime, batch: list[Path]) -> list[list[float]]:
        # Decoding is per item, so an unreadable file is named directly
        # and never reaches a forward pass.
        tensors = [self._decode_image(runtime, path) for path in batch]
        try:
            rows = self._run_image_tower(runtime, tensors)
        except Exception as exc:
            if len(batch) == 1:
                raise ImageEmbeddingError(batch[0], f"{type(exc).__name__}: {exc}") from exc
            # Retry one at a time so the failure is pinned on the item
            # that actually causes it; the rest of the batch survives.
            logger.warning(
                "pe_core.image_batch_failed_retrying_individually",
                batch_size=len(batch),
                error=f"{type(exc).__name__}: {exc}",
            )
            rows = [
                self._run_single_image(runtime, path, tensor)
                for path, tensor in zip(batch, tensors, strict=True)
            ]
        return self._finalize_image_rows(rows, batch)

    def _decode_image(self, runtime: _Runtime, path: Path) -> Any:
        try:
            with runtime.pil_image.open(path) as opened:
                # iPhone photos routinely carry an EXIF orientation;
                # embedding them sideways would silently hurt retrieval.
                upright = runtime.pil_image_ops.exif_transpose(opened)
                rgb = upright.convert("RGB")
        except Exception as exc:
            raise UnreadableImageError(path, f"{type(exc).__name__}: {exc}") from exc
        try:
            return runtime.preprocess(rgb)
        except Exception as exc:
            raise ImageEmbeddingError(
                path, f"preprocessing failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _run_image_tower(self, runtime: _Runtime, tensors: list[Any]) -> list[list[float]]:
        torch = runtime.torch
        with torch.inference_mode():
            batch = torch.stack(tensors).to(runtime.device)
            features = runtime.model.encode_image(batch, normalize=True)
        return _rows(features)

    def _run_single_image(self, runtime: _Runtime, path: Path, tensor: Any) -> list[float]:
        try:
            rows = self._run_image_tower(runtime, [tensor])
        except Exception as exc:
            raise ImageEmbeddingError(path, f"{type(exc).__name__}: {exc}") from exc
        if len(rows) != 1:
            raise PeCoreRuntimeError(f"{self.model_id} returned {len(rows)} vectors for 1 image")
        return rows[0]

    def _finalize_image_rows(self, rows: list[list[float]], batch: list[Path]) -> list[list[float]]:
        if len(rows) != len(batch):
            raise PeCoreRuntimeError(
                f"{self.model_id} returned {len(rows)} vectors for {len(batch)} images"
            )
        out: list[list[float]] = []
        for path, row in zip(batch, rows, strict=True):
            self._check_width(row, "image")
            unit = _l2_normalize(row)
            if unit is None:
                raise ImageEmbeddingError(
                    path, "the image tower produced a zero or non-finite vector"
                )
            out.append(unit)
        return out

    # ------------------------------------------------------------------
    # text
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Text into the same space via the paired text tower — no
        instruction prefix (CLIP-style dual encoder, not Qwen3's
        asymmetric scheme). Input beyond the model's 72-token BPE
        context is truncated by the tokenizer, which is fine for the
        short channel-C queries this serves."""
        return self._embed_text_with(self._load(), text)

    def _embed_text_with(self, runtime: _Runtime, text: str) -> list[float]:
        torch = runtime.torch
        try:
            with torch.inference_mode():
                tokens = runtime.tokenizer([text]).to(runtime.device)
                features = runtime.model.encode_text(tokens, normalize=True)
            rows = _rows(features)
        except Exception as exc:
            raise EmbeddingError(
                f"{self.model_id} text tower failed: {type(exc).__name__}: {exc}"
            ) from exc
        if len(rows) != 1:
            raise PeCoreRuntimeError(f"{self.model_id} returned {len(rows)} vectors for 1 text")
        self._check_width(rows[0], "text")
        unit = _l2_normalize(rows[0])
        if unit is None:
            raise EmbeddingError(f"{self.model_id} text tower produced a zero or non-finite vector")
        return unit

    def _check_width(self, vec: Sequence[float], tower: str) -> None:
        if len(vec) != self.dim:
            raise PeCoreRuntimeError(
                f"{self.model_id} produced a {len(vec)}-wide {tower} embedding, "
                f"expected dim {self.dim}"
            )


__all__ = [
    "MODELS_EXTRA_HINT",
    "OPEN_CLIP_CONFIG_FILENAME",
    "OPEN_CLIP_MIRRORS",
    "OPEN_CLIP_PYTORCH_BIN_FILENAME",
    "OPEN_CLIP_SAFETENSORS_FILENAME",
    "ImageEmbeddingError",
    "PeCoreMultimodalEmbeddingProvider",
    "PeCoreRuntimeError",
    "UnreadableImageError",
    "resolve_weights_repo",
]
