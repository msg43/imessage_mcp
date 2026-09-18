"""One loaded copy of a vision-language model, shared by every provider
in a process that names the same repo and revision (D10.3 remedy,
defect 1).

The problem this exists to solve, measured
-----------------------------------------

`imsg.enrich.mlx_vlm_caption.MlxVlmCaptionProvider` (captioning) and
`imsg.segment.mlx_boundaries.MlxBoundaryProvider` (topical boundary
detection) are pinned to the *same* checkpoint — `imsg.constants`
sets ``CAPTION_MODEL_REPO = BOUNDARY_MODEL_REPO`` and the model
manifest records one entry serving both roles. They nevertheless built
two independent model objects, because one went through ``mlx_vlm.load``
and the other through ``mlx_lm.load``. On the production host MLX's
active memory went 19.00 -> 37.15 GiB as the second one loaded, which is
most of a 64 GB box's spare capacity. Reproduced on the development host
(2026-09-17): 18.99 GiB after ``mlx_vlm.load``, 37.15 GiB after
``mlx_lm.load`` of the same pin in the same process.

Why one copy is enough
----------------------

A vision-language checkpoint contains a complete text-only language
model; ``mlx_vlm``'s model object exposes it as ``.language_model``,
and ``mlx_vlm.stream_generate(..., image=None)`` runs a text-only
conversation through it. Checked on the development host against the
pinned 4-bit 35B (2026-09-17), not assumed:

- the chat template rendered by ``mlx_vlm.prompt_utils.
  apply_chat_template`` for a boundary prompt is **byte-identical** to
  the one ``tokenizer.apply_chat_template`` renders on the ``mlx_lm``
  side (same SHA-256 over the rendered string);
- the last-position logit row for a fixed probe sentence is
  **bit-identical** between ``mlx_vlm``'s ``.language_model`` and
  ``mlx_lm``'s model: the same SHA-256 over all 248,320 float32
  values;
- both produce the same boundary answer for the same window.

So routing boundary detection through an already-loaded VLM is not an
approximation of the ``mlx_lm`` path — it is the same computation on the
same weights. The one measurable difference is that the VLM object also
holds the vision tower, +0.83 GiB (18.99 vs 18.16 GiB), which is the
price of not holding a second 18 GiB copy.

Explicit, not a hidden global
-----------------------------

There is no module-level cache here and no global registry. A caller
constructs a :class:`SharedVlmRuntime` and passes the *same instance* to
every provider that should share; a provider given none makes its own
private one and behaves exactly as it did before. That makes the sharing
an argument you can see at the call site, and lets a test assert on
:attr:`SharedVlmRuntime.loaded_model_ids` that exactly one load happened.

The runtime is **not** thread-safe by itself for concurrent first loads;
the providers that use it serialize their own loads, and the processes
that share one (``imsg enrich``, ``imsg sync``) build their providers up
front on one thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import structlog

from imsg.enrich.model_runtime import (
    ModelRuntimeUnavailableError,
    import_runtime_module,
    resolve_model_snapshot,
)
from imsg.mlx_runtime import bound_buffer_cache

logger = structlog.get_logger(__name__)

MLX_VLM_INSTALL_HINT = "install `mlx-vlm` (Apple silicon only)"

DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES = 4 * 2**30
"""The MLX buffer-cache bound the enrichment-side providers apply when
their weights load (D10.2).

MLX pools freed GPU buffers instead of returning them to the OS, and its
default limit is its *memory* limit — probed at 60.8 GiB on the 64 GB
production host, where one enrichment process was seen holding 37.25 GiB
of pooled freed buffers. The query side has bounded itself at 8 GiB since
`imsg.mlx_runtime.DEFAULT_CACHE_LIMIT_BYTES` and held exactly that across
755 searches; the enrichment providers called nothing.

4 GiB rather than 8: captioning and boundary detection run one prompt at
a time against one resident model, so the activation buffers they free
are a single forward pass's worth, not a 64-row padded embedding batch's.
The bound is `models.enrichment_cache_limit_bytes` in config, so an
operator can raise it without editing code.
"""


@dataclass(frozen=True, slots=True)
class LoadedVlm:
    """One loaded ``mlx_vlm`` model with everything a caller needs to
    prompt it: the model itself, its processor (tokenizer +
    image processor), the checkpoint's config dict (which
    ``mlx_vlm.prompt_utils.apply_chat_template`` needs), and the local
    directory the weights came from."""

    model: Any
    processor: Any
    config: Any
    model_path: str
    model_id: str

    @property
    def tokenizer(self) -> Any:
        """The processor's text tokenizer (``processor`` itself for a
        processor that is already one)."""
        return getattr(self.processor, "tokenizer", self.processor)


def vlm_model_id(model_repo: str, revision: str | None) -> str:
    """``'<repo>@<revision>'``, with ``'main'`` standing in for an
    unpinned revision — the same spelling
    :func:`imsg.mlx_runtime.format_model_id` produces, so a provider's
    recorded ``model_id`` does not depend on which loader built it."""
    return f"{model_repo}@{revision or 'main'}"


class SharedVlmRuntime:
    """Loads at most one ``mlx_vlm`` model per ``(repo, revision)`` and
    hands the same :class:`LoadedVlm` to every caller that asks for that
    pair.

    ``cache_limit_bytes`` bounds MLX's process-wide buffer cache the
    first time any model loads here (``None`` leaves it alone) — the
    treatment `imsg.embed.mlx_text` and `imsg.retrieval.mlx_reranker`
    already give the query side. The bound is process-wide, so applying
    it once covers every MLX provider in the process, including
    `mlx_whisper` transcription, which has no load hook of its own.
    """

    def __init__(self, *, cache_limit_bytes: int | None = DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES) -> None:
        if cache_limit_bytes is not None and cache_limit_bytes < 0:
            raise ValueError(f"cache_limit_bytes must be >= 0 or None, got {cache_limit_bytes}")
        self._cache_limit_bytes = cache_limit_bytes
        self._loaded: dict[tuple[str, str | None], LoadedVlm] = {}
        self._lock = threading.Lock()

    @property
    def loaded_model_ids(self) -> tuple[str, ...]:
        """Every ``'<repo>@<revision>'`` this runtime currently holds
        weights for, in load order — one entry per distinct pin, however
        many providers asked for it. Tests assert on this to prove the
        sharing is real rather than incidental."""
        return tuple(v.model_id for v in self._loaded.values())

    @property
    def cache_limit_bytes(self) -> int | None:
        return self._cache_limit_bytes

    def is_loaded(self, model_repo: str, revision: str | None) -> bool:
        return (model_repo, revision) in self._loaded

    def acquire(self, model_repo: str, revision: str | None) -> LoadedVlm:
        """The loaded model for this pin, loading it on the first ask.

        A load failure is an environment problem (missing or corrupt
        weights, not enough memory for the model at all), so it raises
        :class:`~imsg.enrich.model_runtime.ModelRuntimeUnavailableError`
        — never a per-task error that would burn a queue entry's retry
        budget.
        """
        key = (model_repo, revision)
        with self._lock:
            existing = self._loaded.get(key)
            if existing is not None:
                return existing
            loaded = self._load(model_repo, revision)
            self._loaded[key] = loaded
            return loaded

    def _load(self, model_repo: str, revision: str | None) -> LoadedVlm:
        model_id = vlm_model_id(model_repo, revision)
        mlx_vlm = import_runtime_module("mlx_vlm", install_hint=MLX_VLM_INSTALL_HINT)
        vlm_utils = import_runtime_module("mlx_vlm.utils", install_hint=MLX_VLM_INSTALL_HINT)
        model_path = resolve_model_snapshot(model_repo, revision)
        try:
            model, processor = mlx_vlm.load(model_path)
            config = vlm_utils.load_config(model_path)
        except Exception as exc:
            raise ModelRuntimeUnavailableError(
                f"could not load the vision-language model {model_id}: {exc}"
            ) from exc
        if self._cache_limit_bytes is not None:
            bound_buffer_cache(self._cache_limit_bytes)
        logger.info(
            "shared_vlm.loaded",
            model_id=model_id,
            cache_limit_bytes=self._cache_limit_bytes,
            resident_pins=len(self._loaded) + 1,
        )
        return LoadedVlm(
            model=model,
            processor=processor,
            config=config,
            model_path=model_path,
            model_id=model_id,
        )


__all__ = [
    "DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES",
    "MLX_VLM_INSTALL_HINT",
    "LoadedVlm",
    "SharedVlmRuntime",
    "vlm_model_id",
]
