"""Shared access to the optional MLX runtime (``mlx`` + ``mlx_lm``) for the
real, Apple-silicon model providers — ``imsg.embed.mlx_text``,
``imsg.retrieval.mlx_reranker`` and ``imsg.segment.mlx_boundaries``
(SPEC §4.1: local-only inference, D3/D4 ratified).

Design constraints shared by all three providers:

- ``mlx`` / ``mlx_lm`` are **not** hard dependencies of ``imsg``. They
  are imported lazily, by name, inside functions, so importing a
  provider module (and therefore ``imsg.cli``) never requires them, and
  every test in this repository runs without them by stubbing
  ``sys.modules``. A missing runtime surfaces as
  :class:`MlxRuntimeUnavailableError` (an :class:`~imsg.errors.ImsgError`)
  with an actionable message, never as a raw ``ImportError`` traceback.
- ``mlx_lm.load()`` is the single weight-loading entry point. The
  config's pinned ``revision`` is honoured on every ``mlx_lm`` version:
  passed straight through when ``load()`` accepts ``revision=`` (recent
  releases), otherwise the pinned snapshot is materialised first —
  ``mlx_lm.utils.get_model_path(..., revision=)`` where available,
  falling back to ``huggingface_hub.snapshot_download(..., revision=)``
  — and ``load()`` is handed that local directory.
- Batched inference uses **right padding under the model's own causal
  mask** plus a per-sequence "last real token" gather
  (:func:`gather_last_token_states`). Under a causal mask a real token
  can never attend to a later (padding) position, so every sequence's
  hidden states are exactly the unpadded computation without any
  padding mask: no batch x len x len mask allocation, and no fully-masked
  query rows to turn into NaNs. The Qwen3 reference code left-pads and
  reads position ``-1``; both readings select the same token.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from imsg.errors import ImsgError

_HF_SNAPSHOT_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)
"""The file patterns ``mlx_lm`` itself downloads for a repo — used only on
the ``huggingface_hub`` fallback path so a pinned-revision download does
not pull duplicate weight formats (``*.bin`` next to ``*.safetensors``)."""


DEFAULT_CACHE_LIMIT_BYTES = 8 * 2**30
"""The bound :func:`bound_buffer_cache` applies when a provider's weights
load (D10.2: bound MLX's buffer cache explicitly at process start) —
the text embedder and the reranker both apply it. MLX's default cache
limit equals its memory limit — 121.6 GiB on a 128 GB M2 Ultra
(``mx.device_info`` / ``set_cache_limit``, 2026-09-15) — so every
freed activation buffer is retained and, because consecutive batches
have different shapes, rarely reused: measured with
`scripts/bench_text_embedding.py`, the cache grew by ~1.3 GiB per
batch without bound (44-54 GiB after 8-64 batches; peak *active*
memory never above 28 GiB), and the first production run reached a
102 GB GPU footprint, filled the 30 GB swap and stalled the GPU on
paging (`footprint`/`vm.swapusage`, 2026-09-15). 8 GiB is comfortably
above the active working set of the batches the pipeline plans (peak
9-11 GiB *including* the 8.4 GB of weights for 4k-16k padded
tokens), so bounding it costs nothing and keeps the process resident."""


class MlxRuntimeError(ImsgError):
    """An MLX-backed provider could not load or run its model (weights
    missing/unloadable, an unexpected model class, a tokenizer that does
    not expose what the provider needs)."""


class MlxRuntimeUnavailableError(MlxRuntimeError):
    """``mlx`` / ``mlx_lm`` are not importable in this environment. The
    deterministic ``Fake*`` providers are the dependency-free
    alternative; the real providers need the MLX runtime installed on an
    Apple-silicon host."""


def _import(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MlxRuntimeUnavailableError(
            f"the MLX runtime module '{name}' is not importable ({exc}) — the real "
            f"model providers need the `mlx` and `mlx-lm` packages installed on an "
            f"Apple-silicon host; the deterministic Fake* providers run without them"
        ) from exc


def import_mlx_core() -> ModuleType:
    """Lazily import ``mlx.core`` (array ops)."""
    return _import("mlx.core")


def import_mlx_lm() -> ModuleType:
    """Lazily import ``mlx_lm`` (``load`` / ``stream_generate``)."""
    return _import("mlx_lm")


def import_mlx_lm_cache() -> ModuleType:
    """Lazily import ``mlx_lm.models.cache`` (``make_prompt_cache`` and the
    per-layer key/value caches a model reads earlier tokens from)."""
    return _import("mlx_lm.models.cache")


def bound_buffer_cache(limit_bytes: int) -> None:
    """Bound MLX's buffer cache to at most ``limit_bytes``, keeping any
    tighter bound already in place — and apply this process's MLX memory
    limit, if its command set one (:func:`set_process_memory_limit`).
    Every MLX provider calls this when its weights load, which makes it
    the one point where the runtime is certainly imported.

    The cache limit is process-wide, and several providers share one
    process (the retrieval service loads the text embedder and the
    reranker), so each applies its bound at load without loosening one
    another's. ``mx.set_cache_limit`` returns the previous limit — MLX
    has no getter — so the only way to read the current bound is to set
    one: a previous limit below ``limit_bytes`` was an explicit, tighter
    choice and is put back. The runtime default (the memory limit) is
    never below a sane bound, so a fresh process always ends at
    ``limit_bytes``. A runtime without ``set_cache_limit`` is left alone.
    """
    if limit_bytes < 0:
        raise ValueError(f"limit_bytes must be >= 0, got {limit_bytes}")
    set_cache_limit = getattr(import_mlx_core(), "set_cache_limit", None)
    if set_cache_limit is not None:
        previous = set_cache_limit(limit_bytes)
        if (
            isinstance(previous, int)
            and not isinstance(previous, bool)
            and 0 <= previous < limit_bytes
        ):
            set_cache_limit(previous)
    apply_process_memory_limit()


_process_memory_limit_bytes: int | None = None
"""The MLX memory limit this process's command asked for, or `None` for
MLX's own default. See :func:`set_process_memory_limit`."""


def set_process_memory_limit(limit_bytes: int | None) -> None:
    """Record the MLX memory limit this process runs under — the role's
    `memory.mlx_memory_limits` entry, chosen by the command
    (`imsg.memory_admission.configure_mlx_memory_limit`) — and apply it
    now if MLX is already imported. Otherwise the first provider to load
    applies it (:func:`bound_buffer_cache`), so a command that never
    loads a model never imports MLX. `None` records nothing to apply
    (MLX's default stays).

    What the limit does is in `imsg.config.schema.MlxMemoryLimits`: it
    moves the point at which MLX releases cached buffers and the point
    at which evaluation waits for work in flight; it never refuses an
    allocation and never changes a result."""
    if limit_bytes is not None and limit_bytes <= 0:
        raise ValueError(f"limit_bytes must be > 0 or None, got {limit_bytes}")
    global _process_memory_limit_bytes
    _process_memory_limit_bytes = limit_bytes
    if limit_bytes is not None and "mlx.core" in sys.modules:
        apply_process_memory_limit()


def process_memory_limit() -> int | None:
    return _process_memory_limit_bytes


def apply_process_memory_limit() -> int | None:
    """Set MLX's memory limit to the recorded one; returns what was set,
    or `None` when nothing was recorded or the runtime has no
    ``set_memory_limit``."""
    limit = _process_memory_limit_bytes
    if limit is None:
        return None
    set_memory_limit = getattr(import_mlx_core(), "set_memory_limit", None)
    if set_memory_limit is None:
        return None
    set_memory_limit(limit)
    return limit


def format_model_id(model_repo: str, revision: str | None) -> str:
    """The ``model_id`` every provider records: ``'<repo>@<revision>'``,
    with ``'main'`` standing in for an unpinned revision."""
    return f"{model_repo}@{revision or 'main'}"


def _accepts_parameter(func: Any, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins / C extensions without a signature
        return False


def _resolve_pinned_snapshot(model_repo: str, revision: str) -> Path:
    """Materialise ``model_repo`` at ``revision`` locally for an ``mlx_lm``
    whose ``load()`` cannot pin a revision itself."""
    get_model_path: Any = None
    try:
        get_model_path = importlib.import_module("mlx_lm.utils").get_model_path
    except (ImportError, AttributeError):
        get_model_path = None
    if get_model_path is not None and _accepts_parameter(get_model_path, "revision"):
        resolved = get_model_path(model_repo, revision=revision)
        # mlx_lm 0.26.x returns ``(Path, hf_repo)``; earlier releases a bare ``Path``.
        if isinstance(resolved, tuple):
            resolved = resolved[0]
        return Path(str(resolved))
    hub = _import("huggingface_hub")
    return Path(
        str(
            hub.snapshot_download(
                model_repo, revision=revision, allow_patterns=list(_HF_SNAPSHOT_PATTERNS)
            )
        )
    )


def load_model_and_tokenizer(
    model_repo: str,
    revision: str | None,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """``mlx_lm.load`` with revision pinning on every ``mlx_lm`` version
    (see the module docstring). Returns ``(model, tokenizer)``; the
    tokenizer is ``mlx_lm``'s ``TokenizerWrapper``, which forwards
    attribute access to the underlying Hugging Face tokenizer.
    ``model_config`` entries override the checkpoint's ``config.json``
    before the model class is instantiated (``mlx_lm.load``'s own
    ``model_config``); see ``imsg.embed.mlx_text`` for the one case that
    needs it.

    Raises :class:`MlxRuntimeUnavailableError` when the runtime is
    missing and :class:`MlxRuntimeError` for any load failure.
    """
    mlx_lm = import_mlx_lm()
    kwargs: dict[str, Any] = {}
    if tokenizer_config:
        kwargs["tokenizer_config"] = dict(tokenizer_config)
    if model_config:
        kwargs["model_config"] = dict(model_config)
    try:
        if revision is None:
            loaded = mlx_lm.load(model_repo, **kwargs)
        elif _accepts_parameter(mlx_lm.load, "revision"):
            loaded = mlx_lm.load(model_repo, revision=revision, **kwargs)
        else:
            snapshot = _resolve_pinned_snapshot(model_repo, revision)
            loaded = mlx_lm.load(str(snapshot), **kwargs)
    except MlxRuntimeError:
        raise
    except Exception as exc:  # mlx_lm raises FileNotFoundError/ValueError/hub errors
        raise MlxRuntimeError(
            f"could not load {format_model_id(model_repo, revision)} via mlx_lm: "
            f"{type(exc).__name__}: {exc}. mlx_lm needs an MLX-layout checkpoint "
            f"(e.g. an `mlx-community/*` conversion or the output of `mlx_lm.convert`); "
            f"a repo published only in the raw transformers layout will not load"
        ) from exc
    try:
        model, tokenizer = loaded[0], loaded[1]
    except (TypeError, IndexError) as exc:
        raise MlxRuntimeError(
            f"mlx_lm.load returned an unexpected value for "
            f"{format_model_id(model_repo, revision)}: {type(loaded).__name__}"
        ) from exc
    return model, tokenizer


def hidden_size_of(model: Any) -> int:
    """The model's hidden width (``model.args.hidden_size`` on every
    ``mlx_lm`` decoder-only model class)."""
    try:
        return int(model.args.hidden_size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MlxRuntimeError(
            "mlx_lm model exposes no `args.hidden_size`; this provider only supports "
            "mlx_lm decoder-only model classes"
        ) from exc


def right_pad(sequences: Sequence[Sequence[int]], pad_id: int) -> tuple[list[list[int]], list[int]]:
    """Right-pad token rows to a rectangle. Returns ``(padded, lengths)``;
    ``lengths[i]`` is the number of real tokens in row ``i``, so its last
    real token sits at index ``lengths[i] - 1``."""
    if not sequences:
        raise MlxRuntimeError("right_pad: empty batch")
    lengths = [len(s) for s in sequences]
    if any(n == 0 for n in lengths):
        raise MlxRuntimeError("right_pad: a sequence with no tokens has no last token to read")
    width = max(lengths)
    padded = [[*s, *([pad_id] * (width - len(s)))] for s in sequences]
    return padded, lengths


def base_transformer_hidden_states(model: Any, token_ids: Any, *, cache: Any = None) -> Any:
    """Run the base transformer only — ``model.model`` on every ``mlx_lm``
    decoder-only class (``Qwen3Model`` for ``qwen3``): token embeddings,
    the causal-masked layer stack, and the final norm — returning
    last-layer hidden states of shape ``(batch, seq, hidden)`` without
    ever touching the LM head.

    ``cache`` is one ``mlx_lm`` key/value cache per layer
    (``mlx_lm.models.cache.make_prompt_cache``). The tokens it already
    holds come before ``token_ids``: the new tokens are positioned after
    them and attend to them, and the cache gains the new tokens' keys and
    values."""
    inner = getattr(model, "model", None)
    if inner is None or not callable(inner):
        raise MlxRuntimeError(
            "mlx_lm model exposes no callable `.model` base transformer; this provider "
            "only supports mlx_lm decoder-only model classes"
        )
    if cache is None:
        return inner(token_ids)
    return inner(token_ids, cache=cache)


def gather_last_token_states(mx: ModuleType, hidden: Any, lengths: Sequence[int]) -> Any:
    """Select, per sequence, the hidden state of its last *real* token
    (index ``lengths[i] - 1`` under right padding). ``hidden`` is
    ``(batch, seq, hidden)``; the result is ``(batch, 1, hidden)``."""
    shape = tuple(int(d) for d in hidden.shape)
    if len(shape) != 3:
        raise MlxRuntimeError(f"expected (batch, seq, hidden) hidden states, got shape {shape}")
    if shape[0] != len(lengths):
        raise MlxRuntimeError(
            f"hidden states carry {shape[0]} rows but {len(lengths)} sequence lengths were given"
        )
    for n in lengths:
        if not 0 < n <= shape[1]:
            raise MlxRuntimeError(f"sequence length {n} is outside the padded width {shape[1]}")
    index = mx.array([[[n - 1]] for n in lengths])
    return mx.take_along_axis(hidden, index, axis=1)


def lm_head_logits(model: Any, states: Any) -> Any:
    """Project hidden states to vocabulary logits the way the model's own
    forward pass does: ``lm_head`` when present, else the tied input
    embedding (``model.model.embed_tokens.as_linear``)."""
    tied = bool(getattr(model.args, "tie_word_embeddings", False))
    if tied:
        return model.model.embed_tokens.as_linear(states)
    head = getattr(model, "lm_head", None)
    if head is None or not callable(head):
        raise MlxRuntimeError(
            "mlx_lm model has no `lm_head` and is not weight-tied; cannot read token logits"
        )
    return head(states)


def float_rows(mx: ModuleType, array: Any) -> list[list[float]]:
    """Materialise a ``(batch, 1, features)`` array (the shape
    :func:`gather_last_token_states` and anything derived from it
    produces) as one plain-Python float list per batch row."""
    rows = array.astype(mx.float32).tolist()
    out: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 1 or not isinstance(row[0], list):
            raise MlxRuntimeError("expected a (batch, 1, features) array to read rows from")
        out.append([float(x) for x in row[0]])
    return out


def batched[T](items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """Yield ``items`` in consecutive chunks of at most ``batch_size``."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


__all__ = [
    "DEFAULT_CACHE_LIMIT_BYTES",
    "MlxRuntimeError",
    "MlxRuntimeUnavailableError",
    "apply_process_memory_limit",
    "base_transformer_hidden_states",
    "batched",
    "bound_buffer_cache",
    "float_rows",
    "format_model_id",
    "gather_last_token_states",
    "hidden_size_of",
    "import_mlx_core",
    "import_mlx_lm",
    "import_mlx_lm_cache",
    "lm_head_logits",
    "load_model_and_tokenizer",
    "process_memory_limit",
    "right_pad",
    "set_process_memory_limit",
]
