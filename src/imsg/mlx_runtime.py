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
) -> tuple[Any, Any]:
    """``mlx_lm.load`` with revision pinning on every ``mlx_lm`` version
    (see the module docstring). Returns ``(model, tokenizer)``; the
    tokenizer is ``mlx_lm``'s ``TokenizerWrapper``, which forwards
    attribute access to the underlying Hugging Face tokenizer.

    Raises :class:`MlxRuntimeUnavailableError` when the runtime is
    missing and :class:`MlxRuntimeError` for any load failure.
    """
    mlx_lm = import_mlx_lm()
    kwargs: dict[str, Any] = {}
    if tokenizer_config:
        kwargs["tokenizer_config"] = dict(tokenizer_config)
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


def base_transformer_hidden_states(model: Any, token_ids: Any) -> Any:
    """Run the base transformer only — ``model.model`` on every ``mlx_lm``
    decoder-only class (``Qwen3Model`` for ``qwen3``): token embeddings,
    the causal-masked layer stack, and the final norm — returning
    last-layer hidden states of shape ``(batch, seq, hidden)`` without
    ever touching the LM head."""
    inner = getattr(model, "model", None)
    if inner is None or not callable(inner):
        raise MlxRuntimeError(
            "mlx_lm model exposes no callable `.model` base transformer; this provider "
            "only supports mlx_lm decoder-only model classes"
        )
    return inner(token_ids)


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
    "MlxRuntimeError",
    "MlxRuntimeUnavailableError",
    "base_transformer_hidden_states",
    "batched",
    "float_rows",
    "format_model_id",
    "gather_last_token_states",
    "hidden_size_of",
    "import_mlx_core",
    "import_mlx_lm",
    "lm_head_logits",
    "load_model_and_tokenizer",
    "right_pad",
]
