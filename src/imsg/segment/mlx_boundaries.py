"""Local-LLM topical boundary detection via MLX — the real
:class:`~imsg.segment.boundaries.BoundaryProvider` (SPEC §4.1/§8 S4, D4:
Qwen3.5-35B-A3B 4-bit, temperature 0, fixed prompt, JSON boundary
indices).

Contract with the caller (``imsg.segment.boundaries``): every failure of
the *model's output* — generation timing out, malformed or out-of-range
JSON, a chat template that rejects the conversation — raises
:class:`~imsg.errors.BoundaryDetectionError`, and a returned list is
always sorted, unique and strictly inside ``0 < index < len(window)``.
The caller applies the one-retry-then-session-fallback policy; nothing
here retries. A missing runtime or unloadable weights is *not*
downgraded to that error: it propagates as
:class:`~imsg.mlx_runtime.MlxRuntimeError` so the run aborts. Mapping it
to ``BoundaryDetectionError`` would have every session in the corpus
degrade to a ``fallback:session`` segment under a run that reports
success — the silent failure ``models: backend=real`` exists to prevent.

Prompting: the owner-authored template (config
``segmentation.boundary_prompt``, read by the CLI and passed in as
bytes/text) is followed by the window rendered as numbered messages —
``[index] sender (timestamp): text`` — and wrapped in the model's chat
template with thinking disabled (Qwen3/Qwen3.5 honour
``enable_thinking=False`` by emitting an empty ``<think>`` block, so the
answer starts immediately). Decoding is greedy (``make_sampler(temp=0)``),
so the same window always yields the same boundaries. The response is
expected as ``{"boundaries": [ints]}``; a bare JSON list, code fences and
any stray ``<think>…</think>`` block are tolerated.

Timeouts: MLX generation cannot be interrupted mid-kernel, so the
deadline is enforced between streamed tokens — generation is abandoned
(and :class:`BoundaryDetectionError` raised) at the first token that
arrives past ``timeout_seconds``.

Two loaders, one computation (D10.3 defect 1). The pinned boundary model
*is* the pinned captioning model (``imsg.constants`` sets
``CAPTION_MODEL_REPO = BOUNDARY_MODEL_REPO``), and loading it twice —
once through ``mlx_lm``, once through ``mlx_vlm`` — cost 18 GiB of
duplicate weights on the production host. Given a
:class:`~imsg.shared_vlm_runtime.SharedVlmRuntime` this provider runs
text-only generation through the already-loaded vision-language model
instead; given none it keeps the ``mlx_lm`` path, which is 0.83 GiB
lighter for a process that only detects boundaries. The two paths are
the same computation on the same weights, checked rather than assumed
(2026-09-17, development host, the pinned 4-bit 35B): the rendered chat
prompt is byte-identical, and the last-position logit row for a fixed
probe is bit-identical across all 248,320 float32 values. See
``imsg.shared_vlm_runtime`` for the evidence in full.
"""

from __future__ import annotations

import importlib
import json
import re
import time
from collections.abc import Sequence
from typing import Any

from imsg.enrich.model_runtime import import_runtime_module
from imsg.errors import BoundaryDetectionError, SegmentationError
from imsg.mlx_runtime import (
    DEFAULT_CACHE_LIMIT_BYTES,
    bound_buffer_cache,
    format_model_id,
    import_mlx_lm,
    load_model_and_tokenizer,
)
from imsg.segment.models import MessageForSegmentation
from imsg.shared_vlm_runtime import MLX_VLM_INSTALL_HINT, LoadedVlm, SharedVlmRuntime

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_BOUNDARIES_KEY = "boundaries"

_monotonic = time.monotonic  # module-level so tests can substitute a fake clock


def render_window_messages(window: Sequence[MessageForSegmentation]) -> str:
    """One line per message: ``[index] sender (timestamp): text``. Indices
    are 0-based positions in ``window`` — the coordinate system the
    returned boundaries use. Whitespace runs (including newlines) are
    collapsed so each message stays on one line; empty bodies are
    labelled from the message's flags."""
    lines: list[str] = []
    for index, message in enumerate(window):
        text = " ".join((message.text or "").split())
        if not text:
            if message.is_unsent:
                text = "[unsent message]"
            elif message.has_attachments:
                text = "[attachment]"
            else:
                text = "[empty message]"
        elif message.has_attachments:
            text = f"{text} [attachment]"
        stamp = message.sent_at.isoformat(timespec="minutes")
        lines.append(f"[{index}] {message.sender_short_name} ({stamp}): {text}")
    return "\n".join(lines)


def render_boundary_prompt(template: str, window: Sequence[MessageForSegmentation]) -> str:
    """The full user-turn prompt: the template, a blank line, a header
    naming the index range, then the numbered messages."""
    header = f"Messages ({len(window)} total, indexed 0 to {len(window) - 1}):"
    return f"{template.rstrip()}\n\n{header}\n{render_window_messages(window)}\n"


def _extract_json_text(text: str) -> str:
    stripped = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in stripped:  # an unmatched closing tag: keep what follows it
        stripped = stripped.rsplit("</think>", 1)[1]
    fence = _CODE_FENCE_RE.search(stripped)
    if fence is not None:
        stripped = fence.group(1)
    return stripped.strip()


def _first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for position, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        return value
    raise BoundaryDetectionError(
        f"boundary model response contains no parseable JSON object or list "
        f"({len(text)} chars after stripping)"
    )


def parse_boundary_response(text: str) -> list[int]:
    """Extract the raw boundary indices from a model response: strips
    ``<think>`` blocks and code fences, accepts ``{"boundaries": [...]}``
    or a bare list, rejects anything that is not a list of integers.
    Range/order are validated separately by :func:`validate_boundaries`."""
    body = _extract_json_text(text)
    if not body:
        raise BoundaryDetectionError("boundary model returned an empty response")
    try:
        payload: Any = json.loads(body)
    except json.JSONDecodeError:
        payload = _first_json_value(body)
    if isinstance(payload, dict):
        if _BOUNDARIES_KEY not in payload:
            raise BoundaryDetectionError(
                f"boundary model JSON object has no {_BOUNDARIES_KEY!r} key"
            )
        items = payload[_BOUNDARIES_KEY]
    else:
        items = payload
    if not isinstance(items, list):
        raise BoundaryDetectionError(
            f"boundary model returned {type(items).__name__}, expected a list of integers"
        )
    indices: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise BoundaryDetectionError(
                f"boundary model returned a non-integer index ({type(item).__name__})"
            )
        indices.append(item)
    return indices


def validate_boundaries(indices: Sequence[int], window_size: int) -> list[int]:
    """Enforce the ``BoundaryProvider`` contract: every index strictly
    inside ``(0, window_size)``; the result sorted and de-duplicated."""
    for index in indices:
        if not 0 < index < window_size:
            raise BoundaryDetectionError(
                f"boundary index {index} is outside the valid range 1..{window_size - 1} "
                f"for a window of {window_size} messages"
            )
    return sorted(set(indices))


def _coerce_template(prompt_template: str | bytes) -> str:
    if isinstance(prompt_template, bytes):
        try:
            text = prompt_template.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SegmentationError("boundary prompt template is not valid UTF-8") from exc
    else:
        text = prompt_template
    if not text.strip():
        raise SegmentationError("boundary prompt template is empty")
    return text


class MlxBoundaryProvider:
    """A local instruction-tuned LLM (Qwen3.5-35B-A3B 4-bit per D4) through
    ``mlx_lm``, prompted with the owner-authored boundary template.

    Constructor arguments are plain values; the CLI reads
    ``segmentation.boundary_model`` and the prompt file and passes them
    here. Weights load lazily on the first :meth:`detect_boundaries` or
    explicitly via :meth:`load`; either way a load failure is the
    underlying ``MlxRuntimeError`` (see the module docstring on why it
    is never downgraded to ``BoundaryDetectionError``).

    ``shared_runtime`` selects the loader. Given a
    :class:`~imsg.shared_vlm_runtime.SharedVlmRuntime` — the same
    instance the captioner was handed — this provider generates through
    the already-loaded vision-language model and no second copy of the
    weights is created. Given ``None`` it loads its own through
    ``mlx_lm``, bounding MLX's buffer cache at ``cache_limit_bytes``
    the way every other MLX provider in this codebase does (D10.2:
    unbounded, the pool reached 37.25 GiB in one enrichment process).
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        prompt_template: str | bytes,
        *,
        max_tokens: int = 512,
        timeout_seconds: float = 120.0,
        shared_runtime: SharedVlmRuntime | None = None,
        cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT_BYTES,
    ) -> None:
        if not model_repo:
            raise ValueError("model_repo must be a non-empty repo id or local path")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        if cache_limit_bytes is not None and cache_limit_bytes < 0:
            raise ValueError(f"cache_limit_bytes must be >= 0 or None, got {cache_limit_bytes}")
        self.model_id = format_model_id(model_repo, revision)
        self.prompt_template = _coerce_template(prompt_template)
        self._model_repo = model_repo
        self._revision = revision
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._shared_runtime = shared_runtime
        self._cache_limit_bytes = cache_limit_bytes
        self._model: Any = None
        self._tokenizer: Any = None
        self._vlm: LoadedVlm | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._vlm is not None

    @property
    def shares_vlm_weights(self) -> bool:
        """Whether this provider generates through a shared
        vision-language model rather than loading its own copy. Exposed so
        a caller (and the tests) can tell which of the two paths in the
        module docstring a given instance took, without reaching into
        private state."""
        return self._shared_runtime is not None

    def load(self) -> None:
        """Load weights + tokenizer now (idempotent)."""
        if self.is_loaded:
            return
        if self._shared_runtime is not None:
            self._vlm = self._shared_runtime.acquire(self._model_repo, self._revision)
            return
        model, tokenizer = load_model_and_tokenizer(self._model_repo, self._revision)
        if self._cache_limit_bytes is not None:
            bound_buffer_cache(self._cache_limit_bytes)
        self._model = model
        self._tokenizer = tokenizer

    def unload(self) -> None:
        """Drop this provider's hold on the weights (idempotent); the next
        :meth:`detect_boundaries` loads them again. Dropping references is
        all it does — the caller returns the freed memory to the system
        (`imsg.retrieval.idle_unload.release_freed_memory`). A provider
        on a shared runtime drops only its own reference: the runtime
        keeps the model for the captioner that shares it."""
        self._model = None
        self._tokenizer = None
        self._vlm = None

    def detect_boundaries(self, window: Sequence[MessageForSegmentation]) -> list[int]:
        if len(window) < 2:
            return []  # no index can satisfy 0 < i < len(window); nothing to ask the model
        # Deliberately not wrapped: a missing runtime / unloadable weights
        # is an environment failure, not a property of this window, and
        # must abort the run rather than trigger the caller's fallback
        # (module docstring).
        self.load()
        prompt = self._chat_prompt(render_boundary_prompt(self.prompt_template, window))
        response = self._generate(prompt)
        indices = parse_boundary_response(response)
        return validate_boundaries(indices, len(window))

    def _chat_prompt(self, user_text: str) -> str:
        """Wrap the rendered prompt in the tokenizer's chat template with
        thinking disabled; a tokenizer without a chat template (a base
        model) gets the raw text."""
        if self._vlm is not None:
            return self._vlm_chat_prompt(self._vlm, user_text)
        tokenizer = self._tokenizer
        if not getattr(tokenizer, "chat_template", None):
            return user_text
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except Exception as exc:
            raise BoundaryDetectionError(
                f"boundary model {self.model_id}: chat template failed: {type(exc).__name__}: {exc}"
            ) from exc
        return str(rendered)

    def _vlm_chat_prompt(self, vlm: LoadedVlm, user_text: str) -> str:
        """The same wrapping through ``mlx_vlm``'s own chat-template
        helper, with no image placeholders (``num_images=0``): this is a
        text-only turn through a vision-language model. Checked to render
        byte-identically to the ``mlx_lm`` branch above for the pinned
        model (module docstring)."""
        prompt_utils = import_runtime_module(
            "mlx_vlm.prompt_utils", install_hint=MLX_VLM_INSTALL_HINT
        )
        try:
            rendered = prompt_utils.apply_chat_template(
                vlm.processor, vlm.config, user_text, num_images=0, enable_thinking=False
            )
        except Exception as exc:
            raise BoundaryDetectionError(
                f"boundary model {self.model_id}: chat template failed: {type(exc).__name__}: {exc}"
            ) from exc
        return str(rendered)

    def _generate(self, prompt: str) -> str:
        """Greedy generation with the between-token deadline described in
        the module docstring. Every runtime failure becomes
        :class:`BoundaryDetectionError`."""
        if self._vlm is not None:
            return self._generate_through_vlm(self._vlm, prompt)
        mlx_lm = import_mlx_lm()
        try:
            sample_utils = importlib.import_module("mlx_lm.sample_utils")
            sampler = sample_utils.make_sampler(temp=0.0)
        except Exception as exc:
            raise BoundaryDetectionError(
                f"boundary model {self.model_id}: could not build a greedy sampler: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        deadline = _monotonic() + self._timeout_seconds
        pieces: list[str] = []
        try:
            for response in mlx_lm.stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=self._max_tokens,
                sampler=sampler,
            ):
                pieces.append(str(response.text))
                if _monotonic() > deadline:
                    raise BoundaryDetectionError(
                        f"boundary model {self.model_id} exceeded {self._timeout_seconds:g}s "
                        f"after {len(pieces)} generated tokens"
                    )
        except BoundaryDetectionError:
            raise
        except Exception as exc:
            raise BoundaryDetectionError(
                f"boundary model {self.model_id} generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        return "".join(pieces)

    def _generate_through_vlm(self, vlm: LoadedVlm, prompt: str) -> str:
        """Text-only greedy generation through the shared
        vision-language model (``image=None``), under the same
        between-token deadline and the same error contract as the
        ``mlx_lm`` branch. ``temperature=0.0`` is the greedy sampler
        ``mlx_vlm`` builds for itself — the counterpart of
        ``make_sampler(temp=0)`` above, and the same value the captioner
        passes."""
        mlx_vlm = import_runtime_module("mlx_vlm", install_hint=MLX_VLM_INSTALL_HINT)
        deadline = _monotonic() + self._timeout_seconds
        pieces: list[str] = []
        try:
            for response in mlx_vlm.stream_generate(
                vlm.model,
                vlm.processor,
                prompt,
                image=None,
                max_tokens=self._max_tokens,
                temperature=0.0,
            ):
                pieces.append(str(response.text))
                if _monotonic() > deadline:
                    raise BoundaryDetectionError(
                        f"boundary model {self.model_id} exceeded {self._timeout_seconds:g}s "
                        f"after {len(pieces)} generated tokens"
                    )
        except BoundaryDetectionError:
            raise
        except Exception as exc:
            raise BoundaryDetectionError(
                f"boundary model {self.model_id} generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        return "".join(pieces)


__all__ = [
    "MlxBoundaryProvider",
    "parse_boundary_response",
    "render_boundary_prompt",
    "render_window_messages",
    "validate_boundaries",
]
