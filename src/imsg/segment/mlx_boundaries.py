"""Local-LLM topical boundary detection via MLX — the real
:class:`~imsg.segment.boundaries.BoundaryProvider` (SPEC §4.1/§8 S4, D4:
Qwen3.5-35B-A3B 4-bit, temperature 0, fixed prompt, JSON boundary
indices).

Contract with the caller (``imsg.segment.boundaries``): every failure —
runtime missing, weights unloadable, generation timing out, malformed
or out-of-range output — raises :class:`~imsg.errors.
BoundaryDetectionError`, and a returned list is always sorted, unique
and strictly inside ``0 < index < len(window)``. The caller applies the
one-retry-then-session-fallback policy; nothing here retries.

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
"""

from __future__ import annotations

import importlib
import json
import re
import time
from collections.abc import Sequence
from typing import Any

from imsg.errors import BoundaryDetectionError, SegmentationError
from imsg.mlx_runtime import (
    MlxRuntimeError,
    format_model_id,
    import_mlx_lm,
    load_model_and_tokenizer,
)
from imsg.segment.models import MessageForSegmentation

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
    here. Weights load lazily on the first :meth:`detect_boundaries` (or
    explicitly via :meth:`load`, which raises the underlying
    ``MlxRuntimeError`` rather than ``BoundaryDetectionError`` so a
    startup check fails loudly).
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        prompt_template: str | bytes,
        *,
        max_tokens: int = 512,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not model_repo:
            raise ValueError("model_repo must be a non-empty repo id or local path")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        self.model_id = format_model_id(model_repo, revision)
        self.prompt_template = _coerce_template(prompt_template)
        self._model_repo = model_repo
        self._revision = revision
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load weights + tokenizer now (idempotent)."""
        if self._model is not None:
            return
        model, tokenizer = load_model_and_tokenizer(self._model_repo, self._revision)
        self._model = model
        self._tokenizer = tokenizer

    def detect_boundaries(self, window: Sequence[MessageForSegmentation]) -> list[int]:
        if len(window) < 2:
            return []  # no index can satisfy 0 < i < len(window); nothing to ask the model
        try:
            self.load()
        except MlxRuntimeError as exc:
            raise BoundaryDetectionError(
                f"boundary model {self.model_id} unavailable: {exc}"
            ) from exc
        prompt = self._chat_prompt(render_boundary_prompt(self.prompt_template, window))
        response = self._generate(prompt)
        indices = parse_boundary_response(response)
        return validate_boundaries(indices, len(window))

    def _chat_prompt(self, user_text: str) -> str:
        """Wrap the rendered prompt in the tokenizer's chat template with
        thinking disabled; a tokenizer without a chat template (a base
        model) gets the raw text."""
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

    def _generate(self, prompt: str) -> str:
        """Greedy generation with the between-token deadline described in
        the module docstring. Every runtime failure becomes
        :class:`BoundaryDetectionError`."""
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


__all__ = [
    "MlxBoundaryProvider",
    "parse_boundary_response",
    "render_boundary_prompt",
    "render_window_messages",
    "validate_boundaries",
]
