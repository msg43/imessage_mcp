"""Pass 2 (SPEC §8 S4, D4): LLM topical-boundary detection within a
session, windowed at a fixed 4k-token budget with 10-message overlap,
then hard-capped at `max_messages`/`max_tokens` and floored at a
minimum 2 messages per segment ("Enforce min 2 / max 50 messages / max
2000 tokens (hard split at cap)").

The boundary window size (4,000 tokens) and overlap (10 messages) are
fixed pipeline constants, not `segmentation.*` config fields — SPEC §6
does not expose them, and D4 freezes them alongside the numeric
thresholds that are configurable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from imsg.errors import BoundaryDetectionError
from imsg.segment.models import MessageForSegmentation, SegmentDraft, Session
from imsg.tokens import estimate_tokens

BOUNDARY_WINDOW_TOKENS = 4000
BOUNDARY_WINDOW_OVERLAP_MESSAGES = 10
MIN_MESSAGES_PER_SEGMENT = 2


@runtime_checkable
class BoundaryProvider(Protocol):
    """SPEC §4.1/§8 S4: local boundary-detection LLM, temperature 0,
    fixed prompt, JSON boundary indices. The real MLX-backed
    implementation (Qwen3.5-35B-A3B, D4/§4.1) drops in behind this
    Protocol at Phase 3/5; `FakeBoundaryProvider` below is the
    deterministic stand-in every test in this build uses.
    """

    model_id: str
    """Identifies the model (+ revision, where applicable) actually
    used — folded into `seg_config_hash` by the caller, not by the
    provider itself."""

    def detect_boundaries(self, window: Sequence[MessageForSegmentation]) -> list[int]:
        """Return sorted, unique boundary indices into `window`
        (`0 < index < len(window)`; a new segment starts *at* each
        returned index). Must raise `BoundaryDetectionError` — never
        return a malformed/out-of-range index — on any failure
        (timeout, malformed JSON, model unavailable); the caller
        applies the one-retry-then-fallback policy (SPEC §8 S4)."""
        ...


class FakeBoundaryProvider:
    """Deterministic stand-in for tests (no model weights in this
    environment — see the build task's provider-interface requirement).

    Splits a window every `messages_per_segment` messages — a trivial,
    fully deterministic rule good enough to exercise windowing/merge/
    hard-split logic without pretending to understand conversation
    topics. `always_fail=True` exercises the fallback-to-session path.
    """

    model_id = "fake/boundary-detector@test"

    def __init__(
        self,
        *,
        messages_per_segment: int = 5,
        always_fail: bool = False,
        fail_first_n_calls: int = 0,
    ) -> None:
        self._messages_per_segment = messages_per_segment
        self._always_fail = always_fail
        self._fail_first_n_calls = fail_first_n_calls
        self._calls = 0

    def detect_boundaries(self, window: Sequence[MessageForSegmentation]) -> list[int]:
        self._calls += 1
        if self._always_fail or self._calls <= self._fail_first_n_calls:
            raise BoundaryDetectionError("FakeBoundaryProvider configured to fail")
        return list(range(self._messages_per_segment, len(window), self._messages_per_segment))


def _extend_window(
    messages: Sequence[MessageForSegmentation], start: int, max_tokens: int
) -> int:
    """Return the exclusive end index of a window starting at `start`
    whose estimated token cost stays under `max_tokens` — always
    including at least one message, even if that message alone exceeds
    the budget."""
    total = 0
    i = start
    n = len(messages)
    while i < n:
        cost = estimate_tokens(messages[i].text or "")
        if i > start and total + cost > max_tokens:
            break
        total += cost
        i += 1
    return max(i, start + 1)


def _detect_boundaries_for_window(
    provider: BoundaryProvider, window: Sequence[MessageForSegmentation]
) -> list[int]:
    """One retry on failure (SPEC §8 S4: "malformed LLM JSON -> one
    retry, then fallback") — a second failure propagates to the caller,
    which triggers the session-as-segment fallback."""
    try:
        return provider.detect_boundaries(window)
    except BoundaryDetectionError:
        return provider.detect_boundaries(window)


def _detect_boundaries_windowed(
    messages: Sequence[MessageForSegmentation], provider: BoundaryProvider
) -> list[int]:
    n = len(messages)
    found: set[int] = set()
    window_start = 0
    while window_start < n:
        window_end = _extend_window(messages, window_start, BOUNDARY_WINDOW_TOKENS)
        window = messages[window_start:window_end]
        local_boundaries = _detect_boundaries_for_window(provider, window)
        for b in local_boundaries:
            if not (0 < b < len(window)):
                raise BoundaryDetectionError(
                    f"boundary provider {provider.model_id!r} returned out-of-range "
                    f"index {b} for a window of {len(window)} messages"
                )
            found.add(window_start + b)
        if window_end >= n:
            break
        next_start = window_end - BOUNDARY_WINDOW_OVERLAP_MESSAGES
        if next_start <= window_start:
            next_start = window_start + 1  # guard against non-progress
        window_start = next_start
    return sorted(found)


def _boundaries_to_groups(n: int, boundaries: Sequence[int]) -> list[tuple[int, int]]:
    points = [0, *sorted(set(boundaries)), n]
    return [
        (points[i], points[i + 1]) for i in range(len(points) - 1) if points[i] < points[i + 1]
    ]


def _merge_undersized_groups(
    groups: list[tuple[int, int]], *, min_messages: int
) -> list[tuple[int, int]]:
    if not groups:
        return groups
    merged: list[list[int]] = [list(groups[0])]
    for start, end in groups[1:]:
        if merged[-1][1] - merged[-1][0] < min_messages:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < min_messages:
        merged[-2][1] = merged[-1][1]
        merged.pop()
    return [(s, e) for s, e in merged]


def _hard_split_groups(
    groups: Sequence[tuple[int, int]],
    messages: Sequence[MessageForSegmentation],
    *,
    max_messages: int,
    max_tokens: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in groups:
        chunk_start = start
        count = 0
        token_total = 0
        for i in range(start, end):
            cost = estimate_tokens(messages[i].text or "")
            would_count = count + 1
            would_tokens = token_total + cost
            if count > 0 and (would_count > max_messages or would_tokens > max_tokens):
                result.append((chunk_start, i))
                chunk_start = i
                count = 1
                token_total = cost
            else:
                count = would_count
                token_total = would_tokens
        result.append((chunk_start, end))
    return result


def segment_session(
    session: Session,
    *,
    topical_min_messages: int,
    max_messages: int,
    max_tokens: int,
    boundary_provider: BoundaryProvider,
) -> tuple[list[SegmentDraft], bool]:
    """Turn one `Session` into one or more `SegmentDraft`s.

    Returns `(drafts, used_fallback)`. `used_fallback` is True only
    when the boundary provider failed and the session degraded to a
    single "fallback:session"-labeled segment (SPEC §8 S4) — sessions
    at/below `topical_min_messages` also become one segment, but that
    is normal behavior, not a fallback, so it reports `False`.
    """
    messages = session.messages
    n = len(messages)

    if n <= topical_min_messages:
        return (
            [SegmentDraft(session_started_at=session.started_at, seq_in_session=0, messages=messages)],
            False,
        )

    try:
        boundaries = _detect_boundaries_windowed(messages, boundary_provider)
    except BoundaryDetectionError:
        return (
            [
                SegmentDraft(
                    session_started_at=session.started_at,
                    seq_in_session=0,
                    messages=messages,
                    topic_label="fallback:session",
                )
            ],
            True,
        )

    groups = _boundaries_to_groups(n, boundaries)
    groups = _merge_undersized_groups(groups, min_messages=MIN_MESSAGES_PER_SEGMENT)
    groups = _hard_split_groups(groups, messages, max_messages=max_messages, max_tokens=max_tokens)

    drafts = [
        SegmentDraft(
            session_started_at=session.started_at,
            seq_in_session=i,
            messages=tuple(messages[start:end]),
        )
        for i, (start, end) in enumerate(groups)
    ]
    return drafts, False


__all__ = [
    "BOUNDARY_WINDOW_OVERLAP_MESSAGES",
    "BOUNDARY_WINDOW_TOKENS",
    "MIN_MESSAGES_PER_SEGMENT",
    "BoundaryProvider",
    "FakeBoundaryProvider",
    "segment_session",
]
