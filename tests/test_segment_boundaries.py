"""Pass 2 (SPEC §8 S4, D4): windowed boundary detection, the min-2/
max-50/max-2000-token caps, and the fallback-on-failure path."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from imsg.segment.boundaries import (
    BOUNDARY_WINDOW_TOKENS,
    FakeBoundaryProvider,
    segment_session,
)
from imsg.segment.models import MessageForSegmentation, Session
from imsg.tokens import estimate_tokens

_BASE = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _session(n: int, *, text_len: int = 10) -> Session:
    messages = tuple(
        MessageForSegmentation(
            message_id=i,
            source_guid=f"guid-{i}",
            chat_id=1,
            sent_at=_BASE + timedelta(minutes=i),
            is_from_me=(i % 2 == 0),
            sender_short_name="owner" if i % 2 == 0 else "alice",
            text="x" * text_len,
            is_unsent=False,
            is_edited=False,
            has_attachments=False,
        )
        for i in range(n)
    )
    return Session(
        chat_id=1,
        started_at=messages[0].sent_at,
        ended_at=messages[-1].sent_at,
        messages=messages,
        gap_hours=3.0,
    )


def test_session_at_or_below_topical_min_becomes_one_segment() -> None:
    session = _session(10)
    drafts, used_fallback = segment_session(
        session,
        topical_min_messages=10,
        max_messages=50,
        max_tokens=2000,
        boundary_provider=FakeBoundaryProvider(),
    )
    assert len(drafts) == 1
    assert drafts[0].message_count == 10
    assert used_fallback is False


def test_session_above_threshold_uses_boundary_provider() -> None:
    session = _session(20)
    provider = FakeBoundaryProvider(messages_per_segment=5)
    drafts, used_fallback = segment_session(
        session,
        topical_min_messages=10,
        max_messages=50,
        max_tokens=2000,
        boundary_provider=provider,
    )
    assert used_fallback is False
    assert sum(d.message_count for d in drafts) == 20
    assert all(d.message_count >= 2 for d in drafts)


def test_boundary_provider_failure_falls_back_to_session_as_segment() -> None:
    session = _session(20)
    provider = FakeBoundaryProvider(always_fail=True)
    drafts, used_fallback = segment_session(
        session,
        topical_min_messages=10,
        max_messages=50,
        max_tokens=2000,
        boundary_provider=provider,
    )
    assert used_fallback is True
    assert len(drafts) == 1
    assert drafts[0].message_count == 20
    assert drafts[0].topic_label == "fallback:session"


def test_out_of_range_boundary_index_falls_back_to_session_as_segment() -> None:
    """`segment_session` treats an out-of-range index the same as any
    other boundary-detection failure (SPEC §8 S4): fall back, don't
    propagate — the whole run must not abort over one bad session."""

    class BadProvider:
        model_id = "bad"

        def detect_boundaries(self, window: object) -> list[int]:
            return [9999]

    session = _session(20)
    drafts, used_fallback = segment_session(
        session,
        topical_min_messages=10,
        max_messages=50,
        max_tokens=2000,
        boundary_provider=BadProvider(),
    )
    assert used_fallback is True
    assert drafts[0].topic_label == "fallback:session"


def test_one_retry_before_fallback() -> None:
    """"malformed LLM JSON -> one retry, then fallback" (SPEC §8 S4):
    a provider that fails once then succeeds should NOT fall back."""
    session = _session(20)
    provider = FakeBoundaryProvider(messages_per_segment=5, fail_first_n_calls=1)
    drafts, used_fallback = segment_session(
        session,
        topical_min_messages=10,
        max_messages=50,
        max_tokens=2000,
        boundary_provider=provider,
    )
    assert used_fallback is False
    assert sum(d.message_count for d in drafts) == 20


def test_hard_cap_splits_on_max_messages() -> None:
    session = _session(100)
    # No boundaries detected at all -> one huge group -> must be hard-split.
    provider = FakeBoundaryProvider(messages_per_segment=10_000)
    drafts, _ = segment_session(
        session,
        topical_min_messages=10,
        max_messages=20,
        max_tokens=100_000,
        boundary_provider=provider,
    )
    assert all(d.message_count <= 20 for d in drafts)
    assert sum(d.message_count for d in drafts) == 100


def test_hard_cap_splits_on_max_tokens() -> None:
    # ~1000 messages of 40 chars (~10 tokens each) -> 25 messages hits 250 tokens.
    session = _session(60, text_len=40)
    provider = FakeBoundaryProvider(messages_per_segment=10_000)
    drafts, _ = segment_session(
        session,
        topical_min_messages=10,
        max_messages=1000,
        max_tokens=250,
        boundary_provider=provider,
    )
    assert len(drafts) > 1
    for d in drafts:
        total_tokens = sum(estimate_tokens(m.text or "") for m in d.messages)
        # Allow the single-message-over-budget edge case; otherwise capped.
        assert total_tokens <= 250 or d.message_count == 1


def test_windowing_covers_a_session_larger_than_one_window() -> None:
    """A session whose text volume exceeds one 4k-token boundary window
    must still get every message classified into some segment — no
    message silently dropped by the windowing/overlap logic."""
    # ~50 tokens/message * 200 messages = ~10k tokens, several windows.
    session = _session(200, text_len=200)
    assert sum(len(m.text or "") for m in session.messages) // 4 > BOUNDARY_WINDOW_TOKENS
    provider = FakeBoundaryProvider(messages_per_segment=7)
    drafts, used_fallback = segment_session(
        session,
        topical_min_messages=10,
        max_messages=50,
        max_tokens=2000,
        boundary_provider=provider,
    )
    assert used_fallback is False
    assert sum(d.message_count for d in drafts) == 200
    seen_ids = sorted(m.message_id for d in drafts for m in d.messages)
    assert seen_ids == list(range(200))
