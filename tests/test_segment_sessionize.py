"""Pass 1 (SPEC §8 S4) + the incremental-frontier fix (D6/v1.1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from imsg.segment.models import MessageForSegmentation, PersistedSessionSpan
from imsg.segment.sessionize import compute_recompute_start, sessionize

_BASE = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _msg(offset_minutes: float, message_id: int = 0, text: str = "hi") -> MessageForSegmentation:
    return MessageForSegmentation(
        message_id=message_id or int(offset_minutes * 1000),
        source_guid=f"guid-{offset_minutes}",
        chat_id=1,
        sent_at=_BASE + timedelta(minutes=offset_minutes),
        is_from_me=False,
        sender_short_name="alice",
        text=text,
        is_unsent=False,
        is_edited=False,
        has_attachments=False,
    )


def test_sessionize_empty_input() -> None:
    assert sessionize([], chat_id=1, session_gap_hours=3.0) == []


def test_sessionize_single_message_is_one_session() -> None:
    sessions = sessionize([_msg(0)], chat_id=1, session_gap_hours=3.0)
    assert len(sessions) == 1
    assert sessions[0].messages == (_msg(0),)


def test_sessionize_keeps_close_messages_in_one_session() -> None:
    messages = [_msg(0), _msg(30), _msg(60)]
    sessions = sessionize(messages, chat_id=1, session_gap_hours=3.0)
    assert len(sessions) == 1
    assert len(sessions[0].messages) == 3


def test_sessionize_splits_on_gap_over_threshold() -> None:
    messages = [_msg(0), _msg(10), _msg(10 + 4 * 60)]  # 4h gap > 3h threshold
    sessions = sessionize(messages, chat_id=1, session_gap_hours=3.0)
    assert len(sessions) == 2
    assert len(sessions[0].messages) == 2
    assert len(sessions[1].messages) == 1


def test_sessionize_gap_exactly_at_threshold_does_not_split() -> None:
    messages = [_msg(0), _msg(180)]  # exactly 3h — spec says "> session_gap_hours"
    sessions = sessionize(messages, chat_id=1, session_gap_hours=3.0)
    assert len(sessions) == 1


def test_sessionize_stamps_gap_hours_used() -> None:
    sessions = sessionize([_msg(0)], chat_id=1, session_gap_hours=1.5)
    assert sessions[0].gap_hours == 1.5


# --- compute_recompute_start ---


def _span(session_id: int, start_offset: float, end_offset: float) -> PersistedSessionSpan:
    return PersistedSessionSpan(
        session_id=session_id,
        started_at=_BASE + timedelta(minutes=start_offset),
        ended_at=_BASE + timedelta(minutes=end_offset),
    )


def test_recompute_start_no_history_returns_the_change_itself() -> None:
    changed_at = _BASE + timedelta(hours=1)
    assert compute_recompute_start([], changed_at, session_gap_hours=3.0) == changed_at


def test_recompute_start_tail_extension_reopens_the_last_session() -> None:
    """The v1.1 fix: a new message arriving within the gap of the last
    session's end must reopen that session, not just append a fresh one."""
    sessions = [_span(1, 0, 60)]  # session ended at BASE+60min
    changed_at = _BASE + timedelta(minutes=60 + 30)  # 30 min later, well within 3h gap
    start = compute_recompute_start(sessions, changed_at, session_gap_hours=3.0)
    assert start == sessions[0].started_at


def test_recompute_start_new_message_beyond_gap_does_not_touch_history() -> None:
    sessions = [_span(1, 0, 60)]
    changed_at = _BASE + timedelta(minutes=60) + timedelta(hours=4)  # beyond 3h gap
    start = compute_recompute_start(sessions, changed_at, session_gap_hours=3.0)
    assert start == changed_at


def test_recompute_start_only_touches_the_affected_tail() -> None:
    """Two sealed sessions, then an edit inside a message near the end of
    the second — only that (already-tail) session should be rebuilt,
    not the first."""
    sessions = [
        _span(1, 0, 60),
        _span(2, 60 + 4 * 60, 60 + 4 * 60 + 90),  # starts 4h after session 1 ends
    ]
    changed_at = sessions[1].started_at + timedelta(minutes=10)  # edit inside session 2
    start = compute_recompute_start(sessions, changed_at, session_gap_hours=3.0)
    assert start == sessions[1].started_at


def test_recompute_start_edit_far_in_the_past_rebuilds_from_first_session() -> None:
    sessions = [
        _span(1, 0, 60),
        _span(2, 60 + 4 * 60, 60 + 4 * 60 + 90),
    ]
    changed_at = sessions[0].started_at + timedelta(minutes=5)  # edit inside session 1
    start = compute_recompute_start(sessions, changed_at, session_gap_hours=3.0)
    assert start == sessions[0].started_at


def test_recompute_start_change_in_a_hole_between_two_sessions_is_not_skipped() -> None:
    """The frontier is where re-fetching *starts*, so it must never land
    after the earliest changed message.

    New messages can arrive in the hole between two persisted sessions —
    after one that the session gap already sealed, and before the next
    one begins. Returning the next session's `started_at` unclamped
    skips straight over them: `_fetch_messages_from` never fetches them,
    they are never segmented, and the chat stays dirty on every
    subsequent run.
    """
    sessions = [
        _span(1, 0, 60),  # sealed: ends 20h before the change
        _span(2, 30 * 60, 30 * 60 + 30),  # starts 10h after the change
    ]
    changed_at = _BASE + timedelta(hours=20)  # lands in the hole between them
    start = compute_recompute_start(sessions, changed_at, session_gap_hours=3.0)
    assert start == changed_at


def test_recompute_start_change_predating_all_history_is_not_skipped() -> None:
    """The fall-through path (`no sealed session`) has the same shape: a
    change older than every persisted session must not be skipped by
    rebuilding from the first session's start."""
    sessions = [
        _span(1, 0, 60),
        _span(2, 60 + 4 * 60, 60 + 4 * 60 + 90),
    ]
    changed_at = _BASE - timedelta(hours=2)  # older than every session
    start = compute_recompute_start(sessions, changed_at, session_gap_hours=3.0)
    assert start == changed_at


def test_recompute_start_change_inside_a_middle_session_rebuilds_from_that_session() -> None:
    """A change *inside* an existing session still rebuilds from that
    session's start (earlier than the change), not from the change —
    the whole session has to be re-sessionized as a unit."""
    sessions = [
        _span(1, 0, 60),
        _span(2, 300, 390),
        _span(3, 700, 760),
    ]
    changed_at = sessions[1].started_at + timedelta(minutes=20)  # inside session 2
    start = compute_recompute_start(sessions, changed_at, session_gap_hours=3.0)
    assert start == sessions[1].started_at
    assert start < changed_at


def test_recompute_start_is_never_later_than_the_change() -> None:
    """Property: for any arrangement of persisted sessions and any
    `earliest_changed_at`, the returned frontier is <= the change. The
    caller re-fetches `sent_at >= frontier`, so a frontier past the
    change makes the changed messages permanently unfetchable."""
    span_starts = (0, 300, 700, 1500)
    durations = (0, 45, 200)
    changed_offsets = (-600, -1, 0, 30, 120, 305, 500, 705, 1490, 1600, 5000)
    for gap_hours in (1.0, 3.0, 12.0):
        for n_sessions in range(len(span_starts) + 1):
            for duration in durations:
                sessions = [
                    _span(i + 1, span_starts[i], span_starts[i] + duration)
                    for i in range(n_sessions)
                ]
                for offset in changed_offsets:
                    changed_at = _BASE + timedelta(minutes=offset)
                    start = compute_recompute_start(
                        sessions, changed_at, session_gap_hours=gap_hours
                    )
                    assert start <= changed_at, (
                        f"frontier {start} is after the change {changed_at} "
                        f"(gap={gap_hours}h, sessions={[(s.started_at, s.ended_at) for s in sessions]})"
                    )
