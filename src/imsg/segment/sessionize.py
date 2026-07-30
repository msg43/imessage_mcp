"""Pass 1 (SPEC §8 S4): split a chat's message stream into sessions on
time gaps, plus the incremental-frontier fix (SPEC v1.1 / D6) for
finding where a re-segmentation run must start.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import datetime, timedelta

from imsg.segment.models import MessageForSegmentation, PersistedSessionSpan, Session


def sessionize(
    messages: Sequence[MessageForSegmentation],
    *,
    chat_id: int,
    session_gap_hours: float,
) -> list[Session]:
    """Split `messages` (must already be sorted by `sent_at` ascending,
    all belonging to `chat_id`) into sessions wherever the gap between
    consecutive messages exceeds `session_gap_hours`.

    Empty input returns an empty list. A single message is its own
    one-message session (pass 2 / `topical_min_messages` decides
    whether it becomes its own segment).
    """
    if not messages:
        return []

    gap = timedelta(hours=session_gap_hours)
    sessions: list[Session] = []
    current: list[MessageForSegmentation] = [messages[0]]

    for prev, msg in itertools.pairwise(messages):
        if msg.sent_at - prev.sent_at > gap:
            sessions.append(
                Session(
                    chat_id=chat_id,
                    started_at=current[0].sent_at,
                    ended_at=current[-1].sent_at,
                    messages=tuple(current),
                    gap_hours=session_gap_hours,
                )
            )
            current = [msg]
        else:
            current.append(msg)

    sessions.append(
        Session(
            chat_id=chat_id,
            started_at=current[0].sent_at,
            ended_at=current[-1].sent_at,
            messages=tuple(current),
            gap_hours=session_gap_hours,
        )
    )
    return sessions


def compute_recompute_start(
    existing_sessions: Sequence[PersistedSessionSpan],
    earliest_changed_at: datetime,
    session_gap_hours: float,
) -> datetime:
    """Where a re-segmentation run for this chat must start rebuilding
    from (SPEC §8 S4 "Incremental frontier", the v1.1 fix for the bug
    where a bare max-message watermark wrongly opens a new session for
    a reply that actually belongs in the still-open tail session).

    `existing_sessions` must be sorted by `started_at` ascending.
    Returns a timestamp `T` such that:

    - every persisted session with `started_at < T` is provably
      unaffected by whatever changed at `earliest_changed_at` (its
      trailing gap to the next session — or to `earliest_changed_at`
      itself if it's the last one — is already `> session_gap_hours`,
      so no possible edit *at* `earliest_changed_at` could pull it into
      that session), and
    - the caller re-fetches every message with `sent_at >= T` for this
      chat from Postgres (not just previously-segmented ones — new
      rows in that range must be included) and re-runs `sessionize` +
      pass 2 over them from scratch.

    If no persisted session is safely sealed before
    `earliest_changed_at` (including the empty-history case), this
    returns the start of the earliest session (or `earliest_changed_at`
    itself when there is no history at all) — i.e. rebuild everything.
    """
    if not existing_sessions:
        return earliest_changed_at

    gap = timedelta(hours=session_gap_hours)
    for i in range(len(existing_sessions) - 1, -1, -1):
        span = existing_sessions[i]
        if span.ended_at + gap <= earliest_changed_at:
            # `span` is sealed: nothing at or after earliest_changed_at
            # could ever have joined it, gap-wise. Recompute starts
            # either at the next persisted session, or — if `span` was
            # the last one — at the change itself (a fresh session,
            # nothing existing needs touching).
            if i + 1 < len(existing_sessions):
                return existing_sessions[i + 1].started_at
            return earliest_changed_at

    return existing_sessions[0].started_at


__all__ = ["compute_recompute_start", "sessionize"]
