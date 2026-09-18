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

    `T` is never later than `earliest_changed_at`: it is where the
    caller's `sent_at >= T` re-fetch *starts*, so a `T` past the change
    would leave the changed rows unfetchable, hence unsegmentable, on
    that run and on every run after it. See the clamp below.
    """
    gap = timedelta(hours=session_gap_hours)
    frontier = existing_sessions[0].started_at if existing_sessions else earliest_changed_at

    for i in range(len(existing_sessions) - 1, -1, -1):
        span = existing_sessions[i]
        if span.ended_at + gap <= earliest_changed_at:
            # `span` is sealed: nothing at or after earliest_changed_at
            # could ever have joined it, gap-wise. Recompute starts
            # either at the next persisted session, or — if `span` was
            # the last one — at the change itself (a fresh session,
            # nothing existing needs touching).
            frontier = (
                existing_sessions[i + 1].started_at
                if i + 1 < len(existing_sessions)
                else earliest_changed_at
            )
            break

    # The clamp, and the reason this function has a single exit: the
    # candidate above is only ever a *lower* bound relaxation — "you may
    # safely skip everything before this persisted session". It says
    # nothing about where the changed rows are, and two arrangements put
    # them earlier than it:
    #
    #   1. new messages in the *hole* between a sealed session and the
    #      next persisted one (`existing_sessions[i + 1].started_at` is
    #      after them), and
    #   2. a change predating every persisted session
    #      (`existing_sessions[0].started_at` is after it).
    #
    # In both, an unclamped frontier makes those messages permanently
    # unreachable: `_fetch_messages_from` never sees them, so they are
    # never segmented, the chat stays dirty, and the next run computes
    # the same overshooting frontier again. Re-fetching a little earlier
    # than strictly necessary only costs work; re-fetching too late
    # loses messages.
    return min(frontier, earliest_changed_at)


def compute_recompute_end(
    existing_sessions: Sequence[PersistedSessionSpan],
    latest_changed_at: datetime,
    session_gap_hours: float,
) -> datetime | None:
    """Where a re-segmentation run for this chat may stop, given that
    `latest_changed_at` is the last point in time anything changed.

    Returns the `started_at` of the first persisted session that begins
    more than `session_gap_hours` after `latest_changed_at`, or `None`
    when no persisted session does (rebuild to the end of the chat, the
    behaviour before this bound existed). `existing_sessions` must be
    sorted by `started_at` ascending, same as
    `compute_recompute_start`.

    The caller treats the result as an *exclusive* upper bound: it
    fetches messages with `sent_at < T` and rebuilds only persisted
    sessions with `started_at < T`.

    **Why that session and everything after it is provably unaffected.**
    Let `B` be the returned bound and `P` the latest `sent_at` strictly
    before it. A session boundary sits at `B` exactly when
    `B - P > gap`, so the only way to destroy it is to move `P` later.
    Three things can change, and none of them can:

    - **An edit or an identity change never moves `sent_at`.** It
      rewrites a body or a `sender_person_id` and bumps `updated_at`;
      the timestamp a message is ordered and gap-split by is untouched.
      So no existing message moves closer to `B`.
    - **A deletion (or a retraction filtered out by
      `policy.index_unsent`) only ever removes a message**, which moves
      `P` *earlier* and widens the gap. A boundary cannot be destroyed
      by widening the gap in front of it.
    - **Every added message is at or before `latest_changed_at`.** A
      message that is not yet in any segment is reported by
      `find_dirty_chats` with its own `sent_at`, and the span it
      returns takes `MAX` over exactly those rows — so an added message
      later than `latest_changed_at` cannot exist without having moved
      `latest_changed_at` to itself. Since `B - latest_changed_at >
      gap`, any added message is at least a full `gap` before `B` and
      cannot pull `P` into reach.

    The same argument covers every persisted session after `B`: they
    start later still, so they too begin more than `gap` after the last
    change.

    The one case this cannot see is a `message` row that is *hard
    deleted* from Postgres rather than retracted — there is no
    `updated_at` left to observe, so `find_dirty_chats` cannot report
    it and the span above never widens to cover it. That predates this
    bound (`find_dirty_chats` would not even report the chat), and
    nothing in the extract stage deletes `message` rows; `imsg segment
    --rebuild --chat <id>` is the repair path if one ever does.
    """
    gap = timedelta(hours=session_gap_hours)
    for span in existing_sessions:
        if span.started_at - latest_changed_at > gap:
            return span.started_at
    return None


__all__ = ["compute_recompute_end", "compute_recompute_start", "sessionize"]
