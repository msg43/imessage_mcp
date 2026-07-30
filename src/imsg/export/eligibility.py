"""The eligibility engine (SPEC §11.2, hard requirement 5): default
deny, computed fresh from the live database every time it is asked.

Design rules this module enforces mechanically:

1. **Absence denies.** A chat with no `allowlist_person` coverage, no
   participants, or missing from the returned index is ineligible. No
   query here can widen the eligible set — every SQL statement below
   *collects deny evidence*; eligibility is the absence of evidence
   plus a positive participant count.
2. **NULL denies.** An unresolved participant handle (a
   `chat_participant_source` row with no resolution), a message whose
   effective sender cannot be established, or a tapback with an
   unresolved non-owner sender are each treated as an outsider in the
   thread. Predicates are written so NULL lands on the deny side —
   `coalesce(x, false)`, never a bare `x = true` a NULL could slip
   past.
3. **Senders count, not just current participants.** chat.db's
   participant list is the *current* membership; someone who left a
   group still has messages in it. The spec's "every chat_participant
   allowlisted" check alone would let a former member's messages ride
   along — so every distinct message sender and attributed tapback
   sender in the chat must also be resolved and text-allowed. This is
   deliberately stricter than §11.2's letter; deny-more is always
   within the spec's spirit ("one outsider excludes the thread").
4. **The owner is gated through content, not membership.** chat.db
   does not reliably list the owner in a chat's participant rows, and
   `is_from_me` rows may carry a NULL `sender_person_id` (S3's
   invariant only promises *non-owner* resolution). Relying on the
   participant check alone would therefore silently exempt the owner —
   the exact `is_owner`-bypass D6 forbids. Instead, the *effective
   sender* of an `is_from_me` message or tapback is the singleton
   owner person, so any thread containing the owner's own content
   requires the owner to be explicitly `text_allowed`. No owner person
   row at all → every `is_from_me` row has no effective sender → deny.
5. **Attachments are gated separately** (`attachments_allowed`), per
   message link, within the parent segment (SPEC §11.2/§11.3).

Unsent messages: excluded from *documents* unconditionally (D1), but
their senders still count as deny evidence here — an outsider's
retracted message is still an outsider in the thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imsg.export.models import (
    DENY_PARTICIPANT_NOT_ALLOWLISTED,
    DENY_SENDER_NOT_ALLOWLISTED,
    DENY_SOURCE_PERSON_NOT_ALLOWLISTED,
    DENY_TAPBACK_SENDER,
    DENY_UNRESOLVED_SENDER,
    DENY_UNRESOLVED_SOURCE_PARTICIPANT,
    ChatEligibility,
)

if TYPE_CHECKING:
    import psycopg

# `%(owner)s` below is the singleton owner's person_id, or NULL when no
# owner person exists — in which case every is_from_me row's effective
# sender is NULL and the affected chats are denied (fail closed).
_EFFECTIVE_SENDER = "coalesce(m.sender_person_id, CASE WHEN m.is_from_me THEN %(owner)s::bigint END)"
_EFFECTIVE_TAPBACK_SENDER = (
    "coalesce(t.sender_person_id, CASE WHEN t.is_from_me THEN %(owner)s::bigint END)"
)

# Each query returns (chat_id, ...) rows that are *evidence for deny*.
# Adding a query can only shrink the eligible set; removing one can
# only widen it — treat deletions here as security-relevant.

_PARTICIPANT_COUNTS_SQL = """
    SELECT c.chat_id, count(cp.person_id)
    FROM chat c
    LEFT JOIN chat_participant cp ON cp.chat_id = c.chat_id
    GROUP BY c.chat_id
"""

_PARTICIPANT_NOT_ALLOWLISTED_SQL = """
    SELECT DISTINCT cp.chat_id
    FROM chat_participant cp
    LEFT JOIN allowlist_person al ON al.person_id = cp.person_id
    WHERE NOT coalesce(al.text_allowed, false)
"""

_UNRESOLVED_SOURCE_PARTICIPANT_SQL = """
    SELECT DISTINCT cps.chat_id
    FROM chat_participant_source cps
    LEFT JOIN source_handle_resolution shr
           ON shr.source_handle_id = cps.source_handle_id
    WHERE shr.handle_id IS NULL
"""

_SOURCE_PERSON_NOT_ALLOWLISTED_SQL = """
    SELECT DISTINCT cps.chat_id
    FROM chat_participant_source cps
    JOIN source_handle_resolution shr
      ON shr.source_handle_id = cps.source_handle_id
    JOIN handle h ON h.handle_id = shr.handle_id
    LEFT JOIN allowlist_person al ON al.person_id = h.person_id
    WHERE NOT coalesce(al.text_allowed, false)
"""

_UNRESOLVED_SENDER_SQL = f"""
    SELECT DISTINCT m.chat_id
    FROM message m
    WHERE {_EFFECTIVE_SENDER} IS NULL
"""

_SENDER_NOT_ALLOWLISTED_SQL = f"""
    SELECT DISTINCT m.chat_id
    FROM message m
    LEFT JOIN allowlist_person al ON al.person_id = {_EFFECTIVE_SENDER}
    WHERE {_EFFECTIVE_SENDER} IS NOT NULL
      AND NOT coalesce(al.text_allowed, false)
"""

# Tapbacks are attributed to a chat through their target message; a
# tapback whose target is unresolved cannot enter any document and
# cannot be attributed to a chat, so it contributes no evidence.
_TAPBACK_SENDER_SQL = f"""
    SELECT DISTINCT m.chat_id
    FROM tapback t
    JOIN message m ON m.message_id = t.target_message_id
    LEFT JOIN allowlist_person al ON al.person_id = {_EFFECTIVE_TAPBACK_SENDER}
    WHERE {_EFFECTIVE_TAPBACK_SENDER} IS NULL
       OR NOT coalesce(al.text_allowed, false)
"""

_REASON_QUERIES: tuple[tuple[str, str], ...] = (
    (DENY_PARTICIPANT_NOT_ALLOWLISTED, _PARTICIPANT_NOT_ALLOWLISTED_SQL),
    (DENY_UNRESOLVED_SOURCE_PARTICIPANT, _UNRESOLVED_SOURCE_PARTICIPANT_SQL),
    (DENY_SOURCE_PERSON_NOT_ALLOWLISTED, _SOURCE_PERSON_NOT_ALLOWLISTED_SQL),
    (DENY_UNRESOLVED_SENDER, _UNRESOLVED_SENDER_SQL),
    (DENY_SENDER_NOT_ALLOWLISTED, _SENDER_NOT_ALLOWLISTED_SQL),
    (DENY_TAPBACK_SENDER, _TAPBACK_SENDER_SQL),
)


def owner_person_id(conn: psycopg.Connection) -> int | None:
    """The singleton owner person's id, or None when no owner exists
    (in which case every chat containing is_from_me content denies)."""
    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE is_owner")
        row = cur.fetchone()
        return int(row[0]) if row else None


def compute_chat_eligibility(conn: psycopg.Connection) -> dict[int, ChatEligibility]:
    """Every chat's verdict, computed fresh. Callers MUST treat a
    chat_id missing from the result as denied (default deny)."""
    owner = owner_person_id(conn)
    params = {"owner": owner}
    with conn.cursor() as cur:
        cur.execute(_PARTICIPANT_COUNTS_SQL)
        counts: dict[int, int] = {int(chat_id): int(n) for chat_id, n in cur.fetchall()}

        reasons: dict[int, set[str]] = {chat_id: set() for chat_id in counts}
        for reason, sql in _REASON_QUERIES:
            cur.execute(sql, params)
            for (chat_id,) in cur.fetchall():
                reasons.setdefault(int(chat_id), set()).add(reason)

    return {
        chat_id: ChatEligibility(
            chat_id=chat_id,
            participant_count=counts.get(chat_id, 0),
            deny_reasons=frozenset(chat_reasons),
        )
        for chat_id, chat_reasons in reasons.items()
    }


def eligible_chat_ids(conn: psycopg.Connection) -> set[int]:
    """The set of chats whose segments may export. Everything else is
    denied — including chats this function has never heard of."""
    return {
        chat_id
        for chat_id, verdict in compute_chat_eligibility(conn).items()
        if verdict.eligible
    }


def compute_attachment_eligibility(
    conn: psycopg.Connection, segment_ids: list[int]
) -> dict[tuple[int, int], bool]:
    """The separate attachment gate: `(segment_id, attachment_id) ->
    content may export`, for attachments linked into the given segments
    by at least one non-unsent message.

    True requires EVERY non-unsent message link inside that segment to
    have a resolvable effective sender with `attachments_allowed =
    true` (SPEC §11.2 "via every message link through which it enters
    the document"). `bool_and` over `coalesce(..., false)` means a
    single unresolvable sender or missing allowlist row flips the whole
    pair to deny.

    Pairs absent from the result (e.g. linked only via unsent messages)
    must be treated as denied by callers — the attachment then simply
    never enters any document.
    """
    if not segment_ids:
        return {}
    owner = owner_person_id(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT sm.segment_id, ma.attachment_id,
                   bool_and(
                       {_EFFECTIVE_SENDER} IS NOT NULL
                       AND coalesce(al.attachments_allowed, false)
                   )
            FROM segment_message sm
            JOIN message m ON m.message_id = sm.message_id
            JOIN message_attachment ma ON ma.message_id = m.message_id
            LEFT JOIN allowlist_person al ON al.person_id = {_EFFECTIVE_SENDER}
            WHERE sm.segment_id = ANY(%(segment_ids)s)
              AND NOT m.is_unsent
            GROUP BY sm.segment_id, ma.attachment_id
            """,
            {"segment_ids": segment_ids, "owner": owner},
        )
        return {
            (int(segment_id), int(attachment_id)): bool(ok)
            for segment_id, attachment_id, ok in cur.fetchall()
        }


def snapshot_allowlist(conn: psycopg.Connection) -> list[dict[str, object]]:
    """The full `allowlist_person` state as a canonical, sorted list —
    frozen into `export_run.allowlist_snapshot` at plan time and
    compared byte-for-byte at push time (SPEC §11.4: approval pins
    bytes, not intent; a changed allowlist voids the plan)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT al.person_id, p.short_name, al.text_allowed, al.attachments_allowed
            FROM allowlist_person al
            JOIN person p ON p.person_id = al.person_id
            ORDER BY al.person_id
            """
        )
        return [
            {
                "person_id": int(person_id),
                "short_name": str(short_name),
                "text_allowed": bool(text_allowed),
                "attachments_allowed": bool(attachments_allowed),
            }
            for person_id, short_name, text_allowed, attachments_allowed in cur.fetchall()
        ]


__all__ = [
    "compute_attachment_eligibility",
    "compute_chat_eligibility",
    "eligible_chat_ids",
    "owner_person_id",
    "snapshot_allowlist",
]
