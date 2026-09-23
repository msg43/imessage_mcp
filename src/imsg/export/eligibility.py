"""The eligibility engine (SPEC §11.2, hard requirement 5): default
deny, computed fresh from the live database every time it is asked.

It is also the public MCP surface's `allowlist` scope. SPEC §10.3a:
"Public `allowlist` scope applies the same eligibility predicate to
`search_messages`, `get_conversation`, `list_people`,
`get_attachment_text`, and any future tool". `imsg.retrieval.access`
calls `eligible_chat_ids` and `compute_attachment_eligibility` below
rather than keeping a second copy of the rule, so a change here changes
what export ships and what the public surface serves together.

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
retracted message is still an outsider in the thread. Messages in
Apple's "Recently Deleted" (`message.deleted_at`, D13) are excluded the
same way and count the same way: `exportable_message_sql` is the one
statement of the content rule, used by export and the allowlist scope
alike.

**Holding chats are denied outright** (D13, 2026-09-23). Extraction files
a message whose chat cannot be named into a holding chat
(`chat.unfiled_key`): one per lost group, per sender, or for the owner.
Its participants are whoever happened to be filed there, not the members
of a conversation anyone allowlisted, so no allowlist can make one
eligible. The rule reads only `chat`, so it runs in the first pass.

**Two ways to evaluate the same rules.** `compute_chat_eligibility` runs
every rule over every chat and keeps each chat's reasons, for the export
review report. `eligible_chat_ids` only needs the answer, and the public
surface asks for it on every request, so it runs the rules in two
passes: the participant rules, which read small tables, over every chat;
then the message rules, which read `message` and `tapback`, over only
the chats the first pass left. A chat the first pass denies is denied
whatever its messages say, so the two answers are the same set — which
`tests/test_retrieval_allowlist_parity_integration.py` checks on a
fixture database. Measured on the real corpus (2026-09-22, 13,074 chats,
675,556 messages, allowlists simulated read-only): the seven queries of
one full evaluation summed to 166 ms; the two passes took 17 ms with the
allowlist empty, 12 ms with only the owner allowlisted, 21-31 ms with
the owner and the people of 50 direct-message chats, and 188 ms with
every person allowlisted.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.export.models import (
    DENY_HOLDING_CHAT,
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


def effective_sender_sql(alias: str) -> str:
    """The person a `message` or `tapback` row (aliased `alias`) counts as
    sent by: its resolved sender, or the owner for an `is_from_me` row.

    `%(owner)s` is the singleton owner's person_id, or NULL when no owner
    person exists — in which case every is_from_me row's effective sender
    is NULL and the affected chats are denied (fail closed). Callers bind
    it from `owner_person_id`."""
    return f"coalesce({alias}.sender_person_id, CASE WHEN {alias}.is_from_me THEN %(owner)s::bigint END)"


_EFFECTIVE_SENDER = effective_sender_sql("m")
_EFFECTIVE_TAPBACK_SENDER = effective_sender_sql("t")


def exportable_message_sql(alias: str) -> str:
    """True for a `message` row (aliased `alias`) whose content may appear
    in an export document or under `allowlist` scope: never an unsent
    message (D1) and never one in Apple's "Recently Deleted" (D13). Not a
    parameter anywhere, so no caller can switch it off; under `full`
    scope the retrieval service shows deleted messages, labelled, by not
    applying it."""
    return f"(NOT {alias}.is_unsent AND {alias}.deleted_at IS NULL)"


@dataclass(frozen=True, slots=True)
class _DenyRule:
    """One kind of deny evidence: every chat for which `evidence` holds
    on at least one row of `source`.

    `reads_messages` puts the rule in the second pass of
    `eligible_chat_ids` (it reads `message`/`tapback`, the large tables);
    the other rules read participant tables only."""

    reason: str
    chat_column: str
    source: str
    evidence: str
    reads_messages: bool

    def sql(self, *, among: bool) -> str:
        """`SELECT DISTINCT ... AS chat_id` for every chat with this
        evidence — with `among`, only chats in `%(among)s`. The evidence is
        parenthesized so the restriction can never bind to one arm of an
        `OR` inside it."""
        restriction = f" AND {self.chat_column} = ANY(%(among)s::bigint[])" if among else ""
        return (
            f"SELECT DISTINCT {self.chat_column} AS chat_id FROM {self.source} "
            f"WHERE ({self.evidence}){restriction}"
        )


# Each rule returns chat ids that are *evidence for deny*. Adding a rule
# can only shrink the eligible set; removing one can only widen it —
# treat deletions here as security-relevant.
_DENY_RULES: tuple[_DenyRule, ...] = (
    _DenyRule(
        reason=DENY_HOLDING_CHAT,
        chat_column="c.chat_id",
        source="chat c",
        evidence="c.unfiled_key IS NOT NULL",
        reads_messages=False,
    ),
    _DenyRule(
        reason=DENY_PARTICIPANT_NOT_ALLOWLISTED,
        chat_column="cp.chat_id",
        source=(
            "chat_participant cp "
            "LEFT JOIN allowlist_person al ON al.person_id = cp.person_id"
        ),
        evidence="NOT coalesce(al.text_allowed, false)",
        reads_messages=False,
    ),
    _DenyRule(
        reason=DENY_UNRESOLVED_SOURCE_PARTICIPANT,
        chat_column="cps.chat_id",
        source=(
            "chat_participant_source cps "
            "LEFT JOIN source_handle_resolution shr "
            "ON shr.source_handle_id = cps.source_handle_id"
        ),
        evidence="shr.handle_id IS NULL",
        reads_messages=False,
    ),
    _DenyRule(
        reason=DENY_SOURCE_PERSON_NOT_ALLOWLISTED,
        chat_column="cps.chat_id",
        source=(
            "chat_participant_source cps "
            "JOIN source_handle_resolution shr ON shr.source_handle_id = cps.source_handle_id "
            "JOIN handle h ON h.handle_id = shr.handle_id "
            "LEFT JOIN allowlist_person al ON al.person_id = h.person_id"
        ),
        evidence="NOT coalesce(al.text_allowed, false)",
        reads_messages=False,
    ),
    _DenyRule(
        reason=DENY_UNRESOLVED_SENDER,
        chat_column="m.chat_id",
        source="message m",
        evidence=f"{_EFFECTIVE_SENDER} IS NULL",
        reads_messages=True,
    ),
    _DenyRule(
        reason=DENY_SENDER_NOT_ALLOWLISTED,
        chat_column="m.chat_id",
        source=f"message m LEFT JOIN allowlist_person al ON al.person_id = {_EFFECTIVE_SENDER}",
        evidence=f"{_EFFECTIVE_SENDER} IS NOT NULL AND NOT coalesce(al.text_allowed, false)",
        reads_messages=True,
    ),
    # Tapbacks are attributed to a chat through their target message; a
    # tapback whose target is unresolved cannot enter any document and
    # cannot be attributed to a chat, so it contributes no evidence.
    _DenyRule(
        reason=DENY_TAPBACK_SENDER,
        chat_column="m.chat_id",
        source=(
            "tapback t "
            "JOIN message m ON m.message_id = t.target_message_id "
            f"LEFT JOIN allowlist_person al ON al.person_id = {_EFFECTIVE_TAPBACK_SENDER}"
        ),
        evidence=f"{_EFFECTIVE_TAPBACK_SENDER} IS NULL OR NOT coalesce(al.text_allowed, false)",
        reads_messages=True,
    ),
)


def _participant_counts_sql(*, among: bool) -> str:
    restriction = " WHERE c.chat_id = ANY(%(among)s::bigint[])" if among else ""
    return (
        "SELECT c.chat_id, count(cp.person_id) FROM chat c "
        f"LEFT JOIN chat_participant cp ON cp.chat_id = c.chat_id{restriction} "
        "GROUP BY c.chat_id"
    )


def owner_person_id(conn: psycopg.Connection) -> int | None:
    """The singleton owner person's id, or None when no owner exists
    (in which case every chat containing is_from_me content denies)."""
    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE is_owner")
        row = cur.fetchone()
        return int(row[0]) if row else None


def compute_chat_eligibility(
    conn: psycopg.Connection, *, among: Collection[int] | None = None
) -> dict[int, ChatEligibility]:
    """Every chat's verdict, computed fresh — or, with `among`, only
    those chats' verdicts. Callers MUST treat a chat_id missing from the
    result as denied (default deny)."""
    if among is not None and not among:
        return {}
    restricted = among is not None
    params: dict[str, object] = {
        "owner": owner_person_id(conn),
        "among": sorted({int(c) for c in among}) if among is not None else None,
    }
    with conn.cursor() as cur:
        cur.execute(_participant_counts_sql(among=restricted), params)
        counts: dict[int, int] = {int(chat_id): int(n) for chat_id, n in cur.fetchall()}

        reasons: dict[int, set[str]] = {chat_id: set() for chat_id in counts}
        for rule in _DENY_RULES:
            cur.execute(rule.sql(among=restricted), params)
            for (chat_id,) in cur.fetchall():
                reasons.setdefault(int(chat_id), set()).add(rule.reason)

    return {
        chat_id: ChatEligibility(
            chat_id=chat_id,
            participant_count=counts.get(chat_id, 0),
            deny_reasons=frozenset(chat_reasons),
        )
        for chat_id, chat_reasons in reasons.items()
    }


def eligible_chat_ids(
    conn: psycopg.Connection, *, among: Collection[int] | None = None
) -> set[int]:
    """The set of chats whose segments may export — and the only chats
    the public surface serves under `allowlist` scope. Everything else is
    denied, including chats this function has never heard of. With
    `among`, only those chats are evaluated, so only they can be returned.

    The same answer as keeping the eligible verdicts of
    `compute_chat_eligibility`, reached in two passes (module docstring):
    chats with at least one participant and no participant-rule evidence,
    then, of those, the ones with no message-rule evidence. Both passes
    are set differences (`EXCEPT`) built from the rules above, so this
    function cannot admit a chat that any rule denies."""
    if among is not None and not among:
        return set()
    restricted = among is not None
    owner = owner_person_id(conn)
    members = "SELECT DISTINCT cp.chat_id AS chat_id FROM chat_participant cp" + (
        " WHERE cp.chat_id = ANY(%(among)s::bigint[])" if restricted else ""
    )
    first_pass = members + "".join(
        f" EXCEPT ({rule.sql(among=restricted)})"
        for rule in _DENY_RULES
        if not rule.reads_messages
    )
    second_pass = "SELECT unnest(%(among)s::bigint[]) AS chat_id" + "".join(
        f" EXCEPT ({rule.sql(among=True)})" for rule in _DENY_RULES if rule.reads_messages
    )
    with conn.cursor() as cur:
        cur.execute(
            first_pass,
            {"owner": owner, "among": sorted({int(c) for c in among}) if among is not None else None},
        )
        survivors = sorted(int(chat_id) for (chat_id,) in cur.fetchall())
        if not survivors:
            return set()
        cur.execute(second_pass, {"owner": owner, "among": survivors})
        return {int(chat_id) for (chat_id,) in cur.fetchall()}


def compute_attachment_eligibility(
    conn: psycopg.Connection, segment_ids: list[int]
) -> dict[tuple[int, int], bool]:
    """The separate attachment gate: `(segment_id, attachment_id) ->
    content may export`, for attachments linked into the given segments
    by at least one exportable message (`exportable_message_sql`: not
    unsent, not deleted).

    True requires EVERY exportable message link inside that segment to
    have a resolvable effective sender with `attachments_allowed =
    true` (SPEC §11.2 "via every message link through which it enters
    the document"). `bool_and` over `coalesce(..., false)` means a
    single unresolvable sender or missing allowlist row flips the whole
    pair to deny.

    Pairs absent from the result (e.g. linked only via unsent or deleted
    messages) must be treated as denied by callers — the attachment then
    simply never enters any document.
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
              AND {exportable_message_sql('m')}
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
    "effective_sender_sql",
    "eligible_chat_ids",
    "exportable_message_sql",
    "owner_person_id",
    "snapshot_allowlist",
]
