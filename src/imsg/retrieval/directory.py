"""`list_people` and `get_attachment_text` (SPEC §10.2) — simple
directory-style lookups that do not go through the hybrid query flow.

Both apply the request's scope from `imsg.retrieval.access`: under
`allowlist`, `list_people` lists only the people who appear in an
eligible chat, with activity counted inside eligible chats only, and
`get_attachment_text` returns text only through a parent segment where
export's chat rule and its separate attachment gate both pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.export.eligibility import effective_sender_sql, exportable_message_sql
from imsg.retrieval.access import (
    AccessContext,
    RequestScope,
    attachment_gate,
    resolve_request_scope,
    visible_person_ids,
)
from imsg.retrieval.errors import NotEnrichedError, NotFoundError

if TYPE_CHECKING:
    import psycopg


# --------------------------------------------------------------------------
# list_people
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PersonListing:
    short_name: str
    display_name: str
    organization: str | None
    message_count: int
    first_message: str | None
    last_message: str | None
    handles: tuple[str, ...] | None = None
    """`None` unless `include_handles=True` — SPEC §10.2: "The
    **local** registration extends the schema with `include_handles:
    true`; the public registration omits the property entirely." This
    build is the local surface, so the parameter exists; a future
    public-transport build should never pass `include_handles=True`."""


def list_people(
    conn: psycopg.Connection,
    scope: RequestScope,
    *,
    query: str | None,
    limit: int,
    include_handles: bool = False,
) -> list[PersonListing]:
    """SPEC §10.2 `list_people`. `query` filters on `short_name` OR
    `display_name` (case-insensitive substring).

    Under `full` scope: every person, with counts over all their
    messages. Under `allowlist` (D6: what the public surface lists is
    scope-filtered, like `imsg.retrieval.people`'s suggestions): only
    `imsg.retrieval.access.visible_person_ids` — people who appear in an
    eligible chat — with `message_count`, `first_message` and
    `last_message` over their exportable messages (not unsent, not
    deleted) in eligible chats only, so nothing about an ineligible
    thread shows through the counts. An
    allowlisted person with no eligible chat is not listed: nothing the
    caller could search for involves them.

    Never raw handles on the public surface (SPEC §10.2);
    `include_handles` is a local-only extension the service refuses on
    any other surface."""
    clauses = ["TRUE"]
    params: dict[str, object] = {"limit": limit}
    if query:
        clauses.append("(p.short_name ILIKE %(q)s OR p.display_name ILIKE %(q)s)")
        params["q"] = f"%{query}%"

    visible = visible_person_ids(conn, scope)
    if visible is None:
        message_join = "m.sender_person_id = p.person_id"
    else:
        if not visible:
            return []
        clauses.append("p.person_id = ANY(%(visible)s::bigint[])")
        params["visible"] = sorted(visible)
        chats = scope.chat_predicate("m.chat_id")
        params.update(chats.params)
        params["owner"] = scope.owner_person_id
        message_join = (
            f"{effective_sender_sql('m')} = p.person_id AND {chats.sql} "
            f"AND {exportable_message_sql('m')}"
        )

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT p.person_id, p.short_name, p.display_name, p.organization,
                   count(m.message_id) AS message_count,
                   min(m.sent_at) AS first_message, max(m.sent_at) AS last_message
            FROM person p
            LEFT JOIN message m ON {message_join}
            WHERE {" AND ".join(clauses)}
            GROUP BY p.person_id
            ORDER BY p.display_name
            LIMIT %(limit)s
            """,
            params,
        )
        rows = cur.fetchall()

        handles_by_person: dict[int, list[str]] = {}
        if include_handles and rows:
            person_ids = [r[0] for r in rows]
            cur.execute(
                "SELECT person_id, normalized_value FROM handle "
                "WHERE person_id = ANY(%s) ORDER BY person_id, normalized_value",
                (person_ids,),
            )
            for person_id, normalized_value in cur.fetchall():
                handles_by_person.setdefault(person_id, []).append(normalized_value)

    return [
        PersonListing(
            short_name=short_name,
            display_name=display_name,
            organization=organization,
            message_count=message_count,
            first_message=first_message.isoformat() if first_message else None,
            last_message=last_message.isoformat() if last_message else None,
            handles=tuple(handles_by_person.get(person_id, ())) if include_handles else None,
        )
        for person_id, short_name, display_name, organization, message_count, first_message, last_message in rows
    ]


# --------------------------------------------------------------------------
# get_attachment_text
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttachmentText:
    attachment_key: str
    filename: str | None
    mime_type: str | None
    texts: tuple[dict[str, str], ...]
    """`{kind, model, text}` per done enrichment row (SPEC §10.2)."""


def get_attachment_text(
    conn: psycopg.Connection, context: AccessContext, attachment_key: str
) -> AttachmentText:
    """SPEC §10.2 `get_attachment_text`. "Authorization checks every
    linked message/segment through `message_attachment`; text is
    returned only when at least one authorized parent exists" (D6).

    A parent is a segment containing a message that links the
    attachment. Under `full` scope any parent authorizes. Under
    `allowlist` a parent authorizes only if its chat passes export's chat
    rule AND the attachment passes export's separate attachment gate in
    that segment — every exportable message linking it there has a
    sender with `attachments_allowed` (SPEC §11.2: "An attachment's text
    exports iff its segment is eligible and the attachment's sender (via
    every message link through which it enters the document) has
    `attachments_allowed = true`"). §10.3a applies that same predicate to
    this tool, so the public surface returns an attachment's text exactly
    when export would ship it under at least one parent.

    An attachment that exists but has no authorized parent is
    indistinguishable from one that does not exist at all (D6's
    existence-oracle rule) — both raise the same `NotFoundError`."""
    not_found = NotFoundError(f"no attachment found for attachment_key {attachment_key!r}")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, filename, mime_type FROM attachment WHERE attachment_key = %s",
            (attachment_key,),
        )
        row = cur.fetchone()
        if row is None:
            raise not_found
        attachment_id, filename, mime_type = row

        cur.execute(
            """
            SELECT DISTINCT sm.segment_id, s.chat_id
            FROM message_attachment ma
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE ma.attachment_id = %(attachment_id)s
            """,
            {"attachment_id": attachment_id},
        )
        parents = [(int(segment_id), int(chat_id)) for segment_id, chat_id in cur.fetchall()]

    scope = resolve_request_scope(
        conn, context, among_chat_ids={chat_id for _, chat_id in parents}
    )
    admitted = [segment_id for segment_id, chat_id in parents if scope.admits_chat(chat_id)]
    gate = attachment_gate(conn, scope, admitted)
    if not any(gate.allows(segment_id, int(attachment_id)) for segment_id in admitted):
        raise not_found

    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind, model, state, text FROM enrichment WHERE attachment_id = %s ORDER BY kind",
            (attachment_id,),
        )
        enrichment_rows = cur.fetchall()

    done = [
        {"kind": kind, "model": model or "", "text": text or ""}
        for kind, model, state, text in enrichment_rows
        if state == "done"
    ]
    if not done:
        if not enrichment_rows:
            raise NotEnrichedError(
                f"attachment {attachment_key!r} has no enrichment queued yet"
            )
        states = sorted({state for _, _, state, _ in enrichment_rows})
        raise NotEnrichedError(
            f"attachment {attachment_key!r} exists but is not enriched yet (state(s): "
            f"{', '.join(states)})"
        )

    return AttachmentText(
        attachment_key=attachment_key, filename=filename, mime_type=mime_type, texts=tuple(done)
    )


__all__ = [
    "AttachmentText",
    "PersonListing",
    "get_attachment_text",
    "list_people",
]
