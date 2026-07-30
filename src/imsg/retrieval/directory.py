"""`list_people` and `get_attachment_text` (SPEC §10.2) — simple
directory-style lookups that do not go through the hybrid query flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from imsg.retrieval.access import AccessContext
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
    context: AccessContext,
    *,
    query: str | None,
    limit: int,
    include_handles: bool = False,
) -> list[PersonListing]:
    """SPEC §10.2 `list_people`. `query` filters on `short_name` OR
    `display_name` (case-insensitive substring); omitted, every person
    with at least one message. Scope-filtered exactly like
    `imsg.retrieval.people`'s fuzzy suggestions (D6): under
    `scope='allowlist'`, only allowlisted persons are ever listed —
    never raw handles regardless of scope (SPEC §10.2: "never raw
    handles on the public surface"; `include_handles` is a **local**-
    only extension so it is meaningless to gate on scope here, but the
    caller — the local MCP server — never sets `scope != 'full'`
    anyway)."""
    clauses = ["1=1"]
    params: dict[str, object] = {"limit": limit}
    if query:
        clauses.append("(p.short_name ILIKE %(q)s OR p.display_name ILIKE %(q)s)")
        params["q"] = f"%{query}%"
    if context.scope == "allowlist":
        clauses.append(
            "EXISTS (SELECT 1 FROM allowlist_person ap "
            "WHERE ap.person_id = p.person_id AND ap.text_allowed)"
        )

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT p.person_id, p.short_name, p.display_name, p.organization,
                   count(m.message_id) AS message_count,
                   min(m.sent_at) AS first_message, max(m.sent_at) AS last_message
            FROM person p
            LEFT JOIN message m ON m.sender_person_id = p.person_id
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
    """SPEC §10.2 `get_attachment_text`. Authorization: at least one
    message/segment linking to this attachment must be eligible under
    `context` (D6: "Authorization checks every linked message/segment
    through `message_attachment`; text is returned only when at least
    one authorized parent exists"). An attachment that exists but has
    no authorized parent is indistinguishable from one that does not
    exist at all (D6's existence-oracle rule) — both raise
    `NotFoundError`."""
    from imsg.retrieval.access import segment_eligibility_predicate

    predicate = segment_eligibility_predicate(context, chat_id_expr="s.chat_id")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, filename, mime_type FROM attachment WHERE attachment_key = %s",
            (attachment_key,),
        )
        row = cur.fetchone()
        if row is None:
            raise NotFoundError(f"no attachment found for attachment_key {attachment_key!r}")
        attachment_id, filename, mime_type = row

        cur.execute(
            f"""
            SELECT EXISTS (
                SELECT 1
                FROM message_attachment ma
                JOIN segment_message sm ON sm.message_id = ma.message_id
                JOIN segment s ON s.segment_id = sm.segment_id
                WHERE ma.attachment_id = %(attachment_id)s AND ({predicate})
            )
            """,
            {"attachment_id": attachment_id},
        )
        authorized = bool(cur.fetchone()[0])  # type: ignore[index]
        if not authorized:
            raise NotFoundError(f"no attachment found for attachment_key {attachment_key!r}")

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
