"""Where a message came from: the page's Details panel.

The panel answers the questions a citation leaves open: the exact time
with its zone offset, who sent it and over which service, when it was
edited and what it said before, when it was deleted, whether it was
unsent, how extraction filed it into its conversation, which sources hold
it (the live Messages database, older or recovered copies) with the run
that last read each, its attachments with their SHA-256, and its IDs.

Two fields widen what the page shows, so each has its own switch in
`search_page.details`, off by default, and the panel says when a field is
hidden by one:

- `show_edit_history`: the text of earlier versions of an edited message
  (`message_version`). The edit time itself always shows.
- `show_raw_handles`: the sender's raw number or email as the Messages
  database recorded it (`source_handle.raw_value`, S2's provenance, which
  nothing else downstream reads). Only incoming messages carry one;
  extraction records no handle on the owner's own messages.

Visibility follows the rest of the page: an unsent message has no panel
unless `policy.index_unsent` is on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from imsg.search_page.threads import chat_views, is_opaque_key

if TYPE_CHECKING:
    import psycopg

FILED_BY: dict[str, str] = {
    "chat_message_join": "the Messages database links it to this conversation",
    "recoverable_join": "Recently Deleted in the Messages database names this conversation",
    "ck_1to1": "its one-to-one conversation id names the sender",
    "ck_group_match": "its group id is carried by this conversation alone",
    "holding_lost_group": (
        "its group id is carried by no single conversation, so it is filed with other "
        "messages of that lost group"
    ),
    "holding_sender": (
        "nothing names a conversation, so it is filed with the sender's other unlinked messages"
    ),
}
"""`message.chat_evidence` (migration 0007) in plain words."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_name: str
    source_rowid: int
    merge_mode: str | None
    """`live` (the host's own Messages database) or `seed` (an older or
    recovered copy that only fills gaps); `None` for runs before 0006."""
    read_at: datetime | None
    """When the extraction run that last read this row finished (or
    started, if it never finished)."""


@dataclass(frozen=True, slots=True)
class VersionRecord:
    version_idx: int
    text: str
    edited_at: datetime | None


@dataclass(frozen=True, slots=True)
class AttachmentDetail:
    filename: str | None
    mime_type: str | None
    byte_size: int | None
    sha256: str | None
    state: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MessageDetails:
    message_key: str
    source_guid: str
    sent_at: datetime
    is_from_me: bool
    sender_name: str
    service: str
    raw_handle: str | None
    """`None` when hidden by setting, or when no handle was recorded."""
    raw_handle_service: str | None
    handles_hidden: bool
    is_edited: bool
    date_edited: datetime | None
    versions: tuple[VersionRecord, ...] | None
    """`None` when hidden by setting."""
    deleted_at: datetime | None
    is_unsent: bool
    chat_title: str
    chat_kind: str
    is_holding: bool
    chat_evidence: str
    sources: tuple[SourceRecord, ...]
    attachments: tuple[AttachmentDetail, ...]


def message_details(
    pg: psycopg.Connection,
    message_key: str,
    *,
    index_unsent: bool,
    show_edit_history: bool,
    show_raw_handles: bool,
) -> MessageDetails | None:
    """Everything the panel shows for one message, or `None` when the key
    names no message the page may show."""
    if not is_opaque_key(message_key):
        return None
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT m.message_id, m.message_key, m.source_guid, m.chat_id, m.sent_at, m.is_from_me,
                   m.service::text, m.is_unsent, m.is_edited, m.date_edited, m.deleted_at,
                   m.chat_evidence, m.sender_source_handle_id, p.display_name
            FROM message m LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE m.message_key = %s
            """,
            (message_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        (
            message_id,
            key,
            source_guid,
            chat_id,
            sent_at,
            is_from_me,
            service,
            is_unsent,
            is_edited,
            date_edited,
            deleted_at,
            chat_evidence,
            handle_id,
            display_name,
        ) = row
        if is_unsent and not index_unsent:
            return None
        raw_handle: str | None = None
        raw_handle_service: str | None = None
        if show_raw_handles and handle_id is not None and not is_from_me:
            cur.execute(
                "SELECT raw_value, service::text FROM source_handle WHERE source_handle_id = %s",
                (handle_id,),
            )
            handle = cur.fetchone()
            if handle is not None:
                raw_handle, raw_handle_service = str(handle[0]), str(handle[1])
        versions: tuple[VersionRecord, ...] | None = None
        if show_edit_history:
            cur.execute(
                "SELECT version_idx, text, edited_at FROM message_version "
                "WHERE message_id = %s ORDER BY version_idx",
                (message_id,),
            )
            versions = tuple(VersionRecord(int(i), str(t), e) for i, t, e in cur.fetchall())
        cur.execute(
            """
            SELECT ms.source_name, ms.source_rowid, er.merge_mode,
                   coalesce(er.finished_at, er.started_at)
            FROM message_source ms JOIN extraction_run er ON er.run_id = ms.extraction_run_id
            WHERE ms.message_id = %s
            ORDER BY (er.merge_mode = 'live') DESC NULLS LAST, ms.source_name
            """,
            (message_id,),
        )
        sources = tuple(
            SourceRecord(str(name), int(rowid), mode, read_at)
            for name, rowid, mode, read_at in cur.fetchall()
        )
        cur.execute(
            """
            SELECT a.filename, a.mime_type, a.byte_size, a.sha256, a.state::text,
                   coalesce(array_agg(s.source_name ORDER BY s.source_name)
                            FILTER (WHERE s.source_name IS NOT NULL), '{}')
            FROM message_attachment ma
            JOIN attachment a ON a.attachment_id = ma.attachment_id
            LEFT JOIN attachment_source s ON s.attachment_id = a.attachment_id
            WHERE ma.message_id = %s
            GROUP BY a.attachment_id, ma.ordinal
            ORDER BY ma.ordinal NULLS LAST, a.attachment_id
            """,
            (message_id,),
        )
        attachments = tuple(
            AttachmentDetail(
                filename=filename,
                mime_type=mime,
                byte_size=int(size) if size is not None else None,
                sha256=sha,
                state=str(state),
                sources=tuple(str(s) for s in names),
            )
            for filename, mime, size, sha, state, names in cur.fetchall()
        )
    chat = chat_views(pg, [int(chat_id)])[int(chat_id)]
    return MessageDetails(
        message_key=str(key),
        source_guid=str(source_guid),
        sent_at=sent_at,
        is_from_me=bool(is_from_me),
        sender_name="Me" if is_from_me else (str(display_name) if display_name else "Unknown"),
        service=str(service),
        raw_handle=raw_handle,
        raw_handle_service=raw_handle_service,
        handles_hidden=not show_raw_handles,
        is_edited=bool(is_edited),
        date_edited=date_edited,
        versions=versions,
        deleted_at=deleted_at,
        is_unsent=bool(is_unsent),
        chat_title=chat.title,
        chat_kind=chat.kind,
        is_holding=chat.is_holding,
        chat_evidence=str(chat_evidence),
        sources=sources,
        attachments=attachments,
    )


__all__ = [
    "FILED_BY",
    "AttachmentDetail",
    "MessageDetails",
    "SourceRecord",
    "VersionRecord",
    "message_details",
]
