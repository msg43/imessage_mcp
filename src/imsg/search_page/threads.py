"""Conversation data for the page: chat headers, message windows that
scroll both ways, and the messages that make each search hit readable.

Everything is keyed by the opaque keys the MCP surfaces already use
(`thread_key`, `message_key`, `attachment_key`: sha256 hex, `imsg.keys`),
never by database ids or source GUIDs. Messages come back as
`MessageView`s carrying what the thread view shows: sender, time, text,
reactions (tapbacks), edited / unsent / deleted markers (D1, D13),
attachments and link previews.

Visibility follows the local surface's rules under full scope: deleted
messages show, labelled (D13); unsent messages show only when
`policy.index_unsent` is on (D1), exactly as `get_conversation` does.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import psycopg

OPAQUE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_WINDOW = 200


def is_opaque_key(value: str | None) -> bool:
    return bool(value) and OPAQUE_KEY_RE.match(value or "") is not None


_TAPBACK_SYMBOLS = {
    "loved": "♥",
    "liked": "\U0001f44d",
    "disliked": "\U0001f44e",
    "laughed": "\U0001f602",
    "emphasized": "‼",
    "questioned": "❓",
    "sticker": "\U0001f3f7",
}


def tapback_symbol(kind: str) -> str:
    if kind.startswith("emoji:"):
        return kind.split(":", 1)[1] or "?"
    return _TAPBACK_SYMBOLS.get(kind, kind)


def attachment_kind(mime_type: str | None, filename: str | None) -> str:
    """image | video | audio | pdf | other — the display kind."""
    mime = (mime_type or "").lower()
    name = (filename or "").lower()
    if mime == "application/pdf" or name.endswith(".pdf"):
        return "pdf"
    if mime.startswith("image/") or name.endswith((".heic", ".heif", ".jpg", ".jpeg", ".png", ".gif")):
        return "image"
    if mime.startswith("video/") or name.endswith((".mov", ".mp4", ".m4v")):
        return "video"
    if mime.startswith("audio/") or name.endswith((".caf", ".m4a", ".mp3", ".amr", ".wav", ".aac")):
        return "audio"
    return "other"


@dataclass(frozen=True, slots=True)
class ChatView:
    chat_id: int
    thread_key: str
    kind: str
    title: str
    participants: tuple[str, ...]
    """Display names of everyone but the owner, in name order."""
    is_holding: bool
    """A holding chat (D13): messages no real chat could be named for."""
    service: str


@dataclass(slots=True)
class AttachmentView:
    attachment_id: int
    attachment_key: str
    filename: str | None
    mime_type: str | None
    kind: str
    byte_size: int | None
    available: bool
    state: str
    is_sticker: bool
    caption: str | None = None
    transcript: str | None = None
    text_excerpt: str | None = None


@dataclass(slots=True)
class MessageView:
    message_id: int
    message_key: str
    source_guid: str
    chat_id: int
    sent_at: datetime
    is_from_me: bool
    sender_name: str
    text: str | None
    is_unsent: bool
    is_edited: bool
    is_deleted: bool
    has_attachments: bool
    reply_to_guid: str | None = None
    attachments: list[AttachmentView] = field(default_factory=list)
    reactions: list[tuple[str, str]] = field(default_factory=list)
    reply_to_key: str | None = None
    reply_to_text: str | None = None
    link_previews: list[tuple[str, str | None, str | None]] = field(default_factory=list)


# --------------------------------------------------------------------------
# chats
# --------------------------------------------------------------------------


def chat_views(pg: psycopg.Connection, chat_ids: Iterable[int]) -> dict[int, ChatView]:
    ids = sorted({int(c) for c in chat_ids})
    if not ids:
        return {}
    with pg.cursor() as cur:
        cur.execute(
            "SELECT chat_id, thread_key, kind, display_name, unfiled_key IS NOT NULL, service "
            "FROM chat WHERE chat_id = ANY(%(ids)s::bigint[])",
            {"ids": ids},
        )
        base = cur.fetchall()
        cur.execute(
            """
            SELECT cp.chat_id, p.display_name
            FROM chat_participant cp JOIN person p ON p.person_id = cp.person_id
            WHERE cp.chat_id = ANY(%(ids)s::bigint[]) AND NOT p.is_owner
            ORDER BY cp.chat_id, p.display_name
            """,
            {"ids": ids},
        )
        people: dict[int, list[str]] = {}
        for chat_id, display_name in cur.fetchall():
            people.setdefault(int(chat_id), []).append(str(display_name))
    out: dict[int, ChatView] = {}
    for chat_id, thread_key, kind, display_name, is_holding, service in base:
        participants = tuple(people.get(int(chat_id), ()))
        if display_name:
            title = str(display_name)
        elif participants:
            title = ", ".join(participants[:4]) + (
                f" and {len(participants) - 4} more" if len(participants) > 4 else ""
            )
        else:
            title = "Conversation"
        out[int(chat_id)] = ChatView(
            chat_id=int(chat_id),
            thread_key=str(thread_key),
            kind=str(kind),
            title=title,
            participants=participants,
            is_holding=bool(is_holding),
            service=str(service),
        )
    return out


def chat_by_thread_key(pg: psycopg.Connection, thread_key: str) -> ChatView | None:
    if not is_opaque_key(thread_key):
        return None
    with pg.cursor() as cur:
        cur.execute("SELECT chat_id FROM chat WHERE thread_key = %s", (thread_key,))
        row = cur.fetchone()
    if row is None:
        return None
    return chat_views(pg, [int(row[0])]).get(int(row[0]))


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------

_MESSAGE_COLUMNS = """
    m.message_id, m.message_key, m.source_guid, m.chat_id, m.sent_at, m.is_from_me,
    m.text_original, m.is_unsent, m.is_edited, m.deleted_at, m.has_attachments,
    m.reply_to_guid, p.display_name
"""


def _row_to_view(row: Sequence[Any]) -> MessageView:
    (
        message_id,
        message_key,
        source_guid,
        chat_id,
        sent_at,
        is_from_me,
        text,
        is_unsent,
        is_edited,
        deleted_at,
        has_attachments,
        reply_to_guid,
        display_name,
    ) = row
    return MessageView(
        message_id=int(message_id),
        message_key=str(message_key),
        source_guid=str(source_guid),
        chat_id=int(chat_id),
        sent_at=sent_at,
        is_from_me=bool(is_from_me),
        sender_name="Me" if is_from_me else (str(display_name) if display_name else "Unknown"),
        text=text,
        is_unsent=bool(is_unsent),
        is_edited=bool(is_edited),
        is_deleted=deleted_at is not None,
        has_attachments=bool(has_attachments),
        reply_to_guid=reply_to_guid,
    )


def messages_by_id(pg: psycopg.Connection, message_ids: Iterable[int]) -> dict[int, MessageView]:
    ids = sorted({int(m) for m in message_ids})
    if not ids:
        return {}
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM message m LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE m.message_id = ANY(%(ids)s::bigint[])
            """,
            {"ids": ids},
        )
        return {int(r[0]): _row_to_view(r) for r in cur.fetchall()}


def segment_messages(
    pg: psycopg.Connection, segment_ids: Iterable[int], *, index_unsent: bool
) -> dict[int, list[MessageView]]:
    """Each segment's messages in order, without attachments yet."""
    ids = sorted({int(s) for s in segment_ids})
    if not ids:
        return {}
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT sm.segment_id, {_MESSAGE_COLUMNS}
            FROM segment_message sm
            JOIN message m ON m.message_id = sm.message_id
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE sm.segment_id = ANY(%(ids)s::bigint[])
              AND (%(unsent)s OR NOT m.is_unsent)
            ORDER BY sm.segment_id, m.sent_at, m.message_id
            """,
            {"ids": ids, "unsent": index_unsent},
        )
        out: dict[int, list[MessageView]] = {}
        for segment_id, *row in cur.fetchall():
            out.setdefault(int(segment_id), []).append(_row_to_view(row))
    return out


def decorate(pg: psycopg.Connection, messages: Sequence[MessageView]) -> None:
    """Attach attachments, reactions, reply targets and link previews to
    `messages`, in a handful of queries for the whole batch."""
    if not messages:
        return
    by_id = {m.message_id: m for m in messages}
    ids = sorted(by_id)
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT ma.message_id, a.attachment_id, a.attachment_key, a.filename, a.mime_type,
                   a.byte_size, a.state::text, a.sha256, a.is_sticker
            FROM message_attachment ma JOIN attachment a ON a.attachment_id = ma.attachment_id
            WHERE ma.message_id = ANY(%(ids)s::bigint[])
            ORDER BY ma.message_id, ma.ordinal NULLS LAST, a.attachment_id
            """,
            {"ids": ids},
        )
        attachments: dict[int, AttachmentView] = {}
        for message_id, att_id, key, filename, mime, size, state, sha, sticker in cur.fetchall():
            view = AttachmentView(
                attachment_id=int(att_id),
                attachment_key=str(key),
                filename=filename,
                mime_type=mime,
                kind=attachment_kind(mime, filename),
                byte_size=int(size) if size is not None else None,
                available=(state == "materialized" and bool(sha)),
                state=str(state),
                is_sticker=bool(sticker),
            )
            by_id[int(message_id)].attachments.append(view)
            attachments[int(att_id)] = view

        if attachments:
            cur.execute(
                """
                SELECT attachment_id, kind::text, text FROM enrichment
                WHERE attachment_id = ANY(%(ids)s::bigint[]) AND state = 'done' AND text IS NOT NULL
                """,
                {"ids": sorted(attachments)},
            )
            for att_id, kind, text in cur.fetchall():
                view = attachments[int(att_id)]
                if kind == "caption":
                    view.caption = str(text)
                elif kind == "transcript":
                    view.transcript = str(text)
                elif kind in ("pdf_text", "doc_text", "ocr", "frame_ocr") and not view.text_excerpt:
                    view.text_excerpt = str(text)[:2000]

        cur.execute(
            """
            SELECT t.target_message_id, t.kind, t.is_from_me, p.display_name
            FROM tapback t LEFT JOIN person p ON p.person_id = t.sender_person_id
            WHERE t.target_message_id = ANY(%(ids)s::bigint[]) AND NOT t.removed
            ORDER BY t.acted_at NULLS LAST, t.tapback_id
            """,
            {"ids": ids},
        )
        for target, kind, from_me, name in cur.fetchall():
            who = "Me" if from_me else (str(name) if name else "Unknown")
            by_id[int(target)].reactions.append((tapback_symbol(str(kind)), who))

        reply_guids = sorted({m.reply_to_guid for m in messages if m.reply_to_guid})
        if reply_guids:
            cur.execute(
                "SELECT source_guid, message_key, left(text_original, 120) FROM message "
                "WHERE source_guid = ANY(%(g)s::text[])",
                {"g": reply_guids},
            )
            targets = {str(g): (str(k), t) for g, k, t in cur.fetchall()}
            for message in messages:
                target = targets.get(message.reply_to_guid or "")
                if target is not None:
                    message.reply_to_key, message.reply_to_text = target

        cur.execute(
            "SELECT message_id, url, title, site_name FROM link_preview "
            "WHERE message_id = ANY(%(ids)s::bigint[]) ORDER BY message_id, url",
            {"ids": ids},
        )
        for message_id, url, title, site in cur.fetchall():
            by_id[int(message_id)].link_previews.append((str(url), title, site))


def attachment_chunk_text(
    pg: psycopg.Connection, attachment_ids: Iterable[int], *, max_chars: int = 20000
) -> dict[int, str]:
    ids = sorted({int(a) for a in attachment_ids})
    if not ids:
        return {}
    with pg.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, text FROM attachment_chunk "
            "WHERE attachment_id = ANY(%(ids)s::bigint[]) ORDER BY attachment_id, kind, seq",
            {"ids": ids},
        )
        out: dict[int, list[str]] = {}
        for att_id, text in cur.fetchall():
            parts = out.setdefault(int(att_id), [])
            if sum(len(p) for p in parts) < max_chars:
                parts.append(str(text))
    return {k: " ".join(v)[:max_chars] for k, v in out.items()}


# --------------------------------------------------------------------------
# the thread view: windows that scroll both ways
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Window:
    messages: list[MessageView]
    has_older: bool
    has_newer: bool
    anchor_key: str | None


def _message_position(
    pg: psycopg.Connection, chat_id: int, message_key: str
) -> tuple[datetime, int] | None:
    if not is_opaque_key(message_key):
        return None
    with pg.cursor() as cur:
        cur.execute(
            "SELECT sent_at, message_id FROM message WHERE message_key = %s AND chat_id = %s",
            (message_key, chat_id),
        )
        row = cur.fetchone()
    return (row[0], int(row[1])) if row else None


def _fetch_side(
    pg: psycopg.Connection,
    chat_id: int,
    position: tuple[datetime, int] | None,
    *,
    older: bool,
    limit: int,
    inclusive: bool,
    index_unsent: bool,
) -> list[MessageView]:
    """Up to `limit` messages before (`older`) or after `position`, in
    chronological order. With no position: the newest (`older`) or oldest
    messages of the chat."""
    if older:
        comparison = "<=" if inclusive else "<"
        order = "DESC"
    else:
        comparison = ">=" if inclusive else ">"
        order = "ASC"
    params: dict[str, object] = {"chat": chat_id, "limit": limit, "unsent": index_unsent}
    where = ["m.chat_id = %(chat)s", "(%(unsent)s OR NOT m.is_unsent)"]
    if position is not None:
        where.append(f"(m.sent_at, m.message_id) {comparison} (%(at)s, %(id)s)")
        params["at"], params["id"] = position
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM message m LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE {" AND ".join(where)}
            ORDER BY m.sent_at {order}, m.message_id {order}
            LIMIT %(limit)s
            """,
            params,
        )
        rows = [_row_to_view(r) for r in cur.fetchall()]
    if older:
        rows.reverse()
    return rows


def thread_window(
    pg: psycopg.Connection,
    chat: ChatView,
    *,
    anchor_key: str | None,
    before: int,
    after: int,
    index_unsent: bool,
) -> Window:
    """Messages around `anchor_key` (or the newest messages when there is
    no anchor), plus whether more exist in each direction."""
    before = max(1, min(before, MAX_WINDOW))
    after = max(1, min(after, MAX_WINDOW))
    position = _message_position(pg, chat.chat_id, anchor_key) if anchor_key else None
    if position is None:
        older_rows = _fetch_side(
            pg, chat.chat_id, None, older=True, limit=before + 1, inclusive=True,
            index_unsent=index_unsent,
        )
        has_older = len(older_rows) > before
        messages = older_rows[-before:] if has_older else older_rows
        decorate(pg, messages)
        return Window(messages=messages, has_older=has_older, has_newer=False, anchor_key=None)
    older_rows = _fetch_side(
        pg, chat.chat_id, position, older=True, limit=before + 1, inclusive=False,
        index_unsent=index_unsent,
    )
    newer_rows = _fetch_side(
        pg, chat.chat_id, position, older=False, limit=after + 2, inclusive=True,
        index_unsent=index_unsent,
    )
    has_older = len(older_rows) > before
    has_newer = len(newer_rows) > after + 1
    messages = (older_rows[-before:] if has_older else older_rows) + newer_rows[: after + 1]
    decorate(pg, messages)
    return Window(messages=messages, has_older=has_older, has_newer=has_newer, anchor_key=anchor_key)


def thread_page(
    pg: psycopg.Connection,
    chat: ChatView,
    *,
    cursor_key: str,
    older: bool,
    limit: int,
    index_unsent: bool,
) -> Window:
    """The next `limit` messages beyond `cursor_key`, for infinite scroll."""
    limit = max(1, min(limit, MAX_WINDOW))
    position = _message_position(pg, chat.chat_id, cursor_key)
    if position is None:
        return Window(messages=[], has_older=False, has_newer=False, anchor_key=None)
    rows = _fetch_side(
        pg, chat.chat_id, position, older=older, limit=limit + 1, inclusive=False,
        index_unsent=index_unsent,
    )
    more = len(rows) > limit
    if more:
        rows = rows[-limit:] if older else rows[:limit]
    decorate(pg, rows)
    return Window(
        messages=rows,
        has_older=more if older else False,
        has_newer=more if not older else False,
        anchor_key=None,
    )


__all__ = [
    "MAX_WINDOW",
    "AttachmentView",
    "ChatView",
    "MessageView",
    "Window",
    "attachment_chunk_text",
    "attachment_kind",
    "chat_by_thread_key",
    "chat_views",
    "decorate",
    "is_opaque_key",
    "messages_by_id",
    "segment_messages",
    "tapback_symbol",
    "thread_page",
    "thread_window",
]
