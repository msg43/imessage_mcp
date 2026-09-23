"""Segment-result and conversation-window fetching for the retrieval
service (SPEC §9.1, §10.2 `search_messages`/`get_conversation`).

Deliberately self-contained rather than reusing
`imsg.segment.pipeline`'s private row-fetch helpers
(`_fetch_messages_from`, `_fetch_attachments`, ...): those are
module-private, chat/session-shaped, and owned by the segmentation
build; `get_conversation`'s access pattern (an arbitrary anchor
timestamp, possibly not aligned to any segment boundary, with a
before/after window) is different enough that copying the handful of
SQL statements here — reusing the same *domain* dataclasses
(`imsg.segment.models`) and the same per-message renderer
(`imsg.segment.render.render_message_line`) — was judged lower-risk
than reaching into another package's private functions. Flagged as a
small, acceptable duplication in the build report.

**Under `allowlist` scope nothing here returns stored text.** A
conversation window hides unsent messages and edit history whatever
`policy.*` says, hides messages in Apple's "Recently Deleted" (D13), and
shows an attachment's content only where the request's
`imsg.retrieval.access.AttachmentGate` allows it. `render_segment_texts`
re-renders search results under the same rules, because
`segment.rendered_text` was rendered with every attachment's text, every
deleted message, and `policy.*` as it stood at segmentation time. Under
`full` scope a deleted message is shown, labelled `[deleted]`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from imsg.export.eligibility import exportable_message_sql
from imsg.retrieval.access import AttachmentGate, RequestScope, attachment_gate
from imsg.retrieval.errors import InvalidArgumentError, NotFoundError
from imsg.segment.models import (
    AttachmentSnippet,
    EditVersion,
    MessageForSegmentation,
    SegmentDraft,
)
from imsg.segment.render import render_message_line, render_segment

if TYPE_CHECKING:
    import psycopg

THREAD_NOT_FOUND_MESSAGE = "no thread found for the given thread_id"
"""The one `NOT_FOUND` message `get_conversation` gives, whether the
identifier matches nothing or matches a thread the scope denies. The
public surface returns an error's text verbatim, so two different
messages would tell a caller which threads exist (SPEC §10.2's
existence-oracle rule)."""


def _classify_attachment_kind(mime_type: str | None) -> str:
    """Mirrors `imsg.segment.pipeline._classify_attachment_kind`
    exactly (mime_type -> the display-label kind `AttachmentSnippet`
    and `render_message_line` expect) — deliberately re-implemented
    rather than importing that module-private function; see this
    module's docstring."""
    if not mime_type:
        return "other"
    if mime_type == "application/pdf":
        return "pdf"
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("video/"):
        return "video"
    return "other"


# --------------------------------------------------------------------------
# search_messages result rows
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentSummary:
    segment_id: int
    segment_key: str
    thread_key: str
    chat_kind: str
    chat_display_name: str | None
    people: tuple[str, ...]
    started_at: datetime
    ended_at: datetime
    message_count: int
    has_attachments: bool
    text: str


def fetch_segment_summaries(
    conn: psycopg.Connection, segment_ids: list[int]
) -> dict[int, SegmentSummary]:
    """One row per id in `segment_ids` that still exists, keyed by
    `segment_id` — callers reorder by their own ranking, this makes no
    ordering promise."""
    if not segment_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.segment_id, s.stable_key, s.chat_id, c.thread_key, c.kind,
                   c.display_name, s.started_at, s.ended_at, s.message_count,
                   s.rendered_text,
                   EXISTS (
                       SELECT 1 FROM segment_message sm
                       JOIN message m ON m.message_id = sm.message_id
                       WHERE sm.segment_id = s.segment_id AND m.has_attachments
                   ) AS has_attachments
            FROM segment s
            JOIN chat c ON c.chat_id = s.chat_id
            WHERE s.segment_id = ANY(%(ids)s)
            """,
            {"ids": segment_ids},
        )
        rows = cur.fetchall()

        chat_ids = list({r[2] for r in rows})
        people_by_chat: dict[int, list[str]] = {}
        if chat_ids:
            cur.execute(
                """
                SELECT cp.chat_id, p.short_name
                FROM chat_participant cp
                JOIN person p ON p.person_id = cp.person_id
                WHERE cp.chat_id = ANY(%(chat_ids)s)
                ORDER BY p.short_name
                """,
                {"chat_ids": chat_ids},
            )
            for chat_id, short_name in cur.fetchall():
                people_by_chat.setdefault(chat_id, []).append(short_name)

    out: dict[int, SegmentSummary] = {}
    for (
        segment_id,
        stable_key,
        chat_id,
        thread_key,
        kind,
        display_name,
        started_at,
        ended_at,
        message_count,
        rendered_text,
        has_attachments,
    ) in rows:
        out[segment_id] = SegmentSummary(
            segment_id=segment_id,
            segment_key=stable_key,
            thread_key=thread_key,
            chat_kind=kind,
            chat_display_name=display_name,
            people=tuple(people_by_chat.get(chat_id, ())),
            started_at=started_at,
            ended_at=ended_at,
            message_count=message_count,
            has_attachments=bool(has_attachments),
            text=rendered_text,
        )
    return out


# --------------------------------------------------------------------------
# get_conversation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThreadResolution:
    chat_id: int
    thread_key: str
    default_anchor: datetime | None
    """`None` when `thread_id` resolved via a bare `thread_key` with no
    associated segment — `resolve_anchor` falls back to the chat's most
    recent segment in that case (see its docstring)."""


def resolve_thread(conn: psycopg.Connection, thread_id: str) -> ThreadResolution:
    """`thread_id` is a `segment_key` (`segment.stable_key`) or a
    `thread_key` (`chat.thread_key`) — SPEC §10.2 `get_conversation`
    accepts either. Raises `NotFoundError` if neither matches."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.chat_id, c.thread_key, s.started_at "
            "FROM segment s JOIN chat c ON c.chat_id = s.chat_id "
            "WHERE s.stable_key = %s",
            (thread_id,),
        )
        row = cur.fetchone()
        if row is not None:
            return ThreadResolution(chat_id=row[0], thread_key=row[1], default_anchor=row[2])

        cur.execute("SELECT chat_id, thread_key FROM chat WHERE thread_key = %s", (thread_id,))
        row = cur.fetchone()
        if row is not None:
            return ThreadResolution(chat_id=row[0], thread_key=row[1], default_anchor=None)

    raise NotFoundError(THREAD_NOT_FOUND_MESSAGE)


def _latest_segment_start(conn: psycopg.Connection, chat_id: int) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT started_at FROM segment WHERE chat_id = %s ORDER BY started_at DESC LIMIT 1",
            (chat_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def resolve_anchor(
    conn: psycopg.Connection, resolution: ThreadResolution, anchor: str | None
) -> datetime:
    """Resolve `get_conversation`'s `anchor` argument (SPEC §10.2: "an
    opaque `message_key`, or an ISO-8601 timestamp within the thread")
    to a concrete point in time to window around.

    Detection is unambiguous: `message_key` is a 64-character sha256
    hex digest (`imsg.keys`), which never parses as `datetime.
    fromisoformat`, so an ISO-8601-parseable string is always the
    timestamp form and anything else is looked up as a message key.

    Default (no `anchor` given): the resolved segment's start (SPEC
    §10.2: "Default anchor: segment start") when `thread_id` was a
    `segment_key`; when it was a bare `thread_key` (no specific
    segment), this build's judgment call is the chat's *most recent*
    segment start — showing the tail of the conversation by default
    reads as more useful than its very first message ever, and the
    spec does not define a default for that case."""
    if anchor is None:
        if resolution.default_anchor is not None:
            return resolution.default_anchor
        latest = _latest_segment_start(conn, resolution.chat_id)
        if latest is not None:
            return latest
        raise NotFoundError(f"thread {resolution.thread_key!r} has no segments to anchor on")

    try:
        return datetime.fromisoformat(anchor)
    except ValueError:
        pass

    with conn.cursor() as cur:
        cur.execute(
            "SELECT sent_at FROM message WHERE message_key = %s AND chat_id = %s",
            (anchor, resolution.chat_id),
        )
        row = cur.fetchone()
    if row is None:
        raise InvalidArgumentError(
            f"anchor {anchor!r} is neither a valid ISO-8601 timestamp nor a known "
            f"message_key within this thread"
        )
    sent_at: datetime = row[0]
    return sent_at


_MESSAGE_COLUMNS = """
    m.message_id, m.message_key, m.source_guid, m.sent_at, m.is_from_me,
    m.text_original, m.is_unsent, m.is_edited, m.has_attachments,
    m.sender_person_id, p.short_name, m.deleted_at
"""


def _rows_to_messages(
    cur: psycopg.Cursor,
    chat_id: int,
    rows: Sequence[tuple[Any, ...]],
    *,
    include_edit_history: bool,
    gate: AttachmentGate,
    segment_by_message: dict[int, int],
) -> list[MessageForSegmentation]:
    message_ids: list[int] = [row[0] for row in rows]
    attachments_by_message = _fetch_attachments(
        cur, message_ids, gate=gate, segment_by_message=segment_by_message
    )
    tapbacks_by_message = _fetch_tapback_suffixes(cur, message_ids)
    edit_history_by_message = _fetch_edit_history(cur, message_ids) if include_edit_history else {}

    messages: list[MessageForSegmentation] = []
    for row in rows:
        message_id: int = row[0]
        message_key: str = row[1]
        source_guid: str = row[2]
        sent_at: datetime = row[3]
        is_from_me: bool = row[4]
        text_original: str | None = row[5]
        is_unsent: bool = row[6]
        is_edited: bool = row[7]
        has_attachments: bool = row[8]
        short_name: str | None = row[10]
        is_deleted = row[11] is not None
        sender_short_name = "owner" if is_from_me else (short_name or "unknown")
        messages.append(
            MessageForSegmentation(
                message_id=message_id,
                source_guid=message_key or source_guid,
                chat_id=chat_id,
                sent_at=sent_at,
                is_from_me=is_from_me,
                sender_short_name=sender_short_name,
                text=text_original,
                is_unsent=is_unsent,
                is_edited=is_edited,
                has_attachments=has_attachments,
                attachments=tuple(attachments_by_message.get(message_id, ())),
                tapback_suffixes=tuple(tapbacks_by_message.get(message_id, ())),
                edit_history=tuple(edit_history_by_message.get(message_id, ())),
                is_deleted=is_deleted,
            )
        )
    return messages


def _fetch_attachments(
    cur: psycopg.Cursor,
    message_ids: list[int],
    *,
    gate: AttachmentGate,
    segment_by_message: dict[int, int],
) -> dict[int, list[AttachmentSnippet]]:
    """Each message's attachments in display order. An attachment `gate`
    denies in its message's segment becomes
    `AttachmentSnippet.withheld_placeholder()` before any of its fields
    are read into a snippet."""
    if not message_ids:
        return {}
    cur.execute(
        """
        SELECT ma.message_id, a.attachment_key, a.filename, a.mime_type, e.kind, e.text,
               a.attachment_id
        FROM message_attachment ma
        JOIN attachment a ON a.attachment_id = ma.attachment_id
        LEFT JOIN enrichment e ON e.attachment_id = a.attachment_id AND e.state = 'done'
        WHERE ma.message_id = ANY(%s)
        ORDER BY ma.message_id, ma.ordinal
        """,
        (message_ids,),
    )
    by_key: dict[tuple[int, str], dict[str, object]] = {}
    order: dict[int, list[str]] = {}
    withheld: set[tuple[int, str]] = set()
    for (
        message_id,
        attachment_key,
        filename,
        mime_type,
        enrich_kind,
        enrich_text,
        attachment_id,
    ) in cur.fetchall():
        entry_key = (message_id, attachment_key)
        if entry_key in withheld:
            continue
        if entry_key not in by_key and not gate.allows(
            segment_by_message.get(message_id), int(attachment_id)
        ):
            withheld.add(entry_key)
            order.setdefault(message_id, []).append(attachment_key)
            continue
        if entry_key not in by_key:
            by_key[entry_key] = {
                "filename": filename,
                "kind": _classify_attachment_kind(mime_type),
                "pdf_text": None,
                "caption": None,
                "ocr_text": None,
                "transcript": None,
                "document_text": None,
            }
            order.setdefault(message_id, []).append(attachment_key)
        if enrich_kind == "pdf_text":
            by_key[entry_key]["pdf_text"] = enrich_text
        elif enrich_kind == "caption":
            by_key[entry_key]["caption"] = enrich_text
        elif enrich_kind in ("ocr", "frame_ocr"):
            by_key[entry_key]["ocr_text"] = enrich_text
        elif enrich_kind == "transcript":
            by_key[entry_key]["transcript"] = enrich_text
        elif enrich_kind == "doc_text":
            by_key[entry_key]["document_text"] = enrich_text

    result: dict[int, list[AttachmentSnippet]] = {}
    for message_id, keys in order.items():
        result[message_id] = [
            AttachmentSnippet.withheld_placeholder()
            if (message_id, key) in withheld
            else AttachmentSnippet(
                attachment_key=key,
                kind=str(by_key[(message_id, key)]["kind"]),
                filename=by_key[(message_id, key)]["filename"],  # type: ignore[arg-type]
                caption=by_key[(message_id, key)]["caption"],  # type: ignore[arg-type]
                ocr_text=by_key[(message_id, key)]["ocr_text"],  # type: ignore[arg-type]
                transcript=by_key[(message_id, key)]["transcript"],  # type: ignore[arg-type]
                pdf_text=by_key[(message_id, key)]["pdf_text"],  # type: ignore[arg-type]
                document_text=by_key[(message_id, key)]["document_text"],  # type: ignore[arg-type]
            )
            for key in keys
        ]
    return result


_TAPBACK_SYMBOLS = {
    "loved": "♥",
    "liked": "\U0001F44D",
    "disliked": "\U0001F44E",
    "laughed": "\U0001F602",
    "emphasized": "‼",
    "questioned": "❓",
    "sticker": "\U0001F3F7",
}


def _tapback_symbol(kind: str) -> str:
    if kind.startswith("emoji:"):
        return kind.split(":", 1)[1]
    return _TAPBACK_SYMBOLS.get(kind, kind)


def _fetch_tapback_suffixes(cur: psycopg.Cursor, message_ids: list[int]) -> dict[int, list[str]]:
    if not message_ids:
        return {}
    cur.execute(
        """
        SELECT t.target_message_id, t.kind, t.is_from_me, p.short_name
        FROM tapback t
        LEFT JOIN person p ON p.person_id = t.sender_person_id
        WHERE t.target_message_id = ANY(%s) AND NOT t.removed
        ORDER BY t.acted_at NULLS LAST
        """,
        (message_ids,),
    )
    result: dict[int, list[str]] = {}
    for target_message_id, kind, is_from_me, short_name in cur.fetchall():
        sender = "owner" if is_from_me else (short_name or "unknown")
        result.setdefault(target_message_id, []).append(f"({_tapback_symbol(kind)} {sender})")
    return result


def _fetch_edit_history(cur: psycopg.Cursor, message_ids: list[int]) -> dict[int, list[EditVersion]]:
    if not message_ids:
        return {}
    cur.execute(
        "SELECT message_id, version_idx, text, edited_at FROM message_version "
        "WHERE message_id = ANY(%s) ORDER BY message_id, version_idx",
        (message_ids,),
    )
    result: dict[int, list[EditVersion]] = {}
    for message_id, version_idx, text, edited_at in cur.fetchall():
        result.setdefault(message_id, []).append(
            EditVersion(version_idx=version_idx, text=text, edited_at=edited_at)
        )
    return result


def fetch_conversation_window(
    conn: psycopg.Connection,
    resolution: ThreadResolution,
    anchor_dt: datetime,
    window: int,
    *,
    scope: RequestScope,
    index_unsent: bool,
    include_edit_history: bool,
    timezone: str,
    attachment_snippet_chars: int,
) -> list[dict[str, object]]:
    """`window` messages on each side of `anchor_dt`, plus the message
    exactly at the anchor itself (`2 * window + 1` total in the
    common case) — SPEC §10.2 `get_conversation`: "messages on each
    side of the anchor", rendered per-message (SPEC §9.1 format) with
    opaque keys. The "at-or-after" query below fetches `window + 1`
    rows precisely so the anchor message occupies its own slot instead
    of silently consuming one of the `window` after-anchor slots.

    `index_unsent` / `include_edit_history` are the `policy.*` values;
    `scope` decides whether they apply (never under `allowlist`), whether
    deleted messages show (labelled, under `full` only) and which
    attachments show their content. The caller has already authorized
    `resolution.chat_id` under `scope`."""
    index_unsent = scope.shows_unsent(index_unsent)
    include_edit_history = scope.shows_edit_history(include_edit_history)
    show_deleted = scope.shows_deleted
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM message m
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE m.chat_id = %(chat_id)s AND m.sent_at < %(anchor)s
              AND (%(index_unsent)s OR NOT m.is_unsent)
              AND (%(show_deleted)s OR m.deleted_at IS NULL)
            ORDER BY m.sent_at DESC, m.message_id DESC
            LIMIT %(window)s
            """,
            {
                "chat_id": resolution.chat_id,
                "anchor": anchor_dt,
                "index_unsent": index_unsent,
                "show_deleted": show_deleted,
                "window": window,
            },
        )
        before_rows = list(reversed(cur.fetchall()))

        cur.execute(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM message m
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE m.chat_id = %(chat_id)s AND m.sent_at >= %(anchor)s
              AND (%(index_unsent)s OR NOT m.is_unsent)
              AND (%(show_deleted)s OR m.deleted_at IS NULL)
            ORDER BY m.sent_at ASC, m.message_id ASC
            LIMIT %(limit)s
            """,
            {
                "chat_id": resolution.chat_id,
                "anchor": anchor_dt,
                "index_unsent": index_unsent,
                "show_deleted": show_deleted,
                "limit": window + 1,  # +1 for the anchor message itself
            },
        )
        after_rows = cur.fetchall()

        all_rows = before_rows + after_rows
        message_ids = [int(row[0]) for row in all_rows]
        segment_by_message = _segment_by_message(cur, message_ids) if not scope.is_full else {}
        gate = attachment_gate(conn, scope, segment_by_message.values())
        messages = _rows_to_messages(
            cur,
            resolution.chat_id,
            all_rows,
            include_edit_history=include_edit_history,
            gate=gate,
            segment_by_message=segment_by_message,
        )

    tz = ZoneInfo(timezone)
    out: list[dict[str, object]] = []
    for row, message in zip(all_rows, messages, strict=True):
        message_key: str = row[1]
        line = render_message_line(
            message, timezone=timezone, attachment_snippet_chars=attachment_snippet_chars
        )
        out.append(
            {
                "message_key": message_key,
                "sent_at": message.sent_at.astimezone(tz).isoformat(),
                "sender_short_name": message.sender_short_name,
                "is_from_me": message.is_from_me,
                "is_unsent": message.is_unsent,
                "is_edited": message.is_edited,
                "is_deleted": message.is_deleted,
                "text": line,
                "untrusted_content": True,
            }
        )
    return out


def _segment_by_message(cur: psycopg.Cursor, message_ids: list[int]) -> dict[int, int]:
    """Each message's segment; a message not yet segmented is absent."""
    if not message_ids:
        return {}
    cur.execute(
        "SELECT message_id, segment_id FROM segment_message "
        "WHERE message_id = ANY(%(ids)s::bigint[])",
        {"ids": message_ids},
    )
    return {int(message_id): int(segment_id) for message_id, segment_id in cur.fetchall()}


@dataclass(frozen=True, slots=True)
class _ChatHeader:
    kind: str
    display_name: str | None
    other_participant_display_names: tuple[str, ...]
    unfiled: bool = False


def _chat_headers(cur: psycopg.Cursor, chat_ids: list[int]) -> dict[int, _ChatHeader]:
    """What `render_segment`'s header needs, for many chats in two
    queries. The participant list mirrors `imsg.segment.pipeline.
    fetch_chat_context` (non-owner participants' display names, ordered by
    display name) — `tests/test_retrieval_allowlist_parity_integration.py`
    checks a re-render against the pipeline's own rendering."""
    if not chat_ids:
        return {}
    cur.execute(
        "SELECT chat_id, kind, display_name, unfiled_key IS NOT NULL FROM chat "
        "WHERE chat_id = ANY(%(ids)s::bigint[])",
        {"ids": chat_ids},
    )
    base = {
        int(chat_id): (str(kind), display_name, bool(unfiled))
        for chat_id, kind, display_name, unfiled in cur.fetchall()
    }
    cur.execute(
        """
        SELECT cp.chat_id, p.display_name
        FROM chat_participant cp
        JOIN person p ON p.person_id = cp.person_id
        WHERE cp.chat_id = ANY(%(ids)s::bigint[]) AND NOT p.is_owner
        ORDER BY cp.chat_id, p.display_name
        """,
        {"ids": chat_ids},
    )
    others: dict[int, list[str]] = {}
    for chat_id, display_name in cur.fetchall():
        others.setdefault(int(chat_id), []).append(str(display_name))
    return {
        chat_id: _ChatHeader(
            kind=kind,
            display_name=display_name,
            other_participant_display_names=tuple(others.get(chat_id, ())),
            unfiled=unfiled,
        )
        for chat_id, (kind, display_name, unfiled) in base.items()
    }


def render_segment_texts(
    conn: psycopg.Connection,
    scope: RequestScope,
    segment_ids: list[int],
    *,
    timezone: str,
    attachment_snippet_chars: int,
) -> dict[int, str]:
    """Each segment's §9.1 text rendered now, from its rows, under
    `scope`: only segments whose chat `scope` admits, each attachment
    through the request's `AttachmentGate`, and never an unsent message,
    a deleted one or an edit history — this is the renderer for scopes
    that withhold content, so it applies D1 and D13's exclusions
    (`exportable_message_sql`) unconditionally rather than reading
    `policy.*`. Used in place of the stored `segment.rendered_text`
    whenever `scope.withholds_ineligible_content` (module docstring). A
    segment with no message left to show is absent from the result, as
    export plans no document for it."""
    if not segment_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.segment_id, s.chat_id, s.seq_in_session, se.started_at
            FROM segment s
            JOIN session se ON se.session_id = s.session_id
            WHERE s.segment_id = ANY(%(ids)s::bigint[])
            """,
            {"ids": segment_ids},
        )
        segment_rows = [
            (int(segment_id), int(chat_id), int(seq), started_at)
            for segment_id, chat_id, seq, started_at in cur.fetchall()
            if scope.admits_chat(int(chat_id))
        ]
        if not segment_rows:
            return {}
        admitted = [row[0] for row in segment_rows]
        cur.execute(
            f"""
            SELECT sm.segment_id, {_MESSAGE_COLUMNS}
            FROM segment_message sm
            JOIN message m ON m.message_id = sm.message_id
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE sm.segment_id = ANY(%(ids)s::bigint[])
              AND {exportable_message_sql('m')}
            ORDER BY sm.segment_id, m.sent_at, m.message_id
            """,
            {"ids": admitted},
        )
        rows_by_segment: dict[int, list[tuple[Any, ...]]] = {}
        segment_by_message: dict[int, int] = {}
        for segment_id, *row in cur.fetchall():
            rows_by_segment.setdefault(int(segment_id), []).append(tuple(row))
            segment_by_message[int(row[0])] = int(segment_id)

        gate = attachment_gate(conn, scope, admitted)
        headers = _chat_headers(cur, sorted({row[1] for row in segment_rows}))
        texts: dict[int, str] = {}
        for segment_id, chat_id, seq_in_session, session_started_at in segment_rows:
            rows = rows_by_segment.get(segment_id)
            header = headers.get(chat_id)
            if not rows or header is None:
                continue
            messages = _rows_to_messages(
                cur,
                chat_id,
                rows,
                include_edit_history=False,
                gate=gate,
                segment_by_message=segment_by_message,
            )
            draft = SegmentDraft(
                session_started_at=session_started_at,
                seq_in_session=seq_in_session,
                messages=tuple(messages),
            )
            texts[segment_id] = render_segment(
                draft,
                participants=header.other_participant_display_names,
                chat_kind=header.kind,
                chat_display_name=header.display_name,
                timezone=timezone,
                attachment_snippet_chars=attachment_snippet_chars,
                unfiled=header.unfiled,
            )
    return texts


__all__ = [
    "THREAD_NOT_FOUND_MESSAGE",
    "SegmentSummary",
    "ThreadResolution",
    "fetch_conversation_window",
    "fetch_segment_summaries",
    "render_segment_texts",
    "resolve_anchor",
    "resolve_thread",
]
