"""Browse the corpus without a search word: a timeline and a media grid.

- **Timeline**: every message across all conversations, in time order,
  for a chosen day or range, with a count per day. Paged 200 at a time by
  `(sent_at, message_id)`, which the `message_sent_idx` index serves.
- **Media**: photos, videos, voice notes and other audio, PDFs and other
  files, newest first, filtered by person, date and type, with a count
  per type. Paged 60 at a time by `(sent_at, message_id, attachment_id)`.

Both honour the page's visibility rules (an unsent message shows only
under `policy.index_unsent`) and both open a conversation at the chosen
message. People filters: `people` keeps conversations every listed
person is in (the search page's People filter); `sender` keeps what one
person sent (`me` for the owner). Dates are local days in
`render.timezone`, as on the search page.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from imsg.search_page.threads import (
    MESSAGE_COLUMNS,
    AttachmentView,
    MessageView,
    attachment_kind,
    decorate,
    row_to_view,
)

if TYPE_CHECKING:
    import psycopg

TIMELINE_PAGE = 200
MEDIA_PAGE = 60
MEDIA_KINDS = ("image", "video", "audio", "pdf", "other")
MEDIA_KIND_LABELS = {
    "image": "Photos",
    "video": "Videos",
    "audio": "Voice notes and audio",
    "pdf": "PDFs",
    "other": "Other files",
}


def attachment_kind_sql(alias: str = "a") -> str:
    """`imsg.search_page.threads.attachment_kind` as a SQL expression over
    an `attachment` row, so the media grid can filter and count by kind in
    Postgres. The tests check the two agree."""
    mime = f"lower(coalesce({alias}.mime_type, ''))"
    name = f"lower(coalesce({alias}.filename, ''))"
    pdf = f"({mime} = 'application/pdf' OR {name} LIKE '%%.pdf')"
    image = f"({mime} LIKE 'image/%%' OR {name} ~ '\\.(heic|heif|jpg|jpeg|png|gif)$')"
    video = f"({mime} LIKE 'video/%%' OR {name} ~ '\\.(mov|mp4|m4v)$')"
    audio = f"({mime} LIKE 'audio/%%' OR {name} ~ '\\.(caf|m4a|mp3|amr|wav|aac)$')"
    return (
        f"CASE WHEN {pdf} THEN 'pdf' WHEN {image} THEN 'image' WHEN {video} THEN 'video' "
        f"WHEN {audio} THEN 'audio' ELSE 'other' END"
    )


@dataclass(frozen=True, slots=True)
class BrowseFilters:
    """Resolved filters: person ids, a sender, and a local-day range."""

    people: tuple[int, ...] = ()
    sender: int | None = None
    sender_is_owner: bool = False
    start: datetime | None = None
    """Inclusive, local midnight."""
    end: datetime | None = None
    """Exclusive, local midnight after the last day."""


def day_bounds(first: date, last: date, timezone: str) -> tuple[datetime, datetime]:
    zone = ZoneInfo(timezone)
    start = datetime(first.year, first.month, first.day, tzinfo=zone)
    after = last + timedelta(days=1)
    return start, datetime(after.year, after.month, after.day, tzinfo=zone)


def _where(filters: BrowseFilters, *, index_unsent: bool) -> tuple[list[str], dict[str, object]]:
    clauses = ["(%(b_unsent)s OR NOT m.is_unsent)"]
    params: dict[str, object] = {"b_unsent": index_unsent}
    if filters.start is not None:
        clauses.append("m.sent_at >= %(b_start)s")
        params["b_start"] = filters.start
    if filters.end is not None:
        clauses.append("m.sent_at < %(b_end)s")
        params["b_end"] = filters.end
    if filters.sender_is_owner:
        clauses.append("m.is_from_me")
    elif filters.sender is not None:
        clauses.append("m.sender_person_id = %(b_sender)s AND NOT m.is_from_me")
        params["b_sender"] = filters.sender
    for i, person_id in enumerate(filters.people):
        clauses.append(
            f"EXISTS (SELECT 1 FROM chat_participant __bp_{i} "
            f"WHERE __bp_{i}.chat_id = m.chat_id AND __bp_{i}.person_id = %(b_p{i})s)"
        )
        params[f"b_p{i}"] = person_id
    return clauses, params


def latest_day(
    pg: psycopg.Connection, filters: BrowseFilters, *, index_unsent: bool, timezone: str
) -> date | None:
    """The local day of the latest message under `filters` (dates
    ignored): the timeline's default day."""
    clauses, params = _where(
        BrowseFilters(people=filters.people, sender=filters.sender, sender_is_owner=filters.sender_is_owner),
        index_unsent=index_unsent,
    )
    with pg.cursor() as cur:
        cur.execute(f"SELECT max(m.sent_at) FROM message m WHERE {' AND '.join(clauses)}", params)
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    latest: datetime = row[0]
    return latest.astimezone(ZoneInfo(timezone)).date()


@dataclass(frozen=True, slots=True)
class TimelinePage:
    messages: list[MessageView]
    next_cursor: str | None


def encode_cursor(at: datetime, *ids: int) -> str:
    return ".".join([str(int(at.timestamp() * 1_000_000)), *(str(i) for i in ids)])


def decode_cursor(raw: str | None, parts: int) -> tuple[datetime, tuple[int, ...]] | None:
    """`None` for an absent or malformed cursor."""
    if not raw:
        return None
    pieces = raw.split(".")
    if len(pieces) != parts + 1 or not all(p.isdigit() and len(p) <= 20 for p in pieces):
        return None
    micros = int(pieces[0])
    at = datetime.fromtimestamp(micros // 1_000_000, tz=ZoneInfo("UTC")).replace(
        microsecond=micros % 1_000_000
    )
    return at, tuple(int(p) for p in pieces[1:])


def timeline_page(
    pg: psycopg.Connection,
    filters: BrowseFilters,
    *,
    index_unsent: bool,
    cursor: str | None,
    limit: int = TIMELINE_PAGE,
) -> TimelinePage:
    clauses, params = _where(filters, index_unsent=index_unsent)
    position = decode_cursor(cursor, 1)
    if position is not None:
        clauses.append("(m.sent_at, m.message_id) > (%(b_at)s, %(b_id)s)")
        params["b_at"], params["b_id"] = position[0], position[1][0]
    params["b_limit"] = limit + 1
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT {MESSAGE_COLUMNS}
            FROM message m LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE {' AND '.join(clauses)}
            ORDER BY m.sent_at, m.message_id
            LIMIT %(b_limit)s
            """,
            params,
        )
        rows = [row_to_view(r) for r in cur.fetchall()]
    more = len(rows) > limit
    rows = rows[:limit]
    decorate(pg, rows)
    next_cursor = encode_cursor(rows[-1].sent_at, rows[-1].message_id) if more and rows else None
    return TimelinePage(messages=rows, next_cursor=next_cursor)


def day_counts(
    pg: psycopg.Connection, filters: BrowseFilters, *, index_unsent: bool, timezone: str
) -> list[tuple[date, int]]:
    clauses, params = _where(filters, index_unsent=index_unsent)
    params["b_zone"] = timezone
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT (m.sent_at AT TIME ZONE %(b_zone)s)::date AS day, count(*)
            FROM message m WHERE {' AND '.join(clauses)}
            GROUP BY 1 ORDER BY 1
            """,
            params,
        )
        return [(d, int(n)) for d, n in cur.fetchall()]


def conversation_count(pg: psycopg.Connection, filters: BrowseFilters, *, index_unsent: bool) -> int:
    clauses, params = _where(filters, index_unsent=index_unsent)
    with pg.cursor() as cur:
        cur.execute(f"SELECT count(DISTINCT m.chat_id) FROM message m WHERE {' AND '.join(clauses)}", params)
        row = cur.fetchone()
    return int(row[0]) if row is not None else 0


@dataclass(slots=True)
class MediaItem:
    attachment: AttachmentView
    message_key: str
    message_id: int
    chat_id: int
    sent_at: datetime
    sender_name: str


@dataclass(frozen=True, slots=True)
class MediaPage:
    items: list[MediaItem]
    next_cursor: str | None


def _media_where(
    filters: BrowseFilters, kind: str | None, *, index_unsent: bool
) -> tuple[list[str], dict[str, object]]:
    clauses, params = _where(filters, index_unsent=index_unsent)
    clauses.append("m.has_attachments AND NOT a.is_sticker")
    if kind is not None:
        clauses.append(f"({attachment_kind_sql('a')}) = %(b_kind)s")
        params["b_kind"] = kind
    return clauses, params


_MEDIA_FROM = """
    FROM message m
    JOIN message_attachment ma ON ma.message_id = m.message_id
    JOIN attachment a ON a.attachment_id = ma.attachment_id
    LEFT JOIN person p ON p.person_id = m.sender_person_id
"""


def media_page(
    pg: psycopg.Connection,
    filters: BrowseFilters,
    *,
    kind: str | None,
    index_unsent: bool,
    cursor: str | None,
    limit: int = MEDIA_PAGE,
) -> MediaPage:
    clauses, params = _media_where(filters, kind, index_unsent=index_unsent)
    position = decode_cursor(cursor, 2)
    if position is not None:
        clauses.append("(m.sent_at, m.message_id, a.attachment_id) < (%(b_at)s, %(b_mid)s, %(b_aid)s)")
        params["b_at"] = position[0]
        params["b_mid"], params["b_aid"] = position[1]
    params["b_limit"] = limit + 1
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT a.attachment_id, a.attachment_key, a.filename, a.mime_type, a.byte_size,
                   a.state::text, a.sha256, a.is_sticker,
                   m.message_key, m.message_id, m.chat_id, m.sent_at, m.is_from_me, p.display_name
            {_MEDIA_FROM}
            WHERE {' AND '.join(clauses)}
            ORDER BY m.sent_at DESC, m.message_id DESC, a.attachment_id DESC
            LIMIT %(b_limit)s
            """,
            params,
        )
        rows = cur.fetchall()
    more = len(rows) > limit
    items: list[MediaItem] = []
    for (att_id, key, filename, mime, size, state, sha, sticker, mkey, mid, chat_id, sent_at, from_me, name) in rows[:limit]:
        items.append(
            MediaItem(
                attachment=AttachmentView(
                    attachment_id=int(att_id),
                    attachment_key=str(key),
                    filename=filename,
                    mime_type=mime,
                    kind=attachment_kind(mime, filename),
                    byte_size=int(size) if size is not None else None,
                    available=(state == "materialized" and bool(sha)),
                    state=str(state),
                    is_sticker=bool(sticker),
                ),
                message_key=str(mkey),
                message_id=int(mid),
                chat_id=int(chat_id),
                sent_at=sent_at,
                sender_name="Me" if from_me else (str(name) if name else "Unknown"),
            )
        )
    last = items[-1] if items else None
    next_cursor = (
        encode_cursor(last.sent_at, last.message_id, last.attachment.attachment_id)
        if more and last is not None
        else None
    )
    return MediaPage(items=items, next_cursor=next_cursor)


def media_counts(
    pg: psycopg.Connection, filters: BrowseFilters, *, index_unsent: bool
) -> dict[str, int]:
    clauses, params = _media_where(filters, None, index_unsent=index_unsent)
    with pg.cursor() as cur:
        cur.execute(
            f"SELECT {attachment_kind_sql('a')} AS kind, count(*) {_MEDIA_FROM} "
            f"WHERE {' AND '.join(clauses)} GROUP BY 1",
            params,
        )
        found = {str(k): int(n) for k, n in cur.fetchall()}
    return {kind: found.get(kind, 0) for kind in MEDIA_KINDS}


__all__ = [
    "MEDIA_KINDS",
    "MEDIA_KIND_LABELS",
    "MEDIA_PAGE",
    "TIMELINE_PAGE",
    "BrowseFilters",
    "MediaItem",
    "MediaPage",
    "TimelinePage",
    "attachment_kind_sql",
    "conversation_count",
    "day_bounds",
    "day_counts",
    "decode_cursor",
    "encode_cursor",
    "latest_day",
    "media_counts",
    "media_page",
    "timeline_page",
]
