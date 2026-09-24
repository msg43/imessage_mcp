"""S4 Postgres integration (SPEC §8 S4): the only module in this
package that talks to a live connection. Wires the pure logic in
`sessionize`/`boundaries`/`render`/`hashing` to the schema from
migration 0001 — dirty-chat detection, the incremental-frontier
recompute, and the transactional re-segmentation + outbox emission.

Takes an already-open `psycopg.Connection` and never owns its
lifecycle, per the foundation's DB convention. Honors D1's
`policy.index_unsent` / `policy.index_edit_history` flags and stamps
`seg_config_hash` (D4's freeze mechanism) on every segment it writes.

**How little a re-segmentation run is allowed to touch (2026-09-18).**
A run used to rebuild every session from the earliest change to the end
of the chat, deleting and re-inserting each one. On the live corpus
1,582 genuinely changed messages re-segmented 236,300 messages and
re-wrote 32,473 segments; every re-inserted segment is a new
`segment_id`, so its `segment_embedding` row cascaded away and S6
re-embedded all of them — a multi-hour stage for a few thousand edits.
Two bounds now hold, and both are load-bearing:

1. **The range has an end.** `find_dirty_chats` reports the last
   changed point as well as the first, and the rebuild stops at the
   first persisted session that starts more than one session gap after
   it (`imsg.segment.sessionize.compute_recompute_end`, which carries
   the safety argument).
2. **Inside the range, identical rows are left alone.** A recomputed
   segment that reproduces the stored row exactly — `_segment_is_
   unchanged` says what "exactly" has to mean — is not deleted, not
   re-inserted, emits no `search_index_event`, and keeps its
   `segment_id`, so nothing downstream (`segment_embedding`,
   `export_document.segment_id`) is disturbed.

Neither bound may weaken D4's freeze: `seg_config_hash` is an input to
every `stable_key` *and* is compared on its own, so a config change
matches nothing and rewrites everything.

**Cross-stage dependency:** dirty-chat detection below assumes S2
(edits/retractions) and S3 (identity curation) bump
`message.updated_at` on any change to what a segment renders. Migration
0003's trigger covers every UPDATE of a `message` row, so S2's upserts
and S3's sender repoints are caught by construction. A name change is
not an UPDATE of any message row, though — the rendered text carries
`person.display_name` (header) and `person.short_name` (message lines,
tapback suffixes) — so `rename_person` / `merge_persons` /
`assign_handle` (`imsg.stages.identity._mark_chats_dirty_for_persons`)
bump every message of every chat whose rendered segments name the
persons involved, which makes `find_dirty_chats` report the chat from
its first message and the incremental frontier rebuild it from the
start. Before 2026-09-15 they bumped only `person.updated_at`, which
nothing here watches, and a rename after segmentation was never
re-rendered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imsg.background_gate import BackgroundWorkDeferred, StopCheck
from imsg.config.schema import Config
from imsg.errors import SegmentationError
from imsg.hashing import sha256_text
from imsg.segment.boundaries import BoundaryProvider, segment_session
from imsg.segment.hashing import compute_seg_config_hash, compute_stable_key
from imsg.segment.models import (
    AttachmentSnippet,
    DirtyChatSpan,
    EditVersion,
    MessageForSegmentation,
    PersistedSessionSpan,
    RenderedSegment,
    SegmentationRunReport,
    SegmentDraft,
    Session,
)
from imsg.segment.render import render_segment
from imsg.segment.sessionize import compute_recompute_end, compute_recompute_start, sessionize
from imsg.tokens import estimate_tokens

if TYPE_CHECKING:
    import psycopg

REBUILD_ALL_SENTINEL = datetime.min.replace(tzinfo=UTC)
"""Pass as `earliest_changed_at` to `run_segment_for_chat` to force a
full rebuild of a chat (e.g. after a `segmentation.*`/`policy.*` config
change — `imsg segment --rebuild --chat <id>`, SPEC §8 S4). Sorts
before every real timestamp, so `compute_recompute_start` finds no
sealed session and rebuilds from the beginning."""

_TAPBACK_SYMBOLS = {
    "loved": "♥",
    "liked": "👍",
    "disliked": "👎",
    "laughed": "😂",
    "emphasized": "‼",
    "questioned": "❓",
    "sticker": "🏷",
}


def _tapback_symbol(kind: str) -> str:
    if kind.startswith("emoji:"):
        return kind.split(":", 1)[1]
    return _TAPBACK_SYMBOLS.get(kind, kind)


def _classify_attachment_kind(mime_type: str | None) -> str:
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


@dataclass(frozen=True, slots=True)
class ChatContext:
    chat_id: int
    source_guid: str
    kind: str  # 'dm' | 'group'
    display_name: str | None
    other_participant_display_names: tuple[str, ...]
    """Every chat participant's `display_name` except the owner (SPEC
    §9.1's rendered header never lists the owner in their own "Chat:
    ..." line)."""
    is_unfiled: bool = False
    """A holding chat (`chat.unfiled_key` set, D13): its header says so."""


def fetch_chat_context(conn: psycopg.Connection, chat_id: int) -> ChatContext:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_guid, kind, display_name, unfiled_key IS NOT NULL "
            "FROM chat WHERE chat_id = %s",
            (chat_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SegmentationError(f"chat_id {chat_id} not found")
        source_guid, kind, display_name, is_unfiled = row

        cur.execute(
            """
            SELECT p.display_name
            FROM chat_participant cp
            JOIN person p ON p.person_id = cp.person_id
            WHERE cp.chat_id = %s AND NOT p.is_owner
            ORDER BY p.display_name
            """,
            (chat_id,),
        )
        others = tuple(r[0] for r in cur.fetchall())

    return ChatContext(
        chat_id=chat_id,
        source_guid=source_guid,
        kind=kind,
        display_name=display_name,
        other_participant_display_names=others,
        is_unfiled=bool(is_unfiled),
    )


def fetch_persisted_sessions(
    conn: psycopg.Connection, chat_id: int
) -> list[PersistedSessionSpan]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_id, started_at, ended_at FROM session "
            "WHERE chat_id = %s ORDER BY started_at",
            (chat_id,),
        )
        return [
            PersistedSessionSpan(session_id=sid, started_at=started, ended_at=ended)
            for sid, started, ended in cur.fetchall()
        ]


def find_dirty_chats(
    conn: psycopg.Connection, *, index_unsent: bool
) -> dict[int, DirtyChatSpan]:
    """`{chat_id: DirtyChatSpan}` for every chat with segmentation work
    pending: messages not yet in any segment, or messages whose
    `updated_at` moved past their current segment's `created_at` (edits,
    retractions, identity-merge sender reassignment — see the module
    docstring's cross-stage note).

    The span's `earliest_changed_at` is where the rebuild must start;
    `latest_changed_at` is the last point it must reach, and lets
    `run_segment_for_chat` stop at the first sealed session past it
    instead of rebuilding to the end of the chat.

    The `changed` arm takes `GREATEST(m.sent_at, sess.ended_at)` rather
    than the message's own `sent_at`, so the span always reaches the end
    of the session that *currently* holds the changed message. That
    costs nothing in the ordinary case (an in-place edit sits inside its
    own session), and it removes the end bound's dependence on
    `message.sent_at` never moving: `sent_at` is `Merge.ASSERTED` in
    `imsg.stages.extract`, so a re-extraction from a second source *can*
    rewrite it. Were the span keyed on the new `sent_at` alone, a
    timestamp that moved **backwards** would leave the session still
    holding the message beyond the bound, never rebuilt and permanently
    stale. Reaching the holding session's end closes that.

    **A message that changed chat (D13, 2026-09-23)** makes two chats
    dirty. Extraction moves a message out of a holding chat when stronger
    evidence names its chat (`imsg.stages.unlinked_filing`), and the move
    is an UPDATE, so the message is a `changed` row: the chat it joined is
    reported from its `sent_at`, and the chat whose segment still holds it
    (`s.chat_id`) is reported over that segment's session, so the stale
    segment is rebuilt without it. `run_segment_for_chat` also drops such
    a segment itself before inserting the message anywhere
    (`segment_message` allows each message in one segment only), so the
    order the two chats run in does not matter.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH unsegmented AS (
                SELECT m.chat_id,
                       MIN(m.sent_at) AS earliest,
                       MAX(m.sent_at) AS latest
                FROM message m
                LEFT JOIN segment_message sm ON sm.message_id = m.message_id
                WHERE sm.message_id IS NULL
                  AND m.sender_person_id IS NOT NULL
                  AND (%(index_unsent)s OR NOT m.is_unsent)
                GROUP BY m.chat_id
            ),
            changed_rows AS (
                SELECT m.chat_id, s.chat_id AS segment_chat_id, m.sent_at,
                       sess.started_at AS session_started_at,
                       sess.ended_at AS session_ended_at
                FROM message m
                JOIN segment_message sm ON sm.message_id = m.message_id
                JOIN segment s ON s.segment_id = sm.segment_id
                JOIN session sess ON sess.session_id = s.session_id
                WHERE m.updated_at > s.created_at
            ),
            changed AS (
                SELECT chat_id,
                       MIN(sent_at) AS earliest,
                       MAX(CASE WHEN segment_chat_id = chat_id
                                THEN GREATEST(sent_at, session_ended_at)
                                ELSE sent_at END) AS latest
                FROM changed_rows
                GROUP BY chat_id
            ),
            moved_out AS (
                SELECT segment_chat_id AS chat_id,
                       MIN(session_started_at) AS earliest,
                       MAX(session_ended_at) AS latest
                FROM changed_rows
                WHERE segment_chat_id <> chat_id
                GROUP BY segment_chat_id
            )
            SELECT chat_id, MIN(earliest), MAX(latest) FROM (
                SELECT * FROM unsegmented
                UNION ALL
                SELECT * FROM changed
                UNION ALL
                SELECT * FROM moved_out
            ) combined
            GROUP BY chat_id
            """,
            {"index_unsent": index_unsent},
        )
        return {
            chat_id: DirtyChatSpan(earliest_changed_at=earliest, latest_changed_at=latest)
            for chat_id, earliest, latest in cur.fetchall()
        }


def find_config_stale_chat_ids(
    conn: psycopg.Connection, *, current_seg_config_hash: str
) -> set[int]:
    """Chats with at least one segment whose `seg_config_hash` no longer
    matches the current config — candidates for an explicit
    `--rebuild` (SPEC §8 S4's D4 freeze mechanism)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT chat_id FROM segment WHERE seg_config_hash <> %s",
            (current_seg_config_hash,),
        )
        return {row[0] for row in cur.fetchall()}


def _rows_to_messages(
    cur: psycopg.Cursor,
    chat_id: int,
    rows: list[Any],
    *,
    include_edit_history: bool,
) -> list[MessageForSegmentation]:
    """Shared row->domain-object step for both `_fetch_messages_from`
    (a chat's messages from a timestamp forward) and
    `_fetch_messages_by_id` (an exact, already-known message set — used
    to re-render one segment after attachment enrichment completes,
    SPEC §8 S5b). `rows` is untyped (`Any`) same as every other raw
    psycopg row tuple in this module — see `fetch_chat_context` etc.
    """
    message_ids: list[int] = [r[0] for r in rows]
    attachments_by_message = _fetch_attachments(cur, message_ids)
    tapbacks_by_message = _fetch_tapback_suffixes(cur, message_ids)
    edit_history_by_message = (
        _fetch_edit_history(cur, message_ids) if include_edit_history else {}
    )

    messages: list[MessageForSegmentation] = []
    for (
        message_id,
        source_guid,
        sent_at,
        is_from_me,
        text_original,
        is_unsent,
        is_edited,
        has_attachments,
        sender_person_id,
        short_name,
        deleted_at,
    ) in rows:
        if not is_from_me and sender_person_id is None:
            raise SegmentationError(
                f"message_id {message_id} in chat {chat_id} has no resolved "
                f"sender_person_id — S3's pre-S4 invariant (SPEC §8 S3) should "
                f"have refused to let segmentation run at all"
            )
        sender_short_name = "owner" if is_from_me else short_name
        messages.append(
            MessageForSegmentation(
                message_id=message_id,
                source_guid=source_guid,
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
                is_deleted=deleted_at is not None,
            )
        )
    return messages


_MESSAGE_SELECT_COLUMNS = """
    m.message_id, m.source_guid, m.sent_at, m.is_from_me,
    m.text_original, m.is_unsent, m.is_edited, m.has_attachments,
    m.sender_person_id, p.short_name, m.deleted_at
"""


def _fetch_messages_from(
    conn: psycopg.Connection,
    chat_id: int,
    from_ts: datetime,
    *,
    to_ts: datetime | None = None,
    index_unsent: bool,
    include_edit_history: bool,
) -> list[MessageForSegmentation]:
    """`[from_ts, to_ts)` — `to_ts` is exclusive, because it is the
    `started_at` of the first persisted session the rebuild must leave
    alone (`compute_recompute_end`), and that session's own first
    message must not be pulled into the rebuilt range. `to_ts=None`
    means no upper bound (rebuild to the end of the chat)."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_MESSAGE_SELECT_COLUMNS}
            FROM message m
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE m.chat_id = %(chat_id)s
              AND m.sent_at >= %(from_ts)s
              AND (%(to_ts)s::timestamptz IS NULL OR m.sent_at < %(to_ts)s)
              AND (%(index_unsent)s OR NOT m.is_unsent)
            ORDER BY m.sent_at, m.message_id
            """,
            {
                "chat_id": chat_id,
                "from_ts": from_ts,
                "to_ts": to_ts,
                "index_unsent": index_unsent,
            },
        )
        rows = cur.fetchall()
        return _rows_to_messages(cur, chat_id, rows, include_edit_history=include_edit_history)


def _fetch_messages_by_id(
    conn: psycopg.Connection,
    chat_id: int,
    message_ids: list[int],
    *,
    include_edit_history: bool,
) -> list[MessageForSegmentation]:
    """An exact, already-known set of messages (e.g. a segment's current
    `segment_message` membership) — unlike `_fetch_messages_from`, this
    applies no `policy.index_unsent` filter: the set was already decided
    when the segment was built, and re-rendering must reproduce exactly
    that membership, not re-derive it."""
    if not message_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_MESSAGE_SELECT_COLUMNS}
            FROM message m
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE m.message_id = ANY(%(message_ids)s)
            ORDER BY m.sent_at, m.message_id
            """,
            {"message_ids": message_ids},
        )
        rows = cur.fetchall()
        return _rows_to_messages(cur, chat_id, rows, include_edit_history=include_edit_history)


def _fetch_attachments(
    cur: psycopg.Cursor, message_ids: list[int]
) -> dict[int, list[AttachmentSnippet]]:
    if not message_ids:
        return {}
    cur.execute(
        """
        SELECT ma.message_id, a.attachment_key, a.filename, a.mime_type,
               e.kind, e.text
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
    for message_id, attachment_key, filename, mime_type, enrich_kind, enrich_text in cur.fetchall():
        entry_key = (message_id, attachment_key)
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
            AttachmentSnippet(
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


def _fetch_tapback_suffixes(
    cur: psycopg.Cursor, message_ids: list[int]
) -> dict[int, list[str]]:
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
        result.setdefault(target_message_id, []).append(
            f"({_tapback_symbol(kind)} {sender})"
        )
    return result


def _fetch_edit_history(
    cur: psycopg.Cursor, message_ids: list[int]
) -> dict[int, list[EditVersion]]:
    if not message_ids:
        return {}
    cur.execute(
        """
        SELECT message_id, version_idx, text, edited_at
        FROM message_version
        WHERE message_id = ANY(%s)
        ORDER BY message_id, version_idx
        """,
        (message_ids,),
    )
    result: dict[int, list[EditVersion]] = {}
    for message_id, version_idx, text, edited_at in cur.fetchall():
        result.setdefault(message_id, []).append(
            EditVersion(version_idx=version_idx, text=text, edited_at=edited_at)
        )
    return result


@dataclass(frozen=True, slots=True)
class _StoredSegment:
    """One persisted `segment` row as the reuse check sees it: every
    column `_insert_segments` would write, plus its ordered
    `segment_message` membership and whether any of those messages is
    itself flagged changed."""

    segment_id: int
    seq_in_session: int
    stable_key: str
    started_at: datetime
    ended_at: datetime
    message_count: int
    token_count: int | None
    rendered_sha256: str
    topic_label: str | None
    seg_config_hash: str
    message_ids: tuple[int, ...]
    holds_a_changed_message: bool
    """True when some message in this segment satisfies `updated_at >
    segment.created_at` — exactly `find_dirty_chats`'s `changed`
    predicate. Such a segment is never reused even when it re-renders
    identically, because reuse keeps the old `created_at` and the chat
    would stay dirty forever: every later run would recompute it, skip
    it again, and never clear the flag."""


def _fetch_stored_segments(
    conn: psycopg.Connection, session_ids: list[int]
) -> dict[int, list[_StoredSegment]]:
    """`{session_id: [segments ordered by seq_in_session]}` for the
    sessions inside the rebuild range."""
    if not session_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.session_id, s.segment_id, s.seq_in_session, s.stable_key,
                   s.started_at, s.ended_at, s.message_count, s.token_count,
                   s.rendered_sha256, s.topic_label, s.seg_config_hash,
                   array_agg(sm.message_id ORDER BY m.sent_at, m.message_id),
                   bool_or(m.updated_at > s.created_at)
            FROM segment s
            JOIN segment_message sm ON sm.segment_id = s.segment_id
            JOIN message m ON m.message_id = sm.message_id
            WHERE s.session_id = ANY(%s)
            GROUP BY s.segment_id
            ORDER BY s.session_id, s.seq_in_session
            """,
            (session_ids,),
        )
        stored: dict[int, list[_StoredSegment]] = {sid: [] for sid in session_ids}
        for row in cur.fetchall():
            stored[row[0]].append(
                _StoredSegment(
                    segment_id=row[1],
                    seq_in_session=row[2],
                    stable_key=row[3],
                    started_at=row[4],
                    ended_at=row[5],
                    message_count=row[6],
                    token_count=row[7],
                    rendered_sha256=row[8],
                    topic_label=row[9],
                    seg_config_hash=row[10],
                    message_ids=tuple(row[11]),
                    holds_a_changed_message=bool(row[12]),
                )
            )
    return stored


def _segment_is_unchanged(
    stored: _StoredSegment, recomputed: RenderedSegment, *, seq_in_session: int
) -> bool:
    """**What "unchanged" has to mean.** Every column `_insert_segments`
    would write must already hold the value it would write, and the
    stored `segment_message` membership must be the exact ordered set
    the recompute produced. Anything less leaves a stored row that
    disagrees with what the current config says it should be, which is
    the same defect the rebuild exists to repair.

    Why each part carries its own weight:

    - `stable_key` pins the chat, the **first and last** message guid,
      and `seg_config_hash`. It is the coarse boundary check and,
      because `seg_config_hash` is one of its inputs, it alone is
      already enough to force a full rebuild on a config change.
    - `seg_config_hash` is compared anyway, so the config-change
      guarantee does not silently depend on `compute_stable_key`
      continuing to fold it in.
    - `rendered_sha256` covers everything the renderer emits: bodies,
      short names, the participant header, timestamps, attachment
      snippets, tapbacks, edit history.
    - `message_ids` is the one thing neither of the above can see. The
      key pins only the endpoints and the render is text, so a message
      appearing or vanishing in the **middle** could in principle leave
      both untouched — and `segment_message` is what
      `find_dirty_chats` and `find_segment_ids_for_attachment` join on,
      so a wrong membership is silently wrong forever. Comparing the
      ordered tuple also covers ordering and `message_count`, which are
      checked separately only because they are stored columns.
    - `started_at` / `ended_at` are the span; they follow from the
      endpoints given that `sent_at` does not move, but `sent_at` is
      `Merge.ASSERTED` in extract and *can* be rewritten by a second
      source, so they are compared rather than assumed.
    - `token_count` is derived from the rendered text by
      `imsg.tokens.estimate_tokens`, which is **not** an input to
      `seg_config_hash`. Without this comparison a change to the
      estimator would leave every stored `token_count` stale forever.
    - `seq_in_session` is positional: the recomputed segment list of a
      session is compared to the stored one element by element, so a
      segment only survives if it is still the same segment in the same
      place.
    - `holds_a_changed_message` is not about equality at all — see its
      docstring above. It is what guarantees the run makes progress.
    """
    return (
        not stored.holds_a_changed_message
        and stored.seq_in_session == seq_in_session
        and recomputed.draft.seq_in_session == seq_in_session
        and stored.stable_key == recomputed.stable_key
        and stored.seg_config_hash == recomputed.seg_config_hash
        and stored.rendered_sha256 == recomputed.rendered_sha256
        and stored.started_at == recomputed.draft.started_at
        and stored.ended_at == recomputed.draft.ended_at
        and stored.message_count == recomputed.draft.message_count
        and stored.token_count == recomputed.token_count
        and stored.topic_label == recomputed.draft.topic_label
        and stored.message_ids == tuple(m.message_id for m in recomputed.draft.messages)
    )


@dataclass(frozen=True, slots=True)
class _WritePlan:
    """What a run would do, decided before `dry_run` branches so the
    preview and the real run report the same numbers."""

    sessions_to_delete: tuple[int, ...]
    segments_to_delete: tuple[int, ...]
    """Segments dropped individually out of a session that is being
    kept — disjoint from the segments that cascade off
    `sessions_to_delete` — plus any segment of *another* chat that still
    holds a message now filed in this one (D13: a message moved out of a
    holding chat), which has to go before this chat can place it."""
    reused_session_ids: dict[datetime, int]
    ended_at_updates: tuple[tuple[int, datetime], ...]
    sessions_to_insert: tuple[Session, ...]
    segments_to_insert: dict[datetime, tuple[RenderedSegment, ...]]
    segments_deleted: int
    """Every segment row that disappears, cascades included."""
    segments_written: int
    skipped_unchanged: int


def _plan_writes(
    conn: psycopg.Connection,
    *,
    stale_sessions: list[PersistedSessionSpan],
    recomputed_sessions: list[Session],
    rendered_by_session_start: dict[datetime, list[RenderedSegment]],
    foreign_segment_ids: tuple[int, ...] = (),
) -> _WritePlan:
    """Diff the recomputed sessions/segments against what is already
    stored in the rebuild range.

    `foreign_segment_ids` are other chats' segments that still hold a
    message this chat is about to place (`_foreign_segments_holding`);
    they are deleted with this chat's stale segments.

    A persisted session is matched by `started_at` — `session` has a
    `UNIQUE (chat_id, started_at)`, so within one chat that is a real
    natural key — and its row is **reused** when at least one of its
    segments survives the comparison above. Its `ended_at` is UPDATEd if
    the session grew or shrank (the nightly steady state: new messages
    landing in the still-open tail session), which is a plain column
    write: no `search_index_event`, no `segment_id` churn, so the
    segments that did not change are not re-embedded either.

    A session where *nothing* survives is deleted and re-inserted whole.
    That is also what keeps `session.gap_hours` honest without comparing
    it here: `session_gap_hours` is an input to `seg_config_hash`, which
    is an input to every `stable_key`, so a changed gap invalidates
    every segment of every session, and every session is therefore
    rewritten.
    """
    stored_by_session = _fetch_stored_segments(conn, [s.session_id for s in stale_sessions])
    stale_by_start = {s.started_at: s for s in stale_sessions}
    recomputed_by_start = {s.started_at: s for s in recomputed_sessions}

    sessions_to_delete: list[int] = []
    segments_to_delete: list[int] = []
    reused_session_ids: dict[datetime, int] = {}
    ended_at_updates: list[tuple[int, datetime]] = []
    segments_to_insert: dict[datetime, tuple[RenderedSegment, ...]] = {}
    skipped_unchanged = 0
    cascaded_deletions = 0

    for start, stale in stale_by_start.items():
        stored = stored_by_session.get(stale.session_id, [])
        recomputed = recomputed_by_start.get(start)
        if recomputed is None:
            sessions_to_delete.append(stale.session_id)
            cascaded_deletions += len(stored)
            continue

        rendered = rendered_by_session_start.get(start, [])
        keep: list[bool] = [
            i < len(rendered) and _segment_is_unchanged(st, rendered[i], seq_in_session=i)
            for i, st in enumerate(stored)
        ]
        if not any(keep):
            sessions_to_delete.append(stale.session_id)
            cascaded_deletions += len(stored)
            continue

        reused_session_ids[start] = stale.session_id
        if stale.ended_at != recomputed.ended_at:
            ended_at_updates.append((stale.session_id, recomputed.ended_at))
        segments_to_delete.extend(
            st.segment_id for st, kept in zip(stored, keep, strict=True) if not kept
        )
        skipped_unchanged += sum(keep)
        segments_to_insert[start] = tuple(
            r for i, r in enumerate(rendered) if not (i < len(keep) and keep[i])
        )

    sessions_to_insert = [s for s in recomputed_sessions if s.started_at not in reused_session_ids]
    for s in sessions_to_insert:
        segments_to_insert[s.started_at] = tuple(rendered_by_session_start.get(s.started_at, []))

    segments_to_delete.extend(foreign_segment_ids)

    return _WritePlan(
        sessions_to_delete=tuple(sessions_to_delete),
        segments_to_delete=tuple(segments_to_delete),
        reused_session_ids=reused_session_ids,
        ended_at_updates=tuple(ended_at_updates),
        sessions_to_insert=tuple(sessions_to_insert),
        segments_to_insert=segments_to_insert,
        segments_deleted=cascaded_deletions + len(segments_to_delete),
        segments_written=sum(len(v) for v in segments_to_insert.values()),
        skipped_unchanged=skipped_unchanged,
    )


def _foreign_segments_holding(
    conn: psycopg.Connection, chat_id: int, message_ids: list[int]
) -> tuple[int, ...]:
    """Segments of other chats that still hold one of `message_ids`, all
    of which are filed in `chat_id` now. Only a D13 move out of a holding
    chat makes one (`imsg.stages.unlinked_filing.refile_message`): the
    message's `chat_id` changed while its old segment stood. Dropping that
    segment here, in the same transaction that places the message, is what
    keeps `segment_message`'s one-segment-per-message index from refusing
    the insert when this chat is rebuilt before the one the message left.
    The chat it left is re-segmented on its own (`find_dirty_chats`)."""
    if not message_ids:
        return ()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT sm.segment_id
            FROM segment_message sm
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE sm.message_id = ANY(%(ids)s::bigint[]) AND s.chat_id <> %(chat_id)s
            ORDER BY sm.segment_id
            """,
            {"ids": message_ids, "chat_id": chat_id},
        )
        return tuple(int(row[0]) for row in cur.fetchall())


def _delete_stale_and_emit_delete_events(
    conn: psycopg.Connection, *, session_ids: tuple[int, ...], segment_ids: tuple[int, ...]
) -> int:
    """Emit one `delete` outbox row per segment about to disappear, then
    drop the rows. `session_ids` take their segments with them (ON
    DELETE CASCADE covers segment / segment_message /
    segment_embedding); `segment_ids` are dropped individually out of
    sessions that are being kept, and cascade the same way one level
    down."""
    if not session_ids and not segment_ids:
        return 0
    doomed: list[int] = list(segment_ids)
    with conn.cursor() as cur:
        if session_ids:
            cur.execute(
                "SELECT segment_id FROM segment WHERE session_id = ANY(%s)",
                (list(session_ids),),
            )
            doomed.extend(row[0] for row in cur.fetchall())
        if doomed:
            cur.executemany(
                "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
                "VALUES ('segment', %s, 'delete', NULL)",
                [(sid,) for sid in doomed],
            )
        if segment_ids:
            cur.execute("DELETE FROM segment WHERE segment_id = ANY(%s)", (list(segment_ids),))
        if session_ids:
            cur.execute("DELETE FROM session WHERE session_id = ANY(%s)", (list(session_ids),))
    return len(doomed)


def _insert_sessions(
    conn: psycopg.Connection, sessions: tuple[Session, ...]
) -> dict[datetime, int]:
    ids: dict[datetime, int] = {}
    with conn.cursor() as cur:
        for s in sessions:
            cur.execute(
                "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
                "VALUES (%s, %s, %s, %s) RETURNING session_id",
                (s.chat_id, s.started_at, s.ended_at, s.gap_hours),
            )
            row = cur.fetchone()
            if row is None:  # pragma: no cover - INSERT ... RETURNING always returns a row
                raise SegmentationError("session insert did not return a session_id")
            ids[s.started_at] = row[0]
    return ids


def _update_session_spans(
    conn: psycopg.Connection, ended_at_updates: tuple[tuple[int, datetime], ...]
) -> None:
    """A reused session whose tail grew or shrank. Deliberately not an
    outbox event: `session` is not an indexed entity, and its segments
    carry their own events when they change."""
    if not ended_at_updates:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE session SET ended_at = %s WHERE session_id = %s",
            [(ended_at, session_id) for session_id, ended_at in ended_at_updates],
        )


def _insert_segments(
    conn: psycopg.Connection,
    chat_id: int,
    segments_to_insert: dict[datetime, tuple[RenderedSegment, ...]],
    session_ids: dict[datetime, int],
) -> int:
    written = 0
    with conn.cursor() as cur:
        for session_started_at, rendered in sorted(segments_to_insert.items()):
            session_id = session_ids[session_started_at]
            for r in rendered:
                cur.execute(
                    """
                    INSERT INTO segment (
                        stable_key, chat_id, session_id, seq_in_session,
                        started_at, ended_at, message_count, token_count,
                        rendered_text, rendered_sha256, topic_label, seg_config_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING segment_id
                    """,
                    (
                        r.stable_key,
                        chat_id,
                        session_id,
                        r.draft.seq_in_session,
                        r.draft.started_at,
                        r.draft.ended_at,
                        r.draft.message_count,
                        r.token_count,
                        r.rendered_text,
                        r.rendered_sha256,
                        r.draft.topic_label,
                        r.seg_config_hash,
                    ),
                )
                row = cur.fetchone()
                if row is None:  # pragma: no cover
                    raise SegmentationError("segment insert did not return a segment_id")
                segment_id = row[0]

                cur.executemany(
                    "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
                    [(segment_id, m.message_id) for m in r.draft.messages],
                )
                cur.execute(
                    "INSERT INTO search_index_event "
                    "(entity_kind, entity_id, operation, content_sha256) "
                    "VALUES ('segment', %s, 'upsert', %s)",
                    (segment_id, r.rendered_sha256),
                )
                written += 1
    return written


def refresh_segment_rendering(
    conn: psycopg.Connection, segment_id: int, config: Config
) -> tuple[str, str]:
    """Re-render one segment's text in place, without touching its
    boundaries/membership/`stable_key` (SPEC §8 S5b: attachment
    enrichment completing "re-renders every current parent segment
    reached through `message_attachment`, updates `rendered_sha256`...
    segment boundaries do not change merely because attachment text
    arrived").

    Returns `(rendered_text, rendered_sha256)`. Deliberately does not
    touch `search_index_event` or `segment_embedding` itself — S6
    notices the new `rendered_sha256` no longer matches the stored
    `text_sha256` on its own (that *is* S6's idempotency check) and the
    caller (`imsg.enrich.pipeline`) is responsible for emitting the FTS
    upsert event as part of whatever transaction it's already in.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chat_id, started_at, seq_in_session, topic_label "
            "FROM segment WHERE segment_id = %s",
            (segment_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SegmentationError(f"segment_id {segment_id} not found")
        chat_id, session_started_at, seq_in_session, topic_label = row

        cur.execute(
            "SELECT message_id FROM segment_message WHERE segment_id = %s", (segment_id,)
        )
        message_ids = [r[0] for r in cur.fetchall()]

    chat_ctx = fetch_chat_context(conn, chat_id)
    messages = _fetch_messages_by_id(
        conn, chat_id, message_ids, include_edit_history=config.policy.index_edit_history
    )
    if not messages:
        raise SegmentationError(
            f"segment_id {segment_id} has no messages in segment_message — cannot re-render"
        )

    draft = SegmentDraft(
        session_started_at=session_started_at,
        seq_in_session=seq_in_session,
        messages=tuple(messages),
        topic_label=topic_label,
    )
    text = render_segment(
        draft,
        participants=chat_ctx.other_participant_display_names,
        chat_kind=chat_ctx.kind,
        chat_display_name=chat_ctx.display_name,
        timezone=config.render.timezone,
        attachment_snippet_chars=config.render.attachment_snippet_chars,
        unfiled=chat_ctx.is_unfiled,
    )
    rendered_sha = sha256_text(text)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE segment SET rendered_text = %s, rendered_sha256 = %s, token_count = %s "
            "WHERE segment_id = %s",
            (text, rendered_sha, estimate_tokens(text), segment_id),
        )
    return text, rendered_sha


def find_segment_ids_for_attachment(conn: psycopg.Connection, attachment_id: int) -> list[int]:
    """Every *current* parent segment reached through
    `message_attachment` -> `message` -> `segment_message` (SPEC §8
    S5b) — used by `imsg.enrich.pipeline` to know which segments need
    `refresh_segment_rendering` after an attachment's enrichment text
    changes."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT sm.segment_id
            FROM message_attachment ma
            JOIN segment_message sm ON sm.message_id = ma.message_id
            WHERE ma.attachment_id = %s
            """,
            (attachment_id,),
        )
        return [row[0] for row in cur.fetchall()]


def run_segment_for_chat(
    conn: psycopg.Connection,
    chat_id: int,
    config: Config,
    boundary_provider: BoundaryProvider,
    boundary_prompt_bytes: bytes,
    *,
    earliest_changed_at: datetime,
    latest_changed_at: datetime | None = None,
    dry_run: bool = False,
) -> SegmentationRunReport:
    """Re-segment one chat over the range the incremental frontier says
    is affected, in one Postgres transaction. Pass
    `earliest_changed_at=REBUILD_ALL_SENTINEL` to force a full rebuild
    (config change).

    The range is `[compute_recompute_start(...), compute_recompute_end(
    ...))`. `latest_changed_at` — `find_dirty_chats`'s
    `DirtyChatSpan.latest_changed_at` — supplies the end bound; `None`
    (the default, and what `REBUILD_ALL_SENTINEL` forces) means no
    bound, i.e. rebuild to the end of the chat, which is what every run
    did before the bound existed. See `compute_recompute_end` for why
    the first persisted session starting more than one session gap
    after the last change, and everything after it, is provably
    unaffected.

    Inside the range, a recomputed segment that reproduces the stored
    row exactly (`_segment_is_unchanged`) is left alone: no DELETE, no
    INSERT, no `search_index_event`, and the `segment_id` survives — so
    its `segment_embedding` row is not cascaded away and S6 does not
    re-embed it. Those are counted in `skipped_unchanged`.

    `dry_run=True` (SPEC §8: "takes --dry-run where writes leave the
    machine") runs the exact same sessionize/boundary-detection/render
    computation *and the same diff against what is stored* —
    `boundary_provider` still runs, since it is a scoring call, not a
    write — but skips the `with conn.transaction():` write block
    entirely. Every count it reports comes from the same `_WritePlan`
    the real path executes, so the preview and the run agree by
    construction.
    """
    seg_cfg = config.segmentation
    policy = config.policy

    current_hash = compute_seg_config_hash(
        session_gap_hours=seg_cfg.session_gap_hours,
        topical_min_messages=seg_cfg.topical_min_messages,
        max_messages=seg_cfg.max_messages,
        max_tokens=seg_cfg.max_tokens,
        boundary_model=seg_cfg.boundary_model,
        boundary_revision=seg_cfg.boundary_revision,
        boundary_prompt_bytes=boundary_prompt_bytes,
        index_unsent=policy.index_unsent,
        index_edit_history=policy.index_edit_history,
    )

    chat_ctx = fetch_chat_context(conn, chat_id)
    existing_sessions = fetch_persisted_sessions(conn, chat_id)
    recompute_start = compute_recompute_start(
        existing_sessions, earliest_changed_at, seg_cfg.session_gap_hours
    )

    recompute_end = (
        None
        if latest_changed_at is None or earliest_changed_at == REBUILD_ALL_SENTINEL
        else compute_recompute_end(existing_sessions, latest_changed_at, seg_cfg.session_gap_hours)
    )

    stale_sessions = [
        s
        for s in existing_sessions
        if s.started_at >= recompute_start
        and (recompute_end is None or s.started_at < recompute_end)
    ]

    messages = _fetch_messages_from(
        conn,
        chat_id,
        recompute_start,
        to_ts=recompute_end,
        index_unsent=policy.index_unsent,
        include_edit_history=policy.index_edit_history,
    )

    sessions = sessionize(messages, chat_id=chat_id, session_gap_hours=seg_cfg.session_gap_hours)

    rendered_by_session_start: dict[datetime, list[RenderedSegment]] = {}
    fallback_count = 0
    for session in sessions:
        drafts, used_fallback = segment_session(
            session,
            topical_min_messages=seg_cfg.topical_min_messages,
            max_messages=seg_cfg.max_messages,
            max_tokens=seg_cfg.max_tokens,
            boundary_provider=boundary_provider,
        )
        if used_fallback:
            fallback_count += 1
        segment_list: list[RenderedSegment] = []
        for draft in drafts:
            text = render_segment(
                draft,
                participants=chat_ctx.other_participant_display_names,
                chat_kind=chat_ctx.kind,
                chat_display_name=chat_ctx.display_name,
                timezone=config.render.timezone,
                attachment_snippet_chars=config.render.attachment_snippet_chars,
                unfiled=chat_ctx.is_unfiled,
            )
            rendered_sha = sha256_text(text)
            stable_key = compute_stable_key(
                chat_source_guid=chat_ctx.source_guid,
                first_message_guid=draft.messages[0].source_guid,
                last_message_guid=draft.messages[-1].source_guid,
                seg_config_hash=current_hash,
            )
            segment_list.append(
                RenderedSegment(
                    draft=draft,
                    rendered_text=text,
                    rendered_sha256=rendered_sha,
                    token_count=estimate_tokens(text),
                    seg_config_hash=current_hash,
                    stable_key=stable_key,
                )
            )
        rendered_by_session_start[session.started_at] = segment_list

    plan = _plan_writes(
        conn,
        stale_sessions=stale_sessions,
        recomputed_sessions=sessions,
        rendered_by_session_start=rendered_by_session_start,
        foreign_segment_ids=_foreign_segments_holding(
            conn, chat_id, [m.message_id for m in messages]
        ),
    )

    if dry_run:
        return SegmentationRunReport(
            chat_id=chat_id,
            sessions_written=len(plan.sessions_to_insert),
            segments_written=plan.segments_written,
            segments_deleted=plan.segments_deleted,
            fallback_sessions=fallback_count,
            skipped_unchanged=plan.skipped_unchanged,
            dry_run=True,
            notes=("dry run — no sessions/segments were written or deleted",),
        )

    written = 0
    deleted = 0
    if (
        plan.sessions_to_delete
        or plan.segments_to_delete
        or plan.sessions_to_insert
        or plan.ended_at_updates
        or plan.segments_written
    ):
        # Deletes first, inside the same transaction: `segment_message`
        # has a UNIQUE index on `message_id` alone, so a message moving
        # between segments would collide if the new row went in before
        # the old one came out.
        with conn.transaction():
            deleted = _delete_stale_and_emit_delete_events(
                conn,
                session_ids=plan.sessions_to_delete,
                segment_ids=plan.segments_to_delete,
            )
            _update_session_spans(conn, plan.ended_at_updates)
            session_ids = {
                **plan.reused_session_ids,
                **_insert_sessions(conn, plan.sessions_to_insert),
            }
            written = _insert_segments(conn, chat_id, plan.segments_to_insert, session_ids)

    return SegmentationRunReport(
        chat_id=chat_id,
        sessions_written=len(plan.sessions_to_insert),
        segments_written=written,
        segments_deleted=deleted,
        fallback_sessions=fallback_count,
        skipped_unchanged=plan.skipped_unchanged,
    )


def run_segment(
    conn: psycopg.Connection,
    config: Config,
    boundary_provider: BoundaryProvider,
    boundary_prompt_bytes: bytes,
    *,
    chat_ids: set[int] | None = None,
    dry_run: bool = False,
    stop_check: StopCheck | None = None,
) -> list[SegmentationRunReport]:
    """Top-level incremental entry point (SPEC §8 S7's S4 step): find
    every dirty chat and re-segment each. `chat_ids`, if given,
    restricts the run to that subset (still driven by dirtiness, not a
    forced rebuild — use `run_segment_for_chat` with
    `REBUILD_ALL_SENTINEL` directly for `--rebuild`).

    `dry_run` is threaded straight through to each
    `run_segment_for_chat` call (SPEC §8: "takes --dry-run where
    writes leave the machine").

    `stop_check` (`imsg.background_gate`) is asked before each chat. A
    chat is one transaction, so on a reason every chat already done is
    committed and the rest stay dirty for the next run; the run raises
    `BackgroundWorkDeferred` with the finished chats' reports."""
    dirty = find_dirty_chats(conn, index_unsent=config.policy.index_unsent)
    if chat_ids is not None:
        dirty = {cid: ts for cid, ts in dirty.items() if cid in chat_ids}

    reports: list[SegmentationRunReport] = []
    for chat_id, span in dirty.items():
        reason = stop_check() if stop_check is not None else None
        if reason is not None:
            raise BackgroundWorkDeferred(reason, partial=reports)
        reports.append(
            run_segment_for_chat(
                conn,
                chat_id,
                config,
                boundary_provider,
                boundary_prompt_bytes,
                earliest_changed_at=span.earliest_changed_at,
                latest_changed_at=span.latest_changed_at,
                dry_run=dry_run,
            )
        )
    return reports


__all__ = [
    "REBUILD_ALL_SENTINEL",
    "ChatContext",
    "fetch_chat_context",
    "fetch_persisted_sessions",
    "find_config_stale_chat_ids",
    "find_dirty_chats",
    "find_segment_ids_for_attachment",
    "refresh_segment_rendering",
    "run_segment",
    "run_segment_for_chat",
]
