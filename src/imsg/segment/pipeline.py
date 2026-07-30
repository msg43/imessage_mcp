"""S4 Postgres integration (SPEC §8 S4): the only module in this
package that talks to a live connection. Wires the pure logic in
`sessionize`/`boundaries`/`render`/`hashing` to the schema from
migration 0001 — dirty-chat detection, the incremental-frontier
recompute, and the transactional re-segmentation + outbox emission.

Takes an already-open `psycopg.Connection` and never owns its
lifecycle, per the foundation's DB convention. Honors D1's
`policy.index_unsent` / `policy.index_edit_history` flags and stamps
`seg_config_hash` (D4's freeze mechanism) on every segment it writes.

**Cross-stage dependency (flagged for downstream agents):** dirty-chat
detection below assumes S2 (edits/retractions) and S3 (identity merges)
bump `message.updated_at` on any content-relevant change. Nothing in
migration 0001 does this automatically (no trigger) — if a stage
mutates `message` without setting `updated_at = now()`, the affected
chat will not be picked up by `find_dirty_chats` until an explicit
`--rebuild`. Worth confirming against S2/S3's actual UPDATE statements
once they land.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imsg.config.schema import Config
from imsg.errors import SegmentationError
from imsg.hashing import sha256_text
from imsg.segment.boundaries import BoundaryProvider, segment_session
from imsg.segment.hashing import compute_seg_config_hash, compute_stable_key
from imsg.segment.models import (
    AttachmentSnippet,
    EditVersion,
    MessageForSegmentation,
    PersistedSessionSpan,
    RenderedSegment,
    SegmentationRunReport,
    SegmentDraft,
    Session,
)
from imsg.segment.render import render_segment
from imsg.segment.sessionize import compute_recompute_start, sessionize
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


def fetch_chat_context(conn: psycopg.Connection, chat_id: int) -> ChatContext:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_guid, kind, display_name FROM chat WHERE chat_id = %s",
            (chat_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SegmentationError(f"chat_id {chat_id} not found")
        source_guid, kind, display_name = row

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
) -> dict[int, datetime]:
    """`{chat_id: earliest_changed_at}` for every chat with segmentation
    work pending: messages not yet in any segment, or messages whose
    `updated_at` moved past their current segment's `created_at` (edits,
    retractions, identity-merge sender reassignment — see the module
    docstring's cross-stage note)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH unsegmented AS (
                SELECT m.chat_id, MIN(m.sent_at) AS earliest
                FROM message m
                LEFT JOIN segment_message sm ON sm.message_id = m.message_id
                WHERE sm.message_id IS NULL
                  AND m.sender_person_id IS NOT NULL
                  AND (%(index_unsent)s OR NOT m.is_unsent)
                GROUP BY m.chat_id
            ),
            changed AS (
                SELECT m.chat_id, MIN(m.sent_at) AS earliest
                FROM message m
                JOIN segment_message sm ON sm.message_id = m.message_id
                JOIN segment s ON s.segment_id = sm.segment_id
                WHERE m.updated_at > s.created_at
                GROUP BY m.chat_id
            )
            SELECT chat_id, MIN(earliest) FROM (
                SELECT * FROM unsegmented
                UNION ALL
                SELECT * FROM changed
            ) combined
            GROUP BY chat_id
            """,
            {"index_unsent": index_unsent},
        )
        return dict(cur.fetchall())


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
            )
        )
    return messages


_MESSAGE_SELECT_COLUMNS = """
    m.message_id, m.source_guid, m.sent_at, m.is_from_me,
    m.text_original, m.is_unsent, m.is_edited, m.has_attachments,
    m.sender_person_id, p.short_name
"""


def _fetch_messages_from(
    conn: psycopg.Connection,
    chat_id: int,
    from_ts: datetime,
    *,
    index_unsent: bool,
    include_edit_history: bool,
) -> list[MessageForSegmentation]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_MESSAGE_SELECT_COLUMNS}
            FROM message m
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE m.chat_id = %(chat_id)s
              AND m.sent_at >= %(from_ts)s
              AND (%(index_unsent)s OR NOT m.is_unsent)
            ORDER BY m.sent_at, m.message_id
            """,
            {"chat_id": chat_id, "from_ts": from_ts, "index_unsent": index_unsent},
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


def _delete_stale_and_emit_delete_events(
    conn: psycopg.Connection, stale_session_ids: list[int]
) -> int:
    if not stale_session_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT segment_id FROM segment WHERE session_id = ANY(%s)",
            (stale_session_ids,),
        )
        stale_segment_ids = [row[0] for row in cur.fetchall()]
        if stale_segment_ids:
            cur.executemany(
                "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
                "VALUES ('segment', %s, 'delete', NULL)",
                [(sid,) for sid in stale_segment_ids],
            )
        # ON DELETE CASCADE takes care of segment / segment_message / segment_embedding.
        cur.execute("DELETE FROM session WHERE session_id = ANY(%s)", (stale_session_ids,))
    return len(stale_segment_ids)


def _insert_sessions(conn: psycopg.Connection, sessions: list[Session]) -> dict[datetime, int]:
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


def _insert_segments(
    conn: psycopg.Connection,
    chat_id: int,
    sessions: list[Session],
    rendered_by_session_start: dict[datetime, list[RenderedSegment]],
    session_ids: dict[datetime, int],
) -> int:
    written = 0
    with conn.cursor() as cur:
        for s in sessions:
            session_id = session_ids[s.started_at]
            for r in rendered_by_session_start.get(s.started_at, []):
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
) -> SegmentationRunReport:
    """Re-segment one chat from the point the incremental frontier
    (`imsg.segment.sessionize.compute_recompute_start`) says is safe,
    in one Postgres transaction. Pass `earliest_changed_at=
    REBUILD_ALL_SENTINEL` to force a full rebuild (config change).
    """
    seg_cfg = config.segmentation
    policy = config.policy

    current_hash = compute_seg_config_hash(
        session_gap_hours=seg_cfg.session_gap_hours,
        topical_min_messages=seg_cfg.topical_min_messages,
        max_messages=seg_cfg.max_messages,
        max_tokens=seg_cfg.max_tokens,
        boundary_model=seg_cfg.boundary_model,
        boundary_prompt_bytes=boundary_prompt_bytes,
        index_unsent=policy.index_unsent,
        index_edit_history=policy.index_edit_history,
    )

    chat_ctx = fetch_chat_context(conn, chat_id)
    existing_sessions = fetch_persisted_sessions(conn, chat_id)
    recompute_start = compute_recompute_start(
        existing_sessions, earliest_changed_at, seg_cfg.session_gap_hours
    )

    stale_session_ids = [s.session_id for s in existing_sessions if s.started_at >= recompute_start]

    messages = _fetch_messages_from(
        conn,
        chat_id,
        recompute_start,
        index_unsent=policy.index_unsent,
        include_edit_history=policy.index_edit_history,
    )

    sessions = sessionize(messages, chat_id=chat_id, session_gap_hours=seg_cfg.session_gap_hours)

    rendered_by_session_start: dict[datetime, list[RenderedSegment]] = {}
    fallback_count = 0
    segments_written_count = 0
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
        segments_written_count += len(segment_list)

    with conn.transaction():
        deleted = _delete_stale_and_emit_delete_events(conn, stale_session_ids)
        session_ids = _insert_sessions(conn, sessions)
        written = _insert_segments(conn, chat_id, sessions, rendered_by_session_start, session_ids)

    return SegmentationRunReport(
        chat_id=chat_id,
        sessions_written=len(sessions),
        segments_written=written,
        segments_deleted=deleted,
        fallback_sessions=fallback_count,
    )


def run_segment(
    conn: psycopg.Connection,
    config: Config,
    boundary_provider: BoundaryProvider,
    boundary_prompt_bytes: bytes,
    *,
    chat_ids: set[int] | None = None,
) -> list[SegmentationRunReport]:
    """Top-level incremental entry point (SPEC §8 S7's S4 step): find
    every dirty chat and re-segment each. `chat_ids`, if given,
    restricts the run to that subset (still driven by dirtiness, not a
    forced rebuild — use `run_segment_for_chat` with
    `REBUILD_ALL_SENTINEL` directly for `--rebuild`)."""
    dirty = find_dirty_chats(conn, index_unsent=config.policy.index_unsent)
    if chat_ids is not None:
        dirty = {cid: ts for cid, ts in dirty.items() if cid in chat_ids}

    reports = []
    for chat_id, earliest in dirty.items():
        reports.append(
            run_segment_for_chat(
                conn,
                chat_id,
                config,
                boundary_provider,
                boundary_prompt_bytes,
                earliest_changed_at=earliest,
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
