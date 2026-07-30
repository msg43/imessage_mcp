"""The weekly unclassified-threads report (SPEC §11.5): keeps the
allowlist from rotting by surfacing active chats whose participants
the owner has never classified.

Leak discipline: this report describes threads that are NOT
allowlisted, so it must never carry content. It contains identities
and counters only — short/display names, last-activity date, message
count. NO message text, NO attachment names, NO raw handles (hard
requirement 3: unresolved participants appear only as a count). It is
written to `$DATA_ROOT/export/` — on the encrypted volume, and NOT
under `export/staging/`, so no push path can ever pick it up (push
reads only manifest-listed relpaths inside a run's own staging dir).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import psycopg

    from imsg.config.schema import Config

ACTIVITY_WINDOW_DAYS = 90

_UNCLASSIFIED_SQL = """
    WITH recent AS (
        SELECT chat_id, max(sent_at) AS last_activity, count(*) AS recent_messages
        FROM message
        WHERE sent_at >= %(cutoff)s
        GROUP BY chat_id
    ),
    unlisted AS (
        SELECT DISTINCT cp.chat_id
        FROM chat_participant cp
        LEFT JOIN allowlist_person al ON al.person_id = cp.person_id
        WHERE al.person_id IS NULL
        UNION
        SELECT DISTINCT cps.chat_id
        FROM chat_participant_source cps
        LEFT JOIN source_handle_resolution shr
               ON shr.source_handle_id = cps.source_handle_id
        WHERE shr.handle_id IS NULL
    )
    SELECT c.chat_id, c.kind, c.display_name, r.last_activity, r.recent_messages
    FROM chat c
    JOIN recent r ON r.chat_id = c.chat_id
    JOIN unlisted u ON u.chat_id = c.chat_id
    LEFT JOIN thread_classification tc ON tc.chat_id = c.chat_id
    WHERE coalesce(tc.state, 'unreviewed') = 'unreviewed'
    ORDER BY r.last_activity DESC
"""


def _participants_line(conn: psycopg.Connection, chat_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.short_name, (al.person_id IS NOT NULL) AS listed
            FROM chat_participant cp
            JOIN person p ON p.person_id = cp.person_id
            LEFT JOIN allowlist_person al ON al.person_id = cp.person_id
            WHERE cp.chat_id = %s
            ORDER BY p.short_name
            """,
            (chat_id,),
        )
        names = [
            f"{short_name}" + ("" if listed else "*")
            for short_name, listed in cur.fetchall()
        ]
        cur.execute(
            """
            SELECT count(*)
            FROM chat_participant_source cps
            LEFT JOIN source_handle_resolution shr
                   ON shr.source_handle_id = cps.source_handle_id
            WHERE cps.chat_id = %s AND shr.handle_id IS NULL
            """,
            (chat_id,),
        )
        row = cur.fetchone()
        unresolved = int(row[0]) if row else 0
    line = ", ".join(names) if names else "(no resolved participants)"
    if unresolved:
        line += f"  [+{unresolved} unresolved handle(s)]"
    return line


def unclassified_summary(conn: psycopg.Connection, *, now: datetime | None = None) -> int:
    """Count of unclassified active threads — for `imsg status`."""
    cutoff = (now or datetime.now(UTC)) - timedelta(days=ACTIVITY_WINDOW_DAYS)
    with conn.cursor() as cur:
        cur.execute(_UNCLASSIFIED_SQL, {"cutoff": cutoff})
        return len(cur.fetchall())


def write_unclassified_report(
    conn: psycopg.Connection, config: Config, *, now: datetime | None = None
) -> Path:
    """Write `$DATA_ROOT/export/unclassified-<date>.txt` and return its
    path. Idempotent for a given date (overwrites)."""
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=ACTIVITY_WINDOW_DAYS)
    with conn.cursor() as cur:
        cur.execute(_UNCLASSIFIED_SQL, {"cutoff": cutoff})
        rows = cur.fetchall()

    lines = [
        f"Unclassified threads report — {now:%Y-%m-%d}",
        f"Chats with activity since {cutoff:%Y-%m-%d} having participants not yet",
        "classified for export. Contains identities and counts only — never",
        "message content. (* = participant has no allowlist entry)",
        "",
    ]
    if not rows:
        lines.append("Nothing to classify — every active thread is fully classified.")
    for chat_id, kind, display_name, last_activity, recent_messages in rows:
        label = f'group "{display_name}"' if display_name else str(kind)
        lines.append(
            f"- chat {chat_id} ({label}): last activity {last_activity:%Y-%m-%d}, "
            f"{recent_messages} message(s) in the last {ACTIVITY_WINDOW_DAYS}d"
        )
        lines.append(f"  participants: {_participants_line(conn, int(chat_id))}")
    lines.append("")

    out_dir = config.paths.data_root / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"unclassified-{now:%Y-%m-%d}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


__all__ = ["ACTIVITY_WINDOW_DAYS", "unclassified_summary", "write_unclassified_report"]
