"""AT-2 — seed completeness (SPEC §12 AT-2; flagged homeless by
`docs/DECISIONS.md` D8; SPEC §8 S7: "Seed completeness check (AT-2)
runs before the seed source is marked done").

**Why this lives under `imsg.verify`, not a pipeline stage or
`imsg.eval`**: AT-2 is a one-shot diagnostic verdict ("did the Studio
seed transfer completely?"), not an ongoing pipeline step and not a
retrieval-quality measurement — see the package docstring. Phase 1's
exit criteria depend on it (SPEC §15).

**Why this compares against an exported file, not a live second
database**: the build brief that produced this module works from a
single reachable Postgres instance at a time (the mini and the Studio
are never simultaneously reachable in this environment, and won't
generally be in production either — they're two different Macs). SPEC
§12's literal `imsg verify-seed --reference <studio-snapshot.db>`
implies a live snapshot db is open on both sides at once; this
implementation instead works from a **snapshot file exported by this
same command run on the other host** (`--export`), which is strictly
more portable (works over the tailnet via a copied file, no live
connection required) and is what "counts file exported from the other
machine" in the build brief calls for. The exactness requirement is
unchanged: SPEC §12 AT-2 / D6 — "compares exact canonical message GUID
sets, not only annual counts ... a 0.5% per-year tolerance could hide
thousands of missing messages" — `verify_against_reference` diffs the
full GUID set, not just per-year totals; per-year counts, timestamp
range, decode-null counts, and attachment-join counts are carried as
*diagnostics* alongside the exact-set verdict, matching the spec text.

**`--reference-db` (added later)**: the `--export` exchange assumes
the other side is a host running this software. A *recovered* or
*merged* corpus has no such host — it is just a `chat.db`-shaped file —
which left AT-2 unrunnable for exactly the kind of seed it was written
to check. `build_seed_snapshot_from_chat_db` builds the same
`SeedSnapshot` straight from that file, applying in SQL the same three
inclusion filters `imsg.stages.extract._do_extract` applies in Python,
so the verdict keeps its meaning: a missing GUID is a message the
extractor should have landed and did not.

Duplicate messages across sources are expected and are not a
completeness gap (SPEC §12 AT-2: "duplicates across sources are
expected and visible in `message_source`") — `message.source_guid` is
already the canonical dedupe key (`UNIQUE`, SPEC §7.2), so a message
ingested from both the mini and the Studio appears exactly once in the
GUID set being compared here regardless of how many `message_source`
rows back it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg

SEED_SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SeedSnapshot:
    """One host's exportable completeness fingerprint — the "counts
    file exported from the other machine." `guids` is the exact,
    canonical set this build's completeness verdict is computed from;
    everything else is diagnostic context (SPEC §12 AT-2: "reports
    per-year counts, min/max timestamps, body-decode-null counts, and
    attachment-join counts as diagnostics")."""

    source_label: str
    generated_at: str  # ISO-8601
    guids: frozenset[str]
    per_year_counts: dict[str, int]  # "2023" -> message count, sent_at year in UTC
    min_sent_at: str | None
    max_sent_at: str | None
    body_decode_null_count: int
    """`message.text_original IS NULL` count — a message the extractor
    (or the upstream `imsg-dump` shim, SPEC §4.2) could not decode a
    body for. Not itself a completeness failure (the row exists and is
    counted), but worth surfacing loudly per-host."""
    attachment_join_count: int
    """Distinct messages with >= 1 linked attachment (`message_attachment`)."""


def build_seed_snapshot(conn: psycopg.Connection, *, source_label: str) -> SeedSnapshot:
    """Compute this host's `SeedSnapshot` from its live Postgres. Read
    only — this never touches `chat.db`, live or snapshotted; it reads
    the already-extracted `message` table (SPEC §7.2)."""
    with conn.cursor() as cur:
        cur.execute("SELECT source_guid FROM message")
        guids = frozenset(str(row[0]) for row in cur.fetchall())

        cur.execute(
            """
            SELECT extract(year FROM sent_at AT TIME ZONE 'UTC')::int, count(*)
            FROM message
            GROUP BY 1
            ORDER BY 1
            """
        )
        per_year = {str(int(year)): int(count) for year, count in cur.fetchall()}

        cur.execute("SELECT min(sent_at), max(sent_at) FROM message")
        row = cur.fetchone()
        min_sent, max_sent = (row[0], row[1]) if row is not None else (None, None)

        cur.execute("SELECT count(*) FROM message WHERE text_original IS NULL")
        null_row = cur.fetchone()
        null_count = int(null_row[0]) if null_row is not None else 0

        cur.execute("SELECT count(DISTINCT message_id) FROM message_attachment")
        attach_row = cur.fetchone()
        attach_count = int(attach_row[0]) if attach_row is not None else 0

    return SeedSnapshot(
        source_label=source_label,
        generated_at=datetime.now(UTC).isoformat(),
        guids=guids,
        per_year_counts=per_year,
        min_sent_at=min_sent.isoformat() if min_sent is not None else None,
        max_sent_at=max_sent.isoformat() if max_sent is not None else None,
        body_decode_null_count=null_count,
        attachment_join_count=attach_count,
    )


APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
"""chat.db's `message.date` epoch. Mirrors `imsg.stages.extract.APPLE_EPOCH`."""

APPLE_EPOCH_UNIX_OFFSET = 978_307_200
"""Seconds from the Unix epoch to `APPLE_EPOCH`, for SQLite's `unixepoch`."""

CHAT_DB_TAPBACK_TYPES_SQL = (
    "COALESCE(m.associated_message_type, 0) = 1000 "
    "OR COALESCE(m.associated_message_type, 0) BETWEEN 2000 AND 3999"
)
"""Which `chat.db` rows the extractor routes to `tapback` instead of
`message`.

`imsg.stages.extract._do_extract` does not decide this from SQL at all
— it asks `imsg-dump` (the typedstream decoder) and `continue`s on any
row that comes back with a `tapback`. This clause is the SQL-only
restatement of that decision, needed because a reference built from a
bare SQLite file has no decoder to ask. The `2000..3999` half is the
documented reaction range; the `1000` half (stickers) is the part that
is easy to get wrong, and is included on evidence, not on inference:
in the 2026-08-20 merged corpus all 17 `associated_message_type = 1000`
rows are present in Postgres's `tapback` table and absent from
`message`, so the live extractor did treat them as tapbacks.

Getting this wrong does not corrupt anything — it makes AT-2 report
tapbacks as "missing messages" — but it makes the verdict useless, so
it is stated here once rather than re-derived per call site."""

_ELIGIBLE_MESSAGE_WHERE = f"""
    m.guid IS NOT NULL
    AND NOT ({CHAT_DB_TAPBACK_TYPES_SQL})
    AND COALESCE(m.item_type, 0) = 0
    AND EXISTS (SELECT 1 FROM chat_message_join j WHERE j.message_id = m.ROWID)
"""
"""The three filters `_do_extract` applies before it upserts a `message`
row, in the same order: tapbacks go to `tapback`, `item_type != 0` is a
skipped system row, and a row with no `chat_message_join` is logged as
`extract.message_without_chat` and dropped."""


def _open_chat_db_read_only(db_path: Path) -> sqlite3.Connection:
    """Open a `chat.db`-shaped file strictly read-only.

    `immutable=1` is not just belt-and-braces: without it SQLite may
    create `-wal`/`-shm` sidecar files next to the database even for a
    read, and CLAUDE.md's first non-negotiable is that nothing in this
    system writes anything next to a `chat.db`. Stdlib `sqlite3` rather
    than `apsw` (used elsewhere in this package) because its URI form
    carries both flags with no flag arithmetic.
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"reference database not found: {db_path}")
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (table,)
    ).fetchone()
    return row is not None


def build_seed_snapshot_from_chat_db(db_path: Path, *, source_label: str) -> SeedSnapshot:
    """Build a reference `SeedSnapshot` from a `chat.db`-shaped SQLite
    file instead of from another host's Postgres.

    Why this exists alongside `build_seed_snapshot`: the `--export`
    path answers "do these two Postgres instances agree?", which needs
    the other host to be running this software. AT-2's actual question
    during a seed is narrower and one-sided — "did every message in the
    corpus file I just ingested land in `message`?" — and the corpus
    file is right here. Requiring an export from a host that may not
    exist any more (a recovered/merged corpus has no host) made AT-2
    unrunnable for exactly the case it was written for.

    The GUID set is **not** every row in the file: it is the set the
    extractor is supposed to land in `message`, per
    `_ELIGIBLE_MESSAGE_WHERE`. Comparing against every row would report
    tapbacks and system rows as completeness failures, which inverts
    the verdict's meaning — those rows are correctly elsewhere or
    correctly absent.

    Two diagnostics are weaker here than on the `--export` path and are
    labelled as such:

    * `body_decode_null_count` is a **proxy**. Postgres counts
      `text_original IS NULL`, i.e. what `imsg-dump`'s typedstream
      decoder actually produced; SQL alone cannot know that, so this
      counts rows with neither a `text` value nor an `attributedBody`
      blob — rows no decoder could have produced a body for. A row with
      an `attributedBody` the decoder then failed on is counted here as
      decodable and in Postgres as null, so reference <= local is
      normal and is not a finding on its own.
    * `attachment_join_count` counts eligible messages with at least one
      `message_attachment_join` row. Postgres counts distinct
      `message_attachment.message_id` across every source, so the two
      agree only when the local database holds nothing outside this
      corpus.

    Timestamps use the same conversion as the extractor
    (`imsg.stages.extract._apple_ns_to_datetime`): Apple-epoch
    nanoseconds, divided by 1e9. SQLite's integer division floors, but
    only ever by less than a second and only ever earlier, so it cannot
    move a row across a year boundary that Postgres would not also
    cross.
    """
    conn = _open_chat_db_read_only(db_path)
    try:
        if not _has_table(conn, "message"):
            raise ValueError(f"{db_path} has no 'message' table — not a chat.db-shaped database")
        if not _has_table(conn, "chat_message_join"):
            raise ValueError(
                f"{db_path} has no 'chat_message_join' table — not a chat.db-shaped database"
            )

        guids = frozenset(
            str(row[0])
            for row in conn.execute(f"SELECT m.guid FROM message m WHERE {_ELIGIBLE_MESSAGE_WHERE}")
        )

        per_year = {
            str(int(year)): int(count)
            for year, count in conn.execute(
                f"""
                SELECT CAST(strftime('%Y', datetime(
                           m.date / 1000000000 + {APPLE_EPOCH_UNIX_OFFSET}, 'unixepoch'
                       )) AS INTEGER) AS yr,
                       count(*)
                FROM message m
                WHERE {_ELIGIBLE_MESSAGE_WHERE}
                GROUP BY yr
                ORDER BY yr
                """
            )
            if year is not None
        }

        bounds = conn.execute(
            f"SELECT min(m.date), max(m.date) FROM message m WHERE {_ELIGIBLE_MESSAGE_WHERE}"
        ).fetchone()
        min_ns, max_ns = (bounds[0], bounds[1]) if bounds is not None else (None, None)

        message_columns = _table_columns(conn, "message")
        if {"text", "attributedBody"} <= message_columns:
            undecodable = conn.execute(
                f"""
                SELECT count(*) FROM message m
                WHERE {_ELIGIBLE_MESSAGE_WHERE}
                  AND COALESCE(m.text, '') = '' AND m.attributedBody IS NULL
                """
            ).fetchone()
            body_null_count = int(undecodable[0]) if undecodable is not None else 0
        else:
            body_null_count = 0

        if _has_table(conn, "message_attachment_join"):
            attach = conn.execute(
                f"""
                SELECT count(DISTINCT j.message_id)
                FROM message_attachment_join j
                JOIN message m ON m.ROWID = j.message_id
                WHERE {_ELIGIBLE_MESSAGE_WHERE}
                """
            ).fetchone()
            attach_count = int(attach[0]) if attach is not None else 0
        else:
            attach_count = 0
    finally:
        conn.close()

    return SeedSnapshot(
        source_label=source_label,
        generated_at=datetime.now(UTC).isoformat(),
        guids=guids,
        per_year_counts=per_year,
        min_sent_at=_apple_ns_to_iso(min_ns),
        max_sent_at=_apple_ns_to_iso(max_ns),
        body_decode_null_count=body_null_count,
        attachment_join_count=attach_count,
    )


def _apple_ns_to_iso(value: int | None) -> str | None:
    if value is None or value == 0:
        return None
    return (APPLE_EPOCH + timedelta(seconds=value / 1_000_000_000)).isoformat()


def snapshot_to_json(snapshot: SeedSnapshot) -> str:
    payload = {
        "format_version": SEED_SNAPSHOT_FORMAT_VERSION,
        "source_label": snapshot.source_label,
        "generated_at": snapshot.generated_at,
        "guids": sorted(snapshot.guids),
        "per_year_counts": snapshot.per_year_counts,
        "min_sent_at": snapshot.min_sent_at,
        "max_sent_at": snapshot.max_sent_at,
        "body_decode_null_count": snapshot.body_decode_null_count,
        "attachment_join_count": snapshot.attachment_join_count,
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def snapshot_from_json(text: str) -> SeedSnapshot:
    payload = json.loads(text)
    version = payload.get("format_version")
    if version != SEED_SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            f"seed snapshot format_version {version!r} is not supported "
            f"(expected {SEED_SNAPSHOT_FORMAT_VERSION})"
        )
    return SeedSnapshot(
        source_label=str(payload["source_label"]),
        generated_at=str(payload["generated_at"]),
        guids=frozenset(str(g) for g in payload["guids"]),
        per_year_counts={str(k): int(v) for k, v in payload["per_year_counts"].items()},
        min_sent_at=payload["min_sent_at"],
        max_sent_at=payload["max_sent_at"],
        body_decode_null_count=int(payload["body_decode_null_count"]),
        attachment_join_count=int(payload["attachment_join_count"]),
    )


@dataclass(frozen=True, slots=True)
class SeedVerificationReport:
    """The AT-2 verdict: `reference` (the other host's exported
    snapshot) vs. `local` (this host's live database, computed fresh).
    """

    reference_label: str
    local_label: str
    reference_message_count: int
    local_message_count: int
    missing_guids: tuple[str, ...]
    """GUIDs present in `reference` but absent locally, MINUS any
    `accepted_exceptions` — SPEC §12 AT-2: "Any accepted exception is
    enumerated by GUID and owner-approved," never silently dropped."""
    accepted_missing_guids: tuple[str, ...]
    """The subset of the raw missing set that `accepted_exceptions`
    excused — reported, not hidden, so an owner-approved exception
    list is itself auditable."""
    per_year_diff: dict[str, tuple[int, int]]  # year -> (reference_count, local_count)
    reference_body_decode_null_count: int
    local_body_decode_null_count: int
    reference_attachment_join_count: int
    local_attachment_join_count: int
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def verify_against_reference(
    conn: psycopg.Connection,
    reference: SeedSnapshot,
    *,
    local_label: str,
    accepted_exceptions: frozenset[str] = frozenset(),
) -> SeedVerificationReport:
    """SPEC §12 AT-2's exact-GUID-set diff, computed against a live
    fresh snapshot of `conn`. **Fails** on any reference GUID missing
    locally that isn't in `accepted_exceptions` — a source message
    without a matching local row is "a hard extraction anomaly,
    enumerated and failed" (SPEC §12 AT-2), not averaged away by a
    per-year tolerance.
    """
    local = build_seed_snapshot(conn, source_label=local_label)

    raw_missing = reference.guids - local.guids
    accepted_missing = raw_missing & accepted_exceptions
    unexplained_missing = raw_missing - accepted_exceptions

    years = sorted(set(reference.per_year_counts) | set(local.per_year_counts))
    per_year_diff = {
        year: (reference.per_year_counts.get(year, 0), local.per_year_counts.get(year, 0))
        for year in years
    }

    reasons: list[str] = []
    if unexplained_missing:
        reasons.append(
            f"{len(unexplained_missing)} message GUID(s) present in "
            f"'{reference.source_label}' but missing locally and not in the "
            f"accepted-exceptions list"
        )

    return SeedVerificationReport(
        reference_label=reference.source_label,
        local_label=local_label,
        reference_message_count=len(reference.guids),
        local_message_count=len(local.guids),
        missing_guids=tuple(sorted(unexplained_missing)),
        accepted_missing_guids=tuple(sorted(accepted_missing)),
        per_year_diff=per_year_diff,
        reference_body_decode_null_count=reference.body_decode_null_count,
        local_body_decode_null_count=local.body_decode_null_count,
        reference_attachment_join_count=reference.attachment_join_count,
        local_attachment_join_count=local.attachment_join_count,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def format_report_text(report: SeedVerificationReport) -> str:
    lines = [
        f"AT-2 seed completeness: '{report.local_label}' vs reference "
        f"'{report.reference_label}'",
        f"  reference messages: {report.reference_message_count}",
        f"  local messages:     {report.local_message_count}",
        f"  missing (unexplained): {len(report.missing_guids)}",
        f"  missing (accepted exceptions): {len(report.accepted_missing_guids)}",
        f"  body-decode-null: reference={report.reference_body_decode_null_count} "
        f"local={report.local_body_decode_null_count}",
        f"  attachment-join count: reference={report.reference_attachment_join_count} "
        f"local={report.local_attachment_join_count}",
        "  per-year (reference, local):",
    ]
    for year, (ref_count, local_count) in sorted(report.per_year_diff.items()):
        flag = "" if ref_count == local_count else "  <-- differs"
        lines.append(f"    {year}: ({ref_count}, {local_count}){flag}")
    if report.missing_guids:
        lines.append("  unexplained missing GUIDs:")
        for guid in report.missing_guids[:50]:
            lines.append(f"    {guid}")
        if len(report.missing_guids) > 50:
            lines.append(f"    ... and {len(report.missing_guids) - 50} more")
    lines.append(f"  VERDICT: {'PASS' if report.passed else 'FAIL'}")
    for reason in report.reasons:
        lines.append(f"    - {reason}")
    return "\n".join(lines)


__all__ = [
    "APPLE_EPOCH",
    "APPLE_EPOCH_UNIX_OFFSET",
    "CHAT_DB_TAPBACK_TYPES_SQL",
    "SEED_SNAPSHOT_FORMAT_VERSION",
    "SeedSnapshot",
    "SeedVerificationReport",
    "build_seed_snapshot",
    "build_seed_snapshot_from_chat_db",
    "format_report_text",
    "snapshot_from_json",
    "snapshot_to_json",
    "verify_against_reference",
]
