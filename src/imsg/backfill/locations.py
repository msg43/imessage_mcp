"""The `attachment_location` store (migration 0008; owner decisions D12, D13).

One row per attachment and candidate location: "a copy of this attachment
may be at `path` in `location`". Three writers, each owning its columns:

- **S2 extraction** records the path every source's chat.db recorded
  (`record_recorded_path`), filed under that source's name. Insert only,
  never rewritten, so no source's path is ever lost (D12) — before this,
  `attachment.source_path` held one path and every other source's was
  dropped.
- **`imsg locate-attachments`** (`imsg.backfill.locate`) records copies
  found in listings, drive catalogs, seed databases and this host's own
  Messages folder (`observe_location`), and refreshes their evidence.
- **The backfill** (`imsg.backfill.fetch`) and the push record step
  (`imsg.backfill.push`) write only the attempt columns
  (`record_attempt`, `mark_location_fetched`).

**Match quality** is why a file is believed to be the attachment. Measured
2026-09-23 against another Mac's Messages folder, on attachments already
materialized with a known content hash: a GUID folder plus the file name
found the right content 57,532 of 57,532 times; the same name and byte
size in another folder 38,018 of 38,210 times (99.5%). A name alone is
never accepted (D13), and the database type has no value for it — a
name-only match cannot be stored, so it can never be fetched.

**Paths.** A host path (`~/...` or absolute) names a file on a Mac, and is
read relative to that Mac's Messages attachments folder: every Mac lays
attachments out the same way below it, so a path one Mac recorded names
the same file on another. A path that does not start with `~` or `/` is
relative to a drive's root, as a drive catalog records it.
`fetch_relative_path` turns either into the path to read below the
location's root, or `None` when it cannot be read from any root (a path
into a temporary directory, for example).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import psycopg


class MatchQuality(StrEnum):
    """`attachment_match_quality` (migration 0008), best first."""

    RECORDED_PATH = "recorded_path"
    """A chat.db recorded this path for the attachment."""
    GUID_FOLDER = "guid_folder"
    """The file sits in a folder named after the attachment's GUID, under
    the attachment's file name."""
    NAME_SIZE = "name_size"
    """Same file name and byte size, found elsewhere. Flagged: right 99.5%
    of the time when measured, not certain."""

    @property
    def rank(self) -> int:
        """0 is best. The declaration order of the Postgres enum agrees,
        so `ORDER BY match_quality` sorts the same way."""
        return _MATCH_RANK[self]

    @property
    def flagged(self) -> bool:
        """A copy accepted on this match is right most of the time, not
        certainly: only `name_size`."""
        return self is MatchQuality.NAME_SIZE


_MATCH_RANK: dict[MatchQuality, int] = {
    MatchQuality.RECORDED_PATH: 0,
    MatchQuality.GUID_FOLDER: 1,
    MatchQuality.NAME_SIZE: 2,
}


class LocationOutcome(StrEnum):
    """`attachment_location.last_outcome`: how the last attempt ended."""

    FETCHED = "fetched"
    """This copy was verified and materialized into the cache."""
    STAGED = "staged"
    """Another host copied it into this host's staging directory. The
    backfill verifies it there; until then the location stays open."""
    REJECTED = "rejected"
    """A file was there, but its size or content hash differed from what
    the location's listing said."""
    ABSENT = "absent"
    """No file at the path."""
    REFUSED = "refused"
    """The path is outside every root the fetcher may read, or is not a
    regular file (a directory, a symlink). Never read."""
    UNREACHABLE = "unreachable"
    """The host or drive could not be read (SSH failed, the drive is not
    mounted, a transient read error). Retried on the next run."""


RETRY_OUTCOMES: frozenset[LocationOutcome] = frozenset(
    {LocationOutcome.STAGED, LocationOutcome.UNREACHABLE}
)
"""Outcomes that leave a location open. Every other outcome closes it
until `imsg locate-attachments` records different evidence for it (a
better match, a new size or hash), which reopens it."""

MESSAGES_ATTACHMENTS_DIR = "Library/Messages/Attachments"
MESSAGES_PATH_PREFIX = "~/" + MESSAGES_ATTACHMENTS_DIR + "/"
"""How a Mac's chat.db records an attachment path, and how this module
records a copy found in a Mac's Messages folder."""

_MESSAGES_MARKER = "/" + MESSAGES_ATTACHMENTS_DIR + "/"
_LOCATION_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_valid_location_code(code: str) -> bool:
    """A code becomes one directory name under the staging directory, so
    it must be a single safe path component."""
    return bool(_LOCATION_CODE_RE.fullmatch(code)) and code not in {".", ".."}


def is_sha256_hex(value: str | None) -> bool:
    return value is not None and bool(_SHA256_RE.fullmatch(value))


def safe_relative_path(rel: str) -> bool:
    """True for a relative path that cannot climb out of whatever root it
    is joined to: no leading `/`, no empty, `.` or `..` component, no NUL.
    Containment is still checked against the real filesystem wherever a
    joined path is read; this is the check that needs no filesystem."""
    if not rel or "\x00" in rel or rel.startswith("/"):
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def messages_relative_path(path: str) -> str | None:
    """The part of `path` below a Messages attachments folder, or `None`.

    Accepts the chat.db form (`~/Library/Messages/Attachments/ab/cd/X/a.heic`),
    an absolute home path (`/Users/<name>/Library/...`) and a drive path
    that holds a copy of a Messages folder anywhere below its root. The
    last occurrence wins, so a backup of a backup resolves to the
    innermost folder."""
    probe = path[1:] if path.startswith("~/") else path
    if not probe.startswith("/"):
        probe = "/" + probe
    index = probe.rfind(_MESSAGES_MARKER)
    if index < 0:
        return None
    rel = probe[index + len(_MESSAGES_MARKER) :]
    return rel if safe_relative_path(rel) else None


def fetch_relative_path(path: str) -> str | None:
    """The path to read below a location's root (module docstring)."""
    if path.startswith(("~", "/")):
        return messages_relative_path(path)
    return path if safe_relative_path(path) else None


def name_key(name: str) -> str:
    """How file names are compared: case-insensitively, and with Unicode
    composed the same way (a Mac may store a name decomposed where chat.db
    recorded it composed). Stored paths are never normalized; only the
    comparison is."""
    if not name.isascii():
        name = unicodedata.normalize("NFC", name)
    return name.casefold()


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------


def record_recorded_path(
    cur: psycopg.Cursor[Any], *, attachment_id: int, source_name: str, path: str
) -> bool:
    """S2: the path `source_name`'s chat.db recorded for this attachment.
    Insert only, and never rewritten — a later run of the same source
    finds its row already there. Returns True when the row is new."""
    cur.execute(
        """
        INSERT INTO attachment_location (attachment_id, location, path, match_quality, reported_by)
        VALUES (%s, %s, %s, 'recorded_path', ARRAY[%s]::text[])
        ON CONFLICT (attachment_id, location, path) DO NOTHING
        RETURNING location_id
        """,
        (attachment_id, source_name, path, source_name),
    )
    return cur.fetchone() is not None


@dataclass(frozen=True, slots=True)
class LocationObservation:
    """One candidate copy, as a listing, catalog, seed database or folder
    walk reported it."""

    attachment_id: int
    location: str
    path: str
    match: MatchQuality
    reported_by: str
    byte_size: int | None = None
    sha256: str | None = None


class ObservationOutcome(StrEnum):
    INSERTED = "inserted"
    UPDATED = "updated"
    """The row existed and its evidence changed: a better match, or a size
    or hash it did not have or that differs. Reopens a closed location."""
    UNCHANGED = "unchanged"


_EVIDENCE_CHANGED = """(
            LEAST(l.match_quality, EXCLUDED.match_quality) IS DISTINCT FROM l.match_quality
            OR COALESCE(EXCLUDED.byte_size, l.byte_size) IS DISTINCT FROM l.byte_size
            OR COALESCE(EXCLUDED.sha256, l.sha256) IS DISTINCT FROM l.sha256
        ) AND l.fetched_at IS NULL"""

_OBSERVE_SQL = f"""
WITH prior AS (
    SELECT match_quality, byte_size, sha256
      FROM attachment_location
     WHERE attachment_id = %(attachment_id)s AND location = %(location)s AND path = %(path)s
), written AS (
    INSERT INTO attachment_location AS l
        (attachment_id, location, path, match_quality, reported_by, byte_size, sha256)
    VALUES (%(attachment_id)s, %(location)s, %(path)s, %(match)s::attachment_match_quality,
            ARRAY[%(reported_by)s]::text[], %(byte_size)s, %(sha256)s)
    ON CONFLICT (attachment_id, location, path) DO UPDATE SET
        match_quality = LEAST(l.match_quality, EXCLUDED.match_quality),
        reported_by = CASE WHEN EXCLUDED.reported_by[1] = ANY (l.reported_by)
                           THEN l.reported_by
                           ELSE l.reported_by || EXCLUDED.reported_by END,
        byte_size = COALESCE(EXCLUDED.byte_size, l.byte_size),
        sha256 = COALESCE(EXCLUDED.sha256, l.sha256),
        last_seen_at = now(),
        -- New evidence makes the location new again: an attempt that
        -- failed against the old evidence says nothing about the new.
        last_tried_at = CASE WHEN {_EVIDENCE_CHANGED} THEN NULL ELSE l.last_tried_at END,
        last_outcome = CASE WHEN {_EVIDENCE_CHANGED} THEN NULL ELSE l.last_outcome END,
        last_error = CASE WHEN {_EVIDENCE_CHANGED} THEN NULL ELSE l.last_error END
    RETURNING (xmax = 0) AS inserted, match_quality, byte_size, sha256
)
SELECT written.inserted,
       (prior.match_quality, prior.byte_size, prior.sha256)
         IS DISTINCT FROM (written.match_quality, written.byte_size, written.sha256)
  FROM written LEFT JOIN prior ON true
"""


def observe_location(cur: psycopg.Cursor[Any], obs: LocationObservation) -> ObservationOutcome:
    """Insert a candidate copy, or merge new evidence into the row for the
    same attachment, location and path: the better match is kept, a new
    reporter is appended, and a size or hash is filled or replaced by the
    newer observation (a file's size is what its location says now)."""
    if obs.sha256 is not None and not is_sha256_hex(obs.sha256):
        raise ValueError(f"not a lowercase hex sha256: {obs.sha256!r}")
    cur.execute(
        _OBSERVE_SQL,
        {
            "attachment_id": obs.attachment_id,
            "location": obs.location,
            "path": obs.path,
            "match": obs.match.value,
            "reported_by": obs.reported_by,
            "byte_size": obs.byte_size,
            "sha256": obs.sha256,
        },
    )
    row = cur.fetchone()
    assert row is not None, "attachment_location upsert returned no row"
    if row[0]:
        return ObservationOutcome.INSERTED
    return ObservationOutcome.UPDATED if row[1] else ObservationOutcome.UNCHANGED


def record_attempt(
    conn: psycopg.Connection,
    location_id: int,
    outcome: LocationOutcome,
    error: str | None = None,
) -> None:
    """How one attempt on one location ended. `FETCHED` goes through
    `mark_location_fetched` instead, together with the attachment row."""
    if outcome is LocationOutcome.FETCHED:
        raise ValueError("a fetched location is recorded by mark_location_fetched")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE attachment_location SET last_tried_at = now(), last_outcome = %s, "
            "last_error = %s WHERE location_id = %s AND fetched_at IS NULL",
            (outcome.value, None if error is None else error[:2000], location_id),
        )
    conn.commit()


def mark_location_fetched(
    conn: psycopg.Connection,
    *,
    attachment_id: int,
    location_id: int,
    sha256: str,
    byte_size: int,
    cache_path: str,
) -> None:
    """The attachment is materialized from this location's copy: both rows
    change in one transaction, so a crash between them cannot leave a
    materialized attachment with no record of where its bytes came from."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            UPDATE attachment
            SET state = 'materialized', sha256 = %s, byte_size = %s, cache_path = %s,
                materialization_last_error = NULL, updated_at = now()
            WHERE attachment_id = %s
            """,
            (sha256, byte_size, cache_path, attachment_id),
        )
        cur.execute(
            "UPDATE attachment_location SET fetched_at = now(), last_tried_at = now(), "
            "last_outcome = 'fetched', last_error = NULL WHERE location_id = %s",
            (location_id,),
        )
    conn.commit()


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateLocation:
    """One open candidate copy of an attachment not yet materialized."""

    location_id: int
    attachment_id: int
    location: str
    path: str
    match: MatchQuality
    byte_size: int | None
    sha256: str | None
    last_outcome: LocationOutcome | None

    @property
    def rel(self) -> str | None:
        return fetch_relative_path(self.path)


_OPEN_CANDIDATES_SQL = """
SELECT l.location_id, l.attachment_id, l.location, l.path, l.match_quality::text,
       l.byte_size, l.sha256, l.last_outcome
  FROM attachment_location l
  JOIN attachment a USING (attachment_id)
 WHERE a.state <> 'materialized'
   AND l.fetched_at IS NULL
   AND (l.last_tried_at IS NULL OR l.last_outcome IN ('staged', 'unreachable'))
 ORDER BY l.attachment_id, l.match_quality, l.location_id
"""


def load_open_candidates(conn: psycopg.Connection) -> dict[int, list[CandidateLocation]]:
    """Every open candidate of every attachment that is not materialized,
    in any state (`unsupported` and `missing` included — a new location is
    exactly what can change their outcome), grouped by attachment and
    ordered best match first. Open means never tried, or last left
    retryable (`RETRY_OUTCOMES`)."""
    with conn.cursor() as cur:
        cur.execute(_OPEN_CANDIDATES_SQL)
        rows = cur.fetchall()
    grouped: dict[int, list[CandidateLocation]] = {}
    for location_id, attachment_id, location, path, match, size, sha, outcome in rows:
        grouped.setdefault(int(attachment_id), []).append(
            CandidateLocation(
                location_id=int(location_id),
                attachment_id=int(attachment_id),
                location=str(location),
                path=str(path),
                match=MatchQuality(match),
                byte_size=None if size is None else int(size),
                sha256=sha,
                last_outcome=None if outcome is None else LocationOutcome(outcome),
            )
        )
    return grouped


def other_open_reference_exists(
    conn: psycopg.Connection, *, location: str, path: str, exclude_location_ids: Iterable[int] = ()
) -> bool:
    """Whether any attachment not yet materialized still has an open row
    for this exact file. A staged copy is released only when none does:
    one file can be the candidate copy of more than one attachment."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM attachment_location l JOIN attachment a USING (attachment_id)
             WHERE l.location = %s AND l.path = %s
               AND a.state <> 'materialized' AND l.fetched_at IS NULL
               AND (l.last_outcome IS NULL OR l.last_outcome IN ('staged', 'unreachable'))
               AND NOT (l.location_id = ANY(%s::bigint[]))
             LIMIT 1
            """,
            (location, path, list(exclude_location_ids)),
        )
        return cur.fetchone() is not None


def attachment_kind(mime_type: str | None, uti: str | None, filename: str | None) -> str:
    """A coarse kind for reports: what a gap is, not just how many."""
    mime = (mime_type or "").lower()
    type_id = (uti or "").lower()
    name = (filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if "pluginpayload" in type_id or ext == "pluginpayloadattachment":
        return "link-preview"
    if mime.startswith("image/") or "image" in type_id or ext in {"heic", "jpg", "jpeg", "png", "gif"}:
        return "image"
    if mime.startswith("video/") or "movie" in type_id or "video" in type_id or ext in {"mov", "mp4"}:
        return "video"
    if mime.startswith("audio/") or "audio" in type_id or ext in {"caf", "m4a", "amr"}:
        return "audio"
    if "pdf" in mime or ext == "pdf":
        return "pdf"
    if "vcard" in mime or ext == "vcf":
        return "vcard"
    return "other"


__all__ = [
    "MESSAGES_ATTACHMENTS_DIR",
    "MESSAGES_PATH_PREFIX",
    "RETRY_OUTCOMES",
    "CandidateLocation",
    "LocationObservation",
    "LocationOutcome",
    "MatchQuality",
    "ObservationOutcome",
    "attachment_kind",
    "fetch_relative_path",
    "is_sha256_hex",
    "is_valid_location_code",
    "load_open_candidates",
    "mark_location_fetched",
    "messages_relative_path",
    "name_key",
    "observe_location",
    "other_open_reference_exists",
    "record_attempt",
    "record_recorded_path",
    "safe_relative_path",
]
