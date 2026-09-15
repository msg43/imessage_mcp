"""Open SQLite files read-only *and* without leaving sidecar files behind.

Hard requirement 1 (CLAUDE.md, non-negotiable #1): never write to the
live `chat.db`. The pipeline reads that one file only through S1's
online backup (`imsg.stages.snapshot`); every other database it opens
is a completed snapshot or a prepared seed, and nothing may be written
next to those either — a `--snapshot` argument can name any file the
operator has, and "read-only" has to mean the directory is left
exactly as it was found.

`SQLITE_OPEN_READONLY` on its own does not deliver that. Every
chat.db-shaped file this pipeline meets carries the WAL journal mode
in its header (the live database does, and SQLite's online-backup API
copies page 1 verbatim, so S1's own output and any corpus built from
one inherit it). For a WAL-mode file SQLite's read path opens the
`-shm` wal-index read-write — creating it, and an empty `-wal`, if
they are missing — and may rewrite the index, even on a read-only
connection. Observed 2026-09-14: `imsg extract --snapshot <seed>` moved
the seed's `-shm` mtime to the run's start, and the pipeline's own
`snapshots/` directory held orphaned `.tmp-snapshot-*.db-shm`/`-wal`
files left by the post-backup verify open.

The URI form `file:<path>?mode=ro&immutable=1` is the fix. With
`immutable=1` SQLite treats the file as read-only media: no locks, and
the `-wal`/`-shm` sidecars are neither read nor created. The trade,
and the reason S1's *live* source must not use this module: the flag
asserts the file cannot change while open, and SQLite honours that by
ignoring the write-ahead log entirely — every committed row not yet
checkpointed into the main file is invisible, and a concurrent
checkpoint is a torn read. Measured 2026-09-14 on a live `chat.db`:
the immutable open was four messages behind the plain read-only open.
Callers therefore check :func:`wal_frame_bytes` first and refuse a
file whose `-wal` holds frames; a file with no `-wal`, or an empty
one, reads completely and identically either way.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import apsw

from imsg.paths import resolve_path

WAL_HEADER_BYTES = 32
"""Size of a write-ahead log that has been created but never had a frame
written (SQLite's fixed WAL header). A `-wal` at or below this size
holds no frames."""


def wal_sidecar_path(path: Path | str) -> Path:
    """The `-wal` sidecar SQLite would use for `path`, whether or not it exists."""
    return Path(f"{path}-wal")


def wal_frame_bytes(path: Path | str) -> int:
    """Bytes of frame data in `path`'s `-wal` sidecar; 0 when there is no
    sidecar or it holds no frames.

    A `stat` only — deliberately cheap. It cannot tell whether those
    frames are already checkpointed into the main file (that state lives
    in the `-shm` this module refuses to touch), so a nonzero result
    means an immutable open *may* hide committed rows, not that it will.
    The remedy that makes the answer 0 is `PRAGMA wal_checkpoint(TRUNCATE)`
    run by whoever owns the file: it folds every frame into the main file
    and truncates the log to zero bytes.
    """
    try:
        size = wal_sidecar_path(path).stat().st_size
    except FileNotFoundError:
        return 0
    return max(0, size - WAL_HEADER_BYTES)


def readonly_immutable_uri(path: Path | str) -> str:
    """The `file:` URI that opens `path` with `mode=ro&immutable=1`.

    The path is resolved (`~`, symlinks, `..`) and percent-encoded: `?`,
    `#` and `%` are URI metacharacters, and a raw path containing one
    would be silently misread as a query string or fragment.
    """
    return f"file:{quote(resolve_path(path).as_posix(), safe='/')}?mode=ro&immutable=1"


def open_readonly_immutable(path: Path | str) -> apsw.Connection:
    """Open `path` read-only and immutable, leaving its directory untouched.

    Raises `apsw.Error` (`apsw.CantOpenError` for a missing file) exactly
    like a plain `apsw.Connection`, so callers wrap it in their own stage
    error the way they already wrap the plain open.
    """
    return apsw.Connection(
        readonly_immutable_uri(path), flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI
    )


__all__ = [
    "WAL_HEADER_BYTES",
    "open_readonly_immutable",
    "readonly_immutable_uri",
    "wal_frame_bytes",
    "wal_sidecar_path",
]
