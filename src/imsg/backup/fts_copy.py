"""The FTS5 sidecar half of the nightly recovery copy.

SPEC is in two minds about this file and the two statements are worth
reading together. §5.5's agent table says the daily job is "pg_dump +
fts.db copy"; §14 says "FTS is rebuilt from Postgres (**or** copied with
SQLite's backup API only after a checkpoint + integrity check)". §14 is
the more careful sentence and it is the one implemented here: the
sidecar *is* copied, and the conditions §14 attaches are the whole
implementation.

Why copy at all, given `imsg.embed.fts.rebuild` can regenerate it: the
rebuild streams every segment and attachment chunk out of Postgres and
re-tokenizes them, which on a 700k-message corpus is minutes of work
during an incident, and the copy is small next to the dump beside it.
Why the conditions matter: `cp` of a WAL-mode SQLite database is not a
database (the same lesson `imsg.stages.snapshot` records for `chat.db`,
and the private repo's `archive-messages-corpus.sh` records in shell) —
the main file can be mid-write and the committed-but-uncheckpointed
frames live in a `-wal` the copy does not include.

So: a **passive** checkpoint (never `TRUNCATE` — the sidecar is read by
a KeepAlive MCP server, and a blocking checkpoint at 04:00 would stall
live queries to save a few pages), an integrity check on the source, the
online-backup API for the copy itself, and an integrity check plus a
schema check on **the copy** — because the only thing that proves what
was written is reading what was written.

A **missing** sidecar is not a failure: a machine that has not reached
Phase 3 has no `fts/fts.db`, and the backup set records it as absent. A
**corrupt** sidecar is a failure, and it fails the whole run: propagating
a corrupt index into every one of the 14 retained sets would quietly
destroy exactly the thing these copies exist to protect against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import apsw

from imsg.errors import BackupError
from imsg.hashing import sha256_file
from imsg.sqlite_readonly import open_readonly_immutable

FTS_FILENAME = "fts.db"
FTS_SUBDIR = "fts"

EXPECTED_FTS_TABLES = frozenset({"meta", "seg_map", "seg_fts", "att_map", "att_fts"})
"""The sidecar's own schema (`imsg.embed.fts.schema`). Checked on the
*copy*, so a backup of an empty file that happened to open cleanly is
caught rather than retained for fourteen nights."""

BACKUP_STEP_PAGES = 100
"""Same step size `imsg.stages.snapshot` uses: copy in bounded chunks so
the backup yields between steps instead of holding the source for the
whole transfer."""


@dataclass(frozen=True, slots=True)
class FtsCopyResult:
    path: Path | None
    """`None` when the source sidecar does not exist."""

    byte_size: int | None
    sha256: str | None
    present: bool


def fts_sidecar_path(data_root: Path) -> Path:
    """`$DATA_ROOT/fts/fts.db` — the path `imsg.cli._fts_db_path` builds."""
    return data_root / FTS_SUBDIR / FTS_FILENAME


def _checkpoint_source(source: Path) -> None:
    """`PRAGMA wal_checkpoint(PASSIVE)`, best effort.

    Best effort on purpose: PASSIVE returns immediately when a reader
    holds the WAL rather than waiting, and a sidecar that could not be
    checkpointed is still copied correctly by the backup API — which
    reads through the write-ahead log. The checkpoint is an optimization
    (a smaller `-wal` to walk), not a correctness precondition, so a
    busy sidecar must not fail the nightly job.
    """
    try:
        conn = apsw.Connection(str(source))
    except apsw.Error:
        return
    try:
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except apsw.Error:
        return
    finally:
        conn.close()


def _integrity_check(conn: apsw.Connection, *, label: str, path: Path) -> None:
    # `apsw.Connection(...)` is lazy — it does not touch the file until the
    # first statement, so a file that is not a database at all surfaces
    # here as `apsw.NotADBError` rather than at open time. Caught and
    # converted, because `imsg backup` runs unattended and every failure
    # mode owes the operator one `imsg: …` line, not a traceback.
    try:
        rows = list(conn.execute("PRAGMA integrity_check"))
    except apsw.Error as exc:
        raise BackupError(
            f"the {label} FTS sidecar at '{path}' could not be read: {exc}"
        ) from exc
    result = rows[0][0] if rows else None
    if result != "ok":
        raise BackupError(
            f"the {label} FTS sidecar at '{path}' failed 'PRAGMA integrity_check': "
            f"{rows!r}. Backing a corrupt index into every retained set would "
            f"destroy the thing these copies protect — rebuild it first "
            f"(the sidecar is disposable; it regenerates from Postgres)."
        )


def _verify_copy(path: Path) -> None:
    """Integrity + schema check on the written copy.

    Opened through `imsg.sqlite_readonly` (read-only *and* `immutable=1`)
    for the same reason `imsg.stages.snapshot._verify_snapshot` does: the
    copy inherits the source's WAL journal mode in its header, so a plain
    read-only open would create `-wal`/`-shm` sidecars next to it and
    strand them inside the backup set for good.
    """
    try:
        conn = open_readonly_immutable(path)
    except apsw.Error as exc:
        raise BackupError(f"the copied FTS sidecar at '{path}' will not open: {exc}") from exc
    try:
        _integrity_check(conn, label="copied", path=path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
        missing = EXPECTED_FTS_TABLES - tables
        if missing:
            raise BackupError(
                f"the copied FTS sidecar at '{path}' is missing expected table(s) "
                f"{sorted(missing)} — it is not a usable copy of the sidecar"
            )
    finally:
        conn.close()


def copy_fts_sidecar(*, source: Path, dest: Path) -> FtsCopyResult:
    """Checkpoint, integrity-check, copy via the backup API, verify the copy.

    Returns a result with `present=False` and no path when `source` does
    not exist. Any other failure raises `BackupError` after removing the
    partial copy — a set is never promoted holding one.
    """
    if not source.is_file():
        return FtsCopyResult(path=None, byte_size=None, sha256=None, present=False)

    _checkpoint_source(source)

    try:
        src_conn = apsw.Connection(str(source), flags=apsw.SQLITE_OPEN_READONLY)
    except apsw.Error as exc:
        raise BackupError(f"the live FTS sidecar at '{source}' will not open: {exc}") from exc
    try:
        _integrity_check(src_conn, label="live", path=source)
        try:
            dest_conn = apsw.Connection(str(dest))
        except apsw.Error as exc:
            raise BackupError(f"could not create the FTS copy at '{dest}': {exc}") from exc
        try:
            with dest_conn.backup("main", src_conn, "main") as backup:
                while not backup.done:
                    backup.step(BACKUP_STEP_PAGES)
        except apsw.Error as exc:
            raise BackupError(f"copying the FTS sidecar to '{dest}' failed: {exc}") from exc
        finally:
            dest_conn.close()
    except BackupError:
        dest.unlink(missing_ok=True)
        raise
    finally:
        src_conn.close()

    try:
        _verify_copy(dest)
    except BackupError:
        dest.unlink(missing_ok=True)
        raise

    return FtsCopyResult(
        path=dest,
        byte_size=dest.stat().st_size,
        sha256=sha256_file(dest),
        present=True,
    )


__all__ = [
    "EXPECTED_FTS_TABLES",
    "FTS_FILENAME",
    "FTS_SUBDIR",
    "FtsCopyResult",
    "copy_fts_sidecar",
    "fts_sidecar_path",
]
