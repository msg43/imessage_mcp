"""The Postgres half of the nightly recovery copy: `pg_dump`, then a read-back.

SPEC §14: "nightly `pg_dump`". Deliberately **not** a copy of the live
`pg17/` data directory: a running cluster's files are not a backup —
they are a mid-flight snapshot of a process holding dirty buffers and a
write-ahead log, and restoring them is undefined unless the cluster was
stopped or the copy went through `pg_basebackup`'s backup protocol. A
logical dump is also the right shape for what these copies are *for*
(SPEC §14: they "protect against logical corruption, **not** theft or
disk failure"), because it survives the one class of damage a file copy
faithfully reproduces.

Three things here exist because the obvious version of each is wrong:

**The version pre-check.** `pg_dump` refuses to dump a server newer than
itself. It refuses *after* creating the output file, so a naive run
leaves a zero-byte `postgres.dump` behind and reports a failure the
operator will not see until the restore drill. Measured on the build
machine 2026-09-18: `pg_dump` 16.13 against a 17.9 server exits nonzero
with "aborting because of server version mismatch" and leaves a 0-byte
file. This is not a hypothetical — Homebrew puts `postgresql@16`'s
`pg_dump` first on `PATH` while SPEC §5.3 pins the instance at `pg17/`.
So the major versions are compared before the subprocess starts, and the
error names both versions and the fix.

**The password never reaches `argv`.** `pg_dump` takes no password
option; it reads `PGPASSWORD` from the environment (or `.pgpass`). The
environment of a child process is not world-readable on macOS the way
`argv` is (`ps -ww` shows every argument to every user), so the resolved
secret goes in `env` and nowhere else, and this module never logs it.

**Verification reads the whole archive.** `pg_restore --list` is the
tempting check and it is worthless for this: measured 2026-09-18, a
custom-format dump **cut in half** still lists all of its TOC entries
and exits 0, because the table of contents lives at the front of the
archive and the data blocks follow it. `pg_restore -f <devnull>` reads
and decompresses every block; the same experiment showed it catching a
half truncation, a **one-byte** truncation, and a single flipped byte
2 kB from the end. The TOC listing is still run — but only for what it
is actually good for, which is asserting the expected tables are present
by name.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from imsg.errors import BackupError
from imsg.hashing import sha256_file

DUMP_FILENAME = "postgres.dump"
DUMP_FORMAT_FLAG = "-Fc"
"""Custom format: compressed, and the only format `pg_restore` can read
selectively (a plain SQL dump can only be replayed whole, and cannot be
verified without a live server to replay it into)."""

PG_DUMP_TIMEOUT_SECONDS = 3 * 60 * 60
PG_RESTORE_TIMEOUT_SECONDS = 3 * 60 * 60

EXPECTED_DUMPED_TABLES = frozenset({"message", "chat", "person", "segment"})
"""Tables whose absence means the dump is not of this project's schema —
the same shape of check `imsg.stages.snapshot.EXPECTED_CORE_TABLES`
makes for a chat.db snapshot. Deliberately a small, stable core: a
longer list would turn every future migration into a false alarm here."""

_VERSION_RE = re.compile(r"\(PostgreSQL\)\s+(?P<major>\d+)")
_TOC_TABLE_RE = re.compile(r"^\d+;\s+\d+\s+\d+\s+TABLE\s+\S+\s+(?P<name>\S+)\s", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class DumpResult:
    path: Path
    byte_size: int
    sha256: str
    tables_in_toc: tuple[str, ...]
    pg_dump_major: int
    server_major: int


def resolve_pg_dump(explicit: Path | None = None) -> Path:
    """`shutil.which('pg_dump')`, or a `BackupError` naming the gap.

    Never guesses a hardcoded Homebrew/Postgres.app path — same rule
    `imsg.cli._resolve_binary_or_die` follows for `install-agents`. An
    explicit path is checked for existence rather than trusted.
    """
    if explicit is not None:
        candidate = explicit.expanduser()
        if not candidate.is_file():
            raise BackupError(f"pg_dump not found at '{candidate}'")
        return candidate
    found = shutil.which("pg_dump")
    if found is None:
        raise BackupError(
            "'pg_dump' was not found on $PATH — install the PostgreSQL client "
            "tools (or pass --pg-dump <path>) before running 'imsg backup'"
        )
    return Path(found)


def _sibling_pg_restore(pg_dump: Path) -> Path:
    """`pg_restore` from the *same* installation as `pg_dump`.

    Not `shutil.which('pg_restore')`: on a machine with several
    PostgreSQL versions installed, `PATH` order can hand back a
    `pg_restore` of a different major than the `pg_dump` that wrote the
    archive, and verification would then fail (or, worse, pass) for a
    reason that has nothing to do with the dump.
    """
    candidate = pg_dump.parent / "pg_restore"
    if not candidate.is_file():
        raise BackupError(
            f"'pg_restore' is not installed next to '{pg_dump}' — the dump "
            f"cannot be verified without it, and an unverified dump is not a backup"
        )
    return candidate


def binary_major_version(binary: Path) -> int:
    """Major version reported by `<binary> --version`."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackupError(f"could not run '{binary} --version': {exc}") from exc
    if proc.returncode != 0:
        raise BackupError(f"'{binary} --version' exited {proc.returncode}: {proc.stderr.strip()}")
    match = _VERSION_RE.search(proc.stdout)
    if match is None:
        raise BackupError(f"could not parse a version out of '{binary} --version': {proc.stdout!r}")
    return int(match.group("major"))


def check_version_compatibility(*, pg_dump_major: int, server_major: int, pg_dump: Path) -> None:
    """Refuse before the subprocess runs if `pg_dump` is older than the server.

    A newer `pg_dump` against an older server is supported by PostgreSQL
    and is not refused here.
    """
    if pg_dump_major < server_major:
        raise BackupError(
            f"refusing to back up: '{pg_dump}' is PostgreSQL {pg_dump_major} but the "
            f"instance is PostgreSQL {server_major}. pg_dump cannot dump a newer "
            f"server — it aborts partway and leaves a zero-byte file that looks like "
            f"a backup. Install the PostgreSQL {server_major} client tools and put "
            f"them first on $PATH (or pass --pg-dump <path> to that version's pg_dump)."
        )


def server_major_version(conn: object) -> int:
    """Major version of the connected server, from `server_version_num`.

    Typed loosely because the only thing wanted from the connection is
    one integer; the caller has already verified it is the dedicated
    cluster (`imsg.db.fingerprint.verify_data_directory`).
    """
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("SHOW server_version_num")
        row = cur.fetchone()
    if row is None:  # pragma: no cover - a server that answers SHOW with no row
        raise BackupError("the instance did not report 'server_version_num'")
    return int(row[0]) // 10000


def database_size_bytes(conn: object) -> int:
    """`pg_database_size(current_database())` — the free-space floor."""
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("SELECT pg_database_size(current_database())")
        row = cur.fetchone()
    if row is None:  # pragma: no cover
        raise BackupError("the instance did not report its database size")
    return int(row[0])


def run_pg_dump(*, dsn: str, password: str, dest: Path, pg_dump: Path) -> None:
    """Dump `dsn` into `dest` in custom format.

    On any failure the (possibly zero-byte, possibly partial) output file
    is removed before raising, so no caller can mistake a failed dump for
    a small one.
    """
    env = dict(os.environ)
    env["PGPASSWORD"] = password
    try:
        proc = subprocess.run(
            [
                str(pg_dump),
                DUMP_FORMAT_FLAG,
                "--no-password",
                "--file",
                str(dest),
                dsn,
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=PG_DUMP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        dest.unlink(missing_ok=True)
        raise BackupError(
            f"pg_dump did not finish within {PG_DUMP_TIMEOUT_SECONDS}s and was killed"
        ) from exc
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise BackupError(f"could not run '{pg_dump}': {exc}") from exc
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        raise BackupError(f"pg_dump exited {proc.returncode}: {proc.stderr.strip() or '(no stderr)'}")
    if not dest.is_file():
        raise BackupError(f"pg_dump reported success but wrote no file at '{dest}'")


def _toc_tables(*, dump: Path, pg_restore: Path) -> tuple[str, ...]:
    try:
        proc = subprocess.run(
            [str(pg_restore), "--list", str(dump)],
            capture_output=True,
            text=True,
            check=False,
            timeout=PG_RESTORE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackupError(f"could not list the dump's table of contents: {exc}") from exc
    if proc.returncode != 0:
        raise BackupError(
            f"the dump at '{dump}' has no readable table of contents "
            f"(pg_restore --list exited {proc.returncode}): {proc.stderr.strip()}"
        )
    return tuple(sorted({m.group("name") for m in _TOC_TABLE_RE.finditer(proc.stdout)}))


def _read_whole_archive(*, dump: Path, pg_restore: Path) -> None:
    """`pg_restore -f <devnull>`: decompress every block, write nothing.

    This is the check that can actually fail. Restoring to a file rather
    than a database means no server is contacted and nothing is created
    — the archive is read end to end and the generated SQL is discarded.
    """
    try:
        proc = subprocess.run(
            [str(pg_restore), "--file", os.devnull, str(dump)],
            capture_output=True,
            text=True,
            check=False,
            timeout=PG_RESTORE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackupError(f"could not read back the dump at '{dump}': {exc}") from exc
    if proc.returncode != 0:
        raise BackupError(
            f"the dump at '{dump}' failed its read-back and is NOT a usable backup "
            f"(pg_restore exited {proc.returncode}): {proc.stderr.strip() or '(no stderr)'}"
        )


def verify_dump(*, dump: Path, pg_dump: Path) -> tuple[str, ...]:
    """Read the whole archive back, then assert the expected tables are in it.

    Returns the table names found in the table of contents. Raises
    `BackupError` naming the specific failure — truncation, corruption,
    an unreadable TOC, or a dump of something that is not this schema.
    """
    pg_restore = _sibling_pg_restore(pg_dump)
    if not dump.is_file():
        raise BackupError(f"no dump file at '{dump}' to verify")
    if dump.stat().st_size == 0:
        raise BackupError(f"the dump at '{dump}' is zero bytes — pg_dump wrote nothing")
    _read_whole_archive(dump=dump, pg_restore=pg_restore)
    tables = _toc_tables(dump=dump, pg_restore=pg_restore)
    missing = EXPECTED_DUMPED_TABLES - set(tables)
    if missing:
        raise BackupError(
            f"the dump at '{dump}' is missing expected table(s) {sorted(missing)} — "
            f"it does not look like a dump of the imessage-index schema"
        )
    return tables


def dump_postgres(
    *,
    conn: object,
    dsn: str,
    password: str,
    dest: Path,
    pg_dump: Path,
    server_major: int,
) -> DumpResult:
    """Version-check, dump, verify. Any failure removes `dest` and raises."""
    del conn  # the caller already read what it needed off the connection
    pg_dump_major = binary_major_version(pg_dump)
    check_version_compatibility(
        pg_dump_major=pg_dump_major, server_major=server_major, pg_dump=pg_dump
    )
    run_pg_dump(dsn=dsn, password=password, dest=dest, pg_dump=pg_dump)
    try:
        tables = verify_dump(dump=dest, pg_dump=pg_dump)
    except BackupError:
        dest.unlink(missing_ok=True)
        raise
    return DumpResult(
        path=dest,
        byte_size=dest.stat().st_size,
        sha256=sha256_file(dest),
        tables_in_toc=tables,
        pg_dump_major=pg_dump_major,
        server_major=server_major,
    )


__all__ = [
    "DUMP_FILENAME",
    "EXPECTED_DUMPED_TABLES",
    "DumpResult",
    "binary_major_version",
    "check_version_compatibility",
    "database_size_bytes",
    "dump_postgres",
    "resolve_pg_dump",
    "run_pg_dump",
    "server_major_version",
    "verify_dump",
]
