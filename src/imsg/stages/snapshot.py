"""S1 — Snapshot the live `chat.db` via SQLite's online-backup API (SPEC §8 S1).

Hard requirement 1 (CLAUDE.md): never write to
`~/Library/Messages/chat.db`. This module is the one place in the
pipeline allowed to touch the live path at all, and it does so through
`apsw` (the pinned, wheel-bundled SQLite build — SPEC §4) opened
**`SQLITE_OPEN_READONLY`**, using SQLite's own online-backup API to copy
into a fresh file under `paths.data_root/snapshots/`. Every other stage
only ever sees a completed snapshot path.

Per SPEC §8 S1 / D6, deliberately **not**:

- `cp` / `shutil.copy` — not WAL-correct against a database Messages.app
  may be actively writing to.
- a read-write source connection — even a connection that never issues
  a write can still take locks a plain-readonly connection would not;
  hard requirement 1 wants a connection that is *incapable* of writing.
- `?immutable=1` on the source URI — that flag asserts the file will
  not change for the lifetime of the connection, which is false for a
  live, actively-synced `chat.db`.

This module never loads config or opens its own long-lived database
handle for anything beyond the backup itself — callers pass in
`live_chat_db` / `data_root` directly (mirrors `imsg.mount.guard`'s
shape: a pure, dependency-injectable function plus small connection/
clock seams for testing) rather than a `Config`, since S1 needs nothing
else from it.
"""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import apsw

from imsg.errors import SnapshotError
from imsg.hashing import sha256_file
from imsg.paths import resolve_path

SNAPSHOT_SUBDIR = "snapshots"
SNAPSHOT_FILENAME = "snapshot.db"
PREVIOUS_SUFFIX = ".previous"

MAX_PREVIOUS_RETAINED = 2
"""SPEC §5.3: 'snapshot.db + retained previous (2 kept)'."""

BUSY_TIMEOUT_MS = 30_000
MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 60.0
FREE_SPACE_MULTIPLIER = 2.0
BACKUP_STEP_PAGES = 100

# The real `chat.db` tables a backed-up file must contain to be treated
# as a genuine snapshot rather than, say, an empty or unrelated SQLite
# file that happened to open cleanly (SPEC §8 S1: "expected core tables
# verified"). These are chat.db's own schema's table names, not this
# project's Postgres schema.
EXPECTED_CORE_TABLES = frozenset(
    {
        "chat",
        "handle",
        "message",
        "attachment",
        "chat_message_join",
        "chat_handle_join",
        "message_attachment_join",
    }
)


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """What S1 handed back for S2 to consume."""

    path: Path
    sha256: str
    byte_size: int
    reused_existing: bool
    """True when this run's backup was byte-identical (by sha256) to
    the prior current snapshot, so the prior file was kept in place
    rather than rotated (SPEC §8 S1 idempotency: "identical sha reuses
    the existing snapshot")."""
    dry_run: bool = False
    """True when this result came from a `run_snapshot(dry_run=True)`
    preview (SPEC §8: "takes --dry-run where writes leave the
    machine") — no backup, verification, or rotation happened; `path`
    is the *would-be* destination and `sha256`/`byte_size` were
    computed directly from the live source file, not a real backup."""


# --- dependency-injection seams (mirrors imsg.mount.guard's DiskutilInfoFn pattern) ---

OpenSourceFn = Callable[[str], "apsw.Connection"]
"""Opens the *source* (live chat.db) connection, already
`SQLITE_OPEN_READONLY` with a busy timeout set. Overridable in tests to
simulate a locked live database (`apsw.BusyError`) without needing a
real second writer process holding a lock."""

SleepFn = Callable[[float], None]


def _default_open_source(path: str) -> apsw.Connection:
    conn = apsw.Connection(path, flags=apsw.SQLITE_OPEN_READONLY)
    conn.set_busy_timeout(BUSY_TIMEOUT_MS)
    return conn


def _free_space_bytes(existing_ancestor: Path) -> int:
    candidate = existing_ancestor
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise SnapshotError(f"cannot determine free space near '{existing_ancestor}'")
        candidate = parent
    return shutil.disk_usage(candidate).free


def _backup_once(*, source_path: Path, dest_path: Path, open_source: OpenSourceFn) -> None:
    """One backup attempt. Raises `apsw.BusyError`/`apsw.LockedError` if
    the source stayed locked; any other exception is a hard failure.
    Leaves no partial file behind on any exception (caller's
    responsibility to unlink `dest_path`, since this function does not
    know the caller's retry policy)."""
    source = open_source(str(source_path))
    try:
        dest = apsw.Connection(str(dest_path))
        try:
            with dest.backup("main", source, "main") as backup:
                while not backup.done:
                    backup.step(BACKUP_STEP_PAGES)
        finally:
            dest.close()
    finally:
        source.close()


def _verify_snapshot(path: Path) -> None:
    """`quick_check` + expected-core-tables verification (SPEC §8 S1).

    Raises `SnapshotError` naming the specific problem. Opens the
    freshly-backed-up file readonly — it is our own temp file at this
    point, not the live source, so there is no hard-requirement-1
    concern, but there is no reason to open it writable either.
    """
    try:
        check = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY)
    except apsw.Error as exc:
        raise SnapshotError(f"backed-up snapshot at '{path}' will not even open: {exc}") from exc
    try:
        rows = list(check.execute("PRAGMA quick_check"))
        result = rows[0][0] if rows else None
        if result != "ok":
            raise SnapshotError(
                f"backed-up snapshot at '{path}' failed 'PRAGMA quick_check': {rows!r}"
            )
        tables = {
            row[0] for row in check.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing = EXPECTED_CORE_TABLES - tables
        if missing:
            raise SnapshotError(
                f"backed-up snapshot at '{path}' is missing expected chat.db table(s) "
                f"{sorted(missing)} — does not look like a real chat.db snapshot"
            )
    finally:
        check.close()


def _rotate_previous(snapshots_dir: Path, current_path: Path, max_previous_retained: int) -> None:
    """Shift `snapshot.db` -> `snapshot.db.previous.1` -> `.previous.2`
    -> deleted, making room for a new current snapshot (SPEC §5.3: "2
    kept"). No-op if there is no current snapshot yet."""
    if not current_path.is_file():
        return

    def _slot(n: int) -> Path:
        return snapshots_dir / f"{SNAPSHOT_FILENAME}{PREVIOUS_SUFFIX}.{n}"

    oldest = _slot(max_previous_retained)
    oldest.unlink(missing_ok=True)
    for n in range(max_previous_retained - 1, 0, -1):
        src = _slot(n)
        if src.is_file():
            src.replace(_slot(n + 1))
    current_path.replace(_slot(1))


def run_snapshot(
    *,
    live_chat_db: Path,
    data_root: Path,
    open_source: OpenSourceFn = _default_open_source,
    sleep: SleepFn = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
    retry_wait_seconds: float = RETRY_WAIT_SECONDS,
    free_space_multiplier: float = FREE_SPACE_MULTIPLIER,
    max_previous_retained: int = MAX_PREVIOUS_RETAINED,
    dry_run: bool = False,
) -> SnapshotResult:
    """Back up `live_chat_db` into `data_root/snapshots/snapshot.db`.

    Does **not** run the encrypted-volume mount gate itself — every CLI
    entry point / service start already runs `imsg.mount.guard` before
    reaching any stage (SPEC §5.4), and a stage re-checking it here
    would need its own `Config`-shaped inputs this function
    deliberately avoids requiring. Callers that invoke this directly
    (rather than through the CLI, once wired) are responsible for
    having run the gate first.

    Raises `SnapshotError` for every documented S1 failure mode (SPEC
    §8 S1): the live database stayed locked past `max_attempts`, the
    destination volume has less than `free_space_multiplier` times the
    live database's size free, or the backed-up file fails its
    post-backup integrity check. On any failure, no partial snapshot
    file is left behind under `snapshots/`.

    `dry_run=True` (SPEC §8: "takes --dry-run where writes leave the
    machine") runs only the read-only precondition checks this
    function already does first — the live database exists, and the
    destination volume has the required free-space margin — and
    returns without ever opening SQLite's online-backup API, creating
    `snapshots/`, writing a temp file, verifying anything, or rotating
    `.previous` files. The returned `SnapshotResult.path` is the
    *would-be* destination (the same fixed `snapshots/snapshot.db`
    path a real run would (re)write); `sha256`/`byte_size` are
    computed directly from the live source file — a best-effort
    stand-in, not a guaranteed match for what a real backup's hash
    would be. **`byte_size` is exact** (S1's backup is content-
    faithful, so the source file's own size is the real answer), but
    **`sha256` is only approximate**: SQLite's online-backup API
    writes a fresh logical copy of the database (its own page
    layout), not a byte-for-byte clone of the source file, so even a
    live db nobody wrote to between the preview and a subsequent real
    run can legitimately produce a different file hash despite
    carrying the same data — hashing the source directly is the
    cheapest available preview, not a promise the two hashes will
    match. `reused_existing` is always `False` in dry-run mode: whether
    a real run would reuse the existing snapshot depends on comparing
    hashes of two backups, and only one side of that comparison (the
    source) was actually read.
    """
    source_path = resolve_path(live_chat_db)
    if not source_path.is_file():
        raise SnapshotError(f"live chat.db not found at '{source_path}'")

    snapshots_dir = resolve_path(data_root) / SNAPSHOT_SUBDIR
    live_size = source_path.stat().st_size

    if dry_run:
        free = _free_space_bytes(snapshots_dir)
        required = int(live_size * free_space_multiplier)
        if free < required:
            raise SnapshotError(
                f"refusing to snapshot: {free} bytes free under '{snapshots_dir}', need >= "
                f"{required} bytes ({free_space_multiplier}x the live chat.db's {live_size} bytes)"
            )
        return SnapshotResult(
            path=snapshots_dir / SNAPSHOT_FILENAME,
            sha256=sha256_file(source_path),
            byte_size=live_size,
            reused_existing=False,
            dry_run=True,
        )

    snapshots_dir.mkdir(parents=True, exist_ok=True)

    free = _free_space_bytes(snapshots_dir)
    required = int(live_size * free_space_multiplier)
    if free < required:
        raise SnapshotError(
            f"refusing to snapshot: {free} bytes free under '{snapshots_dir}', need >= "
            f"{required} bytes ({free_space_multiplier}x the live chat.db's {live_size} bytes)"
        )

    tmp_path = snapshots_dir / f".tmp-snapshot-{uuid.uuid4().hex}.db"

    last_busy_exc: apsw.Error | None = None
    for attempt in range(1, max_attempts + 1):
        last_busy_exc = None
        try:
            _backup_once(source_path=source_path, dest_path=tmp_path, open_source=open_source)
            break
        except (apsw.BusyError, apsw.LockedError) as exc:
            last_busy_exc = exc
            tmp_path.unlink(missing_ok=True)
            if attempt < max_attempts:
                sleep(retry_wait_seconds)
                continue
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    if last_busy_exc is not None:
        raise SnapshotError(
            f"live chat.db at '{source_path}' stayed locked after {max_attempts} "
            f"attempts, {retry_wait_seconds}s apart: {last_busy_exc}"
        ) from last_busy_exc

    try:
        _verify_snapshot(tmp_path)
    except SnapshotError:
        tmp_path.unlink(missing_ok=True)
        raise

    new_sha256 = sha256_file(tmp_path)
    byte_size = tmp_path.stat().st_size
    current_path = snapshots_dir / SNAPSHOT_FILENAME

    if current_path.is_file() and sha256_file(current_path) == new_sha256:
        tmp_path.unlink(missing_ok=True)
        return SnapshotResult(
            path=current_path, sha256=new_sha256, byte_size=byte_size, reused_existing=True
        )

    _rotate_previous(snapshots_dir, current_path, max_previous_retained)
    tmp_path.replace(current_path)  # atomic within the same filesystem

    return SnapshotResult(
        path=current_path, sha256=new_sha256, byte_size=byte_size, reused_existing=False
    )


__all__ = [
    "BUSY_TIMEOUT_MS",
    "EXPECTED_CORE_TABLES",
    "MAX_ATTEMPTS",
    "MAX_PREVIOUS_RETAINED",
    "SNAPSHOT_FILENAME",
    "SNAPSHOT_SUBDIR",
    "OpenSourceFn",
    "SnapshotResult",
    "run_snapshot",
]
