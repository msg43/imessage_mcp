"""`imsg backup` — the nightly local recovery copy (SPEC §5.3, §14).

The `com.imsgindex.backup` LaunchAgent runs this at 04:00 every day. It
writes one **set** per run into `$DATA_ROOT/backups/`, verifies every
byte it wrote, and keeps the newest 14 (`imsg.backup.retention`).

--------------------------------------------------------------------
What is in scope, and what is not
--------------------------------------------------------------------

**In: the Postgres dump.** Everything the pipeline derived — identity
resolution, sessionization, segments, embeddings, the allowlist, the
export ledger, the audit log — exists only here. It is also the only
component that can suffer *logical* corruption, which is the one thing
same-device copies protect against (SPEC §14 is explicit that they do
not protect against theft or disk failure). Dumped with `pg_dump`, not
copied from `pg17/`: see `imsg.backup.postgres_dump` for why a running
cluster's data directory is not a backup.

**In: the FTS5 sidecar.** SPEC §5.5 names it in the job description;
SPEC §14 permits copying it only "after a checkpoint + integrity check".
It is rebuildable from Postgres, so it is included for recovery *speed*
rather than necessity — and its conditions are enforced strictly
(`imsg.backup.fts_copy`). Absent is fine; corrupt fails the run.

**Out: the model directory (`models/`).** It is a Hugging Face cache of
**public** weights whose exact identity is already pinned, in git, by
`models/manifest.lock.yaml` — repo plus immutable revision plus
dimension, verified by `imsg models verify`. Backing it up nightly would
copy tens of gigabytes of re-downloadable bytes to defend against a
failure mode the lockfile already covers. Recovery path: `uv sync --extra
models`, then `imsg models verify`.

**Out: the attachment cache (`attachments/`, ~147 GB).** Three reasons,
in descending order of how much they matter:

1. *It would not be a backup, it would be a disk-filling bug.* 147 GB per
   run against 14 retained sets is over two terabytes, on the same volume
   as the thing being protected. Even deduplicated, the first run alone
   doubles the corpus's footprint on the one device that holds it.
2. *It is a cache with a documented rebuild.* The directory is
   content-addressed materialization of originals that live in
   `~/Library/Messages/Attachments` and iCloud;
   `imsg backfill-attachments` rebuilds it and `imsg
   reconcile-attachments` (AT-3) produces the exception manifest saying
   what could not be rebuilt.
3. *Logical corruption is not its failure mode.* A content-addressed blob
   store cannot be damaged by a bad migration the way a relational schema
   can; the risks it actually faces are disk failure and theft, and SPEC
   §14 says in as many words that these copies do not address either.

**The caveat that belongs with that decision, and which this command
prints rather than leaves implicit:** anything iCloud has already purged
exists nowhere but `attachments/`. Excluding it from the nightly copy is
correct, and it is *not* a statement that those bytes are safe. They need
an independently-encrypted off-box archive — which SPEC §14 leaves as an
explicit owner decision, out of scope for v1, and which `imsg status`
already reports as "no disaster backup".

**Out, and flagged rather than silently widened:** `ops/` (approvals,
AT-1 auth-test records, risk acceptances) is small and genuinely
irreplaceable, but SPEC §5.5 scopes this job to "pg_dump + fts.db copy"
and widening a nightly job's scope is an owner decision, not an
implementer's. It is named in this command's output so the gap is
visible. `snapshots/` (regenerable from the live `chat.db` in seconds),
`artifacts/` (transient, 30-day GC), `logs/`, `export/staging/` and
`run/` are out for the ordinary reason that they are derived or
ephemeral.

--------------------------------------------------------------------
How it refuses
--------------------------------------------------------------------

Every precondition is checked **before** the staging directory is
created, and every failure after that point removes the staging
directory and raises. `backups/` therefore only ever contains sets that
passed verification, plus debris left by a process that was killed —
which retention classifies as partial and never deletes.

Idempotency: a run writes to `backups/.incomplete-<uuid4>/` and promotes
it with a single `rename` once the manifest is written. Two runs on one
day produce two differently-named sets and neither can write into the
other's directory; a run that dies leaves a dot-prefixed directory that
no later run will mistake for a backup.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imsg.backup.fts_copy import FtsCopyResult, copy_fts_sidecar, fts_sidecar_path
from imsg.backup.postgres_dump import (
    DUMP_FILENAME,
    DumpResult,
    binary_major_version,
    check_version_compatibility,
    database_size_bytes,
    dump_postgres,
    resolve_pg_dump,
    server_major_version,
)
from imsg.backup.retention import (
    BACKUP_SUBDIR,
    DEFAULT_KEEP,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    STAGING_PREFIX,
    RetentionPlan,
    apply_retention,
    index_backups,
    plan_retention,
)
from imsg.errors import BackupError
from imsg.paths import resolve_path

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config

FREE_SPACE_MULTIPLIER = 1.0
"""Required free space = this many times `pg_database_size()`, plus the
sidecar's size. Deliberately conservative: a custom-format dump is
compressed and normally lands well under the logical database size, so
1.0x leaves real headroom. The guard exists because the failure it
prevents — filling the volume the live index runs on — takes the whole
system down, not just the backup."""

OUT_OF_SCOPE_NOTE = (
    "not backed up: attachments/ (~147 GB content-addressed cache — rebuildable "
    "via 'imsg backfill-attachments'; anything iCloud already purged lives ONLY "
    "there and needs an off-box archive, which this command is not), models/ "
    "(public weights pinned by models/manifest.lock.yaml), ops/ (small and "
    "irreplaceable, but outside SPEC §5.5's scope for this job — owner decision), "
    "snapshots/, artifacts/, logs/, export/staging/, run/"
)

SAME_DEVICE_CAVEAT = (
    "these copies share the physical device with the data they copy: they protect "
    "against logical corruption, NOT theft or disk failure (SPEC §14 — 'no disaster "
    "backup' until an independently encrypted off-box destination is chosen)"
)


@dataclass(frozen=True, slots=True)
class BackupReport:
    set_path: Path | None
    """`None` on a dry run — nothing was promoted."""

    created_at: str
    dump: DumpResult | None
    fts: FtsCopyResult | None
    retention: RetentionPlan
    deleted: tuple[Path, ...]
    total_bytes: int
    free_bytes_before: int
    required_bytes: int
    dry_run: bool = False


def backups_dir_for(config: Config) -> Path:
    return resolve_path(config.paths.data_root) / BACKUP_SUBDIR


def _free_space_bytes(path: Path) -> int:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise BackupError(f"cannot determine free space near '{path}'")
        candidate = parent
    try:
        return shutil.disk_usage(candidate).free
    except OSError as exc:
        raise BackupError(f"cannot determine free space near '{candidate}': {exc}") from exc


def _prepare_destination(backups_dir: Path) -> None:
    """Refuse unless `backups/` exists (or can be created) and is writable.

    The writability probe creates and removes a real file rather than
    consulting `os.access`, which reports what the *permission bits* say
    and not what the filesystem will actually allow — a read-only mount,
    an ACL, or a full volume all pass `os.access(W_OK)` and then fail the
    first write.
    """
    parent = backups_dir.parent
    if not parent.exists():
        raise BackupError(
            f"refusing to back up: the data root '{parent}' does not exist "
            f"(is the encrypted volume mounted?)"
        )
    if not parent.is_dir():
        raise BackupError(f"refusing to back up: '{parent}' is not a directory")
    try:
        backups_dir.mkdir(parents=False, exist_ok=True)
    except OSError as exc:
        raise BackupError(
            f"refusing to back up: could not create the backups directory "
            f"'{backups_dir}': {exc}"
        ) from exc
    if not backups_dir.is_dir():
        raise BackupError(
            f"refusing to back up: '{backups_dir}' exists but is not a directory"
        )
    probe = backups_dir / f".writable-probe-{uuid.uuid4().hex}"
    try:
        probe.write_bytes(b"")
    except OSError as exc:
        raise BackupError(
            f"refusing to back up: the backups directory '{backups_dir}' is not "
            f"writable: {exc}"
        ) from exc
    finally:
        probe.unlink(missing_ok=True)


def _check_free_space(*, backups_dir: Path, db_bytes: int, fts_bytes: int) -> tuple[int, int]:
    required = int(db_bytes * FREE_SPACE_MULTIPLIER) + fts_bytes
    free = _free_space_bytes(backups_dir)
    if free < required:
        raise BackupError(
            f"refusing to back up: {free} bytes free under '{backups_dir}', need >= "
            f"{required} bytes ({FREE_SPACE_MULTIPLIER}x the {db_bytes}-byte database "
            f"plus the {fts_bytes}-byte FTS sidecar). Free space or prune "
            f"'{backups_dir}' before the next run — a dump that fills this volume "
            f"takes the live index down with it."
        )
    return free, required


def _set_name(now: datetime) -> str:
    return f"backup-{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _write_manifest(
    staging: Path,
    *,
    created_at: str,
    dump: DumpResult,
    fts: FtsCopyResult,
) -> None:
    """Written LAST. Its presence, and only its presence, marks the set complete."""
    files: list[dict[str, Any]] = [
        {
            "name": dump.path.name,
            "kind": "postgres-dump",
            "format": "pg_dump custom (-Fc)",
            "bytes": dump.byte_size,
            "sha256": dump.sha256,
            "pg_dump_major": dump.pg_dump_major,
            "server_major": dump.server_major,
            "tables_in_toc": list(dump.tables_in_toc),
            "verified": "pg_restore read-back of every block, plus expected tables in TOC",
        }
    ]
    if fts.present and fts.path is not None:
        files.append(
            {
                "name": fts.path.name,
                "kind": "fts5-sidecar",
                "format": "sqlite (online-backup API)",
                "bytes": fts.byte_size,
                "sha256": fts.sha256,
                "verified": "PRAGMA integrity_check + expected tables, on the copy",
            }
        )
    manifest = {
        "format": MANIFEST_FORMAT,
        "complete": True,
        "created_at": created_at,
        "files": files,
        "fts_sidecar_present": fts.present,
        "out_of_scope": OUT_OF_SCOPE_NOTE,
        "caveat": SAME_DEVICE_CAVEAT,
    }
    (staging / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_backup(
    *,
    conn: psycopg.Connection,
    config: Config,
    keep: int = DEFAULT_KEEP,
    pg_dump_binary: Path | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
) -> BackupReport:
    """Write one verified backup set, then retain the newest `keep`.

    `conn` must already be verified as the dedicated cluster
    (`imsg.db.fingerprint.verify_data_directory`) — the caller owns it
    and closes it. Raises `BackupError` for every documented refusal;
    leaves `backups/` untouched apart from the promoted set and whatever
    retention deleted.

    `dry_run=True` runs every precondition — destination, writability,
    `pg_dump` presence and version, free space — and reports the
    retention plan, without creating a staging directory, dumping, or
    deleting anything.
    """
    now = now or datetime.now(UTC)
    created_at = now.astimezone(UTC).isoformat()
    data_root = resolve_path(config.paths.data_root)
    backups_dir = data_root / BACKUP_SUBDIR

    # --- preconditions, all before anything is created -------------------
    _prepare_destination(backups_dir)
    pg_dump = resolve_pg_dump(pg_dump_binary)
    server_major = server_major_version(conn)
    check_version_compatibility(
        pg_dump_major=binary_major_version(pg_dump),
        server_major=server_major,
        pg_dump=pg_dump,
    )
    sidecar = fts_sidecar_path(data_root)
    fts_bytes = sidecar.stat().st_size if sidecar.is_file() else 0
    free, required = _check_free_space(
        backups_dir=backups_dir, db_bytes=database_size_bytes(conn), fts_bytes=fts_bytes
    )

    if dry_run:
        plan = plan_retention(index_backups(backups_dir), keep=keep)
        return BackupReport(
            set_path=None,
            created_at=created_at,
            dump=None,
            fts=None,
            retention=plan,
            deleted=(),
            total_bytes=0,
            free_bytes_before=free,
            required_bytes=required,
            dry_run=True,
        )

    # --- write into staging, promote only on success ---------------------
    staging = backups_dir / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        dump = dump_postgres(
            conn=conn,
            dsn=config.database.dsn,
            password=config.database.password.resolve(),
            dest=staging / DUMP_FILENAME,
            pg_dump=pg_dump,
            server_major=server_major,
        )
        fts = copy_fts_sidecar(source=sidecar, dest=staging / sidecar.name)
        _write_manifest(staging, created_at=created_at, dump=dump, fts=fts)
        final = backups_dir / _set_name(now)
        staging.rename(final)
    except BaseException:
        # Includes KeyboardInterrupt/SystemExit: a job killed mid-dump must
        # not leave a directory behind that a later run has to reason about.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    total_bytes = dump.byte_size + (fts.byte_size or 0)

    # --- retention, over the index that now includes this run's set ------
    plan = plan_retention(index_backups(backups_dir), keep=keep)
    deleted = apply_retention(plan)

    return BackupReport(
        set_path=final,
        created_at=created_at,
        dump=dump,
        fts=fts,
        retention=plan,
        deleted=deleted,
        total_bytes=total_bytes,
        free_bytes_before=free,
        required_bytes=required,
    )


__all__ = [
    "FREE_SPACE_MULTIPLIER",
    "OUT_OF_SCOPE_NOTE",
    "SAME_DEVICE_CAVEAT",
    "BackupReport",
    "backups_dir_for",
    "run_backup",
]
