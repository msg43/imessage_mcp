"""Migration discovery, planning, and the Postgres runner (SPEC §7.1).

Two layers, deliberately separated for testability:

- :func:`discover_migrations` and :func:`compute_plan` are pure
  functions over dataclasses — no database connection required. This
  is what `tests/test_migrations.py` exercises without Postgres.
- :class:`PostgresMigrationRunner` is the thin psycopg-backed layer
  that actually talks to a database. It is exercised by
  `tests/test_migrations_integration.py`, which skips cleanly when no
  Postgres is reachable (see that module for the connection probe).

Rules transcribed from SPEC §7.1: migrations are strictly ordered,
immutable once merged; the runner applies each pending file inside its
own transaction, records `(version, sha256, applied_at)` in
`schema_migrations`, and refuses to run at all if a previously-applied
file's sha256 no longer matches what's on disk. No down-migrations —
roll forward only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.errors import MigrationError

if TYPE_CHECKING:
    import psycopg

MIGRATION_FILENAME_RE = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z][a-z0-9_]*)\.sql$")

SCHEMA_MIGRATIONS_TABLE = "schema_migrations"


@dataclass(frozen=True, slots=True)
class MigrationFile:
    """A migration as it exists on disk right now."""

    version: int
    name: str
    path: Path
    sha256: str
    sql: str


@dataclass(frozen=True, slots=True)
class AppliedMigration:
    """A migration as recorded in `schema_migrations`."""

    version: int
    sha256: str
    applied_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class HashMismatch:
    """An applied migration whose disk content no longer matches its recorded hash."""

    version: int
    applied_sha256: str
    disk_sha256: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    applied: tuple[AppliedMigration, ...]
    pending: tuple[MigrationFile, ...]
    mismatches: tuple[HashMismatch, ...]

    @property
    def is_clean(self) -> bool:
        return not self.mismatches


def discover_migrations(migrations_dir: Path) -> list[MigrationFile]:
    """Parse every `NNNN_description.sql` file in `migrations_dir`, sorted by version.

    Raises `MigrationError` for a malformed filename or a duplicate
    version number — both indicate a broken migration set rather than
    something to skip quietly.
    """
    if not migrations_dir.is_dir():
        raise MigrationError(f"migrations directory not found: '{migrations_dir}'")

    files: list[MigrationFile] = []
    seen_versions: dict[int, Path] = {}
    for path in sorted(migrations_dir.glob("*.sql")):
        m = MIGRATION_FILENAME_RE.match(path.name)
        if not m:
            raise MigrationError(
                f"migration filename '{path.name}' does not match the required "
                f"'NNNN_description.sql' pattern (SPEC §7.1)"
            )
        version = int(m.group("version"))
        if version in seen_versions:
            raise MigrationError(
                f"duplicate migration version {version:04d}: "
                f"'{seen_versions[version].name}' and '{path.name}'"
            )
        seen_versions[version] = path
        sql = path.read_text()
        sha256 = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        files.append(
            MigrationFile(
                version=version, name=m.group("name"), path=path, sha256=sha256, sql=sql
            )
        )
    files.sort(key=lambda f: f.version)
    return files


def compute_plan(
    discovered: list[MigrationFile], applied: list[AppliedMigration]
) -> MigrationPlan:
    """Pure diff between what's on disk and what `schema_migrations` recorded."""
    applied_by_version = {a.version: a for a in applied}
    discovered_by_version = {d.version: d for d in discovered}

    mismatches: list[HashMismatch] = []
    for version, a in sorted(applied_by_version.items()):
        d = discovered_by_version.get(version)
        if d is None:
            mismatches.append(
                HashMismatch(version=version, applied_sha256=a.sha256, disk_sha256="<file missing>")
            )
        elif d.sha256 != a.sha256:
            mismatches.append(
                HashMismatch(version=version, applied_sha256=a.sha256, disk_sha256=d.sha256)
            )

    pending = tuple(d for d in discovered if d.version not in applied_by_version)
    return MigrationPlan(
        applied=tuple(sorted(applied, key=lambda a: a.version)),
        pending=pending,
        mismatches=tuple(mismatches),
    )


def format_mismatches(mismatches: tuple[HashMismatch, ...]) -> str:
    return "; ".join(
        f"version {m.version:04d}: applied sha256 {m.applied_sha256[:12]}… != "
        f"disk sha256 {m.disk_sha256[:12] if m.disk_sha256 != '<file missing>' else m.disk_sha256}…"
        for m in mismatches
    )


class PostgresMigrationRunner:
    """The live-database half of the migration runner.

    Takes an already-open `psycopg.Connection` (autocommit off — each
    migration applies inside its own transaction) rather than owning
    connection lifecycle itself, so callers control pooling/retry.
    """

    def __init__(self, conn: psycopg.Connection, migrations_dir: Path) -> None:
        self._conn = conn
        self._migrations_dir = migrations_dir

    def fetch_applied(self) -> list[AppliedMigration]:
        """Read `schema_migrations`, tolerating "table doesn't exist yet".

        `schema_migrations` is itself created by migration 0001 — before
        that runs, "no migrations applied" is the correct answer, not an
        error.
        """
        with self._conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"public.{SCHEMA_MIGRATIONS_TABLE}",))
            row = cur.fetchone()
            if row is None or row[0] is None:
                return []
            cur.execute(
                f"SELECT version, sha256, applied_at FROM {SCHEMA_MIGRATIONS_TABLE} "
                f"ORDER BY version"
            )
            return [
                AppliedMigration(version=v, sha256=s, applied_at=a) for v, s, a in cur.fetchall()
            ]

    def plan(self) -> MigrationPlan:
        discovered = discover_migrations(self._migrations_dir)
        applied = self.fetch_applied()
        return compute_plan(discovered, applied)

    def apply_pending(self) -> list[MigrationFile]:
        """Apply every pending migration, in order, each in its own transaction.

        Idempotent: if nothing is pending, this is a no-op. Refuses to
        apply anything (raises `MigrationError`, applies nothing) if any
        previously-applied migration's disk content has drifted from its
        recorded hash.
        """
        current_plan = self.plan()
        if not current_plan.is_clean:
            raise MigrationError(
                "refusing to run: one or more applied migrations no longer match "
                f"disk ({format_mismatches(current_plan.mismatches)}) — migrations "
                f"are immutable once merged (SPEC §7.1)"
            )

        applied_now: list[MigrationFile] = []
        for migration in current_plan.pending:
            with self._conn.transaction(), self._conn.cursor() as cur:
                cur.execute(migration.sql)
                cur.execute(
                    f"INSERT INTO {SCHEMA_MIGRATIONS_TABLE} (version, sha256) "
                    f"VALUES (%s, %s)",
                    (migration.version, migration.sha256),
                )
            applied_now.append(migration)
        return applied_now

    def verify(self) -> MigrationPlan:
        """Re-check state against disk; raises if anything is inconsistent.

        Unlike `plan()`, this raises on mismatches rather than reporting
        them, for use as a hard gate (`imsg migrate --verify`).
        """
        current_plan = self.plan()
        if not current_plan.is_clean:
            raise MigrationError(
                f"verification failed: {format_mismatches(current_plan.mismatches)}"
            )
        return current_plan


__all__ = [
    "AppliedMigration",
    "HashMismatch",
    "MigrationFile",
    "MigrationPlan",
    "PostgresMigrationRunner",
    "compute_plan",
    "discover_migrations",
    "format_mismatches",
]
