"""Postgres connection, migration runner, and cluster fingerprint (SPEC §5.2, §7)."""

from imsg.db.connection import connect
from imsg.db.migrations import (
    AppliedMigration,
    HashMismatch,
    MigrationFile,
    MigrationPlan,
    PostgresMigrationRunner,
    compute_plan,
    discover_migrations,
)

__all__ = [
    "AppliedMigration",
    "HashMismatch",
    "MigrationFile",
    "MigrationPlan",
    "PostgresMigrationRunner",
    "compute_plan",
    "connect",
    "discover_migrations",
]
