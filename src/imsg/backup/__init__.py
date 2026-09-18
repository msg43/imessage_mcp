"""Nightly local recovery copies (SPEC §5.3, §14) — `imsg backup`.

`pipeline.run_backup` is the entry point; `postgres_dump`, `fts_copy`
and `retention` are the three halves worth reading separately (the
scope decision and its reasoning live in `pipeline`'s docstring).
"""

from imsg.backup.pipeline import BackupReport, run_backup
from imsg.backup.retention import (
    BACKUP_SUBDIR,
    DEFAULT_KEEP,
    BackupIndex,
    RetentionPlan,
    index_backups,
    plan_retention,
)

__all__ = [
    "BACKUP_SUBDIR",
    "DEFAULT_KEEP",
    "BackupIndex",
    "BackupReport",
    "RetentionPlan",
    "index_backups",
    "plan_retention",
    "run_backup",
]
