"""Two-sided cluster fingerprint (SPEC §5.2, D6) — CLAUDE.md non-negotiable #6.

v1.0 of the spec had an `i_know_this_is_the_dedicated_instance` config
bypass around the "is this really our own Postgres instance, not some
other system's" check. D6 removed it: there is no override flag here,
only "the filesystem-side and database-side UUIDs already match" or a
hard failure. Port 5433 (checked at config-parse time,
`imsg.config.schema`) is a useful default, not proof of isolation —
this module is the actual proof, checked against a live connection.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.errors import ClusterFingerprintError
from imsg.paths import is_contained_in, join_under_root, resolve_path

if TYPE_CHECKING:
    import psycopg

PG_DATA_SUBDIR = "pg17"


def verify_data_directory(conn: psycopg.Connection, data_root: Path) -> Path:
    """The connected instance's `data_directory` must resolve under `$DATA_ROOT/pg17`."""
    with conn.cursor() as cur:
        cur.execute("SHOW data_directory")
        row = cur.fetchone()
    if row is None:
        raise ClusterFingerprintError("could not read 'SHOW data_directory' from the connection")

    reported = resolve_path(Path(str(row[0])))
    expected_root = resolve_path(join_under_root(data_root, PG_DATA_SUBDIR))
    if reported != expected_root and not is_contained_in(reported, expected_root):
        raise ClusterFingerprintError(
            f"connected Postgres instance's data_directory ('{reported}') does not "
            f"resolve under '{expected_root}' — refusing to treat it as the "
            f"dedicated imessage-index cluster"
        )
    return reported


def ensure_cluster_fingerprint(
    conn: psycopg.Connection, data_root: Path, fingerprint_relpath: Path
) -> str:
    """Bootstrap the fingerprint on first run, or verify it on every run after.

    First run (no filesystem file, no `imsg_meta.cluster_uuid` row):
    generates a UUID and writes both sides in one go. Every later call:
    both sides must exist and agree, or this raises
    `ClusterFingerprintError` — there is no bypass.

    Requires migration 0001 to already be applied (`imsg_meta` must
    exist); callers should run this after `PostgresMigrationRunner.
    apply_pending()`, not before.
    """
    verify_data_directory(conn, data_root)

    fingerprint_path = resolve_path(join_under_root(data_root, fingerprint_relpath))

    with conn.cursor() as cur:
        cur.execute("SELECT value FROM imsg_meta WHERE key = 'cluster_uuid'")
        row = cur.fetchone()
    db_uuid: str | None = row[0] if row else None

    file_uuid: str | None = (
        fingerprint_path.read_text().strip() if fingerprint_path.is_file() else None
    )

    if db_uuid is None and file_uuid is None:
        new_uuid = str(uuid.uuid4())
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path.write_text(new_uuid + "\n")
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO imsg_meta (key, value) VALUES ('cluster_uuid', %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (new_uuid,),
            )
        conn.commit()
        return new_uuid

    if db_uuid is None or file_uuid is None or db_uuid != file_uuid:
        raise ClusterFingerprintError(
            f"cluster fingerprint mismatch: filesystem side ('{fingerprint_path}') "
            f"= {file_uuid!r}, database side (imsg_meta.cluster_uuid) = {db_uuid!r} "
            f"— refusing to proceed; there is no bypass for this check (D6)"
        )
    return db_uuid


__all__ = ["ensure_cluster_fingerprint", "verify_data_directory"]
