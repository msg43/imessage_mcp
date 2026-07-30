"""`imsg export --purge-person` (SPEC §11.4): revocation as a
first-class reconciled operation.

Flow: flag the person's allowlist row to deny (the row is RETAINED for
audit — deleting it would erase the record that they were ever
allowed), then produce a `mode='purge'` plan whose reconciliation
naturally emits deletes for every now-ineligible document. The plan
contains deletes; per D9 a purge run is exempt from the §11.4 approval
gate (retraction only narrows scope), though every drift check still
applies. `push_export` then
executes the deletes and positively verifies absence by document id
before recording anything as purged.

Honesty note (also in the package docstring): this removes content
from the Discovery Engine index and the GCS bucket. Copies already
swept into organizational retention, backups, or another person's
hands are beyond reach. Revocation is damage control; the gate at
export time is the actual protection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imsg.export.errors import ExportPlanError
from imsg.export.models import PlanResult
from imsg.export.planner import plan_export

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config


def purge_person(conn: psycopg.Connection, config: Config, short_name: str) -> PlanResult:
    """Revoke `short_name` and plan the deletions. Returns the purge
    plan; the owner then approves and pushes it like any other run.
    The caller commits.

    If the person has no allowlist row yet, one is created with both
    flags false: they were already denied by absence (default deny),
    but the explicit row records the revocation decision and prevents
    a later accidental 'add' from looking like a first-time add."""
    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE short_name = %s", (short_name,))
        row = cur.fetchone()
        if row is None:
            raise ExportPlanError(
                f"no person with short_name '{short_name}' — nothing to revoke"
            )
        person_id = int(row[0])
        cur.execute(
            """
            INSERT INTO allowlist_person
                (person_id, text_allowed, attachments_allowed, note)
            VALUES (%s, false, false, 'revoked via purge-person')
            ON CONFLICT (person_id) DO UPDATE SET
                text_allowed = false,
                attachments_allowed = false
            """,
            (person_id,),
        )
    return plan_export(conn, config, mode="purge")


__all__ = ["purge_person"]
