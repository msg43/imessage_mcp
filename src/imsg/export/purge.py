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

import psycopg

from imsg.export.errors import ExportPlanError
from imsg.export.models import PlanPreview, PlanResult
from imsg.export.planner import plan_export, preview_plan

if TYPE_CHECKING:
    from imsg.config.schema import Config


def revoke_person(conn: psycopg.Connection, short_name: str) -> int:
    """Flip `short_name`'s allowlist row to deny both gates and return
    the person_id. The row is RETAINED, never deleted — deleting it
    would erase the record that this person was ever allowed, and
    default-deny already covers absence.

    If the person has no allowlist row yet, one is created with both
    flags false: they were already denied by absence, but the explicit
    row records the revocation decision and prevents a later accidental
    'add' from looking like a first-time add.

    Split from `purge_person` so `preview_purge` can apply exactly this
    revocation inside a transaction it then rolls back.
    """
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
    return person_id


def purge_person(conn: psycopg.Connection, config: Config, short_name: str) -> PlanResult:
    """Revoke `short_name` and plan the deletions. Returns the purge
    plan; the owner then pushes it — a purge run is exempt from the
    §11.4 approval gate (D9.3). The caller commits."""
    revoke_person(conn, short_name)
    return plan_export(conn, config, mode="purge")


def preview_purge(
    conn: psycopg.Connection, config: Config, short_name: str
) -> PlanPreview:
    """What `purge_person` WOULD delete, with nothing left behind.

    The revocation is applied for real inside a transaction and the
    reconcile is run against that revoked state, so the answer comes
    from the same code the real purge runs rather than from a
    parallel "what probably happens" query — then the transaction is
    rolled back and `preview_plan` never wrote anything to disk in the
    first place. The allowlist ends exactly as it started.
    """
    preview: PlanPreview | None = None
    with conn.transaction():
        revoke_person(conn, short_name)
        preview = preview_plan(conn, config)
        raise psycopg.Rollback
    if preview is None:  # pragma: no cover — the block above always assigns first
        raise ExportPlanError("purge preview produced no result")
    return preview


__all__ = ["preview_purge", "purge_person", "revoke_person"]
