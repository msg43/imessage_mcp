"""MCP tool-call handlers (SPEC §10.2): each function takes
already-JSON-Schema-validated arguments and returns a JSON-serializable
dict, or raises an `imsg.retrieval.RetrievalError` — mapped to the
SPEC §10.1 error model by `imsg.mcp.tools.dispatch`, not here. Four of
the five are thin adapters over `imsg.retrieval.RetrievalService`;
`check_permissions` is diagnostics, not retrieval, so it is
implemented directly against `imsg.diagnostics` plus the sync/watermark
state those pipeline stages now write (SPEC §10.2's `check_permissions`
response shape).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imsg.diagnostics import check_at_rest_posture, check_full_disk_access, check_mount
from imsg.errors import ClusterFingerprintError

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config
    from imsg.retrieval import AccessContext
    from imsg.retrieval.service import RetrievalService

# --------------------------------------------------------------------------
# search_messages / get_conversation / list_people / get_attachment_text
# --------------------------------------------------------------------------


def search_messages(
    service: RetrievalService, context: AccessContext, params: dict[str, Any]
) -> dict[str, Any]:
    result = service.search_messages(
        context,
        query=params["query"],
        people=params.get("people"),
        after=params.get("after"),
        before=params.get("before"),
        has_attachment=params.get("has_attachment"),
        limit=params.get("limit"),
    )
    return {
        "results": result.results,
        "candidate_lists": result.candidate_lists,
        "scan_cap_reached": result.scan_cap_reached,
    }


def get_conversation(
    service: RetrievalService, context: AccessContext, params: dict[str, Any]
) -> dict[str, Any]:
    return service.get_conversation(
        context,
        thread_id=params["thread_id"],
        anchor=params.get("anchor"),
        window=params.get("window", 20),
    )


def list_people(
    service: RetrievalService, context: AccessContext, params: dict[str, Any]
) -> dict[str, Any]:
    return service.list_people(
        context,
        query=params.get("query"),
        limit=params.get("limit", 100),
        include_handles=params.get("include_handles", False),
    )


def get_attachment_text(
    service: RetrievalService, context: AccessContext, params: dict[str, Any]
) -> dict[str, Any]:
    return service.get_attachment_text(context, attachment_key=params["attachment_key"])


# --------------------------------------------------------------------------
# check_permissions
# --------------------------------------------------------------------------


def _contacts_access_status() -> bool | None:
    """Best-effort, non-prompting Contacts TCC check (mirrors
    `imsg.stages.identity._default_contacts_importer`'s own
    authorization check, re-implemented rather than imported since
    that is a module-private helper): `None` when the `Contacts`
    framework itself is unusable in this environment (pyobjc not
    installed/importable), `True`/`False` otherwise. Never calls
    `requestAccessForEntityType_` — that triggers a GUI prompt, which
    has no answer on an unattended headless pipeline (SPEC §5.1a)."""
    try:
        from Contacts import (
            CNAuthorizationStatusAuthorized,
            CNContactStore,
            CNEntityTypeContacts,
        )
    except ImportError:
        return None
    status = CNContactStore.authorizationStatusForEntityType_(CNEntityTypeContacts)
    return bool(status == CNAuthorizationStatusAuthorized)


def _fetch_last_sync_at(conn: psycopg.Connection) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(finished_at) FROM extraction_run WHERE status = 'ok'")
        row = cur.fetchone()
    return row[0] if row else None


def _fetch_watermarks(conn: psycopg.Connection) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT key, value FROM sync_state WHERE key LIKE 'watermark.rowid.%'")
        rows = cur.fetchall()
    return {key.removeprefix("watermark.rowid."): value for key, value in rows}


def _pg_ok(conn: psycopg.Connection, config: Config) -> bool:
    from imsg.db.fingerprint import verify_data_directory

    try:
        verify_data_directory(conn, config.paths.data_root)
    except ClusterFingerprintError:
        return False
    return True


def check_permissions(*, config: Config, conn: psycopg.Connection) -> dict[str, Any]:
    """SPEC §10.2 `check_permissions` — "Same checks as `imsg
    check-permissions`", plus the DB-backed freshness fields that CLI
    command reports as unavailable ("pipeline stages not yet built") —
    S1/S2/S3/S7 exist as of this build, so `check_permissions` (unlike
    the CLI command, which this build does not also revise) computes
    them for real."""
    mount = check_mount(config.paths.data_root)
    posture = check_at_rest_posture(config.paths.data_root)
    fda = check_full_disk_access(config.paths.live_chat_db)
    pg_ok = _pg_ok(conn, config)

    last_sync_at = _fetch_last_sync_at(conn)
    watermarks = _fetch_watermarks(conn)
    index_fresh = None
    if last_sync_at is not None:
        age_seconds = (datetime.now(UTC) - last_sync_at).total_seconds()
        index_fresh = age_seconds < 2 * config.sync.interval_seconds

    return {
        "full_disk_access": fda,
        "contacts_access": _contacts_access_status(),
        "at_rest_posture": posture.label,
        "boot_volume_encrypted": posture.boot_volume_encrypted,
        "auto_login_enabled": posture.auto_login_enabled,
        "data_volume_encrypted": posture.data_volume_encrypted,
        "mount_ok": mount.ok,
        "pg_ok": pg_ok,
        "last_sync_at": last_sync_at.isoformat() if last_sync_at else None,
        "index_fresh": index_fresh,
        "watermarks": watermarks,
    }


__all__ = [
    "check_permissions",
    "get_attachment_text",
    "get_conversation",
    "list_people",
    "search_messages",
]
