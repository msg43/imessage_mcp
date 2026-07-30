"""Pre-push review: the approval-requirement rules, the owner-facing
review report, and `imsg export approve` (SPEC §11.4).

Approval pins BYTES, not intent (D6): `approve_run` re-reads the
staged manifest and re-hashes every staged document file before
recording `approved_manifest_sha256`. `push` then refuses unless the
approved sha still matches — so nothing that happened between the
owner's eyes and the wire can widen what ships.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from imsg.export.documents import canonical_json
from imsg.export.errors import ExportApprovalError
from imsg.export.models import ApprovalResult
from imsg.hashing import sha256_file
from imsg.paths import is_contained_in

if TYPE_CHECKING:
    from pathlib import Path

    import psycopg

    from imsg.config.schema import Config

APPROVAL_FIRST_PUSH = "first-push"
APPROVAL_DELETES = "plan-contains-deletes"
APPROVAL_NEW_PERSON = "new-person"
APPROVAL_NEW_THREAD = "new-thread"
APPROVAL_NEW_MIME = "new-attachment-mime-class"
APPROVAL_CONFIG_CHANGED = "config-or-policy-changed"


def _last_ok_run(conn: psycopg.Connection) -> tuple[Any, str] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT allowlist_snapshot, config_sha256 FROM export_run "
            "WHERE status = 'ok' ORDER BY export_run_id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return (row[0], str(row[1])) if row else None


def compute_approval_requirements(
    conn: psycopg.Connection, manifest: dict[str, Any]
) -> tuple[str, ...]:
    """The §11.4 triggers, computed conservatively: anything that
    cannot be positively shown to be previously-approved territory
    counts as new (e.g. a pushed document whose segment row has since
    vanished no longer vouches for its thread). Returns () only when
    the plan is a pure content-update of already-approved scope."""
    reasons: list[str] = []
    upserts: list[dict[str, Any]] = list(manifest.get("upserts", []))
    deletes: list[dict[str, Any]] = list(manifest.get("deletes", []))

    last = _last_ok_run(conn)
    if last is None:
        return (APPROVAL_FIRST_PUSH,)

    if deletes:
        reasons.append(APPROVAL_DELETES)

    last_snapshot, last_config_sha = last
    previously_allowed = {
        str(entry["short_name"])
        for entry in (last_snapshot or [])
        if isinstance(entry, dict) and entry.get("text_allowed") is True
    }
    plan_people = {str(p) for u in upserts for p in u.get("people", [])}
    if plan_people - previously_allowed:
        reasons.append(APPROVAL_NEW_PERSON)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT s.chat_id
            FROM export_document ed
            JOIN segment s ON s.segment_id = ed.segment_id
            WHERE ed.state = 'pushed'
            """
        )
        pushed_chats = {int(r[0]) for r in cur.fetchall()}
        cur.execute(
            """
            SELECT DISTINCT coalesce(a.mime_type, 'unknown')
            FROM export_document ed
            JOIN attachment_chunk ac ON ac.chunk_id = ed.attachment_chunk_id
            JOIN attachment a ON a.attachment_id = ac.attachment_id
            WHERE ed.state = 'pushed'
            """
        )
        pushed_mimes = {str(r[0]) for r in cur.fetchall()}

    plan_chats = {int(u["chat_id"]) for u in upserts}
    if plan_chats - pushed_chats:
        reasons.append(APPROVAL_NEW_THREAD)

    plan_mimes = {
        str(u.get("mime_type") or "unknown")
        for u in upserts
        if u.get("kind") == "attachment_chunk"
    }
    if plan_mimes - pushed_mimes:
        reasons.append(APPROVAL_NEW_MIME)

    if str(manifest.get("config_sha256", "")) != last_config_sha:
        reasons.append(APPROVAL_CONFIG_CHANGED)

    return tuple(reasons)


# ---------------------------------------------------------------------------
# Review report (SPEC §11.4: per thread — participants, message count,
# date range, three sample lines; attachment-document counts + MIME
# classes). Local file on the encrypted volume; contains content of
# ELIGIBLE threads only, by construction (it is built from staged docs).
# ---------------------------------------------------------------------------


def _sample_lines(doc_text: str) -> list[str]:
    body = doc_text.split("\n---\n", 1)
    if len(body) != 2:
        return []
    lines = [ln for ln in body[1].splitlines() if ln.strip()]
    if not lines:
        return []
    picks = {0, len(lines) // 2, len(lines) - 1}
    return [lines[i] for i in sorted(picks)]


def build_review_report(
    conn: psycopg.Connection,
    *,
    manifest: dict[str, Any],
    staging_dir: Path,
    approval_reasons: tuple[str, ...],
    unchanged_count: int,
) -> str:
    upserts: list[dict[str, Any]] = list(manifest.get("upserts", []))
    deletes: list[dict[str, Any]] = list(manifest.get("deletes", []))

    by_chat: dict[int, list[dict[str, Any]]] = {}
    for upsert in upserts:
        by_chat.setdefault(int(upsert["chat_id"]), []).append(upsert)

    chat_labels: dict[int, str] = {}
    if by_chat:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chat_id, kind, display_name FROM chat WHERE chat_id = ANY(%s)",
                (sorted(by_chat),),
            )
            for chat_id, kind, display_name in cur.fetchall():
                label = f'group "{display_name}"' if display_name else str(kind)
                chat_labels[int(chat_id)] = label

    out: list[str] = [
        f"Export plan review — mode: {manifest.get('mode')}",
        f"Upserts: {len(upserts)}   Deletes: {len(deletes)}   Unchanged: {unchanged_count}",
        "",
    ]
    if approval_reasons:
        out.append("OWNER APPROVAL REQUIRED before push — reasons: " + ", ".join(approval_reasons))
    else:
        out.append("No new scope: push may proceed without fresh approval (§11.4).")
    out.append("")

    for chat_id in sorted(by_chat):
        entries = by_chat[chat_id]
        segment_docs = [e for e in entries if e["kind"] == "segment"]
        chunk_docs = [e for e in entries if e["kind"] == "attachment_chunk"]
        people = sorted({str(p) for e in entries for p in e.get("people", [])})
        started = min(str(e["started_at"]) for e in entries)
        ended = max(str(e["ended_at"]) for e in entries)
        mimes = sorted({str(e.get("mime_type") or "unknown") for e in chunk_docs})

        message_lines = 0
        samples: list[str] = []
        for entry in segment_docs:
            staged = staging_dir / str(entry["staged_relpath"])
            if staged.is_file():
                text = staged.read_text(encoding="utf-8")
                body = text.split("\n---\n", 1)
                message_lines += len(body[1].splitlines()) if len(body) == 2 else 0
                if not samples:
                    samples = _sample_lines(text)

        out.append(f"Thread {chat_id} ({chat_labels.get(chat_id, 'unknown')})")
        out.append(f"  participants: {', '.join(people)}")
        out.append(f"  segment documents: {len(segment_docs)}   message lines: {message_lines}")
        out.append(f"  date range: {started} \u2013 {ended}")
        out.append(
            f"  attachment documents: {len(chunk_docs)}"
            + (f"   mime classes: {', '.join(mimes)}" if mimes else "")
        )
        for sample in samples:
            out.append(f"  sample: {sample}")
        out.append("")

    if deletes:
        out.append(f"Deletes ({len(deletes)} documents, by id):")
        out.extend(f"  {d['document_id']}" for d in deletes)
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def approve_run(
    conn: psycopg.Connection,
    config: Config,
    run_id: int,
    *,
    approval_id: str | None = None,
) -> ApprovalResult:
    """Record owner approval of a planned run, pinning the exact staged
    bytes (SPEC §11.4). Re-verifies the manifest hash against the DB
    row AND every staged file against the manifest before recording —
    an approval can never be minted for bytes the owner did not stage.
    The caller commits."""
    # Import here to avoid a planner<->review import cycle.
    from imsg.export.planner import load_manifest, staging_dir_for

    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, manifest_sha256 FROM export_run "
            "WHERE export_run_id = %s FOR UPDATE",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ExportApprovalError(f"export run {run_id} does not exist")
    status, manifest_sha_db = str(row[0]), str(row[1])
    if status != "planned":
        raise ExportApprovalError(
            f"export run {run_id} is '{status}', not 'planned' — only a freshly "
            f"planned run can be approved"
        )

    staging_dir = staging_dir_for(config, run_id)
    manifest, manifest_sha_disk = load_manifest(staging_dir)
    if manifest_sha_disk != manifest_sha_db:
        raise ExportApprovalError(
            f"staged manifest hash {manifest_sha_disk[:12]}… does not match the "
            f"recorded plan hash {manifest_sha_db[:12]}… — staging was modified "
            f"after planning; re-run `imsg export plan`"
        )

    for entry in manifest.get("upserts", []):
        relpath = str(entry["staged_relpath"])
        staged = staging_dir / relpath
        if not is_contained_in(staged, staging_dir):
            raise ExportApprovalError(
                f"staged path '{relpath}' escapes the staging directory — refusing"
            )
        if not staged.is_file():
            raise ExportApprovalError(f"staged file missing: {staged}")
        actual = sha256_file(staged)
        if actual != str(entry["content_sha256"]):
            raise ExportApprovalError(
                f"staged file {relpath} hashes to {actual[:12]}…, manifest says "
                f"{str(entry['content_sha256'])[:12]}… — staging was modified; "
                f"re-run `imsg export plan`"
            )

    final_approval_id = approval_id or f"approval-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE export_run
            SET approved_at = now(),
                approved_manifest_sha256 = %s,
                approval_id = %s
            WHERE export_run_id = %s
            """,
            (manifest_sha_db, final_approval_id, run_id),
        )
    return ApprovalResult(
        run_id=run_id,
        approval_id=final_approval_id,
        approved_manifest_sha256=manifest_sha_db,
    )


def allowlist_matches_snapshot(
    current: list[dict[str, object]], stored: object
) -> bool:
    """Byte-level comparison of allowlist states via the canonical JSON
    encoding (used by push's TOCTOU check)."""
    return canonical_json(current) == canonical_json(stored)


__all__ = [
    "APPROVAL_CONFIG_CHANGED",
    "APPROVAL_DELETES",
    "APPROVAL_FIRST_PUSH",
    "APPROVAL_NEW_MIME",
    "APPROVAL_NEW_PERSON",
    "APPROVAL_NEW_THREAD",
    "allowlist_matches_snapshot",
    "approve_run",
    "build_review_report",
    "compute_approval_requirements",
]
