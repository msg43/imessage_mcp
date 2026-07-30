"""`imsg export push --from-run` (SPEC §11.1, §11.4): promote exactly
one approved plan — or refuse.

`push` never re-renders (D6). It verifies, in order, before the first
byte leaves the machine:

1. the run is in a pushable state;
2. the current config still hashes to the plan's `config_sha256`;
3. the current `allowlist_person` table still matches the plan's
   frozen snapshot byte-for-byte;
4. the staged manifest still hashes to `manifest_sha256`, and — when
   approval is required — to `approved_manifest_sha256`;
5. every staged file still hashes to its manifest entry and resolves
   inside the staging directory;
6. **eligibility still holds**, re-derived from the live database: the
   hash pin alone proves the bytes didn't change, NOT that the world
   didn't — a participant added to a group, an identity merge, or a
   re-segmentation between approval and push all invalidate the plan
   even though the staged bytes verify perfectly. Every planned upsert
   must still map to a live, eligible segment/chunk with an unchanged
   eligible-attachment set, or the whole push aborts.

Any verification failure raises and uploads nothing. Failures *during*
execution (network, per-document import errors) are recorded per item
as `failed` and are safely retryable — a retry re-runs all of the
verification above first, so a stale retry cannot outlive drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from imsg.export.documents import (
    attachment_chunk_document_id,
    metadata_jsonl_line,
    segment_document_id,
    structdata_for,
)
from imsg.export.eligibility import (
    compute_attachment_eligibility,
    eligible_chat_ids,
    snapshot_allowlist,
)
from imsg.export.errors import ExportDriftError, ExportPushError
from imsg.export.models import PushResult
from imsg.export.planner import export_config_sha256, load_manifest, staging_dir_for
from imsg.export.review import allowlist_matches_snapshot, compute_approval_requirements
from imsg.export.transport import ExportTransport, ImportEntry, TransportError
from imsg.hashing import sha256_file
from imsg.paths import is_contained_in

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config

_PUSHABLE_STATES = ("planned", "pushing", "failed")


def _verify_eligibility_unchanged(
    conn: psycopg.Connection, upserts: list[dict[str, Any]]
) -> None:
    """The TOCTOU closure (step 6 above). Deny-biased throughout: a
    vanished segment, a re-parented chunk, or ANY difference in the
    eligible-attachment set aborts — including an attachment that
    became newly eligible, because 'the world changed' voids the plan
    in either direction (SPEC §11.1: any drift requires a new plan)."""
    if not upserts:
        return
    eligible_now = eligible_chat_ids(conn)
    segment_ids = sorted({int(u["segment_id"]) for u in upserts})

    with conn.cursor() as cur:
        cur.execute(
            "SELECT segment_id, stable_key, chat_id FROM segment WHERE segment_id = ANY(%s)",
            (segment_ids,),
        )
        segments = {
            int(sid): (str(stable_key), int(chat_id))
            for sid, stable_key, chat_id in cur.fetchall()
        }
        chunk_ids = sorted(
            {
                int(u["attachment_chunk_id"])
                for u in upserts
                if u.get("attachment_chunk_id") is not None
            }
        )
        chunks: dict[int, tuple[int, str, str, int]] = {}
        if chunk_ids:
            cur.execute(
                """
                SELECT ac.chunk_id, ac.attachment_id, a.source_guid, ac.kind, ac.seq
                FROM attachment_chunk ac
                JOIN attachment a ON a.attachment_id = ac.attachment_id
                WHERE ac.chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            chunks = {
                int(cid): (int(aid), str(guid), str(kind), int(seq))
                for cid, aid, guid, kind, seq in cur.fetchall()
            }

        # Current eligible-attachment guid sets per segment.
        attachment_eligibility = compute_attachment_eligibility(conn, segment_ids)
        cur.execute(
            """
            SELECT DISTINCT sm.segment_id, ma.attachment_id, a.source_guid
            FROM segment_message sm
            JOIN message m ON m.message_id = sm.message_id
            JOIN message_attachment ma ON ma.message_id = m.message_id
            JOIN attachment a ON a.attachment_id = ma.attachment_id
            WHERE sm.segment_id = ANY(%s) AND NOT m.is_unsent
            """,
            (segment_ids,),
        )
        eligible_guids_now: dict[int, set[str]] = {sid: set() for sid in segment_ids}
        guid_rows = cur.fetchall()
    attachment_id_by_pair: dict[tuple[int, str], int] = {}
    for sid, aid, guid in guid_rows:
        attachment_id_by_pair[(int(sid), str(guid))] = int(aid)
        if attachment_eligibility.get((int(sid), int(aid)), False):
            eligible_guids_now[int(sid)].add(str(guid))

    for upsert in upserts:
        document_id = str(upsert["document_id"])
        segment_id = int(upsert["segment_id"])
        live = segments.get(segment_id)
        if live is None:
            raise ExportDriftError(
                f"planned document {document_id[:12]}… references segment "
                f"{segment_id}, which no longer exists — the corpus changed "
                f"since planning; re-run `imsg export plan`"
            )
        stable_key, chat_id_now = live
        if chat_id_now not in eligible_now:
            raise ExportDriftError(
                f"planned document {document_id[:12]}… belongs to chat "
                f"{chat_id_now}, which is no longer export-eligible (allowlist, "
                f"participant, or identity change since planning) — re-run "
                f"`imsg export plan`"
            )
        if int(upsert["chat_id"]) != chat_id_now:
            raise ExportDriftError(
                f"planned document {document_id[:12]}… moved from chat "
                f"{upsert['chat_id']} to chat {chat_id_now} since planning — "
                f"re-run `imsg export plan`"
            )
        if upsert.get("kind") == "segment":
            if segment_document_id(stable_key) != document_id:
                raise ExportDriftError(
                    f"segment {segment_id} was re-segmented since planning "
                    f"(stable key changed) — re-run `imsg export plan`"
                )
            planned_guids = {str(g) for g in upsert.get("eligible_attachment_guids", [])}
            if planned_guids != eligible_guids_now.get(segment_id, set()):
                raise ExportDriftError(
                    f"the eligible-attachment set of segment {segment_id} changed "
                    f"since planning — re-run `imsg export plan`"
                )
        else:
            chunk_id = upsert.get("attachment_chunk_id")
            chunk = chunks.get(int(chunk_id)) if chunk_id is not None else None
            if chunk is None:
                raise ExportDriftError(
                    f"planned attachment document {document_id[:12]}… references "
                    f"a chunk that no longer exists — re-run `imsg export plan`"
                )
            attachment_id, guid, kind, seq = chunk
            if attachment_chunk_document_id(stable_key, guid, kind, seq) != document_id:
                raise ExportDriftError(
                    f"attachment chunk {chunk_id} no longer derives document id "
                    f"{document_id[:12]}… — re-run `imsg export plan`"
                )
            if not attachment_eligibility.get((segment_id, attachment_id), False):
                raise ExportDriftError(
                    f"attachment {guid} is no longer attachments-eligible in "
                    f"segment {segment_id} — re-run `imsg export plan`"
                )


def push_export(
    conn: psycopg.Connection,
    config: Config,
    run_id: int,
    transport: ExportTransport,
) -> PushResult:
    """Verify every pin, then promote the plan through `transport`.
    The caller commits (and should commit promptly after return: the
    external side effects have already happened; re-pushing after a
    crash is idempotent but works from the recorded item states)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, mode, manifest_sha256, approved_manifest_sha256,
                   allowlist_snapshot, config_sha256
            FROM export_run WHERE export_run_id = %s FOR UPDATE
            """,
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ExportPushError(f"export run {run_id} does not exist")
    status, _mode, manifest_sha_db, approved_sha, allowlist_stored, config_sha_db = row
    if str(status) not in _PUSHABLE_STATES:
        raise ExportPushError(
            f"export run {run_id} is '{status}' — only {_PUSHABLE_STATES} can push"
        )

    # (2) config pin
    config_sha_now = export_config_sha256(config)
    if config_sha_now != str(config_sha_db):
        raise ExportDriftError(
            "the export-relevant config changed since this plan was created — "
            "re-run `imsg export plan`"
        )

    # (3) allowlist pin — byte-for-byte against the frozen snapshot
    if not allowlist_matches_snapshot(snapshot_allowlist(conn), allowlist_stored):
        raise ExportDriftError(
            "allowlist_person changed since this plan was created — the plan is "
            "void; re-run `imsg export plan` (and re-approve)"
        )

    # (4) manifest pin
    staging_dir = staging_dir_for(config, run_id)
    manifest, manifest_sha_disk = load_manifest(staging_dir)
    if manifest_sha_disk != str(manifest_sha_db):
        raise ExportDriftError(
            "the staged manifest no longer hashes to the planned manifest sha — "
            "staging was modified; re-run `imsg export plan`"
        )

    upserts: list[dict[str, Any]] = list(manifest.get("upserts", []))
    deletes: list[dict[str, Any]] = list(manifest.get("deletes", []))

    # Approval: recomputed NOW, not read from the plan — drift in the
    # delta conditions themselves (e.g. the last ok run changed) must
    # be reflected. First push and every qualifying delta require the
    # pinned approval (SPEC §11.4).
    approval_reasons = compute_approval_requirements(conn, manifest)
    if approval_reasons:
        if approved_sha is None:
            raise ExportPushError(
                f"export run {run_id} requires owner approval before push "
                f"(reasons: {', '.join(approval_reasons)}) — run "
                f"`imsg export approve {run_id}` after reviewing the report"
            )
        if str(approved_sha) != manifest_sha_disk:
            raise ExportDriftError(
                "the approved manifest sha does not match the staged manifest — "
                "approval pins bytes (D6); re-plan and re-approve"
            )

    # (5) staged-file pins (all upserts, including already-pushed ones —
    # tampered staging voids the run outright)
    for entry in upserts:
        relpath = str(entry["staged_relpath"])
        staged = staging_dir / relpath
        if not is_contained_in(staged, staging_dir):
            raise ExportDriftError(
                f"staged path '{relpath}' escapes the staging directory — refusing"
            )
        if not staged.is_file() or sha256_file(staged) != str(entry["content_sha256"]):
            raise ExportDriftError(
                f"staged file '{relpath}' is missing or no longer hashes to its "
                f"manifest entry — staging was modified; re-run `imsg export plan`"
            )

    # (6) eligibility re-verification against the live database
    _verify_eligibility_unchanged(conn, upserts)

    # --- verification complete; execute ------------------------------------
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE export_run SET status = 'pushing' WHERE export_run_id = %s",
            (run_id,),
        )
        cur.execute(
            "SELECT document_id, result_state FROM export_run_item "
            "WHERE export_run_id = %s",
            (run_id,),
        )
        item_states = {str(doc_id): str(state) for doc_id, state in cur.fetchall()}

    pushed = deleted = failed = skipped = 0
    notes: list[str] = []

    # Upserts: upload, then one incremental import batch.
    to_import: list[tuple[dict[str, Any], ImportEntry, str]] = []  # (entry, import, uri)
    for entry in upserts:
        document_id = str(entry["document_id"])
        if item_states.get(document_id) == "pushed":
            skipped += 1
            continue
        staged = staging_dir / str(entry["staged_relpath"])
        data = staged.read_bytes()
        gcs_object = str(entry["gcs_object"])
        try:
            uri = transport.upload_document(gcs_object=gcs_object, data=data)
        except TransportError as exc:
            failed += 1
            _record_item(conn, run_id, document_id, "failed", str(exc))
            continue
        jsonl = metadata_jsonl_line(
            document_id=document_id,
            gcs_bucket=config.export.gcs_bucket,
            gcs_object=gcs_object,
            struct_data=structdata_for(
                people=tuple(str(p) for p in entry.get("people", [])),
                started_at=str(entry["started_at"]),
                ended_at=str(entry["ended_at"]),
                segment_key=str(entry["segment_key"]),
                document_kind=str(entry["kind"]),
            ),
        )
        to_import.append(
            (entry, ImportEntry(document_id=document_id, gcs_object=gcs_object,
                                metadata_jsonl_line=jsonl), uri)
        )

    if to_import:
        try:
            import_result = transport.import_documents([imp for _, imp, _ in to_import])
            import_failures = set(import_result.failed_document_ids)
        except TransportError as exc:
            import_failures = {imp.document_id for _, imp, _ in to_import}
            notes.append(f"import batch failed: {exc}")
        for entry, imp, uri in to_import:
            if imp.document_id in import_failures:
                failed += 1
                _record_item(conn, run_id, imp.document_id, "failed", "import failed")
            else:
                pushed += 1
                _record_item(conn, run_id, imp.document_id, "pushed", None)
                _upsert_export_document(conn, entry, uri)

    # Deletes: delete, then POSITIVELY verify absence (SPEC §11.4/D6) —
    # a document is only recorded 'purged' after the store says gone.
    for entry in deletes:
        document_id = str(entry["document_id"])
        if item_states.get(document_id) == "deleted":
            skipped += 1
            continue
        gcs_object = str(entry["gcs_object"])
        try:
            transport.delete_document(document_id=document_id, gcs_object=gcs_object)
        except TransportError as exc:
            failed += 1
            _record_item(conn, run_id, document_id, "failed", str(exc))
            continue
        if transport.document_absent(document_id=document_id, gcs_object=gcs_object):
            deleted += 1
            _record_item(conn, run_id, document_id, "deleted", None)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE export_document SET state = 'purged' WHERE document_id = %s",
                    (document_id,),
                )
        else:
            failed += 1
            _record_item(
                conn,
                run_id,
                document_id,
                "failed",
                "delete reported success but the document is still present — "
                "absence NOT verified",
            )

    # Any item still failed (from this or a previous attempt) keeps the
    # run failed and retryable.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM export_run_item "
            "WHERE export_run_id = %s AND result_state NOT IN ('pushed','deleted')",
            (run_id,),
        )
        row = cur.fetchone()
        remaining = int(row[0]) if row else 0
        final_status = "ok" if remaining == 0 else "failed"
        cur.execute(
            "UPDATE export_run SET status = %s, finished_at = now() "
            "WHERE export_run_id = %s",
            (final_status, run_id),
        )

    return PushResult(
        run_id=run_id,
        status=final_status,
        pushed=pushed,
        deleted=deleted,
        failed=failed,
        skipped_already_done=skipped,
        notes=tuple(notes),
    )


def _record_item(
    conn: psycopg.Connection,
    run_id: int,
    document_id: str,
    result_state: str,
    error: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE export_run_item SET result_state = %s, error = %s "
            "WHERE export_run_id = %s AND document_id = %s",
            (result_state, error, run_id, document_id),
        )


def _upsert_export_document(
    conn: psycopg.Connection, entry: dict[str, Any], gcs_uri: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO export_document
                (document_id, kind, segment_id, attachment_chunk_id, gcs_uri,
                 current_content_sha256, state)
            VALUES (%s, %s, %s, %s, %s, %s, 'pushed')
            ON CONFLICT (document_id) DO UPDATE SET
                kind = EXCLUDED.kind,
                segment_id = EXCLUDED.segment_id,
                attachment_chunk_id = EXCLUDED.attachment_chunk_id,
                gcs_uri = EXCLUDED.gcs_uri,
                current_content_sha256 = EXCLUDED.current_content_sha256,
                state = 'pushed'
            """,
            (
                str(entry["document_id"]),
                str(entry["kind"]),
                int(entry["segment_id"]),
                entry.get("attachment_chunk_id"),
                gcs_uri,
                str(entry["content_sha256"]),
            ),
        )


__all__ = ["push_export"]
