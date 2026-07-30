"""`imsg export plan` (SPEC §11.1): eligibility query → reconcile
desired vs current `export_document` state → render immutable staging
bytes → manifest of upserts AND deletes → review report.

The planner is the only module that decides *what* may leave the
machine; `push` decides only whether the world still matches what was
planned. Everything staged here is re-derived from the live database
through `imsg.export.eligibility` — `segment.rendered_text` is never
reused, because it may legally contain unsent text or edit history
under local policy flags (D1: export re-renders with the exclusions
hard-coded).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from psycopg.types.json import Jsonb

from imsg.config.schema import Config
from imsg.export.documents import (
    DOCS_SUBDIR,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT_VERSION,
    METADATA_FILENAME,
    REPORT_FILENAME,
    attachment_chunk_document_id,
    canonical_json,
    gcs_object_for,
    metadata_jsonl_line,
    segment_document_id,
    staged_relpath_for,
    structdata_for,
)
from imsg.export.eligibility import (
    compute_attachment_eligibility,
    eligible_chat_ids,
    owner_person_id,
    snapshot_allowlist,
)
from imsg.export.errors import ExportPlanError
from imsg.export.models import (
    ExportAttachment,
    ExportChunk,
    ExportMessage,
    ExportSegment,
    PlannedDelete,
    PlannedUpsert,
    PlanResult,
)
from imsg.export.render import (
    RENDERER_VERSION,
    render_chunk_document,
    render_segment_document,
)
from imsg.export.review import build_review_report, compute_approval_requirements
from imsg.hashing import sha256_text

if TYPE_CHECKING:
    from pathlib import Path

    import psycopg

_TAPBACK_SYMBOLS = {
    "loved": "♥",
    "liked": "👍",
    "disliked": "👎",
    "laughed": "😂",
    "emphasized": "‼",
    "questioned": "❓",
    "sticker": "🏷",
}


def export_config_sha256(config: Config) -> str:
    """Hash of every config value that can change export *content or
    destination*, plus the renderer version. Stored on the run and
    re-verified at push (SPEC §11.1) — a policy flip, a render change,
    or a retargeted bucket all void a standing plan. `policy.*` is
    included even though export rendering ignores it (D1): §11.4 lists
    "policy change" as an approval trigger, so it must show up as
    config drift."""
    projection = {
        "renderer_version": RENDERER_VERSION,
        "export": config.export.model_dump(mode="json"),
        "render": config.render.model_dump(mode="json"),
        "policy": config.policy.model_dump(mode="json"),
    }
    return sha256_text(canonical_json(projection))


def staging_dir_for(config: Config, run_id: int) -> Path:
    return config.paths.data_root / "export" / "staging" / str(run_id)


def load_manifest(staging_dir: Path) -> tuple[dict[str, Any], str]:
    """Read a staged manifest; returns (parsed, sha256-of-file-text).
    The hash is of the exact file bytes, which `plan` wrote as
    canonical JSON — the value compared against `export_run.
    manifest_sha256` and `approved_manifest_sha256`."""
    manifest_path = staging_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ExportPlanError(f"staged manifest missing: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ExportPlanError(f"staged manifest is not a JSON object: {manifest_path}")
    return parsed, sha256_text(text)


# ---------------------------------------------------------------------------
# Fetching gated rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ChatInfo:
    chat_id: int
    kind: str
    display_name: str | None
    people: tuple[str, ...]  # sorted short_names: participants + effective senders


def _fetch_chat_infos(conn: psycopg.Connection, chat_ids: list[int]) -> dict[int, _ChatInfo]:
    if not chat_ids:
        return {}
    owner = owner_person_id(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chat_id, kind, display_name FROM chat WHERE chat_id = ANY(%s)",
            (chat_ids,),
        )
        base = {int(cid): (str(kind), dname) for cid, kind, dname in cur.fetchall()}

        # People = current participants plus everyone whose messages
        # appear in the chat (former members, the owner) — eligibility
        # has already required all of them to be allowlisted.
        cur.execute(
            """
            SELECT cp.chat_id, p.short_name
            FROM chat_participant cp
            JOIN person p ON p.person_id = cp.person_id
            WHERE cp.chat_id = ANY(%(chat_ids)s)
            UNION
            SELECT m.chat_id, p.short_name
            FROM message m
            JOIN person p ON p.person_id =
                coalesce(m.sender_person_id, CASE WHEN m.is_from_me THEN %(owner)s::bigint END)
            WHERE m.chat_id = ANY(%(chat_ids)s)
            """,
            {"chat_ids": chat_ids, "owner": owner},
        )
        people: dict[int, set[str]] = {}
        for cid, short_name in cur.fetchall():
            people.setdefault(int(cid), set()).add(str(short_name))

    return {
        cid: _ChatInfo(
            chat_id=cid,
            kind=kind,
            display_name=dname,
            people=tuple(sorted(people.get(cid, set()))),
        )
        for cid, (kind, dname) in base.items()
    }


def _fetch_attachment_rows(
    conn: psycopg.Connection,
    message_ids: list[int],
    eligibility_by_pair: dict[tuple[int, int], bool],
    segment_by_message: dict[int, int],
) -> dict[int, list[ExportAttachment]]:
    """Attachments per message, each already stamped with its SEPARATE
    content gate verdict. An ineligible attachment keeps only its ids —
    filename/MIME/enrichment text are dropped here, before the renderer
    ever sees them, so no later formatting bug can leak them."""
    if not message_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ma.message_id, a.attachment_id, a.source_guid, a.filename,
                   a.mime_type, e.kind, e.text
            FROM message_attachment ma
            JOIN attachment a ON a.attachment_id = ma.attachment_id
            LEFT JOIN enrichment e
                   ON e.attachment_id = a.attachment_id AND e.state = 'done'
            WHERE ma.message_id = ANY(%s)
            ORDER BY ma.message_id, ma.ordinal, a.attachment_id
            """,
            (message_ids,),
        )
        rows = cur.fetchall()

    collected: dict[tuple[int, int], dict[str, Any]] = {}
    order: dict[int, list[int]] = {}
    for message_id, attachment_id, source_guid, filename, mime_type, ekind, etext in rows:
        message_id = int(message_id)
        attachment_id = int(attachment_id)
        key = (message_id, attachment_id)
        if key not in collected:
            segment_id = segment_by_message.get(message_id)
            eligible = (
                bool(eligibility_by_pair.get((segment_id, attachment_id), False))
                if segment_id is not None
                else False
            )
            collected[key] = {
                "source_guid": str(source_guid),
                "filename": filename if eligible else None,
                "mime_type": mime_type if eligible else None,
                "eligible": eligible,
                "caption": None,
                "ocr_text": None,
                "transcript": None,
                "pdf_text": None,
            }
            order.setdefault(message_id, []).append(attachment_id)
        entry = collected[key]
        if entry["eligible"] and etext is not None:
            if ekind == "pdf_text":
                entry["pdf_text"] = etext
            elif ekind == "caption":
                entry["caption"] = etext
            elif ekind in ("ocr", "frame_ocr"):
                entry["ocr_text"] = etext
            elif ekind == "transcript":
                entry["transcript"] = etext

    result: dict[int, list[ExportAttachment]] = {}
    for message_id, attachment_ids in order.items():
        result[message_id] = [
            ExportAttachment(
                attachment_id=attachment_id,
                source_guid=str(collected[(message_id, attachment_id)]["source_guid"]),
                filename=collected[(message_id, attachment_id)]["filename"],
                mime_type=collected[(message_id, attachment_id)]["mime_type"],
                content_eligible=bool(collected[(message_id, attachment_id)]["eligible"]),
                caption=collected[(message_id, attachment_id)]["caption"],
                ocr_text=collected[(message_id, attachment_id)]["ocr_text"],
                transcript=collected[(message_id, attachment_id)]["transcript"],
                pdf_text=collected[(message_id, attachment_id)]["pdf_text"],
            )
            for attachment_id in attachment_ids
        ]
    return result


def _tapback_symbol(kind: str) -> str:
    if kind.startswith("emoji:"):
        return kind.split(":", 1)[1]
    return _TAPBACK_SYMBOLS.get(kind, kind)


def _fetch_tapback_suffixes(
    conn: psycopg.Connection, message_ids: list[int], *, owner_short_name: str | None
) -> dict[int, list[str]]:
    if not message_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.target_message_id, t.kind, t.is_from_me, p.short_name
            FROM tapback t
            LEFT JOIN person p ON p.person_id = t.sender_person_id
            WHERE t.target_message_id = ANY(%s) AND NOT t.removed
            ORDER BY t.acted_at NULLS LAST, t.tapback_id
            """,
            (message_ids,),
        )
        rows = cur.fetchall()
    result: dict[int, list[str]] = {}
    for target_message_id, kind, is_from_me, short_name in rows:
        sender = short_name or (owner_short_name if is_from_me else None)
        if sender is None:
            # Unattributable reaction — omit rather than guess. (An
            # eligible chat cannot reach here; this is defense in depth.)
            continue
        result.setdefault(int(target_message_id), []).append(
            f"({_tapback_symbol(str(kind))} {sender})"
        )
    return result


def _fetch_export_segments(
    conn: psycopg.Connection, chat_ids: list[int]
) -> tuple[list[ExportSegment], dict[tuple[int, int], bool]]:
    """All segments of the given (already eligibility-vetted) chats,
    with messages (never unsent, latest text only), gated attachments,
    and tapback suffixes. Returns the attachment-eligibility map too so
    the chunk pass reuses the same verdicts."""
    if not chat_ids:
        return [], {}
    chat_infos = _fetch_chat_infos(conn, chat_ids)
    owner = owner_person_id(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT segment_id, stable_key, chat_id, started_at, ended_at
            FROM segment
            WHERE chat_id = ANY(%s)
            ORDER BY chat_id, started_at, segment_id
            """,
            (chat_ids,),
        )
        segment_rows = cur.fetchall()
        segment_ids = [int(r[0]) for r in segment_rows]

        # D1 is enforced structurally here: `NOT m.is_unsent` is not a
        # parameter, and only `text_original` (the latest text) is
        # selected — there is no code path that could include a
        # `message_version` row in an export document.
        cur.execute(
            """
            SELECT sm.segment_id, m.message_id, m.sent_at, m.text_original,
                   p.short_name
            FROM segment_message sm
            JOIN message m ON m.message_id = sm.message_id
            LEFT JOIN person p ON p.person_id =
                coalesce(m.sender_person_id, CASE WHEN m.is_from_me THEN %(owner)s::bigint END)
            WHERE sm.segment_id = ANY(%(segment_ids)s)
              AND NOT m.is_unsent
            ORDER BY m.sent_at, m.message_id
            """,
            {"segment_ids": segment_ids, "owner": owner},
        )
        message_rows = cur.fetchall()

    attachment_eligibility = compute_attachment_eligibility(conn, segment_ids)

    message_ids = [int(r[1]) for r in message_rows]
    segment_by_message = {int(r[1]): int(r[0]) for r in message_rows}
    attachments_by_message = _fetch_attachment_rows(
        conn, message_ids, attachment_eligibility, segment_by_message
    )
    owner_short = _owner_short_name(conn)
    tapbacks_by_message = _fetch_tapback_suffixes(
        conn, message_ids, owner_short_name=owner_short
    )

    messages_by_segment: dict[int, list[ExportMessage]] = {}
    for segment_id, message_id, sent_at, text_original, short_name in message_rows:
        segment_id = int(segment_id)
        message_id = int(message_id)
        if short_name is None:
            # Eligibility guarantees every exported message has a
            # resolvable, allowlisted effective sender; hitting this
            # means the world changed mid-plan. Refuse, don't guess.
            raise ExportPlanError(
                f"message {message_id} has no resolvable sender short_name during "
                f"planning — eligibility drift mid-plan; re-run `imsg export plan`"
            )
        messages_by_segment.setdefault(segment_id, []).append(
            ExportMessage(
                message_id=message_id,
                sent_at=sent_at,
                sender_short_name=str(short_name),
                text=text_original,
                attachments=tuple(attachments_by_message.get(message_id, ())),
                tapback_suffixes=tuple(tapbacks_by_message.get(message_id, ())),
            )
        )

    segments: list[ExportSegment] = []
    for segment_id, stable_key, chat_id, started_at, ended_at in segment_rows:
        chat_info = chat_infos.get(int(chat_id))
        if chat_info is None:
            raise ExportPlanError(f"segment {segment_id} references unknown chat {chat_id}")
        segments.append(
            ExportSegment(
                segment_id=int(segment_id),
                stable_key=str(stable_key),
                chat_id=int(chat_id),
                chat_kind=chat_info.kind,
                chat_display_name=chat_info.display_name,
                participant_short_names=chat_info.people,
                started_at=started_at,
                ended_at=ended_at,
                messages=tuple(messages_by_segment.get(int(segment_id), ())),
            )
        )
    return segments, attachment_eligibility


def _owner_short_name(conn: psycopg.Connection) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT short_name FROM person WHERE is_owner")
        row = cur.fetchone()
        return str(row[0]) if row else None


def _fetch_chunks_for(
    conn: psycopg.Connection,
    segments: list[ExportSegment],
    attachment_eligibility: dict[tuple[int, int], bool],
) -> list[ExportChunk]:
    """Attachment-chunk documents: one per (eligible attachment,
    authorized parent segment, chunk row). Only attachments whose
    SEPARATE gate passed in that parent contribute (SPEC §11.3)."""
    eligible_pairs = [
        (segment_id, attachment_id)
        for (segment_id, attachment_id), ok in attachment_eligibility.items()
        if ok
    ]
    if not eligible_pairs:
        return []
    segments_by_id = {s.segment_id: s for s in segments}
    attachment_ids = sorted({attachment_id for _, attachment_id in eligible_pairs})
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ac.chunk_id, ac.attachment_id, a.source_guid, a.mime_type,
                   ac.kind, ac.seq, ac.text
            FROM attachment_chunk ac
            JOIN attachment a ON a.attachment_id = ac.attachment_id
            WHERE ac.attachment_id = ANY(%s)
            ORDER BY ac.attachment_id, ac.kind, ac.seq
            """,
            (attachment_ids,),
        )
        chunk_rows = cur.fetchall()

    chunks_by_attachment: dict[int, list[tuple[int, str, str | None, str, int, str]]] = {}
    for chunk_id, attachment_id, source_guid, mime_type, kind, seq, text in chunk_rows:
        chunks_by_attachment.setdefault(int(attachment_id), []).append(
            (int(chunk_id), str(source_guid), mime_type, str(kind), int(seq), str(text))
        )

    result: list[ExportChunk] = []
    for segment_id, attachment_id in sorted(eligible_pairs):
        parent = segments_by_id.get(segment_id)
        if parent is None:
            continue  # segment vanished mid-plan; denying is the safe default
        for chunk_id, source_guid, mime_type, kind, seq, text in chunks_by_attachment.get(
            attachment_id, []
        ):
            result.append(
                ExportChunk(
                    chunk_id=chunk_id,
                    attachment_id=attachment_id,
                    attachment_source_guid=source_guid,
                    mime_type=mime_type,
                    kind=kind,
                    seq=seq,
                    text=text,
                    parent=parent,
                )
            )
    return result


# ---------------------------------------------------------------------------
# Desired-document construction + reconcile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DesiredDocument:
    upsert: PlannedUpsert
    text: str


def build_desired_documents(
    conn: psycopg.Connection, config: Config
) -> dict[str, _DesiredDocument]:
    """The full desired external state under the CURRENT database and
    config — the reconciliation target. Every document here has passed
    the chat gate; every attachment byte has passed the separate
    attachment gate."""
    chat_ids = sorted(eligible_chat_ids(conn))
    segments, attachment_eligibility = _fetch_export_segments(conn, chat_ids)
    chunks = _fetch_chunks_for(conn, segments, attachment_eligibility)

    tz = config.render.timezone
    snippet_chars = config.render.attachment_snippet_chars

    desired: dict[str, _DesiredDocument] = {}
    for segment in segments:
        text = render_segment_document(
            segment, timezone=tz, attachment_snippet_chars=snippet_chars
        )
        if text is None:
            continue  # nothing renderable (e.g. all-unsent segment)
        document_id = segment_document_id(segment.stable_key)
        eligible_guids = tuple(
            sorted(
                {
                    att.source_guid
                    for message in segment.messages
                    for att in message.attachments
                    if att.content_eligible
                }
            )
        )
        desired[document_id] = _DesiredDocument(
            upsert=PlannedUpsert(
                document_id=document_id,
                kind="segment",
                content_sha256=sha256_text(text),
                staged_relpath=staged_relpath_for(document_id),
                gcs_object=gcs_object_for(document_id),
                chat_id=segment.chat_id,
                segment_id=segment.segment_id,
                attachment_chunk_id=None,
                people=segment.participant_short_names,
                mime_type=None,
                eligible_attachment_guids=eligible_guids,
                started_at=segment.started_at.isoformat(),
                ended_at=segment.ended_at.isoformat(),
                segment_key=segment.stable_key,
            ),
            text=text,
        )

    for chunk in chunks:
        text = render_chunk_document(chunk, timezone=tz)
        document_id = attachment_chunk_document_id(
            chunk.parent.stable_key, chunk.attachment_source_guid, chunk.kind, chunk.seq
        )
        desired[document_id] = _DesiredDocument(
            upsert=PlannedUpsert(
                document_id=document_id,
                kind="attachment_chunk",
                content_sha256=sha256_text(text),
                staged_relpath=staged_relpath_for(document_id),
                gcs_object=gcs_object_for(document_id),
                chat_id=chunk.parent.chat_id,
                segment_id=chunk.parent.segment_id,
                attachment_chunk_id=chunk.chunk_id,
                people=chunk.parent.participant_short_names,
                mime_type=chunk.mime_type or "unknown",
                eligible_attachment_guids=(chunk.attachment_source_guid,),
                started_at=chunk.parent.started_at.isoformat(),
                ended_at=chunk.parent.ended_at.isoformat(),
                segment_key=chunk.parent.stable_key,
            ),
            text=text,
        )
    return desired


def _current_pushed_state(conn: psycopg.Connection) -> dict[str, str | None]:
    """document_id -> current_content_sha256 for docs the external
    store currently holds."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT document_id, current_content_sha256 FROM export_document "
            "WHERE state = 'pushed'"
        )
        return {str(doc_id): sha for doc_id, sha in cur.fetchall()}


def _upsert_to_manifest_entry(upsert: PlannedUpsert) -> dict[str, Any]:
    return {
        "document_id": upsert.document_id,
        "kind": upsert.kind,
        "content_sha256": upsert.content_sha256,
        "staged_relpath": upsert.staged_relpath,
        "gcs_object": upsert.gcs_object,
        "chat_id": upsert.chat_id,
        "segment_id": upsert.segment_id,
        "attachment_chunk_id": upsert.attachment_chunk_id,
        "people": list(upsert.people),
        "mime_type": upsert.mime_type,
        "eligible_attachment_guids": list(upsert.eligible_attachment_guids),
        "started_at": upsert.started_at,
        "ended_at": upsert.ended_at,
        "segment_key": upsert.segment_key,
    }


def plan_export(
    conn: psycopg.Connection, config: Config, *, mode: str = "reconcile"
) -> PlanResult:
    """Produce one hash-pinned export plan. Writes staging bytes under
    `$DATA_ROOT/export/staging/<run>/`, records `export_run` +
    `export_run_item`, and returns the review summary. Never uploads
    anything. The caller commits."""
    if mode not in ("reconcile", "purge"):
        raise ExportPlanError(f"unknown export plan mode '{mode}'")

    desired = build_desired_documents(conn, config)
    current = _current_pushed_state(conn)

    upserts: list[PlannedUpsert] = []
    unchanged = 0
    for document_id in sorted(desired):
        doc = desired[document_id]
        if current.get(document_id) == doc.upsert.content_sha256:
            unchanged += 1
        else:
            upserts.append(doc.upsert)
    deletes = [
        PlannedDelete(document_id=document_id, gcs_object=gcs_object_for(document_id))
        for document_id in sorted(current)
        if document_id not in desired
    ]

    allowlist = snapshot_allowlist(conn)
    allowlist_json = canonical_json(allowlist)
    config_sha = export_config_sha256(config)

    manifest: dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "mode": mode,
        "allowlist_sha256": sha256_text(allowlist_json),
        "config_sha256": config_sha,
        "upserts": [_upsert_to_manifest_entry(u) for u in upserts],
        "deletes": [
            {"document_id": d.document_id, "gcs_object": d.gcs_object} for d in deletes
        ],
    }
    manifest_text = canonical_json(manifest)
    manifest_sha = sha256_text(manifest_text)

    approval_reasons = compute_approval_requirements(conn, manifest)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO export_run
                (mode, allowlist_snapshot, config_sha256, manifest_sha256,
                 doc_count, status)
            VALUES (%s, %s, %s, %s, %s, 'running')
            RETURNING export_run_id
            """,
            (mode, Jsonb(allowlist), config_sha, manifest_sha, len(upserts) + len(deletes)),
        )
        row = cur.fetchone()
        if row is None:
            raise ExportPlanError("failed to create export_run row")
        run_id = int(row[0])

    staging_dir = staging_dir_for(config, run_id)
    if staging_dir.exists():
        raise ExportPlanError(
            f"staging dir already exists: {staging_dir} — staging is immutable "
            f"per run; refusing to overwrite"
        )
    (staging_dir / DOCS_SUBDIR).mkdir(parents=True)

    metadata_lines: list[str] = []
    for upsert in upserts:
        doc = desired[upsert.document_id]
        staged_path = staging_dir / upsert.staged_relpath
        staged_path.write_text(doc.text, encoding="utf-8")
        metadata_lines.append(
            metadata_jsonl_line(
                document_id=upsert.document_id,
                gcs_bucket=config.export.gcs_bucket,
                gcs_object=upsert.gcs_object,
                struct_data=structdata_for(
                    people=upsert.people,
                    started_at=upsert.started_at,
                    ended_at=upsert.ended_at,
                    segment_key=upsert.segment_key,
                    document_kind=upsert.kind,
                ),
            )
        )
    (staging_dir / METADATA_FILENAME).write_text(
        "\n".join(metadata_lines) + ("\n" if metadata_lines else ""), encoding="utf-8"
    )
    (staging_dir / MANIFEST_FILENAME).write_text(manifest_text, encoding="utf-8")

    report_text = build_review_report(
        conn,
        manifest=manifest,
        staging_dir=staging_dir,
        approval_reasons=approval_reasons,
        unchanged_count=unchanged,
    )
    report_path = staging_dir / REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8")

    with conn.cursor() as cur:
        for upsert in upserts:
            cur.execute(
                """
                INSERT INTO export_run_item
                    (export_run_id, document_id, action, content_sha256,
                     staged_relpath, result_state)
                VALUES (%s, %s, 'upsert', %s, %s, 'staged')
                """,
                (run_id, upsert.document_id, upsert.content_sha256, upsert.staged_relpath),
            )
        for delete in deletes:
            cur.execute(
                """
                INSERT INTO export_run_item
                    (export_run_id, document_id, action, result_state)
                VALUES (%s, %s, 'delete', 'staged')
                """,
                (run_id, delete.document_id),
            )
        cur.execute(
            "UPDATE export_run SET status = 'planned', finished_at = now() "
            "WHERE export_run_id = %s",
            (run_id,),
        )

    return PlanResult(
        run_id=run_id,
        mode=mode,
        manifest_sha256=manifest_sha,
        staging_dir=str(staging_dir),
        upsert_count=len(upserts),
        delete_count=len(deletes),
        unchanged_count=unchanged,
        approval_required=bool(approval_reasons),
        approval_reasons=approval_reasons,
        report_path=str(report_path),
    )


__all__ = [
    "build_desired_documents",
    "export_config_sha256",
    "load_manifest",
    "plan_export",
    "staging_dir_for",
]
