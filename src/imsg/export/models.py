"""Domain objects for the export gate (SPEC §11).

Pure dataclasses — `eligibility`, `render`, and `documents` operate on
these without a database; `planner`/`push` are the only modules that
build them from a live connection (mirroring the segment package's
layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

DENY_NO_PARTICIPANTS = "no-participants"
DENY_PARTICIPANT_NOT_ALLOWLISTED = "participant-not-allowlisted"
DENY_UNRESOLVED_SOURCE_PARTICIPANT = "unresolved-source-participant"
DENY_SOURCE_PERSON_NOT_ALLOWLISTED = "resolved-source-person-not-allowlisted"
DENY_UNRESOLVED_SENDER = "unresolved-message-sender"
DENY_SENDER_NOT_ALLOWLISTED = "sender-not-allowlisted"
DENY_TAPBACK_SENDER = "tapback-sender-unresolved-or-not-allowlisted"


@dataclass(frozen=True, slots=True)
class ChatEligibility:
    """One chat's verdict. `eligible` is derived, never stored: it is
    true only when the chat has at least one participant and *zero*
    deny reasons. Any chat absent from the eligibility index is denied
    by construction (callers must treat a missing key as deny)."""

    chat_id: int
    participant_count: int
    deny_reasons: frozenset[str]

    @property
    def eligible(self) -> bool:
        return self.participant_count > 0 and not self.deny_reasons


# ---------------------------------------------------------------------------
# Renderable rows (built by planner from the schema, consumed by render)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExportAttachment:
    """One attachment as it appears on one message inside a segment.

    `content_eligible` is the SEPARATE attachment gate (SPEC §11.2):
    false means the renderer must emit a content-free placeholder —
    no filename, no MIME type, no enrichment text."""

    attachment_id: int
    source_guid: str
    filename: str | None
    mime_type: str | None
    content_eligible: bool
    caption: str | None = None
    ocr_text: str | None = None
    transcript: str | None = None
    pdf_text: str | None = None


@dataclass(frozen=True, slots=True)
class ExportMessage:
    """One non-unsent message with its *latest* text only. The planner
    never populates prior edit versions here — there is deliberately no
    field to put them in (D1: hard-coded, not filtered)."""

    message_id: int
    sent_at: datetime
    sender_short_name: str
    text: str | None
    attachments: tuple[ExportAttachment, ...] = ()
    tapback_suffixes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExportSegment:
    """A segment plus everything the renderer needs, already gated."""

    segment_id: int
    stable_key: str
    chat_id: int
    chat_kind: str  # 'dm' | 'group'
    chat_display_name: str | None
    participant_short_names: tuple[str, ...]  # sorted, owner included
    started_at: datetime
    ended_at: datetime
    messages: tuple[ExportMessage, ...]


@dataclass(frozen=True, slots=True)
class ExportChunk:
    """One `attachment_chunk` row eligible for export under one parent
    segment (SPEC §11.3: one document per authorized parent)."""

    chunk_id: int
    attachment_id: int
    attachment_source_guid: str
    mime_type: str | None
    kind: str
    seq: int
    text: str
    parent: ExportSegment


# ---------------------------------------------------------------------------
# Plan / manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannedUpsert:
    document_id: str
    kind: str  # 'segment' | 'attachment_chunk'
    content_sha256: str
    staged_relpath: str
    gcs_object: str
    chat_id: int
    segment_id: int
    attachment_chunk_id: int | None
    people: tuple[str, ...]
    mime_type: str | None
    eligible_attachment_guids: tuple[str, ...]
    started_at: str  # ISO-8601 — carried into metadata structData
    ended_at: str
    segment_key: str


@dataclass(frozen=True, slots=True)
class PlannedDelete:
    document_id: str
    gcs_object: str


@dataclass(frozen=True, slots=True)
class PlanResult:
    run_id: int
    mode: str
    manifest_sha256: str
    staging_dir: str
    upsert_count: int
    delete_count: int
    unchanged_count: int
    approval_required: bool
    approval_reasons: tuple[str, ...]
    report_path: str


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    run_id: int
    approval_id: str
    approved_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class PushResult:
    run_id: int
    status: str  # 'ok' | 'failed'
    pushed: int
    deleted: int
    failed: int
    skipped_already_done: int
    notes: tuple[str, ...] = field(default_factory=tuple)
