"""Domain objects for S4 sessionization/segmentation (SPEC §8 S4).

Deliberately decoupled from psycopg row shapes: `imsg.segment.pipeline`
is the only module that knows how to build these from a live connection
(and the only one exercised by the Postgres integration test); every
other module in this package (`sessionize`, `boundaries`, `render`,
`hashing`) is pure and takes/returns these dataclasses, so it can be
unit tested without a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AttachmentSnippet:
    """One attachment rendered inline in a message (SPEC §9.1 example line)."""

    attachment_key: str
    kind: str  # 'pdf' | 'image' | 'audio' | 'video' | 'other' — display label only
    filename: str | None
    caption: str | None = None
    ocr_text: str | None = None
    transcript: str | None = None
    pdf_text: str | None = None


@dataclass(frozen=True, slots=True)
class EditVersion:
    """A prior version of an edited message, from `message_version`."""

    version_idx: int
    text: str
    edited_at: datetime | None


@dataclass(frozen=True, slots=True)
class MessageForSegmentation:
    """One message as segmentation sees it: already policy-filtered and
    identity-resolved (S3's invariant guarantees `sender_short_name` is
    never a raw handle) by the caller (`imsg.segment.pipeline`).

    `text` is always the *current* (latest-edit) body — D1: prior
    versions never replace it, they are additive via `edit_history`
    and only populated when `policy.index_edit_history` is true.
    """

    message_id: int
    source_guid: str
    chat_id: int
    sent_at: datetime
    is_from_me: bool
    sender_short_name: str
    text: str | None
    is_unsent: bool
    is_edited: bool
    has_attachments: bool
    attachments: tuple[AttachmentSnippet, ...] = ()
    tapback_suffixes: tuple[str, ...] = ()
    edit_history: tuple[EditVersion, ...] = ()


@dataclass(frozen=True, slots=True)
class Session:
    """Pass-1 output: one time-gap-bounded run of messages in one chat."""

    chat_id: int
    started_at: datetime
    ended_at: datetime
    messages: tuple[MessageForSegmentation, ...]
    gap_hours: float
    """The `session_gap_hours` threshold used to build this session —
    stamped onto the `session` row (SPEC §7.2 `session.gap_hours`)."""


@dataclass(frozen=True, slots=True)
class SegmentDraft:
    """Pass-2 output: one segment within a session, not yet rendered/hashed."""

    session_started_at: datetime
    seq_in_session: int
    messages: tuple[MessageForSegmentation, ...]
    topic_label: str | None = None

    @property
    def started_at(self) -> datetime:
        return self.messages[0].sent_at

    @property
    def ended_at(self) -> datetime:
        return self.messages[-1].sent_at

    @property
    def message_count(self) -> int:
        return len(self.messages)


@dataclass(frozen=True, slots=True)
class RenderedSegment:
    """A `SegmentDraft` plus its rendered text, ready to persist."""

    draft: SegmentDraft
    rendered_text: str
    rendered_sha256: str
    token_count: int
    seg_config_hash: str
    stable_key: str


@dataclass(frozen=True, slots=True)
class PersistedSessionSpan:
    """The subset of a `session` row the incremental-frontier logic needs."""

    session_id: int
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True, slots=True)
class SegmentationRunReport:
    """Summary of one `run_segment_for_chat` call, for logging/CLI output."""

    chat_id: int
    sessions_written: int = 0
    segments_written: int = 0
    segments_deleted: int = 0
    fallback_sessions: int = 0
    """Sessions that fell back to session-as-segment because the
    boundary model failed or returned malformed output (SPEC §8 S4)."""
    skipped_unchanged: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)
    dry_run: bool = False
    """True when this report came from `run_segment_for_chat(dry_run=
    True)` (SPEC §8: "takes --dry-run where writes leave the
    machine") — `sessions_written`/`segments_written`/`segments_deleted`
    are the counts that *would* have been written/deleted (the real
    sessionize/boundary-detection/render computation still ran), but
    nothing was actually written to Postgres."""
