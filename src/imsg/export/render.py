"""Export-document rendering (SPEC §11.3) — deliberately independent of
`imsg.segment.render`.

The local segment renderer honors `policy.index_unsent` /
`policy.index_edit_history` and may therefore embed unsent text and
edit history in `segment.rendered_text`. Export documents must NEVER
contain either, regardless of policy (D1, hard-coded) — so this module
re-renders from gated `ExportMessage` rows that structurally cannot
carry edit history, and the planner excludes unsent messages in SQL.
Reusing the local renderer here would make a local policy flip a
corporate-data leak; keep the two renderers separate.

Other export-specific differences from the local format:

- Participants render as `short_name`s (SPEC §11.3), owner included.
- No `get_attachment_text(...)` MCP affordances — meaningless in a
  Discovery Engine document, and no reason to ship opaque local keys.
- An attachment whose sender is not attachments-allowed renders as a
  single content-free placeholder: no filename, no MIME type, no
  enrichment text (a filename alone can carry private content).
- Message lines carry full dates — a corporate search hit must be
  self-describing without the surrounding thread.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from imsg.export.models import ExportAttachment, ExportChunk, ExportMessage, ExportSegment

RENDERER_VERSION = "1"
"""Folded into the export config hash by the planner — bump on any
output-format change so existing plans are recognized as stale."""

WITHHELD_PLACEHOLDER = "[attachment not exported]"

_EMPTY = "—"  # em dash, matching the local renderer's empty marker


def _truncate(text: str | None, *, snippet_chars: int) -> str:
    if not text:
        return _EMPTY
    stripped = text.strip()
    if not stripped:
        return _EMPTY
    if len(stripped) > snippet_chars:
        return stripped[:snippet_chars].rstrip() + "…"
    return stripped


def _classify(mime_type: str | None) -> str:
    if not mime_type:
        return "other"
    if mime_type == "application/pdf":
        return "pdf"
    for prefix in ("image/", "audio/", "video/"):
        if mime_type.startswith(prefix):
            return prefix.rstrip("/")
    return "other"


def _format_attachment(att: ExportAttachment, *, snippet_chars: int) -> str:
    if not att.content_eligible:
        # Content-free by design: no filename, no MIME, no text.
        return WITHHELD_PLACEHOLDER

    def trunc(s: str | None) -> str:
        return _truncate(s, snippet_chars=snippet_chars)

    kind = _classify(att.mime_type)
    if kind == "pdf":
        label = att.filename or "attachment.pdf"
        return f'[pdf "{label}": "{trunc(att.pdf_text)}"]'
    if kind == "image":
        return f'[image: caption "{trunc(att.caption)}" | ocr "{trunc(att.ocr_text)}"]'
    if kind == "audio":
        return f'[audio: transcript "{trunc(att.transcript)}"]'
    if kind == "video":
        return (
            f'[video: caption "{trunc(att.caption)}" | ocr "{trunc(att.ocr_text)}" '
            f'| transcript "{trunc(att.transcript)}"]'
        )
    label = att.filename or "attachment"
    return f'[attachment "{label}"]'


def _format_message(message: ExportMessage, *, tz: ZoneInfo, snippet_chars: int) -> str:
    local_time = message.sent_at.astimezone(tz)
    parts: list[str] = []
    if message.text:
        parts.append(message.text)
    parts.extend(
        _format_attachment(att, snippet_chars=snippet_chars) for att in message.attachments
    )
    body = " ".join(parts)
    for suffix in message.tapback_suffixes:
        body = f"{body} {suffix}" if body else suffix
    return f"[{local_time:%Y-%m-%d %H:%M}] {message.sender_short_name}: {body}"


def _header_lines(segment: ExportSegment, *, timezone: str) -> list[str]:
    tz = ZoneInfo(timezone)
    people = ", ".join(segment.participant_short_names)
    if segment.chat_kind == "group":
        title = (
            f' (group "{segment.chat_display_name}")'
            if segment.chat_display_name
            else " (group)"
        )
    else:
        title = ""
    started = segment.started_at.astimezone(tz)
    ended = segment.ended_at.astimezone(tz)
    return [
        f"Chat: {people}{title}",
        f"Time: {started:%Y-%m-%dT%H:%M} \u2013 {ended:%Y-%m-%dT%H:%M} {timezone}",
    ]


def render_segment_document(
    segment: ExportSegment, *, timezone: str, attachment_snippet_chars: int
) -> str | None:
    """The full text of one segment document, or None when nothing is
    renderable (e.g. every message in the segment was unsent — the
    planner then simply plans no document; an empty shell would still
    leak participant metadata for no benefit)."""
    if not segment.messages:
        return None
    tz = ZoneInfo(timezone)
    lines = _header_lines(segment, timezone=timezone)
    lines.append("---")
    lines.extend(
        _format_message(m, tz=tz, snippet_chars=attachment_snippet_chars)
        for m in segment.messages
    )
    return "\n".join(lines) + "\n"


def render_chunk_document(chunk: ExportChunk, *, timezone: str) -> str:
    """One attachment-chunk document: the full eligible chunk text plus
    parent-segment metadata (SPEC §11.3). The parent's stable key ships
    in metadata structData, not the body."""
    lines = _header_lines(chunk.parent, timezone=timezone)
    lines.insert(0, f"Attachment content ({chunk.kind}, part {chunk.seq})")
    lines.append("---")
    lines.append(chunk.text)
    return "\n".join(lines) + "\n"


__all__ = [
    "RENDERER_VERSION",
    "WITHHELD_PLACEHOLDER",
    "render_chunk_document",
    "render_segment_document",
]
