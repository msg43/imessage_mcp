"""Segment rendering (SPEC §9.1) — builds `segment.rendered_text`, the
unit that is returned by `search_messages` *and* fed to FTS/embedding
(after `imsg.textnorm.normalize_text`, applied by S6 at the point of
use — see that module's docstring for why normalization is not baked
into the stored copy).

`RENDERER_VERSION` is folded into `seg_config_hash` (`imsg.segment.
hashing`) — bump it any time this module's output format changes for
the same input, so existing segments are recognized as stale and
re-rendered rather than silently left in an old format.
"""

from __future__ import annotations

from collections.abc import Sequence
from zoneinfo import ZoneInfo

from imsg.segment.models import AttachmentSnippet, MessageForSegmentation, SegmentDraft

RENDERER_VERSION = "1"
"""Unchanged by `WITHHELD_ATTACHMENT`: segmentation never renders a
withheld attachment, so no stored text renders differently.

Unchanged by `DELETED_LABEL` and the holding-chat header (D13,
2026-09-23) for the same reason: both render only for rows that did not
exist before migration 0007 (a message with `deleted_at`, a chat with
`unfiled_key`), and every other input renders byte for byte as before.
A message that gains `deleted_at` moves its own `updated_at`, so its
chat is re-segmented and picks up the label then. Bumping the version
would instead re-render and re-embed the whole corpus for nothing."""

DELETED_LABEL = "[deleted]"
"""Leads the body of a message in Apple's "Recently Deleted" (D13:
deleted messages are searchable, labelled, and not hidden)."""

_EMPTY_SNIPPET = "—"  # em dash, matches the SPEC §9.1 example's "ocr \"—\""

WITHHELD_ATTACHMENT = "[attachment withheld]"
"""What an attachment the separate attachment gate denies renders as on
the public surface under `allowlist` scope (SPEC §11.2): content-free by
design, like export's placeholder — a filename, a kind or an opaque key
can each say more than the owner allowlisted."""


def _truncate(text: str | None, *, snippet_chars: int) -> str:
    if not text:
        return _EMPTY_SNIPPET
    stripped = text.strip()
    if not stripped:
        return _EMPTY_SNIPPET
    if len(stripped) > snippet_chars:
        return stripped[:snippet_chars].rstrip() + "…"
    return stripped


def _format_attachment(att: AttachmentSnippet, *, snippet_chars: int) -> str:
    if att.withheld:
        return WITHHELD_ATTACHMENT

    def trunc(s: str | None) -> str:
        return _truncate(s, snippet_chars=snippet_chars)

    if att.kind == "pdf":
        label = att.filename or "attachment.pdf"
        return (
            f'[pdf "{label}": "{trunc(att.pdf_text)}" — full text via '
            f'get_attachment_text("{att.attachment_key}")]'
        )
    if att.kind == "image":
        return f'[image: caption "{trunc(att.caption)}" | ocr "{trunc(att.ocr_text)}"]'
    if att.kind == "audio":
        return f'[audio: transcript "{trunc(att.transcript)}"]'
    if att.kind == "video":
        return (
            f'[video: caption "{trunc(att.caption)}" | ocr "{trunc(att.ocr_text)}" '
            f'| transcript "{trunc(att.transcript)}"]'
        )
    label = att.filename or att.attachment_key
    return f'[attachment "{label}"]'


def _format_message(message: MessageForSegmentation, *, tz: ZoneInfo, snippet_chars: int) -> str:
    local_time = message.sent_at.astimezone(tz)
    parts: list[str] = []
    if message.text:
        parts.append(message.text)
    parts.extend(_format_attachment(att, snippet_chars=snippet_chars) for att in message.attachments)
    body = " ".join(parts)
    if message.is_edited and message.edit_history:
        history = " -> ".join(f'"{v.text}"' for v in message.edit_history)
        body = f"{body} [edited from: {history}]" if body else f"[edited from: {history}]"
    for suffix in message.tapback_suffixes:
        body = f"{body} {suffix}" if body else suffix
    if message.is_deleted:
        body = f"{DELETED_LABEL} {body}" if body else DELETED_LABEL
    return f"[{local_time:%H:%M}] {message.sender_short_name}: {body}"


def render_message_line(
    message: MessageForSegmentation, *, timezone: str, attachment_snippet_chars: int
) -> str:
    """Public one-message wrapper around the same per-message rendering
    `render_segment` uses internally (SPEC §9.1's `[HH:MM] sender: ...`
    line shape) — reused by `imsg.retrieval` for `get_conversation`
    (§10.2), which returns a message-granular window rather than a
    whole rendered segment. Keeping exactly one implementation of the
    per-message line format means a segment's rendered text and a
    `get_conversation` window can never drift out of sync with each
    other."""
    tz = ZoneInfo(timezone)
    return _format_message(message, tz=tz, snippet_chars=attachment_snippet_chars)


def render_segment(
    draft: SegmentDraft,
    *,
    participants: Sequence[str],
    chat_kind: str,
    chat_display_name: str | None,
    timezone: str,
    attachment_snippet_chars: int,
    unfiled: bool = False,
) -> str:
    """SPEC §9.1: build the rendered/indexed/returned text for one segment.

    `unfiled` marks a holding chat (D13: messages no real chat could be
    named for). Its name says so (`Unfiled: ...`) and is shown in the
    header whatever the chat's kind, so a reader never takes a holding
    chat for a real 1:1 conversation. A real chat renders exactly as
    before.

    `participants` are the chat's *other* participants' `display_name`s
    (the owner is never listed in their own "Chat: ..." header, matching
    the SPEC §9.1 example) — never raw handles (hard requirement 3);
    per-message sender labels use the finer-grained `short_name`
    (`"owner"` for `is_from_me`) set by the caller on each
    `MessageForSegmentation.sender_short_name`. `chat_kind` is `'dm'` or
    `'group'` (`imsg.config.schema` doesn't own this enum; it mirrors
    the `chat_kind` Postgres enum from migration 0001).
    """
    tz = ZoneInfo(timezone)
    started_local = draft.started_at.astimezone(tz)
    ended_local = draft.ended_at.astimezone(tz)

    header_people = ", ".join(participants)
    if chat_kind == "group":
        title = f' (group "{chat_display_name}")' if chat_display_name else " (group)"
    elif unfiled and chat_display_name:
        title = f" ({chat_display_name})"
    else:
        title = ""
    chat_line = f"Chat: {header_people}{title}"

    if started_local.date() == ended_local.date():
        time_line = f"Time: {started_local:%Y-%m-%d %H:%M} \u2013 {ended_local:%H:%M} {timezone}"
    else:
        time_line = (
            f"Time: {started_local:%Y-%m-%d %H:%M} \u2013 "
            f"{ended_local:%Y-%m-%d %H:%M} {timezone}"
        )

    lines = [chat_line, time_line, "---"]
    lines.extend(
        _format_message(m, tz=tz, snippet_chars=attachment_snippet_chars) for m in draft.messages
    )
    return "\n".join(lines)


__all__ = [
    "DELETED_LABEL",
    "RENDERER_VERSION",
    "WITHHELD_ATTACHMENT",
    "render_message_line",
    "render_segment",
]
