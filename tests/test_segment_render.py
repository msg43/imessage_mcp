"""Segment rendering (SPEC §9.1) — fictional personas only (D5)."""

from __future__ import annotations

from datetime import UTC, datetime

from imsg.segment.models import AttachmentSnippet, MessageForSegmentation, SegmentDraft
from imsg.segment.render import render_segment

_T0 = datetime(2023, 4, 12, 13, 58, tzinfo=UTC)
_T1 = datetime(2023, 4, 12, 14, 41, tzinfo=UTC)


def _msg(**overrides: object) -> MessageForSegmentation:
    base = {
        "message_id": 1,
        "source_guid": "g1",
        "chat_id": 1,
        "sent_at": _T0,
        "is_from_me": True,
        "sender_short_name": "owner",
        "text": "hello",
        "is_unsent": False,
        "is_edited": False,
        "has_attachments": False,
    }
    base.update(overrides)
    return MessageForSegmentation(**base)  # type: ignore[arg-type]


def test_dm_header_has_no_group_suffix() -> None:
    draft = SegmentDraft(session_started_at=_T0, seq_in_session=0, messages=(_msg(),))
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert text.startswith("Chat: Alice Example\n")
    assert "group" not in text.splitlines()[0]


def test_group_header_includes_display_name() -> None:
    draft = SegmentDraft(session_started_at=_T0, seq_in_session=0, messages=(_msg(),))
    text = render_segment(
        draft,
        participants=["Alice Example", "Bob Builder"],
        chat_kind="group",
        chat_display_name="Deck project",
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert text.splitlines()[0] == 'Chat: Alice Example, Bob Builder (group "Deck project")'


def test_owner_never_listed_in_participants() -> None:
    """The header's participant list is caller-supplied and must already
    exclude the owner (SPEC §9.1) — this test locks the render function's
    contract: it renders whatever it's given, it does not itself filter."""
    draft = SegmentDraft(session_started_at=_T0, seq_in_session=0, messages=(_msg(),))
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert "owner" not in text.splitlines()[0]


def test_message_line_format() -> None:
    draft = SegmentDraft(
        session_started_at=_T0,
        seq_in_session=0,
        messages=(_msg(text="did the revised bid come through?"),),
    )
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert "[13:58] owner: did the revised bid come through?" in text


def test_time_line_same_day() -> None:
    draft = SegmentDraft(
        session_started_at=_T0,
        seq_in_session=0,
        messages=(_msg(sent_at=_T0), _msg(message_id=2, sent_at=_T1)),
    )
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert text.splitlines()[1] == "Time: 2023-04-12 13:58 \u2013 14:41 UTC"


def test_pdf_attachment_renders_with_full_text_pointer() -> None:
    att = AttachmentSnippet(
        attachment_key="att_deadbeef",
        kind="pdf",
        filename="bid-rev3.pdf",
        pdf_text="Deck rebuild, materials $14,200",
    )
    draft = SegmentDraft(
        session_started_at=_T0,
        seq_in_session=0,
        messages=(_msg(text=None, has_attachments=True, attachments=(att,)),),
    )
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert 'get_attachment_text("att_deadbeef")' in text
    assert "Deck rebuild" in text


def test_image_attachment_renders_caption_and_ocr() -> None:
    att = AttachmentSnippet(
        attachment_key="att_1",
        kind="image",
        filename="deck.jpg",
        caption="photo of concrete deck footing with rebar",
        ocr_text=None,
    )
    draft = SegmentDraft(
        session_started_at=_T0,
        seq_in_session=0,
        messages=(_msg(text=None, has_attachments=True, attachments=(att,)),),
    )
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert 'caption "photo of concrete deck footing with rebar"' in text
    assert 'ocr "—"' in text


def test_attachment_snippet_truncates_at_configured_length() -> None:
    att = AttachmentSnippet(
        attachment_key="att_1",
        kind="pdf",
        filename="long.pdf",
        pdf_text="x" * 500,
    )
    draft = SegmentDraft(
        session_started_at=_T0,
        seq_in_session=0,
        messages=(_msg(text=None, has_attachments=True, attachments=(att,)),),
    )
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=50,
    )
    assert "x" * 51 not in text
    assert "…" in text


def test_tapback_renders_as_inline_suffix() -> None:
    draft = SegmentDraft(
        session_started_at=_T0,
        seq_in_session=0,
        messages=(_msg(text="see attached", tapback_suffixes=("(♥ alice)",)),),
    )
    text = render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=200,
    )
    assert "see attached (♥ alice)" in text


def test_render_is_deterministic() -> None:
    draft = SegmentDraft(session_started_at=_T0, seq_in_session=0, messages=(_msg(),))
    kwargs = {
        "participants": ["Alice Example"],
        "chat_kind": "dm",
        "chat_display_name": None,
        "timezone": "UTC",
        "attachment_snippet_chars": 200,
    }
    assert render_segment(draft, **kwargs) == render_segment(draft, **kwargs)  # type: ignore[arg-type]


def _render_one(att: AttachmentSnippet, *, snippet_chars: int = 200) -> str:
    draft = SegmentDraft(
        session_started_at=_T0,
        seq_in_session=0,
        messages=(_msg(text=None, has_attachments=True, attachments=(att,)),),
    )
    return render_segment(
        draft,
        participants=["Alice Example"],
        chat_kind="dm",
        chat_display_name=None,
        timezone="UTC",
        attachment_snippet_chars=snippet_chars,
    )


def test_other_attachment_with_document_text_renders_a_snippet_and_the_full_text_pointer() -> None:
    """A contact card, a Word document or a text file: once `doc_text`
    has run, the segment carries its text the way a PDF's does, so the
    reranker and the caller see what matched."""
    att = AttachmentSnippet(
        attachment_key="att_card",
        kind="other",
        filename="Alice Example.vcf",
        document_text="Name: Alice Example\nOrganization: Acme Construction",
    )
    text = _render_one(att)
    assert (
        '[attachment "Alice Example.vcf": "Name: Alice Example\nOrganization: Acme Construction" '
        '— full text via get_attachment_text("att_card")]'
    ) in text


def test_document_text_snippet_truncates_like_every_other_attachment_text() -> None:
    att = AttachmentSnippet(
        attachment_key="att_doc", kind="other", filename="plan.docx", document_text="x" * 500
    )
    text = _render_one(att, snippet_chars=20)
    assert '"' + "x" * 20 + "…" + '"' in text


def test_other_attachment_without_document_text_renders_exactly_as_before() -> None:
    """Why `RENDERER_VERSION` does not move: every segment stored before
    `doc_text` existed has no document text, and renders byte for byte
    the same, so nothing already indexed goes stale."""
    att = AttachmentSnippet(attachment_key="att_zip", kind="other", filename="bundle.zip")
    assert '[attachment "bundle.zip"]' in _render_one(att)
    unnamed = AttachmentSnippet(attachment_key="att_zip", kind="other", filename=None)
    assert '[attachment "att_zip"]' in _render_one(unnamed)
