"""Pure unit tests: export rendering (D1 hard-coding, withheld
attachments) and document identity (content-independent ids, D6)."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from imsg.export.documents import (
    attachment_chunk_document_id,
    canonical_json,
    gcs_object_for,
    metadata_jsonl_line,
    segment_document_id,
)
from imsg.export.models import ExportAttachment, ExportChunk, ExportMessage, ExportSegment
from imsg.export.render import (
    WITHHELD_PLACEHOLDER,
    render_chunk_document,
    render_segment_document,
)

_T = datetime(2024, 6, 1, 16, 0, tzinfo=UTC)


def _segment(messages: tuple[ExportMessage, ...]) -> ExportSegment:
    return ExportSegment(
        segment_id=1,
        stable_key="stable-abc",
        chat_id=1,
        chat_kind="group",
        chat_display_name="Deck project",
        participant_short_names=("alice", "bob", "owner"),
        started_at=_T,
        ended_at=_T,
        messages=messages,
    )


def _message(**overrides: object) -> ExportMessage:
    defaults: dict[str, object] = {
        "message_id": 10,
        "sent_at": _T,
        "sender_short_name": "alice",
        "text": "did the revised bid come through?",
        "attachments": (),
        "tapback_suffixes": (),
    }
    defaults.update(overrides)
    return ExportMessage(**defaults)  # type: ignore[arg-type]


def test_segment_document_renders_header_and_lines() -> None:
    doc = render_segment_document(
        _segment((_message(),)), timezone="America/Los_Angeles", attachment_snippet_chars=200
    )
    assert doc is not None
    assert doc.startswith('Chat: alice, bob, owner (group "Deck project")\n')
    assert "Time: 2024-06-01T09:00" in doc  # UTC 16:00 -> LA 09:00
    assert "[2024-06-01 09:00] alice: did the revised bid come through?" in doc


def test_empty_segment_renders_nothing_at_all() -> None:
    assert (
        render_segment_document(
            _segment(()), timezone="America/Los_Angeles", attachment_snippet_chars=200
        )
        is None
    )


def test_withheld_attachment_leaks_nothing_even_if_fields_are_populated() -> None:
    """Defense in depth: the planner strips filename/MIME/text from
    ineligible attachments before rendering, but even a fully populated
    ineligible attachment must render as the bare placeholder."""
    att = ExportAttachment(
        attachment_id=5,
        source_guid="att-guid-5",
        filename="very-private-photo-name.jpg",
        mime_type="image/jpeg",
        content_eligible=False,
        caption="a private caption",
        ocr_text="private ocr text",
        transcript="private transcript",
        pdf_text="private pdf text",
    )
    doc = render_segment_document(
        _segment((_message(attachments=(att,)),)),
        timezone="America/Los_Angeles",
        attachment_snippet_chars=200,
    )
    assert doc is not None
    assert WITHHELD_PLACEHOLDER in doc
    for leak in (
        "very-private-photo-name",
        "image/jpeg",
        "a private caption",
        "private ocr text",
        "private transcript",
        "private pdf text",
    ):
        assert leak not in doc


def test_eligible_attachment_renders_snippets_with_truncation() -> None:
    att = ExportAttachment(
        attachment_id=5,
        source_guid="att-guid-5",
        filename="bid-rev3.pdf",
        mime_type="application/pdf",
        content_eligible=True,
        pdf_text="Deck rebuild, materials $14,200 and quite a lot more text here",
    )
    doc = render_segment_document(
        _segment((_message(attachments=(att,)),)),
        timezone="America/Los_Angeles",
        attachment_snippet_chars=20,
    )
    assert doc is not None
    assert '[pdf "bid-rev3.pdf": "Deck rebuild, materi…"]' in doc
    # No MCP affordance in a corporate document:
    assert "get_attachment_text" not in doc


def test_rendering_is_deterministic() -> None:
    seg = _segment((_message(), _message(message_id=11, sender_short_name="bob", text="yes")))
    kwargs: dict[str, object] = {"timezone": "America/Los_Angeles", "attachment_snippet_chars": 200}
    assert render_segment_document(seg, **kwargs) == render_segment_document(seg, **kwargs)  # type: ignore[arg-type]


def test_chunk_document_carries_parent_metadata_and_full_text() -> None:
    chunk = ExportChunk(
        chunk_id=7,
        attachment_id=5,
        attachment_source_guid="att-guid-5",
        mime_type="application/pdf",
        kind="pdf_text",
        seq=2,
        text="the full chunk text, untruncated",
        parent=_segment((_message(),)),
    )
    doc = render_chunk_document(chunk, timezone="America/Los_Angeles")
    assert doc.startswith("Attachment content (pdf_text, part 2)\n")
    assert 'Chat: alice, bob, owner (group "Deck project")' in doc
    assert doc.endswith("the full chunk text, untruncated\n")


# --- document identity ------------------------------------------------------


def test_segment_document_id_is_content_independent() -> None:
    """D6: the id depends only on the stable key — the same document is
    UPDATED when content changes, never duplicated."""
    a = segment_document_id("stable-abc")
    b = segment_document_id("stable-abc")
    c = segment_document_id("stable-other")
    assert a == b
    assert a != c
    _assert_rfc1034_document_id(a)


def _assert_rfc1034_document_id(document_id: str) -> None:
    """D9: Discovery Engine constrains `Document.id` to RFC-1034, 1-63 chars.

    Asserted as properties rather than a fixed digest so the test fails on
    a real constraint violation instead of on any change to the hash input.
    Both properties matter and only one is about length: a bare hex digest
    starting with a digit is a valid 63-char string that still violates
    RFC-1034's leading-letter rule.
    """
    assert 1 <= len(document_id) <= 63, f"exceeds RFC-1034 limit: {len(document_id)}"
    assert re.fullmatch(r"[a-z][a-z0-9-]*", document_id), (
        f"not RFC-1034 preferred syntax (must start with a letter): {document_id!r}"
    )


def test_chunk_document_id_uses_structural_coordinates_only() -> None:
    a = attachment_chunk_document_id("stable-abc", "att-guid", "pdf_text", 0)
    same = attachment_chunk_document_id("stable-abc", "att-guid", "pdf_text", 0)
    other_seq = attachment_chunk_document_id("stable-abc", "att-guid", "pdf_text", 1)
    other_parent = attachment_chunk_document_id("stable-zzz", "att-guid", "pdf_text", 0)
    assert a == same
    assert len({a, other_seq, other_parent}) == 3
    for document_id in (a, other_seq, other_parent):
        _assert_rfc1034_document_id(document_id)


def test_gcs_object_is_a_pure_function_of_the_document_id() -> None:
    doc_id = segment_document_id("stable-abc")
    assert gcs_object_for(doc_id) == f"segments/{doc_id}.txt"


def test_canonical_json_is_order_insensitive_and_stable() -> None:
    assert canonical_json({"b": 1, "a": [2, 1]}) == canonical_json({"a": [2, 1], "b": 1})
    assert canonical_json({"a": "é"}) == '{"a":"\\u00e9"}'


def test_metadata_jsonl_line_shape() -> None:
    line = metadata_jsonl_line(
        document_id="doc1",
        gcs_bucket="acme-bucket",
        gcs_object="segments/doc1.txt",
        struct_data={"people": ["alice"], "document_kind": "segment"},
    )
    assert '"id":"doc1"' in line
    assert '"uri":"gs://acme-bucket/segments/doc1.txt"' in line
    assert '"mimeType":"text/plain"' in line
