"""MIME-based enrichment routing (SPEC §8 S5b)."""

from __future__ import annotations

import pytest

from imsg.enrich.router import (
    DEFAULT_CLAIM_ORDER,
    ENRICHMENT_KINDS,
    document_format_for_mime,
    kinds_for_mime,
    parse_kinds,
    route_for_mime,
)
from imsg.errors import ConfigError, UntrustedAttachmentError


def test_pdf_routes_to_pdf_text_only() -> None:
    assert kinds_for_mime("application/pdf") == ("pdf_text",)


def test_image_routes_to_ocr_and_caption_always() -> None:
    assert kinds_for_mime("image/jpeg") == ("ocr", "caption")
    assert kinds_for_mime("image/png") == ("ocr", "caption")
    assert kinds_for_mime("image/heic") == ("ocr", "caption")
    # Link-preview payloads and favicons sniff as ordinary raster types.
    assert kinds_for_mime("image/vnd.microsoft.icon") == ("ocr", "caption")
    assert kinds_for_mime("image/webp") == ("ocr", "caption")


def test_audio_routes_to_transcript_only() -> None:
    assert kinds_for_mime("audio/mpeg") == ("transcript",)
    assert kinds_for_mime("audio/x-caf") == ("transcript",)
    assert kinds_for_mime("audio/amr") == ("transcript",)


def test_video_routes_to_frame_ocr_caption_transcript() -> None:
    assert kinds_for_mime("video/mp4") == ("frame_ocr", "caption", "transcript")
    assert kinds_for_mime("video/quicktime") == ("frame_ocr", "caption", "transcript")


def test_unroutable_mime_raises() -> None:
    with pytest.raises(UntrustedAttachmentError):
        kinds_for_mime("application/x-msdownload")


@pytest.mark.parametrize(
    ("mime", "fmt"),
    [
        ("text/plain", "plain"),
        ("text/markdown", "plain"),
        ("text/csv", "plain"),
        ("text/calendar", "plain"),
        ("text/x-shellscript", "plain"),
        ("application/json", "plain"),
        ("text/xml", "plain"),
        ("text/vcard", "vcard"),
        ("text/x-vcard", "vcard"),
        ("text/directory", "vcard"),
        ("text/html", "html"),
        ("image/svg+xml", "html"),
        ("text/rtf", "rtf"),
        ("application/rtf", "rtf"),
        ("application/msword", "doc"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
        ("application/vnd.oasis.opendocument.text", "odt"),
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"),
    ],
)
def test_text_bearing_documents_route_to_doc_text(mime: str, fmt: str) -> None:
    assert kinds_for_mime(mime) == ("doc_text",)
    assert document_format_for_mime(mime) == fmt


@pytest.mark.parametrize(
    "mime",
    [
        "application/zip",
        "application/octet-stream",
        "inode/x-empty",
        "application/x-pkcs12",
        # Not raster images, whatever their top-level type says: Apple
        # Vision and PIL cannot decode them, so OCR and captioning could
        # only ever fail on them.
        "image/vnd.dwg",
    ],
)
def test_still_unroutable_types_route_nowhere(mime: str) -> None:
    assert route_for_mime(mime) == ()
    assert document_format_for_mime(mime) is None
    with pytest.raises(UntrustedAttachmentError):
        kinds_for_mime(mime)


def test_svg_is_text_not_a_raster_image() -> None:
    """Vision cannot open SVG (checked on macOS 26), so an SVG gets its
    text nodes, not OCR and a caption."""
    assert route_for_mime("image/svg+xml") == ("doc_text",)


def test_route_for_mime_agrees_with_kinds_for_mime_where_both_answer() -> None:
    for mime in ("application/pdf", "image/png", "audio/x-m4a", "video/mp4", "text/vcard"):
        assert route_for_mime(mime) == kinds_for_mime(mime)


def test_default_claim_order_runs_cheap_kinds_first_and_captions_last() -> None:
    assert set(DEFAULT_CLAIM_ORDER) == set(ENRICHMENT_KINDS)
    assert DEFAULT_CLAIM_ORDER[-1] == "caption"
    assert DEFAULT_CLAIM_ORDER.index("doc_text") < DEFAULT_CLAIM_ORDER.index("ocr")
    assert DEFAULT_CLAIM_ORDER.index("pdf_text") < DEFAULT_CLAIM_ORDER.index("ocr")
    for cheap in ("doc_text", "pdf_text", "ocr", "transcript", "frame_ocr"):
        assert DEFAULT_CLAIM_ORDER.index(cheap) < DEFAULT_CLAIM_ORDER.index("caption")


def test_parse_kinds_keeps_the_given_order_and_drops_repeats() -> None:
    assert parse_kinds("ocr, pdf_text,ocr ,transcript") == ("ocr", "pdf_text", "transcript")


@pytest.mark.parametrize("value", ["", " , ", "ocr,captions", "OCR"])
def test_parse_kinds_rejects_unknown_or_empty_lists(value: str) -> None:
    with pytest.raises(ConfigError):
        parse_kinds(value)
