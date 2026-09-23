"""MIME-based enrichment routing table (SPEC §8 S5b): "vision is a
branch, not the trunk" (architecture §5.5) — which enrichment kinds an
attachment needs is a pure function of its sniffed MIME type, never the
filename extension, and never an ML classifier's decision.

The table:

| Sniffed MIME type                              | Kinds                               |
|------------------------------------------------|-------------------------------------|
| `application/pdf`                              | `pdf_text` (a scan adds `ocr`)      |
| text-bearing documents (`document_format_for_mime`) | `doc_text`                     |
| `image/*`, raster only                         | `ocr`, `caption`                    |
| `audio/*`                                      | `transcript`                        |
| `video/*`                                      | `frame_ocr`, `caption`, `transcript`|

Text-bearing documents are contact cards (`text/vcard`, which is also
what Apple's shared-location cards sniff as), every other `text/*` type,
HTML and SVG, RTF and Word documents (read by macOS `textutil`), and
OOXML spreadsheets and slides. SVG sits here rather than under images:
Apple Vision cannot open it, and its only text is its text nodes.

Everything else routes nowhere: archives (`application/zip`, which is
also what iWork documents and wallet passes sniff as), unidentified
binary (`application/octet-stream`), empty files (`inode/x-empty`), and
image types that are not raster images (`image/vnd.dwg`).
"""

from __future__ import annotations

from imsg.errors import ConfigError, UntrustedAttachmentError

PDF_MIME = "application/pdf"

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
ODT_MIME = "application/vnd.oasis.opendocument.text"

ENRICHMENT_KINDS: tuple[str, ...] = (
    "doc_text",
    "pdf_text",
    "ocr",
    "transcript",
    "frame_ocr",
    "caption",
)
"""Every `enrichment_kind` value (migrations 0001 and 0009), in the
default claim order."""

DEFAULT_CLAIM_ORDER: tuple[str, ...] = ENRICHMENT_KINDS
"""The order a worker claims kinds in when `imsg enrich --kinds` does not
name one: cheapest and highest-recall first, captions last.

`doc_text` and `pdf_text` are text-layer reads measured in milliseconds;
`ocr` runs Apple Vision; `transcript` and `frame_ocr` decode media; one
`caption` costs about 14 s of GPU on the production host (CHANGELOG
2026-09-17) and there are tens of thousands of them. A caption backlog
never delays the text that is cheapest to make searchable."""

_DOCUMENT_FORMATS: dict[str, str] = {
    # contact cards, including Apple's shared-location cards
    "text/vcard": "vcard",
    "text/x-vcard": "vcard",
    "text/directory": "vcard",
    "text/x-vlocation": "vcard",
    # markup: visible text and text nodes only, parsed in-process
    "text/html": "html",
    "application/xhtml+xml": "html",
    "image/svg+xml": "html",
    # read by macOS textutil, sandboxed (`imsg.enrich.doc_text`)
    "text/rtf": "rtf",
    "application/rtf": "rtf",
    "application/msword": "doc",
    DOCX_MIME: "docx",
    ODT_MIME: "odt",
    # OOXML that textutil cannot read, parsed in-process
    XLSX_MIME: "xlsx",
    PPTX_MIME: "pptx",
    # structured text that is still text
    "application/json": "plain",
    "application/xml": "plain",
}

_NOT_RASTER_IMAGES = frozenset({"image/vnd.dwg", "image/vnd.dxf"})
"""`image/*` types that Apple Vision and PIL cannot decode: OCR and a
caption could only fail on them, five attempts each."""


def document_format_for_mime(mime_type: str) -> str | None:
    """Which `imsg.enrich.doc_text` reader handles this sniffed type —
    `vcard`, `html`, `plain`, `rtf`, `doc`, `docx`, `odt`, `xlsx` or
    `pptx` — or `None` when it is not a text-bearing document."""
    fmt = _DOCUMENT_FORMATS.get(mime_type)
    if fmt is not None:
        return fmt
    if mime_type.startswith("text/"):
        return "plain"
    return None


def route_for_mime(mime_type: str) -> tuple[str, ...]:
    """The `enrichment_kind` values this sniffed MIME type unconditionally
    enqueues, or `()` when there is no route. PDF's conditional `ocr` kind
    (SPEC: scanned pages fall under `pdf_scanned_threshold_chars_per_page`
    also enqueue OCR) is NOT decided here — that's
    `imsg.enrich.pdf_text.is_scanned`'s job, applied after the text layer
    has actually been extracted."""
    if mime_type == PDF_MIME:
        return ("pdf_text",)
    if document_format_for_mime(mime_type) is not None:
        return ("doc_text",)
    if mime_type.startswith("image/") and mime_type not in _NOT_RASTER_IMAGES:
        return ("ocr", "caption")
    if mime_type.startswith("audio/"):
        return ("transcript",)
    if mime_type.startswith("video/"):
        return ("frame_ocr", "caption", "transcript")
    return ()


def kinds_for_mime(mime_type: str) -> tuple[str, ...]:
    """`route_for_mime`, for a caller that must have a route.

    Raises `UntrustedAttachmentError` for any MIME type this pipeline
    has no enrichment route for — an unroutable attachment is a typed
    permanent failure (SPEC §8 S5b), not a silent no-op.
    """
    kinds = route_for_mime(mime_type)
    if not kinds:
        raise UntrustedAttachmentError(
            f"MIME type {mime_type!r} has no enrichment route — not a PDF, text-bearing "
            f"document, raster image, audio, or video type this pipeline understands"
        )
    return kinds


def parse_kinds(value: str) -> tuple[str, ...]:
    """`imsg enrich --kinds`: a comma-separated list of enrichment kinds.
    Order is kept, because it is the claim order; a repeat is dropped.
    Anything that is not exactly an `enrichment_kind` value is refused
    with the valid list, before any work starts."""
    items = [part.strip() for part in value.split(",") if part.strip()]
    valid = ", ".join(ENRICHMENT_KINDS)
    if not items:
        raise ConfigError(f"--kinds needs at least one enrichment kind (valid: {valid})")
    unknown = [item for item in items if item not in ENRICHMENT_KINDS]
    if unknown:
        raise ConfigError(
            f"unknown enrichment kind(s): {', '.join(unknown)} (valid: {valid})"
        )
    return tuple(dict.fromkeys(items))


__all__ = [
    "DEFAULT_CLAIM_ORDER",
    "DOCX_MIME",
    "ENRICHMENT_KINDS",
    "ODT_MIME",
    "PDF_MIME",
    "PPTX_MIME",
    "XLSX_MIME",
    "document_format_for_mime",
    "kinds_for_mime",
    "parse_kinds",
    "route_for_mime",
]
