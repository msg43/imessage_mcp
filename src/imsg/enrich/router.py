"""MIME-based enrichment routing table (SPEC §8 S5b): "vision is a
branch, not the trunk" (architecture §5.5) — which enrichment kinds an
attachment needs is a pure function of its sniffed MIME type, never the
filename extension, and never an ML classifier's decision.
"""

from __future__ import annotations

from imsg.errors import UntrustedAttachmentError

PDF_MIME = "application/pdf"


def kinds_for_mime(mime_type: str) -> tuple[str, ...]:
    """The `enrichment_kind` values this MIME type unconditionally
    enqueues. PDF's conditional `ocr` kind (SPEC: scanned pages fall
    under `pdf_scanned_threshold_chars_per_page` also enqueue OCR) is
    NOT decided here — that's `imsg.enrich.pdf_text.is_scanned`'s job,
    applied after the text layer has actually been extracted.

    Raises `UntrustedAttachmentError` for any MIME type this pipeline
    has no enrichment route for — an unroutable attachment is a typed
    permanent failure (SPEC §8 S5b), not a silent no-op.
    """
    if mime_type == PDF_MIME:
        return ("pdf_text",)
    if mime_type.startswith("image/"):
        return ("ocr", "caption")
    if mime_type.startswith("audio/"):
        return ("transcript",)
    if mime_type.startswith("video/"):
        return ("frame_ocr", "caption", "transcript")
    raise UntrustedAttachmentError(
        f"MIME type {mime_type!r} has no enrichment route — not a PDF, image, "
        f"audio, or video type this pipeline understands"
    )


__all__ = ["PDF_MIME", "kinds_for_mime"]
