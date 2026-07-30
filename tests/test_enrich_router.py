"""MIME-based enrichment routing (SPEC §8 S5b)."""

from __future__ import annotations

import pytest

from imsg.enrich.router import kinds_for_mime
from imsg.errors import UntrustedAttachmentError


def test_pdf_routes_to_pdf_text_only() -> None:
    assert kinds_for_mime("application/pdf") == ("pdf_text",)


def test_image_routes_to_ocr_and_caption_always() -> None:
    assert kinds_for_mime("image/jpeg") == ("ocr", "caption")
    assert kinds_for_mime("image/png") == ("ocr", "caption")
    assert kinds_for_mime("image/heic") == ("ocr", "caption")


def test_audio_routes_to_transcript_only() -> None:
    assert kinds_for_mime("audio/mpeg") == ("transcript",)
    assert kinds_for_mime("audio/x-caf") == ("transcript",)


def test_video_routes_to_frame_ocr_caption_transcript() -> None:
    assert kinds_for_mime("video/mp4") == ("frame_ocr", "caption", "transcript")
    assert kinds_for_mime("video/quicktime") == ("frame_ocr", "caption", "transcript")


def test_unroutable_mime_raises() -> None:
    with pytest.raises(UntrustedAttachmentError):
        kinds_for_mime("application/x-msdownload")


def test_unroutable_mime_text_raises() -> None:
    with pytest.raises(UntrustedAttachmentError):
        kinds_for_mime("text/plain")
