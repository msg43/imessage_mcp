"""`pdfinfo` page counts and page sizes (`imsg.enrich.pdf_info`), read
before anything extracts or renders a PDF. Real poppler, synthetic PDFs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from _hostile_attachments import write_pdf
from imsg.enrich.pdf_info import PdfPage, read_pdf_info
from imsg.enrich.sandboxed_decoder import DecoderBudget
from imsg.errors import EnrichmentError, UntrustedAttachmentError

pytestmark = pytest.mark.skipif(
    shutil.which("pdfinfo") is None or not Path("/usr/bin/sandbox-exec").exists(),
    reason="needs poppler's pdfinfo and macOS sandbox-exec",
)


def _budget(tmp_path: Path) -> DecoderBudget:
    return DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)


def test_reads_the_page_count(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", pages=7)
    info = read_pdf_info(pdf, budget=_budget(tmp_path), max_pages=1000)
    assert info.page_count == 7
    assert info.pages == ()


def test_reads_every_pages_media_box(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", pages=3, media_box=(0, 0, 612, 792))
    info = read_pdf_info(pdf, budget=_budget(tmp_path), max_pages=1000, page_sizes=True)
    assert info.pages == tuple(PdfPage(n, 612.0, 792.0) for n in (1, 2, 3))


def test_a_small_crop_box_does_not_hide_a_huge_media_box(tmp_path: Path) -> None:
    """`pdftoppm` renders the media box; `pdfinfo`'s "Page size" line is the
    crop box. A 2000 pt media box behind a 100 pt crop box must be seen
    as 2000 pt."""
    pdf = write_pdf(
        tmp_path / "doc.pdf", media_box=(0, 0, 2000, 2000), crop_box=(0, 0, 100, 100)
    )
    info = read_pdf_info(pdf, budget=_budget(tmp_path), max_pages=1000, page_sizes=True)
    assert info.pages == (PdfPage(1, 2000.0, 2000.0),)


def test_too_many_pages_is_refused_permanently(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", pages=12)
    with pytest.raises(UntrustedAttachmentError, match="max_pdf_pages"):
        read_pdf_info(pdf, budget=_budget(tmp_path), max_pages=10, page_sizes=True)


def test_a_file_that_is_not_a_pdf_is_an_ordinary_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(EnrichmentError) as excinfo:
        read_pdf_info(bad, budget=_budget(tmp_path), max_pages=1000)
    assert not isinstance(excinfo.value, UntrustedAttachmentError)
