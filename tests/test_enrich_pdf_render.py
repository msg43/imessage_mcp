"""Real `pdftoppm` page rasterization for the scanned-OCR path (SPEC §8
S5b): one page at a time, each within the pixel ceiling, the page count
checked before anything renders."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from _hostile_attachments import png_size, write_pdf
from _pdf_fixtures import write_minimal_pdf
from imsg import constants
from imsg.enrich.pdf_info import PdfPage
from imsg.enrich.pdf_render import render_dpi, render_pdf_page, rendered_pages, rendered_pixels
from imsg.enrich.sandboxed_decoder import DecoderBudget
from imsg.errors import EnrichmentError, UntrustedAttachmentError

needs_poppler = pytest.mark.skipif(
    shutil.which("pdftoppm") is None or not Path("/usr/bin/sandbox-exec").exists(),
    reason="needs poppler and macOS sandbox-exec",
)


def _budget(tmp_path: Path) -> DecoderBudget:
    return DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)


# --------------------------------------------------------------------------
# choosing the resolution (pure)
# --------------------------------------------------------------------------


def test_an_ordinary_page_renders_at_300_dpi() -> None:
    letter = PdfPage(1, 612, 792)
    assert render_dpi(letter, max_pixels=constants.DEFAULT_MAX_IMAGE_PIXELS) == 300
    assert rendered_pixels(letter, 300) == 2550 * 3300


def test_a_poster_page_renders_at_the_highest_dpi_that_fits() -> None:
    """50 inches square is 225 megapixels at 300 dpi; 267 dpi is the most
    that fits under the default ceiling."""
    poster = PdfPage(1, 3600, 3600)
    ceiling = constants.DEFAULT_MAX_IMAGE_PIXELS
    dpi = render_dpi(poster, max_pixels=ceiling)
    assert dpi == 267
    assert rendered_pixels(poster, dpi) <= ceiling < rendered_pixels(poster, dpi + 1)


def test_the_largest_page_a_pdf_can_declare_still_fits() -> None:
    page = PdfPage(1, 14400, 14400)  # 200 inches, the PDF 1.7 maximum
    dpi = render_dpi(page, max_pixels=constants.DEFAULT_MAX_IMAGE_PIXELS)
    assert dpi == 66
    assert rendered_pixels(page, dpi) <= constants.DEFAULT_MAX_IMAGE_PIXELS


def test_a_page_too_large_even_at_1_dpi_is_refused() -> None:
    with pytest.raises(UntrustedAttachmentError, match="max_image_pixels"):
        render_dpi(PdfPage(3, 7_200_000, 7_200_000), max_pixels=1_000_000)


# --------------------------------------------------------------------------
# rendering, for real
# --------------------------------------------------------------------------


@needs_poppler
def test_renders_one_page_at_the_requested_resolution(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    write_minimal_pdf(pdf, ["Page One", "Page Two", "Page Three"])
    budget = _budget(tmp_path)
    image = render_pdf_page(pdf, 2, dpi=72, budget=budget)
    assert image.parent == budget.work_dir
    assert png_size(image) == (612, 792)


@needs_poppler
def test_pages_come_one_at_a_time_and_each_is_deleted_before_the_next(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    write_minimal_pdf(pdf, ["A", "B", "C", "D"])
    budget = _budget(tmp_path)
    seen: list[int] = []
    for page in rendered_pages(pdf, budget=budget, max_pages=1000, max_pixels=10**9):
        on_disk = sorted(p.name for p in budget.work_dir.glob("*.png"))
        assert on_disk == [page.path.name], f"more than one page image at once: {on_disk}"
        assert page.dpi == 300
        seen.append(page.number)
    assert seen == [1, 2, 3, 4]
    assert list(budget.work_dir.glob("*.png")) == []
    assert list(budget.work_dir.glob("pdftoppm-*.err")) == [], "per-page logs piled up"


@needs_poppler
def test_a_page_over_the_pixel_ceiling_is_rendered_smaller(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "tabloid.pdf", media_box=(0, 0, 1224, 1584))  # 17 x 22 in
    budget = _budget(tmp_path)
    ceiling = 2_000_000
    pages = [
        (page.dpi, png_size(page.path))
        for page in rendered_pages(pdf, budget=budget, max_pages=1000, max_pixels=ceiling)
    ]
    ((dpi, (width, height)),) = pages
    assert dpi < 300
    assert width * height <= ceiling


@needs_poppler
def test_a_document_over_the_page_ceiling_is_refused_before_any_page_renders(
    tmp_path: Path,
) -> None:
    pdf = write_pdf(tmp_path / "long.pdf", pages=40)
    budget = _budget(tmp_path)
    with pytest.raises(UntrustedAttachmentError, match="max_pdf_pages"):
        next(rendered_pages(pdf, budget=budget, max_pages=3, max_pixels=10**9))
    assert list(budget.work_dir.glob("*.png")) == []


@needs_poppler
def test_nonexistent_pdf_raises(tmp_path: Path) -> None:
    with pytest.raises(EnrichmentError):
        next(
            rendered_pages(
                tmp_path / "missing.pdf", budget=_budget(tmp_path), max_pages=10, max_pixels=10**9
            )
        )
