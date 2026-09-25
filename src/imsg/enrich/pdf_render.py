"""Real `pdftoppm` (poppler) PDF-page rasterization for the scanned-OCR
path (SPEC §8 S5b: "pdftoppm 300 dpi -> Apple Vision per page").

**One page at a time, each within the pixel ceiling (2026-09-24, QA
review).** `pdftoppm` used to render every page of the document at 300
dpi into the task directory in one call, and the page count was checked
only afterwards, so every page of a 5,000-page PDF was written to disk
before the document was refused. Now:

- the page count and every page's media box are read first
  (`imsg.enrich.pdf_info`), and a document over
  `enrichment.limits.max_pdf_pages` is refused before anything renders;
- pages are rendered one at a time (`rendered_pages`), and each image is
  deleted before the next page renders, so the task directory holds one
  page image at most;
- each page renders at 300 dpi unless that would exceed
  `enrichment.limits.max_image_pixels`, in which case it renders at the
  highest resolution that fits (`render_dpi`). A 50-inch-square poster
  page would be 225 megapixels at 300 dpi; it renders at 267 dpi. A page
  too large to fit even at 1 dpi is refused;
- `pdftoppm` runs sandboxed and inside the task's budget
  (`imsg.enrich.sandboxed_decoder`), which also stops it on the memory
  ceiling: a page's embedded image is decoded at its own resolution,
  whatever the page size.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from imsg.enrich.pdf_info import PdfPage, read_pdf_info
from imsg.enrich.sandboxed_decoder import DecoderBudget, run_decoder
from imsg.errors import EnrichmentError, UntrustedAttachmentError

DEFAULT_DPI = 300
_POINTS_PER_INCH = 72.0


def rendered_pixels(page: PdfPage, dpi: int) -> int:
    """How many pixels `pdftoppm` draws for `page` at `dpi`, rounding each
    side up, so the figure is never below the real one."""
    width = math.ceil(page.width_pt * dpi / _POINTS_PER_INCH)
    height = math.ceil(page.height_pt * dpi / _POINTS_PER_INCH)
    return width * height


def render_dpi(page: PdfPage, *, max_pixels: int, dpi: int = DEFAULT_DPI) -> int:
    """`dpi`, or the highest resolution below it at which `page` fits in
    `max_pixels`. Raises `UntrustedAttachmentError` for a page that does
    not fit even at 1 dpi."""
    if rendered_pixels(page, dpi) <= max_pixels:
        return dpi
    square_inches = (page.width_pt / _POINTS_PER_INCH) * (page.height_pt / _POINTS_PER_INCH)
    fitted = min(dpi, max(1, math.floor(math.sqrt(max_pixels / square_inches))))
    while fitted > 1 and rendered_pixels(page, fitted) > max_pixels:
        fitted -= 1
    if rendered_pixels(page, fitted) > max_pixels:
        raise UntrustedAttachmentError(
            f"page {page.number} is {page.width_pt:g} x {page.height_pt:g} pt, more than "
            f"enrichment.limits.max_image_pixels ({max_pixels}) even at 1 dpi"
        )
    return fitted


def render_pdf_page(pdf_path: Path, page_number: int, *, dpi: int, budget: DecoderBudget) -> Path:
    """Render one page to a PNG in the work directory and return its path."""
    stem = budget.work_dir / f"page-{page_number:05d}"
    run = run_decoder(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            os.fspath(pdf_path.absolute()),
            os.fspath(stem),
        ],
        budget=budget,
        name="pdftoppm",
        subject=pdf_path,
    )
    try:
        if run.returncode != 0:
            raise EnrichmentError(
                f"pdftoppm failed on page {page_number} of '{pdf_path}': {run.stderr_tail()}"
            )
    finally:
        # One log per page would pile up over a long document.
        run.stderr_path.unlink(missing_ok=True)
    image = stem.with_suffix(".png")
    if not image.is_file():
        raise EnrichmentError(f"pdftoppm produced no image for page {page_number} of '{pdf_path}'")
    return image


@dataclass(frozen=True, slots=True)
class RenderedPage:
    number: int
    path: Path
    dpi: int


def rendered_pages(
    pdf_path: Path,
    *,
    budget: DecoderBudget,
    max_pages: int,
    max_pixels: int,
    dpi: int = DEFAULT_DPI,
) -> Iterator[RenderedPage]:
    """Every page in order, rendered one at a time: each image is deleted
    once the caller moves on, before the next page renders. The task's
    deadline is checked before each page, since the caller's work on a
    page (OCR) runs outside any subprocess watch."""
    info = read_pdf_info(pdf_path, budget=budget, max_pages=max_pages, page_sizes=True)
    for page in info.pages:
        budget.check_deadline(f"rendering page {page.number} of '{pdf_path}'")
        page_dpi = render_dpi(page, max_pixels=max_pixels, dpi=dpi)
        image = render_pdf_page(pdf_path, page.number, dpi=page_dpi, budget=budget)
        try:
            yield RenderedPage(number=page.number, path=image, dpi=page_dpi)
        finally:
            image.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_DPI",
    "RenderedPage",
    "render_dpi",
    "render_pdf_page",
    "rendered_pages",
    "rendered_pixels",
]
