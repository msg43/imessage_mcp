"""A PDF's page count and page sizes, read with poppler's `pdfinfo` before
anything extracts or renders it (SPEC §8 S5b, D6).

`pdftotext` and `pdftoppm` used to run first and the page count was
checked on their output, so a 5,000-page PDF was rendered in full before
it was refused. `pdfinfo` answers from the document's page tree without
drawing anything, so the ceiling is checked first.

**The media box, not "Page size".** `pdfinfo`'s "Page size" line is the
crop box, while `pdftoppm` renders the media box unless told otherwise.
Checked 2026-09-24 on poppler 26.08: a page with a 2000x2000 pt media box
and a 100x100 pt crop box reports "Page size: 100 x 100 pts" and renders
2000x2000 at 72 dpi. The render budget (`imsg.enrich.pdf_render`) is
therefore computed from each page's media box, which `-box` prints.
Poppler ignores `/UserUnit` in both tools alike (a page scaled by 4
printed and rendered at its unscaled size), so the two agree.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from imsg.enrich.limits import check_pdf_page_count
from imsg.enrich.sandboxed_decoder import DecoderBudget, run_decoder
from imsg.errors import EnrichmentError, UntrustedAttachmentError

MAX_PDFINFO_OUTPUT_BYTES = 4 * 1024 * 1024
"""`pdfinfo` prints the document's metadata too, which the sender
controls; a thousand pages of boxes come to about 400 KB."""

_PAGES_RE = re.compile(r"^Pages:\s+(\d+)\s*$", re.MULTILINE)
_MEDIA_BOX_RE = re.compile(
    r"^Page\s+(\d+)\s+MediaBox:\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class PdfPage:
    """One page's media box size, in PDF points (1/72 inch)."""

    number: int
    width_pt: float
    height_pt: float


@dataclass(frozen=True, slots=True)
class PdfInfo:
    page_count: int
    pages: tuple[PdfPage, ...]
    """Every page's size, in order, when they were asked for; else empty."""


def read_pdf_info(
    pdf_path: Path, *, budget: DecoderBudget, max_pages: int, page_sizes: bool = False
) -> PdfInfo:
    """The page count, refused over `max_pages`
    (`enrichment.limits.max_pdf_pages`) before anything else runs; with
    `page_sizes`, every page's media box too. `pdfinfo` runs sandboxed and
    inside `budget` (`imsg.enrich.sandboxed_decoder`)."""
    argv = ["pdfinfo"]
    if page_sizes:
        argv += ["-box", "-f", "1", "-l", str(max_pages)]
    argv.append(os.fspath(pdf_path.absolute()))
    run = run_decoder(
        argv,
        budget=budget,
        name="pdfinfo",
        subject=pdf_path,
        stdout_name="pdfinfo.txt",
        max_stdout_bytes=MAX_PDFINFO_OUTPUT_BYTES,
    )
    if run.returncode != 0:
        raise EnrichmentError(f"pdfinfo failed on '{pdf_path}': {run.stderr_tail()}")
    assert run.stdout_path is not None
    text = run.stdout_path.read_bytes().decode("utf-8", errors="replace")
    match = _PAGES_RE.search(text)
    if match is None:
        raise EnrichmentError(f"pdfinfo reported no page count for '{pdf_path}'")
    page_count = int(match.group(1))
    check_pdf_page_count(page_count, max_pages=max_pages)
    if not page_sizes:
        return PdfInfo(page_count=page_count, pages=())
    boxes: dict[int, PdfPage] = {}
    for box in _MEDIA_BOX_RE.finditer(text):
        number = int(box.group(1))
        x1, y1, x2, y2 = (float(box.group(i)) for i in range(2, 6))
        boxes[number] = PdfPage(number=number, width_pt=abs(x2 - x1), height_pt=abs(y2 - y1))
    missing = [n for n in range(1, page_count + 1) if n not in boxes]
    if missing:
        raise UntrustedAttachmentError(
            f"pdfinfo reported no media box for page {missing[0]} of '{pdf_path}'; "
            f"refusing to render a page of unknown size"
        )
    return PdfInfo(page_count=page_count, pages=tuple(boxes[n] for n in range(1, page_count + 1)))


__all__ = ["MAX_PDFINFO_OUTPUT_BYTES", "PdfInfo", "PdfPage", "read_pdf_info"]
