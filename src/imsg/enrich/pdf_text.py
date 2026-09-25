"""Real `pdftotext` (poppler) text-layer extraction, plus the
scanned-PDF detection that decides whether OCR also needs to run
(SPEC §8 S5b: "< pdf_scanned_threshold_chars_per_page avg -> also
enqueue ocr"). Not a model — a deterministic subprocess tool,
implemented for real (see `imsg.enrich.provider`'s docstring for the
line between "model" and "tool" this build draws).

**Bounded (2026-09-24, QA review).** The text used to be read whole into
memory with `capture_output=True`, and only then was the page count
checked, so a PDF whose content streams inflate to gigabytes of text
reached memory before any ceiling. Now the page count is read first
(`imsg.enrich.pdf_info`), `pdftotext` is told to stop at
`enrichment.limits.max_pdf_pages`, and its output streams into a file in
the task's work directory, stopped once it passes `MAX_PDF_TEXT_BYTES`
(the same ceiling `textutil` has). It runs sandboxed like every decoder
(`imsg.enrich.sandboxed_decoder`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from imsg.enrich.pdf_info import read_pdf_info
from imsg.enrich.sandboxed_decoder import MAX_DECODER_OUTPUT_BYTES, DecoderBudget, run_decoder
from imsg.errors import EnrichmentError

_FORM_FEED = "\x0c"  # pdftotext's default page separator (no -nopgbrk)

MAX_PDF_TEXT_BYTES = MAX_DECODER_OUTPUT_BYTES
"""`pdftotext` is stopped once its text passes this: a typed permanent
failure, not a truncation. Read at call time, so a test can lower it."""


@dataclass(frozen=True, slots=True)
class PdfTextResult:
    pages: tuple[str, ...]  # one entry per page, in order

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p for p in self.pages if p)

    @property
    def avg_chars_per_page(self) -> float:
        if not self.pages:
            return 0.0
        return sum(len(p) for p in self.pages) / len(self.pages)


def extract_pdf_text(pdf_path: Path, *, budget: DecoderBudget, max_pages: int) -> PdfTextResult:
    """The text layer, one entry per page. Refuses a PDF over `max_pages`
    before extracting anything, and a text layer over `MAX_PDF_TEXT_BYTES`
    while extracting it (`UntrustedAttachmentError`, permanent); a
    `pdftotext` that fails is an `EnrichmentError`, retried."""
    read_pdf_info(pdf_path, budget=budget, max_pages=max_pages)
    run = run_decoder(
        ["pdftotext", "-layout", "-l", str(max_pages), os.fspath(pdf_path.absolute()), "-"],
        budget=budget,
        name="pdftotext",
        subject=pdf_path,
        stdout_name="pdftotext.txt",
        max_stdout_bytes=MAX_PDF_TEXT_BYTES,
    )
    if run.returncode != 0:
        raise EnrichmentError(f"pdftotext failed on '{pdf_path}': {run.stderr_tail()}")
    assert run.stdout_path is not None
    text = run.stdout_path.read_bytes().decode("utf-8", errors="replace")
    pages = text.split(_FORM_FEED)
    if pages and pages[-1] == "":  # trailing form-feed after the last page
        pages = pages[:-1]
    return PdfTextResult(pages=tuple(pages))


def is_scanned(result: PdfTextResult, *, threshold_chars_per_page: int) -> bool:
    """True if the text layer is sparse enough to be a scan with little
    or no real text (SPEC §8 S5b's `pdf_scanned_threshold_chars_per_page`
    check). A zero-page result (extraction produced nothing at all)
    counts as scanned — there is no text layer to speak of."""
    if result.page_count == 0:
        return True
    return result.avg_chars_per_page < threshold_chars_per_page


__all__ = ["MAX_PDF_TEXT_BYTES", "PdfTextResult", "extract_pdf_text", "is_scanned"]
