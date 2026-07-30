"""Real `pdftotext` (poppler) text-layer extraction, plus the
scanned-PDF detection that decides whether OCR also needs to run
(SPEC §8 S5b: "< pdf_scanned_threshold_chars_per_page avg -> also
enqueue ocr"). Not a model — a deterministic subprocess tool,
implemented for real (see `imsg.enrich.provider`'s docstring for the
line between "model" and "tool" this build draws).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from imsg.errors import EnrichmentError

_FORM_FEED = "\x0c"  # pdftotext's default page separator (no -nopgbrk)


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


def extract_pdf_text(pdf_path: Path, *, timeout_seconds: int) -> PdfTextResult:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnrichmentError(
            f"pdftotext timed out after {timeout_seconds}s on '{pdf_path}'"
        ) from exc
    except OSError as exc:
        raise EnrichmentError(f"pdftotext could not run: {exc}") from exc
    if proc.returncode != 0:
        raise EnrichmentError(f"pdftotext failed on '{pdf_path}': {proc.stderr.strip()}")

    pages = proc.stdout.split(_FORM_FEED)
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


__all__ = ["PdfTextResult", "extract_pdf_text", "is_scanned"]
