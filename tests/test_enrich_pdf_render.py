"""Real `pdftoppm` page rasterization for the scanned-OCR path
(SPEC §8 S5b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from _pdf_fixtures import write_minimal_pdf
from imsg.enrich.pdf_render import render_pdf_pages_to_png
from imsg.errors import EnrichmentError


def test_renders_one_png_per_page(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    write_minimal_pdf(pdf, ["Page One", "Page Two", "Page Three"])
    out_dir = tmp_path / "rendered"
    pages = render_pdf_pages_to_png(pdf, out_dir, dpi=72, timeout_seconds=15)
    assert len(pages) == 3
    for p in pages:
        assert p.is_file()
        assert p.suffix == ".png"
        assert p.stat().st_size > 0


def test_pages_are_returned_in_order(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    write_minimal_pdf(pdf, ["A", "B", "C", "D"])
    pages = render_pdf_pages_to_png(pdf, tmp_path / "out", dpi=72, timeout_seconds=15)
    names = [p.name for p in pages]
    assert names == sorted(names)


def test_nonexistent_pdf_raises(tmp_path: Path) -> None:
    with pytest.raises(EnrichmentError):
        render_pdf_pages_to_png(tmp_path / "missing.pdf", tmp_path / "out", timeout_seconds=10)
