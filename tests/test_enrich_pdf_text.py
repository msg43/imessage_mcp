"""Real `pdftotext` extraction + scanned-PDF detection (SPEC §8 S5b).
Uses hand-built minimal PDFs (`tests/_pdf_fixtures.py`) — real poppler,
no mocking, because `pdftotext` is available wherever this build runs.
Every call runs sandboxed inside a task budget
(`imsg.enrich.sandboxed_decoder`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from _pdf_fixtures import write_minimal_pdf
from imsg.enrich.pdf_text import extract_pdf_text, is_scanned
from imsg.enrich.sandboxed_decoder import DecoderBudget
from imsg.errors import EnrichmentError


def _budget(tmp_path: Path) -> DecoderBudget:
    return DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)


def test_extracts_single_page_text(tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    write_minimal_pdf(pdf, ["Hello World"])
    result = extract_pdf_text(pdf, budget=_budget(tmp_path), max_pages=1000)
    assert result.page_count == 1
    assert "Hello World" in result.full_text


def test_extracts_multi_page_text_in_order(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    write_minimal_pdf(pdf, ["Page One Text", "Page Two Text", "Page Three Text"])
    result = extract_pdf_text(pdf, budget=_budget(tmp_path), max_pages=1000)
    assert result.page_count == 3
    assert "Page One Text" in result.pages[0]
    assert "Page Two Text" in result.pages[1]
    assert "Page Three Text" in result.pages[2]
    idx1 = result.full_text.index("Page One")
    idx2 = result.full_text.index("Page Two")
    idx3 = result.full_text.index("Page Three")
    assert idx1 < idx2 < idx3


def test_nonexistent_file_raises_enrichment_error(tmp_path: Path) -> None:
    with pytest.raises(EnrichmentError):
        extract_pdf_text(tmp_path / "does-not-exist.pdf", budget=_budget(tmp_path), max_pages=1000)


def test_corrupt_pdf_raises_enrichment_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a real pdf at all")
    with pytest.raises(EnrichmentError):
        extract_pdf_text(bad, budget=_budget(tmp_path), max_pages=1000)


def test_is_scanned_true_for_sparse_text_layer(tmp_path: Path) -> None:
    pdf = tmp_path / "sparse.pdf"
    write_minimal_pdf(pdf, ["x"])  # one character on the page
    result = extract_pdf_text(pdf, budget=_budget(tmp_path), max_pages=1000)
    assert is_scanned(result, threshold_chars_per_page=50) is True


def test_is_scanned_false_for_dense_text_layer(tmp_path: Path) -> None:
    pdf = tmp_path / "dense.pdf"
    write_minimal_pdf(pdf, ["Hello World this line has plenty of visible characters on it"])
    result = extract_pdf_text(pdf, budget=_budget(tmp_path), max_pages=1000)
    assert is_scanned(result, threshold_chars_per_page=10) is False
