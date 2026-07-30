"""Real content-based MIME sniffing via `file` (SPEC §8 S5b, D6: "the
extension is not trusted")."""

from __future__ import annotations

from pathlib import Path

import pytest

from _pdf_fixtures import write_minimal_pdf
from imsg.enrich.mime import real_sniff_mime
from imsg.errors import UntrustedAttachmentError


def test_sniffs_pdf_by_content_even_with_wrong_extension(tmp_path: Path) -> None:
    misnamed = tmp_path / "totally_an_image.jpg"
    write_minimal_pdf(misnamed, ["Hello"])
    assert real_sniff_mime(misnamed) == "application/pdf"


def test_sniffs_plain_text(tmp_path: Path) -> None:
    f = tmp_path / "notes.dat"
    f.write_text("just some plain text content here\n")
    assert real_sniff_mime(f) == "text/plain"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UntrustedAttachmentError):
        real_sniff_mime(tmp_path / "does-not-exist")
