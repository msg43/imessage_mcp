"""Untrusted-attachment resource ceilings (SPEC §8 S5b, D6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from imsg.enrich.limits import (
    check_file_size,
    check_media_duration,
    check_path_containment,
    check_pdf_page_count,
)
from imsg.errors import UntrustedAttachmentError


def test_path_containment_accepts_a_file_under_root(tmp_path: Path) -> None:
    root = tmp_path / "Attachments"
    root.mkdir()
    f = root / "sub" / "photo.jpg"
    f.parent.mkdir()
    f.write_bytes(b"data")
    resolved = check_path_containment(f, root)
    assert resolved == f.resolve()


def test_path_containment_rejects_traversal_escape(tmp_path: Path) -> None:
    root = tmp_path / "Attachments"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"data")
    escaping = root / ".." / "outside.jpg"
    with pytest.raises(UntrustedAttachmentError):
        check_path_containment(escaping, root)


def test_path_containment_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "Attachments"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"data")
    link = root / "escape_link.jpg"
    link.symlink_to(outside)
    with pytest.raises(UntrustedAttachmentError):
        check_path_containment(link, root)


def test_path_containment_rejects_directory(tmp_path: Path) -> None:
    root = tmp_path / "Attachments"
    sub = root / "sub"
    sub.mkdir(parents=True)
    with pytest.raises(UntrustedAttachmentError):
        check_path_containment(sub, root)


def test_file_size_within_limit_ok(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 100)
    assert check_file_size(f, max_bytes=1000) == 100


def test_file_size_over_limit_raises(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 1000)
    with pytest.raises(UntrustedAttachmentError):
        check_file_size(f, max_bytes=100)


def test_pdf_page_count_within_limit_ok() -> None:
    check_pdf_page_count(10, max_pages=100)  # must not raise


def test_pdf_page_count_over_limit_raises() -> None:
    with pytest.raises(UntrustedAttachmentError):
        check_pdf_page_count(200, max_pages=100)


def test_media_duration_within_limit_ok() -> None:
    check_media_duration(60.0, max_seconds=3600)  # must not raise


def test_media_duration_over_limit_raises() -> None:
    with pytest.raises(UntrustedAttachmentError):
        check_media_duration(7200.0, max_seconds=3600)
