"""Attachment files come only from the content-addressed cache, with an
allowlisted content type; anything a browser could run is a download."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from imsg.search_page.attachments import (
    AttachmentRecord,
    cache_file,
    content_disposition,
    content_type,
    safe_filename,
    serve_plan,
)


def _record(filename: str | None, mime: str | None, sha: str | None = None) -> AttachmentRecord:
    return AttachmentRecord(
        attachment_id=1,
        attachment_key="k" * 64,
        filename=filename,
        mime_type=mime,
        byte_size=10,
        sha256=sha,
        state="materialized",
    )


def _cached(root: Path, content: bytes) -> str:
    sha = hashlib.sha256(content).hexdigest()
    path = root / "attachments" / sha[:2] / sha
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha


def test_cache_file_resolves_only_inside_the_cache(tmp_path: Path) -> None:
    root = tmp_path / "root"
    sha = _cached(root, b"hello")
    assert cache_file(root, sha) == (root / "attachments" / sha[:2] / sha).resolve()
    for bad in (None, "", "../../etc/passwd", "A" * 64, "0" * 63, sha + "/", "../" + sha[3:]):
        assert cache_file(root, bad) is None
    assert cache_file(root, "cd" * 32) is None  # missing file


def test_symlinks_and_directories_in_the_cache_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    sha = "ab" * 32
    target = root / "attachments" / "ab" / sha
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)
    assert cache_file(root, sha) is None
    sha2 = "cd" * 32
    (root / "attachments" / "cd" / sha2).mkdir(parents=True)
    assert cache_file(root, sha2) is None


@pytest.mark.parametrize(
    ("filename", "mime", "served", "inline", "sandboxed"),
    [
        ("a.jpg", "image/jpeg", "image/jpeg", True, True),
        ("IMG_1.HEIC", None, "image/heic", True, True),
        ("clip.mov", "video/quicktime", "video/quicktime", True, True),
        ("note.caf", None, "audio/x-caf", True, True),
        ("bid.pdf", "application/pdf", "application/pdf", True, False),
        ("readme.txt", "text/plain", "text/plain; charset=utf-8", True, True),
        ("page.html", "text/html", "application/octet-stream", False, True),
        ("logo.svg", "image/svg+xml", "application/octet-stream", False, True),
        ("run.js", "application/javascript", "application/octet-stream", False, True),
        ("mystery", None, "application/octet-stream", False, True),
        ("weird", "not a type", "application/octet-stream", False, True),
    ],
)
def test_serve_plan(filename: str, mime: str | None, served: str, inline: bool, sandboxed: bool) -> None:
    plan = serve_plan(_record(filename, mime), download=False)
    assert plan.media_type == served
    assert plan.inline is inline
    assert plan.headers["Content-Disposition"].startswith("inline" if inline else "attachment")
    assert plan.headers["X-Content-Type-Options"] == "nosniff"
    assert ("sandbox" in plan.headers["Content-Security-Policy"]) is sandboxed


def test_download_forces_attachment() -> None:
    plan = serve_plan(_record("a.jpg", "image/jpeg"), download=True)
    assert plan.headers["Content-Disposition"].startswith("attachment")


def test_filenames_cannot_break_the_header() -> None:
    record = _record('bad"\r\nSet-Cookie: x=1/../name.pdf', "application/pdf")
    name = safe_filename(record)
    assert "\r" not in name and "\n" not in name and '"' not in name and "/" not in name
    header = content_disposition("inline", name)
    assert "\r" not in header and "\n" not in header
    assert content_type(_record("x.PDF", None)) == "application/pdf"
