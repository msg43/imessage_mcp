"""Thumbnails and previews: the converter runs sandboxed, writes only to
its work directory, caches results, and remembers failures."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from _search_page_fixtures import tiny_png
from imsg.search_page.media import MediaConverter, sandbox_profile

MACOS = sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def test_sandbox_profile_denies_network_and_limits_writes() -> None:
    profile = sandbox_profile(Path('/data/work "x"'))
    assert "(deny network*)" in profile and "(deny file-write*)" in profile
    assert '(allow file-write* (subpath "/data/work \\"x\\""))' in profile


@pytest.mark.skipif(not MACOS, reason="needs macOS sandbox-exec and ImageIO")
def test_image_thumbnail_is_made_once_and_cached(tmp_path: Path) -> None:
    source = tmp_path / "source-without-extension"
    source.write_bytes(tiny_png(64, 48))
    converter = MediaConverter(tmp_path / "thumbs")
    thumb = converter.image_thumbnail(source, "ab" * 32)
    assert thumb is not None and thumb.read_bytes()[:3] == b"\xff\xd8\xff"  # JPEG
    assert thumb.parent.parent == tmp_path / "thumbs"
    assert converter.image_thumbnail(source, "ab" * 32) == thumb
    assert not any((tmp_path / "thumbs" / ".work").iterdir())  # work dirs removed


@pytest.mark.skipif(not MACOS, reason="needs macOS sandbox-exec and ImageIO")
def test_unreadable_input_fails_once_and_is_not_retried(tmp_path: Path) -> None:
    source = tmp_path / "junk"
    source.write_bytes(b"not an image at all")
    converter = MediaConverter(tmp_path / "thumbs")
    assert converter.image_thumbnail(source, "cd" * 32) is None
    marker = tmp_path / "thumbs" / "cd" / f"{'cd' * 32}.thumb.jpg.failed"
    assert marker.exists()
    source.write_bytes(tiny_png())
    assert converter.image_thumbnail(source, "cd" * 32) is None  # remembered for a day
