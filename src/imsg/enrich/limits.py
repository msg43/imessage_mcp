"""Untrusted-attachment resource ceilings (SPEC §8 S5b, D6): attachment
content is attacker-supplied bytes. Every check here runs *before* any
decoder touches the file's content, and every violation is a typed
`UntrustedAttachmentError`, never a hang or a silent skip.
"""

from __future__ import annotations

from pathlib import Path

from imsg.errors import UntrustedAttachmentError
from imsg.paths import is_contained_in, resolve_path


def check_path_containment(source_path: Path, attachments_root: Path) -> Path:
    """Resolve `source_path` (symlinks + `..`) and refuse anything that
    doesn't land under `attachments_root`, isn't a regular file, or is
    itself a symlink (no traversal, no symlink escape, no special
    files — SPEC §8 S5b)."""
    if source_path.is_symlink():
        raise UntrustedAttachmentError(f"'{source_path}' is a symlink — refusing to follow it")
    resolved = resolve_path(source_path)
    if not is_contained_in(resolved, attachments_root):
        raise UntrustedAttachmentError(
            f"'{source_path}' does not resolve under the Messages attachments "
            f"root ('{attachments_root}') — refusing to read it"
        )
    if not resolved.is_file():
        raise UntrustedAttachmentError(f"'{source_path}' is not a regular file")
    return resolved


def check_file_size(path: Path, *, max_bytes: int) -> int:
    size = path.stat().st_size
    if size > max_bytes:
        raise UntrustedAttachmentError(
            f"'{path}' is {size} bytes, exceeding enrichment.limits.max_file_bytes ({max_bytes})"
        )
    return size


def check_pdf_page_count(page_count: int, *, max_pages: int) -> None:
    if page_count > max_pages:
        raise UntrustedAttachmentError(
            f"PDF has {page_count} pages, exceeding enrichment.limits.max_pdf_pages ({max_pages})"
        )


def check_media_duration(seconds: float, *, max_seconds: int) -> None:
    if seconds > max_seconds:
        raise UntrustedAttachmentError(
            f"media is {seconds:.1f}s, exceeding enrichment.limits.max_media_seconds ({max_seconds})"
        )


__all__ = [
    "check_file_size",
    "check_media_duration",
    "check_path_containment",
    "check_pdf_page_count",
]
