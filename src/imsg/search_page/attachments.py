"""Serving attachment files from the content-addressed cache.

An attachment is looked up by its opaque `attachment_key`. The file path
is never taken from the database or the request: it is rebuilt from the
row's `sha256` (`<data_root>/attachments/<sha[:2]>/<sha>`, the layout
`imsg.backfill.materialize.cache_path_for` writes), the sha must be 64
hex characters, and the resolved path (symlinks followed) must still be
a regular file beneath `<data_root>/attachments`. Anything else is 404.

Content types come from an allowlist. Browser-renderable media, PDFs and
plain text are served inline; everything else is a download
(`Content-Disposition: attachment`, `application/octet-stream`), so an
HTML or SVG attachment can never run script in the page's origin. Every
response also carries `X-Content-Type-Options: nosniff` and, except for
PDFs (whose viewer a sandbox would break), a `sandbox` Content Security
Policy.
"""

from __future__ import annotations

import mimetypes
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from imsg.paths import is_contained_in, resolve_path
from imsg.search_page.threads import attachment_kind, is_opaque_key

if TYPE_CHECKING:
    import psycopg

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

INLINE_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/avif",
        "image/heic",
        "image/heif",
        "image/tiff",
        "video/mp4",
        "video/quicktime",
        "video/x-m4v",
        "video/webm",
        "audio/mpeg",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/x-caf",
        "audio/amr",
        "application/pdf",
        "text/plain",
    }
)
"""Types a browser renders itself, with no script of the page's origin."""

NEEDS_IMAGE_PREVIEW: frozenset[str] = frozenset({"image/heic", "image/heif", "image/tiff"})
"""Most browsers cannot draw these; the page shows a converted JPEG."""
NEEDS_AUDIO_PREVIEW: frozenset[str] = frozenset({"audio/x-caf", "audio/amr", "audio/3gpp"})
"""iMessage voice notes: converted to AAC so every browser can play them."""

_EXTRA_TYPES = {
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".caf": "audio/x-caf",
    ".amr": "audio/amr",
    ".m4a": "audio/x-m4a",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".vcf": "text/vcard",
}

PDF_HEADERS = {"Content-Security-Policy": "frame-ancestors 'self'"}
SANDBOX_HEADERS = {
    "Content-Security-Policy": (
        "sandbox; default-src 'none'; img-src 'self'; media-src 'self'; "
        "style-src 'unsafe-inline'; frame-ancestors 'self'"
    )
}


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    attachment_id: int
    attachment_key: str
    filename: str | None
    mime_type: str | None
    byte_size: int | None
    sha256: str | None
    state: str

    @property
    def kind(self) -> str:
        return attachment_kind(self.mime_type, self.filename)


def lookup_attachment(pg: psycopg.Connection, attachment_key: str) -> AttachmentRecord | None:
    if not is_opaque_key(attachment_key):
        return None
    with pg.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, attachment_key, filename, mime_type, byte_size, sha256, "
            "state::text FROM attachment WHERE attachment_key = %s",
            (attachment_key,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return AttachmentRecord(
        attachment_id=int(row[0]),
        attachment_key=str(row[1]),
        filename=row[2],
        mime_type=row[3],
        byte_size=int(row[4]) if row[4] is not None else None,
        sha256=row[5],
        state=str(row[6]),
    )


def attachments_root(data_root: Path) -> Path:
    return resolve_path(data_root / "attachments")


def cache_file(data_root: Path, sha256: str | None) -> Path | None:
    """The cached file for `sha256`, or `None` if the sha is malformed or
    the file is missing, not regular, or resolves outside the cache."""
    if not sha256 or not SHA256_RE.match(sha256):
        return None
    root = attachments_root(data_root)
    candidate = resolve_path(root / sha256[:2] / sha256)
    if not is_contained_in(candidate, root) or candidate == root:
        return None
    try:
        info = os.stat(candidate)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return candidate


def content_type(record: AttachmentRecord) -> str:
    """A normalized content type: the stored one when it looks sane, else a
    guess from the file name, else `application/octet-stream`."""
    stored = (record.mime_type or "").split(";", 1)[0].strip().lower()
    if stored and stored != "application/octet-stream" and re.match(r"^[a-z0-9.+\-]+/[a-z0-9.+\-]+$", stored):
        return stored
    name = (record.filename or "").lower()
    ext = os.path.splitext(name)[1]
    if ext in _EXTRA_TYPES:
        return _EXTRA_TYPES[ext]
    guessed = mimetypes.guess_type(name)[0] if name else None
    return guessed or "application/octet-stream"


def safe_filename(record: AttachmentRecord) -> str:
    raw = os.path.basename(record.filename or "") or f"attachment-{record.attachment_key[:12]}"
    cleaned = "".join(ch for ch in unicodedata.normalize("NFC", raw) if ch.isprintable())
    cleaned = cleaned.replace('"', "'").replace("\\", "_").replace("/", "_").strip()
    return cleaned[:180] or f"attachment-{record.attachment_key[:12]}"


def content_disposition(disposition: str, filename: str) -> str:
    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace("?", "_")
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@dataclass(frozen=True, slots=True)
class ServePlan:
    media_type: str
    headers: dict[str, str]
    inline: bool


def serve_plan(record: AttachmentRecord, *, download: bool) -> ServePlan:
    media = content_type(record)
    inline = media in INLINE_TYPES and not download
    served_type = media if media in INLINE_TYPES else "application/octet-stream"
    if served_type == "text/plain":
        served_type = "text/plain; charset=utf-8"
    headers = {
        "Content-Disposition": content_disposition(
            "inline" if inline else "attachment", safe_filename(record)
        ),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-store",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    headers.update(PDF_HEADERS if media == "application/pdf" and inline else SANDBOX_HEADERS)
    return ServePlan(media_type=served_type, headers=headers, inline=inline)


def needs_image_preview(record: AttachmentRecord) -> bool:
    return content_type(record) in NEEDS_IMAGE_PREVIEW


def needs_audio_preview(record: AttachmentRecord) -> bool:
    return content_type(record) in NEEDS_AUDIO_PREVIEW


__all__ = [
    "INLINE_TYPES",
    "NEEDS_AUDIO_PREVIEW",
    "NEEDS_IMAGE_PREVIEW",
    "AttachmentRecord",
    "ServePlan",
    "attachments_root",
    "cache_file",
    "content_disposition",
    "content_type",
    "lookup_attachment",
    "needs_audio_preview",
    "needs_image_preview",
    "safe_filename",
    "serve_plan",
]
