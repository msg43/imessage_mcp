"""Small shared sha256 helpers.

Every stage that needs a content hash (S1's snapshot integrity check,
S2's dedupe/provenance hashing, `imsg.keys`'s opaque-key derivation, the
migration runner) should use these rather than re-deriving
``hashlib.sha256(...).hexdigest()`` inline, so the encoding/chunking
choices stay consistent across the codebase.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hex sha256 digest of a file's contents, read in fixed-size chunks
    so this is safe to call on multi-GB files (a `chat.db` snapshot can
    be several GB) without loading the whole thing into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Hex sha256 digest of a string, encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["sha256_file", "sha256_text"]
