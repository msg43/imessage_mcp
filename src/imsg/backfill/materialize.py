"""Materialize one attachment (SPEC §8 S5a): reading the file in full is
what triggers APFS/iCloud to download a dataless placeholder's real
content; once read, copy it content-addressed into
`$DATA_ROOT/attachments/sha256[0:2]/sha256` (SPEC §5.3 layout).

A copy found somewhere else (D13: another Mac, a drive, a NAS share) is
verified before it reaches the cache: `expected_size` and
`expected_sha256` are what that location's own listing said about the
file, and a copy that differs is refused before it is moved into the
cache (`ContentMismatchError`), so the cache never holds bytes no
attachment row points at.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from imsg.hashing import sha256_file
from imsg.paths import is_contained_in, resolve_path

_READ_CHUNK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MaterializeResult:
    sha256: str
    byte_size: int
    cache_path: Path


class ContentMismatchError(Exception):
    """A copy's size or content hash is not what its location said. Not an
    `OSError`: the read succeeded, and the copy is wrong, which retrying
    cannot fix."""

    def __init__(self, what: str, expected: str | int, actual: str | int) -> None:
        super().__init__(f"{what} mismatch: expected {expected}, got {actual}")
        self.what = what
        self.expected = expected
        self.actual = actual


def cache_path_for(data_root: Path, sha256: str) -> Path:
    return data_root / "attachments" / sha256[:2] / sha256


def materialize_attachment(
    source_path: Path,
    data_root: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> MaterializeResult:
    """Read `source_path` in full and copy it into the content-addressed
    cache under `data_root`. Raises `OSError` on any read/write failure
    — the caller (`imsg.backfill.pipeline`) translates that into the
    attachment state machine's retry/backoff/`missing` handling; this
    function has no opinion about retries.

    With `expected_size` / `expected_sha256`, the copy is checked after it
    is read and before it enters the cache; a mismatch raises
    `ContentMismatchError` and leaves the cache untouched.

    Deduplicates by content: if a file with the same sha256 is already
    cached, the freshly-read bytes are discarded rather than
    overwritten (the cache is content-addressed, so this is a no-op
    either way, but avoids a redundant write for a large file).
    """
    resolved_source = resolve_path(source_path)
    tmp_root = data_root / "attachments" / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    # The source's own name is kept (truncated) only as a debugging aid
    # for a stray partial; it must never make the temp path itself
    # over-long, or the cleanup in `finally` raises ENAMETOOLONG and
    # masks the real error from the read that failed first.
    tmp_path = tmp_root / f"{resolved_source.name[:64]}.{id(resolved_source)}.partial"

    byte_size = 0
    try:
        with resolved_source.open("rb") as src, tmp_path.open("wb") as dst:
            while chunk := src.read(_READ_CHUNK):
                dst.write(chunk)
                byte_size += len(chunk)

        if expected_size is not None and byte_size != expected_size:
            raise ContentMismatchError("size", expected_size, byte_size)
        digest = sha256_file(tmp_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ContentMismatchError("sha256", expected_sha256, digest)
        final_path = cache_path_for(data_root, digest)
        if not is_contained_in(final_path, data_root):  # pragma: no cover - defensive
            raise OSError(f"materialized cache path escaped data_root: {final_path}")

        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            tmp_path.unlink(missing_ok=True)  # identical content already cached
        else:
            shutil.move(str(tmp_path), str(final_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return MaterializeResult(sha256=digest, byte_size=byte_size, cache_path=final_path)


def materialize_from_cache(
    data_root: Path, sha256: str, *, expected_size: int | None = None
) -> MaterializeResult | None:
    """The copy's content is already in the cache when a location's listing
    gave its hash and a cached file has that hash: nothing needs to be
    fetched. The cached file is re-hashed, so a damaged cache file is not
    trusted. `None` when there is no such cached file, or it does not
    verify."""
    path = cache_path_for(data_root, sha256)
    try:
        st = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    if expected_size is not None and st.st_size != expected_size:
        return None
    try:
        if sha256_file(path) != sha256:
            return None
    except OSError:
        return None
    return MaterializeResult(sha256=sha256, byte_size=st.st_size, cache_path=path)


__all__ = [
    "ContentMismatchError",
    "MaterializeResult",
    "cache_path_for",
    "materialize_attachment",
    "materialize_from_cache",
]
