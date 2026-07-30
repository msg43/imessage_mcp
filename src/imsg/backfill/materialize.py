"""Materialize one attachment (SPEC §8 S5a): reading the file in full is
what triggers APFS/iCloud to download a dataless placeholder's real
content; once read, copy it content-addressed into
`$DATA_ROOT/attachments/sha256[0:2]/sha256` (SPEC §5.3 layout).
"""

from __future__ import annotations

import shutil
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


def cache_path_for(data_root: Path, sha256: str) -> Path:
    return data_root / "attachments" / sha256[:2] / sha256


def materialize_attachment(source_path: Path, data_root: Path) -> MaterializeResult:
    """Read `source_path` in full and copy it into the content-addressed
    cache under `data_root`. Raises `OSError` on any read/write failure
    — the caller (`imsg.backfill.pipeline`) translates that into the
    attachment state machine's retry/backoff/`missing` handling; this
    function has no opinion about retries.

    Deduplicates by content: if a file with the same sha256 is already
    cached, the freshly-read bytes are discarded rather than
    overwritten (the cache is content-addressed, so this is a no-op
    either way, but avoids a redundant write for a large file).
    """
    resolved_source = resolve_path(source_path)
    tmp_root = data_root / "attachments" / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_root / f"{resolved_source.name}.{id(resolved_source)}.partial"

    byte_size = 0
    try:
        with resolved_source.open("rb") as src, tmp_path.open("wb") as dst:
            while chunk := src.read(_READ_CHUNK):
                dst.write(chunk)
                byte_size += len(chunk)

        digest = sha256_file(tmp_path)
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


__all__ = ["MaterializeResult", "cache_path_for", "materialize_attachment"]
