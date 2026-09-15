"""Deterministic-failure classification for S5a materialization (SPEC §8
S5a; migration 0003's `unsupported` state).

An attachment row belongs in `unsupported` when materialization can
never succeed no matter how often it is retried — the reason is a
property of the row, not of the moment. Two families:

- the recorded `source_path` is refused *before any read* because it
  resolves outside the Messages attachments root. That containment
  check (`imsg.backfill.pipeline`) is load-bearing and stays exactly as
  strict; this module only names *why* the path fell outside — a
  system temporary directory, the sticker cache, or something else —
  so the AT-3 report can say more than "error".
- the read itself failed with an errno that cannot change on retry:
  the path names a directory, or a path component exceeds the
  filesystem's name limit.

Everything else raised as `OSError` (I/O errors, an iCloud fetch that
timed out, a placeholder that is not there yet) stays on the
retry/backoff ladder — `error`, then `missing` after
`MAX_MATERIALIZATION_ATTEMPTS` — exactly as before.

A row with **no** `source_path` at all is a third, simpler case: there
is nothing on disk to read, so it is `missing` from the start
(`NO_SOURCE_PATH_ERROR`), never `dataless` — "retrying" would be false.

`materialization_last_error` for an `unsupported` row is written as
`unsupported[<reason>]: <detail>` (`format_unsupported_error`) so the
report can sub-count reasons by parsing that prefix
(`parse_unsupported_reason`) instead of pattern-matching free text.
"""

from __future__ import annotations

import errno
import re
from enum import StrEnum
from pathlib import Path

NO_SOURCE_PATH_ERROR = (
    "no source path recorded in chat.db — there is no local placeholder to read"
)
"""`materialization_last_error` for a row whose `source_path` is NULL.
Set by S2 at insert time and by the backfill pass when it reclassifies
a pre-existing NULL-path row out of `dataless`."""


class UnsupportedReason(StrEnum):
    """Why a row can never be materialized. The value is the token that
    appears inside `unsupported[...]` in `materialization_last_error`."""

    TEMP_DIRECTORY_PATH = "temp-directory-path"
    STICKER_CACHE_PATH = "sticker-cache-path"
    OUT_OF_ROOT_PATH = "out-of-root-path"
    IS_A_DIRECTORY = "is-a-directory"
    FILE_NAME_TOO_LONG = "file-name-too-long"

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_DESCRIPTIONS: dict[UnsupportedReason, str] = {
    UnsupportedReason.TEMP_DIRECTORY_PATH: (
        "source path points into a system temporary directory, not the Messages "
        "attachments root; refused without reading"
    ),
    UnsupportedReason.STICKER_CACHE_PATH: (
        "source path points into the Messages sticker cache, not the attachments "
        "root; refused without reading"
    ),
    UnsupportedReason.OUT_OF_ROOT_PATH: (
        "source path resolves outside the Messages attachments root; refused without reading"
    ),
    UnsupportedReason.IS_A_DIRECTORY: "source path names a directory, not a file",
    UnsupportedReason.FILE_NAME_TOO_LONG: (
        "a component of the source path exceeds the filesystem's name limit"
    ),
}

_TEMP_DIRECTORY_ROOTS: tuple[Path, ...] = (
    Path("/private/var/folders"),  # macOS per-user temp/cache tree (`$TMPDIR` lives here)
    Path("/var/folders"),  # the same tree before `/var -> /private/var` is resolved
    Path("/private/tmp"),
    Path("/tmp"),  # classification only; nothing is ever written here
)
_STICKER_CACHE_MARKER: tuple[str, ...] = ("Library", "Messages", "StickerCache")

_UNSUPPORTED_PREFIX = re.compile(r"^unsupported\[([a-z-]+)\]")


def classify_out_of_root(resolved_source: Path) -> UnsupportedReason:
    """Name why an already-resolved path that FAILED the containment
    check fell outside the attachments root. Precedence is most-specific
    first: the sticker cache (which sits next to the attachments root
    under the Messages directory) before the temp-directory tree, then
    everything else. Callers must have established non-containment
    already — this function never re-checks it and never touches the
    filesystem."""
    parts = resolved_source.parts
    marker_len = len(_STICKER_CACHE_MARKER)
    for i in range(len(parts) - marker_len + 1):
        if parts[i : i + marker_len] == _STICKER_CACHE_MARKER:
            return UnsupportedReason.STICKER_CACHE_PATH
    if any(resolved_source.is_relative_to(root) for root in _TEMP_DIRECTORY_ROOTS):
        return UnsupportedReason.TEMP_DIRECTORY_PATH
    return UnsupportedReason.OUT_OF_ROOT_PATH


def classify_os_error(exc: OSError) -> UnsupportedReason | None:
    """The deterministic errno classes. `None` means "treat as transient":
    back off and retry, which is the only safe default for an errno this
    function has not explicitly reasoned about."""
    if exc.errno == errno.EISDIR:
        return UnsupportedReason.IS_A_DIRECTORY
    if exc.errno == errno.ENAMETOOLONG:
        return UnsupportedReason.FILE_NAME_TOO_LONG
    return None


def format_unsupported_error(reason: UnsupportedReason, detail: str) -> str:
    return f"unsupported[{reason.value}]: {detail}"


def parse_unsupported_reason(last_error: str | None) -> UnsupportedReason | None:
    """Inverse of `format_unsupported_error`. `None` for anything that
    does not carry the prefix (including a NULL) — an `unsupported` row
    written by something other than this module is reported as
    "unspecified" rather than guessed at."""
    if not last_error:
        return None
    match = _UNSUPPORTED_PREFIX.match(last_error)
    if match is None:
        return None
    try:
        return UnsupportedReason(match.group(1))
    except ValueError:
        return None


__all__ = [
    "NO_SOURCE_PATH_ERROR",
    "UnsupportedReason",
    "classify_os_error",
    "classify_out_of_root",
    "format_unsupported_error",
    "parse_unsupported_reason",
]
