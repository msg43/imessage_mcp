"""MIME sniffing via the `file` command's content-based magic detection
(SPEC §8 S5b, D6: "the extension is not trusted").

macOS ships `file` (and its bundled magic database) at `/usr/bin/file`;
shelling out to it gets the same content-based sniffing SPEC §4's
`python-magic / libmagic` dependency line calls for, without adding a
compiled libmagic *binding* dependency — not available via Homebrew in
this build/test sandbox. Same outcome (content sniffed, not the
filename extension trusted), subprocess boundary instead of a C
extension; worth revisiting at Phase 3/5 if a bundled libmagic wheel
turns out to be the simpler real-deployment path.

One refinement on top of `file`: the `file` 5.41 that macOS 26 ships has
no magic for Apple's Core Audio Format, the container Messages records
voice messages in, and reports it as `application/octet-stream`. A CAF
file's first six bytes are fixed ('caff', then file version 1), so
`refine_sniffed_mime` reads them and reports `audio/x-caf`. Without it a
voice message routes nowhere and is never transcribed (183 of them on
the production index, 2026-09-23).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from imsg.errors import UntrustedAttachmentError

MimeSnifferFn = Callable[[Path], str]

GENERIC_BINARY_MIME = "application/octet-stream"
CAF_MIME = "audio/x-caf"
_CAF_SIGNATURE = b"caff\x00\x01"  # 'caff', then mFileVersion 1 (big-endian UInt16)

DEFAULT_SNIFF_BATCH_SIZE = 256
"""Paths per `file` process in `sniff_mime_batch`. On the production host
one `file` per 400 paths sniffed all 100,046 materialized attachments in
about 41 s (2026-09-23); one process per path costs a few milliseconds
each in process start-up alone."""

_SINGLE_TIMEOUT_SECONDS = 10
_BATCH_TIMEOUT_SECONDS = 120

# `type/subtype` — deliberately simple; only used to reject `file`'s prose
# error messages ("cannot open `...' (No such file or directory)"), which
# `file` prints to *stdout* with a **0** exit code (verified empirically —
# do not trust returncode/stderr alone to catch this).
_MIME_TYPE_RE = re.compile(r"^[\w.+-]+/[\w.+-]+$")


def refine_sniffed_mime(path: Path, mime_type: str) -> str:
    """Correct the one content type `file` 5.41 is known to miss here: a
    Core Audio Format file it calls `application/octet-stream`. Decided
    from the file's own leading bytes, never its name. Any other type is
    returned unchanged; so is octet-stream when the file cannot be read
    again (it was readable a moment ago, when `file` sniffed it)."""
    if mime_type != GENERIC_BINARY_MIME:
        return mime_type
    try:
        with path.open("rb") as fh:
            head = fh.read(len(_CAF_SIGNATURE))
    except OSError:
        return mime_type
    return CAF_MIME if head == _CAF_SIGNATURE else mime_type


def real_sniff_mime(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["file", "--brief", "--mime-type", "--", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=_SINGLE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UntrustedAttachmentError(f"MIME sniffing failed for '{path}': {exc}") from exc
    output = proc.stdout.strip()
    if proc.returncode != 0 or not _MIME_TYPE_RE.match(output):
        raise UntrustedAttachmentError(
            f"MIME sniffing failed for '{path}': 'file' exited {proc.returncode} "
            f"with unparseable output {output!r} (stderr: {proc.stderr.strip()!r})"
        )
    return refine_sniffed_mime(path, output)


@dataclass(frozen=True, slots=True)
class BatchSniffResult:
    mime_by_path: dict[Path, str] = field(default_factory=dict)
    """Every path `file` could sniff, with its (refined) MIME type."""
    errors: dict[Path, str] = field(default_factory=dict)
    """Every path it could not, with what `file` said instead."""


def _sniff_one_into(path: Path, result: BatchSniffResult) -> None:
    try:
        result.mime_by_path[path] = real_sniff_mime(path)
    except UntrustedAttachmentError as exc:
        result.errors[path] = str(exc)


def sniff_mime_batch(
    paths: Sequence[Path], *, batch_size: int = DEFAULT_SNIFF_BATCH_SIZE
) -> BatchSniffResult:
    """Sniff many files with one `file` process per `batch_size` paths.

    `file --brief` prints exactly one line per path, in order, including
    for a path it cannot open (that line is prose, not a MIME type, and
    lands in `errors`). When a batch's output does not have one line per
    path, or the process fails outright, that batch is sniffed again one
    path at a time, so a single odd file cannot cost its neighbours their
    answer. No shell is involved, and `--` keeps any path from being read
    as an option."""
    result = BatchSniffResult()
    for start in range(0, len(paths), max(1, batch_size)):
        chunk = list(paths[start : start + max(1, batch_size)])
        try:
            proc = subprocess.run(
                ["file", "--brief", "--mime-type", "--", *(str(p) for p in chunk)],
                capture_output=True,
                text=True,
                check=False,
                timeout=_BATCH_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        lines = proc.stdout.splitlines() if proc is not None else []
        if proc is None or proc.returncode != 0 or len(lines) != len(chunk):
            for path in chunk:
                _sniff_one_into(path, result)
            continue
        for path, line in zip(chunk, lines, strict=True):
            output = line.strip()
            if _MIME_TYPE_RE.match(output):
                result.mime_by_path[path] = refine_sniffed_mime(path, output)
            else:
                result.errors[path] = f"MIME sniffing failed for '{path}': {output!r}"
    return result


__all__ = [
    "CAF_MIME",
    "DEFAULT_SNIFF_BATCH_SIZE",
    "GENERIC_BINARY_MIME",
    "BatchSniffResult",
    "MimeSnifferFn",
    "real_sniff_mime",
    "refine_sniffed_mime",
    "sniff_mime_batch",
]
