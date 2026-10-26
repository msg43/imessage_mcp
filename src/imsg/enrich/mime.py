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

**`file` runs sandboxed (2026-09-29, QA review).** `file` parses the
attachment's bytes, which a stranger can choose, so it runs like every
other decoder (`imsg.enrich.sandboxed_decoder`): under `sandbox-exec`
with no network and writes only in a work directory under `data_root`,
inside a `DecoderBudget`, and within its own time bound (10 s for one
path, 120 s for a batch). In a task the budget is the task's; the
planner, which runs outside any task, makes a scratch one
(`scratch_budget`). A sniff that passes a ceiling or its time bound, or
that the sandbox refuses, raises `UntrustedAttachmentError`, the failure
a sniff has always raised, and the task is recorded `failed`.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from imsg.enrich.sandboxed_decoder import DecoderBudget, run_decoder, scratch_budget
from imsg.errors import EnrichmentError, UntrustedAttachmentError

MimeSnifferFn = Callable[[Path], str]
"""A stand-in sniffer a test can hand the pipeline or the planner in
place of the sandboxed `file`."""

GENERIC_BINARY_MIME = "application/octet-stream"
CAF_MIME = "audio/x-caf"
_CAF_SIGNATURE = b"caff\x00\x01"  # 'caff', then mFileVersion 1 (big-endian UInt16)

DEFAULT_SNIFF_BATCH_SIZE = 256
"""Paths per `file` process in `sniff_mime_batch`. On the production host
one `file` per 400 paths sniffed all 100,046 materialized attachments in
about 41 s (2026-09-23); one process per path costs a few milliseconds
each in process start-up alone."""

SINGLE_SNIFF_TIMEOUT_SECONDS = 10
BATCH_SNIFF_TIMEOUT_SECONDS = 120

SNIFF_MAX_TEMP_BYTES = 64 * 1024 * 1024
"""The work-directory ceiling of a planner sniff's scratch budget. `file`
writes nothing; only its captured stdout and stderr land there."""

_MAX_LINE_BYTES = 4096
"""Per path, a generous bound on one line of `file --brief` output, used
to bound what is read back."""

_SNIFF_OUTPUT_NAME = "file-mime.out"

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


def _run_file(
    paths: Sequence[Path], *, budget: DecoderBudget, max_seconds: float
) -> tuple[int, list[str], str]:
    """Run `file --brief --mime-type -- <paths>` sandboxed, inside
    `budget` and `max_seconds`. Returns its exit code, its stdout lines
    and the tail of its stderr. A ceiling hit raises
    `UntrustedAttachmentError`; a `file` that could not be started at all
    raises the same, as `subprocess.run`'s `OSError` used to."""
    subject = paths[0] if len(paths) == 1 else Path(f"{len(paths)} attachments")
    try:
        run = run_decoder(
            ["file", "--brief", "--mime-type", "--", *(str(p) for p in paths)],
            budget=budget,
            name="file",
            subject=subject,
            stdout_name=_SNIFF_OUTPUT_NAME,
            max_stdout_bytes=_MAX_LINE_BYTES * len(paths),
            max_seconds=max_seconds,
        )
    except UntrustedAttachmentError:
        raise
    except EnrichmentError as exc:
        raise UntrustedAttachmentError(f"MIME sniffing failed for '{subject}': {exc}") from exc
    assert run.stdout_path is not None
    try:
        lines = run.stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
    finally:
        with contextlib.suppress(OSError):
            run.stdout_path.unlink()
    stderr = run.stderr_tail()
    with contextlib.suppress(OSError):
        run.stderr_path.unlink()
    return run.returncode, lines, stderr


def sniff_mime(
    path: Path, *, budget: DecoderBudget, max_seconds: float = SINGLE_SNIFF_TIMEOUT_SECONDS
) -> str:
    """`path`'s content-sniffed MIME type, from `file` run sandboxed
    inside `budget` (module docstring). Raises `UntrustedAttachmentError`
    when `file` cannot sniff it, passes a ceiling, or is refused."""
    returncode, lines, stderr = _run_file([path], budget=budget, max_seconds=max_seconds)
    output = lines[0].strip() if len(lines) == 1 else "\n".join(lines)
    if returncode != 0 or not _MIME_TYPE_RE.match(output):
        raise UntrustedAttachmentError(
            f"MIME sniffing failed for '{path}': 'file' exited {returncode} "
            f"with unparseable output {output!r} (stderr: {stderr!r})"
        )
    return refine_sniffed_mime(path, output)


def sniff_mime_in_scratch(path: Path, *, data_root: Path) -> str:
    """`sniff_mime` for a caller outside any enrichment task (S5a's
    enqueue hook), in a scratch work directory under `data_root`."""
    with scratch_budget(
        data_root,
        "sniff",
        timeout_seconds=SINGLE_SNIFF_TIMEOUT_SECONDS,
        max_temp_bytes=SNIFF_MAX_TEMP_BYTES,
    ) as budget:
        return sniff_mime(path, budget=budget)


@dataclass(frozen=True, slots=True)
class BatchSniffResult:
    mime_by_path: dict[Path, str] = field(default_factory=dict)
    """Every path `file` could sniff, with its (refined) MIME type."""
    errors: dict[Path, str] = field(default_factory=dict)
    """Every path it could not, with what `file` said instead."""


def _sniff_one_into(path: Path, result: BatchSniffResult, budget: DecoderBudget) -> None:
    try:
        result.mime_by_path[path] = sniff_mime(path, budget=budget)
    except UntrustedAttachmentError as exc:
        result.errors[path] = str(exc)


def sniff_mime_batch(
    paths: Sequence[Path], *, data_root: Path, batch_size: int = DEFAULT_SNIFF_BATCH_SIZE
) -> BatchSniffResult:
    """Sniff many files with one sandboxed `file` process per
    `batch_size` paths, each inside a scratch budget under `data_root`
    and a 120 s bound.

    `file --brief` prints exactly one line per path, in order, including
    for a path it cannot open (that line is prose, not a MIME type, and
    lands in `errors`). When a batch's output does not have one line per
    path, or the process fails outright or passes a ceiling, that batch
    is sniffed again one path at a time (each under the 10 s bound), so a
    single odd file cannot cost its neighbours their answer. No shell is
    involved, and `--` keeps any path from being read as an option."""
    result = BatchSniffResult()
    step = max(1, batch_size)
    for start in range(0, len(paths), step):
        chunk = list(paths[start : start + step])
        with scratch_budget(
            data_root,
            "sniff",
            timeout_seconds=BATCH_SNIFF_TIMEOUT_SECONDS + SINGLE_SNIFF_TIMEOUT_SECONDS * len(chunk),
            max_temp_bytes=SNIFF_MAX_TEMP_BYTES,
        ) as budget:
            try:
                returncode, lines, _ = _run_file(
                    chunk, budget=budget, max_seconds=BATCH_SNIFF_TIMEOUT_SECONDS
                )
            except UntrustedAttachmentError:
                returncode, lines = None, []
            if returncode != 0 or len(lines) != len(chunk):
                for path in chunk:
                    _sniff_one_into(path, result, budget)
                continue
        for path, line in zip(chunk, lines, strict=True):
            output = line.strip()
            if _MIME_TYPE_RE.match(output):
                result.mime_by_path[path] = refine_sniffed_mime(path, output)
            else:
                result.errors[path] = f"MIME sniffing failed for '{path}': {output!r}"
    return result


__all__ = [
    "BATCH_SNIFF_TIMEOUT_SECONDS",
    "CAF_MIME",
    "DEFAULT_SNIFF_BATCH_SIZE",
    "GENERIC_BINARY_MIME",
    "SINGLE_SNIFF_TIMEOUT_SECONDS",
    "BatchSniffResult",
    "MimeSnifferFn",
    "refine_sniffed_mime",
    "sniff_mime",
    "sniff_mime_batch",
    "sniff_mime_in_scratch",
]
