"""Run an external decoder on attacker-supplied bytes, fenced in (SPEC §8
S5b, D6).

Every decoder the enrichment pipeline shells out to (`pdfinfo`,
`pdftotext`, `pdftoppm`, `ffprobe`, `ffmpeg`) reads bytes a stranger can
choose: anyone who can send the owner an iMessage can send an attachment.
Each one runs:

- **Under `sandbox-exec`**, with `TASK_SANDBOX_PROFILE`: no network at
  all, and no file writes anywhere except inside the task's own work
  directory. Reading stays allowed, because the decoders need their own
  libraries and fonts and the input lives in the attachment cache.
  Checked 2026-09-24 on macOS 26.6 with poppler 26.08 and ffmpeg 8.0.1: a
  write outside the work directory fails with "Operation not permitted",
  and an `http://` input opens no connection (a listener on 127.0.0.1
  counted 0 connections sandboxed, 1 unsandboxed). `textutil` has had the
  same kind of sandbox since 2026-09-23 (`imsg.enrich.doc_text`), with no
  writes at all.
- **Without a shell**, from an argument list, with stdin closed, the work
  directory as its current directory and as `TMPDIR`.
- **Inside a budget** (`DecoderBudget`), checked every `POLL_SECONDS`
  while it runs: the task's wall-clock deadline
  (`enrichment.limits.task_timeout_seconds`, for the whole task rather
  than for each call), the bytes in the work directory
  (`temp_bytes_per_task`), the decoder's physical memory footprint
  (`max_decoder_memory_bytes`), and, where the caller keeps the decoder's
  output, the size of that output. The first ceiling passed stops the
  decoder with SIGKILL, which it cannot catch or ignore.

A ceiling hit is a typed permanent failure (`UntrustedAttachmentError`):
the same file passes the same ceiling on every try, so it is recorded
`failed` at once rather than retried. A decoder that exits non-zero is a
plain `EnrichmentError`, retried with backoff, as before.

Measured 2026-09-24 on the development host, the largest legitimate
footprints were 689 MiB (`pdftoppm` rendering a 50-inch page at the
pixel ceiling) and 620 MiB (`ffmpeg` sampling frames from 4K H.264);
`pdftotext` extracting 50 MB of text peaked at 13 MiB. The 4 GiB default
memory ceiling is about six times the largest of those.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

from imsg import constants
from imsg.errors import EnrichmentError, UntrustedAttachmentError
from imsg.host_memory import process_footprint_bytes
from imsg.paths import is_contained_in, resolve_path

if TYPE_CHECKING:
    from imsg.config.schema import EnrichmentLimits

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

WORK_DIR_PARAM = "WORK_DIR"
"""The profile parameter carrying the work directory. It is passed with
`sandbox-exec -D`, never spliced into the profile text, so no path can
change the profile's meaning."""

TASK_SANDBOX_PROFILE = (
    "(version 1)"
    "(allow default)"
    "(deny network*)"
    "(deny file-write*)"
    f'(allow file-write* (subpath (param "{WORK_DIR_PARAM}")))'
    '(allow file-write-data (literal "/dev/null"))'
)
"""No network; writes only beneath the task's work directory (and to
`/dev/null`). A later rule wins over an earlier one for the same
operation, which is what lets the last two carve exceptions out of the
blanket write denial."""

MAX_DECODER_OUTPUT_BYTES = 64 * 1024 * 1024
"""The largest output a decoder may write for the pipeline to read back
into memory (`textutil`'s text, `pdftotext`'s text). A decoder is stopped
once its output passes this."""

DEFAULT_MAX_DECODER_MEMORY_BYTES = constants.DEFAULT_MAX_DECODER_MEMORY_BYTES
"""`enrichment.limits.max_decoder_memory_bytes`'s default (see the
module docstring for the measurements behind it)."""

POLL_SECONDS = 0.05


def directory_bytes(root: Path) -> int:
    """The total size of the regular files beneath `root`, not following
    symlinks. A file that disappears while it is being counted is
    skipped."""
    total = 0
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
    return total


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


@dataclass(frozen=True, slots=True)
class DecoderBudget:
    """What one enrichment task may spend on decoders, from the moment it
    started: the directory they may write into, the deadline, and the
    temp-space and memory ceilings. Made once per task
    (`DecoderBudget.for_task`) and handed to every decoder the task runs,
    so the deadline covers the whole task rather than each call."""

    work_dir: Path
    """Resolved, so the sandbox profile matches the path the kernel sees
    (`/var` is a symlink to `/private/var` on macOS)."""
    deadline: float
    timeout_seconds: float
    max_temp_bytes: int
    max_memory_bytes: int
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    @classmethod
    def start(
        cls,
        work_dir: Path,
        *,
        timeout_seconds: float,
        max_temp_bytes: int,
        max_memory_bytes: int = DEFAULT_MAX_DECODER_MEMORY_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> DecoderBudget:
        work_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            work_dir=resolve_path(work_dir),
            deadline=clock() + timeout_seconds,
            timeout_seconds=timeout_seconds,
            max_temp_bytes=max_temp_bytes,
            max_memory_bytes=max_memory_bytes,
            clock=clock,
        )

    @classmethod
    def for_task(
        cls,
        work_dir: Path,
        limits: EnrichmentLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> DecoderBudget:
        """The budget `enrichment.limits` gives one task."""
        return cls.start(
            work_dir,
            timeout_seconds=limits.task_timeout_seconds,
            max_temp_bytes=limits.temp_bytes_per_task,
            max_memory_bytes=limits.max_decoder_memory_bytes,
            clock=clock,
        )

    def remaining_seconds(self) -> float:
        return self.deadline - self.clock()

    def check_deadline(self, doing: str) -> None:
        """Raise the typed failure once the task's deadline has passed.
        Called between units of in-process work (one page, one frame),
        which no subprocess watch can interrupt."""
        if self.clock() > self.deadline:
            raise UntrustedAttachmentError(
                f"{doing}: the task ran past its {self.timeout_seconds:g}s ceiling "
                f"(enrichment.limits.task_timeout_seconds)"
            )

    def temp_bytes(self) -> int:
        return directory_bytes(self.work_dir)

    def check_temp_bytes(self, doing: str) -> None:
        used = self.temp_bytes()
        if used > self.max_temp_bytes:
            raise UntrustedAttachmentError(
                f"{doing}: the task's work directory holds {used} bytes, over "
                f"enrichment.limits.temp_bytes_per_task ({self.max_temp_bytes})"
            )

    def contains(self, path: Path) -> bool:
        return is_contained_in(path, self.work_dir)


@dataclass(frozen=True, slots=True)
class DecoderRun:
    """A decoder that ran to its own exit inside its budget."""

    returncode: int
    stdout_path: Path | None
    """The file its stdout went to, or `None` when it was discarded."""
    stderr_path: Path

    def stderr_tail(self, limit: int = 500) -> str:
        """The last `limit` bytes of its stderr, as text: enough for an
        error message, never the whole of a log an attacker can grow."""
        try:
            with self.stderr_path.open("rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                fh.seek(max(0, size - limit))
                return fh.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def stderr_lines(self, max_line_bytes: int = 64 * 1024) -> Iterator[str]:
        """Its stderr line by line, each line cut at `max_line_bytes`, so
        a pathological log is scanned without being held in memory."""
        with self.stderr_path.open("rb") as fh:
            while True:
                raw = fh.readline(max_line_bytes)
                if not raw:
                    return
                yield raw.decode("utf-8", errors="replace")


def resolve_executable(program: str) -> str:
    """`program`'s absolute path (a bare name is looked up on `PATH`), or
    `EnrichmentError` naming what is missing — an environment problem,
    retried with backoff like any failure to run a decoder."""
    if os.path.isabs(program):
        return program
    found = shutil.which(program)
    if found is None:
        raise EnrichmentError(f"{program} is not installed (not found on PATH)")
    return found


def decoder_command(argv: Sequence[str], work_dir: Path, *, profile: str = TASK_SANDBOX_PROFILE) -> list[str]:
    """The exact argv that runs `argv` sandboxed: `sandbox-exec` with the
    work directory as a profile parameter, then the decoder by absolute
    path. No shell anywhere."""
    return [
        SANDBOX_EXEC,
        "-D",
        f"{WORK_DIR_PARAM}={work_dir}",
        "-p",
        profile,
        resolve_executable(argv[0]),
        *argv[1:],
    ]


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL cannot be caught or ignored, so the wait always returns."""
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    proc.wait()


def run_decoder(
    argv: Sequence[str],
    *,
    budget: DecoderBudget,
    name: str,
    subject: Path,
    stdout_name: str | None = None,
    max_stdout_bytes: int | None = None,
) -> DecoderRun:
    """Run `argv` sandboxed, inside `budget` (module docstring), and
    return once it exits on its own.

    `stdout_name` is a file name in the work directory that receives the
    decoder's stdout; `None` discards it. `max_stdout_bytes` stops the
    decoder once that file passes it. `name` and `subject` (the
    attachment) only word the errors. Every ceiling is checked once more
    after the decoder exits, since the last burst of writing can land
    between two checks."""
    budget.check_deadline(f"{name} on '{subject}'")
    work_dir = budget.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    command = decoder_command(argv, work_dir)
    stdout_path = None
    if stdout_name is not None:
        if os.sep in stdout_name or stdout_name in ("", ".", ".."):
            raise EnrichmentError(f"decoder output name {stdout_name!r} must be a plain file name")
        stdout_path = work_dir / stdout_name
    err_fd, err_name = tempfile.mkstemp(dir=work_dir, prefix=f"{name}-", suffix=".err")
    stderr_path = Path(err_name)
    env = {**os.environ, "TMPDIR": f"{work_dir}{os.sep}"}
    with contextlib.ExitStack() as stack:
        err = stack.enter_context(os.fdopen(err_fd, "wb"))
        out: IO[bytes] | int = subprocess.DEVNULL
        if stdout_path is not None:
            out = stack.enter_context(stdout_path.open("wb"))
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=work_dir,
                env=env,
            )
        except OSError as exc:
            raise EnrichmentError(f"{name} could not run: {exc}") from exc
        try:
            returncode = _watch(
                proc,
                budget=budget,
                name=name,
                subject=subject,
                stdout_path=stdout_path,
                max_stdout_bytes=max_stdout_bytes,
            )
        except BaseException:
            _stop(proc)
            raise
    if stdout_path is not None and max_stdout_bytes is not None:
        _check_output(stdout_path, max_stdout_bytes, name=name, subject=subject)
    budget.check_temp_bytes(f"{name} on '{subject}'")
    return DecoderRun(returncode=returncode, stdout_path=stdout_path, stderr_path=stderr_path)


def _check_output(path: Path, max_bytes: int, *, name: str, subject: Path) -> None:
    size = _file_size(path)
    if size > max_bytes:
        raise UntrustedAttachmentError(
            f"{name} output for '{subject}' passed the decoder output ceiling "
            f"({max_bytes} bytes)"
        )


def _watch(
    proc: subprocess.Popen[bytes],
    *,
    budget: DecoderBudget,
    name: str,
    subject: Path,
    stdout_path: Path | None,
    max_stdout_bytes: int | None,
) -> int:
    """Wait for `proc`, checking every ceiling each `POLL_SECONDS`; raise
    the first one passed (the caller stops the process). A footprint that
    cannot be read — the process exiting between two checks — is skipped
    for that check."""
    doing = f"{name} on '{subject}'"
    while True:
        try:
            return proc.wait(timeout=POLL_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        if stdout_path is not None and max_stdout_bytes is not None:
            _check_output(stdout_path, max_stdout_bytes, name=name, subject=subject)
        budget.check_temp_bytes(doing)
        footprint = process_footprint_bytes(proc.pid)
        if footprint is not None and footprint > budget.max_memory_bytes:
            raise UntrustedAttachmentError(
                f"{doing}: the decoder's memory footprint reached {footprint} bytes, over "
                f"enrichment.limits.max_decoder_memory_bytes ({budget.max_memory_bytes})"
            )
        budget.check_deadline(doing)


__all__ = [
    "DEFAULT_MAX_DECODER_MEMORY_BYTES",
    "MAX_DECODER_OUTPUT_BYTES",
    "POLL_SECONDS",
    "SANDBOX_EXEC",
    "TASK_SANDBOX_PROFILE",
    "WORK_DIR_PARAM",
    "DecoderBudget",
    "DecoderRun",
    "decoder_command",
    "directory_bytes",
    "resolve_executable",
    "run_decoder",
]
