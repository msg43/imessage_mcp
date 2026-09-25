"""Size-based rotation for the logs under `<data_root>/logs` (SPEC §14:
"50 MB rotation, 90-day retention").

macOS's own rotator, `newsyslog`, reads its rules from `/etc/newsyslog.d`,
which needs root, and nothing in this project runs as root. So the
application rotates its own logs: the nightly `imsg backup` calls
:func:`rotate_logs` after its backup, and `imsg logs rotate` runs it by
hand (`--dry-run` shows what it would do).

**Copy, then truncate — never rename.** Every writer of these files holds
them open: launchd's standard-output files, the supervisor's log files
(`imsg.agents.supervise`), the scheduled sync's append handle, and a
KeepAlive server can hold its descriptor for weeks. Renaming the file
would leave each writer appending to the renamed copy. Instead the
current contents are copied into a compressed generation and the file is
truncated in place: same inode, and a writer that opened it for
appending (`O_APPEND`, `>>`, `open(..., "a")`) continues at the new end.
Two costs, both accepted for operational logs and stated here so nobody
rediscovers them: a line written between the last copy and the truncate
(microseconds apart) is lost; and a writer that did *not* open for
appending keeps its old offset and leaves a run of zero bytes before its
next line. Everything this project starts opens its logs for appending.

**What is touched.** Only regular files directly in the logs directory
whose names end in `.log` — never a symlink, a directory, or anything
else an operator keeps there (`.tsv` progress files, pid files). A file
is rotated when it is at least `rotate_bytes` long. Generations are
`<name>.1.gz` (newest) … `<name>.<keep>.gz`; older ones are deleted, and
so is any generation whose file is older than `retention_days`. One
rotation runs at a time (an exclusive lock on `logs/.rotate.lock`).
"""

from __future__ import annotations

import fcntl
import gzip
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from imsg.errors import ImsgError
from imsg.paths import is_contained_in, resolve_path

LOGS_SUBDIR = "logs"
LOCK_FILENAME = ".rotate.lock"
DEFAULT_ROTATE_BYTES = 50 * 10**6
"""SPEC §14: 50 MB."""
DEFAULT_KEEP = 10
DEFAULT_RETENTION_DAYS = 90
"""SPEC §14: 90-day retention."""

_COPY_CHUNK_BYTES = 1 << 20
_CATCH_UP_ROUNDS = 3
"""After the first copy, bytes appended meanwhile are copied too, up to
this many rounds, so the window in which a line can be lost is the last
round only."""
_GENERATION_RE = re.compile(r"^(?P<base>.+\.log)\.(?P<n>\d+)\.gz$")


class LogRotationError(ImsgError):
    """The logs directory could not be rotated safely (it resolves outside
    the data root, or the rotation lock is held by another run)."""


@dataclass(slots=True)
class RotationReport:
    rotated: list[tuple[str, int]] = field(default_factory=list)
    """`(file name, bytes moved into the new generation)`."""
    deleted: list[str] = field(default_factory=list)
    """Generation files removed for age or count."""
    skipped: list[tuple[str, str]] = field(default_factory=list)
    """`(entry name, why)` for `.log` entries that were not rotated."""
    dry_run: bool = False

    def describe(self) -> list[str]:
        verb = "would rotate" if self.dry_run else "rotated"
        lines = [f"{verb} {name} ({size:,} bytes)" for name, size in self.rotated]
        lines += [f"{'would delete' if self.dry_run else 'deleted'} {name}" for name in self.deleted]
        lines += [f"skipped {name}: {why}" for name, why in self.skipped]
        return lines


def logs_dir_for(data_root: Path) -> Path:
    """`<data_root>/logs`, resolved and proven to stay under the data root
    (a symlinked `logs` pointing elsewhere is refused, not followed)."""
    root = resolve_path(data_root)
    directory = resolve_path(root / LOGS_SUBDIR)
    if not is_contained_in(directory, root):
        raise LogRotationError(
            f"{root / LOGS_SUBDIR} resolves to {directory}, outside the data root {root}"
        )
    return directory


@contextmanager
def _rotation_lock(directory: Path) -> Iterator[None]:
    path = directory / LOCK_FILENAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LogRotationError(f"another log rotation holds {path}") from exc
        yield
    finally:
        os.close(fd)


def _generation(path: Path, n: int) -> Path:
    return path.with_name(f"{path.name}.{n}.gz")


def _copy_then_truncate(path: Path) -> int:
    """Compress `path`'s current contents into `<name>.1.gz` (older
    generations already shifted by the caller) and truncate `path` in
    place. Returns the bytes copied."""
    target = _generation(path, 1)
    staging = target.with_name(f".{target.name}.partial")
    copied = 0
    with path.open("rb") as source, gzip.open(staging, "wb") as sink:
        for _ in range(_CATCH_UP_ROUNDS):
            chunk = source.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            while chunk:
                sink.write(chunk)
                copied += len(chunk)
                chunk = source.read(_COPY_CHUNK_BYTES)
        # Take whatever arrived during the copy, then empty the file in
        # place (same inode, so appending writers carry on).
        with path.open("r+b") as handle:
            tail = source.read()
            if tail:
                sink.write(tail)
                copied += len(tail)
            handle.truncate(0)
    os.chmod(staging, 0o600)
    os.replace(staging, target)
    return copied


def rotate_logs(
    data_root: Path,
    *,
    rotate_bytes: int = DEFAULT_ROTATE_BYTES,
    keep: int = DEFAULT_KEEP,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    now: float | None = None,
) -> RotationReport:
    """Rotate every `*.log` in `<data_root>/logs` that has reached
    `rotate_bytes`, then prune generations beyond `keep` or older than
    `retention_days`. A missing logs directory is nothing to rotate."""
    if rotate_bytes < 1 or keep < 1 or retention_days < 1:
        raise LogRotationError("rotate_bytes, keep and retention_days must all be at least 1")
    report = RotationReport(dry_run=dry_run)
    directory = logs_dir_for(data_root)
    if not directory.is_dir():
        return report
    moment = time.time() if now is None else now
    cutoff = moment - retention_days * 86400

    with _rotation_lock(directory):
        for entry in sorted(directory.iterdir()):
            if not entry.name.endswith(".log"):
                continue
            if entry.is_symlink() or not entry.is_file():
                report.skipped.append((entry.name, "not a regular file"))
                continue
            size = entry.stat().st_size
            if size < rotate_bytes:
                continue
            report.rotated.append((entry.name, size))
            if dry_run:
                continue
            # Shift older generations up by one, dropping the oldest.
            for n in range(keep, 0, -1):
                older = _generation(entry, n)
                if not older.exists():
                    continue
                if n == keep:
                    older.unlink()
                    report.deleted.append(older.name)
                else:
                    os.replace(older, _generation(entry, n + 1))
            report.rotated[-1] = (entry.name, _copy_then_truncate(entry))

        for entry in sorted(directory.iterdir()):
            match = _GENERATION_RE.match(entry.name)
            if match is None or entry.is_symlink() or not entry.is_file():
                continue
            too_many = int(match.group("n")) > keep
            too_old = entry.stat().st_mtime < cutoff
            if not (too_many or too_old) or entry.name in report.deleted:
                continue
            report.deleted.append(entry.name)
            if not dry_run:
                entry.unlink()
    return report


def total_log_bytes(data_root: Path) -> int:
    """Bytes held by `*.log` files and their generations — for a status
    line that shows whether rotation is keeping up."""
    directory = logs_dir_for(data_root)
    if not directory.is_dir():
        return 0
    return sum(
        entry.stat().st_size
        for entry in directory.iterdir()
        if entry.is_file()
        and not entry.is_symlink()
        and (entry.name.endswith(".log") or _GENERATION_RE.match(entry.name))
    )


__all__ = [
    "DEFAULT_KEEP",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_ROTATE_BYTES",
    "LOCK_FILENAME",
    "LOGS_SUBDIR",
    "LogRotationError",
    "RotationReport",
    "logs_dir_for",
    "rotate_logs",
    "total_log_bytes",
]
