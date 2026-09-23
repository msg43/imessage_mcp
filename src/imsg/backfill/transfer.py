"""Read-only copies of attachment files into this index's staging directory
(owner decision D13's fetcher).

Two copies exist, and both are plain `rsync` runs of just the listed files,
with the file list on stdin (`--from0 --files-from=-`, NUL-separated, so no
list is ever written to disk, and no path is split or shell-parsed):

- **push** — on the host that holds the files (another Mac, with its
  attached drives): a local root is the source, this index host's staging
  directory is the destination, over SSH. For hosts the index host cannot
  reach itself.
- **pull** — on the index host: a directory on a host it can reach over
  SSH (a NAS share) is the source, local staging is the destination.

**Neither copy can write to its source.** In both, the source side is
rsync's *sender*, and a sender only reads — unless it is told to remove
what it sent, and the flags that tell it to (`SOURCE_WRITE_FLAGS`) are
refused before any command runs (`assert_read_only`), as are the flags
that delete anything anywhere (`DELETE_FLAGS`). The destination is always
a staging directory under `paths.data_root`: never a source, and on the
encrypted volume (non-negotiable 2). Remote paths are checked to be plain
absolute paths before they reach a remote shell.

rsync's stderr names files, so it is never printed: callers write it to a
log under `paths.data_root` (on the encrypted volume) and print counts.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from imsg.backfill.locations import safe_relative_path

SOURCE_WRITE_FLAGS: frozenset[str] = frozenset({"--remove-source-files", "--remove-sent-files"})
"""The only rsync options that make a sender modify its own files."""

DELETE_FLAGS: frozenset[str] = frozenset(
    {
        "--delete", "--del", "--delete-before", "--delete-during", "--delete-delay",
        "--delete-after", "--delete-excluded", "--delete-missing-args", "--force",
        "--force-delete", "--max-delete",
    }
)
"""Options that delete files. None is ever needed: a copy only adds."""

_BASE_FLAGS: tuple[str, ...] = (
    "--from0",
    "--files-from=-",
    "--perms",
    "--chmod=D700,F600",
    "--timeout=300",
    "--no-motd",
)
"""`--files-from` implies `--relative`, so each listed path is recreated
below the destination. Symlinks at the source are skipped (no `--links`,
no `--copy-links`), and staged copies are private to the owner."""

REACHED_EXIT_CODES: frozenset[int] = frozenset({0, 23, 24})
"""rsync ran against both ends: 0 is complete; 23 is a partial transfer
(some listed file could not be read or was not there) and 24 means a
listed file vanished. Anything else means the copy did not run, and every
file in it counts as unreachable."""

_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+@-]*$")
_SSH_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")


class TransferError(ValueError):
    """A copy command that must not run: a flag that writes to or deletes
    from a source, an unsafe remote path, or an unsafe listed path."""


@dataclass(frozen=True, slots=True)
class CopyCommand:
    argv: tuple[str, ...]
    file_list: bytes
    """NUL-separated relative paths, fed to rsync on stdin."""
    destination: str

    @property
    def file_count(self) -> int:
        return self.file_list.count(b"\0")


@dataclass(frozen=True, slots=True)
class CopyResult:
    returncode: int
    stderr: str

    @property
    def reached(self) -> bool:
        return self.returncode in REACHED_EXIT_CODES


CopyRunner = Callable[[CopyCommand], CopyResult]


def assert_read_only(argv: Sequence[str]) -> None:
    """Refuse any argv carrying a flag that writes to a source or deletes
    anything, in either `--flag` or `--flag=value` form."""
    for arg in argv:
        name = arg.split("=", 1)[0]
        if name in SOURCE_WRITE_FLAGS or name in DELETE_FLAGS:
            raise TransferError(f"refusing an rsync flag that writes to a source or deletes: {arg}")


def _file_list(relpaths: Sequence[str]) -> bytes:
    for rel in relpaths:
        if not safe_relative_path(rel):
            raise TransferError("refusing an unsafe relative path in a copy list")
    return b"".join(rel.encode("utf-8") + b"\0" for rel in relpaths)


def _check_remote_path(path: str) -> str:
    if not _REMOTE_PATH_RE.fullmatch(path) or "/../" in f"{path}/" or "/./" in f"{path}/":
        raise TransferError(
            f"remote path must be a plain absolute path of [A-Za-z0-9._/+@-], got {path!r}"
        )
    return path.rstrip("/") or "/"


def _check_ssh_host(host: str) -> str:
    if not _SSH_HOST_RE.fullmatch(host):
        raise TransferError(f"ssh host must be a plain host name or alias, got {host!r}")
    return host


def _argv(rsync: str, ssh: str, source: str, destination: str, *, dry_run: bool,
          extra: Sequence[str] = ()) -> tuple[str, ...]:
    argv = (
        *shlex.split(rsync), *_BASE_FLAGS, *extra, *(("--dry-run",) if dry_run else ()),
        "-e", ssh, source, destination,
    )
    assert_read_only(argv)
    return argv


def push_command(
    *,
    rsync: str,
    ssh: str,
    local_root: Path,
    ssh_host: str,
    remote_staging_dir: str,
    relpaths: Sequence[str],
    dry_run: bool = False,
) -> CopyCommand:
    """Copy `relpaths` below `local_root` (this host, read only) into
    `remote_staging_dir` on `ssh_host` (the index host's staging directory
    for this location). `--mkpath` creates that directory on the index
    host."""
    destination = f"{_check_ssh_host(ssh_host)}:{_check_remote_path(remote_staging_dir)}/"
    source = f"{local_root.as_posix().rstrip('/')}/"
    return CopyCommand(
        argv=_argv(rsync, ssh, source, destination, dry_run=dry_run, extra=("--mkpath",)),
        file_list=_file_list(relpaths),
        destination=destination,
    )


def pull_command(
    *,
    rsync: str,
    ssh: str,
    ssh_host: str,
    remote_root: str,
    local_staging_dir: Path,
    relpaths: Sequence[str],
    dry_run: bool = False,
) -> CopyCommand:
    """Copy `relpaths` below `remote_root` on `ssh_host` (read only: the
    remote side is rsync's sender) into `local_staging_dir`."""
    source = f"{_check_ssh_host(ssh_host)}:{_check_remote_path(remote_root)}/"
    destination = f"{local_staging_dir.as_posix().rstrip('/')}/"
    return CopyCommand(
        argv=_argv(rsync, ssh, source, destination, dry_run=dry_run),
        file_list=_file_list(relpaths),
        destination=destination,
    )


def run_copy(command: CopyCommand, *, timeout_seconds: float = 6 * 3600) -> CopyResult:
    """Run one copy. The read-only check runs again here, so a command
    built by hand cannot skip it."""
    assert_read_only(command.argv)
    try:
        proc = subprocess.run(
            list(command.argv),
            input=command.file_list,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return CopyResult(returncode=127, stderr=f"rsync not found: {exc}")
    except subprocess.TimeoutExpired:
        return CopyResult(returncode=30, stderr=f"rsync timed out after {timeout_seconds}s")
    return CopyResult(returncode=proc.returncode, stderr=proc.stderr.decode("utf-8", "replace"))


__all__ = [
    "DELETE_FLAGS",
    "REACHED_EXIT_CODES",
    "SOURCE_WRITE_FLAGS",
    "CopyCommand",
    "CopyResult",
    "CopyRunner",
    "TransferError",
    "assert_read_only",
    "pull_command",
    "push_command",
    "run_copy",
]
