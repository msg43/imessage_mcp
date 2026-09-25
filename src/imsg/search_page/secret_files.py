"""Owner-only files on the encrypted volume: the password hash, the
session store and the internal model API's shared secret.

Reading refuses a file that is a symlink, is not a regular file, is not
owned by this user, or grants any permission to group or others (the
rule `ssh` applies to private keys). Writing creates the file 0600 from
the start (never chmod-after-write, which leaves a window) inside 0700
directories, then renames it into place so a reader never sees half a
file. Every path is resolved and checked against `paths.data_root`
first, so neither a `..` nor a planted symlink can move a secret off
the encrypted volume.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

from imsg.paths import is_contained_in, join_under_root, resolve_path
from imsg.search_page.errors import SecretFileError

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
MAX_SECRET_FILE_BYTES = 1024 * 1024


def data_file(data_root: Path, relative: Path) -> Path:
    """`relative` joined under `data_root`, resolved, and refused unless it
    stays beneath `data_root` after every symlink is followed."""
    candidate = join_under_root(data_root, relative)
    resolved = resolve_path(candidate)
    if not is_contained_in(resolved, data_root):
        raise SecretFileError(
            f"{relative} resolves to {resolved}, outside paths.data_root ({data_root}); "
            f"search-page files must stay on the encrypted volume"
        )
    return resolved


def check_private_file(path: Path) -> os.stat_result:
    """The `lstat` of `path` if it is a regular, owner-only file of ours.
    Raises `SecretFileError` naming the rule it breaks."""
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise SecretFileError(f"{path} does not exist") from exc
    if stat.S_ISLNK(info.st_mode):
        raise SecretFileError(f"{path} is a symlink; secret files must be regular files")
    if not stat.S_ISREG(info.st_mode):
        raise SecretFileError(f"{path} is not a regular file")
    if info.st_uid != os.getuid():
        raise SecretFileError(f"{path} is not owned by this user (uid {os.getuid()})")
    if info.st_mode & 0o077:
        raise SecretFileError(
            f"{path} is mode {stat.S_IMODE(info.st_mode):04o}; it must be 0600 "
            f"(no access for group or others)"
        )
    if info.st_size > MAX_SECRET_FILE_BYTES:
        raise SecretFileError(f"{path} is larger than {MAX_SECRET_FILE_BYTES} bytes")
    return info


def read_private_file(path: Path) -> bytes:
    """Read an owner-only file, re-checking the open descriptor so a swap
    between the check and the read cannot slip another file in."""
    before = check_private_file(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise SecretFileError(f"{path} changed while it was being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def ensure_private_dir(directory: Path) -> None:
    """Create `directory` (and parents) 0700 if missing; tighten an
    existing one we own to 0700."""
    directory.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    info = os.lstat(directory)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SecretFileError(f"{directory} is not a plain directory")
    if info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) != PRIVATE_DIR_MODE:
        os.chmod(directory, PRIVATE_DIR_MODE)


def write_private_file(path: Path, data: bytes) -> None:
    """Atomically replace `path` with `data`, 0600 from creation."""
    ensure_private_dir(path.parent)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temp, flags, PRIVATE_FILE_MODE)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        temp.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.chmod(temp, PRIVATE_FILE_MODE)
    os.replace(temp, path)


__all__ = [
    "MAX_SECRET_FILE_BYTES",
    "PRIVATE_DIR_MODE",
    "PRIVATE_FILE_MODE",
    "check_private_file",
    "data_file",
    "ensure_private_dir",
    "read_private_file",
    "write_private_file",
]
