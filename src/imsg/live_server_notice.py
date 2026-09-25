"""The file a live MCP server holds while it waits to load its models, so
background work gives way to it.

**Why (2026-09-25).** The public MCP server was restarted with new code
while a scheduled `imsg sync` was segmenting. Memory admission
(`imsg.memory_admission`) counted the sync's reservation against the
server: 9.9 GiB of the 21.0 GiB the sync had been admitted for about
nine minutes earlier, which it had not taken up. The server refused to
load its models, and Gemini had no working search for about 12 minutes,
until the sync was stopped by hand. The MCP servers answer people;
background work (segment, embed, the enrich worker, sync's heavy steps,
eval) can wait. This file is how a live server tells every background
process on the host that it is waiting.

**The notice.** A live server that wants its models and cannot load them
yet posts `<data_root>/run/live-servers-waiting/<pid>.json`, naming
itself, and holds an exclusive `flock` on it for as long as it is posted.
Background processes read the directory:

- before a model load (`imsg.memory_admission`): a posted notice is a
  refusal, so no background load starts while a live server waits;
- between units of work (`imsg.background_gate`): a posted notice stops
  the command after the unit in hand, which drops its models, its memory
  reservation and the heavy-model lock, and exits 75 (`deferred: memory`).

When the live server posts and withdraws its notice is decided in
`imsg.memory_admission.LiveServerAdmission`.

**Liveness comes from the kernel, as with the heavy-model lock.** A
notice counts only while its file is locked, and the kernel drops the
lock the moment the process exits, however it exits. A server that
crashes or is killed stops holding background work back at once, and a
leftover file counts for nothing even if another process later gets the
same pid, because nobody holds its lock. The file's content (pid, role,
command, when it was posted) is only for `imsg status` and log lines.

**Posted atomically.** The file is written and locked under a temporary
name, then renamed into place, so a file visible under its final name is
always either locked by its live owner or left behind by a dead one; a
reader never catches it before its lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from imsg.errors import ImsgError
from imsg.paths import is_contained_in, join_under_root, resolve_path

RUN_SUBDIR = "run"
NOTICES_SUBDIR = "live-servers-waiting"
"""`<data_root>/run/live-servers-waiting/`, one `<pid>.json` per waiting
live server."""

_NOTICE_NAME = re.compile(r"^(\d+)\.json$")
_MAX_NOTICE_BYTES = 4096


class LiveServerNoticePathError(ImsgError):
    """The notices directory resolves outside `data_root` (a `run`
    symlink pointing elsewhere)."""


def notices_directory(data_root: Path) -> Path:
    path = join_under_root(data_root, Path(RUN_SUBDIR) / NOTICES_SUBDIR)
    if not is_contained_in(path, data_root):
        raise LiveServerNoticePathError(f"'{path}' resolves outside data_root '{data_root}'")
    return resolve_path(path)


@dataclass(frozen=True, slots=True)
class WaitingLiveServer:
    """A posted notice, as its owner described itself."""

    pid: int
    role: str
    command: str
    since: str
    """When the notice was posted, ISO 8601 UTC."""

    def describe(self) -> str:
        return f"{self.command} (pid {self.pid}) since {self.since}"


def _parse(raw: bytes, pid: int) -> WaitingLiveServer:
    """The notice's content. The lock, not the content, says the notice is
    posted, so a file that cannot be parsed still counts, described as
    unknown."""
    unknown = WaitingLiveServer(pid=pid, role="unknown", command="a live MCP server", since="unknown")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return unknown
    if not isinstance(document, dict):
        return unknown
    role, command, since = document.get("role"), document.get("command"), document.get("since")
    if not (isinstance(role, str) and isinstance(command, str) and isinstance(since, str)):
        return unknown
    return WaitingLiveServer(pid=pid, role=role, command=command, since=since)


def _posted_notice(path: Path, pid: int) -> WaitingLiveServer | None:
    """The notice at `path` if its owner holds it, else `None`. Takes
    nothing an owner would wait for: a shared, non-blocking lock, released
    at once, and owners never take the lock again after posting."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return None  # withdrawn since the directory was listed
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _parse(os.pread(fd, _MAX_NOTICE_BYTES, 0), pid)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None  # nobody holds it: its owner has exited
    finally:
        os.close(fd)


def waiting_live_servers(data_root: Path) -> list[WaitingLiveServer]:
    """Every live server whose notice is posted right now, in pid order.
    Raises `OSError` or `ImsgError` when the directory cannot be read;
    callers decide what that means (background work treats it as a
    reason to hold back: `imsg.memory_admission`, `imsg.background_gate`)."""
    directory = notices_directory(data_root)
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return []
    found: list[WaitingLiveServer] = []
    for name in sorted(names, key=lambda n: (len(n), n)):
        match = _NOTICE_NAME.match(name)
        if match is None:
            continue
        notice = _posted_notice(directory / name, int(match.group(1)))
        if notice is not None:
            found.append(notice)
    return found


def _same_file(path: Path, fd: int) -> bool:
    try:
        named, held = os.stat(path), os.fstat(fd)
    except OSError:
        return False
    return (named.st_dev, named.st_ino) == (held.st_dev, held.st_ino)


def _sweep_stale(directory: Path, own_pid: int) -> None:
    """Remove notices whose owners have exited. Best-effort: a file left
    behind counts for nothing anyway (module docstring)."""
    with contextlib.suppress(OSError):
        names = os.listdir(directory)
        for name in names:
            match = _NOTICE_NAME.match(name)
            if match is None or int(match.group(1)) == own_pid:
                continue
            path = directory / name
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                # Shared, so a reader probing the same file at this instant
                # is never told it is held.
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                if _same_file(path, fd):
                    path.unlink()
            except OSError:
                pass  # held by a live owner, or already gone
            finally:
                os.close(fd)


class LiveServerNotice:
    """This process's notice. `post()` and `withdraw()` are idempotent and
    safe from any thread; `posted_for()` says how long the current notice
    has been up, by `clock`."""

    def __init__(
        self,
        data_root: Path,
        *,
        role: str,
        command: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._data_root = data_root
        self._role = role
        self._command = command
        self._clock = clock
        self._guard = threading.Lock()
        self._fd: int | None = None
        self._path: Path | None = None
        self._posted_at: float | None = None
        self._since: str | None = None

    @property
    def posted(self) -> bool:
        with self._guard:
            return self._fd is not None

    @property
    def since(self) -> str | None:
        """When the current notice was posted, ISO 8601 UTC."""
        with self._guard:
            return self._since

    def posted_for(self) -> float | None:
        """Seconds since the current notice was posted; `None` when none is."""
        with self._guard:
            if self._posted_at is None:
                return None
            return max(0.0, self._clock() - self._posted_at)

    def post(self) -> None:
        """Post the notice, if it is not up already. Raises what the
        filesystem raises; the caller decides whether that matters."""
        with self._guard:
            if self._fd is not None:
                return
            directory = notices_directory(self._data_root)
            directory.mkdir(parents=True, exist_ok=True)
            pid = os.getpid()
            _sweep_stale(directory, pid)
            since = datetime.now(UTC).isoformat(timespec="seconds")
            document = {"pid": pid, "role": self._role, "command": self._command, "since": since}
            fd, temporary = tempfile.mkstemp(dir=directory, prefix=f".{pid}-", suffix=".tmp")
            final = directory / f"{pid}.json"
            try:
                os.write(fd, json.dumps(document).encode("utf-8"))
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.replace(temporary, final)
            except BaseException:
                os.close(fd)
                Path(temporary).unlink(missing_ok=True)
                raise
            self._fd, self._path = fd, final
            self._posted_at, self._since = self._clock(), since

    def withdraw(self) -> None:
        """Take the notice down, if it is up. Never raises: the file goes
        first, then the lock, so a reader never sees a notice without its
        lock that its owner still means."""
        with self._guard:
            fd, path = self._fd, self._path
            self._fd = self._path = None
            self._posted_at = self._since = None
        if fd is None:
            return
        if path is not None and _same_file(path, fd):
            with contextlib.suppress(OSError):
                path.unlink()
        with contextlib.suppress(OSError):
            os.close(fd)


__all__ = [
    "NOTICES_SUBDIR",
    "LiveServerNotice",
    "LiveServerNoticePathError",
    "WaitingLiveServer",
    "notices_directory",
    "waiting_live_servers",
]
