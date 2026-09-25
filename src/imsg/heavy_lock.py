"""One model-heavy command at a time on a host.

Why this exists (2026-09-24): a 64 GiB index host ran out of memory and
hung. At the crash, `imsg embed` (24.9 GiB resident: the text embedder
plus the PE-Core image embedder) and the 15-minute scheduled `imsg sync`
(22.8 GiB: the boundary LLM plus the same two embedders) were running at
the same time. Every one of those models is loaded by a separate
process, so nothing inside a process can share or bound them; only
serializing the processes can.

**The lock.** An exclusive `fcntl.flock` on
`<data_root>/run/heavy-models.lock`. `flock` belongs to the open file,
so the kernel drops it the moment the holding process exits, however it
exits: a normal return, an exception, `SIGKILL`, or the machine
crashing. There is no stale-lock cleanup to get wrong. The file is never
deleted, because unlinking a file other processes may be blocked on
lets two processes each lock a different inode.

**Holder info.** The holder writes its pid, its command and when it took
the lock into the file, as JSON. That is diagnostics only: a waiter
logs it, `imsg status` prints it. The file's content is trusted only
while the lock is actually held (`inspect_heavy_lock` checks), because a
killed holder leaves its content behind.

**Re-entrant within one process.** `flock` locks conflict between two
open files even in the same process, so a second `HeavyModelLock` for
the same path in the same process would wait on itself forever. A
process-wide registry counts holders per path instead: `sync` takes the
lock in its segment step and again in its embed step, and the second
take is a no-op.

**Not inherited by child processes.** `os.open` returns a
non-inheritable descriptor (PEP 446) and `subprocess` closes descriptors
by default, so a child (`imsg-dump`, `textutil`, `file`) never keeps the
lock alive after its parent has died.

Which commands take it, and when, is decided in `imsg.cli` and
`imsg.eval.cli`: every command that loads a large model, at the point
it is about to, and never the MCP servers, which keep their models
resident and answer queries.

**Waiting commands go first (2026-09-24, QA review).** `imsg enrich`
used to hold the lock for its whole run, and a 5,000-task chunk ran from
22:36 to 05:04, so the 15-minute `imsg sync` waited hours to segment and
embed new messages. Now:

- A command that has to wait for the lock first takes a second lock,
  `<data_root>/run/heavy-models.waiting`, exclusively, holds it while it
  waits, and lets it go once it has the heavy lock. `flock`s drop with
  the process, so a waiter that is killed stops announcing itself.
- A holder created with `yields_to_waiters=True` (the enrichment worker)
  asks `waiter_present()` between units of work. When a command is
  waiting it unloads its models, releases the lock, and takes it back
  afterwards (`imsg.enrich.worker.HeavyLockHandoff`).
- Such a holder never overtakes a waiter: it takes the lock only while
  nobody is waiting, polling every `yield_poll_seconds`, and gives it
  straight back if a waiter announced itself in the instant it took it.

Only one model-heavy process ever holds the lock, as before: a hand-over
releases it before the waiting command can take it, so the two model
sets are never both loaded under it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import structlog

from imsg.errors import HeavyModelLockBusyError, HeavyModelLockError
from imsg.paths import is_contained_in, join_under_root, resolve_path

logger = structlog.get_logger(__name__)

RUN_SUBDIR = "run"
"""`<data_root>/run`: where this build already keeps state about processes
that are running now (the Postgres sockets, the public MCP warm-up file)."""

LOCK_FILENAME = "heavy-models.lock"
WAITING_FILENAME = "heavy-models.waiting"
"""Held exclusively by a command while it waits for the lock (module
docstring, "Waiting commands go first")."""

DEFAULT_YIELD_POLL_SECONDS = 0.2
"""How often a yielding command re-checks for its turn while it waits."""


def _run_file_path(data_root: Path, filename: str) -> Path:
    path = join_under_root(data_root, Path(RUN_SUBDIR) / filename)
    if not is_contained_in(path, data_root):
        raise HeavyModelLockError(
            f"the heavy-model lock path '{path}' resolves outside data_root '{data_root}'"
        )
    return resolve_path(path)


def heavy_lock_path(data_root: Path) -> Path:
    """`<data_root>/run/heavy-models.lock`, refused if it resolves outside
    `data_root` (a `run` symlink pointing elsewhere)."""
    return _run_file_path(data_root, LOCK_FILENAME)


def heavy_waiting_path(data_root: Path) -> Path:
    """`<data_root>/run/heavy-models.waiting`, with the same containment
    check."""
    return _run_file_path(data_root, WAITING_FILENAME)


@dataclass(frozen=True, slots=True)
class LockHolder:
    """What the holding process wrote into the lock file."""

    pid: int
    command: str
    acquired_at: str

    def describe(self) -> str:
        return f"pid {self.pid} ({self.command}) since {self.acquired_at}"


@dataclass(frozen=True, slots=True)
class HeavyLockStatus:
    """`held` is decided by the kernel, not by the file's content.
    `holder` is None when nobody holds the lock, or when the holder has
    not written its info yet (the moment between taking it and writing)."""

    held: bool
    holder: LockHolder | None


def _parse_holder(raw: bytes) -> LockHolder | None:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    pid, command, acquired_at = (
        document.get("pid"),
        document.get("command"),
        document.get("acquired_at"),
    )
    if not isinstance(pid, int) or not isinstance(command, str) or not isinstance(acquired_at, str):
        return None
    return LockHolder(pid=pid, command=command, acquired_at=acquired_at)


def _read_holder_fd(fd: int) -> LockHolder | None:
    try:
        return _parse_holder(os.pread(fd, 4096, 0))
    except OSError:
        return None


def inspect_heavy_lock(data_root: Path) -> HeavyLockStatus:
    """Is the lock held right now, and by whom. Takes nothing: a probe that
    finds the lock free releases it again at once."""
    path = heavy_lock_path(data_root)
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return HeavyLockStatus(held=False, holder=None)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return HeavyLockStatus(held=True, holder=_read_holder_fd(fd))
        fcntl.flock(fd, fcntl.LOCK_UN)
        return HeavyLockStatus(held=False, holder=None)
    finally:
        os.close(fd)


@dataclass
class _ProcessHold:
    fd: int
    count: int


_registry_guard = threading.Lock()
_process_holds: dict[Path, _ProcessHold] = {}
"""Paths this process holds the lock on, with how many `HeavyModelLock`
objects currently hold each. See the module docstring on re-entrancy."""


class HeavyModelLock:
    """The host-wide lock for one command. `acquire()` is idempotent, so a
    command can call it at the start of every model phase; `release()` (or
    leaving the `with` block) drops it. Waits when another process holds
    it, unless `wait=False`, which raises `HeavyModelLockBusyError`.

    `yields_to_waiters=True` makes it the lock of a command that steps
    aside for waiting commands (module docstring, "Waiting commands go
    first"): it never takes the lock while another command waits for it,
    and `waiter_present()` tells it, between units of work, when to hand
    the lock over."""

    def __init__(
        self,
        data_root: Path,
        *,
        command: str,
        wait: bool = True,
        yields_to_waiters: bool = False,
        yield_poll_seconds: float = DEFAULT_YIELD_POLL_SECONDS,
    ) -> None:
        self._path = heavy_lock_path(data_root)
        self._waiting_path = heavy_waiting_path(data_root)
        self._command = command
        self._wait = wait
        self._yields = yields_to_waiters
        self._yield_poll_seconds = yield_poll_seconds
        self._held = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._held

    @property
    def yields_to_waiters(self) -> bool:
        return self._yields

    def acquire(self, *, wait: bool | None = None) -> None:
        """Take the lock. `wait` overrides the constructor's choice for this
        call: a command started with `--no-wait` still waits to get the
        lock back after handing it over."""
        if self._held:
            return
        with _registry_guard:
            existing = _process_holds.get(self._path)
            if existing is not None:
                existing.count += 1
                self._held = True
                return
        fd = self._lock_new_fd(self._wait if wait is None else wait)
        with _registry_guard:
            _process_holds[self._path] = _ProcessHold(fd=fd, count=1)
        self._held = True

    def waiter_present(self) -> bool:
        """Whether another command is waiting for the lock right now. Takes
        nothing that a waiter would wait for: the probe is a shared,
        non-blocking lock on the waiting file, released at once."""
        room = self._open(self._waiting_path)
        try:
            try:
                fcntl.flock(room, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(room, fcntl.LOCK_UN)
            return False
        finally:
            os.close(room)

    def _open(self, path: Path) -> int:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise HeavyModelLockError(
                f"could not open the heavy-model lock '{path}': {exc.strerror or exc}"
            ) from exc

    def _lock_new_fd(self, wait: bool) -> int:
        fd = self._open(self._path)
        try:
            if self._yields:
                self._take_when_nobody_waits(fd, wait)
            else:
                self._take_announced(fd, wait)
            self._write_holder(fd)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _held_by(self, fd: int) -> str:
        holder = _read_holder_fd(fd)
        return holder.describe() if holder is not None else "a process that has not written its details yet"

    def _busy(self, held_by: str) -> HeavyModelLockBusyError:
        return HeavyModelLockBusyError(
            f"another model-heavy command holds the host-wide lock ({held_by}); "
            f"'{self._command}' did not wait because of --no-wait. Run it again "
            f"without --no-wait to queue behind it"
        )

    def _log_waiting(self, held_by: str) -> float:
        logger.info(
            "heavy_lock.waiting",
            command=self._command,
            held_by=held_by,
            lock_path=str(self._path),
        )
        return time.monotonic()

    def _log_acquired(self, started: float) -> None:
        logger.info(
            "heavy_lock.acquired",
            command=self._command,
            waited_seconds=round(time.monotonic() - started, 1),
        )

    def _take_announced(self, fd: int, wait: bool) -> None:
        """Take the lock at once if it is free. Otherwise announce this
        command in the waiting file, wait behind any command announced
        before it and then behind the holder, and leave the waiting file
        once the lock is taken."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            pass
        held_by = self._held_by(fd)
        if not wait:
            raise self._busy(held_by)
        started = self._log_waiting(held_by)
        room = self._open(self._waiting_path)
        try:
            # Both block in the kernel. A signal with a Python handler that
            # raises (Ctrl-C) ends the wait; any other interruption is
            # retried by Python itself (PEP 475). Closing `room` below
            # leaves the waiting file however this ends.
            fcntl.flock(room, fcntl.LOCK_EX)
            fcntl.flock(fd, fcntl.LOCK_EX)
        finally:
            os.close(room)
        self._log_acquired(started)

    def _take_when_nobody_waits(self, fd: int, wait: bool) -> None:
        """Take the lock only while no other command waits for it, checking
        again once it is taken: a waiter that announced itself in that
        instant gets the lock handed straight back. Polls every
        `yield_poll_seconds` while the lock is held or a command waits."""
        started: float | None = None
        while True:
            if not self.waiter_present():
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if not self.waiter_present():
                        break
                    fcntl.flock(fd, fcntl.LOCK_UN)
            if not wait:
                raise self._busy(f"{self._held_by(fd)}, or another command is waiting for it")
            if started is None:
                started = self._log_waiting(
                    f"{self._held_by(fd)}; commands waiting for it go first"
                )
            time.sleep(self._yield_poll_seconds)
        if started is not None:
            self._log_acquired(started)

    def _write_holder(self, fd: int) -> None:
        document = {
            "pid": os.getpid(),
            "command": self._command,
            "acquired_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        data = (json.dumps(document) + "\n").encode("utf-8")
        os.ftruncate(fd, 0)
        os.pwrite(fd, data, 0)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        with _registry_guard:
            hold = _process_holds.get(self._path)
            if hold is None:  # pragma: no cover - registry and flag out of step
                return
            hold.count -= 1
            if hold.count > 0:
                return
            del _process_holds[self._path]
        # Blank the holder info before unlocking, so a reader never sees a
        # released lock's details as if they described a live holder.
        with contextlib.suppress(OSError):
            os.ftruncate(hold.fd, 0)
        with contextlib.suppress(OSError):
            fcntl.flock(hold.fd, fcntl.LOCK_UN)
        os.close(hold.fd)

    def __enter__(self) -> HeavyModelLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


__all__ = [
    "DEFAULT_YIELD_POLL_SECONDS",
    "LOCK_FILENAME",
    "RUN_SUBDIR",
    "WAITING_FILENAME",
    "HeavyLockStatus",
    "HeavyModelLock",
    "LockHolder",
    "heavy_lock_path",
    "heavy_waiting_path",
    "inspect_heavy_lock",
]
