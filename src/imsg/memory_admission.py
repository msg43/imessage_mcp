"""Load a model set only when the host has room for it.

**Why (2026-09-24).** A 64 GiB index host ran out of memory and hung
with five idle MCP servers holding about 62 GiB between them, next to
enrichment and a scheduled sync. The host-wide heavy-model lock
(`imsg.heavy_lock`) stops two background commands loading at once, and
the idle unloader (`imsg.retrieval.idle_unload`) gives an idle server's
memory back, but nothing asked, before a load started, whether the host
had the memory for it. This module is that question, asked the same way
by every process that loads models: the MCP servers at startup and at
every reload after an unload (`imsg.retrieval.background_warm_up`), and
segment, embed, the enrich worker, sync's heavy steps and eval
(`imsg.background_gate`, `imsg.eval.cli`).

**The rule.** A model set may load when both hold:

- the kernel's pressure level is no worse than
  `memory.admission_max_pressure` (normal by default; no setting admits
  a load at critical), and
- the host's available memory (`imsg.host_memory`), less what other
  processes were admitted for and have not taken up yet, covers the
  role's expected footprint (`memory.footprints`) plus
  `memory.reserve_bytes` for the rest of the host.

**Fail closed.** When the host cannot be measured, or anything else in
the check goes wrong, the answer is no, with the cause in the reason. A
refused MCP server keeps answering, with a retryable `WARMING_UP` error
that says the host's memory is busy; a refused background command exits
`deferred: memory`. Neither weakens anything else: the public server's
auth gate runs before any tool call can learn the host is busy.

**Two processes admitted at once.** A load takes tens of seconds after
the check, so two processes that both check before either loads would
each be admitted for the same free memory, and five `imsg mcp local`
started by a client restart in the same second would all load. So the
check runs under a short host-wide `flock`
(`<data_root>/run/memory-admission.lock`, held for one `vm_stat` run),
and every admitted process leaves a reservation
(`<data_root>/run/memory-reservations/<pid>.json`) naming what it was
admitted for. The next check subtracts, for each other live process with
a reservation, the part it has not taken up yet: its reservation less
its current footprint, never below zero. A process removes its
reservation when it unloads its models or finishes; one whose process
has gone (killed, crashed) is swept by the next check.

**Live servers first (2026-09-25).** The public MCP server was restarted
while a scheduled `imsg sync` segmented. The sync had been admitted for
21.0 GiB about nine minutes earlier and held 11.1 GiB, so 9.9 GiB of
"promised" memory counted against the server: 29.4 GiB available less
9.9 was short of the 17.2 GiB it needs plus the 8.0 GiB reserve, though
29.4 alone covered it. The server stayed unloaded, and Gemini had no
working search for about 12 minutes, until the sync was stopped by hand.
The MCP servers (the live roles: `ModelRole.PUBLIC_SERVER`,
`ModelRole.LOCAL_SERVER`) answer people; every other load is background
work and gives way to them:

- A live server whose load is refused posts a notice
  (`imsg.live_server_notice`). While any is posted, no background load
  is admitted, and a running background command stops after the unit of
  work it is on (`imsg.background_gate`), dropping its models and its
  reservation, so the live server loads at its next check.
- A background job that cannot stop soon (a long unit of work, eval,
  which has no units, or an older build) cannot hold a live server off
  for long: once the server has waited `memory.background_yield_seconds`
  (60 s by default), what background jobs were promised and have not
  taken up stops counting against it. It still has to fit in the
  memory that is really available, with the reserve, at an admissible
  pressure level: background promises give way, the host's real memory
  never does. A server admitted past background promises keeps its
  notice up until those jobs have stopped, so none of them grows into
  the memory it was promised after the server has taken it.
- Other live servers' promises always count, for live servers too: five
  local servers started in the same second are still admitted one at a
  time.

Which live server holds which notice, and when it takes it down, is
`LiveServerAdmission`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from imsg import host_memory, mlx_runtime
from imsg.errors import ImsgError
from imsg.host_memory import (
    HostMemory,
    HostMemoryProbe,
    PressureLevel,
    format_gib,
    process_footprint_bytes,
)
from imsg.live_server_notice import LiveServerNotice, WaitingLiveServer, waiting_live_servers
from imsg.paths import is_contained_in, join_under_root, resolve_path

if TYPE_CHECKING:
    from imsg.config.schema import Config, MemoryConfig

RUN_SUBDIR = "run"
ADMISSION_LOCK_FILENAME = "memory-admission.lock"
RESERVATIONS_SUBDIR = "memory-reservations"


class ModelRole(StrEnum):
    PUBLIC_SERVER = "public_server"
    LOCAL_SERVER = "local_server"
    EMBED = "embed"
    ENRICH = "enrich"
    SEGMENT = "segment"

    @property
    def models(self) -> str:
        """What the role loads, for log lines and refusals."""
        return _ROLE_MODELS[self]

    @property
    def live(self) -> bool:
        """An MCP server, which answers people; every other role is
        background work (module docstring, "Live servers first")."""
        return self in LIVE_ROLES


_ROLE_MODELS: dict[ModelRole, str] = {
    ModelRole.PUBLIC_SERVER: "the public MCP server's models",
    ModelRole.LOCAL_SERVER: "the local MCP server's models",
    ModelRole.EMBED: "the embedding models",
    ModelRole.ENRICH: "the enrichment models",
    ModelRole.SEGMENT: "the segmentation boundary model",
}

LIVE_ROLES: frozenset[ModelRole] = frozenset({ModelRole.PUBLIC_SERVER, ModelRole.LOCAL_SERVER})


def footprint_bytes(memory: MemoryConfig, role: ModelRole) -> int:
    footprints = memory.footprints
    match role:
        case ModelRole.PUBLIC_SERVER:
            return footprints.public_server_bytes
        case ModelRole.LOCAL_SERVER:
            return footprints.local_server_bytes
        case ModelRole.EMBED:
            return footprints.embed_bytes
        case ModelRole.ENRICH:
            return footprints.enrich_bytes
        case ModelRole.SEGMENT:
            return footprints.segment_bytes


def mlx_memory_limit_bytes(memory: MemoryConfig, role: ModelRole) -> int:
    limits = memory.mlx_memory_limits
    match role:
        case ModelRole.PUBLIC_SERVER:
            return limits.public_server_bytes
        case ModelRole.LOCAL_SERVER:
            return limits.local_server_bytes
        case ModelRole.EMBED:
            return limits.embed_bytes
        case ModelRole.ENRICH:
            return limits.enrich_bytes
        case ModelRole.SEGMENT:
            return limits.segment_bytes


def configure_mlx_memory_limit(cfg: Config, role: ModelRole) -> int | None:
    """Set this process's MLX memory limit to `role`'s
    (`memory.mlx_memory_limits`), through
    `imsg.mlx_runtime.set_process_memory_limit`: applied now if MLX is
    loaded, else when the first MLX provider loads. Returns the limit, or
    `None` when nothing is set: the fake backend loads no MLX, and a
    configured 0 keeps MLX's own default."""
    if cfg.models.backend != "real":
        return None
    limit = mlx_memory_limit_bytes(cfg.memory, role)
    mlx_runtime.set_process_memory_limit(limit or None)
    return limit or None


# --------------------------------------------------------------------------
# reservations: what other processes were admitted for
# --------------------------------------------------------------------------


def pid_is_running(pid: int) -> bool:
    """Whether `pid` is alive. A `PermissionError` means it exists and
    belongs to another user, which still counts."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class Reservation:
    pid: int
    role: str
    reserved_bytes: int
    admitted_at: str
    command: str
    live: bool | None = None
    """Whether a live MCP server holds it. `None` in a file written before
    this field existed (2026-09-25): `is_live` then goes by the role, so
    such a file counts against everyone, as it did when it was written."""

    @property
    def is_live(self) -> bool:
        if self.live is not None:
            return self.live
        return self.role in _LIVE_ROLE_VALUES


_LIVE_ROLE_VALUES = frozenset(role.value for role in LIVE_ROLES)


@dataclass(frozen=True, slots=True)
class OtherReservation:
    """Another running process's reservation, as one check saw it."""

    reservation: Reservation
    held_bytes: int | None
    """Its footprint at the check; `None` when it could not be read."""

    @property
    def promised_bytes(self) -> int:
        """What it was admitted for and does not hold yet."""
        return max(0, self.reservation.reserved_bytes - (self.held_bytes or 0))

    def describe(self) -> str:
        r = self.reservation
        kind = "live server" if r.is_live else "background"
        return f"{r.command} pid {r.pid}, {kind}: {format_gib(self.promised_bytes)}"

    def as_dict(self, *, counted: bool) -> dict[str, object]:
        """For the public server's warm-up file and `imsg status`: numbers,
        the role and the command name, which never carries arguments."""
        r = self.reservation
        return {
            "pid": r.pid,
            "command": r.command,
            "role": r.role,
            "live": r.is_live,
            "reserved_bytes": r.reserved_bytes,
            "held_bytes": self.held_bytes,
            "promised_bytes": self.promised_bytes,
            "counted": counted,
            "admitted_at": r.admitted_at,
        }


class AdmissionPathError(ImsgError):
    """The admission lock or the reservations directory resolves outside
    `data_root` (a `run` symlink pointing elsewhere)."""


def _run_path(data_root: Path, name: str) -> Path:
    path = join_under_root(data_root, Path(RUN_SUBDIR) / name)
    if not is_contained_in(path, data_root):
        raise AdmissionPathError(f"'{path}' resolves outside data_root '{data_root}'")
    return resolve_path(path)


class ReservationBook:
    """The reservations under `<data_root>/run` (module docstring)."""

    def __init__(
        self,
        data_root: Path,
        *,
        is_running: Callable[[int], bool] = pid_is_running,
        footprint: Callable[[int], int | None] = process_footprint_bytes,
    ) -> None:
        self._data_root = data_root
        self._is_running = is_running
        self._footprint = footprint

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def directory(self) -> Path:
        return _run_path(self._data_root, RESERVATIONS_SUBDIR)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the host-wide admission lock. Blocking: every holder keeps
        it for one measurement, well under a second."""
        path = _run_path(self._data_root, ADMISSION_LOCK_FILENAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def reservations(self) -> list[Reservation]:
        """Every readable reservation file, live or not. Read without the
        lock: a file is replaced by rename, so it is never half-written."""
        found: list[Reservation] = []
        directory = self.directory
        if not directory.is_dir():
            return found
        for path in sorted(directory.glob("*.json")):
            reservation = _read_reservation(path)
            if reservation is not None:
                found.append(reservation)
        return found

    def running_reservations(self) -> list[Reservation]:
        """The reservations whose process is still running. Read without
        the lock and sweeps nothing, so it is safe from any thread."""
        return [r for r in self.reservations() if self._is_running(r.pid)]

    def others(self, own_pid: int) -> list[OtherReservation]:
        """Every live process other than `own_pid` with a reservation, and
        how much of it that process holds now; sweeps the files of
        processes that are gone. Call it under `locked()`."""
        found: list[OtherReservation] = []
        directory = self.directory
        if not directory.is_dir():
            return found
        for path in sorted(directory.glob("*.json")):
            reservation = _read_reservation(path)
            if reservation is None:
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            if reservation.pid == own_pid:
                continue
            if not self._is_running(reservation.pid):
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            found.append(OtherReservation(reservation, self._footprint(reservation.pid)))
        return found

    def promised_to_others(self, own_pid: int) -> int:
        """What live processes other than `own_pid` were admitted for and
        do not hold yet (`others`). Call it under `locked()`."""
        return sum(other.promised_bytes for other in self.others(own_pid))

    def write(self, reservation: Reservation) -> None:
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{reservation.pid}.json"
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pid": reservation.pid,
                        "role": reservation.role,
                        "reserved_bytes": reservation.reserved_bytes,
                        "admitted_at": reservation.admitted_at,
                        "command": reservation.command,
                        "live": reservation.is_live,
                    },
                    handle,
                )
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def remove(self, pid: int) -> None:
        with contextlib.suppress(OSError, ImsgError):
            (self.directory / f"{pid}.json").unlink(missing_ok=True)


def _read_reservation(path: Path) -> Reservation | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    pid, role, reserved = document.get("pid"), document.get("role"), document.get("reserved_bytes")
    admitted_at, command = document.get("admitted_at"), document.get("command")
    live = document.get("live")
    if not (
        isinstance(pid, int)
        and isinstance(role, str)
        and isinstance(reserved, int)
        and isinstance(admitted_at, str)
        and isinstance(command, str)
        and (live is None or isinstance(live, bool))
    ):
        return None
    return Reservation(pid, role, reserved, admitted_at, command, live)


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------


class RefusalCause(StrEnum):
    PRESSURE = "pressure"
    MEMORY = "memory"
    LIVE_SERVER_WAITING = "live_server_waiting"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    """One sentence an operator can act on, naming the numbers."""
    required_bytes: int
    reserve_bytes: int
    promised_to_others_bytes: int = 0
    memory: HostMemory | None = None
    """What the host looked like; `None` when it could not be measured."""
    cause: RefusalCause | None = None
    """Why it was refused; `None` when admitted."""
    counted: tuple[OtherReservation, ...] = ()
    """Other processes' reservations that counted against this load."""
    set_aside: tuple[OtherReservation, ...] = ()
    """Background reservations whose promises did not count, because a
    live server had waited long enough for them to give way."""
    live_servers_waiting: tuple[WaitingLiveServer, ...] = ()
    """For a background load: the live servers it was refused for."""

    def public_summary(self) -> str:
        """The decision in numbers only — what the public surface may say
        (SPEC §10.1: no filesystem paths or error text in public errors;
        `reason` can carry the text of a failed measurement)."""
        needs = (
            f"the models need {format_gib(self.required_bytes)} plus a reserve of "
            f"{format_gib(self.reserve_bytes)}"
        )
        if self.memory is None:
            return f"the host's memory could not be measured; {needs}"
        return f"{self.memory.describe()}; {needs}"

    @property
    def background_promises_set_aside_bytes(self) -> int:
        return sum(other.promised_bytes for other in self.set_aside)

    def waiting_on(self) -> list[dict[str, object]]:
        """Every other reservation this check saw, whether its promise
        counted, and any live server a background load was refused for —
        what `imsg status` shows a waiting server is waiting on."""
        rows = [other.as_dict(counted=True) for other in self.counted]
        rows += [other.as_dict(counted=False) for other in self.set_aside]
        rows += [
            {
                "pid": waiting.pid,
                "command": waiting.command,
                "role": waiting.role,
                "live": True,
                "waiting_since": waiting.since,
            }
            for waiting in self.live_servers_waiting
        ]
        return rows

    def waiting_key(self) -> tuple[object, ...]:
        """What changes when what the load waits on changes, and not when
        only the numbers move: for deciding when a refusal is worth a new
        log line (`imsg.retrieval.background_warm_up`)."""
        return (
            self.admitted,
            self.cause,
            tuple(sorted(o.reservation.pid for o in self.counted if o.promised_bytes > 0)),
            tuple(sorted(o.reservation.pid for o in self.set_aside)),
            tuple(sorted(w.pid for w in self.live_servers_waiting)),
        )


def _listed(others: Sequence[OtherReservation]) -> str:
    return "; ".join(other.describe() for other in others if other.promised_bytes > 0)


def decide(
    memory: HostMemory,
    *,
    what: str,
    required_bytes: int,
    reserve_bytes: int,
    max_pressure: PressureLevel,
    promised_to_others_bytes: int = 0,
    counted: Sequence[OtherReservation] = (),
    set_aside: Sequence[OtherReservation] = (),
    live_servers_waiting: Sequence[WaitingLiveServer] = (),
) -> AdmissionDecision:
    """The rule in the module docstring, on numbers already read.
    `counted` and `set_aside` only name, in the reason, the processes
    behind `promised_to_others_bytes` and the promises that gave way;
    `live_servers_waiting`, for a background load, is a refusal."""

    def outcome(admitted: bool, reason: str, cause: RefusalCause | None) -> AdmissionDecision:
        return AdmissionDecision(
            admitted=admitted,
            reason=reason,
            required_bytes=required_bytes,
            reserve_bytes=reserve_bytes,
            promised_to_others_bytes=promised_to_others_bytes,
            memory=memory,
            cause=cause,
            counted=tuple(counted),
            set_aside=tuple(set_aside),
            live_servers_waiting=tuple(live_servers_waiting),
        )

    if memory.pressure.severity > max_pressure.severity or memory.pressure is PressureLevel.CRITICAL:
        return outcome(
            False,
            f"host memory busy: the kernel reports {memory.pressure.value} memory pressure "
            f"and loads wait for {max_pressure.value}; {memory.describe()}",
            RefusalCause.PRESSURE,
        )
    if live_servers_waiting:
        who = "; ".join(waiting.describe() for waiting in live_servers_waiting)
        return outcome(
            False,
            f"a live MCP server is waiting to load its models ({who}); background work "
            f"loads only after it has; {memory.describe()}",
            RefusalCause.LIVE_SERVER_WAITING,
        )
    names = _listed(counted)
    promised = (
        f", {format_gib(promised_to_others_bytes)} of it promised to other model processes "
        f"still loading{f' ({names})' if names else ''}"
        if promised_to_others_bytes > 0
        else ""
    )
    aside = sum(other.promised_bytes for other in set_aside)
    set_aside_text = (
        f"; {format_gib(aside)} promised to background work not counted ({_listed(set_aside)}): "
        f"background work gives way to a live server"
        if aside > 0
        else ""
    )
    needs = (
        f"{what} need {format_gib(required_bytes)} plus a reserve of "
        f"{format_gib(reserve_bytes)} for the rest of the host"
    )
    if memory.available_bytes - promised_to_others_bytes < required_bytes + reserve_bytes:
        return outcome(
            False,
            f"host memory busy: {memory.describe()}{promised}{set_aside_text}; {needs}",
            RefusalCause.MEMORY,
        )
    return outcome(True, f"{memory.describe()}{promised}{set_aside_text}; {needs}", None)


def refused(what: str, cause: str, *, required_bytes: int, reserve_bytes: int) -> AdmissionDecision:
    return AdmissionDecision(
        admitted=False,
        reason=f"host memory could not be measured ({cause}); not loading {what} on a guess",
        required_bytes=required_bytes,
        reserve_bytes=reserve_bytes,
        cause=RefusalCause.UNMEASURED,
    )


# --------------------------------------------------------------------------
# admission for one process
# --------------------------------------------------------------------------


class MemoryAdmission:
    """The admission check for one process's model set. `check()` decides
    now and, on yes, records the reservation; `release()` drops the
    reservation once the models are unloaded or the process is done.
    Background commands wait for a yes through
    `imsg.background_gate.BackgroundGate.admit`; live servers ask through
    `LiveServerAdmission`."""

    def __init__(
        self,
        *,
        probe: HostMemoryProbe,
        book: ReservationBook,
        role: ModelRole,
        required_bytes: int,
        reserve_bytes: int,
        max_pressure: PressureLevel,
        command: str,
        live: bool | None = None,
        live_servers_waiting: Callable[[], Sequence[WaitingLiveServer]] | None = None,
    ) -> None:
        """`live` defaults to the role's (`ModelRole.live`); eval passes
        False, since it borrows the local server's footprint but is batch
        work. `live_servers_waiting` is how a background check reads the
        live servers' notices, the real directory unless a test says
        otherwise."""
        self._probe = probe
        self._book = book
        self._role = role
        self._required = required_bytes
        self._reserve = reserve_bytes
        self._max_pressure = max_pressure
        self._command = command
        self._live = role.live if live is None else live
        self._live_servers_waiting = (
            live_servers_waiting
            if live_servers_waiting is not None
            else lambda: waiting_live_servers(book.data_root)
        )

    @classmethod
    def for_role(
        cls,
        cfg: Config,
        role: ModelRole,
        *,
        command: str,
        copies: int = 1,
        probe: HostMemoryProbe | None = None,
        live: bool | None = None,
    ) -> MemoryAdmission:
        """`copies` model sets of `role` (eval pool builds one per variant)."""
        memory = cfg.memory
        return cls(
            probe=probe if probe is not None else host_memory.default_probe(),
            book=ReservationBook(cfg.paths.data_root),
            role=role,
            required_bytes=footprint_bytes(memory, role) * max(1, copies),
            reserve_bytes=memory.reserve_bytes,
            max_pressure=PressureLevel(memory.admission_max_pressure),
            command=command,
            live=live,
        )

    @property
    def role(self) -> ModelRole:
        return self._role

    @property
    def live(self) -> bool:
        return self._live

    @property
    def book(self) -> ReservationBook:
        return self._book

    @property
    def required_bytes(self) -> int:
        return self._required

    def check(
        self,
        *,
        count_background_promises: bool = True,
        while_locked: Callable[[AdmissionDecision], None] | None = None,
    ) -> AdmissionDecision:
        """Decide now. Never raises: a failure to measure is a refusal
        (module docstring, "Fail closed"). For a live load,
        `count_background_promises=False` lets background jobs' untaken
        promises give way (`LiveServerAdmission` decides when); a
        background load always counts every promise, and is refused while
        any live server's notice is posted. `while_locked` sees the
        decision before the admission lock is released, so a live server
        posts its notice before any other process's check can run; it
        can never change the decision (anything it raises is dropped)."""
        what = self._role.models
        try:
            with self._book.locked():
                memory = self._probe.read()
                others = self._book.others(os.getpid())
                waiting: Sequence[WaitingLiveServer] = ()
                if self._live:
                    counted = [
                        o
                        for o in others
                        if count_background_promises
                        or o.reservation.is_live
                        or o.promised_bytes == 0
                    ]
                    set_aside = [o for o in others if o not in counted]
                else:
                    counted, set_aside = others, []
                    waiting = self._live_servers_waiting()
                decision = decide(
                    memory,
                    what=what,
                    required_bytes=self._required,
                    reserve_bytes=self._reserve,
                    max_pressure=self._max_pressure,
                    promised_to_others_bytes=sum(o.promised_bytes for o in counted),
                    counted=counted,
                    set_aside=set_aside,
                    live_servers_waiting=waiting,
                )
                if decision.admitted:
                    self._book.write(
                        Reservation(
                            pid=os.getpid(),
                            role=self._role.value,
                            reserved_bytes=self._required,
                            admitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
                            command=self._command,
                            live=self._live,
                        )
                    )
                if while_locked is not None:
                    with contextlib.suppress(Exception):
                        while_locked(decision)
                return decision
        except Exception as exc:
            cause = f"{type(exc).__name__}: {exc}" if not isinstance(exc, ImsgError) else str(exc)
            return refused(
                what, cause, required_bytes=self._required, reserve_bytes=self._reserve
            )

    def release(self) -> None:
        """Remove this process's reservation. Idempotent; never raises."""
        self._book.remove(os.getpid())


# --------------------------------------------------------------------------
# a live server: its loads come first
# --------------------------------------------------------------------------

NOTICE_LAPSE_SECONDS = 120.0
"""A notice posted because a load was refused comes down when no load
has been tried for this long. The public server tries again by itself
every `memory.admission_retry_seconds`, so its notice never lapses while
it waits; a local server tries again only when a client calls, and a
client that has stopped asking should not hold background work back.
Two minutes is several of the retries a refused call is told to make
(`HostMemoryBusyError` names the retry interval, 15 s by default)."""


class _NoticeCause(StrEnum):
    WAITING = "waiting"
    """A load was refused; the server is trying again."""
    PAST_BACKGROUND = "past_background"
    """Admitted past background promises; up until those jobs stop."""
    RELOAD_PENDING = "reload_pending"
    """Unloaded under memory pressure; the server loads again by itself."""


class LiveServerAdmission:
    """Memory admission for a live MCP server (the public one, a local
    one): the same rule as every load, except that background work gives
    way to it (module docstring, "Live servers first").

    - `check()` is the warm-up's admission (`imsg.retrieval.
      background_warm_up`): refused, it posts the server's notice
      (`imsg.live_server_notice`), which holds background work back;
      admitted, it takes the notice down, unless background promises were
      set aside, and then leaves it up until those jobs have stopped.
    - Background promises count for the first `background_yield_seconds`
      of a wait, while background jobs stop between units of work; after
      that they do not.
    - `release()` after an idle unload or at exit: the reservation and
      the notice go. `release_for_reload()` after an unload under memory
      pressure, when the server loads again by itself: the reservation
      goes and the notice goes up, so no background load takes the memory
      during the cooldown.
    - `tick()`, from the server's watchdog every few seconds, takes the
      notice down when its reason has passed.

    Posting or withdrawing the notice never changes a decision: a
    filesystem error there is logged once and the load goes on as the
    memory allows."""

    def __init__(
        self,
        admission: MemoryAdmission,
        notice: LiveServerNotice,
        *,
        background_yield_seconds: float,
        lapse_seconds: float = NOTICE_LAPSE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if background_yield_seconds < 0:
            raise ValueError(
                f"background_yield_seconds must be >= 0, got {background_yield_seconds}"
            )
        self._admission = admission
        self._notice = notice
        self._yield_seconds = float(background_yield_seconds)
        self._lapse_seconds = float(lapse_seconds)
        self._clock = clock
        self._log = log
        self._guard = threading.Lock()
        self._cause: _NoticeCause | None = None
        self._last_refusal_at: float | None = None
        self._notice_error_logged = False

    @classmethod
    def for_role(
        cls,
        cfg: Config,
        role: ModelRole,
        *,
        command: str,
        probe: HostMemoryProbe | None = None,
        log: Callable[[str], None] | None = None,
    ) -> LiveServerAdmission:
        admission = MemoryAdmission.for_role(cfg, role, command=command, probe=probe, live=True)
        notice = LiveServerNotice(cfg.paths.data_root, role=role.value, command=command)
        return cls(
            admission,
            notice,
            background_yield_seconds=cfg.memory.background_yield_seconds,
            log=log,
        )

    @property
    def admission(self) -> MemoryAdmission:
        return self._admission

    @property
    def notice(self) -> LiveServerNotice:
        return self._notice

    @property
    def background_yield_seconds(self) -> float:
        return self._yield_seconds

    def background_promises_count(self) -> bool:
        """Whether background jobs' untaken promises still count: from the
        start of a wait until it is `background_yield_seconds` old."""
        if self._yield_seconds <= 0:
            return False
        waited = self._notice.posted_for()
        return waited is None or waited < self._yield_seconds

    def check(self) -> AdmissionDecision:
        """The warm-up's admission: decide, and post or take down the
        notice under the same admission lock (`MemoryAdmission.check`)."""
        counting = self.background_promises_count()

        def settle(decision: AdmissionDecision) -> None:
            with self._guard:
                if decision.admitted and decision.set_aside:
                    self._post(_NoticeCause.PAST_BACKGROUND)
                elif decision.admitted:
                    self._withdraw()
                else:
                    self._last_refusal_at = self._clock()
                    self._post(_NoticeCause.WAITING)

        decision = self._admission.check(count_background_promises=counting, while_locked=settle)
        if decision.admitted:
            return decision
        return dataclasses.replace(decision, reason=decision.reason + self._waiting_note(counting))

    def _waiting_note(self, counting: bool) -> str:
        if not self._notice.posted:
            return ""
        note = (
            "; background work has been asked to give way: running jobs stop after their "
            "current unit of work and none starts a load"
        )
        if counting and self._yield_seconds > 0:
            note += (
                f"; what background jobs were promised counts against this load for the "
                f"first {self._yield_seconds:g} s of the wait"
            )
        return note

    def release(self) -> None:
        """The models are unloaded for good (idle) or the process is done.
        Idempotent; never raises."""
        self._admission.release()
        with self._guard:
            self._withdraw()

    def release_for_reload(self) -> None:
        """The models were unloaded under memory pressure and the server
        loads them again by itself after its cooldown. Never raises."""
        self._admission.release()
        with self._guard:
            self._post(_NoticeCause.RELOAD_PENDING)

    def tick(self) -> None:
        """Take the notice down once its reason has passed: a refused load
        nobody has tried again for `lapse_seconds`, or background jobs a
        load was admitted past that have all stopped. Never raises."""
        with self._guard:
            cause = self._cause
            if cause is _NoticeCause.WAITING:
                last = self._last_refusal_at
                if last is not None and self._clock() - last >= self._lapse_seconds:
                    self._withdraw()
                    self._say(
                        f"no load has been tried for {self._lapse_seconds:g} s; background "
                        f"work may load again"
                    )
            elif cause is _NoticeCause.PAST_BACKGROUND and not self._background_jobs_running():
                self._withdraw()
                self._say("the background jobs this server loaded past have stopped")

    def _background_jobs_running(self) -> bool:
        own = os.getpid()
        try:
            reservations = self._admission.book.running_reservations()
        except (OSError, ImsgError):
            return True  # unknown: keep holding background work back
        return any(r.pid != own and not r.is_live for r in reservations)

    # Callers hold self._guard.

    def _post(self, cause: _NoticeCause) -> None:
        self._cause = cause
        try:
            self._notice.post()
        except (OSError, ImsgError) as exc:
            if not self._notice_error_logged:
                self._notice_error_logged = True
                self._say(
                    f"could not post the notice that holds background work back "
                    f"({type(exc).__name__}: {exc}); loads still wait for memory"
                )

    def _withdraw(self) -> None:
        self._cause = None
        self._last_refusal_at = None
        self._notice.withdraw()

    def _say(self, line: str) -> None:
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log(line)


__all__ = [
    "ADMISSION_LOCK_FILENAME",
    "LIVE_ROLES",
    "NOTICE_LAPSE_SECONDS",
    "RESERVATIONS_SUBDIR",
    "AdmissionDecision",
    "AdmissionPathError",
    "LiveServerAdmission",
    "MemoryAdmission",
    "ModelRole",
    "OtherReservation",
    "RefusalCause",
    "Reservation",
    "ReservationBook",
    "configure_mlx_memory_limit",
    "decide",
    "footprint_bytes",
    "mlx_memory_limit_bytes",
    "pid_is_running",
    "refused",
]
