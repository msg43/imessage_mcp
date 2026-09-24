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
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Iterator
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


_ROLE_MODELS: dict[ModelRole, str] = {
    ModelRole.PUBLIC_SERVER: "the public MCP server's models",
    ModelRole.LOCAL_SERVER: "the local MCP server's models",
    ModelRole.EMBED: "the embedding models",
    ModelRole.ENRICH: "the enrichment models",
    ModelRole.SEGMENT: "the segmentation boundary model",
}


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
# the decision
# --------------------------------------------------------------------------


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


def decide(
    memory: HostMemory,
    *,
    what: str,
    required_bytes: int,
    reserve_bytes: int,
    max_pressure: PressureLevel,
    promised_to_others_bytes: int = 0,
) -> AdmissionDecision:
    """The rule in the module docstring, on numbers already read."""

    def outcome(admitted: bool, reason: str) -> AdmissionDecision:
        return AdmissionDecision(
            admitted=admitted,
            reason=reason,
            required_bytes=required_bytes,
            reserve_bytes=reserve_bytes,
            promised_to_others_bytes=promised_to_others_bytes,
            memory=memory,
        )

    if memory.pressure.severity > max_pressure.severity or memory.pressure is PressureLevel.CRITICAL:
        return outcome(
            False,
            f"host memory busy: the kernel reports {memory.pressure.value} memory pressure "
            f"and loads wait for {max_pressure.value}; {memory.describe()}",
        )
    promised = (
        f", {format_gib(promised_to_others_bytes)} of it promised to other model processes "
        f"still loading"
        if promised_to_others_bytes > 0
        else ""
    )
    needs = (
        f"{what} need {format_gib(required_bytes)} plus a reserve of "
        f"{format_gib(reserve_bytes)} for the rest of the host"
    )
    if memory.available_bytes - promised_to_others_bytes < required_bytes + reserve_bytes:
        return outcome(False, f"host memory busy: {memory.describe()}{promised}; {needs}")
    return outcome(True, f"{memory.describe()}{promised}; {needs}")


def refused(what: str, cause: str, *, required_bytes: int, reserve_bytes: int) -> AdmissionDecision:
    return AdmissionDecision(
        admitted=False,
        reason=f"host memory could not be measured ({cause}); not loading {what} on a guess",
        required_bytes=required_bytes,
        reserve_bytes=reserve_bytes,
    )


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

    def promised_to_others(self, own_pid: int) -> int:
        """What live processes other than `own_pid` were admitted for and
        do not hold yet; sweeps the files of processes that are gone.
        Call it under `locked()`."""
        total = 0
        directory = self.directory
        if not directory.is_dir():
            return 0
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
            held = self._footprint(reservation.pid) or 0
            total += max(0, reservation.reserved_bytes - held)
        return total

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
    if not (
        isinstance(pid, int)
        and isinstance(role, str)
        and isinstance(reserved, int)
        and isinstance(admitted_at, str)
        and isinstance(command, str)
    ):
        return None
    return Reservation(pid, role, reserved, admitted_at, command)


# --------------------------------------------------------------------------
# admission for one process
# --------------------------------------------------------------------------


class MemoryAdmission:
    """The admission check for one process's model set. `check()` decides
    now and, on yes, records the reservation; `release()` drops the
    reservation once the models are unloaded or the process is done.
    Background commands wait for a yes through
    `imsg.background_gate.BackgroundGate.admit`."""

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
    ) -> None:
        self._probe = probe
        self._book = book
        self._role = role
        self._required = required_bytes
        self._reserve = reserve_bytes
        self._max_pressure = max_pressure
        self._command = command

    @classmethod
    def for_role(
        cls,
        cfg: Config,
        role: ModelRole,
        *,
        command: str,
        copies: int = 1,
        probe: HostMemoryProbe | None = None,
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
        )

    @property
    def role(self) -> ModelRole:
        return self._role

    @property
    def required_bytes(self) -> int:
        return self._required

    def check(self) -> AdmissionDecision:
        """Decide now. Never raises: a failure to measure is a refusal
        (module docstring, "Fail closed")."""
        what = self._role.models
        try:
            with self._book.locked():
                memory = self._probe.read()
                promised = self._book.promised_to_others(os.getpid())
                decision = decide(
                    memory,
                    what=what,
                    required_bytes=self._required,
                    reserve_bytes=self._reserve,
                    max_pressure=self._max_pressure,
                    promised_to_others_bytes=promised,
                )
                if decision.admitted:
                    self._book.write(
                        Reservation(
                            pid=os.getpid(),
                            role=self._role.value,
                            reserved_bytes=self._required,
                            admitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
                            command=self._command,
                        )
                    )
                return decision
        except Exception as exc:
            cause = f"{type(exc).__name__}: {exc}" if not isinstance(exc, ImsgError) else str(exc)
            return refused(
                what, cause, required_bytes=self._required, reserve_bytes=self._reserve
            )

    def release(self) -> None:
        """Remove this process's reservation. Idempotent; never raises."""
        self._book.remove(os.getpid())


__all__ = [
    "ADMISSION_LOCK_FILENAME",
    "RESERVATIONS_SUBDIR",
    "AdmissionDecision",
    "AdmissionPathError",
    "MemoryAdmission",
    "ModelRole",
    "Reservation",
    "ReservationBook",
    "configure_mlx_memory_limit",
    "decide",
    "footprint_bytes",
    "mlx_memory_limit_bytes",
    "pid_is_running",
    "refused",
]
