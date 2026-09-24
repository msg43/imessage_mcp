"""How much memory the host has right now, read without root.

**Why this exists (2026-09-24).** A 64 GiB index host ran out of memory
and hung. Every process that loads models decided to load on its own,
without asking whether the host had room; `imsg.memory_admission` now
asks first, `imsg.background_gate` asks again between units of work, and
the MCP servers' watchdog (`imsg.retrieval.idle_unload`) asks every few
seconds. This module is the one place that reads the answer. It decides
nothing.

**What is read, and why these.**

- **Available memory**: `vm_stat`'s free + inactive + speculative pages,
  times its page size. Free pages cost nothing to use; inactive and
  speculative pages are the ones the kernel reclaims first. Inactive
  also holds anonymous pages that must be compressed or swapped to
  reclaim, so under pressure this figure errs high. The pressure level
  is the second signal for exactly that case.
- **The kernel's pressure level**: `sysctl kern.memorystatus_vm_pressure_level`.
  The value is libdispatch's encoding: 1 normal, 2 warn, 4 critical
  (`DISPATCH_MEMORYPRESSURE_*` in `<dispatch/source.h>`; XNU's handler
  converts its internal level to it in
  `bsd/kern/kern_memorystatus_notify.c`). On macOS the handler checks
  no privilege, so it works without root. It is registered with
  `CTLFLAG_MASKED`, which is why `sysctl -a` never lists it.
- **The kernel's free percentage**: `kern.memorystatus_level`, the
  "System-wide memory free percentage" `memory_pressure` prints.
  Reported, not used in any decision.
- **Swap used**: `vm.swapusage`. Reported, not used in any decision.
- **A process's footprint**: `proc_pid_rusage(RUSAGE_INFO_V4)`'s
  `ri_phys_footprint`, the number the kernel's own jetsam and panic
  reports call footprint. Readable without root for processes of the
  same user, which every `imsg` process on a host is. `ps`'s RSS is no
  substitute: it leaves out MLX's GPU buffers. Checked on an M2 Ultra
  (2026-09-24): a process holding a 2 GiB MLX array had an RSS of 33 MB
  and a footprint of 2.16 GB.

Everything but `vm_stat` is read through `ctypes`, so the MCP servers'
watchdog can ask for the pressure level every few seconds without
forking a process that holds 16 GiB of models.

Tests never read the machine they run on: `tests/conftest.py` replaces
:func:`default_probe` and :func:`list_model_processes` for every test,
and the tests of this module call the real functions by name.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import re
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any, Protocol

from imsg.errors import ImsgError

GIB = 2**30


def format_gib(n_bytes: int) -> str:
    return f"{n_bytes / GIB:.1f} GiB"


class HostMemoryUnavailableError(ImsgError):
    """The host's memory could not be measured. Every caller that has to
    decide something treats this as "no": a load is refused, never
    admitted on a guess (`imsg.memory_admission`)."""


class PressureLevel(StrEnum):
    NORMAL = "normal"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    def at_least(self, other: PressureLevel) -> bool:
        return self.severity >= other.severity


_SEVERITY: Mapping[PressureLevel, int] = {
    PressureLevel.NORMAL: 0,
    PressureLevel.WARN: 1,
    PressureLevel.CRITICAL: 2,
}

KERNEL_PRESSURE_LEVELS: Mapping[int, PressureLevel] = {
    1: PressureLevel.NORMAL,
    2: PressureLevel.WARN,
    4: PressureLevel.CRITICAL,
}
"""`kern.memorystatus_vm_pressure_level` values (module docstring)."""


@dataclass(frozen=True, slots=True)
class HostMemory:
    total_bytes: int
    free_bytes: int
    inactive_bytes: int
    speculative_bytes: int
    pressure: PressureLevel
    kernel_free_percent: int | None = None
    swap_used_bytes: int | None = None

    @property
    def available_bytes(self) -> int:
        """Free + inactive + speculative (module docstring)."""
        return self.free_bytes + self.inactive_bytes + self.speculative_bytes

    def describe(self) -> str:
        return (
            f"{format_gib(self.available_bytes)} available (free + inactive + speculative) "
            f"of {format_gib(self.total_bytes)}, pressure {self.pressure.value}"
        )


class HostMemoryProbe(Protocol):
    """What every caller needs from the host. `read` is the full picture
    (one `vm_stat` run); `pressure` is the cheap part alone."""

    def read(self) -> HostMemory: ...

    def pressure(self) -> PressureLevel: ...


# --------------------------------------------------------------------------
# vm_stat
# --------------------------------------------------------------------------

VM_STAT = "/usr/bin/vm_stat"

_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_LINE_RE = re.compile(r'^\s*"?([A-Za-z][A-Za-z -]*?)"?:\s+(\d+)\.?\s*$')


@dataclass(frozen=True, slots=True)
class VmStatPages:
    page_size: int
    free: int
    inactive: int
    speculative: int


def parse_vm_stat(text: str) -> VmStatPages:
    """`vm_stat`'s page size and its free, inactive and speculative page
    counts. Raises `HostMemoryUnavailableError` when any of the four is
    missing, rather than reading a missing line as zero."""
    size = _PAGE_SIZE_RE.search(text)
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = _VM_STAT_LINE_RE.match(line)
        if match:
            counts[match.group(1).strip().lower()] = int(match.group(2))
    wanted = ("pages free", "pages inactive", "pages speculative")
    missing = [name for name in wanted if name not in counts]
    if size is None or missing:
        what = ", ".join(([] if size else ["page size"]) + missing)
        raise HostMemoryUnavailableError(f"vm_stat output has no {what}")
    return VmStatPages(
        page_size=int(size.group(1)),
        free=counts["pages free"],
        inactive=counts["pages inactive"],
        speculative=counts["pages speculative"],
    )


# --------------------------------------------------------------------------
# sysctl, through ctypes
# --------------------------------------------------------------------------


@cache
def _libc() -> Any:
    path = ctypes.util.find_library("c")
    if path is None:
        raise HostMemoryUnavailableError("the C library could not be found")
    return ctypes.CDLL(path, use_errno=True)


def sysctl_bytes(name: str, size: int) -> bytes:
    """The raw value of sysctl `name` (at most `size` bytes)."""
    try:
        sysctlbyname = _libc().sysctlbyname
    except AttributeError as exc:  # not macOS
        raise HostMemoryUnavailableError(f"sysctl {name}: sysctlbyname is not available") from exc
    buffer = ctypes.create_string_buffer(size)
    length = ctypes.c_size_t(size)
    rc = sysctlbyname(name.encode(), buffer, ctypes.byref(length), None, ctypes.c_size_t(0))
    if rc != 0:
        errno = ctypes.get_errno()
        raise HostMemoryUnavailableError(f"sysctl {name}: {os.strerror(errno)}")
    return buffer.raw[: length.value]


def sysctl_uint(name: str) -> int:
    raw = sysctl_bytes(name, 8)
    if len(raw) == 4:
        return int(struct.unpack("=I", raw)[0])
    if len(raw) == 8:
        return int(struct.unpack("=Q", raw)[0])
    raise HostMemoryUnavailableError(f"sysctl {name}: unexpected {len(raw)}-byte value")


_XSW_USAGE = struct.Struct("=QQQIi")
"""`struct xsw_usage` from `<sys/sysctl.h>`: total, avail, used (u_int64),
pagesize (u_int32), encrypted (boolean_t)."""


def swap_used_bytes() -> int:
    raw = sysctl_bytes("vm.swapusage", _XSW_USAGE.size)
    if len(raw) != _XSW_USAGE.size:
        raise HostMemoryUnavailableError(f"sysctl vm.swapusage: unexpected {len(raw)}-byte value")
    _total, _avail, used, _pagesize, _encrypted = _XSW_USAGE.unpack(raw)
    return int(used)


def kernel_pressure_level() -> PressureLevel:
    raw = sysctl_uint("kern.memorystatus_vm_pressure_level")
    level = KERNEL_PRESSURE_LEVELS.get(raw)
    if level is None:
        raise HostMemoryUnavailableError(
            f"kern.memorystatus_vm_pressure_level is {raw}, not one of the known levels "
            f"(1 normal, 2 warn, 4 critical)"
        )
    return level


class SystemHostMemoryProbe:
    """Reads this host (module docstring)."""

    def __init__(self, *, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> None:
        self._run = run

    def pressure(self) -> PressureLevel:
        return kernel_pressure_level()

    def read(self) -> HostMemory:
        try:
            proc = self._run(
                [VM_STAT], capture_output=True, text=True, check=False, timeout=10
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HostMemoryUnavailableError(f"vm_stat could not run: {exc}") from exc
        if proc.returncode != 0:
            raise HostMemoryUnavailableError(f"vm_stat exited {proc.returncode}")
        pages = parse_vm_stat(str(proc.stdout))
        free_percent: int | None = None
        swap: int | None = None
        with contextlib.suppress(HostMemoryUnavailableError):
            free_percent = sysctl_uint("kern.memorystatus_level")
        with contextlib.suppress(HostMemoryUnavailableError):
            swap = swap_used_bytes()
        return HostMemory(
            total_bytes=sysctl_uint("hw.memsize"),
            free_bytes=pages.free * pages.page_size,
            inactive_bytes=pages.inactive * pages.page_size,
            speculative_bytes=pages.speculative * pages.page_size,
            pressure=self.pressure(),
            kernel_free_percent=free_percent,
            swap_used_bytes=swap,
        )


def default_probe() -> HostMemoryProbe:
    """The probe every caller uses unless it is handed one. Called through
    the module (`host_memory.default_probe()`) so the test suite can
    replace it in one place."""
    return SystemHostMemoryProbe()


# --------------------------------------------------------------------------
# process footprints
# --------------------------------------------------------------------------


class _RusageInfoV4(ctypes.Structure):
    """`struct rusage_info_v4` from `<sys/resource.h>`, whole: the kernel
    writes every field of the requested flavour, so the buffer must be
    the full size even though only the footprint is read."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
            )
        ],
    ]


_RUSAGE_INFO_V4 = 4


@cache
def _libproc() -> Any:
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    return library


def process_footprint_bytes(pid: int) -> int | None:
    """`pid`'s physical footprint, or `None` when it cannot be read: the
    process is gone, belongs to another user (reading those needs root),
    or this is not macOS."""
    try:
        libproc = _libproc()
    except OSError:
        return None
    info = _RusageInfoV4()
    if libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint)


# --------------------------------------------------------------------------
# which processes can hold a model set
# --------------------------------------------------------------------------

MODEL_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("mcp", "local"),
    ("mcp", "public"),
    ("embed",),
    ("enrich",),
    ("segment",),
    ("sync",),
    ("eval", "run"),
    ("eval", "pool"),
)
"""The `imsg` subcommands that load models."""


@dataclass(frozen=True, slots=True)
class ModelProcess:
    pid: int
    command: str
    """`imsg <subcommand>`, without its options (a `--config` path says
    nothing about memory)."""
    footprint_bytes: int | None


def model_command_of(command_line: str) -> str | None:
    """`imsg <subcommand>` when `command_line` is an `imsg` process running
    one of `MODEL_COMMANDS`, else `None`. The `imsg` script has to be the
    program itself or the script its Python interpreter runs: an `ssh`
    client whose remote command names `imsg mcp local`, or a shell
    wrapper, holds no models of its own and is left out."""
    tokens = command_line.split()
    if not tokens:
        return None
    if os.path.basename(tokens[0]) == "imsg":
        rest = tokens[1:]
    elif os.path.basename(tokens[0]).startswith("python") and len(tokens) > 1 and (
        os.path.basename(tokens[1]) == "imsg"
    ):
        rest = tokens[2:]
    else:
        return None
    for command in MODEL_COMMANDS:
        if tuple(rest[: len(command)]) == command:
            return " ".join(("imsg", *command))
    return None


def list_model_processes(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    footprint: Callable[[int], int | None] = process_footprint_bytes,
) -> list[ModelProcess]:
    """Every running `imsg` process whose subcommand loads models, with
    its footprint. `ps` needs no root to list commands; the footprint
    comes from `process_footprint_bytes`. Empty when `ps` cannot run.
    Called through the module, like `default_probe`."""
    try:
        proc = run(
            ["/bin/ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    found: list[ModelProcess] = []
    own_pid = os.getpid()
    for line in str(proc.stdout).splitlines():
        pid_text, _, command_line = line.strip().partition(" ")
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        command = model_command_of(command_line)
        if command is None or pid == own_pid:
            continue
        found.append(ModelProcess(pid=pid, command=command, footprint_bytes=footprint(pid)))
    return found


def is_darwin() -> bool:
    return sys.platform == "darwin"


__all__ = [
    "GIB",
    "KERNEL_PRESSURE_LEVELS",
    "MODEL_COMMANDS",
    "HostMemory",
    "HostMemoryProbe",
    "HostMemoryUnavailableError",
    "ModelProcess",
    "PressureLevel",
    "SystemHostMemoryProbe",
    "VmStatPages",
    "default_probe",
    "format_gib",
    "is_darwin",
    "kernel_pressure_level",
    "list_model_processes",
    "model_command_of",
    "parse_vm_stat",
    "process_footprint_bytes",
    "swap_used_bytes",
    "sysctl_uint",
]
