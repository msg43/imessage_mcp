"""The host-memory probe reads real numbers without root (`imsg.host_memory`).

Two kinds of test. Parsing and mapping run everywhere on fixed input. The
live ones run on macOS against the machine running the tests and compare
the probe with the tools an operator would use (`vm_stat`, `sysctl`), so a
wrong `ctypes` layout or a changed tool output fails here rather than in a
load decision on the index host.

`tests/conftest.py` replaces `imsg.host_memory.default_probe` and
`list_model_processes` on the module for every test; the names imported
below are bound at import time and are the real functions.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from imsg.host_memory import (
    GIB,
    KERNEL_PRESSURE_LEVELS,
    HostMemory,
    HostMemoryUnavailableError,
    PressureLevel,
    SystemHostMemoryProbe,
    kernel_pressure_level,
    list_model_processes,
    model_command_of,
    parse_vm_stat,
    process_footprint_bytes,
    sysctl_uint,
)

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="reads a macOS host")

VM_STAT_SAMPLE = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  1356383.
Pages active:                                 895252.
Pages inactive:                               839920.
Pages speculative:                            192121.
Pages throttled:                                   0.
Pages wired down:                             189386.
Pages purgeable:                               26219.
"Translation faults":                       39118507.
Pages copy-on-write:                         3592943.
File-backed pages:                           1022568.
Anonymous pages:                              904725.
Pages stored in compressor:                  1354639.
Pages occupied by compressor:                 672858.
Swapouts:                                    1795586.
"""


def test_vm_stat_free_inactive_and_speculative_pages_are_read_with_the_page_size() -> None:
    pages = parse_vm_stat(VM_STAT_SAMPLE)
    assert (pages.page_size, pages.free, pages.inactive, pages.speculative) == (
        16384,
        1356383,
        839920,
        192121,
    )


@pytest.mark.parametrize("missing", ["Pages free", "Pages inactive", "Pages speculative"])
def test_vm_stat_output_missing_a_count_is_refused_not_read_as_zero(missing: str) -> None:
    text = "\n".join(line for line in VM_STAT_SAMPLE.splitlines() if not line.startswith(missing))
    with pytest.raises(HostMemoryUnavailableError, match=missing.lower()):
        parse_vm_stat(text)


def test_vm_stat_output_without_a_page_size_is_refused() -> None:
    with pytest.raises(HostMemoryUnavailableError, match="page size"):
        parse_vm_stat(VM_STAT_SAMPLE.replace("page size of 16384 bytes", "no size here"))


def test_available_memory_is_free_plus_inactive_plus_speculative() -> None:
    memory = HostMemory(
        total_bytes=64 * GIB,
        free_bytes=20 * GIB,
        inactive_bytes=12 * GIB,
        speculative_bytes=3 * GIB,
        pressure=PressureLevel.NORMAL,
    )
    assert memory.available_bytes == 35 * GIB
    assert memory.describe() == (
        "35.0 GiB available (free + inactive + speculative) of 64.0 GiB, pressure normal"
    )


def test_the_kernel_levels_are_libdispatchs_encoding() -> None:
    assert dict(KERNEL_PRESSURE_LEVELS) == {
        1: PressureLevel.NORMAL,
        2: PressureLevel.WARN,
        4: PressureLevel.CRITICAL,
    }
    assert PressureLevel.CRITICAL.at_least(PressureLevel.WARN)
    assert not PressureLevel.WARN.at_least(PressureLevel.CRITICAL)


def test_an_unknown_kernel_level_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import imsg.host_memory as module

    monkeypatch.setattr(module, "sysctl_uint", lambda name: 3)
    with pytest.raises(HostMemoryUnavailableError, match="not one of the known levels"):
        kernel_pressure_level()


def test_a_vm_stat_that_cannot_run_is_an_unmeasurable_host() -> None:
    def broken(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("vm_stat")

    with pytest.raises(HostMemoryUnavailableError, match="vm_stat could not run"):
        SystemHostMemoryProbe(run=broken).read()


# --------------------------------------------------------------------------
# the live host
# --------------------------------------------------------------------------


def _sysctl_text(name: str) -> str:
    return subprocess.run(
        ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, check=True
    ).stdout.strip()


@darwin_only
def test_the_live_probe_agrees_with_sysctl_and_vm_stat() -> None:
    memory = SystemHostMemoryProbe().read()

    assert memory.total_bytes == int(_sysctl_text("hw.memsize"))
    assert memory.pressure is KERNEL_PRESSURE_LEVELS[
        int(_sysctl_text("kern.memorystatus_vm_pressure_level"))
    ]
    assert 0 < memory.available_bytes <= memory.total_bytes

    # A second, independent vm_stat reading: free memory moves between
    # the two reads, so compare within a generous tolerance.
    pages = parse_vm_stat(
        subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=True).stdout
    )
    again = (pages.free + pages.inactive + pages.speculative) * pages.page_size
    assert abs(again - memory.available_bytes) < 4 * GIB

    used = re.search(r"used = ([\d.]+)M", _sysctl_text("vm.swapusage"))
    assert used is not None and memory.swap_used_bytes is not None
    assert abs(memory.swap_used_bytes - float(used.group(1)) * 2**20) < 64 * 2**20

    assert memory.kernel_free_percent is not None
    assert 0 <= memory.kernel_free_percent <= 100


@darwin_only
def test_sysctl_reads_the_same_integers_as_the_sysctl_tool() -> None:
    assert sysctl_uint("hw.memsize") == int(_sysctl_text("hw.memsize"))
    assert sysctl_uint("vm.pagesize") == int(_sysctl_text("vm.pagesize"))


@darwin_only
def test_a_processs_footprint_is_read_without_root_and_counts_what_it_holds(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib, sys, time\n"
            "held = bytearray(256 * 2**20)\n"
            "for i in range(0, len(held), 16384): held[i] = 1\n"
            f"pathlib.Path({str(ready)!r}).write_text('x')\n"
            f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.05)\n",
        ]
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert time.monotonic() < deadline, "the child never allocated"
            time.sleep(0.05)
        footprint = process_footprint_bytes(child.pid)
        assert footprint is not None
        assert footprint >= 256 * 2**20
    finally:
        release.write_text("x")
        child.wait(timeout=30)
    # A process that has gone, and one that belongs to root, read as None.
    assert process_footprint_bytes(child.pid) is None
    assert process_footprint_bytes(1) is None


# --------------------------------------------------------------------------
# which processes can hold models
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command_line", "expected"),
    [
        ("/venv/bin/python3 /venv/bin/imsg mcp local", "imsg mcp local"),
        ("/venv/bin/python3.14 /venv/bin/imsg mcp public --config /x.yaml", "imsg mcp public"),
        ("/venv/bin/imsg enrich --config /x.yaml", "imsg enrich"),
        ("/venv/bin/python3 /venv/bin/imsg sync --config /x.yaml", "imsg sync"),
        ("/venv/bin/python3 /venv/bin/imsg eval pool --out w.yaml", "imsg eval pool"),
        ("/venv/bin/python3 /venv/bin/imsg status", None),
        ("/venv/bin/python3 /venv/bin/imsg background pause", None),
        # An ssh client whose remote command runs the server holds nothing.
        ("ssh -o BatchMode=yes -T host exec /venv/bin/imsg mcp local", None),
        ("/bin/sh -c /venv/bin/imsg guard-mount && exec postgres", None),
        ("", None),
    ],
)
def test_model_commands_are_recognised_only_in_the_imsg_process_itself(
    command_line: str, expected: str | None
) -> None:
    assert model_command_of(command_line) == expected


def test_model_processes_are_listed_with_their_footprints() -> None:
    ps_output = (
        "  101 /venv/bin/python3 /venv/bin/imsg mcp public --config /x.yaml\n"
        "  202 /venv/bin/python3 /venv/bin/imsg mcp local\n"
        "  303 /usr/sbin/sshd\n"
        "  404 ssh host /venv/bin/imsg mcp local\n"
        "  505 /venv/bin/imsg embed --config /x.yaml\n"
    )

    def fake_ps(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=ps_output, stderr="")

    footprints = {101: 17 * GIB, 202: 350 * 2**20, 505: None}
    found = list_model_processes(run=fake_ps, footprint=footprints.get)
    assert [(p.pid, p.command, p.footprint_bytes) for p in found] == [
        (101, "imsg mcp public", 17 * GIB),
        (202, "imsg mcp local", 350 * 2**20),
        (505, "imsg embed", None),
    ]
