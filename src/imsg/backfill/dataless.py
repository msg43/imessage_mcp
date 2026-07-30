"""APFS dataless (iCloud-optimized) placeholder detection (SPEC §8 S5a).

Two heuristics, matching the spec's phrasing verbatim ("os.stat
blocks==0 heuristic + ls -lO dataless attr"):

1. `st_blocks == 0` while `st_size > 0` — no data blocks allocated on
   disk for a file that reports nonzero size.
2. `ls -lO <path>` includes the `dataless` file-flag token — the flags
   column macOS actually surfaces for these placeholders.

Both probes are injectable (`stat_fn`, `ls_probe`) because a real
dataless placeholder can only be produced by an actual unoptimized
iCloud Photos/Messages sync — nothing this build can fabricate in a
test sandbox (no corpus, no real Messages attachments here). Tests
exercise the *combination logic* against fake probes standing in for
real filesystem state.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

StatFn = Callable[[Path], os.stat_result]
LsProbeFn = Callable[[Path], str]


def _real_stat(path: Path) -> os.stat_result:
    return os.stat(path)


def _real_ls_probe(path: Path) -> str:
    proc = subprocess.run(
        ["ls", "-lO", str(path)], capture_output=True, text=True, check=False, timeout=5
    )
    return proc.stdout


def blocks_zero_heuristic(path: Path, *, stat_fn: StatFn = _real_stat) -> bool | None:
    """True if `st_blocks == 0` while `st_size > 0` — the strongest
    single dataless signal. `None` if the path can't be stat'd, or the
    platform doesn't report `st_blocks` at all."""
    try:
        st = stat_fn(path)
    except OSError:
        return None
    st_blocks = getattr(st, "st_blocks", None)
    if st_blocks is None:
        return None
    return bool(st_blocks == 0 and st.st_size > 0)


def ls_dataless_flag(path: Path, *, ls_probe: LsProbeFn = _real_ls_probe) -> bool | None:
    """True if `ls -lO` reports the `dataless` flag for `path`. `None`
    if the probe produced no usable output (path gone, command failed).

    Parses only the flags *field* (`ls -lO`'s 5th whitespace-separated
    column: mode, links, owner, group, flags, ...) rather than
    substring-matching the whole line — the line also contains the
    full path, and a path that happens to contain the literal word
    "dataless" (a test tmp-dir named after this very function did) must
    not be misread as the file-flags column saying so.
    """
    try:
        output = ls_probe(path)
    except (OSError, subprocess.SubprocessError):
        return None
    stripped = output.strip()
    if not stripped:
        return None
    fields = stripped.splitlines()[0].split(None, 5)
    if len(fields) < 5:
        return None
    flags_field = fields[4]
    if flags_field == "-":
        return False
    return "dataless" in flags_field.split(",")


def is_dataless(
    path: Path,
    *,
    stat_fn: StatFn = _real_stat,
    ls_probe: LsProbeFn = _real_ls_probe,
) -> bool:
    """Best-effort combination: either signal firing is enough to treat
    the file as an iCloud placeholder that needs materializing before
    it can be read for real. Both signals absent/unavailable is treated
    as "not dataless" — a file that is genuinely unreadable for some
    other reason surfaces as a read error later in the pipeline, not
    here (this function only classifies, never raises)."""
    if blocks_zero_heuristic(path, stat_fn=stat_fn):
        return True
    return bool(ls_dataless_flag(path, ls_probe=ls_probe))


__all__ = ["blocks_zero_heuristic", "is_dataless", "ls_dataless_flag"]
