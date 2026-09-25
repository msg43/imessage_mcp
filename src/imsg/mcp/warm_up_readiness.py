"""Where an operator reads whether the public MCP surface is warm.

The public transport has no unauthenticated path — by design (SPEC §10.4
hard requirement 4: every request, `initialize` and `tools/list`
included, carries a valid bearer token). So the obvious readiness probe
— a `/healthz` anyone can `curl` — is exactly the thing this surface
must not grow: it would be the one route past the only access control
the project has.

Readiness is therefore reported two ways that already exist, neither of
which touches the transport:

1. **stderr.** `imsg mcp public` logs `warm-up started: …` and `warm-up
   done: … ready in N s` (or `warm-up FAILED …`) as they happen. Under
   launchd that is `<data_root>/logs/imsgindex-mcp-public.err.log`, so
   `tail -f` on the host answers "is it up yet?" live.
2. **This file, and `imsg status`.** The server publishes its warm-up
   phase to `<data_root>/run/mcp-public-warm-up.json` on every phase and
   step change (`BackgroundWarmUp(on_status=…)`); `imsg status` reads it
   back. That makes readiness scriptable without opening a port — the
   reader needs filesystem access to the encrypted volume, which is a
   strictly higher bar than an HTTP request from anywhere.

**Liveness, not just the last thing written.** The agent is `KeepAlive`;
launchd relaunches it after a crash and every relaunch is a cold load. A
file that still says `ready` from the process that died five minutes ago
would be worse than no file at all, so the writer records its pid and
the reader checks that pid is still running before believing the phase.
The check is `os.kill(pid, 0)`, which cannot distinguish a live
`imsg mcp public` from an unrelated process that later reused the pid —
so the reader's answer is "the process that published this is still
alive", not "an MCP server is serving". Over the ~4 million pid space of
a machine that reboots rarely, that is the right trade against the
alternative (a lock file, which adds a failure mode of its own).

**What a refused load is waiting on (2026-09-25).** While the phase is
`memory_busy` the file also carries how often the load is tried
(`memory_retry_seconds`) and every other process's memory reservation the
last check saw (`waiting_on`: pid, command name, role, whether it is a
live server or background work, what it was admitted for, holds and has
not taken up, and whether that counted against the load). Command names
are the fixed `imsg <subcommand>` strings; no arguments, paths or error
text.

Nothing here is on the request path: a readiness file that cannot be
written costs the operator a status field and costs the warm-up nothing
(`BackgroundWarmUp._publish` swallows what the publisher raises).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imsg.retrieval.background_warm_up import WarmUpStatus

RUN_SUBDIR = "run"
"""`<data_root>/run` — the same directory the dedicated Postgres instance
puts its unix sockets in (`imsg.agents.plists`), i.e. this build's
existing home for "state about processes that are running right now"."""

READINESS_FILENAME = "mcp-public-warm-up.json"

NOT_RUNNING = "not_running"
"""No live process has published a phase: either none ever did, or the
one that did is gone."""

UNREADABLE = "unreadable"
"""A file is there but says nothing usable — reported as its own state
rather than folded into `not_running`, because the two call for different
operator actions and an empty read is a fact about the read, not the
world."""


def readiness_path(data_root: Path) -> Path:
    return data_root / RUN_SUBDIR / READINESS_FILENAME


@dataclass(frozen=True, slots=True)
class WarmUpReadinessReport:
    """What `imsg status` shows. `state` is a warm-up phase
    (`warming`/`ready`/`failed`) when a live process published it, and
    `NOT_RUNNING`/`UNREADABLE` otherwise — so a stale `ready` can never
    be mistaken for a warm server."""

    state: str
    detail: str
    pid: int | None = None
    failure: str | None = None
    waiting_on: tuple[str, ...] = ()
    """While `memory_busy`: one line per reservation the last check saw
    (`describe_waiting_on`)."""


class WarmUpReadinessFile:
    """Publishes a warm-up's phase to `path`, atomically.

    Atomically because `imsg status` reads the same file with no lock:
    the write goes to a temporary file in the same directory and is
    renamed over the target, so a reader sees either the previous
    complete document or the new one, never half of either.

    Deliberately not `fsync`ed. The rename is what readers on this
    machine need, and `fsync` would only add durability across a power
    loss — which cannot matter for a file whose entire content is "this
    pid is warming", since no pid survives one. It is also the slow part
    of the write, and `BackgroundWarmUp` calls this while holding the
    lock every waiting tool call takes.
    """

    def __init__(self, path: Path, *, pid: int | None = None) -> None:
        self._path = path
        self._pid = os.getpid() if pid is None else pid

    @property
    def path(self) -> Path:
        return self._path

    def publish(self, status: WarmUpStatus) -> None:
        """Write `status`. Raises what the filesystem raises — the caller
        (`BackgroundWarmUp._publish`) is the one that decides a readiness
        write must never change whether the models load."""
        document = {
            "pid": self._pid,
            "phase": status.phase.value,
            "loading": status.loading,
            "steps_done": status.steps_done,
            "steps_total": status.steps_total,
            "elapsed_seconds": round(status.elapsed_seconds, 1),
            "seconds_remaining": (
                None if status.settled else max(1, math.ceil(status.seconds_remaining))
            ),
            "failure": status.failure,
            "memory_detail": status.detail,
            "memory_retry_seconds": (
                status.retry_seconds if status.phase.value == "memory_busy" else None
            ),
            "waiting_on": status.admission.waiting_on() if status.admission is not None else [],
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            os.replace(temporary, self._path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _is_running(pid: int) -> bool:
    """Whether `pid` names a live process. A `PermissionError` means it
    exists and belongs to someone else — still running, which is the
    question being asked."""
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


def read_warm_up_readiness(data_root: Path) -> WarmUpReadinessReport:
    """The public surface's warm-up as `imsg status` reports it."""
    path = readiness_path(data_root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return WarmUpReadinessReport(
            state=NOT_RUNNING,
            detail="no 'imsg mcp public' process has published a warm-up",
        )
    except OSError as exc:
        return WarmUpReadinessReport(
            state=UNREADABLE, detail=f"readiness file could not be read: {type(exc).__name__}"
        )

    try:
        document = json.loads(raw)
    except ValueError:
        return WarmUpReadinessReport(state=UNREADABLE, detail="readiness file is not valid JSON")
    if not isinstance(document, dict):
        return WarmUpReadinessReport(
            state=UNREADABLE, detail="readiness file is not a JSON object"
        )

    phase = document.get("phase")
    pid = document.get("pid")
    if not isinstance(phase, str) or not isinstance(pid, int):
        return WarmUpReadinessReport(
            state=UNREADABLE, detail="readiness file has no phase/pid to report"
        )

    failure = document.get("failure")
    failure = failure if isinstance(failure, str) else None
    updated_at = document.get("updated_at")
    when = f" at {updated_at}" if isinstance(updated_at, str) else ""
    age = _seconds_since(updated_at)

    if not _is_running(pid):
        return WarmUpReadinessReport(
            state=NOT_RUNNING,
            detail=f"process {pid} is gone; its last published phase was '{phase}'{when}",
            pid=None,
            failure=failure,
        )

    rows = document.get("waiting_on")
    waiting_on = tuple(
        line
        for row in (rows if isinstance(rows, list) else [])
        if (line := describe_waiting_on(row)) is not None
    )
    return WarmUpReadinessReport(
        state=phase,
        detail=_describe(document, phase, failure, age),
        pid=pid,
        failure=failure,
        waiting_on=waiting_on if phase == "memory_busy" else (),
    )


def _gib(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value / 2**30:.1f} GiB"
    return None


def describe_waiting_on(row: object) -> str | None:
    """One `waiting_on` entry as a line: a reservation, or a live server
    another load waits behind. `None` for an entry that is not one."""
    if not isinstance(row, dict):
        return None
    pid, command = row.get("pid"), row.get("command")
    if not isinstance(pid, int) or not isinstance(command, str):
        return None
    kind = "live server" if row.get("live") is True else "background"
    since = row.get("waiting_since")
    if isinstance(since, str):
        return f"{command} pid {pid} ({kind}), waiting for memory since {since}"
    promised, reserved = _gib(row.get("promised_bytes")), _gib(row.get("reserved_bytes"))
    held = _gib(row.get("held_bytes")) or "an unreadable footprint"
    role = row.get("role")
    what = f"{kind}, {role}" if isinstance(role, str) else kind
    counted = (
        "counted against this load"
        if row.get("counted") is True
        else "not counted: background work gives way"
    )
    return (
        f"{command} pid {pid} ({what}): admitted for {reserved or '?'}, holds {held}, "
        f"{promised or '?'} not taken up yet, {counted}"
    )


def _seconds_since(updated_at: object) -> float:
    """How long ago the file was written, or 0 if that cannot be told.

    The file is rewritten only when the phase or the running step
    changes, so a long step leaves its estimate sitting there — `about
    91 s remaining` would still read `about 91 s remaining` a minute
    later, which is the kind of number that looks precise and is not.
    Ageing it here costs one subtraction and makes the estimate tick
    down the way a reader assumes it does. Wall clock rather than a
    monotonic clock because the writer is another process; for a figure
    already labelled "about", clock skew of a few seconds is noise.
    """
    if not isinstance(updated_at, str):
        return 0.0
    try:
        written = datetime.fromisoformat(updated_at)
    except ValueError:
        return 0.0
    if written.tzinfo is None:
        written = written.replace(tzinfo=UTC)
    return max((datetime.now(UTC) - written).total_seconds(), 0.0)


def _describe(document: dict[str, Any], phase: str, failure: str | None, age: float) -> str:
    if phase == "failed":
        return f"models failed to load ({failure or 'cause unknown'}); restart the agent"
    if phase == "unloaded":
        return (
            "models unloaded (after sitting idle, mcp.public.idle_unload_seconds, or for "
            "memory pressure, memory.public_server_release_at); the next tool call reloads "
            "them and is answered WARMING_UP until they are warm"
        )
    if phase == "memory_busy":
        detail = document.get("memory_detail")
        why = f": {detail}" if isinstance(detail, str) and detail else ""
        retry = document.get("memory_retry_seconds")
        every = (
            f" every {retry:g} s"
            if isinstance(retry, int | float) and not isinstance(retry, bool)
            else ""
        )
        return (
            f"models not loaded — the host did not have the memory for them{why}. Tool "
            f"calls answer WARMING_UP (host memory busy); the server retries by itself{every}"
        )
    if phase == "ready":
        elapsed = document.get("elapsed_seconds")
        took = f" in {elapsed} s" if isinstance(elapsed, int | float) else ""
        return f"models warm{took}; queries are answered at full speed"
    loading = document.get("loading")
    remaining = document.get("seconds_remaining")
    done, total = document.get("steps_done"), document.get("steps_total")
    what = f"loading the {loading}" if isinstance(loading, str) else "starting to load"
    progress = f" ({done}/{total} done)" if isinstance(done, int) and isinstance(total, int) else ""
    left = (
        f", about {max(1, math.ceil(remaining - age))} s remaining"
        if isinstance(remaining, int)
        else ""
    )
    return f"{what}{progress}{left} — tool calls answer WARMING_UP until this finishes"


__all__ = [
    "NOT_RUNNING",
    "READINESS_FILENAME",
    "RUN_SUBDIR",
    "UNREADABLE",
    "WarmUpReadinessFile",
    "WarmUpReadinessReport",
    "describe_waiting_on",
    "read_warm_up_readiness",
    "readiness_path",
]
