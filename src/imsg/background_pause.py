"""The switch that pauses heavy background work, so another job on the
host can have its memory.

**What it pauses**: segmentation, embedding, the enrichment worker,
attachment backfill, and `imsg sync`'s segmentation and embedding steps.
A paused command does no heavy work and exits `deferred: paused`; one
already running stops after the unit of work it is on
(`imsg.background_gate`).

**What it never pauses**: the MCP servers, which keep answering queries
(and keep deciding for themselves whether the host has memory for their
models, `imsg.memory_admission`), and `imsg sync`'s light steps —
snapshot, extract, identity — so no message is lost while paused: every
new message is extracted into the database as usual, and is segmented,
embedded and made searchable after the pause ends.

**Two ways to set it; either one pauses.**

1. `imsg background pause [--reason TEXT] [--until TIME]` and
   `imsg background resume`, which write and remove
   `<data_root>/run/background-pause.json` on the encrypted volume.
2. The host-wide file `background.host_pause_file` (default
   `~/.config/imessage-index/pause-background`), for another project on
   the host that should not need this project's config, virtualenv or
   encrypted volume: create the file to pause, remove it to resume. It
   is only ever read here. Its content is optional, one `key=value` per
   line: `reason=` (shown in `imsg status`), `until=` (an ISO 8601 time,
   after which it no longer pauses) and `pid=` (it pauses only while
   that process lives, so an importer that crashes cannot leave the
   index paused for good). Anything else in it is shown as the reason.

A pause that cannot be read is treated as a pause: someone meant to set
it, and the cost of honoring a garbled pause (background work waits) is
smaller than the cost of ignoring a real one (an import the operator
protected runs out of memory). `imsg background resume` clears it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from imsg.errors import ImsgError
from imsg.memory_admission import pid_is_running
from imsg.paths import is_contained_in, join_under_root, resolve_path

RUN_SUBDIR = "run"
PAUSE_FILENAME = "background-pause.json"
MAX_REASON_CHARS = 200
IMSG_COMMAND_SOURCE = "imsg background pause"


class BackgroundPauseError(ImsgError):
    """The pause file's path resolves outside `data_root`, it cannot be
    written, or an `--until` value cannot be read."""


def pause_file_path(data_root: Path) -> Path:
    path = join_under_root(data_root, Path(RUN_SUBDIR) / PAUSE_FILENAME)
    if not is_contained_in(path, data_root):
        raise BackgroundPauseError(
            f"the pause file '{path}' resolves outside data_root '{data_root}'"
        )
    return resolve_path(path)


def clean_reason(text: str | None) -> str | None:
    """One line of at most `MAX_REASON_CHARS` printable characters, or
    `None` for nothing left."""
    if text is None:
        return None
    flattened = " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())
    if not flattened:
        return None
    if len(flattened) > MAX_REASON_CHARS:
        flattened = flattened[: MAX_REASON_CHARS - 1] + "…"
    return flattened


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([mhd])\s*$")


def parse_until(text: str, *, now: datetime | None = None) -> datetime:
    """`--until` as a time: an ISO 8601 date-time (without an offset it is
    this host's local time) or a duration from now — `90m`, `6h`, `2d`."""
    current = now or datetime.now(UTC)
    duration = _DURATION_RE.match(text)
    if duration is not None:
        amount, unit = float(duration.group(1)), duration.group(2)
        step = {"m": timedelta(minutes=1), "h": timedelta(hours=1), "d": timedelta(days=1)}[unit]
        return current + amount * step
    try:
        parsed = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise BackgroundPauseError(
            f"--until {text!r} is neither an ISO 8601 time (2026-09-26T08:00) nor a "
            f"duration (90m, 6h, 2d)"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()  # this host's local time
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class PauseRequest:
    """One reason background work is (or would be) paused."""

    source: str
    """`imsg background pause`, or `host pause file <path>`."""
    reason: str | None = None
    until: datetime | None = None
    pid: int | None = None
    set_at: datetime | None = None

    def describe(self) -> str:
        set_by = f"set by {self.source}"
        if self.set_at is not None:
            set_by += f" at {self.set_at.isoformat(timespec='seconds')}"
        parts = [self.reason or "no reason given", set_by]
        if self.until is not None:
            parts.append(f"until {self.until.isoformat(timespec='seconds')}")
        if self.pid is not None:
            parts.append(f"while process {self.pid} runs")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class PauseState:
    active: tuple[PauseRequest, ...] = ()
    """What pauses background work right now."""
    lapsed: tuple[str, ...] = ()
    """Pause requests still on disk that no longer pause anything (their
    `until` passed, or their process is gone), described."""

    @property
    def paused(self) -> bool:
        return bool(self.active)

    def describe(self) -> str:
        if not self.active:
            return "not paused"
        return "paused: " + " | ".join(request.describe() for request in self.active)


def write_pause(
    data_root: Path,
    *,
    reason: str | None,
    until: datetime | None,
    now: datetime | None = None,
) -> PauseRequest:
    """Set (or replace) the `imsg background pause` request, atomically."""
    request = PauseRequest(
        source=IMSG_COMMAND_SOURCE,
        reason=clean_reason(reason),
        until=until,
        set_at=(now or datetime.now(UTC)).replace(microsecond=0),
    )
    path = pause_file_path(data_root)
    document = {
        "reason": request.reason,
        "until": request.until.isoformat(timespec="seconds") if request.until else None,
        "set_at": request.set_at.isoformat(timespec="seconds") if request.set_at else None,
        "set_by_pid": os.getpid(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise BackgroundPauseError(f"could not write the pause file '{path}': {exc}") from exc
    return request


def clear_pause(data_root: Path) -> bool:
    """Remove the `imsg background pause` request; whether there was one."""
    path = pause_file_path(data_root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise BackgroundPauseError(f"could not remove the pause file '{path}': {exc}") from exc
    return True


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(UTC)


def _read_imsg_pause(path: Path) -> PauseRequest | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return PauseRequest(
            source=IMSG_COMMAND_SOURCE,
            reason=f"the pause file could not be read ({type(exc).__name__}); treated as paused",
        )
    try:
        document = json.loads(raw)
    except ValueError:
        document = None
    if not isinstance(document, dict):
        return PauseRequest(
            source=IMSG_COMMAND_SOURCE,
            reason="the pause file is not valid; treated as paused until "
            "`imsg background resume` clears it",
        )
    reason = document.get("reason")
    until = document.get("until")
    return PauseRequest(
        source=IMSG_COMMAND_SOURCE,
        reason=clean_reason(reason) if isinstance(reason, str) else None,
        until=_parse_time(until),
        set_at=_parse_time(document.get("set_at")),
    )


def _read_host_pause(path: Path) -> PauseRequest | None:
    source = f"host pause file {path}"
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except FileNotFoundError:
        return None
    except IsADirectoryError:
        return PauseRequest(source=source)
    except OSError as exc:
        return PauseRequest(
            source=source,
            reason=f"present but unreadable ({type(exc).__name__}); treated as paused",
        )
    fields: dict[str, str] = {}
    loose: list[str] = []
    for line in raw.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().lower() in ("reason", "until", "pid"):
            fields[key.strip().lower()] = value.strip()
        elif line.strip():
            loose.append(line.strip())
    pid_text = fields.get("pid", "")
    set_at: datetime | None = None
    with contextlib.suppress(OSError):
        set_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0)
    return PauseRequest(
        source=source,
        reason=clean_reason(fields.get("reason") or " ".join(loose) or None),
        until=_parse_time(fields.get("until")),
        pid=int(pid_text) if pid_text.isdigit() else None,
        set_at=set_at,
    )


def read_pause_state(
    data_root: Path,
    *,
    host_pause_file: Path | None,
    now: datetime | None = None,
    is_running: Callable[[int], bool] = pid_is_running,
) -> PauseState:
    """Whether heavy background work is paused, and by what."""
    current = now or datetime.now(UTC)
    requests: list[PauseRequest] = []
    try:
        own = _read_imsg_pause(pause_file_path(data_root))
    except BackgroundPauseError as exc:
        own = PauseRequest(source=IMSG_COMMAND_SOURCE, reason=f"{exc}; treated as paused")
    if own is not None:
        requests.append(own)
    if host_pause_file is not None:
        host = _read_host_pause(host_pause_file)
        if host is not None:
            requests.append(host)

    active: list[PauseRequest] = []
    lapsed: list[str] = []
    for request in requests:
        if request.until is not None and request.until <= current:
            lapsed.append(f"{request.describe()} (expired)")
        elif request.pid is not None and not is_running(request.pid):
            lapsed.append(f"{request.describe()} (that process is gone)")
        else:
            active.append(request)
    return PauseState(active=tuple(active), lapsed=tuple(lapsed))


__all__ = [
    "IMSG_COMMAND_SOURCE",
    "PAUSE_FILENAME",
    "BackgroundPauseError",
    "PauseRequest",
    "PauseState",
    "clean_reason",
    "clear_pause",
    "parse_until",
    "pause_file_path",
    "read_pause_state",
    "write_pause",
]
