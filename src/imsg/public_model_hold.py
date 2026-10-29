"""The switch that makes the public MCP server give its models back for a
while, so a scheduled job can use that memory.

**Why (2026-10-29).** The public server keeps its models loaded all day so
Gemini's searches never meet a cold server (`mcp.public.idle_unload_seconds`
0). Captioning needs about 25 GiB of the 64 GiB host and does not fit beside
them, so captions run in a nightly window (01:00-09:00, owner decision
2026-10-29): the window's job sets this hold, the public server unloads, the
captions run, and the hold ends.

**What it does.** While a hold is active, the public server unloads its
models as soon as no call is in flight, starts no reload, and answers
retrieval calls `WARMING_UP`. When the hold ends, by `imsg mcp release-models`
or because its `until` has passed, the server loads again by itself
(`imsg.retrieval.idle_unload`), through the usual memory admission: a
background job still holding the memory stops after its current unit of work
and the server loads.

**Always time-limited.** A hold must name an `until` no more than
`MAX_HOLD` away, so a window job that dies cannot leave Gemini without search.

**An unreadable hold is no hold.** Unlike the background pause, which fails
toward pausing, this file fails toward serving: a garbled hold is ignored
(and reported), because the cost of ignoring it is only that captions wait
for memory, while honoring a bad one would take search away.

The file is `<data_root>/run/public-models-hold.json`, on the encrypted
volume, written atomically. Only the public server reads it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from imsg.background_pause import clean_reason
from imsg.errors import ImsgError
from imsg.paths import is_contained_in, join_under_root, resolve_path

RUN_SUBDIR = "run"
HOLD_FILENAME = "public-models-hold.json"
MAX_HOLD = timedelta(hours=12)
"""The longest a hold may last. The nightly window is 8 hours."""


class PublicModelHoldError(ImsgError):
    """The hold file's path resolves outside `data_root`, it cannot be
    written or removed, or the requested `until` is not usable."""


def hold_file_path(data_root: Path) -> Path:
    path = join_under_root(data_root, Path(RUN_SUBDIR) / HOLD_FILENAME)
    if not is_contained_in(path, data_root):
        raise PublicModelHoldError(
            f"the model hold file '{path}' resolves outside data_root '{data_root}'"
        )
    return resolve_path(path)


@dataclass(frozen=True, slots=True)
class ModelHold:
    """One hold on the public server's models."""

    until: datetime
    reason: str | None = None
    set_at: datetime | None = None

    def active(self, now: datetime) -> bool:
        return now < self.until

    def describe(self) -> str:
        parts = [self.reason or "no reason given"]
        parts.append(f"until {self.until.astimezone().isoformat(timespec='minutes')}")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class HoldReading:
    """What the hold file says right now."""

    hold: ModelHold | None = None
    problem: str | None = None
    """Why a file that exists was ignored (unreadable, invalid)."""

    def active_hold(self, now: datetime | None = None) -> ModelHold | None:
        current = now or datetime.now(UTC)
        if self.hold is not None and self.hold.active(current):
            return self.hold
        return None


def write_hold(
    data_root: Path,
    *,
    until: datetime,
    reason: str | None,
    now: datetime | None = None,
) -> ModelHold:
    """Set (or replace) the hold, atomically. `until` must be in the future
    and no more than `MAX_HOLD` away."""
    current = now or datetime.now(UTC)
    if until.tzinfo is None:
        raise PublicModelHoldError("the hold's end time needs a time zone")
    if until <= current:
        raise PublicModelHoldError(
            f"the hold's end {until.isoformat(timespec='seconds')} is already in the past"
        )
    if until - current > MAX_HOLD:
        raise PublicModelHoldError(
            f"a hold may last at most {MAX_HOLD.total_seconds() / 3600:.0f} hours; "
            f"{until.isoformat(timespec='seconds')} is further away"
        )
    hold = ModelHold(
        until=until.astimezone(UTC),
        reason=clean_reason(reason),
        set_at=current.replace(microsecond=0),
    )
    path = hold_file_path(data_root)
    document = {
        "until": hold.until.isoformat(timespec="seconds"),
        "reason": hold.reason,
        "set_at": hold.set_at.isoformat(timespec="seconds") if hold.set_at else None,
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
        raise PublicModelHoldError(f"could not write the hold file '{path}': {exc}") from exc
    return hold


def clear_hold(data_root: Path) -> bool:
    """Remove the hold; whether there was one."""
    path = hold_file_path(data_root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PublicModelHoldError(f"could not remove the hold file '{path}': {exc}") from exc
    return True


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def read_hold(data_root: Path) -> HoldReading:
    """The hold on disk. Never raises: a path or file problem comes back as
    `problem` with no hold (module docstring: fails toward serving)."""
    try:
        path = hold_file_path(data_root)
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HoldReading()
    except (OSError, PublicModelHoldError) as exc:
        return HoldReading(problem=f"the model hold file could not be read ({exc}); ignored")
    try:
        document = json.loads(raw)
    except ValueError:
        document = None
    if not isinstance(document, dict):
        return HoldReading(problem="the model hold file is not valid JSON; ignored")
    until = _parse_time(document.get("until"))
    if until is None:
        return HoldReading(problem="the model hold file has no usable `until`; ignored")
    reason = document.get("reason")
    return HoldReading(
        hold=ModelHold(
            until=until,
            reason=clean_reason(reason) if isinstance(reason, str) else None,
            set_at=_parse_time(document.get("set_at")),
        )
    )


__all__ = [
    "HOLD_FILENAME",
    "MAX_HOLD",
    "HoldReading",
    "ModelHold",
    "PublicModelHoldError",
    "clear_hold",
    "hold_file_path",
    "read_hold",
    "write_hold",
]
