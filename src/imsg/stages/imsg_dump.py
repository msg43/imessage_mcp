"""Subprocess client for `tools/imsg-dump` (SPEC §4.2, §8 S2).

`imsg-dump` is a GPL-3.0 Rust binary (its own directory, its own
`LICENSE`) that links the `imessage-database` crate to decode the parts
of `chat.db` that are not plain SQL columns: the `attributedBody`
typedstream body blob, an edited message's prior-version history, and
tapback (reaction) classification. The core (this module included)
never links that crate — it only ever shells out to the compiled
binary and reads NDJSON off its stdout, one object per line, arm's
length across a process boundary (SPEC §4.2, same pattern as the
`imessage-exporter` fallback).

Contract (one NDJSON object per line, per message row in scope):

    {rowid, guid, chat_guid, handle, is_from_me, date, date_edited,
     date_retracted, service, body_text, edit_history[], is_unsent,
     tapback{...}|null, attachment_rowids[], reply_to_guid}

`imsg.stages.extract` (S2) treats SQL over the snapshot as the
authoritative source for *structural* fields it can already read as
plain columns (dates, `is_unsent`/`is_edited` inferred from
`date_retracted`/`date_edited` being non-null, `thread_originator_guid`
for `reply_to_guid`, chat/attachment linkage) — see that module's
docstring for the reasoning. This client is consulted purely for the
fields that genuinely require typedstream decoding: `body_text`,
`edit_history[]` text, and `tapback` classification.

Two seams for testability, deliberately separate:

- `run_process`: how a subprocess gets started at all (default
  `subprocess.Popen`). Tests can point this at a small stand-in script
  (e.g. `python3 -c "..."`) to exercise real streaming/exit-code/stderr
  behavior without needing the compiled Rust binary.
- Line parsing (`_parse_line`) is exercised directly with hand-written
  NDJSON strings, independent of any subprocess at all.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog

from imsg.errors import ImsgDumpError

logger = structlog.get_logger(__name__)

DEFAULT_BINARY_RELPATH = Path("tools/imsg-dump/target/release/imsg-dump")
DEBUG_BINARY_RELPATH = Path("tools/imsg-dump/target/debug/imsg-dump")


def default_binary_path(repo_root: Path) -> Path:
    """The compiled `imsg-dump` binary, preferring a release build.

    Falls back to the debug build (what a plain `cargo build` produces)
    so local/dev runs work without requiring `--release`. Returns the
    release path (even if absent) when neither exists, so callers get a
    consistent "not found" error naming the expected release location.
    """
    release = repo_root / DEFAULT_BINARY_RELPATH
    if release.is_file():
        return release
    debug = repo_root / DEBUG_BINARY_RELPATH
    if debug.is_file():
        return debug
    return release


@dataclass(frozen=True, slots=True)
class EditVersion:
    text: str | None
    edited_at: str | None


@dataclass(frozen=True, slots=True)
class TapbackInfo:
    kind: str
    """One of loved|liked|disliked|laughed|emphasized|questioned|emoji|sticker
    — matches the real `imsg-dump` implementation's `tapback_kind` mapping.
    Note this is the *bare* kind ("emoji", not "emoji:<char>"); combine with
    `emoji` to get the Postgres `tapback.kind` column's documented
    `emoji:<char>` form (SPEC §7.2) — see `imsg.stages.extract._upsert_tapback`."""
    target_guid: str
    emoji: str | None = None
    """The literal emoji character, only present when `kind == "emoji"`."""
    action: str = "added"
    """"added" or "removed" — chat.db models un-reacting as a second event
    referencing the same target, not a mutation of the first. Distinct from
    `ImsgDumpMessage.is_unsent`, which is about a *message* being retracted,
    not a *tapback* being removed — do not conflate the two."""


@dataclass(frozen=True, slots=True)
class ImsgDumpMessage:
    """One decoded row from `imsg-dump`'s NDJSON output."""

    rowid: int
    guid: str
    chat_guid: str | None
    handle: str | None
    is_from_me: bool
    date: str | None
    date_edited: str | None
    date_retracted: str | None
    service: str
    body_text: str | None
    edit_history: tuple[EditVersion, ...]
    is_unsent: bool
    tapback: TapbackInfo | None
    attachment_rowids: tuple[int, ...]
    reply_to_guid: str | None


RunProcessFn = Callable[[list[str]], "subprocess.Popen[str]"]


def _default_run_process(argv: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )


def _parse_tapback(raw: object) -> TapbackInfo | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ImsgDumpError(f"'tapback' field must be an object or null, got {raw!r}")
    kind = raw.get("kind")
    target_guid = raw.get("target_guid")
    if not isinstance(kind, str) or not isinstance(target_guid, str):
        raise ImsgDumpError(
            f"'tapback' object must have string 'kind' and 'target_guid', got {raw!r}"
        )
    emoji = raw.get("emoji")
    if emoji is not None and not isinstance(emoji, str):
        raise ImsgDumpError(f"'tapback.emoji' must be a string or null, got {emoji!r}")
    action = raw.get("action", "added")
    if action not in ("added", "removed"):
        raise ImsgDumpError(f"'tapback.action' must be 'added' or 'removed', got {action!r}")
    return TapbackInfo(kind=kind, target_guid=target_guid, emoji=emoji, action=action)


def _parse_edit_history(raw: object) -> tuple[EditVersion, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ImsgDumpError(f"'edit_history' must be an array, got {raw!r}")
    versions: list[EditVersion] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ImsgDumpError(f"'edit_history' entries must be objects, got {item!r}")
        versions.append(
            EditVersion(text=item.get("text"), edited_at=item.get("edited_at"))
        )
    return tuple(versions)


def _parse_line(line: str) -> ImsgDumpMessage:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ImsgDumpError(f"imsg-dump emitted a non-JSON line: {line!r}") from exc
    if not isinstance(raw, dict):
        raise ImsgDumpError(f"imsg-dump line did not decode to a JSON object: {line!r}")

    try:
        rowid = raw["rowid"]
        guid = raw["guid"]
    except KeyError as exc:
        raise ImsgDumpError(f"imsg-dump line missing required field {exc}: {line!r}") from exc
    if not isinstance(rowid, int) or not isinstance(guid, str):
        raise ImsgDumpError(f"imsg-dump line has wrong types for rowid/guid: {line!r}")

    try:
        attachment_rowids = tuple(int(x) for x in raw.get("attachment_rowids") or ())
    except (TypeError, ValueError) as exc:
        raise ImsgDumpError(f"'attachment_rowids' must be an array of integers: {line!r}") from exc

    return ImsgDumpMessage(
        rowid=rowid,
        guid=guid,
        chat_guid=raw.get("chat_guid"),
        handle=raw.get("handle"),
        is_from_me=bool(raw.get("is_from_me", False)),
        date=raw.get("date"),
        date_edited=raw.get("date_edited"),
        date_retracted=raw.get("date_retracted"),
        service=str(raw.get("service", "unknown")),
        body_text=raw.get("body_text"),
        edit_history=_parse_edit_history(raw.get("edit_history")),
        is_unsent=bool(raw.get("is_unsent", False)),
        tapback=_parse_tapback(raw.get("tapback")),
        attachment_rowids=attachment_rowids,
        reply_to_guid=raw.get("reply_to_guid"),
    )


@dataclass(frozen=True, slots=True)
class ImsgDumpRun:
    messages: tuple[ImsgDumpMessage, ...]
    stderr_lines: tuple[str, ...]
    """Captured stderr — per SPEC §4.2, the shim logs per-row decode
    failures here and continues rather than aborting; these lines are
    surfaced to the caller for the run report/log, never raised."""


def run_imsg_dump(
    *,
    binary_path: Path,
    snapshot_path: Path,
    since_rowid: int,
    run_process: RunProcessFn = _default_run_process,
) -> ImsgDumpRun:
    """Run `imsg-dump --db <snapshot_path> --since-rowid <since_rowid>`
    and parse its NDJSON stdout.

    Raises `ImsgDumpError` if the binary is missing, cannot be started,
    exits nonzero, or emits a line that fails to parse — all boundary
    failures. An individual message's *internal* decode failure is
    specified to degrade inside the shim (stderr line + `body_text:
    null`), not to raise here.
    """
    if not binary_path.is_file():
        raise ImsgDumpError(
            f"imsg-dump binary not found at '{binary_path}' — build it with "
            f"'cargo build --release' in tools/imsg-dump/ first"
        )

    argv = [
        str(binary_path),
        "--db",
        str(snapshot_path),
        "--since-rowid",
        str(since_rowid),
    ]

    try:
        proc = run_process(argv)
    except OSError as exc:
        raise ImsgDumpError(f"could not start imsg-dump ({argv}): {exc}") from exc

    messages: list[ImsgDumpMessage] = []
    stderr_lines: list[str] = []

    assert proc.stdout is not None  # guaranteed by stdout=PIPE in every run_process impl we ship
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            messages.append(_parse_line(line))
    finally:
        proc.stdout.close()
        if proc.stderr is not None:
            stderr_lines = [ln.rstrip("\n") for ln in proc.stderr if ln.strip()]
            proc.stderr.close()
        returncode = proc.wait()

    for stderr_line in stderr_lines:
        logger.warning("imsg_dump.stderr", line=stderr_line)

    if returncode != 0:
        tail = "; ".join(stderr_lines[-5:])
        raise ImsgDumpError(
            f"imsg-dump exited {returncode} ({argv}); stderr tail: {tail or '<empty>'}"
        )

    return ImsgDumpRun(messages=tuple(messages), stderr_lines=tuple(stderr_lines))


__all__ = [
    "DEBUG_BINARY_RELPATH",
    "DEFAULT_BINARY_RELPATH",
    "EditVersion",
    "ImsgDumpMessage",
    "ImsgDumpRun",
    "RunProcessFn",
    "TapbackInfo",
    "default_binary_path",
    "run_imsg_dump",
]
