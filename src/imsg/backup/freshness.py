"""Is the nightly backup still being made? The check that runs unattended.

`imsg status` already reports `backup_stale` (a newest complete set older
than `BACKUP_STALE_AFTER_SECONDS`), but it is a command someone has to
run. A backup job can fail every night while nobody looks, and the newest
set simply ages. `imsg backup-check` is that same test as a command with
an exit status: 0 when the newest complete set is fresh, 1 when it is
stale, absent or could not be judged. Run daily by the optional
`backup-check` LaunchAgent (`imsg install-agents --only backup-check`),
a failing exit shows in launchd's last exit status, and `--notify` also
raises a macOS notification, the same mechanism `scripts/copy_backup_offsite.py
--notify` uses.

A check that cannot tell fails loudly rather than passing: an unreadable
`backups/` directory, a manifest with an unparseable timestamp, a
directory with no complete set at all, a config file that cannot be read
and an encrypted volume that is not mounted all exit 1, because "nothing
to report" and "could not look" must not read as healthy. The last is the
likeliest way for the nightly backup to stop, so the command runs the mount
gate itself and reports its refusal (the agent is also rendered without the
supervisor's mount wait, which would exit before the command ran).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from imsg.diagnostics import BACKUP_STALE_AFTER_SECONDS, BackupStatus

STALE_TITLE = "iMessage index backup is stale"
COULD_NOT_CHECK_TITLE = "iMessage index backup could not be checked"


@dataclass(frozen=True, slots=True)
class FreshnessVerdict:
    ok: bool
    message: str
    title: str = STALE_TITLE
    """The notification's title when `ok` is false."""


def judge_backup_freshness(status: BackupStatus) -> FreshnessVerdict:
    """Turn `imsg.diagnostics.check_backups`'s answer into pass or fail with a reason."""
    hours = BACKUP_STALE_AFTER_SECONDS // 3600
    partial = (
        f" ({status.partial_sets} partial set(s) present)" if status.partial_sets else ""
    )
    if status.backup_stale is False:
        return FreshnessVerdict(
            True,
            f"newest complete backup set is {status.last_backup_at}, within {hours} hours; "
            f"{status.complete_sets} complete set(s) kept{partial}",
        )
    if status.backup_stale is True:
        return FreshnessVerdict(
            False,
            f"the newest complete backup set is from {status.last_backup_at}, more than "
            f"{hours} hours ago, so the nightly backup has not produced a new set"
            f"{partial}. Run 'imsg backup' by hand and read its error line, or read the "
            f"backup job's error log under <data_root>/logs.",
        )
    return FreshnessVerdict(
        False,
        f"backup freshness could not be established: {status.reason or 'no reason given'}"
        f"{partial}",
        COULD_NOT_CHECK_TITLE,
    )


def notify_macos(
    title: str,
    message: str,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> str | None:
    """Post a macOS notification; return None on success or the failure's words.

    The text goes to `osascript` as arguments of the script's `run`
    handler, never spliced into the script, so a quote or backslash in a
    message cannot break (or inject into) the AppleScript.
    """
    command = [
        "osascript",
        "-e",
        "on run argv",
        "-e",
        "display notification (item 2 of argv) with title (item 1 of argv)",
        "-e",
        "end run",
        title,
        message,
    ]
    try:
        proc = run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return f"osascript exited {proc.returncode}: {proc.stderr.strip() or '(no stderr)'}"
    return None


__all__ = [
    "COULD_NOT_CHECK_TITLE",
    "STALE_TITLE",
    "FreshnessVerdict",
    "judge_backup_freshness",
    "notify_macos",
]
