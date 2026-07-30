"""The mount gate (SPEC §5.4) — CLAUDE.md non-negotiable #2.

`guard_mount` refuses to proceed unless `paths.data_root` resolves to a
path that is actually inside a *mounted, encrypted* volume, and that
volume carries the sentinel file that proves it is really the intended
data volume rather than an unmounted mount point silently resolving to
the boot volume underneath it.

This is deliberately importable (call it at the top of every CLI entry
point and service start — see `imsg.cli`) as well as runnable as
`imsg guard-mount`. On failure, the CLI wrapper exits `EX_CONFIG` (78)
and appends one content-free line to
`~/Library/Logs/imsgindex-guard.log` — the only permitted off-mount log
write (SPEC §5.4).
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from imsg.errors import MountGateError
from imsg.paths import is_contained_in, resolve_path

EX_CONFIG = 78
"""sysexits.h EX_CONFIG — configuration error. SPEC §5.4."""

SENTINEL_FILENAME = ".imsgindex-volume"
GUARD_LOG_PATH = Path("~/Library/Logs/imsgindex-guard.log").expanduser()


@dataclass(frozen=True, slots=True)
class MountInfo:
    """The subset of `diskutil info -plist` this gate cares about."""

    mount_point: Path | None
    encrypted: bool
    volume_name: str | None


DiskutilInfoFn = Callable[[Path], MountInfo]


def real_diskutil_info(path: Path) -> MountInfo:
    """Query `diskutil info -plist` for the volume containing `path`.

    Raises `MountGateError` if `diskutil` is unavailable or reports
    failure (e.g. no volume found for that path) — never a raw
    `subprocess`/`plistlib` exception, so callers only ever need to
    handle one error type.
    """
    try:
        proc = subprocess.run(
            ["diskutil", "info", "-plist", str(path)],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MountGateError(
            "the 'diskutil' CLI is not available — the mount gate only runs on macOS"
        ) from exc
    if proc.returncode != 0:
        raise MountGateError(
            f"'diskutil info' could not find a volume containing '{path}' "
            f"(exit {proc.returncode}) — is the encrypted volume mounted?"
        )
    try:
        data = plistlib.loads(proc.stdout)
    except (plistlib.InvalidFileException, ValueError) as exc:
        raise MountGateError("could not parse 'diskutil info -plist' output") from exc

    mount_point_raw = data.get("MountPoint")
    return MountInfo(
        mount_point=Path(mount_point_raw) if mount_point_raw else None,
        encrypted=bool(data.get("Encryption", False)),
        volume_name=data.get("VolumeName"),
    )


def guard_mount(
    data_root: Path,
    *,
    diskutil_info: DiskutilInfoFn = real_diskutil_info,
    sentinel_filename: str = SENTINEL_FILENAME,
) -> MountInfo:
    """Refuse to proceed unless `data_root` is really on a mounted, encrypted volume.

    Returns the `MountInfo` on success. Raises `MountGateError` on any
    failure — absent mount, unencrypted volume, or missing sentinel
    file (SPEC §5.4 steps 1-2). Never infers mount status from a string
    prefix: `data_root` is fully resolved (symlinks + `..`) before any
    comparison, per SPEC §5.4 step 1.
    """
    resolved_root = resolve_path(data_root)

    try:
        info = diskutil_info(resolved_root)
    except MountGateError:
        raise
    except Exception as exc:
        raise MountGateError(f"mount inspection failed for '{resolved_root}': {exc}") from exc

    if info.mount_point is None:
        raise MountGateError(
            f"no mounted volume found containing data_root '{resolved_root}'"
        )

    resolved_mount_point = resolve_path(info.mount_point)
    if resolved_root != resolved_mount_point and not is_contained_in(
        resolved_root, resolved_mount_point
    ):
        raise MountGateError(
            f"data_root '{resolved_root}' does not resolve under its reported "
            f"mount point '{resolved_mount_point}'"
        )

    if not info.encrypted:
        raise MountGateError(
            f"volume '{info.volume_name or resolved_mount_point}' containing "
            f"data_root is not encrypted (CLAUDE.md non-negotiable #2)"
        )

    sentinel = resolved_root / sentinel_filename
    if not sentinel.is_file():
        raise MountGateError(
            f"sentinel file '{sentinel}' is missing — refusing to treat an "
            f"unmounted or wrong path as the data volume (SPEC §5.4 step 2)"
        )

    return info


def _log_guard_failure(reason: str) -> None:
    """Append one content-free line to the off-mount guard failure log."""
    GUARD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    with GUARD_LOG_PATH.open("a") as f:
        f.write(f"{timestamp} guard-mount failed: {reason}\n")


def run_guard_mount_or_exit(data_root: Path) -> MountInfo:
    """The CLI-facing wrapper: run the gate, or log + `sys.exit(EX_CONFIG)`.

    Every CLI entry point that touches `data_root` should call this
    before doing anything else.
    """
    try:
        return guard_mount(data_root)
    except MountGateError as exc:
        _log_guard_failure(str(exc))
        print(f"imsg: mount gate failed: {exc}", file=sys.stderr)
        raise SystemExit(EX_CONFIG) from exc


__all__ = [
    "EX_CONFIG",
    "GUARD_LOG_PATH",
    "SENTINEL_FILENAME",
    "MountInfo",
    "guard_mount",
    "real_diskutil_info",
    "run_guard_mount_or_exit",
]
