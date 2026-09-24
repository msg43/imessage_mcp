"""The mount gate (SPEC §5.4) — CLAUDE.md non-negotiable #2.

`guard_mount` refuses to proceed unless `paths.data_root` resolves to a
path that is actually inside a *mounted, encrypted* volume, and that
volume carries the sentinel file that proves it is really the intended
data volume rather than an unmounted mount point silently resolving to
the boot volume underneath it.

As of 2026-09-24 (owner decision), "encrypted volume" includes the
FileVault-encrypted startup (boot) volume, not only a separate
encrypted APFS volume — non-experts installing this on a single-disk
Mac would otherwise have no feasible `data_root`. The boot volume is
accepted only for a `data_root` outside `/Volumes/` (e.g.
`~/imsgindex-data`), and only when FileVault is confirmed ON for it
(via `fdesetup status`); an unencrypted boot volume, and any case
where that status cannot be determined, still fail closed. A
`data_root` under `/Volumes/` that resolves to the boot volume is
always refused, fail-closed, with no FileVault check at all — that
shape only arises when the intended separate volume is unmounted, per
the 2026-08-12 regression this module documents below.

This is deliberately importable (call it at the top of every CLI entry
point and service start — see `imsg.cli`) as well as runnable as
`imsg guard-mount`. On failure, the CLI wrapper exits `EX_CONFIG` (78)
and appends one content-free line to
`~/Library/Logs/imsgindex-guard.log` — the only permitted off-mount log
write (SPEC §5.4).
"""

from __future__ import annotations

import os
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
FileVaultStatusFn = Callable[[], bool | None]


def containing_mount_point(path: Path) -> Path:
    """Walk `path` up to the mount point of the volume containing it.

    `diskutil info` accepts a device node or a **mount point** — never
    an arbitrary path inside a volume. `data_root` is always inside one
    (`/Volumes/IMSG-Data/imsgindex`, per SPEC §6 and the
    implementation guide §0.6), so passing it to `diskutil` directly
    fails with exit 1 on every valid deployment.
    """
    p = path
    while p != p.parent and not os.path.ismount(p):
        p = p.parent
    return p


def real_diskutil_info(path: Path) -> MountInfo:
    """Query `diskutil info -plist` for the volume containing `path`.

    Raises `MountGateError` if `diskutil` is unavailable or reports
    failure (e.g. no volume found for that path) — never a raw
    `subprocess`/`plistlib` exception, so callers only ever need to
    handle one error type.
    """
    try:
        proc = subprocess.run(
            ["diskutil", "info", "-plist", str(containing_mount_point(path))],
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


def real_filevault_status() -> bool | None:
    """Query `fdesetup status` for whether FileVault is on for the startup disk.

    Returns `True`/`False` when `fdesetup` gives an unambiguous answer,
    `None` when it is unavailable, times out, or prints something this
    parser doesn't recognize. `None` must be treated as "unknown" by
    callers, never as "off" or "on" — `guard_mount` fails closed on it,
    the same way it fails closed on a `real_diskutil_info` error.
    """
    try:
        proc = subprocess.run(
            ["fdesetup", "status"], capture_output=True, text=True, check=False, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lowered = proc.stdout.lower()
    if "filevault is on" in lowered:
        return True
    if "filevault is off" in lowered:
        return False
    return None


def guard_mount(
    data_root: Path,
    *,
    diskutil_info: DiskutilInfoFn = real_diskutil_info,
    filevault_status: FileVaultStatusFn = real_filevault_status,
    sentinel_filename: str = SENTINEL_FILENAME,
) -> MountInfo:
    """Refuse to proceed unless `data_root` is really on a mounted, encrypted volume.

    Returns the `MountInfo` on success. Raises `MountGateError` on any
    failure — absent mount, unencrypted volume, or missing sentinel
    file (SPEC §5.4 steps 1-2). Never infers mount status from a string
    prefix: `data_root` is fully resolved (symlinks + `..`) before any
    comparison, per SPEC §5.4 step 1.

    The startup (boot) volume is accepted as an encrypted volume only
    for a `data_root` outside `/Volumes/`, and only when
    `filevault_status()` confirms FileVault is on for it. A `data_root`
    under `/Volumes/` that resolves to the boot volume is always
    refused — see the comment above the check. A separate, non-boot
    volume is still accepted purely on `diskutil`'s `Encryption` field,
    as before.
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

    # `/` is the startup (boot) volume. As of 2026-09-24 (owner decision,
    # CLAUDE.md non-negotiable #2) it is a legitimate data_root location when
    # FileVault protects it — but ONLY for a data_root that was actually
    # meant to live there.
    #
    # The failure mode found 2026-08-15 still applies: when a *separate*
    # encrypted volume is configured but UNMOUNTED, os.path.ismount is False
    # for its would-be mount point, so the walk-up lands on '/' and diskutil
    # answers for the boot volume instead. On macOS a separate volume always
    # mounts under /Volumes — nothing legitimately configured outside
    # /Volumes ever resolves to a non-boot volume — so a resolved data_root
    # under /Volumes that nonetheless landed on '/' is unambiguous proof of
    # exactly this failure mode, never a genuine boot-volume deployment. That
    # case is refused unconditionally, before any FileVault check, closing
    # the gap the original (2026-09-24) boot-volume allowance left open: a
    # stale sentinel under an unmounted /Volumes path could otherwise pass
    # once the boot-volume path itself became acceptable.
    #
    # A data_root outside /Volumes (e.g. `~/imsgindex-data`) that resolves to
    # '/' has no such ambiguity — there is no unmounted-separate-volume
    # explanation for it — so it is judged purely on FileVault status. The
    # sentinel file below still applies in that case; since there is no
    # "unmounted" scenario for the disk the OS is running from, it instead
    # guards against a missing or renamed `data_root` directory being
    # silently treated as valid (e.g. a typo'd path that happens to exist for
    # an unrelated reason).
    if resolved_mount_point == Path("/"):
        if is_contained_in(resolved_root, Path("/Volumes")):
            raise MountGateError(
                f"'{resolved_root}' is under /Volumes/ but that volume is not mounted; "
                "unlock/mount it (e.g. the encrypted data volume) and retry."
            )
        boot_encrypted = filevault_status()
        if boot_encrypted is not True:
            status_desc = "is off" if boot_encrypted is False else "could not be determined"
            raise MountGateError(
                f"data_root '{resolved_root}' resolved to the startup volume ('/'), whose "
                f"FileVault status {status_desc} (CLAUDE.md non-negotiable #2). Fix: turn on "
                "FileVault for the startup disk (System Settings > Privacy & Security > "
                "FileVault), or point paths.data_root at a separate encrypted volume instead."
            )
    elif not info.encrypted:
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
    "real_filevault_status",
    "run_guard_mount_or_exit",
]
