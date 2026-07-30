"""Shared diagnostic checks behind `imsg check-permissions` and `imsg status`.

Foundation-build scope note: the full field set SPEC §10.2/§14 describes
for these commands (watermarks, queue depths, FTS outbox lag, coverage,
audit-rejection counts, unclassified-thread counts, last sync/export/
backup) depends on pipeline stages this build does not implement yet
(S1-S8). Those fields are reported as `None` with an explanatory note
rather than faked — a later build wires them up once the corresponding
stage exists. Everything checkable *without* a live pipeline (mount,
Postgres reachability + cluster fingerprint, at-rest posture, Full Disk
Access, disk free space) is implemented for real.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from imsg.config.schema import Config
from imsg.db.connection import connect
from imsg.db.fingerprint import verify_data_directory
from imsg.errors import ClusterFingerprintError, MountGateError, SecretResolutionError
from imsg.mount.guard import MountInfo, guard_mount, real_diskutil_info


@dataclass(frozen=True, slots=True)
class MountCheck:
    ok: bool
    reason: str | None
    info: MountInfo | None


def check_mount(data_root: Path) -> MountCheck:
    try:
        info = guard_mount(data_root)
    except MountGateError as exc:
        return MountCheck(ok=False, reason=str(exc), info=None)
    return MountCheck(ok=True, reason=None, info=info)


@dataclass(frozen=True, slots=True)
class AtRestPosture:
    label: str
    """One of 'unattended', 'secure', 'mixed-or-unknown'."""

    boot_volume_encrypted: bool | None
    auto_login_enabled: bool | None
    data_volume_encrypted: bool | None
    caveat: str


_UNATTENDED_CAVEAT = (
    "unattended posture (SPEC §5.1, ratified D6): protects against bare-disk "
    "theft only — NOT whole-host theft, because auto-login unlocks the "
    "encrypted volume without a human present. Do not describe this as "
    "'at-rest protected' without this qualification."
)
_SECURE_CAVEAT = (
    "secure posture (SPEC §5.1): full at-rest protection, but the index is "
    "down after any reboot until an operator logs in."
)
_UNKNOWN_CAVEAT = (
    "does not match either documented posture (SPEC §5.1) — review manually."
)


def _fdesetup_status() -> bool | None:
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


def _auto_login_enabled() -> bool | None:
    try:
        proc = subprocess.run(
            [
                "defaults",
                "read",
                "/Library/Preferences/com.apple.loginwindow",
                "autoLoginUser",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        # Key absent (readable but unset) is the common "auto-login off" case;
        # a read failure for another reason is genuinely unknown, but we have
        # no reliable way to tell those apart without root, so this is a
        # best-effort signal, not a guarantee.
        return False
    return bool(proc.stdout.strip())


def check_at_rest_posture(data_root: Path) -> AtRestPosture:
    boot_encrypted = _fdesetup_status()
    auto_login = _auto_login_enabled()
    try:
        info = real_diskutil_info(data_root)
        data_encrypted: bool | None = info.encrypted
    except Exception:
        data_encrypted = None

    if boot_encrypted is False and auto_login is True:
        return AtRestPosture(
            label="unattended",
            boot_volume_encrypted=boot_encrypted,
            auto_login_enabled=auto_login,
            data_volume_encrypted=data_encrypted,
            caveat=_UNATTENDED_CAVEAT,
        )
    if boot_encrypted is True and auto_login is False:
        return AtRestPosture(
            label="secure",
            boot_volume_encrypted=boot_encrypted,
            auto_login_enabled=auto_login,
            data_volume_encrypted=data_encrypted,
            caveat=_SECURE_CAVEAT,
        )
    return AtRestPosture(
        label="mixed-or-unknown",
        boot_volume_encrypted=boot_encrypted,
        auto_login_enabled=auto_login,
        data_volume_encrypted=data_encrypted,
        caveat=_UNKNOWN_CAVEAT,
    )


def check_full_disk_access(live_chat_db: Path) -> bool | None:
    """Best-effort FDA probe: try to open+read a few bytes of the live chat.db.

    Returns `True`/`False` when determinable, `None` when the path is
    simply absent (can't distinguish "no FDA" from "no Messages set up
    yet"). Read-only, and reads at most 16 bytes — never anything that
    could be message content.
    """
    path = live_chat_db.expanduser()
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.read(16)
    except PermissionError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class PostgresCheck:
    reachable: bool
    cluster_fingerprint_ok: bool | None
    reason: str | None


def check_postgres(config: Config) -> PostgresCheck:
    try:
        conn = connect(config.database, autocommit=True)
    except SecretResolutionError as exc:
        return PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason=str(exc))
    except Exception as exc:
        return PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason=str(exc))

    try:
        try:
            verify_data_directory(conn, config.paths.data_root)
        except ClusterFingerprintError as exc:
            return PostgresCheck(reachable=True, cluster_fingerprint_ok=False, reason=str(exc))
        return PostgresCheck(reachable=True, cluster_fingerprint_ok=True, reason=None)
    finally:
        conn.close()


def disk_free_bytes(path: Path) -> int | None:
    """`shutil.disk_usage` on the nearest existing ancestor of `path`."""
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    try:
        return shutil.disk_usage(candidate).free
    except OSError:
        return None


__all__ = [
    "AtRestPosture",
    "MountCheck",
    "PostgresCheck",
    "check_at_rest_posture",
    "check_full_disk_access",
    "check_mount",
    "check_postgres",
    "disk_free_bytes",
]
