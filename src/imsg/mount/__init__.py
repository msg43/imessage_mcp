"""The encrypted-volume mount gate (SPEC §5.4)."""

from imsg.mount.guard import EX_CONFIG, MountInfo, guard_mount, run_guard_mount_or_exit

__all__ = ["EX_CONFIG", "MountInfo", "guard_mount", "run_guard_mount_or_exit"]
