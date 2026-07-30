from __future__ import annotations

from pathlib import Path

import pytest

from imsg.errors import MountGateError
from imsg.mount.guard import (
    EX_CONFIG,
    MountInfo,
    guard_mount,
    run_guard_mount_or_exit,
)


def test_guard_mount_succeeds_when_encrypted_mounted_with_sentinel(
    data_root: Path, encrypted_mount_info: object
) -> None:
    info = guard_mount(data_root, diskutil_info=encrypted_mount_info)  # type: ignore[arg-type]
    assert info.encrypted is True


def test_guard_mount_fails_when_sentinel_missing(tmp_path: Path, encrypted_mount_info: object) -> None:
    root = tmp_path / "no_sentinel"
    root.mkdir()
    with pytest.raises(MountGateError, match="sentinel"):
        guard_mount(root, diskutil_info=encrypted_mount_info)  # type: ignore[arg-type]


def test_guard_mount_fails_when_not_encrypted(data_root: Path) -> None:
    def unencrypted(path: Path) -> MountInfo:
        return MountInfo(mount_point=path, encrypted=False, volume_name="Boot Volume")

    with pytest.raises(MountGateError, match="not encrypted"):
        guard_mount(data_root, diskutil_info=unencrypted)


def test_guard_mount_fails_when_no_mount_point_found(data_root: Path) -> None:
    def absent(path: Path) -> MountInfo:
        return MountInfo(mount_point=None, encrypted=False, volume_name=None)

    with pytest.raises(MountGateError, match="no mounted volume"):
        guard_mount(data_root, diskutil_info=absent)


def test_guard_mount_never_infers_containment_by_string_prefix(
    tmp_path: Path, encrypted_mount_info: object
) -> None:
    """A data_root that is a symlink pointing OUTSIDE the reported (encrypted,
    mounted) volume must fail even though the string prefix would look fine."""
    real_mount = tmp_path / "real_encrypted_volume"
    real_mount.mkdir()
    (real_mount / ".imsgindex-volume").write_text("")

    decoy_root = tmp_path / "decoy_root"
    decoy_root.mkdir()
    (decoy_root / ".imsgindex-volume").write_text("")

    # data_root path is a symlink whose name suggests it's under decoy_root,
    # but it actually points at a directory outside the volume diskutil
    # reports for `decoy_root` itself.
    outside = tmp_path / "outside_everything"
    outside.mkdir()
    escape_link = decoy_root / "escape"
    escape_link.symlink_to(outside)

    def reports_decoy_as_the_volume(path: Path) -> MountInfo:
        return MountInfo(mount_point=decoy_root, encrypted=True, volume_name="Decoy")

    with pytest.raises(MountGateError):
        guard_mount(escape_link, diskutil_info=reports_decoy_as_the_volume)


def test_diskutil_failure_is_normalized_to_mount_gate_error(data_root: Path) -> None:
    def broken(path: Path) -> MountInfo:
        raise RuntimeError("diskutil exploded")

    with pytest.raises(MountGateError, match="diskutil exploded"):
        guard_mount(data_root, diskutil_info=broken)


def test_run_guard_mount_or_exit_exits_78_and_logs_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import imsg.mount.guard as guard_module

    fake_log = tmp_path / "guard.log"
    monkeypatch.setattr(guard_module, "GUARD_LOG_PATH", fake_log)

    broken_root = tmp_path / "not_mounted_anywhere"

    with pytest.raises(SystemExit) as exc_info:
        run_guard_mount_or_exit(broken_root)
    # run_guard_mount_or_exit uses the real diskutil_info by default; on a
    # dev machine without that exact mount it will still fail the gate
    # (either "no volume found" or similar) - assert only the exit code
    # and log behavior, which is what SPEC §5.4 actually mandates.
    assert exc_info.value.code == EX_CONFIG

    assert fake_log.exists()
    content = fake_log.read_text()
    assert "guard-mount failed" in content
    # Content-free: never the data_root path's contents, only the failure
    # reason and a timestamp. We can at least assert it stays a single line
    # per failure and never embeds anything resembling message content.
    assert content.count("\n") == 1
