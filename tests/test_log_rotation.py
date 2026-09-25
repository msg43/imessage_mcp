"""Size-based rotation of `<data_root>/logs` (`imsg.log_rotation`, SPEC
§14). Before 2026-09-24 nothing rotated these files: the scheduled
sync's log grew by about 800 KB a day and the public server's output
files were plain redirects (QA review 2026-09-24).

The property that matters most is the one a rename-based rotator gets
wrong: a writer that holds the log open — launchd, the supervisor, a
KeepAlive server — keeps writing to the same file after rotation, from
the start, with no hole."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import pytest

from imsg.log_rotation import (
    LOCK_FILENAME,
    LogRotationError,
    _rotation_lock,
    logs_dir_for,
    rotate_logs,
    total_log_bytes,
)

MB = 10**6


@pytest.fixture
def root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data-root"
    (data_root / "logs").mkdir(parents=True)
    return data_root


def _log(root: Path, name: str, size: int, fill: bytes = b"x") -> Path:
    path = root / "logs" / name
    line = fill * 99 + b"\n"
    path.write_bytes(line * (size // len(line)) + line[: size % len(line)])
    return path


def test_a_log_at_the_threshold_is_compressed_and_truncated_in_place(root: Path) -> None:
    path = _log(root, "imsgindex-mcp-public.err.log", 2 * MB)
    original = path.read_bytes()
    inode = path.stat().st_ino

    report = rotate_logs(root, rotate_bytes=MB)

    assert report.rotated == [("imsgindex-mcp-public.err.log", 2 * MB)]
    assert path.stat().st_size == 0
    assert path.stat().st_ino == inode, "same file, so open writers carry on"
    with gzip.open(root / "logs" / "imsgindex-mcp-public.err.log.1.gz", "rb") as handle:
        assert handle.read() == original
    assert oct((root / "logs" / "imsgindex-mcp-public.err.log.1.gz").stat().st_mode & 0o777) == "0o600"


def test_an_appending_writer_keeps_writing_to_the_rotated_file_without_a_hole(root: Path) -> None:
    path = root / "logs" / "scheduled-sync.log"
    with path.open("a", buffering=1) as writer:  # how the scheduled sync writes
        writer.write("before rotation\n" * 70_000)
        writer.flush()
        rotate_logs(root, rotate_bytes=MB)
        writer.write("after rotation\n")
    assert path.read_bytes() == b"after rotation\n"
    with gzip.open(root / "logs" / "scheduled-sync.log.1.gz", "rb") as handle:
        assert handle.read() == b"before rotation\n" * 70_000


def test_small_logs_other_files_and_symlinks_are_left_alone(root: Path, tmp_path: Path) -> None:
    small = _log(root, "pg.log", MB - 1)
    progress = root / "logs" / "phase3-progress.tsv"
    progress.write_bytes(b"x" * 3 * MB)
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"y" * 3 * MB)
    (root / "logs" / "linked.log").symlink_to(outside)

    report = rotate_logs(root, rotate_bytes=MB)

    assert report.rotated == []
    assert report.skipped == [("linked.log", "not a regular file")]
    assert small.stat().st_size == MB - 1
    assert progress.stat().st_size == 3 * MB
    assert outside.stat().st_size == 3 * MB, "a symlink is never followed"


def test_generations_shift_and_the_oldest_beyond_keep_is_deleted(root: Path) -> None:
    for round_number in range(3):
        _log(root, "a.log", 2 * MB, fill=str(round_number).encode())
        rotate_logs(root, rotate_bytes=MB, keep=2)
    names = sorted(p.name for p in (root / "logs").iterdir() if p.name != LOCK_FILENAME)
    assert names == ["a.log", "a.log.1.gz", "a.log.2.gz"]
    with gzip.open(root / "logs" / "a.log.1.gz", "rb") as handle:
        assert handle.read(1) == b"2", "newest generation is .1"
    with gzip.open(root / "logs" / "a.log.2.gz", "rb") as handle:
        assert handle.read(1) == b"1"


def test_generations_older_than_the_retention_window_are_deleted(root: Path) -> None:
    _log(root, "a.log", 2 * MB)
    rotate_logs(root, rotate_bytes=MB)
    generation = root / "logs" / "a.log.1.gz"
    old = generation.stat().st_mtime - 91 * 86400
    os.utime(generation, (old, old))

    report = rotate_logs(root, rotate_bytes=MB, retention_days=90)

    assert report.deleted == ["a.log.1.gz"]
    assert not generation.exists()


def test_dry_run_reports_and_changes_nothing(root: Path) -> None:
    path = _log(root, "a.log", 2 * MB)
    report = rotate_logs(root, rotate_bytes=MB, dry_run=True)
    assert report.rotated == [("a.log", 2 * MB)]
    assert report.describe() == [f"would rotate a.log ({2 * MB:,} bytes)"]
    assert path.stat().st_size == 2 * MB
    assert not (root / "logs" / "a.log.1.gz").exists()


def test_a_logs_directory_that_escapes_the_data_root_is_refused(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (data_root / "logs").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(LogRotationError, match="outside the data root"):
        rotate_logs(data_root)


def test_one_rotation_at_a_time(root: Path) -> None:
    _log(root, "a.log", 2 * MB)
    with _rotation_lock(logs_dir_for(root)), pytest.raises(LogRotationError, match="another log rotation"):
        rotate_logs(root, rotate_bytes=MB)


def test_a_missing_logs_directory_is_nothing_to_rotate(tmp_path: Path) -> None:
    assert rotate_logs(tmp_path).rotated == []
    assert total_log_bytes(tmp_path) == 0
