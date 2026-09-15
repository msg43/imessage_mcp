"""Unit tests for `imsg.backfill.classify` — no database, no filesystem:
every function here decides from a path's *shape* or an errno alone.
Paths are synthetic; none of them is stat'd or opened."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from imsg.backfill.classify import (
    NO_SOURCE_PATH_ERROR,
    UnsupportedReason,
    classify_os_error,
    classify_out_of_root,
    format_unsupported_error,
    parse_unsupported_reason,
)


@pytest.mark.parametrize(
    ("resolved", "expected"),
    [
        (
            Path("/private/var/folders/ab/cdef0123/T/com.example.app/blob.bin"),
            UnsupportedReason.TEMP_DIRECTORY_PATH,
        ),
        (
            Path("/var/folders/ab/cdef0123/C/cache.bin"),
            UnsupportedReason.TEMP_DIRECTORY_PATH,
        ),
        (Path("/tmp/dropped.bin"), UnsupportedReason.TEMP_DIRECTORY_PATH),
        (Path("/private/tmp/dropped.bin"), UnsupportedReason.TEMP_DIRECTORY_PATH),
        (
            Path("/Users/alice/Library/Messages/StickerCache/pack/sticker.heic"),
            UnsupportedReason.STICKER_CACHE_PATH,
        ),
        (Path("/Users/alice/Downloads/elsewhere.pdf"), UnsupportedReason.OUT_OF_ROOT_PATH),
        (Path("/Volumes/Other/Library/Messages/Attachments/x.jpg"), UnsupportedReason.OUT_OF_ROOT_PATH),
    ],
)
def test_classify_out_of_root_names_the_path_shape(
    resolved: Path, expected: UnsupportedReason
) -> None:
    assert classify_out_of_root(resolved) is expected


def test_sticker_cache_wins_over_temp_directory() -> None:
    # A fake home under the temp tree (exactly what pytest's tmp_path
    # produces on macOS) must still classify by the more specific marker.
    resolved = Path("/private/var/folders/ab/cdef0123/T/home/Library/Messages/StickerCache/s.heic")
    assert classify_out_of_root(resolved) is UnsupportedReason.STICKER_CACHE_PATH


def test_sticker_cache_marker_must_be_contiguous() -> None:
    # "StickerCache" somewhere else in the path is not the Messages sticker cache.
    resolved = Path("/Users/alice/Library/StickerCache/Messages/x.heic")
    assert classify_out_of_root(resolved) is UnsupportedReason.OUT_OF_ROOT_PATH


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (errno.EISDIR, UnsupportedReason.IS_A_DIRECTORY),
        (errno.ENAMETOOLONG, UnsupportedReason.FILE_NAME_TOO_LONG),
    ],
)
def test_classify_os_error_deterministic_classes(err: int, expected: UnsupportedReason) -> None:
    exc = OSError(err, "synthetic")
    assert classify_os_error(exc) is expected


@pytest.mark.parametrize("err", [errno.ENOENT, errno.EIO, errno.EACCES, errno.ETIMEDOUT])
def test_classify_os_error_leaves_transient_classes_on_the_retry_ladder(err: int) -> None:
    assert classify_os_error(OSError(err, "synthetic")) is None


def test_classify_os_error_without_errno_is_transient() -> None:
    assert classify_os_error(OSError("no errno at all")) is None


def test_format_and_parse_round_trip() -> None:
    for reason in UnsupportedReason:
        text = format_unsupported_error(reason, "detail with 'quotes' and — dashes")
        assert text.startswith(f"unsupported[{reason.value}]: ")
        assert parse_unsupported_reason(text) is reason


@pytest.mark.parametrize(
    "last_error",
    [None, "", "[Errno 5] Input/output error", "unsupported[not-a-real-class]: x", NO_SOURCE_PATH_ERROR],
)
def test_parse_unsupported_reason_rejects_anything_else(last_error: str | None) -> None:
    assert parse_unsupported_reason(last_error) is None


def test_every_reason_has_a_description() -> None:
    for reason in UnsupportedReason:
        assert reason.description
        assert reason.value in format_unsupported_error(reason, "x")
