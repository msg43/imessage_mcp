"""Dataless-placeholder detection (SPEC §8 S5a) — exercised against fake
stat/ls probes since a real dataless placeholder can't be fabricated in
a test sandbox."""

from __future__ import annotations

import os
from pathlib import Path

from imsg.backfill.dataless import blocks_zero_heuristic, is_dataless, ls_dataless_flag


class _FakeStat:
    def __init__(self, st_size: int, st_blocks: int) -> None:
        self.st_size = st_size
        self.st_blocks = st_blocks


def test_blocks_zero_heuristic_true_for_zero_blocks_nonzero_size(tmp_path: Path) -> None:
    p = tmp_path / "f"
    assert blocks_zero_heuristic(p, stat_fn=lambda _: _FakeStat(1024, 0)) is True  # type: ignore[arg-type,return-value]


def test_blocks_zero_heuristic_false_for_normal_file(tmp_path: Path) -> None:
    p = tmp_path / "f"
    assert blocks_zero_heuristic(p, stat_fn=lambda _: _FakeStat(1024, 8)) is False  # type: ignore[arg-type,return-value]


def test_blocks_zero_heuristic_false_for_empty_file(tmp_path: Path) -> None:
    """A genuinely empty (0-byte) file also has 0 blocks — that's not a
    dataless placeholder, just an empty file."""
    p = tmp_path / "f"
    assert blocks_zero_heuristic(p, stat_fn=lambda _: _FakeStat(0, 0)) is False  # type: ignore[arg-type,return-value]


def test_blocks_zero_heuristic_none_on_stat_error(tmp_path: Path) -> None:
    def _raise(_: Path) -> os.stat_result:
        raise OSError("gone")

    assert blocks_zero_heuristic(tmp_path / "missing", stat_fn=_raise) is None


def test_ls_dataless_flag_true_when_present(tmp_path: Path) -> None:
    p = tmp_path / "f"
    output = "-rw-r--r--  1 user  staff  dataless 1024 Jan  1 00:00 f\n"
    assert ls_dataless_flag(p, ls_probe=lambda _: output) is True


def test_ls_dataless_flag_false_when_absent(tmp_path: Path) -> None:
    p = tmp_path / "f"
    output = "-rw-r--r--  1 user  staff  - 1024 Jan  1 00:00 f\n"
    assert ls_dataless_flag(p, ls_probe=lambda _: output) is False


def test_ls_dataless_flag_none_on_empty_output(tmp_path: Path) -> None:
    assert ls_dataless_flag(tmp_path / "f", ls_probe=lambda _: "") is None


def test_is_dataless_true_if_either_signal_fires(tmp_path: Path) -> None:
    p = tmp_path / "f"
    assert (
        is_dataless(
            p,
            stat_fn=lambda _: _FakeStat(1024, 0),  # type: ignore[arg-type,return-value]
            ls_probe=lambda _: "-rw-r--r--  1 user  staff  - 1024 Jan  1 00:00 f\n",
        )
        is True
    )
    assert (
        is_dataless(
            p,
            stat_fn=lambda _: _FakeStat(1024, 8),  # type: ignore[arg-type,return-value]
            ls_probe=lambda _: "-rw-r--r--  1 user  staff  dataless 1024 Jan  1 00:00 f\n",
        )
        is True
    )


def test_is_dataless_false_when_neither_signal_fires(tmp_path: Path) -> None:
    p = tmp_path / "f"
    assert (
        is_dataless(
            p,
            stat_fn=lambda _: _FakeStat(1024, 8),  # type: ignore[arg-type,return-value]
            ls_probe=lambda _: "-rw-r--r--  1 user  staff  - 1024 Jan  1 00:00 f\n",
        )
        is False
    )


def test_is_dataless_against_a_real_normal_file(tmp_path: Path) -> None:
    """No fakes: a real, freshly-written file on this machine's real
    filesystem must never be classified dataless. Also a regression
    test in its own right: this test's own tmp-dir name (derived from
    the test function name) contains the literal substring "dataless",
    which once caused a false positive from a naive whole-line
    substring search over `ls -lO` output — `ls_dataless_flag` must
    only look at the flags *field*, never the full line/path."""
    p = tmp_path / "real.txt"
    p.write_text("hello world")
    assert is_dataless(p) is False
