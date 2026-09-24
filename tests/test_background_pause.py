"""The pause switch for heavy background work (`imsg.background_pause`):
set by `imsg background pause`, or by another project creating the host
pause file; either one pauses."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import ConfigDictFactory
from imsg.background_pause import (
    BackgroundPauseError,
    clear_pause,
    parse_until,
    pause_file_path,
    read_pause_state,
    write_pause,
)
from imsg.config.loader import load_config_dict
from imsg.errors import ConfigError

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_pause_then_resume(data_root: Path) -> None:
    assert not read_pause_state(data_root, host_pause_file=None).paused

    request = write_pause(data_root, reason="library import", until=None, now=NOW)
    state = read_pause_state(data_root, host_pause_file=None, now=NOW)
    assert state.paused
    assert state.active == (request,)
    assert state.describe() == (
        "paused: library import; set by imsg background pause at 2026-09-24T12:00:00+00:00"
    )
    assert pause_file_path(data_root) == data_root / "run" / "background-pause.json"

    assert clear_pause(data_root) is True
    assert clear_pause(data_root) is False
    assert not read_pause_state(data_root, host_pause_file=None).paused


def test_a_pause_with_until_ends_by_itself(data_root: Path) -> None:
    write_pause(data_root, reason=None, until=NOW + timedelta(hours=6), now=NOW)
    assert read_pause_state(data_root, host_pause_file=None, now=NOW + timedelta(hours=5)).paused
    later = read_pause_state(data_root, host_pause_file=None, now=NOW + timedelta(hours=6))
    assert not later.paused
    assert len(later.lapsed) == 1 and later.lapsed[0].endswith("(expired)")


def test_until_takes_an_iso_time_or_a_duration() -> None:
    assert parse_until("90m", now=NOW) == NOW + timedelta(minutes=90)
    assert parse_until("6h", now=NOW) == NOW + timedelta(hours=6)
    assert parse_until("2d", now=NOW) == NOW + timedelta(days=2)
    assert parse_until("2026-09-26T08:00:00+00:00", now=NOW) == datetime(
        2026, 9, 26, 8, 0, tzinfo=UTC
    )
    naive = parse_until("2026-09-26T08:00", now=NOW)
    assert naive.tzinfo is not None  # read as this host's local time
    with pytest.raises(BackgroundPauseError, match="neither an ISO 8601 time"):
        parse_until("next tuesday", now=NOW)


def test_a_reason_is_one_short_printable_line(data_root: Path) -> None:
    request = write_pause(data_root, reason="import\n\tphotos\x07" + "x" * 500, until=None)
    assert request.reason is not None
    assert "\n" not in request.reason and "\x07" not in request.reason
    assert len(request.reason) == 200


def test_the_host_pause_file_pauses_by_existing(data_root: Path, tmp_path: Path) -> None:
    flag = tmp_path / "pause-background"
    assert not read_pause_state(data_root, host_pause_file=flag).paused
    flag.write_text("")
    state = read_pause_state(data_root, host_pause_file=flag)
    assert state.paused
    assert state.active[0].source == f"host pause file {flag}"
    flag.unlink()
    assert not read_pause_state(data_root, host_pause_file=flag).paused


def test_the_host_pause_file_can_carry_a_reason(data_root: Path, tmp_path: Path) -> None:
    flag = tmp_path / "pause-background"
    flag.write_text("reason=photo library import\n")
    assert read_pause_state(data_root, host_pause_file=flag).active[0].reason == (
        "photo library import"
    )
    flag.write_text("importing photos, back tomorrow\n")
    assert read_pause_state(data_root, host_pause_file=flag).active[0].reason == (
        "importing photos, back tomorrow"
    )


def test_a_host_pause_tied_to_a_process_ends_when_the_process_does(
    data_root: Path, tmp_path: Path
) -> None:
    flag = tmp_path / "pause-background"
    importer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        flag.write_text(f"reason=import\npid={importer.pid}\n")
        assert read_pause_state(data_root, host_pause_file=flag).paused
    finally:
        importer.kill()
        importer.wait(timeout=30)
    state = read_pause_state(data_root, host_pause_file=flag)
    assert not state.paused
    assert state.lapsed[0].endswith("(that process is gone)")


def test_a_host_pause_with_a_passed_until_no_longer_pauses(data_root: Path, tmp_path: Path) -> None:
    flag = tmp_path / "pause-background"
    flag.write_text("until=2026-09-24T11:00:00+00:00\n")
    assert not read_pause_state(data_root, host_pause_file=flag, now=NOW).paused
    flag.write_text("until=2026-09-24T13:00:00+00:00\n")
    assert read_pause_state(data_root, host_pause_file=flag, now=NOW).paused


def test_an_unreadable_pause_file_counts_as_a_pause(data_root: Path) -> None:
    path = pause_file_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    state = read_pause_state(data_root, host_pause_file=None)
    assert state.paused
    assert "treated as paused" in state.describe()
    clear_pause(data_root)
    assert not read_pause_state(data_root, host_pause_file=None).paused


def test_either_switch_pauses_and_both_are_reported(data_root: Path, tmp_path: Path) -> None:
    flag = tmp_path / "pause-background"
    flag.write_text("reason=other project\n")
    write_pause(data_root, reason="this project", until=None)
    state = read_pause_state(data_root, host_pause_file=flag)
    assert [r.reason for r in state.active] == ["this project", "other project"]
    clear_pause(data_root)
    assert read_pause_state(data_root, host_pause_file=flag).paused  # the host file still pauses


def test_the_host_pause_file_is_on_by_default_outside_the_data_root(
    config_dict_factory: ConfigDictFactory,
) -> None:
    cfg = load_config_dict(config_dict_factory())
    expected = Path(os.environ["HOME"]) / ".config" / "imessage-index" / "pause-background"
    assert cfg.background.host_pause_file == expected
    off = load_config_dict(config_dict_factory(**{"background": {"host_pause_file": None}}))
    assert off.background.host_pause_file is None


def test_the_host_pause_file_may_not_live_under_the_messages_folder(
    config_dict_factory: ConfigDictFactory, messages_dir: Path
) -> None:
    with pytest.raises(ConfigError, match=r"background\.host_pause_file"):
        load_config_dict(
            config_dict_factory(**{"background": {"host_pause_file": str(messages_dir / "pause")}})
        )
