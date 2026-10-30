"""The public server's model hold (`imsg.public_model_hold`), the
`imsg mcp hold-models --until` parser, and what an enrich run is admitted
for (2026-10-29: the nightly caption window, and the OCR-only run that was
charged for the captioner)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import ConfigDictFactory
from imsg.cli import _parse_hold_until
from imsg.config.loader import load_config_dict
from imsg.host_memory import GIB
from imsg.memory_admission import MemoryAdmission, ModelRole, enrich_footprint_bytes
from imsg.public_model_hold import (
    MAX_HOLD,
    PublicModelHoldError,
    clear_hold,
    hold_file_path,
    read_hold,
    write_hold,
)

NOW = datetime(2026, 10, 30, 5, 0, tzinfo=UTC)  # 01:00 EDT


def test_a_written_hold_reads_back_and_is_active_until_its_end(tmp_path: Path) -> None:
    until = NOW + timedelta(hours=8)
    hold = write_hold(tmp_path, until=until, reason="nightly caption window", now=NOW)
    assert hold_file_path(tmp_path).is_file()
    reading = read_hold(tmp_path)
    assert reading.problem is None
    assert reading.hold == hold
    assert reading.active_hold(NOW + timedelta(hours=7, minutes=59)) is not None
    assert reading.active_hold(until) is None  # ends at `until`, by itself


def test_clear_removes_the_hold_and_says_whether_there_was_one(tmp_path: Path) -> None:
    write_hold(tmp_path, until=NOW + timedelta(hours=1), reason=None, now=NOW)
    assert clear_hold(tmp_path) is True
    assert clear_hold(tmp_path) is False
    assert read_hold(tmp_path).hold is None


@pytest.mark.parametrize(
    ("until", "message"),
    [
        (NOW - timedelta(minutes=1), "already in the past"),
        (NOW + MAX_HOLD + timedelta(minutes=1), "at most 12 hours"),
        (datetime(2026, 10, 30, 9, 0), "time zone"),
    ],
)
def test_a_hold_must_end_soon_and_in_the_future(
    tmp_path: Path, until: datetime, message: str
) -> None:
    with pytest.raises(PublicModelHoldError, match=message):
        write_hold(tmp_path, until=until, reason=None, now=NOW)
    assert not hold_file_path(tmp_path).exists()


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps(["a list"]),
        json.dumps({"reason": "no end"}),
        json.dumps({"until": "2026-10-30T09:00"}),
    ],
)
def test_an_unusable_hold_file_is_ignored_with_a_reason(tmp_path: Path, content: str) -> None:
    """Fails toward serving: a garbled hold never takes search away."""
    path = hold_file_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    reading = read_hold(tmp_path)
    assert reading.hold is None
    assert reading.active_hold(NOW) is None
    assert reading.problem is not None and "ignored" in reading.problem


def test_until_takes_a_clock_time_as_the_next_one_on_this_host() -> None:
    local_midnight_ish = datetime(2026, 10, 30, 1, 0).astimezone()
    nine = _parse_hold_until("09:00", now=local_midnight_ish)
    assert nine.astimezone().hour == 9 and nine - local_midnight_ish == timedelta(hours=8)
    # 09:00 asked at 10:00 means tomorrow's.
    ten = local_midnight_ish.replace(hour=10)
    assert _parse_hold_until("09:00", now=ten) - ten == timedelta(hours=23)


def test_until_also_takes_durations_and_iso_times() -> None:
    assert _parse_hold_until("8h", now=NOW) == NOW + timedelta(hours=8)
    assert _parse_hold_until("2026-10-30T09:00:00+00:00", now=NOW) == datetime(
        2026, 10, 30, 9, 0, tzinfo=UTC
    )
    with pytest.raises(PublicModelHoldError):
        _parse_hold_until("25:00", now=NOW)
    with pytest.raises(PublicModelHoldError):
        _parse_hold_until("tomorrow", now=NOW)


def test_an_enrich_run_that_never_captions_is_admitted_for_six_gib(
    config_dict_factory: ConfigDictFactory,
) -> None:
    cfg = load_config_dict(config_dict_factory())
    fast = ("doc_text", "pdf_text", "transcript", "ocr", "frame_ocr")
    assert enrich_footprint_bytes(cfg.memory, fast) == int(6.0 * GIB)
    assert enrich_footprint_bytes(cfg.memory, ("caption",)) == int(24.8 * GIB)
    assert enrich_footprint_bytes(cfg.memory, (*fast, "caption")) == int(24.8 * GIB)
    admission = MemoryAdmission.for_role(
        cfg,
        ModelRole.ENRICH,
        command="imsg enrich",
        footprint=enrich_footprint_bytes(cfg.memory, fast),
    )
    assert admission.required_bytes == int(6.0 * GIB)
    assert MemoryAdmission.for_role(cfg, ModelRole.ENRICH, command="x").required_bytes == int(
        24.8 * GIB
    )
