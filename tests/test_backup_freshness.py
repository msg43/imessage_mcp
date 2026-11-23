"""`imsg backup-check`: the unattended "is the nightly backup still being
made?" test, and the optional LaunchAgent that runs it daily.

No database and no real notification: the backup directory is a temporary
one holding hand-made sets (the same shape `index_backups` accepts), and
the macOS notification is replaced by a recorder. The test that matters
most is the one where the newest set is old: the check must exit 1 and
say why, because a quiet exit 0 there is the failure it exists to catch.

Fictional paths only (D5).
"""

from __future__ import annotations

import json
import plistlib
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import imsg.cli as cli_module
from conftest import ConfigDictFactory
from imsg.agents.plists import (
    AGENT_NAMES,
    BACKUP_CHECK_HOUR,
    LABEL_PREFIX,
    OPTIONAL_AGENT_NAMES,
    render_agent_plists,
)
from imsg.backup.freshness import judge_backup_freshness, notify_macos
from imsg.backup.retention import MANIFEST_FILENAME, MANIFEST_FORMAT
from imsg.cli import app
from imsg.config.loader import load_config_dict
from imsg.diagnostics import BACKUP_STALE_AFTER_SECONDS, BackupStatus
from imsg.mount.guard import MountInfo

runner = CliRunner()


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def _status(
    *, stale: bool | None, last: str | None = "2026-11-20T04:00:00+00:00",
    complete: int = 14, partial: int = 0, reason: str | None = None,
) -> BackupStatus:
    return BackupStatus(
        last_backup_at=last, backup_stale=stale, complete_sets=complete,
        partial_sets=partial, reason=reason,
    )


def test_a_fresh_newest_set_passes() -> None:
    verdict = judge_backup_freshness(_status(stale=False))
    assert verdict.ok
    assert "2026-11-20T04:00:00+00:00" in verdict.message and "within 26 hours" in verdict.message


def test_a_stale_newest_set_fails_and_says_when_it_is_from_and_what_to_do() -> None:
    verdict = judge_backup_freshness(_status(stale=True, last="2026-11-01T04:00:00+00:00"))
    assert not verdict.ok
    assert "2026-11-01T04:00:00+00:00" in verdict.message
    assert "more than 26 hours" in verdict.message
    assert "imsg backup" in verdict.message and "error log" in verdict.message


def test_the_limit_the_message_quotes_is_the_one_status_uses() -> None:
    assert BACKUP_STALE_AFTER_SECONDS == 26 * 3600


def test_no_complete_set_fails_rather_than_passing_quietly() -> None:
    verdict = judge_backup_freshness(
        _status(stale=None, last=None, complete=0, reason="no complete backup set under backups/")
    )
    assert not verdict.ok
    assert "could not be established" in verdict.message
    assert "no complete backup set under backups/" in verdict.message


def test_an_unjudgeable_timestamp_fails() -> None:
    verdict = judge_backup_freshness(
        _status(stale=None, reason="newest set's created_at 'x' is not ISO-8601")
    )
    assert not verdict.ok and "not ISO-8601" in verdict.message


def test_partial_sets_are_mentioned_but_never_make_a_stale_set_pass() -> None:
    verdict = judge_backup_freshness(_status(stale=True, partial=2))
    assert not verdict.ok and "2 partial set(s) present" in verdict.message


# ---------------------------------------------------------------------------
# The notification
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, *, returncode: int = 0, stderr: str = "", error: Exception | None = None):
        self.calls: list[list[str]] = []
        self._result = SimpleNamespace(returncode=returncode, stderr=stderr)
        self._error = error

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        self.calls.append(command)
        if self._error is not None:
            raise self._error
        return self._result


def test_the_message_travels_as_an_argument_never_inside_the_script() -> None:
    hostile = 'newest set is old" & do shell script "echo oops'
    recorder = _Recorder()
    assert notify_macos("iMessage index backup is stale", hostile, run=recorder) is None
    (command,) = recorder.calls
    assert command[0] == "osascript"
    script_parts = [command[i + 1] for i, part in enumerate(command) if part == "-e"]
    assert all(hostile not in part for part in script_parts)
    assert command[-2:] == ["iMessage index backup is stale", hostile]


def test_a_failing_osascript_is_reported_not_swallowed() -> None:
    failure = notify_macos("t", "m", run=_Recorder(returncode=1, stderr="not authorized"))
    assert failure is not None and "osascript exited 1" in failure and "not authorized" in failure


def test_a_missing_osascript_is_reported_not_swallowed() -> None:
    failure = notify_macos("t", "m", run=_Recorder(error=FileNotFoundError("osascript")))
    assert failure is not None and "FileNotFoundError" in failure


def test_a_hung_osascript_is_reported_not_swallowed() -> None:
    timeout = subprocess.TimeoutExpired(cmd="osascript", timeout=30)
    failure = notify_macos("t", "m", run=_Recorder(error=timeout))
    assert failure is not None and "TimeoutExpired" in failure


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _write_set(backups: Path, created: datetime, uid: str = "aaaaaa") -> None:
    path = backups / f"backup-{created.strftime('%Y%m%dT%H%M%SZ')}-{uid}"
    path.mkdir(parents=True)
    (path / "postgres.dump").write_bytes(b"not-a-real-dump")
    (path / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "complete": True,
                "created_at": created.isoformat(),
                "files": [],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def check(
    config_dict_factory: ConfigDictFactory, data_root: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Any]:
    """Runs `imsg backup-check` against a temporary data_root; returns the result."""
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda root: MountInfo(mount_point=root, encrypted=True, volume_name="scratch"),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict_factory()), encoding="utf-8")

    def _run(*extra: str) -> Any:
        return runner.invoke(app, ["backup-check", "--config", str(config_path), *extra])

    return _run


def test_a_fresh_set_exits_zero_and_is_quiet_about_notifications(
    check: Callable[..., Any], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(cli_module, "notify_macos", lambda t, m: posted.append((t, m)))
    _write_set(data_root / "backups", datetime.now(UTC) - timedelta(hours=6))
    result = check("--notify")
    assert result.exit_code == 0, result.output
    assert "backup-check: newest complete backup set" in result.output
    assert posted == []


def test_an_old_newest_set_exits_one_with_the_reason_and_notifies(
    check: Callable[..., Any], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cli_module, "notify_macos", lambda t, m: posted.append((t, m)) or None
    )
    _write_set(data_root / "backups", datetime.now(UTC) - timedelta(days=13))
    result = check("--notify")
    assert result.exit_code == 1
    assert result.output.strip().startswith("imsg: backup-check: the newest complete backup set")
    assert "Traceback" not in result.output
    assert len(posted) == 1
    title, message = posted[0]
    assert "stale" in title and "more than 26 hours" in message


def test_without_the_flag_nothing_is_posted(
    check: Callable[..., Any], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[object] = []
    monkeypatch.setattr(cli_module, "notify_macos", lambda t, m: posted.append(m))
    _write_set(data_root / "backups", datetime.now(UTC) - timedelta(days=3))
    assert check().exit_code == 1
    assert posted == []


def test_no_backups_directory_at_all_exits_one(check: Callable[..., Any]) -> None:
    result = check()
    assert result.exit_code == 1
    assert "could not be established" in result.output
    assert "no complete backup set" in result.output


def test_a_failed_notification_is_said_aloud_and_the_exit_stays_one(
    check: Callable[..., Any], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "notify_macos", lambda t, m: "osascript exited 1: no GUI")
    _write_set(data_root / "backups", datetime.now(UTC) - timedelta(days=5))
    result = check("--notify")
    assert result.exit_code == 1
    assert "notification not posted: osascript exited 1: no GUI" in result.output


def test_it_reads_the_backup_directory_only(
    check: Callable[..., Any], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No database and no secret: the Postgres password variable is unset
    here and the command still answers."""
    monkeypatch.delenv("IMSG_TEST_PG_PASSWORD", raising=False)
    _write_set(data_root / "backups", datetime.now(UTC) - timedelta(hours=1))
    assert check().exit_code == 0


# ---------------------------------------------------------------------------
# The optional LaunchAgent
# ---------------------------------------------------------------------------

DATA_ROOT = "/Volumes/IMSG-Data/imsgindex"
INTERPRETER = Path("/opt/example/python3.12")
SUPERVISOR = Path("/opt/example/imsg/agents/supervise.py")
IMSG = Path("/usr/local/bin/imsg")
POSTGRES = Path("/opt/homebrew/opt/postgresql@17/bin/postgres")
CONFIG_PATH = Path("/Volumes/IMSG-Data/imsgindex/private/config.yaml")
LABEL = f"{LABEL_PREFIX}backup-check"


def _render(config_dict_factory: ConfigDictFactory, **kwargs: object) -> dict[str, bytes]:
    config = load_config_dict(config_dict_factory(**{"paths.data_root": DATA_ROOT}))
    return render_agent_plists(
        config,
        imsg_binary=IMSG,
        postgres_binary=POSTGRES,
        cloudflared_binary=None,
        config_path=CONFIG_PATH,
        interpreter=INTERPRETER,
        supervisor=SUPERVISOR,
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_agent_is_optional_and_never_in_the_default_set(
    config_dict_factory: ConfigDictFactory,
) -> None:
    assert "backup-check" in OPTIONAL_AGENT_NAMES
    assert "backup-check" not in AGENT_NAMES
    rendered = _render(config_dict_factory, only=[n for n in AGENT_NAMES if n != "tunnel"])
    assert LABEL not in rendered


def test_the_agent_runs_daily_under_the_supervisor_with_no_database_and_no_secret(
    config_dict_factory: ConfigDictFactory,
) -> None:
    rendered = _render(config_dict_factory, only=["backup-check"])
    assert set(rendered) == {LABEL}
    plist = plistlib.loads(rendered[LABEL])
    assert plist["StartCalendarInterval"] == {"Hour": BACKUP_CHECK_HOUR, "Minute": 0}
    assert "KeepAlive" not in plist and "RunAtLoad" not in plist
    assert plist["ProcessType"] == "Background"
    arguments = plist["ProgramArguments"]
    cut = arguments.index("--")
    supervisor, command = arguments[:cut], arguments[cut + 1 :]
    assert supervisor[supervisor.index("--service") + 1] == "backup-check"
    assert "--wait-for-postgres" not in supervisor
    assert "--env" not in supervisor
    assert command == [str(IMSG), "backup-check", "--config", str(CONFIG_PATH), "--notify"]


def test_the_check_time_catches_a_missed_night_the_same_morning_and_passes_a_good_one() -> None:
    """The backup runs at 04:00. At check time a good night's newest set is
    `BACKUP_CHECK_HOUR - 4` hours old and a missed night's is that plus 24.
    The first must be under the limit and the second over it."""
    limit_hours = BACKUP_STALE_AFTER_SECONDS / 3600
    good_night_age = BACKUP_CHECK_HOUR - 4
    missed_night_age = good_night_age + 24
    assert good_night_age < limit_hours < missed_night_age
