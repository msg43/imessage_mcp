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
import sys
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
from imsg.errors import MountGateError
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
        "guard_mount",
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
# A check that cannot look must still alert (QA 2026-11-23)
# ---------------------------------------------------------------------------
#
# With the encrypted volume not mounted the nightly backup cannot run either,
# so this is the case the check exists for. It used to exit through the mount
# gate before it reached `--notify`, so nothing was ever posted.


def test_an_unmounted_volume_notifies_and_exits_one_instead_of_dying_in_the_gate(
    tmp_path: Path,
    config_dict_factory: ConfigDictFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real gate, not a stand-in: a data root with no volume sentinel
    is refused, and the command must turn that into a failing verdict."""
    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cli_module, "notify_macos", lambda t, m: posted.append((t, m)) or None
    )
    config_path = tmp_path / "config.yaml"
    config = config_dict_factory(**{"paths.data_root": str(tmp_path / "volume-absent")})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = runner.invoke(app, ["backup-check", "--config", str(config_path), "--notify"])
    assert result.exit_code == 1, result.output
    assert "backup check could not look" in result.output
    assert "Traceback" not in result.output
    assert len(posted) == 1, "the notification is the whole point of the check"
    title, message = posted[0]
    assert "could not" in title and "stale" not in title
    assert "backup check could not look" in message


def test_a_refused_mount_gate_exits_one_with_the_reason_and_without_the_flag_posts_nothing(
    check: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[object] = []
    monkeypatch.setattr(cli_module, "notify_macos", lambda t, m: posted.append(m))

    def refuse(root: Path) -> MountInfo:
        raise MountGateError("sentinel .imsgindex-volume is missing")

    monkeypatch.setattr(cli_module, "guard_mount", refuse)
    result = check()
    assert result.exit_code == 1
    assert "sentinel .imsgindex-volume is missing" in result.output
    assert posted == []


def test_a_config_that_cannot_be_read_also_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config file can sit on the very volume that is missing."""
    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cli_module, "notify_macos", lambda t, m: posted.append((t, m)) or None
    )
    missing = tmp_path / "volume-absent" / "config.yaml"
    result = runner.invoke(app, ["backup-check", "--config", str(missing), "--notify"])
    assert result.exit_code == 1
    assert "backup check could not look" in result.output
    assert len(posted) == 1 and "config file not found" in posted[0][1]


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


def test_the_agent_is_not_held_back_by_the_mount_gate_so_it_can_report_a_missing_volume(
    config_dict_factory: ConfigDictFactory,
) -> None:
    """The other agents wait for the encrypted volume and exit 78 without it.
    This one must run exactly then, so its supervisor is told not to gate
    and to read no secret."""
    plist = plistlib.loads(_render(config_dict_factory, only=["backup-check"])[LABEL])
    arguments = plist["ProgramArguments"]
    supervisor = arguments[: arguments.index("--")]
    assert "--no-mount-gate" in supervisor
    assert "--env" not in supervisor
    for name in AGENT_NAMES:
        if name == "tunnel":
            continue
        other = plistlib.loads(
            _render(config_dict_factory, only=[name])[f"{LABEL_PREFIX}{name}"]
        )["ProgramArguments"]
        assert "--no-mount-gate" not in other[: other.index("--")], name


def test_the_rendered_agent_runs_the_check_when_the_volume_is_absent(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    """The agent exactly as rendered, run for real against a data root with
    no volume in it: the command must start (here a stand-in `imsg` that
    records its arguments and exits 1, as a failing check does), its exit
    status must reach launchd, and nothing may be written under the data
    root. Before the fix the supervisor exited 78 and the command never ran."""
    seen = tmp_path / "seen"
    stand_in = tmp_path / "bin" / "imsg"
    stand_in.parent.mkdir()
    stand_in.write_text(f'#!/bin/sh\necho "$@" > {seen}\nexit 1\n', encoding="utf-8")
    stand_in.chmod(0o755)
    absent_root = tmp_path / "volume-absent"
    absent_root.mkdir()
    config = load_config_dict(config_dict_factory(**{"paths.data_root": str(absent_root)}))
    plist = plistlib.loads(
        render_agent_plists(
            config,
            imsg_binary=stand_in,
            postgres_binary=POSTGRES,
            cloudflared_binary=None,
            config_path=tmp_path / "config.yaml",
            interpreter=Path(sys.executable),
            only=["backup-check"],
        )[LABEL]
    )
    done = subprocess.run(
        plist["ProgramArguments"], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 1, done.stderr
    assert seen.read_text(encoding="utf-8").split() == [
        "backup-check", "--config", str(tmp_path / "config.yaml"), "--notify",
    ]
    assert list(absent_root.iterdir()) == []


def test_the_check_time_catches_a_missed_night_the_same_morning_and_passes_a_good_one() -> None:
    """The backup runs at 04:00. At check time a good night's newest set is
    `BACKUP_CHECK_HOUR - 4` hours old and a missed night's is that plus 24.
    The first must be under the limit and the second over it."""
    limit_hours = BACKUP_STALE_AFTER_SECONDS / 3600
    good_night_age = BACKUP_CHECK_HOUR - 4
    missed_night_age = good_night_age + 24
    assert good_night_age < limit_hours < missed_night_age
