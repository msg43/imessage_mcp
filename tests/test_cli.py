"""CLI wiring tests. No live database or real mounted volume required —
DB/mount-touching commands are exercised with monkeypatched collaborators
so this file stays in the "no network, no live Postgres" unit suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import imsg.cli as cli_module
from imsg.cli import app
from imsg.db.migrations import AppliedMigration, MigrationFile, MigrationPlan
from imsg.diagnostics import AtRestPosture, MountCheck, PostgresCheck
from imsg.mount.guard import MountInfo

runner = CliRunner()


def _write_config(path: Path, data_root: Path, messages_dir: Path) -> None:
    live_chat_db = messages_dir / "chat.db"
    live_chat_db.parent.mkdir(parents=True, exist_ok=True)
    live_chat_db.write_text("")
    path.write_text(
        f"""
paths:
  data_root: {data_root}
  live_chat_db: {live_chat_db}
database:
  dsn: postgresql://imsg@127.0.0.1:5433/imsgindex
  password: env:IMSG_TEST_PG_PASSWORD
sync:
  interval_seconds: 900
  sources:
    - name: mini
      chat_db: {live_chat_db}
embedding:
  revision: deadbeef
  query_instruction: "test instruction"
  multimodal:
    revision: cafef00d
retrieval:
  reranker_revision: f00dcafe
mcp:
  public:
    scope: allowlist
export:
  gcp_project: example-project
  gcs_bucket: example-bucket
  data_store_id: example-datastore
"""
    )


@pytest.fixture
def cli_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)

    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, data_root, messages_dir)
    return config_path


# --------------------------------------------------------------------------
# Stub pipeline-stage commands
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "snapshot",
        "extract",
        "identity",
        "segment",
        "embed",
        "sync",
        "enrich",
        "backfill-attachments",
        "export",
        "install-agents",
    ],
)
def test_stub_stage_exits_nonzero_and_names_the_stage(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 1
    assert command.replace("-", "-") in result.output  # the stage name appears
    assert "not implemented" in result.output


def test_help_lists_every_stage() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["migrate", "status", "check-permissions", "snapshot", "sync", "export"]:
        assert command in result.output


# --------------------------------------------------------------------------
# check-permissions / status: real diagnostics logic, fake system probes
# --------------------------------------------------------------------------


def test_check_permissions_json_output(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module,
        "check_mount",
        lambda data_root: MountCheck(ok=True, reason=None, info=None),
    )
    monkeypatch.setattr(
        cli_module,
        "check_at_rest_posture",
        lambda data_root: AtRestPosture(
            label="unattended",
            boot_volume_encrypted=False,
            auto_login_enabled=True,
            data_volume_encrypted=True,
            caveat="test caveat",
        ),
    )
    monkeypatch.setattr(cli_module, "check_full_disk_access", lambda path: True)
    monkeypatch.setattr(
        cli_module,
        "check_postgres",
        lambda config: PostgresCheck(reachable=True, cluster_fingerprint_ok=True, reason=None),
    )

    result = runner.invoke(app, ["check-permissions", "--config", str(cli_config), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mount_ok"] is True
    assert payload["pg_ok"] is True
    assert payload["at_rest_posture"] == "unattended"
    assert payload["contacts_access"] is None


def test_status_json_output(cli_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module, "check_mount", lambda data_root: MountCheck(ok=False, reason="nope", info=None)
    )
    monkeypatch.setattr(
        cli_module,
        "check_at_rest_posture",
        lambda data_root: AtRestPosture(
            label="mixed-or-unknown",
            boot_volume_encrypted=None,
            auto_login_enabled=None,
            data_volume_encrypted=None,
            caveat="unknown",
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "check_postgres",
        lambda config: PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason="down"),
    )
    monkeypatch.setattr(cli_module, "disk_free_bytes", lambda path: 123456)

    result = runner.invoke(app, ["status", "--config", str(cli_config), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mount_ok"] is False
    assert payload["disk_free_bytes"] == 123456
    assert payload["watermarks_per_source"] is None


# --------------------------------------------------------------------------
# migrate: fully mocked DB layer, no live Postgres or mount required
# --------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self, plan: MigrationPlan, applied_to_return: list[MigrationFile]) -> None:
        self._plan = plan
        self._applied_to_return = applied_to_return
        self.apply_called = False

    def plan(self) -> MigrationPlan:
        return self._plan

    def verify(self) -> MigrationPlan:
        return self._plan

    def apply_pending(self) -> list[MigrationFile]:
        self.apply_called = True
        return self._applied_to_return


class _FakeConn:
    def close(self) -> None:
        pass


@pytest.fixture
def mocked_migrate_env(cli_config: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakeConn())
    monkeypatch.setattr(cli_module, "ensure_cluster_fingerprint", lambda *a, **kw: "fake-uuid")
    return cli_config


def test_migrate_status_reports_pending(
    mocked_migrate_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = MigrationPlan(applied=(), pending=(), mismatches=())
    fake_runner = _FakeRunner(plan, [])
    monkeypatch.setattr(cli_module, "PostgresMigrationRunner", lambda conn, mdir: fake_runner)

    result = runner.invoke(app, ["migrate", "--config", str(mocked_migrate_env), "--status"])
    assert result.exit_code == 0, result.output
    assert fake_runner.apply_called is False


def test_migrate_status_flags_mismatches(
    mocked_migrate_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.db.migrations import HashMismatch

    plan = MigrationPlan(
        applied=(AppliedMigration(version=1, sha256="a" * 64),),
        pending=(),
        mismatches=(HashMismatch(version=1, applied_sha256="a" * 64, disk_sha256="b" * 64),),
    )
    fake_runner = _FakeRunner(plan, [])
    monkeypatch.setattr(cli_module, "PostgresMigrationRunner", lambda conn, mdir: fake_runner)

    result = runner.invoke(app, ["migrate", "--config", str(mocked_migrate_env), "--status"])
    assert result.exit_code == 1


def test_migrate_applies_pending_and_reports_fingerprint(
    mocked_migrate_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migration_file = MigrationFile(
        version=1, name="initial", path=Path("0001_initial.sql"), sha256="a" * 64, sql="SELECT 1;"
    )
    plan = MigrationPlan(applied=(), pending=(migration_file,), mismatches=())
    fake_runner = _FakeRunner(plan, [migration_file])
    monkeypatch.setattr(cli_module, "PostgresMigrationRunner", lambda conn, mdir: fake_runner)

    result = runner.invoke(app, ["migrate", "--config", str(mocked_migrate_env)])
    assert result.exit_code == 0, result.output
    assert fake_runner.apply_called is True
    assert "fake-uuid" in result.output


def test_migrate_status_and_verify_are_mutually_exclusive(mocked_migrate_env: Path) -> None:
    result = runner.invoke(
        app, ["migrate", "--config", str(mocked_migrate_env), "--status", "--verify"]
    )
    assert result.exit_code == 2


def test_migrate_with_bad_config_reports_config_error_not_a_traceback(tmp_path: Path) -> None:
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text("paths: {data_root: /nonexistent}\n")
    result = runner.invoke(app, ["migrate", "--config", str(bad_config)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
