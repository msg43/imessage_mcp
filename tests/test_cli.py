"""CLI wiring tests. No live database or real mounted volume required —
DB/mount-touching commands are exercised with monkeypatched collaborators
so this file stays in the "no network, no live Postgres" unit suite."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import imsg.cli as cli_module
import imsg.providers.factory as factory_module
from imsg.cli import app
from imsg.db.enrichment_yield_locks import YieldReport, YieldState
from imsg.db.migrations import AppliedMigration, MigrationFile, MigrationPlan
from imsg.diagnostics import AtRestPosture, BufferPoolCheck, MountCheck, PostgresCheck
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
models:
  backend: fake
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
# No stubs left
# --------------------------------------------------------------------------


def test_no_cli_command_is_a_stage_stub() -> None:
    """`export` was the last `StageNotImplementedError` stub and is now
    wired (see tests/test_export_cli_integration.py). This asserts the
    class of failure is gone rather than one instance of it: no command
    anywhere in the tree may answer "not implemented".

    `imsg backup` — which the daily LaunchAgent invokes — is a different
    thing: it is ABSENT, not stubbed, and is tracked in the cli module
    docstring. A stub would at least name itself when run; an absent
    command is why that note exists.

    The tree is walked through click's own command objects rather than by
    parsing rendered help, because help text wraps and a parser of it
    finds words like "Engine" and calls them commands.
    """
    import typer.main

    def walk(command: Any, path: tuple[str, ...]) -> list[tuple[str, ...]]:
        assert "not implemented" not in (command.help or "").lower(), path
        found = [path]
        # Duck-typed on `.commands` rather than `isinstance(_, click.Group)`:
        # typer's TyperGroup does not satisfy that isinstance check under
        # click 8.4, which silently made an earlier version of this walk
        # visit exactly one node and pass.
        for name, child in getattr(command, "commands", {}).items():
            found.extend(walk(child, (*path, name)))
        return found

    root = typer.main.get_command(app)
    paths = walk(root, ())
    assert len(paths) > 20, paths  # the walk actually walked something
    # And the one that used to be a stub really runs now.
    assert ("export", "plan") in paths
    assert ("export", "unclassified-report") in paths


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


# --------------------------------------------------------------------------
# Wired pipeline-stage commands: mocked DB/mount layer, real CLI plumbing.
# Each test only asserts that the CLI command loads config, gates on the
# mount, connects+verifies the fingerprint, and calls the real stage
# function with the right arguments, reporting its result — the stage
# functions' own behavior is exercised by their own modules' test suites.
# --------------------------------------------------------------------------


class _FakePgConn:
    """A no-op stand-in wherever `_connect_and_verify_or_die` hands back a
    connection — every stage function itself is monkeypatched in these
    tests, so nothing here needs real query behavior.

    It DOES have to model transaction control, though. This double
    originally exposed only `close()`, so no test could observe whether a
    command committed — which is exactly how the 2026-08-14 defect
    survived: `extract` reported success while every write rolled back at
    `conn.close()`. `commit`/`rollback` are recorded here so a test can
    assert on them rather than on the absence of an exception."""

    def __init__(self, transaction_status: Any = None) -> None:
        self.committed = 0
        self.rolled_back = 0
        self.closed = False
        self.statements: list[str] = []
        # Model psycopg's transaction status. The CLI asserts the connection is
        # IDLE after the fingerprint check; a double left this unmodelled and
        # the suite could not see the difference between a committed run and a
        # discarded one.
        import psycopg

        status = (
            psycopg.pq.TransactionStatus.IDLE if transaction_status is None else transaction_status
        )
        self.info = SimpleNamespace(transaction_status=status)

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1

    def close(self) -> None:
        self.closed = True

    def cursor(self) -> Any:
        """Enough of a cursor for the two things the CLI asks this fake
        database on the paths these tests drive:

        - the warm-up's buffer-pool step, which asks whether the
          `pg_prewarm` function exists (migration 0004) before it does
          anything — this database has no extensions, so the answer is no
          and the step reports that instead of prewarming;
        - the enrichment worker's yield gate, which probes the advisory
          lock the MCP server holds while answering a query
          (`imsg.db.enrichment_yield_locks`). `True` means nobody holds
          it, which is the no-query-running case these tests are in.
        """
        statements = self.statements

        class _Cursor:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def execute(self, sql: str, params: Any = None) -> None:
                statements.append(sql)
                allowed = ("proname = 'pg_prewarm'", "pg_try_advisory_lock", "pg_advisory_unlock")
                assert any(fragment in sql for fragment in allowed), (
                    f"unexpected statement: {sql}"
                )

            def fetchone(self) -> tuple[Any, ...]:
                return (True,) if "advisory" in statements[-1] else (False,)

        return _Cursor()


@pytest.fixture
def mocked_pg_env(cli_config: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakePgConn())
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    return cli_config


def _data_root_from_config(config_path: Path) -> Path:
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("data_root:"):
            return Path(stripped.split(":", 1)[1].strip())
    raise AssertionError("data_root not found in test config fixture")


def test_snapshot_wires_run_snapshot(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.snapshot import SnapshotResult

    captured: dict[str, Any] = {}

    def fake_run_snapshot(*, live_chat_db: Path, data_root: Path, **kw: Any) -> SnapshotResult:
        captured["live_chat_db"] = live_chat_db
        captured["data_root"] = data_root
        captured["kw"] = kw
        return SnapshotResult(
            path=data_root / "snapshots" / "snapshot.db",
            sha256="a" * 64,
            byte_size=10,
            reused_existing=False,
        )

    monkeypatch.setattr(cli_module, "run_snapshot", fake_run_snapshot)
    result = runner.invoke(app, ["snapshot", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "sha256=" + "a" * 64 in result.output
    assert captured["data_root"] == _data_root_from_config(mocked_pg_env)
    assert "DRY RUN" not in result.output
    assert captured["kw"] == {"dry_run": False}


def test_snapshot_dry_run_passes_the_flag_and_prints_the_marker(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.snapshot import SnapshotResult

    captured: dict[str, Any] = {}

    def fake_run_snapshot(*, live_chat_db: Path, data_root: Path, **kw: Any) -> SnapshotResult:
        captured["kw"] = kw
        return SnapshotResult(
            path=data_root / "snapshots" / "snapshot.db",
            sha256="b" * 64,
            byte_size=10,
            reused_existing=False,
            dry_run=True,
        )

    monkeypatch.setattr(cli_module, "run_snapshot", fake_run_snapshot)
    result = runner.invoke(app, ["snapshot", "--config", str(mocked_pg_env), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert captured["kw"] == {"dry_run": True}
    assert "DRY RUN — nothing was written" in result.output


def test_extract_without_a_snapshot_exits_cleanly(mocked_pg_env: Path) -> None:
    result = runner.invoke(app, ["extract", "--config", str(mocked_pg_env)])
    assert result.exit_code == 1
    assert "run 'imsg snapshot' first" in result.output


def test_extract_wires_run_extract(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.extract import ExtractResult

    data_root = _data_root_from_config(mocked_pg_env)
    snapshot_dir = data_root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "snapshot.db").write_text("")

    captured: dict[str, Any] = {}

    def fake_run_extract(**kwargs: Any) -> ExtractResult:
        captured.update(kwargs)
        return ExtractResult(
            run_id=1,
            watermark_before=0,
            watermark_after=5,
            chats_upserted=1,
            handles_upserted=1,
            messages_upserted=5,
            tapbacks_upserted=0,
            system_messages_skipped=0,
            attachments_upserted=0,
            link_previews_upserted=0,
            bodies_missing=0,
            dump_stderr_line_count=0,
        )

    monkeypatch.setattr(cli_module, "run_extract", fake_run_extract)
    result = runner.invoke(app, ["extract", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "messages_upserted=5" in result.output
    assert captured["source_name"] == "mini"
    assert captured["dry_run"] is False
    assert "DRY RUN" not in result.output


def test_extract_dry_run_passes_the_flag_and_prints_the_marker(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.extract import ExtractResult

    data_root = _data_root_from_config(mocked_pg_env)
    snapshot_dir = data_root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "snapshot.db").write_text("")

    captured: dict[str, Any] = {}

    def fake_run_extract(**kwargs: Any) -> ExtractResult:
        captured.update(kwargs)
        return ExtractResult(
            run_id=1, watermark_before=0, watermark_after=0, chats_upserted=0,
            handles_upserted=0, messages_upserted=0, tapbacks_upserted=0,
            system_messages_skipped=0, attachments_upserted=0, link_previews_upserted=0,
            bodies_missing=0, dump_stderr_line_count=0, dry_run=True,
        )

    monkeypatch.setattr(cli_module, "run_extract", fake_run_extract)
    result = runner.invoke(app, ["extract", "--config", str(mocked_pg_env), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True
    assert "DRY RUN — nothing was written" in result.output


def _extract_via_cli(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str, result: Any = None
) -> tuple[dict[str, Any], str]:
    """Run `imsg extract` against a fake `run_extract`; return the kwargs
    it was called with and the command's output."""
    from imsg.stages.extract import ExtractResult

    data_root = _data_root_from_config(config_path)
    (data_root / "snapshots").mkdir(parents=True, exist_ok=True)
    (data_root / "snapshots" / "snapshot.db").write_text("")
    captured: dict[str, Any] = {}

    def fake_run_extract(**kwargs: Any) -> ExtractResult:
        captured.update(kwargs)
        if result is not None:
            return result  # type: ignore[no-any-return]
        return ExtractResult(
            run_id=1, watermark_before=0, watermark_after=0, chats_upserted=0,
            handles_upserted=0, messages_upserted=0, tapbacks_upserted=0,
            system_messages_skipped=0, attachments_upserted=0, link_previews_upserted=0,
            bodies_missing=0, dump_stderr_line_count=0,
            merge_mode=kwargs["merge_mode"],
        )

    monkeypatch.setattr(cli_module, "run_extract", fake_run_extract)
    invoked = runner.invoke(app, ["extract", "--config", str(config_path), *args])
    assert invoked.exit_code == 0, invoked.output
    return captured, invoked.output


def test_extract_of_the_pipeline_snapshot_is_live_when_it_can_only_be_the_live_database(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D12: only this machine's own live database may replace a non-empty
    value. With one configured source, and that source the live
    `chat.db`, the pipeline snapshot can only have come from it."""
    from imsg.stages.extract import MergeMode

    captured, output = _extract_via_cli(mocked_pg_env, monkeypatch)

    assert captured["merge_mode"] is MergeMode.LIVE
    assert "mode=live" in output


def test_extract_of_a_seed_file_is_a_seed(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from imsg.stages.extract import MergeMode

    seed = tmp_path / "recovered.db"
    seed.write_text("")
    captured, output = _extract_via_cli(
        mocked_pg_env, monkeypatch, "--snapshot", str(seed), "--source", "seed-2026"
    )

    assert captured["merge_mode"] is MergeMode.SEED
    assert "mode=seed" in output


def test_extract_of_the_pipeline_snapshot_is_a_seed_when_another_source_shares_it(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every configured source's S1 copy lands in the same
    `snapshots/snapshot.db`. Once a second source points at another Mac's
    database, `imsg extract` cannot tell whose copy is there, so it takes
    the rule that loses nothing -- even when asked for the live source."""
    from imsg.stages.extract import MergeMode

    studio = tmp_path / "studio-chat.db"
    studio.write_text("")
    text = mocked_pg_env.read_text()
    mocked_pg_env.write_text(
        text.replace("  sources:\n", f"  sources:\n    - name: studio\n      chat_db: {studio}\n", 1)
    )

    captured, _ = _extract_via_cli(mocked_pg_env, monkeypatch, "--source", "mini")

    assert captured["merge_mode"] is MergeMode.SEED


def test_extract_reports_every_table_with_fills_told_apart(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D12 rule 5. The 2026-09-22 dry run printed only the message counts
    and hid every chat and attachment rewrite."""
    from imsg.stages.extract import ExtractResult, MergeMode, UpsertCounts

    reported = ExtractResult(
        run_id=1, watermark_before=0, watermark_after=9, chats_upserted=2,
        handles_upserted=2, messages_upserted=3, tapbacks_upserted=1,
        system_messages_skipped=0, attachments_upserted=1, link_previews_upserted=0,
        bodies_missing=0, dump_stderr_line_count=0,
        merge_mode=MergeMode.SEED,
        chat_upserts=UpsertCounts(filled=1, unchanged=1),
        message_upserts=UpsertCounts(inserted=1, filled=1, newer_edit=1),
        attachment_upserts=UpsertCounts(unchanged=1),
        tapback_upserts=UpsertCounts(inserted=1),
    )
    _, output = _extract_via_cli(mocked_pg_env, monkeypatch, result=reported)

    assert "messages_upserted=3 (inserted=1 updated=2 unchanged=0)" in output
    table_lines = [line for line in output.splitlines() if "table=" in line]
    assert len(table_lines) == len(reported.table_counts())
    assert (
        "extract: table=chat inserted=0 filled=1 newer_edit=0 replaced=0 unchanged=1"
        in table_lines
    )
    assert (
        "extract: table=message inserted=1 filled=1 newer_edit=1 replaced=0 unchanged=0"
        in table_lines
    )
    assert any(line.startswith("extract: table=tapback inserted=1 ") for line in table_lines)
    assert "mode=seed" in output


def test_identity_wires_run_identity_and_warns_on_degraded_contacts(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.identity import ContactsImportOutcome, IdentityResult, InvariantReport

    def fake_run_identity(*, conn: Any, config: Any, **kw: Any) -> IdentityResult:
        return IdentityResult(
            source_handles_processed=1,
            persons_created=1,
            handles_created=1,
            messages_resolved=1,
            tapbacks_resolved=0,
            chat_participants_resolved=1,
            contacts=ContactsImportOutcome(
                attempted=True, contacts_loaded=0, degraded=True, degraded_reason="no TCC grant"
            ),
            invariant=InvariantReport(
                unresolved_message_senders=0,
                unresolved_tapback_senders=0,
                unresolved_chat_participants=0,
                owner_person_count=1,
            ),
        )

    monkeypatch.setattr(cli_module, "run_identity", fake_run_identity)
    result = runner.invoke(app, ["identity", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "invariant_ok=True" in result.output
    assert "no TCC grant" in result.output


def test_identity_rematch_stubs_wires_the_stage_and_prints_counts(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.identity_rematch import RematchStubsResult, StubOutcome

    captured: dict[str, Any] = {}

    def fake_rematch(*, conn: Any, config: Any, dry_run: bool = False, **kw: Any) -> RematchStubsResult:
        captured.update(dry_run=dry_run, region=config.identity.default_region)
        return RematchStubsResult(
            outcomes=(
                StubOutcome(
                    7, "+14155552671", (("+14155552671", "phone"),), "matched",
                    new_display_name="Alice Example", new_short_name="alice-example",
                ),
                StubOutcome(8, "+14155552672", (("+14155552672", "phone"),), "ambiguous"),
                StubOutcome(9, "24273", (("24273", "unknown"),), "unmatched"),
            ),
            contacts_loaded=2,
            dry_run=dry_run,
            chats_marked_dirty=3,
        )

    monkeypatch.setattr(cli_module, "run_rematch_stubs", fake_rematch)

    result = runner.invoke(app, ["identity", "rematch-stubs", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is False
    assert captured["region"] == "US"
    assert "identity: contacts loaded=2" in result.output
    assert "identity: rematched person 7 '+14155552671' -> 'Alice Example'" in result.output
    assert "rematch: stubs=3 matched=1 ambiguous=1 unmatched=1" in result.output
    assert "identity: chats marked for re-segmentation: 3" in result.output
    assert "run `imsg segment` and then `imsg embed`" in result.output
    assert "DRY RUN" not in result.output

    result = runner.invoke(
        app, ["identity", "rematch-stubs", "--config", str(mocked_pg_env), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True
    assert "rematch: stubs=3 matched=1 ambiguous=1 unmatched=1" in result.output
    assert "rematched person" not in result.output  # a dry run prints counts only
    assert "identity: chats marked for re-segmentation: 3" in result.output
    assert "DRY RUN — nothing was written" in result.output


def test_identity_rematch_stubs_help_says_segment_and_embed_must_follow() -> None:
    result = runner.invoke(app, ["identity", "rematch-stubs", "--help"])
    assert result.exit_code == 0, result.output
    assert "imsg segment" in result.output and "imsg embed" in result.output

    result = runner.invoke(app, ["identity", "apply-overrides", "--help"])
    assert result.exit_code == 0, result.output
    assert "imsg segment" in result.output and "imsg embed" in result.output


def _recording_connect(monkeypatch: pytest.MonkeyPatch) -> list[_FakePgConn]:
    """Like `mocked_pg_env`'s `connect`, but keeps every connection it hands
    out so a test can assert the command committed on it."""
    conns: list[_FakePgConn] = []

    def fake_connect(database: Any, **kw: Any) -> _FakePgConn:
        conn = _FakePgConn()
        conns.append(conn)
        return conn

    monkeypatch.setattr(cli_module, "connect", fake_connect)
    return conns


def test_identity_rename_wires_rename_person_and_prints_the_dirty_chat_count(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    conns = _recording_connect(monkeypatch)

    def fake_rename(conn: Any, **kw: Any) -> frozenset[int]:
        captured.update(kw)
        return frozenset({11, 12})

    monkeypatch.setattr(cli_module, "rename_person", fake_rename)

    result = runner.invoke(
        app,
        ["identity", "rename", "--person", "7", "--name", "Alice Example", "--short", "alice",
         "--config", str(mocked_pg_env)],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "person_id": 7, "display_name": "Alice Example", "short_name": "alice", "allow_owner": False,
    }
    assert conns[-1].committed == 1
    assert "identity: renamed person 7 to 'Alice Example'" in result.output
    assert "identity: chats marked for re-segmentation: 2" in result.output
    assert "run `imsg segment` and then `imsg embed`" in result.output
    assert "WARNING" not in result.output


def test_identity_rename_owner_needs_yes_owner(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the flag the library refuses the owner and the CLI relays
    that (exit 1, the flag named); with it the CLI warns, threads
    `allow_owner=True`, and reports what was marked."""
    from imsg.stages.identity import IdentityError

    captured: dict[str, Any] = {}

    def fake_rename(conn: Any, **kw: Any) -> frozenset[int]:
        captured.update(kw)
        if not kw["allow_owner"]:
            raise IdentityError(
                "refusing to rename person 3: it is the singleton owner person, which merge, "
                "apply-overrides and rematch-stubs never touch — pass --yes-owner "
                "(allow_owner=True) to rename it deliberately"
            )
        return frozenset()

    monkeypatch.setattr(cli_module, "rename_person", fake_rename)

    result = runner.invoke(
        app, ["identity", "rename", "--person", "3", "--name", "Jamie", "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 1
    assert captured["allow_owner"] is False
    assert "imsg: refusing to rename person 3" in result.output
    assert "--yes-owner" in result.output
    assert "renamed person" not in result.output

    result = runner.invoke(
        app,
        ["identity", "rename", "--person", "3", "--name", "Jamie", "--yes-owner",
         "--config", str(mocked_pg_env)],
    )
    assert result.exit_code == 0, result.output
    assert captured["allow_owner"] is True
    assert "identity: WARNING --yes-owner" in result.output
    assert "identity: renamed person 3 to 'Jamie'" in result.output
    assert "identity: chats marked for re-segmentation: 0" in result.output
    assert "run `imsg segment`" not in result.output  # nothing to re-segment, no nag


def test_identity_merge_and_assign_print_the_dirty_chat_count(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    conns = _recording_connect(monkeypatch)

    def fake_merge(conn: Any, **kw: Any) -> frozenset[int]:
        captured["merge"] = kw
        return frozenset({5})

    def fake_assign(conn: Any, **kw: Any) -> frozenset[int]:
        captured["assign"] = kw
        return frozenset({5, 6, 7})

    monkeypatch.setattr(cli_module, "merge_persons", fake_merge)
    monkeypatch.setattr(cli_module, "assign_handle", fake_assign)

    result = runner.invoke(
        app, ["identity", "merge", "--keep", "7", "--absorb", "8", "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 0, result.output
    assert captured["merge"] == {"keep_person_id": 7, "absorb_person_id": 8}
    assert conns[-1].committed == 1
    assert "identity: merged person 8 into 7" in result.output
    assert "identity: chats marked for re-segmentation: 1" in result.output
    assert "run `imsg segment` and then `imsg embed`" in result.output

    result = runner.invoke(
        app,
        ["identity", "assign", "--value", "+14155552671", "--kind", "phone", "--person", "7",
         "--config", str(mocked_pg_env)],
    )
    assert result.exit_code == 0, result.output
    assert captured["assign"] == {"normalized_value": "+14155552671", "kind": "phone", "person_id": 7}
    assert conns[-1].committed == 1
    assert "identity: assigned phone '+14155552671' to person 7" in result.output
    assert "identity: chats marked for re-segmentation: 3" in result.output


def test_identity_rematch_stubs_fails_loudly_without_contacts(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contacts is the whole point: no grant means a clean error and exit 1,
    never a counts line that reads as "looked, matched nothing"."""
    from imsg.stages.identity import ContactsAccessDeniedError

    def denied(**kw: Any) -> Any:
        raise ContactsAccessDeniedError(
            "rematch-stubs needs Contacts and could not use it; nothing was changed. "
            "Contacts access is not authorized (CNAuthorizationStatus=0)"
        )

    monkeypatch.setattr(cli_module, "run_rematch_stubs", denied)
    result = runner.invoke(
        app, ["identity", "rematch-stubs", "--config", str(mocked_pg_env), "--dry-run"]
    )
    assert result.exit_code == 1
    assert "imsg: " in result.output and "Contacts access is not authorized" in result.output
    assert "rematch:" not in result.output
    assert "DRY RUN" not in result.output
    assert "Traceback" not in result.output


def _write_overrides_fixture(path: Path) -> None:
    """A tiny, fictional decisions file (schema: imsg.stages.identity_overrides)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "note": "fixture",
                "decided": "2026-08-15",
                "overrides": [
                    {"value": "+14155552671", "kind": "phone", "name": "Alice Example", "was": []},
                    {"value": "24273", "kind": "unknown", "name": "Acme Bank Alerts", "was": [], "why": "alerts"},
                ],
            }
        )
    )


def _clean_invariant() -> Any:
    from imsg.stages.identity import InvariantReport

    return InvariantReport(
        unresolved_message_senders=0,
        unresolved_tapback_senders=0,
        unresolved_chat_participants=0,
        owner_person_count=1,
    )


def test_identity_apply_overrides_wires_the_replay_and_prints_the_summary(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.identity_overrides import ApplyOverridesResult, OverrideOutcome

    overrides_path = _data_root_from_config(mocked_pg_env) / "private" / "identity-overrides.json"
    _write_overrides_fixture(overrides_path)
    captured: dict[str, Any] = {}

    def fake_apply(conn: Any, *, overrides: Any, default_region: str, dry_run: bool, force: bool) -> Any:
        captured.update(overrides=overrides, default_region=default_region, dry_run=dry_run, force=force)
        return ApplyOverridesResult(
            outcomes=(
                OverrideOutcome(overrides[0], "applied", "renamed person 7 '+14155552671' -> 'Alice Example'"),
                OverrideOutcome(overrides[1], "unmatched", "no handle in the index for unknown '24273'"),
            ),
            invariant=_clean_invariant(),
            dry_run=dry_run,
            chats_marked_dirty=1,
        )

    monkeypatch.setattr(cli_module, "apply_overrides", fake_apply)

    result = runner.invoke(
        app, ["identity", "apply-overrides", str(overrides_path), "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 0, result.output
    assert [o.name for o in captured["overrides"]] == ["Alice Example", "Acme Bank Alerts"]
    assert captured["default_region"] == "US"
    assert captured["dry_run"] is False
    assert captured["force"] is False
    assert "identity: 2 decision(s) from" in result.output
    assert "identity: applied renamed person 7" in result.output
    assert "identity: unmatched no handle in the index" in result.output
    assert "identity: invariant" in result.output and "ok=True" in result.output
    assert "identity: applied=1 already=0 unmatched=1 conflicts=0" in result.output
    assert "identity: chats marked for re-segmentation: 1" in result.output
    assert "run `imsg segment` and then `imsg embed`" in result.output
    assert "DRY RUN" not in result.output

    result = runner.invoke(
        app,
        ["identity", "apply-overrides", str(overrides_path), "--config", str(mocked_pg_env),
         "--dry-run", "--force"],
    )
    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True
    assert captured["force"] is True
    assert "DRY RUN — nothing was written" in result.output


def test_identity_apply_overrides_rejects_a_bad_file_before_touching_the_database(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides_path = _data_root_from_config(mocked_pg_env) / "private" / "identity-overrides.json"
    overrides_path.parent.mkdir(parents=True)
    overrides_path.write_text(json.dumps({"overrides": [{"value": "+14155552671", "kind": "phone"}]}))

    def must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("apply_overrides must not be called for an invalid file")

    monkeypatch.setattr(cli_module, "apply_overrides", must_not_run)
    result = runner.invoke(
        app, ["identity", "apply-overrides", str(overrides_path), "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 1
    assert "imsg: " in result.output and "overrides[0]: 'name' must be" in result.output
    assert "Traceback" not in result.output


def test_identity_export_overrides_refuses_a_path_outside_data_root(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("export_overrides must not be called for a path outside data_root")

    monkeypatch.setattr(cli_module, "export_overrides", must_not_run)
    outside = tmp_path / "elsewhere" / "identity-overrides.json"
    result = runner.invoke(
        app, ["identity", "export-overrides", str(outside), "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 1
    assert "refusing to write" in result.output and "non-negotiable #2" in result.output
    assert not outside.exists()


def test_identity_export_overrides_writes_the_file_and_merges_the_existing_one(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.identity_overrides import (
        ExportOverridesResult,
        IdentityOverride,
        IdentityOverridesFile,
        load_overrides,
    )

    overrides_path = _data_root_from_config(mocked_pg_env) / "private" / "identity-overrides.json"
    _write_overrides_fixture(overrides_path)
    captured: dict[str, Any] = {}

    def fake_export(conn: Any, *, default_region: str, previous: Any) -> ExportOverridesResult:
        captured.update(default_region=default_region, previous=previous)
        file = IdentityOverridesFile(
            overrides=(
                IdentityOverride(value="+14155552671", kind="phone", name="Alice Example"),
                IdentityOverride(value="24273", kind="unknown", name="Acme Bank Alerts", why="alerts"),
            ),
            note="fixture",
            decided="2026-09-15",
        )
        return ExportOverridesResult(file=file, exported=1, carried_forward=1)

    monkeypatch.setattr(cli_module, "export_overrides", fake_export)
    result = runner.invoke(
        app, ["identity", "export-overrides", str(overrides_path), "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 0, result.output
    assert captured["default_region"] == "US"
    assert [o.name for o in captured["previous"].overrides] == ["Alice Example", "Acme Bank Alerts"]
    assert "identity: exported 1 decision(s) to" in result.output
    assert "(carried_forward=1, total=2)" in result.output
    assert load_overrides(overrides_path).decided == "2026-09-15"


def test_segment_rebuild_requires_chat(mocked_pg_env: Path) -> None:
    result = runner.invoke(app, ["segment", "--rebuild", "--config", str(mocked_pg_env)])
    assert result.exit_code == 2


def test_segment_wires_run_segment(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.segment.models import SegmentationRunReport

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("segment this")

    captured: dict[str, Any] = {}

    def fake_run_segment(conn: Any, config: Any, provider: Any, prompt_bytes: bytes, **kw: Any) -> list[SegmentationRunReport]:
        captured["prompt_bytes"] = prompt_bytes
        captured["kw"] = kw
        return [SegmentationRunReport(chat_id=1, segments_written=3)]

    monkeypatch.setattr(cli_module, "run_segment", fake_run_segment)
    result = runner.invoke(app, ["segment", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "3 segment(s) written" in result.output
    assert captured["prompt_bytes"] == b"segment this"
    assert captured["kw"]["chat_ids"] is None


def _hide_the_shipped_prompts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Simulate a checkout without `prompts/`: the data_root copy is then
    the only candidate, so the "missing prompt" error paths are reachable."""
    monkeypatch.setattr(factory_module, "default_prompt_root", lambda: tmp_path / "no-checkout")


def test_segment_missing_boundary_prompt_exits_cleanly(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _hide_the_shipped_prompts(monkeypatch, tmp_path)
    result = runner.invoke(app, ["segment", "--config", str(mocked_pg_env)])
    assert result.exit_code == 1
    assert "boundary prompt not found" in result.output


def test_segment_falls_back_to_the_repo_shipped_boundary_prompt(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh data_root has no prompts yet. The shipped
    `prompts/segment_boundaries.txt` is used — its exact bytes, since they
    feed `seg_config_hash` — and the run says so."""
    from imsg.segment.models import SegmentationRunReport

    shipped = factory_module.default_prompt_root() / "prompts" / "segment_boundaries.txt"
    assert shipped.is_file()
    captured: dict[str, Any] = {}

    def fake_run_segment(conn: Any, config: Any, provider: Any, prompt_bytes: bytes, **kw: Any) -> list[SegmentationRunReport]:
        captured["prompt_bytes"] = prompt_bytes
        return [SegmentationRunReport(chat_id=1, segments_written=1)]

    monkeypatch.setattr(cli_module, "run_segment", fake_run_segment)
    result = runner.invoke(app, ["segment", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert captured["prompt_bytes"] == shipped.read_bytes()
    assert f"segmentation prompt: {shipped} (repo-shipped default)" in result.output


def test_segment_prefers_the_data_root_boundary_prompt_and_says_so(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.segment.models import SegmentationRunReport

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("operator override")
    captured: dict[str, Any] = {}

    def fake_run_segment(conn: Any, config: Any, provider: Any, prompt_bytes: bytes, **kw: Any) -> list[SegmentationRunReport]:
        captured["prompt_bytes"] = prompt_bytes
        return [SegmentationRunReport(chat_id=1, segments_written=1)]

    monkeypatch.setattr(cli_module, "run_segment", fake_run_segment)
    result = runner.invoke(app, ["segment", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert captured["prompt_bytes"] == b"operator override"
    assert f"segmentation prompt: {prompt_path} (data_root)" in result.output


def test_segment_rebuild_wires_run_segment_for_chat(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.segment.models import SegmentationRunReport

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("x")

    captured: dict[str, Any] = {}

    def fake_run_segment_for_chat(
        conn: Any, chat_id: int, config: Any, provider: Any, prompt_bytes: bytes, **kw: Any
    ) -> SegmentationRunReport:
        captured["chat_id"] = chat_id
        captured["kw"] = kw
        return SegmentationRunReport(chat_id=chat_id, segments_written=2, segments_deleted=1)

    monkeypatch.setattr(cli_module, "run_segment_for_chat", fake_run_segment_for_chat)
    result = runner.invoke(
        app, ["segment", "--rebuild", "--chat", "42", "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 0, result.output
    assert "chat 42 rebuilt" in result.output
    assert captured["chat_id"] == 42
    from imsg.segment.pipeline import REBUILD_ALL_SENTINEL

    assert captured["kw"]["earliest_changed_at"] == REBUILD_ALL_SENTINEL


def test_embed_wires_run_embed_and_fts_sync(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.embed.fts.sync import SyncReport
    from imsg.embed.pipeline import EmbedRunReport

    def fake_run_embed(conn: Any, provider: Any, **kw: Any) -> EmbedRunReport:
        return EmbedRunReport(segments_embedded=2, chunks_embedded=1, attachments_embedded=0)

    def fake_sync_fts(pg_conn: Any, fts_conn: Any, **kw: Any) -> SyncReport:
        return SyncReport(events_applied=3, upserts=2, deletes=1)

    monkeypatch.setattr(cli_module, "run_embed", fake_run_embed)
    monkeypatch.setattr(cli_module, "sync_fts", fake_sync_fts)
    result = runner.invoke(app, ["embed", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "segments_embedded=2" in result.output
    assert "events_applied=3" in result.output


def test_sync_missing_boundary_prompt_exits_cleanly(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _hide_the_shipped_prompts(monkeypatch, tmp_path)
    result = runner.invoke(app, ["sync", "--config", str(mocked_pg_env)])
    assert result.exit_code == 1
    assert "boundary prompt not found" in result.output


def test_sync_wires_run_sync_all_sources(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.extract import ExtractResult
    from imsg.stages.identity import ContactsImportOutcome, IdentityResult, InvariantReport
    from imsg.stages.sync import SyncResult

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("x")

    captured: dict[str, Any] = {}

    def fake_run_sync_all_sources(**kwargs: Any) -> list[SyncResult]:
        captured.update(kwargs)
        extract = ExtractResult(
            run_id=1, watermark_before=0, watermark_after=1, chats_upserted=0,
            handles_upserted=0, messages_upserted=7, tapbacks_upserted=0,
            system_messages_skipped=0, attachments_upserted=0, link_previews_upserted=0,
            bodies_missing=0, dump_stderr_line_count=0,
        )
        identity = IdentityResult(
            source_handles_processed=0, persons_created=0, handles_created=0,
            messages_resolved=0, tapbacks_resolved=0, chat_participants_resolved=0,
            contacts=ContactsImportOutcome(attempted=False, contacts_loaded=0, degraded=False, degraded_reason=None),
            invariant=InvariantReport(0, 0, 0, 1),
        )
        return [
            SyncResult(
                source_name="mini", snapshot=None, extract=extract, identity=identity,
                segment_ran=True, embed_ran=True,
            )
        ]

    monkeypatch.setattr(cli_module, "run_sync_all_sources", fake_run_sync_all_sources)
    result = runner.invoke(app, ["sync", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "messages_upserted=7" in result.output
    assert "segment_ran=True" in result.output
    assert callable(captured["segment_fn"])
    assert callable(captured["embed_fn"])
    assert captured["dry_run"] is False
    assert "DRY RUN" not in result.output


def test_sync_dry_run_passes_the_flag_and_prints_the_marker(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.sync import SyncResult

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("x")

    captured: dict[str, Any] = {}

    def fake_run_sync_all_sources(**kwargs: Any) -> list[SyncResult]:
        captured.update(kwargs)
        return [
            SyncResult(
                source_name="mini", snapshot=None, extract=None, identity=None,
                segment_ran=False, embed_ran=False, dry_run=True,
                note="dry run stopped after S1 snapshot — no real snapshot file exists yet",
            )
        ]

    monkeypatch.setattr(cli_module, "run_sync_all_sources", fake_run_sync_all_sources)
    result = runner.invoke(app, ["sync", "--config", str(mocked_pg_env), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True
    assert "dry run stopped after S1 snapshot" in result.output
    assert "DRY RUN — nothing was written" in result.output


def test_enrich_wires_claim_and_process(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.enrich.queue import EnrichmentTask

    tasks = [EnrichmentTask(attachment_id=1, kind="ocr", attempts=0), EnrichmentTask(attachment_id=2, kind="caption", attempts=0)]
    monkeypatch.setattr(cli_module, "claim_tasks", lambda conn, **kw: tasks)
    monkeypatch.setattr(cli_module, "process_one_task", lambda conn, config, providers, task: "done")

    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "claimed=2" in result.output
    assert "done=2" in result.output
    assert "DRY RUN" not in result.output


def test_enrich_dry_run_uses_preview_claimable_tasks(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.enrich.queue import EnrichPreviewReport

    def boom_claim(conn: Any, **kw: Any) -> Any:
        raise AssertionError("claim_tasks must not be called in dry-run mode")

    def boom_process(conn: Any, config: Any, providers: Any, task: Any) -> str:
        raise AssertionError("process_one_task must not be called in dry-run mode")

    monkeypatch.setattr(cli_module, "claim_tasks", boom_claim)
    monkeypatch.setattr(cli_module, "process_one_task", boom_process)
    monkeypatch.setattr(
        cli_module,
        "preview_claimable_tasks",
        lambda conn, **kw: EnrichPreviewReport(total=3, by_kind={"ocr": 2, "caption": 1}),
    )

    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "claimable=3" in result.output
    assert "ocr=2" in result.output
    assert "caption=1" in result.output
    assert "DRY RUN — nothing was written" in result.output


def test_enrich_retry_failed_resets_rows_before_claiming(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed: list[str] = []

    class _FakeCursor:
        def __enter__(self) -> _FakeCursor:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def execute(self, sql: str, *a: object) -> None:
            executed.append(sql)

        def fetchone(self) -> tuple[object, ...]:
            # The yield gate's advisory-lock probe: True means nobody
            # holds the query-in-flight lock, which is this test's world.
            return (True,)

    class _FakeTxnConn(_FakePgConn):
        def transaction(self) -> Any:
            from contextlib import contextmanager

            @contextmanager
            def _cm() -> Any:
                yield None

            return _cm()

        def cursor(self) -> _FakeCursor:
            return _FakeCursor()

    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakeTxnConn())
    monkeypatch.setattr(cli_module, "claim_tasks", lambda conn, **kw: [])
    result = runner.invoke(
        app, ["enrich", "--retry-failed", "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 0, result.output
    assert any("state = 'pending'" in sql for sql in executed)


def test_backfill_attachments_wires_run_backfill(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.backfill.pipeline import BackfillRunReport

    captured: dict[str, Any] = {}

    def fake_run_backfill(conn: Any, data_root: Path, attachments_root: Path, **kw: Any) -> BackfillRunReport:
        captured["attachments_root"] = attachments_root
        captured["kw"] = kw
        return BackfillRunReport(considered=5, materialized=4, errored=1, marked_missing=0)

    monkeypatch.setattr(cli_module, "run_backfill", fake_run_backfill)
    result = runner.invoke(
        app, ["backfill-attachments", "--yes-full-run", "--config", str(mocked_pg_env)]
    )
    assert result.exit_code == 0, result.output
    assert "materialized=4" in result.output
    assert captured["attachments_root"].name == "Attachments"
    assert captured["kw"]["yes_full_run"] is True
    assert captured["kw"]["retry_failed"] is False


def test_backfill_attachments_retry_failed_flag_and_reclassification_line(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.backfill.pipeline import BackfillRunReport

    captured: dict[str, Any] = {}

    def fake_run_backfill(conn: Any, data_root: Path, attachments_root: Path, **kw: Any) -> BackfillRunReport:
        captured["kw"] = kw
        return BackfillRunReport(
            considered=7, materialized=2, errored=1, marked_missing=1, marked_unsupported=3,
            reclassified_unsupported=435, reclassified_missing_no_source=1362, retry_reset=445,
        )

    monkeypatch.setattr(cli_module, "run_backfill", fake_run_backfill)
    result = runner.invoke(
        app,
        ["backfill-attachments", "--retry-failed", "--yes-full-run", "--config", str(mocked_pg_env)],
    )
    assert result.exit_code == 0, result.output
    assert captured["kw"]["retry_failed"] is True
    # Every terminal state a run can produce is printed, from the same
    # counters the database was written from.
    assert "considered=7 materialized=2 unsupported=3 errored=1 marked_missing=1" in result.output
    assert "unsupported=435 missing_no_source_path=1362; retry_failed_reset=445" in result.output


# --------------------------------------------------------------------------
# install-agents (SPEC §5.5) — real, no live Postgres/mount required
# --------------------------------------------------------------------------


def _fake_which(name: str) -> str | None:
    return {
        "imsg": "/usr/local/bin/imsg",
        "postgres": "/opt/homebrew/bin/postgres",
        "cloudflared": "/opt/homebrew/bin/cloudflared",
    }.get(name)


_FIXED_DATA_ROOT = "/Volumes/Data-Encrypted/imsgindex"
"""Deliberately NOT `tmp_path`-derived: pytest's own tmp dirs
(`/…/pytest-of-<local-username>/…`) embed the real local OS username,
which would make the leak-substring check below fire for a reason that
has nothing to do with `install-agents`'s own output."""


def test_install_agents_writes_all_seven_plists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plistlib
    import shutil

    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, Path(_FIXED_DATA_ROOT), messages_dir)

    monkeypatch.setattr(shutil, "which", _fake_which)

    dest = tmp_path / "LaunchAgents"
    result = runner.invoke(
        app, ["install-agents", "--config", str(config_path), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output

    expected_labels = {
        "com.imsgindex.pg",
        "com.imsgindex.sync",
        "com.imsgindex.enrich",
        "com.imsgindex.mcp-public",
        "com.imsgindex.tunnel",
        "com.imsgindex.report",
        "com.imsgindex.backup",
    }
    written = {p.stem for p in dest.glob("*.plist")}
    assert written == expected_labels

    # Note: this end-to-end CLI invocation necessarily writes a real
    # config.yaml under pytest's own tmp dir and passes that real path
    # via --config, so scanning *this* test's rendered output for
    # forbidden substrings would just be checking pytest's tmp-dir
    # naming, not this command's own behavior — the substring leak
    # check that actually matters (given a config_path this test does
    # not control) lives in test_launchagents.py's
    # `render_agent_plists` unit tests, which hold every input path
    # fixed and fictional.
    for plist_path in dest.glob("*.plist"):
        content = plist_path.read_bytes()
        parsed = plistlib.loads(content)  # must round-trip as valid XML plist
        assert parsed["Label"] == plist_path.stem
        assert isinstance(parsed["ProgramArguments"], list) and parsed["ProgramArguments"]

    pg_plist = plistlib.loads((dest / "com.imsgindex.pg.plist").read_bytes())
    assert pg_plist["KeepAlive"] is True
    assert "5433" in " ".join(pg_plist["ProgramArguments"])

    sync_plist = plistlib.loads((dest / "com.imsgindex.sync.plist").read_bytes())
    assert sync_plist["StartInterval"] == 900

    enrich_plist = plistlib.loads((dest / "com.imsgindex.enrich.plist").read_bytes())
    assert isinstance(enrich_plist["StartCalendarInterval"], list)
    assert len(enrich_plist["StartCalendarInterval"]) > 1

    report_plist = plistlib.loads((dest / "com.imsgindex.report.plist").read_bytes())
    assert report_plist["StartCalendarInterval"] == {"Weekday": 1, "Hour": 8, "Minute": 0}

    backup_plist = plistlib.loads((dest / "com.imsgindex.backup.plist").read_bytes())
    assert backup_plist["StartCalendarInterval"] == {"Hour": 4, "Minute": 0}


def test_install_agents_missing_postgres_binary_exits_cleanly(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = runner.invoke(
        app,
        ["install-agents", "--config", str(cli_config), "--dest", str(tmp_path / "LaunchAgents")],
    )
    assert result.exit_code == 1
    assert "postgres" in result.output


def test_install_agents_missing_cloudflared_binary_exits_cleanly(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda name: "/opt/homebrew/bin/postgres" if name == "postgres" else None
    )
    result = runner.invoke(
        app,
        ["install-agents", "--config", str(cli_config), "--dest", str(tmp_path / "LaunchAgents")],
    )
    assert result.exit_code == 1
    assert "cloudflared" in result.output


def test_mcp_local_disabled_in_config_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")
    live_chat_db = messages_dir / "chat.db"
    live_chat_db.write_text("")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
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
models:
  backend: fake
mcp:
  local:
    enabled: false
  public:
    scope: allowlist
export:
  gcp_project: example-project
  gcs_bucket: example-bucket
  data_store_id: example-datastore
"""
    )

    result = runner.invoke(app, ["mcp", "local", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "mcp.local.enabled is false" in result.output


def test_mcp_local_wires_server_and_runs_it(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    from imsg.mcp.tools.local_server import LocalMcpServer, run_local_server

    captured: dict[str, Any] = {}

    def fake_anyio_run(func: Any, *args: Any) -> None:
        # `anyio.run` is itself a synchronous, blocking call that drives the
        # event loop internally — the real thing never returns until the
        # server stops, so the fake must not be a coroutine function either.
        # Patching the `anyio` module object directly (not `cli_module.anyio`)
        # is equivalent — `cli.py`'s `import anyio` binds the same module
        # object from `sys.modules`.
        captured["func"] = func
        captured["args"] = args

    monkeypatch.setattr(anyio, "run", fake_anyio_run)
    result = runner.invoke(app, ["mcp", "local", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert captured["func"] is run_local_server
    local = captured["args"][0]
    assert isinstance(local, LocalMcpServer)


# --------------------------------------------------------------------------
# mcp public (SPEC §10.4) — mirrors the mcp local tests immediately above,
# with uvicorn.run monkeypatched instead of anyio.run.
# --------------------------------------------------------------------------


def test_mcp_public_disabled_in_config_exits_cleanly(mocked_pg_env: Path) -> None:
    # `cli_config`'s base fixture never sets `mcp.public.enabled` — the
    # schema default (`False`) applies, matching "flipped on at Phase 6,
    # never before" (SPEC §6).
    result = runner.invoke(app, ["mcp", "public", "--config", str(mocked_pg_env)])
    assert result.exit_code == 1
    assert "mcp.public.enabled is false" in result.output


def _write_public_enabled_config(path: Path, data_root: Path, messages_dir: Path) -> None:
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
models:
  backend: fake
mcp:
  public:
    enabled: true
    external_url: https://mcp.fictional.example/mcp
    allowed_origins: [https://vertexaisearch.fictional.example]
    allowed_hosts: [mcp.fictional.example]
    scope: allowlist
    oauth:
      client_id: fictional-client-id.apps.example
      owner_subject: env:IMSG_TEST_OWNER_SUBJECT
export:
  gcp_project: example-project
  gcs_bucket: example-bucket
  data_store_id: example-datastore
"""
    )


@pytest.fixture
def mocked_pg_env_public_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Same DB/mount mocking as `mocked_pg_env`, but with `mcp.public`
    fully enabled — `mocked_pg_env` itself can't be reused here since it
    is bound to `cli_config`'s fixed (public-disabled) config content."""
    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", "300000000000000000009")
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)

    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")

    config_path = tmp_path / "config.yaml"
    _write_public_enabled_config(config_path, data_root, messages_dir)

    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakePgConn())
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    return config_path


def test_mcp_public_wires_server_and_runs_it(
    mocked_pg_env_public_enabled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    from imsg.mcp.tools.public_server import TransportGuardASGIApp

    captured: dict[str, Any] = {}

    def fake_uvicorn_run(app_arg: Any, **kw: Any) -> None:
        # Real `uvicorn.run` blocks until the server stops; the fake must
        # not actually bind a socket or serve anything in a unit test.
        captured["app"] = app_arg
        captured["kw"] = kw

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    result = runner.invoke(
        app, ["mcp", "public", "--config", str(mocked_pg_env_public_enabled)]
    )
    assert result.exit_code == 0, result.output
    assert isinstance(captured["app"], TransportGuardASGIApp)
    assert captured["kw"]["host"] == "127.0.0.1"
    assert captured["kw"]["port"] == 8700


# --------------------------------------------------------------------------
# Regression, 2026-08-14 — the persistence defect.
#
# `connect()` is autocommit=False and `verify_data_directory` issues a
# `SHOW`, so a transaction is open before any stage runs. Every
# `conn.transaction()` then nested as a SAVEPOINT, and since no command
# called `conn.commit()`, `finally: conn.close()` discarded every write
# while the command printed its success line and exited 0. A full extract
# reported messages_upserted=655494 and left n_live_tup=0.
#
# The fix is in `_connect_and_verify_or_die`; this pins it there so a
# future refactor cannot quietly drop it again.
# --------------------------------------------------------------------------


def test_connect_and_verify_rejects_a_connection_left_in_a_transaction(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structural guarantee: stages must never receive a connection with an
    open transaction, because their `conn.transaction()` blocks would nest as
    SAVEPOINTs and be discarded at close() — silently, while reporting success.

    `connect()` sets autocommit=True so reads leave the connection idle. This
    pins the assertion that catches a regression of that setting.
    """
    import psycopg

    conn = _FakePgConn(transaction_status=psycopg.pq.TransactionStatus.INTRANS)
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: conn)
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda c, data_root: Path(str(data_root))
    )
    cfg = cli_module._load_config_or_die(cli_config)
    with pytest.raises(typer.Exit) as exc:
        cli_module._connect_and_verify_or_die(cfg)
    assert exc.value.exit_code == 1
    assert conn.closed


def test_connect_and_verify_accepts_an_idle_connection(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _FakePgConn()
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: conn)
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda c, data_root: Path(str(data_root))
    )
    cfg = cli_module._load_config_or_die(cli_config)
    assert cli_module._connect_and_verify_or_die(cfg) is conn
    assert not conn.closed


def test_connect_defaults_to_autocommit() -> None:
    """autocommit=True is the whole fix — pin the default so it cannot drift."""
    import inspect

    from imsg.db.connection import connect as real_connect

    assert inspect.signature(real_connect).parameters["autocommit"].default is True


# --------------------------------------------------------------------------
# The one-shot seed path (SPEC §8 S7): `--snapshot` bypasses S1 and feeds a
# prepared database straight to S2.
#
# The guard matters more than the feature. A seed advances the ROWID watermark
# of whatever source it lands on, and ROWIDs are meaningless across database
# files — seed a file whose ROWIDs run past the live source's and every real
# message below the new watermark is skipped forever, with no error. So the
# seed must carry its own --source, and that source must not be one the
# pipeline snapshots from a live chat.db.
# --------------------------------------------------------------------------


def test_seed_requires_a_source(mocked_pg_env: Path, tmp_path: Path) -> None:
    seed = tmp_path / "seed.db"
    seed.write_text("")
    result = runner.invoke(
        app, ["extract", "--config", str(mocked_pg_env), "--snapshot", str(seed)]
    )
    assert result.exit_code == 1
    assert "--snapshot requires --source" in result.output


def test_seed_refuses_a_configured_live_source(mocked_pg_env: Path, tmp_path: Path) -> None:
    """`mini` is the configured sync.sources entry in the test config — the
    exact shape of the collision this guard exists to prevent."""
    seed = tmp_path / "seed.db"
    seed.write_text("")
    result = runner.invoke(
        app,
        ["extract", "--config", str(mocked_pg_env), "--snapshot", str(seed), "--source", "mini"],
    )
    assert result.exit_code == 1
    assert "configured sync.sources entry" in result.output
    assert "skipped forever" in result.output


def test_seed_refuses_a_missing_file(mocked_pg_env: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "extract", "--config", str(mocked_pg_env),
            "--snapshot", str(tmp_path / "nope.db"), "--source", "seed-2026",
        ],
    )
    assert result.exit_code == 1
    assert "--snapshot file not found" in result.output


def _live_chat_db_from_config(config_path: Path) -> Path:
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("live_chat_db:"):
            return Path(stripped.split(":", 1)[1].strip())
    raise AssertionError("live_chat_db not found in test config fixture")


def _refuse_to_extract(**kwargs: Any) -> Any:
    raise AssertionError("run_extract must not be reached for a refused seed")


def test_seed_refuses_the_live_chat_db_itself(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-negotiable #1: the live chat.db is read exactly once, by S1's
    SQLite backup. `--snapshot` handing it straight to S2 would be a
    second reader of the live file — refused by name, before anything
    is opened."""
    live = _live_chat_db_from_config(mocked_pg_env)
    assert live.is_file()
    monkeypatch.setattr(cli_module, "run_extract", _refuse_to_extract)
    result = runner.invoke(
        app,
        ["extract", "--config", str(mocked_pg_env), "--snapshot", str(live), "--source", "seed-2026"],
    )
    assert result.exit_code == 1
    assert "is the live Messages database" in result.output
    assert "non-negotiable #1" in result.output


def test_seed_refuses_a_symlink_to_the_live_chat_db(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = _live_chat_db_from_config(mocked_pg_env)
    link = tmp_path / "innocent-looking-seed.db"
    link.symlink_to(live)
    monkeypatch.setattr(cli_module, "run_extract", _refuse_to_extract)
    result = runner.invoke(
        app,
        ["extract", "--config", str(mocked_pg_env), "--snapshot", str(link), "--source", "seed-2026"],
    )
    assert result.exit_code == 1
    assert "is the live Messages database" in result.output


def test_seed_refuses_anything_inside_the_live_messages_directory(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy placed next to the live database is still inside the
    directory S1 alone may touch — and Messages keeps -wal/-shm sidecars
    there that a second SQLite reader would interact with."""
    live = _live_chat_db_from_config(mocked_pg_env)
    beside = live.parent / "chat-copy.db"
    beside.write_text("")
    monkeypatch.setattr(cli_module, "run_extract", _refuse_to_extract)
    result = runner.invoke(
        app,
        ["extract", "--config", str(mocked_pg_env), "--snapshot", str(beside), "--source", "seed-2026"],
    )
    assert result.exit_code == 1
    assert "inside the live Messages directory" in result.output
    assert "non-negotiable #1" in result.output


def test_sync_seed_refuses_the_live_chat_db(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _live_chat_db_from_config(mocked_pg_env)

    def refuse(**kwargs: Any) -> Any:
        raise AssertionError("run_sync must not be reached for a refused seed")

    monkeypatch.setattr(cli_module, "run_sync", refuse)
    result = runner.invoke(
        app,
        ["sync", "--config", str(mocked_pg_env), "--snapshot", str(live), "--source", "seed-2026"],
    )
    assert result.exit_code == 1
    assert "is the live Messages database" in result.output


def test_seed_refuses_a_hard_link_to_the_live_chat_db(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hard link resolves to a different string for the same inode — the
    shape of the macOS `/System/Volumes/Data/…` firmlink alias too, which
    cannot be built under tmp_path. Resolved-path equality alone would let
    it through; `is_same_file` compares inodes as well."""
    live = _live_chat_db_from_config(mocked_pg_env)
    link = tmp_path / "innocent-looking-seed.db"
    os.link(live, link)
    monkeypatch.setattr(cli_module, "run_extract", _refuse_to_extract)
    result = runner.invoke(
        app,
        ["extract", "--config", str(mocked_pg_env), "--snapshot", str(link), "--source", "seed-2026"],
    )
    assert result.exit_code == 1
    assert "is the live Messages database" in result.output


def test_extract_seed_uses_the_given_file_and_source(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The positive case, and the reason it is checked by value: the whole
    point is that S2 reads the seed and NOT `snapshots/snapshot.db`."""
    from imsg.stages.extract import ExtractResult

    data_root = _data_root_from_config(mocked_pg_env)
    pipeline_snapshot = data_root / "snapshots" / "snapshot.db"
    pipeline_snapshot.parent.mkdir(parents=True, exist_ok=True)
    pipeline_snapshot.write_text("")  # present, and must still be ignored

    seed = tmp_path / "corpus-merged.db"
    seed.write_text("")

    captured: dict[str, Any] = {}

    def fake_run_extract(**kwargs: Any) -> ExtractResult:
        captured.update(kwargs)
        return ExtractResult(
            run_id=1, watermark_before=0, watermark_after=9195, chats_upserted=0,
            handles_upserted=0, messages_upserted=9195, tapbacks_upserted=0,
            system_messages_skipped=0, attachments_upserted=0, link_previews_upserted=0,
            bodies_missing=0, dump_stderr_line_count=0,
        )

    monkeypatch.setattr(cli_module, "run_extract", fake_run_extract)
    result = runner.invoke(
        app,
        [
            "extract", "--config", str(mocked_pg_env),
            "--snapshot", str(seed), "--source", "recovered-2026",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["snapshot_path"] == seed
    assert captured["snapshot_path"] != pipeline_snapshot
    assert captured["source_name"] == "recovered-2026"


def test_sync_seed_skips_s1_and_targets_one_source(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SPEC §8 S7's shape: `imsg sync --source <seed> --snapshot <path>`
    forwards to `run_sync(snapshot_override=...)`, which skips S1."""
    from imsg.stages.sync import SyncResult

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("x")

    seed = tmp_path / "corpus-merged.db"
    seed.write_text("")
    captured: dict[str, Any] = {}

    def fake_run_sync(**kwargs: Any) -> SyncResult:
        captured.update(kwargs)
        return SyncResult(
            source_name=kwargs["source_name"], snapshot=None, extract=None,
            identity=None, segment_ran=False, embed_ran=False, note="seeded",
        )

    def fail_all_sources(**kwargs: Any) -> list[SyncResult]:
        raise AssertionError("--source must not fan out to every configured source")

    monkeypatch.setattr(cli_module, "run_sync", fake_run_sync)
    monkeypatch.setattr(cli_module, "run_sync_all_sources", fail_all_sources)
    result = runner.invoke(
        app,
        [
            "sync", "--config", str(mocked_pg_env),
            "--snapshot", str(seed), "--source", "recovered-2026",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["snapshot_override"] == seed
    assert captured["source_name"] == "recovered-2026"


def test_sync_without_overrides_still_syncs_every_source(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.stages.sync import SyncResult

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("x")

    called: dict[str, Any] = {}

    def fake_all(**kwargs: Any) -> list[SyncResult]:
        called.update(kwargs)
        return []

    monkeypatch.setattr(cli_module, "run_sync_all_sources", fake_all)
    result = runner.invoke(app, ["sync", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert called, "the no-override path must still fan out to all sources"


# --------------------------------------------------------------------------
# Model providers come from imsg.providers.factory (`models.backend`);
# every provider-building command prints `models: backend=<real|fake>`
# --------------------------------------------------------------------------


def _patch_status_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module, "check_mount", lambda data_root: MountCheck(ok=True, reason=None, info=None)
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
    monkeypatch.setattr(
        cli_module,
        "check_postgres",
        lambda config: PostgresCheck(reachable=True, cluster_fingerprint_ok=True, reason=None),
    )
    monkeypatch.setattr(cli_module, "disk_free_bytes", lambda path: 1)
    monkeypatch.setattr(
        cli_module,
        "check_buffer_pool",
        lambda config: BufferPoolCheck(3 * 2**30, 1_250_557_952, None),
    )
    monkeypatch.setattr(
        cli_module,
        "check_enrichment_yield",
        lambda config: YieldState(query_in_flight=False, enrichment_paused=False),
    )


def _strip_models_section(config_path: Path) -> None:
    """Remove the fixture's explicit `models: backend: fake` so the config
    exercises the schema default (`real`)."""
    text = config_path.read_text()
    assert "models:\n  backend: fake\n" in text
    config_path.write_text(text.replace("models:\n  backend: fake\n", ""))


def _point_provider_at_a_missing_module(monkeypatch: pytest.MonkeyPatch, role: str) -> str:
    from imsg.providers import factory as factory_module

    module = f"imsg.absent_provider_module_for_{role}"
    spec = factory_module.RealProviderSpec(role, module, "Provider")
    monkeypatch.setitem(factory_module.REAL_PROVIDERS, role, spec)
    return module


def test_status_prints_the_backend_line_and_reports_it_in_json(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status_probes(monkeypatch)

    result = runner.invoke(app, ["status", "--config", str(cli_config)])
    assert result.exit_code == 0, result.output
    assert "models: backend=fake" in result.output
    assert "models_backend:" not in result.output  # printed once, in its canonical form

    result = runner.invoke(app, ["status", "--config", str(cli_config), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["models_backend"] == "fake"


def test_status_reports_shared_buffers_against_the_hnsw_indexes(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vector search is only fast while its index pages are cached, and a
    pool smaller than the indexes can never hold them — so `status` says
    both sizes, and says so loudly when the pool is too small."""
    _patch_status_probes(monkeypatch)
    result = runner.invoke(app, ["status", "--config", str(cli_config), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["postgres_shared_buffers_bytes"] == 3 * 2**30
    assert payload["hnsw_index_bytes"] == 1_250_557_952
    assert payload["shared_buffers_holds_hnsw_indexes"] is True
    assert payload["shared_buffers_warning"] is None

    warning = "shared_buffers (0.12 GiB) is smaller than the HNSW indexes (1.16 GiB): ..."
    monkeypatch.setattr(
        cli_module,
        "check_buffer_pool",
        lambda config: BufferPoolCheck(128 * 2**20, 1_250_557_952, warning),
    )
    result = runner.invoke(app, ["status", "--config", str(cli_config)])
    assert result.exit_code == 0, result.output
    assert warning in result.output
    assert "shared_buffers_holds_hnsw_indexes: False" in result.output


def test_status_does_not_ask_an_unreachable_database_about_its_buffer_pool(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status_probes(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "check_postgres",
        lambda config: PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason="down"),
    )

    def never(config: object) -> object:
        raise AssertionError("the buffer pool is not probed when Postgres is down")

    monkeypatch.setattr(cli_module, "check_buffer_pool", never)
    result = runner.invoke(app, ["status", "--config", str(cli_config), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["postgres_shared_buffers_bytes"] is None
    assert payload["shared_buffers_holds_hnsw_indexes"] is None


def test_status_reports_the_real_default_when_config_is_silent(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status_probes(monkeypatch)
    _strip_models_section(cli_config)
    result = runner.invoke(app, ["status", "--config", str(cli_config)])
    assert result.exit_code == 0, result.output
    assert "models: backend=real" in result.output


def test_segment_prints_the_backend_line_and_uses_the_factory(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.segment.boundaries import FakeBoundaryProvider
    from imsg.segment.models import SegmentationRunReport

    data_root = _data_root_from_config(mocked_pg_env)
    prompt_path = data_root / "prompts" / "segment_boundaries.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("segment this")
    captured: dict[str, Any] = {}

    def fake_run_segment(conn: Any, config: Any, provider: Any, prompt_bytes: bytes, **kw: Any) -> list[SegmentationRunReport]:
        captured["provider"] = provider
        return [SegmentationRunReport(chat_id=1, segments_written=1)]

    monkeypatch.setattr(cli_module, "run_segment", fake_run_segment)
    result = runner.invoke(app, ["segment", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "models: backend=fake" in result.output
    assert isinstance(captured["provider"], FakeBoundaryProvider)


def test_embed_prints_the_backend_line_and_uses_the_factory(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.embed.pipeline import EmbedRunReport
    from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider

    captured: dict[str, Any] = {}

    def fake_run_embed(conn: Any, provider: Any, **kw: Any) -> EmbedRunReport:
        captured["provider"] = provider
        captured["kw"] = kw
        return EmbedRunReport(segments_embedded=0, chunks_embedded=0, attachments_embedded=0)

    monkeypatch.setattr(cli_module, "run_embed", fake_run_embed)
    result = runner.invoke(app, ["embed", "--config", str(mocked_pg_env), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "models: backend=fake" in result.output
    assert "DRY RUN — nothing was written" in result.output
    assert isinstance(captured["provider"], FakeTextEmbeddingProvider)
    assert isinstance(captured["kw"]["multimodal_provider"], FakeMultimodalEmbeddingProvider)


def test_sync_prints_the_backend_line_before_building_providers(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No boundary prompt anywhere: sync exits 1 while building its segment
    # function — the backend line must already be out by then.
    _hide_the_shipped_prompts(monkeypatch, tmp_path)
    result = runner.invoke(app, ["sync", "--config", str(mocked_pg_env)])
    assert result.exit_code == 1
    assert "models: backend=fake" in result.output
    assert "boundary prompt not found" in result.output


def test_enrich_prints_the_backend_line_and_uses_the_factory(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider
    from imsg.enrich.queue import EnrichmentTask

    captured: dict[str, Any] = {}

    def fake_process(conn: Any, config: Any, providers: Any, task: Any) -> str:
        captured["providers"] = providers
        return "done"

    monkeypatch.setattr(
        cli_module, "claim_tasks", lambda conn, **kw: [EnrichmentTask(attachment_id=1, kind="ocr", attempts=0)]
    )
    monkeypatch.setattr(cli_module, "process_one_task", fake_process)
    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "models: backend=fake" in result.output
    providers = captured["providers"]
    assert isinstance(providers.ocr, FakeOcrProvider)
    assert isinstance(providers.caption, FakeCaptionProvider)
    assert isinstance(providers.transcription, FakeTranscriptionProvider)


def test_enrich_real_backend_prints_the_caption_prompt_it_used(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the real backend, `enrich` reads the fixed caption prompt —
    the shipped `prompts/caption.txt` when data_root has none — hands its
    exact text to the factory, and prints which file it used."""
    from imsg.enrich.pipeline import EnrichmentProviders
    from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider

    _strip_models_section(mocked_pg_env)  # schema default: backend=real
    shipped = factory_module.default_prompt_root() / "prompts" / "caption.txt"
    captured: dict[str, Any] = {}

    def fake_build(
        cfg: Any, *, caption_prompt: str | None = None, shared_runtime: Any = None
    ) -> EnrichmentProviders:
        captured["caption_prompt"] = caption_prompt
        captured["shared_runtime"] = shared_runtime
        return EnrichmentProviders(
            ocr=FakeOcrProvider(), caption=FakeCaptionProvider(), transcription=FakeTranscriptionProvider()
        )

    monkeypatch.setattr(cli_module, "build_enrichment_providers", fake_build)
    monkeypatch.setattr(cli_module, "claim_tasks", lambda conn, **kw: [])
    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "models: backend=real" in result.output
    assert f"caption prompt: {shipped} (repo-shipped default)" in result.output
    assert captured["caption_prompt"] == shipped.read_bytes().decode("utf-8")


def test_mcp_local_sends_the_backend_line_to_stderr_not_the_stdio_channel(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    monkeypatch.setattr(anyio, "run", lambda func, *args: None)
    result = runner.invoke(app, ["mcp", "local", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "models: backend=fake" in result.stderr
    assert "models: backend=fake" not in result.stdout


def test_mcp_local_serves_first_and_warms_every_provider_in_the_background(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No model is touched before the server starts serving — an MCP
    client drops a server that has not answered `initialize` within its
    startup timeout. `run_local_server` starts the warm-up once serving;
    the fake below does the same, and every provider then runs one
    throwaway input on the model thread, with progress on stderr only."""
    import threading

    import anyio

    from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
    from imsg.mcp.tools.local_server import LocalMcpServer
    from imsg.retrieval.background_warm_up import WarmUpPhase
    from imsg.retrieval.reranker import FakeRerankerProvider

    order: list[str] = []
    threads: set[threading.Thread] = set()

    def note(name: str) -> None:
        order.append(name)
        threads.add(threading.current_thread())

    class _Text(FakeTextEmbeddingProvider):
        def embed_query(self, text: str, *, instruction: str) -> list[float]:
            note("text")
            return super().embed_query(text, instruction=instruction)

    class _Multimodal(FakeMultimodalEmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            note("multimodal")
            return super().embed_text(text)

    class _Reranker(FakeRerankerProvider):
        def score(self, query: str, documents: list[str]) -> list[float]:
            note("reranker")
            return super().score(query, documents)

    monkeypatch.setattr(cli_module, "build_text_provider", lambda cfg: _Text(dim=cfg.embedding.dim))
    monkeypatch.setattr(
        cli_module,
        "build_multimodal_provider",
        lambda cfg: _Multimodal(dim=cfg.embedding.multimodal.dim),
    )
    monkeypatch.setattr(cli_module, "build_reranker", lambda cfg: _Reranker())
    phases: list[WarmUpPhase] = []

    def fake_serve(func: Any, local: LocalMcpServer) -> None:
        order.append("serve")
        phases.append(local.warm_up.status().phase)
        local.warm_up.start()
        phases.append(local.warm_up.wait(timeout=10).phase)

    monkeypatch.setattr(anyio, "run", fake_serve)

    result = runner.invoke(app, ["mcp", "local", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert order == ["serve", "text", "multimodal", "reranker"]
    assert "mcp local: database buffer pool ready in " in result.stderr
    assert "pg_prewarm is not installed" in result.stderr  # this test's database has no extensions
    assert phases == [WarmUpPhase.NOT_STARTED, WarmUpPhase.READY]
    assert len(threads) == 1 and threading.main_thread() not in threads
    assert "mcp local: warm-up started: 4 steps in the background" in result.stderr
    for model in ("text embedder", "multimodal text tower", "reranker"):
        assert f"mcp local: {model} ready in " in result.stderr
    assert "mcp local: warm-up done: 4 steps ready in " in result.stderr
    assert "warm-up" not in result.stdout


def test_mcp_local_keeps_serving_when_a_provider_cannot_warm_up(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider that cannot load no longer stops the server — it would
    then vanish from the client with nothing to say why. The server keeps
    answering, the cause is logged once on stderr, and every tool call
    reports it (tests/test_mcp_local_server_warm_up.py)."""
    import anyio

    from imsg.errors import ProviderUnavailableError
    from imsg.mcp.tools.local_server import LocalMcpServer
    from imsg.retrieval.background_warm_up import WarmUpPhase
    from imsg.retrieval.reranker import FakeRerankerProvider

    class _Broken(FakeRerankerProvider):
        def score(self, query: str, documents: list[str]) -> list[float]:
            raise ProviderUnavailableError("reranker weights could not load")

    failures: list[str | None] = []

    def fake_serve(func: Any, local: LocalMcpServer) -> None:
        local.warm_up.start()
        status = local.warm_up.wait(timeout=10)
        assert status.phase is WarmUpPhase.FAILED
        failures.append(status.failure)

    monkeypatch.setattr(cli_module, "build_reranker", lambda cfg: _Broken())
    monkeypatch.setattr(anyio, "run", fake_serve)
    result = runner.invoke(app, ["mcp", "local", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert failures == ["reranker: reranker weights could not load"]
    assert result.stderr.count("warm-up FAILED") == 1
    assert "reranker: reranker weights could not load" in result.stderr
    assert "warm-up done" not in result.stderr
    assert "FAILED" not in result.stdout


def test_real_backend_with_a_missing_provider_module_exits_cleanly(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default backend, before a provider branch is merged (or with
    the `models` extra absent): one `imsg: ...` line, exit 1, nothing
    run — never a traceback."""
    _strip_models_section(mocked_pg_env)
    module = _point_provider_at_a_missing_module(monkeypatch, "text_embedding")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run_embed must not run without providers")

    monkeypatch.setattr(cli_module, "run_embed", boom)
    result = runner.invoke(app, ["embed", "--config", str(mocked_pg_env)])
    assert result.exit_code == 1
    assert "models: backend=real" in result.output
    assert "imsg: this build has no real 'text_embedding' provider" in result.output
    assert module in result.output
    assert "models.backend: fake" in result.output
    assert "Traceback" not in result.output


def test_enrich_dry_run_builds_no_providers_even_on_the_real_backend(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.enrich.queue import EnrichPreviewReport

    _strip_models_section(mocked_pg_env)
    _point_provider_at_a_missing_module(monkeypatch, "ocr")
    monkeypatch.setattr(
        cli_module, "preview_claimable_tasks", lambda conn, **kw: EnrichPreviewReport(total=0, by_kind={})
    )
    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "models: backend=real" in result.output
    assert "DRY RUN — nothing was written" in result.output


def test_help_lists_the_models_command_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "models" in result.output
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.output


# --------------------------------------------------------------------------
# enrichment yields to in-flight queries (D10.3)
# --------------------------------------------------------------------------


def test_status_reports_whether_enrichment_is_yielding_right_now(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status_probes(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "check_enrichment_yield",
        lambda config: YieldState(query_in_flight=True, enrichment_paused=True),
    )
    payload = json.loads(
        runner.invoke(app, ["status", "--config", str(cli_config), "--json"]).output
    )
    assert payload["enrichment_yield_enabled"] is True  # the schema default
    assert payload["enrichment_yielding_now"] is True
    assert payload["query_in_flight"] is True
    assert payload["enrichment_yield_reason"] is None


def test_status_does_not_ask_an_unreachable_database_about_advisory_locks(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status_probes(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "check_postgres",
        lambda config: PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason="down"),
    )

    def _must_not_run(config: Any) -> YieldState:
        raise AssertionError("check_enrichment_yield ran against an unreachable database")

    monkeypatch.setattr(cli_module, "check_enrichment_yield", _must_not_run)
    payload = json.loads(
        runner.invoke(app, ["status", "--config", str(cli_config), "--json"]).output
    )
    assert payload["enrichment_yielding_now"] is None
    assert payload["query_in_flight"] is None


def test_enrich_checks_the_yield_gate_before_claiming_and_between_tasks(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claimed task holds a lease and cannot be abandoned half-run, so
    the gate is consulted before the claim and between units of work —
    never during one."""
    from imsg.enrich.queue import EnrichmentTask

    order: list[str] = []
    tasks = [
        EnrichmentTask(attachment_id=1, kind="ocr", attempts=0),
        EnrichmentTask(attachment_id=2, kind="ocr", attempts=0),
        EnrichmentTask(attachment_id=3, kind="ocr", attempts=0),
    ]

    class _Gate:
        def __init__(self, conn: Any, **kwargs: Any) -> None:
            captured["gate_kwargs"] = kwargs

        def wait_until_clear(self) -> Any:
            order.append("gate")
            return YieldReport(paused=False, waited_seconds=0.0)

    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli_module, "EnrichmentYieldGate", _Gate)

    def _claim(conn: Any, **kw: Any) -> list[EnrichmentTask]:
        order.append("claim")
        return tasks

    monkeypatch.setattr(cli_module, "claim_tasks", _claim)
    monkeypatch.setattr(
        cli_module,
        "process_one_task",
        lambda conn, cfg, providers, task: order.append(f"task{task.attachment_id}") or "done",
    )

    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert order == ["gate", "claim", "task1", "gate", "task2", "gate", "task3"]
    assert captured["gate_kwargs"]["enabled"] is True  # the schema default


def test_enrich_reports_the_time_it_spent_yielding(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.enrich.queue import EnrichmentTask

    class _Gate:
        def __init__(self, conn: Any, **kwargs: Any) -> None:
            pass

        def wait_until_clear(self) -> Any:
            return YieldReport(paused=True, waited_seconds=1.5)

    monkeypatch.setattr(cli_module, "EnrichmentYieldGate", _Gate)
    monkeypatch.setattr(
        cli_module,
        "claim_tasks",
        lambda conn, **kw: [EnrichmentTask(attachment_id=1, kind="ocr", attempts=0)],
    )
    monkeypatch.setattr(cli_module, "process_one_task", lambda *a: "done")

    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "yielded to in-flight queries 1 time(s), 1.5s total" in result.output


def test_enrich_says_nothing_about_yielding_when_it_never_paused(
    mocked_pg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overnight common case: nobody is searching, so the gate is one
    round trip per unit of work and the run reads exactly as before."""
    monkeypatch.setattr(cli_module, "claim_tasks", lambda conn, **kw: [])
    result = runner.invoke(app, ["enrich", "--config", str(mocked_pg_env)])
    assert result.exit_code == 0, result.output
    assert "yielded" not in result.output
