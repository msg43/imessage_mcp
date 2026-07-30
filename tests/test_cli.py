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


@pytest.mark.parametrize("command", ["export"])
def test_stub_stage_exits_nonzero_and_names_the_stage(command: str) -> None:
    """`export` (a parallel agent's scope this wave) remains a stub;
    `install-agents` is now real (exercised below), and every other
    stage this build wires up is exercised elsewhere in this file."""
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
    tests, so nothing here needs real query behavior."""

    def close(self) -> None:
        pass


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


def test_segment_missing_boundary_prompt_exits_cleanly(mocked_pg_env: Path) -> None:
    result = runner.invoke(app, ["segment", "--config", str(mocked_pg_env)])
    assert result.exit_code == 1
    assert "boundary prompt not found" in result.output


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


def test_sync_missing_boundary_prompt_exits_cleanly(mocked_pg_env: Path) -> None:
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
