"""`imsg status` reports SPEC §14's pipeline fields from the tables that
hold them: watermarks per source, last sync (and whether it is stale),
enrichment queue depths, FTS applied event and outbox lag, unresolved
identities, attachment coverage, last export, last backup (and whether it
is stale) and the 7-day audit-rejection count.

Before 2026-09-24 every one of these printed `None` with a note saying it
was not wired, so nothing flagged a stale sync or a missing backup (QA
review 2026-09-24). These tests seed a scratch database with known rows
and assert the exact numbers, then prove the degraded cases: no
database, no FTS sidecar, no backups — each a `None` with a reason and
exit 0, never a crash.

Fictional personas only (D5): Jamie Owner, Alice Example.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import apsw
import psycopg
import pytest
import yaml
from typer.testing import CliRunner

import imsg.cli as cli_module
import imsg.diagnostics as diagnostics_module
from _export_fixtures import (
    admin_reachable,
    create_scratch_db,
    drop_scratch_db,
    dsn,
    insert_attachment,
    insert_chat,
    insert_message,
    insert_person,
)
from imsg.cli import app
from imsg.embed.fts.schema import create_schema, set_meta
from imsg.mount.guard import MountInfo

TEST_DB_NAME = "imsg_index_status_pipeline_test"
runner = CliRunner()

requires_pg = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(TEST_DB_NAME)


@pytest.fixture
def status_cli(
    config_dict_factory: Callable[..., dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Path]:
    """Every collaborator `status` uses, pointed at the scratch database
    (the unclassified-thread test's pattern)."""
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused-under-trust-auth")
    monkeypatch.setattr(
        cli_module,
        "check_mount",
        lambda data_root: diagnostics_module.MountCheck(
            ok=True, reason=None, info=MountInfo(data_root, True, "scratch")
        ),
    )
    monkeypatch.setattr(
        diagnostics_module,
        "connect",
        lambda database, **kw: psycopg.connect(dsn(TEST_DB_NAME), autocommit=True),
    )
    monkeypatch.setattr(diagnostics_module, "verify_data_directory", lambda conn, root: root)

    def _write(**overrides: Any) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config_dict_factory(**overrides)), encoding="utf-8")
        return path

    return _write


def _status(config_path: Path) -> dict[str, Any]:
    result = runner.invoke(app, ["status", "--config", str(config_path), "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)  # type: ignore[no-any-return]


def _data_root(config_path: Path) -> Path:
    return Path(yaml.safe_load(config_path.read_text(encoding="utf-8"))["paths"]["data_root"])


def _write_backup_set(data_root: Path, name: str, created_at: datetime) -> Path:
    directory = data_root / "backups" / name
    directory.mkdir(parents=True)
    (directory / "MANIFEST.json").write_text(
        json.dumps({"format": 1, "complete": True, "created_at": created_at.isoformat()}),
        encoding="utf-8",
    )
    return directory


def _write_fts_sidecar(data_root: Path, applied_event_id: int) -> None:
    path = data_root / "fts" / "fts.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(path))
    try:
        create_schema(conn)
        set_meta(conn, "applied_event_id", str(applied_event_id))
    finally:
        conn.close()


def _seed(db: psycopg.Connection) -> dict[str, int]:
    """Known rows in every table a pipeline field reads."""
    now = datetime.now(UTC)
    owner = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice = insert_person(db, display_name="Alice Example", short_name="alice")
    db.execute("UPDATE person SET needs_review = true WHERE person_id = %s", (alice,))
    chat = insert_chat(db, source_guid="chat-status")
    insert_message(db, chat_id=chat, sender_person_id=owner, is_from_me=True, sent_at=now, text="a")
    insert_message(db, chat_id=chat, sender_person_id=None, is_from_me=False, sent_at=now, text="b")
    insert_message(db, chat_id=chat, sender_person_id=None, is_from_me=False, sent_at=now, text="c")

    db.execute(
        "INSERT INTO sync_state (key, value) VALUES "
        "('watermark.rowid.mini', '812345'), ('watermark.rowid.studio-seed', '500'), "
        "('unrelated.key', 'x')"
    )
    db.execute(
        "INSERT INTO extraction_run (source_name, snapshot_path, snapshot_sha256, started_at, "
        "finished_at, status) VALUES "
        "('mini', '/s', 'h', %(a)s, %(b)s, 'ok'), "
        "('studio-seed', '/s', 'h', %(c)s, %(d)s, 'ok'), "
        "('mini', '/s', 'h', %(e)s, NULL, 'failed')",
        {
            "a": now - timedelta(minutes=6),
            "b": now - timedelta(minutes=5),
            "c": now - timedelta(days=3, minutes=1),
            "d": now - timedelta(days=3),
            "e": now - timedelta(minutes=1),
        },
    )

    attachments = [insert_attachment(db, filename=f"f{i}.jpg", mime_type="image/jpeg") for i in range(5)]
    db.execute(
        "UPDATE attachment SET state = 'dataless' WHERE attachment_id = %s", (attachments[3],)
    )
    db.execute("UPDATE attachment SET state = 'missing' WHERE attachment_id = %s", (attachments[4],))
    db.execute(
        "INSERT INTO enrichment (attachment_id, kind, state, next_attempt_at) VALUES "
        "(%(a0)s, 'ocr', 'pending', now() - interval '1 minute'), "
        "(%(a1)s, 'ocr', 'pending', now() + interval '1 hour'), "
        "(%(a2)s, 'caption', 'failed', now()), "
        "(%(a0)s, 'caption', 'done', now()), "
        "(%(a1)s, 'caption', 'pending', now() - interval '1 minute')",
        {"a0": attachments[0], "a1": attachments[1], "a2": attachments[2]},
    )

    for entity_id in range(1, 6):
        db.execute(
            "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
            "VALUES ('segment', %s, 'upsert', 'sha')",
            (entity_id,),
        )
    events = db.execute("SELECT event_id FROM search_index_event ORDER BY event_id").fetchall()

    db.execute(
        "INSERT INTO export_run (mode, allowlist_snapshot, config_sha256, manifest_sha256, "
        "started_at, finished_at, status) VALUES "
        "('reconcile', '{}', 'c', 'm', %(a)s, %(b)s, 'ok'), "
        "('reconcile', '{}', 'c', 'm', %(c)s, NULL, 'failed')",
        {"a": now - timedelta(days=1, minutes=5), "b": now - timedelta(days=1), "c": now},
    )

    db.execute(
        "INSERT INTO mcp_audit (ts, surface, subject, subject_ok, tool, error) VALUES "
        "(now() - interval '1 hour', 'public', NULL, false, NULL, 'UNAUTHORIZED'), "
        "(now() - interval '2 days', 'public', 'someone', false, 'search_messages', 'UNAUTHORIZED'), "
        "(now() - interval '6 days', 'public', NULL, false, NULL, 'RATE_LIMITED'), "
        "(now() - interval '10 days', 'public', NULL, false, NULL, 'UNAUTHORIZED'), "
        "(now() - interval '1 hour', 'public', 'owner', true, 'search_messages', NULL), "
        "(now() - interval '1 hour', 'local', 'local', true, 'search_messages', NULL)"
    )
    db.commit()
    return {"third_event": int(events[2][0]), "max_event": int(events[-1][0])}


@requires_pg
def test_status_reports_every_pipeline_field_from_the_tables(
    db: psycopg.Connection, status_cli: Callable[..., Path]
) -> None:
    seeded = _seed(db)
    config_path = status_cli()
    data_root = _data_root(config_path)
    _write_fts_sidecar(data_root, seeded["third_event"])
    _write_backup_set(data_root, "backup-20260924T020000Z-a1b2c3", datetime.now(UTC) - timedelta(hours=2))
    (data_root / "backups" / ".incomplete-20260924T030000Z-d4e5f6").mkdir()

    report = _status(config_path)

    assert report["watermarks_per_source"].keys() == {"mini", "studio-seed"}
    assert report["watermarks_per_source"]["mini"]["rowid"] == 812345
    assert report["watermarks_per_source"]["studio-seed"]["rowid"] == 500
    assert report["watermarks_per_source"]["mini"]["updated_at"]

    assert report["last_sync_by_source"].keys() == {"mini", "studio-seed"}
    assert report["last_sync_at"] == report["last_sync_by_source"]["mini"]
    assert report["sync_stale"] is False

    depths = report["enrichment_queue_depths"]
    assert (depths["pending"], depths["failed"], depths["done"], depths["running"]) == (3, 1, 1, 0)
    assert depths["ready_now"] == 2
    assert depths["by_kind"] == {"ocr": {"pending": 2}, "caption": {"pending": 1, "failed": 1}}

    assert report["fts_applied_event_id"] == seeded["third_event"]
    assert report["fts_outbox_max_event_id"] == seeded["max_event"]
    assert report["fts_outbox_lag"] == 2

    assert report["unresolved_identity_count"] == 2
    assert report["unresolved_identity_detail"] == {
        "message_senders": 2,
        "tapback_senders": 0,
        "chat_participants": 0,
        "owner_persons": 1,
        "persons_needing_review": 1,
    }

    coverage = report["attachment_materialization_coverage"]
    assert (coverage["materialized"], coverage["dataless"], coverage["missing"]) == (3, 1, 1)
    assert coverage["total"] == 5 and coverage["materialized_fraction"] == 0.6

    assert report["last_export_at"] is not None
    assert report["last_export_run_status"] == "failed"
    assert report["audit_rejection_count_7d"] == 3

    assert report["last_backup_at"] is not None
    assert report["backup_stale"] is False
    assert (report["backup_complete_sets"], report["backup_partial_sets"]) == (1, 1)
    assert report["pipeline_reasons"] == {}
    assert "pipeline_note" not in report, "no field is reported as unwired any more"


@requires_pg
def test_stale_sync_and_stale_backup_are_flagged(
    db: psycopg.Connection, status_cli: Callable[..., Path]
) -> None:
    db.execute(
        "INSERT INTO extraction_run (source_name, snapshot_path, snapshot_sha256, finished_at, "
        "status) VALUES ('mini', '/s', 'h', now() - interval '2 hours', 'ok')"
    )
    db.commit()
    config_path = status_cli()
    _write_backup_set(
        _data_root(config_path), "backup-20260920T040000Z-a1b2c3", datetime.now(UTC) - timedelta(days=3)
    )
    report = _status(config_path)
    assert report["sync_stale"] is True
    assert report["backup_stale"] is True


@requires_pg
def test_rolled_up_rejections_are_counted_when_the_rollup_table_exists(
    db: psycopg.Connection, status_cli: Callable[..., Path]
) -> None:
    """Rejections the public server counts in memory, and rows retention
    rolls up, live in `mcp_audit_rollup` with a `request_count` each; the
    7-day count must include them."""
    db.execute(
        "CREATE TABLE mcp_audit_rollup (period_start timestamptz NOT NULL, period_end timestamptz "
        "NOT NULL, source text NOT NULL, surface text NOT NULL, subject_ok boolean NOT NULL, tool "
        "text, error text, request_count int NOT NULL)"
    )
    db.execute(
        "INSERT INTO mcp_audit_rollup VALUES "
        "(now() - interval '2 hours', now() - interval '1 hour', 'unauthenticated', 'public', "
        "false, NULL, 'UNAUTHORIZED', 31), "
        "(now() - interval '9 days', now() - interval '8 days', 'retention', 'public', false, "
        "NULL, 'UNAUTHORIZED', 1000), "
        "(now() - interval '2 hours', now() - interval '1 hour', 'retention', 'public', true, "
        "'search_messages', NULL, 7)"
    )
    db.execute(
        "INSERT INTO mcp_audit (surface, subject, subject_ok, error) VALUES "
        "('public', NULL, false, 'UNAUTHORIZED')"
    )
    db.commit()
    report = _status(status_cli())
    assert report["audit_rejection_count_7d"] == 32


@requires_pg
def test_missing_fts_sidecar_costs_only_the_fts_fields(
    db: psycopg.Connection, status_cli: Callable[..., Path]
) -> None:
    config_path = status_cli()
    report = _status(config_path)
    assert report["fts_applied_event_id"] is None and report["fts_outbox_lag"] is None
    assert "has not been built" in report["pipeline_reasons"]["fts_applied_event_id"]
    assert report["fts_outbox_max_event_id"] == 0
    assert report["enrichment_queue_depths"]["pending"] == 0
    assert not (_data_root(config_path) / "fts" / "fts.db").exists(), "status must not create it"
    assert report["last_sync_at"] is None
    assert "no successful extraction_run" in report["pipeline_reasons"]["last_sync_at"]
    assert report["last_backup_at"] is None and report["backup_reason"]


def test_unreachable_postgres_reports_every_pipeline_field_as_unknown_with_a_reason(
    config_dict_factory: Callable[..., dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused")
    monkeypatch.setattr(
        cli_module,
        "check_mount",
        lambda data_root: diagnostics_module.MountCheck(
            ok=True, reason=None, info=MountInfo(data_root, True, "scratch")
        ),
    )

    def refuse(database: Any, **kw: Any) -> Any:
        raise psycopg.OperationalError("connection refused (test)")

    monkeypatch.setattr(diagnostics_module, "connect", refuse)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict_factory()), encoding="utf-8")
    report = _status(path)
    from imsg.diagnostics import PIPELINE_FIELDS

    for name in PIPELINE_FIELDS:
        assert report[name] is None, name
        assert "connection refused" in report["pipeline_reasons"][name], name


def test_check_backups_ignores_partial_and_foreign_entries(
    config_dict_factory: Callable[..., dict[str, Any]], tmp_path: Path
) -> None:
    from imsg.config.loader import load_config_dict
    from imsg.diagnostics import check_backups

    cfg = load_config_dict(config_dict_factory())
    root = cfg.paths.data_root
    empty = check_backups(cfg)
    assert (empty.last_backup_at, empty.backup_stale, empty.complete_sets) == (None, None, 0)
    assert empty.reason == "no complete backup set under backups/"

    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    _write_backup_set(root, "backup-20260924T040000Z-a1b2c3", now - timedelta(hours=8))
    _write_backup_set(root, "backup-20260923T040000Z-a1b2c4", now - timedelta(hours=32))
    (root / "backups" / "backup-20260924T050000Z-ffffff").mkdir()  # no manifest: partial
    (root / "backups" / "notes.txt").write_text("operator file", encoding="utf-8")  # foreign
    status = check_backups(cfg, now=now)
    assert status.last_backup_at == (now - timedelta(hours=8)).isoformat()
    assert (status.backup_stale, status.complete_sets, status.partial_sets) == (False, 2, 1)
    assert check_backups(cfg, now=now + timedelta(hours=20)).backup_stale is True
