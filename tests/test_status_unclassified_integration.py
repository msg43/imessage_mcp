"""`imsg status` surfaces the unclassified-thread count (SPEC §11.5, §14).

The reason this needed care rather than a one-line addition: `status` is
a health check, and a health check that hangs or crashes when the thing
it is checking is unhealthy is worse than one that reports "unknown".

A previous pass declined the wiring on the grounds that `status` never
opens a query connection. That premise is false — `check_postgres`,
`check_buffer_pool` and `check_enrichment_yield` each already open one —
so the objection does not apply. The *real* hazard is that, unlike those
three, this query aggregates over `message`, so a large corpus under
contention could make `status` slow rather than wrong.

These tests therefore prove the degradation, not just the happy path:
the count is right when the database is healthy, and `status` still
exits 0 with a reason when the database is absent, pre-migration, or —
the one that actually needed a statement timeout — **busy**, which is
exercised here by holding a real ACCESS EXCLUSIVE lock on `message` from
a second session.

Fictional personas only (D5): Jamie Owner, Alice Example, Bob Builder.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from typer.testing import CliRunner

import imsg.cli as cli_module
import imsg.diagnostics as diagnostics_module
from _export_fixtures import (
    add_participant,
    admin_reachable,
    allow,
    create_scratch_db,
    drop_scratch_db,
    dsn,
    insert_chat,
    insert_message,
    insert_person,
)
from imsg.cli import app
from imsg.mount.guard import MountInfo

TEST_DB_NAME = "imsg_index_status_unclassified_test"
runner = CliRunner()

pytestmark = pytest.mark.skipif(
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
    """Point every collaborator `status` uses at the scratch database.

    `diagnostics` is patched as well as `cli`: the unclassified check
    opens its own connection through `imsg.diagnostics.connect`, exactly
    like the buffer-pool and enrichment-yield checks beside it.
    """
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused-under-trust-auth")
    monkeypatch.setattr(
        cli_module,
        "check_mount",
        lambda data_root: diagnostics_module.MountCheck(
            ok=True, reason=None, info=MountInfo(data_root, True, "scratch")
        ),
    )
    monkeypatch.setattr(
        diagnostics_module, "connect", lambda database, **kw: psycopg.connect(
            dsn(TEST_DB_NAME), autocommit=True
        )
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


def _seed_one_unclassified_thread(db: psycopg.Connection) -> None:
    """One fully-classified DM that must NOT count, and one chat with an
    unallowlisted participant that must."""
    now = datetime.now(UTC)
    owner = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice = insert_person(db, display_name="Alice Example", short_name="alice")
    bob = insert_person(db, display_name="Bob Builder", short_name="bob")
    allow(db, owner)
    allow(db, alice)

    classified = insert_chat(db, source_guid="chat-classified")
    add_participant(db, classified, owner)
    add_participant(db, classified, alice)
    insert_message(
        db, chat_id=classified, sender_person_id=owner, is_from_me=True, sent_at=now,
        text="already allowlisted on both sides",
    )

    unclassified = insert_chat(db, source_guid="chat-unclassified", kind="group")
    add_participant(db, unclassified, owner)
    add_participant(db, unclassified, bob)
    insert_message(
        db, chat_id=unclassified, sender_person_id=bob, is_from_me=False, sent_at=now,
        text="bob is not on the allowlist",
    )
    db.commit()


def test_status_reports_the_real_unclassified_count(
    db: psycopg.Connection, status_cli: Callable[..., Path]
) -> None:
    assert _status(status_cli())["unclassified_thread_count"] == 0
    _seed_one_unclassified_thread(db)
    report = _status(status_cli())
    assert report["unclassified_thread_count"] == 1
    assert report["unclassified_thread_reason"] is None


def test_status_no_longer_claims_the_field_is_unwired(
    db: psycopg.Connection, status_cli: Callable[..., Path]
) -> None:
    """The old note said every listed field reports None until wired. A
    stale note on a live field is how an operator learns to ignore the note.
    Since 2026-09-24 every SPEC §14 field is wired, so there is no note at
    all; a None carries its own reason instead."""
    report = _status(status_cli())
    assert "pipeline_note" not in report
    assert "unclassified_thread_count" in report
    assert "pipeline_reasons" in report


# ---------------------------------------------------------------------------
# Degradation: the point of the whole exercise
# ---------------------------------------------------------------------------


def test_status_still_succeeds_when_the_database_is_absent(
    config_dict_factory: Callable[..., dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`status` must still be a working health check when nothing answers,
    reporting what it could not read rather than failing.

    Absence is injected by making `connect` raise the error libpq raises,
    rather than by pointing the config at an unreachable address. Two
    reasons: an unroutable address costs a 75-second TCP timeout per
    check, and the config schema pins `database.dsn` to port 5433 — which
    on a real deployment host is the live index. A test must not connect
    to that, even read-only.
    """
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused")
    monkeypatch.setattr(
        cli_module,
        "check_mount",
        lambda data_root: diagnostics_module.MountCheck(
            ok=True, reason=None, info=MountInfo(data_root, True, "scratch")
        ),
    )

    def _nothing_listening(database: Any, **kw: Any) -> psycopg.Connection:
        raise psycopg.OperationalError(
            'connection to server at "127.0.0.1", port 5433 failed: Connection refused'
        )

    monkeypatch.setattr(diagnostics_module, "connect", _nothing_listening)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict_factory()), encoding="utf-8")

    result = runner.invoke(app, ["status", "--config", str(path), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["postgres_reachable"] is False
    assert report["unclassified_thread_count"] is None
    assert "Traceback" not in result.output


def test_status_survives_a_pre_migration_database(
    status_cli: Callable[..., Path],
) -> None:
    """A database with no schema yet: the query's tables do not exist, and
    that is a reason string, not a crash."""
    admin = psycopg.connect(dsn("postgres"), autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
            cur.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    finally:
        admin.close()
    try:
        report = _status(status_cli())
        assert report["unclassified_thread_count"] is None
        assert "unclassified thread count not read" in report["unclassified_thread_reason"]
    finally:
        drop_scratch_db(TEST_DB_NAME)


def test_status_does_not_hang_when_the_database_is_busy(
    db: psycopg.Connection,
    status_cli: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real lock, not a simulated one.

    Another session holds ACCESS EXCLUSIVE on `message`; the count query
    needs ACCESS SHARE and therefore blocks. Without the statement timeout
    this call would wait for as long as the other session holds the lock —
    a health check that hangs exactly when the system is unhealthy. The
    timeout is lowered here only to keep the test quick.
    """
    _seed_one_unclassified_thread(db)
    monkeypatch.setattr(diagnostics_module, "STATEMENT_TIMEOUT_MS", 250)

    blocker = psycopg.connect(dsn(TEST_DB_NAME))
    try:
        with blocker.cursor() as cur:
            cur.execute("LOCK TABLE message IN ACCESS EXCLUSIVE MODE")
        report = _status(status_cli())
        assert report["unclassified_thread_count"] is None
        assert "unclassified thread count not read" in report["unclassified_thread_reason"]
        # Everything else still answered — one blocked field must not take
        # the rest of the health check down with it.
        assert report["postgres_reachable"] is True
        assert report["mount_ok"] is True
    finally:
        blocker.rollback()
        blocker.close()


def test_a_busy_database_does_not_make_status_exit_nonzero(
    db: psycopg.Connection,
    status_cli: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics_module, "STATEMENT_TIMEOUT_MS", 250)
    blocker = psycopg.connect(dsn(TEST_DB_NAME))
    try:
        with blocker.cursor() as cur:
            cur.execute("LOCK TABLE message IN ACCESS EXCLUSIVE MODE")
        result = runner.invoke(app, ["status", "--config", str(status_cli())])
        assert result.exit_code == 0
        assert "unclassified_thread_count: None" in result.output
    finally:
        blocker.rollback()
        blocker.close()
