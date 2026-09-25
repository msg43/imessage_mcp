"""Audit retention and the aggregate rows, against a real `mcp_audit` and
`mcp_audit_rollup` (migration 0010).

- `prune_audit` rolls detailed rows older than the window into one row
  per UTC day and outcome and deletes them, keeps accepted public rows
  whole (AT-1's standing check reads the whole history), merges old
  refusal-count intervals into days, and a dry run changes nothing.
- `imsg mcp audit-prune` runs it and says what it did.
- Refusal counts reach `mcp_audit_rollup` through the pooled sink
  `imsg mcp public` uses, and per-request rows through the same pool
  reuse its connections instead of opening one per row.

Same skip pattern as the other integration tests. Subjects are fictional.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner
from imsg.db.pool import postgres_pool
from imsg.mcp.audit import (
    ACCEPTED_FOREIGN_SQL,
    AuditRecord,
    PostgresAuditSink,
    RejectionTally,
    RejectionTallyWriter,
    prune_audit,
    retention_cutoff,
)

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_audit_retention_test"
REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

OWNER_SUB = "700000000000000000007"
OTHER_SUB = "800000000000000000008"


def _dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


def _reachable() -> bool:
    try:
        conn = psycopg.connect(_dsn("postgres"), connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason=(
        f"no reachable scratch Postgres instance (tried {TEST_PG_HOST}:{TEST_PG_PORT}) — "
        "set IMSG_TEST_PG_HOST/IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


@pytest.fixture
def scratch_db() -> Iterator[psycopg.Connection]:
    admin = psycopg.connect(_dsn("postgres"), autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
            cur.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    finally:
        admin.close()
    conn = psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
    try:
        yield conn
    finally:
        conn.close()
        admin = psycopg.connect(_dsn("postgres"), autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        finally:
            admin.close()


NOW = datetime.now(UTC).replace(microsecond=0)
"""The real clock: `imsg mcp audit-prune` reads it, so the seeded rows are
placed relative to it."""
OLD_DAY = datetime(2026, 3, 1, tzinfo=UTC)  # well past 90 days before any run of this test
OLDER_DAY = datetime(2026, 2, 27, tzinfo=UTC)
RECENT = NOW - timedelta(days=2)


def _audit(
    conn: psycopg.Connection,
    ts: datetime,
    *,
    surface: str = "public",
    subject: str | None = None,
    subject_ok: bool = False,
    tool: str | None = None,
    error: str | None = None,
    count: int = 1,
) -> None:
    with conn.cursor() as cur:
        for _ in range(count):
            cur.execute(
                "INSERT INTO mcp_audit (ts, surface, subject, subject_ok, tool, error) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (ts, surface, subject, subject_ok, tool, error),
            )


def _interval(conn: psycopg.Connection, start: datetime, count: int, error: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO mcp_audit_rollup (period_start, period_end, source, surface, "
            "subject_ok, tool, error, request_count) "
            "VALUES (%s, %s, 'unauthenticated', 'public', false, NULL, %s, %s)",
            (start, start + timedelta(seconds=60), error, count),
        )


def _seed(conn: psycopg.Connection) -> None:
    # Old detailed rows: rejections on two days, local-surface calls, and
    # accepted public rows — the owner's, and one foreign subject's, the
    # breach AT-1's standing check exists to catch.
    _audit(conn, OLD_DAY + timedelta(hours=3), subject=OTHER_SUB, error="UNAUTHORIZED", count=3)
    _audit(conn, OLDER_DAY + timedelta(hours=9), error="UNAVAILABLE", count=2)
    _audit(
        conn,
        OLD_DAY + timedelta(hours=4),
        surface="local",
        subject="local",
        subject_ok=True,
        tool="search_messages",
        count=4,
    )
    _audit(conn, OLD_DAY + timedelta(hours=5), subject=OWNER_SUB, subject_ok=True, tool="list_people", count=2)
    _audit(conn, OLD_DAY + timedelta(hours=6), subject=OTHER_SUB, subject_ok=True, tool="list_people")
    # Recent rows, inside the window.
    _audit(conn, RECENT, subject=OTHER_SUB, error="UNAUTHORIZED")
    _audit(conn, RECENT, subject=OWNER_SUB, subject_ok=True, tool="search_messages")
    _audit(conn, RECENT, surface="local", subject="local", subject_ok=True, tool="list_people")
    # Refusal counts: three old intervals on one day, one recent.
    _interval(conn, OLD_DAY + timedelta(hours=1), 31, "UNAUTHORIZED")
    _interval(conn, OLD_DAY + timedelta(hours=2), 7, "UNAUTHORIZED")
    _interval(conn, OLD_DAY + timedelta(hours=2), 5, "RATE_LIMITED")
    _interval(conn, RECENT, 4, "UNAUTHORIZED")


def _rows(conn: psycopg.Connection, sql: str) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql)
        return list(cur.fetchall())


def _state(conn: psycopg.Connection) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    return (
        _rows(conn, "SELECT audit_id, ts, surface, subject, subject_ok, tool, error FROM mcp_audit ORDER BY audit_id"),
        _rows(
            conn,
            "SELECT period_start, period_end, source, surface, subject_ok, tool, error, request_count "
            "FROM mcp_audit_rollup ORDER BY rollup_id",
        ),
    )


def test_the_cutoff_is_a_utc_midnight_the_given_days_back() -> None:
    now = datetime(2026, 9, 24, 15, 30, tzinfo=UTC)
    assert retention_cutoff(now, 90) == datetime(2026, 6, 26, tzinfo=UTC)
    with pytest.raises(ValueError):
        retention_cutoff(now, 0)
    with pytest.raises(ValueError):
        retention_cutoff(now.replace(tzinfo=None), 90)


def test_a_dry_run_reports_exactly_what_a_real_run_does_and_changes_nothing(
    scratch_db: psycopg.Connection,
) -> None:
    _seed(scratch_db)
    before = _state(scratch_db)
    cutoff = retention_cutoff(NOW, 90)

    dry = prune_audit(scratch_db, cutoff=cutoff, dry_run=True)
    assert _state(scratch_db) == before
    real = prune_audit(scratch_db, cutoff=cutoff)

    assert dry.dry_run is True and real.dry_run is False
    assert (
        dry.detail_rows_rolled_up,
        dry.daily_rows_written,
        dry.accepted_public_rows_kept,
        dry.interval_rows_merged,
        dry.merged_rows_written,
    ) == (
        real.detail_rows_rolled_up,
        real.daily_rows_written,
        real.accepted_public_rows_kept,
        real.interval_rows_merged,
        real.merged_rows_written,
    )


def test_old_rows_become_daily_counts_and_accepted_public_rows_stay_whole(
    scratch_db: psycopg.Connection,
) -> None:
    _seed(scratch_db)

    report = prune_audit(scratch_db, cutoff=retention_cutoff(NOW, 90))

    assert report.detail_rows_rolled_up == 3 + 2 + 4
    assert report.daily_rows_written == 3
    assert report.accepted_public_rows_kept == 3
    assert report.interval_rows_merged == 3
    assert report.merged_rows_written == 2  # one day, two codes

    remaining = _rows(
        scratch_db, "SELECT ts, surface, subject, subject_ok, tool FROM mcp_audit ORDER BY audit_id"
    )
    old_kept = [r for r in remaining if r[0] < RECENT]
    assert len(old_kept) == 3
    assert all(r[1] == "public" and r[3] is True for r in old_kept)
    assert len(remaining) == 3 + 3  # the kept old rows, and every recent row

    # AT-1's standing check still sees the old accepted foreign row.
    with scratch_db.cursor() as cur:
        cur.execute(ACCEPTED_FOREIGN_SQL, {"owner_subject": OWNER_SUB})
        row = cur.fetchone()
    assert row is not None and row[0] == 1

    daily = _rows(
        scratch_db,
        "SELECT period_start, period_end, surface, subject_ok, tool, error, request_count "
        "FROM mcp_audit_rollup WHERE source = 'retention' ORDER BY period_start, surface, error",
    )
    day = timedelta(days=1)
    assert daily == [
        (OLDER_DAY, OLDER_DAY + day, "public", False, None, "UNAVAILABLE", 2),
        (OLD_DAY, OLD_DAY + day, "local", True, "search_messages", None, 4),
        (OLD_DAY, OLD_DAY + day, "public", False, None, "UNAUTHORIZED", 3),
    ]

    counted = _rows(
        scratch_db,
        "SELECT period_start, period_end, error, request_count FROM mcp_audit_rollup "
        "WHERE source = 'unauthenticated' ORDER BY period_start, error",
    )
    assert counted == [
        (OLD_DAY, OLD_DAY + day, "RATE_LIMITED", 5),
        (OLD_DAY, OLD_DAY + day, "UNAUTHORIZED", 38),
        (RECENT, RECENT + timedelta(seconds=60), "UNAUTHORIZED", 4),
    ]


def test_pruning_twice_changes_nothing_the_second_time(scratch_db: psycopg.Connection) -> None:
    _seed(scratch_db)
    cutoff = retention_cutoff(NOW, 90)
    prune_audit(scratch_db, cutoff=cutoff)
    after_first = _state(scratch_db)

    second = prune_audit(scratch_db, cutoff=cutoff)

    assert _state(scratch_db) == after_first
    assert (second.detail_rows_rolled_up, second.interval_rows_merged) == (0, 0)
    assert second.accepted_public_rows_kept == 3


def test_refusal_counts_reach_the_rollup_table_through_the_pooled_sink(
    scratch_db: psycopg.Connection,
) -> None:
    pool = postgres_pool(
        lambda: psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True), max_size=2, name="audit"
    )
    try:
        sink = PostgresAuditSink(pool.lease)
        tally = RejectionTally()
        writer = RejectionTallyWriter(tally, sink, interval_seconds=60)
        for _ in range(31):
            tally.add("UNAUTHORIZED")
        for _ in range(3):
            tally.add("RATE_LIMITED")

        assert writer.write_now() == 2
    finally:
        pool.close()

    rows = _rows(
        scratch_db,
        "SELECT source, surface, subject_ok, tool, error, request_count, period_end >= period_start "
        "FROM mcp_audit_rollup ORDER BY error",
    )
    assert rows == [
        ("unauthenticated", "public", False, None, "RATE_LIMITED", 3, True),
        ("unauthenticated", "public", False, None, "UNAUTHORIZED", 31, True),
    ]
    assert _rows(scratch_db, "SELECT count(*) FROM mcp_audit") == [(0,)]


def test_per_request_rows_through_the_pool_reuse_its_connections(
    scratch_db: psycopg.Connection,
) -> None:
    """The old sink opened a connection per row; through the pool, 50 rows
    take at most the pool's two."""
    opened = 0

    def open_one() -> psycopg.Connection:
        nonlocal opened
        opened += 1
        return psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True)

    pool = postgres_pool(open_one, max_size=2, name="audit")
    try:
        sink = PostgresAuditSink(pool.lease)
        for i in range(50):
            sink.record(
                AuditRecord(
                    surface="public",
                    subject=OWNER_SUB,
                    subject_ok=True,
                    tool=f"tool-{i}",
                    params_sha256=None,
                    result_count=1,
                    latency_ms=3,
                    error=None,
                )
            )
    finally:
        pool.close()

    assert opened == 1  # one writer at a time needs one connection
    assert _rows(scratch_db, "SELECT count(*) FROM mcp_audit") == [(50,)]


def test_imsg_mcp_audit_prune_runs_the_prune_and_says_what_it_did(
    scratch_db: psycopg.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import imsg.cli as cli_module
    import imsg.config.schema as schema_module
    from imsg.mount.guard import MountInfo

    _seed(scratch_db)
    before = _state(scratch_db)
    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
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
  audit_retention_days: 90
  public:
    scope: allowlist
export:
  gcp_project: example-project
  gcs_bucket: example-bucket
  data_store_id: example-datastore
"""
    )
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda root: MountInfo(mount_point=root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(
        cli_module,
        "_connect_and_verify_or_die",
        lambda cfg: psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True),
    )
    runner = CliRunner()

    dry = runner.invoke(
        cli_module.app, ["mcp", "audit-prune", "--config", str(config_path), "--dry-run"]
    )
    assert dry.exit_code == 0, dry.output
    assert "dry run, nothing changed" in dry.output
    assert "detailed rows rolled into daily counts: 9 (into 3 daily rows)" in dry.output
    assert "accepted public rows kept in full: 3" in dry.output
    assert _state(scratch_db) == before

    real = runner.invoke(cli_module.app, ["mcp", "audit-prune", "--config", str(config_path)])
    assert real.exit_code == 0, real.output
    assert "detailed rows rolled into daily counts: 9 (into 3 daily rows)" in real.output
    assert "refused-request counts merged by day: 3 rows (into 2)" in real.output
    assert _rows(scratch_db, "SELECT count(*) FROM mcp_audit") == [(6,)]
