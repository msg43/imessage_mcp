"""Every normal operation works when the database role is not a superuser
(QA review 2026-09-24: the `imsg` role was a superuser on a
trust-authenticated instance).

The role these tests run as owns the database and everything in it, has
`pg_read_all_settings` (the cluster fingerprint reads `data_directory`,
which only superusers and that role may see), and nothing else. The
`vector` and `pg_prewarm` extensions are created by the admin first —
neither is a trusted extension, so a non-superuser can never create
them, which is why the operator guide creates them as `postgres`.

When the test user is a superuser the tests create such a role; when it
already is not one (a suite run as a least-privilege user), they use it
directly. Fictional content only.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest

from _export_fixtures import (
    ADMIN_DSN,
    REAL_MIGRATIONS_DIR,
    TEST_PG_HOST,
    TEST_PG_PORT,
    admin_reachable,
    insert_chat,
    insert_message,
    insert_person,
    insert_segment,
)
from imsg.backup.postgres_dump import dump_postgres, server_major_version
from imsg.db.migrations import PostgresMigrationRunner
from imsg.db.prewarm import prewarm_query_path, query_path_relations

pytestmark = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)

DB_NAME = "imsg_index_least_privilege_test"
ROLE_NAME = "imsg_least_privilege_test"


def _dsn(user: str, dbname: str) -> str:
    return f"postgresql://{user}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


@dataclass(frozen=True, slots=True)
class LeastPrivilegeDb:
    user: str
    dsn: str
    created_role: bool


def _is_superuser(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()
    return bool(row and row[0])


@pytest.fixture
def lp_db() -> Iterator[LeastPrivilegeDb]:
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        superuser = _is_superuser(admin)
        current = str(admin.execute("SELECT current_user").fetchone()[0])  # type: ignore[index]
        admin.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
        if superuser:
            admin.execute(f"DROP ROLE IF EXISTS {ROLE_NAME}")
            admin.execute(
                f"CREATE ROLE {ROLE_NAME} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                f"NOREPLICATION NOBYPASSRLS"
            )
            admin.execute(f"GRANT pg_read_all_settings TO {ROLE_NAME}")
            admin.execute(f"CREATE DATABASE {DB_NAME} OWNER {ROLE_NAME}")
            with psycopg.connect(_dsn(current, DB_NAME), autocommit=True) as setup:
                setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
                setup.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
            user = ROLE_NAME
        else:
            # Already least-privileged: the extensions must come from the
            # template (the admin put them there), exactly as in production.
            admin.execute(f"CREATE DATABASE {DB_NAME}")
            user = current
    finally:
        admin.close()
    try:
        yield LeastPrivilegeDb(user=user, dsn=_dsn(user, DB_NAME), created_role=superuser)
    finally:
        cleanup = psycopg.connect(ADMIN_DSN, autocommit=True)
        try:
            cleanup.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
            if superuser:
                cleanup.execute(f"DROP ROLE IF EXISTS {ROLE_NAME}")
        finally:
            cleanup.close()


def _migrate_and_seed(db: LeastPrivilegeDb) -> None:
    """Every migration as the non-superuser owner, then rows whose
    rendered text is long and incompressible enough to live out of line,
    in the TOAST relation the prewarm has to reach."""
    with psycopg.connect(db.dsn) as conn:
        assert not _is_superuser(conn)
        applied = PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
        conn.commit()
        # Counted from the directory, so a new migration is exercised here
        # as the non-superuser too rather than breaking the count.
        migration_count = len(sorted(REAL_MIGRATIONS_DIR.glob("*.sql")))
        assert [m.version for m in applied] == list(range(1, migration_count + 1))
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        owner = insert_person(conn, display_name="Jamie Owner", short_name="owner", is_owner=True)
        chat = insert_chat(conn, source_guid="chat-least-privilege")
        message = insert_message(
            conn, chat_id=chat, sender_person_id=owner, is_from_me=True, sent_at=now, text="hi"
        )
        insert_segment(
            conn,
            chat_id=chat,
            started_at=now,
            ended_at=now,
            message_ids=[message],
            rendered_text=secrets.token_hex(40_000),
        )
        conn.commit()


def test_prewarm_reaches_toast_relations_without_superuser(lp_db: LeastPrivilegeDb) -> None:
    """Before 2026-09-24 the prewarm named each relation, and naming a
    `pg_toast.` relation needs USAGE on that schema, which only superusers
    have: every TOAST relation and TOAST index failed with "permission
    denied for schema pg_toast"."""
    _migrate_and_seed(lp_db)
    with psycopg.connect(lp_db.dsn, autocommit=True) as conn:
        toast = [
            r for r in query_path_relations(conn) if r.name.startswith("pg_toast.") and r.size_bytes
        ]
        assert toast, "the seed must put rows in a TOAST relation, or this proves nothing"
        report = prewarm_query_path(conn)
    assert report.failures == ()
    assert report.error is None
    assert report.relations >= len(toast)


def test_the_cluster_fingerprint_needs_only_pg_read_all_settings(lp_db: LeastPrivilegeDb) -> None:
    """`imsg.db.fingerprint` reads `data_directory` on every connection."""
    with psycopg.connect(lp_db.dsn, autocommit=True) as conn:
        assert conn.execute("SHOW data_directory").fetchone()
    if not lp_db.created_role:
        pytest.skip("cannot revoke a grant from the suite's own role")
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        admin.execute(f"REVOKE pg_read_all_settings FROM {ROLE_NAME}")
        with (
            psycopg.connect(lp_db.dsn, autocommit=True) as conn,
            pytest.raises(psycopg.errors.InsufficientPrivilege, match="data_directory"),
        ):
            conn.execute("SHOW data_directory")
    finally:
        admin.execute(f"GRANT pg_read_all_settings TO {ROLE_NAME}")
        admin.close()


def test_backup_vacuum_and_rerun_migrations_work_and_server_programs_do_not(
    lp_db: LeastPrivilegeDb, tmp_path: Path
) -> None:
    _migrate_and_seed(lp_db)
    with psycopg.connect(lp_db.dsn, autocommit=True) as conn:
        # `imsg backup`'s own dump-and-verify, as the owner.
        result = dump_postgres(
            conn=conn,
            dsn=lp_db.dsn,
            password=os.environ.get("IMSG_TEST_PG_PASSWORD", "unused"),
            dest=tmp_path / "postgres.dump",
            pg_dump=_pg_binary("pg_dump", server_major_version(conn)),
            server_major=server_major_version(conn),
        )
        assert result.byte_size > 0 and "message" in result.tables_in_toc

        notices: list[str] = []
        conn.add_notice_handler(lambda diag: notices.append(str(diag.message_primary)))
        conn.execute("VACUUM (ANALYZE)")
        analyzed = conn.execute(
            "SELECT last_analyze IS NOT NULL FROM pg_stat_user_tables WHERE relname = 'segment'"
        ).fetchone()
        assert analyzed and analyzed[0]
        # Only the shared catalogs are skipped, each with a warning.
        assert all("skipping it" in notice for notice in notices)

        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("COPY (SELECT 1) TO PROGRAM 'true'")
    with psycopg.connect(lp_db.dsn) as conn:
        assert PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending() == []


def _pg_binary(name: str, major: int) -> Path:
    """The client binary matching the scratch server's major version:
    Homebrew's versioned keg first (the PATH copy may be older, and an
    older pg_dump refuses a newer server), then PATH."""
    import shutil

    for candidate in (
        f"/opt/homebrew/opt/postgresql@{major}/bin/{name}",
        f"/usr/local/opt/postgresql@{major}/bin/{name}",
        shutil.which(name),
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    pytest.skip(f"{name} for PostgreSQL {major} not found")
