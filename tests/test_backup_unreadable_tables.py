"""A refused `pg_dump` names every table the backup role could not read.

`pg_dump` locks all tables first and stops at the first one its role may
not read, so its own message names one table even when several are
blocked. `imsg.backup.postgres_dump.dump_postgres` now asks the catalog,
on the connection it already holds, which tables that role cannot SELECT,
and lists them all with the remedy.

The unit tests need no database: the connection is a stub that answers
the one catalog query, and `pg_dump` is a small shell script that fails
the way the real one does (the real stderr was captured against a scratch
cluster and is reproduced below). The integration tests at the end run
the real `pg_dump` against a scratch Postgres, with a table owned by a
second role, and skip cleanly when none is reachable.

Table names here are invented. Fictional personas only (D5).
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from _export_fixtures import ADMIN_DSN, TEST_PG_HOST, TEST_PG_PORT, admin_reachable
from imsg.backup.postgres_dump import (
    UNREADABLE_TABLES_SQL,
    DumpResult,
    PgDumpExitError,
    binary_major_version,
    dump_postgres,
    explain_unreadable_tables,
    unreadable_tables,
)
from imsg.errors import BackupError

REAL_STDERR = (
    "pg_dump: error: query failed: ERROR:  permission denied for table acme_scratch_one\n"
    "pg_dump: detail: Query was: LOCK TABLE public.message, public.chat, public.person, "
    "public.segment, public.acme_scratch_one, public.acme_scratch_two IN ACCESS SHARE MODE"
)
"""What `pg_dump` 17 printed when run as a role that cannot read two tables
(scratch cluster, 2026-11-23): one table named, the rest only inside the
quoted LOCK statement."""


class _Cursor:
    def __init__(self, conn: StubConn) -> None:
        self._conn = conn
        self._rows: list[tuple[str, str]] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str) -> None:
        self._conn.statements.append(sql)
        if self._conn.error is not None:
            raise self._conn.error
        self._rows = list(self._conn.rows)

    def fetchall(self) -> list[tuple[str, str]]:
        return self._rows


class StubConn:
    """Answers the catalog query with fixed rows, or raises."""

    def __init__(
        self, rows: list[tuple[str, str]] | None = None, error: Exception | None = None
    ) -> None:
        self.rows = rows or []
        self.error = error
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)


def _fake_pg_dump(directory: Path, *, major: int = 17, stderr: str = REAL_STDERR, code: int = 1) -> Path:
    """A `pg_dump` that reports a version, then writes an empty output
    file and fails, as the real one does on a permission error."""
    script = directory / "pg_dump"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "pg_dump (PostgreSQL) '
        f'{major}.9"; exit 0; fi\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "--file" ]; then : > "$2"; fi\n'
        "  shift\n"
        "done\n"
        f"cat >&2 <<'EOF_STDERR'\n{stderr}\nEOF_STDERR\n"
        f"exit {code}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _dump(tmp_path: Path, conn: object, pg_dump: Path) -> Path:
    dest = tmp_path / "postgres.dump"
    dump_postgres(
        conn=conn,
        dsn="postgresql://acme_backup@127.0.0.1:5433/acme_index",
        password="unused-by-these-tests",
        dest=dest,
        pg_dump=pg_dump,
        server_major=17,
    )
    return dest


# ---------------------------------------------------------------------------
# The catalog question and the sentence built from it
# ---------------------------------------------------------------------------


def test_unreadable_tables_are_schema_qualified_and_in_catalog_order() -> None:
    conn = StubConn(rows=[("public", "acme_scratch_one"), ("reports", "acme_scratch_two")])
    assert unreadable_tables(conn) == ("public.acme_scratch_one", "reports.acme_scratch_two")
    assert conn.statements == [UNREADABLE_TABLES_SQL]


def test_the_catalog_question_is_about_the_connected_role_and_skips_system_schemas() -> None:
    sql = " ".join(UNREADABLE_TABLES_SQL.split())
    assert "has_table_privilege(current_user" in sql
    assert "'SELECT'" in sql
    assert "information_schema" in sql and "'^pg_'" in sql


def test_nothing_unreadable_adds_no_words() -> None:
    assert explain_unreadable_tables(StubConn(rows=[])) == ""


def test_the_sentence_names_every_table_and_the_remedy() -> None:
    sentence = explain_unreadable_tables(
        StubConn(rows=[("public", "acme_scratch_one"), ("public", "acme_scratch_two")])
    )
    assert "cannot read 2 table(s)" in sentence
    assert "public.acme_scratch_one, public.acme_scratch_two" in sentence
    assert "owner" in sentence and "SELECT" in sentence and "imsg backup" in sentence


def test_a_catalog_that_cannot_be_read_says_so_instead_of_staying_silent() -> None:
    sentence = explain_unreadable_tables(StubConn(error=RuntimeError("connection reset")))
    assert "could not list the tables this role cannot read" in sentence
    assert "RuntimeError: connection reset" in sentence


# ---------------------------------------------------------------------------
# Through dump_postgres, with a pg_dump that fails like the real one
# ---------------------------------------------------------------------------


def test_a_refused_dump_names_every_unreadable_table_not_just_the_first(tmp_path: Path) -> None:
    conn = StubConn(rows=[("public", "acme_scratch_one"), ("public", "acme_scratch_two")])
    with pytest.raises(PgDumpExitError) as caught:
        _dump(tmp_path, conn, _fake_pg_dump(tmp_path))
    message = str(caught.value)
    # pg_dump's own words are kept ...
    assert message.startswith("pg_dump exited 1: pg_dump: error: query failed")
    # ... and the second table, which pg_dump's one-line error never names
    # outside a quoted LOCK statement, is now stated as unreadable.
    assert "cannot read 2 table(s): public.acme_scratch_one, public.acme_scratch_two" in message
    assert not (tmp_path / "postgres.dump").exists()
    assert isinstance(caught.value, BackupError)


def test_a_failed_dump_with_nothing_unreadable_keeps_pg_dumps_words_alone(tmp_path: Path) -> None:
    stderr = "pg_dump: error: connection to server failed: Connection refused"
    with pytest.raises(PgDumpExitError) as caught:
        _dump(tmp_path, StubConn(rows=[]), _fake_pg_dump(tmp_path, stderr=stderr))
    assert str(caught.value) == f"pg_dump exited 1: {stderr}"


def test_a_failed_dump_whose_catalog_cannot_be_read_still_raises_the_real_error(
    tmp_path: Path,
) -> None:
    conn = StubConn(error=RuntimeError("no connection"))
    with pytest.raises(PgDumpExitError, match="pg_dump exited 1: pg_dump: error: query failed"):
        _dump(tmp_path, conn, _fake_pg_dump(tmp_path))


def test_a_pg_dump_that_cannot_start_does_not_ask_the_catalog(tmp_path: Path) -> None:
    """Only a nonzero exit is explained; a binary that will not run is a
    different problem and must not pick up a table list."""
    not_executable = tmp_path / "pg_dump"
    not_executable.write_text("#!/bin/sh\necho 'pg_dump (PostgreSQL) 17.9'\n", encoding="utf-8")
    not_executable.chmod(0o600)
    conn = StubConn(rows=[("public", "acme_scratch_one")])
    with pytest.raises(BackupError):
        _dump(tmp_path, conn, not_executable)
    assert conn.statements == []


def test_the_fake_reports_a_version_the_way_the_real_binary_does(tmp_path: Path) -> None:
    assert binary_major_version(_fake_pg_dump(tmp_path, major=17)) == 17


# ---------------------------------------------------------------------------
# The real thing: a real pg_dump, a real role that cannot read a table
# ---------------------------------------------------------------------------


def _server_major() -> int | None:
    try:
        conn = psycopg.connect(ADMIN_DSN, connect_timeout=2)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version_num")
            row = cur.fetchone()
        return int(row[0]) // 10000 if row else None
    finally:
        conn.close()


def _new_enough_pg_dump(server_major: int) -> Path | None:
    candidates: list[Path] = []
    on_path = shutil.which("pg_dump")
    if on_path:
        candidates.append(Path(on_path))
    for major in range(server_major, server_major + 4):
        candidates.append(Path(f"/opt/homebrew/opt/postgresql@{major}/bin/pg_dump"))
        candidates.append(Path(f"/usr/local/opt/postgresql@{major}/bin/pg_dump"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if binary_major_version(candidate) >= server_major:
                return candidate
        except BackupError:  # pragma: no cover
            continue
    return None


SERVER_MAJOR = _server_major()
PG_DUMP = _new_enough_pg_dump(SERVER_MAJOR) if SERVER_MAJOR else None

needs_postgres = pytest.mark.skipif(
    not admin_reachable() or PG_DUMP is None,
    reason="needs a reachable scratch Postgres (IMSG_TEST_PG_HOST/PORT/USER) and a pg_dump "
    "at least as new as its server",
)


def _role_dsn(role: str, dbname: str) -> str:
    return f"postgresql://{role}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


def _admin_execute(statements: list[str], dbname: str = "postgres") -> None:
    conn = psycopg.connect(
        f"postgresql://{os.environ.get('IMSG_TEST_PG_USER', 'postgres')}@/{dbname}"
        f"?host={TEST_PG_HOST}&port={TEST_PG_PORT}",
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)  # type: ignore[arg-type]
    finally:
        conn.close()


@pytest.fixture
def blocked_database() -> Iterator[tuple[str, str]]:
    """A database whose app role owns the core tables, plus two scratch
    tables created by the administrator role, which the app role cannot
    read. Yields (role, database)."""
    suffix = uuid.uuid4().hex[:8]
    role, db = f"acme_backup_{suffix}", f"acme_index_{suffix}"
    _admin_execute([f"CREATE ROLE {role} LOGIN", f"CREATE DATABASE {db} OWNER {role}"])
    app = psycopg.connect(_role_dsn(role, db), autocommit=True)
    try:
        with app.cursor() as cur:
            for table in ("message", "chat", "person", "segment"):
                cur.execute(f"CREATE TABLE {table} (id integer)")  # type: ignore[arg-type]
    finally:
        app.close()
    _admin_execute(
        [
            "CREATE TABLE public.acme_scratch_one (id integer)",
            "CREATE TABLE public.acme_scratch_two (id integer)",
        ],
        dbname=db,
    )
    try:
        yield role, db
    finally:
        _admin_execute([f"DROP DATABASE IF EXISTS {db}"])
        _admin_execute([f"DROP ROLE IF EXISTS {role}"])


@needs_postgres
def test_real_catalog_finds_the_tables_owned_by_another_role(
    blocked_database: tuple[str, str],
) -> None:
    role, db = blocked_database
    conn = psycopg.connect(_role_dsn(role, db), autocommit=True)
    try:
        assert unreadable_tables(conn) == ("public.acme_scratch_one", "public.acme_scratch_two")
    finally:
        conn.close()


@needs_postgres
def test_real_pg_dump_refusal_names_both_tables_then_succeeds_once_ownership_is_fixed(
    blocked_database: tuple[str, str], tmp_path: Path
) -> None:
    role, db = blocked_database
    assert PG_DUMP is not None and SERVER_MAJOR is not None
    conn = psycopg.connect(_role_dsn(role, db), autocommit=True)
    try:
        dest = tmp_path / "postgres.dump"

        def attempt() -> DumpResult:
            return dump_postgres(
                conn=conn,
                dsn=_role_dsn(role, db),
                password="unused-under-trust-auth",
                dest=dest,
                pg_dump=PG_DUMP,
                server_major=SERVER_MAJOR,
            )

        with pytest.raises(PgDumpExitError) as caught:
            attempt()
        message = str(caught.value)
        assert "permission denied for table" in message  # pg_dump's own words
        assert "cannot read 2 table(s): public.acme_scratch_one, public.acme_scratch_two" in message
        assert not dest.exists()

        # The remedy the message gives, applied: the same call now succeeds
        # and the catalog has nothing left to report.
        _admin_execute(
            [
                f"ALTER TABLE public.acme_scratch_one OWNER TO {role}",
                f"ALTER TABLE public.acme_scratch_two OWNER TO {role}",
            ],
            dbname=db,
        )
        assert unreadable_tables(conn) == ()
        result = attempt()
        assert dest.is_file() and dest.stat().st_size > 0
        assert result.path == dest
    finally:
        conn.close()
