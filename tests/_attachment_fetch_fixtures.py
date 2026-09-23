"""Shared setup for the attachment fetcher's tests (owner decision D13).

Scratch Postgres databases, rows written with plain SQL (so a test module
that imports only these helpers and `imsg.cli` still collects on code that
predates the fetcher, and fails on its assertions rather than on an
import), a stand-in `ssh` that runs the remote half of a command locally,
and a check for an rsync new enough for `--mkpath`.

Fictional content only: file names are generic camera names and GUIDs are
made up (D5).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from imsg.db.migrations import PostgresMigrationRunner

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


def _admin_reachable() -> bool:
    try:
        conn = psycopg.connect(dsn("postgres"), connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


REACHABLE = _admin_reachable()

requires_postgres = pytest.mark.skipif(
    not REACHABLE,
    reason=(
        "no reachable scratch Postgres instance "
        f"(tried {TEST_PG_HOST}:{TEST_PG_PORT}) — set IMSG_TEST_PG_HOST/"
        "IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


def scratch_database(name: str) -> Iterator[psycopg.Connection]:
    """A fresh database with every migration applied; an autocommit
    connection, the shape production's `connect()` returns."""
    admin = psycopg.connect(dsn("postgres"), autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {name}")
            cur.execute(f"CREATE DATABASE {name}")
    finally:
        admin.close()
    conn = psycopg.connect(dsn(name), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
    try:
        yield conn
    finally:
        conn.close()
        admin = psycopg.connect(dsn("postgres"), autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {name}")
        finally:
            admin.close()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def insert_attachment(
    conn: psycopg.Connection,
    *,
    guid: str | None = None,
    filename: str | None = "IMG_0001.jpeg",
    source_path: str | None = None,
    byte_size: int | None = None,
    state: str = "missing",
    mime_type: str | None = "image/jpeg",
    uti: str | None = "public.jpeg",
    last_error: str | None = None,
) -> int:
    guid = guid or f"{uuid.uuid4()}".upper()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, filename, source_path, uti,
                                    mime_type, byte_size, state, materialization_last_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::materialization_state, %s)
            RETURNING attachment_id
            """,
            (guid, f"key-{guid}", filename, source_path, uti, mime_type, byte_size, state,
             last_error),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def insert_location(
    conn: psycopg.Connection,
    *,
    attachment_id: int,
    location: str,
    path: str,
    match: str = "recorded_path",
    byte_size: int | None = None,
    sha256: str | None = None,
    reported_by: str = "test",
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment_location
                (attachment_id, location, path, match_quality, reported_by, byte_size, sha256)
            VALUES (%s, %s, %s, %s::attachment_match_quality, ARRAY[%s]::text[], %s, %s)
            RETURNING location_id
            """,
            (attachment_id, location, path, match, reported_by, byte_size, sha256),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def attachment_row(conn: psycopg.Connection, attachment_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state::text, sha256, byte_size, cache_path, materialization_last_error "
            "FROM attachment WHERE attachment_id = %s",
            (attachment_id,),
        )
        row = cur.fetchone()
    assert row is not None
    return dict(zip(("state", "sha256", "byte_size", "cache_path", "last_error"), row, strict=True))


def location_row(conn: psycopg.Connection, location_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT last_outcome, last_tried_at, fetched_at, last_error, match_quality::text, "
            "reported_by, byte_size, sha256 FROM attachment_location WHERE location_id = %s",
            (location_id,),
        )
        row = cur.fetchone()
    assert row is not None
    keys = ("outcome", "tried_at", "fetched_at", "error", "match", "reported_by", "byte_size",
            "sha256")
    return dict(zip(keys, row, strict=True))


def write_file(root: Path, rel: str, data: bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def tree_state(root: Path) -> dict[str, tuple[int, int, int, str]]:
    """Every entry below `root`: (mode, size, mtime_ns, sha256 of a file's
    bytes). Two equal results mean nothing below `root` was written,
    renamed, deleted or had its permissions changed."""
    state: dict[str, tuple[int, int, int, str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in sorted(dirnames + filenames):
            path = Path(dirpath) / name
            st = os.lstat(path)
            digest = ""
            if stat.S_ISREG(st.st_mode):
                digest = sha256_bytes(path.read_bytes())
            state[str(path.relative_to(root))] = (st.st_mode, st.st_size, st.st_mtime_ns, digest)
    return state


def write_fake_ssh(directory: Path) -> Path:
    """An `ssh` that drops its options and the host name and runs the
    remote command here, through `sh -c`, as sshd would on the far end."""
    script = directory / "fake-ssh"
    script.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        "    -o|-l|-p|-i) shift 2 ;;\n"
        "    -*) shift ;;\n"
        "    *) break ;;\n"
        "  esac\n"
        "done\n"
        "shift\n"
        'exec /bin/sh -c "$*"\n'
    )
    script.chmod(0o755)
    return script


def modern_rsync() -> str | None:
    """An rsync that supports `--mkpath` (3.2.3+), or None."""
    for candidate in (shutil.which("rsync"), "/opt/homebrew/bin/rsync", "/usr/local/bin/rsync"):
        if not candidate or not Path(candidate).exists():
            continue
        out = subprocess.run([candidate, "--version"], capture_output=True, text=True, check=False)
        found = re.search(r"rsync\s+version\s+v?(\d+)\.(\d+)\.(\d+)", out.stdout)
        if found and tuple(int(g) for g in found.groups()) >= (3, 2, 3):
            return candidate
    return None


RSYNC = modern_rsync()
requires_rsync = pytest.mark.skipif(RSYNC is None, reason="needs rsync 3.2.3 or newer")
