"""FTS5 sidecar schema (SPEC §7.3, D2) — real SQLite (apsw) throughout,
no mocking: this is exactly the "test the FTS5 sidecar for real"
requirement."""

from __future__ import annotations

from pathlib import Path

import apsw
import pytest

from imsg.embed.fts.schema import (
    FTS_SCHEMA_VERSION,
    assert_schema_current,
    config_fingerprint,
    create_schema,
    get_applied_event_id,
    get_meta,
    set_meta,
)
from imsg.errors import FtsSidecarError


def test_create_schema_creates_every_table(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    cur = conn.cursor()
    names = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    for expected in ("meta", "seg_map", "att_map"):
        assert expected in names
    fts_tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%fts%'")}
    assert {"seg_fts", "seg_fts_tri", "att_fts", "att_fts_tri"} <= fts_tables


def test_create_schema_is_idempotent(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    create_schema(conn)  # must not raise or duplicate anything
    assert get_applied_event_id(conn) == 0


def test_create_schema_sets_version_and_fingerprint(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    assert get_meta(conn, "fts_schema_version") == FTS_SCHEMA_VERSION
    assert get_meta(conn, "config_fingerprint") == config_fingerprint()


def test_applied_event_id_defaults_to_zero(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    assert get_applied_event_id(conn) == 0


def test_set_meta_then_get_meta_roundtrips(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    set_meta(conn, "applied_event_id", "42")
    assert get_applied_event_id(conn) == 42


def test_assert_schema_current_passes_for_a_fresh_sidecar(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    assert_schema_current(conn)  # must not raise


def test_assert_schema_current_raises_on_stale_fingerprint(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    set_meta(conn, "config_fingerprint", "stale-value-from-an-old-build")
    with pytest.raises(FtsSidecarError):
        assert_schema_current(conn)


def test_get_applied_event_id_raises_if_never_initialized(tmp_path: Path) -> None:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    with pytest.raises(FtsSidecarError):
        get_applied_event_id(conn)
