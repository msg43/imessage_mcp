"""FTS5 sidecar schema (SPEC §7.3, D2) — versioned by
`fts_schema_version` + tokenizer config in the sidecar's own `meta`
table; any change to either forces a full rebuild rather than a
silently mixed index (D2's own framing: "cheap now, annoying later").

`seg_fts`/`att_fts` are *contentless* FTS5 tables (`content=''`,
`contentless_delete=1`, requiring SQLite >= 3.45) serving ranked BM25
search; `seg_fts_tri`/`att_fts_tri` are ordinary (non-contentless) FTS5
tables with the `trigram` tokenizer serving exact-substring/verbatim
queries. `seg_map`/`att_map` hold the mapping back to Postgres ids —
`fts_rowid` is set equal to the Postgres `segment_id`/`chunk_id`
directly (both fit in SQLite's 64-bit `INTEGER PRIMARY KEY`), so no
separate id-allocation scheme is needed.
"""

from __future__ import annotations

import hashlib

import apsw

from imsg.errors import FtsSidecarError

FTS_SCHEMA_VERSION = "1"
"""Bump whenever this module's DDL shape changes — `sync.sync_fts`
refuses to run against a sidecar whose stored version/tokenizer
fingerprint doesn't match this module's; `imsg fts rebuild` is the only
recovery path (SPEC §7.3/§9.3)."""

PRIMARY_TOKENIZER = "unicode61 remove_diacritics 2"
"""D2-settled primary tokenizer."""

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS seg_map (
  fts_rowid INTEGER PRIMARY KEY,
  segment_id INTEGER NOT NULL UNIQUE,
  stable_key TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS seg_fts USING fts5(
  text,
  tokenize = "unicode61 remove_diacritics 2",
  content = '', contentless_delete = 1
);
CREATE VIRTUAL TABLE IF NOT EXISTS seg_fts_tri USING fts5(text, tokenize = "trigram");

CREATE TABLE IF NOT EXISTS att_map (
  fts_rowid INTEGER PRIMARY KEY,
  chunk_id INTEGER NOT NULL UNIQUE,
  attachment_id INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS att_fts USING fts5(
  text,
  tokenize = "unicode61 remove_diacritics 2",
  content = '', contentless_delete = 1
);
CREATE VIRTUAL TABLE IF NOT EXISTS att_fts_tri USING fts5(text, tokenize = "trigram");
"""


def config_fingerprint() -> str:
    return hashlib.sha256(f"{FTS_SCHEMA_VERSION}:{PRIMARY_TOKENIZER}".encode()).hexdigest()


def get_meta(conn: apsw.Connection, key: str) -> str | None:
    """`None` both when the key is absent *and* when `meta` doesn't
    exist yet (a sidecar file `create_schema` has never run against) —
    callers that need to distinguish "not initialized" from "key not
    set" should check `get_applied_event_id`, which raises
    `FtsSidecarError` for the former."""
    cur = conn.cursor()
    try:
        row = cur.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except apsw.SQLError as exc:
        if "no such table" in str(exc):
            return None
        raise
    return row[0] if row else None


def set_meta(conn: apsw.Connection, key: str, value: str) -> None:
    conn.cursor().execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def create_schema(conn: apsw.Connection) -> None:
    """Idempotent: safe to call against a fresh file or an
    already-current one (every DDL statement is `IF NOT EXISTS`)."""
    with conn:
        conn.cursor().execute(_DDL)
        set_meta(conn, "fts_schema_version", FTS_SCHEMA_VERSION)
        set_meta(conn, "config_fingerprint", config_fingerprint())
        if get_meta(conn, "applied_event_id") is None:
            set_meta(conn, "applied_event_id", "0")


def assert_schema_current(conn: apsw.Connection) -> None:
    """Raises `FtsSidecarError` if the sidecar's stored schema/tokenizer
    fingerprint doesn't match this module's current one — the caller's
    recovery path is `imsg fts rebuild`, never a silent partial-mix."""
    stored = get_meta(conn, "config_fingerprint")
    if stored != config_fingerprint():
        raise FtsSidecarError(
            f"FTS sidecar config fingerprint {stored!r} does not match the current "
            f"schema/tokenizer fingerprint {config_fingerprint()!r} — run 'imsg fts rebuild' "
            f"(SPEC §7.3: a schema or tokenizer change forces a full rebuild, never a "
            f"silently mixed index)"
        )


def get_applied_event_id(conn: apsw.Connection) -> int:
    value = get_meta(conn, "applied_event_id")
    if value is None:
        raise FtsSidecarError(
            "FTS sidecar has no 'applied_event_id' in its meta table — has "
            "create_schema() been run against it?"
        )
    return int(value)


__all__ = [
    "FTS_SCHEMA_VERSION",
    "PRIMARY_TOKENIZER",
    "assert_schema_current",
    "config_fingerprint",
    "create_schema",
    "get_applied_event_id",
    "get_meta",
    "set_meta",
]
