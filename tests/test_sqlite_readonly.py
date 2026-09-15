"""`imsg.sqlite_readonly` — the read-only open that leaves a database's
directory exactly as it found it (CLAUDE.md non-negotiable #1, extended
to every chat.db-shaped file the pipeline only reads).

Every assertion here is about files on disk, not flags: the failure this
module exists to prevent was invisible to a flags-only test —
`SQLITE_OPEN_READONLY` was set, and the `-shm` sidecar moved anyway."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import apsw
import pytest

from imsg.sqlite_readonly import (
    WAL_HEADER_BYTES,
    open_readonly_immutable,
    readonly_immutable_uri,
    wal_frame_bytes,
    wal_sidecar_path,
)


def _fingerprint(directory: Path) -> dict[str, tuple[int, int, str]]:
    """(size, mtime_ns, sha256) of every entry in `directory` — untouched
    *and* byte-identical, which is stronger than a name listing."""
    out: dict[str, tuple[int, int, str]] = {}
    for p in sorted(directory.iterdir()):
        st = p.stat()
        out[p.name] = (st.st_size, st.st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def _wal_mode_db(path: Path, rows: int = 3) -> Path:
    """A WAL-header SQLite file, closed cleanly, alone in its directory —
    the shape of the live chat.db, of S1's backup output, and of a
    prepared seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT)")
    conn.executemany("INSERT INTO message (guid) VALUES (?)", [(f"msg-{i}",) for i in range(rows)])
    conn.close()
    assert path.read_bytes()[18:20] == b"\x02\x02", "fixture must carry the WAL header byte"
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    return path


def _count(conn: apsw.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM message").fetchone()
    assert row is not None
    return int(row[0])


def test_open_leaves_the_directory_byte_identical(tmp_path: Path) -> None:
    db = _wal_mode_db(tmp_path / "seed" / "corpus.db")
    before = _fingerprint(db.parent)

    conn = open_readonly_immutable(db)
    try:
        assert _count(conn) == 3
    finally:
        conn.close()

    assert _fingerprint(db.parent) == before


def test_plain_readonly_open_creates_sidecars_which_is_why_this_module_exists(
    tmp_path: Path,
) -> None:
    """The control for the test above: the bare `SQLITE_OPEN_READONLY` open
    the pipeline used to do writes next to the file. If SQLite ever stops
    doing that, this fails and the module's premise should be revisited."""
    db = _wal_mode_db(tmp_path / "seed" / "corpus.db")

    conn = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        assert _count(conn) == 3
    finally:
        conn.close()

    assert sorted(p.name for p in db.parent.iterdir()) == [
        "corpus.db",
        "corpus.db-shm",
        "corpus.db-wal",
    ]


def test_open_refuses_writes(tmp_path: Path) -> None:
    db = _wal_mode_db(tmp_path / "corpus.db")
    conn = open_readonly_immutable(db)
    try:
        with pytest.raises(apsw.ReadOnlyError):
            conn.execute("DELETE FROM message")
    finally:
        conn.close()


def test_open_missing_file_raises_like_a_plain_open(tmp_path: Path) -> None:
    with pytest.raises(apsw.CantOpenError):
        open_readonly_immutable(tmp_path / "nope.db")
    assert not (tmp_path / "nope.db").exists()  # immutable never creates


def test_uri_percent_encodes_metacharacters(tmp_path: Path) -> None:
    """`?`, `#` and `%` are URI metacharacters: a raw path containing one
    would be split into query string / fragment, or mis-decoded."""
    db = _wal_mode_db(tmp_path / "odd dir #1 100% ?" / "corpus.db")

    uri = readonly_immutable_uri(db)
    assert uri.startswith("file:/")
    assert uri.endswith("?mode=ro&immutable=1")
    assert "%23" in uri and "%25" in uri and "%3F" in uri and "%20" in uri
    assert uri.count("?") == 1  # only the real query separator survives

    conn = open_readonly_immutable(db)
    try:
        assert _count(conn) == 3
    finally:
        conn.close()


def test_wal_frame_bytes_counts_only_frames(tmp_path: Path) -> None:
    db = _wal_mode_db(tmp_path / "corpus.db")
    wal = wal_sidecar_path(db)
    assert wal == tmp_path / "corpus.db-wal"

    assert wal_frame_bytes(db) == 0  # no sidecar at all
    wal.write_bytes(b"")
    assert wal_frame_bytes(db) == 0  # the 0-byte -wal a plain read-only open leaves behind
    wal.write_bytes(b"\x00" * WAL_HEADER_BYTES)
    assert wal_frame_bytes(db) == 0  # header only, no frames
    wal.unlink()

    writer = sqlite3.connect(str(db), isolation_level=None)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO message (guid) VALUES ('in-the-log')")
        assert wal_frame_bytes(db) > 0
        # The remedy the extract stage's refusal names, verified to make the answer 0.
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert wal_frame_bytes(db) == 0
    finally:
        writer.close()


def test_immutable_open_ignores_uncheckpointed_frames(tmp_path: Path) -> None:
    """The trade, pinned: rows committed only to the write-ahead log are
    invisible to an immutable open and visible to a plain one. This is
    why S1's *live* source must not use this module, and why the extract
    stage refuses a file whose `-wal` holds frames."""
    db = _wal_mode_db(tmp_path / "corpus.db")
    writer = sqlite3.connect(str(db), isolation_level=None)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.executemany(
            "INSERT INTO message (guid) VALUES (?)", [("in-the-log-1",), ("in-the-log-2",)]
        )

        plain = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            assert _count(plain) == 5
        finally:
            plain.close()

        immutable = open_readonly_immutable(db)
        try:
            assert _count(immutable) == 3
        finally:
            immutable.close()
    finally:
        writer.close()
