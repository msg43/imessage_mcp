"""S2's snapshot open (`imsg.stages.extract._default_open_snapshot`) — no
Postgres needed, so this lives outside `test_extract.py`, whose module-level
skip would otherwise hide these in the no-database unit run.

S2 must read a snapshot or a `--snapshot` seed without writing anything
next to it (CLAUDE.md non-negotiable #1): the plain `SQLITE_OPEN_READONLY`
open it used to do moved the seed corpus's `-shm` mtime on every run
(observed 2026-09-14)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
from imsg.errors import ExtractionError
from imsg.hashing import sha256_file
from imsg.stages.extract import _default_open_snapshot


def _wal_mode_seed(path: Path) -> Path:
    """A WAL-header chat.db-shaped file, closed cleanly, alone in its dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    seed = builder.build(path)
    conn = sqlite3.connect(str(seed), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    assert seed.read_bytes()[18:20] == b"\x02\x02"
    assert sorted(p.name for p in seed.parent.iterdir()) == [seed.name]
    return seed


def _fingerprint(directory: Path) -> dict[str, tuple[int, int, str]]:
    return {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns, sha256_file(p))
        for p in sorted(directory.iterdir())
    }


def test_default_open_snapshot_leaves_the_directory_byte_identical(tmp_path: Path) -> None:
    seed = _wal_mode_seed(tmp_path / "seed" / "corpus-merged.db")
    before = _fingerprint(seed.parent)

    conn = _default_open_snapshot(str(seed))
    try:
        assert [row[0] for row in conn.execute("SELECT guid FROM message")] == ["msg-1"]
    finally:
        conn.close()

    assert _fingerprint(seed.parent) == before


def test_default_open_snapshot_accepts_an_empty_wal_sidecar(tmp_path: Path) -> None:
    """The real seed corpus sits next to a 0-byte `-wal` that an earlier
    plain read-only open created. Empty means no frames: nothing to hide."""
    seed = _wal_mode_seed(tmp_path / "seed" / "corpus-merged.db")
    (tmp_path / "seed" / "corpus-merged.db-wal").write_bytes(b"")
    before = _fingerprint(seed.parent)

    conn = _default_open_snapshot(str(seed))
    try:
        assert [row[0] for row in conn.execute("SELECT guid FROM message")] == ["msg-1"]
    finally:
        conn.close()

    assert _fingerprint(seed.parent) == before


def test_default_open_snapshot_refuses_a_wal_with_frames_and_names_the_fix(
    tmp_path: Path,
) -> None:
    """A seed copied together with a `-wal` that holds frames: an immutable
    open would silently skip every message committed to that log. Refuse,
    name the remedy, and prove the remedy makes the same file readable."""
    seed = _wal_mode_seed(tmp_path / "seed" / "corpus-merged.db")
    writer = sqlite3.connect(str(seed), isolation_level=None)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO message (guid) VALUES ('msg-only-in-the-wal')")

        with pytest.raises(ExtractionError) as excinfo:
            _default_open_snapshot(str(seed))
        message = str(excinfo.value)
        assert str(seed) + "-wal" in message
        assert "wal_checkpoint(TRUNCATE)" in message
        assert "never on the live chat.db" in message

        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        writer.close()

    conn = _default_open_snapshot(str(seed))
    try:
        guids = {row[0] for row in conn.execute("SELECT guid FROM message")}
        assert guids == {"msg-1", "msg-only-in-the-wal"}
    finally:
        conn.close()
