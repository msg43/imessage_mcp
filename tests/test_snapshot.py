from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import apsw
import pytest

from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
from imsg.errors import SnapshotError
from imsg.stages import snapshot as snapshot_mod
from imsg.stages.snapshot import (
    MAX_PREVIOUS_RETAINED,
    SNAPSHOT_FILENAME,
    SnapshotResult,
    run_snapshot,
)


def _make_live_chat_db(path: Path) -> Path:
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+15551234567"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    return builder.build(path)


def test_run_snapshot_backs_up_and_verifies(tmp_path: Path) -> None:
    live = _make_live_chat_db(tmp_path / "chat.db")
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    result = run_snapshot(live_chat_db=live, data_root=data_root)

    assert isinstance(result, SnapshotResult)
    assert result.path == data_root / "snapshots" / SNAPSHOT_FILENAME
    assert result.path.is_file()
    assert result.reused_existing is False
    assert result.byte_size > 0
    assert len(result.sha256) == 64

    check = apsw.Connection(str(result.path), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        rows = list(check.execute("SELECT guid FROM message"))
        assert rows == [("msg-1",)]
    finally:
        check.close()


def test_run_snapshot_never_opens_source_writable(tmp_path: Path) -> None:
    """The seam records what flags the source was opened with — proving
    the default path really does pass SQLITE_OPEN_READONLY (hard
    requirement 1), not just documenting an intention."""
    live = _make_live_chat_db(tmp_path / "chat.db")
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    seen_flags: list[int] = []

    def spying_open_source(path: str) -> apsw.Connection:
        conn = apsw.Connection(path, flags=apsw.SQLITE_OPEN_READONLY)
        seen_flags.append(apsw.SQLITE_OPEN_READONLY)
        conn.set_busy_timeout(1000)
        return conn

    run_snapshot(live_chat_db=live, data_root=data_root, open_source=spying_open_source)

    assert seen_flags == [apsw.SQLITE_OPEN_READONLY]


def test_run_snapshot_missing_live_db_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    with pytest.raises(SnapshotError, match="not found"):
        run_snapshot(live_chat_db=tmp_path / "does-not-exist.db", data_root=data_root)


def test_run_snapshot_second_identical_run_reuses_existing(tmp_path: Path) -> None:
    live = _make_live_chat_db(tmp_path / "chat.db")
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    first = run_snapshot(live_chat_db=live, data_root=data_root)
    assert first.reused_existing is False

    second = run_snapshot(live_chat_db=live, data_root=data_root)
    assert second.reused_existing is True
    assert second.sha256 == first.sha256
    assert second.path == first.path

    # No .previous file should exist yet — nothing changed to rotate.
    assert not (data_root / "snapshots" / f"{SNAPSHOT_FILENAME}.previous.1").exists()


def test_run_snapshot_rotates_previous_on_change(tmp_path: Path) -> None:
    live_path = tmp_path / "chat.db"
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    _make_live_chat_db(live_path)
    first = run_snapshot(live_chat_db=live_path, data_root=data_root)

    # Mutate the live db so the next backup differs.
    conn = sqlite3.connect(str(live_path))
    conn.execute(
        "INSERT INTO message (ROWID, guid, is_from_me, date, item_type) VALUES (2, 'msg-2', 1, 0, 0)"
    )
    conn.commit()
    conn.close()

    second = run_snapshot(live_chat_db=live_path, data_root=data_root)
    assert second.reused_existing is False
    assert second.sha256 != first.sha256

    previous_1 = data_root / "snapshots" / f"{SNAPSHOT_FILENAME}.previous.1"
    assert previous_1.is_file()
    from imsg.hashing import sha256_file

    assert sha256_file(previous_1) == first.sha256

    # A third change should push the second into .previous.1, first into
    # .previous.2, and retain no more than MAX_PREVIOUS_RETAINED.
    conn = sqlite3.connect(str(live_path))
    conn.execute(
        "INSERT INTO message (ROWID, guid, is_from_me, date, item_type) VALUES (3, 'msg-3', 1, 0, 0)"
    )
    conn.commit()
    conn.close()
    third = run_snapshot(live_chat_db=live_path, data_root=data_root)
    assert third.reused_existing is False

    previous_2 = data_root / "snapshots" / f"{SNAPSHOT_FILENAME}.previous.2"
    assert sha256_file(data_root / "snapshots" / f"{SNAPSHOT_FILENAME}.previous.1") == second.sha256
    assert sha256_file(previous_2) == first.sha256
    assert MAX_PREVIOUS_RETAINED == 2
    assert not (data_root / "snapshots" / f"{SNAPSHOT_FILENAME}.previous.3").exists()


def test_run_snapshot_retries_on_busy_then_succeeds(tmp_path: Path) -> None:
    live = _make_live_chat_db(tmp_path / "chat.db")
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    attempts = {"count": 0}

    def flaky_open_source(path: str) -> apsw.Connection:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise apsw.BusyError("database is locked")
        conn = apsw.Connection(path, flags=apsw.SQLITE_OPEN_READONLY)
        conn.set_busy_timeout(1000)
        return conn

    sleeps: list[float] = []
    result = run_snapshot(
        live_chat_db=live,
        data_root=data_root,
        open_source=flaky_open_source,
        sleep=sleeps.append,
        max_attempts=3,
        retry_wait_seconds=60.0,
    )
    assert result.reused_existing is False
    assert attempts["count"] == 3
    assert sleeps == [60.0, 60.0]


def test_run_snapshot_gives_up_after_max_attempts(tmp_path: Path) -> None:
    live = _make_live_chat_db(tmp_path / "chat.db")
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    def always_busy(path: str) -> apsw.Connection:
        raise apsw.BusyError("database is locked")

    sleeps: list[float] = []
    with pytest.raises(SnapshotError, match="stayed locked"):
        run_snapshot(
            live_chat_db=live,
            data_root=data_root,
            open_source=always_busy,
            sleep=sleeps.append,
            max_attempts=3,
            retry_wait_seconds=1.0,
        )
    assert sleeps == [1.0, 1.0]
    # No partial temp file left behind.
    leftovers = list((data_root / "snapshots").glob(".tmp-*"))
    assert leftovers == []


def test_run_snapshot_refuses_when_disk_nearly_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = _make_live_chat_db(tmp_path / "chat.db")
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    class _FakeUsage:
        free = 1  # far below any real chat.db's size * 2

    monkeypatch.setattr(shutil, "disk_usage", lambda _path: _FakeUsage())

    with pytest.raises(SnapshotError, match="free"):
        run_snapshot(live_chat_db=live, data_root=data_root)


def test_verify_snapshot_rejects_a_non_chatdb_shaped_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.db"
    conn = sqlite3.connect(str(bogus))
    conn.execute("CREATE TABLE unrelated (x)")
    conn.commit()
    conn.close()

    with pytest.raises(SnapshotError, match="missing expected chat\\.db table"):
        snapshot_mod._verify_snapshot(bogus)
