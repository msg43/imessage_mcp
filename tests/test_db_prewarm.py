"""`imsg.db.prewarm`: which relations the search path's prewarm covers, the
blocks/bytes/seconds it reports, and the failures it survives — a cold
buffer pool makes search slow, not wrong, so nothing here may raise.

No database: the connection is a scripted fake that answers each statement
by its SQL, which also pins the statements this module is allowed to run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import psycopg
import pytest

from imsg import diagnostics
from imsg.db.prewarm import (
    PREWARM_UNAVAILABLE,
    QUERY_PATH_INDEXES,
    QUERY_PATH_TABLES,
    PrewarmRelation,
    hnsw_index_bytes,
    hnsw_indexes,
    prewarm_query_path,
    query_path_relations,
    shared_buffers_bytes,
)

BLOCK = 8192
HNSW_ROWS = [("segment_embedding_hnsw", 30 * BLOCK), ("attachment_mm_embedding_hnsw", 20 * BLOCK)]
"""(index, size) as the catalog reports them, largest first."""


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self._rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self._rows = self._conn.respond(sql, params)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class _FakeConn:
    """Answers the five statements `imsg.db.prewarm` issues. `sizes` gives
    each named table/index a size (absent means the relation does not
    exist); `toast` names the tables whose TOAST relation exists."""

    def __init__(
        self,
        *,
        sizes: dict[str, int] | None = None,
        toast: dict[str, tuple[int, int]] | None = None,
        has_prewarm: bool = True,
        blocks: int = 3,
        failing: frozenset[str] = frozenset(),
        shared_buffers: int = 512 * BLOCK,
    ) -> None:
        self.sizes = dict.fromkeys((*QUERY_PATH_TABLES, *QUERY_PATH_INDEXES), 10 * BLOCK)
        self.sizes.update(sizes or {})
        self.toast = toast or {}
        self.has_prewarm = has_prewarm
        self.blocks = blocks
        self.failing = failing
        self.shared_buffers = shared_buffers
        self.prewarmed: list[str] = []
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True

    def respond(self, sql: str, params: dict[str, Any] | None) -> list[tuple[Any, ...]]:
        if "proname = 'pg_prewarm'" in sql:
            return [(self.has_prewarm,)]
        if "amname = 'hnsw'" in sql:
            return [(name, size) for name, size in HNSW_ROWS]
        if "block_size" in sql and "shared_buffers" in sql:
            return [(self.shared_buffers,)]
        if "block_size" in sql:
            return [(BLOCK,)]
        if "pg_prewarm(" in sql:
            name = str((params or {})["name"])
            if name in self.failing:
                raise psycopg.errors.InsufficientPrivilege(f"permission denied for {name}")
            self.prewarmed.append(name)
            return [(self.blocks,)]
        if "c.relname = ANY" in sql:
            rows: list[tuple[Any, ...]] = []
            for name in (params or {})["names"]:
                if name not in self.sizes:
                    continue
                toast = self.toast.get(name)
                rows.append(
                    (
                        name,
                        name,
                        self.sizes[name],
                        f"pg_toast.pg_toast_{name}" if toast else None,
                        toast[0] if toast else None,
                        f"pg_toast.pg_toast_{name}_index" if toast else None,
                        toast[1] if toast else None,
                    )
                )
            return rows
        raise AssertionError(f"unexpected statement: {sql}")


def _conn(**kwargs: Any) -> psycopg.Connection:
    return cast(psycopg.Connection, _FakeConn(**kwargs))


def test_the_hnsw_indexes_come_from_the_catalog_not_a_list() -> None:
    conn = _conn()
    assert hnsw_indexes(conn) == [PrewarmRelation(name, size) for name, size in HNSW_ROWS]
    assert hnsw_index_bytes(conn) == sum(size for _, size in HNSW_ROWS)
    assert shared_buffers_bytes(_conn(shared_buffers=99 * BLOCK)) == 99 * BLOCK


def test_relations_are_the_hnsw_indexes_then_the_tables_with_toast_then_the_indexes() -> None:
    conn = _conn(toast={"segment_embedding": (77 * BLOCK, 5 * BLOCK)})
    names = [relation.name for relation in query_path_relations(conn)]
    assert names[:2] == [name for name, _ in HNSW_ROWS]
    assert names[2:5] == [
        "segment_embedding",
        "pg_toast.pg_toast_segment_embedding",
        "pg_toast.pg_toast_segment_embedding_index",
    ]
    assert set(names) == {
        *(name for name, _ in HNSW_ROWS),
        *QUERY_PATH_TABLES,
        *QUERY_PATH_INDEXES,
        "pg_toast.pg_toast_segment_embedding",
        "pg_toast.pg_toast_segment_embedding_index",
    }
    assert len(names) == len(set(names))
    # `message`'s heap is deliberately not prewarmed; its primary key is.
    assert "message" not in QUERY_PATH_TABLES and "message_pkey" in QUERY_PATH_INDEXES


def test_a_relation_that_does_not_exist_yet_is_simply_absent() -> None:
    sizes = dict.fromkeys((*QUERY_PATH_TABLES, *QUERY_PATH_INDEXES), 10 * BLOCK)
    del sizes["attachment_chunk_embedding"]
    del sizes["segment_message_single"]
    conn = _FakeConn()
    conn.sizes = sizes
    names = [r.name for r in query_path_relations(cast(psycopg.Connection, conn))]
    assert "attachment_chunk_embedding" not in names and "segment_message_single" not in names
    assert "segment_message_pkey" in names


def test_prewarm_reports_relations_blocks_bytes_and_seconds() -> None:
    fake = _FakeConn(toast={"segment": (4 * BLOCK, BLOCK)}, blocks=7)
    clock = iter([100.0, 104.5]).__next__
    report = prewarm_query_path(cast(psycopg.Connection, fake), clock=clock)
    expected = len(HNSW_ROWS) + len(QUERY_PATH_TABLES) + len(QUERY_PATH_INDEXES) + 2
    assert report.relations == expected
    assert fake.prewarmed == [r.name for r in query_path_relations(cast(psycopg.Connection, fake))]
    assert report.blocks == 7 * expected
    assert report.bytes_prewarmed == 7 * expected * BLOCK
    assert report.seconds == pytest.approx(4.5)
    assert report.error is None and report.failures == ()
    assert f"in {expected} relation(s), 4.5 s" in report.summary


def test_an_empty_relation_is_not_prewarmed() -> None:
    fake = _FakeConn(sizes={"attachment_chunk_embedding": 0, "person": 0})
    report = prewarm_query_path(cast(psycopg.Connection, fake))
    assert "attachment_chunk_embedding" not in fake.prewarmed
    assert "person" not in fake.prewarmed and "person_pkey" in fake.prewarmed
    assert report.relations == len(fake.prewarmed)


def test_without_the_extension_nothing_is_prewarmed_and_nothing_raises() -> None:
    fake = _FakeConn(has_prewarm=False)
    report = prewarm_query_path(cast(psycopg.Connection, fake))
    assert (report.relations, report.blocks, report.bytes_prewarmed) == (0, 0, 0)
    assert report.error == PREWARM_UNAVAILABLE
    assert report.summary == PREWARM_UNAVAILABLE
    assert fake.prewarmed == []
    assert "migration 0004" in PREWARM_UNAVAILABLE


def test_one_relation_that_cannot_be_prewarmed_does_not_stop_the_rest() -> None:
    fake = _FakeConn(failing=frozenset({"segment_embedding_hnsw", "person_pkey"}))
    report = prewarm_query_path(cast(psycopg.Connection, fake))
    assert "segment_embedding_hnsw" not in fake.prewarmed
    assert "attachment_mm_embedding_hnsw" in fake.prewarmed
    assert report.relations == len(fake.prewarmed) > 0
    assert len(report.failures) == 2
    assert report.error is not None and "2 relation(s) could not be prewarmed" in report.error
    assert "permission denied" in report.failures[0]
    assert report.summary.startswith(f"{report.bytes_prewarmed / 2**20:,.0f} MiB")
    assert "could not be prewarmed" in report.summary


# --------------------------------------------------------------------------
# `imsg status`: shared_buffers against the HNSW indexes it has to hold
# --------------------------------------------------------------------------


def _patch_connect(monkeypatch: pytest.MonkeyPatch, conn: Any) -> None:
    monkeypatch.setattr(diagnostics, "connect", lambda database, **kw: conn)


def _config() -> Any:
    return SimpleNamespace(database=SimpleNamespace(dsn="postgresql://unused"))


def test_check_buffer_pool_reports_both_sizes_and_is_quiet_when_the_pool_is_big_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hnsw = sum(size for _, size in HNSW_ROWS)
    _patch_connect(monkeypatch, _FakeConn(shared_buffers=hnsw * 3))
    check = diagnostics.check_buffer_pool(_config())
    assert (check.shared_buffers_bytes, check.hnsw_index_bytes) == (hnsw * 3, hnsw)
    assert check.holds_hnsw_indexes is True and check.warning is None


def test_check_buffer_pool_warns_when_the_pool_cannot_hold_the_hnsw_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hnsw = sum(size for _, size in HNSW_ROWS)
    _patch_connect(monkeypatch, _FakeConn(shared_buffers=hnsw // 4))
    check = diagnostics.check_buffer_pool(_config())
    assert check.holds_hnsw_indexes is False
    assert check.warning is not None
    assert "smaller than the HNSW indexes" in check.warning
    assert "300-900 ms cold" in check.warning and "postgresql.conf" in check.warning


def test_check_buffer_pool_says_why_when_the_database_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(database: Any, **kw: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(diagnostics, "connect", boom)
    check = diagnostics.check_buffer_pool(_config())
    assert (check.shared_buffers_bytes, check.hnsw_index_bytes) == (None, None)
    assert check.holds_hnsw_indexes is None
    assert check.warning == "buffer pool not read: connection refused"
