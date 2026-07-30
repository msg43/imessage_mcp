"""Cluster-fingerprint tests (SPEC §5.2, D6) against a fake connection.

`test_migrations_integration.py` covers the "real Postgres correctly
rejected as a non-dedicated cluster" case against a live database.
Here we cover the bootstrap/match/mismatch state machine, which needs
a connection whose reported `data_directory` we control precisely —
not available from any real ambient Postgres instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from imsg.db.fingerprint import ensure_cluster_fingerprint, verify_data_directory
from imsg.errors import ClusterFingerprintError


class _FakeCursor:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn
        self._result: list[tuple[Any, ...]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        stripped = sql.strip()
        if stripped == "SHOW data_directory":
            self._result = [(str(self._conn.data_directory),)]
        elif "SELECT value FROM imsg_meta" in stripped:
            value = self._conn.meta.get("cluster_uuid")
            self._result = [(value,)] if value is not None else []
        elif stripped.startswith("INSERT INTO imsg_meta"):
            assert params is not None
            self._conn.meta["cluster_uuid"] = params[0]
            self._result = []
        else:
            raise AssertionError(f"unexpected SQL against fake connection: {sql!r}")

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._result[0] if self._result else None


class _FakeConnection:
    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory
        self.meta: dict[str, str] = {}
        self.committed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.committed = True


def test_verify_data_directory_accepts_exact_pg17_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=data_root / "pg17")
    resolved = verify_data_directory(conn, data_root)  # type: ignore[arg-type]
    assert resolved == (data_root / "pg17").resolve()


def test_verify_data_directory_accepts_nested_subdir(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=data_root / "pg17" / "base" / "1")
    verify_data_directory(conn, data_root)  # type: ignore[arg-type]  # must not raise


def test_verify_data_directory_rejects_foreign_cluster(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=tmp_path / "some" / "other" / "cluster")
    with pytest.raises(ClusterFingerprintError):
        verify_data_directory(conn, data_root)  # type: ignore[arg-type]


def test_ensure_cluster_fingerprint_bootstraps_on_first_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=data_root / "pg17")
    fingerprint_relpath = Path("pg17/.imsgindex-cluster")

    uuid_value = ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]

    assert conn.meta["cluster_uuid"] == uuid_value
    assert conn.committed is True
    written = (data_root / fingerprint_relpath).read_text().strip()
    assert written == uuid_value


def test_ensure_cluster_fingerprint_is_idempotent(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=data_root / "pg17")
    fingerprint_relpath = Path("pg17/.imsgindex-cluster")

    first = ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]
    second = ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]
    assert first == second


def test_ensure_cluster_fingerprint_detects_file_db_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=data_root / "pg17")
    fingerprint_relpath = Path("pg17/.imsgindex-cluster")

    ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]  # bootstrap

    # Tamper with the on-disk side only.
    (data_root / fingerprint_relpath).write_text("not-the-real-uuid\n")

    with pytest.raises(ClusterFingerprintError, match="mismatch"):
        ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]


def test_ensure_cluster_fingerprint_detects_db_only_missing_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=data_root / "pg17")
    fingerprint_relpath = Path("pg17/.imsgindex-cluster")

    ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]  # bootstrap
    (data_root / fingerprint_relpath).unlink()

    with pytest.raises(ClusterFingerprintError, match="mismatch"):
        ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]


def test_ensure_cluster_fingerprint_rejects_foreign_cluster_before_bootstrapping(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data_root"
    conn = _FakeConnection(data_directory=tmp_path / "unrelated_cluster")
    fingerprint_relpath = Path("pg17/.imsgindex-cluster")

    with pytest.raises(ClusterFingerprintError):
        ensure_cluster_fingerprint(conn, data_root, fingerprint_relpath)  # type: ignore[arg-type]

    assert not (data_root / fingerprint_relpath).exists()
