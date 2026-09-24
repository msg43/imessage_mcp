"""Tests for scripts/doctor.py -- the read-only setup diagnostic.

Every check must (a) correctly detect its failure mode and print a
plain-English "do this next" fix line, and (b) never crash on an
unexpected condition. The most important guarantee this file checks:
the Full Disk Access check NEVER opens (reads the contents of)
chat.db -- it only checks readability -- which is why the "unreadable
chat.db" test below uses a real 000-permission file rather than
grepping the script's source.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import stat
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.usefixtures("messages_dir")

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCTOR_PATH = REPO_ROOT / "scripts" / "doctor.py"


def _load_doctor() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("imsg_doctor_under_test", DOCTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Python 3.14's dataclasses looks the defining module up via
    # sys.modules[cls.__module__] -- register it before executing so
    # doctor.py's frozen/slotted CheckResult dataclass can find itself.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


doctor = _load_doctor()


def _fake_config(**overrides: Any) -> types.SimpleNamespace:
    """A minimal stand-in for the pieces of `Config` doctor.py reads."""
    base = types.SimpleNamespace(
        paths=types.SimpleNamespace(data_root=Path("/nonexistent"), live_chat_db=Path("/nonexistent/chat.db")),
        database=types.SimpleNamespace(dsn="postgresql://imsg@127.0.0.1:1/imsgindex"),
        models=types.SimpleNamespace(backend="real"),
        retrieval=types.SimpleNamespace(reranker_model="models/qwen3-reranker-0.6b-mxfp8-e61197ed"),
    )
    for dotted, value in overrides.items():
        obj = base
        parts = dotted.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return base


# --------------------------------------------------------------------------
# uv present
# --------------------------------------------------------------------------


def test_check_uv_fails_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    result = doctor.check_uv()
    assert result.passed is False
    assert "install uv" in result.next_step.lower()


def test_check_uv_passes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/local/bin/uv")
    result = doctor.check_uv()
    assert result.passed is True


# --------------------------------------------------------------------------
# Rust shim built
# --------------------------------------------------------------------------


def test_check_rust_shim_fails_when_missing(tmp_path: Path) -> None:
    result = doctor.check_rust_shim(tmp_path / "does-not-exist")
    assert result.passed is False
    assert "cargo build --release" in result.next_step


def test_check_rust_shim_passes_when_executable(tmp_path: Path) -> None:
    binary = tmp_path / "imsg-dump"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    result = doctor.check_rust_shim(binary)
    assert result.passed is True


# --------------------------------------------------------------------------
# Postgres reachable
# --------------------------------------------------------------------------


def test_check_postgres_reachable_fails_when_closed() -> None:
    # Port 1 is a privileged port essentially never listening locally.
    result = doctor.check_postgres_reachable("postgresql://imsg@127.0.0.1:1/imsgindex", timeout=1.0)
    assert result.passed is False
    assert "bootstrap_local_postgres.sh" in result.next_step


def test_check_postgres_reachable_passes_when_open() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        result = doctor.check_postgres_reachable(f"postgresql://imsg@127.0.0.1:{port}/imsgindex")
    assert result.passed is True


def test_check_postgres_reachable_fails_on_unparseable_dsn() -> None:
    result = doctor.check_postgres_reachable("not-a-dsn")
    assert result.passed is False


# --------------------------------------------------------------------------
# pgvector + pg_prewarm present
# --------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows

    def execute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def fetchall(self) -> list[tuple[str]]:
        return self._rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


def test_check_extensions_fails_when_both_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _FakeConn([]))
    result = doctor.check_extensions("postgresql://imsg@127.0.0.1:5433/imsgindex")
    assert result.passed is False
    assert "CREATE EXTENSION" in result.next_step
    assert "vector" in result.next_step
    assert "pg_prewarm" in result.next_step


def test_check_extensions_fails_when_one_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _FakeConn([("vector",)]))
    result = doctor.check_extensions("postgresql://imsg@127.0.0.1:5433/imsgindex")
    assert result.passed is False
    assert "pg_prewarm" in result.next_step
    assert "vector" not in result.next_step.split("missing:")[-1]


def test_check_extensions_passes_when_both_present(monkeypatch: pytest.MonkeyPatch) -> None:
    import psycopg

    monkeypatch.setattr(
        psycopg, "connect", lambda *a, **k: _FakeConn([("vector",), ("pg_prewarm",)])
    )
    result = doctor.check_extensions("postgresql://imsg@127.0.0.1:5433/imsgindex")
    assert result.passed is True


def test_check_extensions_fails_when_unreachable() -> None:
    result = doctor.check_extensions("postgresql://imsg@127.0.0.1:1/imsgindex", timeout=1.0)
    assert result.passed is False
    assert "Postgres reachable" in result.next_step


# --------------------------------------------------------------------------
# Data volume mounted + encrypted + sentinel -- must call the guard's own
# function rather than re-implement its policy.
# --------------------------------------------------------------------------


def test_check_data_volume_fails_and_names_the_guard_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from imsg.errors import MountGateError

    def _raise(data_root: Path, **kwargs: Any) -> None:
        raise MountGateError("sentinel file is missing -- this is the exact guard message")

    monkeypatch.setattr("imsg.mount.guard.guard_mount", _raise)
    result = doctor.check_data_volume(tmp_path)
    assert result.passed is False
    assert "sentinel file is missing" in result.next_step


def test_check_data_volume_passes_when_guard_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from imsg.mount.guard import MountInfo

    def _accept(data_root: Path, **kwargs: Any) -> MountInfo:
        return MountInfo(mount_point=data_root, encrypted=True, volume_name="IMSG-Data-Test")

    monkeypatch.setattr("imsg.mount.guard.guard_mount", _accept)
    result = doctor.check_data_volume(tmp_path)
    assert result.passed is True


def test_check_data_volume_calls_the_real_guard_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """doctor.py must call imsg.mount.guard.guard_mount, not re-implement its policy."""
    calls: list[Path] = []
    from imsg.mount.guard import MountInfo

    def _spy(data_root: Path, **kwargs: Any) -> MountInfo:
        calls.append(data_root)
        return MountInfo(mount_point=data_root, encrypted=True, volume_name="IMSG-Data-Test")

    monkeypatch.setattr("imsg.mount.guard.guard_mount", _spy)
    doctor.check_data_volume(tmp_path)
    assert calls == [tmp_path]


# --------------------------------------------------------------------------
# config loads
# --------------------------------------------------------------------------


def test_check_config_loads_fails_on_missing_file(tmp_path: Path) -> None:
    result, config = doctor.check_config_loads(tmp_path / "no-such-config.yaml")
    assert result.passed is False
    assert config is None
    assert "fix your config file" in result.next_step.lower()


def test_check_config_loads_passes_on_valid_config(
    tmp_path: Path, config_dict_factory: Any
) -> None:
    raw = config_dict_factory()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    result, config = doctor.check_config_loads(config_path)
    assert result.passed is True
    assert config is not None
    assert config.database.dsn.endswith(":5433/imsgindex")


# --------------------------------------------------------------------------
# models present
# --------------------------------------------------------------------------


def test_check_models_present_fails_when_packages_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)
    config = _fake_config()
    result = doctor.check_models_present(config)
    assert result.passed is False
    assert "uv sync --extra models" in result.next_step


def test_check_models_present_passes_when_fake_backend() -> None:
    config = _fake_config(**{"models.backend": "fake"})
    result = doctor.check_models_present(config)
    assert result.passed is True


def test_check_models_present_passes_when_packages_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object())
    config = _fake_config()
    result = doctor.check_models_present(config)
    assert result.passed is True


def test_check_models_present_fails_when_config_is_none() -> None:
    result = doctor.check_models_present(None)
    assert result.passed is False


# --------------------------------------------------------------------------
# reranker converted directory present
# --------------------------------------------------------------------------


def test_check_reranker_present_fails_when_local_conversion_missing(tmp_path: Path) -> None:
    config = _fake_config(**{"paths.data_root": tmp_path})
    result = doctor.check_reranker_present(config)
    assert result.passed is False
    assert "manifest.lock.yaml" in result.next_step


def test_check_reranker_present_passes_when_directory_exists(tmp_path: Path) -> None:
    reranker_dir = tmp_path / "models" / "qwen3-reranker-0.6b-mxfp8-e61197ed"
    reranker_dir.mkdir(parents=True)
    config = _fake_config(**{"paths.data_root": tmp_path})
    result = doctor.check_reranker_present(config)
    assert result.passed is True


def test_check_reranker_present_passes_for_hf_repo_id(tmp_path: Path) -> None:
    config = _fake_config(
        **{"paths.data_root": tmp_path, "retrieval.reranker_model": "Qwen/Qwen3-Reranker-8B"}
    )
    result = doctor.check_reranker_present(config)
    assert result.passed is True
    assert "repo id" in result.next_step


def test_check_reranker_present_fails_when_config_is_none() -> None:
    result = doctor.check_reranker_present(None)
    assert result.passed is False


# --------------------------------------------------------------------------
# Full Disk Access -- must NEVER open (read the contents of) chat.db.
# --------------------------------------------------------------------------


def test_check_full_disk_access_fails_on_missing_file(tmp_path: Path) -> None:
    result = doctor.check_full_disk_access(tmp_path / "chat.db")
    assert result.passed is False
    assert "does not exist" in result.next_step


def test_check_full_disk_access_never_reads_a_zero_permission_file(tmp_path: Path) -> None:
    """The core doctor guarantee: it decides readability without opening the file."""
    chat_db = tmp_path / "chat.db"
    chat_db.write_bytes(b"this must never be read by doctor.py")
    chat_db.chmod(0o000)
    try:
        result = doctor.check_full_disk_access(chat_db)
    except PermissionError:
        pytest.fail("doctor.py tried to open chat.db instead of just checking readability")
    finally:
        chat_db.chmod(0o644)

    if os.geteuid() != 0:
        assert result.passed is False
        assert "Full Disk Access" in result.next_step


def test_check_full_disk_access_passes_when_readable(tmp_path: Path) -> None:
    chat_db = tmp_path / "chat.db"
    chat_db.write_bytes(b"fine to have this exist and be readable")
    result = doctor.check_full_disk_access(chat_db)
    assert result.passed is True


# --------------------------------------------------------------------------
# Full run: exit code and never crashing on a locked-down chat.db.
# --------------------------------------------------------------------------


def test_main_never_raises_permission_error_with_locked_chat_db(
    tmp_path: Path, config_dict_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = config_dict_factory()
    chat_db = Path(raw["paths"]["live_chat_db"])
    chat_db.chmod(0o000)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    try:
        exit_code = doctor.main(["--config", str(config_path)])
    except PermissionError:
        pytest.fail("doctor.py's full run tried to open chat.db")
    finally:
        chat_db.chmod(0o644)

    assert isinstance(exit_code, int)
    captured = capsys.readouterr()
    assert "Full Disk Access" in captured.out


def test_main_reports_missing_volume_and_exits_nonzero(
    tmp_path: Path, config_dict_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = config_dict_factory()
    # Point data_root at a directory that is never going to be a mounted,
    # encrypted, sentinel-bearing volume.
    missing_volume = tmp_path / "nonexistent-volume"
    raw["paths"]["data_root"] = str(missing_volume)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    exit_code = doctor.main(["--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code != 0
    combined = captured.out.lower()
    assert "volume" in combined or "sentinel" in combined
    assert "mounted" in combined or "sentinel" in combined


def test_main_all_checks_pass_returns_zero(
    tmp_path: Path,
    config_dict_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sanity check that a fully-faked-healthy environment reports success."""
    raw = config_dict_factory()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    monkeypatch.setattr(doctor, "check_uv", lambda: doctor._pass("uv present"))
    monkeypatch.setattr(doctor, "check_rust_shim", lambda *a: doctor._pass("Rust shim built (imsg-dump)"))
    monkeypatch.setattr(doctor, "check_postgres_reachable", lambda *a, **k: doctor._pass("Postgres reachable on 5433"))
    monkeypatch.setattr(doctor, "check_extensions", lambda *a, **k: doctor._pass("pgvector + pg_prewarm present"))

    from imsg.mount.guard import MountInfo

    monkeypatch.setattr(
        "imsg.mount.guard.guard_mount",
        lambda data_root, **k: MountInfo(mount_point=data_root, encrypted=True, volume_name="x"),
    )

    exit_code = doctor.main(["--config", str(config_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "All checks passed." in captured.out
