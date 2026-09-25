"""`imsg search-page ...` commands, with the mount guard stubbed: the
password lands as an scrypt hash in a 0600 file and ends every session,
the model-API secret is created once, and `check` reports what is
missing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import imsg.search_page.cli as cli_module
from imsg.cli import app
from imsg.search_page import auth as auth_module
from imsg.search_page.auth import PasswordFile, SessionStore, verify_password

runner = CliRunner()


@pytest.fixture
def config_file(config_dict_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli_module, "run_guard_mount_or_exit", lambda root: None)
    monkeypatch.setattr(auth_module, "SCRYPT_N", 2**14)
    raw = config_dict_factory()
    raw["search_page"] = {"enabled": True, "model_api": {"enabled": True}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_set_password_writes_a_0600_hash_and_ends_sessions(config_file: Path, data_root: Path) -> None:
    sessions = SessionStore(data_root / "private/search-page/sessions.json", lifetime_seconds=60)
    sessions.create("old-fingerprint")
    result = runner.invoke(
        app,
        ["search-page", "set-password", "--config", str(config_file), "--password-stdin"],
        input="a very good password\n",
    )
    assert result.exit_code == 0, result.output
    assert "1 session(s) ended" in result.output
    path = data_root / "private/search-page/owner-password"
    assert (path.stat().st_mode & 0o777) == 0o600
    encoded, _ = PasswordFile(path).load()
    assert verify_password("a very good password", encoded)
    assert "a very good password" not in path.read_text()


def test_set_password_refuses_a_short_one(config_file: Path) -> None:
    result = runner.invoke(
        app, ["search-page", "set-password", "--config", str(config_file), "--password-stdin"], input="short\n"
    )
    assert result.exit_code == 2 and "at least" in result.output


def test_init_model_secret_once(config_file: Path, data_root: Path) -> None:
    first = runner.invoke(app, ["search-page", "init-model-secret", "--config", str(config_file)])
    assert first.exit_code == 0, first.output
    path = data_root / "private/search-page/model-api.secret"
    value = path.read_text()
    assert (path.stat().st_mode & 0o777) == 0o600 and len(value.strip()) >= 32
    again = runner.invoke(app, ["search-page", "init-model-secret", "--config", str(config_file)])
    assert "already present" in again.output and path.read_text() == value
    rotated = runner.invoke(app, ["search-page", "init-model-secret", "--config", str(config_file), "--rotate"])
    assert rotated.exit_code == 0 and path.read_text() != value


def test_check_reports_missing_files(config_file: Path) -> None:
    result = runner.invoke(app, ["search-page", "check", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "password file: MISSING" in result.output
    assert "listen: 127.0.0.1:8710" in result.output
