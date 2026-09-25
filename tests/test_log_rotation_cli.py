"""`imsg logs rotate`, and the nightly `imsg backup` rotating the logs
under `<data_root>/logs` even when its backup fails (SPEC §14; QA review
2026-09-24: nothing rotated them). The rotation itself is tested in
`tests/test_log_rotation.py`; these tests need only the CLI, so they run
— and fail — against code that has no rotation at all."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import typer
import yaml
from typer.testing import CliRunner

import imsg.cli as cli_module
from imsg.cli import app

runner = CliRunner()
MB = 10**6


def _log(root: Path, name: str, size: int) -> Path:
    path = root / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


@pytest.fixture
def cli_config(
    config_dict_factory: Callable[..., dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(cli_module, "run_guard_mount_or_exit", lambda data_root: None)
    raw = config_dict_factory()
    raw.setdefault("logging", {})["rotate_bytes"] = MB
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _config_root(path: Path) -> Path:
    return Path(yaml.safe_load(path.read_text(encoding="utf-8"))["paths"]["data_root"])


def test_logs_rotate_command_uses_the_configured_threshold(cli_config: Path) -> None:
    root = _config_root(cli_config)

    _log(root, "scheduled-sync.log", 2 * MB)
    result = runner.invoke(app, ["logs", "rotate", "--config", str(cli_config)])
    assert result.exit_code == 0, result.output
    assert "rotated scheduled-sync.log" in result.output
    assert (root / "logs" / "scheduled-sync.log.1.gz").is_file()


def test_backup_rotates_logs_even_when_the_backup_cannot_connect(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Logs grow whether or not a backup could be taken; the nightly job
    rotates them either way, and still exits non-zero for the backup."""
    root = _config_root(cli_config)

    _log(root, "imsgindex-pg.err.log", 2 * MB)

    def refuse(cfg: Any) -> Any:
        raise typer.Exit(code=1)

    monkeypatch.setattr(cli_module, "_connect_and_verify_or_die", refuse)
    result = runner.invoke(app, ["backup", "--config", str(cli_config)])
    assert result.exit_code == 1
    assert "backup: logs: rotated imsgindex-pg.err.log" in result.output
    assert (root / "logs" / "imsgindex-pg.err.log").stat().st_size == 0
