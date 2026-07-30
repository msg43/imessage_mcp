"""CLI wiring smoke tests for `imsg eval ...` / `imsg verify-seed` /
`imsg reconcile-attachments` — argument-validation paths only, no live
database or config file needed (same "no network, no live Postgres"
unit-suite spirit as `tests/test_cli.py`)."""

from __future__ import annotations

from typer.testing import CliRunner

from imsg.cli import app

runner = CliRunner()


def test_eval_command_group_is_registered() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "import-queries" in result.output
    assert "run" in result.output
    assert "diff" in result.output
    assert "pool" in result.output


def test_verify_seed_is_registered() -> None:
    result = runner.invoke(app, ["verify-seed", "--help"])
    assert result.exit_code == 0
    assert "--export" in result.output
    assert "--reference" in result.output


def test_reconcile_attachments_is_registered() -> None:
    result = runner.invoke(app, ["reconcile-attachments", "--help"])
    assert result.exit_code == 0
    assert "--csv-out" in result.output


def test_verify_seed_requires_exactly_one_of_export_or_reference() -> None:
    result = runner.invoke(app, ["verify-seed"])
    assert result.exit_code == 2
    assert "exactly one of --export or --reference" in result.output


def test_verify_seed_rejects_both_export_and_reference(tmp_path: object) -> None:
    result = runner.invoke(
        app, ["verify-seed", "--export", "/tmp/x.json", "--reference", "/tmp/y.json"]
    )
    assert result.exit_code == 2
    assert "exactly one of --export or --reference" in result.output


def test_eval_run_rejects_non_local_target() -> None:
    result = runner.invoke(app, ["eval", "run", "--target", "gemini"])
    assert result.exit_code == 1
    assert "not wired into this CLI build" in result.output


def test_eval_pool_requires_at_least_two_variants() -> None:
    result = runner.invoke(app, ["eval", "pool", "--out", "/tmp/pool.yaml", "--variants", "default"])
    assert result.exit_code == 2
    assert "needs >= 2 variants" in result.output
