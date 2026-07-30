"""`imsg verify-seed` (AT-2) and `imsg reconcile-attachments` (AT-3)
(SPEC §12). Top-level commands, matching the exact names SPEC §12
uses, registered onto the root `app` from `imsg.cli` via
`app.command("verify-seed")(verify_seed)` / `app.command(
"reconcile-attachments")(reconcile_attachments)` — see that module's
docstring for why this file doesn't import `imsg.cli` itself
(circularity) and instead re-implements the same small "load config,
connect, verify fingerprint" helpers as `imsg.eval.cli`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from imsg.config.loader import load_config
from imsg.db.connection import connect
from imsg.db.fingerprint import verify_data_directory
from imsg.errors import ImsgError
from imsg.verify.attachments import build_at3_report, format_report_text, report_to_csv
from imsg.verify.seed import (
    build_seed_snapshot,
    snapshot_from_json,
    snapshot_to_json,
    verify_against_reference,
)
from imsg.verify.seed import format_report_text as format_seed_report_text

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to config.yaml. Defaults to $IMSG_CONFIG, then ./config.yaml."),
]


def _load_config_or_die(config_path: Path | None) -> Config:
    try:
        return load_config(config_path)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _connect_and_verify_or_die(cfg: Config) -> psycopg.Connection:
    try:
        conn = connect(cfg.database)
    except ImsgError as exc:
        typer.echo(f"imsg: could not connect to Postgres: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        verify_data_directory(conn, cfg.paths.data_root)
    except ImsgError as exc:
        conn.close()
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return conn


# --------------------------------------------------------------------------
# AT-2 — imsg verify-seed
# --------------------------------------------------------------------------


def verify_seed(
    config: ConfigOption = None,
    export: Annotated[
        Path | None,
        typer.Option(help="Write this host's seed snapshot here (run this on the OTHER host first)."),
    ] = None,
    reference: Annotated[
        Path | None,
        typer.Option(help="A snapshot exported (via --export) from the other host; verify against it."),
    ] = None,
    source_label: Annotated[
        str, typer.Option(help="Label for this host in the snapshot/report (e.g. 'mini', 'studio').")
    ] = "local",
    accept_missing: Annotated[
        Path | None,
        typer.Option(help="A file with one owner-approved-exception GUID per line."),
    ] = None,
    report_out: Annotated[Path | None, typer.Option(help="Also write the text report here.")] = None,
) -> None:
    """AT-2 — seed completeness (SPEC §12 AT-2). Exactly one of
    `--export`/`--reference` is required: run with `--export <file>`
    on the *other* machine, copy that file here (e.g. `rsync` over the
    tailnet — SPEC §8 S7's Studio-seed transfer already assumes exactly
    that path), then run with `--reference <file>` on this one. See
    `imsg.verify.seed` for why this two-step file exchange replaces the
    literal `--reference <studio-snapshot.db>` spec text (both hosts
    are never reachable at once here)."""
    if (export is None) == (reference is None):
        typer.echo("imsg: verify-seed requires exactly one of --export or --reference", err=True)
        raise typer.Exit(code=2)

    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        if export is not None:
            snapshot = build_seed_snapshot(conn, source_label=source_label)
            export.write_text(snapshot_to_json(snapshot), encoding="utf-8")
            typer.echo(
                f"verify-seed: exported {len(snapshot.guids)} message GUID(s) as "
                f"'{source_label}' to {export}"
            )
            return

        assert reference is not None
        ref_snapshot = snapshot_from_json(reference.read_text(encoding="utf-8"))
        accepted: frozenset[str] = frozenset()
        if accept_missing is not None:
            accepted = frozenset(
                line.strip()
                for line in accept_missing.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        report = verify_against_reference(
            conn, ref_snapshot, local_label=source_label, accepted_exceptions=accepted
        )
    finally:
        conn.close()

    text = format_seed_report_text(report)
    typer.echo(text)
    if report_out is not None:
        report_out.write_text(text, encoding="utf-8")
    if not report.passed:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------
# AT-3 — imsg reconcile-attachments
# --------------------------------------------------------------------------


def reconcile_attachments(
    config: ConfigOption = None,
    csv_out: Annotated[Path | None, typer.Option(help="Write the CSV manifest here.")] = None,
    report_out: Annotated[Path | None, typer.Option(help="Also write the text summary here.")] = None,
    sample_size: Annotated[
        int, typer.Option(help="Stratified integrity-sample size over materialized files.")
    ] = 25,
    seed: Annotated[int | None, typer.Option(help="Sampling seed, for reproducible runs.")] = None,
) -> None:
    """AT-3 — attachment reconciliation (SPEC §12 AT-3). Layers on top
    of the S5a-built `imsg.backfill.reconcile.build_reconciliation_report`
    (does not duplicate it): by-year/by-type materialization rates, a
    four-category exception manifest, CSV output, and a stratified
    hash/open integrity check over materialized files."""
    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        report = build_at3_report(conn, integrity_sample_size=sample_size, seed=seed)
    finally:
        conn.close()

    text = format_report_text(report)
    typer.echo(text)
    if report_out is not None:
        report_out.write_text(text, encoding="utf-8")
    if csv_out is not None:
        csv_out.write_text(report_to_csv(report), encoding="utf-8")
        typer.echo(f"reconcile-attachments: wrote CSV to {csv_out}")
    if not report.passed:
        raise typer.Exit(code=1)


__all__ = ["reconcile_attachments", "verify_seed"]
