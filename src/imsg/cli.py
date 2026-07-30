"""The `imsg` CLI (SPEC §8, §14).

Wired up for real: `migrate`, `check-permissions`, `status`,
`guard-mount`. Every pipeline-stage subcommand below that is not yet
implemented raises `StageNotImplementedError` naming the stage — a
later build fills each one in; the subcommand names and signatures
here are the contract those builds implement against, so keep them
stable.

Pattern for downstream agents: every real command loads config via
`imsg.config.loader.load_config` exactly once near the top, then runs
the mount gate before touching anything under `data_root` (writes) —
or, for read-only diagnostics, calls `imsg.diagnostics.check_mount`
and reports the result instead of hard-exiting. Catch `ImsgError` at
the command boundary and print a clean message; never let a bare
traceback reach the terminal for an expected failure mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from imsg.config.loader import load_config
from imsg.db.connection import connect
from imsg.db.fingerprint import ensure_cluster_fingerprint
from imsg.db.migrations import PostgresMigrationRunner, format_mismatches
from imsg.diagnostics import (
    check_at_rest_posture,
    check_full_disk_access,
    check_mount,
    check_postgres,
    disk_free_bytes,
)
from imsg.errors import ImsgError, StageNotImplementedError
from imsg.mount.guard import run_guard_mount_or_exit

app = typer.Typer(
    name="imsg",
    help="Local-first iMessage retrieval index: extraction, identity, "
    "segmentation, hybrid search, and a scoped MCP surface.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to config.yaml. Defaults to $IMSG_CONFIG, then ./config.yaml.",
    ),
]


def _default_migrations_dir() -> Path:
    # src/imsg/cli.py -> src/imsg -> src -> repo root -> migrations/
    return Path(__file__).resolve().parents[2] / "migrations"


def _load_config_or_die(config_path: Path | None) -> object:
    try:
        return load_config(config_path)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.callback()
def _root() -> None:
    """See `imsg <command> --help` for details on each subcommand."""


# --------------------------------------------------------------------------
# Real commands
# --------------------------------------------------------------------------


@app.command("guard-mount")
def guard_mount_cmd(config: ConfigOption = None) -> None:
    """Refuse to proceed unless data_root is on a mounted, encrypted volume (SPEC §5.4)."""
    cfg = _load_config_or_die(config)
    info = run_guard_mount_or_exit(cfg.paths.data_root)  # type: ignore[attr-defined]
    typer.echo(f"guard-mount: ok — '{cfg.paths.data_root}' is on encrypted volume "  # type: ignore[attr-defined]
               f"'{info.volume_name}'")


@app.command()
def migrate(
    config: ConfigOption = None,
    status: Annotated[bool, typer.Option("--status", help="List applied/pending state; do not apply.")] = False,
    verify: Annotated[bool, typer.Option("--verify", help="Verify applied migrations match disk; do not apply.")] = False,
    migrations_dir: Annotated[
        Path | None, typer.Option(help="Override the migrations directory (mainly for testing).")
    ] = None,
) -> None:
    """Apply pending Postgres migrations (SPEC §7.1). Idempotent; roll-forward only."""
    if status and verify:
        typer.echo("imsg: --status and --verify are mutually exclusive", err=True)
        raise typer.Exit(code=2)

    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)  # type: ignore[attr-defined]

    mdir = migrations_dir or _default_migrations_dir()

    try:
        conn = connect(cfg.database)  # type: ignore[attr-defined]
    except ImsgError as exc:
        typer.echo(f"imsg: could not connect to Postgres: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        runner = PostgresMigrationRunner(conn, mdir)
        if status:
            plan = runner.plan()
            typer.echo(f"applied:  {[a.version for a in plan.applied]}")
            typer.echo(f"pending:  {[p.version for p in plan.pending]}")
            if plan.mismatches:
                typer.echo(f"MISMATCH: {format_mismatches(plan.mismatches)}", err=True)
                raise typer.Exit(code=1)
            return
        if verify:
            try:
                runner.verify()
            except ImsgError as exc:
                typer.echo(f"imsg: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            typer.echo("migrate --verify: clean")
            return

        try:
            applied = runner.apply_pending()
        except ImsgError as exc:
            typer.echo(f"imsg: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        if not applied:
            typer.echo("migrate: nothing to apply (already up to date)")
        else:
            for m in applied:
                typer.echo(f"migrate: applied {m.version:04d}_{m.name}.sql")
            fp = ensure_cluster_fingerprint(
                conn, cfg.paths.data_root, cfg.database.cluster_fingerprint_file  # type: ignore[attr-defined]
            )
            typer.echo(f"migrate: cluster fingerprint {fp}")
    finally:
        conn.close()


@app.command("check-permissions")
def check_permissions(
    config: ConfigOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Report FDA, Contacts, mount, at-rest posture, and Postgres reachability (SPEC §5.1a)."""
    cfg = _load_config_or_die(config)

    mount = check_mount(cfg.paths.data_root)  # type: ignore[attr-defined]
    posture = check_at_rest_posture(cfg.paths.data_root)  # type: ignore[attr-defined]
    fda = check_full_disk_access(cfg.paths.live_chat_db)  # type: ignore[attr-defined]
    pg = check_postgres(cfg)  # type: ignore[arg-type]

    report = {
        "full_disk_access": fda,
        "contacts_access": None,
        "contacts_access_note": (
            "not checked — requires the Contacts framework (pyobjc); wired up "
            "when the S3 identity-resolution stage is built"
        ),
        "at_rest_posture": posture.label,
        "at_rest_posture_caveat": posture.caveat,
        "boot_volume_encrypted": posture.boot_volume_encrypted,
        "auto_login_enabled": posture.auto_login_enabled,
        "data_volume_encrypted": posture.data_volume_encrypted,
        "mount_ok": mount.ok,
        "mount_reason": mount.reason,
        "pg_ok": pg.reachable and bool(pg.cluster_fingerprint_ok),
        "pg_reachable": pg.reachable,
        "pg_cluster_fingerprint_ok": pg.cluster_fingerprint_ok,
        "pg_reason": pg.reason,
        "last_sync_at": None,
        "index_fresh": None,
        "watermarks": None,
        "pipeline_note": "sync/index-freshness/watermarks unavailable — pipeline stages not yet built",
    }

    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return

    for key, value in report.items():
        typer.echo(f"{key}: {value}")


@app.command()
def status(
    config: ConfigOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Mount, Postgres, disk free, at-rest posture (SPEC §14). Pipeline fields report as unavailable until built."""
    cfg = _load_config_or_die(config)

    mount = check_mount(cfg.paths.data_root)  # type: ignore[attr-defined]
    posture = check_at_rest_posture(cfg.paths.data_root)  # type: ignore[attr-defined]
    pg = check_postgres(cfg)  # type: ignore[arg-type]
    free_bytes = disk_free_bytes(cfg.paths.data_root)  # type: ignore[attr-defined]

    report = {
        "mount_ok": mount.ok,
        "mount_reason": mount.reason,
        "postgres_reachable": pg.reachable,
        "postgres_cluster_fingerprint_ok": pg.cluster_fingerprint_ok,
        "postgres_reason": pg.reason,
        "at_rest_posture": posture.label,
        "at_rest_posture_caveat": posture.caveat,
        "disk_free_bytes": free_bytes,
        "watermarks_per_source": None,
        "enrichment_queue_depths": None,
        "fts_applied_event_id": None,
        "fts_outbox_lag": None,
        "unresolved_identity_count": None,
        "attachment_materialization_coverage": None,
        "last_sync_at": None,
        "last_export_at": None,
        "last_backup_at": None,
        "audit_rejection_count_7d": None,
        "unclassified_thread_count": None,
        "pipeline_note": "fields above report None until the corresponding pipeline stage is built",
    }

    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return

    for key, value in report.items():
        typer.echo(f"{key}: {value}")


# --------------------------------------------------------------------------
# Pipeline-stage stubs (SPEC §8) — not this build's scope; later builds fill these in
# --------------------------------------------------------------------------


def _stub(stage: str) -> None:
    try:
        raise StageNotImplementedError(stage)
    except StageNotImplementedError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def snapshot() -> None:
    """S1 — snapshot the live chat.db via the SQLite online-backup API. Not implemented yet."""
    _stub("snapshot")


@app.command()
def extract() -> None:
    """S2 — extract chats/messages/attachments from the snapshot. Not implemented yet."""
    _stub("extract")


@app.command()
def identity() -> None:
    """S3 — resolve handles to person_id via Contacts + manual curation. Not implemented yet."""
    _stub("identity")


@app.command()
def segment() -> None:
    """S4 — sessionize and segment messages for indexing. Not implemented yet."""
    _stub("segment")


@app.command()
def embed() -> None:
    """S6 — embed segments/attachment chunks and update the FTS sidecar. Not implemented yet."""
    _stub("embed")


@app.command()
def sync() -> None:
    """S7 — incremental S1→S2→S3→S4→S6 sync. Not implemented yet."""
    _stub("sync")


@app.command()
def enrich() -> None:
    """S5b — OCR/caption/transcribe/pdftotext enrichment queue worker. Not implemented yet."""
    _stub("enrich")


@app.command("backfill-attachments")
def backfill_attachments() -> None:
    """S5a — materialize iCloud-optimized attachments locally. Not implemented yet."""
    _stub("backfill-attachments")


@app.command()
def export() -> None:
    """S8 — allowlisted export to GCS / Discovery Engine. Not implemented yet."""
    _stub("export")


@app.command("install-agents")
def install_agents() -> None:
    """Render and install the thin LaunchAgent plists (SPEC §5.5). Not implemented yet."""
    _stub("install-agents")


if __name__ == "__main__":  # pragma: no cover
    app()
