"""The `imsg` CLI (SPEC §8, §14).

Wired up for real: `migrate`, `check-permissions`, `status`,
`guard-mount`, `snapshot`, `extract`, `identity`, `segment`, `embed`,
`sync`, `enrich`, `backfill-attachments`, `mcp local`, `mcp public`,
`models verify`, `install-agents`. `export` remains a
`StageNotImplementedError` stub — it is a parallel agent's scope
(SPEC §11, `src/imsg/export/`, untouched here).

**Model providers** come from `imsg.providers.factory` — the only place
a model-backed command (`segment`, `embed`, `sync`, `enrich`, `mcp
local`, `mcp public`, and `eval run`/`eval pool` in `imsg.eval.cli`)
obtains its embedding / boundary / reranker / OCR / caption /
transcription providers. `models.backend` in config selects between the
real implementations (`real`, the default: MLX, Apple Vision and
PE-Core, pinned by repo + revision in config and in
`models/manifest.lock.yaml`) and the deterministic `Fake*` stand-ins
(`fake`, explicit opt-in only). Every command that builds providers
prints `models: backend=<real|fake>` first, so a fake run — which
reports success while its search results are meaningless — can never
be mistaken for a real one. A real provider whose module or runtime
packages are missing fails as one clean `imsg: ...` line telling the
operator to install the `models` extra, never as a traceback.

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
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import anyio
import apsw
import psycopg
import typer
import uvicorn

from imsg.agents.plists import render_agent_plists
from imsg.backfill.pipeline import DEFAULT_RATE_PER_MINUTE, run_backfill
from imsg.config.loader import default_config_path, load_config
from imsg.db.connection import connect
from imsg.db.fingerprint import ensure_cluster_fingerprint, verify_data_directory
from imsg.db.migrations import PostgresMigrationRunner, format_mismatches
from imsg.diagnostics import (
    check_at_rest_posture,
    check_full_disk_access,
    check_mount,
    check_postgres,
    disk_free_bytes,
)
from imsg.embed.fts.schema import assert_schema_current, create_schema
from imsg.embed.fts.sync import sync_fts
from imsg.embed.pipeline import run_embed
from imsg.enrich.pipeline import process_one_task
from imsg.enrich.queue import claim_tasks, preview_claimable_tasks
from imsg.errors import AgentInstallError, ImsgError, StageNotImplementedError
from imsg.eval.cli import eval_app
from imsg.mcp.audit import PostgresAuditSink
from imsg.mcp.auth import build_public_gate
from imsg.mcp.tools.local_server import LocalMcpServer, run_local_server
from imsg.mcp.tools.public_server import PublicMcpServer, build_public_asgi_app, parse_bind_address
from imsg.mount.guard import run_guard_mount_or_exit
from imsg.paths import is_contained_in, is_same_file, resolve_path
from imsg.providers.factory import (
    ResolvedPrompt,
    backend_status_line,
    build_boundary_provider,
    build_enrichment_providers,
    build_multimodal_provider,
    build_reranker,
    build_text_provider,
    read_prompt_text,
    resolve_caption_prompt,
    resolve_prompt_path,
)
from imsg.providers.manifest import verify_manifest
from imsg.retrieval.service import RetrievalService
from imsg.segment.pipeline import REBUILD_ALL_SENTINEL, run_segment, run_segment_for_chat
from imsg.stages.extract import run_extract
from imsg.stages.identity import (
    assign_handle,
    compute_invariant_report,
    merge_persons,
    rename_person,
    run_identity,
)
from imsg.stages.identity_overrides import (
    apply_overrides,
    export_overrides,
    load_overrides,
    write_overrides,
)
from imsg.stages.imsg_dump import default_binary_path
from imsg.stages.snapshot import SNAPSHOT_FILENAME, SNAPSHOT_SUBDIR, run_snapshot
from imsg.stages.sync import EmbedFn, SegmentFn, run_sync, run_sync_all_sources
from imsg.verify.cli import reconcile_attachments, verify_seed

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config

app = typer.Typer(
    name="imsg",
    help="Local-first iMessage retrieval index: extraction, identity, "
    "segmentation, hybrid search, and a scoped MCP surface.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

mcp_app = typer.Typer(name="mcp", help="MCP surfaces (SPEC §10).", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")
models_app = typer.Typer(
    name="models",
    help="Model pins: models/manifest.lock.yaml (SPEC model-manifest requirement).",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")
app.command("verify-seed")(verify_seed)
app.command("reconcile-attachments")(reconcile_attachments)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to config.yaml. Defaults to $IMSG_CONFIG, then ./config.yaml.",
    ),
]

DryRunOption = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Preview what this stage would do without writing anything (SPEC §8).",
    ),
]

DRY_RUN_MARKER = "DRY RUN — nothing was written"
"""Printed verbatim (and grep-able) on every `--dry-run` invocation's
output, alongside the normal report line (SPEC §8: "takes --dry-run
where writes leave the machine")."""


def _repo_root() -> Path:
    # src/imsg/cli.py -> src/imsg -> src -> repo root
    return Path(__file__).resolve().parents[2]


def _default_migrations_dir() -> Path:
    return _repo_root() / "migrations"


def _load_config_or_die(config_path: Path | None) -> Config:
    try:
        return load_config(config_path)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _connect_and_verify_or_die(cfg: Config) -> psycopg.Connection:
    """Every stage command's standard DB entry point: connect, then
    verify the two-sided cluster fingerprint (CLAUDE.md non-negotiable
    #6, SPEC §5.2) before touching anything — "Every process, before
    migrations or queries, connects and checks ... there is no bypass
    flag." `migrate` is the one exception (it *bootstraps* the
    fingerprint after applying migration 0001, so it cannot verify one
    that does not exist yet)."""
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

    # Assert the connection is idle before any stage touches it.
    #
    # `connect()` sets autocommit=True precisely so that reads — including
    # the fingerprint `SHOW` above — never open a transaction, which makes
    # every `with conn.transaction():` in the stages a genuine top-level
    # transaction that commits on exit. See db/connection.py for the full
    # incident: with autocommit off, every stage silently rolled back its
    # writes while reporting success.
    #
    # An earlier fix issued `conn.rollback()` here instead. That was wrong in
    # kind, not just degree: it closed only the transaction THIS function
    # opened, so any stage whose first statement was a bare read re-opened
    # one immediately — `segment`, `embed`, `enrich` and `sync` all still
    # lost their writes. This assertion is what a regression should hit,
    # loudly, instead of silently discarding a multi-hour run.
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        conn.close()
        typer.echo(
            "imsg: internal error — connection is not idle after the fingerprint "
            "check, so stage writes would nest as savepoints and be discarded at "
            "close. Check that db.connection.connect() still sets autocommit=True.",
            err=True,
        )
        raise typer.Exit(code=1)
    return conn


def _fts_db_path(cfg: Config) -> Path:
    return cfg.paths.data_root / "fts" / "fts.db"


def _open_fts_conn(cfg: Config) -> apsw.Connection:
    path = _fts_db_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(path))
    create_schema(conn)
    return conn


def _echo_prompt_line(label: str, resolved: ResolvedPrompt) -> None:
    """Which prompt file a run used and where it came from — printed like
    the backend line, because the bytes feed `seg_config_hash` /
    `prompt_sha256` and a silent fallback to the shipped copy would be
    invisible in the run's output otherwise."""
    typer.echo(f"{label}: {resolved.path} ({resolved.description})")


def _boundary_prompt_bytes_or_die(cfg: Config) -> bytes:
    relative = cfg.segmentation.boundary_prompt
    resolved = resolve_prompt_path(cfg.paths.data_root, relative)
    if resolved is None:
        typer.echo(
            f"imsg: segmentation boundary prompt not found at "
            f"'{cfg.paths.data_root / relative}' (and the repository ships no '{relative}' "
            f"to fall back to) — author it before running segmentation "
            f"(SPEC §6 segmentation.boundary_prompt)",
            err=True,
        )
        raise typer.Exit(code=1)
    _echo_prompt_line("segmentation prompt", resolved)
    try:
        return resolved.path.read_bytes()
    except OSError as exc:
        typer.echo(f"imsg: segmentation boundary prompt '{resolved.path}' could not be read: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _caption_prompt_or_die(cfg: Config) -> str | None:
    """The fixed captioning prompt for the real backend, printed with its
    provenance like the boundary prompt. The fake backend never reads one."""
    if cfg.models.backend != "real":
        return None
    resolved = _build_or_die(lambda: resolve_caption_prompt(cfg))
    _echo_prompt_line("caption prompt", resolved)
    return _build_or_die(
        lambda: read_prompt_text(resolved.path, field_name="enrichment.caption_prompt")
    )


def _echo_backend_line(cfg: Config, *, err: bool = False) -> None:
    """`models: backend=<real|fake>` — every command that builds
    providers prints this before doing anything else (to stderr for the
    stdio MCP server, whose stdout is the JSON-RPC channel)."""
    typer.echo(backend_status_line(cfg), err=err)


def _build_or_die[T](build: Callable[[], T]) -> T:
    """Construct providers at the command boundary: a missing provider
    module, missing runtime package, missing prompt file, or failed
    model load is one clean `imsg: ...` line and exit 1, never a
    traceback (see `imsg.providers.factory`)."""
    try:
        return build()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _decode_prompt(prompt_bytes: bytes) -> str:
    # Same decode the segmentation config hash applies to these bytes
    # (imsg.segment.hashing), so the model sees exactly what was hashed.
    return prompt_bytes.decode("utf-8", "replace")


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
    info = run_guard_mount_or_exit(cfg.paths.data_root)
    typer.echo(f"guard-mount: ok — '{cfg.paths.data_root}' is on encrypted volume "
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
    run_guard_mount_or_exit(cfg.paths.data_root)

    mdir = migrations_dir or _default_migrations_dir()

    try:
        conn = connect(cfg.database)
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
                conn, cfg.paths.data_root, cfg.database.cluster_fingerprint_file
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

    mount = check_mount(cfg.paths.data_root)
    posture = check_at_rest_posture(cfg.paths.data_root)
    fda = check_full_disk_access(cfg.paths.live_chat_db)
    pg = check_postgres(cfg)

    report = {
        "full_disk_access": fda,
        "contacts_access": None,
        "contacts_access_note": (
            "not checked by this CLI command — see the 'check_permissions' MCP "
            "tool (imsg.mcp.tools.handlers), which does check it via the "
            "Contacts framework"
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
        "pipeline_note": "sync/index-freshness/watermarks unavailable from this "
        "CLI command — see the 'check_permissions' MCP tool, which reports them "
        "from Postgres now that S1-S3/S7 are built",
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

    mount = check_mount(cfg.paths.data_root)
    posture = check_at_rest_posture(cfg.paths.data_root)
    pg = check_postgres(cfg)
    free_bytes = disk_free_bytes(cfg.paths.data_root)

    report = {
        "models_backend": cfg.models.backend,
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
        "pipeline_note": "fields above report None until wired to the now-built "
        "pipeline stages (a later revision of this command's own scope, not this "
        "build's task list) — 'imsg check-permissions'/the MCP check_permissions "
        "tool already reports last_sync_at/watermarks",
    }

    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return

    _echo_backend_line(cfg)
    for key, value in report.items():
        if key == "models_backend":
            continue  # already printed in its canonical `models: backend=...` form
        typer.echo(f"{key}: {value}")


@models_app.command("verify")
def models_verify(
    lock: Annotated[
        Path | None,
        typer.Option(
            "--lock",
            help="Path to manifest.lock.yaml (defaults to the repo's models/manifest.lock.yaml).",
        ),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(
            "--write", help="Accept drift: rewrite drifted entries' revision/license in the lock."
        ),
    ] = False,
    data_root: Annotated[
        Path | None,
        typer.Option(
            "--data-root",
            help="paths.data_root, under which local conversions' output_dir live "
            "(default: the config schema's default data root).",
        ),
    ] = None,
    skip_remote: Annotated[
        bool, typer.Option("--skip-remote", help="Do not contact the Hugging Face API.")
    ] = False,
    skip_artifacts: Annotated[
        bool,
        typer.Option(
            "--skip-artifacts",
            help="Do not check local conversions' directories under data_root.",
        ),
    ] = False,
    skip_runtime: Annotated[
        bool, typer.Option("--skip-runtime", help="Do not check installed runtime packages.")
    ] = False,
) -> None:
    """Re-resolve every pinned model's current revision and license from
    the Hugging Face API (a local conversion's upstream repo) and report
    drift versus models/manifest.lock.yaml; check every local conversion's
    directory under data_root against its recorded artifact_sha256; check
    installed runtime packages against its `min_runtime` floors. Never
    modifies the lock without --write (SPEC: a build must not silently
    advance a model because 'latest' changed), and never advances a local
    conversion's upstream pin."""
    code = verify_manifest(
        lock,
        data_root=data_root,
        write=write,
        skip_remote=skip_remote,
        skip_artifacts=skip_artifacts,
        skip_runtime=skip_runtime,
        out=sys.stdout,
    )
    if code != 0:
        raise typer.Exit(code=code)


@app.command()
def snapshot(config: ConfigOption = None, dry_run: DryRunOption = False) -> None:
    """S1 — snapshot the live chat.db via the SQLite online-backup API (SPEC §8 S1)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    try:
        result = run_snapshot(
            live_chat_db=cfg.paths.live_chat_db, data_root=cfg.paths.data_root, dry_run=dry_run
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"snapshot: {result.path} sha256={result.sha256} "
        f"reused_existing={result.reused_existing}"
    )
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


def _validate_seed_or_die(cfg: Config, snapshot: Path | None, source: str | None) -> None:
    """Gate the one-shot seed path (SPEC §8 S7): `--snapshot` feeds a prepared
    database straight to S2, bypassing S1.

    Three rules, all fail-closed, because the failures they prevent are
    silent and permanent. A seed advances the **ROWID watermark of whatever
    source it is ingested under**, and a ROWID means something only inside
    one database file. Seed a file whose ROWIDs run past the live source's
    and the watermark jumps past real messages that were never looked at —
    they are then below the watermark forever, and nothing reports rows it
    never read. So a seed must carry its own `--source`, and that source
    must not be one the pipeline is actively snapshotting. And the file
    must not be the live `chat.db` (or anything beside it): the pipeline
    reads the live database exactly once, through S1's SQLite `.backup`,
    and nothing else may ever open it (CLAUDE.md non-negotiable #1). That
    comparison is by inode as well as by resolved path — a hard link, or
    the macOS `/System/Volumes/Data/…` firmlink alias of `~`, is a
    different string for the same file.
    """
    if snapshot is None:
        return
    if source is None:
        typer.echo(
            "imsg: --snapshot requires --source. A seed must be ingested under its own "
            "source name so it gets its own ROWID watermark namespace.",
            err=True,
        )
        raise typer.Exit(code=1)
    configured = {s.name for s in cfg.sync.sources}
    if source in configured:
        typer.echo(
            f"imsg: --snapshot may not be used with --source '{source}' — that is a "
            f"configured sync.sources entry, which the pipeline snapshots from the live "
            f"chat.db. Seeding it would advance that source's ROWID watermark to this "
            f"file's row count; every live message below the new watermark is then "
            f"skipped forever, with no error. Use a fresh source name for the seed.",
            err=True,
        )
        raise typer.Exit(code=1)
    # Compared by *resolved* path, so a symlink or `..` cannot dodge the
    # check (SPEC §5.4: never infer containment from a string prefix), and
    # by inode, so neither can a hard link or the macOS firmlink alias.
    seed_resolved = resolve_path(snapshot)
    live_databases = [("paths.live_chat_db", resolve_path(cfg.paths.live_chat_db))]
    live_databases += [
        (f"sync.sources[{i}] ({s.name}).chat_db", resolve_path(s.chat_db))
        for i, s in enumerate(cfg.sync.sources)
    ]
    for label, live in live_databases:
        if is_same_file(seed_resolved, live):
            typer.echo(
                f"imsg: --snapshot '{snapshot}' is the live Messages database ({label}). "
                f"A seed must be a snapshot or a prepared copy — the pipeline never opens "
                f"chat.db directly; S1's SQLite backup is its only reader (CLAUDE.md "
                f"non-negotiable #1: never write to the live chat.db). Run 'imsg snapshot', "
                f"or copy the file under data_root, and seed from that.",
                err=True,
            )
            raise typer.Exit(code=1)
    live_dir = resolve_path(cfg.paths.live_chat_db).parent
    if is_contained_in(seed_resolved, live_dir):
        typer.echo(
            f"imsg: --snapshot '{snapshot}' resolves inside the live Messages directory "
            f"'{live_dir}' (the live chat.db, its -wal/-shm sidecars and the attachment "
            f"tree live there). Nothing in this pipeline may open a database there "
            f"directly (CLAUDE.md non-negotiable #1) — copy the file under data_root first.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not snapshot.is_file():
        typer.echo(f"imsg: --snapshot file not found: '{snapshot}'", err=True)
        raise typer.Exit(code=1)


@app.command()
def extract(
    config: ConfigOption = None,
    source: Annotated[
        str | None,
        typer.Option(help="Source name from sync.sources; defaults to the first configured source."),
    ] = None,
    snapshot: Annotated[
        Path | None,
        typer.Option(
            help="One-shot seed: extract from this prepared database instead of the "
            "pipeline snapshot. Requires --source, which must not name a configured "
            "sync.sources entry.",
        ),
    ] = None,
    dry_run: DryRunOption = False,
) -> None:
    """S2 — extract chats/messages/attachments from the current snapshot (SPEC §8 S2).

    `--snapshot` is the seed path: it never touches `snapshots/snapshot.db`,
    which S1 atomically replaces from the live chat.db every
    `sync.interval_seconds` — anything staged there is destroyed on the next
    tick. It requires `--source` because a seed advances that source's ROWID
    watermark, and ROWIDs mean nothing across database files.
    """
    cfg = _load_config_or_die(config)
    _validate_seed_or_die(cfg, snapshot, source)
    run_guard_mount_or_exit(cfg.paths.data_root)

    source_name = source or cfg.sync.sources[0].name
    snapshot_path = snapshot or cfg.paths.data_root / SNAPSHOT_SUBDIR / SNAPSHOT_FILENAME
    if not snapshot_path.is_file():
        typer.echo(
            f"imsg: no snapshot found at '{snapshot_path}' — run 'imsg snapshot' first", err=True
        )
        raise typer.Exit(code=1)

    conn = _connect_and_verify_or_die(cfg)
    try:
        result = run_extract(
            conn=conn,
            source_name=source_name,
            snapshot_path=snapshot_path,
            imsg_dump_binary=default_binary_path(_repo_root()),
            dry_run=dry_run,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(
        f"extract: messages_upserted={result.messages_upserted} "
        f"watermark {result.watermark_before}->{result.watermark_after} "
        f"bodies_missing={result.bodies_missing}"
    )
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


identity_app = typer.Typer(
    name="identity",
    help="S3 — resolve handles to person_id, then curate them (SPEC §8 S3).",
)
app.add_typer(identity_app, name="identity")


def _identity_import(cfg: Config, dry_run: bool) -> None:
    """The S3 resolve pass. Shared by `identity` and `identity import` so the
    bare form keeps working — it was the only form that existed before the
    curation subcommands landed."""
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        result = run_identity(conn=conn, config=cfg, dry_run=dry_run)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"identity: persons_created={result.persons_created} "
        f"handles_created={result.handles_created} invariant_ok={result.invariant.ok}"
    )
    # Always report the Contacts outcome, not only on degrade.
    #
    # `contacts_loaded` was computed and discarded before 2026-08-15, so a run
    # that reached Contacts successfully and matched NOTHING was indistinguishable
    # from a healthy one: the summary line looked identical either way. That is
    # exactly what happened on the first real run — 10,725 stubs, one named
    # person (the owner), and no signal at all that anything was wrong.
    # A zero here is the single number that separates "Contacts returned nothing"
    # from "returned plenty and none of it matched", which are different bugs.
    typer.echo(
        f"identity: contacts attempted={result.contacts.attempted} "
        f"loaded={result.contacts.contacts_loaded}"
    )
    if result.contacts.attempted and result.contacts.contacts_loaded == 0 and not result.contacts.degraded:
        typer.echo(
            "identity: WARNING Contacts access succeeded but returned 0 records — "
            "every handle will become an unnamed review stub. Check that the "
            "account holding your contacts has Contacts enabled and has synced.",
            err=True,
        )
    if result.contacts.degraded:
        typer.echo(
            f"identity: WARNING contacts import degraded: {result.contacts.degraded_reason}",
            err=True,
        )
    if not result.invariant.ok:
        typer.echo(
            "identity: invariant NOT satisfied — segmentation (S4) must not run "
            "until this is clean (SPEC §8 S3)",
            err=True,
        )
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


@identity_app.callback(invoke_without_command=True)
def identity(
    ctx: typer.Context,
    config: ConfigOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """S3 — resolve handles to person_id via Contacts + manual curation (SPEC §8 S3)."""
    if ctx.invoked_subcommand is not None:
        # Options parsed HERE belong to the group, not the subcommand, and
        # typer will not forward them. Silently dropping them was dangerous:
        # `identity --config other.yaml merge --keep A --absorb B` merged
        # persons in the DEFAULT database, and `identity --dry-run import`
        # performed real writes. Fail loudly instead of guessing.
        if config is not None or dry_run:
            typer.echo(
                f"imsg: put --config/--dry-run AFTER the subcommand "
                f"(e.g. 'imsg identity {ctx.invoked_subcommand} --config ...'). "
                "Options before a subcommand apply to the group and would be ignored.",
                err=True,
            )
            raise typer.Exit(code=2)
        return
    _identity_import(_load_config_or_die(config), dry_run)


@identity_app.command("import")
def identity_import(config: ConfigOption = None, dry_run: DryRunOption = False) -> None:
    """Resolve every source handle to a person_id (Contacts + review stubs)."""
    _identity_import(_load_config_or_die(config), dry_run)


@identity_app.command("review-report")
def identity_review_report(
    config: ConfigOption = None,
    limit: Annotated[int, typer.Option(help="How many persons to list.")] = 40,
    all_persons: Annotated[
        bool, typer.Option("--all", help="Include persons already reviewed, not just needs_review.")
    ] = False,
    sample: Annotated[
        bool,
        typer.Option(
            "--sample",
            help="Show one representative message per person — usually identifies them outright.",
        ),
    ] = False,
) -> None:
    """The curation worklist — persons ranked by message volume (SPEC §8 S3).

    Ordered by messages descending on purpose: the value of this review is
    concentrated in your top correspondents, and a long tail of one-message
    strangers never needs a name.
    """
    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        report = compute_invariant_report(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.person_id, p.display_name, p.needs_review,
                       count(DISTINCT h.handle_id) AS handles,
                       count(DISTINCT m.message_id) AS messages
                FROM person p
                LEFT JOIN handle h ON h.person_id = p.person_id
                LEFT JOIN message m ON m.sender_person_id = p.person_id
                WHERE (%s OR p.needs_review)
                GROUP BY p.person_id, p.display_name, p.needs_review
                ORDER BY messages DESC, handles DESC, p.person_id
                LIMIT %s
                """,
                (all_persons, limit),
            )
            rows = cur.fetchall()
            samples: dict[int, str] = {}
            if sample and rows:
                # One message per person, and it is very often decisive: automated
                # senders announce themselves outright ("Acme Marine: ... Battery
                # Low", "Your HomeGuard system..."), which no amount of contact
                # lookup would ever have revealed. Added 2026-08-15 after chasing
                # several unknown numbers through Gmail and web search when the
                # answer was sitting in the corpus the whole time.
                cur.execute(
                    """
                    SELECT DISTINCT ON (m.sender_person_id)
                           m.sender_person_id,
                           left(regexp_replace(m.text_original, E'[\n\r]+', ' ', 'g'), 100)
                    FROM message m
                    WHERE m.sender_person_id = ANY(%s)
                      AND m.text_original IS NOT NULL
                      AND length(m.text_original) BETWEEN 20 AND 400
                    ORDER BY m.sender_person_id, m.sent_at DESC
                    """,
                    ([r[0] for r in rows],),
                )
                samples = {int(pid): txt for pid, txt in cur.fetchall()}
            cur.execute("SELECT count(*) FROM person WHERE needs_review")
            pending = cur.fetchone()
            cur.execute("SELECT count(*) FROM person")
            total = cur.fetchone()
    finally:
        conn.close()

    typer.echo(f"persons: {total[0] if total else 0} total, {pending[0] if pending else 0} needing review")
    typer.echo(
        "invariant: "
        f"unresolved_message_senders={report.unresolved_message_senders} "
        f"unresolved_tapback_senders={report.unresolved_tapback_senders} "
        f"unresolved_chat_participants={report.unresolved_chat_participants} "
        f"ok={report.ok}"
    )
    typer.echo("")
    typer.echo(f"{'id':>7}  {'msgs':>8}  {'hdls':>5}  {'review':<7} name")
    for pid, name, needs, handles, messages in rows:
        flag = "YES" if needs else "-"
        typer.echo(f"{pid:>7}  {messages:>8}  {handles:>5}  {flag:<7} {name}")
        if sample:
            typer.echo(f"{'':>7}  {'':>8}  {'':>5}  {'':<7} \u21b3 {samples.get(pid, '(no text messages)')}")


@identity_app.command("duplicate-candidates")
def identity_duplicate_candidates(
    config: ConfigOption = None,
    limit: Annotated[int, typer.Option(help="Max candidate pairs to list.")] = 60,
) -> None:
    """Persons that may be the same human — ranked, for a human to scan.

    Deliberately RANKS rather than merges. Merging is irreversible and a
    wrong call fuses two real people's messages under one `person_id`,
    which non-negotiable #3 makes load-bearing everywhere downstream. So
    this surfaces evidence and lets the operator decide.

    Signals, strongest first:
      * one Contacts card whose identifiers landed on several persons —
        the address book itself says they are one human;
      * one name being an abbreviation/prefix of another (Jeff/Jeffrey),
        which the override applier's exact match cannot catch;
      * identical names (already handled on rename, listed for completeness);
      * shared group chats plus non-overlapping active windows, the
        signature of somebody who changed number.
    """
    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH pm AS (
                  SELECT p.person_id, p.display_name, p.needs_review,
                         (SELECT count(*) FROM message m WHERE m.sender_person_id=p.person_id) AS msgs,
                         (SELECT min(m.sent_at) FROM message m WHERE m.sender_person_id=p.person_id) AS first_at,
                         (SELECT max(m.sent_at) FROM message m WHERE m.sender_person_id=p.person_id) AS last_at
                  FROM person p WHERE NOT p.is_owner
                )
                SELECT a.person_id, a.display_name, a.msgs,
                       b.person_id, b.display_name, b.msgs,
                       (SELECT count(*) FROM chat_participant ca
                         JOIN chat_participant cb ON cb.chat_id=ca.chat_id
                        WHERE ca.person_id=a.person_id AND cb.person_id=b.person_id) AS shared_chats,
                       (a.last_at < b.first_at OR b.last_at < a.first_at) AS disjoint
                FROM pm a JOIN pm b ON a.person_id < b.person_id
                WHERE a.msgs > 0 AND b.msgs > 0
                  AND (
                    lower(a.display_name) = lower(b.display_name)
                    OR (length(a.display_name) > 3
                        AND position(lower(a.display_name) in lower(b.display_name)) = 1)
                    OR (length(b.display_name) > 3
                        AND position(lower(b.display_name) in lower(a.display_name)) = 1)
                  )
                ORDER BY (a.msgs + b.msgs) DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        typer.echo("no duplicate candidates found")
        return
    typer.echo(f"{len(rows)} candidate pair(s) — same person? merge with:")
    typer.echo("  imsg identity merge --keep <id> --absorb <id>\n")
    conn2 = _connect_and_verify_or_die(cfg)
    try:
        handles: dict[int, list[str]] = {}
        with conn2.cursor() as cur:
            cur.execute("SELECT person_id, normalized_value FROM handle ORDER BY person_id")
            for pid, val in cur.fetchall():
                handles.setdefault(int(pid), []).append(str(val))
    finally:
        conn2.close()

    def _links(a: list[str], b: list[str]) -> str:
        """Evidence connecting two candidates, computed from their identifiers.

        Cheap signals that a human would use anyway: a shared email domain,
        an email whose local part echoes the other side, or numbers from the
        same area code. None of these are conclusive — they are shown so the
        operator can decide faster, never so the tool can decide for them.
        """
        out = []
        dom_a = {x.split("@", 1)[1] for x in a if "@" in x}
        dom_b = {x.split("@", 1)[1] for x in b if "@" in x}
        shared = dom_a & dom_b
        if shared:
            out.append("same email domain: " + ", ".join(sorted(shared)))
        ac_a = {x[2:5] for x in a if x.startswith("+1") and len(x) >= 12}
        ac_b = {x[2:5] for x in b if x.startswith("+1") and len(x) >= 12}
        if ac_a & ac_b:
            out.append("same area code: " + ", ".join(sorted(ac_a & ac_b)))
        return "; ".join(out)

    for a_id, a_name, a_n, b_id, b_name, b_n, chats, disjoint in rows:
        flags = []
        if disjoint:
            flags.append("no time overlap")
        if chats:
            flags.append(f"{chats} shared chat(s)")
        ha, hb = handles.get(a_id, []), handles.get(b_id, [])
        link = _links(ha, hb)
        if link:
            flags.append(link)
        typer.echo("")
        typer.echo(f"  {a_n + b_n:>6} total   {' | '.join(flags) if flags else ''}")
        typer.echo(f"    {a_id:>7} {a_n:>6}  {a_name}")
        typer.echo(f"            {'':>6}  {', '.join(ha) or '(no handles)'}")
        typer.echo(f"    {b_id:>7} {b_n:>6}  {b_name}")
        typer.echo(f"            {'':>6}  {', '.join(hb) or '(no handles)'}")


@identity_app.command("merge")
def identity_merge(
    keep: Annotated[int, typer.Option(help="person_id to KEEP.")],
    absorb: Annotated[int, typer.Option(help="person_id to absorb and delete.")],
    config: ConfigOption = None,
) -> None:
    """Fold `absorb` into `keep` — handles, messages, tapbacks, participants."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        merge_persons(conn, keep_person_id=keep, absorb_person_id=absorb)
        conn.commit()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"identity: merged person {absorb} into {keep}")


@identity_app.command("rename")
def identity_rename(
    person: Annotated[int, typer.Option(help="person_id to rename.")],
    name: Annotated[str, typer.Option(help="New display_name.")],
    short: Annotated[str | None, typer.Option(help="Optional new short_name.")] = None,
    config: ConfigOption = None,
) -> None:
    """Set a person's display name (and optionally short name)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        rename_person(conn, person_id=person, display_name=name, short_name=short)
        conn.commit()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"identity: renamed person {person} to {name!r}")


@identity_app.command("assign")
def identity_assign(
    value: Annotated[str, typer.Option(help="Normalized handle value (E.164 phone or email).")],
    kind: Annotated[str, typer.Option(help="Handle kind: 'phone' or 'email'.")],
    person: Annotated[int, typer.Option(help="person_id to attach it to.")],
    config: ConfigOption = None,
) -> None:
    """Repoint one canonical handle onto a different person."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        assign_handle(conn, normalized_value=value, kind=kind, person_id=person)
        conn.commit()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"identity: assigned {kind} {value!r} to person {person}")


OverridesPathArgument = Annotated[
    Path,
    typer.Argument(
        help="The curated-decisions file (JSON; schema in imsg.stages.identity_overrides). "
        "It carries real names and identifiers: keep it under paths.data_root.",
    ),
]


@identity_app.command("apply-overrides")
def identity_apply_overrides(
    path: OverridesPathArgument,
    config: ConfigOption = None,
    dry_run: DryRunOption = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Also apply decisions that conflict with the current person table (a handle "
            "now on a person named since, or a person renamed since). The owner person and an "
            "ambiguous target are never forced.",
        ),
    ] = False,
) -> None:
    """Replay curated decisions (rename/assign/merge) idempotently — run after any identity rebuild.

    Every decision is keyed on its normalized identifier, so it also finds
    handles created after the file was written. Already-applied decisions
    are no-ops; identifiers absent from the index are reported as
    `unmatched`, never invented; conflicts are reported and skipped unless
    `--force`. Ends with the S3 invariant report and a one-line summary.
    """
    cfg = _load_config_or_die(config)
    try:
        decisions = load_overrides(path)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        result = apply_overrides(
            conn,
            overrides=decisions.overrides,
            default_region=cfg.identity.default_region,
            dry_run=dry_run,
            force=force,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(f"identity: {len(decisions.overrides)} decision(s) from {path}")
    for outcome in result.outcomes:
        if outcome.status == "applied":
            forced = " (forced)" if outcome.forced else ""
            typer.echo(f"identity: applied{forced} {outcome.detail}")
        elif outcome.status in ("unmatched", "conflict"):
            typer.echo(f"identity: {outcome.status} {outcome.detail}", err=True)
    typer.echo(
        "identity: invariant "
        f"unresolved_message_senders={result.invariant.unresolved_message_senders} "
        f"unresolved_tapback_senders={result.invariant.unresolved_tapback_senders} "
        f"unresolved_chat_participants={result.invariant.unresolved_chat_participants} "
        f"ok={result.invariant.ok}"
    )
    if not result.invariant.ok:
        typer.echo(
            "identity: invariant NOT satisfied — segmentation (S4) must not run "
            "until this is clean (SPEC §8 S3)",
            err=True,
        )
    typer.echo(f"identity: {result.summary}")
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


@identity_app.command("export-overrides")
def identity_export_overrides(path: OverridesPathArgument, config: ConfigOption = None) -> None:
    """Write the current curated decisions as an overrides file — regenerate it after hand-curation.

    One decision per handle of every reviewed (or multi-handle) non-owner
    person. An existing file at `path` is merged in: its `was`/`why` are
    kept, and decisions the index cannot currently express are carried
    forward rather than dropped.
    """
    cfg = _load_config_or_die(config)
    if not is_contained_in(path, cfg.paths.data_root):
        typer.echo(
            f"imsg: refusing to write {path}: the decisions file carries every named handle "
            f"and belongs under paths.data_root ({cfg.paths.data_root}) — CLAUDE.md "
            "non-negotiable #2",
            err=True,
        )
        raise typer.Exit(code=1)
    previous = None
    if path.exists():
        try:
            previous = load_overrides(path)
        except ImsgError as exc:
            typer.echo(f"imsg: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        exported = export_overrides(
            conn, default_region=cfg.identity.default_region, previous=previous
        )
    finally:
        conn.close()
    try:
        write_overrides(path, exported.file)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"identity: exported {exported.exported} decision(s) to {path} "
        f"(carried_forward={exported.carried_forward}, total={len(exported.file.overrides)})"
    )


@app.command()
def segment(
    config: ConfigOption = None,
    chat: Annotated[int | None, typer.Option(help="Restrict to one chat_id.")] = None,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild", help="Force a full rebuild of --chat (e.g. after a config change)."
        ),
    ] = False,
    dry_run: DryRunOption = False,
) -> None:
    """S4 — sessionize and segment messages for indexing (SPEC §8 S4)."""
    cfg = _load_config_or_die(config)
    if rebuild and chat is None:
        typer.echo("imsg: --rebuild requires --chat <id>", err=True)
        raise typer.Exit(code=2)
    run_guard_mount_or_exit(cfg.paths.data_root)

    prompt_bytes = _boundary_prompt_bytes_or_die(cfg)
    _echo_backend_line(cfg)
    provider = _build_or_die(lambda: build_boundary_provider(cfg, _decode_prompt(prompt_bytes)))

    conn = _connect_and_verify_or_die(cfg)
    try:
        if rebuild:
            assert chat is not None
            report = run_segment_for_chat(
                conn,
                chat,
                cfg,
                provider,
                prompt_bytes,
                earliest_changed_at=REBUILD_ALL_SENTINEL,
                dry_run=dry_run,
            )
            typer.echo(
                f"segment: chat {chat} rebuilt — segments_written={report.segments_written} "
                f"segments_deleted={report.segments_deleted}"
            )
        else:
            chat_ids = {chat} if chat is not None else None
            reports = run_segment(conn, cfg, provider, prompt_bytes, chat_ids=chat_ids, dry_run=dry_run)
            total_written = sum(r.segments_written for r in reports)
            total_fallback = sum(r.fallback_sessions for r in reports)
            typer.echo(
                f"segment: {len(reports)} chat(s) processed, {total_written} segment(s) "
                f"written, {total_fallback} fallback session(s)"
            )
        if dry_run:
            typer.echo(DRY_RUN_MARKER)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()


@app.command()
def embed(config: ConfigOption = None, dry_run: DryRunOption = False) -> None:
    """S6 — embed segments/attachment chunks and update the FTS sidecar (SPEC §8 S6)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    _echo_backend_line(cfg)
    # Providers first, connection second: a missing model runtime fails
    # fast, before anything is opened.
    text_provider = _build_or_die(lambda: build_text_provider(cfg))
    multimodal_provider = _build_or_die(lambda: build_multimodal_provider(cfg))

    conn = _connect_and_verify_or_die(cfg)
    # `_open_fts_conn` creates `fts/` and the sqlite sidecar file on disk
    # (a real filesystem write) — skip it entirely in dry-run mode, since
    # `run_embed(dry_run=True)` never needs it and a dry run must not
    # write anything (SPEC §8).
    fts_conn = _open_fts_conn(cfg) if not dry_run else None
    try:
        report = run_embed(
            conn,
            text_provider,
            multimodal_provider=multimodal_provider,
            batch_size=cfg.embedding.batch_size,
            max_batch_tokens=cfg.embedding.max_batch_tokens,
            dry_run=dry_run,
        )
        if not dry_run:
            assert fts_conn is not None
            sync_report = sync_fts(conn, fts_conn)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if fts_conn is not None:
            fts_conn.close()
        conn.close()

    typer.echo(
        f"embed: segments_embedded={report.segments_embedded} "
        f"chunks_embedded={report.chunks_embedded} "
        f"attachments_embedded={report.attachments_embedded} "
        f"attachments_failed={report.attachments_failed}"
    )
    if dry_run:
        typer.echo(DRY_RUN_MARKER)
    else:
        typer.echo(
            f"embed: fts events_applied={sync_report.events_applied} "
            f"(upserts={sync_report.upserts} deletes={sync_report.deletes})"
        )


def _make_segment_fn(cfg: Config) -> SegmentFn:
    prompt_bytes = _boundary_prompt_bytes_or_die(cfg)
    provider = _build_or_die(lambda: build_boundary_provider(cfg, _decode_prompt(prompt_bytes)))

    def _segment_fn(conn: psycopg.Connection, config: Config, *, dry_run: bool = False) -> object:
        return run_segment(conn, config, provider, prompt_bytes, dry_run=dry_run)

    return _segment_fn


def _make_embed_fn(cfg: Config) -> EmbedFn:
    # Built once per `sync` process, up front: a real embedder loads model
    # weights, and `run_sync_all_sources` invokes this once per source.
    text_provider = _build_or_die(lambda: build_text_provider(cfg))
    multimodal_provider = _build_or_die(lambda: build_multimodal_provider(cfg))

    def _embed_fn(conn: psycopg.Connection, config: Config, *, dry_run: bool = False) -> object:
        report = run_embed(
            conn,
            text_provider,
            multimodal_provider=multimodal_provider,
            batch_size=config.embedding.batch_size,
            max_batch_tokens=config.embedding.max_batch_tokens,
            dry_run=dry_run,
        )
        if dry_run:
            # See `embed`'s own CLI command: opening the FTS sidecar
            # creates it on disk, a real write a dry run must not do.
            return report
        fts_conn = _open_fts_conn(config)
        try:
            sync_fts(conn, fts_conn)
        finally:
            fts_conn.close()
        return report

    return _embed_fn


@app.command()
def sync(
    config: ConfigOption = None,
    source: Annotated[
        str | None,
        typer.Option(help="Sync only this source instead of every configured one."),
    ] = None,
    snapshot: Annotated[
        Path | None,
        typer.Option(
            help="One-shot seed: skip S1 and feed this prepared database straight to S2 "
            "(SPEC §8 S7). Requires --source, which must not name a configured "
            "sync.sources entry.",
        ),
    ] = None,
    dry_run: DryRunOption = False,
) -> None:
    """S7 — incremental S1→S2→S3→S4→S6 sync for every configured source (SPEC §8 S7).

    With `--source`, syncs that one source. With `--source` and `--snapshot`,
    runs the studio-seed one-shot path SPEC §8 S7 specifies: S1 is skipped
    entirely and the already-prepared file feeds S2→S3→S4→S6, so the seed
    lands under its own source name and its own ROWID watermark.
    """
    cfg = _load_config_or_die(config)
    _validate_seed_or_die(cfg, snapshot, source)
    _echo_backend_line(cfg)
    segment_fn = _make_segment_fn(cfg)  # validates the boundary prompt exists up front
    embed_fn = _make_embed_fn(cfg)

    conn = _connect_and_verify_or_die(cfg)
    try:
        if source is not None:
            results = [
                run_sync(
                    conn=conn,
                    config=cfg,
                    source_name=source,
                    imsg_dump_binary=default_binary_path(_repo_root()),
                    snapshot_override=snapshot,
                    segment_fn=segment_fn,
                    embed_fn=embed_fn,
                    dry_run=dry_run,
                )
            ]
        else:
            results = run_sync_all_sources(
                conn=conn,
                config=cfg,
                imsg_dump_binary=default_binary_path(_repo_root()),
                segment_fn=segment_fn,
                embed_fn=embed_fn,
                dry_run=dry_run,
            )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    for r in results:
        if r.extract is None:
            typer.echo(f"sync: source={r.source_name} {r.note}")
            continue
        typer.echo(
            f"sync: source={r.source_name} messages_upserted={r.extract.messages_upserted} "
            f"segment_ran={r.segment_ran} embed_ran={r.embed_ran}"
        )
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


@app.command()
def enrich(
    config: ConfigOption = None,
    limit: Annotated[int, typer.Option(help="Max tasks to claim and process this run.")] = 100,
    worker_id: Annotated[
        str, typer.Option(help="Lease owner id (SPEC §8 S5b lease/backoff).")
    ] = "cli",
    retry_failed: Annotated[
        bool,
        typer.Option("--retry-failed", help="Reset permanently-failed tasks to pending first."),
    ] = False,
    dry_run: DryRunOption = False,
) -> None:
    """S5b — OCR/caption/transcribe/pdftotext enrichment queue worker (SPEC §8 S5b)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    _echo_backend_line(cfg)
    # Providers before the DB connection so a missing model runtime fails
    # fast; a dry run claims and dispatches nothing, so it builds none.
    providers = None
    if not dry_run:
        caption_prompt = _caption_prompt_or_die(cfg)
        providers = _build_or_die(
            lambda: build_enrichment_providers(cfg, caption_prompt=caption_prompt)
        )

    conn = _connect_and_verify_or_die(cfg)

    if dry_run:
        # No lease claim, no dispatch — see `preview_claimable_tasks`'s
        # docstring for why full per-task dry-run isn't meaningful here.
        try:
            preview = preview_claimable_tasks(conn)
        except ImsgError as exc:
            typer.echo(f"imsg: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        finally:
            conn.close()

        by_kind = " ".join(f"{kind}={count}" for kind, count in sorted(preview.by_kind.items())) or "none"
        typer.echo(f"enrich: claimable={preview.total} {by_kind}")
        typer.echo(DRY_RUN_MARKER)
        return

    assert providers is not None  # built above for every non-dry run
    try:
        if retry_failed:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    "UPDATE enrichment SET state = 'pending', attempts = 0, "
                    "next_attempt_at = now(), last_error = NULL WHERE state = 'failed'"
                )

        tasks = claim_tasks(conn, worker_id=worker_id, limit=limit)
        outcomes: dict[str, int] = {}
        for task in tasks:
            outcome = process_one_task(conn, cfg, providers, task)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    summary = " ".join(f"{k}={v}" for k, v in sorted(outcomes.items())) or "none"
    typer.echo(f"enrich: claimed={len(tasks)} {summary}")


@app.command("backfill-attachments")
def backfill_attachments(
    config: ConfigOption = None,
    rate: Annotated[float, typer.Option(help="Files per minute (SPEC §8 S5a).")] = DEFAULT_RATE_PER_MINUTE,
    yes_full_run: Annotated[
        bool,
        typer.Option(
            "--yes-full-run", help="Skip the first-run 12-file trial gate (SPEC §8 S5a)."
        ),
    ] = False,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed",
            help="Put failed rows (state error or missing, with a source path) back on the "
            "retry ladder first — attempts 0, eligible now — so they are re-examined this run.",
        ),
    ] = False,
    dry_run: DryRunOption = False,
) -> None:
    """S5a — materialize iCloud-optimized attachments locally (SPEC §8 S5a)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)

    conn = _connect_and_verify_or_die(cfg)
    attachments_root = cfg.paths.live_chat_db.parent / "Attachments"
    try:
        report = run_backfill(
            conn,
            cfg.paths.data_root,
            attachments_root,
            rate_per_minute=rate,
            yes_full_run=yes_full_run,
            dry_run=dry_run,
            retry_failed=retry_failed,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    # Two lines, two kinds of write: what this run attempted, and what it
    # reclassified without reading (see `imsg.backfill.pipeline`'s
    # docstring). Every count here is a row the database now holds in
    # that state.
    typer.echo(
        f"backfill-attachments: considered={report.considered} "
        f"materialized={report.materialized} unsupported={report.marked_unsupported} "
        f"errored={report.errored} marked_missing={report.marked_missing}"
    )
    typer.echo(
        f"backfill-attachments: reclassified without reading — "
        f"unsupported={report.reclassified_unsupported} "
        f"missing_no_source_path={report.reclassified_missing_no_source}; "
        f"retry_failed_reset={report.retry_reset}"
    )
    if report.trial_gate_capped:
        typer.echo(
            "backfill-attachments: first-run trial gate active — pass --yes-full-run "
            "to process the rest",
            err=True,
        )
    if report.halted_low_disk_space:
        typer.echo("backfill-attachments: halted — low disk space", err=True)
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


@mcp_app.command("local")
def mcp_local(config: ConfigOption = None) -> None:
    """`imsg mcp local` — stdio MCP server, tailnet/SSH only, full corpus scope (SPEC §10.3)."""
    cfg = _load_config_or_die(config)
    if not cfg.mcp.local.enabled:
        typer.echo("imsg: mcp.local.enabled is false in config", err=True)
        raise typer.Exit(code=1)
    run_guard_mount_or_exit(cfg.paths.data_root)
    # stderr: this server's stdout IS the stdio JSON-RPC channel.
    _echo_backend_line(cfg, err=True)
    text_provider = _build_or_die(lambda: build_text_provider(cfg))
    reranker = _build_or_die(lambda: build_reranker(cfg))
    multimodal_provider = _build_or_die(lambda: build_multimodal_provider(cfg))

    conn = _connect_and_verify_or_die(cfg)
    fts_conn = _open_fts_conn(cfg)
    try:
        assert_schema_current(fts_conn)
    except ImsgError as exc:
        fts_conn.close()
        conn.close()
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    service = RetrievalService(
        pg_conn=conn,
        fts_conn=fts_conn,
        config=cfg,
        text_provider=text_provider,
        reranker=reranker,
        multimodal_provider=multimodal_provider,
    )
    audit = PostgresAuditSink(lambda: connect(cfg.database, autocommit=True))
    local = LocalMcpServer(service=service, audit=audit, config=cfg, conn=conn)
    try:
        anyio.run(run_local_server, local)
    finally:
        fts_conn.close()
        conn.close()


@mcp_app.command("public")
def mcp_public(config: ConfigOption = None) -> None:
    """`imsg mcp public` — StreamableHTTP MCP server behind cloudflared,
    OAuth subject validation, fail closed (SPEC §10.4). Binds loopback
    only (`mcp.public.bind`); cloudflared is the only process that ever
    faces a public interface. Every request — including `initialize`
    and `tools/list`, not only tool calls — passes through
    `imsg.mcp.auth.PublicAuthGate` before anything else runs (hard
    requirement 4: no unauthenticated path, no config flag to disable
    it); scope (`mcp.public.scope`, REQUIRED with no default, SPEC
    §10.3a/D6) is fixed for the life of this process.
    """
    cfg = _load_config_or_die(config)
    if not cfg.mcp.public.enabled:
        typer.echo("imsg: mcp.public.enabled is false in config", err=True)
        raise typer.Exit(code=1)
    run_guard_mount_or_exit(cfg.paths.data_root)
    _echo_backend_line(cfg, err=True)
    text_provider = _build_or_die(lambda: build_text_provider(cfg))
    reranker = _build_or_die(lambda: build_reranker(cfg))
    multimodal_provider = _build_or_die(lambda: build_multimodal_provider(cfg))

    conn = _connect_and_verify_or_die(cfg)
    fts_conn = _open_fts_conn(cfg)
    try:
        assert_schema_current(fts_conn)
    except ImsgError as exc:
        fts_conn.close()
        conn.close()
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        host, port = parse_bind_address(cfg.mcp.public.bind)
    except ImsgError as exc:
        fts_conn.close()
        conn.close()
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    service = RetrievalService(
        pg_conn=conn,
        fts_conn=fts_conn,
        config=cfg,
        text_provider=text_provider,
        reranker=reranker,
        multimodal_provider=multimodal_provider,
    )
    audit = PostgresAuditSink(lambda: connect(cfg.database, autocommit=True))
    try:
        # `build_public_gate` refuses to construct (raises, never
        # allow-all) if `owner_subject`/`client_id` are missing or
        # unresolvable — hard requirement 4's fail-closed startup.
        gate = build_public_gate(cfg.mcp.public, audit=audit)
    except ImsgError as exc:
        fts_conn.close()
        conn.close()
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    public = PublicMcpServer(service=service, gate=gate, scope=cfg.mcp.public.scope)
    asgi_app = build_public_asgi_app(
        public,
        allowed_hosts=cfg.mcp.public.allowed_hosts,
        allowed_origins=cfg.mcp.public.allowed_origins,
        external_url=cfg.mcp.public.external_url,
    )

    typer.echo(
        f"mcp public: listening on {host}:{port}, scope={cfg.mcp.public.scope}, "
        f"external_url={cfg.mcp.public.external_url}"
    )
    try:
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    finally:
        fts_conn.close()
        conn.close()


# --------------------------------------------------------------------------
# Pipeline-stage stubs still pending (SPEC §8/§5.5) — not this build's scope
# --------------------------------------------------------------------------


def _stub(stage: str) -> None:
    try:
        raise StageNotImplementedError(stage)
    except StageNotImplementedError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def export() -> None:
    """S8 — allowlisted export to GCS / Discovery Engine. Not implemented yet (parallel agent's scope this wave)."""
    _stub("export")


# --------------------------------------------------------------------------
# install-agents (SPEC §5.5) — real, wired up
# --------------------------------------------------------------------------


def _resolve_binary_or_die(name: str) -> Path:
    """`shutil.which(name)`, or a clean `AgentInstallError` — never a
    silently-guessed hardcoded fallback path (SPEC §5.5: `install-
    agents` must not invent a path for a binary it cannot find)."""
    found = shutil.which(name)
    if found is None:
        raise AgentInstallError(
            f"'{name}' binary not found on $PATH — install it (or make it "
            f"reachable on PATH) before running 'imsg install-agents'"
        )
    return Path(found)


def _resolve_imsg_binary() -> Path:
    """The installed `imsg` console-script, if this environment has
    one on `PATH`; otherwise the sibling `bin/imsg` next to the
    running interpreter (`sys.executable`) — the shape a `uv`/venv
    install normally takes. Never guesses a hardcoded absolute path."""
    found = shutil.which("imsg")
    if found is not None:
        return Path(found)
    return Path(sys.executable).resolve().parent / "imsg"


@app.command("install-agents")
def install_agents(
    config: ConfigOption = None,
    dest: Annotated[
        Path,
        typer.Option(
            help="Directory to write rendered plists into. Defaults to the real "
            "~/Library/LaunchAgents; tests point this at a tmp_path instead."
        ),
    ] = Path("~/Library/LaunchAgents"),
) -> None:
    """Render and install the thin, content-free LaunchAgent plists
    (SPEC §5.5) into `~/Library/LaunchAgents` — the only place launchd
    discovers user agents. The rendered plists reference `--config
    <path>` and fixed bootstrap paths only; every real value (hostname,
    secrets) lives in that config file, never in the plist itself, and
    the rendered plists are written only here, on a real machine at
    install time — never committed to this repo.
    """
    cfg = _load_config_or_die(config)
    resolved_config_path = (config if config is not None else default_config_path()).resolve()

    try:
        imsg_binary = _resolve_imsg_binary()
        postgres_binary = _resolve_binary_or_die("postgres")
        cloudflared_binary = _resolve_binary_or_die("cloudflared")
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    plists = render_agent_plists(
        cfg,
        imsg_binary=imsg_binary,
        postgres_binary=postgres_binary,
        cloudflared_binary=cloudflared_binary,
        config_path=resolved_config_path,
    )

    dest_dir = dest.expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for label, content in sorted(plists.items()):
        out_path = dest_dir / f"{label}.plist"
        out_path.write_bytes(content)
        typer.echo(f"install-agents: wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
