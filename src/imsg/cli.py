"""The `imsg` CLI (SPEC §8, §14).

Wired up for real: `migrate`, `check-permissions`, `status`,
`guard-mount`, `snapshot`, `extract`, `identity`, `segment`, `embed`,
`sync`, `enrich`, `backfill-attachments`, `locate-attachments`,
`push-attachments` (with the index-host halves `push-attachments-plan`
and `push-attachments-record`), `backup`, `mcp local`, `mcp
public` (serve, and `--probe` for AT-1), `models verify`,
`install-agents`, and `export` (`plan`, `approve`, `push`,
`purge-person`, `unclassified-report` — SPEC §11, `src/imsg/export/`).
No CLI stub remains.

**Every command an installed LaunchAgent invokes now exists.** All
seven `com.imsgindex.*` plists render commands this CLI provides —
`install-agents` no longer schedules a job that fails nightly. `backup`
(SPEC §5.3/§14, `src/imsg/backup/`) is the daily 04:00 one; read that
package's `pipeline` docstring for what it deliberately does *not* copy
and why.

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
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import anyio
import apsw
import psycopg
import typer
import uvicorn

from imsg import host_memory
from imsg.agents.plists import render_agent_plists
from imsg.backfill.fetch import LocationFetchReport, access_from_config, settings_from_config
from imsg.backfill.locate import CoverageReport, LocatedFile, build_coverage, run_locate
from imsg.backfill.pipeline import DEFAULT_RATE_PER_MINUTE, run_backfill
from imsg.backfill.push import (
    PushError,
    build_push_plan,
    parse_plan,
    parse_results,
    parse_root,
    plan_to_lines,
    record_push_results,
    results_to_lines,
    run_push,
)
from imsg.backfill.transfer import TransferError, run_copy
from imsg.background_gate import BackgroundGate, BackgroundWorkDeferred, StopReason
from imsg.background_pause import (
    BackgroundPauseError,
    clear_pause,
    parse_until,
    read_pause_state,
    write_pause,
)
from imsg.backup.pipeline import OUT_OF_SCOPE_NOTE, SAME_DEVICE_CAVEAT, run_backup
from imsg.backup.retention import DEFAULT_KEEP
from imsg.config.loader import default_config_path, load_config
from imsg.db.connection import connect
from imsg.db.enrichment_yield_locks import (
    EnrichmentYieldGate,
    QueryInFlightMarker,
)
from imsg.db.fingerprint import ensure_cluster_fingerprint, verify_data_directory
from imsg.db.migrations import PostgresMigrationRunner, format_mismatches
from imsg.diagnostics import (
    BufferPoolCheck,
    check_at_rest_posture,
    check_buffer_pool,
    check_enrichment_yield,
    check_full_disk_access,
    check_mount,
    check_postgres,
    check_unclassified_threads,
    disk_free_bytes,
)
from imsg.embed.fts.schema import assert_schema_current, create_schema
from imsg.embed.fts.sync import sync_fts
from imsg.embed.pipeline import EmbedRunReport, run_embed
from imsg.enrich.pipeline import process_one_task
from imsg.enrich.planner import EnrichmentPlanReport, plan_enrichment
from imsg.enrich.queue import (
    claim_tasks,
    missing_enrichment_kinds,
    preview_claimable_tasks,
    reset_failed_tasks,
)
from imsg.enrich.router import DEFAULT_CLAIM_ORDER, ENRICHMENT_KINDS, parse_kinds
from imsg.enrich.worker import run_enrich_worker
from imsg.errors import AgentInstallError, ImsgError
from imsg.eval.cli import eval_app
from imsg.export import (
    ExportPlanError,
    ExportPushError,
    approve_run,
    plan_export,
    preview_plan,
    preview_purge,
    purge_person,
    push_export,
    unclassified_summary,
    verify_push_preconditions,
    write_unclassified_report,
)
from imsg.heavy_lock import HeavyModelLock, inspect_heavy_lock
from imsg.host_memory import PressureLevel, format_gib
from imsg.mcp.audit import PostgresAuditSink
from imsg.mcp.auth import build_public_gate
from imsg.mcp.probe import run_auth_probe
from imsg.mcp.probe_cli import (
    EXIT_CONFIG,
    ProbeConfigurationError,
    check_probe_preconditions,
    format_probe_report,
    verdict_exit_code,
)
from imsg.mcp.tools.local_server import LocalMcpServer, run_local_server
from imsg.mcp.tools.public_server import PublicMcpServer, build_public_asgi_app, parse_bind_address
from imsg.mcp.warm_up_readiness import (
    WarmUpReadinessFile,
    read_warm_up_readiness,
    readiness_path,
)
from imsg.memory_admission import (
    MemoryAdmission,
    ModelRole,
    ReservationBook,
    configure_mlx_memory_limit,
    pid_is_running,
)
from imsg.mount.guard import run_guard_mount_or_exit
from imsg.paths import is_contained_in, is_same_file, resolve_path
from imsg.providers.factory import (
    ResolvedPrompt,
    backend_status_line,
    build_boundary_provider,
    build_enrichment_providers,
    build_multimodal_provider,
    build_reranker,
    build_shared_vlm_runtime,
    build_text_provider,
    read_prompt_text,
    resolve_caption_prompt,
    resolve_prompt_path,
)
from imsg.providers.manifest import verify_manifest
from imsg.retrieval.background_warm_up import BackgroundWarmUp
from imsg.retrieval.idle_unload import IdleModelUnloader, PressureRelease, release_freed_memory
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import RetrievalService
from imsg.segment.pipeline import REBUILD_ALL_SENTINEL, run_segment, run_segment_for_chat
from imsg.stages.extract import ExtractResult, MergeMode, merge_mode_for_source, run_extract
from imsg.stages.identity import (
    assign_handle,
    compute_invariant_report,
    merge_persons,
    rename_person,
    run_identity,
)
from imsg.stages.identity_filtered_twins import run_merge_filtered_twins
from imsg.stages.identity_overrides import (
    apply_overrides,
    export_overrides,
    load_overrides,
    write_overrides,
)
from imsg.stages.identity_rematch import run_rematch_stubs
from imsg.stages.imsg_dump import default_binary_path
from imsg.stages.snapshot import SNAPSHOT_FILENAME, SNAPSHOT_SUBDIR, run_snapshot
from imsg.stages.sync import EmbedFn, SegmentFn, run_sync, run_sync_all_sources
from imsg.verify.cli import reconcile_attachments, verify_seed

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config
    from imsg.export.transport import ExportTransport

app = typer.Typer(
    name="imsg",
    help="Local-first iMessage retrieval index: extraction, identity, "
    "segmentation, hybrid search, and a scoped MCP surface.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

mcp_app = typer.Typer(
    name="mcp", help="MCP (Model Context Protocol) servers: local and public search surfaces.", no_args_is_help=True
)
app.add_typer(mcp_app, name="mcp")
models_app = typer.Typer(
    name="models",
    help="Check and manage the pinned model versions in models/manifest.lock.yaml.",
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
        help="Preview what this stage would do without writing anything.",
    ),
]

NoWaitOption = Annotated[
    bool,
    typer.Option(
        "--no-wait",
        help="If another model-heavy command holds the host-wide lock, exit with an error "
        "instead of waiting for it (imsg.heavy_lock).",
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


def _heavy_lock_or_die(cfg: Config, command: str, *, no_wait: bool) -> HeavyModelLock:
    """The host-wide heavy-model lock for `command`, not yet taken
    (`imsg.heavy_lock`). One model-heavy process at a time: two of them
    together exhausted a 64 GiB host's memory (2026-09-24)."""
    try:
        return HeavyModelLock(cfg.paths.data_root, command=command, wait=not no_wait)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _acquire_heavy_lock_or_die(lock: HeavyModelLock) -> None:
    try:
        lock.acquire()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _background_gate(cfg: Config, command: str) -> BackgroundGate:
    """The pause switch, memory admission and between-units checks for a
    heavy background command (`imsg.background_gate`); its progress lines
    go to stderr, prefixed with the command."""
    return BackgroundGate.from_config(
        cfg, log=lambda line: typer.echo(f"{command}: {line}", err=True)
    )


def _exit_deferred(command: str, reason: StopReason) -> NoReturn:
    """`<command>: deferred: <paused|memory> — <why>` and the matching
    exit code (`imsg.background_gate`: 76 paused, 75 memory)."""
    typer.echo(f"{command}: {reason.line()}")
    raise typer.Exit(code=reason.exit_code)


def _exit_if_paused(gate: BackgroundGate, command: str) -> None:
    reason = gate.paused()
    if reason is not None:
        _exit_deferred(command, reason)


def _admit_or_exit(
    gate: BackgroundGate, admission: MemoryAdmission, heavy_lock: HeavyModelLock, command: str
) -> None:
    """Memory admission for a command that holds the heavy lock: waits
    (bounded, logged) for the host to have room; still refused, the lock
    is released and the command exits `deferred: memory`."""
    reason = gate.admit(admission)
    if reason is not None:
        heavy_lock.release()
        _exit_deferred(command, reason)


def _release_models(*providers: object) -> None:
    """Drop the weights of every provider that can, then hand the freed
    memory back to the system — between sync's heavy steps, so the
    boundary model and the embedders are never resident together."""
    dropped = False
    for provider in providers:
        unload = getattr(provider, "unload", None)
        if callable(unload):
            unload()
            dropped = True
    if dropped:
        release_freed_memory()


def _query_marker(cfg: Config) -> QueryInFlightMarker:
    """The "a query is in flight" publisher an MCP server holds while it
    answers (`imsg.db.enrichment_yield_locks`), so the nightly enrichment
    worker pauses between its units of work.

    On its own connection, opened lazily on the first query: a
    session-level advisory lock has to live on a session that lasts as
    long as the marker, and the retrieval connection is busy answering the
    query being marked. `enrichment.yield_to_queries` turns the whole
    mechanism off on both sides at once."""
    return QueryInFlightMarker(
        lambda: connect(cfg.database, autocommit=True),
        enabled=cfg.enrichment.yield_to_queries,
    )


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
    """Refuse to proceed unless data_root is on a mounted, encrypted volume."""
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
    """Apply pending Postgres migrations. Idempotent; roll-forward only."""
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
    """Report Full Disk Access, Contacts, mount, at-rest posture, and Postgres reachability."""
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


def _heavy_lock_status_fields(cfg: Config) -> dict[str, object]:
    """`imsg status`'s view of the host-wide heavy-model lock: held or not
    (asked of the kernel), and the holder's own description of itself."""
    try:
        state = inspect_heavy_lock(cfg.paths.data_root)
    except (ImsgError, OSError) as exc:
        return {"heavy_models_lock_held": None, "heavy_models_lock_holder": f"unreadable: {exc}"}
    holder = state.holder.describe() if state.holder is not None else None
    return {"heavy_models_lock_held": state.held, "heavy_models_lock_holder": holder}


def _host_memory_status_fields() -> dict[str, object]:
    """`imsg status`'s view of the host's memory (`imsg.host_memory`): what
    admission would see now, and the kernel's pressure level."""
    try:
        memory = host_memory.default_probe().read()
    except Exception as exc:
        return {
            "host_memory_available_bytes": None,
            "host_memory_total_bytes": None,
            "host_memory_pressure": None,
            "host_memory_kernel_free_percent": None,
            "host_swap_used_bytes": None,
            "host_memory_summary": f"could not be measured: {exc}",
        }
    return {
        "host_memory_available_bytes": memory.available_bytes,
        "host_memory_total_bytes": memory.total_bytes,
        "host_memory_pressure": memory.pressure.value,
        "host_memory_kernel_free_percent": memory.kernel_free_percent,
        "host_swap_used_bytes": memory.swap_used_bytes,
        "host_memory_summary": memory.describe(),
    }


def _background_pause_status_fields(cfg: Config) -> dict[str, object]:
    """`imsg status`'s view of the pause switch (`imsg.background_pause`)."""
    state = read_pause_state(cfg.paths.data_root, host_pause_file=cfg.background.host_pause_file)
    untils = [r.until for r in state.active if r.until is not None]
    return {
        "background_paused": state.paused,
        "background_pause_reason": state.describe() if state.paused else None,
        "background_paused_until": (
            max(untils).isoformat(timespec="seconds")
            if untils and len(untils) == len(state.active)
            else None
        ),
        "background_pause_lapsed": list(state.lapsed),
        "background_host_pause_file": (
            str(cfg.background.host_pause_file) if cfg.background.host_pause_file else None
        ),
    }


def _model_process_status_fields(cfg: Config) -> dict[str, object]:
    """Every running `imsg` process that loads models, its footprint (read
    without root, `imsg.host_memory`), and what memory admission reserved
    for it (`imsg.memory_admission`)."""
    try:
        reservations = {r.pid: r for r in ReservationBook(cfg.paths.data_root).reservations()}
    except (ImsgError, OSError):
        reservations = {}
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for process in host_memory.list_model_processes():
        reservation = reservations.get(process.pid)
        seen.add(process.pid)
        rows.append(
            {
                "pid": process.pid,
                "command": process.command,
                "footprint_bytes": process.footprint_bytes,
                "reserved_bytes": reservation.reserved_bytes if reservation else None,
            }
        )
    for pid, reservation in sorted(reservations.items()):
        if pid in seen or not pid_is_running(pid):
            continue
        rows.append(
            {
                "pid": pid,
                "command": reservation.command,
                "footprint_bytes": host_memory.process_footprint_bytes(pid),
                "reserved_bytes": reservation.reserved_bytes,
            }
        )
    total = sum(
        footprint for row in rows if isinstance(footprint := row["footprint_bytes"], int)
    )
    return {"model_processes": rows, "model_processes_footprint_bytes": total}


def _describe_model_processes(rows: object) -> list[str]:
    if not isinstance(rows, list) or not rows:
        return ["  none running"]
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        footprint = row.get("footprint_bytes")
        reserved = row.get("reserved_bytes")
        held = format_gib(footprint) if isinstance(footprint, int) else "footprint unreadable"
        extra = f", admitted for {format_gib(reserved)}" if isinstance(reserved, int) else ""
        lines.append(f"  pid {row.get('pid')} {row.get('command')}: {held}{extra}")
    return lines


@app.command()
def status(
    config: ConfigOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Mount, Postgres, disk free, at-rest posture. Pipeline fields report as unavailable until built."""
    cfg = _load_config_or_die(config)

    mount = check_mount(cfg.paths.data_root)
    posture = check_at_rest_posture(cfg.paths.data_root)
    pg = check_postgres(cfg)
    pool = check_buffer_pool(cfg) if pg.reachable else BufferPoolCheck(None, None, None)
    yield_state = check_enrichment_yield(cfg) if pg.reachable else None
    # SPEC §11.5/§14: the count belongs here so a rotting allowlist is
    # visible at a glance rather than only in the weekly report. Gated on
    # reachability and bounded by a statement timeout inside the check —
    # a busy or absent database yields None plus a reason, never a crash.
    unclassified = check_unclassified_threads(cfg) if pg.reachable else None
    free_bytes = disk_free_bytes(cfg.paths.data_root)
    # The public MCP surface's own readiness. Read from a file the server
    # publishes rather than from the server itself: its transport requires
    # a bearer token on every request by design (SPEC §10.4 hard
    # requirement 4), and an unauthenticated readiness endpoint would be
    # the one way past the only access control this project has.
    public_warm_up = read_warm_up_readiness(cfg.paths.data_root)
    heavy_lock_state = _heavy_lock_status_fields(cfg)
    # Memory: what admission sees now, whether heavy background work is
    # paused, and which processes hold models and how much.
    memory_state = _host_memory_status_fields()
    pause_state = _background_pause_status_fields(cfg)
    process_state = _model_process_status_fields(cfg)

    report = {
        "models_backend": cfg.models.backend,
        "mount_ok": mount.ok,
        "mount_reason": mount.reason,
        "postgres_reachable": pg.reachable,
        "postgres_cluster_fingerprint_ok": pg.cluster_fingerprint_ok,
        "postgres_reason": pg.reason,
        "postgres_shared_buffers_bytes": pool.shared_buffers_bytes,
        "hnsw_index_bytes": pool.hnsw_index_bytes,
        "shared_buffers_holds_hnsw_indexes": pool.holds_hnsw_indexes,
        "shared_buffers_warning": pool.warning,
        "at_rest_posture": posture.label,
        "at_rest_posture_caveat": posture.caveat,
        "disk_free_bytes": free_bytes,
        # D10.3: is enrichment standing aside for the query side right now?
        "enrichment_yield_enabled": cfg.enrichment.yield_to_queries,
        "enrichment_yielding_now": yield_state.enrichment_paused if yield_state else None,
        "query_in_flight": yield_state.query_in_flight if yield_state else None,
        "enrichment_yield_reason": yield_state.reason if yield_state else None,
        "mcp_public_warm_up": public_warm_up.state,
        "mcp_public_warm_up_detail": public_warm_up.detail,
        "mcp_public_pid": public_warm_up.pid,
        **memory_state,
        **pause_state,
        **process_state,
        **heavy_lock_state,
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
        "unclassified_thread_count": unclassified.count if unclassified else None,
        "unclassified_thread_reason": unclassified.reason if unclassified else None,
        "pipeline_note": "the remaining None fields above are not yet wired to the "
        "now-built pipeline stages (a later revision of this command's own scope) — "
        "'imsg check-permissions'/the MCP check_permissions tool already reports "
        "last_sync_at/watermarks. unclassified_thread_count IS live (SPEC §11.5); "
        "None there means the count could not be read, and "
        "unclassified_thread_reason says why",
    }

    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return

    _echo_backend_line(cfg)
    for key, value in report.items():
        if key == "models_backend":
            continue  # already printed in its canonical `models: backend=...` form
        if key == "model_processes":
            typer.echo("model_processes:")
            for line in _describe_model_processes(value):
                typer.echo(line)
            continue
        typer.echo(f"{key}: {value}")


# --------------------------------------------------------------------------
# background: the pause switch for heavy background work
# --------------------------------------------------------------------------

background_app = typer.Typer(
    name="background",
    help="Pause and resume heavy background work — segment, embed, the enrich worker, "
    "backfill-attachments and sync's segmentation and embedding (imsg.background_pause). "
    "MCP servers are never paused; sync's snapshot, extract and identity steps always run.",
    no_args_is_help=True,
)
app.add_typer(background_app, name="background")


def _pause_state_report(cfg: Config) -> dict[str, object]:
    state = read_pause_state(cfg.paths.data_root, host_pause_file=cfg.background.host_pause_file)
    return {
        "paused": state.paused,
        "summary": state.describe(),
        "active": [
            {
                "source": r.source,
                "reason": r.reason,
                "set_at": r.set_at.isoformat(timespec="seconds") if r.set_at else None,
                "until": r.until.isoformat(timespec="seconds") if r.until else None,
                "while_pid": r.pid,
            }
            for r in state.active
        ],
        "lapsed": list(state.lapsed),
        "host_pause_file": (
            str(cfg.background.host_pause_file) if cfg.background.host_pause_file else None
        ),
    }


@background_app.command("pause")
def background_pause(
    config: ConfigOption = None,
    reason: Annotated[
        str | None, typer.Option("--reason", help="Why, shown by `imsg status`.")
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(
            "--until",
            help="Resume by itself at this time: ISO 8601 (2026-09-26T08:00, this host's "
            "local time without an offset) or a duration from now (90m, 6h, 2d).",
        ),
    ] = None,
) -> None:
    """Pause heavy background work until `imsg background resume` (or --until).

    Commands that start while paused exit 76 without loading a model; running
    ones stop after their current unit of work. The MCP servers keep serving,
    and `imsg sync` keeps snapshotting, extracting and resolving identities, so
    no message is lost."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    try:
        until_at = parse_until(until) if until is not None else None
    except BackgroundPauseError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if until_at is not None and until_at <= datetime.now(UTC):
        typer.echo(f"imsg: --until {until!r} is already in the past", err=True)
        raise typer.Exit(code=2)
    try:
        request = write_pause(cfg.paths.data_root, reason=reason, until=until_at)
    except BackgroundPauseError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"background: paused — {request.describe()}")
    typer.echo(
        "background: running heavy work stops after its current unit; MCP servers keep "
        "serving; `imsg sync` still snapshots, extracts and resolves identities"
    )


@background_app.command("resume")
def background_resume(config: ConfigOption = None) -> None:
    """Clear the `imsg background pause` switch. A host pause file set by
    another project keeps background work paused until it is removed."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    try:
        cleared = clear_pause(cfg.paths.data_root)
    except BackgroundPauseError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        "background: resumed — the `imsg background pause` switch is cleared"
        if cleared
        else "background: no `imsg background pause` switch was set"
    )
    state = read_pause_state(cfg.paths.data_root, host_pause_file=cfg.background.host_pause_file)
    if state.paused:
        typer.echo(f"background: still {state.describe()}")


@background_app.command("status")
def background_status(
    config: ConfigOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Whether heavy background work is paused, by what, and until when."""
    cfg = _load_config_or_die(config)
    report = _pause_state_report(cfg)
    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return
    typer.echo(f"background: {report['summary']}")
    lapsed = report["lapsed"]
    if isinstance(lapsed, list):
        for line in lapsed:
            typer.echo(f"background: no longer in force: {line}")
    host_file = report["host_pause_file"]
    typer.echo(
        f"background: host pause file: {host_file}" if host_file else "background: host pause file: off"
    )


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
    modifies the lock without --write (a build must not silently
    advance a model just because its upstream 'latest' changed), and never
    advances a local conversion's upstream pin."""
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
    """S1 — snapshot the live chat.db via the SQLite online-backup API."""
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


def _pipeline_snapshot_merge_mode(cfg: Config, source_name: str) -> MergeMode:
    """The D12 merge rule for `imsg extract` without `--snapshot`.

    S1 writes every configured source's copy to the same
    `snapshots/snapshot.db`, so this command cannot tell whose copy is
    there. It can vouch for it only when every configured source is this
    machine's own live database; otherwise it takes the rule that loses
    nothing. A source name that is not configured is a seed too.
    (`imsg sync` does not need this: it runs S1 itself and knows.)"""
    if source_name not in {source.name for source in cfg.sync.sources}:
        return MergeMode.SEED
    modes = {
        merge_mode_for_source(source.chat_db, cfg.paths.live_chat_db)
        for source in cfg.sync.sources
    }
    return MergeMode.LIVE if modes == {MergeMode.LIVE} else MergeMode.SEED


_MERGE_MODE_MEANING = {
    MergeMode.LIVE: "this machine's own chat.db: may replace a non-empty value",
    MergeMode.SEED: "inserts and fills only; a body changes only for a strictly newer edit",
}


def _extract_report_lines(result: ExtractResult) -> list[str]:
    """What S2 did to every table it wrote (D12 rule 5), one grep-able
    line each. `replaced` must be 0 on every line of a seed run."""
    lines = [f"mode={result.merge_mode.value} ({_MERGE_MODE_MEANING[result.merge_mode]})"]
    for table, counts in result.table_counts().items():
        lines.append(
            f"table={table} inserted={counts.inserted} filled={counts.filled} "
            f"newer_edit={counts.newer_edit} replaced={counts.replaced} "
            f"unchanged={counts.unchanged}"
        )
    lines.append(
        f"messages_marked_for_resegmentation={result.messages_marked_for_resegmentation} "
        f"bodies_kept_as_history={result.bodies_kept_as_history} "
        f"tapback_targets_resolved={result.tapback_targets_resolved}"
    )
    # D13: messages with no chat link. `rescanned` rows came from below the
    # watermark; the next five say which rule placed each message this run
    # inserted or moved out of a holding chat.
    u = result.unlinked
    lines.append(
        f"unlinked: rescanned={u.rescanned} recoverable_join={u.recoverable_join} "
        f"ck_1to1={u.ck_1to1} ck_group_match={u.ck_group_match} "
        f"holding_lost_group={u.holding_lost_group} holding_sender={u.holding_sender} "
        f"moved_from_holding={u.moved_from_holding} evidence_raised={u.evidence_raised} "
        f"in_recently_deleted={u.in_recently_deleted} chats_created={u.chats_created} "
        f"holding_chats_created={u.holding_chats_created} "
        f"skipped_without_date={u.skipped_without_date}"
    )
    return lines


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
    """S2 — extract chats/messages/attachments from the current snapshot.

    `--snapshot` is the seed path: it never touches `snapshots/snapshot.db`,
    which S1 atomically replaces from the live chat.db every
    `sync.interval_seconds` — anything staged there is destroyed on the next
    tick. It requires `--source` because a seed advances that source's ROWID
    watermark, and ROWIDs mean nothing across database files.

    A seed only adds (D12): it inserts rows and fills empty values, and never
    replaces a non-empty one. The output names the rule the run applied and
    reports every table it wrote.
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
    merge_mode = (
        MergeMode.SEED if snapshot is not None else _pipeline_snapshot_merge_mode(cfg, source_name)
    )

    conn = _connect_and_verify_or_die(cfg)
    try:
        result = run_extract(
            conn=conn,
            source_name=source_name,
            snapshot_path=snapshot_path,
            imsg_dump_binary=default_binary_path(_repo_root()),
            dry_run=dry_run,
            merge_mode=merge_mode,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    # The breakdown, not just the total: "messages_upserted=662683" reads
    # the same whether the run corrected the corpus or rewrote it with its
    # own values, and only the second proposes re-segmenting everything.
    typer.echo(
        f"extract: messages_upserted={result.messages_upserted} "
        f"(inserted={result.message_upserts.inserted} "
        f"updated={result.message_upserts.updated} "
        f"unchanged={result.message_upserts.unchanged}) "
        f"watermark {result.watermark_before}->{result.watermark_after} "
        f"bodies_missing={result.bodies_missing}"
    )
    for line in _extract_report_lines(result):
        typer.echo(f"extract: {line}")
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


identity_app = typer.Typer(
    name="identity",
    help="S3 — resolve handles to person_id, then curate them.",
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
    """S3 — resolve handles to person_id via Contacts + manual curation."""
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


def _echo_chats_marked_dirty(count: int) -> None:
    """Every curation command ends with this: the chats whose rendered
    segments named the persons involved are marked so `imsg segment`
    re-renders them and `imsg embed` re-embeds the result (the rendered
    hash changes, which is what S6 treats as pending). Printed on dry
    runs too, as the count the real run would mark."""
    typer.echo(f"identity: chats marked for re-segmentation: {count}")
    if count:
        typer.echo(
            "identity: run `imsg segment` and then `imsg embed` — until then the index "
            "keeps the old names in those chats' segments, embeddings and FTS rows"
        )


@identity_app.command("rematch-stubs")
def identity_rematch_stubs(config: ConfigOption = None, dry_run: DryRunOption = False) -> None:
    """Name the review stubs an import that ran without Contacts access left behind.

    Contacts access is granted per application, so an import run over SSH
    or by an agent creates stubs named after their raw handle that a later
    import never revisits. Run this from a terminal that HAS the grant:
    every stub still named after its own handle is looked up in Contacts
    under the import's own rules (unique match only) and, on a match, named
    exactly as the import would have named it. Reviewed persons, the owner,
    ambiguous handles and handles no card carries are never touched. Fails
    outright, rather than degrading, when Contacts is unavailable.

    Every rename marks the chats whose segments carry the stub's old name
    for re-segmentation — on a Contacts-less index that is most chats —
    and the last line reports how many. Run `imsg segment` and then
    `imsg embed` afterwards; until then the index keeps the raw-handle
    names in those chats' segments, embeddings and FTS rows.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        result = run_rematch_stubs(conn=conn, config=cfg, dry_run=dry_run)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(f"identity: contacts loaded={result.contacts_loaded}")
    if not dry_run:
        for outcome in result.outcomes:
            if outcome.status == "matched":
                typer.echo(
                    f"identity: rematched person {outcome.person_id} "
                    f"{outcome.display_name!r} -> {outcome.new_display_name!r}"
                )
    typer.echo(f"rematch: {result.summary}")
    _echo_chats_marked_dirty(result.chats_marked_dirty)
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


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
    """The curation worklist — persons ranked by message volume.

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
      * one name being an abbreviation/prefix of another (Mike/Michael),
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
    """Fold `absorb` into `keep` — handles, messages, tapbacks, participants.

    The absorbed person's chats (and the kept person's) are marked for
    re-segmentation, since their segments now render the kept person's
    names; run `imsg segment` and then `imsg embed` afterwards.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        dirty = merge_persons(conn, keep_person_id=keep, absorb_person_id=absorb)
        conn.commit()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"identity: merged person {absorb} into {keep}")
    _echo_chats_marked_dirty(len(dirty))


@identity_app.command("rename")
def identity_rename(
    person: Annotated[int, typer.Option(help="person_id to rename.")],
    name: Annotated[str, typer.Option(help="New display_name.")],
    short: Annotated[str | None, typer.Option(help="Optional new short_name.")] = None,
    config: ConfigOption = None,
    yes_owner: Annotated[
        bool,
        typer.Option(
            "--yes-owner",
            help="Required to rename the owner person — the one identity merge, "
            "apply-overrides and rematch-stubs never touch. Without it, renaming the "
            "owner is refused.",
        ),
    ] = False,
) -> None:
    """Set a person's display name (and optionally short name).

    Every chat whose segments carry the person's name is marked for
    re-segmentation; run `imsg segment` and then `imsg embed` afterwards.
    The owner person is refused unless --yes-owner is given.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    if yes_owner:
        typer.echo(
            f"identity: WARNING --yes-owner: person {person} is renamed even if it is the "
            "owner person, which every automated curation path leaves alone",
            err=True,
        )
    conn = _connect_and_verify_or_die(cfg)
    try:
        dirty = rename_person(
            conn, person_id=person, display_name=name, short_name=short, allow_owner=yes_owner
        )
        conn.commit()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"identity: renamed person {person} to {name!r}")
    _echo_chats_marked_dirty(len(dirty))


@identity_app.command("assign")
def identity_assign(
    value: Annotated[str, typer.Option(help="Normalized handle value (E.164 phone or email).")],
    kind: Annotated[str, typer.Option(help="Handle kind: 'phone' or 'email'.")],
    person: Annotated[int, typer.Option(help="person_id to attach it to.")],
    config: ConfigOption = None,
) -> None:
    """Repoint one canonical handle onto a different person.

    The chats of the handle's previous person and of its new one are marked
    for re-segmentation; run `imsg segment` and then `imsg embed` afterwards.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        dirty = assign_handle(conn, normalized_value=value, kind=kind, person_id=person)
        conn.commit()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"identity: assigned {kind} {value!r} to person {person}")
    _echo_chats_marked_dirty(len(dirty))


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

    Every applied decision marks the chats whose segments carry the names
    it replaced for re-segmentation, and the last line reports how many.
    Run `imsg segment` and then `imsg embed` afterwards; until then the
    index keeps the old names in those chats' segments, embeddings and
    FTS rows.
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
    _echo_chats_marked_dirty(result.chats_marked_dirty)
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


@identity_app.command("merge-filtered-twins")
def identity_merge_filtered_twins(
    config: ConfigOption = None,
    dry_run: DryRunOption = False,
    show_names: Annotated[
        bool,
        typer.Option(
            "--show-names",
            help="Also print the names and the handle value of each refused pair. Without it, "
            "refused pairs are listed by person_id only.",
        ),
    ] = False,
) -> None:
    """Merge the persons an early import split on iOS filter tags ("(filtered)", "(smsft…)").

    Source handles resolved before the filter-tag fix of 2026-08-15 still
    point at a canonical handle that carries the tag, owned by a person of
    its own, so one sender appears as two persons. This repoints them to the
    clean handle and merges the two persons (keeping a curated name), and
    gives a tagged handle with no clean twin its clean value. Two persons
    with different curated names, and the owner, are never merged: those
    pairs are listed and left as they are. Messages, chats and raw source
    handles are never deleted. Run with --dry-run first: it reports the same
    counts and writes nothing.

    Every merge and stub rename marks the chats whose segments name the
    persons involved for re-segmentation, and the last line reports how
    many. Run `imsg segment` and then `imsg embed` afterwards; until then
    the index keeps the old names in those chats' segments, embeddings and
    FTS rows.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        result = run_merge_filtered_twins(
            conn=conn, default_region=cfg.identity.default_region, dry_run=dry_run
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"identity: tagged_source_handles={result.tagged_source_handles} "
        f"legacy_source_handles={result.legacy_source_handles}"
    )
    typer.echo(f"identity: {result.summary}")
    for pair in result.refused:
        why = (
            "both carry curated names that differ"
            if pair.reason == "curated_conflict"
            else "one of them is the owner person"
        )
        line = (
            f"identity: refused person {pair.legacy_person_id} (tagged handle) and person "
            f"{pair.target_person_id} (clean handle): {why}"
        )
        if show_names:
            line += (
                f" — {pair.legacy_display_name!r} / {pair.target_display_name!r}, "
                f"{pair.clean_kind} {pair.clean_value!r}"
            )
        typer.echo(line, err=True)
    if result.refused:
        typer.echo(
            "identity: refused pairs were left unchanged; decide them by hand with "
            "`imsg identity merge` or `imsg identity rename`, then run this again",
            err=True,
        )
    typer.echo(
        f"identity: allowlist rows removed with merged-away persons="
        f"{result.allowlist_rows_absorbed} kept persons narrowed={result.allowlist_rows_narrowed}"
    )
    if result.other_stale_source_handles:
        typer.echo(
            f"identity: WARNING {result.other_stale_source_handles} untagged source handle(s) "
            "resolve to a canonical handle that today's normalizer would not produce; this "
            "command does not repair them",
            err=True,
        )
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
    _echo_chats_marked_dirty(result.chats_marked_dirty)
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
    no_wait: NoWaitOption = False,
) -> None:
    """S4 — sessionize and segment messages for indexing.

    Holds the host-wide heavy-model lock (`imsg.heavy_lock`) for the whole
    run, `--dry-run` included: a dry run still asks the boundary model.

    Heavy background work (`imsg.background_gate`): exits 76 without
    loading anything while background work is paused, waits for memory
    and exits 75 if the host never has room, and between chats stops
    (exit 75/76) when paused or when the host's memory pressure rises."""
    cfg = _load_config_or_die(config)
    if rebuild and chat is None:
        typer.echo("imsg: --rebuild requires --chat <id>", err=True)
        raise typer.Exit(code=2)
    run_guard_mount_or_exit(cfg.paths.data_root)
    gate = _background_gate(cfg, "segment")
    _exit_if_paused(gate, "segment")

    prompt_bytes = _boundary_prompt_bytes_or_die(cfg)
    _echo_backend_line(cfg)
    provider = _build_or_die(lambda: build_boundary_provider(cfg, _decode_prompt(prompt_bytes)))
    heavy_lock = _heavy_lock_or_die(cfg, "imsg segment", no_wait=no_wait)
    admission = MemoryAdmission.for_role(
        cfg, ModelRole.SEGMENT, command="imsg segment", probe=gate.probe
    )

    conn = _connect_and_verify_or_die(cfg)
    try:
        _acquire_heavy_lock_or_die(heavy_lock)
        _admit_or_exit(gate, admission, heavy_lock, "segment")
        configure_mlx_memory_limit(cfg, ModelRole.SEGMENT)
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
                f"segments_deleted={report.segments_deleted} "
                f"segments_unchanged={report.skipped_unchanged}"
            )
        else:
            chat_ids = {chat} if chat is not None else None
            stopped: StopReason | None = None
            try:
                reports = run_segment(
                    conn,
                    cfg,
                    provider,
                    prompt_bytes,
                    chat_ids=chat_ids,
                    dry_run=dry_run,
                    stop_check=gate.between_units,
                )
            except BackgroundWorkDeferred as exc:
                stopped = exc.reason
                reports = exc.partial if isinstance(exc.partial, list) else []
            total_written = sum(r.segments_written for r in reports)
            total_fallback = sum(r.fallback_sessions for r in reports)
            total_unchanged = sum(r.skipped_unchanged for r in reports)
            typer.echo(
                f"segment: {len(reports)} chat(s) processed, {total_written} segment(s) "
                f"written, {total_unchanged} left unchanged, "
                f"{total_fallback} fallback session(s)"
            )
            if stopped is not None:
                if dry_run:
                    typer.echo(DRY_RUN_MARKER)
                _exit_deferred("segment", stopped)
        if dry_run:
            typer.echo(DRY_RUN_MARKER)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        admission.release()
        heavy_lock.release()
        conn.close()


@app.command()
def embed(
    config: ConfigOption = None, dry_run: DryRunOption = False, no_wait: NoWaitOption = False
) -> None:
    """S6 — embed segments/attachment chunks and update the FTS sidecar.

    Holds the host-wide heavy-model lock (`imsg.heavy_lock`) for the run.
    `--dry-run` only counts pending rows, loads no model and takes no lock.

    Heavy background work (`imsg.background_gate`): exits 76 while
    background work is paused, exits 75 if the host never has memory for
    the models, and between batches stops (committing what is done) when
    paused or when the host's memory pressure rises."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    gate = _background_gate(cfg, "embed")
    if not dry_run:
        _exit_if_paused(gate, "embed")
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
    heavy_lock = _heavy_lock_or_die(cfg, "imsg embed", no_wait=no_wait)
    admission = MemoryAdmission.for_role(cfg, ModelRole.EMBED, command="imsg embed", probe=gate.probe)
    stopped: StopReason | None = None
    try:
        if not dry_run:
            _acquire_heavy_lock_or_die(heavy_lock)
            _admit_or_exit(gate, admission, heavy_lock, "embed")
            configure_mlx_memory_limit(cfg, ModelRole.EMBED)
        try:
            report = run_embed(
                conn,
                text_provider,
                multimodal_provider=multimodal_provider,
                batch_size=cfg.embedding.batch_size,
                max_batch_tokens=cfg.embedding.max_batch_tokens,
                dry_run=dry_run,
                stop_check=None if dry_run else gate.between_units,
            )
        except BackgroundWorkDeferred as exc:
            if not isinstance(exc.partial, EmbedRunReport):  # pragma: no cover - run_embed always sets it
                raise
            stopped, report = exc.reason, exc.partial
        if not dry_run:
            # The FTS sidecar needs no model, so a run stopped for memory
            # still brings it up to date with what did commit.
            assert fts_conn is not None
            sync_report = sync_fts(conn, fts_conn)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        admission.release()
        heavy_lock.release()
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
    if stopped is not None:
        _exit_deferred("embed", stopped)


class _SyncHeavySteps:
    """`sync`'s S4 and S6 under the pause switch and memory admission
    (`imsg.background_gate`). Before each heavy step: paused, or refused
    memory after the bounded wait, raises `BackgroundWorkDeferred`, which
    `run_sync` records while S1-S3's work stands. A deferral sticks for
    the rest of the command, so later sources skip their heavy steps at
    once instead of waiting again, and it releases the heavy lock at once
    so other heavy commands are not held up by the light work left."""

    def __init__(self, cfg: Config, heavy_lock: HeavyModelLock, gate: BackgroundGate) -> None:
        self._cfg = cfg
        self._heavy_lock = heavy_lock
        self._gate = gate
        self._admission: MemoryAdmission | None = None
        self.deferred: StopReason | None = None

    def begin(self, role: ModelRole) -> None:
        if self.deferred is not None:
            raise BackgroundWorkDeferred(self.deferred)
        paused = self._gate.paused()
        if paused is not None:
            self.defer(paused)
        self._heavy_lock.acquire()
        admission = MemoryAdmission.for_role(
            self._cfg, role, command="imsg sync", probe=self._gate.probe
        )
        self._admission = admission
        refused = self._gate.admit(admission)
        if refused is not None:
            self.defer(refused)
        configure_mlx_memory_limit(self._cfg, role)

    def stop_check(self) -> StopReason | None:
        return self._gate.between_units()

    def defer(self, reason: StopReason) -> NoReturn:
        self.note_stopped(reason)
        raise BackgroundWorkDeferred(reason)

    def note_stopped(self, reason: StopReason) -> None:
        self.deferred = reason
        self.release_admission()
        self._heavy_lock.release()

    def release_admission(self) -> None:
        if self._admission is not None:
            self._admission.release()
            self._admission = None


def _make_segment_fn(cfg: Config, steps: _SyncHeavySteps) -> SegmentFn:
    prompt_bytes = _boundary_prompt_bytes_or_die(cfg)
    provider = _build_or_die(lambda: build_boundary_provider(cfg, _decode_prompt(prompt_bytes)))

    def _segment_fn(conn: psycopg.Connection, config: Config, *, dry_run: bool = False) -> object:
        # A dry run asks the boundary model too, so it takes the lock and
        # passes the gate too.
        steps.begin(ModelRole.SEGMENT)
        try:
            return run_segment(
                conn, config, provider, prompt_bytes, dry_run=dry_run, stop_check=steps.stop_check
            )
        except BackgroundWorkDeferred as exc:
            steps.note_stopped(exc.reason)
            raise
        finally:
            # The boundary model is not needed again until the next
            # source's S4: drop it now, so it and the embedders below are
            # never resident together.
            _release_models(provider)
            steps.release_admission()

    return _segment_fn


def _make_embed_fn(cfg: Config, steps: _SyncHeavySteps) -> EmbedFn:
    # Built once per `sync` process, up front (a missing model runtime
    # fails fast); the weights load lazily, on the first embedding.
    text_provider = _build_or_die(lambda: build_text_provider(cfg))
    multimodal_provider = _build_or_die(lambda: build_multimodal_provider(cfg))

    def _embed_fn(conn: psycopg.Connection, config: Config, *, dry_run: bool = False) -> object:
        if dry_run:
            # A dry run only counts pending rows: no model, no lock, no
            # gate (see `embed`), and no FTS sidecar — opening it creates
            # it on disk, a real write a dry run must not do.
            return run_embed(
                conn,
                text_provider,
                multimodal_provider=multimodal_provider,
                batch_size=config.embedding.batch_size,
                max_batch_tokens=config.embedding.max_batch_tokens,
                dry_run=True,
            )
        steps.begin(ModelRole.EMBED)
        stopped: BackgroundWorkDeferred | None = None
        try:
            try:
                report: object = run_embed(
                    conn,
                    text_provider,
                    multimodal_provider=multimodal_provider,
                    batch_size=config.embedding.batch_size,
                    max_batch_tokens=config.embedding.max_batch_tokens,
                    stop_check=steps.stop_check,
                )
            except BackgroundWorkDeferred as exc:
                stopped, report = exc, exc.partial
            fts_conn = _open_fts_conn(config)
            try:
                sync_fts(conn, fts_conn)
            finally:
                fts_conn.close()
        finally:
            _release_models(text_provider, multimodal_provider)
            steps.release_admission()
        if stopped is not None:
            steps.note_stopped(stopped.reason)
            raise stopped
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
            help="One-shot seed: skip S1 and feed this prepared database straight to S2. "
            "Requires --source, which must not name a configured "
            "sync.sources entry.",
        ),
    ] = None,
    dry_run: DryRunOption = False,
    no_wait: NoWaitOption = False,
) -> None:
    """S7 — incremental S1→S2→S3→S4→S6 sync for every configured source.

    With `--source`, syncs that one source. With `--source` and `--snapshot`,
    runs the one-shot seed path: S1 is skipped
    entirely and the already-prepared file feeds S2→S3→S4→S6, so the seed
    lands under its own source name and its own ROWID watermark.

    The host-wide heavy-model lock (`imsg.heavy_lock`) is taken when the
    first source reaches S4, not at start: snapshot and extraction never
    wait behind another process's models. Once taken it is held until the
    command ends.

    S4 and S6 are heavy background work (`imsg.background_gate`). While
    background work is paused, or when the host has no memory for their
    models, sync does its light work only — snapshot, extract, identity,
    so no message is lost — skips segmentation and embedding, says so,
    and exits 76 (paused) or 75 (memory). Between S4 and S6 the boundary
    model is dropped, so the two model sets are never resident together.
    """
    cfg = _load_config_or_die(config)
    _validate_seed_or_die(cfg, snapshot, source)
    _echo_backend_line(cfg)
    heavy_lock = _heavy_lock_or_die(cfg, "imsg sync", no_wait=no_wait)
    steps = _SyncHeavySteps(cfg, heavy_lock, _background_gate(cfg, "sync"))
    segment_fn = _make_segment_fn(cfg, steps)  # validates the boundary prompt exists up front
    embed_fn = _make_embed_fn(cfg, steps)

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
        steps.release_admission()
        heavy_lock.release()
        conn.close()

    deferred: StopReason | None = None
    for r in results:
        if r.extract is None:
            typer.echo(f"sync: source={r.source_name} {r.note}")
            continue
        typer.echo(
            f"sync: source={r.source_name} messages_upserted={r.extract.messages_upserted} "
            f"(inserted={r.extract.message_upserts.inserted} "
            f"updated={r.extract.message_upserts.updated} "
            f"unchanged={r.extract.message_upserts.unchanged}) "
            f"segment_ran={r.segment_ran} embed_ran={r.embed_ran}"
        )
        for line in _extract_report_lines(r.extract):
            typer.echo(f"sync: source={r.source_name} {line}")
        if r.deferred is not None:
            skipped = "segmentation and embedding" if not r.segment_ran else "embedding"
            typer.echo(
                f"sync: source={r.source_name} light work done (snapshot, extract, identity); "
                f"{skipped} skipped — {r.deferred.line()}"
            )
            deferred = deferred or r.deferred
    if dry_run:
        typer.echo(DRY_RUN_MARKER)
    if deferred is not None:
        raise typer.Exit(code=deferred.exit_code)


def _kind_counts(counts: dict[str, int]) -> str:
    """`total k1=n k2=n`, kinds in claim order, other keys (MIME types)
    after them in descending count."""
    ordered = [k for k in ENRICHMENT_KINDS if k in counts]
    rest = sorted((k for k in counts if k not in ENRICHMENT_KINDS), key=lambda k: (-counts[k], k))
    parts = " ".join(f"{k}={counts[k]}" for k in [*ordered, *rest])
    return f"{sum(counts.values())} {parts}".rstrip()


def _echo_plan_report(report: EnrichmentPlanReport) -> None:
    typer.echo(
        f"enrich plan: attachments={report.attachments} refused={report.refused} "
        f"sniff_failed={report.sniff_failed}"
    )
    verb = "would_enqueue" if report.dry_run else "enqueued"
    typer.echo(f"enrich plan: {verb}={_kind_counts(report.enqueued)}")
    typer.echo(f"enrich plan: already_queued={_kind_counts(report.already_queued)}")
    if report.not_selected:
        typer.echo(f"enrich plan: not_selected={_kind_counts(report.not_selected)}")
    typer.echo(f"enrich plan: unroutable={_kind_counts(report.unroutable)}")


def _enrich_plan(cfg: Config, conn: psycopg.Connection, kinds: tuple[str, ...] | None, dry_run: bool) -> None:
    try:
        report = plan_enrichment(conn, data_root=cfg.paths.data_root, kinds=kinds, dry_run=dry_run)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    _echo_plan_report(report)
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


@app.command()
def enrich(
    config: ConfigOption = None,
    limit: Annotated[int, typer.Option(help="Max tasks to claim and process this run.")] = 100,
    worker_id: Annotated[
        str, typer.Option(help="Lease owner id, used for the claim/backoff bookkeeping.")
    ] = "cli",
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed",
            help="Reset permanently-failed tasks (of the selected kinds) to pending first.",
        ),
    ] = False,
    kinds: Annotated[
        str | None,
        typer.Option(
            "--kinds",
            help="Comma-separated enrichment kinds to work on, in the order to claim them "
            f"(default: {','.join(DEFAULT_CLAIM_ORDER)} — cheap kinds first, captions last). "
            "With --plan, the kinds to enqueue.",
        ),
    ] = None,
    plan: Annotated[
        bool,
        typer.Option(
            "--plan",
            help="Fill the queue instead of draining it: enqueue the router's kinds for every "
            "materialized attachment that lacks them (by content-sniffed MIME type), then exit.",
        ),
    ] = False,
    dry_run: DryRunOption = False,
    no_wait: NoWaitOption = False,
) -> None:
    """S5b — OCR/caption/transcribe/text-extraction enrichment queue worker.

    Claims one task at a time: `--kinds` in the order given, else cheap kinds
    first and captions last; within a kind, the newest attachment first.
    `--plan` fills the queue instead (see `imsg.enrich.planner`).

    A worker run holds the host-wide heavy-model lock (`imsg.heavy_lock`),
    taken before the first claim so no task sits leased while it waits.
    `--plan` and `--dry-run` load no model and take no lock.

    A worker run is heavy background work (`imsg.background_gate`): it
    exits 76 without claiming anything while background work is paused,
    exits 75 if the host never has memory for the models, and between
    tasks — holding no lease — stops when paused or when the host's
    memory pressure rises (`imsg.enrich.worker`).
    """
    cfg = _load_config_or_die(config)
    if plan and retry_failed:
        typer.echo(
            "imsg: --retry-failed resets tasks for a worker run; --plan only adds missing "
            "tasks and resets nothing — run them separately",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        selected = parse_kinds(kinds) if kinds is not None else None
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    claim_order = selected if selected is not None else DEFAULT_CLAIM_ORDER
    run_guard_mount_or_exit(cfg.paths.data_root)
    gate = _background_gate(cfg, "enrich")
    if not dry_run and not plan:
        _exit_if_paused(gate, "enrich")
    _echo_backend_line(cfg)
    # Providers before the DB connection so a missing model runtime fails
    # fast; a dry run claims and dispatches nothing, and a plan run fills
    # the queue without running a model, so neither builds any.
    providers = None
    if not dry_run and not plan:
        caption_prompt = _caption_prompt_or_die(cfg)
        # One runtime for every vision-language role this process builds,
        # so a composition that captions *and* detects boundaries against
        # the same pin loads one copy of the weights rather than two
        # (D10.3 defect 1; `imsg.shared_vlm_runtime`). This process ships
        # with only the captioner, so today it holds one either way — but
        # the runtime is what makes that a property of the code rather
        # than of which providers happen to be wired here.
        shared_vlm = build_shared_vlm_runtime(cfg)
        providers = _build_or_die(
            lambda: build_enrichment_providers(
                cfg, caption_prompt=caption_prompt, shared_runtime=shared_vlm
            )
        )

    conn = _connect_and_verify_or_die(cfg)
    try:
        missing = missing_enrichment_kinds(conn, claim_order)
    except ImsgError as exc:
        conn.close()
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if missing:
        conn.close()
        typer.echo(
            f"imsg: this database has no enrichment kind {', '.join(missing)} yet — "
            f"run `imsg migrate` (migration 0009) first",
            err=True,
        )
        raise typer.Exit(code=1)

    if plan:
        _enrich_plan(cfg, conn, selected, dry_run)
        return

    if dry_run:
        # No lease claim, no dispatch — see `preview_claimable_tasks`'s
        # docstring for why full per-task dry-run isn't meaningful here.
        try:
            preview = preview_claimable_tasks(conn, kinds=claim_order)
        except ImsgError as exc:
            typer.echo(f"imsg: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        finally:
            conn.close()

        by_kind = (
            " ".join(f"{k}={preview.by_kind[k]}" for k in claim_order if k in preview.by_kind)
            or "none"
        )
        typer.echo(f"enrich: claimable={preview.total} {by_kind}")
        typer.echo(DRY_RUN_MARKER)
        return

    assert providers is not None  # built above for every non-dry, non-plan run
    if selected is not None:
        typer.echo(f"enrich: kinds={','.join(selected)}")
    # Enrichment yields to the MCP server, never the reverse (D10.3).
    # Checked BEFORE each claim, never during a task: a claimed task
    # holds a lease and a half-run model call cannot be represented in
    # the queue, so pausing between units of work is the only point at
    # which nothing can be abandoned or corrupted. One task is claimed at
    # a time, so no task sits leased while the worker waits here, a long
    # run never outlives the leases of tasks it has not started yet, and
    # a cheap task enqueued mid-run is claimed before the next caption.
    # The same point is where the worker stops for a pause or for memory
    # pressure (`imsg.enrich.worker`).
    yield_gate = EnrichmentYieldGate(
        conn,
        enabled=cfg.enrichment.yield_to_queries,
        poll_interval_seconds=cfg.enrichment.yield_poll_interval_seconds,
        max_pause_seconds=cfg.enrichment.yield_max_pause_seconds,
    )
    heavy_lock = _heavy_lock_or_die(cfg, "imsg enrich", no_wait=no_wait)
    admission = MemoryAdmission.for_role(
        cfg, ModelRole.ENRICH, command="imsg enrich", probe=gate.probe
    )
    try:
        # Before any claim or reset: waiting here holds no lease.
        _acquire_heavy_lock_or_die(heavy_lock)
        _admit_or_exit(gate, admission, heavy_lock, "enrich")
        configure_mlx_memory_limit(cfg, ModelRole.ENRICH)
        if retry_failed:
            with conn.transaction():
                reset = reset_failed_tasks(conn, kinds=claim_order)
            typer.echo(f"enrich: reset {reset} failed task(s) to pending")

        worker = run_enrich_worker(
            conn,
            cfg,
            providers,
            worker_id=worker_id,
            claim_order=claim_order,
            limit=limit,
            yield_gate=yield_gate,
            stop_check=gate.between_units,
            claim=claim_tasks,
            process=process_one_task,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        admission.release()
        heavy_lock.release()
        conn.close()

    summary = " ".join(f"{k}={v}" for k, v in sorted(worker.outcomes.items())) or "none"
    typer.echo(f"enrich: claimed={worker.processed} {summary}")
    if worker.yield_pauses:
        typer.echo(
            f"enrich: yielded to in-flight queries {worker.yield_pauses} time(s), "
            f"{worker.yielded_seconds:.1f}s total"
        )
    if worker.stopped is not None:
        _exit_deferred("enrich", worker.stopped)


@app.command("backfill-attachments")
def backfill_attachments(
    config: ConfigOption = None,
    rate: Annotated[float, typer.Option(help="Files per minute.")] = DEFAULT_RATE_PER_MINUTE,
    yes_full_run: Annotated[
        bool,
        typer.Option(
            "--yes-full-run", help="Skip the first-run 12-file trial gate."
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
    no_pull: Annotated[
        bool,
        typer.Option(
            "--no-pull",
            help="Skip the attachments.pull locations this run (no SSH copies); every other "
            "location is still tried.",
        ),
    ] = False,
    dry_run: DryRunOption = False,
) -> None:
    """S5a — materialize attachments into the cache.

    First each attachment's own path on this host, as before. Then every
    other known copy of every attachment still not materialized — missing
    and unsupported ones included — best first: content already in the
    cache, this host's Messages folder, copies another host pushed into
    staging, then attachments.pull locations over SSH. Each copy is checked
    against the size and hash its location reported before it enters the
    cache (D13). Sources are only ever read. Counts only are printed.

    Heavy background work (`imsg.background_gate`), though it loads no
    model: a real run exits 76 while background work is paused, and
    between files stops (exit 75/76) when paused or when the host's
    memory pressure rises."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    gate = _background_gate(cfg, "backfill-attachments")
    if not dry_run:
        _exit_if_paused(gate, "backfill-attachments")

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
            locations=settings_from_config(cfg, run_pulls=not no_pull),
            stop_check=None if dry_run else gate.between_units,
        )
    except (ImsgError, TransferError) as exc:
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
    if report.locations is not None:
        for line in _location_fetch_lines(report.locations):
            typer.echo(f"backfill-attachments: {line}")
    if report.enrichment_enqueued or report.enrichment_unroutable or report.enrichment_plan_errors:
        typer.echo(
            f"backfill-attachments: enrichment tasks enqueued={report.enrichment_enqueued} "
            f"unroutable={report.enrichment_unroutable} "
            f"plan_errors={report.enrichment_plan_errors}"
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
    if report.stopped is not None:
        _exit_deferred("backfill-attachments", report.stopped)


def _counter_text(counts: Mapping[Any, int]) -> str:
    if not counts:
        return "none"
    parts = []
    for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        label = "/".join(key) if isinstance(key, tuple) else key
        parts.append(f"{label}={n}")
    return " ".join(parts)


def _location_fetch_lines(report: LocationFetchReport) -> list[str]:
    """The location phase, counts only (no path is ever printed: the
    terminal output of a run on another host lands on that host's disk)."""
    if report.dry_run:
        return [
            f"locations (dry run): attachments with an open copy="
            f"{report.attachments_with_candidates}; first copy each would try: "
            f"{_counter_text(report.would_try)}",
            f"locations (dry run): waiting on a push: {_counter_text(report.awaiting_push)}; "
            f"no host reads: {_counter_text(report.no_access)}",
        ]
    return [
        f"locations: attachments with an open copy={report.attachments_with_candidates} "
        f"attempted={report.attempted} materialized={report.materialized_total} "
        f"(flagged name+size={report.materialized_flagged})",
        f"locations: materialized by tier/location/match: {_counter_text(report.materialized)}",
        f"locations: rejected (size or hash differed): {_counter_text(report.rejected)}; "
        f"absent: {_counter_text(report.absent)}; refused: {_counter_text(report.refused)}; "
        f"unreachable: {_counter_text(report.unreachable)}",
        f"locations: pulled: {_counter_text(report.pulled)}; "
        f"pull runs that did not reach their host: {_counter_text(report.pull_runs_failed)}",
        f"locations: still waiting on a push: {_counter_text(report.awaiting_push)}; "
        f"only on locations no host reads: {_counter_text(report.no_access)}; "
        f"pulls not run (--no-pull): {_counter_text(report.pull_skipped)}; "
        f"name+size copies held back for a pending push={report.deferred_flagged}",
    ]


def _coverage_lines(coverage: CoverageReport) -> list[str]:
    not_materialized = sum(coverage.not_materialized.values())
    return [
        f"coverage: not materialized={not_materialized} "
        f"({_counter_text(coverage.not_materialized)})",
        f"coverage: next copy would come from: {_counter_text(coverage.best)}; "
        f"of which a name+size match only (flagged)={coverage.best_is_flagged}",
        f"coverage: unfetchable={coverage.unfetchable}; every copy tried and closed, by "
        f"state/kind: {_counter_text(coverage.exhausted)}; no copy found anywhere, by "
        f"state/kind: {_counter_text(coverage.no_candidate)}",
        f"coverage: candidate rows by location/match: {_counter_text(coverage.rows)}",
        f"coverage: materialized from a location, by location/match: "
        f"{_counter_text(coverage.fetched)}",
    ]


def _located_files(specs: list[str] | None, flag: str) -> list[LocatedFile]:
    located: list[LocatedFile] = []
    for spec in specs or []:
        code, sep, raw = spec.partition("=")
        if not sep or not code or not raw:
            typer.echo(f"imsg: {flag} must be CODE=PATH, got {spec!r}", err=True)
            raise typer.Exit(code=2)
        path = Path(raw).expanduser()
        if not path.is_file():
            typer.echo(f"imsg: {flag} {code}: no such file", err=True)
            raise typer.Exit(code=1)
        located.append(LocatedFile(code, path))
    return located


@app.command("locate-attachments")
def locate_attachments(
    config: ConfigOption = None,
    listing: Annotated[
        list[str] | None,
        typer.Option(
            "--listing",
            help="CODE=PATH: a tab-separated listing (rel, size, mtime, sha256) of another "
            "Mac's Messages attachments folder, CODE being that Mac. Repeatable.",
        ),
    ] = None,
    catalog: Annotated[
        list[str] | None,
        typer.Option(
            "--catalog",
            help="[CODE=]PATH: a drive catalog (path, size, mtime, ext; gzip or plain), or a "
            "directory searched for them. The drive code comes from the file or run-directory "
            "name unless given. Repeatable.",
        ),
    ] = None,
    seed_db: Annotated[
        list[str] | None,
        typer.Option(
            "--seed-db",
            help="CODE=PATH: a chat.db-shaped database (opened read-only) whose recorded "
            "attachment paths are stored under CODE, the Mac it came from. Repeatable.",
        ),
    ] = None,
    no_local_walk: Annotated[
        bool, typer.Option("--no-local-walk", help="Do not walk this host's Messages folder.")
    ] = False,
    report_only: Annotated[
        bool, typer.Option("--report-only", help="Print coverage from the table; read nothing.")
    ] = False,
    dry_run: DryRunOption = False,
) -> None:
    """Find every candidate copy of every attachment not yet materialized (D13).

    Records each in attachment_location with how it matched: a recorded
    path, a GUID folder plus name, or name plus size (flagged). A name alone
    is counted and never stored. Then prints where each attachment's next
    copy would come from and what remains unfetchable. Reads listings,
    catalogs and databases only; copies nothing. Counts only are printed:
    the paths stay in the database."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    seeds = _located_files(seed_db, "--seed-db")
    listings = _located_files(listing, "--listing")
    access = access_from_config(cfg)

    conn = _connect_and_verify_or_die(cfg)
    try:
        if report_only:
            coverage = build_coverage(conn, access, cfg.paths.data_root)
            for line in _coverage_lines(coverage):
                typer.echo(f"locate-attachments: {line}")
            return
        report, coverage = run_locate(
            conn,
            access=access,
            data_root=cfg.paths.data_root,
            seeds=seeds,
            listings=listings,
            catalogs=catalog or [],
            walk_local=not no_local_walk,
            dry_run=dry_run,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"locate-attachments: attachments not materialized={report.targets} "
        f"({_counter_text(report.targets_by_state)})"
    )
    typer.echo(
        f"locate-attachments: read seed_rows={report.seed_rows} local_files={report.local_files} "
        f"listing_rows={report.listing_rows} catalog_files={report.catalog_files} "
        f"catalog_rows={report.catalog_rows} catalogs_without_a_code={report.catalogs_skipped}"
    )
    typer.echo(f"locate-attachments: rows by location/match/outcome: {_counter_text(report.stored)}")
    typer.echo(
        f"locate-attachments: weak matches (name only, or a GUID folder holding another name) — "
        f"never stored, never fetched: {_counter_text(report.weak)}; over the per-location "
        f"cap={report.capped}; paths not valid UTF-8={report.undecodable_paths}"
    )
    for line in _coverage_lines(coverage):
        typer.echo(f"locate-attachments: {line}")
    if dry_run:
        typer.echo(DRY_RUN_MARKER)


@app.command("push-attachments-plan")
def push_attachments_plan(
    location: Annotated[
        list[str],
        typer.Option("--location", help="A location code the pushing host serves. Repeatable."),
    ],
    config: ConfigOption = None,
) -> None:
    """Index host: print the push plan for `imsg push-attachments` (machine-readable).

    For the pushing host's own use over SSH. The plan names file paths, so
    it is written only to a pipe: run from a terminal it refuses. Counts go
    to stderr. Reads only."""
    if sys.stdout.isatty():
        typer.echo(
            "imsg: push-attachments-plan writes file paths; it is read by "
            "'imsg push-attachments' over SSH and refuses to print to a terminal",
            err=True,
        )
        raise typer.Exit(code=2)
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        plan = build_push_plan(
            conn, locations=location, access=access_from_config(cfg), data_root=cfg.paths.data_root
        )
    finally:
        conn.close()
    for line in plan_to_lines(plan):
        sys.stdout.write(line + "\n")
    sys.stdout.flush()
    candidates = sum(len(item.candidates) for item in plan.items)
    typer.echo(
        f"push-attachments-plan: attachments={len(plan.items)} candidates={candidates}", err=True
    )


@app.command("push-attachments-record")
def push_attachments_record(config: ConfigOption = None) -> None:
    """Index host: record what `imsg push-attachments` did (results on stdin).

    Writes only the attempt columns of the planned attachment_location
    rows; rsync's messages go to a log under data_root/logs."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    try:
        results, rsync_log = parse_results(sys.stdin)
    except (PushError, ValueError) as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    conn = _connect_and_verify_or_die(cfg)
    try:
        counts = record_push_results(
            conn, results, rsync_log=rsync_log, log_dir=cfg.paths.data_root / "logs"
        )
    finally:
        conn.close()
    typer.echo(f"push-attachments-record: recorded {_counter_text(counts)}")


def _remote_imsg(
    ssh: str, ssh_host: str, argv: list[str], *, stdin: bytes | None, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*shlex.split(ssh), ssh_host, shlex.join(argv)],
        input=stdin,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


@app.command("push-attachments")
def push_attachments(
    ssh_host: Annotated[str, typer.Option(help="The index host, as ssh names it.")],
    remote_imsg: Annotated[
        str, typer.Option(help="Absolute path of the imsg executable on the index host.")
    ],
    root: Annotated[
        list[str],
        typer.Option(
            "--root",
            help="CODE=PATH: a location this host serves and the directory its paths are "
            "below (this Mac's ~/Library/Messages/Attachments, or a drive's mount point). "
            "Repeatable, in preference order.",
        ),
    ],
    remote_config: Annotated[
        str | None, typer.Option(help="Absolute path of config.yaml on the index host.")
    ] = None,
    ssh: Annotated[str, typer.Option(help="ssh command, with options.")] = "ssh -o BatchMode=yes",
    rsync: Annotated[str, typer.Option(help="rsync command on this host.")] = "rsync",
    dry_run: DryRunOption = False,
) -> None:
    """Run on a host the index host cannot reach: copy attachment files the
    index is missing into its staging directory, read-only (D13).

    Asks the index host for a plan over SSH, checks each planned file here
    (below its --root, a regular file, the listed size), and copies the first
    good copy of each attachment with one rsync per location; this host is
    rsync's sender and its files are only read. Then sends back what
    happened to each candidate. The index host verifies and materializes
    the copies at its next `imsg backfill-attachments`. Needs no config and
    no database here. Prints counts only; neither the plan nor any file
    list touches this host's disk."""
    try:
        roots = [parse_root(spec) for spec in root]
    except PushError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    base = [remote_imsg]
    config_args = ["--config", remote_config] if remote_config else []
    plan_argv = [*base, "push-attachments-plan", *config_args]
    for push_root in roots:
        plan_argv += ["--location", push_root.location]

    fetched = _remote_imsg(ssh, ssh_host, plan_argv, stdin=None, timeout=1800)
    for line in fetched.stderr.decode("utf-8", "replace").splitlines():
        typer.echo(f"index host: {line}", err=True)
    if fetched.returncode != 0:
        typer.echo(f"imsg: the index host did not return a plan (exit {fetched.returncode})",
                   err=True)
        raise typer.Exit(code=1)
    try:
        # Split on "\n" only: `splitlines` also breaks on Unicode line
        # separators, which a file name inside a JSON string may contain.
        plan = parse_plan(fetched.stdout.decode("utf-8").split("\n"))
        report, results, rsync_log = run_push(
            plan, roots, ssh_host=ssh_host, ssh=ssh, rsync=rsync, runner=run_copy, dry_run=dry_run
        )
    except (PushError, TransferError, ValueError) as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"push-attachments: plan attachments={report.items} "
        f"candidates checked here={report.candidates_checked} "
        f"candidates at locations with no --root here={report.no_root_here}"
    )
    typer.echo(
        f"push-attachments: {'would send' if dry_run else 'sent'}: "
        f"{_counter_text(report.selected)}; rsync exit codes: "
        f"{_counter_text(report.copy_exit_codes)}"
    )
    typer.echo(
        f"push-attachments: not sent, by location/outcome: {_counter_text(report.not_sent)}; "
        f"attachments with nothing sendable here={report.unserved}"
    )
    if dry_run:
        typer.echo(DRY_RUN_MARKER)
        return
    record_argv = [*base, "push-attachments-record", *config_args]
    payload = "".join(line + "\n" for line in results_to_lines(results, rsync_log))
    recorded = _remote_imsg(ssh, ssh_host, record_argv, stdin=payload.encode("utf-8"),
                            timeout=600)
    for line in (recorded.stdout + recorded.stderr).decode("utf-8", "replace").splitlines():
        typer.echo(f"index host: {line}")
    if recorded.returncode != 0:
        typer.echo(
            f"imsg: the index host did not record the results (exit {recorded.returncode}); "
            f"the staged copies are still verified at its next backfill",
            err=True,
        )
        raise typer.Exit(code=1)


@mcp_app.command("local")
def mcp_local(config: ConfigOption = None) -> None:
    """`imsg mcp local` — stdio MCP server for use on this Mac or over SSH, full corpus scope."""
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

    # Every model call — the background warm-up and every query — runs on
    # this one thread (imsg.retrieval.model_thread explains why).
    model_thread = ModelThread()
    query_marker = _query_marker(cfg)
    service = RetrievalService(
        pg_conn=conn,
        fts_conn=fts_conn,
        config=cfg,
        text_provider=text_provider,
        reranker=reranker,
        multimodal_provider=multimodal_provider,
        model_thread=model_thread,
        query_marker=query_marker,
    )
    # Every load — at startup and after every unload — first asks whether
    # the host has the memory (imsg.memory_admission); refused, retrieval
    # calls answer WARMING_UP (host memory busy) and the next call asks
    # again. This server is never paused: it only ever defers a load.
    memory_probe = host_memory.default_probe()
    admission = MemoryAdmission.for_role(
        cfg, ModelRole.LOCAL_SERVER, command="imsg mcp local", probe=memory_probe
    )
    configure_mlx_memory_limit(cfg, ModelRole.LOCAL_SERVER)
    # Not warmed here: the server answers the MCP handshake first, and
    # this starts on the first retrieval call (or, with
    # mcp.local.warm_at_start, once run_local_server is serving). Tool
    # calls wait for it (imsg.mcp.tools.local_server); its progress goes
    # to stderr.
    warm_up = BackgroundWarmUp(
        service.warm_up_steps(),
        model_thread=model_thread,
        log=lambda line: typer.echo(f"mcp local: {line}", err=True),
        admission=admission.check,
        admission_retry_seconds=cfg.memory.admission_retry_seconds,
    )
    # Drops the models after mcp.local.idle_unload_seconds with no
    # retrieval call: every client session runs its own copy of this
    # server, so idle sessions would otherwise hold a model set each
    # (imsg.retrieval.idle_unload). Drops them at once, idle or not, when
    # the kernel reports memory.local_server_release_at pressure.
    idle_unloader = IdleModelUnloader(
        warm_up=warm_up,
        model_thread=model_thread,
        unload=service.unload_models,
        idle_seconds=cfg.mcp.local.idle_unload_seconds,
        log=lambda line: typer.echo(f"mcp local: {line}", err=True),
        pressure_release=PressureRelease(
            read_pressure=memory_probe.pressure,
            release_at=PressureLevel(cfg.memory.local_server_release_at),
            check_seconds=cfg.memory.pressure_check_seconds,
        ),
        after_unload=admission.release,
    )
    audit = PostgresAuditSink(lambda: connect(cfg.database, autocommit=True))
    # Every client session runs its own copy of this server and most never
    # search, so by default nothing loads until the first retrieval call
    # (mcp.local.warm_at_start).
    local = LocalMcpServer(
        service=service,
        audit=audit,
        config=cfg,
        conn=conn,
        warm_up=warm_up,
        idle_unloader=idle_unloader,
        warm_at_start=cfg.mcp.local.warm_at_start,
    )
    if not cfg.mcp.local.warm_at_start:
        typer.echo(
            "mcp local: the models load on the first retrieval call "
            "(mcp.local.warm_at_start is false)",
            err=True,
        )
    try:
        anyio.run(run_local_server, local)
    finally:
        model_thread.close()
        admission.release()
        query_marker.close()
        fts_conn.close()
        conn.close()


def _run_at1_probe(cfg: Config, owner_token_ref: str | None, foreign_token_ref: str | None) -> None:
    """`imsg mcp public --probe` — AT-1's synthetic auth probe (SPEC §12).

    The precondition to exposing a personal corpus to the internet, so
    the order here is load-bearing: **every** refusal happens in
    `check_probe_preconditions`, which is a pure function over config and
    two reference strings, before the audit sink or the auth gate exists.
    Nothing on a refusal path can have contacted Google, because nothing
    that could has been constructed yet.

    No model providers are built and no server is started — the probe
    tests the gate, not the service, and SPEC AT-1 step 0 has it run
    "while the real server stays disabled". `mcp.public.enabled` is
    therefore deliberately NOT required here.
    """
    try:
        tokens = check_probe_preconditions(
            cfg, owner_token_ref=owner_token_ref, foreign_token_ref=foreign_token_ref
        )
    except ProbeConfigurationError as exc:
        typer.echo(f"imsg: AT-1 probe did not run — {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG) from exc

    run_guard_mount_or_exit(cfg.paths.data_root)

    # The audit log is not a detail of this test, it is half of it: AT-1
    # steps 3-4 turn on the difference between "the non-owner saw nothing"
    # and "the non-owner was rejected and the rejection is recorded", and
    # the standing invariant is asserted over the log's entire history.
    # So this runs against the real `mcp_audit` table and writes real rows.
    conn = _connect_and_verify_or_die(cfg)
    try:
        audit = PostgresAuditSink(lambda: connect(cfg.database, autocommit=True))
        try:
            gate = build_public_gate(cfg.mcp.public, audit=audit)
        except ImsgError as exc:
            typer.echo(f"imsg: AT-1 probe did not run — {exc}", err=True)
            raise typer.Exit(code=EXIT_CONFIG) from exc

        owner_subject = cfg.mcp.public.oauth.owner_subject
        assert owner_subject is not None  # check_probe_preconditions proved this
        report = run_auth_probe(
            gate,
            audit,
            owner_token=tokens.owner,
            foreign_token=tokens.foreign,
            owner_subject=owner_subject.resolve(),
        )
    finally:
        conn.close()

    for line in format_probe_report(report, scope=cfg.mcp.public.scope):
        typer.echo(line)
    raise typer.Exit(code=verdict_exit_code(report.verdict))


@mcp_app.command("public")
def mcp_public(
    config: ConfigOption = None,
    probe: Annotated[
        bool,
        typer.Option(
            "--probe",
            help="Run the AT-1 synthetic auth probe instead of serving. "
            "Requires --owner-token-ref and --foreign-token-ref.",
        ),
    ] = False,
    owner_token_ref: Annotated[
        str | None,
        typer.Option(
            "--owner-token-ref",
            help="Secret REFERENCE to the owner's bearer token: 'keychain:<item>' "
            "or 'env:<VAR>'. Never the token itself — that would land in your "
            "shell history and in 'ps' output. --probe only.",
        ),
    ] = None,
    foreign_token_ref: Annotated[
        str | None,
        typer.Option(
            "--foreign-token-ref",
            help="Secret REFERENCE to a NON-owner account's bearer token, same "
            "form as --owner-token-ref. --probe only.",
        ),
    ] = None,
) -> None:
    """`imsg mcp public` — StreamableHTTP MCP server behind cloudflared,
    OAuth subject validation, fail closed. Binds loopback
    only (`mcp.public.bind`); cloudflared is the only process that ever
    faces a public interface. Every request — including `initialize`
    and `tools/list`, not only tool calls — passes through
    `imsg.mcp.auth.PublicAuthGate` before anything else runs (hard
    requirement 4: no unauthenticated path, no config flag to disable
    it); scope (`mcp.public.scope`, REQUIRED with no default) is fixed
    for the life of this process.

    `--probe` runs AT-1's synthetic auth probe instead of serving — the
    gate that must pass before any corpus is exposed.
    """
    cfg = _load_config_or_die(config)
    if probe:
        _run_at1_probe(cfg, owner_token_ref, foreign_token_ref)
        return  # pragma: no cover - _run_at1_probe always exits
    if owner_token_ref is not None or foreign_token_ref is not None:
        typer.echo(
            "imsg: --owner-token-ref/--foreign-token-ref are only meaningful with "
            "--probe; refusing to start the server with token references that would "
            "go unused",
            err=True,
        )
        raise typer.Exit(code=2)
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

    # Every model call — the background warm-up and every query — runs on
    # this one thread, exactly as on the local surface
    # (imsg.retrieval.model_thread explains why: MLX gives each OS thread
    # its own default GPU stream, so an array whose computation was set up
    # on the warm-up's thread cannot be evaluated on the event loop's).
    # Without it, warming in the background would make queries fail rather
    # than fast.
    model_thread = ModelThread()
    query_marker = _query_marker(cfg)
    service = RetrievalService(
        pg_conn=conn,
        fts_conn=fts_conn,
        config=cfg,
        text_provider=text_provider,
        reranker=reranker,
        multimodal_provider=multimodal_provider,
        model_thread=model_thread,
        query_marker=query_marker,
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

    # The models load in the background while the transport serves. There
    # is no unauthenticated readiness endpoint to ask — adding one would
    # be the single hole in the only access control this project has — so
    # progress goes to stderr (launchd captures it to
    # <data_root>/logs/imsgindex-mcp-public.err.log) and to a readiness
    # file `imsg status` reads (imsg.mcp.warm_up_readiness).
    readiness = WarmUpReadinessFile(readiness_path(cfg.paths.data_root))
    # As on the local surface, every load first asks whether the host has
    # the memory (imsg.memory_admission). Unlike it, this server loads
    # again by itself once admitted (rewarm), since it has to stay warm
    # for its latency budget, and it releases its models only at
    # memory.public_server_release_at (critical by default, or never).
    memory_probe = host_memory.default_probe()
    admission = MemoryAdmission.for_role(
        cfg, ModelRole.PUBLIC_SERVER, command="imsg mcp public", probe=memory_probe
    )
    configure_mlx_memory_limit(cfg, ModelRole.PUBLIC_SERVER)
    warm_up = BackgroundWarmUp(
        service.warm_up_steps(),
        model_thread=model_thread,
        log=lambda line: typer.echo(f"mcp public: {line}", err=True),
        on_status=readiness.publish,
        admission=admission.check,
        admission_retry_seconds=cfg.memory.admission_retry_seconds,
    )
    release_at = cfg.memory.public_server_release_at
    idle_unloader = IdleModelUnloader(
        warm_up=warm_up,
        model_thread=model_thread,
        unload=service.unload_models,
        idle_seconds=cfg.mcp.public.idle_unload_seconds,
        log=lambda line: typer.echo(f"mcp public: {line}", err=True),
        pressure_release=PressureRelease(
            read_pressure=memory_probe.pressure,
            release_at=None if release_at == "never" else PressureLevel(release_at),
            check_seconds=cfg.memory.pressure_check_seconds,
            rewarm=True,
            rewarm_cooldown_seconds=cfg.memory.public_rewarm_cooldown_seconds,
        ),
        after_unload=admission.release,
    )
    public = PublicMcpServer(
        service=service,
        gate=gate,
        scope=cfg.mcp.public.scope,
        warm_up=warm_up,
        idle_unloader=idle_unloader,
    )
    asgi_app = build_public_asgi_app(
        public,
        allowed_hosts=cfg.mcp.public.allowed_hosts,
        allowed_origins=cfg.mcp.public.allowed_origins,
        external_url=cfg.mcp.public.external_url,
    )

    typer.echo(
        f"mcp public: listening on {host}:{port}, scope={cfg.mcp.public.scope}, "
        f"external_url={cfg.mcp.public.external_url}",
        err=True,
    )
    # Started before the listener binds, and off this thread: the agent is
    # KeepAlive, so every relaunch is a cold load, and warming only once a
    # request arrives would put the whole load inside that request. Nothing
    # here blocks — `start()` hands the steps to the model thread and
    # returns, so uvicorn binds and serves while the weights load.
    warm_up.start()
    idle_unloader.start_watchdog()
    try:
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    finally:
        idle_unloader.stop()
        model_thread.close()
        admission.release()
        query_marker.close()
        fts_conn.close()
        conn.close()


# --------------------------------------------------------------------------
# export (S8, SPEC §11) — the allowlisted export gate
#
# The only path by which message content can leave this machine, so the
# shape of every command here is "refuse, unless". Each one loads config,
# runs the mount gate, connects, and turns any `ImsgError` into one clean
# line — an operator who sees a traceback from this surface cannot tell
# whether something was uploaded.
#
# `push` is the only command that can reach the network, and it cannot do
# so without `export.gcp_credentials` naming a credential: see
# `_export_transport_or_die`, which refuses BEFORE importing a Google
# client library. `push --dry-run` runs every verification the real push
# runs and builds no transport at all, so rehearsing the gate is provably
# network-free.
# --------------------------------------------------------------------------

export_app = typer.Typer(
    name="export",
    help="S8 — the default-deny export gate to GCS / Discovery Engine.",
    no_args_is_help=True,
)
app.add_typer(export_app, name="export")

RunIdArgument = Annotated[
    int,
    typer.Argument(
        metavar="RUN-ID",
        help="The export run id printed by `imsg export plan`.",
    ),
]


def _allowlist_person_count(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM allowlist_person")
        row = cur.fetchone()
        return int(row[0]) if row else 0


def _resolve_export_person_or_die(conn: psycopg.Connection, token: str) -> tuple[int, str]:
    """`<id-or-name>` -> (person_id, short_name).

    `short_name` is tried first and a numeric `person_id` only as a
    fallback, so a person whose short_name happens to be digits is still
    reachable by the name the operator sees in the review report. A token
    matching neither is a refusal, never a no-op: a purge that silently
    revoked nobody would read as success.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT person_id, short_name FROM person WHERE short_name = %s", (token,)
        )
        row = cur.fetchone()
        if row is None and token.isdigit():
            cur.execute(
                "SELECT person_id, short_name FROM person WHERE person_id = %s",
                (int(token),),
            )
            row = cur.fetchone()
    if row is None:
        raise ExportPlanError(
            f"no person matches '{token}' — pass the short_name shown in the export "
            f"review report (or a numeric person_id). Nothing was revoked."
        )
    return int(row[0]), str(row[1])


def _export_transport_or_die(cfg: Config) -> ExportTransport:
    """Build the real GCS / Discovery Engine transport, or refuse.

    The credential check comes first and the Google client libraries are
    imported only after it passes. That ordering is the mechanism, not a
    style choice: with `export.gcp_credentials` unset, this process never
    loads a Google client at all, so there is no code path from a stock
    checkout to a network call. There is deliberately no flag that
    substitutes a stand-in transport here — the fake one exists for tests,
    which construct it themselves.
    """
    ref = cfg.export.gcp_credentials
    if ref is None:
        raise ExportPushError(
            "export.gcp_credentials is not set — `imsg export push` has no credential "
            "to authenticate with and refuses rather than trying. Set it to a "
            "'keychain:<item>' or 'env:<VAR>' reference naming a GCP service-account "
            "key (SPEC §6; secrets never live in config.yaml itself). Nothing was "
            "uploaded, and no plan state changed."
        )
    from imsg.export.gcp_transport import (
        build_gcs_discovery_engine_transport,
        resolve_gcp_credentials,
    )
    from imsg.export.transport import TransportError

    try:
        credentials = resolve_gcp_credentials(ref)
    except TransportError as exc:
        # Never interpolate the resolved value — only the reference.
        raise ExportPushError(
            f"export.gcp_credentials ('{ref.raw}') did not resolve to a usable GCP "
            f"service-account key: {exc}"
        ) from exc
    return build_gcs_discovery_engine_transport(
        gcp_project=cfg.export.gcp_project,
        gcs_bucket=cfg.export.gcs_bucket,
        data_store_id=cfg.export.data_store_id,
        credentials=credentials,
    )


@export_app.command("plan")
def export_plan(config: ConfigOption = None, dry_run: DryRunOption = False) -> None:
    """Compute eligibility, stage the documents, write the review report.

    Stages immutable bytes under `$DATA_ROOT/export/staging/<run>/` and
    records the run; nothing leaves the machine. Read the report it names
    before approving — that review is the actual control.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        if _allowlist_person_count(conn) == 0:
            # Not merely "an empty plan": with nobody classified, a reconcile
            # would also schedule the deletion of everything previously
            # pushed. Refuse and say which of the two directions the operator
            # probably wants.
            raise ExportPlanError(
                "allowlist_person is empty — every thread is denied by default "
                "(hard requirement 5), so this plan would stage nothing and would "
                "schedule the deletion of any document already pushed. Classify at "
                "least one person before planning; to retract one person, use "
                "`imsg export purge-person`."
            )
        if dry_run:
            preview = preview_plan(conn, cfg)
            typer.echo(
                f"export plan: would stage {preview.upsert_count} document(s) across "
                f"{len(preview.chat_ids)} thread(s), delete {preview.delete_count}, "
                f"leave {preview.unchanged_count} unchanged"
            )
            for document_id in preview.delete_document_ids:
                typer.echo(f"export plan: would delete {document_id}")
            typer.echo(DRY_RUN_MARKER)
            return
        with conn.transaction():
            result = plan_export(conn, cfg)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"export plan: run {result.run_id} (mode={result.mode}) — "
        f"{result.upsert_count} upsert(s), {result.delete_count} delete(s), "
        f"{result.unchanged_count} unchanged"
    )
    typer.echo(f"export plan: staged under {result.staging_dir}")
    typer.echo(f"export plan: review report {result.report_path}")
    if result.approval_required:
        typer.echo(
            "export plan: OWNER APPROVAL REQUIRED before push — reasons: "
            + ", ".join(result.approval_reasons)
        )
        typer.echo(
            f"export plan: read the report, then `imsg export approve {result.run_id}`"
        )
    else:
        typer.echo(
            f"export plan: no new scope — `imsg export push {result.run_id}` may "
            f"proceed without fresh approval (SPEC §11.4)"
        )


@export_app.command("approve")
def export_approve(
    run_id: RunIdArgument,
    config: ConfigOption = None,
    approval_id: Annotated[
        str | None,
        typer.Option(
            help="Record this approval id instead of a generated one (audit trails).",
        ),
    ] = None,
) -> None:
    """Record owner approval of a planned run, pinning its exact bytes.

    Re-reads the staged manifest and re-hashes every staged file first:
    an approval can never be minted for bytes the owner did not stage.
    Refuses an unknown run id, a run that is not in 'planned' state, and
    any staging that changed since the plan.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        with conn.transaction():
            result = approve_run(conn, cfg, run_id, approval_id=approval_id)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(f"export approve: run {result.run_id} approved as {result.approval_id}")
    typer.echo(
        f"export approve: pinned manifest sha256 {result.approved_manifest_sha256}"
    )
    typer.echo(
        f"export approve: `imsg export push {result.run_id}` may now promote exactly "
        f"those bytes — any later change voids this approval"
    )


@export_app.command("push")
def export_push(
    run_id: RunIdArgument,
    config: ConfigOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Promote one approved plan to GCS / Discovery Engine, or refuse.

    Re-verifies every hash pin AND re-derives eligibility from the live
    database before the first byte leaves: the pins prove the bytes did
    not change, not that the world did not (D9). Any drift aborts and
    requires a new plan. `--dry-run` runs all of that and stops — it
    builds no transport, so it cannot reach the network even with a
    credential configured.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    if dry_run:
        _export_push_rehearse(cfg, run_id)
        return
    # The credential gate runs BEFORE the database is opened, so a push with
    # nothing configured refuses having touched nothing at all — and no
    # Google client library is ever imported into the process.
    try:
        transport = _export_transport_or_die(cfg)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _export_push_execute(cfg, run_id, transport)


def _export_push_rehearse(cfg: Config, run_id: int) -> None:
    """`push --dry-run`: every check the real push runs, then stop. Builds
    no transport, so it needs no credential and cannot reach the network."""
    conn = _connect_and_verify_or_die(cfg)
    try:
        preflight = verify_push_preconditions(conn, cfg, run_id)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"export push: run {preflight.run_id} (mode={preflight.mode}) verifies — "
        f"would push {preflight.upsert_count} upsert(s) and "
        f"{preflight.delete_count} delete(s), "
        f"{preflight.already_done_count} item(s) already done"
    )
    typer.echo(
        "export push: approval "
        + (
            "satisfied for: " + ", ".join(preflight.approval_reasons)
            if preflight.approval_reasons
            else "not required for this run (SPEC §11.4)"
        )
    )
    typer.echo(DRY_RUN_MARKER)


def _export_push_execute(cfg: Config, run_id: int, transport: ExportTransport) -> None:
    conn = _connect_and_verify_or_die(cfg)
    try:
        # Deliberately NOT wrapped in `conn.transaction()`: the connection is
        # autocommit, so each item's recorded outcome lands as it happens.
        # Wrapping would mean a crash after a successful upload rolls back
        # the `export_document` row that records it — leaving a document in
        # the corporate store that this system's reconciler cannot see and
        # `purge-person` therefore cannot delete. Redundant re-uploads on a
        # retry are cheap and idempotent; an invisible document is not.
        result = push_export(conn, cfg, run_id, transport)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"export push: run {result.run_id} {result.status} — pushed={result.pushed} "
        f"deleted={result.deleted} failed={result.failed} "
        f"skipped_already_done={result.skipped_already_done}"
    )
    for note in result.notes:
        typer.echo(f"export push: {note}", err=True)
    if result.status != "ok":
        typer.echo(
            f"export push: run {result.run_id} has failed items and stays retryable — "
            f"re-run `imsg export push {result.run_id}` (it re-verifies every pin and "
            f"re-derives eligibility first)",
            err=True,
        )
        raise typer.Exit(code=1)


@export_app.command("purge-person")
def export_purge_person(
    person: Annotated[
        str,
        typer.Argument(
            metavar="ID-OR-NAME",
            help="short_name (preferred) or numeric person_id of the person to revoke.",
        ),
    ],
    config: ConfigOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Revoke one person and plan the deletion of everything they touched.

    Flags their allowlist row to deny — the row is kept, so the record
    that they were ever allowed survives — then plans the removal of every
    now-ineligible document. The resulting run is exempt from the approval
    gate (D9.3: retraction only narrows scope), and is recorded in full;
    unapproved does not mean unlogged. Push it to execute the deletions,
    which are verified absent by document id.

    Honest limit: this reaches the Discovery Engine index and the GCS
    bucket. Copies already in organizational retention, backups, or
    another person's hands are beyond it.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        person_id, short_name = _resolve_export_person_or_die(conn, person)
        if dry_run:
            preview = preview_purge(conn, cfg, short_name)
            typer.echo(
                f"export purge-person: would revoke {short_name} (person {person_id}) "
                f"and delete {preview.delete_count} document(s)"
            )
            for document_id in preview.delete_document_ids:
                typer.echo(f"export purge-person: would delete {document_id}")
            typer.echo(DRY_RUN_MARKER)
            return
        with conn.transaction():
            result = purge_person(conn, cfg, short_name)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"export purge-person: revoked {short_name} (person {person_id}) — "
        f"text_allowed and attachments_allowed are now false; the row is retained "
        f"for audit"
    )
    typer.echo(
        f"export purge-person: run {result.run_id} plans {result.delete_count} "
        f"deletion(s) and {result.upsert_count} upsert(s)"
    )
    typer.echo(f"export purge-person: review report {result.report_path}")
    typer.echo(
        f"export purge-person: purges are exempt from the approval gate (D9.3) — "
        f"run `imsg export push {result.run_id}` to execute the deletions"
    )


@export_app.command("unclassified-report")
def export_unclassified_report(
    config: ConfigOption = None, dry_run: DryRunOption = False
) -> None:
    """Write the weekly unclassified-threads report.

    Active chats whose participants have never been classified for
    export, so a static allowlist does not quietly rot as people join and
    leave. Identities and counts only, never message content, and written
    outside `export/staging/` so no push can ever select it. This is the
    command the weekly LaunchAgent invokes.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        count = unclassified_summary(conn)
        if dry_run:
            typer.echo(
                f"export unclassified-report: {count} unclassified active thread(s)"
            )
            typer.echo(DRY_RUN_MARKER)
            return
        report_path = write_unclassified_report(conn, cfg)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"export unclassified-report: {count} unclassified active thread(s) — "
        f"wrote {report_path}"
    )


# --------------------------------------------------------------------------
# backup (SPEC §5.3, §14) — the daily 04:00 LaunchAgent's command
# --------------------------------------------------------------------------


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    gib = n / float(2**30)
    return f"{n} B ({gib:.2f} GiB)" if gib >= 0.01 else f"{n} B"


@app.command()
def backup(
    config: ConfigOption = None,
    keep: Annotated[
        int,
        typer.Option(
            "--keep",
            help="How many complete backup sets to retain (default: 14). "
            "Clamped to a minimum of 1 — the newest set is never a deletion candidate.",
        ),
    ] = DEFAULT_KEEP,
    pg_dump_binary: Annotated[
        Path | None,
        typer.Option(
            "--pg-dump",
            help="Path to a specific pg_dump. Defaults to the one on $PATH; pass "
            "this when $PATH's pg_dump is older than the instance's major version.",
        ),
    ] = None,
    dry_run: DryRunOption = False,
) -> None:
    """Nightly local recovery copy: verified `pg_dump` + FTS sidecar, 14 kept.

    What this does NOT cover is as important as what it does — the
    attachment cache (~147 GB), the model directory and `ops/` are
    deliberately out of scope, for the reasons in
    `imsg.backup.pipeline`'s docstring, and these copies share the
    physical device with the data they copy. Both facts are printed on
    every run rather than left to be inferred.

    Refuses rather than half-running: a missing or unwritable
    destination, a missing `pg_dump`, a `pg_dump` older than the
    instance, insufficient free space, a dump that fails its read-back,
    or a corrupt FTS sidecar each abort with one `imsg: …` line and
    leave `backups/` exactly as it was.
    """
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    conn = _connect_and_verify_or_die(cfg)
    try:
        report = run_backup(
            conn=conn,
            config=cfg,
            keep=keep,
            pg_dump_binary=pg_dump_binary,
            dry_run=dry_run,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    plan = report.retention
    if dry_run:
        typer.echo(
            f"backup: preconditions OK — {_format_bytes(report.free_bytes_before)} free, "
            f"{_format_bytes(report.required_bytes)} required"
        )
        typer.echo(
            f"backup: would retain {len(plan.retained)} set(s), delete "
            f"{len(plan.delete)} (keep={plan.keep})"
        )
        for path in plan.delete:
            typer.echo(f"backup:   would delete {path.name}")
        typer.echo(DRY_RUN_MARKER)
    else:
        assert report.dump is not None and report.set_path is not None
        fts_line = (
            f"fts.db {_format_bytes(report.fts.byte_size)}"
            if report.fts is not None and report.fts.present
            else "fts.db absent (not yet built — rebuildable from Postgres)"
        )
        typer.echo(f"backup: wrote {report.set_path}")
        typer.echo(
            f"backup:   postgres.dump {_format_bytes(report.dump.byte_size)} "
            f"(pg_dump {report.dump.pg_dump_major} -> server {report.dump.server_major}, "
            f"read back in full), {fts_line}"
        )
        typer.echo(
            f"backup:   retained {len(plan.retained)} set(s), deleted "
            f"{len(report.deleted)} (keep={plan.keep})"
        )

    if plan.partial:
        typer.echo(
            f"backup: WARNING {len(plan.partial)} partial/incomplete set(s) left by an "
            f"interrupted run are present and were NOT deleted (retention only removes "
            f"sets it can prove are complete): "
            f"{', '.join(p.name for p in plan.partial)}"
        )
    if plan.foreign:
        typer.echo(
            f"backup: note {len(plan.foreign)} unrecognized entr(ies) in backups/ were "
            f"left untouched: {', '.join(p.name for p in plan.foreign)}"
        )
    typer.echo(f"backup: {OUT_OF_SCOPE_NOTE}")
    typer.echo(f"backup: {SAME_DEVICE_CAVEAT}")


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
    into `~/Library/LaunchAgents` — the only place launchd
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
