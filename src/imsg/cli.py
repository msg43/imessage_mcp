"""The `imsg` CLI (SPEC §8, §14).

Wired up for real: `migrate`, `check-permissions`, `status`,
`guard-mount`, `snapshot`, `extract`, `identity`, `segment`, `embed`,
`sync`, `enrich`, `backfill-attachments`, `mcp local`. `export` and
`install-agents` remain `StageNotImplementedError` stubs — `export` is
a parallel agent's scope this wave (SPEC §11, `src/imsg/export/`,
untouched here), and `install-agents` (SPEC §5.5) is not yet built by
anyone.

**Placeholder model providers (flagged prominently)**: `segment`,
`embed`, `sync`, and `mcp local` construct `Fake*EmbeddingProvider` /
`FakeBoundaryProvider` / `FakeRerankerProvider` — no MLX-backed
provider loader exists anywhere in this codebase yet (SPEC §4.1's real
models are explicitly Phase 3/5 work), so there is nothing else to
wire these CLI commands to. This makes every stage genuinely runnable
end-to-end today, but retrieval/segmentation *quality* from these
commands is not representative of the real system. A follow-up build
should add a real provider-loader module (reading `embedding.model` /
`revision` / `quantization` etc. from config) and swap it in exactly
where `_text_provider`/`_multimodal_provider`/`FakeBoundaryProvider()`/
`FakeRerankerProvider()` are constructed below.

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
from typing import TYPE_CHECKING, Annotated

import anyio
import apsw
import typer

from imsg.backfill.pipeline import DEFAULT_RATE_PER_MINUTE, run_backfill
from imsg.config.loader import load_config
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
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.enrich.pipeline import EnrichmentProviders, process_one_task
from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider
from imsg.enrich.queue import claim_tasks
from imsg.errors import ImsgError, StageNotImplementedError
from imsg.mcp.audit import PostgresAuditSink
from imsg.mcp.tools.local_server import LocalMcpServer, run_local_server
from imsg.mount.guard import run_guard_mount_or_exit
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.service import RetrievalService
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import REBUILD_ALL_SENTINEL, run_segment, run_segment_for_chat
from imsg.stages.extract import run_extract
from imsg.stages.identity import run_identity
from imsg.stages.imsg_dump import default_binary_path
from imsg.stages.snapshot import SNAPSHOT_FILENAME, SNAPSHOT_SUBDIR, run_snapshot
from imsg.stages.sync import EmbedFn, SegmentFn, run_sync_all_sources

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

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to config.yaml. Defaults to $IMSG_CONFIG, then ./config.yaml.",
    ),
]


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
    return conn


def _fts_db_path(cfg: Config) -> Path:
    return cfg.paths.data_root / "fts" / "fts.db"


def _open_fts_conn(cfg: Config) -> apsw.Connection:
    path = _fts_db_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(path))
    create_schema(conn)
    return conn


def _boundary_prompt_bytes_or_die(cfg: Config) -> bytes:
    path = cfg.paths.data_root / cfg.segmentation.boundary_prompt
    try:
        return path.read_bytes()
    except OSError as exc:
        typer.echo(
            f"imsg: segmentation boundary prompt not found at '{path}' — author "
            f"it before running segmentation (SPEC §6 segmentation.boundary_prompt)",
            err=True,
        )
        raise typer.Exit(code=1) from exc


def _text_provider(cfg: Config) -> FakeTextEmbeddingProvider:
    # PLACEHOLDER — see module docstring: no real provider loader exists yet.
    return FakeTextEmbeddingProvider(dim=cfg.embedding.dim)


def _multimodal_provider(cfg: Config) -> FakeMultimodalEmbeddingProvider | None:
    # PLACEHOLDER — see module docstring.
    if not cfg.embedding.multimodal.enabled:
        return None
    return FakeMultimodalEmbeddingProvider(dim=cfg.embedding.multimodal.dim)


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

    for key, value in report.items():
        typer.echo(f"{key}: {value}")


@app.command()
def snapshot(config: ConfigOption = None) -> None:
    """S1 — snapshot the live chat.db via the SQLite online-backup API (SPEC §8 S1)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    try:
        result = run_snapshot(live_chat_db=cfg.paths.live_chat_db, data_root=cfg.paths.data_root)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"snapshot: {result.path} sha256={result.sha256} "
        f"reused_existing={result.reused_existing}"
    )


@app.command()
def extract(
    config: ConfigOption = None,
    source: Annotated[
        str | None,
        typer.Option(help="Source name from sync.sources; defaults to the first configured source."),
    ] = None,
) -> None:
    """S2 — extract chats/messages/attachments from the current snapshot (SPEC §8 S2)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)

    source_name = source or cfg.sync.sources[0].name
    snapshot_path = cfg.paths.data_root / SNAPSHOT_SUBDIR / SNAPSHOT_FILENAME
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


@app.command()
def identity(config: ConfigOption = None) -> None:
    """S3 — resolve handles to person_id via Contacts + manual curation (SPEC §8 S3)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)

    conn = _connect_and_verify_or_die(cfg)
    try:
        result = run_identity(conn=conn, config=cfg)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"identity: persons_created={result.persons_created} "
        f"handles_created={result.handles_created} invariant_ok={result.invariant.ok}"
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
) -> None:
    """S4 — sessionize and segment messages for indexing (SPEC §8 S4)."""
    cfg = _load_config_or_die(config)
    if rebuild and chat is None:
        typer.echo("imsg: --rebuild requires --chat <id>", err=True)
        raise typer.Exit(code=2)
    run_guard_mount_or_exit(cfg.paths.data_root)

    prompt_bytes = _boundary_prompt_bytes_or_die(cfg)
    provider = FakeBoundaryProvider()  # PLACEHOLDER — see module docstring.

    conn = _connect_and_verify_or_die(cfg)
    try:
        if rebuild:
            assert chat is not None
            report = run_segment_for_chat(
                conn, chat, cfg, provider, prompt_bytes, earliest_changed_at=REBUILD_ALL_SENTINEL
            )
            typer.echo(
                f"segment: chat {chat} rebuilt — segments_written={report.segments_written} "
                f"segments_deleted={report.segments_deleted}"
            )
        else:
            chat_ids = {chat} if chat is not None else None
            reports = run_segment(conn, cfg, provider, prompt_bytes, chat_ids=chat_ids)
            total_written = sum(r.segments_written for r in reports)
            total_fallback = sum(r.fallback_sessions for r in reports)
            typer.echo(
                f"segment: {len(reports)} chat(s) processed, {total_written} segment(s) "
                f"written, {total_fallback} fallback session(s)"
            )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()


@app.command()
def embed(config: ConfigOption = None) -> None:
    """S6 — embed segments/attachment chunks and update the FTS sidecar (SPEC §8 S6)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)

    conn = _connect_and_verify_or_die(cfg)
    fts_conn = _open_fts_conn(cfg)
    try:
        report = run_embed(
            conn,
            _text_provider(cfg),
            multimodal_provider=_multimodal_provider(cfg),
            batch_size=cfg.embedding.batch_size,
        )
        sync_report = sync_fts(conn, fts_conn)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        fts_conn.close()
        conn.close()

    typer.echo(
        f"embed: segments_embedded={report.segments_embedded} "
        f"chunks_embedded={report.chunks_embedded} "
        f"attachments_embedded={report.attachments_embedded}"
    )
    typer.echo(
        f"embed: fts events_applied={sync_report.events_applied} "
        f"(upserts={sync_report.upserts} deletes={sync_report.deletes})"
    )


def _make_segment_fn(cfg: Config) -> SegmentFn:
    prompt_bytes = _boundary_prompt_bytes_or_die(cfg)
    provider = FakeBoundaryProvider()  # PLACEHOLDER — see module docstring.

    def _segment_fn(conn: psycopg.Connection, config: Config) -> object:
        return run_segment(conn, config, provider, prompt_bytes)

    return _segment_fn


def _make_embed_fn(cfg: Config) -> EmbedFn:
    del cfg  # unused: each invocation reads whatever `config` `run_sync` hands it

    def _embed_fn(conn: psycopg.Connection, config: Config) -> object:
        report = run_embed(
            conn,
            _text_provider(config),
            multimodal_provider=_multimodal_provider(config),
            batch_size=config.embedding.batch_size,
        )
        fts_conn = _open_fts_conn(config)
        try:
            sync_fts(conn, fts_conn)
        finally:
            fts_conn.close()
        return report

    return _embed_fn


@app.command()
def sync(config: ConfigOption = None) -> None:
    """S7 — incremental S1→S2→S3→S4→S6 sync for every configured source (SPEC §8 S7)."""
    cfg = _load_config_or_die(config)
    segment_fn = _make_segment_fn(cfg)  # validates the boundary prompt exists up front
    embed_fn = _make_embed_fn(cfg)

    conn = _connect_and_verify_or_die(cfg)
    try:
        results = run_sync_all_sources(
            conn=conn,
            config=cfg,
            imsg_dump_binary=default_binary_path(_repo_root()),
            segment_fn=segment_fn,
            embed_fn=embed_fn,
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    for r in results:
        typer.echo(
            f"sync: source={r.source_name} messages_upserted={r.extract.messages_upserted} "
            f"segment_ran={r.segment_ran} embed_ran={r.embed_ran}"
        )


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
) -> None:
    """S5b — OCR/caption/transcribe/pdftotext enrichment queue worker (SPEC §8 S5b)."""
    cfg = _load_config_or_die(config)
    run_guard_mount_or_exit(cfg.paths.data_root)

    conn = _connect_and_verify_or_die(cfg)
    providers = EnrichmentProviders(  # PLACEHOLDER — see module docstring.
        ocr=FakeOcrProvider(), caption=FakeCaptionProvider(), transcription=FakeTranscriptionProvider()
    )
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
        )
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.echo(
        f"backfill-attachments: considered={report.considered} "
        f"materialized={report.materialized} errored={report.errored} "
        f"marked_missing={report.marked_missing}"
    )
    if report.trial_gate_capped:
        typer.echo(
            "backfill-attachments: first-run trial gate active — pass --yes-full-run "
            "to process the rest",
            err=True,
        )
    if report.halted_low_disk_space:
        typer.echo("backfill-attachments: halted — low disk space", err=True)


@mcp_app.command("local")
def mcp_local(config: ConfigOption = None) -> None:
    """`imsg mcp local` — stdio MCP server, tailnet/SSH only, full corpus scope (SPEC §10.3)."""
    cfg = _load_config_or_die(config)
    if not cfg.mcp.local.enabled:
        typer.echo("imsg: mcp.local.enabled is false in config", err=True)
        raise typer.Exit(code=1)
    run_guard_mount_or_exit(cfg.paths.data_root)

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
        text_provider=_text_provider(cfg),  # PLACEHOLDER — see module docstring.
        reranker=FakeRerankerProvider(),  # PLACEHOLDER — see module docstring.
        multimodal_provider=_multimodal_provider(cfg),
    )
    audit = PostgresAuditSink(lambda: connect(cfg.database, autocommit=True))
    local = LocalMcpServer(service=service, audit=audit, config=cfg, conn=conn)
    try:
        anyio.run(run_local_server, local)
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


@app.command("install-agents")
def install_agents() -> None:
    """Render and install the thin LaunchAgent plists (SPEC §5.5). Not implemented yet."""
    _stub("install-agents")


if __name__ == "__main__":  # pragma: no cover
    app()
