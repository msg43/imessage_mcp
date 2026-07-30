"""`imsg eval ...` (SPEC §13): query/label interchange, the runner, the
diff table, and the pooling workflow.

Deliberately does **not** import `imsg.cli` (that would be circular —
`imsg.cli` imports `eval_app` from here) so it re-implements the small
number of "load config, connect, verify fingerprint, open the FTS
sidecar" helpers `imsg.cli` also has, rather than reaching into that
module's private functions. `imsg.cli` wires this sub-app in with
`app.add_typer(eval_app, name="eval")` — see that module's docstring
for why: "adding my own subcommands" is this build's only sanctioned
edit to `cli.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import apsw
import typer

from imsg.config.loader import load_config
from imsg.db.connection import connect
from imsg.db.fingerprint import verify_data_directory
from imsg.embed.fts.schema import create_schema
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.errors import ImsgError
from imsg.eval.backend import LocalEvalBackend, PassthroughReranker
from imsg.eval.diff import diff_runs, format_diff_markdown
from imsg.eval.io import (
    config_projection_sha256,
    label_segment_by_key,
    labels_from_yaml,
    labels_to_yaml,
    load_labels,
    load_queries,
    queries_from_yaml,
    queries_to_yaml,
    run_filename,
    run_result_from_json,
    run_result_to_json,
    upsert_label,
    upsert_query,
)
from imsg.eval.models import EvalQuery
from imsg.eval.pool import build_pool, import_pool_worksheet, pool_to_worksheet_yaml
from imsg.eval.runner import run_eval
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.service import RetrievalService

if TYPE_CHECKING:
    import psycopg

    from imsg.config.schema import Config

eval_app = typer.Typer(name="eval", help="Eval harness: queries, labels, runs, diffs (SPEC §13).",
                        no_args_is_help=True)

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


def _open_fts_conn(cfg: Config) -> apsw.Connection:
    path = cfg.paths.data_root / "fts" / "fts.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(path))
    create_schema(conn)
    return conn


# --------------------------------------------------------------------------
# Config variants (SPEC §13.3: "reranker on/off, dual-vector on/off")
# --------------------------------------------------------------------------

VARIANT_REGISTRY: dict[str, dict[str, bool]] = {
    "default": {"no_rerank": False, "no_multimodal": False},
    "no-rerank": {"no_rerank": True, "no_multimodal": False},
    "no-multimodal": {"no_rerank": False, "no_multimodal": True},
    "no-rerank-no-multimodal": {"no_rerank": True, "no_multimodal": True},
}
"""Named, reproducible config variants the CLI can build without a
second `config.yaml` — segmentation-threshold variants are instead
realized by re-segmenting the corpus in place under a different
`--config` and diffing two runs (SPEC §13.3's other listed variant),
since that's how S4's `seg_config_hash` already tracks threshold
changes; nothing here needs to duplicate that."""


def _build_local_backend(
    conn: psycopg.Connection, fts_conn: apsw.Connection, cfg: Config, *, variant: str
) -> LocalEvalBackend:
    if variant not in VARIANT_REGISTRY:
        typer.echo(
            f"imsg: unknown eval variant '{variant}' (known: {sorted(VARIANT_REGISTRY)})",
            err=True,
        )
        raise typer.Exit(code=2)
    flags = VARIANT_REGISTRY[variant]
    multimodal_provider = None
    if not flags["no_multimodal"] and cfg.embedding.multimodal.enabled:
        multimodal_provider = FakeMultimodalEmbeddingProvider(dim=cfg.embedding.multimodal.dim)
    service = RetrievalService(
        pg_conn=conn,
        fts_conn=fts_conn,
        config=cfg,
        text_provider=FakeTextEmbeddingProvider(dim=cfg.embedding.dim),  # PLACEHOLDER — see imsg.cli's module docstring
        reranker=PassthroughReranker() if flags["no_rerank"] else FakeRerankerProvider(),
        multimodal_provider=multimodal_provider,
    )
    return LocalEvalBackend(service=service, context=LOCAL_FULL_ACCESS)


# --------------------------------------------------------------------------
# Query / label interchange (SPEC §13.1)
# --------------------------------------------------------------------------


@eval_app.command("import-queries")
def import_queries(
    file: Annotated[Path, typer.Option(help="queries.yaml to import (SPEC §13.1 shape).")],
    config: ConfigOption = None,
) -> None:
    """Upsert every query in `file` into the canonical `eval_query` table."""
    cfg = _load_config_or_die(config)
    queries = queries_from_yaml(file.read_text(encoding="utf-8"))
    conn = _connect_and_verify_or_die(cfg)
    try:
        with conn.transaction():
            for q in queries:
                upsert_query(conn, q)
    finally:
        conn.close()
    typer.echo(f"eval import-queries: upserted {len(queries)} quer(y/ies) from {file}")


@eval_app.command("export-queries")
def export_queries(
    file: Annotated[Path, typer.Option(help="Where to write queries.yaml.")],
    target: Annotated[str | None, typer.Option(help="Restrict to queries whose targets include this.")] = None,
    config: ConfigOption = None,
) -> None:
    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        queries = load_queries(conn, target=target)
    finally:
        conn.close()
    file.write_text(queries_to_yaml(queries), encoding="utf-8")
    typer.echo(f"eval export-queries: wrote {len(queries)} quer(y/ies) to {file}")


@eval_app.command("import-labels")
def import_labels(
    file: Annotated[Path, typer.Option(help="labels.yaml to import (SPEC §13.1 shape).")],
    config: ConfigOption = None,
) -> None:
    """Upsert every label in `file` into `relevance_label`, anchored on
    the `anchor_guid` already present in the file (this is the raw
    canonical-store shape, unlike `eval pool`'s worksheet — see that
    command for the segment_key-based path)."""
    cfg = _load_config_or_die(config)
    labels = labels_from_yaml(file.read_text(encoding="utf-8"))
    conn = _connect_and_verify_or_die(cfg)
    try:
        with conn.transaction():
            for lbl in labels:
                upsert_label(conn, lbl)
    finally:
        conn.close()
    typer.echo(f"eval import-labels: upserted {len(labels)} label(s) from {file}")


@eval_app.command("export-labels")
def export_labels(
    file: Annotated[Path, typer.Option(help="Where to write labels.yaml.")],
    query_id: Annotated[str | None, typer.Option(help="Restrict to one query_id.")] = None,
    config: ConfigOption = None,
) -> None:
    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        labels = load_labels(conn, query_id=query_id)
    finally:
        conn.close()
    file.write_text(labels_to_yaml(labels), encoding="utf-8")
    typer.echo(f"eval export-labels: wrote {len(labels)} label(s) to {file}")


@eval_app.command("label")
def label_cmd(
    query_id: Annotated[str, typer.Option("--query-id")],
    segment_key: Annotated[str, typer.Option("--segment-key")],
    grade: Annotated[int, typer.Option(min=0, max=2, help="0=not relevant, 1=relevant, 2=highly relevant.")],
    query_text: Annotated[
        str | None,
        typer.Option(help="Create --query-id with this text if it doesn't exist yet."),
    ] = None,
    config: ConfigOption = None,
) -> None:
    """The CLI counterpart to the local `mark_relevant` MCP tool (SPEC
    §10.2/§13.2): upserts one `relevance_label`, anchored on the
    segment's first message GUID."""
    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        with conn.transaction():
            lbl = label_segment_by_key(
                conn,
                query_id=query_id,
                segment_key=segment_key,
                grade=grade,
                source="manual",
                query_text_if_new=query_text,
            )
    except ValueError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"eval label: query={lbl.query_id} anchor_guid={lbl.anchor_guid} grade={lbl.grade}")


# --------------------------------------------------------------------------
# Runner + diff (SPEC §13.3)
# --------------------------------------------------------------------------


@eval_app.command("run")
def run_cmd(
    config: ConfigOption = None,
    target: Annotated[str, typer.Option(help="local|gemini — this build only wires up local.")] = "local",
    k: Annotated[int, typer.Option(help="Top-k per query (SPEC §13.3).")] = 10,
    variant: Annotated[
        str, typer.Option(help=f"Config variant: one of {sorted(VARIANT_REGISTRY)}.")
    ] = "default",
    run_label: Annotated[
        str | None,
        typer.Option("--label", help="Extra tag folded into this run's config hash, e.g. 'baseline'."),
    ] = None,
) -> None:
    """Score every `eval_query` (for `--target`) against a retrieval
    backend and write `eval/runs/<...>.json` (SPEC §13.3). This is the
    AT-4 baseline artifact once the canonical store meets the SPEC §12
    minimums — the command reports pass/fail against those minimums
    every run, not just once."""
    if target != "local":
        typer.echo(
            "imsg: --target gemini is not wired into this CLI build — it needs real "
            "GCP credentials and a completed Phase 7 export; construct a "
            "imsg.eval.backend.GeminiEvalBackend + imsg.eval.runner.run_eval "
            "programmatically instead (see imsg.eval.gemini_client)",
            err=True,
        )
        raise typer.Exit(code=1)

    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    fts_conn = _open_fts_conn(cfg)
    try:
        backend = _build_local_backend(conn, fts_conn, cfg, variant=variant)
        config_sha = config_projection_sha256(
            target=target, k=k, extra={"variant": variant, "label": run_label}
        )
        result = run_eval(conn, backend, target=target, config_sha256=config_sha, k=k)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        fts_conn.close()
        conn.close()

    runs_dir = cfg.paths.data_root / cfg.eval.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / run_filename(
        run_id=result.run_id, target=result.target, config_sha256=result.config_sha256
    )
    out_path.write_text(run_result_to_json(result), encoding="utf-8")

    typer.echo(f"eval run: wrote {out_path}")
    typer.echo(
        f"eval run: queries={len(result.per_query)} nDCG@{k}={result.ndcg_at_k_mean} "
        f"recall@{k}={result.recall_at_k_mean} mrr={result.mrr:.4f} "
        f"success@{k}={result.success_at_k_rate:.4f} judged_coverage={result.judged_coverage:.4f}"
    )
    if result.at4.passed:
        typer.echo("eval run: AT-4 minimums MET")
    else:
        typer.echo(f"eval run: AT-4 minimums NOT met — {'; '.join(result.at4.reasons)}", err=True)


@eval_app.command("diff")
def diff_cmd(
    run_a: Annotated[Path, typer.Argument(help="Path to the baseline run JSON.")],
    run_b: Annotated[Path, typer.Argument(help="Path to the candidate run JSON.")],
    out: Annotated[Path | None, typer.Option(help="Write markdown here instead of stdout.")] = None,
) -> None:
    """`imsg eval diff <run-a> <run-b>` (SPEC §13.3): "the artifact
    every retrieval change must include." Pure file-to-file — no DB
    connection needed."""
    a = run_result_from_json(run_a.read_text(encoding="utf-8"))
    b = run_result_from_json(run_b.read_text(encoding="utf-8"))
    diff = diff_runs(a, b)
    markdown = format_diff_markdown(diff)
    if out is not None:
        out.write_text(markdown, encoding="utf-8")
        typer.echo(f"eval diff: wrote {out}")
    else:
        typer.echo(markdown)


# --------------------------------------------------------------------------
# Pooling (SPEC §13.2)
# --------------------------------------------------------------------------


@eval_app.command("pool")
def pool_cmd(
    out: Annotated[Path, typer.Option(help="Where to write the un-graded judgment worksheet.")],
    config: ConfigOption = None,
    variants: Annotated[
        str, typer.Option(help=f"Comma-separated variant names from {sorted(VARIANT_REGISTRY)}.")
    ] = "default,no-rerank",
    top_n: Annotated[int, typer.Option(help="Pool depth per config (SPEC §13.2 default: 20).")] = 20,
    seed: Annotated[int | None, typer.Option(help="Randomization seed, for reproducible worksheets.")] = None,
    target: Annotated[str | None, typer.Option(help="Restrict to queries whose targets include this.")] = None,
) -> None:
    """SPEC §13.2: run >= 2 materially different configs, pool their
    top-N unique results per query, randomize, and write a worksheet
    for the owner to grade 0/1/2 — import it back with `eval
    import-pool`."""
    names = [v.strip() for v in variants.split(",") if v.strip()]
    if len(names) < 2:
        typer.echo(
            f"imsg: eval pool needs >= 2 variants (SPEC §13.2), got {names!r}", err=True
        )
        raise typer.Exit(code=2)

    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    fts_conn = _open_fts_conn(cfg)
    try:
        backends = {name: _build_local_backend(conn, fts_conn, cfg, variant=name) for name in names}
        queries: list[EvalQuery] = load_queries(conn, target=target)
        entries = build_pool(conn, backends, queries, top_n=top_n, seed=seed)
    finally:
        fts_conn.close()
        conn.close()

    out.write_text(pool_to_worksheet_yaml(entries), encoding="utf-8")
    typer.echo(
        f"eval pool: {len(entries)} candidate(s) across {len(queries)} quer(y/ies) "
        f"and {len(names)} variant(s) -> {out}"
    )


@eval_app.command("import-pool")
def import_pool_cmd(
    file: Annotated[Path, typer.Option(help="A worksheet from 'eval pool', with grades filled in.")],
    config: ConfigOption = None,
) -> None:
    """Import a graded worksheet as `relevance_label` rows
    (`source='pool_judgment'`) — entries still `grade: null` are
    skipped."""
    cfg = _load_config_or_die(config)
    conn = _connect_and_verify_or_die(cfg)
    try:
        with conn.transaction():
            count = import_pool_worksheet(conn, file.read_text(encoding="utf-8"))
    except ValueError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"eval import-pool: imported {count} label(s) from {file}")


__all__ = ["VARIANT_REGISTRY", "eval_app"]
