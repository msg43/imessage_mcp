"""S7 — Incremental sync orchestration (SPEC §8 S7), LaunchAgent every 15 min.

"S1 → S2 → S3 (auto-stub only) → S4 → S6 for new/edited rows. Per-source
watermarks in `sync_state`." This module owns the *ordering and gating*
of that pipeline for one source per call; the per-row watermark/dirty
logic within each stage stays in that stage (S2 owns
`sync_state['watermark.rowid.<source>']`; S3 has nothing to watermark —
it always sweeps every still-unresolved `source_handle` row; S4's own
dirty-chat tracking is that module's concern entirely).

"S3 (auto-stub only)" (SPEC §8 S7): the automated 15-minute sync never
blocks on interactive review — `imsg.stages.identity.run_identity` is
already fully automatic by construction (Contacts import or review-stub
creation, never a prompt), so this note is satisfied without any extra
code here; the owner's separate, manual `imsg identity review-report` /
`identity merge` workflow (SPEC §8 S3) is explicitly not part of this
loop.

**S4/S6 are dependency-injected, not imported** (`segment_fn`/`embed_fn`,
both `None` by default): this build's scope is S1/S2/S3/S7 only, and
S4/S6 are being built by other agents in parallel — hard-importing
their modules here would couple this file's importability to modules
outside this build's scope and outside its control. When absent, S4/S6
are skipped with a logged note rather than treated as an error, so
`run_sync` is fully functional end-to-end (snapshot → extract →
identity) today; a later consolidation step supplies the real callables
(e.g. `functools.partial(imsg.segment.pipeline.run_segment,
boundary_provider=..., boundary_prompt_bytes=...)` for S4, whose own
docstring already names itself "SPEC §8 S7's S4 step" — the two sides
of this seam were designed to meet here).

**Studio seed** (SPEC §8 S7): "one-shot `imsg sync --source studio-seed
--snapshot <path>` against a snapshot transferred from the Studio
(rsync over tailnet)". `run_sync(snapshot_override=...)` skips S1
entirely and feeds that already-transferred file straight to S2 — this
module does not implement the rsync transfer itself (an ops/manual
step, not a pipeline stage) or the "seed completeness check (AT-2)"
that SPEC §8 S7 says must run before the seed source is marked done —
AT-2 is an exact-GUID-set-diff acceptance test (SPEC/D6), which reads
as belonging with the eval/acceptance-test harness (SPEC §13) rather
than this pipeline stage; flagged as a gap in the build report rather
than silently skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import structlog

from imsg.config.schema import Config
from imsg.errors import ImsgError, SyncError
from imsg.mount.guard import guard_mount
from imsg.stages.extract import ExtractResult, RunImsgDumpFn, _default_run_imsg_dump, run_extract
from imsg.stages.identity import (
    ContactsImporterFn,
    IdentityResult,
    _default_contacts_importer,
    assert_invariant_or_raise,
    run_identity,
)
from imsg.stages.snapshot import SnapshotResult, run_snapshot

logger = structlog.get_logger(__name__)

SegmentFn = Callable[..., object]
"""S4's real entry point (`imsg.segment.pipeline.run_segment`, not this
build's scope) takes more than `(conn, config)` — bind the extra
arguments (`boundary_provider`, `boundary_prompt_bytes`) with
`functools.partial` before passing the result here. Widened from
`Callable[[psycopg.Connection, Config], object]` to `Callable[...,
object]` so `run_sync` can call it as `segment_fn(conn, config,
dry_run=dry_run)` (SPEC §8 dry-run) without narrowing every caller's
signature to a fixed keyword set."""

EmbedFn = Callable[..., object]
"""S6's entry point — same binding convention and same widening
rationale as `SegmentFn`."""

RunSnapshotFn = Callable[..., SnapshotResult]
RunExtractFn = Callable[..., ExtractResult]
RunIdentityFn = Callable[..., IdentityResult]
"""S1/S2/S3 injectable seams, mirroring every other stage module's
dependency-injection pattern in this codebase (`imsg.mount.guard`'s
`DiskutilInfoFn`, `imsg.stages.snapshot`'s `OpenSourceFn`, etc.) —
`run_sync`'s own unit tests exercise ordering/error-wrapping/the S3→S4
invariant gate against fakes here, independent of real SQLite/Postgres/
subprocess machinery; a smaller integration suite exercises the real
defaults end-to-end."""


@dataclass(frozen=True, slots=True)
class SyncResult:
    source_name: str
    snapshot: SnapshotResult | None
    """`None` only for a `snapshot_override` run (the studio-seed
    one-shot path), where S1 was skipped entirely because the snapshot
    was already transferred rather than backed up from a live source."""
    extract: ExtractResult | None
    """`None` only when a dry run stopped the chain right after S1 —
    see `note`."""
    identity: IdentityResult | None
    """`None` under the same condition as `extract`."""
    segment_ran: bool
    embed_ran: bool
    dry_run: bool = False
    note: str | None = None
    """Set when a dry run stopped the chain early (SPEC §8 S1/S7
    sequencing tradeoff: S1's dry-run mode never actually creates a
    snapshot file when none exists yet, so there is nothing on disk
    for S2 to preview reading; rather than fabricating a snapshot
    path, the chain stops here and reports why)."""


def _resolve_source_chat_db(config: Config, source_name: str) -> Path:
    for source in config.sync.sources:
        if source.name == source_name:
            return source.chat_db
    raise SyncError(
        f"source '{source_name}' not found in config.sync.sources "
        f"(configured: {[s.name for s in config.sync.sources]})"
    )


def run_sync(
    *,
    conn: psycopg.Connection,
    config: Config,
    source_name: str,
    imsg_dump_binary: Path,
    live_chat_db: Path | None = None,
    snapshot_override: Path | None = None,
    run_imsg_dump_fn: RunImsgDumpFn = _default_run_imsg_dump,
    contacts_importer: ContactsImporterFn = _default_contacts_importer,
    segment_fn: SegmentFn | None = None,
    embed_fn: EmbedFn | None = None,
    run_snapshot_fn: RunSnapshotFn = run_snapshot,
    run_extract_fn: RunExtractFn = run_extract,
    run_identity_fn: RunIdentityFn = run_identity,
    dry_run: bool = False,
) -> SyncResult:
    """S1 → S2 → S3 → S4 → S6 for one source (SPEC §8 S7).

    Runs the mount gate itself (unlike the individual S1/S2/S3 stage
    functions, which take that as a precondition their caller already
    satisfied) — SPEC §8 S7 names this the LaunchAgent-invoked service
    entry point, which is exactly the "service start" hard requirement
    2 (CLAUDE.md) requires the gate at.

    `snapshot_override` selects the studio-seed one-shot path (skips
    S1); otherwise `live_chat_db` is used directly, or resolved from
    `config.sync.sources` by `source_name` if not given.

    Raises `SyncError` naming which stage failed. S4/S6 are skipped
    (not errors) when their callables are `None` — see the module
    docstring's "S4/S6 are dependency-injected" note. The S3→S4
    invariant gate (hard requirement 3) is enforced here even when
    `segment_fn` is `None`: an unresolved identity state is still a
    real failure of this sync run, not silently swallowed just because
    nothing downstream was wired up to need it yet.

    `dry_run=True` (SPEC §8: "takes --dry-run where writes leave the
    machine") threads straight through to every sub-stage call — each
    of S1-S4/S6 already has its own accurate, self-contained dry-run
    mode. The one real sequencing wrinkle: S1's dry-run mode never
    actually creates `snapshots/snapshot.db` when none exists yet, so
    there is nothing on disk for S2 to preview reading. Rather than
    fabricating a fake path, a dry run with no pre-existing real
    snapshot file stops the chain right after S1 and returns a
    `SyncResult` with `extract`/`identity` as `None` and `note`
    explaining why. When a real snapshot *does* already exist on disk
    (from a prior real `imsg sync`/`imsg snapshot` run), S2's dry-run
    preview reads that existing file directly — reading an
    already-materialized file is not a new write, so this is safe even
    in dry-run mode. One further limitation worth flagging: because
    each sub-stage's dry-run wraps its own writes in a rolled-back
    transaction (S2/S3) or simply skips its write block (S4/S6), a
    later stage's dry-run preview in this same chain never sees an
    earlier stage's *hypothetical* writes — S3's preview, for example,
    is computed against whatever identity state is already committed
    in Postgres, not against S2's rolled-back preview extraction. Each
    stage's preview answers "what would this stage do against the
    database as it stands right now," not "what would the whole chain
    do if run for real starting from here."
    """
    guard_mount(config.paths.data_root)

    snapshot_result: SnapshotResult | None = None
    if snapshot_override is not None:
        snapshot_path = snapshot_override
        snapshot_sha256 = None
    else:
        resolved_chat_db = live_chat_db or _resolve_source_chat_db(config, source_name)
        try:
            snapshot_result = run_snapshot_fn(
                live_chat_db=resolved_chat_db, data_root=config.paths.data_root, dry_run=dry_run
            )
        except ImsgError as exc:
            raise SyncError(f"sync for source '{source_name}' failed at S1 snapshot: {exc}") from exc
        snapshot_path = snapshot_result.path
        snapshot_sha256 = snapshot_result.sha256

        if dry_run and not snapshot_path.is_file():
            logger.info(
                "sync.dry_run_stopped_after_snapshot",
                source=source_name,
                reason="no real snapshot file exists yet for S2 to preview reading",
            )
            return SyncResult(
                source_name=source_name,
                snapshot=snapshot_result,
                extract=None,
                identity=None,
                segment_ran=False,
                embed_ran=False,
                dry_run=True,
                note=(
                    "dry run stopped after S1 snapshot — no real snapshot file "
                    "exists yet on disk for S2 to preview reading; run a real "
                    "'imsg snapshot' or 'imsg sync' first, then dry-run again"
                ),
            )

    try:
        extract_result = run_extract_fn(
            conn=conn,
            source_name=source_name,
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha256,
            imsg_dump_binary=imsg_dump_binary,
            run_imsg_dump_fn=run_imsg_dump_fn,
            dry_run=dry_run,
        )
    except ImsgError as exc:
        raise SyncError(f"sync for source '{source_name}' failed at S2 extract: {exc}") from exc

    try:
        identity_result = run_identity_fn(
            conn=conn, config=config, contacts_importer=contacts_importer, dry_run=dry_run
        )
    except ImsgError as exc:
        raise SyncError(f"sync for source '{source_name}' failed at S3 identity: {exc}") from exc

    try:
        assert_invariant_or_raise(identity_result.invariant)
    except ImsgError as exc:
        raise SyncError(
            f"sync for source '{source_name}' stopped before S4: {exc}"
        ) from exc

    segment_ran = False
    if segment_fn is not None:
        try:
            segment_fn(conn, config, dry_run=dry_run)
            segment_ran = True
        except ImsgError as exc:
            raise SyncError(f"sync for source '{source_name}' failed at S4 segment: {exc}") from exc
    else:
        logger.info(
            "sync.segment_skipped", source=source_name, reason="no segment_fn supplied"
        )

    embed_ran = False
    if embed_fn is not None:
        try:
            embed_fn(conn, config, dry_run=dry_run)
            embed_ran = True
        except ImsgError as exc:
            raise SyncError(f"sync for source '{source_name}' failed at S6 embed: {exc}") from exc
    else:
        logger.info("sync.embed_skipped", source=source_name, reason="no embed_fn supplied")

    return SyncResult(
        source_name=source_name,
        snapshot=snapshot_result,
        extract=extract_result,
        identity=identity_result,
        segment_ran=segment_ran,
        embed_ran=embed_ran,
        dry_run=dry_run,
    )


def run_sync_all_sources(
    *,
    conn: psycopg.Connection,
    config: Config,
    imsg_dump_binary: Path,
    dry_run: bool = False,
    **kwargs: Any,
) -> list[SyncResult]:
    """SPEC §8 S7's regular (non-studio-seed) path: `imsg sync` with no
    `--source`/`--snapshot` override syncs every configured
    `sync.sources` entry in turn. `**kwargs` forwards straight to
    `run_sync` (e.g. `segment_fn`, `embed_fn`, `contacts_importer`) —
    kept generic here so this function does not need updating every
    time `run_sync`'s injectable seams change. `dry_run` is a named
    parameter (not left inside `**kwargs`) purely so its default is
    visible at this call site too.

    Stops at the first source that raises rather than continuing past
    a failure — a partial sync leaves each already-completed source's
    Postgres state exactly as that source's own commits left it (S2/S3
    each commit their own transactions), so re-running is safe and
    resumes from each source's own watermark; it does not silently
    skip a source that broke.
    """
    results: list[SyncResult] = []
    for source in config.sync.sources:
        results.append(
            run_sync(
                conn=conn,
                config=config,
                source_name=source.name,
                imsg_dump_binary=imsg_dump_binary,
                dry_run=dry_run,
                **kwargs,
            )
        )
    return results


__all__ = [
    "EmbedFn",
    "SegmentFn",
    "SyncResult",
    "run_sync",
    "run_sync_all_sources",
]
