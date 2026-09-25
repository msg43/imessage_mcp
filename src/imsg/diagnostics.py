"""Shared diagnostic checks behind `imsg check-permissions` and `imsg status`.

Every field SPEC §14 lists for `imsg status` is read from its source:
mount, Postgres reachability + cluster fingerprint, at-rest posture, Full
Disk Access, disk free space, buffer pool against the HNSW indexes,
enrichment yield state, the unclassified-thread count (§11.5), and — since
2026-09-24 — the pipeline fields: watermarks, last sync, enrichment queue
depths, FTS outbox lag, unresolved identities, attachment coverage, last
export and backup, and the 7-day audit-rejection count
(:func:`check_pipeline_status`, :func:`check_backups`). A `None` always
means "could not be read", with a reason saying why; nothing here raises.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import apsw
import psycopg

from imsg.backup.pipeline import backups_dir_for
from imsg.backup.retention import index_backups
from imsg.config.schema import Config
from imsg.db.connection import connect
from imsg.db.enrichment_yield_locks import YieldState, read_yield_state
from imsg.db.fingerprint import verify_data_directory
from imsg.db.prewarm import hnsw_index_bytes, shared_buffers_bytes
from imsg.embed.fts.schema import get_meta
from imsg.errors import ClusterFingerprintError, MountGateError, SecretResolutionError
from imsg.export.unclassified import unclassified_summary
from imsg.mount.guard import MountInfo, guard_mount, real_diskutil_info
from imsg.stages.identity import compute_invariant_report

GIB = float(2**30)


@dataclass(frozen=True, slots=True)
class MountCheck:
    ok: bool
    reason: str | None
    info: MountInfo | None


def check_mount(data_root: Path) -> MountCheck:
    try:
        info = guard_mount(data_root)
    except MountGateError as exc:
        return MountCheck(ok=False, reason=str(exc), info=None)
    return MountCheck(ok=True, reason=None, info=info)


@dataclass(frozen=True, slots=True)
class AtRestPosture:
    label: str
    """One of 'unattended', 'secure', 'mixed-or-unknown'."""

    boot_volume_encrypted: bool | None
    auto_login_enabled: bool | None
    data_volume_encrypted: bool | None
    caveat: str


_UNATTENDED_CAVEAT = (
    "unattended posture (SPEC §5.1, ratified D6): protects against bare-disk "
    "theft only — NOT whole-host theft, because auto-login unlocks the "
    "encrypted volume without a human present. Do not describe this as "
    "'at-rest protected' without this qualification."
)
_SECURE_CAVEAT = (
    "secure posture (SPEC §5.1): full at-rest protection, but the index is "
    "down after any reboot until an operator logs in."
)
_UNKNOWN_CAVEAT = (
    "does not match either documented posture (SPEC §5.1) — review manually."
)


def _fdesetup_status() -> bool | None:
    try:
        proc = subprocess.run(
            ["fdesetup", "status"], capture_output=True, text=True, check=False, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lowered = proc.stdout.lower()
    if "filevault is on" in lowered:
        return True
    if "filevault is off" in lowered:
        return False
    return None


def _auto_login_enabled() -> bool | None:
    try:
        proc = subprocess.run(
            [
                "defaults",
                "read",
                "/Library/Preferences/com.apple.loginwindow",
                "autoLoginUser",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        # Key absent (readable but unset) is the common "auto-login off" case;
        # a read failure for another reason is genuinely unknown, but we have
        # no reliable way to tell those apart without root, so this is a
        # best-effort signal, not a guarantee.
        return False
    return bool(proc.stdout.strip())


def check_at_rest_posture(data_root: Path) -> AtRestPosture:
    boot_encrypted = _fdesetup_status()
    auto_login = _auto_login_enabled()
    try:
        info = real_diskutil_info(data_root)
        data_encrypted: bool | None = info.encrypted
    except Exception:
        data_encrypted = None

    if boot_encrypted is False and auto_login is True:
        return AtRestPosture(
            label="unattended",
            boot_volume_encrypted=boot_encrypted,
            auto_login_enabled=auto_login,
            data_volume_encrypted=data_encrypted,
            caveat=_UNATTENDED_CAVEAT,
        )
    if boot_encrypted is True and auto_login is False:
        return AtRestPosture(
            label="secure",
            boot_volume_encrypted=boot_encrypted,
            auto_login_enabled=auto_login,
            data_volume_encrypted=data_encrypted,
            caveat=_SECURE_CAVEAT,
        )
    return AtRestPosture(
        label="mixed-or-unknown",
        boot_volume_encrypted=boot_encrypted,
        auto_login_enabled=auto_login,
        data_volume_encrypted=data_encrypted,
        caveat=_UNKNOWN_CAVEAT,
    )


def check_full_disk_access(live_chat_db: Path) -> bool | None:
    """Best-effort FDA probe: try to open+read a few bytes of the live chat.db.

    Returns `True`/`False` when determinable, `None` when the path is
    simply absent (can't distinguish "no FDA" from "no Messages set up
    yet"). Read-only, and reads at most 16 bytes — never anything that
    could be message content.
    """
    path = live_chat_db.expanduser()
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.read(16)
    except PermissionError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class PostgresCheck:
    reachable: bool
    cluster_fingerprint_ok: bool | None
    reason: str | None


def check_postgres(config: Config) -> PostgresCheck:
    try:
        conn = connect(config.database, autocommit=True)
    except SecretResolutionError as exc:
        return PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason=str(exc))
    except Exception as exc:
        return PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason=str(exc))

    try:
        try:
            verify_data_directory(conn, config.paths.data_root)
        except ClusterFingerprintError as exc:
            return PostgresCheck(reachable=True, cluster_fingerprint_ok=False, reason=str(exc))
        return PostgresCheck(reachable=True, cluster_fingerprint_ok=True, reason=None)
    finally:
        conn.close()


@dataclass(frozen=True, slots=True)
class BufferPoolCheck:
    """`shared_buffers` against the HNSW indexes it has to hold.

    Vector search is only fast while its index pages are cached (300-900
    ms cold against 7-24 ms warm, measured 2026-09-16/17), and a pool
    smaller than the indexes cannot hold them however often they are read,
    so this is a configuration mistake a status line should name rather
    than leave as unexplained latency. The HNSW total is the floor, not
    the target — the whole query path reads more (`imsg.db.prewarm`)."""

    shared_buffers_bytes: int | None
    hnsw_index_bytes: int | None
    warning: str | None

    @property
    def holds_hnsw_indexes(self) -> bool | None:
        if self.shared_buffers_bytes is None or self.hnsw_index_bytes is None:
            return None
        return self.shared_buffers_bytes >= self.hnsw_index_bytes


def check_buffer_pool(config: Config) -> BufferPoolCheck:
    """`shared_buffers` and the total HNSW index size, read from the live
    instance. An unreachable database is reported as `None`s with the
    reason, not as a failure — `check_postgres` already says it is down."""
    try:
        conn = connect(config.database, autocommit=True)
    except Exception as exc:
        return BufferPoolCheck(None, None, f"buffer pool not read: {exc}")
    try:
        pool = shared_buffers_bytes(conn)
        indexes = hnsw_index_bytes(conn)
    except Exception as exc:
        return BufferPoolCheck(None, None, f"buffer pool not read: {exc}")
    finally:
        conn.close()
    warning = None
    if indexes and pool < indexes:
        warning = (
            f"shared_buffers ({pool / GIB:.2f} GiB) is smaller than the HNSW indexes "
            f"({indexes / GIB:.2f} GiB): vector searches will read index pages from disk "
            f"(measured 300-900 ms cold against 7-24 ms warm). Raise shared_buffers in the "
            f"instance's postgresql.conf and restart."
        )
    return BufferPoolCheck(pool, indexes, warning)


def check_enrichment_yield(config: Config) -> YieldState:
    """Whether a query is in flight and whether an enrichment worker is
    currently standing aside for it (D10.3's ratified remedy;
    `imsg.db.enrichment_yield_locks`).

    Read from `pg_locks` on a connection of its own, taking no lock — a
    status command must be able to observe the contention it reports on
    without joining it. An unreachable database comes back as a `reason`,
    not an exception: `check_postgres` already says it is down."""
    try:
        conn = connect(config.database, autocommit=True)
    except Exception as exc:
        return YieldState(
            query_in_flight=False,
            enrichment_paused=False,
            reason=f"advisory locks not read: {type(exc).__name__}: {exc}",
        )
    try:
        return read_yield_state(conn)
    finally:
        conn.close()


@dataclass(frozen=True, slots=True)
class UnclassifiedCheck:
    """Active threads whose participants have never been classified for export.

    SPEC §11.5 puts this count in `imsg status` ("Surfaced in `imsg
    status`") and §14 lists it among the status fields, because a static
    allowlist rots silently: next month's contractor joins a group, the
    weekly report names them, and nothing in the operator's daily glance
    says so until someone reads the report.

    A previous pass declined to wire this on the grounds that `status`
    never opens a query connection and adding one would change its
    failure profile as a health check. That premise does not hold —
    `check_postgres`, `check_buffer_pool` and `check_enrichment_yield`
    each already open their own connection and run statements, and each
    already reports failure as a `reason` string rather than raising. The
    real hazard is different and specific: unlike those three, which read
    tiny catalogs, this query aggregates over `message`, so on a large
    corpus under load it could make a health check *slow* rather than
    wrong. That is what :data:`STATEMENT_TIMEOUT_MS` is for — a busy
    database yields `count=None` with a reason inside a bounded time,
    which is the same shape as "database absent" and is exactly how a
    health check should degrade.
    """

    count: int | None
    reason: str | None


STATEMENT_TIMEOUT_MS = 5_000
"""Ceiling on the unclassified-threads query. `imsg status` must answer
promptly whether or not the corpus is large and whether or not the
enrichment worker is hammering the instance; an unavailable number with
a reason is a better health check than a command that hangs."""


def check_unclassified_threads(config: Config) -> UnclassifiedCheck:
    """The SPEC §11.5 count, on a connection of its own, bounded by a timeout.

    Every failure mode — unreachable instance, unresolvable password,
    missing tables (pre-migration), statement timeout on a busy corpus —
    comes back as `count=None` plus a reason. Nothing here can raise, by
    the same rule `check_buffer_pool` and `check_enrichment_yield`
    follow: `check_postgres` is the field that says the database is down,
    and the other fields must not each restate it as a crash.
    """
    try:
        conn = connect(config.database, autocommit=True)
    except Exception as exc:
        return UnclassifiedCheck(None, f"unclassified thread count not read: {exc}")
    try:
        conn.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        return UnclassifiedCheck(unclassified_summary(conn), None)
    except Exception as exc:
        return UnclassifiedCheck(
            None, f"unclassified thread count not read: {type(exc).__name__}: {exc}"
        )
    finally:
        conn.close()


SYNC_STALE_AFTER_SECONDS = 3600
"""SPEC §14: `imsg status` flags a sync older than one hour. The
scheduled sync runs every 15 minutes, so an hour is four missed runs."""

BACKUP_STALE_AFTER_SECONDS = 26 * 3600
"""The nightly backup runs at 04:00; a newest complete set older than 26
hours means at least one night was missed."""

AUDIT_REJECTION_WINDOW_DAYS = 7
"""SPEC §14: "audit-rejection count (last 7 d)"."""


@dataclass(frozen=True, slots=True)
class PipelineStatus:
    """SPEC §14's pipeline fields, read from the tables that hold them.

    Each field is `None` only when it could not be read, and then
    `reasons[<field>]` says why — the same rule the other checks here
    follow: a health check reports "unknown, because…", it never raises
    and never guesses. Every query is bounded by
    :data:`STATEMENT_TIMEOUT_MS` and reads through indexes or small
    tables, so a busy corpus costs a reason, not a hang.

    - `watermarks_per_source`: S2's `sync_state['watermark.rowid.<source>']`
      — the highest `chat.db` ROWID extracted — and when it last moved.
    - `last_sync_at` / `last_sync_by_source`: the newest successful
      extraction (`extraction_run.status = 'ok'`), the step every sync
      completes first; `sync_stale` when that is older than
      :data:`SYNC_STALE_AFTER_SECONDS`.
    - `enrichment_queue_depths`: `enrichment` rows by state, `ready_now`
      (pending and due), and pending/failed by kind.
    - `fts_applied_event_id` / `fts_outbox_lag`: the FTS sidecar's
      applied watermark (its own `meta` table) and how many
      `search_index_event` rows are past it.
    - `unresolved_identity_count`: what S3's gate before segmentation
      counts (`imsg.stages.identity.compute_invariant_report`), detailed in
      `unresolved_identity_detail` with persons still awaiting review.
    - `attachment_materialization_coverage`: `attachment` rows by
      materialization state and the materialized fraction.
    - `last_export_at`: the newest `export_run` that finished `ok`, and the
      status of the newest run of any kind.
    - `audit_rejection_count_7d`: rejected requests in the last 7 days —
      rejected `mcp_audit` rows, plus the rejected request counts in
      `mcp_audit_rollup` when that table exists (it holds rejections the
      public server counts in memory and rows retention rolled up).
    """

    watermarks_per_source: dict[str, dict[str, object]] | None
    last_sync_at: str | None
    last_sync_by_source: dict[str, str] | None
    sync_stale: bool | None
    enrichment_queue_depths: dict[str, object] | None
    fts_applied_event_id: int | None
    fts_outbox_max_event_id: int | None
    fts_outbox_lag: int | None
    unresolved_identity_count: int | None
    unresolved_identity_detail: dict[str, int] | None
    attachment_materialization_coverage: dict[str, object] | None
    last_export_at: str | None
    last_export_run_status: str | None
    audit_rejection_count_7d: int | None
    reasons: dict[str, str]


PIPELINE_FIELDS: tuple[str, ...] = (
    "watermarks_per_source",
    "last_sync_at",
    "last_sync_by_source",
    "sync_stale",
    "enrichment_queue_depths",
    "fts_applied_event_id",
    "fts_outbox_max_event_id",
    "fts_outbox_lag",
    "unresolved_identity_count",
    "unresolved_identity_detail",
    "attachment_materialization_coverage",
    "last_export_at",
    "last_export_run_status",
    "audit_rejection_count_7d",
)


def _iso(value: object) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _read_watermarks(conn: psycopg.Connection) -> dict[str, dict[str, object]]:
    rows = conn.execute(
        "SELECT key, value, updated_at FROM sync_state "
        "WHERE key LIKE 'watermark.rowid.%' ORDER BY key"
    ).fetchall()
    out: dict[str, dict[str, object]] = {}
    for key, value, updated_at in rows:
        rowid: object = int(value) if str(value).isdigit() else value
        out[str(key).removeprefix("watermark.rowid.")] = {
            "rowid": rowid,
            "updated_at": _iso(updated_at),
        }
    return out


def _read_last_sync(
    conn: psycopg.Connection, now: datetime
) -> tuple[str | None, dict[str, str], bool | None]:
    rows = conn.execute(
        "SELECT source_name, max(finished_at) FROM extraction_run "
        "WHERE status = 'ok' AND finished_at IS NOT NULL GROUP BY source_name ORDER BY 1"
    ).fetchall()
    by_source = {str(src): _iso(ts) or "" for src, ts in rows if isinstance(ts, datetime)}
    newest = max((ts for _, ts in rows if isinstance(ts, datetime)), default=None)
    if newest is None:
        return None, by_source, None
    stale = (now - newest).total_seconds() > SYNC_STALE_AFTER_SECONDS
    return newest.isoformat(), by_source, stale


def _read_enrichment_queue(conn: psycopg.Connection) -> dict[str, object]:
    rows = conn.execute(
        "SELECT kind::text, state::text, count(*) FROM enrichment GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    totals: dict[str, int] = dict.fromkeys(
        ("pending", "running", "failed", "done", "skipped"), 0
    )
    by_kind: dict[str, dict[str, int]] = {}
    for kind, state, count in rows:
        totals[str(state)] = totals.get(str(state), 0) + int(count)
        if state in ("pending", "failed"):
            by_kind.setdefault(str(kind), {})[str(state)] = int(count)
    ready = conn.execute(
        "SELECT count(*) FROM enrichment WHERE state = 'pending' AND next_attempt_at <= now()"
    ).fetchone()
    return {**totals, "ready_now": int(ready[0]) if ready else 0, "by_kind": by_kind}


def fts_db_path(config: Config) -> Path:
    """The FTS sidecar (SPEC §5.3 `fts/fts.db`)."""
    return config.paths.data_root / "fts" / "fts.db"


def _read_fts_applied_event_id(config: Config) -> int:
    """The sidecar's own applied watermark. Opened without the create
    flag — a status command must never create `fts.db` — and not
    immutable, so a value still in the write-ahead log is read too."""
    path = fts_db_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist (the FTS sidecar has not been built)")
    conn = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READWRITE)
    try:
        conn.set_busy_timeout(STATEMENT_TIMEOUT_MS)
        value = get_meta(conn, "applied_event_id")
    finally:
        conn.close()
    if value is None or not value.strip().isdigit():
        raise ValueError(f"{path} has no numeric applied_event_id in its meta table")
    return int(value)


def _read_attachment_coverage(conn: psycopg.Connection) -> dict[str, object]:
    rows = conn.execute("SELECT state::text, count(*) FROM attachment GROUP BY 1").fetchall()
    counts = {str(state): int(n) for state, n in rows}
    total = sum(counts.values())
    materialized = counts.get("materialized", 0)
    return {
        **counts,
        "total": total,
        "materialized_fraction": round(materialized / total, 4) if total else None,
    }


def _read_last_export(conn: psycopg.Connection) -> tuple[str | None, str | None]:
    last_ok = conn.execute("SELECT max(finished_at) FROM export_run WHERE status = 'ok'").fetchone()
    newest = conn.execute(
        "SELECT status FROM export_run ORDER BY started_at DESC, export_run_id DESC LIMIT 1"
    ).fetchone()
    return (_iso(last_ok[0]) if last_ok else None, str(newest[0]) if newest else None)


def _read_audit_rejections(conn: psycopg.Connection) -> tuple[int, str | None]:
    detailed = conn.execute(
        "SELECT count(*) FROM mcp_audit WHERE NOT subject_ok "
        "AND ts >= now() - make_interval(days => %s)",
        (AUDIT_REJECTION_WINDOW_DAYS,),
    ).fetchone()
    total = int(detailed[0]) if detailed else 0
    exists = conn.execute("SELECT to_regclass('mcp_audit_rollup') IS NOT NULL").fetchone()
    if not (exists and exists[0]):
        return total, None
    try:
        rolled = conn.execute(
            "SELECT coalesce(sum(request_count), 0) FROM mcp_audit_rollup WHERE NOT subject_ok "
            "AND period_end > now() - make_interval(days => %s)",
            (AUDIT_REJECTION_WINDOW_DAYS,),
        ).fetchone()
    except psycopg.Error as exc:
        return total, (
            f"counted mcp_audit only; mcp_audit_rollup could not be read: "
            f"{type(exc).__name__}: {exc}"
        )
    return total + (int(rolled[0]) if rolled else 0), None


def check_pipeline_status(config: Config, *, now: datetime | None = None) -> PipelineStatus:
    """Read every SPEC §14 pipeline field on one connection of its own.

    Never raises. An unreachable database sets every field to `None` with
    the same reason; otherwise each field is read on its own, so one
    failing query (a table missing before a migration, a statement
    timeout) costs that field alone."""
    moment = now or datetime.now(UTC)
    values: dict[str, object] = dict.fromkeys(PIPELINE_FIELDS)
    reasons: dict[str, str] = {}

    def fail(fields: tuple[str, ...], exc: BaseException) -> None:
        for name in fields:
            values[name] = None
            reasons[name] = f"{type(exc).__name__}: {exc}"

    try:
        conn = connect(config.database, autocommit=True)
    except Exception as exc:
        fail(PIPELINE_FIELDS, exc)
        return PipelineStatus(**values, reasons=reasons)  # type: ignore[arg-type]
    try:
        try:
            conn.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        except Exception as exc:
            fail(PIPELINE_FIELDS, exc)
            return PipelineStatus(**values, reasons=reasons)  # type: ignore[arg-type]

        try:
            values["watermarks_per_source"] = _read_watermarks(conn)
        except Exception as exc:
            fail(("watermarks_per_source",), exc)
        try:
            last, by_source, stale = _read_last_sync(conn, moment)
            values.update(last_sync_at=last, last_sync_by_source=by_source, sync_stale=stale)
            if last is None:
                reasons["last_sync_at"] = "no successful extraction_run recorded yet"
                reasons["sync_stale"] = "no successful extraction_run recorded yet"
        except Exception as exc:
            fail(("last_sync_at", "last_sync_by_source", "sync_stale"), exc)
        try:
            values["enrichment_queue_depths"] = _read_enrichment_queue(conn)
        except Exception as exc:
            fail(("enrichment_queue_depths",), exc)
        try:
            max_row = conn.execute(
                "SELECT coalesce(max(event_id), 0) FROM search_index_event"
            ).fetchone()
            values["fts_outbox_max_event_id"] = int(max_row[0]) if max_row else 0
        except Exception as exc:
            fail(("fts_outbox_max_event_id",), exc)
        try:
            applied = _read_fts_applied_event_id(config)
            values["fts_applied_event_id"] = applied
            lag_row = conn.execute(
                "SELECT count(*) FROM search_index_event WHERE event_id > %s", (applied,)
            ).fetchone()
            values["fts_outbox_lag"] = int(lag_row[0]) if lag_row else 0
        except Exception as exc:
            fail(("fts_applied_event_id", "fts_outbox_lag"), exc)
        try:
            report = compute_invariant_report(conn)
            review = conn.execute("SELECT count(*) FROM person WHERE needs_review").fetchone()
            values["unresolved_identity_count"] = (
                report.unresolved_message_senders
                + report.unresolved_tapback_senders
                + report.unresolved_chat_participants
            )
            values["unresolved_identity_detail"] = {
                "message_senders": report.unresolved_message_senders,
                "tapback_senders": report.unresolved_tapback_senders,
                "chat_participants": report.unresolved_chat_participants,
                "owner_persons": report.owner_person_count,
                "persons_needing_review": int(review[0]) if review else 0,
            }
        except Exception as exc:
            fail(("unresolved_identity_count", "unresolved_identity_detail"), exc)
        try:
            values["attachment_materialization_coverage"] = _read_attachment_coverage(conn)
        except Exception as exc:
            fail(("attachment_materialization_coverage",), exc)
        try:
            last_export, newest_status = _read_last_export(conn)
            values.update(last_export_at=last_export, last_export_run_status=newest_status)
            if last_export is None:
                reasons["last_export_at"] = "no export_run has finished ok"
        except Exception as exc:
            fail(("last_export_at", "last_export_run_status"), exc)
        try:
            count, note = _read_audit_rejections(conn)
            values["audit_rejection_count_7d"] = count
            if note is not None:
                reasons["audit_rejection_count_7d"] = note
        except Exception as exc:
            fail(("audit_rejection_count_7d",), exc)
    finally:
        conn.close()
    return PipelineStatus(**values, reasons=reasons)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BackupStatus:
    """The newest complete set under `backups/`, judged by the same rule
    retention uses (`imsg.backup.retention.index_backups`: a correctly
    named directory whose `MANIFEST.json` says it is complete)."""

    last_backup_at: str | None
    backup_stale: bool | None
    complete_sets: int
    partial_sets: int
    reason: str | None


def check_backups(config: Config, *, now: datetime | None = None) -> BackupStatus:
    moment = now or datetime.now(UTC)
    try:
        index = index_backups(backups_dir_for(config))
    except Exception as exc:
        return BackupStatus(None, None, 0, 0, f"backups/ not read: {type(exc).__name__}: {exc}")
    if not index.complete:
        return BackupStatus(
            None, None, 0, len(index.partial), "no complete backup set under backups/"
        )
    newest = index.complete[0].created_at
    try:
        created = datetime.fromisoformat(newest)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        stale: bool | None = (moment - created).total_seconds() > BACKUP_STALE_AFTER_SECONDS
        reason = None
    except ValueError:
        stale, reason = None, f"newest set's created_at {newest!r} is not ISO-8601"
    return BackupStatus(newest, stale, len(index.complete), len(index.partial), reason)


def disk_free_bytes(path: Path) -> int | None:
    """`shutil.disk_usage` on the nearest existing ancestor of `path`."""
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    try:
        return shutil.disk_usage(candidate).free
    except OSError:
        return None


__all__ = [
    "AUDIT_REJECTION_WINDOW_DAYS",
    "BACKUP_STALE_AFTER_SECONDS",
    "GIB",
    "PIPELINE_FIELDS",
    "STATEMENT_TIMEOUT_MS",
    "SYNC_STALE_AFTER_SECONDS",
    "AtRestPosture",
    "BackupStatus",
    "BufferPoolCheck",
    "MountCheck",
    "PipelineStatus",
    "PostgresCheck",
    "UnclassifiedCheck",
    "check_at_rest_posture",
    "check_backups",
    "check_buffer_pool",
    "check_enrichment_yield",
    "check_full_disk_access",
    "check_mount",
    "check_pipeline_status",
    "check_postgres",
    "check_unclassified_threads",
    "disk_free_bytes",
    "fts_db_path",
]
