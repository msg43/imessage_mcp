"""Pushing attachment copies from a host the index host cannot reach
(owner decision D13's fetcher).

The index host (where Postgres and the cache live) cannot open a
connection to the other Mac, but that Mac can reach it over SSH. So the
other Mac runs `imsg push-attachments`, which:

1. asks the index host for a **plan** over SSH
   (`imsg push-attachments-plan`, `build_push_plan` here): for each
   attachment still not materialized, its open candidate copies at the
   locations this Mac serves (its own Messages folder, its attached drives),
   best first;
2. checks each candidate on its own disks, read-only (`check_candidate`):
   below the root it belongs to, a regular file, the size the listing said;
3. copies the first good candidate of each attachment into the index
   host's staging directory with one plain rsync per location
   (`imsg.backfill.transfer.push_command`: this Mac is the sender, so its
   files are only read);
4. sends back what happened to each candidate
   (`imsg push-attachments-record`, `record_push_results`).

The backfill on the index host then verifies each staged copy against the
size and hash its location reported, and materializes it
(`imsg.backfill.fetch`, the staged tier).

**Paths stay on encrypted volumes.** The plan travels over SSH into the
pushing process's memory, and the file list reaches rsync on stdin; neither
is written to the pushing Mac's disk. The pushing command prints counts
only. rsync's stderr names files, so it is sent to the index host with the
results and written to a log under `paths.data_root` there.
"""

from __future__ import annotations

import json
import os
import stat
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imsg.backfill.fetch import LocationAccess, Tier
from imsg.backfill.locations import (
    CandidateLocation,
    LocationOutcome,
    MatchQuality,
    is_sha256_hex,
    is_valid_location_code,
    load_open_candidates,
    safe_relative_path,
)
from imsg.backfill.materialize import cache_path_for
from imsg.backfill.transfer import CopyRunner, push_command
from imsg.errors import AttachmentBackfillError
from imsg.paths import is_contained_in, resolve_path

if TYPE_CHECKING:
    import psycopg

PLAN_FORMAT = "imsg-attachment-push-plan/1"
RESULTS_FORMAT = "imsg-attachment-push-results/1"

PUSH_OUTCOMES: frozenset[LocationOutcome] = frozenset(
    {
        LocationOutcome.STAGED,
        LocationOutcome.ABSENT,
        LocationOutcome.REJECTED,
        LocationOutcome.REFUSED,
        LocationOutcome.UNREACHABLE,
    }
)
"""What a pushing host may report. `fetched` is only ever decided by the
index host, after it has verified the staged copy."""

_MESSAGES_DIR = Path("~/Library/Messages").expanduser()


class PushError(AttachmentBackfillError):
    """A plan or result that cannot be trusted as written, or a root the
    pushing host must not read."""


@dataclass(frozen=True, slots=True)
class PushCandidate:
    location_id: int
    location: str
    rel: str
    """Below the location's root on the pushing host."""
    match: MatchQuality
    byte_size: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class PushPlanItem:
    attachment_id: int
    candidates: tuple[PushCandidate, ...]


@dataclass(frozen=True, slots=True)
class PushPlan:
    staging_dir: str
    """Absolute, on the index host; one directory per location below it."""
    items: tuple[PushPlanItem, ...]


# --------------------------------------------------------------------------
# index host: the plan and the results
# --------------------------------------------------------------------------


def build_push_plan(
    conn: psycopg.Connection,
    *,
    locations: Sequence[str],
    access: LocationAccess,
    data_root: Path,
) -> PushPlan:
    """Read-only. For each attachment still not materialized, its open
    candidates at `locations` (the pushing host's preference order), best
    first: identity matches before `name_size`, then the pusher's order,
    then match rank. An attachment is left out when a copy of it is
    already waiting in staging, or when the cache already holds the
    content one of its copies was listed with — the backfill needs no
    push for those."""
    wanted = [code for code in locations if is_valid_location_code(code)]
    order = {code: i for i, code in enumerate(wanted)}
    items: list[PushPlanItem] = []
    for attachment_id, rows in sorted(load_open_candidates(conn).items()):
        if _satisfiable_here(rows, access, data_root):
            continue
        candidates = [
            row for row in rows
            if row.location in order and row.rel is not None
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda r: (r.match.flagged, order[r.location], r.match.rank, r.location_id)
        )
        items.append(
            PushPlanItem(
                attachment_id=attachment_id,
                candidates=tuple(
                    PushCandidate(
                        location_id=r.location_id,
                        location=r.location,
                        rel=r.rel or "",
                        match=r.match,
                        byte_size=r.byte_size,
                        sha256=r.sha256,
                    )
                    for r in candidates
                ),
            )
        )
    return PushPlan(staging_dir=str(resolve_path(access.staging_root)), items=tuple(items))


def _satisfiable_here(
    rows: list[CandidateLocation], access: LocationAccess, data_root: Path
) -> bool:
    for row in rows:
        if row.sha256 is not None and cache_path_for(data_root, row.sha256).is_file():
            return True
        if row.location != access.local_location and access.tier_of(row) is Tier.STAGED:
            return True
    return False


def plan_to_lines(plan: PushPlan) -> Iterable[str]:
    yield json.dumps(
        {"format": PLAN_FORMAT, "staging_dir": plan.staging_dir, "items": len(plan.items)}
    )
    for item in plan.items:
        yield json.dumps(
            {
                "attachment_id": item.attachment_id,
                "candidates": [
                    {
                        "location_id": c.location_id,
                        "location": c.location,
                        "rel": c.rel,
                        "match": c.match.value,
                        "byte_size": c.byte_size,
                        "sha256": c.sha256,
                    }
                    for c in item.candidates
                ],
            },
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class PushResult:
    location_id: int
    outcome: LocationOutcome
    error: str | None = None


def results_to_lines(results: Sequence[PushResult], rsync_log: Sequence[str]) -> Iterable[str]:
    yield json.dumps({"format": RESULTS_FORMAT, "results": len(results)})
    for result in results:
        yield json.dumps(
            {"location_id": result.location_id, "outcome": result.outcome.value,
             "error": result.error},
            ensure_ascii=False,
        )
    for entry in rsync_log:
        yield json.dumps({"rsync_log": entry}, ensure_ascii=False)


def parse_results(lines: Iterable[str]) -> tuple[list[PushResult], list[str]]:
    results: list[PushResult] = []
    rsync_log: list[str] = []
    header_seen = False
    for raw in lines:
        if not raw.strip():
            continue
        doc = json.loads(raw)
        if not header_seen:
            if doc.get("format") != RESULTS_FORMAT:
                raise PushError(f"not a push result stream (format {doc.get('format')!r})")
            header_seen = True
            continue
        if "rsync_log" in doc:
            rsync_log.append(str(doc["rsync_log"]))
            continue
        try:
            outcome = LocationOutcome(doc["outcome"])
        except (KeyError, ValueError) as exc:
            raise PushError(f"push result with an unknown outcome: {doc.get('outcome')!r}") from exc
        if outcome not in PUSH_OUTCOMES:
            raise PushError(f"a pushing host may not report {outcome.value!r}")
        location_id = doc.get("location_id")
        if not isinstance(location_id, int) or isinstance(location_id, bool):
            raise PushError("push result without an integer location_id")
        error = doc.get("error")
        results.append(PushResult(location_id, outcome, None if error is None else str(error)))
    if not header_seen:
        raise PushError("empty push result stream")
    return results, rsync_log


def record_push_results(
    conn: psycopg.Connection,
    results: Sequence[PushResult],
    *,
    rsync_log: Sequence[str] = (),
    log_dir: Path | None = None,
) -> Counter[str]:
    """Write what the pushing host saw to the location rows it planned.
    Touches only `attachment_location`'s attempt columns, and never a row
    already fetched. rsync's own messages go to a log file under
    `log_dir`. Returns outcome -> rows updated."""
    counts: Counter[str] = Counter()
    with conn.transaction(), conn.cursor() as cur:
        for result in results:
            cur.execute(
                "UPDATE attachment_location SET last_tried_at = now(), last_outcome = %s, "
                "last_error = %s WHERE location_id = %s AND fetched_at IS NULL",
                (result.outcome.value,
                 None if result.error is None else result.error[:2000],
                 result.location_id),
            )
            if cur.rowcount:
                counts[result.outcome.value] += 1
    conn.commit()
    if log_dir is not None and any(entry.strip() for entry in rsync_log):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"attachment-push-{stamp}.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(rsync_log) + "\n")
        os.chmod(log_path, 0o600)
    return counts


# --------------------------------------------------------------------------
# pushing host
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PushRoot:
    """A location this host serves, and the directory its paths are below."""

    location: str
    root: Path


def parse_root(spec: str) -> PushRoot:
    """`CODE=PATH`. A root may not be the Messages folder itself or any
    folder above it: only its `Attachments` folder, or somewhere outside it
    entirely (so no plan can name `chat.db`)."""
    code, sep, raw = spec.partition("=")
    if not sep or not raw or not is_valid_location_code(code):
        raise PushError(f"--root must be CODE=PATH with a plain code, got {spec!r}")
    if not Path(raw).expanduser().is_absolute():
        raise PushError(f"--root path must be absolute: {raw!r}")
    root = resolve_path(raw)
    messages = resolve_path(_MESSAGES_DIR)
    if is_contained_in(messages, root) or (
        is_contained_in(root, messages) and not is_contained_in(root, messages / "Attachments")
    ):
        raise PushError(
            f"--root {code} may not be {raw!r}: below ~/Library/Messages only the Attachments "
            f"folder may be read, never chat.db"
        )
    return PushRoot(location=code, root=root)


def parse_plan(lines: Iterable[str]) -> PushPlan:
    staging_dir: str | None = None
    items: list[PushPlanItem] = []
    for raw in lines:
        if not raw.strip():
            continue
        doc: dict[str, Any] = json.loads(raw)
        if staging_dir is None:
            if doc.get("format") != PLAN_FORMAT:
                raise PushError(f"not a push plan (format {doc.get('format')!r})")
            staging_dir = str(doc.get("staging_dir") or "")
            if not staging_dir.startswith("/"):
                raise PushError("push plan without an absolute staging_dir")
            continue
        candidates: list[PushCandidate] = []
        for c in doc.get("candidates", []):
            location = str(c["location"])
            rel = str(c["rel"])
            sha = c.get("sha256")
            size = c.get("byte_size")
            if not is_valid_location_code(location) or not safe_relative_path(rel):
                raise PushError("push plan with an unsafe location code or path")
            candidates.append(
                PushCandidate(
                    location_id=int(c["location_id"]),
                    location=location,
                    rel=rel,
                    match=MatchQuality(c["match"]),
                    byte_size=None if size is None else int(size),
                    sha256=sha if is_sha256_hex(sha) else None,
                )
            )
        items.append(PushPlanItem(int(doc["attachment_id"]), tuple(candidates)))
    if staging_dir is None:
        raise PushError("empty push plan")
    return PushPlan(staging_dir=staging_dir, items=tuple(items))


def check_candidate(root: Path, candidate: PushCandidate) -> tuple[LocationOutcome | None, str]:
    """Read-only check of one candidate on this host: `(None, "")` when it
    may be sent."""
    path = root / candidate.rel
    try:
        st = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return LocationOutcome.ABSENT, "absent on the pushing host"
    except OSError as exc:
        return LocationOutcome.UNREACHABLE, f"could not stat on the pushing host: {exc.strerror}"
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return LocationOutcome.REFUSED, "not a regular file on the pushing host"
    if not is_contained_in(path, root):
        return LocationOutcome.REFUSED, "resolves outside its root on the pushing host"
    if candidate.byte_size is not None and st.st_size != candidate.byte_size:
        return (
            LocationOutcome.REJECTED,
            f"size {st.st_size} on the pushing host != listed {candidate.byte_size}",
        )
    return None, ""


@dataclass
class PushReport:
    items: int = 0
    candidates_checked: int = 0
    selected: Counter[str] = field(default_factory=Counter)
    """location -> files sent (or, in a dry run, that would be)."""
    not_sent: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(location, outcome) -> candidates this host could not send."""
    no_root_here: int = 0
    """Candidates at locations this host was not given a root for."""
    unserved: int = 0
    """Attachments none of whose candidates this host could send."""
    copy_exit_codes: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False


def run_push(
    plan: PushPlan,
    roots: Sequence[PushRoot],
    *,
    ssh_host: str,
    ssh: str,
    rsync: str,
    runner: CopyRunner,
    dry_run: bool = False,
) -> tuple[PushReport, list[PushResult], list[str]]:
    """Check every planned candidate here and push the first good one of
    each attachment. Returns the report, one result per candidate this host
    decided something about, and rsync's stderr (for the index host's log;
    never printed here)."""
    report = PushReport(items=len(plan.items), dry_run=dry_run)
    by_code = {r.location: r for r in roots}
    mounted = {code: r.root.is_dir() for code, r in by_code.items()}
    results: list[PushResult] = []
    picks: dict[str, list[PushCandidate]] = {}
    for item in plan.items:
        chosen = False
        for candidate in item.candidates:
            root = by_code.get(candidate.location)
            if root is None:
                report.no_root_here += 1
                continue
            report.candidates_checked += 1
            if not mounted[candidate.location]:
                outcome, detail = LocationOutcome.UNREACHABLE, "root not mounted on the pushing host"
            else:
                problem, detail = check_candidate(root.root, candidate)
                if problem is None:
                    picks.setdefault(candidate.location, []).append(candidate)
                    chosen = True
                    break
                outcome = problem
            report.not_sent[(candidate.location, outcome.value)] += 1
            results.append(PushResult(candidate.location_id, outcome, detail))
        if not chosen:
            report.unserved += 1

    rsync_log: list[str] = []
    for code in [r.location for r in roots]:
        chosen_here = picks.get(code)
        if not chosen_here:
            continue
        rels = sorted({c.rel for c in chosen_here})
        command = push_command(
            rsync=rsync,
            ssh=ssh,
            local_root=by_code[code].root,
            ssh_host=ssh_host,
            remote_staging_dir=f"{plan.staging_dir.rstrip('/')}/{code}",
            relpaths=rels,
            dry_run=dry_run,
        )
        copied = runner(command)
        report.copy_exit_codes[code] = copied.returncode
        if copied.stderr.strip():
            rsync_log.append(f"push {code}: exit {copied.returncode}\n{copied.stderr}")
        report.selected[code] += len(chosen_here)
        staged = LocationOutcome.STAGED if copied.reached else LocationOutcome.UNREACHABLE
        why = None if copied.reached else f"push to the index host exited {copied.returncode}"
        for candidate in chosen_here:
            results.append(PushResult(candidate.location_id, staged, why))
    return report, results, rsync_log


__all__ = [
    "PLAN_FORMAT",
    "PUSH_OUTCOMES",
    "RESULTS_FORMAT",
    "PushCandidate",
    "PushError",
    "PushPlan",
    "PushPlanItem",
    "PushReport",
    "PushResult",
    "PushRoot",
    "build_push_plan",
    "check_candidate",
    "parse_plan",
    "parse_results",
    "parse_root",
    "plan_to_lines",
    "record_push_results",
    "results_to_lines",
    "run_push",
]
