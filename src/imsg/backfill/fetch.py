"""The location phase of S5a (owner decision D13): fetch an attachment from
any place a copy of it is known to be, not only the one path this host's
chat.db recorded.

`imsg.backfill.pipeline.run_backfill` first tries each attachment's own
`source_path` on this host, exactly as before. This phase then takes every
attachment that is still not materialized — in any state, `missing` and
`unsupported` included — and has an open row in `attachment_location`
(`imsg.backfill.locations`), and tries its copies **best first**:

1. **cache** — the location's listing gave the file's hash, and the cache
   already holds that content (another attachment row had the same bytes).
   Nothing is fetched; the cached file is re-hashed first.
2. **local** — this host's own Messages attachments folder.
3. **staged** — a copy another host pushed into this host's staging
   directory (`imsg push-attachments`, run on the host that has the files:
   the Studio and its attached drives). Only a staged file is read; this
   host never reaches the other host.
4. **pull** — a location this host can reach over SSH itself (a NAS
   share), copied read-only into staging, then verified like a staged copy.
5. **awaiting push / no access** — a location nobody has copied yet, or a
   drive attached to no host: nothing to try here, and the report says
   which location would supply the copy.

Within that order, copies matched on a recorded path or a GUID folder
(identity matches, measured right every time) come before copies matched
on name and size (`name_size`, flagged: 99.5% right). While an identity
match waits on a push, a flagged copy of the same attachment is held back,
so a certain copy is not displaced by a probable one just because the push
has not run yet.

**Verification.** Every copy is checked against what its location said:
the byte size when known, and the sha256 when known (a listing of another
Mac's folder carries hashes; drive catalogs carry sizes only). A copy that
differs is rejected before it reaches the cache
(`imsg.backfill.materialize.ContentMismatchError`). Paths are read only
below the root they belong to: this host's attachments folder, or the
location's own staging directory, with symlinks and non-regular files
refused.

**Sources are read-only.** This phase reads local files, reads staged
copies, and runs `imsg.backfill.transfer.pull_command`, whose remote side is
rsync's sender. The only files it ever deletes are its own staged copies,
once their content is in the cache and no other open candidate row points
at them.
"""

from __future__ import annotations

import errno
import os
import stat
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.backfill.classify import classify_os_error
from imsg.backfill.locations import (
    CandidateLocation,
    LocationOutcome,
    MatchQuality,
    fetch_relative_path,
    is_valid_location_code,
    load_open_candidates,
    mark_location_fetched,
    other_open_reference_exists,
    record_attempt,
)
from imsg.backfill.materialize import (
    ContentMismatchError,
    cache_path_for,
    materialize_attachment,
    materialize_from_cache,
)
from imsg.backfill.transfer import CopyRunner, pull_command, run_copy
from imsg.background_gate import StopCheck, StopReason
from imsg.enrich.planner import enqueue_for_materialized
from imsg.paths import is_contained_in, is_same_file, join_under_root, resolve_path

if TYPE_CHECKING:
    import psycopg

    from imsg.backfill.throttle import RateThrottle
    from imsg.config.schema import Config


class Tier(StrEnum):
    """Where one attempt reads from, in the order they are tried."""

    CACHE = "cache"
    LOCAL = "local"
    STAGED = "staged"
    AWAITING_PUSH = "awaiting-push"
    PULL = "pull"
    NO_ACCESS = "no-access"


_TIER_ORDER: dict[Tier, int] = {
    Tier.CACHE: 0,
    Tier.LOCAL: 1,
    Tier.STAGED: 2,
    Tier.AWAITING_PUSH: 2,
    Tier.PULL: 3,
    Tier.NO_ACCESS: 4,
}

_ACTIONABLE = frozenset({Tier.CACHE, Tier.LOCAL, Tier.STAGED, Tier.PULL})


@dataclass(frozen=True, slots=True)
class PullLocation:
    """A location this host copies from itself, over SSH, read-only."""

    location: str
    ssh_host: str
    root: str
    """Absolute path on `ssh_host` that the location's paths are relative to."""


@dataclass(frozen=True, slots=True)
class LocationAccess:
    """How each location can be read on this host."""

    local_location: str
    """The code this host's own Messages folder is filed under: the name of
    the sync source that is this host's live chat.db."""
    attachments_root: Path
    """This host's Messages attachments folder."""
    staging_root: Path
    """Where pushed and pulled copies land: `<data_root>/<staging_dir>`,
    one directory per location code."""
    push_locations: tuple[str, ...] = ()
    """Locations another host pushes into staging (`imsg push-attachments`)."""
    pulls: tuple[PullLocation, ...] = ()

    def pull_for(self, location: str) -> PullLocation | None:
        for pull in self.pulls:
            if pull.location == location:
                return pull
        return None

    def staged_path(self, location: str, rel: str) -> Path | None:
        if not is_valid_location_code(location):
            return None
        return self.staging_root / location / rel

    def tier_of(self, row: CandidateLocation) -> Tier:
        """The location tier of a row, ignoring the cache shortcut."""
        if row.location == self.local_location:
            return Tier.LOCAL
        rel = row.rel
        staged = None if rel is None else self.staged_path(row.location, rel)
        if staged is not None and _is_regular_file(staged):
            return Tier.STAGED
        if self.pull_for(row.location) is not None:
            return Tier.PULL
        if row.location in self.push_locations:
            return Tier.AWAITING_PUSH
        return Tier.NO_ACCESS


@dataclass(frozen=True, slots=True)
class LocationFetchSettings:
    access: LocationAccess
    ssh_command: str = "ssh -o BatchMode=yes"
    rsync_command: str = "rsync"
    run_pulls: bool = True
    copy_runner: CopyRunner = run_copy
    log_dir: Path | None = None
    """Where rsync's stderr is written (it names files, so it is never
    printed). `None` discards it."""


@dataclass
class LocationFetchReport:
    attachments_with_candidates: int = 0
    attempted: int = 0
    """Attachments at least one copy was actually tried for."""
    materialized: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    """(tier, location, match quality) -> attachments materialized."""
    rejected: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(location, what differed: size or sha256) -> copies refused."""
    absent: Counter[str] = field(default_factory=Counter)
    refused: Counter[str] = field(default_factory=Counter)
    unreachable: Counter[str] = field(default_factory=Counter)
    awaiting_push: Counter[str] = field(default_factory=Counter)
    """location -> attachments still not materialized whose next copy is
    one another host has not pushed yet."""
    no_access: Counter[str] = field(default_factory=Counter)
    """location -> attachments still not materialized whose only remaining
    copies are somewhere no host can read (a drive attached nowhere)."""
    deferred_flagged: int = 0
    pull_skipped: Counter[str] = field(default_factory=Counter)
    """location -> attachments whose next copy was a pull, not run
    because pulls were switched off (`--no-pull`)."""
    pulled: Counter[str] = field(default_factory=Counter)
    pull_runs_failed: Counter[str] = field(default_factory=Counter)
    would_try: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    """Dry run only: the first copy each attachment would try."""
    budget_capped: bool = False
    halted_low_disk_space: bool = False
    stopped: StopReason | None = None
    """Set when the phase stopped because heavy background work was paused
    or the host was short of memory (`imsg.background_gate`)."""
    notes: list[str] = field(default_factory=list)
    dry_run: bool = False
    enrichment_enqueued: int = 0
    """S5b tasks queued for the attachments this phase materialized
    (`imsg.enrich.planner.enqueue_for_materialized`)."""
    enrichment_unroutable: int = 0
    enrichment_plan_errors: int = 0

    @property
    def materialized_total(self) -> int:
        return sum(self.materialized.values())

    @property
    def materialized_flagged(self) -> int:
        return sum(n for (_, _, match), n in self.materialized.items()
                   if match == MatchQuality.NAME_SIZE.value)


@dataclass(frozen=True, slots=True)
class _Attempt:
    tier: Tier
    row: CandidateLocation

    def sort_key(self) -> tuple[bool, int, int, int]:
        return (self.row.match.flagged, _TIER_ORDER[self.tier], self.row.match.rank,
                self.row.location_id)


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _ordered_attempts(
    rows: list[CandidateLocation], access: LocationAccess, data_root: Path
) -> tuple[list[_Attempt], list[CandidateLocation], bool]:
    """(attempts best first, rows no root can read, whether flagged copies
    were held back for a pending push). Pure: reads only `stat`."""
    attempts: list[_Attempt] = []
    unreadable: list[CandidateLocation] = []
    for row in rows:
        if row.rel is None:
            unreadable.append(row)
            continue
        if row.sha256 is not None and _is_regular_file(cache_path_for(data_root, row.sha256)):
            attempts.append(_Attempt(Tier.CACHE, row))
        attempts.append(_Attempt(access.tier_of(row), row))
    attempts.sort(key=_Attempt.sort_key)
    deferred = any(
        a.tier is Tier.AWAITING_PUSH and not a.row.match.flagged for a in attempts
    ) and any(a.row.match.flagged for a in attempts)
    if deferred:
        attempts = [a for a in attempts if not a.row.match.flagged]
    return attempts, unreadable, deferred


def best_copy(
    rows: list[CandidateLocation], access: LocationAccess, data_root: Path
) -> tuple[Tier, CandidateLocation] | None:
    """The copy the location phase would try first for one attachment: the
    first it can act on, else the first waiting on a push, else the first
    on a location no host reads. `None` when nothing is open."""
    attempts, _, _ = _ordered_attempts(rows, access, data_root)
    for wanted in (_ACTIONABLE, {Tier.AWAITING_PUSH}, {Tier.NO_ACCESS}):
        for attempt in attempts:
            if attempt.tier in wanted:
                return attempt.tier, attempt.row
    return None


def _file_problem(path: Path, root: Path) -> tuple[LocationOutcome | None, int]:
    """(problem, size): `None` when `path` is a regular file that really
    resolves below `root`."""
    try:
        st = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return LocationOutcome.ABSENT, 0
    except OSError:
        return LocationOutcome.UNREACHABLE, 0
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return LocationOutcome.REFUSED, 0
    if not is_contained_in(path, root):
        return LocationOutcome.REFUSED, 0
    return None, st.st_size


DiskFreeFn = Callable[[Path], int]


class _LocationPhase:
    def __init__(
        self,
        conn: psycopg.Connection,
        data_root: Path,
        settings: LocationFetchSettings,
        *,
        throttle: RateThrottle | None,
        dry_run: bool,
        budget: int | None,
        disk_free_fn: DiskFreeFn | None,
        min_free_bytes: int,
        free_space_check_interval: int,
        read_this_run: dict[int, str],
        stop_check: StopCheck | None = None,
    ) -> None:
        self.conn = conn
        self.read_this_run = read_this_run
        self.stop_check = stop_check
        self.data_root = data_root
        self.settings = settings
        self.access = settings.access
        self.throttle = throttle
        self.dry_run = dry_run
        self.budget = budget
        self.disk_free_fn = disk_free_fn
        self.min_free_bytes = min_free_bytes
        self.free_space_check_interval = max(1, free_space_check_interval)
        self.report = LocationFetchReport(dry_run=dry_run)
        self._tries = 0

    # -- bookkeeping ------------------------------------------------------

    def _record(self, row: CandidateLocation, outcome: LocationOutcome, error: str) -> None:
        if outcome is LocationOutcome.ABSENT:
            self.report.absent[row.location] += 1
        elif outcome is LocationOutcome.REFUSED:
            self.report.refused[row.location] += 1
        elif outcome is LocationOutcome.UNREACHABLE:
            self.report.unreachable[row.location] += 1
        if not self.dry_run:
            record_attempt(self.conn, row.location_id, outcome, error)

    def _reject(self, row: CandidateLocation, what: str, error: str) -> None:
        self.report.rejected[(row.location, what)] += 1
        if not self.dry_run:
            record_attempt(self.conn, row.location_id, LocationOutcome.REJECTED, error)

    def _fetched(self, attempt: _Attempt, sha256: str, byte_size: int, cache_path: Path) -> None:
        mark_location_fetched(
            self.conn,
            attachment_id=attempt.row.attachment_id,
            location_id=attempt.row.location_id,
            sha256=sha256,
            byte_size=byte_size,
            cache_path=str(cache_path),
        )
        self.report.materialized[
            (attempt.tier.value, attempt.row.location, attempt.row.match.value)
        ] += 1
        # Queue its S5b enrichment now, exactly as S5a does for a row
        # materialized from its own source path.
        outcome = enqueue_for_materialized(
            self.conn, attempt.row.attachment_id, cache_path, data_root=self.data_root
        )
        self.report.enrichment_enqueued += len(outcome.enqueued)
        self.report.enrichment_unroutable += int(outcome.unroutable)
        self.report.enrichment_plan_errors += int(outcome.error is not None)

    def _keep_going(self) -> bool:
        """False once the stop check has given a reason (recorded)."""
        if self.stop_check is None:
            return True
        reason = self.stop_check()
        if reason is None:
            return True
        self.report.stopped = reason
        self.report.notes.append(f"stopped: {reason.line()}")
        return False

    def _space_ok(self) -> bool:
        if self.disk_free_fn is None:
            return True
        if self._tries != 1 and self._tries % self.free_space_check_interval != 0:
            return True
        free = self.disk_free_fn(self.data_root)
        if free < self.min_free_bytes:
            self.report.halted_low_disk_space = True
            self.report.notes.append(
                f"location phase halted: free space {free} bytes < minimum {self.min_free_bytes}"
            )
            return False
        return True

    def _space_for_pull(self) -> bool:
        """A pull copies into staging, on the same volume as the cache."""
        if self.disk_free_fn is None:
            return True
        free = self.disk_free_fn(self.data_root)
        if free < self.min_free_bytes:
            self.report.halted_low_disk_space = True
            self.report.notes.append(
                f"pulls halted: free space {free} bytes < minimum {self.min_free_bytes}"
            )
            return False
        return True

    def _release_staged(self, row: CandidateLocation, staged: Path) -> None:
        """Delete a staged copy once nothing still needs it. Staging is this
        index's own directory; a source is never touched."""
        if self.dry_run:
            return
        if other_open_reference_exists(
            self.conn, location=row.location, path=row.path, exclude_location_ids=[row.location_id]
        ):
            return
        location_root = self.access.staging_root / row.location
        if not is_contained_in(staged, location_root):  # pragma: no cover - defensive
            return
        staged.unlink(missing_ok=True)
        parent = staged.parent
        while parent != location_root and is_contained_in(parent, location_root):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    # -- attempts ---------------------------------------------------------

    def _try_cache(self, attempt: _Attempt) -> bool:
        row = attempt.row
        assert row.sha256 is not None
        result = materialize_from_cache(self.data_root, row.sha256, expected_size=row.byte_size)
        if result is None:
            return False
        self._fetched(attempt, result.sha256, result.byte_size, result.cache_path)
        return True

    def _verify_and_materialize(
        self, attempt: _Attempt, path: Path, root: Path, *, staged: bool
    ) -> bool:
        row = attempt.row
        problem, size = _file_problem(path, root)
        if problem is LocationOutcome.ABSENT and staged:
            return False  # not staged after all: nothing was tried
        if problem is not None:
            self._record(row, problem, f"{problem.value}: {path}")
            return False
        if row.byte_size is not None and size != row.byte_size:
            self._reject(row, "size", f"rejected: size {size} != expected {row.byte_size}: {path}")
            if staged:
                self._release_staged(row, path)
            return False
        if not staged and self.throttle is not None:
            self.throttle.wait()  # a local read may make iCloud download the file
        try:
            result = materialize_attachment(
                path, self.data_root, expected_size=row.byte_size, expected_sha256=row.sha256
            )
        except ContentMismatchError as exc:
            self._reject(row, exc.what, f"rejected: {exc}: {path}")
            if staged:
                self._release_staged(row, path)
            return False
        except OSError as exc:
            deterministic = classify_os_error(exc)
            outcome = LocationOutcome.REFUSED if deterministic else LocationOutcome.UNREACHABLE
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                outcome = LocationOutcome.ABSENT
            self._record(row, outcome, f"{outcome.value}: {exc}")
            return False
        self._fetched(attempt, result.sha256, result.byte_size, result.cache_path)
        if staged:
            self._release_staged(row, path)
        return True

    def _try(self, attempt: _Attempt) -> bool:
        row = attempt.row
        rel = row.rel
        if attempt.tier is Tier.CACHE:
            return self._try_cache(attempt)
        if rel is None:  # pragma: no cover - such rows are refused before ordering
            return False
        if attempt.tier is Tier.LOCAL:
            root = self.access.attachments_root
            return self._verify_and_materialize(attempt, root / rel, root, staged=False)
        staged = self.access.staged_path(row.location, rel)
        if staged is None:
            return False
        return self._verify_and_materialize(
            attempt, staged, self.access.staging_root / row.location, staged=True
        )

    # -- ordering ---------------------------------------------------------

    def _attempts_for(self, rows: list[CandidateLocation]) -> list[_Attempt]:
        attempts, unreadable, deferred = _ordered_attempts(rows, self.access, self.data_root)
        if rows and rows[0].attachment_id in self.read_this_run:
            # S5a read this attachment's own path moments ago and it failed;
            # its retry ladder owns that path. Reading it again here would
            # only repeat the same iCloud request in the same run.
            own = fetch_relative_path(self.read_this_run[rows[0].attachment_id])
            attempts = [
                a for a in attempts
                if not (a.tier is Tier.LOCAL and own is not None and a.row.rel == own)
            ]
        for row in unreadable:
            # A path no root can read (a temporary directory, the sticker
            # cache): close it now rather than list it on every run.
            self._record(row, LocationOutcome.REFUSED,
                         "refused: the path is not below any readable root")
        if deferred:
            self.report.deferred_flagged += 1
        return attempts

    # -- run --------------------------------------------------------------

    def run(self) -> LocationFetchReport:
        grouped = load_open_candidates(self.conn)
        self.report.attachments_with_candidates = len(grouped)
        queues = {aid: self._attempts_for(rows) for aid, rows in sorted(grouped.items())}

        if self.dry_run:
            for queue in queues.values():
                first = next((a for a in queue if a.tier in _ACTIONABLE), None)
                if first is not None:
                    self.report.would_try[(first.tier.value, first.row.location,
                                           first.row.match.value)] += 1
                else:
                    self._classify_leftover(queue)
            self.report.notes.append(
                "dry run: nothing was read, copied, pulled or written; would_try counts the "
                "first copy each attachment would try"
            )
            return self.report

        started: set[int] = set()
        leftovers: dict[int, list[_Attempt]] = {}
        halted = False
        while queues and not halted:
            pull_requests: dict[str, list[_Attempt]] = {}
            for aid in sorted(queues):
                queue = queues[aid]
                done = False
                while queue:
                    attempt = queue[0]
                    pulling = attempt.tier is Tier.PULL and self.settings.run_pulls
                    if not pulling:
                        queue.pop(0)
                        if attempt.tier not in _ACTIONABLE or attempt.tier is Tier.PULL:
                            leftovers.setdefault(aid, []).append(attempt)
                            continue
                    if aid not in started:
                        if self.budget is not None and len(started) >= self.budget:
                            self.report.budget_capped = True
                            if not pulling:
                                queue.insert(0, attempt)
                            halted = True
                            break
                        started.add(aid)
                    if pulling:
                        pull_requests.setdefault(attempt.row.location, []).append(attempt)
                        break
                    self._tries += 1
                    if not self._space_ok() or not self._keep_going():
                        halted = True
                        break
                    if self._try(attempt):
                        done = True
                        break
                if halted:
                    break
                if done:
                    del queues[aid]
                    leftovers.pop(aid, None)
                elif not queue:
                    del queues[aid]
            if halted or not pull_requests:
                break
            for location, attempts in pull_requests.items():
                if not self._space_for_pull() or not self._keep_going():
                    halted = True
                    break
                done_ids = self._pull(location, attempts)
                for attempt in attempts:
                    aid = attempt.row.attachment_id
                    waiting = queues.get(aid)
                    if waiting and waiting[0] is attempt:
                        waiting.pop(0)
                    if aid in done_ids:
                        queues.pop(aid, None)
                        leftovers.pop(aid, None)
                    elif waiting is not None and not waiting:
                        del queues[aid]

        self.report.attempted = len(started)
        for aid, queue in queues.items():  # halted or capped before these ran
            leftovers.setdefault(aid, []).extend(queue)
        # Every attachment that was materialized has been dropped from
        # `leftovers`; what remains is still missing a copy.
        for attempts in leftovers.values():
            self._classify_leftover(attempts)
        return self.report

    def _classify_leftover(self, attempts: list[_Attempt]) -> None:
        skipped = next((a for a in attempts if a.tier is Tier.PULL), None)
        if skipped is not None:  # only left over when pulls were switched off
            self.report.pull_skipped[skipped.row.location] += 1
            return
        waiting = next((a for a in attempts if a.tier is Tier.AWAITING_PUSH), None)
        if waiting is not None:
            self.report.awaiting_push[waiting.row.location] += 1
            return
        unread = next((a for a in attempts if a.tier is Tier.NO_ACCESS), None)
        if unread is not None:
            self.report.no_access[unread.row.location] += 1

    def _pull(self, location: str, attempts: list[_Attempt]) -> set[int]:
        """One read-only rsync for every copy wanted from this location, then
        each arrival is verified like a pushed copy. Returns the attachment
        ids materialized."""
        pull = self.access.pull_for(location)
        assert pull is not None
        destination = self.access.staging_root / location
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        rels = sorted({a.row.rel for a in attempts if a.row.rel is not None})
        command = pull_command(
            rsync=self.settings.rsync_command,
            ssh=self.settings.ssh_command,
            ssh_host=pull.ssh_host,
            remote_root=pull.root,
            local_staging_dir=destination,
            relpaths=rels,
        )
        result = self.settings.copy_runner(command)
        self._write_log(f"pull-{location}", result.returncode, result.stderr)
        if not result.reached:
            self.report.pull_runs_failed[location] += 1
        done: set[int] = set()
        for attempt in attempts:
            rel = attempt.row.rel
            assert rel is not None
            staged = destination / rel
            if _is_regular_file(staged):
                self.report.pulled[location] += 1
                if self._verify_and_materialize(attempt, staged, destination, staged=True):
                    done.add(attempt.row.attachment_id)
                continue
            outcome = LocationOutcome.ABSENT if result.reached else LocationOutcome.UNREACHABLE
            self._record(attempt.row, outcome,
                         f"{outcome.value}: pull from {location} exited {result.returncode}")
        return done

    def _write_log(self, label: str, returncode: int, stderr: str) -> None:
        log_dir = self.settings.log_dir
        if log_dir is None or not stderr.strip():
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"attachment-{label}-{stamp}.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"exit {returncode}\n{stderr}\n")
        os.chmod(log_path, 0o600)


def local_location_for(cfg: Config) -> str:
    """The code this host's own Messages folder is filed under:
    `attachments.local_location` if set, else the name of the sync source
    that is this host's live chat.db — the name extraction files that
    source's recorded paths under."""
    if cfg.attachments.local_location:
        return cfg.attachments.local_location
    for source in cfg.sync.sources:
        if is_same_file(source.chat_db, cfg.paths.live_chat_db):
            return source.name
    return cfg.sync.sources[0].name


def access_from_config(cfg: Config) -> LocationAccess:
    attachments = cfg.attachments
    return LocationAccess(
        local_location=local_location_for(cfg),
        attachments_root=cfg.paths.live_chat_db.parent / "Attachments",
        staging_root=resolve_path(join_under_root(cfg.paths.data_root, attachments.staging_dir)),
        push_locations=tuple(attachments.push_locations),
        pulls=tuple(PullLocation(p.location, p.ssh_host, p.root) for p in attachments.pull),
    )


def settings_from_config(cfg: Config, *, run_pulls: bool = True) -> LocationFetchSettings:
    return LocationFetchSettings(
        access=access_from_config(cfg),
        ssh_command=cfg.attachments.ssh_command,
        rsync_command=cfg.attachments.rsync_command,
        run_pulls=run_pulls,
        log_dir=cfg.paths.data_root / "logs",
    )


def fetch_from_locations(
    conn: psycopg.Connection,
    data_root: Path,
    settings: LocationFetchSettings,
    *,
    throttle: RateThrottle | None = None,
    dry_run: bool = False,
    budget: int | None = None,
    disk_free_fn: DiskFreeFn | None = None,
    min_free_bytes: int = 0,
    free_space_check_interval: int = 100,
    read_this_run: dict[int, str] | None = None,
    stop_check: StopCheck | None = None,
) -> LocationFetchReport:
    """Run the location phase once (module docstring). `budget` caps how
    many attachments are attempted (the first-run trial gate's remainder).
    `read_this_run` maps the attachments S5a just tried to the path it
    read, so the same local path is not read twice in one run.
    `dry_run=True` classifies every attachment's first copy and reads,
    copies and writes nothing."""
    return _LocationPhase(
        conn,
        data_root,
        settings,
        throttle=throttle,
        dry_run=dry_run,
        budget=budget,
        disk_free_fn=disk_free_fn,
        min_free_bytes=min_free_bytes,
        free_space_check_interval=free_space_check_interval,
        read_this_run=read_this_run or {},
        stop_check=stop_check,
    ).run()


__all__ = [
    "LocationAccess",
    "LocationFetchReport",
    "LocationFetchSettings",
    "PullLocation",
    "Tier",
    "access_from_config",
    "best_copy",
    "fetch_from_locations",
    "local_location_for",
    "settings_from_config",
]
