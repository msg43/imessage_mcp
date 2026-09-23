"""`imsg locate-attachments` (owner decision D13): find every candidate copy
of every attachment the index has not materialized, and record it in
`attachment_location` (`imsg.backfill.locations`).

Evidence comes from four kinds of input, each optional:

- **seed databases** (`--seed-db CODE=PATH`): chat.db-shaped files, opened
  read-only and immutable. The path each recorded for an attachment is
  stored as a `recorded_path` row under `CODE`, the Mac that database came
  from. Extraction already records every source's path from now on; this
  recovers the paths of sources extracted before it did.
- **this host's Messages attachments folder**, walked with `lstat` only.
- **listings of another Mac's Messages attachments folder**
  (`--listing CODE=PATH`): tab-separated `rel, size, mtime, sha256`, `rel`
  below that folder. The hash makes a match verifiable before anything is
  copied, and lets the backfill skip the copy when the cache already holds
  that content.
- **drive catalogs** (`--catalog [CODE=]PATH`, a file or a directory of
  them): gzip or plain tab-separated `path, size, mtime, ext` with `#`
  header lines, `path` relative to the drive's root. The drive code comes
  from the file name (`<CODE>.files.tsv.gz`, `<CODE>.files.2.tsv.gz`) or
  from a sweep-run directory (`<stamp>_<host>_<CODE>_<id>/catalog.files.tsv.gz`),
  unless given.

Each file seen is matched against the attachments still missing a copy:

- `recorded_path`: its path below a Messages attachments folder equals a
  path some source recorded for the attachment (a Time Machine or clone of
  a Mac's home counts: the part below `Library/Messages/Attachments/` is
  compared);
- `guid_folder`: its folder is named after the attachment's GUID and its
  name is one of the attachment's names;
- `name_size`: one of the attachment's names, and the attachment's byte
  size, anywhere else. Flagged.
- A name alone, or a GUID folder holding a different name, is counted in
  the report and never stored.

An attachment's names are its chat.db transfer name and the base name of
every path recorded for it; names compare case-insensitively with Unicode
composed alike (`imsg.backfill.locations.name_key`). At most
`MAX_UNRECORDED_ROWS_PER_LOCATION` `guid_folder`/`name_size` rows are kept
per attachment and location — enough to survive a few bad copies, without
recording every duplicate on a drive of backups. `recorded_path` rows are
never capped.

Nothing here reads a file's content, copies a file, or writes anything but
Postgres rows. It prints counts only: the paths stay in the database, on the
encrypted volume. `dry_run` does all of it inside a transaction that is
rolled back, so its counts are exact.
"""

from __future__ import annotations

import gzip
import io
import os
import re
import stat
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import apsw

from imsg.backfill.fetch import LocationAccess, Tier, best_copy
from imsg.backfill.locations import (
    MESSAGES_ATTACHMENTS_DIR,
    MESSAGES_PATH_PREFIX,
    LocationObservation,
    MatchQuality,
    ObservationOutcome,
    attachment_kind,
    is_sha256_hex,
    is_valid_location_code,
    load_open_candidates,
    messages_relative_path,
    name_key,
    observe_location,
    safe_relative_path,
)
from imsg.errors import AttachmentBackfillError
from imsg.sqlite_readonly import open_readonly_immutable, wal_frame_bytes

if TYPE_CHECKING:
    import psycopg

MAX_UNRECORDED_ROWS_PER_LOCATION = 10
_SQL_CHUNK = 500
_CATALOG_NAME_RE = re.compile(r"^(?P<code>[A-Z]-[A-Z0-9]{5})\.files(?:\.\d+)?\.tsv(?:\.gz)?$")
_CATALOG_RUN_FILE_RE = re.compile(r"^catalog\.files(?:\.\d+)?\.tsv(?:\.gz)?$")
_CATALOG_RUN_DIR_RE = re.compile(r"_(?P<code>[A-Z]-[A-Z0-9]{5})_[0-9a-f]+$")
_MESSAGES_MARKER = MESSAGES_ATTACHMENTS_DIR + "/"


class LocateError(AttachmentBackfillError):
    """An input `imsg locate-attachments` was pointed at cannot be read
    safely (a seed database with an unfolded write-ahead log, a code that
    is not a single path component)."""


@dataclass(frozen=True, slots=True)
class LocatedFile:
    """An input naming a location code and a file."""

    location: str
    path: Path


@dataclass
class LocateReport:
    targets: int = 0
    targets_by_state: Counter[str] = field(default_factory=Counter)
    seed_rows: int = 0
    local_files: int = 0
    listing_rows: int = 0
    catalog_files: int = 0
    catalog_rows: int = 0
    catalogs_skipped: int = 0
    """Catalog files whose drive code could not be told from their name."""
    undecodable_paths: int = 0
    """Matching paths that are not valid UTF-8 and so cannot be stored."""
    capped: int = 0
    stored: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    """(location, match quality, inserted|updated|unchanged) -> rows."""
    weak: Counter[str] = field(default_factory=Counter)
    """location -> (attachment, file) pairs matched on a name alone, or a
    GUID folder holding another name. Never stored, never fetched."""
    dry_run: bool = False


@dataclass
class CoverageReport:
    """Where the next copy of every attachment still not materialized
    would come from, and what is left without one."""

    not_materialized: Counter[str] = field(default_factory=Counter)
    best: Counter[str] = field(default_factory=Counter)
    """tier label -> attachments: `cache`, `local`, `staged:<code>`,
    `push:<code>` (awaiting a push), `pull:<code>`, `no-access:<code>`."""
    best_is_flagged: int = 0
    exhausted: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(state, kind) -> attachments whose every candidate was tried and
    closed (absent, rejected, refused)."""
    no_candidate: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(state, kind) -> attachments no source, listing or catalog placed
    anywhere."""
    rows: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(location, match quality) -> rows in `attachment_location`."""
    fetched: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(location, match quality) -> attachments materialized from there."""

    @property
    def unfetchable(self) -> int:
        return sum(self.exhausted.values()) + sum(self.no_candidate.values())


# --------------------------------------------------------------------------
# targets and matching
# --------------------------------------------------------------------------


def _rel_key(rel: str) -> str:
    return rel if rel.isascii() else unicodedata.normalize("NFC", rel)


def _basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


@dataclass
class _Target:
    attachment_id: int
    guid: str
    state: str
    kind: str
    byte_size: int | None
    names: set[str] = field(default_factory=set)
    rels: set[str] = field(default_factory=set)


class _Index:
    def __init__(self) -> None:
        self.by_id: dict[int, _Target] = {}
        self.by_guid: dict[str, _Target] = {}
        self.by_name: dict[str, list[_Target]] = {}
        self.by_rel: dict[str, list[_Target]] = {}

    def add(self, target: _Target) -> None:
        self.by_id[target.attachment_id] = target
        self.by_guid[target.guid] = target

    def add_name(self, target: _Target, name: str | None) -> None:
        if not name:
            return
        key = name_key(name)
        if key and key not in target.names:
            target.names.add(key)
            self.by_name.setdefault(key, []).append(target)

    def add_recorded_path(self, target: _Target, path: str | None) -> None:
        if not path:
            return
        self.add_name(target, _basename(path))
        rel = messages_relative_path(path)
        if rel is None:
            return
        key = _rel_key(rel)
        if key not in target.rels:
            target.rels.add(key)
            self.by_rel.setdefault(key, []).append(target)

    def might_match(self, name: str, parent: str, recorded_rel: str | None) -> bool:
        return (
            parent in self.by_guid
            or name_key(name) in self.by_name
            or (recorded_rel is not None and _rel_key(recorded_rel) in self.by_rel)
        )

    def match(
        self, *, name: str, parent: str, recorded_rel: str | None, size: int | None
    ) -> tuple[list[tuple[_Target, MatchQuality]], int]:
        """(matches, weak): the best acceptable match per attachment for one
        file, and how many attachments matched it only weakly."""
        found: dict[int, tuple[_Target, MatchQuality]] = {}
        if recorded_rel is not None:
            for target in self.by_rel.get(_rel_key(recorded_rel), ()):
                found[target.attachment_id] = (target, MatchQuality.RECORDED_PATH)
        key = name_key(name)
        weak = 0
        in_folder = self.by_guid.get(parent)
        if in_folder is not None and in_folder.attachment_id not in found:
            if key in in_folder.names:
                found[in_folder.attachment_id] = (in_folder, MatchQuality.GUID_FOLDER)
            else:
                weak += 1
        for target in self.by_name.get(key, ()):
            if target.attachment_id in found:
                continue
            if target.byte_size is not None and size == target.byte_size:
                found[target.attachment_id] = (target, MatchQuality.NAME_SIZE)
            elif target is not in_folder:
                weak += 1
        return list(found.values()), weak


def _load_targets(conn: psycopg.Connection) -> _Index:
    index = _Index()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attachment_id, a.source_guid, a.filename, a.source_path, a.uti,
                   a.mime_type, a.byte_size, a.state::text,
                   COALESCE(array_agg(l.path) FILTER (WHERE l.match_quality = 'recorded_path'),
                            '{}')
              FROM attachment a
              LEFT JOIN attachment_location l ON l.attachment_id = a.attachment_id
             WHERE a.state <> 'materialized'
             GROUP BY a.attachment_id
             ORDER BY a.attachment_id
            """
        )
        rows = cur.fetchall()
    for aid, guid, filename, source_path, uti, mime, size, state, recorded in rows:
        target = _Target(
            attachment_id=int(aid),
            guid=str(guid),
            state=str(state),
            kind=attachment_kind(mime, uti, filename),
            byte_size=int(size) if size is not None and int(size) > 0 else None,
        )
        index.add(target)
        index.add_name(target, filename)
        index.add_recorded_path(target, source_path)
        for path in recorded:
            index.add_recorded_path(target, path)
    return index


class _Collector:
    """Observations for one run, grouped by the row they describe
    (attachment, location, path) and capped per attachment and location.
    Every observation of a row is kept: a seed database and a listing can
    both describe one file, the first with the path and the second with
    its size and hash, and `observe_location` merges them."""

    def __init__(self, report: LocateReport) -> None:
        self.report = report
        self.observations: dict[tuple[int, str, str], list[LocationObservation]] = {}
        self._unrecorded: Counter[tuple[int, str]] = Counter()

    def add(self, obs: LocationObservation) -> None:
        try:
            obs.path.encode("utf-8")
        except UnicodeEncodeError:
            self.report.undecodable_paths += 1
            return
        key = (obs.attachment_id, obs.location, obs.path)
        seen = self.observations.get(key)
        if seen is not None:
            seen.append(obs)
            return
        if obs.match is not MatchQuality.RECORDED_PATH:
            per_location = (obs.attachment_id, obs.location)
            if self._unrecorded[per_location] >= MAX_UNRECORDED_ROWS_PER_LOCATION:
                self.report.capped += 1
                return
            self._unrecorded[per_location] += 1
        self.observations[key] = [obs]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _read_seed(
    location: LocatedFile, index: _Index, collector: _Collector, report: LocateReport
) -> None:
    pending = wal_frame_bytes(location.path)
    if pending:
        raise LocateError(
            f"refusing to read seed database '{location.path}': its write-ahead log holds "
            f"{pending} bytes of frames the read-only immutable open cannot see. Fold it into "
            f"the file first, on that copy only: sqlite3 '{location.path}' "
            f"'PRAGMA wal_checkpoint(TRUNCATE)'"
        )
    try:
        seed = open_readonly_immutable(location.path)
    except apsw.Error as exc:
        raise LocateError(f"seed database '{location.path}' will not open: {exc}") from exc
    try:
        guids = sorted(index.by_guid)
        for start in range(0, len(guids), _SQL_CHUNK):
            chunk = guids[start : start + _SQL_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            for guid, path, transfer_name in seed.execute(
                f"SELECT guid, filename, transfer_name FROM attachment WHERE guid IN ({placeholders})",
                chunk,
            ):
                target = index.by_guid.get(str(guid))
                if target is None:
                    continue
                report.seed_rows += 1
                index.add_name(target, transfer_name)
                if path:
                    index.add_recorded_path(target, str(path))
                    collector.add(
                        LocationObservation(
                            attachment_id=target.attachment_id,
                            location=location.location,
                            path=str(path),
                            match=MatchQuality.RECORDED_PATH,
                            reported_by=location.location,
                        )
                    )
    finally:
        seed.close()


def _observe(
    index: _Index,
    collector: _Collector,
    report: LocateReport,
    *,
    location: str,
    stored_path: str,
    name: str,
    parent: str,
    recorded_rel: str | None,
    size: int | None,
    sha256: str | None,
    reported_by: str,
) -> None:
    matches, weak = index.match(name=name, parent=parent, recorded_rel=recorded_rel, size=size)
    if weak:
        report.weak[location] += weak
    for target, quality in matches:
        collector.add(
            LocationObservation(
                attachment_id=target.attachment_id,
                location=location,
                path=stored_path,
                match=quality,
                reported_by=reported_by,
                byte_size=size,
                sha256=sha256,
            )
        )


def _walk_local(
    root: Path, location: str, index: _Index, collector: _Collector, report: LocateReport
) -> None:
    if not root.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        parent = os.path.basename(dirpath)
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            report.local_files += 1
            rel = Path(os.path.relpath(full, root)).as_posix()
            if not index.might_match(filename, parent, rel):
                continue
            _observe(
                index, collector, report,
                location=location, stored_path=MESSAGES_PATH_PREFIX + rel, name=filename,
                parent=parent, recorded_rel=rel, size=st.st_size, sha256=None,
                reported_by=f"walk:{location}",
            )


def _parse_size(field_value: str) -> int | None:
    try:
        size = int(field_value)
    except ValueError:
        return None
    return size if size >= 0 else None


def _read_listing(
    listing: LocatedFile, index: _Index, collector: _Collector, report: LocateReport
) -> None:
    reported_by = f"listing:{listing.path.name}"
    with listing.path.open("r", encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            report.listing_rows += 1
            rel = parts[0]
            if not safe_relative_path(rel):
                continue
            name = _basename(rel)
            parent = rel.rsplit("/", 2)[-2] if rel.count("/") >= 1 else ""
            if not index.might_match(name, parent, rel):
                continue
            sha = parts[3].strip().lower() if len(parts) >= 4 else ""
            _observe(
                index, collector, report,
                location=listing.location, stored_path=MESSAGES_PATH_PREFIX + rel, name=name,
                parent=parent, recorded_rel=rel, size=_parse_size(parts[1]),
                sha256=sha if is_sha256_hex(sha) else None, reported_by=reported_by,
            )


def catalog_code_for(path: Path) -> str | None:
    """The drive code a catalog file's name or sweep-run directory carries."""
    named = _CATALOG_NAME_RE.fullmatch(path.name)
    if named:
        return named.group("code")
    if _CATALOG_RUN_FILE_RE.fullmatch(path.name):
        run = _CATALOG_RUN_DIR_RE.search(path.parent.name)
        if run:
            return run.group("code")
    return None


def _catalog_files(spec: str) -> Iterator[LocatedFile | Path]:
    """`[CODE=]PATH`, a file or a directory searched for catalogs. Yields a
    `LocatedFile` when the code is known, else the bare path (skipped)."""
    code, sep, raw = spec.partition("=")
    explicit = code if sep and is_valid_location_code(code) and "/" not in code else None
    target = Path(raw if explicit else spec).expanduser()
    if target.is_dir():
        candidates: Iterable[Path] = sorted(
            p for p in target.rglob("*") if p.is_file() and ".tsv" in p.name
        )
    else:
        candidates = [target]
    for candidate in candidates:
        found = explicit or catalog_code_for(candidate)
        yield LocatedFile(found, candidate) if found else candidate


def _open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return io.TextIOWrapper(
            gzip.open(path, "rb"), encoding="utf-8", errors="surrogateescape"
        )
    return path.open("r", encoding="utf-8", errors="surrogateescape")


def _read_catalog(
    catalog: LocatedFile, index: _Index, collector: _Collector, report: LocateReport
) -> None:
    report.catalog_files += 1
    reported_by = f"catalog:{catalog.path.name}"
    with _open_text(catalog.path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            tab = line.find("\t")
            if tab <= 0:
                continue
            report.catalog_rows += 1
            path = line[:tab]
            slash = path.rfind("/")
            name = path[slash + 1 :]
            parent = path[path.rfind("/", 0, slash) + 1 : slash] if slash > 0 else ""
            recorded_rel = messages_relative_path(path) if _MESSAGES_MARKER in path else None
            if not index.might_match(name, parent, recorded_rel):
                continue
            if not safe_relative_path(path):
                continue
            rest = line[tab + 1 :]
            size_field = rest.split("\t", 1)[0].strip()
            _observe(
                index, collector, report,
                location=catalog.location, stored_path=path, name=name, parent=parent,
                recorded_rel=recorded_rel, size=_parse_size(size_field), sha256=None,
                reported_by=reported_by,
            )


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def build_coverage(
    conn: psycopg.Connection, access: LocationAccess, data_root: Path
) -> CoverageReport:
    """Read-only: where each attachment still missing a copy would get one
    next (`imsg.backfill.fetch.best_copy`), and what is left without any."""
    coverage = CoverageReport()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attachment_id, a.state::text, a.mime_type, a.uti, a.filename,
                   EXISTS (SELECT 1 FROM attachment_location l
                            WHERE l.attachment_id = a.attachment_id)
              FROM attachment a
             WHERE a.state <> 'materialized'
            """
        )
        pending = cur.fetchall()
        cur.execute(
            "SELECT location, match_quality::text, count(*), count(fetched_at) "
            "FROM attachment_location GROUP BY 1, 2"
        )
        for location, match, rows, fetched in cur.fetchall():
            coverage.rows[(str(location), str(match))] = int(rows)
            if fetched:
                coverage.fetched[(str(location), str(match))] = int(fetched)
    open_rows = load_open_candidates(conn)
    for aid, state, mime, uti, filename, has_rows in pending:
        coverage.not_materialized[str(state)] += 1
        best = best_copy(open_rows.get(int(aid), []), access, data_root)
        if best is None:
            bucket = coverage.exhausted if has_rows else coverage.no_candidate
            bucket[(str(state), attachment_kind(mime, uti, filename))] += 1
            continue
        tier, row = best
        label = tier.value if tier in (Tier.CACHE, Tier.LOCAL) else {
            Tier.STAGED: "staged", Tier.AWAITING_PUSH: "push", Tier.PULL: "pull",
            Tier.NO_ACCESS: "no-access",
        }[tier] + f":{row.location}"
        coverage.best[label] += 1
        if row.match.flagged:
            coverage.best_is_flagged += 1
    return coverage


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


class _DryRunRollback(Exception):
    def __init__(self, report: LocateReport, coverage: CoverageReport) -> None:
        self.report = report
        self.coverage = coverage


def run_locate(
    conn: psycopg.Connection,
    *,
    access: LocationAccess,
    data_root: Path,
    seeds: Iterable[LocatedFile] = (),
    listings: Iterable[LocatedFile] = (),
    catalogs: Iterable[str] = (),
    walk_local: bool = True,
    dry_run: bool = False,
) -> tuple[LocateReport, CoverageReport]:
    """Record every candidate copy found, then report coverage. Seeds are
    read first, so the paths they recorded are matched against every other
    input."""
    report = LocateReport(dry_run=dry_run)
    seed_list = list(seeds)
    listing_list = list(listings)
    for item in (*seed_list, *listing_list):
        if not is_valid_location_code(item.location):
            raise LocateError(f"location code {item.location!r} is not a single path component")

    index = _load_targets(conn)
    report.targets = len(index.by_id)
    report.targets_by_state.update(t.state for t in index.by_id.values())
    collector = _Collector(report)

    for seed in seed_list:
        _read_seed(seed, index, collector, report)
    if walk_local:
        _walk_local(access.attachments_root, access.local_location, index, collector, report)
    for listing in listing_list:
        _read_listing(listing, index, collector, report)
    for spec in catalogs:
        for found in _catalog_files(spec):
            if isinstance(found, LocatedFile):
                _read_catalog(found, index, collector, report)
            else:
                report.catalogs_skipped += 1

    try:
        with conn.transaction():
            with conn.cursor() as cur:
                for key in sorted(collector.observations):
                    group = collector.observations[key]
                    outcomes = [observe_location(cur, obs) for obs in group]
                    if outcomes[0] is ObservationOutcome.INSERTED:
                        final = ObservationOutcome.INSERTED
                    elif ObservationOutcome.UPDATED in outcomes:
                        final = ObservationOutcome.UPDATED
                    else:
                        final = ObservationOutcome.UNCHANGED
                    best = min((obs.match for obs in group), key=lambda m: m.rank)
                    report.stored[(group[0].location, best.value, final.value)] += 1
            coverage = build_coverage(conn, access, data_root)
            if dry_run:
                raise _DryRunRollback(report, coverage)
    except _DryRunRollback as rolled_back:
        return rolled_back.report, rolled_back.coverage
    conn.commit()
    return report, coverage


__all__ = [
    "MAX_UNRECORDED_ROWS_PER_LOCATION",
    "CoverageReport",
    "LocateError",
    "LocateReport",
    "LocatedFile",
    "ObservationOutcome",
    "build_coverage",
    "catalog_code_for",
    "run_locate",
]
