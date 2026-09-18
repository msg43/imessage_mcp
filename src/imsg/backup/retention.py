"""The retention rule for `$DATA_ROOT/backups/` — deliberately narrow.

SPEC §5.3 asks for "14 kept". The hard part is not *which* to keep, it
is proving that the deletion side can only ever remove something it has
positively identified as a superseded, **complete** backup set. A
retention pass that deletes by age alone will eventually delete a
half-written set that a killed 04:00 run left behind, or an operator's
own file that happens to live in the same directory, and it will do it
silently at four in the morning.

**The rule, in full.** A directory entry under `backups/` is a deletion
candidate only when every one of these holds:

1. it is a **direct child** of the resolved `backups/` directory, and is
   a real directory — not a file, not a symlink (`Path.is_dir()` is
   checked against the un-followed entry via `is_symlink()` first);
2. its name matches :data:`SET_NAME_RE` exactly — `backup-<UTC
   timestamp>-<6 hex>`. This is what keeps the `.incomplete-*` staging
   directories, and anything an operator put here, out of the candidate
   set by construction rather than by a later exclusion;
3. it contains a `MANIFEST.json` that parses as an object, declares
   ``format`` equal to :data:`MANIFEST_FORMAT`, and carries
   ``"complete": true``. The manifest is written **last** by
   :mod:`imsg.backup.pipeline`, so its presence is the only evidence
   that a set finished;
4. after sorting every set satisfying 1-3 by its manifest ``created_at``
   (descending, ties broken by name), it is at index >= ``keep``;
5. ``keep >= 1``, enforced here, so the newest complete set is outside
   the candidate set even if a caller passes ``keep=0``.

Anything failing 1-3 is classified as **partial** or **foreign** and is
never deleted — only counted and reported, so the operator can see that
`backups/` is accumulating debris instead of discovering it when the
volume fills. The one thing `imsg backup` does delete outside this rule
is its own staging directory, on its own failure path: it created that
directory microseconds earlier and holds its path, which is the only
certainty strong enough to justify a `rmtree`.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BACKUP_SUBDIR = "backups"
"""SPEC §5.3: `backups/  nightly local recovery copies, 14 kept`."""

MANIFEST_FILENAME = "MANIFEST.json"
MANIFEST_FORMAT = 1

DEFAULT_KEEP = 14
"""SPEC §5.3/§14: "14 retained". Counted in *sets*, not days — two runs
on one day produce two sets and consume two slots, which is the honest
reading: retention protects a bounded amount of disk, and a day on
which the operator ran the job twice is a day on which twice as much
disk was used."""

STAGING_PREFIX = ".incomplete-"
"""Staging directories are dot-prefixed *and* named with a word that
says what they are, so that a human listing `backups/` and a machine
matching :data:`SET_NAME_RE` both reach the same conclusion."""

SET_NAME_RE = re.compile(r"^backup-(?P<stamp>\d{8}T\d{6}Z)-[0-9a-f]{6}$")
"""`backup-20260918T040000Z-a1b2c3`. The six-hex suffix exists so that
two runs in the same second cannot collide on a directory name — the
idempotency requirement is that a second run of the day adds a set, and
never writes into or over an existing one."""


@dataclass(frozen=True, slots=True)
class BackupSet:
    """A directory under `backups/` that passed rules 1-3 above."""

    path: Path
    created_at: str
    """The manifest's own `created_at` (ISO-8601 UTC). Sorted on in
    preference to the directory name: the name encodes the same instant,
    but the manifest is the artifact the completeness check already read,
    and a set whose name and manifest disagree should sort by what it
    actually recorded."""

    byte_size: int


@dataclass(frozen=True, slots=True)
class BackupIndex:
    """Everything under `backups/`, split by what can be proven about it."""

    complete: tuple[BackupSet, ...]
    """Newest first."""

    partial: tuple[Path, ...]
    """Correctly-named sets with no valid complete manifest, plus every
    leftover `.incomplete-*` staging directory. Never deleted."""

    foreign: tuple[Path, ...]
    """Everything else in `backups/` — files, symlinks, differently-named
    directories. Never deleted, never even opened."""


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    keep: int
    delete: tuple[Path, ...]
    retained: tuple[Path, ...]
    partial: tuple[Path, ...]
    foreign: tuple[Path, ...]

    @property
    def deletes_nothing(self) -> bool:
        return not self.delete


def _directory_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:  # pragma: no cover - racing a concurrent delete
                continue
    return total


def read_manifest(set_dir: Path) -> dict[str, object] | None:
    """The set's manifest, or `None` if it is missing, unreadable, not an
    object, of an unknown `format`, or not marked complete.

    Every one of those is treated identically on purpose: the only
    question this function answers is "can this directory be proven to be
    a finished backup set of a format this code understands", and a
    manifest that fails for any reason cannot prove it.
    """
    manifest_path = set_dir / MANIFEST_FILENAME
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("format") != MANIFEST_FORMAT:
        return None
    if parsed.get("complete") is not True:
        return None
    if not isinstance(parsed.get("created_at"), str):
        return None
    return parsed


def index_backups(backups_dir: Path) -> BackupIndex:
    """Classify every entry in `backups_dir` under the rule at the top.

    A missing `backups_dir` is an empty index, not an error — the first
    ever run has nothing to retain.
    """
    complete: list[BackupSet] = []
    partial: list[Path] = []
    foreign: list[Path] = []

    try:
        entries = sorted(backups_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return BackupIndex((), (), ())
    except OSError:
        return BackupIndex((), (), ())

    for entry in entries:
        # Rule 1. A symlink is rejected before `is_dir()` is consulted:
        # `is_dir()` follows links, so a symlink to a real directory would
        # otherwise be a deletion candidate whose target lives anywhere.
        if entry.is_symlink() or not entry.is_dir():
            foreign.append(entry)
            continue
        if entry.name.startswith(STAGING_PREFIX):
            partial.append(entry)
            continue
        # Rule 2.
        if not SET_NAME_RE.match(entry.name):
            foreign.append(entry)
            continue
        # Rule 3.
        manifest = read_manifest(entry)
        if manifest is None:
            partial.append(entry)
            continue
        complete.append(
            BackupSet(
                path=entry,
                created_at=str(manifest["created_at"]),
                byte_size=_directory_bytes(entry),
            )
        )

    complete.sort(key=lambda s: (s.created_at, s.path.name), reverse=True)
    return BackupIndex(tuple(complete), tuple(partial), tuple(foreign))


def plan_retention(index: BackupIndex, *, keep: int = DEFAULT_KEEP) -> RetentionPlan:
    """Rules 4 and 5: keep the newest `keep` complete sets, delete the rest.

    `keep` is clamped to at least 1 rather than rejected, so a
    misconfigured `--keep 0` prunes hard but can never leave the operator
    with no backup at all. Nothing outside `index.complete` is ever a
    candidate.
    """
    effective_keep = max(1, keep)
    retained = index.complete[:effective_keep]
    doomed = index.complete[effective_keep:]
    return RetentionPlan(
        keep=effective_keep,
        delete=tuple(s.path for s in doomed),
        retained=tuple(s.path for s in retained),
        partial=index.partial,
        foreign=index.foreign,
    )


def apply_retention(plan: RetentionPlan) -> tuple[Path, ...]:
    """Delete exactly `plan.delete`, re-checking each path first.

    The re-check is not paranoia about this module: `plan_retention` is
    a pure function over an index that may have been built minutes ago
    (a dry run the operator then approved), and a `rmtree` is the one
    irreversible thing `imsg backup` does. A candidate that no longer
    looks like a complete set is skipped rather than removed.
    """
    deleted: list[Path] = []
    for path in plan.delete:
        if path.is_symlink() or not path.is_dir():
            continue
        if not SET_NAME_RE.match(path.name):
            continue
        if read_manifest(path) is None:
            continue
        shutil.rmtree(path)
        deleted.append(path)
    return tuple(deleted)


__all__ = [
    "BACKUP_SUBDIR",
    "DEFAULT_KEEP",
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT",
    "SET_NAME_RE",
    "STAGING_PREFIX",
    "BackupIndex",
    "BackupSet",
    "RetentionPlan",
    "apply_retention",
    "index_backups",
    "plan_retention",
    "read_manifest",
]
