"""Replay and export of curated identity decisions (SPEC §8 S3 curation).

`imsg identity rename` / `assign` / `merge` are how a human names the
persons S3 could not name on its own. Those decisions are **derived
state**: `run_identity` only resolves handles that have no resolution
row, so any identity rebuild recreates the review stubs and silently
drops every name that was decided by hand. This module makes the
decisions survive that — `apply_overrides` replays a decisions file
through the same three curation functions the interactive commands
call (`rename_person`, `assign_handle`, `merge_persons`), and
`export_overrides` regenerates the file from the current person table
after the next round of hand-curation.

The file format
---------------
One JSON object. Every example below is fictional.

    {
      "note": "Curated identity decisions. Re-apply after any identity rebuild.",
      "decided": "2026-08-15",
      "overrides": [
        {
          "value": "+14155552671",
          "kind": "phone",
          "name": "Alice Example",
          "was": ["Contacts Conflict"],
          "why": "the household number saved under two cards; Alice is the one who texts"
        },
        {
          "value": "alice@example.com",
          "kind": "email",
          "name": "Alice Example",
          "was": []
        },
        {
          "value": "24273",
          "kind": "unknown",
          "name": "Acme Bank Alerts",
          "was": ["Unknown Sender"],
          "why": "every message is a balance alert signed Acme Bank"
        }
      ]
    }

* `note` (string, optional) — free text for the human reading the file.
* `decided` (string, optional) — ISO date the file was last written.
* `overrides` (list, required) — one decision per handle:
  * `value` (string) — the handle as `imsg identity assign --value`
    takes it: E.164 for phones, lowercased for emails, the raw id for
    `unknown` (short codes and the like).
  * `kind` (string) — `phone`, `email`, `apple_id` or `unknown`; the
    `handle_kind` enum.
  * `name` (string) — the display name this handle's person must carry.
  * `was` (list of strings, optional) — the display name(s) the person
    carried before the decision. A record for the reader, and the
    conflict rule below treats a person still carrying one of these
    names as "not yet decided" rather than "renamed since".
  * `why` (string, optional) — the reason, in the owner's words.

Two decisions naming the same `name` mean "these handles are one
person": the second handle is merged (or, when its person keeps other
handles, assigned) onto the person the first one named. This is how a
human with six identifiers, whom Contacts could not unify because no
card matched, becomes one `person_id` again after a rebuild.

Matching
--------
A decision is matched to a handle by its **normalized** identifier:
both the recorded `(value, kind)` and whatever `normalize_handle`
produces for `value` today are looked up, so a decision recorded before
a normalization fix (an iOS filter tag such as `(filtered)` that once
kept a phone number in `kind='unknown'`) still finds the clean handle a
later rebuild creates — and, before that rebuild, the legacy row too.
Two entries whose identifiers normalize to one handle therefore describe
one sender and must agree on `name`; when they disagree, whichever is
applied second finds a person named by the first and is reported as a
conflict on every run until the file is fixed by hand — never silently
re-decided, and never flip-flopped.

Outcomes, per decision
----------------------
* `already` — every matched handle is on a person whose display name is
  already `name` (case-insensitive). Nothing is written. Running the
  file twice therefore changes nothing.
* `applied` — at least one curation operation ran:
  `rename` when no other person carries `name`; otherwise `merge` of
  the handle's person into the one that does (when the handle was that
  person's only one) or `assign` of just the handle (when the person
  keeps others).
* `unmatched` — no handle exists for the identifier. Nothing is
  invented: S3 alone creates handles and persons.
* `conflict` — a matched handle is on a person somebody has named since
  (reviewed, and named neither `name`, nor its own handle value, nor
  anything in `was`), or the handle belongs to the owner person, or
  more than one person already carries `name` so there is no single
  target. Skipped and reported; `force=True` overrides the first kind
  only — the owner and an ambiguous target are never forced.

A run is one transaction: it ends with the S3 invariant report, and a
dry run rolls the whole transaction back after computing exactly what
a real run would have done (the same savepoint-and-rollback pattern
`run_identity(dry_run=True)` uses).

Export
------
`export_overrides` writes one decision per handle of every non-owner
person that has been reviewed (named by hand, or by a unique Contacts
match — a rebuild without Contacts access loses both) or that holds
more than one handle (a merge or assign somebody made). Entries of a
previous file are carried forward: `was` and `why` are kept for a
handle that is still exported (with the previous `name` prepended to
`was` when the name changed), and a previous decision whose handle is
absent from the index or has fallen back to an unreviewed stub is kept
verbatim — export never drops a decision the index cannot currently
express. A previous decision recorded in a legacy form follows the clean
handle only once the legacy row is gone; while a legacy row and its
clean form coexist (they may sit on different persons), each is
exported as the index holds it and replay reports any disagreement.

The file carries real names and identifiers and belongs under
`paths.data_root` (CLAUDE.md non-negotiable #2); the CLI refuses to
write it anywhere else.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import psycopg

from imsg.errors import IdentityError
from imsg.stages.identity import (
    InvariantReport,
    assign_handle,
    compute_invariant_report,
    merge_persons,
    normalize_handle,
    rename_person,
)

HANDLE_KINDS: frozenset[str] = frozenset({"phone", "email", "apple_id", "unknown"})
"""The `handle_kind` enum (migrations/0001_initial.sql)."""

DEFAULT_NOTE = (
    "Curated identity decisions (imsg identity rename/assign/merge). "
    "Re-apply after any identity rebuild with `imsg identity apply-overrides`; "
    "regenerate after hand-curation with `imsg identity export-overrides`."
)

_TOP_LEVEL_KEYS = frozenset({"note", "decided", "overrides"})
_ENTRY_KEYS = frozenset({"value", "kind", "name", "was", "why"})


# --------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentityOverride:
    """One decision: the handle `(value, kind)` belongs to a person named `name`."""

    value: str
    kind: str
    name: str
    was: tuple[str, ...] = ()
    why: str | None = None


@dataclass(frozen=True, slots=True)
class IdentityOverridesFile:
    overrides: tuple[IdentityOverride, ...]
    note: str | None = None
    decided: str | None = None


def _optional_str(data: dict[str, Any], key: str, *, source: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise IdentityError(f"{source}: '{key}' must be a string when present")
    return value


def _required_str(entry: dict[str, Any], key: str, *, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise IdentityError(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


def _parse_entry(entry: object, *, where: str) -> IdentityOverride:
    if not isinstance(entry, dict):
        raise IdentityError(f"{where}: each override must be a JSON object")
    unknown = set(entry) - _ENTRY_KEYS
    if unknown:
        raise IdentityError(
            f"{where}: unknown key(s) {sorted(unknown)} — the schema is value/kind/name/was/why"
        )
    value = _required_str(entry, "value", where=where)
    kind = _required_str(entry, "kind", where=where)
    if kind not in HANDLE_KINDS:
        raise IdentityError(f"{where}: 'kind' must be one of {sorted(HANDLE_KINDS)}, not {kind!r}")
    name = _required_str(entry, "name", where=where)
    raw_was = entry.get("was", [])
    if not isinstance(raw_was, list) or not all(isinstance(w, str) for w in raw_was):
        raise IdentityError(f"{where}: 'was' must be a list of strings")
    why = entry.get("why")
    if why is not None and not isinstance(why, str):
        raise IdentityError(f"{where}: 'why' must be a string when present")
    return IdentityOverride(
        value=value,
        kind=kind,
        name=name,
        was=tuple(w.strip() for w in raw_was if w.strip()),
        why=why,
    )


def parse_overrides(data: object, *, source: str = "<overrides>") -> IdentityOverridesFile:
    """Validate a decoded JSON document against the schema in the module
    docstring. Every violation is an `IdentityError` naming the entry index
    and key, never a traceback."""
    if not isinstance(data, dict):
        raise IdentityError(
            f"{source}: the top level must be a JSON object with an 'overrides' list"
        )
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise IdentityError(f"{source}: unknown top-level key(s) {sorted(unknown)}")
    raw_overrides = data.get("overrides")
    if not isinstance(raw_overrides, list):
        raise IdentityError(f"{source}: 'overrides' must be a list")

    overrides: list[IdentityOverride] = []
    seen: dict[tuple[str, str], int] = {}
    for index, entry in enumerate(raw_overrides):
        where = f"{source}: overrides[{index}]"
        override = _parse_entry(entry, where=where)
        key = (override.value, override.kind)
        if key in seen:
            raise IdentityError(
                f"{where}: repeats the identifier of overrides[{seen[key]}] — one decision per handle"
            )
        seen[key] = index
        overrides.append(override)

    return IdentityOverridesFile(
        overrides=tuple(overrides),
        note=_optional_str(data, "note", source=source),
        decided=_optional_str(data, "decided", source=source),
    )


def load_overrides(path: Path) -> IdentityOverridesFile:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IdentityError(f"cannot read overrides file {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IdentityError(f"{path}: not valid JSON ({exc})") from exc
    return parse_overrides(data, source=str(path))


def _entry_to_json(override: IdentityOverride) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "value": override.value,
        "kind": override.kind,
        "name": override.name,
        "was": list(override.was),
    }
    if override.why is not None:
        entry["why"] = override.why
    return entry


def write_overrides(path: Path, file: IdentityOverridesFile) -> None:
    """Write the file atomically (temp file beside it, then rename), keys in
    the documented order, so a crash mid-write never leaves a truncated
    decisions file behind."""
    payload: dict[str, Any] = {
        "note": file.note if file.note is not None else DEFAULT_NOTE,
        "decided": file.decided if file.decided is not None else dt.date.today().isoformat(),
        "overrides": [_entry_to_json(o) for o in file.overrides],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        raise IdentityError(f"cannot write overrides file {path}: {exc}") from exc


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------


def identifier_forms(
    override: IdentityOverride, default_region: str
) -> tuple[tuple[str, str], ...]:
    """The `(normalized_value, kind)` pairs a decision can match: the
    recorded pair first, then what `normalize_handle` makes of `value`
    today when that differs (a filter-tagged number recorded as `unknown`
    also matches the clean `phone` handle a rebuild creates)."""
    forms = [(override.value, override.kind)]
    normalized = normalize_handle(override.value, default_region)
    if normalized not in forms:
        forms.append(normalized)
    return tuple(forms)


def _same_name(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


@dataclass(frozen=True, slots=True)
class _HandleRow:
    handle_id: int
    normalized_value: str
    kind: str
    person_id: int
    display_name: str
    needs_review: bool
    is_owner: bool

    @property
    def label(self) -> str:
        return f"{self.kind} {self.normalized_value!r}"


def _lookup_handles(cur: psycopg.Cursor[Any], forms: Iterable[tuple[str, str]]) -> list[_HandleRow]:
    rows: list[_HandleRow] = []
    for value, kind in forms:
        cur.execute(
            """
            SELECT h.handle_id, h.normalized_value, h.kind, p.person_id,
                   p.display_name, p.needs_review, p.is_owner
            FROM handle h JOIN person p ON p.person_id = h.person_id
            WHERE h.normalized_value = %s AND h.kind = %s
            """,
            (value, kind),
        )
        row = cur.fetchone()
        if row is not None:
            rows.append(
                _HandleRow(
                    handle_id=int(row[0]),
                    normalized_value=str(row[1]),
                    kind=str(row[2]),
                    person_id=int(row[3]),
                    display_name=str(row[4]),
                    needs_review=bool(row[5]),
                    is_owner=bool(row[6]),
                )
            )
    return rows


_Classification = Literal["already", "safe", "conflict", "owner"]


def _classify(row: _HandleRow, override: IdentityOverride) -> _Classification:
    """Whether applying `override` to this handle is a no-op, safe, or a
    conflict with a decision somebody made since the file was written."""
    if _same_name(row.display_name, override.name):
        return "already"
    if row.is_owner:
        return "owner"
    if row.needs_review:
        return "safe"  # never hand-named: a review stub, or a Contacts cross-reference onto one
    if row.display_name == row.normalized_value:
        return "safe"  # still carrying the stub name S3 gave it
    if any(_same_name(row.display_name, prior) for prior in override.was):
        return "safe"  # the name the decision was made against
    return "conflict"


# --------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------

OverrideStatus = Literal["applied", "already", "unmatched", "conflict"]


@dataclass(frozen=True, slots=True)
class OverrideOutcome:
    override: IdentityOverride
    status: OverrideStatus
    detail: str
    """What ran (`renamed person 12 … -> …`, `merged person 40 into 12`,
    `assigned phone … to person 12`) or why nothing did."""
    forced: bool = False
    """Applied despite a conflict, because `force=True`."""


@dataclass(frozen=True, slots=True)
class ApplyOverridesResult:
    outcomes: tuple[OverrideOutcome, ...]
    invariant: InvariantReport
    dry_run: bool = False

    def count(self, status: OverrideStatus) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def applied(self) -> int:
        return self.count("applied")

    @property
    def already(self) -> int:
        return self.count("already")

    @property
    def unmatched(self) -> int:
        return self.count("unmatched")

    @property
    def conflicts(self) -> int:
        return self.count("conflict")

    @property
    def forced(self) -> int:
        return sum(1 for o in self.outcomes if o.forced)

    @property
    def summary(self) -> str:
        return (
            f"applied={self.applied} already={self.already} "
            f"unmatched={self.unmatched} conflicts={self.conflicts}"
        )


class _AmbiguousTarget(Exception):
    def __init__(self, name: str, count: int) -> None:
        super().__init__(name)
        self.name = name
        self.count = count


class _DryRunRollback(Exception):
    """Forces the outer transaction of a dry run to ROLLBACK; caught in
    `apply_overrides`, never allowed to escape this module."""

    def __init__(self, result: ApplyOverridesResult) -> None:
        self.result = result


def _find_target_person(
    cur: psycopg.Cursor[Any],
    *,
    name: str,
    exclude_person_id: int,
    group_forms: frozenset[tuple[str, str]],
) -> int | None:
    """The one other person already carrying `name`, or `None`. When
    several do, the one holding a handle this file lists under the same
    name is the target the file itself created; otherwise the file cannot
    say which person it means and nothing is guessed."""
    cur.execute(
        "SELECT person_id, display_name FROM person "
        "WHERE lower(display_name) = lower(%s) AND NOT is_owner AND person_id <> %s "
        "ORDER BY person_id",
        (name, exclude_person_id),
    )
    candidates = [
        int(pid) for pid, display_name in cur.fetchall() if _same_name(str(display_name), name)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    holders: list[int] = []
    for pid in candidates:
        cur.execute("SELECT normalized_value, kind FROM handle WHERE person_id = %s", (pid,))
        if any((str(v), str(k)) in group_forms for v, k in cur.fetchall()):
            holders.append(pid)
    if len(holders) == 1:
        return holders[0]
    raise _AmbiguousTarget(name, len(candidates))


def _apply_to_handle(
    conn: psycopg.Connection,
    cur: psycopg.Cursor[Any],
    row: _HandleRow,
    override: IdentityOverride,
    group_forms: frozenset[tuple[str, str]],
) -> str | None:
    """Run the one curation operation this handle needs, through the same
    function the interactive command calls. Returns a description of what
    ran, or `None` when an earlier operation of this run already put the
    handle where the decision wants it."""
    # Re-read: an earlier decision (or the legacy form of this one) may have
    # renamed or merged this handle's person since the plan was made.
    fresh = _lookup_handles(cur, ((row.normalized_value, row.kind),))
    if not fresh:
        return None
    row = fresh[0]
    if _same_name(row.display_name, override.name):
        return None

    target = _find_target_person(
        cur, name=override.name, exclude_person_id=row.person_id, group_forms=group_forms
    )
    if target is None:
        rename_person(conn, person_id=row.person_id, display_name=override.name)
        return f"renamed person {row.person_id} {row.display_name!r} -> {override.name!r} ({row.label})"

    cur.execute(
        "SELECT count(*) FROM handle WHERE person_id = %s AND handle_id <> %s",
        (row.person_id, row.handle_id),
    )
    others = cur.fetchone()
    if others is None or int(others[0]) == 0:
        merge_persons(conn, keep_person_id=target, absorb_person_id=row.person_id)
        return (
            f"merged person {row.person_id} {row.display_name!r} into person {target} "
            f"{override.name!r} ({row.label})"
        )
    assign_handle(conn, normalized_value=row.normalized_value, kind=row.kind, person_id=target)
    return (
        f"assigned {row.label} from person {row.person_id} {row.display_name!r} "
        f"to person {target} {override.name!r}"
    )


def _apply_one(
    conn: psycopg.Connection,
    cur: psycopg.Cursor[Any],
    override: IdentityOverride,
    *,
    default_region: str,
    force: bool,
    group_forms: frozenset[tuple[str, str]],
) -> OverrideOutcome:
    forms = identifier_forms(override, default_region)
    rows = _lookup_handles(cur, forms)
    if not rows:
        return OverrideOutcome(
            override,
            "unmatched",
            f"no handle in the index for {override.kind} {override.value!r} -> {override.name!r}",
        )

    classified = [(row, _classify(row, override)) for row in rows]
    owner_rows = [row for row, c in classified if c == "owner"]
    if owner_rows:
        return OverrideOutcome(
            override,
            "conflict",
            f"{owner_rows[0].label} belongs to the owner person — the owner is never renamed "
            f"or merged by a decisions file",
        )
    conflicts = [row for row, c in classified if c == "conflict"]
    if conflicts and not force:
        described = "; ".join(
            f"{row.label} is on person {row.person_id} named {row.display_name!r}, "
            f"which is neither the decision's {override.name!r} nor a name it lists as prior"
            for row in conflicts
        )
        return OverrideOutcome(override, "conflict", described + " (skipped; --force applies it)")

    pending = [row for row, c in classified if c != "already"]
    if not pending:
        return OverrideOutcome(
            override, "already", f"{rows[0].label} already on a person named {override.name!r}"
        )

    try:
        with conn.transaction():
            details = [_apply_to_handle(conn, cur, row, override, group_forms) for row in pending]
    except _AmbiguousTarget as exc:
        return OverrideOutcome(
            override,
            "conflict",
            f"{exc.count} persons are already named {exc.name!r}; the decision for "
            f"{override.kind} {override.value!r} cannot say which one it means — merge or "
            f"rename them by hand first (never forced)",
        )
    ran = [d for d in details if d is not None]
    if not ran:
        return OverrideOutcome(
            override, "already", f"{rows[0].label} already on a person named {override.name!r}"
        )
    return OverrideOutcome(override, "applied", "; ".join(ran), forced=bool(conflicts))


def _apply_body(
    conn: psycopg.Connection,
    overrides: Sequence[IdentityOverride],
    *,
    default_region: str,
    force: bool,
) -> ApplyOverridesResult:
    groups: dict[str, set[tuple[str, str]]] = {}
    for override in overrides:
        groups.setdefault(override.name.casefold(), set()).update(
            identifier_forms(override, default_region)
        )
    outcomes: list[OverrideOutcome] = []
    with conn.transaction(), conn.cursor() as cur:
        for override in overrides:
            outcomes.append(
                _apply_one(
                    conn,
                    cur,
                    override,
                    default_region=default_region,
                    force=force,
                    group_forms=frozenset(groups[override.name.casefold()]),
                )
            )
    invariant = compute_invariant_report(conn)
    return ApplyOverridesResult(outcomes=tuple(outcomes), invariant=invariant)


def apply_overrides(
    conn: psycopg.Connection,
    *,
    overrides: Sequence[IdentityOverride],
    default_region: str,
    dry_run: bool = False,
    force: bool = False,
) -> ApplyOverridesResult:
    """Replay `overrides` in order (see the module docstring for the
    per-decision outcomes), as one transaction that ends with the S3
    invariant report. `dry_run=True` performs the same work inside a
    transaction it then rolls back, so the outcomes are exactly what a
    real run would produce and nothing is written."""
    if not dry_run:
        return _apply_body(conn, overrides, default_region=default_region, force=force)
    try:
        with conn.transaction():
            result = _apply_body(conn, overrides, default_region=default_region, force=force)
            raise _DryRunRollback(result)
    except _DryRunRollback as sentinel:
        return replace(sentinel.result, dry_run=True)


# --------------------------------------------------------------------------
# exporting
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExportOverridesResult:
    file: IdentityOverridesFile
    exported: int
    """Decisions taken from the current person table."""
    carried_forward: int
    """Previous decisions the index cannot currently express (handle absent,
    or back on an unreviewed stub), kept verbatim."""


def _current_decisions(cur: psycopg.Cursor[Any]) -> list[tuple[str, str, str]]:
    cur.execute(
        """
        SELECT h.normalized_value, h.kind, p.display_name
        FROM handle h JOIN person p ON p.person_id = h.person_id
        WHERE NOT p.is_owner
          AND (NOT p.needs_review
               OR EXISTS (SELECT 1 FROM handle h2
                          WHERE h2.person_id = p.person_id AND h2.handle_id <> h.handle_id))
        ORDER BY lower(p.display_name), p.person_id, h.kind, h.normalized_value
        """
    )
    return [(str(v), str(k), str(n)) for v, k, n in cur.fetchall()]


def export_overrides(
    conn: psycopg.Connection,
    *,
    default_region: str,
    previous: IdentityOverridesFile | None = None,
    today: dt.date | None = None,
) -> ExportOverridesResult:
    """Build the decisions file for the current person table, carrying a
    previous file's `was`/`why` and its currently-inexpressible decisions
    forward (module docstring, "Export"). Pure read of Postgres."""
    with conn.cursor() as cur:
        current = _current_decisions(cur)
        cur.execute("SELECT normalized_value, kind FROM handle")
        existing_forms = {(str(v), str(k)) for v, k in cur.fetchall()}

    by_recorded: dict[tuple[str, str], IdentityOverride] = {}
    by_normalized: dict[tuple[str, str], IdentityOverride] = {}
    if previous is not None:
        for override in previous.overrides:
            recorded, *normalized = identifier_forms(override, default_region)
            by_recorded[recorded] = override
            for form in normalized:
                by_normalized.setdefault(form, override)

    def prior_for(value: str, kind: str) -> IdentityOverride | None:
        prior = by_recorded.get((value, kind))
        if prior is not None:
            return prior
        # A decision recorded in a legacy form (a filter-tagged number that sat
        # in kind='unknown') follows its clean handle only once the legacy row
        # is gone. While both rows exist they may sit on different persons, and
        # the clean row is exported exactly as the index holds it.
        candidate = by_normalized.get((value, kind))
        if candidate is not None and (candidate.value, candidate.kind) not in existing_forms:
            return candidate
        return None

    exported: list[IdentityOverride] = []
    superseded: set[int] = set()
    for value, kind, name in current:
        prior = prior_for(value, kind)
        if prior is None:
            exported.append(IdentityOverride(value=value, kind=kind, name=name))
            continue
        superseded.add(id(prior))
        was = prior.was
        if not _same_name(prior.name, name):
            was = (prior.name, *(w for w in prior.was if not _same_name(w, prior.name)))
        exported.append(IdentityOverride(value=value, kind=kind, name=name, was=was, why=prior.why))

    carried = [o for o in (previous.overrides if previous else ()) if id(o) not in superseded]
    exported_count = len(exported)
    exported.extend(carried)
    exported.sort(key=lambda o: (o.name.casefold(), o.kind, o.value))

    file = IdentityOverridesFile(
        overrides=tuple(exported),
        note=previous.note if previous is not None and previous.note else DEFAULT_NOTE,
        decided=(today or dt.date.today()).isoformat(),
    )
    return ExportOverridesResult(file=file, exported=exported_count, carried_forward=len(carried))


__all__ = [
    "DEFAULT_NOTE",
    "HANDLE_KINDS",
    "ApplyOverridesResult",
    "ExportOverridesResult",
    "IdentityOverride",
    "IdentityOverridesFile",
    "OverrideOutcome",
    "OverrideStatus",
    "apply_overrides",
    "export_overrides",
    "identifier_forms",
    "load_overrides",
    "parse_overrides",
    "write_overrides",
]
