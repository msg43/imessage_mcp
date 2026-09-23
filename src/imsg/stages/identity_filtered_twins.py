"""Merge the persons an early import split on iOS filter tags (`imsg identity merge-filtered-twins`).

iOS records a filtered sender as `<id>(filtered)` or `<id>(smsft…)`. Since
2026-08-15 `normalize_handle` strips the tag (`strip_ios_filter_suffix`),
so a tagged source handle and its untagged form share one canonical handle
and one person. But `run_identity` resolves only source handles that have
no `source_handle_resolution` row, so every tagged source handle resolved
BEFORE that fix still points at a canonical handle that carries the tag
(`kind='unknown'` for a number or short code, `'email'` for an address) and
at a review stub of its own. No later run looks at them again.

Measured read-only on the production index on 2026-09-23: 3,448 such
canonical handles on 3,448 persons, every person created by the first
identity import on 2026-08-15, hours before the fix. 2,938 of the handles
have an untagged twin on a different person. A person-scoped question about
any of those senders finds only part of what they sent.

What the repair does
--------------------
A *legacy handle* is a canonical handle that tagged source handles resolve
to although `normalize_handle` now gives them a different
`(normalized_value, kind)`, their *clean form*. For each clean form:

1. **Target handle.** The handle that already holds the clean form (the
   *twin*), if there is one. Otherwise the legacy handle with the lowest id
   is rewritten to the clean form, and if its person is still the review
   stub named after the tagged value, the stub is renamed to the clean
   value with a fresh `short_name`. It stays a stub, under the name S3
   would give it today.
2. **Persons.** For every other legacy handle of the clean form whose person
   is not the target's person, the two persons are merged with
   `merge_persons`, the machinery `imsg identity merge` uses. The target's
   person is kept, unless only the legacy person carries a curated name, in
   which case the legacy person is kept; a curated name is never merged
   away. A *curated name* is any display name other than the normalized
   value of one of the person's own handles (the name S3 gives a stub) and
   other than the `Unnamed Contact` placeholder: a hand rename, a
   decisions-file rename, or a unique Contacts match.
   Two kinds of pair are refused, reported, and left exactly as they are:
   both persons carry curated names that differ, since a human has to
   choose; and pairs involving the owner, which is never merged
   automatically.
3. **Handles.** Every source handle of the legacy handle is repointed to the
   target handle. The legacy handle, which nothing resolves to any more, is
   then removed: it has been merged into its twin.

Nothing else changes. No `message`, `tapback`, `attachment`, `chat` or
`source_handle` row is deleted, and raw source-handle values stay exactly
as extraction recorded them. A merge repoints message and tapback senders
and moves `message.updated_at` in the chats it marks, as every merge does.
No message loses its sender and no chat loses a participant (D12):
resolutions are repointed, legacy handles are rewritten or merged into
their twin, and persons are merged.

Allowlist rows follow `merge_persons`: the kept person's row, or its
absence, stands, narrowed by the absorbed person's row when both exist. An
absorbed person's flags never widen access.

Every merge and every stub rename marks the chats whose rendered segments
name the persons involved for re-segmentation (`merge_persons`,
`rename_person`), so the run must be followed by `imsg segment` and then
`imsg embed`. Repointing a resolution or rewriting a handle value changes
no rendered text by itself (segments carry `display_name` and
`short_name`, never handle values), so neither marks anything on its own.

Untagged source handles whose canonical handle also differs from today's
normalization (after a `phonenumbers` metadata update, say) are the same
kind of defect. They are counted in `other_stale_source_handles` and left
alone: the owner's decision (D13) covers the filter tags.

The run is one transaction and ends with the S3 invariant report. A dry run
does the same work inside a transaction it then rolls back, so its counts
are exactly what a real run would produce.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import psycopg

from imsg.errors import IdentityError
from imsg.stages.identity import (
    UNNAMED_CONTACT_DISPLAY_NAME,
    InvariantReport,
    _generate_unique_short_name,
    compute_invariant_report,
    has_ios_filter_suffix,
    merge_persons,
    normalize_handle,
    rename_person,
)

RefusalReason = Literal["curated_conflict", "owner"]
"""Why a pair was left alone: `curated_conflict` — both persons carry
curated names that differ; `owner` — one of them is the owner person."""

CleanForm = tuple[str, str]
"""`(normalized_value, kind)` as `normalize_handle` produces it today."""


@dataclass(frozen=True, slots=True)
class RefusedPair:
    """A legacy person and a target person the repair did not merge. Neither
    person, nor the legacy handle, nor its source handles changed."""

    reason: RefusalReason
    clean_value: str
    clean_kind: str
    legacy_person_id: int
    legacy_display_name: str
    target_person_id: int
    target_display_name: str


@dataclass(frozen=True, slots=True)
class FilteredTwinsResult:
    tagged_source_handles: int
    """Resolved source handles whose raw value carries an iOS filter tag."""
    legacy_source_handles: int
    """Of those, the ones that resolve to a legacy handle."""
    legacy_handles: int
    """Distinct legacy handles those resolve to."""
    source_handles_repointed: int
    handles_rewritten: int
    """Legacy handles with no twin, rewritten to their clean form."""
    legacy_handles_merged: int
    """Legacy handles merged into a target: sources repointed, row removed."""
    persons_merged: int
    stubs_renamed: int
    """Stubs named after a tagged value, renamed to the clean value."""
    refused: tuple[RefusedPair, ...]
    allowlist_rows_absorbed: int
    """Merged-away persons that had an allowlist row; it was removed with them."""
    allowlist_rows_narrowed: int
    """Kept persons whose allowlist flags a merged-away person's row turned off."""
    other_stale_source_handles: int
    """Untagged source handles whose canonical handle is not today's
    normalization either. Counted, not repaired."""
    invariant: InvariantReport
    chats_marked_dirty: int = 0
    """Distinct chats marked for re-segmentation. `imsg segment` and then
    `imsg embed` must follow a run that reports a non-zero count."""
    dry_run: bool = False

    @property
    def refused_curated(self) -> int:
        return sum(1 for pair in self.refused if pair.reason == "curated_conflict")

    @property
    def refused_owner(self) -> int:
        return sum(1 for pair in self.refused if pair.reason == "owner")

    @property
    def summary(self) -> str:
        return (
            f"legacy_handles={self.legacy_handles} persons_merged={self.persons_merged} "
            f"handles_rewritten={self.handles_rewritten} "
            f"legacy_handles_merged={self.legacy_handles_merged} "
            f"source_handles_repointed={self.source_handles_repointed} "
            f"stubs_renamed={self.stubs_renamed} refused_curated={self.refused_curated} "
            f"refused_owner={self.refused_owner}"
        )


class _DryRunRollback(Exception):
    """Forces the outer transaction of a dry run to ROLLBACK; caught in
    `run_merge_filtered_twins`, never allowed to escape this module."""

    def __init__(self, result: FilteredTwinsResult) -> None:
        self.result = result


@dataclass(frozen=True, slots=True)
class _Scan:
    tagged_source_handles: int
    legacy_source_handles: int
    other_stale_source_handles: int
    groups: tuple[tuple[CleanForm, tuple[int, ...]], ...]
    """`(clean form, legacy handle ids ascending)`, ordered by lowest id."""


def _scan(cur: psycopg.Cursor[Any], default_region: str) -> _Scan:
    """Every resolved source handle, classified against today's normalizer."""
    cur.execute(
        """
        SELECT sh.source_handle_id, sh.raw_value, h.handle_id, h.normalized_value, h.kind::text
        FROM source_handle sh
        JOIN source_handle_resolution shr ON shr.source_handle_id = sh.source_handle_id
        JOIN handle h ON h.handle_id = shr.handle_id
        ORDER BY sh.source_handle_id
        """
    )
    tagged = 0
    legacy_sources = 0
    other_stale = 0
    # Every resolution a handle receives, as (is_legacy, clean form), so a
    # legacy handle can be checked to carry nothing else before it is merged.
    by_handle: dict[int, list[tuple[bool, CleanForm]]] = {}
    for _source_handle_id, raw_value, handle_id, value, kind in cur.fetchall():
        clean = normalize_handle(str(raw_value), default_region)
        stale = clean != (str(value), str(kind))
        is_tagged = has_ios_filter_suffix(str(raw_value))
        tagged += is_tagged
        legacy_sources += is_tagged and stale
        other_stale += stale and not is_tagged
        by_handle.setdefault(int(handle_id), []).append((is_tagged and stale, clean))

    groups: dict[CleanForm, list[int]] = {}
    for handle_id, resolutions in sorted(by_handle.items()):
        if not any(is_legacy for is_legacy, _ in resolutions):
            continue
        forms = {clean for _, clean in resolutions}
        if not all(is_legacy for is_legacy, _ in resolutions) or len(forms) != 1:
            # Impossible for rows the pre-fix normalizer wrote: it produced a
            # tagged value only from a tagged raw value, and two raw values
            # reach one tagged value only when they differ in case or
            # surrounding whitespace, which the clean form ignores too.
            # Refuse rather than guess, so nothing is written.
            raise IdentityError(
                f"handle {handle_id} is resolved from source handles that do not share one "
                f"clean form (or include untagged ones); merge-filtered-twins refuses to guess. "
                f"Nothing was changed."
            )
        groups.setdefault(forms.pop(), []).append(handle_id)
    ordered = tuple(
        sorted(((form, tuple(ids)) for form, ids in groups.items()), key=lambda group: group[1][0])
    )
    return _Scan(
        tagged_source_handles=tagged,
        legacy_source_handles=legacy_sources,
        other_stale_source_handles=other_stale,
        groups=ordered,
    )


@dataclass(frozen=True, slots=True)
class _Person:
    person_id: int
    display_name: str
    is_owner: bool
    curated: bool


def _person_of_handle(cur: psycopg.Cursor[Any], handle_id: int) -> _Person:
    cur.execute(
        """
        SELECT p.person_id, p.display_name, p.is_owner,
               EXISTS (SELECT 1 FROM handle own
                       WHERE own.person_id = p.person_id
                         AND own.normalized_value = p.display_name)
        FROM handle h JOIN person p ON p.person_id = h.person_id
        WHERE h.handle_id = %s
        """,
        (handle_id,),
    )
    row = cur.fetchone()
    assert row is not None, handle_id
    person_id, display_name, is_owner, named_after_own_handle = row
    curated = not (
        bool(is_owner)
        or bool(named_after_own_handle)
        or str(display_name) == UNNAMED_CONTACT_DISPLAY_NAME
    )
    return _Person(int(person_id), str(display_name), bool(is_owner), curated)


def _choose(legacy: _Person, target: _Person) -> tuple[int, int] | RefusalReason:
    """`(keep, absorb)` for a pair, or why the pair must be left alone."""
    if legacy.is_owner or target.is_owner:
        return "owner"
    if legacy.curated and target.curated:
        if legacy.display_name.casefold() != target.display_name.casefold():
            return "curated_conflict"
        return target.person_id, legacy.person_id
    if legacy.curated:
        return legacy.person_id, target.person_id
    return target.person_id, legacy.person_id


def _allowlist_effect(cur: psycopg.Cursor[Any], keep: int, absorb: int) -> tuple[bool, bool]:
    """`(absorbed person has a row, the merge turns a kept flag off)`,
    read before `merge_persons` applies its allowlist rule."""
    cur.execute(
        "SELECT person_id, text_allowed, attachments_allowed FROM allowlist_person "
        "WHERE person_id IN (%s, %s)",
        (keep, absorb),
    )
    rows = {int(pid): (bool(text), bool(attachments)) for pid, text, attachments in cur.fetchall()}
    if absorb not in rows:
        return False, False
    if keep not in rows:
        return True, False
    kept, absorbed = rows[keep], rows[absorb]
    return True, any(k and not a for k, a in zip(kept, absorbed, strict=True))


def _handle_id(cur: psycopg.Cursor[Any], form: CleanForm) -> int | None:
    cur.execute(
        "SELECT handle_id FROM handle WHERE normalized_value = %s AND kind = %s", form
    )
    row = cur.fetchone()
    return int(row[0]) if row is not None else None


def _repair_body(conn: psycopg.Connection, default_region: str) -> FilteredTwinsResult:
    dirty_chat_ids: set[int] = set()
    refused: list[RefusedPair] = []
    repointed = rewritten = legacy_merged = merged = renamed = 0
    allowlist_absorbed = allowlist_narrowed = 0

    with conn.transaction(), conn.cursor() as cur:
        scan = _scan(cur, default_region)
        for form, legacy_ids in scan.groups:
            clean_value, clean_kind = form
            pending = list(legacy_ids)
            target_id = _handle_id(cur, form)
            if target_id is None:
                target_id = pending.pop(0)
                cur.execute("SELECT normalized_value FROM handle WHERE handle_id = %s", (target_id,))
                row = cur.fetchone()
                assert row is not None
                tagged_value = str(row[0])
                cur.execute(
                    "UPDATE handle SET normalized_value = %s, kind = %s WHERE handle_id = %s",
                    (clean_value, clean_kind, target_id),
                )
                rewritten += 1
                person = _person_of_handle(cur, target_id)
                if not person.is_owner and person.display_name == tagged_value:
                    dirty_chat_ids |= rename_person(
                        conn,
                        person_id=person.person_id,
                        display_name=clean_value,
                        short_name=_generate_unique_short_name(cur, clean_value),
                        mark_reviewed=False,
                    )
                    renamed += 1

            for legacy_id in pending:
                legacy = _person_of_handle(cur, legacy_id)
                target = _person_of_handle(cur, target_id)
                if legacy.person_id != target.person_id:
                    choice = _choose(legacy, target)
                    if isinstance(choice, str):
                        refused.append(
                            RefusedPair(
                                reason=choice,
                                clean_value=clean_value,
                                clean_kind=clean_kind,
                                legacy_person_id=legacy.person_id,
                                legacy_display_name=legacy.display_name,
                                target_person_id=target.person_id,
                                target_display_name=target.display_name,
                            )
                        )
                        continue
                    keep, absorb = choice
                    had_row, narrowed = _allowlist_effect(cur, keep, absorb)
                    allowlist_absorbed += had_row
                    allowlist_narrowed += narrowed
                    dirty_chat_ids |= merge_persons(conn, keep_person_id=keep, absorb_person_id=absorb)
                    merged += 1
                cur.execute(
                    "UPDATE source_handle_resolution SET handle_id = %s WHERE handle_id = %s",
                    (target_id, legacy_id),
                )
                repointed += cur.rowcount
                cur.execute("DELETE FROM handle WHERE handle_id = %s", (legacy_id,))
                legacy_merged += 1

    invariant = compute_invariant_report(conn)
    return FilteredTwinsResult(
        tagged_source_handles=scan.tagged_source_handles,
        legacy_source_handles=scan.legacy_source_handles,
        legacy_handles=sum(len(ids) for _, ids in scan.groups),
        source_handles_repointed=repointed,
        handles_rewritten=rewritten,
        legacy_handles_merged=legacy_merged,
        persons_merged=merged,
        stubs_renamed=renamed,
        refused=tuple(refused),
        allowlist_rows_absorbed=allowlist_absorbed,
        allowlist_rows_narrowed=allowlist_narrowed,
        other_stale_source_handles=scan.other_stale_source_handles,
        invariant=invariant,
        chats_marked_dirty=len(dirty_chat_ids),
    )


def run_merge_filtered_twins(
    *, conn: psycopg.Connection, default_region: str, dry_run: bool = False
) -> FilteredTwinsResult:
    """Repair every legacy filter-tagged handle (module docstring), as one
    transaction ending with the S3 invariant report. `dry_run=True` does the
    same work inside a transaction it then rolls back: the counts are what a
    real run would produce, and nothing is written."""
    if not dry_run:
        return _repair_body(conn, default_region)
    try:
        with conn.transaction():
            result = _repair_body(conn, default_region)
            raise _DryRunRollback(result)
    except _DryRunRollback as sentinel:
        return replace(sentinel.result, dry_run=True)


__all__ = [
    "FilteredTwinsResult",
    "RefusalReason",
    "RefusedPair",
    "run_merge_filtered_twins",
]
