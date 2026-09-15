"""Name the review stubs a Contacts-less import left behind (`imsg identity rematch-stubs`).

`run_identity` resolves only source handles that have no
`source_handle_resolution` row, so a handle first seen while the process
had no Contacts grant becomes a review stub named after its own handle —
and no later import, with the grant or without, ever looks at it again.
Contacts access on macOS is granted per application: an import run over
SSH or by an agent has none, an import run from the owner's terminal
does, and the same index sees both. Measured 2026-09-15: one Contacts-less
run created 166 persons, every one still raw-named after a Contacts-enabled
import (which reported loading 1,055 cards) revisited none of them. This
will recur whenever the import runs from an SSH or agent context.

This module is the missing second look. It takes every raw stub, asks the
same `ContactsIndex` the import builds the same question the import asks
(`find_unique`: a unique match only — duplicate-account, decorated and
contained-name cards agree, anything else is ambiguous and nothing is
guessed), and on a unique match names the person exactly as the import
names a fresh handle: the card's display name, a `short_name` slugged from
it, the card's organization, the stub note cleared, `needs_review`
cleared. The rename goes through `rename_person`, so `person.updated_at`
moves the way a hand rename moves it.

What a stub is
--------------
A person that is `needs_review`, is not the owner, holds at least one
handle, and whose `display_name` is exactly the normalized value of one of
its own handles — the name `_find_or_create_person` gives a person it could
not name. Nothing else in the schema records that a person was named by
hand: `export_overrides` treats "not `needs_review`" (or holding several
handles) as curated, and `apply_overrides` treats "still named after its
own handle" as not yet decided, so this module draws the same line. A hand
rename, a unique Contacts match at import, and `apply-overrides` all clear
`needs_review` and give a name that is not a handle, so none of them can be
mistaken for a stub. A person holding several handles (a merge or assign
put them together) is named only when every handle finds the same card —
Contacts has to agree with the human who merged them.

Never touched: reviewed persons, the owner, a stub whose handle no card
carries (`unmatched`), a stub whose cards disagree (`ambiguous`), and a
stub whose only match is a card with no name of its own (the import's
`Unnamed Contact` placeholder would clear `needs_review` while naming
nobody — such a stub stays a stub, counted as `unmatched`).

Contacts is the whole point, so its absence is an error, not a degrade:
`identity.contacts_import: false`, a missing or unauthorized grant
(`ContactsAccessDeniedError`, straight from the importer), and an
authorized store that returns no cards all raise before the database is
read. A dry run performs the real work inside a transaction it then rolls
back — the same pattern `run_identity` and `apply_overrides` use — so its
counts are exactly what a real run would produce.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import psycopg

from imsg.config.schema import Config
from imsg.errors import IdentityError
from imsg.stages.identity import (
    UNNAMED_CONTACT_DISPLAY_NAME,
    ContactsAccessDeniedError,
    ContactsImporterFn,
    ContactsIndex,
    MatchStatus,
    _default_contacts_importer,
    _generate_unique_short_name,
    rename_person,
)


@dataclass(frozen=True, slots=True)
class StubOutcome:
    """One raw stub and what Contacts said about it."""

    person_id: int
    display_name: str
    """The name the stub carried: the normalized value of one of its handles."""
    handles: tuple[tuple[str, str], ...]
    """`(normalized_value, kind)` for every handle the person holds."""
    status: MatchStatus
    new_display_name: str | None = None
    new_short_name: str | None = None


@dataclass(frozen=True, slots=True)
class RematchStubsResult:
    outcomes: tuple[StubOutcome, ...]
    contacts_loaded: int
    dry_run: bool = False

    def count(self, status: MatchStatus) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == status)

    @property
    def stubs(self) -> int:
        return len(self.outcomes)

    @property
    def matched(self) -> int:
        return self.count("matched")

    @property
    def ambiguous(self) -> int:
        return self.count("ambiguous")

    @property
    def unmatched(self) -> int:
        return self.count("unmatched")

    @property
    def summary(self) -> str:
        return (
            f"stubs={self.stubs} matched={self.matched} "
            f"ambiguous={self.ambiguous} unmatched={self.unmatched}"
        )


class _DryRunRollback(Exception):
    """Forces the outer transaction of a dry run to ROLLBACK; caught in
    `run_rematch_stubs`, never allowed to escape this module."""

    def __init__(self, result: RematchStubsResult) -> None:
        self.result = result


@dataclass(frozen=True, slots=True)
class _Stub:
    person_id: int
    display_name: str
    handles: tuple[tuple[str, str], ...]


def _load_stubs(cur: psycopg.Cursor[Any]) -> list[_Stub]:
    """Every raw stub (module docstring, "What a stub is") with all of its handles."""
    cur.execute(
        """
        SELECT p.person_id, p.display_name,
               array_agg(h.normalized_value ORDER BY h.handle_id),
               array_agg(h.kind::text ORDER BY h.handle_id)
        FROM person p
        JOIN handle h ON h.person_id = p.person_id
        WHERE p.needs_review AND NOT p.is_owner
        GROUP BY p.person_id, p.display_name
        HAVING bool_or(h.normalized_value = p.display_name)
        ORDER BY p.person_id
        """
    )
    return [
        _Stub(
            person_id=int(person_id),
            display_name=str(display_name),
            handles=tuple(
                zip((str(v) for v in values), (str(k) for k in kinds), strict=True)
            ),
        )
        for person_id, display_name, values, kinds in cur.fetchall()
    ]


def _rematch_body(
    conn: psycopg.Connection, index: ContactsIndex, *, contacts_loaded: int
) -> RematchStubsResult:
    outcomes: list[StubOutcome] = []
    with conn.transaction(), conn.cursor() as cur:
        for stub in _load_stubs(cur):
            found = index.lookup_all(stub.handles)
            contact = found.contact
            if found.status != "matched" or contact is None:
                outcomes.append(
                    StubOutcome(stub.person_id, stub.display_name, stub.handles, found.status)
                )
                continue
            if contact.display_name == UNNAMED_CONTACT_DISPLAY_NAME:
                # The card names nobody. The import would store the placeholder
                # and clear needs_review; here that would drop a stub off the
                # review worklist without naming it, so it stays a stub.
                outcomes.append(
                    StubOutcome(stub.person_id, stub.display_name, stub.handles, "unmatched")
                )
                continue

            # Exactly what `_create_person` gives a Contacts-named person, applied
            # to the existing row: name, a fresh slug, organization, no stub note,
            # off the review worklist. `rename_person` is the same path a hand
            # rename takes, `updated_at` bump included.
            short_name = _generate_unique_short_name(cur, contact.display_name)
            rename_person(
                conn,
                person_id=stub.person_id,
                display_name=contact.display_name,
                short_name=short_name,
            )
            cur.execute(
                "UPDATE person SET organization = %s, notes = NULL WHERE person_id = %s",
                (contact.organization, stub.person_id),
            )
            outcomes.append(
                StubOutcome(
                    stub.person_id,
                    stub.display_name,
                    stub.handles,
                    "matched",
                    new_display_name=contact.display_name,
                    new_short_name=short_name,
                )
            )
    return RematchStubsResult(outcomes=tuple(outcomes), contacts_loaded=contacts_loaded)


def run_rematch_stubs(
    *,
    conn: psycopg.Connection,
    config: Config,
    contacts_importer: ContactsImporterFn = _default_contacts_importer,
    dry_run: bool = False,
) -> RematchStubsResult:
    """Name every raw stub that Contacts can name uniquely (module docstring).

    Raises — before touching the database — when Contacts is switched off
    in config, when the process has no Contacts grant
    (`ContactsAccessDeniedError`), or when Contacts returns no cards: there
    is nothing this function can do without Contacts, and reporting
    `matched=0` as if it had looked would be the silent degrade this
    command exists to replace. `dry_run=True` runs the real logic inside a
    transaction it then rolls back, so the counts are exact and nothing is
    written.
    """
    if not config.identity.contacts_import:
        raise IdentityError(
            "identity.contacts_import is false in config — rematch-stubs names review stubs "
            "from Contacts and has nothing to do without it; nothing was changed"
        )
    try:
        records = contacts_importer(config.identity.default_region)
    except ContactsAccessDeniedError as exc:
        raise ContactsAccessDeniedError(
            f"rematch-stubs needs Contacts and could not use it; nothing was changed. {exc}"
        ) from exc
    if not records:
        raise IdentityError(
            "Contacts access is authorized but returned 0 cards, so no stub could be matched; "
            "nothing was changed. Check that the account holding your contacts has Contacts "
            "enabled and has synced."
        )
    index = ContactsIndex(records)

    if not dry_run:
        return _rematch_body(conn, index, contacts_loaded=len(records))
    try:
        with conn.transaction():
            result = _rematch_body(conn, index, contacts_loaded=len(records))
            raise _DryRunRollback(result)
    except _DryRunRollback as sentinel:
        return replace(sentinel.result, dry_run=True)


__all__ = [
    "RematchStubsResult",
    "StubOutcome",
    "run_rematch_stubs",
]
