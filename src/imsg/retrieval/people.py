"""Person-filter resolution (SPEC §9.4 step 1, D6, §10.1).

"Person resolution is exact `short_name` -> exact display name ->
fuzzy candidates; a non-unique fuzzy match returns `PERSON_AMBIGUOUS`
with candidates, never a silent pick."

Reading applied here (judgment call, since the spec does not spell out
what a *unique* fuzzy match does): the fuzzy tier never auto-resolves,
even when it finds exactly one plausible candidate — "never a silent
pick" is read literally. A single close-but-not-exact fuzzy match still
comes back as `PERSON_AMBIGUOUS` with one candidate, forcing the caller
to retry with that candidate's exact `short_name`. Only the first two
tiers (exact `short_name`, exact `display_name`) resolve automatically,
and only when they are unique.

D6 additionally requires near-match suggestions to respect the
effective access scope ("public person near-match suggestions are
themselves filtered by the effective access scope") — `_fuzzy_pool`
below applies `access.segment_eligibility_predicate`-equivalent
person-level filtering (via `allowlist_person.text_allowed`) whenever
`context.scope != "full"`. The local surface is always full scope, so
this never actually filters anything in this build; it exists so a
later public-surface build gets the D6 behavior for free.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

from imsg.retrieval.access import AccessContext
from imsg.retrieval.errors import (
    InvalidArgumentError,
    PersonAmbiguousError,
    PersonCandidate,
    PersonNotFoundError,
)

if TYPE_CHECKING:
    import psycopg

MAX_CANDIDATES = 5
"""SPEC §10.1: "include up to 5 near-matches"."""

FUZZY_MIN_SCORE = 0.5
"""`difflib.SequenceMatcher.ratio()` floor for a fuzzy candidate to be
worth suggesting at all — below this, nothing is close enough to be a
plausible "did you mean" and the result is `PersonNotFoundError` with
no candidates rather than noise."""


def _fuzzy_pool(
    conn: psycopg.Connection, context: AccessContext
) -> list[tuple[int, str, str]]:
    """`(person_id, short_name, display_name)` for every person the
    fuzzy tier is allowed to suggest, scope-filtered per D6."""
    with conn.cursor() as cur:
        if context.scope == "full":
            cur.execute("SELECT person_id, short_name, display_name FROM person")
        else:
            cur.execute(
                """
                SELECT p.person_id, p.short_name, p.display_name
                FROM person p
                JOIN allowlist_person ap ON ap.person_id = p.person_id
                WHERE ap.text_allowed
                """
            )
        return [(int(pid), sn, dn) for pid, sn, dn in cur.fetchall()]


def _fuzzy_candidates(
    conn: psycopg.Connection, context: AccessContext, query: str
) -> tuple[PersonCandidate, ...]:
    lowered = query.lower()
    scored: list[tuple[float, PersonCandidate]] = []
    for _person_id, short_name, display_name in _fuzzy_pool(conn, context):
        score = max(
            difflib.SequenceMatcher(None, lowered, short_name.lower()).ratio(),
            difflib.SequenceMatcher(None, lowered, display_name.lower()).ratio(),
        )
        if score >= FUZZY_MIN_SCORE:
            scored.append((score, PersonCandidate(short_name=short_name, display_name=display_name)))
    scored.sort(key=lambda t: -t[0])
    return tuple(c for _, c in scored[:MAX_CANDIDATES])


def resolve_person(conn: psycopg.Connection, context: AccessContext, query: str) -> int:
    """Resolve one `people` filter entry (SPEC §10.2 `search_messages`
    schema) to a `person_id`.

    Raises `InvalidArgumentError` for an empty entry,
    `PersonNotFoundError`/`PersonAmbiguousError` per the ladder
    described in this module's docstring.
    """
    stripped = query.strip()
    if not stripped:
        raise InvalidArgumentError("a 'people' filter entry must not be empty")

    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE short_name = %s", (stripped,))
        row = cur.fetchone()
        if row is not None:
            return int(row[0])

        cur.execute("SELECT person_id FROM person WHERE display_name = %s", (stripped,))
        exact_display_matches = cur.fetchall()

    if len(exact_display_matches) == 1:
        return int(exact_display_matches[0][0])
    if len(exact_display_matches) > 1:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT short_name, display_name FROM person WHERE display_name = %s "
                "ORDER BY short_name LIMIT %s",
                (stripped, MAX_CANDIDATES),
            )
            candidates = tuple(
                PersonCandidate(short_name=sn, display_name=dn) for sn, dn in cur.fetchall()
            )
        raise PersonAmbiguousError(stripped, candidates)

    fuzzy = _fuzzy_candidates(conn, context, stripped)
    if not fuzzy:
        raise PersonNotFoundError(stripped, ())
    raise PersonAmbiguousError(stripped, fuzzy)


__all__ = ["FUZZY_MIN_SCORE", "MAX_CANDIDATES", "resolve_person"]
