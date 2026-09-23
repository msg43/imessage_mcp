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
themselves filtered by the effective access scope"). Under `allowlist`
scope every tier here — exact `short_name`, exact `display_name`, the
candidates of an ambiguous display name, and the fuzzy suggestions —
looks only at `imsg.retrieval.access.visible_person_ids`: people who
appear in an eligible chat. A person outside that set is reported as
`PERSON_NOT_FOUND`, exactly like a name that exists nowhere, so a public
caller cannot learn who is in the corpus by trying names.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from imsg.retrieval.access import RequestScope, visible_person_ids
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


def _within(visible: frozenset[int] | None) -> tuple[str, dict[str, object]]:
    """The `AND ...` clause (and its parameter) limiting a `person` query
    to `visible`; nothing under `full` scope."""
    if visible is None:
        return "", {}
    return " AND person_id = ANY(%(visible)s::bigint[])", {"visible": sorted(visible)}


def _fuzzy_candidates(
    conn: psycopg.Connection, query: str, visible: frozenset[int] | None
) -> tuple[PersonCandidate, ...]:
    within, params = _within(visible)
    with conn.cursor() as cur:
        cur.execute(f"SELECT short_name, display_name FROM person WHERE TRUE{within}", params)
        pool = [(str(sn), str(dn)) for sn, dn in cur.fetchall()]
    lowered = query.lower()
    scored: list[tuple[float, PersonCandidate]] = []
    for short_name, display_name in pool:
        score = max(
            difflib.SequenceMatcher(None, lowered, short_name.lower()).ratio(),
            difflib.SequenceMatcher(None, lowered, display_name.lower()).ratio(),
        )
        if score >= FUZZY_MIN_SCORE:
            scored.append((score, PersonCandidate(short_name=short_name, display_name=display_name)))
    scored.sort(key=lambda t: -t[0])
    return tuple(c for _, c in scored[:MAX_CANDIDATES])


def _resolve_one(
    conn: psycopg.Connection, query: str, visible: frozenset[int] | None
) -> int:
    stripped = query.strip()
    if not stripped:
        raise InvalidArgumentError("a 'people' filter entry must not be empty")

    within, visible_params = _within(visible)
    params: dict[str, object] = {"q": stripped, "n": MAX_CANDIDATES, **visible_params}
    with conn.cursor() as cur:
        cur.execute(f"SELECT person_id FROM person WHERE short_name = %(q)s{within}", params)
        row = cur.fetchone()
        if row is not None:
            return int(row[0])

        cur.execute(f"SELECT person_id FROM person WHERE display_name = %(q)s{within}", params)
        exact_display_matches = cur.fetchall()

        if len(exact_display_matches) == 1:
            return int(exact_display_matches[0][0])
        if len(exact_display_matches) > 1:
            cur.execute(
                f"SELECT short_name, display_name FROM person WHERE display_name = %(q)s{within} "
                "ORDER BY short_name LIMIT %(n)s",
                params,
            )
            candidates = tuple(
                PersonCandidate(short_name=sn, display_name=dn) for sn, dn in cur.fetchall()
            )
            raise PersonAmbiguousError(stripped, candidates)

    fuzzy = _fuzzy_candidates(conn, stripped, visible)
    if not fuzzy:
        raise PersonNotFoundError(stripped, ())
    raise PersonAmbiguousError(stripped, fuzzy)


def resolve_people(
    conn: psycopg.Connection, scope: RequestScope, queries: Sequence[str]
) -> tuple[int, ...]:
    """Resolve every `people` filter entry (SPEC §10.2 `search_messages`
    schema) to a `person_id`, in order, looking up the scope's visible
    people once for all of them.

    Raises `InvalidArgumentError` for an empty entry,
    `PersonNotFoundError`/`PersonAmbiguousError` per the ladder
    described in this module's docstring.
    """
    if not queries:
        return ()
    visible = visible_person_ids(conn, scope)
    return tuple(_resolve_one(conn, q, visible) for q in queries)


def resolve_person(conn: psycopg.Connection, scope: RequestScope, query: str) -> int:
    """Resolve one `people` filter entry; see `resolve_people`."""
    return resolve_people(conn, scope, [query])[0]


__all__ = ["FUZZY_MIN_SCORE", "MAX_CANDIDATES", "resolve_people", "resolve_person"]
