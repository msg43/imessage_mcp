"""The combined authorization/filter predicate (SPEC §9.4 steps 1-2):
"Build the authorization/filter predicate **once** (surface scope,
people, dates, attachment flag) and apply it to every candidate path —
public scope is enforced in this repository layer, not in individual
tools (§10.3a, D6)."

`compile_predicate` is that one place. It always folds in
`imsg.retrieval.access.segment_eligibility_predicate`, so every caller
gets scope enforcement "for free" rather than remembering to AND it in
separately — the exact failure mode §10.3a is written to prevent.

Every predicate produced here is a boolean SQL expression over a
`segment` row aliased `s` (default) that must already be joined into
the enclosing query — see `imsg.retrieval.vector_search` /
`imsg.retrieval.fts_search` for how each candidate path supplies that
join.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from imsg.retrieval.access import AccessContext, segment_eligibility_predicate
from imsg.retrieval.errors import DateRangeInvalidError, InvalidArgumentError
from imsg.retrieval.people import resolve_person

if TYPE_CHECKING:
    import psycopg

MAX_PEOPLE_FILTER = 20
"""Matches SPEC §10.2's `search_messages.people` schema: `"maxItems":
20`. Enforced here too (not only by the MCP transport's JSON Schema
validation) so this module is safe to call directly, e.g. from tests
or a future non-MCP caller."""


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Already-resolved filter values — no further DB lookups needed
    once this is built."""

    people: tuple[int, ...] = ()
    """Resolved `person_id`s. Combined with **AND** semantics (every
    listed person must be a participant of the chat) — a judgment call
    the spec does not spell out; documented in `compile_predicate`."""
    after: datetime | None = None
    """Inclusive, at local midnight (SPEC §9.4 step 1)."""
    before: datetime | None = None
    """Exclusive, at local midnight (SPEC §9.4 step 1)."""
    has_attachment: bool | None = None


def _parse_local_midnight(value: str, timezone: str) -> datetime:
    try:
        d = date.fromisoformat(value)
    except ValueError as exc:
        raise InvalidArgumentError(
            f"not a valid date (expected YYYY-MM-DD): {value!r}"
        ) from exc
    return datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(timezone))


def resolve_filters(
    conn: psycopg.Connection,
    context: AccessContext,
    *,
    people: list[str] | None,
    after: str | None,
    before: str | None,
    has_attachment: bool | None,
    timezone: str,
) -> SearchFilters:
    """Parse and resolve the raw `search_messages` filter arguments
    (SPEC §10.2) into a `SearchFilters`. Raises `InvalidArgumentError`,
    `PersonNotFoundError`, `PersonAmbiguousError`, or
    `DateRangeInvalidError` exactly as SPEC §10.2 documents for this
    tool's error set."""
    people = people or []
    if len(people) > MAX_PEOPLE_FILTER:
        raise InvalidArgumentError(
            f"'people' accepts at most {MAX_PEOPLE_FILTER} entries, got {len(people)}"
        )
    person_ids = tuple(resolve_person(conn, context, p) for p in people)

    after_dt = _parse_local_midnight(after, timezone) if after else None
    before_dt = _parse_local_midnight(before, timezone) if before else None
    if after_dt is not None and before_dt is not None and after_dt >= before_dt:
        raise DateRangeInvalidError(
            f"'after' ({after}) must be strictly before 'before' ({before})"
        )

    return SearchFilters(
        people=person_ids, after=after_dt, before=before_dt, has_attachment=has_attachment
    )


@dataclass(frozen=True, slots=True)
class CompiledPredicate:
    sql: str
    params: dict[str, object]


def compile_predicate(
    filters: SearchFilters, context: AccessContext, *, segment_alias: str = "s"
) -> CompiledPredicate:
    """Build the combined boolean SQL expression (+ bound params) for
    `filters`/`context`, safe to `AND` into any query with a `segment`
    row aliased `segment_alias` in scope.

    People-filter semantics (judgment call, not specified by SPEC
    §10.2 beyond "person short_names or display names; resolved to
    person_id"): every listed person must be a chat participant — AND,
    not OR. Reasoning: the common use is narrowing to a specific
    conversation ("my thread with Alice and Bob"), which AND expresses
    directly; OR ("anything involving Alice or Bob") is also
    defensible and flagged in the build report as an alternative worth
    an explicit spec decision.
    """
    clauses = [segment_eligibility_predicate(context, chat_id_expr=f"{segment_alias}.chat_id")]
    params: dict[str, object] = {}

    if filters.after is not None:
        clauses.append(f"{segment_alias}.started_at >= %(f_after)s")
        params["f_after"] = filters.after
    if filters.before is not None:
        clauses.append(f"{segment_alias}.started_at < %(f_before)s")
        params["f_before"] = filters.before

    if filters.has_attachment is not None:
        exists_kw = "EXISTS" if filters.has_attachment else "NOT EXISTS"
        clauses.append(
            f"{exists_kw} (SELECT 1 FROM segment_message __sm_ha "
            f"JOIN message __m_ha ON __m_ha.message_id = __sm_ha.message_id "
            f"WHERE __sm_ha.segment_id = {segment_alias}.segment_id "
            f"AND __m_ha.has_attachments)"
        )

    for i, person_id in enumerate(filters.people):
        key = f"f_person_{i}"
        clauses.append(
            f"EXISTS (SELECT 1 FROM chat_participant __cp_{i} "
            f"WHERE __cp_{i}.chat_id = {segment_alias}.chat_id "
            f"AND __cp_{i}.person_id = %({key})s)"
        )
        params[key] = person_id

    return CompiledPredicate(sql=" AND ".join(clauses), params=params)


__all__ = [
    "MAX_PEOPLE_FILTER",
    "CompiledPredicate",
    "SearchFilters",
    "compile_predicate",
    "resolve_filters",
]
