"""Unit tests for `imsg.retrieval.filters` that do not require a
person lookup (so no database is needed) — date-range validation and
predicate compilation. Person resolution itself, and the full
`resolve_filters` path with `people` non-empty, are exercised against
live Postgres in `tests/test_retrieval_integration.py`."""

from __future__ import annotations

import pytest

from imsg.retrieval.access import (
    LOCAL_FULL_ACCESS,
    SCOPE_CHAT_IDS_PARAM,
    AccessContext,
    RequestScope,
    resolve_request_scope,
)
from imsg.retrieval.errors import DateRangeInvalidError, InvalidArgumentError
from imsg.retrieval.filters import SearchFilters, compile_predicate, resolve_filters

FULL = resolve_request_scope(None, LOCAL_FULL_ACCESS)


def test_resolve_filters_with_no_people_needs_no_connection() -> None:
    filters = resolve_filters(
        None,  # type: ignore[arg-type]
        FULL,
        people=None,
        after="2024-01-01",
        before="2024-02-01",
        has_attachment=None,
        timezone="America/Los_Angeles",
    )
    assert filters.people == ()
    assert filters.after is not None
    assert filters.before is not None
    assert filters.after < filters.before


def test_after_inclusive_before_exclusive_at_local_midnight() -> None:
    filters = resolve_filters(
        None,  # type: ignore[arg-type]
        FULL,
        people=None,
        after="2024-01-01",
        before="2024-01-02",
        has_attachment=None,
        timezone="America/Los_Angeles",
    )
    assert filters.after is not None
    assert filters.after.hour == 0
    assert filters.after.minute == 0
    assert filters.before is not None
    assert (filters.before - filters.after).total_seconds() == 86400


def test_after_equal_to_before_is_invalid() -> None:
    with pytest.raises(DateRangeInvalidError):
        resolve_filters(
            None,  # type: ignore[arg-type]
            FULL,
            people=None,
            after="2024-01-05",
            before="2024-01-05",
            has_attachment=None,
            timezone="UTC",
        )


def test_after_later_than_before_is_invalid() -> None:
    with pytest.raises(DateRangeInvalidError):
        resolve_filters(
            None,  # type: ignore[arg-type]
            FULL,
            people=None,
            after="2024-06-01",
            before="2024-01-01",
            has_attachment=None,
            timezone="UTC",
        )


def test_malformed_date_is_invalid_argument() -> None:
    with pytest.raises(InvalidArgumentError):
        resolve_filters(
            None,  # type: ignore[arg-type]
            FULL,
            people=None,
            after="not-a-date",
            before=None,
            has_attachment=None,
            timezone="UTC",
        )


def test_too_many_people_is_invalid_argument() -> None:
    with pytest.raises(InvalidArgumentError):
        resolve_filters(
            None,  # type: ignore[arg-type]
            FULL,
            people=[f"p{i}" for i in range(21)],
            after=None,
            before=None,
            has_attachment=None,
            timezone="UTC",
        )


# --------------------------------------------------------------------------
# compile_predicate
# --------------------------------------------------------------------------


def test_empty_filters_still_fold_in_scope_predicate() -> None:
    predicate = compile_predicate(SearchFilters(), FULL)
    assert predicate.sql == "TRUE"
    assert predicate.params == {}


def test_date_filters_add_clauses_and_params() -> None:
    from datetime import UTC, datetime

    filters = SearchFilters(
        after=datetime(2024, 1, 1, tzinfo=UTC), before=datetime(2024, 2, 1, tzinfo=UTC)
    )
    predicate = compile_predicate(filters, FULL)
    assert "s.started_at >= %(f_after)s" in predicate.sql
    assert "s.started_at < %(f_before)s" in predicate.sql
    assert predicate.params["f_after"] == filters.after
    assert predicate.params["f_before"] == filters.before


def test_has_attachment_true_uses_exists() -> None:
    predicate = compile_predicate(SearchFilters(has_attachment=True), FULL)
    assert "EXISTS" in predicate.sql
    assert "NOT EXISTS" not in predicate.sql.split("EXISTS", 1)[0]


def test_has_attachment_false_uses_not_exists() -> None:
    predicate = compile_predicate(SearchFilters(has_attachment=False), FULL)
    assert "NOT EXISTS" in predicate.sql


def test_people_filter_is_anded_across_all_listed_persons() -> None:
    predicate = compile_predicate(SearchFilters(people=(1, 2, 3)), FULL)
    assert predicate.sql.count("chat_participant") == 3
    assert predicate.params == {"f_person_0": 1, "f_person_1": 2, "f_person_2": 3}


def test_allowlist_scope_folds_in_the_evaluated_eligible_set() -> None:
    ctx = AccessContext(surface="public", scope="allowlist", subject="1234567890")
    scope = RequestScope(context=ctx, eligible_chat_ids=frozenset({4, 2}))
    predicate = compile_predicate(SearchFilters(people=(9,)), scope)
    assert predicate.sql.startswith(f"s.chat_id = ANY(%({SCOPE_CHAT_IDS_PARAM})s::bigint[])")
    assert predicate.params == {SCOPE_CHAT_IDS_PARAM: [2, 4], "f_person_0": 9}


def test_a_people_filter_under_any_scope_needs_a_connection() -> None:
    with pytest.raises(ValueError, match="needs a database connection"):
        resolve_filters(
            None,
            FULL,
            people=["alice"],
            after=None,
            before=None,
            has_attachment=None,
            timezone="UTC",
        )
