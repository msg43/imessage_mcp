"""Unit tests for `imsg.retrieval.access` (SPEC §10.3a, D6) — no
database required. What `allowlist` scope admits is exercised against
live Postgres in `tests/test_retrieval_allowlist_parity_integration.py`
(export's rule, chat by chat) and `tests/test_retrieval_integration.py`
(the tools end to end)."""

from __future__ import annotations

import pytest

from imsg.retrieval.access import (
    LOCAL_FULL_ACCESS,
    SCOPE_CHAT_IDS_PARAM,
    AccessContext,
    AttachmentGate,
    RequestScope,
    resolve_request_scope,
)

PUBLIC_ALLOWLIST = AccessContext(surface="public", scope="allowlist", subject="1234567890")


def test_local_full_access_constant() -> None:
    assert LOCAL_FULL_ACCESS.surface == "local"
    assert LOCAL_FULL_ACCESS.scope == "full"
    assert LOCAL_FULL_ACCESS.subject is None


def test_local_surface_rejects_allowlist_scope() -> None:
    with pytest.raises(ValueError, match="local surface is always full"):
        AccessContext(surface="local", scope="allowlist")


def test_public_full_scope_is_permitted() -> None:
    ctx = AccessContext(surface="public", scope="full", subject="1234567890")
    assert ctx.scope == "full"


def test_public_allowlist_scope_is_permitted() -> None:
    assert PUBLIC_ALLOWLIST.scope == "allowlist"


@pytest.mark.parametrize(
    ("surface", "scope"), [("public", "everything"), ("public", ""), ("remote", "allowlist")]
)
def test_unknown_scope_or_surface_is_rejected_at_construction(surface: str, scope: str) -> None:
    """A stray value must fail here, not fall through to whichever branch
    a later `if scope == ...` happens to take."""
    with pytest.raises(ValueError, match="unknown"):
        AccessContext(surface=surface, scope=scope)  # type: ignore[arg-type]


def test_full_scope_resolves_without_a_database() -> None:
    scope = resolve_request_scope(None, LOCAL_FULL_ACCESS)
    assert scope.is_full
    assert scope.eligible_chat_ids is None
    assert not scope.admits_nothing
    assert not scope.withholds_ineligible_content


def test_allowlist_scope_needs_a_database() -> None:
    with pytest.raises(ValueError, match="needs a database connection"):
        resolve_request_scope(None, PUBLIC_ALLOWLIST)


def test_full_scope_predicate_is_unconditionally_true() -> None:
    predicate = resolve_request_scope(None, LOCAL_FULL_ACCESS).chat_predicate("s.chat_id")
    assert predicate.sql == "TRUE"
    assert predicate.params == {}


def test_allowlist_predicate_is_membership_in_the_evaluated_set() -> None:
    scope = RequestScope(context=PUBLIC_ALLOWLIST, eligible_chat_ids=frozenset({7, 3}))
    predicate = scope.chat_predicate("s.chat_id")
    assert predicate.sql == f"s.chat_id = ANY(%({SCOPE_CHAT_IDS_PARAM})s::bigint[])"
    assert predicate.params == {SCOPE_CHAT_IDS_PARAM: [3, 7]}


def test_allowlist_predicate_honors_a_custom_chat_id_expr() -> None:
    scope = RequestScope(context=PUBLIC_ALLOWLIST, eligible_chat_ids=frozenset({1}))
    assert scope.chat_predicate("m.chat_id").sql.startswith("m.chat_id = ANY(")


def test_an_empty_eligible_set_admits_nothing_and_its_predicate_matches_nothing() -> None:
    scope = RequestScope(context=PUBLIC_ALLOWLIST, eligible_chat_ids=frozenset())
    assert scope.admits_nothing
    assert not scope.admits_chat(1)
    assert scope.chat_predicate("s.chat_id").params == {SCOPE_CHAT_IDS_PARAM: []}


def test_a_chat_outside_the_evaluated_set_is_denied() -> None:
    """A restricted evaluation (`among_chat_ids`) must stay closed for
    every chat it did not evaluate."""
    scope = RequestScope(context=PUBLIC_ALLOWLIST, eligible_chat_ids=frozenset({5}))
    assert scope.admits_chat(5)
    assert not scope.admits_chat(6)


def test_allowlist_never_shows_unsent_or_edit_history_whatever_the_policy() -> None:
    allowlist = RequestScope(context=PUBLIC_ALLOWLIST, eligible_chat_ids=frozenset({1}))
    full = resolve_request_scope(None, LOCAL_FULL_ACCESS)
    assert allowlist.withholds_ineligible_content
    assert not allowlist.shows_unsent(True)
    assert not allowlist.shows_edit_history(True)
    assert full.shows_unsent(True) and not full.shows_unsent(False)
    assert full.shows_edit_history(True) and not full.shows_edit_history(False)


def test_attachment_gate_full_scope_allows_everything() -> None:
    gate = AttachmentGate(allowed_pairs=None)
    assert gate.allows(1, 2)
    assert gate.allows(None, 2)


def test_attachment_gate_denies_unlisted_pairs_and_unsegmented_messages() -> None:
    gate = AttachmentGate(allowed_pairs=frozenset({(1, 2)}))
    assert gate.allows(1, 2)
    assert not gate.allows(1, 3)
    assert not gate.allows(2, 2)
    assert not gate.allows(None, 2)
