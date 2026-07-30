"""Unit tests for `imsg.retrieval.access` (SPEC §10.3a, D6) — no
database required; `segment_eligibility_predicate` returns SQL text,
asserted here as text, exercised for real against live Postgres in
`tests/test_retrieval_integration.py`."""

from __future__ import annotations

import pytest

from imsg.retrieval.access import LOCAL_FULL_ACCESS, AccessContext, segment_eligibility_predicate


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
    ctx = AccessContext(surface="public", scope="allowlist", subject="1234567890")
    assert ctx.scope == "allowlist"


def test_full_scope_predicate_is_unconditionally_true() -> None:
    assert segment_eligibility_predicate(LOCAL_FULL_ACCESS) == "TRUE"


def test_allowlist_scope_predicate_references_chat_participant() -> None:
    ctx = AccessContext(surface="public", scope="allowlist", subject="1234567890")
    predicate = segment_eligibility_predicate(ctx)
    assert "chat_participant" in predicate
    assert "allowlist_person" in predicate
    assert "text_allowed" in predicate
    assert "s.chat_id" in predicate  # default alias


def test_allowlist_scope_predicate_honors_a_custom_chat_id_expr() -> None:
    ctx = AccessContext(surface="public", scope="allowlist", subject="1234567890")
    predicate = segment_eligibility_predicate(ctx, chat_id_expr="%(chat_id)s")
    assert "%(chat_id)s" in predicate
    assert "s.chat_id" not in predicate
