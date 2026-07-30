"""Boundary tests for the public-surface auth gate (SPEC §10.4, hard req 4).

Every value here is fictional (D5): subjects, client ids, hostnames.
No test touches the network — introspection is stubbed, and the real
introspector's transport handling is exercised by monkeypatching urllib.
"""

from __future__ import annotations

import dataclasses
import inspect
import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from imsg.mcp.audit import AuditRecord, MemoryAuditSink, hash_params
from imsg.mcp.auth import (
    AuthorizedRequest,
    GoogleTokeninfoIntrospector,
    PublicAuthGate,
    Rejection,
    TokenIntrospection,
    ToolOutcome,
    _VerdictCache,
    build_public_gate,
    parse_tokeninfo_response,
    resource_metadata_url_for,
)
from imsg.mcp.errors import (
    AuditWriteError,
    IntrospectionUnavailableError,
    PublicSurfaceStartupError,
    TokenInvalidError,
)

OWNER_SUB = "100000000000000000001"
OTHER_SUB = "200000000000000000002"
CLIENT_ID = "000000000000-fictional.apps.example"
OTHER_CLIENT_ID = "999999999999-someoneelse.apps.example"

OWNER_TOKEN = "fictional-owner-access-token"
OTHER_TOKEN = "fictional-foreign-access-token"


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def claims(
    *,
    sub: str = OWNER_SUB,
    aud: str = CLIENT_ID,
    azp: str | None = None,
    scopes: frozenset[str] = frozenset({"openid", "email", "profile"}),
    expires_in: int = 3600,
) -> TokenIntrospection:
    return TokenIntrospection(
        subject=sub,
        audience=aud,
        authorized_party=azp,
        scopes=scopes,
        expires_in_seconds=expires_in,
    )


class StubIntrospector:
    """Maps tokens to claims or exceptions; records every call."""

    def __init__(self) -> None:
        self.outcomes: dict[str, TokenIntrospection | Exception] = {}
        self.calls: list[str] = []

    def introspect(self, token: str) -> TokenIntrospection:
        self.calls.append(token)
        outcome = self.outcomes.get(token)
        if outcome is None:
            raise TokenInvalidError("unknown token")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FailingSink:
    """An audit sink that always fails — auditability must gate service."""

    def record(self, rec: AuditRecord) -> None:
        raise AuditWriteError("audit store down")


def make_gate(
    *,
    introspector: StubIntrospector | None = None,
    audit: MemoryAuditSink | FailingSink | None = None,
    clock: FakeClock | None = None,
    cache_ttl_seconds: int = 60,
    rate_limit_per_minute: int = 60,
    failure_budget_per_minute: int = 60,
    owner_subject: str = OWNER_SUB,
    resource_metadata_url: str | None = None,
) -> tuple[PublicAuthGate, StubIntrospector, MemoryAuditSink | FailingSink, FakeClock]:
    intro = introspector if introspector is not None else StubIntrospector()
    if not intro.outcomes:
        intro.outcomes[OWNER_TOKEN] = claims()
        intro.outcomes[OTHER_TOKEN] = claims(sub=OTHER_SUB)
    sink = audit if audit is not None else MemoryAuditSink()
    clk = clock if clock is not None else FakeClock()
    gate = PublicAuthGate(
        client_id=CLIENT_ID,
        owner_subject=owner_subject,
        introspector=intro,
        audit=sink,
        cache_ttl_seconds=cache_ttl_seconds,
        rate_limit_per_minute=rate_limit_per_minute,
        failure_budget_per_minute=failure_budget_per_minute,
        resource_metadata_url=resource_metadata_url,
        clock=clk,
    )
    return gate, intro, sink, clk


def bearer(token: str) -> str:
    return f"Bearer {token}"


def assert_rejected(verdict: AuthorizedRequest | Rejection, status: int) -> Rejection:
    assert isinstance(verdict, Rejection), f"expected rejection, got {verdict!r}"
    assert verdict.status == status
    return verdict


# ---------------------------------------------------------------------------
# Bearer header parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "Bearer",
        "Bearer ",
        "Basic Zm9vOmJhcg==",
        "Bearer two tokens",
        "Bearer tok,Bearer tok2",  # comma-joined duplicate headers must not parse
        "Bearer bad\r\ninjected: header",
        "Token abc",
        " Bearer abc",  # leading whitespace is not tolerated
    ],
)
def test_absent_or_malformed_authorization_is_401(header: str | None) -> None:
    gate, intro, sink, _ = make_gate()
    verdict = gate.authorize(header)
    assert_rejected(verdict, 401)
    assert intro.calls == []  # nothing malformed ever reaches the network
    assert isinstance(sink, MemoryAuditSink)
    (row,) = sink.snapshot()
    assert row.subject is None and row.subject_ok is False
    assert row.error == "UNAUTHORIZED"


def test_oversized_token_is_401_without_introspection() -> None:
    gate, intro, _, _ = make_gate()
    verdict = gate.authorize("Bearer " + "a" * 5000)
    assert_rejected(verdict, 401)
    assert intro.calls == []


def test_bearer_scheme_is_case_insensitive_per_rfc() -> None:
    gate, _, _, _ = make_gate()
    verdict = gate.authorize(f"bearer {OWNER_TOKEN}")
    assert isinstance(verdict, AuthorizedRequest)
    assert verdict.subject == OWNER_SUB


# ---------------------------------------------------------------------------
# Claim validation — every rejection axis
# ---------------------------------------------------------------------------


def test_valid_owner_token_is_allowed() -> None:
    gate, _, _, _ = make_gate()
    verdict = gate.authorize(bearer(OWNER_TOKEN))
    assert isinstance(verdict, AuthorizedRequest)
    assert verdict.subject == OWNER_SUB


def test_invalid_token_is_401() -> None:
    gate, intro, _, _ = make_gate()
    intro.outcomes["bad"] = TokenInvalidError("nope")
    assert_rejected(gate.authorize(bearer("bad")), 401)


def test_expired_token_is_401() -> None:
    gate, intro, _, _ = make_gate()
    intro.outcomes["expired"] = claims(expires_in=0)
    assert_rejected(gate.authorize(bearer("expired")), 401)
    intro.outcomes["expired2"] = claims(expires_in=-30)
    assert_rejected(gate.authorize(bearer("expired2")), 401)


def test_wrong_audience_is_401_even_with_owner_subject() -> None:
    """A token the owner minted for ANY other app must never open the corpus."""
    gate, intro, _, _ = make_gate()
    intro.outcomes["cross-app"] = claims(sub=OWNER_SUB, aud=OTHER_CLIENT_ID)
    assert_rejected(gate.authorize(bearer("cross-app")), 401)


def test_mismatched_azp_is_401_even_with_correct_aud() -> None:
    gate, intro, _, _ = make_gate()
    intro.outcomes["odd-azp"] = claims(azp=OTHER_CLIENT_ID)
    assert_rejected(gate.authorize(bearer("odd-azp")), 401)


def test_matching_azp_is_allowed() -> None:
    gate, intro, _, _ = make_gate()
    intro.outcomes["good-azp"] = claims(azp=CLIENT_ID)
    verdict = gate.authorize(bearer("good-azp"))
    assert isinstance(verdict, AuthorizedRequest)


def test_missing_required_scope_is_401() -> None:
    gate, intro, _, _ = make_gate()
    intro.outcomes["no-openid"] = claims(scopes=frozenset({"email"}))
    assert_rejected(gate.authorize(bearer("no-openid")), 401)


def test_wrong_subject_is_401_and_audits_the_raw_subject() -> None:
    gate, _, sink, _ = make_gate()
    assert_rejected(gate.authorize(bearer(OTHER_TOKEN)), 401)
    assert isinstance(sink, MemoryAuditSink)
    (row,) = sink.snapshot()
    assert row.subject == OTHER_SUB  # SPEC §10.4 item 3: raw rejected subject
    assert row.subject_ok is False


@pytest.mark.parametrize(
    "sub",
    [
        "not-numeric",
        "user@fictional.example",  # email is never an acceptable subject
        "１２３",  # noqa: RUF001 — fullwidth digits are the attack being tested
        " 100000000000000000001",  # whitespace variants
        "100000000000000000001 ",
        "",
    ],
)
def test_non_numeric_subject_is_401(sub: str) -> None:
    gate, intro, _, _ = make_gate()
    intro.outcomes["weird"] = claims(sub=sub)
    assert_rejected(gate.authorize(bearer("weird")), 401)


def test_leading_zero_subject_is_not_the_owner() -> None:
    """Exact string comparison — no int() coercion where '0123' == '123'."""
    gate, intro, _, _ = make_gate()
    intro.outcomes["padded"] = claims(sub="0" + OWNER_SUB)
    assert_rejected(gate.authorize(bearer("padded")), 401)


def test_all_401s_are_byte_identical_regardless_of_reason() -> None:
    """No oracle: bad token vs valid-token-wrong-subject must be indistinguishable."""
    gate, intro, _, _ = make_gate(
        resource_metadata_url="https://mcp.fictional.example/.well-known/oauth-protected-resource"
    )
    intro.outcomes["bad"] = TokenInvalidError("nope")
    r_missing = assert_rejected(gate.authorize(None), 401)
    r_invalid = assert_rejected(gate.authorize(bearer("bad")), 401)
    r_foreign = assert_rejected(gate.authorize(bearer(OTHER_TOKEN)), 401)
    assert r_missing == r_invalid == r_foreign
    assert r_foreign.www_authenticate is not None
    assert "resource_metadata" in r_foreign.www_authenticate


# ---------------------------------------------------------------------------
# Fail closed on introspection failure — SPEC §10.4 item 4
# ---------------------------------------------------------------------------


def test_introspection_network_failure_is_503_never_allow() -> None:
    gate, intro, sink, _ = make_gate()
    intro.outcomes[OWNER_TOKEN] = IntrospectionUnavailableError("down")
    verdict = gate.authorize(bearer(OWNER_TOKEN))
    assert_rejected(verdict, 503)
    assert isinstance(sink, MemoryAuditSink)
    (row,) = sink.snapshot()
    assert row.subject_ok is False
    assert row.error == "UNAVAILABLE"


def test_no_local_validation_fallback_exists() -> None:
    """With introspection permanently down, no token — however shaped — is accepted."""
    intro = StubIntrospector()
    for token in ("a", "b", OWNER_TOKEN):
        intro.outcomes[token] = IntrospectionUnavailableError("down")
    gate, _, _, clk = make_gate(introspector=intro)
    for token in ("a", "b", OWNER_TOKEN):
        assert_rejected(gate.authorize(bearer(token)), 503)
        clk.advance(10)  # step past the unavailable cooldown each time


def test_unavailable_cooldown_answers_503_without_hammering_upstream() -> None:
    gate, intro, _, clk = make_gate()
    intro.outcomes["t1"] = IntrospectionUnavailableError("down")
    assert_rejected(gate.authorize(bearer("t1")), 503)
    calls_after_first = len(intro.calls)
    # Within the cooldown, even a *different* token answers 503 with no call.
    assert_rejected(gate.authorize(bearer("t2")), 503)
    assert len(intro.calls) == calls_after_first
    # After the cooldown, introspection resumes.
    clk.advance(6)
    verdict = gate.authorize(bearer(OWNER_TOKEN))
    assert isinstance(verdict, AuthorizedRequest)


# ---------------------------------------------------------------------------
# Verdict cache — TTL, negative caching, bounds
# ---------------------------------------------------------------------------


def test_cache_hit_skips_introspection_within_ttl() -> None:
    gate, intro, _, clk = make_gate(cache_ttl_seconds=60)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    assert intro.calls.count(OWNER_TOKEN) == 1
    clk.advance(61)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    assert intro.calls.count(OWNER_TOKEN) == 2


def test_cache_ttl_is_capped_by_remaining_token_life() -> None:
    """TTL = min(configured, expires_in): a dying token cannot outlive itself in cache."""
    gate, intro, _, clk = make_gate(cache_ttl_seconds=60)
    intro.outcomes["short"] = claims(expires_in=2)
    assert isinstance(gate.authorize(bearer("short")), AuthorizedRequest)
    clk.advance(3)
    intro.outcomes["short"] = claims(expires_in=0)  # now expired upstream
    assert_rejected(gate.authorize(bearer("short")), 401)
    assert intro.calls.count("short") == 2  # cache expired with the token


def test_negative_verdicts_are_cached_briefly_and_still_audited() -> None:
    gate, intro, sink, clk = make_gate(cache_ttl_seconds=300)
    assert_rejected(gate.authorize(bearer(OTHER_TOKEN)), 401)
    assert_rejected(gate.authorize(bearer(OTHER_TOKEN)), 401)
    assert intro.calls.count(OTHER_TOKEN) == 1  # served from negative cache
    assert isinstance(sink, MemoryAuditSink)
    rows = sink.snapshot()
    assert len(rows) == 2  # every request audited, cache hit or not
    assert all(r.subject == OTHER_SUB and not r.subject_ok for r in rows)
    clk.advance(31)  # negative TTL is capped at 30s regardless of config
    assert_rejected(gate.authorize(bearer(OTHER_TOKEN)), 401)
    assert intro.calls.count(OTHER_TOKEN) == 2


def test_verdict_cache_is_a_bounded_lru() -> None:
    clk = FakeClock()
    cache = _VerdictCache(3, clk)
    from imsg.mcp.auth import _CacheEntry

    for i in range(10):
        cache.put(
            f"key{i}",
            _CacheEntry(
                expires_at=clk() + 60, allowed=False, subject=None, scopes=frozenset()
            ),
        )
    assert len(cache._entries) == 3  # attacker-supplied keys cannot grow memory
    assert cache.get("key0") is None
    assert cache.get("key9") is not None


# ---------------------------------------------------------------------------
# Rate limiting — per subject, and the pre-auth failure budget
# ---------------------------------------------------------------------------


def test_subject_rate_limit_returns_429_rate_limited() -> None:
    gate, _, sink, _ = make_gate(rate_limit_per_minute=2)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    verdict = assert_rejected(gate.authorize(bearer(OWNER_TOKEN)), 429)
    assert verdict.code == "RATE_LIMITED"
    assert isinstance(sink, MemoryAuditSink)
    row = sink.snapshot()[-1]
    assert row.subject == OWNER_SUB and row.error == "RATE_LIMITED"


def test_rate_limit_applies_on_cache_hits_too() -> None:
    """A cached ALLOW must not become a rate-limit bypass."""
    gate, intro, _, _ = make_gate(rate_limit_per_minute=2)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    assert_rejected(gate.authorize(bearer(OWNER_TOKEN)), 429)
    assert intro.calls.count(OWNER_TOKEN) == 1


def test_rate_limit_window_slides() -> None:
    gate, _, _, clk = make_gate(rate_limit_per_minute=1)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    assert_rejected(gate.authorize(bearer(OWNER_TOKEN)), 429)
    clk.advance(61)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)


def test_failure_budget_stops_introspection_amplification() -> None:
    """Distinct garbage tokens must not translate 1:1 into upstream calls forever."""
    gate, intro, _, _ = make_gate(failure_budget_per_minute=2)
    intro.outcomes["g1"] = TokenInvalidError("no")
    intro.outcomes["g2"] = TokenInvalidError("no")
    intro.outcomes["g3"] = TokenInvalidError("no")
    assert_rejected(gate.authorize(bearer("g1")), 401)
    assert_rejected(gate.authorize(bearer("g2")), 401)
    verdict = assert_rejected(gate.authorize(bearer("g3")), 429)
    assert verdict.code == "RATE_LIMITED"
    assert "g3" not in intro.calls  # refused before any network call


def test_cached_owner_token_keeps_working_while_failure_budget_is_exhausted() -> None:
    gate, intro, _, _ = make_gate(failure_budget_per_minute=1)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)
    intro.outcomes["g1"] = TokenInvalidError("no")
    assert_rejected(gate.authorize(bearer("g1")), 401)
    # Budget now exhausted for cache misses, but the owner's cached verdict holds.
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)


# ---------------------------------------------------------------------------
# Audit is load-bearing
# ---------------------------------------------------------------------------


def test_audit_failure_denies_rejections_with_503() -> None:
    gate, _, _, _ = make_gate(audit=FailingSink())
    verdict = gate.authorize(None)
    assert_rejected(verdict, 503)


def test_audit_failure_withholds_payload_on_accept_path() -> None:
    gate, _, _, _ = make_gate(audit=FailingSink())
    handler_ran = False

    def handler(req: AuthorizedRequest) -> ToolOutcome[str]:
        nonlocal handler_ran
        handler_ran = True
        return ToolOutcome(payload="corpus-content", result_count=1)

    result = gate.dispatch(bearer(OWNER_TOKEN), tool="search", params={}, handler=handler)
    assert handler_ran  # validation passed; the tool ran
    assert result.rejection is not None and result.rejection.status == 503
    assert result.payload is None  # an unaudited disclosure is never served


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_dispatch_success_audits_tool_params_hash_and_count() -> None:
    gate, _, sink, _ = make_gate()
    params = {"query": "fictional", "limit": 5}

    def handler(req: AuthorizedRequest) -> ToolOutcome[list[str]]:
        return ToolOutcome(payload=["a", "b"], result_count=2)

    result = gate.dispatch(
        bearer(OWNER_TOKEN), tool="search_messages", params=params, handler=handler
    )
    assert result.rejection is None
    assert result.payload == ["a", "b"]
    assert isinstance(sink, MemoryAuditSink)
    (row,) = sink.snapshot()
    assert row.subject == OWNER_SUB and row.subject_ok is True
    assert row.tool == "search_messages"
    assert row.params_sha256 == hash_params(params)  # hashed, never raw
    assert row.result_count == 2
    assert row.latency_ms is not None


def test_dispatch_rejection_never_runs_the_handler() -> None:
    gate, _, _, _ = make_gate()
    called = False

    def handler(req: AuthorizedRequest) -> ToolOutcome[str]:
        nonlocal called
        called = True
        return ToolOutcome(payload="secret")

    result = gate.dispatch(bearer(OTHER_TOKEN), tool="search", params={}, handler=handler)
    assert result.rejection is not None and result.rejection.status == 401
    assert result.payload is None
    assert called is False


def test_dispatch_handler_exception_audits_internal_and_propagates() -> None:
    gate, _, sink, _ = make_gate()

    def handler(req: AuthorizedRequest) -> ToolOutcome[str]:
        raise RuntimeError("path=/fictional/should/never/leak")

    with pytest.raises(RuntimeError):
        gate.dispatch(bearer(OWNER_TOKEN), tool="search", params={}, handler=handler)
    assert isinstance(sink, MemoryAuditSink)
    (row,) = sink.snapshot()
    assert row.error == "INTERNAL"  # never the exception text


def test_authorized_request_is_frozen() -> None:
    req = AuthorizedRequest(subject=OWNER_SUB, scopes=frozenset({"openid"}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.subject = OTHER_SUB  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structural fail-closed guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func",
    [PublicAuthGate.__init__, build_public_gate],
)
def test_no_parameter_can_disable_validation(func: Any) -> None:
    """Hard requirement 4: no config path disables any step. Guard the signature
    so a future 'convenience' bypass fails this test by construction."""
    forbidden = ("disable", "skip", "bypass", "allow_all", "insecure", "dev_mode", "unsafe")
    for name in inspect.signature(func).parameters:
        lowered = name.lower()
        assert not any(f in lowered for f in forbidden), (
            f"parameter {name!r} looks like a validation bypass — "
            f"hard requirement 4 forbids any such path"
        )


@pytest.mark.parametrize(
    "owner_subject",
    ["", "not-numeric", "user@fictional.example", "1" * 256, "12 34"],
)
def test_gate_refuses_to_construct_with_bad_owner_subject(owner_subject: str) -> None:
    with pytest.raises(PublicSurfaceStartupError):
        PublicAuthGate(
            client_id=CLIENT_ID,
            owner_subject=owner_subject,
            introspector=StubIntrospector(),
            audit=MemoryAuditSink(),
        )


@pytest.mark.parametrize("client_id", ["", " padded ", "\t"])
def test_gate_refuses_to_construct_with_bad_client_id(client_id: str) -> None:
    with pytest.raises(PublicSurfaceStartupError):
        PublicAuthGate(
            client_id=client_id,
            owner_subject=OWNER_SUB,
            introspector=StubIntrospector(),
            audit=MemoryAuditSink(),
        )


# ---------------------------------------------------------------------------
# Startup from config — refusal paths (hard requirement 4)
# ---------------------------------------------------------------------------


def _public_config(**overrides: Any) -> Any:
    from imsg.config.schema import McpPublicConfig

    base: dict[str, Any] = {
        "enabled": False,
        "scope": "allowlist",
        "external_url": "https://mcp.fictional.example/mcp",
        "oauth": {
            "issuer": "google",
            "client_id": CLIENT_ID,
            "owner_subject": "env:IMSG_TEST_OWNER_SUBJECT",
            "tokeninfo_cache_ttl_seconds": 60,
        },
    }
    base.update(overrides)
    return McpPublicConfig.model_validate(base)


def test_build_gate_succeeds_with_resolvable_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", OWNER_SUB)
    intro = StubIntrospector()
    intro.outcomes[OWNER_TOKEN] = claims()
    gate = build_public_gate(_public_config(), audit=MemoryAuditSink(), introspector=intro)
    verdict = gate.authorize(bearer(OWNER_TOKEN))
    assert isinstance(verdict, AuthorizedRequest)


def test_build_gate_refuses_when_owner_subject_missing() -> None:
    cfg = _public_config(
        oauth={"issuer": "google", "client_id": CLIENT_ID, "owner_subject": None}
    )
    with pytest.raises(PublicSurfaceStartupError):
        build_public_gate(cfg, audit=MemoryAuditSink(), introspector=StubIntrospector())


def test_build_gate_refuses_when_owner_subject_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IMSG_TEST_OWNER_SUBJECT", raising=False)
    with pytest.raises(PublicSurfaceStartupError):
        build_public_gate(
            _public_config(), audit=MemoryAuditSink(), introspector=StubIntrospector()
        )


def test_build_gate_refuses_when_resolved_subject_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", "")
    with pytest.raises(PublicSurfaceStartupError):
        build_public_gate(
            _public_config(), audit=MemoryAuditSink(), introspector=StubIntrospector()
        )


def test_build_gate_refuses_when_resolved_subject_is_an_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC §10.4: numeric sub, never email — emails get reassigned."""
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", "owner@fictional.example")
    with pytest.raises(PublicSurfaceStartupError):
        build_public_gate(
            _public_config(), audit=MemoryAuditSink(), introspector=StubIntrospector()
        )


def test_build_gate_refuses_when_client_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", OWNER_SUB)
    cfg = _public_config(
        oauth={
            "issuer": "google",
            "client_id": None,
            "owner_subject": "env:IMSG_TEST_OWNER_SUBJECT",
        }
    )
    with pytest.raises(PublicSurfaceStartupError):
        build_public_gate(cfg, audit=MemoryAuditSink(), introspector=StubIntrospector())


def test_build_gate_resolves_env_ref_client_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", OWNER_SUB)
    monkeypatch.setenv("IMSG_TEST_CLIENT_ID", CLIENT_ID)
    cfg = _public_config(
        oauth={
            "issuer": "google",
            "client_id": "env:IMSG_TEST_CLIENT_ID",
            "owner_subject": "env:IMSG_TEST_OWNER_SUBJECT",
        }
    )
    intro = StubIntrospector()
    intro.outcomes[OWNER_TOKEN] = claims()  # aud == the resolved CLIENT_ID
    gate = build_public_gate(cfg, audit=MemoryAuditSink(), introspector=intro)
    assert isinstance(gate.authorize(bearer(OWNER_TOKEN)), AuthorizedRequest)


def test_challenge_carries_resource_metadata_url_from_external_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", OWNER_SUB)
    gate = build_public_gate(
        _public_config(), audit=MemoryAuditSink(), introspector=StubIntrospector()
    )
    rejection = gate.authorize(None)
    assert isinstance(rejection, Rejection)
    assert rejection.www_authenticate is not None
    assert (
        'resource_metadata="https://mcp.fictional.example/.well-known/oauth-protected-resource"'
        in rejection.www_authenticate
    )


def test_resource_metadata_url_strips_path() -> None:
    assert (
        resource_metadata_url_for("https://mcp.fictional.example/mcp")
        == "https://mcp.fictional.example/.well-known/oauth-protected-resource"
    )


# ---------------------------------------------------------------------------
# Google tokeninfo introspector — transport handling (urllib monkeypatched)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_introspector_requires_https() -> None:
    with pytest.raises(PublicSurfaceStartupError):
        GoogleTokeninfoIntrospector("http://tokeninfo.fictional.example/")


def test_introspector_posts_token_in_body_not_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, **kwargs: Any) -> FakeResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = request.data
        return FakeResponse(
            json.dumps(
                {
                    "sub": OWNER_SUB,
                    "aud": CLIENT_ID,
                    "scope": "openid email",
                    "expires_in": "3599",
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = GoogleTokeninfoIntrospector().introspect("fictional-token")
    assert result.subject == OWNER_SUB
    assert result.expires_in_seconds == 3599
    assert captured["method"] == "POST"
    assert "fictional-token" not in captured["url"]  # token never in the URL
    assert b"fictional-token" in captured["body"]


def test_introspector_maps_4xx_to_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, **kwargs: Any) -> FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", None, io.BytesIO(b"")  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(TokenInvalidError):
        GoogleTokeninfoIntrospector().introspect("t")


def test_introspector_maps_5xx_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, **kwargs: Any) -> FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url, 503, "Unavailable", None, io.BytesIO(b"")  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(IntrospectionUnavailableError):
        GoogleTokeninfoIntrospector().introspect("t")


@pytest.mark.parametrize(
    "exc", [urllib.error.URLError("dns"), TimeoutError(), OSError("reset")]
)
def test_introspector_maps_network_errors_to_unavailable(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    def fake_urlopen(request: Any, **kwargs: Any) -> FakeResponse:
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(IntrospectionUnavailableError):
        GoogleTokeninfoIntrospector().introspect("t")


def test_introspector_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(b"x" * (64 * 1024 + 10))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(IntrospectionUnavailableError):
        GoogleTokeninfoIntrospector().introspect("t")


# ---------------------------------------------------------------------------
# Tokeninfo response parsing — strict, closed
# ---------------------------------------------------------------------------


def _body(**fields: Any) -> bytes:
    base: dict[str, Any] = {
        "sub": OWNER_SUB,
        "aud": CLIENT_ID,
        "scope": "openid email profile",
        "expires_in": "3599",
    }
    base.update(fields)
    return json.dumps({k: v for k, v in base.items() if v is not ...}).encode()


def test_parse_valid_response_with_string_expires_in() -> None:
    parsed = parse_tokeninfo_response(_body())
    assert parsed.subject == OWNER_SUB
    assert parsed.audience == CLIENT_ID
    assert parsed.scopes == frozenset({"openid", "email", "profile"})
    assert parsed.expires_in_seconds == 3599


@pytest.mark.parametrize("raw", [b"not json", b"[1,2]", b'"str"', b"\xff\xfe"])
def test_parse_malformed_body_is_unavailable(raw: bytes) -> None:
    with pytest.raises(IntrospectionUnavailableError):
        parse_tokeninfo_response(raw)


@pytest.mark.parametrize(
    "fields",
    [
        {"sub": ...},  # missing entirely
        {"sub": ""},
        {"sub": 12345},  # numeric JSON type, not string
        {"aud": ...},
        {"aud": ""},
        {"expires_in": ...},
        {"expires_in": "soon"},
        {"expires_in": True},
        {"scope": 5},
        {"azp": 5},
    ],
)
def test_parse_missing_or_mistyped_claims_is_invalid(fields: dict[str, Any]) -> None:
    with pytest.raises(TokenInvalidError):
        parse_tokeninfo_response(_body(**fields))
