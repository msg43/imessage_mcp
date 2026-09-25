"""The public-surface security boundary (SPEC §10.4, hard requirement 4).

OAuth subject validation is the ONLY access control on the public MCP
surface — there is no network ACL and no per-user isolation guarantee
from the client platform. Everything in this module is therefore built
fail-closed, with **no configuration path that disables any step**:

- Bearer token required; absent/malformed/oversized → 401.
- Validity is established **only** by Google's tokeninfo introspection.
  Google user access tokens are opaque — they are NOT JWTs and cannot be
  verified by signature (D6 verdict 7; SPEC §10.4). There is deliberately
  no local-verification fallback anywhere in this module, and the
  introspector interface returns claims, not booleans, so a future
  "optimization" back to local decoding has nothing to plug into: an
  introspector that does not perform a live lookup cannot know
  `expires_in`, which validation requires.
- Token validity uses **Google's clock** (`expires_in`), never local
  `exp` arithmetic — a skewed local clock can neither extend nor deny a
  token's life.
- `aud` must equal the registered client id exactly; a present `azp`
  that differs is rejected. Accepting "aud OR azp" would let a token
  minted for *any other* Google OAuth app read the corpus.
- `sub` is compared as an exact string against the pinned numeric owner
  subject, in constant time over fixed-length digests (no length or
  prefix timing signal). No int() coercion — "0123" never equals "123".
- Introspection unreachable/malformed → 503, never allow (SPEC §10.4
  item 4 says 503; a task-level summary elsewhere said 401 — the spec is
  authoritative and 503 is the honest fail-closed answer).
- Every request is audited; an unauditable request that carried a token
  is denied. A request whose token was judged — accepted, rejected,
  rate-limited — gets its own `mcp_audit` row. A request turned away
  before any token was judged (none sent, a malformed header, a client
  already throttled, the failure budget spent, the tokeninfo breaker
  open) has no subject to record, and is counted in memory instead
  (`imsg.mcp.audit.RejectionTally`): one aggregate row per code per
  interval, so a scanner's flood costs a dictionary update rather than a
  database connection each (QA review 2026-09-24). Rejected subjects are
  logged raw (SPEC §10.4 item 3); tokens never are — cache and audit keys
  use sha256(token).
- Rejected requests are throttled per client address
  (`imsg.mcp.ratelimit.ClientFailureThrottle`): a client over its
  failure budget is answered 429 before any tokeninfo call or audit
  write. A token already cached as the owner's is never throttled, and
  one client's failures no longer spend the budget that decides whether
  the owner's next token is introspected.
- All 401 responses are byte-identical regardless of *why* (bad token vs
  valid-token-wrong-subject): the boundary is not an oracle for whether
  a subject exists or a token was close.

Transport contract (the parts that live outside this module, SPEC §10.4):
the HTTP layer MUST validate `Host` and any present `Origin` against
config before dispatch (403 on mismatch), reject duplicate Authorization
headers rather than joining them, attach RESPONSE_CACHE_HEADERS to every
tool response, and route every request through :meth:`PublicAuthGate.dispatch`
— there is no other supported path to a tool handler.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Protocol

from imsg.mcp.audit import AuditRecord, AuditSink, RejectionTally, hash_params
from imsg.mcp.errors import (
    AuditWriteError,
    IntrospectionUnavailableError,
    PublicSurfaceStartupError,
    TokenInvalidError,
)
from imsg.mcp.ratelimit import ClientFailureThrottle, SlidingWindowLimiter

if TYPE_CHECKING:
    from imsg.config.schema import McpPublicConfig

GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"

# Response headers the transport MUST attach to tool responses
# (SPEC §10.4: never permit shared caching of corpus text).
RESPONSE_CACHE_HEADERS: Mapping[str, str] = {
    "Cache-Control": "private, no-store",
    "Vary": "Authorization",
}

# RFC 6750 b64token charset; Google access tokens fit comfortably.
_BEARER_RE = re.compile(r"^Bearer[ ]+(?P<token>[A-Za-z0-9\-._~+/]+=*)$", re.IGNORECASE)
_MAX_TOKEN_CHARS = 4096

# Numeric Google `sub` (SPEC §10.4 item 3: numeric, never email). OIDC caps
# sub at 255 chars. Exact-string semantics: leading zeros are preserved,
# unicode digit lookalikes fail the ASCII class.
_SUBJECT_RE = re.compile(r"^[0-9]{1,255}$")

_MAX_TOKENINFO_BYTES = 64 * 1024
_NEGATIVE_CACHE_CAP_SECONDS = 30.0
_UNAVAILABLE_COOLDOWN_SECONDS = 5.0
_CACHE_MAX_ENTRIES = 4096
DEFAULT_FAILURE_BUDGET_PER_MINUTE = 600
"""Failed introspections per minute across every client before cache
misses are refused 429 without asking Google (D7.3's amplification
guard). Was 60, when it was the only failure limit: one source sending a
bad token a second then locked the owner's next token out (QA review
2026-09-24). With a per-client budget in front of it, this one only has
to bound what many addresses can spend together; the verdict cache
already bounds a repeated token to one call per 30 s. Also the budget
all requests with no identifiable client share
(`imsg.mcp.ratelimit.ClientFailureThrottle`)."""

DEFAULT_CLIENT_FAILURE_BUDGET_PER_MINUTE = 10
"""Rejected requests per minute one client address may make before it is
answered 429 without further work. A legitimate client fails once when
its token expires and then presents a new one; ten a minute is a client
guessing, scanning or broken."""


def _ct_equal(a: str, b: str) -> bool:
    """Constant-time equality over fixed-length digests.

    Hashing first means compare_digest always sees equal-length inputs,
    so neither content nor *length* of the pinned subject leaks through
    timing.
    """
    da = hashlib.sha256(a.encode("utf-8")).digest()
    db = hashlib.sha256(b.encode("utf-8")).digest()
    return hmac.compare_digest(da, db)


def _token_key(token: str) -> str:
    """Cache/audit key for a token. The raw token is never stored or logged."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenIntrospection:
    """Claims returned by a live introspection lookup.

    `expires_in_seconds` comes from the introspection endpoint's own
    clock, which is why it (and not local time vs `exp`) drives both
    validity and cache TTL.
    """

    subject: str
    audience: str
    authorized_party: str | None
    scopes: frozenset[str]
    expires_in_seconds: int


class TokenIntrospector(Protocol):
    """Live token introspection. Implementations must never log the token."""

    def introspect(self, token: str) -> TokenIntrospection:
        """Return claims, or raise TokenInvalidError / IntrospectionUnavailableError."""
        ...


class GoogleTokeninfoIntrospector:
    """Introspects opaque Google access tokens via the tokeninfo endpoint.

    The token travels in the POST body — never in the URL, where proxies
    and server logs would capture it. TLS verification uses the platform
    default trust store and is not configurable off; the endpoint URL
    must be https.
    """

    def __init__(
        self,
        url: str = GOOGLE_TOKENINFO_URL,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme != "https":
            raise PublicSurfaceStartupError(
                "tokeninfo URL must be https — plaintext introspection would "
                "expose bearer tokens in transit (SPEC §10.4)"
            )
        self._url = url
        self._timeout = timeout_seconds
        self._ssl_context = ssl.create_default_context()

    def introspect(self, token: str) -> TokenIntrospection:
        body = urllib.parse.urlencode({"access_token": token}).encode("ascii")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout, context=self._ssl_context
            ) as response:
                raw = response.read(_MAX_TOKENINFO_BYTES + 1)
        except urllib.error.HTTPError as exc:
            # tokeninfo answers 4xx for invalid/expired/revoked tokens.
            if 400 <= exc.code < 500:
                raise TokenInvalidError("token rejected by introspection") from None
            raise IntrospectionUnavailableError("introspection upstream error") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise IntrospectionUnavailableError("introspection unreachable") from None
        if len(raw) > _MAX_TOKENINFO_BYTES:
            raise IntrospectionUnavailableError("introspection response oversized")
        return parse_tokeninfo_response(raw)


def parse_tokeninfo_response(raw: bytes) -> TokenIntrospection:
    """Strictly parse a tokeninfo 200 body.

    Classification is deliberate: a response that is not well-formed JSON
    (or not an object) means we cannot tell anything about the token →
    unavailable (503). A well-formed response *missing required claims*
    means the token verifiably cannot prove the owner's identity — e.g. a
    token without identity scopes has no `sub` — → invalid (401). Both
    are fail-closed.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise IntrospectionUnavailableError("introspection response unparseable") from None
    if not isinstance(data, dict):
        raise IntrospectionUnavailableError("introspection response not an object")

    subject = data.get("sub")
    audience = data.get("aud")
    if not isinstance(subject, str) or not subject:
        raise TokenInvalidError("introspection response lacks a subject")
    if not isinstance(audience, str) or not audience:
        raise TokenInvalidError("introspection response lacks an audience")

    azp = data.get("azp")
    if azp is not None and not isinstance(azp, str):
        raise TokenInvalidError("introspection azp malformed")

    scope_field = data.get("scope")
    if scope_field is None:
        scopes: frozenset[str] = frozenset()
    elif isinstance(scope_field, str):
        scopes = frozenset(scope_field.split())
    else:
        raise TokenInvalidError("introspection scope malformed")

    expires_raw = data.get("expires_in")
    if isinstance(expires_raw, bool) or not isinstance(expires_raw, int | str):
        raise TokenInvalidError("introspection expiry missing")
    try:
        expires_in = int(expires_raw)
    except ValueError:
        raise TokenInvalidError("introspection expiry malformed") from None

    return TokenIntrospection(
        subject=subject,
        audience=audience,
        authorized_party=azp,
        scopes=scopes,
        expires_in_seconds=expires_in,
    )


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rejection:
    """A denied request. `code` is a closed-set machine code (audit.py)."""

    status: int  # 401 | 429 | 503
    code: str  # UNAUTHORIZED | RATE_LIMITED | UNAVAILABLE
    www_authenticate: str | None


@dataclass(frozen=True, slots=True)
class AuthorizedRequest:
    """Snapshot of a validated request context.

    Frozen at validation time: the subject the handler sees is exactly
    the subject that was validated — there is no re-derivation between
    validation and handling (no TOCTOU on identity).
    """

    subject: str
    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class ToolOutcome[T]:
    """What a tool handler returns to `dispatch` for auditing."""

    payload: T
    result_count: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DispatchResult[T]:
    """Either a rejection (payload is None) or the tool payload."""

    rejection: Rejection | None
    payload: T | None


class RateLimitCharge:
    """One event recorded against the subject's rate limit for one HTTP
    request, by the transport's check (:meth:`PublicAuthGate.admit`).

    A tool call is one HTTP request, so it must cost one event — but it is
    checked twice, by the transport guard and again by
    :meth:`PublicAuthGate.dispatch` before the handler runs, and both
    charged the limiter (QA review 2026-09-24: 60 per minute admitted at
    most 30 tool calls). The transport now hands this receipt to
    `dispatch`, which still re-validates the token in full and accepts the
    receipt instead of charging again.

    Spendable once, by the gate that issued it, for the token it was
    issued for — a JSON-RPC batch carrying two tool calls in one request
    pays for the second. `dispatch` without a receipt (the AT-1 probe, a
    test, any caller that is not the transport) charges as before, so
    forgetting to pass one can only over-count."""

    __slots__ = ("_gate", "_lock", "_spent", "_token_key")

    def __init__(self, gate: PublicAuthGate, token_key: str) -> None:
        self._gate = gate
        self._token_key = token_key
        self._spent = False
        self._lock = threading.Lock()

    def spend(self, gate: PublicAuthGate, token_key: str) -> bool:
        """Use the receipt for this gate and token; `False` if it is not
        theirs or was already used."""
        with self._lock:
            if self._spent or self._gate is not gate or self._token_key != token_key:
                return False
            self._spent = True
            return True


@dataclass(frozen=True, slots=True)
class Admission:
    """A request the transport let through: who it is, and the rate-limit
    event it already paid (to hand to :meth:`PublicAuthGate.dispatch`)."""

    request: AuthorizedRequest
    charge: RateLimitCharge | None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float  # monotonic
    allowed: bool
    subject: str | None  # validated subject if allowed; raw rejected subject if known
    scopes: frozenset[str]


class _VerdictCache:
    """Bounded LRU keyed on sha256(token). Thread-safe.

    Bounded because the key space is attacker-controlled: unlimited
    distinct garbage tokens must not grow memory without limit. Eviction
    is always safe — the worst case is one extra introspection.
    """

    def __init__(self, max_entries: int, clock: Callable[[], float]) -> None:
        self._max = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()

    def get(self, key: str) -> _CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._clock():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: _CacheEntry) -> None:
        if entry.expires_at <= self._clock():
            return
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class _NeedsIO:
    """Returned by the core check when deciding would need a tokeninfo
    call or an audit write and the caller asked for none."""


_NEEDS_IO = _NeedsIO()


class PublicAuthGate:
    """Owner-subject validation, rate limiting, and audit for the public surface.

    Construction validates its own inputs and refuses (raises
    PublicSurfaceStartupError) rather than degrade. By design there is no
    parameter that skips a validation step — tests assert this
    structurally against the signature.

    Three entry points share one check (:meth:`_decide`):

    - :meth:`admit` — the transport guard's, for every HTTP request. With
      `io_allowed=False` it decides only what memory can decide (the
      header's shape, a cached verdict, the client throttle) and returns
      `None` for the rest, so the event loop never waits on Google or
      Postgres; the guard then calls it again on a worker thread.
    - :meth:`dispatch` — the only path to a tool handler.
    - :meth:`authorize` — the bare check, for callers that need neither.
    """

    def __init__(
        self,
        *,
        client_id: str,
        owner_subject: str,
        introspector: TokenIntrospector,
        audit: AuditSink,
        cache_ttl_seconds: int = 60,
        rate_limit_per_minute: int = 60,
        failure_budget_per_minute: int = DEFAULT_FAILURE_BUDGET_PER_MINUTE,
        client_failure_budget_per_minute: int = DEFAULT_CLIENT_FAILURE_BUDGET_PER_MINUTE,
        rejection_tally: RejectionTally | None = None,
        required_scopes: frozenset[str] = frozenset({"openid"}),
        resource_metadata_url: str | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not client_id or client_id.strip() != client_id or not client_id.strip():
            raise PublicSurfaceStartupError(
                "public surface refused to start: OAuth client id is empty or "
                "has surrounding whitespace — the audience check would be "
                "meaningless (SPEC §10.4)"
            )
        if not _SUBJECT_RE.match(owner_subject):
            raise PublicSurfaceStartupError(
                "public surface refused to start: pinned owner_subject is not a "
                "plausible numeric Google sub (SPEC §10.4: numeric sub, never "
                "email; hard requirement 4: empty/missing must refuse startup, "
                "never allow-all)"
            )
        if cache_ttl_seconds < 1:
            raise PublicSurfaceStartupError("tokeninfo cache TTL must be >= 1 second")
        self._client_id = client_id
        self._owner_subject = owner_subject
        self._introspector = introspector
        self._audit = audit
        self._cache_ttl = float(cache_ttl_seconds)
        self._required_scopes = required_scopes
        self._clock = clock
        self._cache = _VerdictCache(_CACHE_MAX_ENTRIES, clock)
        self._subject_limiter = SlidingWindowLimiter(rate_limit_per_minute, clock=clock)
        self._failure_limiter = SlidingWindowLimiter(failure_budget_per_minute, clock=clock)
        self._throttle = ClientFailureThrottle(
            per_client_limit=client_failure_budget_per_minute,
            shared_limit=failure_budget_per_minute,
            clock=clock,
        )
        self._tally = rejection_tally if rejection_tally is not None else RejectionTally()
        self._unavailable_until = 0.0
        self._unavailable_lock = threading.Lock()
        challenge = 'Bearer error="invalid_token"'
        if resource_metadata_url is not None:
            challenge += f', resource_metadata="{resource_metadata_url}"'
        # One challenge string for every 401 — never varies by rejection
        # reason, so responses carry no subject-existence oracle.
        self._challenge = challenge

    @property
    def rejection_tally(self) -> RejectionTally:
        """The in-memory counts of requests turned away before any token
        was judged; `imsg mcp public` writes them out
        (`imsg.mcp.audit.RejectionTallyWriter`)."""
        return self._tally

    # -- rejection constructors (single source of response shapes) ----------

    def _unauthorized(self) -> Rejection:
        return Rejection(status=401, code="UNAUTHORIZED", www_authenticate=self._challenge)

    def _rate_limited(self) -> Rejection:
        return Rejection(status=429, code="RATE_LIMITED", www_authenticate=None)

    def _unavailable(self) -> Rejection:
        return Rejection(status=503, code="UNAVAILABLE", www_authenticate=None)

    # -- audit -------------------------------------------------------------

    def _record(
        self,
        *,
        subject: str | None,
        subject_ok: bool,
        tool: str | None,
        params: Mapping[str, object] | None,
        result_count: int | None,
        latency_ms: int | None,
        error: str | None,
    ) -> Rejection | None:
        """Write one audit row; on failure, force a 503 denial.

        The audit trail is what AT-1 step 4 interrogates; a request that
        cannot be audited is not served.
        """
        rec = AuditRecord(
            surface="public",
            subject=subject,
            subject_ok=subject_ok,
            tool=tool,
            params_sha256=hash_params(params),
            result_count=result_count,
            latency_ms=latency_ms,
            error=error,
        )
        try:
            self._audit.record(rec)
        except AuditWriteError:
            return self._unavailable()
        return None

    def count_unauthenticated_rejection(self, client: str | None) -> None:
        """Count one request the transport refused before looking for a
        token — a duplicated header, a bad `Host` or `Origin` — against
        `client`, like any other rejection that judged no token."""
        self._tally.add("UNAUTHORIZED")
        self._throttle.note_failure(client)

    # -- validation core ---------------------------------------------------

    def _validate_claims(self, claims: TokenIntrospection) -> _CacheEntry:
        """Map introspection claims to a cache-ready verdict.

        Every rejection here is a 401 externally; internally we keep the
        raw subject for subject-mismatch so repeat requests keep auditing
        it (SPEC §10.4 item 3).
        """
        now = self._clock()
        negative_ttl = min(self._cache_ttl, _NEGATIVE_CACHE_CAP_SECONDS)

        def deny(subject: str | None) -> _CacheEntry:
            return _CacheEntry(
                expires_at=now + negative_ttl,
                allowed=False,
                subject=subject,
                scopes=frozenset(),
            )

        if claims.expires_in_seconds <= 0:
            return deny(None)
        if not _ct_equal(claims.audience, self._client_id):
            return deny(None)
        if claims.authorized_party is not None and not _ct_equal(
            claims.authorized_party, self._client_id
        ):
            return deny(None)
        if not self._required_scopes.issubset(claims.scopes):
            return deny(None)
        if not _SUBJECT_RE.match(claims.subject):
            # Not a numeric Google sub at all — reject before comparison;
            # audit the raw value (truncated by the sink).
            return deny(claims.subject)
        if not _ct_equal(claims.subject, self._owner_subject):
            return deny(claims.subject)
        positive_ttl = min(self._cache_ttl, float(claims.expires_in_seconds))
        return _CacheEntry(
            expires_at=now + positive_ttl,
            allowed=True,
            subject=self._owner_subject,
            scopes=claims.scopes,
        )

    def _decide(
        self,
        authorization: str | None,
        *,
        tool: str | None,
        params: Mapping[str, object] | None,
        client: str | None,
        charge: RateLimitCharge | None,
        io_allowed: bool,
    ) -> Admission | Rejection | _NeedsIO:
        """Validate one request: header → verdict cache → client throttle →
        circuit breaker → failure budget → live introspection → subject
        rate limit.

        Two kinds of rejection, by whether a token was judged:
        `counted` ones (no token, a throttled client, the budget or the
        breaker) go to the in-memory tally; `recorded` ones (a token
        introspected or cached as rejected, the owner over the rate
        limit) get an audit row, which is I/O. With `io_allowed=False`
        anything needing I/O returns `_NEEDS_IO` before any side effect,
        so the caller can run the same check again off the event loop.

        Only a failure the client caused counts against its throttle: no
        token or a malformed one, a token Google or this gate rejected.
        A 503 while Google is unreachable, a 429 from the shared budget,
        and a 429 from the throttle itself do not — otherwise an outage,
        or a client retrying through one, would keep an address
        throttled after the cause had gone, the owner's included.
        """
        started = self._clock()

        def latency() -> int:
            return int((self._clock() - started) * 1000)

        def counted(rej: Rejection, *, failure: bool) -> Rejection:
            self._tally.add(rej.code)
            if failure:
                self._throttle.note_failure(client)
            return rej

        def recorded(rej: Rejection, subject: str | None, *, failure: bool) -> Rejection:
            if failure:
                self._throttle.note_failure(client)
            audit_failure = self._record(
                subject=subject,
                subject_ok=False,
                tool=tool,
                params=params,
                result_count=None,
                latency_ms=latency(),
                error=rej.code,
            )
            return audit_failure if audit_failure is not None else rej

        match = (
            _BEARER_RE.match(authorization)
            if authorization is not None and len(authorization) <= _MAX_TOKEN_CHARS
            else None
        )
        if match is None:
            # No usable token: nothing to judge and nothing to exempt, so a
            # throttled client is refused as throttled.
            if self._throttle.is_throttled(client):
                return counted(self._rate_limited(), failure=False)
            return counted(self._unauthorized(), failure=True)
        token = match.group("token")
        key = _token_key(token)

        cached = self._cache.get(key)
        # Only a token already cached as the owner's is exempt from the
        # throttle: it must never lock the owner out of a session that is
        # already working, whoever shares its address.
        if (cached is None or not cached.allowed) and self._throttle.is_throttled(client):
            return counted(self._rate_limited(), failure=False)

        if cached is not None:
            entry = cached
        else:
            with self._unavailable_lock:
                if self._clock() < self._unavailable_until:
                    return counted(self._unavailable(), failure=False)
            if not self._failure_limiter.would_allow("introspection-failures"):
                # Failure budget exhausted: refuse before spending an
                # outbound call. 429, uncached (the budget window is its
                # own clock).
                return counted(self._rate_limited(), failure=False)
            if not io_allowed:
                return _NEEDS_IO
            try:
                claims = self._introspector.introspect(token)
            except TokenInvalidError:
                self._failure_limiter.note("introspection-failures")
                entry = _CacheEntry(
                    expires_at=self._clock() + min(self._cache_ttl, _NEGATIVE_CACHE_CAP_SECONDS),
                    allowed=False,
                    subject=None,
                    scopes=frozenset(),
                )
                self._cache.put(key, entry)
            except IntrospectionUnavailableError:
                # Fail closed (503) and cool down: while tokeninfo is down
                # we answer from the breaker instead of stacking slow
                # timeouts.
                with self._unavailable_lock:
                    self._unavailable_until = self._clock() + _UNAVAILABLE_COOLDOWN_SECONDS
                return counted(self._unavailable(), failure=False)
            else:
                entry = self._validate_claims(claims)
                if not entry.allowed:
                    self._failure_limiter.note("introspection-failures")
                self._cache.put(key, entry)

        if not entry.allowed:
            if not io_allowed:
                return _NEEDS_IO
            return recorded(self._unauthorized(), entry.subject, failure=True)

        subject = entry.subject
        if subject is None:
            # Structurally unreachable (allowed entries always carry the
            # owner subject) — but if it ever happens, deny, don't guess.
            if not io_allowed:
                return _NEEDS_IO
            return recorded(self._unauthorized(), None, failure=True)

        # The subject's rate limit applies to every request, cache hit or
        # not — once. A receipt from the transport's check of this same
        # request pays for it instead of a second event.
        new_charge: RateLimitCharge | None = None
        if charge is None or not charge.spend(self, key):
            if not self._subject_limiter.allow(subject):
                if not io_allowed:
                    return _NEEDS_IO
                return recorded(self._rate_limited(), subject, failure=False)
            new_charge = RateLimitCharge(self, key)

        return Admission(
            request=AuthorizedRequest(subject=subject, scopes=entry.scopes), charge=new_charge
        )

    # -- public API --------------------------------------------------------

    def authorize(
        self,
        authorization: str | None,
        *,
        tool: str | None = None,
        params: Mapping[str, object] | None = None,
        client: str | None = None,
        charge: RateLimitCharge | None = None,
    ) -> AuthorizedRequest | Rejection:
        """Validate one request, doing whatever I/O that takes. Audits
        every rejection: an `mcp_audit` row when a token was judged, the
        in-memory tally when none was.

        On ALLOW the audit row is written by :meth:`dispatch` (or
        :meth:`audit_allowed`) once the outcome is known — `dispatch` is
        the supported transport path and guarantees the row.
        """
        verdict = self._decide(
            authorization,
            tool=tool,
            params=params,
            client=client,
            charge=charge,
            io_allowed=True,
        )
        assert not isinstance(verdict, _NeedsIO)  # io_allowed=True always decides
        if isinstance(verdict, Rejection):
            return verdict
        return verdict.request

    def admit(
        self, authorization: str | None, *, client: str | None, io_allowed: bool
    ) -> Admission | Rejection | None:
        """The transport guard's check of one HTTP request (see the class
        docstring). `None` only when `io_allowed` is false and the answer
        needs a tokeninfo call or an audit row: call again with
        `io_allowed=True`, off the event loop. The admission's `charge`
        is the rate-limit event this request paid, for :meth:`dispatch`."""
        verdict = self._decide(
            authorization,
            tool=None,
            params=None,
            client=client,
            charge=None,
            io_allowed=io_allowed,
        )
        if isinstance(verdict, _NeedsIO):
            return None
        return verdict

    def audit_allowed(
        self,
        request: AuthorizedRequest,
        *,
        tool: str | None,
        params: Mapping[str, object] | None,
        result_count: int | None,
        latency_ms: int | None,
        error: str | None,
    ) -> Rejection | None:
        """Write the accept-path audit row. Returns a 503 rejection on audit failure."""
        return self._record(
            subject=request.subject,
            subject_ok=True,
            tool=tool,
            params=params,
            result_count=result_count,
            latency_ms=latency_ms,
            error=error,
        )

    def dispatch[T](
        self,
        authorization: str | None,
        *,
        tool: str,
        params: Mapping[str, object] | None,
        handler: Callable[[AuthorizedRequest], ToolOutcome[T]],
        charge: RateLimitCharge | None = None,
        client: str | None = None,
    ) -> DispatchResult[T]:
        """The supported request path: authorize, run, audit — atomically.

        The handler runs only after validation, receives the frozen
        AuthorizedRequest, and its outcome is audited in the same call.
        If the handler raises, an INTERNAL row is written and the
        exception propagates (the transport maps it to §10.1 INTERNAL —
        never a stack trace). If the accept-path audit row cannot be
        written, the payload is withheld and the request answers 503:
        the tools are read-only, so dropping a computed result is safe,
        and an unaudited disclosure is worse than a retry.

        `charge` is the transport's receipt for this request's rate-limit
        event (:class:`RateLimitCharge`); the token is re-validated in
        full either way.
        """
        started = self._clock()
        verdict = self.authorize(
            authorization, tool=tool, params=params, client=client, charge=charge
        )
        if isinstance(verdict, Rejection):
            return DispatchResult(rejection=verdict, payload=None)

        try:
            outcome = handler(verdict)
        except Exception:
            self._record(
                subject=verdict.subject,
                subject_ok=True,
                tool=tool,
                params=params,
                result_count=None,
                latency_ms=int((self._clock() - started) * 1000),
                error="INTERNAL",
            )
            raise

        audit_failure = self.audit_allowed(
            verdict,
            tool=tool,
            params=params,
            result_count=outcome.result_count,
            latency_ms=int((self._clock() - started) * 1000),
            error=outcome.error,
        )
        if audit_failure is not None:
            return DispatchResult(rejection=audit_failure, payload=None)
        return DispatchResult(rejection=None, payload=outcome.payload)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _resolve_client_id(raw: str) -> str:
    """Accept a literal client id or an env:/keychain:/file: reference.

    The config schema types `oauth.client_id` as a plain string while the
    documented example uses `env:IMSG_OAUTH_CLIENT_ID`; treating a
    ref-shaped value literally would make the audience check unsatisfiable
    (fail closed but silently broken), so refs are resolved here — every
    kind `imsg.config.secrets.SecretRef` accepts, `file:` included since
    secrets may live in owner-only files.
    """
    if raw.startswith(("env:", "keychain:", "file:")):
        from imsg.config.secrets import SecretRef

        return SecretRef.parse(raw).resolve()
    return raw


def resource_metadata_url_for(external_url: str) -> str:
    """RFC 9728 protected-resource metadata URL for the public origin."""
    parts = urllib.parse.urlsplit(external_url)
    return f"{parts.scheme}://{parts.netloc}/.well-known/oauth-protected-resource"


def build_public_gate(
    config: McpPublicConfig,
    *,
    audit: AuditSink,
    introspector: TokenIntrospector | None = None,
    rejection_tally: RejectionTally | None = None,
    clock: Callable[[], float] = monotonic,
) -> PublicAuthGate:
    """Construct the gate from validated config, refusing startup on any gap.

    Hard requirement 4: an empty, missing, or unresolvable pinned subject
    refuses startup — there is no allow-all fallback and no parameter
    that creates one.
    """
    from imsg.errors import ImsgError

    oauth = config.oauth
    if oauth.owner_subject is None:
        raise PublicSurfaceStartupError(
            "public surface refused to start: mcp.public.oauth.owner_subject "
            "is not configured (hard requirement 4: fail closed, never allow-all)"
        )
    try:
        owner_subject = oauth.owner_subject.resolve()
    except ImsgError as exc:
        raise PublicSurfaceStartupError(
            "public surface refused to start: pinned owner_subject could not "
            "be resolved (hard requirement 4)"
        ) from exc

    if oauth.client_id is None:
        raise PublicSurfaceStartupError(
            "public surface refused to start: mcp.public.oauth.client_id is "
            "not configured — the audience check requires it (SPEC §10.4)"
        )
    try:
        client_id = _resolve_client_id(oauth.client_id)
    except (ImsgError, ValueError) as exc:
        raise PublicSurfaceStartupError(
            "public surface refused to start: OAuth client id reference could "
            "not be resolved (SPEC §10.4)"
        ) from exc

    metadata_url = (
        resource_metadata_url_for(config.external_url)
        if config.external_url is not None
        else None
    )
    return PublicAuthGate(
        client_id=client_id,
        owner_subject=owner_subject,
        introspector=introspector if introspector is not None else GoogleTokeninfoIntrospector(),
        audit=audit,
        cache_ttl_seconds=oauth.tokeninfo_cache_ttl_seconds,
        rate_limit_per_minute=config.rate_limit_per_minute,
        failure_budget_per_minute=config.failure_budget_per_minute,
        client_failure_budget_per_minute=config.client_failure_budget_per_minute,
        rejection_tally=rejection_tally,
        resource_metadata_url=metadata_url,
        clock=clock,
    )


__all__ = [
    "DEFAULT_CLIENT_FAILURE_BUDGET_PER_MINUTE",
    "DEFAULT_FAILURE_BUDGET_PER_MINUTE",
    "GOOGLE_TOKENINFO_URL",
    "RESPONSE_CACHE_HEADERS",
    "Admission",
    "AuthorizedRequest",
    "DispatchResult",
    "GoogleTokeninfoIntrospector",
    "PublicAuthGate",
    "RateLimitCharge",
    "Rejection",
    "TokenIntrospection",
    "TokenIntrospector",
    "ToolOutcome",
    "build_public_gate",
    "parse_tokeninfo_response",
    "resource_metadata_url_for",
]
