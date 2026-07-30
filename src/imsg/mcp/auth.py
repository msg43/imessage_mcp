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
- Every request produces an `mcp_audit` row; an unauditable request is
  denied. Rejected subjects are logged raw (SPEC §10.4 item 3); tokens
  never are — cache and audit keys use sha256(token).
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

from imsg.mcp.audit import AuditRecord, AuditSink, hash_params
from imsg.mcp.errors import (
    AuditWriteError,
    IntrospectionUnavailableError,
    PublicSurfaceStartupError,
    TokenInvalidError,
)
from imsg.mcp.ratelimit import SlidingWindowLimiter

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
_DEFAULT_FAILURE_BUDGET_PER_MINUTE = 60


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


class PublicAuthGate:
    """Owner-subject validation, rate limiting, and audit for the public surface.

    Construction validates its own inputs and refuses (raises
    PublicSurfaceStartupError) rather than degrade. By design there is no
    parameter that skips a validation step — tests assert this
    structurally against the signature.
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
        failure_budget_per_minute: int = _DEFAULT_FAILURE_BUDGET_PER_MINUTE,
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
        self._unavailable_until = 0.0
        self._unavailable_lock = threading.Lock()
        challenge = 'Bearer error="invalid_token"'
        if resource_metadata_url is not None:
            challenge += f', resource_metadata="{resource_metadata_url}"'
        # One challenge string for every 401 — never varies by rejection
        # reason, so responses carry no subject-existence oracle.
        self._challenge = challenge

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

    def _resolve_token(self, token: str) -> _CacheEntry | Rejection:
        """Cache → circuit breaker → failure budget → live introspection."""
        key = _token_key(token)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        with self._unavailable_lock:
            if self._clock() < self._unavailable_until:
                return self._unavailable()

        if not self._failure_limiter.would_allow("introspection-failures"):
            # Failure budget exhausted: refuse before spending an outbound
            # call. 429, uncached (the budget window is its own clock).
            return self._rate_limited()

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
            return entry
        except IntrospectionUnavailableError:
            # Fail closed (503) and cool down: while tokeninfo is down we
            # answer from the breaker instead of stacking slow timeouts.
            with self._unavailable_lock:
                self._unavailable_until = self._clock() + _UNAVAILABLE_COOLDOWN_SECONDS
            return self._unavailable()

        entry = self._validate_claims(claims)
        if not entry.allowed:
            self._failure_limiter.note("introspection-failures")
        self._cache.put(key, entry)
        return entry

    # -- public API --------------------------------------------------------

    def authorize(
        self,
        authorization: str | None,
        *,
        tool: str | None = None,
        params: Mapping[str, object] | None = None,
    ) -> AuthorizedRequest | Rejection:
        """Validate one request. Writes the audit row for every rejection.

        On ALLOW the audit row is written by :meth:`dispatch` (or
        :meth:`audit_allowed`) once the outcome is known — `dispatch` is
        the supported transport path and guarantees the row.
        """
        started = self._clock()

        def latency() -> int:
            return int((self._clock() - started) * 1000)

        def rejected(rej: Rejection, subject: str | None) -> Rejection:
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

        if authorization is None:
            return rejected(self._unauthorized(), None)
        if len(authorization) > _MAX_TOKEN_CHARS:
            return rejected(self._unauthorized(), None)
        match = _BEARER_RE.match(authorization)
        if match is None:
            return rejected(self._unauthorized(), None)
        token = match.group("token")

        resolved = self._resolve_token(token)
        if isinstance(resolved, Rejection):
            return rejected(resolved, None)
        if not resolved.allowed:
            return rejected(self._unauthorized(), resolved.subject)

        # Rate limit applies on every request, cache hit or not.
        subject = resolved.subject
        if subject is None:
            # Structurally unreachable (allowed entries always carry the
            # owner subject) — but if it ever happens, deny, don't guess.
            return rejected(self._unauthorized(), None)
        if not self._subject_limiter.allow(subject):
            return rejected(self._rate_limited(), subject)

        return AuthorizedRequest(subject=subject, scopes=resolved.scopes)

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
        """
        started = self._clock()
        verdict = self.authorize(authorization, tool=tool, params=params)
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
    """Accept a literal client id or an env:/keychain: reference.

    The config schema types `oauth.client_id` as a plain string while the
    documented example uses `env:IMSG_OAUTH_CLIENT_ID`; treating a
    ref-shaped value literally would make the audience check unsatisfiable
    (fail closed but silently broken), so refs are resolved here.
    """
    if raw.startswith(("env:", "keychain:")):
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
        resource_metadata_url=metadata_url,
        clock=clock,
    )


__all__ = [
    "GOOGLE_TOKENINFO_URL",
    "RESPONSE_CACHE_HEADERS",
    "AuthorizedRequest",
    "DispatchResult",
    "GoogleTokeninfoIntrospector",
    "PublicAuthGate",
    "Rejection",
    "TokenIntrospection",
    "TokenIntrospector",
    "ToolOutcome",
    "build_public_gate",
    "parse_tokeninfo_response",
    "resource_metadata_url_for",
]
