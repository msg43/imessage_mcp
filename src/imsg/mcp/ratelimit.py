"""Sliding-window rate limiting for the public MCP surface (SPEC §10.4).

Three limiter roles:

- **Per-subject** (`rate_limit_per_minute`, :class:`SlidingWindowLimiter`):
  keys are *validated* OAuth subjects only — in practice exactly one (the
  owner) — so the key space is bounded by construction. Charged once per
  HTTP request, including verdict-cache hits: a cached ALLOW must not
  become a rate-limit bypass. A tool call is one HTTP request, so it is
  one charge (`imsg.mcp.auth.RateLimitCharge` carries the transport's
  charge to `dispatch`).
- **Global failure budget** (beyond spec, defense in depth,
  :class:`SlidingWindowLimiter`): a single shared bucket counting failed
  introspections. Without it, a stream of distinct garbage tokens turns
  this server into an amplifier against Google's tokeninfo (each miss is
  an outbound call) and can exhaust our client's upstream quota. When the
  budget is exhausted, cache-miss requests are refused 429 *before* any
  network call. The deliberate tradeoff (D7.3): under an attack spread
  over many addresses the owner's re-validation can also be throttled
  (availability), which is the correct direction for this system —
  confidentiality failure is irreversible, downtime is not.
- **Per-client failure throttle** (:class:`ClientFailureThrottle`):
  failures per client address — no token, a malformed one, a rejected
  one. A client over its budget within the window is answered 429 before
  any further work — no introspection, no audit row — so one source can
  neither spend the global budget (which is what used to lock the
  owner's next token out) nor make the server write a row per request.

Time source is an injectable monotonic clock; wall-clock changes cannot
widen a window.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Callable
from time import monotonic


class SlidingWindowLimiter:
    """True sliding window over `window_seconds`, per key. Thread-safe."""

    def __init__(
        self,
        limit: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("rate limit must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window must be positive")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        """Record an event for `key` iff it fits in the window; return the verdict."""
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            q = self._events.get(key)
            if q is None:
                q = deque()
                self._events[key] = q
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self._limit:
                return False
            q.append(now)
            return True

    def note(self, key: str) -> None:
        """Record an event for `key` unconditionally (used for failure counting)."""
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            q = self._events.get(key)
            if q is None:
                q = deque()
                self._events[key] = q
            while q and q[0] <= cutoff:
                q.popleft()
            q.append(now)

    def would_allow(self, key: str) -> bool:
        """Check without recording — for pre-network refusal decisions."""
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            q = self._events.get(key)
            if q is None:
                return True
            while q and q[0] <= cutoff:
                q.popleft()
            return len(q) < self._limit


_UNIDENTIFIED = ""
"""The one bucket every request with no identifiable client shares. An
empty string is never a client key (`imsg.mcp.tools.public_server.
client_key` returns an address or `None`)."""

DEFAULT_MAX_TRACKED_CLIENTS = 4096


class ClientFailureThrottle:
    """Rejected requests per client over a sliding window, and the verdict
    "this client has been rejected too often to be worth more work".

    A client is throttled while it has at least `per_client_limit`
    failures in the last `window_seconds`, and let back in once the oldest
    of them leaves the window. The gate notes only failures the client
    caused (`imsg.mcp.auth.PublicAuthGate._decide`): the requests it
    refuses while the client is throttled do not extend the throttle, so
    a client that keeps sending gets at most `per_client_limit` real
    attempts a window, and one caught up in an outage is not kept out
    after it ends.

    Requests whose client cannot be identified (`client=None`: no
    forwarded address, or a direct local connection) share one bucket
    with its own, larger `shared_limit` — the fallback when nothing about
    the request says who sent it.

    Memory is bounded twice, because an attacker chooses the keys: each
    client keeps at most its limit's worth of timestamps (all the verdict
    needs), and at most `max_clients` clients are kept, the least recently
    rejected dropped first. A dropped client starts again with a clean
    record; the global failure budget in `imsg.mcp.auth.PublicAuthGate`
    is what bounds work across many addresses. Thread-safe.
    """

    def __init__(
        self,
        *,
        per_client_limit: int,
        shared_limit: int,
        window_seconds: float = 60.0,
        max_clients: int = DEFAULT_MAX_TRACKED_CLIENTS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if per_client_limit < 1 or shared_limit < 1:
            raise ValueError("failure limits must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window must be positive")
        if max_clients < 1:
            raise ValueError("max_clients must be >= 1")
        self._per_client_limit = per_client_limit
        self._shared_limit = shared_limit
        self._window = window_seconds
        self._max_clients = max_clients
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()

    def _limit_for(self, key: str) -> int:
        return self._shared_limit if key == _UNIDENTIFIED else self._per_client_limit

    def _expire(self, q: deque[float], now: float) -> None:
        cutoff = now - self._window
        while q and q[0] <= cutoff:
            q.popleft()

    def note_failure(self, client: str | None) -> None:
        """Record one rejected request from `client`."""
        key = _UNIDENTIFIED if client is None else client
        now = self._clock()
        with self._lock:
            q = self._failures.get(key)
            if q is None:
                q = deque(maxlen=self._limit_for(key))
                self._failures[key] = q
            self._expire(q, now)
            q.append(now)  # a full deque drops its oldest entry
            self._failures.move_to_end(key)
            while len(self._failures) > self._max_clients:
                self._failures.popitem(last=False)

    def is_throttled(self, client: str | None) -> bool:
        """Whether `client` has used up its failures for the window."""
        key = _UNIDENTIFIED if client is None else client
        now = self._clock()
        with self._lock:
            q = self._failures.get(key)
            if q is None:
                return False
            self._expire(q, now)
            return len(q) >= self._limit_for(key)

    def tracked_clients(self) -> int:
        """How many clients have a record right now — what a test of the
        memory bound asserts on."""
        with self._lock:
            return len(self._failures)


__all__ = ["DEFAULT_MAX_TRACKED_CLIENTS", "ClientFailureThrottle", "SlidingWindowLimiter"]
