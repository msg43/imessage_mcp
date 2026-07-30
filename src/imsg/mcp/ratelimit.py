"""Sliding-window rate limiting for the public MCP surface (SPEC §10.4).

Two limiter roles, both served by :class:`SlidingWindowLimiter`:

- **Per-subject** (`rate_limit_per_minute`): keys are *validated* OAuth
  subjects only — in practice exactly one (the owner) — so the key space
  is bounded by construction. Applied on every request, including verdict-
  cache hits: a cached ALLOW must not become a rate-limit bypass.
- **Pre-auth failure budget** (beyond spec, defense in depth): a single
  shared bucket counting failed introspections. Without it, a stream of
  distinct garbage tokens turns this server into an amplifier against
  Google's tokeninfo (each miss is an outbound call) and can exhaust our
  client's upstream quota. When the budget is exhausted, cache-miss
  requests are refused 429 *before* any network call. The deliberate
  tradeoff: under active attack the owner's re-validation can also be
  throttled (availability), which is the correct direction for this
  system — confidentiality failure is irreversible, downtime is not.

Time source is an injectable monotonic clock; wall-clock changes cannot
widen a window.
"""

from __future__ import annotations

import threading
from collections import deque
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


__all__ = ["SlidingWindowLimiter"]
