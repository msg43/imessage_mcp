"""Rate throttle for S5a materialization (`--rate` files/min, default
60, SPEC §8 S5a) — a fixed-interval pacer: spacing every materialization
call evenly avoids a burst against iCloud followed by a long stall,
which is closer to what "N files per minute, sustained" means than a
bursty token bucket would be.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class RateThrottle:
    """Call `wait()` once, immediately before each file's materialization."""

    def __init__(
        self,
        files_per_minute: float,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if files_per_minute <= 0:
            raise ValueError("files_per_minute must be > 0")
        self._interval = 60.0 / files_per_minute
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self._interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


__all__ = ["RateThrottle"]
