"""Drop a server's models after it has sat idle, and load them again on
the next tool call.

**Why.** Every MCP server process loads its own copy of the retrieval
models — the text embedder, the PE-Core text tower and the reranker —
and before this module kept them until it exited. `imsg mcp local` is one
process per client session (each Claude Code or Claude Desktop session
on another machine runs its own `ssh ... imsg mcp local`), so a handful
of idle sessions held a handful of full model sets. On the index host,
five idle Python processes held about 62 GiB of a 64 GiB machine, one of
them 15.5 GiB and four 11.7 GiB each, and the host ran out of memory and
hung (kernel panic report, 2026-09-24).

**What happens.** A watchdog thread checks every few seconds. When the
models are loaded, no retrieval tool call is in flight, and none has
started or finished for `mcp.idle_unload_seconds`, it:

1. marks the warm-up `unloaded` (`BackgroundWarmUp.mark_unloaded`), and
2. queues the unload on the model thread (`imsg.retrieval.model_thread`),
   where every model call runs: each provider drops its weights, then
   :func:`release_freed_memory` runs a garbage collection and empties
   MLX's buffer cache and torch's MPS cache, so the memory goes back to
   the system rather than staying in an allocator's free list.

Both happen under this object's lock, the same lock a tool call takes to
register itself (:meth:`IdleModelUnloader.call`). So a call either
registers first — and then nothing unloads while it runs — or registers
after, finds the warm-up `unloaded`, and starts the reload, which the
model thread runs after the unload it queued behind. The call then waits
for the reload exactly as a call at startup waits for the first load:
up to the server's warm-up wait (90 s), answering `WARMING_UP` with an
estimate if the reload takes longer (`imsg.mcp.tools.local_server`).

A failed warm-up is never unloaded — it is reported as failed until the
server restarts — and one still warming is never interrupted.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import sys
import threading
import time
from collections.abc import Callable, Iterator

from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpPhase, log_to_stderr
from imsg.retrieval.model_thread import ModelThread

DEFAULT_IDLE_UNLOAD_SECONDS = 600
"""Ten minutes: long enough that a conversation's queries, usually a few
seconds to a few minutes apart, all hit warm models, and short enough that
a session left open overnight gives its memory back the same evening."""

MIN_IDLE_UNLOAD_SECONDS = 60
"""The shortest idle period the config accepts (other than 0, which turns
unloading off). A reload costs tens of seconds; unloading between the
queries of one conversation would cost more than it saves."""

MAX_CHECK_INTERVAL_SECONDS = 15.0
"""The watchdog checks at least this often, so an unload happens at most
this long after the idle period ends."""

_GIB = 2**30


def release_freed_memory() -> str:
    """Hand memory that nothing references any more back to the system:
    one garbage collection, then MLX's buffer cache and torch's MPS cache
    emptied — each only if that runtime is already imported, so a server
    on the fake backend never imports either. Returns a short description
    of what MLX still holds, for the log line (empty when MLX is not
    loaded). Best-effort: an allocator that refuses the hint leaves the
    memory cached, not the process broken, so nothing here raises."""
    gc.collect()
    detail = ""
    if "mlx.core" in sys.modules:
        with contextlib.suppress(Exception):
            mx = importlib.import_module("mlx.core")
            clear_cache = getattr(mx, "clear_cache", None)
            if clear_cache is None:  # MLX before 0.24 kept it under mx.metal
                clear_cache = getattr(getattr(mx, "metal", None), "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
            active = getattr(mx, "get_active_memory", None)
            cache = getattr(mx, "get_cache_memory", None)
            if callable(active) and callable(cache):
                detail = (
                    f"MLX now holds {active() / _GIB:.2f} GiB active, "
                    f"{cache() / _GIB:.2f} GiB cached"
                )
    if "torch" in sys.modules:
        with contextlib.suppress(Exception):
            torch = importlib.import_module("torch")
            mps = getattr(torch, "mps", None)
            empty_cache = getattr(mps, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
    return detail


class IdleModelUnloader:
    """Unloads a server's models after `idle_seconds` with no retrieval
    tool call, and reloads them when the next one arrives.

    `unload` drops the models; it runs on `model_thread`. `idle_seconds`
    of 0 turns the whole thing off: :meth:`call` still counts calls but
    nothing is ever unloaded. `clock` is injectable so tests can move
    time by hand and call :meth:`unload_if_idle` themselves instead of
    starting the watchdog."""

    def __init__(
        self,
        *,
        warm_up: BackgroundWarmUp,
        model_thread: ModelThread,
        unload: Callable[[], object],
        idle_seconds: float,
        log: Callable[[str], None] = log_to_stderr,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if idle_seconds < 0:
            raise ValueError(f"idle_seconds must be >= 0, got {idle_seconds}")
        self._warm_up = warm_up
        self._model_thread = model_thread
        self._unload = unload
        self._idle_seconds = float(idle_seconds)
        self._log_line = log
        self._clock = clock
        self._lock = threading.Lock()
        self._in_flight = 0
        self._last_activity = clock()
        self._ready_seen = False
        self._unloads = 0
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._idle_seconds > 0

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def unloads(self) -> int:
        """How many times the models have been unloaded."""
        with self._lock:
            return self._unloads

    # -- the tool-call side ----------------------------------------------------

    @contextlib.contextmanager
    def call(self) -> Iterator[None]:
        """Wrap one retrieval tool call, from before it waits for the
        warm-up until its answer is computed. While any call is inside,
        nothing unloads. Entering starts the reload if the models were
        unloaded; the caller then waits for the warm-up as it always
        does."""
        with self._lock:
            self._in_flight += 1
            self._last_activity = self._clock()
            # Under the lock, so the reload is queued on the model thread
            # after any unload this lock has already queued.
            if self._warm_up.status().phase is WarmUpPhase.UNLOADED:
                self._warm_up.start()
        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1
                self._last_activity = self._clock()

    # -- the idle side -----------------------------------------------------------

    def unload_if_idle(self) -> bool:
        """Unload now if the models are loaded, nothing is in flight, and
        nothing has happened for `idle_seconds`. Returns whether an
        unload was queued. The idle period counts from the later of the
        last call and the first check that saw the warm-up ready, so a
        server whose warm-up took a while does not unload the moment it
        finishes."""
        if not self.enabled:
            return False
        with self._lock:
            if self._warm_up.status().phase is not WarmUpPhase.READY:
                self._ready_seen = False
                return False
            now = self._clock()
            if not self._ready_seen:
                self._ready_seen = True
                self._last_activity = max(self._last_activity, now)
                return False
            if self._in_flight > 0:
                return False
            idle = now - self._last_activity
            if idle < self._idle_seconds:
                return False
            if not self._warm_up.mark_unloaded():
                return False
            self._ready_seen = False
            self._unloads += 1
            self._log(
                f"idle for {idle:.0f} s with no tool call: unloading the models; "
                f"the next tool call reloads them"
            )
            try:
                self._model_thread.submit(self._run_unload)
            except RuntimeError as exc:  # the model thread is closed: the server is exiting
                self._log(f"unload not queued: {exc}")
            return True

    def _run_unload(self) -> None:
        """On the model thread."""
        started = time.perf_counter()
        try:
            self._unload()
        except Exception as exc:
            # The providers load lazily, so a model that did not unload
            # cleanly is still one the next call can use; say so and go on.
            self._log(f"unload failed ({type(exc).__name__}: {exc}); memory may not be returned")
        detail = release_freed_memory()
        suffix = f"; {detail}" if detail else ""
        self._log(f"models unloaded in {time.perf_counter() - started:.1f} s{suffix}")

    # -- the watchdog ------------------------------------------------------------

    def check_interval_seconds(self) -> float:
        return max(1.0, min(self._idle_seconds / 10, MAX_CHECK_INTERVAL_SECONDS))

    def start_watchdog(self) -> None:
        """Start the daemon thread that calls :meth:`unload_if_idle`
        periodically. Idempotent; does nothing when unloading is off. A
        daemon, like the model thread, so it never keeps a server whose
        client has gone alive."""
        if not self.enabled or self._watchdog is not None:
            return
        self._watchdog = threading.Thread(
            target=self._watch, name="imsg-idle-unload", daemon=True
        )
        self._watchdog.start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        interval = self.check_interval_seconds()
        while not self._stop.wait(interval):
            try:
                self.unload_if_idle()
            except Exception as exc:  # the watchdog must outlive one bad check
                self._log(f"idle check failed: {type(exc).__name__}: {exc}")

    def _log(self, line: str) -> None:
        with contextlib.suppress(Exception):
            self._log_line(line)


__all__ = [
    "DEFAULT_IDLE_UNLOAD_SECONDS",
    "MAX_CHECK_INTERVAL_SECONDS",
    "MIN_IDLE_UNLOAD_SECONDS",
    "IdleModelUnloader",
    "release_freed_memory",
]
