"""Warm the retrieval models — and the database's buffer pool — in the
background while the server already answers.

`imsg mcp local` used to load and warm every model before it answered
anything — 120.9 s in one run on the M2 Ultra host (2026-09-17), with
`initialize` answered 121.9 s after the process started. An MCP client
drops a server that does not answer the handshake in time; Claude Code
waits `MCP_TIMEOUT`, 30 s by default (code.claude.com/docs/en/env-vars).
The server now answers the handshake straight away and warms up here, on
the model thread (`imsg.retrieval.model_thread`), while tool calls wait
for it (`imsg.mcp.tools.local_server`).

What this module promises:

- **One logged pass.** `start()` logs the start, each step's completion is
  logged with its time (and with whatever the step itself returns, such as
  how many bytes a prewarm moved), then the total. A failure is logged once and
  the models after it are not loaded: a server that cannot load one model
  answers no tool call, instead of serving with some models loaded and
  others not.
- **A status any thread can read.** `status()` and `wait(timeout)` report
  the phase (not started, warming, ready, failed), the model loading now,
  and an estimate of the seconds remaining. `wait` never blocks past its
  timeout, and a warm-up that dies in an unexpected way still ends
  `failed`, so nothing waits on it forever.
- **A status another *process* can read, when asked for one.** `on_status`
  is called with the new status every time the phase or the running step
  changes — the public surface uses it to publish readiness to a file
  `imsg status` reads (`imsg.mcp.warm_up_readiness`), since the public
  transport has no unauthenticated endpoint to ask. Like the log
  callback, it can never change the outcome: an `on_status` that raises
  is swallowed.
- **The estimate** is what is left of the running step's estimated
  duration plus the estimated durations of the steps after it. Those are
  constants taken from measured runs (`RetrievalService.warm_up_steps`),
  set toward the slow end so the estimate errs long rather than short; the
  running step is never counted as having less than
  `RUNNING_STEP_FLOOR_SECONDS` left, which also covers a step that has run
  past its estimate.
- **A ready warm-up can be undone and run again.** `mark_unloaded()` moves
  a ready warm-up to `unloaded` once its models have been dropped for
  sitting idle (`imsg.retrieval.idle_unload`); `start()` then runs every
  step again, with the same logging, status and estimate as the first
  pass, so a tool call that arrives after an unload waits for the reload
  exactly the way one that arrives at startup waits for the first load.
- **No load the host has no memory for.** Given an `admission` check
  (`imsg.memory_admission`), every pass — the first and every reload —
  asks it on the model thread before the first step, after any unload
  queued ahead of it has run. Refused, no step runs: the warm-up enters
  `memory_busy`, which a waiting tool call treats as settled and answers
  at once with a retryable `WARMING_UP` error saying the host's memory
  is busy (`HostMemoryBusyError`). `start()` from `memory_busy` asks
  again, but no sooner than `admission_retry_seconds` after the last
  refusal, so a stream of calls cannot turn into a stream of
  measurements; refusals in a row are logged once every
  `REFUSAL_LOG_INTERVAL_SECONDS`.
"""

from __future__ import annotations

import contextlib
import math
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from imsg.errors import ImsgError
from imsg.memory_admission import AdmissionDecision, refused
from imsg.retrieval.errors import HostMemoryBusyError, WarmingUpError, WarmUpFailedError
from imsg.retrieval.model_thread import ModelThread

RUNNING_STEP_FLOOR_SECONDS = 5.0
"""The least time the running step is estimated to have left."""

MAX_FAILURE_CHARS = 500
"""A failure cause is cut to this length for the log line and tool errors."""

DEFAULT_ADMISSION_RETRY_SECONDS = 15.0
"""`memory.admission_retry_seconds`'s default, for callers built without a
config (tests)."""

REFUSAL_LOG_INTERVAL_SECONDS = 300.0
"""Refused loads in a row are logged at most this often: the public server
retries on its own for as long as the host is short, which may be hours."""


class WarmUpPhase(StrEnum):
    NOT_STARTED = "not_started"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"
    UNLOADED = "unloaded"
    """Was ready; the models were dropped after sitting idle. The next
    `start()` loads them again."""
    MEMORY_BUSY = "memory_busy"
    """Not loaded: the memory admission check refused the load. The next
    `start()` asks again (no sooner than the retry interval)."""


_STARTABLE = (WarmUpPhase.NOT_STARTED, WarmUpPhase.UNLOADED, WarmUpPhase.MEMORY_BUSY)
_PENDING = (WarmUpPhase.NOT_STARTED, WarmUpPhase.UNLOADED, WarmUpPhase.WARMING)
"""What `wait()` waits through: a pass that may still start or finish."""


@dataclass(frozen=True, slots=True)
class WarmUpStep:
    """One thing to warm: `run` does it (loads a model and pushes one
    throwaway input through it, or fills the database's buffer pool) and
    `estimated_seconds` is how long that is expected to take.

    A `run` that returns a non-empty string has that appended to its log
    line — the step's own numbers, which only it knows (how many bytes the
    prewarm moved, say)."""

    name: str
    estimated_seconds: float
    run: Callable[[], object]


@dataclass(frozen=True, slots=True)
class WarmUpStatus:
    phase: WarmUpPhase
    loading: str | None
    """The model loading now — set only while warming."""
    steps_done: int
    steps_total: int
    elapsed_seconds: float
    seconds_remaining: float
    """An estimate; 0 once ready or failed."""
    failure: str | None
    """The cause — set only once failed."""
    admission: AdmissionDecision | None = None
    """The refusal — set only while `memory_busy`."""
    retry_seconds: float = DEFAULT_ADMISSION_RETRY_SECONDS
    """How often a refused load is tried again."""

    @property
    def settled(self) -> bool:
        """Nothing a waiting call can gain by waiting longer: ready,
        failed, or refused for memory (retried on a later call)."""
        return self.phase in (WarmUpPhase.READY, WarmUpPhase.FAILED, WarmUpPhase.MEMORY_BUSY)

    @property
    def detail(self) -> str | None:
        """Why the models are not loaded, when the host's memory is why."""
        return self.admission.reason if self.admission is not None else None

    def raise_unless_ready(self, *, wait_bound_seconds: float, public: bool = False) -> None:
        """Return when ready; otherwise raise the tool error for this
        status: `WarmUpFailedError` naming the cause, `HostMemoryBusyError`
        when the host had no memory for the models, or `WarmingUpError`
        with the estimate. `wait_bound_seconds` is how long a call waits,
        which the warming-up message tells the caller. `public` keeps the
        memory refusal to numbers (`AdmissionDecision.public_summary`)."""
        if self.phase is WarmUpPhase.READY:
            return
        if self.phase is WarmUpPhase.FAILED:
            raise WarmUpFailedError(self.failure or "cause unknown")
        if self.phase is WarmUpPhase.MEMORY_BUSY:
            decision = self.admission
            if decision is None:  # pragma: no cover - memory_busy always carries its decision
                detail = "the host's memory is busy"
            else:
                detail = decision.public_summary() if public else decision.reason
            raise HostMemoryBusyError(
                detail=detail,
                retry_seconds=self.retry_seconds,
                wait_bound_seconds=wait_bound_seconds,
            )
        raise WarmingUpError(
            seconds_remaining=max(1, math.ceil(self.seconds_remaining)),
            loading=self.loading,
            wait_bound_seconds=wait_bound_seconds,
        )


def describe_failure(exc: BaseException) -> str:
    """One line naming what went wrong. An `ImsgError` message is written
    for the operator already; anything else keeps its type name, which is
    often the only clue (`RuntimeError: ...` from inside a runtime)."""
    message = " ".join(str(exc).split())
    if isinstance(exc, ImsgError) and message:
        text = message
    else:
        text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if len(text) > MAX_FAILURE_CHARS:
        text = text[: MAX_FAILURE_CHARS - 1] + "…"
    return text


def _steps(count: int) -> str:
    """The warm-up's own unit: a step is a model to load or the database's
    buffer pool to fill, so the log line cannot say "models"."""
    return "1 step" if count == 1 else f"{count} steps"


def log_to_stderr(line: str) -> None:
    with contextlib.suppress(OSError, ValueError):
        print(line, file=sys.stderr, flush=True)


class BackgroundWarmUp:
    """Runs `steps` once, in order, on `model_thread` — after `admission`,
    when one is given, says the host has the memory for them."""

    def __init__(
        self,
        steps: Sequence[WarmUpStep],
        *,
        model_thread: ModelThread,
        log: Callable[[str], None] = log_to_stderr,
        on_status: Callable[[WarmUpStatus], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        admission: Callable[[], AdmissionDecision] | None = None,
        admission_retry_seconds: float = DEFAULT_ADMISSION_RETRY_SECONDS,
    ) -> None:
        if admission_retry_seconds <= 0:
            raise ValueError(f"admission_retry_seconds must be > 0, got {admission_retry_seconds}")
        self._steps = tuple(steps)
        self._model_thread = model_thread
        self._log_line = log
        self._on_status = on_status
        self._clock = clock
        self._admission = admission
        self._retry_seconds = float(admission_retry_seconds)
        self._condition = threading.Condition()
        self._phase = WarmUpPhase.NOT_STARTED
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._step_index = 0
        self._step_started_at: float | None = None
        self._failure: str | None = None
        self._refusal: AdmissionDecision | None = None
        self._refused_at: float | None = None
        self._refusal_logged_at: float | None = None
        self._ever_ready = False

    @property
    def steps(self) -> tuple[WarmUpStep, ...]:
        return self._steps

    def start(self) -> None:
        """Begin warming on the model thread and return at once.
        Idempotent while warming, ready or failed; from `unloaded` it runs
        every step again. Never raises: a warm-up that cannot even be
        queued ends failed, like one whose model cannot load."""
        with self._condition:
            if self._phase not in _STARTABLE:
                return
            retrying = self._phase is WarmUpPhase.MEMORY_BUSY
            if retrying and self._retry_wait_left() > 0:
                return  # refused a moment ago: the answer has not changed yet
            reloading = self._phase is WarmUpPhase.UNLOADED or (retrying and self._ever_ready)
            self._phase = WarmUpPhase.WARMING
            self._started_at = self._clock()
            self._finished_at = None
            self._step_index = 0
            self._step_started_at = None
        if not retrying:
            names = ", ".join(step.name for step in self._steps)
            what = "reload" if reloading else "warm-up"
            self._log(f"{what} started: {_steps(len(self._steps))} in the background ({names})")
        self._publish()
        try:
            self._model_thread.submit(self._run)
        except Exception as exc:
            self._fail(None, describe_failure(exc))

    def mark_unloaded(self) -> bool:
        """Record that the models were dropped, so the next `start()`
        loads them again. Only a ready warm-up can be unloaded: one still
        warming has a pass in progress, and a failed one is reported as
        failed until the server restarts. Returns whether the phase
        changed. The caller drops the models itself, on the model thread,
        and must queue that before anything can call `start()` again
        (`imsg.retrieval.idle_unload.IdleModelUnloader` holds its own lock
        across both)."""
        with self._condition:
            if self._phase is not WarmUpPhase.READY:
                return False
            self._phase = WarmUpPhase.UNLOADED
            self._started_at = None
            self._finished_at = None
            self._step_index = 0
            self._step_started_at = None
            self._publish()
            self._condition.notify_all()
            return True

    def status(self) -> WarmUpStatus:
        with self._condition:
            return self._status()

    def _retry_wait_left(self) -> float:
        """Caller holds the condition."""
        if self._refused_at is None:
            return 0.0
        return max(0.0, self._retry_seconds - (self._clock() - self._refused_at))

    def wait(self, timeout: float) -> WarmUpStatus:
        """Block until the warm-up is ready, failed or refused for memory,
        or `timeout` seconds pass, whichever is first; return the status
        then."""
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._condition:
            while self._phase in _PENDING:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._condition.wait(left)
            return self._status()

    # -- the warm-up itself (model thread) ------------------------------------

    # Each outcome is logged before it is published, so by the time a
    # waiter sees "ready" or "failed" the line saying so is already out.

    def _run(self) -> None:
        try:
            if not self._admitted():
                return
            for index, step in enumerate(self._steps):
                step_started = self._clock()
                with self._condition:
                    self._step_index = index
                    self._step_started_at = step_started
                self._publish()
                try:
                    detail = step.run()
                except Exception as exc:
                    self._fail(step, describe_failure(exc), seconds=self._clock() - step_started)
                    return
                described = f" ({detail})" if isinstance(detail, str) and detail.strip() else ""
                self._log(
                    f"{step.name} ready in {self._clock() - step_started:.1f} s{described}"
                )
            finished = self._clock()
            total = finished - (self._started_at if self._started_at is not None else finished)
            self._log(f"warm-up done: {_steps(len(self._steps))} ready in {total:.1f} s")
            with self._condition:
                self._phase = WarmUpPhase.READY
                self._finished_at = finished
                self._step_index = len(self._steps)
                self._ever_ready = True
                self._publish()
                self._condition.notify_all()
        finally:
            with self._condition:
                unexpected = self._phase is WarmUpPhase.WARMING
            if unexpected:  # something other than a step's Exception ended the pass
                self._fail(None, "the warm-up stopped unexpectedly")

    def _admitted(self) -> bool:
        """Ask the admission check, on the model thread; `True` to go on.
        A refusal, or a check that raises (treated as a refusal: fail
        closed), leaves the warm-up `memory_busy` with nothing loaded."""
        if self._admission is None:
            return True
        try:
            decision = self._admission()
        except Exception as exc:
            decision = refused(
                "the models", describe_failure(exc), required_bytes=0, reserve_bytes=0
            )
        with self._condition:
            after_refusal = self._refusal is not None
        if decision.admitted:
            prefix = "host memory is available again" if after_refusal else "memory admitted"
            self._log(f"{prefix}: {decision.reason}")
            with self._condition:
                self._refusal = None
                self._refused_at = None
                self._refusal_logged_at = None
            return True
        now = self._clock()
        with self._condition:
            due = (
                self._refusal_logged_at is None
                or now - self._refusal_logged_at >= REFUSAL_LOG_INTERVAL_SECONDS
            )
            if due:
                self._refusal_logged_at = now
        if due:
            self._log(
                f"not loading the models — {decision.reason}. Retrieval calls answer "
                f"WARMING_UP (host memory busy) until the host has room; the load is "
                f"tried again at most every {self._retry_seconds:g} s"
            )
        with self._condition:
            self._phase = WarmUpPhase.MEMORY_BUSY
            self._refusal = decision
            self._refused_at = now
            self._started_at = None
            self._finished_at = None
            self._step_index = 0
            self._step_started_at = None
            self._publish()
            self._condition.notify_all()
        return False

    def _fail(self, step: WarmUpStep | None, cause: str, *, seconds: float | None = None) -> None:
        """Called at most once per pass, from the model thread (or from
        `start` when the pass could not be queued at all)."""
        failure = f"{step.name}: {cause}" if step is not None else cause
        after = f" after {seconds:.1f} s" if seconds is not None else ""
        self._log(
            f"warm-up FAILED{after} — {failure}. No tool call will be answered "
            f"until the cause is fixed and the server restarts"
        )
        with self._condition:
            if self._phase is not WarmUpPhase.WARMING:
                return
            self._phase = WarmUpPhase.FAILED
            self._failure = failure
            self._finished_at = self._clock()
            self._publish()
            self._condition.notify_all()

    def _log(self, line: str) -> None:
        with contextlib.suppress(Exception):  # a log line must never change the outcome
            self._log_line(line)

    def _publish(self) -> None:
        """Hand the current status to `on_status`, if one was given.

        Called under the condition at the two terminal transitions, and
        so before `notify_all` — the same ordering the log lines above
        have, and for the same reason: by the time anything can observe
        `ready` or `failed`, the readiness file already says so. A status
        command that raced the transition would otherwise report a warm
        server as still warming, which is exactly the question it exists
        to answer. The lock is therefore held across the publisher, so
        `on_status` must be quick (a local file write) and must never
        call back into this object.

        Swallows everything the publisher raises, for the same reason
        `_log` does — readiness reporting must never decide whether the
        models load."""
        if self._on_status is None:
            return
        with self._condition:
            status = self._status()
            with contextlib.suppress(Exception):
                self._on_status(status)

    # -- status (caller holds the condition) ----------------------------------

    def _status(self) -> WarmUpStatus:
        total = len(self._steps)
        now = self._clock()
        elapsed = 0.0
        if self._started_at is not None:
            end = self._finished_at if self._finished_at is not None else now
            elapsed = max(end - self._started_at, 0.0)
        if self._phase is WarmUpPhase.READY:
            return WarmUpStatus(self._phase, None, total, total, elapsed, 0.0, None)
        if self._phase is WarmUpPhase.FAILED:
            return WarmUpStatus(
                self._phase, None, self._step_index, total, elapsed, 0.0, self._failure
            )
        if self._phase is WarmUpPhase.MEMORY_BUSY:
            return WarmUpStatus(
                self._phase,
                None,
                0,
                total,
                0.0,
                self._retry_wait_left(),
                None,
                admission=self._refusal,
                retry_seconds=self._retry_seconds,
            )
        if self._phase in _STARTABLE:
            remaining = sum(step.estimated_seconds for step in self._steps)
            return WarmUpStatus(self._phase, None, 0, total, 0.0, remaining, None)
        if self._step_index >= total:  # between the last step and READY
            return WarmUpStatus(self._phase, None, total, total, elapsed, 0.0, None)
        running = self._steps[self._step_index]
        in_step = now - self._step_started_at if self._step_started_at is not None else 0.0
        remaining = max(running.estimated_seconds - in_step, RUNNING_STEP_FLOOR_SECONDS) + sum(
            step.estimated_seconds for step in self._steps[self._step_index + 1 :]
        )
        return WarmUpStatus(
            self._phase, running.name, self._step_index, total, elapsed, remaining, None
        )


__all__ = [
    "DEFAULT_ADMISSION_RETRY_SECONDS",
    "MAX_FAILURE_CHARS",
    "REFUSAL_LOG_INTERVAL_SECONDS",
    "RUNNING_STEP_FLOOR_SECONDS",
    "BackgroundWarmUp",
    "WarmUpPhase",
    "WarmUpStatus",
    "WarmUpStep",
    "describe_failure",
    "log_to_stderr",
]
