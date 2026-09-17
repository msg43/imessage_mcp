"""One dedicated thread for every model call a long-lived server makes.

`imsg mcp local` warms its models in the background while it already
answers the MCP handshake (`imsg.retrieval.background_warm_up`), so the
models are touched by two parties: the warm-up, and the tool calls that
follow it. Both run their model calls here, on the same thread, one at
a time.

Why one thread rather than a lock. A lock would stop the two from
running models at the same moment, but not from running them on
*different threads*, and MLX cares about that: MLX 0.32 gives each OS
thread its own default GPU stream, and an array whose computation was
set up on one thread cannot be evaluated from another. Probed on the M2
Ultra host (2026-09-17): an array or model evaluated on a worker thread
is usable from the main thread, but one still unevaluated fails there
with `RuntimeError: There is no Stream(gpu, 1) in current thread`.
Loading on the warm-up thread and querying on the event-loop thread
would therefore work only as long as the warm-up happened to evaluate
every array a query touches. Confining every model call to this thread
removes that dependency, and it serializes model access as well: MLX's
own documentation for cross-thread streams says "all nodes in a graph
must be evaluated in sequence", and PE-Core's torch model (MPS) is
never run concurrently either.

The thread is a daemon, so a client that disconnects in the middle of a
weight load does not keep the process alive until the load finishes.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

DEFAULT_THREAD_NAME = "imsg-models"


@dataclass(frozen=True, slots=True)
class _Job:
    call: Callable[[], Any]
    future: Future[Any]


class ModelThread:
    """Runs submitted calls on one long-lived daemon thread, in
    submission order."""

    def __init__(self, name: str = DEFAULT_THREAD_NAME) -> None:
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self._thread.start()

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    def submit[T](self, call: Callable[[], T]) -> Future[T]:
        """Queue `call`; its result or exception arrives on the future."""
        future: Future[T] = Future()
        with self._close_lock:
            if self._closed:
                raise RuntimeError("the model thread is closed")
            self._jobs.put(_Job(call, future))
        return future

    def run[T](self, call: Callable[[], T]) -> T:
        """Run `call` on the model thread and return its result, raising
        what it raised. Called from the model thread itself (a warm-up
        step, say) it runs inline — queueing would deadlock."""
        if threading.current_thread() is self._thread:
            return call()
        return self.submit(call).result()

    def close(self) -> None:
        """Stop accepting calls; the thread exits after the ones already
        queued. Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._jobs.put(None)

    def _serve(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            if not job.future.set_running_or_notify_cancel():
                continue
            try:
                result = job.call()
            except BaseException as exc:  # handed to the caller, never kills the thread
                job.future.set_exception(exc)
            else:
                job.future.set_result(result)


__all__ = ["DEFAULT_THREAD_NAME", "ModelThread"]
