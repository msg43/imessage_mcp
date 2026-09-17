"""`ModelThread`: every model call on one long-lived thread, one at a time."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from imsg.retrieval.model_thread import ModelThread


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-models")
    yield thread
    thread.close()


def test_every_call_runs_on_the_same_thread_whoever_submits_it(model_thread: ModelThread) -> None:
    seen: list[threading.Thread] = []
    lock = threading.Lock()

    def call() -> None:
        with lock:
            seen.append(threading.current_thread())

    callers = [threading.Thread(target=lambda: model_thread.run(call)) for _ in range(4)]
    for caller in callers:
        caller.start()
    model_thread.run(call)
    for caller in callers:
        caller.join(timeout=5)
    assert len(seen) == 5
    assert set(seen) == {model_thread.thread}
    assert model_thread.thread is not threading.current_thread()
    assert model_thread.thread.daemon


def test_run_returns_the_result_and_raises_what_the_call_raised(model_thread: ModelThread) -> None:
    assert model_thread.run(lambda: 41 + 1) == 42

    def broken() -> int:
        raise ValueError("weights missing")

    with pytest.raises(ValueError, match="weights missing"):
        model_thread.run(broken)
    assert model_thread.run(lambda: "still serving") == "still serving"


def test_a_call_from_the_model_thread_itself_runs_inline(model_thread: ModelThread) -> None:
    # Queueing it would wait on the very thread that is waiting.
    assert model_thread.run(lambda: model_thread.run(lambda: "nested")) == "nested"


def test_calls_run_one_at_a_time_in_submission_order(model_thread: ModelThread) -> None:
    release = threading.Event()
    events: list[str] = []

    def first() -> None:
        events.append("first started")
        release.wait(timeout=5)
        events.append("first finished")

    def second() -> None:
        events.append("second started")

    first_done = model_thread.submit(first)
    second_done = model_thread.submit(second)
    assert not second_done.done()
    release.set()
    first_done.result(timeout=5)
    second_done.result(timeout=5)
    assert events == ["first started", "first finished", "second started"]


def test_close_finishes_queued_calls_then_refuses_new_ones() -> None:
    thread = ModelThread()
    queued = thread.submit(lambda: "queued before close")
    thread.close()
    thread.close()  # idempotent
    assert queued.result(timeout=5) == "queued before close"
    with pytest.raises(RuntimeError, match="closed"):
        thread.submit(lambda: None)
    thread.thread.join(timeout=5)
    assert not thread.thread.is_alive()
