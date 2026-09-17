"""`BackgroundWarmUp`: warms the models on the model thread while the
caller carries on, logs start / each model / total, fails loudly and
once, and reports a status — with an estimate of the seconds remaining —
that tool calls are gated on."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import pytest

from imsg.errors import ProviderUnavailableError
from imsg.retrieval.background_warm_up import (
    MAX_FAILURE_CHARS,
    RUNNING_STEP_FLOOR_SECONDS,
    BackgroundWarmUp,
    WarmUpPhase,
    WarmUpStep,
    describe_failure,
)
from imsg.retrieval.errors import WarmingUpError, WarmUpFailedError
from imsg.retrieval.model_thread import ModelThread


@pytest.fixture
def model_thread() -> Iterator[ModelThread]:
    thread = ModelThread(name="test-warm-up")
    yield thread
    thread.close()


class _Steps:
    """Warm-up steps that record where and in what order they ran, and can
    be held until the test releases them."""

    def __init__(self, *names: str, held: bool = False) -> None:
        self.names = names
        self.ran: list[str] = []
        self.threads: list[threading.Thread] = []
        self.release = {name: threading.Event() for name in names}
        if not held:
            for event in self.release.values():
                event.set()
        self.failures: dict[str, BaseException] = {}

    def step(self, name: str, estimated_seconds: float = 10.0) -> WarmUpStep:
        def run() -> None:
            self.threads.append(threading.current_thread())
            assert self.release[name].wait(timeout=10), f"{name} was never released"
            if name in self.failures:
                raise self.failures[name]
            self.ran.append(name)

        return WarmUpStep(name, estimated_seconds, run)

    def all(self) -> list[WarmUpStep]:
        return [self.step(name) for name in self.names]


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def test_warms_every_model_in_order_on_the_model_thread_and_logs_each(
    model_thread: ModelThread,
) -> None:
    steps = _Steps("text embedder", "multimodal text tower", "reranker")
    lines: list[str] = []
    warm_up = BackgroundWarmUp(steps.all(), model_thread=model_thread, log=lines.append)

    warm_up.start()
    status = warm_up.wait(timeout=5)

    assert status.phase is WarmUpPhase.READY
    assert (status.steps_done, status.steps_total, status.seconds_remaining) == (3, 3, 0.0)
    assert steps.ran == ["text embedder", "multimodal text tower", "reranker"]
    assert set(steps.threads) == {model_thread.thread}
    assert lines[0] == (
        "warm-up started: 3 steps in the background "
        "(text embedder, multimodal text tower, reranker)"
    )
    assert [line.split(" ready in ")[0] for line in lines[1:4]] == steps.ran
    assert all(line.endswith(" s") for line in lines[1:])
    assert lines[4].startswith("warm-up done: 3 steps ready in ")
    assert len(lines) == 5


def test_a_step_that_returns_its_own_numbers_has_them_logged(
    model_thread: ModelThread,
) -> None:
    """Only the step knows what it did — how many bytes a prewarm moved,
    say — so a string it returns is appended to its line."""
    lines: list[str] = []
    steps = [
        WarmUpStep("text embedder", 1.0, lambda: None),
        WarmUpStep("database buffer pool", 1.0, lambda: "2,296 MiB in 17 relation(s), 16.2 s"),
        WarmUpStep("quiet step", 1.0, lambda: "   "),
    ]
    warm_up = BackgroundWarmUp(steps, model_thread=model_thread, log=lines.append)
    warm_up.start()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    assert lines[1].startswith("text embedder ready in ") and lines[1].endswith(" s")
    assert lines[2].endswith(" s (2,296 MiB in 17 relation(s), 16.2 s)")
    assert lines[3].endswith(" s")  # whitespace is not a detail


def test_start_returns_at_once_and_the_status_names_the_model_loading(
    model_thread: ModelThread,
) -> None:
    steps = _Steps("text embedder", "reranker", held=True)
    warm_up = BackgroundWarmUp(steps.all(), model_thread=model_thread, log=lambda line: None)
    assert warm_up.status().phase is WarmUpPhase.NOT_STARTED

    began = time.monotonic()
    warm_up.start()
    assert time.monotonic() - began < 1.0

    _wait_until(lambda: len(steps.threads) == 1)
    status = warm_up.status()
    assert (status.phase, status.loading, status.steps_done) == (
        WarmUpPhase.WARMING,
        "text embedder",
        0,
    )
    steps.release["text embedder"].set()
    _wait_until(lambda: warm_up.status().loading == "reranker")
    assert warm_up.status().steps_done == 1
    steps.release["reranker"].set()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY


def test_start_is_idempotent(model_thread: ModelThread) -> None:
    steps = _Steps("text embedder")
    lines: list[str] = []
    warm_up = BackgroundWarmUp(steps.all(), model_thread=model_thread, log=lines.append)
    warm_up.start()
    warm_up.start()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY
    warm_up.start()
    model_thread.run(lambda: None)  # anything queued by a second start would have run by now
    assert steps.ran == ["text embedder"]
    assert sum(line.startswith("warm-up started") for line in lines) == 1


def test_wait_gives_up_at_its_timeout_while_a_model_is_still_loading(
    model_thread: ModelThread,
) -> None:
    steps = _Steps("reranker", held=True)
    warm_up = BackgroundWarmUp(steps.all(), model_thread=model_thread, log=lambda line: None)
    warm_up.start()
    began = time.monotonic()
    status = warm_up.wait(timeout=0.2)
    waited = time.monotonic() - began
    assert status.phase is WarmUpPhase.WARMING
    assert 0.2 <= waited < 3.0
    steps.release["reranker"].set()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY


def test_the_estimate_is_what_is_left_of_the_running_step_plus_every_later_one(
    model_thread: ModelThread,
) -> None:
    now = [100.0]
    steps = _Steps("text embedder", "multimodal text tower", "reranker", held=True)
    warm_up = BackgroundWarmUp(
        [
            steps.step("text embedder", 10.0),
            steps.step("multimodal text tower", 30.0),
            steps.step("reranker", 5.0),
        ],
        model_thread=model_thread,
        log=lambda line: None,
        clock=lambda: now[0],
    )
    assert warm_up.status().seconds_remaining == 45.0  # not started: everything

    warm_up.start()
    _wait_until(lambda: len(steps.threads) == 1)
    now[0] = 104.0
    assert warm_up.status().seconds_remaining == 6.0 + 30.0 + 5.0
    now[0] = 109.0  # 1 s of its estimated 10 left: the floor applies
    assert warm_up.status().seconds_remaining == RUNNING_STEP_FLOOR_SECONDS + 35.0

    now[0] = 110.0
    steps.release["text embedder"].set()
    _wait_until(lambda: len(steps.threads) == 2)
    now[0] = 120.0
    status = warm_up.status()
    assert (status.loading, status.seconds_remaining) == ("multimodal text tower", 20.0 + 5.0)
    assert status.elapsed_seconds == 20.0
    now[0] = 150.0  # past its estimated 30 s: counted as the floor
    assert warm_up.status().seconds_remaining == RUNNING_STEP_FLOOR_SECONDS + 5.0

    steps.release["multimodal text tower"].set()
    _wait_until(lambda: len(steps.threads) == 3)
    now[0] = 152.0
    steps.release["reranker"].set()
    status = warm_up.wait(timeout=5)
    assert (status.phase, status.seconds_remaining, status.elapsed_seconds) == (
        WarmUpPhase.READY,
        0.0,
        52.0,
    )
    now[0] = 500.0  # elapsed stops counting once settled
    assert warm_up.status().elapsed_seconds == 52.0


def test_a_step_that_fails_is_logged_once_and_the_models_after_it_are_not_loaded(
    model_thread: ModelThread,
) -> None:
    steps = _Steps("text embedder", "multimodal text tower", "reranker")
    steps.failures["multimodal text tower"] = ProviderUnavailableError(
        "PE-Core weights not found in the model cache"
    )
    lines: list[str] = []
    warm_up = BackgroundWarmUp(steps.all(), model_thread=model_thread, log=lines.append)

    warm_up.start()
    status = warm_up.wait(timeout=5)

    assert status.phase is WarmUpPhase.FAILED
    assert status.failure == "multimodal text tower: PE-Core weights not found in the model cache"
    assert (status.steps_done, status.seconds_remaining) == (1, 0.0)
    assert steps.ran == ["text embedder"]
    assert len(steps.threads) == 2  # the reranker step never started
    failed = [line for line in lines if "FAILED" in line]
    assert len(failed) == 1
    assert "multimodal text tower: PE-Core weights not found in the model cache" in failed[0]
    assert not any(line.startswith("warm-up done") for line in lines)
    assert warm_up.wait(timeout=0).failure == status.failure


def test_a_warm_up_that_cannot_even_be_queued_ends_failed() -> None:
    closed = ModelThread()
    closed.close()
    lines: list[str] = []
    warm_up = BackgroundWarmUp(_Steps("reranker").all(), model_thread=closed, log=lines.append)
    warm_up.start()  # must not raise into the server
    status = warm_up.status()
    assert status.phase is WarmUpPhase.FAILED
    assert status.failure == "RuntimeError: the model thread is closed"
    assert sum("FAILED" in line for line in lines) == 1


def test_a_warm_up_ended_by_something_other_than_an_exception_still_ends_failed(
    model_thread: ModelThread,
) -> None:
    steps = _Steps("reranker")
    steps.failures["reranker"] = SystemExit(3)
    warm_up = BackgroundWarmUp(steps.all(), model_thread=model_thread, log=lambda line: None)
    warm_up.start()
    status = warm_up.wait(timeout=5)
    assert status.phase is WarmUpPhase.FAILED
    assert status.failure == "the warm-up stopped unexpectedly"
    assert model_thread.run(lambda: "the thread survived") == "the thread survived"


def test_a_log_line_that_cannot_be_written_changes_nothing(model_thread: ModelThread) -> None:
    def broken_log(line: str) -> None:
        raise BrokenPipeError("stderr is gone")

    warm_up = BackgroundWarmUp(
        _Steps("text embedder").all(), model_thread=model_thread, log=broken_log
    )
    warm_up.start()
    assert warm_up.wait(timeout=5).phase is WarmUpPhase.READY


def test_describe_failure_keeps_one_line_and_the_type_name_when_it_helps() -> None:
    assert describe_failure(ProviderUnavailableError("install the models extra")) == (
        "install the models extra"
    )
    assert describe_failure(RuntimeError("There is no Stream(gpu, 1)\n in current thread.")) == (
        "RuntimeError: There is no Stream(gpu, 1) in current thread."
    )
    assert describe_failure(RuntimeError()) == "RuntimeError"
    assert describe_failure(ProviderUnavailableError()) == "ProviderUnavailableError"
    long = describe_failure(ValueError("x" * 2000))
    assert len(long) == MAX_FAILURE_CHARS and long.endswith("…")


def test_each_phase_maps_to_its_tool_error(model_thread: ModelThread) -> None:
    held = _Steps("reranker", held=True)
    warming = BackgroundWarmUp(
        [held.step("reranker", 12.4)], model_thread=model_thread, log=lambda line: None
    )
    with pytest.raises(WarmingUpError) as not_started:
        warming.status().raise_unless_ready(wait_bound_seconds=90.0)
    assert "have not started loading" in str(not_started.value)

    warming.start()
    _wait_until(lambda: len(held.threads) == 1)
    with pytest.raises(WarmingUpError) as still_warming:
        warming.status().raise_unless_ready(wait_bound_seconds=90.0)
    error = still_warming.value
    assert error.code == "WARMING_UP"
    assert error.seconds_remaining == 13  # rounded up, never "0 s remaining"
    assert str(error) == (
        "the index is warming up (loading the reranker): about 13 s remaining (estimate). "
        "Retry this call; each call waits up to 90 s for the models to finish loading"
    )
    held.release["reranker"].set()
    warming.wait(timeout=5).raise_unless_ready(wait_bound_seconds=90.0)  # ready: no error

    broken = _Steps("text embedder")
    broken.failures["text embedder"] = ProviderUnavailableError("weights not found")
    failing = BackgroundWarmUp(broken.all(), model_thread=model_thread, log=lambda line: None)
    failing.start()
    with pytest.raises(WarmUpFailedError) as failed:
        failing.wait(timeout=5).raise_unless_ready(wait_bound_seconds=90.0)
    assert failed.value.code == "WARM_UP_FAILED"
    assert "(text embedder: weights not found)" in str(failed.value)
    assert "restart the MCP server" in str(failed.value)
