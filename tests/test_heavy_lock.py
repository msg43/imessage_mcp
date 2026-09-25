"""The host-wide heavy-model lock (`imsg.heavy_lock`), with real processes.

No database. Each contention test runs the other holder in a real child
process, because the property under test is between processes: `flock`
never conflicts with itself inside one process's registry."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from imsg.errors import HeavyModelLockBusyError, HeavyModelLockError
from imsg.heavy_lock import HeavyModelLock, heavy_lock_path, inspect_heavy_lock

_HOLDER = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from imsg.heavy_lock import HeavyModelLock

    lock = HeavyModelLock(Path(sys.argv[1]), command=sys.argv[2])
    lock.acquire()
    print("acquired", flush=True)
    sys.stdin.readline()  # hold until the test says release (or kills us)
    lock.release()
    print("released", flush=True)
    """
)


def _start_holder(data_root: Path, command: str = "imsg embed") -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(data_root), command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line == "acquired", f"holder did not start: {line!r}"
    return proc


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data_root"
    root.mkdir()
    return root


def test_lock_file_lives_under_data_root_run(data_root: Path) -> None:
    assert heavy_lock_path(data_root) == (data_root / "run" / "heavy-models.lock").resolve()


def test_a_run_symlink_pointing_outside_data_root_is_refused(
    data_root: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (data_root / "run").symlink_to(elsewhere)
    with pytest.raises(HeavyModelLockError, match="outside data_root"):
        HeavyModelLock(data_root, command="imsg embed")


def test_second_process_waits_until_the_first_releases(data_root: Path) -> None:
    holder = _start_holder(data_root, "imsg embed")
    waiter = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(data_root), "imsg sync"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert waiter.stdout is not None and holder.stdin is not None
        waiter_stdout = waiter.stdout
        lines: list[str] = []
        acquired = threading.Event()

        def _read() -> None:
            for line in waiter_stdout:
                lines.append(line.strip())
                if line.strip() == "acquired":
                    acquired.set()
                    return

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        # The waiter is blocked in flock: no "acquired" line while held,
        # only its structlog line saying who it is waiting for.
        assert not acquired.wait(timeout=1.5), f"waiter acquired while held: {lines}"
        assert len(lines) == 1
        assert "heavy_lock.waiting" in lines[0]
        assert f"pid {holder.pid} (imsg embed)" in lines[0]
        before = inspect_heavy_lock(data_root).holder
        assert before is not None and before.command == "imsg embed"

        holder.stdin.write("\n")
        holder.stdin.flush()
        assert acquired.wait(timeout=10), f"waiter never acquired: {lines}"
        status = inspect_heavy_lock(data_root)
        assert status.held
        assert status.holder is not None
        assert status.holder.pid == waiter.pid
        assert status.holder.command == "imsg sync"
    finally:
        _stop(holder)
        _stop(waiter)


def test_lock_is_released_when_the_holder_is_sigkilled(data_root: Path) -> None:
    holder = _start_holder(data_root)
    try:
        assert inspect_heavy_lock(data_root).held
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=10)
    finally:
        _stop(holder)

    # The killed holder's details are still in the file, but the kernel
    # dropped its lock: nobody holds it, and the stale content is not
    # reported as a holder.
    assert heavy_lock_path(data_root).read_text(encoding="utf-8").strip() != ""
    status = inspect_heavy_lock(data_root)
    assert not status.held
    assert status.holder is None
    lock = HeavyModelLock(data_root, command="imsg enrich", wait=False)
    lock.acquire()  # would raise HeavyModelLockBusyError if still held
    lock.release()


def test_holder_info_is_readable_while_held(data_root: Path) -> None:
    holder = _start_holder(data_root, "imsg enrich")
    try:
        status = inspect_heavy_lock(data_root)
        assert status.held
        assert status.holder is not None
        assert status.holder.pid == holder.pid
        assert status.holder.command == "imsg enrich"
        assert status.holder.acquired_at.endswith("+00:00")
    finally:
        _stop(holder)


def test_no_wait_raises_a_busy_error_naming_the_holder(data_root: Path) -> None:
    holder = _start_holder(data_root, "imsg embed")
    try:
        lock = HeavyModelLock(data_root, command="imsg sync", wait=False)
        with pytest.raises(HeavyModelLockBusyError) as excinfo:
            lock.acquire()
        assert f"pid {holder.pid} (imsg embed)" in str(excinfo.value)
        assert "--no-wait" in str(excinfo.value)
        assert not lock.held
    finally:
        _stop(holder)


def test_waiting_logs_who_holds_it_then_logs_the_acquisition(data_root: Path) -> None:
    holder = _start_holder(data_root, "imsg embed")
    assert holder.stdin is not None
    stdin = holder.stdin

    def _release_soon() -> None:
        time.sleep(0.5)
        stdin.write("\n")
        stdin.flush()

    releaser = threading.Thread(target=_release_soon)
    try:
        lock = HeavyModelLock(data_root, command="imsg sync")
        with capture_logs() as logs:
            releaser.start()
            lock.acquire()
        releaser.join(timeout=10)
        events = [entry["event"] for entry in logs]
        assert events == ["heavy_lock.waiting", "heavy_lock.acquired"]
        assert f"pid {holder.pid} (imsg embed)" in logs[0]["held_by"]
        assert logs[0]["command"] == "imsg sync"
        assert logs[1]["waited_seconds"] >= 0.3
        lock.release()
    finally:
        _stop(holder)


def test_uncontended_acquire_logs_nothing(data_root: Path) -> None:
    with capture_logs() as logs, HeavyModelLock(data_root, command="imsg embed"):
        pass
    assert logs == []


def test_reentrant_within_one_process_and_released_by_the_last_holder(data_root: Path) -> None:
    """`sync` takes the lock in its segment step and again in its embed
    step. Two `flock`s on two open files in one process would conflict,
    so without the registry the second take would wait on itself."""
    outer = HeavyModelLock(data_root, command="imsg sync")
    inner = HeavyModelLock(data_root, command="imsg sync")
    outer.acquire()
    outer.acquire()  # idempotent on the same object
    inner.acquire()  # no self-deadlock
    inner.release()
    assert inspect_heavy_lock(data_root).held  # outer still holds it
    outer.release()
    status = inspect_heavy_lock(data_root)
    assert not status.held
    assert heavy_lock_path(data_root).read_text(encoding="utf-8") == ""


def test_release_without_acquire_is_a_no_op(data_root: Path) -> None:
    HeavyModelLock(data_root, command="imsg embed").release()
    assert not inspect_heavy_lock(data_root).held


_HOLDER_WITH_CHILD = textwrap.dedent(
    """
    import subprocess, sys
    from pathlib import Path
    from imsg.heavy_lock import HeavyModelLock

    lock = HeavyModelLock(Path(sys.argv[1]), command="imsg enrich")
    lock.acquire()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], close_fds=False
    )
    print(child.pid, flush=True)
    sys.stdin.readline()
    """
)


def test_a_child_outliving_a_killed_holder_does_not_keep_the_lock(data_root: Path) -> None:
    """A child (`imsg-dump`, `textutil`) that outlives its SIGKILLed parent
    must not keep the lock alive, which it would if it had inherited the
    lock's open file."""
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_WITH_CHILD, str(data_root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    child_pid = int(holder.stdout.readline().strip())
    try:
        assert inspect_heavy_lock(data_root).held
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=10)
        os.kill(child_pid, 0)  # the child is still alive
        assert not inspect_heavy_lock(data_root).held
    finally:
        _stop(holder)
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# waiting commands go first (QA review 2026-09-24): a holder created with
# yields_to_waiters=True sees a waiting command, and never takes the lock
# ahead of one
# --------------------------------------------------------------------------


def _start_waiter(data_root: Path, command: str = "imsg sync") -> tuple[subprocess.Popen[str], threading.Event]:
    """A child that waits for the lock the way `imsg sync` does, holds it
    until told to release, and reports each step on stdout. The event is
    set once it prints "acquired"."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(data_root), command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    acquired = threading.Event()
    stdout = proc.stdout
    assert stdout is not None

    def _read() -> None:
        for line in stdout:
            if line.strip() == "acquired":
                acquired.set()

    threading.Thread(target=_read, daemon=True).start()
    return proc, acquired


def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_the_holder_sees_a_command_waiting_for_the_lock(data_root: Path) -> None:
    lock = HeavyModelLock(data_root, command="imsg enrich", yields_to_waiters=True)
    lock.acquire()
    waiter, acquired = _start_waiter(data_root)
    try:
        assert _until(lock.waiter_present), "the waiting command never announced itself"
        assert not acquired.is_set()
        lock.release()
        assert acquired.wait(timeout=10), "the waiting command never got the lock"
        # Once it holds the lock it is no longer waiting.
        assert _until(lambda: not lock.waiter_present())
    finally:
        lock.release()
        _stop(waiter)


def test_nobody_waiting_is_the_ordinary_answer(data_root: Path) -> None:
    lock = HeavyModelLock(data_root, command="imsg enrich", yields_to_waiters=True)
    with lock:
        assert not lock.waiter_present()


def test_a_yielding_command_never_takes_the_lock_ahead_of_a_waiting_one(data_root: Path) -> None:
    """Three processes: an embed holds the lock, a sync is waiting for it,
    and the enrichment worker (this process) asks for it too. When the
    embed finishes the sync must get the lock, and the worker only once
    the sync is done. Plain `flock` would let either of the two waiters
    win."""
    holder = _start_holder(data_root, "imsg embed")
    waiter, waiter_acquired = _start_waiter(data_root)
    worker = HeavyModelLock(
        data_root, command="imsg enrich", yields_to_waiters=True, yield_poll_seconds=0.02
    )
    worker_acquired = threading.Event()

    def _take() -> None:
        worker.acquire()
        worker_acquired.set()

    try:
        assert _until(worker.waiter_present), "the sync never announced itself"
        taker = threading.Thread(target=_take, daemon=True)
        taker.start()
        time.sleep(0.2)
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.flush()
        assert waiter_acquired.wait(timeout=10), "the waiting sync did not get the lock"
        assert not worker_acquired.wait(timeout=1.0), "the worker took the lock ahead of the sync"
        status = inspect_heavy_lock(data_root)
        assert status.holder is not None and status.holder.command == "imsg sync"
        assert waiter.stdin is not None
        waiter.stdin.write("\n")
        waiter.stdin.flush()
        assert worker_acquired.wait(timeout=10), "the worker never got the lock back"
        taker.join(timeout=5)
    finally:
        worker.release()
        _stop(holder)
        _stop(waiter)


def test_a_yielding_command_told_not_to_wait_is_busy_while_another_command_waits(
    data_root: Path,
) -> None:
    holder = _start_holder(data_root, "imsg embed")
    waiter, _ = _start_waiter(data_root)
    worker = HeavyModelLock(
        data_root, command="imsg enrich", wait=False, yields_to_waiters=True
    )
    try:
        assert _until(worker.waiter_present)
        with pytest.raises(HeavyModelLockBusyError, match="--no-wait"):
            worker.acquire()
        assert not worker.held
    finally:
        _stop(holder)
        _stop(waiter)
