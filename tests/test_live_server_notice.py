"""The file a live MCP server holds while it waits to load its models
(`imsg.live_server_notice`).

A notice counts only while its owner holds its lock, so these tests use
real processes wherever liveness is the point: a server that is killed
must stop holding background work back at once, and a file left behind,
even one named after a running process, must count for nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from imsg.live_server_notice import (
    LiveServerNotice,
    LiveServerNoticePathError,
    notices_directory,
    waiting_live_servers,
)

_CHILD_POSTS = """
import pathlib, sys, time
from imsg.live_server_notice import LiveServerNotice

data_root, ready = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
notice = LiveServerNotice(data_root, role="public_server", command="imsg mcp public")
notice.post()
ready.write_text("posted")
time.sleep(60)
"""


def _child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}


def _wait_for(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.01)


def test_a_posted_notice_names_the_server_and_withdrawing_it_removes_it(data_root: Path) -> None:
    notice = LiveServerNotice(data_root, role="public_server", command="imsg mcp public")
    assert waiting_live_servers(data_root) == []
    assert notice.posted_for() is None

    notice.post()
    notice.post()  # idempotent
    [waiting] = waiting_live_servers(data_root)
    assert (waiting.pid, waiting.role, waiting.command) == (
        os.getpid(),
        "public_server",
        "imsg mcp public",
    )
    assert waiting.since == notice.since
    assert waiting.describe().startswith(f"imsg mcp public (pid {os.getpid()}) since ")
    assert notice.posted and notice.posted_for() is not None

    notice.withdraw()
    notice.withdraw()  # idempotent
    assert waiting_live_servers(data_root) == []
    assert not notice.posted and notice.posted_for() is None
    assert list(notices_directory(data_root).iterdir()) == []


def test_posted_for_counts_from_the_post_on_the_given_clock(data_root: Path) -> None:
    now = [100.0]
    notice = LiveServerNotice(
        data_root, role="local_server", command="imsg mcp local", clock=lambda: now[0]
    )
    notice.post()
    now[0] += 42.0
    assert notice.posted_for() == 42.0
    notice.post()  # already up: the wait does not start again
    assert notice.posted_for() == 42.0
    notice.withdraw()


def test_a_killed_servers_notice_stops_counting_at_once(data_root: Path, tmp_path: Path) -> None:
    """The kernel drops the lock when the process dies, however it dies;
    the file it leaves behind counts for nothing."""
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_POSTS, str(data_root), str(ready)], env=_child_env()
    )
    try:
        _wait_for(ready)
        assert [w.pid for w in waiting_live_servers(data_root)] == [child.pid]
        child.kill()
        child.wait(timeout=30)
        assert (notices_directory(data_root) / f"{child.pid}.json").exists()  # left behind
        assert waiting_live_servers(data_root) == []
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)


def test_a_leftover_file_counts_for_nothing_even_under_a_running_pid(data_root: Path) -> None:
    """Checking the pid would believe this file: pid 1 is always running.
    Nobody holds its lock, so it is not a notice."""
    directory = notices_directory(data_root)
    directory.mkdir(parents=True)
    (directory / "1.json").write_text(
        json.dumps({"pid": 1, "role": "public_server", "command": "imsg mcp public", "since": "x"})
    )
    assert waiting_live_servers(data_root) == []


def test_posting_sweeps_the_files_of_servers_that_have_exited(data_root: Path) -> None:
    directory = notices_directory(data_root)
    directory.mkdir(parents=True)
    leftover = directory / "999999.json"
    leftover.write_text("{}")
    notice = LiveServerNotice(data_root, role="public_server", command="imsg mcp public")
    notice.post()
    try:
        assert not leftover.exists()
        assert [p.name for p in directory.glob("*.json")] == [f"{os.getpid()}.json"]
    finally:
        notice.withdraw()


def test_a_held_notice_whose_content_is_garbled_still_counts(data_root: Path) -> None:
    """The lock says whether a notice is up; the content only describes it."""
    notice = LiveServerNotice(data_root, role="public_server", command="imsg mcp public")
    notice.post()
    try:
        (notices_directory(data_root) / f"{os.getpid()}.json").write_bytes(b"not json")
        [waiting] = waiting_live_servers(data_root)
        assert (waiting.pid, waiting.command) == (os.getpid(), "a live MCP server")
    finally:
        notice.withdraw()


def test_the_notices_directory_must_stay_inside_data_root(data_root: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (data_root / "run").symlink_to(elsewhere, target_is_directory=True)
    notice = LiveServerNotice(data_root, role="public_server", command="imsg mcp public")
    with pytest.raises(LiveServerNoticePathError):
        notice.post()
    with pytest.raises(LiveServerNoticePathError):
        waiting_live_servers(data_root)
    assert not notice.posted
    assert list(elsewhere.iterdir()) == []
