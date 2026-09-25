"""The LaunchAgent supervisor (`imsg.agents.supervise`), run the way
launchd runs it: as a script, in its own process, against a temporary
data root, with a stand-in `imsg` for the mount gate.

What these tests pin down is what supervision has to guarantee (QA
review 2026-09-24: the public server, Postgres and the backups had no
supervisor): nothing starts before the volume and the gate; secrets come
only from owner-only files and never reach a log; the service's exit
status reaches launchd; a stop request reaches the service as the signal
it expects; and a crash loop slows down."""

from __future__ import annotations

import ast
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import imsg.agents.supervise as supervise_module
from imsg.agents.supervise import EX_CONFIG, EX_TEMPFAIL, SENTINEL_NAME, record_start_and_count

SUPERVISOR = Path(supervise_module.__file__).resolve()
SECRET = "owner-subject-value-4242"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data-root"
    (data_root / "private" / "env").mkdir(parents=True)
    (data_root / SENTINEL_NAME).write_text("")
    return data_root


def _stub_imsg(tmp_path: Path, *, guard_status: int = 0) -> Path:
    """An `imsg` whose `guard-mount` exits with `guard_status`."""
    path = tmp_path / "bin" / "imsg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!/bin/sh\n[ \"$1\" = guard-mount ] && {{ echo 'guard said no' >&2; exit {guard_status}; }}\nexit 0\n"
    )
    path.chmod(0o755)
    return path


def _secret_file(root: Path, name: str = "IMSG_OWNER_SUB", mode: int = 0o600) -> str:
    path = root / "private" / "env" / name
    path.write_text(SECRET + "\n")
    path.chmod(mode)
    return f"{name}=private/env/{name}"


def _argv(root: Path, imsg: Path, tmp_path: Path, *extra: str, command: list[str]) -> list[str]:
    return [
        sys.executable,
        str(SUPERVISOR),
        "--service",
        "svc",
        "--data-root",
        str(root),
        "--imsg",
        str(imsg),
        "--config",
        str(root / "private" / "config.yaml"),
        "--guard-log",
        str(tmp_path / "guard.log"),
        *extra,
        "--",
        *command,
    ]


def _run(argv: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _child(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def test_nothing_starts_until_the_volume_is_mounted(root: Path, tmp_path: Path) -> None:
    (root / SENTINEL_NAME).unlink()
    marker = tmp_path / "ran"
    done = _run(
        _argv(
            root, _stub_imsg(tmp_path), tmp_path, "--mount-wait", "0.3",
            command=_child(f"open({str(marker)!r}, 'w')"),
        )
    )
    assert done.returncode == EX_CONFIG
    assert not marker.exists()
    lines = (tmp_path / "guard.log").read_text().splitlines()
    assert len(lines) == 1 and "svc" in lines[0] and "not mounted" in lines[0]
    assert not (root / "logs").exists(), "nothing is written to the data root before the gate"


def test_a_refused_mount_gate_stops_the_start(root: Path, tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    done = _run(
        _argv(
            root, _stub_imsg(tmp_path, guard_status=78), tmp_path,
            command=_child(f"open({str(marker)!r}, 'w')"),
        )
    )
    assert done.returncode == EX_CONFIG
    assert not marker.exists()
    assert "guard-mount refused (guard said no)" in (tmp_path / "guard.log").read_text()


def test_the_service_gets_its_secret_from_a_0600_file_and_its_status_reaches_launchd(
    root: Path, tmp_path: Path
) -> None:
    seen = tmp_path / "seen"
    script = (
        "import os, sys\n"
        f"open({str(seen)!r}, 'w').write(os.environ['IMSG_OWNER_SUB'])\n"
        "print('service stdout'); print('service stderr', file=sys.stderr)\n"
        "sys.exit(3)\n"
    )
    done = _run(
        _argv(root, _stub_imsg(tmp_path), tmp_path, "--env", _secret_file(root), command=_child(script))
    )
    assert done.returncode == 3
    assert seen.read_text() == SECRET, "trailing newline removed, value intact"
    out_log = root / "logs" / "imsgindex-svc.out.log"
    err_log = root / "logs" / "imsgindex-svc.err.log"
    assert out_log.read_text() == "service stdout\n"
    err = err_log.read_text()
    assert "service stderr" in err
    assert "starting (env: IMSG_OWNER_SUB" in err and "exited with status 3" in err
    assert SECRET not in err + out_log.read_text() + done.stdout + done.stderr
    assert oct(err_log.stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize("problem", ["group-readable", "symlink", "missing", "empty", "escape"])
def test_secret_files_that_are_not_owner_only_regular_files_are_refused(
    root: Path, tmp_path: Path, problem: str
) -> None:
    marker = tmp_path / "ran"
    spec = "IMSG_OWNER_SUB=private/env/IMSG_OWNER_SUB"
    target = root / "private" / "env" / "IMSG_OWNER_SUB"
    if problem == "group-readable":
        _secret_file(root, mode=0o640)
    elif problem == "symlink":
        real = tmp_path / "elsewhere"
        real.write_text(SECRET)
        real.chmod(0o600)
        target.symlink_to(real)
    elif problem == "empty":
        target.write_text("\n")
        target.chmod(0o600)
    elif problem == "escape":
        spec = "IMSG_OWNER_SUB=../outside"
    done = _run(
        _argv(
            root, _stub_imsg(tmp_path), tmp_path, "--env", spec,
            command=_child(f"open({str(marker)!r}, 'w')"),
        )
    )
    assert done.returncode == EX_CONFIG
    assert not marker.exists()
    logged = (root / "logs" / "imsgindex-svc.err.log").read_text() if problem != "escape" else done.stderr
    assert "IMSG_OWNER_SUB" in logged
    assert SECRET not in logged


def _start_supervised(root: Path, tmp_path: Path, *extra: str, script: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        _argv(root, _stub_imsg(tmp_path), tmp_path, *extra, command=_child(script)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = tmp_path / "ready"
    deadline = time.monotonic() + 20
    while not ready.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "the service never started"
        time.sleep(0.05)
    return process


def _signal_recorder(tmp_path: Path) -> str:
    return (
        "import signal, sys, time\n"
        f"out = {str(tmp_path / 'got')!r}\n"
        "def handler(signum, frame):\n"
        "    open(out, 'w').write(signal.Signals(signum).name)\n"
        "    sys.exit(0)\n"
        "for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):\n"
        "    signal.signal(s, handler)\n"
        f"open({str(tmp_path / 'ready')!r}, 'w')\n"
        "time.sleep(30)\n"
    )


def test_launchds_stop_request_reaches_postgres_as_a_fast_shutdown(root: Path, tmp_path: Path) -> None:
    """launchd stops a job with TERM; Postgres treats TERM as a *smart*
    shutdown that waits for every client to disconnect, so the pg agent
    asks the supervisor to deliver INT (fast shutdown) instead."""
    process = _start_supervised(root, tmp_path, "--stop-signal", "INT", script=_signal_recorder(tmp_path))
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=20) == 0
    assert (tmp_path / "got").read_text() == "SIGINT"


def test_other_signals_are_forwarded_unchanged(root: Path, tmp_path: Path) -> None:
    """`launchctl kill HUP` must reach Postgres as HUP: a configuration reload."""
    process = _start_supervised(root, tmp_path, script=_signal_recorder(tmp_path))
    process.send_signal(signal.SIGHUP)
    assert process.wait(timeout=20) == 0
    assert (tmp_path / "got").read_text() == "SIGHUP"


def test_a_crash_loop_waits_before_the_next_start(root: Path, tmp_path: Path) -> None:
    starts = root / "run" / "supervise-svc.starts"
    starts.parent.mkdir(parents=True)
    now = time.time()
    starts.write_text("".join(f"{now - i:.3f}\n" for i in range(3)))
    began = time.monotonic()
    done = _run(
        _argv(
            root, _stub_imsg(tmp_path), tmp_path,
            "--crash-limit", "2", "--crash-window", "600", "--crash-backoff", "0.7",
            command=_child("pass"),
        )
    )
    assert done.returncode == 0
    assert time.monotonic() - began >= 0.7
    assert "4 starts in the last 600 s" in (root / "logs" / "imsgindex-svc.err.log").read_text()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_service_that_needs_postgres_waits_for_it(root: Path, tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    port = _free_port()
    absent = _run(
        _argv(
            root, _stub_imsg(tmp_path), tmp_path,
            "--wait-for-postgres", f"127.0.0.1:{port}", "--postgres-wait", "0.5",
            command=_child(f"open({str(marker)!r}, 'w')"),
        )
    )
    assert absent.returncode == EX_TEMPFAIL
    assert not marker.exists()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()
        present = _run(
            _argv(
                root, _stub_imsg(tmp_path), tmp_path,
                "--wait-for-postgres", f"127.0.0.1:{port}", "--postgres-wait", "5",
                command=_child(f"open({str(marker)!r}, 'w')"),
            )
        )
    assert present.returncode == 0
    assert marker.exists()


def test_start_history_keeps_only_the_window(tmp_path: Path) -> None:
    path = tmp_path / "run" / "x.starts"
    path.parent.mkdir()
    path.write_text("1.0\nnot-a-number\n9990.0\n9995.0\n")
    assert record_start_and_count(path, now=10_000.0, window=100.0) == 3
    assert path.read_text() == "9990.000\n9995.000\n10000.000\n"


def test_the_supervisor_needs_only_the_standard_library() -> None:
    """launchd starts it with the interpreter that holds the Full Disk
    Access grant, outside any virtual environment."""
    tree = ast.parse(SUPERVISOR.read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported - {"__future__"} <= set(sys.stdlib_module_names), imported
    isolated = subprocess.run(
        [sys.executable, "-I", str(SUPERVISOR), "--help"],
        capture_output=True, text=True, check=False, env={"PATH": os.defpath},
    )
    assert isolated.returncode == 0, isolated.stderr
