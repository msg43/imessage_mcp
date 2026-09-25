"""The one program every `com.imsgindex.*` LaunchAgent runs (SPEC §5.4,
§5.5): wait for the encrypted volume, pass the mount gate, load the
service's environment from 0600 files on that volume, then run the
service as a child and stay with it until it exits.

Why a wrapper, and why this shape
---------------------------------

- **The plist stays content-free.** SPEC §5.5 forbids secrets and
  hostnames in a plist. Values the services need from the environment —
  the OAuth client id, the owner's subject, the database password — are
  read here from files under `<data_root>` (`--env NAME=RELPATH`), each
  of which must be a regular file owned by this user and readable by no
  one else. The plist names only the variable and the file.
- **Nothing starts before the mount.** The wrapper waits for the volume
  sentinel (`<data_root>/.imsgindex-volume`), then runs the real gate,
  `imsg guard-mount`. Either failing is exit 78 (`EX_CONFIG`) with one
  line in the off-mount guard log (`~/Library/Logs/imsgindex-guard.log`,
  SPEC §5.4's one permitted off-mount write), and launchd tries again
  after the agent's `ThrottleInterval`. Under FileVault with manual
  login (posture 2, SPEC §5.1) that is exactly the login sequence: the
  agents load at login, and each service starts once the volume has
  unlocked from the login keychain.
- **launchd keeps the service alive.** The service is a child, not an
  `exec`: the child inherits this process's macOS privacy attribution,
  which is how the existing agents give `imsg` Full Disk Access (grant
  it to the interpreter the plist names — this file runs under that
  interpreter, with the standard library only). launchd's `KeepAlive`
  restarts the whole unit whenever it exits, and when this wrapper
  dies launchd removes whatever is left in its process group, so a
  service is never orphaned. The wrapper forwards `TERM` as
  `--stop-signal` (Postgres takes `INT`, its fast shutdown, which
  disconnects clients instead of waiting for them) and every other
  common signal as-is, so `launchctl kill HUP` reaches Postgres as a
  configuration reload.
- **A crash loop slows down.** More than `--crash-limit` starts within
  `--crash-window` seconds and the next start waits `--crash-backoff`
  seconds first, so a service that dies at startup does not reload its
  models every minute.
- **Logs are opened for appending**, on the volume
  (`<data_root>/logs/imsgindex-<service>.{out,err}.log`), so the size
  rotation in `imsg.log_rotation` can truncate them in place while the
  service runs.

This module imports nothing outside the standard library, so the
interpreter the plist names can run it without the project's virtual
environment: `python3 <path>/supervise.py --service ... -- <command>`.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

EX_TEMPFAIL = 75
EX_CONFIG = 78
SENTINEL_NAME = ".imsgindex-volume"
GUARD_LOG = Path("~/Library/Logs/imsgindex-guard.log")
_SERVICE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
FORWARDED_SIGNALS = (
    signal.SIGTERM,
    signal.SIGINT,
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGUSR1,
    signal.SIGUSR2,
)
STOP_SIGNALS = {"TERM": signal.SIGTERM, "INT": signal.SIGINT, "QUIT": signal.SIGQUIT}


class SuperviseError(Exception):
    """A precondition failed. Carries the exit status launchd will see."""

    def __init__(self, message: str, status: int = EX_CONFIG) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class EnvFile:
    name: str
    relative_path: str


def parse_env_file(spec: str) -> EnvFile:
    """`NAME=RELPATH`: an environment variable and the file under the data
    root that holds its value."""
    name, sep, relative = spec.partition("=")
    if not sep or not _ENV_NAME_RE.match(name):
        raise SuperviseError(f"--env must be NAME=RELPATH with an upper-case NAME, got {spec!r}")
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or relative.startswith("~"):
        raise SuperviseError(
            f"--env {name}: the file must be a path relative to the data root with no "
            f"'..', got {relative!r}"
        )
    return EnvFile(name, relative)


def read_env_file(data_root: Path, entry: EnvFile) -> str:
    """The variable's value: the file's text without its trailing line
    ending. Refused unless the file is a regular file (not a symlink)
    under the data root, owned by this user, and readable by nobody else
    (mode 0600 or 0400). Error messages name the file, never the value."""
    path = data_root / entry.relative_path
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SuperviseError(f"{entry.name}: {path} does not exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SuperviseError(f"{entry.name}: {path} is not a regular file (symlinks are refused)")
    if not Path(os.path.realpath(path)).is_relative_to(os.path.realpath(data_root)):
        raise SuperviseError(f"{entry.name}: {path} resolves outside the data root")
    if info.st_uid != os.getuid():
        raise SuperviseError(f"{entry.name}: {path} is not owned by this user")
    if info.st_mode & 0o077:
        raise SuperviseError(
            f"{entry.name}: {path} has mode {stat.S_IMODE(info.st_mode):04o}; it must be "
            f"readable by its owner only (chmod 600)"
        )
    try:
        value = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SuperviseError(f"{entry.name}: {path} is not UTF-8 text") from exc
    value = value.rstrip("\r\n")
    if not value or "\n" in value or "\r" in value or "\x00" in value:
        raise SuperviseError(f"{entry.name}: {path} must hold exactly one non-empty line")
    return value


def guard_log_line(message: str, *, guard_log: Path = GUARD_LOG) -> None:
    """One line in the off-mount guard log: a time, the service, what
    failed. Never message content (SPEC §5.4). Best effort — if even this
    file cannot be written, launchd's exit status still says 78."""
    try:
        path = guard_log.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} supervise: {message}\n")
    except OSError:
        pass


def wait_for_sentinel(data_root: Path, *, timeout: float, poll: float = 5.0) -> bool:
    """Whether `<data_root>/.imsgindex-volume` appeared within `timeout`
    seconds — the sign the encrypted volume is mounted (SPEC §5.4 item
    2), checked before anything else touches the data root."""
    sentinel = data_root / SENTINEL_NAME
    deadline = time.monotonic() + timeout
    while True:
        if sentinel.is_file():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))


def run_guard(imsg: Path, config: Path, *, timeout: float = 120.0) -> tuple[bool, str]:
    """`imsg guard-mount --config <config>`: the full gate (the volume is
    mounted and encrypted, the sentinel is present, `data_root` resolves
    inside it)."""
    try:
        done = subprocess.run(
            [str(imsg), "guard-mount", "--config", str(config)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (done.stderr or done.stdout).strip().splitlines()
    return done.returncode == 0, (detail[-1] if detail else f"exit {done.returncode}")


def record_start_and_count(state_file: Path, *, now: float, window: float) -> int:
    """Append this start to `state_file` and return how many starts fall
    within the last `window` seconds, this one included. Older lines are
    dropped as the file is rewritten."""
    recent: list[float] = []
    try:
        for line in state_file.read_text(encoding="utf-8").splitlines():
            try:
                stamp = float(line)
            except ValueError:
                continue
            if now - window < stamp <= now:
                recent.append(stamp)
    except FileNotFoundError:
        pass
    recent.append(now)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_name(state_file.name + ".tmp")
    tmp.write_text("".join(f"{stamp:.3f}\n" for stamp in recent), encoding="utf-8")
    os.replace(tmp, state_file)
    return len(recent)


def postgres_ready(host: str, port: int, *, pg_isready: Path | None, timeout: float = 5.0) -> bool:
    """`pg_isready` when its path is given (it knows "accepting
    connections" from "starting up"); otherwise a TCP connect."""
    if pg_isready is not None:
        try:
            done = subprocess.run(
                [str(pg_isready), "-q", "-h", host, "-p", str(port), "-t", str(int(timeout))],
                check=False,
                timeout=timeout + 5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return done.returncode == 0
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def open_log(path: Path) -> int:
    """An append-only descriptor, created 0600 if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)


class _Logger:
    def __init__(self, fd: int, service: str) -> None:
        self._fd = fd
        self._service = service

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} supervise[{self._service}]: {message}\n"
        with contextlib.suppress(OSError):
            os.write(self._fd, line.encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supervise",
        description="Mount-gated, environment-loading supervisor for one imsg service "
        "(module docstring). Everything after -- is the service's command.",
    )
    parser.add_argument("--service", required=True, help="short name, e.g. mcp-public")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--imsg", required=True, type=Path, help="the imsg executable")
    parser.add_argument("--config", required=True, type=Path, help="config for imsg guard-mount")
    parser.add_argument(
        "--env", action="append", default=[], metavar="NAME=RELPATH", type=parse_env_file,
        help="set NAME from the 0600 file RELPATH under the data root (repeatable)",
    )
    parser.add_argument("--wait-for-postgres", metavar="HOST:PORT", default=None)
    parser.add_argument("--pg-isready", type=Path, default=None)
    parser.add_argument("--postgres-wait", type=float, default=900.0)
    parser.add_argument("--stop-signal", choices=sorted(STOP_SIGNALS), default="TERM")
    parser.add_argument("--mount-wait", type=float, default=900.0)
    parser.add_argument("--crash-window", type=float, default=600.0)
    parser.add_argument("--crash-limit", type=int, default=5)
    parser.add_argument("--crash-backoff", type=float, default=300.0)
    parser.add_argument("--guard-log", type=Path, default=GUARD_LOG, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _child_status(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SuperviseError as exc:
        print(f"supervise: {exc}", file=sys.stderr)
        return exc.status
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no service command after --")
    if not _SERVICE_RE.match(args.service):
        parser.error(f"--service must be lower-case letters, digits and '-', got {args.service!r}")
    data_root: Path = args.data_root
    if not data_root.is_absolute():
        parser.error("--data-root must be an absolute path")

    # 1. The mount: the sentinel, then the real gate.
    if not wait_for_sentinel(data_root, timeout=args.mount_wait):
        guard_log_line(
            f"{args.service}: {data_root / SENTINEL_NAME} absent after {args.mount_wait:.0f} s — "
            f"encrypted volume not mounted; exit {EX_CONFIG}, launchd will retry",
            guard_log=args.guard_log,
        )
        return EX_CONFIG
    ok, detail = run_guard(args.imsg, args.config)
    if not ok:
        guard_log_line(
            f"{args.service}: imsg guard-mount refused ({detail}); exit {EX_CONFIG}",
            guard_log=args.guard_log,
        )
        return EX_CONFIG

    # 2. From here on the service's own logs, on the volume.
    logs = data_root / "logs"
    out_fd = open_log(logs / f"imsgindex-{args.service}.out.log")
    err_fd = open_log(logs / f"imsgindex-{args.service}.err.log")
    log = _Logger(err_fd, args.service)

    # 3. Crash-loop brake.
    starts = record_start_and_count(
        data_root / "run" / f"supervise-{args.service}.starts",
        now=time.time(),
        window=args.crash_window,
    )
    if starts > args.crash_limit:
        log(
            f"{starts} starts in the last {args.crash_window:.0f} s (limit {args.crash_limit}); "
            f"waiting {args.crash_backoff:.0f} s before starting again"
        )
        time.sleep(args.crash_backoff)

    # 4. The environment, from 0600 files. Names are logged, values never.
    env = dict(os.environ)
    try:
        for entry in args.env:
            env[entry.name] = read_env_file(data_root, entry)
    except SuperviseError as exc:
        log(f"refusing to start: {exc}")
        return exc.status

    # 5. Postgres first, for the services that need it.
    if args.wait_for_postgres:
        host, _, port_text = args.wait_for_postgres.rpartition(":")
        try:
            port = int(port_text)
        except ValueError:
            log(f"--wait-for-postgres must be HOST:PORT, got {args.wait_for_postgres!r}")
            return EX_CONFIG
        deadline = time.monotonic() + args.postgres_wait
        while not postgres_ready(host, port, pg_isready=args.pg_isready):
            if time.monotonic() >= deadline:
                log(f"Postgres at {host}:{port} not ready after {args.postgres_wait:.0f} s")
                return EX_TEMPFAIL
            time.sleep(2.0)

    # 6. The service, supervised. Handlers go in before the spawn, so a
    # stop request that arrives in between is honoured, not lost.
    stop_signal = STOP_SIGNALS[args.stop_signal]
    running: list[subprocess.Popen[bytes]] = []
    pending: list[int] = []

    def forward(signum: int, _frame: object) -> None:
        target = stop_signal if signum == signal.SIGTERM else signum
        if not running:
            pending.append(target)
            return
        with contextlib.suppress(ProcessLookupError):
            running[0].send_signal(target)

    for signum in FORWARDED_SIGNALS:
        signal.signal(signum, forward)
    log(
        f"starting (env: {', '.join(e.name for e in args.env) or 'none'}; stop signal "
        f"{args.stop_signal}): {' '.join(command)}"
    )
    try:
        child = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=out_fd, stderr=err_fd
        )
    except OSError as exc:
        log(f"could not start: {type(exc).__name__}: {exc}")
        return EX_CONFIG
    running.append(child)
    for target in pending:
        child.send_signal(target)
    started = time.monotonic()
    returncode = child.wait()
    status = _child_status(returncode)
    log(f"exited with status {status} after {time.monotonic() - started:.1f} s")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
