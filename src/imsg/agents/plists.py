"""Renders the `com.imsgindex.*` LaunchAgent plists (SPEC §5.5).

launchd only discovers user agents from `~/Library/LaunchAgents`, so
installation writes **thin, content-free plists** there. Since
2026-09-24 every one of them runs the same program: the supervisor,
`imsg/agents/supervise.py`, under one fixed interpreter, wrapping the
service's command. The supervisor waits for the encrypted volume, passes
`imsg guard-mount`, reads the service's `env:` secrets from 0600 files
under `data_root`, opens the service's logs on the volume, and keeps the
service as its child so launchd's `KeepAlive` restarts it after a crash
(QA review 2026-09-24: the public server, Postgres and the backups had
no supervisor).

Every `ProgramArguments` list built here MUST be assembled from generic,
fixed bootstrap values only — no literal hostname, secret, person name,
or GCP identifier anywhere in the rendered dict (SPEC §5.5: "the plist
MUST contain no secret, hostname, person name, GCP identifier, or message
path beyond the fixed bootstrap paths"). A secret reference contributes
its variable NAME and the path of the file holding it, never a value.

Everything here is a pure function: build a `dict` shaped for
`plistlib.dumps(..., fmt=plistlib.FMT_XML)`, no filesystem writes, no
templating engine — plain Python dict construction is both sufficient
and more testable than a string template (round-trip through
`plistlib.loads` in tests instead of parsing XML by hand). Only
`imsg.cli`'s `install-agents` command actually writes the rendered
bytes to disk.
"""

from __future__ import annotations

import plistlib
import re
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.config.schema import REQUIRED_DB_PORT
from imsg.db.fingerprint import PG_DATA_SUBDIR

if TYPE_CHECKING:
    from imsg.config.schema import Config

LABEL_PREFIX = "com.imsgindex."

_WINDOW_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")
"""Same shape `imsg.config.schema`'s `EnrichmentConfig.window` validator
already enforces (`HH:MM-HH:MM`) — duplicated here rather than imported
since that one is a private module-level constant; keeping this
module's own copy also means it does not silently drift if the schema
module's regex changes in a way this parser cannot handle (see the
wrap-past-midnight note in `calendar_intervals_for_window`)."""


def calendar_intervals_for_window(window: str, *, every_minutes: int = 30) -> list[dict[str, int]]:
    """Parse an `HH:MM-HH:MM` window into launchd `StartCalendarInterval`
    entries (SPEC §5.5's `…enrich` agent: "nightly window ... S5
    enrichment queue + attachment backfill throttle").

    launchd has no native "recurring every N minutes within a window"
    primitive — `imsg enrich` itself claims and processes one bounded
    batch and exits, so an *array* of discrete
    `{"Hour": h, "Minute": m}` entries, `every_minutes` apart from the
    window's start up to (and including, if it lands exactly on the
    boundary) its end, is what gives "runs repeatedly within the
    window" semantics.

    Only supports a window where `end > start` — SPEC's example
    `"01:00-07:00"` doesn't wrap midnight, and `enrichment.window`'s
    own schema validator enforces the `HH:MM-HH:MM` shape but not
    ordering. A window that wraps past midnight (e.g. `"22:00-02:00"`)
    is unsupported: raises `ValueError` naming the gap explicitly
    rather than silently producing a truncated/wrong schedule. TODO if
    this is ever needed: split into two ranges, `[start, 24:00)` and
    `[00:00, end]`.
    """
    match = _WINDOW_RE.match(window)
    if not match:
        raise ValueError(f"window must look like 'HH:MM-HH:MM', got {window!r}")
    start_h, start_m, end_h, end_m = (int(g) for g in match.groups())
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    if end_minutes <= start_minutes:
        raise ValueError(
            f"window {window!r} wraps past midnight (end <= start) — "
            f"calendar_intervals_for_window does not support that yet "
            f"(TODO: split into [start, 24:00) and [00:00, end])"
        )
    if every_minutes <= 0:
        raise ValueError(f"every_minutes must be positive, got {every_minutes}")

    intervals: list[dict[str, int]] = []
    minute = start_minutes
    while minute <= end_minutes:
        intervals.append({"Hour": minute // 60, "Minute": minute % 60})
        minute += every_minutes
    return intervals


LOGS_NOTE = (
    "launchd's own standard output and error go to /dev/null: the supervisor "
    "(imsg.agents.supervise) opens <data_root>/logs/imsgindex-<service>.{out,err}.log "
    "itself, for appending, once the mount gate has passed — so nothing is written "
    "before the encrypted volume is there, and the size rotation can truncate them"
)

AGENT_NAMES: tuple[str, ...] = ("pg", "sync", "enrich", "mcp-public", "tunnel", "report", "backup")
"""SPEC §5.5's table, in its order."""

THROTTLE_SECONDS = 60
"""SPEC §5.4: agents "use ThrottleInterval 60 and simply retry until the
mount appears" — also the least time between two starts of a KeepAlive
service that keeps exiting."""

PG_EXIT_TIMEOUT_SECONDS = 120
"""How long launchd waits after the stop signal before it kills Postgres.
The supervisor turns launchd's TERM into Postgres's fast shutdown (INT),
which ends with a checkpoint of up to `shared_buffers` of dirty pages."""

MCP_EXIT_TIMEOUT_SECONDS = 30
POSTGRES_HOST = "127.0.0.1"

DEFAULT_ENV_DIR = "private/env"
"""Where, under `data_root`, the supervisor looks for the file holding an
`env:NAME` secret when the installer names no other: `private/env/NAME`,
mode 0600. `render_agent_plists(env_files=...)` points a name elsewhere
(an existing 0600 file, say)."""

SYSTEM_PATH: tuple[str, ...] = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)


@dataclass(frozen=True, slots=True)
class AgentBinaries:
    """Fixed bootstrap paths every plist is built from — never read from
    the config's contents, never a secret or hostname (SPEC §5.5)."""

    interpreter: Path
    """The interpreter launchd starts. Grant Full Disk Access to this
    binary (SPEC §5.1a): the services run as its children."""
    supervisor: Path
    """`imsg/agents/supervise.py` — standard library only, so
    `interpreter` needs no virtual environment to run it."""
    imsg: Path
    postgres: Path
    cloudflared: Path | None = None


def default_interpreter() -> Path:
    """The interpreter running this process, symlinks resolved: for a
    virtual environment, the base interpreter its `python` links to —
    one stable binary to hold the Full Disk Access grant."""
    return Path(sys.executable).resolve()


def default_supervisor() -> Path:
    return (Path(__file__).resolve().parent / "supervise.py").resolve()


def env_secret_names(config: Config, *, public: bool) -> list[str]:
    """The `env:NAME` secrets a service built from `config` resolves at
    start: the database password for every service that connects, plus
    the public surface's OAuth references for `mcp-public`. Names only —
    their values are read by the supervisor from 0600 files."""
    refs: list[str] = []
    if config.database.password.kind == "env":
        refs.append(config.database.password.name)
    if public:
        oauth = config.mcp.public.oauth
        if oauth.client_id is not None and oauth.client_id.startswith("env:"):
            refs.append(oauth.client_id.removeprefix("env:"))
        for ref in (oauth.owner_subject, oauth.client_secret):
            if ref is not None and ref.kind == "env":
                refs.append(ref.name)
    return list(dict.fromkeys(refs))


def _search_path(binaries: AgentBinaries) -> str:
    dirs = [str(binaries.postgres.parent), str(binaries.imsg.parent), *SYSTEM_PATH]
    return ":".join(dict.fromkeys(dirs))


def _supervised(
    name: str,
    *,
    binaries: AgentBinaries,
    data_root: Path,
    guard_config: Path,
    command: list[str],
    env_names: Sequence[str] = (),
    env_files: Mapping[str, str] | None = None,
    wait_for_postgres: bool = False,
    stop_signal: str = "TERM",
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """One plist whose program is the supervisor wrapping `command`."""
    overrides = env_files or {}
    arguments = [
        str(binaries.interpreter),
        str(binaries.supervisor),
        "--service",
        name,
        "--data-root",
        str(data_root),
        "--imsg",
        str(binaries.imsg),
        "--config",
        str(guard_config),
    ]
    for env_name in env_names:
        relative = overrides.get(env_name, f"{DEFAULT_ENV_DIR}/{env_name}")
        arguments += ["--env", f"{env_name}={relative}"]
    if wait_for_postgres:
        arguments += [
            "--wait-for-postgres",
            f"{POSTGRES_HOST}:{REQUIRED_DB_PORT}",
            "--pg-isready",
            str(binaries.postgres.parent / "pg_isready"),
        ]
    if stop_signal != "TERM":
        arguments += ["--stop-signal", stop_signal]
    return {
        "Label": f"{LABEL_PREFIX}{name}",
        "ProgramArguments": [*arguments, "--", *command],
        "EnvironmentVariables": {"PATH": _search_path(binaries), **(environment or {})},
        "StandardOutPath": "/dev/null",
        "StandardErrorPath": "/dev/null",
        "ThrottleInterval": THROTTLE_SECONDS,
    }


def render_agent_plists(
    config: Config,
    *,
    imsg_binary: Path,
    postgres_binary: Path,
    cloudflared_binary: Path | None,
    config_path: Path,
    interpreter: Path | None = None,
    supervisor: Path | None = None,
    mcp_public_config: Config | None = None,
    mcp_public_config_path: Path | None = None,
    env_files: Mapping[str, str] | None = None,
    only: Collection[str] | None = None,
) -> dict[str, bytes]:
    """Render SPEC §5.5's `com.imsgindex.*` plists as `label -> XML bytes`,
    fully unit-testable: paths in, bytes out, nothing written.

    Every agent runs `interpreter supervisor … -- <service command>`
    (`imsg.agents.supervise`): the supervisor waits for the encrypted
    volume, passes `imsg guard-mount`, loads the service's `env:` secrets
    from 0600 files under `data_root` (`env_files` maps a name to a file
    other than `private/env/NAME`), and keeps the service as its child.
    Nothing in a plist comes from the config's *values* except
    `data_root`, the sync interval and the enrichment window; secret
    references contribute their variable *names* only (SPEC §5.5).

    `mcp_public_config`/`_path` let the public server run from its own
    config file (the instance renders one with the public hostname);
    both default to `config`/`config_path`. `only` renders a subset of
    `AGENT_NAMES`; the tunnel needs `cloudflared_binary`.

    Service by service:

    - `pg` — KeepAlive; `postgres` in the foreground on port 5433, data in
      `data_root/pg17`, TCP on 127.0.0.1 only, its Unix socket in
      `data_root/run` (not the world-writable `/tmp`); `LC_ALL=C` (or
      the postmaster dies at startup on macOS); launchd's TERM becomes a
      fast shutdown; `ProcessType` Interactive, since every query waits on it.
    - `mcp-public` — KeepAlive; waits for Postgres, then `imsg mcp public`;
      Interactive.
    - `backup` — daily 04:00; waits for Postgres, then `imsg backup` with
      the `pg_dump` beside `postgres`, which also rotates the logs;
      `ProcessType` Background.
    - `sync`, `enrich`, `report` — their schedules, unchanged in shape.
    - `tunnel` — KeepAlive `cloudflared`, only when asked for.
    """
    data_root = config.paths.data_root
    binaries = AgentBinaries(
        interpreter=interpreter or default_interpreter(),
        supervisor=supervisor or default_supervisor(),
        imsg=imsg_binary,
        postgres=postgres_binary,
        cloudflared=cloudflared_binary,
    )
    selected = list(AGENT_NAMES) if only is None else [n for n in AGENT_NAMES if n in set(only)]
    unknown = set(only or ()) - set(AGENT_NAMES)
    if unknown:
        raise ValueError(f"unknown agent(s) {sorted(unknown)}; choose from {list(AGENT_NAMES)}")
    public_config = mcp_public_config or config
    public_path = mcp_public_config_path or config_path
    db_env = env_secret_names(config, public=False)
    imsg = str(imsg_binary)
    cfg = str(config_path)

    builders: dict[str, Callable[[], dict[str, object]]] = {
        "pg": lambda: _pg_plist(binaries, data_root, config_path),
        "sync": lambda: {
            **_supervised(
                "sync",
                binaries=binaries,
                data_root=data_root,
                guard_config=config_path,
                command=[imsg, "sync", "--config", cfg],
                env_names=db_env,
                env_files=env_files,
            ),
            "StartInterval": config.sync.interval_seconds,
        },
        "enrich": lambda: {
            **_supervised(
                "enrich",
                binaries=binaries,
                data_root=data_root,
                guard_config=config_path,
                command=[imsg, "enrich", "--config", cfg],
                env_names=db_env,
                env_files=env_files,
            ),
            "StartCalendarInterval": calendar_intervals_for_window(config.enrichment.window),
        },
        "mcp-public": lambda: {
            **_supervised(
                "mcp-public",
                binaries=binaries,
                data_root=data_root,
                guard_config=public_path,
                command=[imsg, "mcp", "public", "--config", str(public_path)],
                env_names=env_secret_names(public_config, public=True),
                env_files=env_files,
                wait_for_postgres=True,
            ),
            "KeepAlive": True,
            "RunAtLoad": True,
            "ExitTimeOut": MCP_EXIT_TIMEOUT_SECONDS,
            "ProcessType": "Interactive",
        },
        "tunnel": lambda: _tunnel_plist(binaries, data_root, config_path),
        "report": lambda: {
            **_supervised(
                "report",
                binaries=binaries,
                data_root=data_root,
                guard_config=config_path,
                command=[imsg, "export", "unclassified-report", "--config", cfg],
                env_names=db_env,
                env_files=env_files,
                wait_for_postgres=True,
            ),
            # launchd's Weekday is 0/7 = Sunday, so Monday is 1.
            "StartCalendarInterval": {"Weekday": 1, "Hour": 8, "Minute": 0},
        },
        "backup": lambda: {
            **_supervised(
                "backup",
                binaries=binaries,
                data_root=data_root,
                guard_config=config_path,
                command=[
                    imsg,
                    "backup",
                    "--config",
                    cfg,
                    "--pg-dump",
                    str(postgres_binary.parent / "pg_dump"),
                ],
                env_names=db_env,
                env_files=env_files,
                wait_for_postgres=True,
            ),
            "StartCalendarInterval": {"Hour": 4, "Minute": 0},
            "ProcessType": "Background",
        },
    }
    return {
        f"{LABEL_PREFIX}{name}": plistlib.dumps(builders[name](), fmt=plistlib.FMT_XML)
        for name in selected
    }


def _pg_plist(binaries: AgentBinaries, data_root: Path, config_path: Path) -> dict[str, object]:
    """`…pg` — the dedicated instance (SPEC §5.2) under launchd as its only
    owner. `postgres` itself refuses to start while another postmaster
    holds the data directory's `postmaster.pid`, so a start that overlaps
    an old instance fails and retries; it never runs two."""
    plist = _supervised(
        "pg",
        binaries=binaries,
        data_root=data_root,
        guard_config=config_path,
        command=[
            str(binaries.postgres),
            "-D",
            str(data_root / PG_DATA_SUBDIR),
            "-p",
            str(REQUIRED_DB_PORT),
            "-c",
            f"listen_addresses={POSTGRES_HOST}",
            "-k",
            str(data_root / "run"),
        ],
        stop_signal="INT",
        environment={"LC_ALL": "C"},
    )
    plist.update(
        {
            "KeepAlive": True,
            "RunAtLoad": True,
            "ExitTimeOut": PG_EXIT_TIMEOUT_SECONDS,
            "ProcessType": "Interactive",
        }
    )
    return plist


def _tunnel_plist(binaries: AgentBinaries, data_root: Path, config_path: Path) -> dict[str, object]:
    """`…tunnel` — KeepAlive `cloudflared tunnel --config <rendered> run`.
    The cloudflared config is a fixed path under `data_root/private`; the
    tunnel hostname lives inside it, never here."""
    if binaries.cloudflared is None:
        raise ValueError("the tunnel agent needs the cloudflared binary")
    plist = _supervised(
        "tunnel",
        binaries=binaries,
        data_root=data_root,
        guard_config=config_path,
        command=[
            str(binaries.cloudflared),
            "tunnel",
            "--config",
            str(data_root / "private" / "cloudflared.yaml"),
            "run",
        ],
    )
    plist.update({"KeepAlive": True, "RunAtLoad": True})
    return plist


__all__ = [
    "AGENT_NAMES",
    "DEFAULT_ENV_DIR",
    "LABEL_PREFIX",
    "LOGS_NOTE",
    "AgentBinaries",
    "calendar_intervals_for_window",
    "default_interpreter",
    "default_supervisor",
    "env_secret_names",
    "render_agent_plists",
]
