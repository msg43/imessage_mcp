"""Renders the 7 `com.imsgindex.*` LaunchAgent plists (SPEC §5.5).

launchd only discovers user agents from `~/Library/LaunchAgents`, so
installation writes **thin, content-free plists** there: each one
invokes a fixed core wrapper (the installed `imsg` binary, or a raw
`postgres`/`cloudflared` binary gated by an explicit `imsg guard-mount`
shell wrapper) that reads real config/runtime files from
`$DATA_ROOT` at run time. Every `ProgramArguments` list built here MUST
be assembled from generic, fixed bootstrap values only — no literal
hostname, secret, person name, or GCP identifier anywhere in the
rendered dict (SPEC §5.5: "the plist MUST contain no secret, hostname,
person name, GCP identifier, or message path beyond the fixed
bootstrap paths"). Real values live only in the config.yaml the
rendered plist points at via `--config <path>`, which this module
never reads the contents of.

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


def _log_paths(data_root: Path, agent_name: str) -> tuple[Path, Path]:
    """SPEC §5.3: `logs/` lives under `data_root` ("all derived state
    on the encrypted volume" — CLAUDE.md non-negotiable #2). Unlike the
    mount gate's own off-mount failure log (the one deliberate
    exception, SPEC §5.4), a LaunchAgent's stdout/stderr capture is
    ordinary pipeline output and belongs on the encrypted volume."""
    logs_dir = data_root / "logs"
    return (
        logs_dir / f"imsgindex-{agent_name}.out.log",
        logs_dir / f"imsgindex-{agent_name}.err.log",
    )


def _guard_mount_and_exec(*, imsg_binary: Path, config_path: Path, exec_cmd: str) -> list[str]:
    """The explicit `sh -c "imsg guard-mount ... && exec ..."` wrapper
    SPEC §5.5 requires for agents that invoke a **raw** binary
    (`postgres`, `cloudflared`) with no `imsg` CLI wrapper of its own
    to gate itself. Agents that shell out to `imsg <subcommand>`
    directly do NOT need this: every real `imsg` stage command already
    runs the mount gate itself before touching `data_root` (see
    `imsg.cli`'s per-command `run_guard_mount_or_exit` calls, and
    `imsg.stages.sync.run_sync`'s own internal `guard_mount` call for
    `imsg sync` specifically) — SPEC §5.4's "every CLI entry point ...
    runs guard-mount" already covers those.
    """
    return [
        "/bin/sh",
        "-c",
        f"{imsg_binary} guard-mount --config {config_path} && exec {exec_cmd}",
    ]


def _base_plist(
    label: str, *, program_arguments: list[str], data_root: Path, agent_name: str
) -> dict[str, object]:
    out_log, err_log = _log_paths(data_root, agent_name)
    return {
        "Label": label,
        "ProgramArguments": program_arguments,
        "StandardOutPath": str(out_log),
        "StandardErrorPath": str(err_log),
    }


def _pg_plist(
    *, data_root: Path, imsg_binary: Path, postgres_binary: Path, config_path: Path
) -> dict[str, object]:
    """`…pg` — KeepAlive; gated (raw binary, no `imsg` wrapper of its
    own) then invokes `postgres` directly with the dedicated
    instance's fixed flags (SPEC §5.2: port 5433, `pg17` data dir under
    `data_root`, unix sockets under `data_root/run`)."""
    pg_data_dir = data_root / PG_DATA_SUBDIR
    run_dir = data_root / "run"
    exec_cmd = (
        f"{postgres_binary} -D {pg_data_dir} -p {REQUIRED_DB_PORT} "
        f"-c listen_addresses=127.0.0.1 -k {run_dir}"
    )
    plist = _base_plist(
        f"{LABEL_PREFIX}pg",
        program_arguments=_guard_mount_and_exec(
            imsg_binary=imsg_binary, config_path=config_path, exec_cmd=exec_cmd
        ),
        data_root=data_root,
        agent_name="pg",
    )
    plist["KeepAlive"] = True
    return plist


def _sync_plist(
    *, data_root: Path, imsg_binary: Path, config_path: Path, interval_seconds: int
) -> dict[str, object]:
    """`…sync` — `StartInterval` from `config.sync.interval_seconds`;
    `imsg sync` gates itself (see `_guard_mount_and_exec`'s docstring),
    so no extra shell wrapper is needed here."""
    plist = _base_plist(
        f"{LABEL_PREFIX}sync",
        program_arguments=[str(imsg_binary), "sync", "--config", str(config_path)],
        data_root=data_root,
        agent_name="sync",
    )
    plist["StartInterval"] = interval_seconds
    return plist


def _enrich_plist(
    *, data_root: Path, imsg_binary: Path, config_path: Path, window: str
) -> dict[str, object]:
    """`…enrich` — nightly window (default `01:00-07:00`), periodic
    `StartCalendarInterval` array via `calendar_intervals_for_window`;
    `imsg enrich` gates itself the same way `imsg sync` does."""
    plist = _base_plist(
        f"{LABEL_PREFIX}enrich",
        program_arguments=[str(imsg_binary), "enrich", "--config", str(config_path)],
        data_root=data_root,
        agent_name="enrich",
    )
    plist["StartCalendarInterval"] = calendar_intervals_for_window(window)
    return plist


def _mcp_public_plist(*, data_root: Path, imsg_binary: Path, config_path: Path) -> dict[str, object]:
    """`…mcp-public` — KeepAlive; invokes `imsg mcp public --config
    <path>`. That subcommand is being built in parallel this same wave
    (`src/imsg/mcp/*`, someone else's territory) — this function only
    references the command name/args; it will exist by the time
    everything is integrated (SPEC §10.4)."""
    plist = _base_plist(
        f"{LABEL_PREFIX}mcp-public",
        program_arguments=[str(imsg_binary), "mcp", "public", "--config", str(config_path)],
        data_root=data_root,
        agent_name="mcp-public",
    )
    plist["KeepAlive"] = True
    return plist


def _tunnel_plist(
    *, data_root: Path, imsg_binary: Path, cloudflared_binary: Path, config_path: Path
) -> dict[str, object]:
    """`…tunnel` — KeepAlive; gated (raw binary, no `imsg` wrapper of
    its own), then `cloudflared tunnel --config <rendered config> run`.
    The cloudflared config path is itself a generic, fixed bootstrap
    path under `data_root` (SPEC §5.3 `private/` — "instance config /
    overlay checkout") — never a literal hostname; the real tunnel
    hostname lives inside that rendered config file, which this module
    never reads or writes."""
    cloudflared_config_path = data_root / "private" / "cloudflared.yaml"
    exec_cmd = f"{cloudflared_binary} tunnel --config {cloudflared_config_path} run"
    plist = _base_plist(
        f"{LABEL_PREFIX}tunnel",
        program_arguments=_guard_mount_and_exec(
            imsg_binary=imsg_binary, config_path=config_path, exec_cmd=exec_cmd
        ),
        data_root=data_root,
        agent_name="tunnel",
    )
    plist["KeepAlive"] = True
    return plist


def _report_plist(*, data_root: Path, imsg_binary: Path, config_path: Path) -> dict[str, object]:
    """`…report` — weekly, Monday 08:00 (launchd's `Weekday` is
    0/7=Sunday, so Monday is `1`). Invokes `imsg export
    unclassified-report --config <path>`, a CLI subcommand that does
    not exist yet as of this build (`src/imsg/export/unclassified.py`
    has the logic; wiring `imsg export` is a parallel agent's scope
    this wave, out of bounds to touch here) — the plist is still
    rendered correctly; the target command's existence is a separate,
    already-flagged gap."""
    plist = _base_plist(
        f"{LABEL_PREFIX}report",
        program_arguments=[
            str(imsg_binary), "export", "unclassified-report", "--config", str(config_path)
        ],
        data_root=data_root,
        agent_name="report",
    )
    plist["StartCalendarInterval"] = {"Weekday": 1, "Hour": 8, "Minute": 0}
    return plist


def _backup_plist(*, data_root: Path, imsg_binary: Path, config_path: Path) -> dict[str, object]:
    """`…backup` — daily 04:00. Invokes `imsg backup --config <path>`
    (SPEC §5.3/§14: "backups/ nightly local recovery copies, 14
    kept")."""
    plist = _base_plist(
        f"{LABEL_PREFIX}backup",
        program_arguments=[str(imsg_binary), "backup", "--config", str(config_path)],
        data_root=data_root,
        agent_name="backup",
    )
    plist["StartCalendarInterval"] = {"Hour": 4, "Minute": 0}
    return plist


def render_agent_plists(
    config: Config,
    *,
    imsg_binary: Path,
    postgres_binary: Path,
    cloudflared_binary: Path,
    config_path: Path,
) -> dict[str, bytes]:
    """Render all 7 `com.imsgindex.*` plists (SPEC §5.5's table) as
    `label -> XML plist bytes`, fully unit-testable without touching
    the filesystem: pass paths in, get bytes out (round-trip through
    `plistlib.loads` to assert on the dict; grep the bytes for
    forbidden substrings).

    `config` supplies `config.sync.interval_seconds` and
    `config.enrichment.window`; every other value here is a fixed
    bootstrap path derived from `config.paths.data_root` or the
    explicitly-passed binary/config paths — never a literal hostname,
    secret, or person name (SPEC §5.5).
    """
    data_root = config.paths.data_root
    builders: dict[str, dict[str, object]] = {
        f"{LABEL_PREFIX}pg": _pg_plist(
            data_root=data_root,
            imsg_binary=imsg_binary,
            postgres_binary=postgres_binary,
            config_path=config_path,
        ),
        f"{LABEL_PREFIX}sync": _sync_plist(
            data_root=data_root,
            imsg_binary=imsg_binary,
            config_path=config_path,
            interval_seconds=config.sync.interval_seconds,
        ),
        f"{LABEL_PREFIX}enrich": _enrich_plist(
            data_root=data_root,
            imsg_binary=imsg_binary,
            config_path=config_path,
            window=config.enrichment.window,
        ),
        f"{LABEL_PREFIX}mcp-public": _mcp_public_plist(
            data_root=data_root, imsg_binary=imsg_binary, config_path=config_path
        ),
        f"{LABEL_PREFIX}tunnel": _tunnel_plist(
            data_root=data_root,
            imsg_binary=imsg_binary,
            cloudflared_binary=cloudflared_binary,
            config_path=config_path,
        ),
        f"{LABEL_PREFIX}report": _report_plist(
            data_root=data_root, imsg_binary=imsg_binary, config_path=config_path
        ),
        f"{LABEL_PREFIX}backup": _backup_plist(
            data_root=data_root, imsg_binary=imsg_binary, config_path=config_path
        ),
    }
    return {label: plistlib.dumps(plist, fmt=plistlib.FMT_XML) for label, plist in builders.items()}


__all__ = [
    "LABEL_PREFIX",
    "calendar_intervals_for_window",
    "render_agent_plists",
]
