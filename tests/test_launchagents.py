"""Unit tests for `imsg.agents.plists` (SPEC §5.5) — no filesystem
writes, no live Postgres. Fictional personas/paths only (D5): this
module renders `~/Library/LaunchAgents`-bound content, so these tests
also double as the leak-check the project's CLAUDE.md mandates for
this specific surface."""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from conftest import ConfigDictFactory
from imsg.agents.plists import (
    AGENT_NAMES,
    LABEL_PREFIX,
    calendar_intervals_for_window,
    render_agent_plists,
)
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.fingerprint import PG_DATA_SUBDIR

# --------------------------------------------------------------------------
# calendar_intervals_for_window
# --------------------------------------------------------------------------


def test_calendar_intervals_default_window() -> None:
    intervals = calendar_intervals_for_window("01:00-07:00")
    assert intervals[0] == {"Hour": 1, "Minute": 0}
    assert intervals[-1] == {"Hour": 7, "Minute": 0}
    # Spaced 30 minutes apart by default.
    assert intervals[1] == {"Hour": 1, "Minute": 30}
    assert len(intervals) == 13  # 01:00, 01:30, ..., 07:00


def test_calendar_intervals_custom_spacing() -> None:
    intervals = calendar_intervals_for_window("02:00-03:00", every_minutes=20)
    assert intervals == [
        {"Hour": 2, "Minute": 0},
        {"Hour": 2, "Minute": 20},
        {"Hour": 2, "Minute": 40},
        {"Hour": 3, "Minute": 0},
    ]


def test_calendar_intervals_rejects_malformed_window() -> None:
    with pytest.raises(ValueError, match="HH:MM-HH:MM"):
        calendar_intervals_for_window("not-a-window")


def test_calendar_intervals_rejects_midnight_wrap() -> None:
    with pytest.raises(ValueError, match="wraps past midnight"):
        calendar_intervals_for_window("22:00-02:00")


def test_calendar_intervals_rejects_nonpositive_spacing() -> None:
    with pytest.raises(ValueError, match="every_minutes must be positive"):
        calendar_intervals_for_window("01:00-02:00", every_minutes=0)


# --------------------------------------------------------------------------
# render_agent_plists
# --------------------------------------------------------------------------


_FIXED_DATA_ROOT = "/Volumes/IMSG-Data/imsgindex"
"""A fixed, fictional `data_root` — deliberately NOT derived from
pytest's own `tmp_path` fixture, whose default location
(`/…/pytest-of-<local-username>/…`) embeds the real local OS username
and would make the leak-substring check below fire on every run for a
reason that has nothing to do with this module's own output."""


@pytest.fixture
def config(config_dict_factory: ConfigDictFactory) -> Config:
    return load_config_dict(config_dict_factory(**{"paths.data_root": _FIXED_DATA_ROOT}))


INTERPRETER = Path("/opt/example/python3.12")
SUPERVISOR = Path("/opt/example/imsg/agents/supervise.py")
IMSG = Path("/usr/local/bin/imsg")
POSTGRES = Path("/opt/homebrew/opt/postgresql@17/bin/postgres")
CONFIG_PATH = Path("/Volumes/IMSG-Data/imsgindex/private/config.yaml")


def _render(config: Config, **kwargs: object) -> dict[str, bytes]:
    options: dict[str, object] = {
        "imsg_binary": IMSG,
        "postgres_binary": POSTGRES,
        "cloudflared_binary": Path("/opt/homebrew/bin/cloudflared"),
        "config_path": CONFIG_PATH,
        "interpreter": INTERPRETER,
        "supervisor": SUPERVISOR,
    }
    options.update(kwargs)
    return render_agent_plists(config, **options)  # type: ignore[arg-type]


@pytest.fixture
def rendered(config: Config) -> dict[str, bytes]:
    return _render(config)


def _parsed(rendered: dict[str, bytes], name: str) -> dict[str, object]:
    parsed: dict[str, object] = plistlib.loads(rendered[f"{LABEL_PREFIX}{name}"])
    return parsed


def _split(parsed: dict[str, object]) -> tuple[list[str], list[str]]:
    """`(supervisor arguments, service command)` — either side of `--`."""
    arguments = parsed["ProgramArguments"]
    assert isinstance(arguments, list)
    cut = arguments.index("--")
    return arguments[:cut], arguments[cut + 1 :]


def _option_values(arguments: list[str], flag: str) -> list[str]:
    return [arguments[i + 1] for i, arg in enumerate(arguments) if arg == flag]


def test_renders_exactly_seven_labeled_agents(rendered: dict[str, bytes]) -> None:
    assert set(rendered.keys()) == {f"{LABEL_PREFIX}{name}" for name in AGENT_NAMES}
    assert len(rendered) == 7


def test_every_plist_round_trips_and_has_a_label_and_program_arguments(
    rendered: dict[str, bytes],
) -> None:
    for label, content in rendered.items():
        parsed = plistlib.loads(content)
        assert parsed["Label"] == label
        assert isinstance(parsed["ProgramArguments"], list)
        assert parsed["ProgramArguments"]  # non-empty
        assert all(isinstance(arg, str) for arg in parsed["ProgramArguments"])


def test_every_agent_runs_the_supervisor_with_the_mount_gate(
    rendered: dict[str, bytes], config: Config
) -> None:
    """One program for every agent: the interpreter that holds the Full
    Disk Access grant, running the supervisor, which gates on the mount
    (`--data-root`, `--imsg ... guard-mount --config`) before the service."""
    for label, content in rendered.items():
        parsed = plistlib.loads(content)
        supervisor, command = _split(parsed)
        assert supervisor[:2] == [str(INTERPRETER), str(SUPERVISOR)], label
        assert _option_values(supervisor, "--service") == [label.removeprefix(LABEL_PREFIX)]
        assert _option_values(supervisor, "--data-root") == [str(config.paths.data_root)]
        assert _option_values(supervisor, "--imsg") == [str(IMSG)]
        assert _option_values(supervisor, "--config") == [str(CONFIG_PATH)]
        assert command, label
        assert parsed["ThrottleInterval"] == 60, "SPEC §5.4: retry every 60 s until the mount appears"


def test_launchd_output_goes_nowhere_and_the_supervisor_logs_on_the_volume(
    rendered: dict[str, bytes],
) -> None:
    """launchd opens StandardOutPath before anything runs — before the
    volume is mounted — so the supervisor opens the logs itself, on the
    volume, after the gate."""
    for content in rendered.values():
        parsed = plistlib.loads(content)
        assert parsed["StandardOutPath"] == "/dev/null"
        assert parsed["StandardErrorPath"] == "/dev/null"


def test_pg_agent_is_keepalive_and_uses_dedicated_port_and_pg17_dir(
    rendered: dict[str, bytes], config: Config
) -> None:
    parsed = _parsed(rendered, "pg")
    assert parsed["KeepAlive"] is True and parsed["RunAtLoad"] is True
    supervisor, command = _split(parsed)
    assert command[0] == str(POSTGRES)
    joined = " ".join(command)
    assert "-p 5433" in joined
    assert f"-D {config.paths.data_root / PG_DATA_SUBDIR}" in joined
    assert f"-k {config.paths.data_root / 'run'}" in joined
    assert "listen_addresses=127.0.0.1" in joined
    # launchd's TERM would be a smart shutdown that waits on every client.
    assert _option_values(supervisor, "--stop-signal") == ["INT"]
    assert _option_values(supervisor, "--env") == [], "Postgres gets no secrets"
    environment = parsed["EnvironmentVariables"]
    assert isinstance(environment, dict)
    assert environment["LC_ALL"] == "C"
    assert str(POSTGRES.parent) in environment["PATH"].split(":")
    assert parsed["ExitTimeOut"] >= 60  # type: ignore[operator]
    assert parsed["ProcessType"] == "Interactive"


def test_sync_agent_uses_configured_interval(rendered: dict[str, bytes], config: Config) -> None:
    parsed = _parsed(rendered, "sync")
    assert parsed["StartInterval"] == config.sync.interval_seconds
    supervisor, command = _split(parsed)
    assert command == [str(IMSG), "sync", "--config", str(CONFIG_PATH)]
    assert _option_values(supervisor, "--env") == [
        "IMSG_TEST_PG_PASSWORD=private/env/IMSG_TEST_PG_PASSWORD"
    ]


def test_enrich_agent_uses_configured_window(rendered: dict[str, bytes], config: Config) -> None:
    parsed = _parsed(rendered, "enrich")
    assert parsed["StartCalendarInterval"] == calendar_intervals_for_window(config.enrichment.window)
    _, command = _split(parsed)
    assert command[1] == "enrich"


def test_mcp_public_agent_is_keepalive_and_waits_for_postgres(rendered: dict[str, bytes]) -> None:
    parsed = _parsed(rendered, "mcp-public")
    assert parsed["KeepAlive"] is True and parsed["RunAtLoad"] is True
    assert parsed["ProcessType"] == "Interactive"
    supervisor, command = _split(parsed)
    assert command == [str(IMSG), "mcp", "public", "--config", str(CONFIG_PATH)]
    assert _option_values(supervisor, "--wait-for-postgres") == ["127.0.0.1:5433"]
    assert _option_values(supervisor, "--pg-isready") == [str(POSTGRES.parent / "pg_isready")]


def test_mcp_public_agent_names_its_secrets_and_never_holds_their_values(
    config_dict_factory: ConfigDictFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public server's `env:` references become `--env NAME=FILE`
    pairs; the values — present in this process's environment here, as
    they would be on an operator's shell — never reach a plist."""
    monkeypatch.setenv("IMSG_OWNER_SUB", "sentinel-subject-value")
    monkeypatch.setenv("IMSG_OAUTH_CLIENT_ID", "sentinel-client-value")
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "sentinel-password-value")
    base = load_config_dict(config_dict_factory(**{"paths.data_root": _FIXED_DATA_ROOT}))
    public = load_config_dict(
        config_dict_factory(
            **{
                "paths.data_root": _FIXED_DATA_ROOT,
                "mcp.public.oauth": {
                    "client_id": "env:IMSG_OAUTH_CLIENT_ID",
                    "owner_subject": "env:IMSG_OWNER_SUB",
                },
            }
        )
    )
    public_path = Path(_FIXED_DATA_ROOT) / "private" / "config.public.yaml"
    rendered = _render(
        base,
        mcp_public_config=public,
        mcp_public_config_path=public_path,
        env_files={"IMSG_OWNER_SUB": "private/owner-sub"},
    )
    supervisor, command = _split(_parsed(rendered, "mcp-public"))
    assert command[-1] == str(public_path)
    assert _option_values(supervisor, "--config") == [str(public_path)]
    assert _option_values(supervisor, "--env") == [
        "IMSG_TEST_PG_PASSWORD=private/env/IMSG_TEST_PG_PASSWORD",
        "IMSG_OAUTH_CLIENT_ID=private/env/IMSG_OAUTH_CLIENT_ID",
        "IMSG_OWNER_SUB=private/owner-sub",
    ]
    for content in rendered.values():
        assert b"sentinel-" not in content


def test_tunnel_agent_is_keepalive_and_uses_bootstrap_config_path(
    rendered: dict[str, bytes], config: Config
) -> None:
    parsed = _parsed(rendered, "tunnel")
    assert parsed["KeepAlive"] is True
    _, command = _split(parsed)
    assert command[0] == "/opt/homebrew/bin/cloudflared"
    assert str(config.paths.data_root / "private" / "cloudflared.yaml") in command


def test_report_agent_fires_monday_eight_am(rendered: dict[str, bytes]) -> None:
    parsed = _parsed(rendered, "report")
    assert parsed["StartCalendarInterval"] == {"Weekday": 1, "Hour": 8, "Minute": 0}
    _, command = _split(parsed)
    assert command[1:3] == ["export", "unclassified-report"]


def test_backup_agent_fires_daily_four_am_with_the_matching_pg_dump(rendered: dict[str, bytes]) -> None:
    parsed = _parsed(rendered, "backup")
    assert parsed["StartCalendarInterval"] == {"Hour": 4, "Minute": 0}
    assert parsed["ProcessType"] == "Background"
    assert "KeepAlive" not in parsed
    supervisor, command = _split(parsed)
    assert command[1] == "backup"
    assert command[-2:] == ["--pg-dump", str(POSTGRES.parent / "pg_dump")]
    assert _option_values(supervisor, "--wait-for-postgres") == ["127.0.0.1:5433"]


def test_only_renders_the_requested_agents_and_the_tunnel_needs_cloudflared(config: Config) -> None:
    subset = _render(config, cloudflared_binary=None, only=["pg", "mcp-public", "backup"])
    assert set(subset) == {f"{LABEL_PREFIX}{n}" for n in ("pg", "mcp-public", "backup")}
    with pytest.raises(ValueError, match="cloudflared"):
        _render(config, cloudflared_binary=None, only=["tunnel"])
    with pytest.raises(ValueError, match="unknown agent"):
        _render(config, only=["pg", "nonsense"])


def _render_with_data_root(config_dict_factory: ConfigDictFactory, data_root: str) -> dict[str, bytes]:
    config = load_config_dict(config_dict_factory(**{"paths.data_root": data_root}))
    return _render(config, config_path=Path(f"{data_root}/private/config.yaml"))


def test_plists_are_config_driven_not_hardcoded(
    config_dict_factory: ConfigDictFactory,
) -> None:
    """Instance-specific values reach a plist ONLY via config (D5, SPEC §3).

    Asserted with a fictional sentinel rather than a denylist of real
    terms. An earlier version of this test hardcoded the project's
    actual leak-check denylist — which made the test guarding against
    leaks into the leak itself, since the core repo is published and
    that denylist enumerates the sensitive terms by name. Per SPEC
    §3.2 the denylist belongs in the private overlay and runs as a
    local pre-commit hook, never here.

    This formulation is also strictly stronger. A denylist catches only
    the terms someone remembered to list; proving the renderer is
    config-driven catches *any* hardcoded instance value, including
    ones nobody thought to forbid.
    """
    sentinel = "sentinel-a1b2c3"
    with_sentinel = _render_with_data_root(config_dict_factory, f"/Volumes/{sentinel}/imsgindex")
    without_sentinel = _render_with_data_root(config_dict_factory, _FIXED_DATA_ROOT)

    # Guard against a vacuous test: if no config value ever reaches a
    # plist, the absence check below would pass for the wrong reason.
    assert any(sentinel.encode() in content for content in with_sentinel.values()), (
        "no rendered plist reflected the configured data_root — this test would "
        "otherwise pass vacuously and prove nothing"
    )

    for label, content in without_sentinel.items():
        assert sentinel.encode() not in content, (
            f"{label} plist contains a value this config never supplied — it is hardcoded"
        )


def test_a_file_referenced_database_password_is_not_passed_through_the_environment(
    config_dict_factory: ConfigDictFactory,
) -> None:
    """With `database.password: file:<path>` the service reads the file
    itself, so no agent gets an `--env IMSG_…PASSWORD=…` pair for it."""
    config = load_config_dict(
        config_dict_factory(
            **{
                "paths.data_root": _FIXED_DATA_ROOT,
                "database.password": f"file:{_FIXED_DATA_ROOT}/private/env/IMSG_PG_PASSWORD",
            }
        )
    )
    rendered = _render(config)
    for name, content in rendered.items():
        parsed: dict[str, object] = plistlib.loads(content)
        arguments = parsed["ProgramArguments"]
        assert isinstance(arguments, list)
        if "--" not in arguments:
            continue
        supervisor = arguments[: arguments.index("--")]
        assert not any("PASSWORD" in value for value in _option_values(supervisor, "--env")), name
