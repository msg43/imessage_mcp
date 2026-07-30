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


_FIXED_DATA_ROOT = "/Volumes/Data-Encrypted/imsgindex"
"""A fixed, fictional `data_root` — deliberately NOT derived from
pytest's own `tmp_path` fixture, whose default location
(`/…/pytest-of-<local-username>/…`) embeds the real local OS username
and would make the leak-substring check below fire on every run for a
reason that has nothing to do with this module's own output."""


@pytest.fixture
def config(config_dict_factory: ConfigDictFactory) -> Config:
    return load_config_dict(config_dict_factory(**{"paths.data_root": _FIXED_DATA_ROOT}))


@pytest.fixture
def rendered(config: Config) -> dict[str, bytes]:
    return render_agent_plists(
        config,
        imsg_binary=Path("/usr/local/bin/imsg"),
        postgres_binary=Path("/opt/homebrew/bin/postgres"),
        cloudflared_binary=Path("/opt/homebrew/bin/cloudflared"),
        config_path=Path("/Volumes/Data-Encrypted/imsgindex/private/config.yaml"),
    )


def test_renders_exactly_seven_labeled_agents(rendered: dict[str, bytes]) -> None:
    assert set(rendered.keys()) == {
        f"{LABEL_PREFIX}pg",
        f"{LABEL_PREFIX}sync",
        f"{LABEL_PREFIX}enrich",
        f"{LABEL_PREFIX}mcp-public",
        f"{LABEL_PREFIX}tunnel",
        f"{LABEL_PREFIX}report",
        f"{LABEL_PREFIX}backup",
    }


def test_every_plist_round_trips_and_has_a_label_and_program_arguments(
    rendered: dict[str, bytes],
) -> None:
    for label, content in rendered.items():
        parsed = plistlib.loads(content)
        assert parsed["Label"] == label
        assert isinstance(parsed["ProgramArguments"], list)
        assert parsed["ProgramArguments"]  # non-empty
        assert all(isinstance(arg, str) for arg in parsed["ProgramArguments"])


def test_pg_agent_is_keepalive_and_uses_dedicated_port_and_pg17_dir(
    rendered: dict[str, bytes], config: Config
) -> None:
    parsed = plistlib.loads(rendered[f"{LABEL_PREFIX}pg"])
    assert parsed["KeepAlive"] is True
    joined = " ".join(parsed["ProgramArguments"])
    assert "guard-mount" in joined  # raw binary -> explicit guard-mount wrapper
    assert "/opt/homebrew/bin/postgres" in joined
    assert "-p 5433" in joined
    assert str(config.paths.data_root / PG_DATA_SUBDIR) in joined
    assert str(config.paths.data_root / "run") in joined


def test_sync_agent_uses_configured_interval_and_no_guard_mount_wrapper(
    rendered: dict[str, bytes], config: Config
) -> None:
    parsed = plistlib.loads(rendered[f"{LABEL_PREFIX}sync"])
    assert parsed["StartInterval"] == config.sync.interval_seconds
    assert parsed["ProgramArguments"][0] == "/usr/local/bin/imsg"
    assert parsed["ProgramArguments"][1] == "sync"
    # imsg sync gates itself internally -> no extra guard-mount shell wrapper.
    assert "/bin/sh" not in parsed["ProgramArguments"]


def test_enrich_agent_uses_configured_window(rendered: dict[str, bytes], config: Config) -> None:
    parsed = plistlib.loads(rendered[f"{LABEL_PREFIX}enrich"])
    assert parsed["StartCalendarInterval"] == calendar_intervals_for_window(config.enrichment.window)
    assert "enrich" in parsed["ProgramArguments"]


def test_mcp_public_agent_references_command_not_yet_built(rendered: dict[str, bytes]) -> None:
    parsed = plistlib.loads(rendered[f"{LABEL_PREFIX}mcp-public"])
    assert parsed["KeepAlive"] is True
    assert parsed["ProgramArguments"][1:3] == ["mcp", "public"]


def test_tunnel_agent_is_keepalive_gated_and_uses_bootstrap_config_path(
    rendered: dict[str, bytes], config: Config
) -> None:
    parsed = plistlib.loads(rendered[f"{LABEL_PREFIX}tunnel"])
    assert parsed["KeepAlive"] is True
    joined = " ".join(parsed["ProgramArguments"])
    assert "guard-mount" in joined
    assert "/opt/homebrew/bin/cloudflared" in joined
    assert str(config.paths.data_root / "private" / "cloudflared.yaml") in joined


def test_report_agent_fires_monday_eight_am(rendered: dict[str, bytes]) -> None:
    parsed = plistlib.loads(rendered[f"{LABEL_PREFIX}report"])
    assert parsed["StartCalendarInterval"] == {"Weekday": 1, "Hour": 8, "Minute": 0}
    assert parsed["ProgramArguments"][1:3] == ["export", "unclassified-report"]


def test_backup_agent_fires_daily_four_am(rendered: dict[str, bytes]) -> None:
    parsed = plistlib.loads(rendered[f"{LABEL_PREFIX}backup"])
    assert parsed["StartCalendarInterval"] == {"Hour": 4, "Minute": 0}
    assert parsed["ProgramArguments"][1] == "backup"


def test_all_agents_log_under_data_root_logs(rendered: dict[str, bytes], config: Config) -> None:
    logs_dir = config.paths.data_root / "logs"
    for content in rendered.values():
        parsed = plistlib.loads(content)
        assert str(parsed["StandardOutPath"]).startswith(str(logs_dir))
        assert str(parsed["StandardErrorPath"]).startswith(str(logs_dir))


def _render_with_data_root(config_dict_factory: ConfigDictFactory, data_root: str) -> dict[str, bytes]:
    config = load_config_dict(config_dict_factory(**{"paths.data_root": data_root}))
    return render_agent_plists(
        config,
        imsg_binary=Path("/usr/local/bin/imsg"),
        postgres_binary=Path("/opt/homebrew/bin/postgres"),
        cloudflared_binary=Path("/opt/homebrew/bin/cloudflared"),
        config_path=Path(f"{data_root}/private/config.yaml"),
    )


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
