"""The search page's KeepAlive agent (D14), rendered by `imsg
install-agents --only search-page` under the shared supervisor: waits for
Postgres, passes the mount guard, reads the database secret from a 0600
file the plist names (never a value), and is never part of the default
set. Fixed, fictional paths only: this output lands in LaunchAgents."""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import ConfigDictFactory
from imsg.agents.plists import AGENT_NAMES, LABEL_PREFIX, OPTIONAL_AGENT_NAMES, render_agent_plists
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config

DATA_ROOT = "/Volumes/IMSG-Data/imsgindex"
INTERPRETER = Path("/opt/example/python3.12")
SUPERVISOR = Path("/opt/example/imsg/agents/supervise.py")
IMSG = Path("/usr/local/bin/imsg")
POSTGRES = Path("/opt/homebrew/opt/postgresql@17/bin/postgres")
CONFIG_PATH = Path("/Volumes/IMSG-Data/imsgindex/private/config.yaml")
LABEL = f"{LABEL_PREFIX}search-page"


@pytest.fixture
def config(config_dict_factory: ConfigDictFactory) -> Config:
    raw = config_dict_factory(**{"paths.data_root": DATA_ROOT})
    raw["search_page"] = {"enabled": True}
    return load_config_dict(raw)


def _render(config: Config, **kwargs: object) -> dict[str, bytes]:
    return render_agent_plists(
        config,
        imsg_binary=IMSG,
        postgres_binary=POSTGRES,
        cloudflared_binary=None,
        config_path=CONFIG_PATH,
        interpreter=INTERPRETER,
        supervisor=SUPERVISOR,
        **kwargs,  # type: ignore[arg-type]
    )


def test_never_in_the_default_set_even_when_enabled(config: Config) -> None:
    rendered = _render(config, only=[n for n in AGENT_NAMES if n != "tunnel"])
    assert LABEL not in rendered
    assert "search-page" in OPTIONAL_AGENT_NAMES


def test_rendered_only_on_request_and_supervised(config: Config) -> None:
    rendered = _render(config, only=["search-page"])
    assert set(rendered) == {LABEL}
    plist = plistlib.loads(rendered[LABEL])
    assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True
    assert plist["ProcessType"] == "Interactive"
    assert plist["ThrottleInterval"] == 60
    arguments = plist["ProgramArguments"]
    cut = arguments.index("--")
    supervisor, command = arguments[:cut], arguments[cut + 1 :]
    assert supervisor[:2] == [str(INTERPRETER), str(SUPERVISOR)]
    assert supervisor[supervisor.index("--service") + 1] == "search-page"
    assert supervisor[supervisor.index("--config") + 1] == str(CONFIG_PATH)
    assert "--wait-for-postgres" in supervisor
    # The database password is named, with the 0600 file that holds it.
    assert "IMSG_TEST_PG_PASSWORD=private/env/IMSG_TEST_PG_PASSWORD" in supervisor
    assert command == [str(IMSG), "search-page", "serve", "--config", str(CONFIG_PATH)]
    # Logs go to the volume through the supervisor, never through launchd.
    assert plist["StandardOutPath"] == "/dev/null" and plist["StandardErrorPath"] == "/dev/null"


def test_holds_no_secret_value(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "hunter2-not-a-real-password")
    raw = _render(config, only=["search-page"])[LABEL]
    assert b"hunter2" not in raw
    assert b"Bearer" not in raw


def test_unknown_agent_still_refused(config: Config) -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        _render(config, only=["search-pages"])


def test_install_agents_renders_it_into_a_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_dict_factory: ConfigDictFactory
) -> None:
    import shutil

    import yaml

    from imsg.cli import app

    raw = config_dict_factory()
    raw["search_page"] = {"enabled": True}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/example/bin/{name}")
    dest = tmp_path / "agents"
    result = CliRunner().invoke(
        app,
        ["install-agents", "--config", str(config_file), "--dest", str(dest), "--only", "search-page"],
    )
    assert result.exit_code == 0, result.output
    assert [p.name for p in dest.iterdir()] == [f"{LABEL}.plist"]
    assert "nothing was loaded" in result.output
