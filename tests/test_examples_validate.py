"""W2 acceptance: the example files a non-expert copies must actually work.

`examples/config.single-mac.yaml` must load through the real config
loader (`imsg.config.loader.load_config`) once its host-specific paths
are pointed at a throwaway tmp_path, the same way `tests/conftest.py`'s
`data_root`/`messages_dir` fixtures redirect them for every other
config test. `examples/claude_desktop_config.json` must be valid JSON
shaped the way Claude Desktop expects (an `mcpServers` object).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from imsg.config.loader import load_config

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_single_mac_example_config_loads(
    tmp_path: Path, data_root: Path, messages_dir: Path
) -> None:
    """The example, with only its host-specific paths swapped in, must
    pass real schema validation — same shape a copy-pasting user gets."""
    raw = yaml.safe_load((EXAMPLES_DIR / "config.single-mac.yaml").read_text())

    live_chat_db = messages_dir / "chat.db"
    live_chat_db.write_text("")

    raw["paths"]["data_root"] = str(data_root)
    raw["paths"]["live_chat_db"] = str(live_chat_db)
    raw["sync"]["sources"][0]["chat_db"] = str(live_chat_db)
    # The password field only needs to look like a valid secret
    # reference — load_config() parses and validates, it never resolves
    # a secret (that is assert_secrets_resolvable's job, exercised
    # elsewhere), so no real Keychain item is needed here.
    raw["database"]["password"] = "env:IMSG_TEST_PG_PASSWORD"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    cfg = load_config(config_path)

    assert cfg.paths.data_root == data_root
    assert cfg.paths.live_chat_db == live_chat_db
    assert cfg.database.dsn.endswith(":5433/imsgindex")
    assert cfg.mcp.public.enabled is False
    assert cfg.mcp.public.scope == "allowlist"
    assert cfg.models.backend == "real"


def test_single_mac_example_config_is_not_using_a_real_username(tmp_path: Path) -> None:
    """Guard against the example regressing into a real name/path: every
    path in the file is the fictional 'alice' persona (CLAUDE.md's
    public-safety rule), never left as a placeholder some other real
    name slipped into."""
    text = (EXAMPLES_DIR / "config.single-mac.yaml").read_text()
    assert "/Users/alice/" in text
    # Every home-directory path must be the persona's. Checking for "only
    # alice" rather than "not <some real name>" keeps real names out of
    # this file too.
    users = set(re.findall(r"/Users/([^/\s\"']+)", text))
    assert users == {"alice"}


def test_claude_desktop_config_is_valid_mcp_json() -> None:
    data = json.loads((EXAMPLES_DIR / "claude_desktop_config.json").read_text())

    assert "mcpServers" in data
    servers = data["mcpServers"]
    assert isinstance(servers, dict) and servers

    server = servers["imsg-local"]
    # Claude Desktop does not inherit the user's shell environment, so
    # both the binary and IMSG_CONFIG must be absolute paths — a bare
    # "imsg" on PATH or a relative config path would silently fail.
    assert Path(server["command"]).is_absolute()
    assert server["command"].endswith(".venv/bin/imsg")
    assert server["args"] == ["mcp", "local"]
    assert "env" in server and "IMSG_CONFIG" in server["env"]
    assert Path(server["env"]["IMSG_CONFIG"]).is_absolute()


def test_claude_code_mcp_add_script_is_valid_bash() -> None:
    script = EXAMPLES_DIR / "claude-code-mcp-add.sh"
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, cwd=_repo_root()
    )
    assert result.returncode == 0, result.stderr


def test_install_guide_imsg_commands_are_real(tmp_path: Path) -> None:
    """Every `imsg <cmd>` the guide tells a reader to run must be a real
    CLI command. This mirrors the acceptance check run by hand
    (`grep -oE ... docs/install-macos.md`) so a future edit that
    introduces a typo'd command fails CI, not just a manual re-check."""
    import re

    guide = (_repo_root() / "docs" / "install-macos.md").read_text()
    pattern = re.compile(r"(?:uv run )?imsg [a-z-]+(?: [a-z-]+)?")
    commands = sorted({m.group(0).removeprefix("uv run ").strip() for m in pattern.finditer(guide)})
    assert commands, "expected to find at least one 'imsg <cmd>' invocation in the guide"

    for command in commands:
        parts = command.split()
        assert parts[0] == "imsg"
        rest = parts[1:]
        args = rest if rest and rest[-1] == "--help" else [*rest, "--help"]
        result = subprocess.run(
            ["uv", "run", "imsg", *args],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
        )
        assert result.returncode == 0, (
            f"'uv run imsg {' '.join(args)}' exited {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
