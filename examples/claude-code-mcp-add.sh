#!/usr/bin/env bash
# examples/claude-code-mcp-add.sh — register the local imessage-index MCP
# server with Claude Code's `claude mcp add` (as opposed to Claude
# Desktop, which reads examples/claude_desktop_config.json instead).
#
# Usage:
#   cd /path/to/your/imessage-index/checkout
#   ./examples/claude-code-mcp-add.sh
#
# Run this from inside the repo checkout you built `imsg` in — it
# derives the absolute paths to the venv binary and your config from
# the current directory, so it works for any username or install
# location without editing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMSG_BIN="$REPO_ROOT/.venv/bin/imsg"
CONFIG_PATH="${IMSG_CONFIG:-$REPO_ROOT/config.yaml}"

if [[ ! -x "$IMSG_BIN" ]]; then
  echo "error: $IMSG_BIN not found or not executable." >&2
  echo "Run 'uv sync --extra models' from $REPO_ROOT first." >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "error: config file not found at $CONFIG_PATH." >&2
  echo "Copy examples/config.single-mac.yaml to config.yaml (or set" >&2
  echo "IMSG_CONFIG) and fill in your own paths first." >&2
  exit 1
fi

echo "Registering imsg-local with Claude Code..."
echo "  command: $IMSG_BIN"
echo "  args:    mcp local"
echo "  env:     IMSG_CONFIG=$CONFIG_PATH"

claude mcp add \
  --transport stdio \
  --scope user \
  -e "IMSG_CONFIG=$CONFIG_PATH" \
  imsg-local \
  -- "$IMSG_BIN" mcp local

echo "Done. Verify with: claude mcp list"
