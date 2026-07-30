"""The local MCP surface's tool implementations (SPEC §10.2, §10.3).

Layered underneath `imsg.mcp` (the public-surface security boundary,
untouched by this package — see that package's own docstring):

- `schemas` — hand-authored, spec-exact `inputSchema`/annotations for
  the five in-scope tools.
- `handlers` — pure adapters from parsed tool arguments to
  `imsg.retrieval.RetrievalService` calls (plus `check_permissions`,
  which is diagnostics rather than retrieval).
- `dispatch` — the SPEC §10.1 error model and `surface='local'` audit
  logging, reusing `imsg.mcp.audit` without modifying it.
- `local_server` — wires the above to the official `mcp` Python SDK's
  low-level `Server` over stdio (SPEC §10.3).
"""

from imsg.mcp.tools.dispatch import LOCAL_SUBJECT, ToolCallResult, call_tool
from imsg.mcp.tools.local_server import LocalMcpServer, run_local_server
from imsg.mcp.tools.schemas import TOOL_DEFINITIONS, TOOL_DEFINITIONS_BY_NAME, ToolDefinition

__all__ = [
    "LOCAL_SUBJECT",
    "TOOL_DEFINITIONS",
    "TOOL_DEFINITIONS_BY_NAME",
    "LocalMcpServer",
    "ToolCallResult",
    "ToolDefinition",
    "call_tool",
    "run_local_server",
]
