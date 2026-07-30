"""The local MCP surface (SPEC §10.3): `imsg mcp local` — stdio,
reached from the Studio as `ssh mini imsg mcp local` over the tailnet,
full corpus scope. Registers exactly the five in-scope tools
(`imsg.mcp.tools.schemas.TOOL_DEFINITIONS`) against the official `mcp`
Python SDK.

Uses the low-level `mcp.server.lowlevel.Server` API (`on_list_tools`/
`on_call_tool` callbacks) rather than the higher-level `MCPServer`
convenience wrapper: the latter's `add_tool` always derives
`inputSchema` from a Python function's signature via pydantic, which
cannot reproduce SPEC-exact constraints (`minLength`, `maxItems`,
`"format": "date"`, `additionalProperties: false`, ...) without a
bespoke pydantic model per tool — handing the SDK the exact
hand-authored dict from `imsg.mcp.tools.schemas` directly is simpler
and more obviously faithful to SPEC §10.2.

Because a hand-authored `inputSchema` is not automatically enforced by
the low-level SDK the way a derived one would be, arguments are
explicitly validated against it here (via `jsonschema`) before a
handler ever runs — a schema violation becomes an ordinary SPEC §10.1
`INVALID_ARGUMENT` tool error (audited like any other call), never a
raw exception.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jsonschema
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from imsg.mcp.tools import handlers
from imsg.mcp.tools.dispatch import call_tool
from imsg.mcp.tools.schemas import TOOL_DEFINITIONS, TOOL_DEFINITIONS_BY_NAME, ToolDefinition
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.errors import InvalidArgumentError

if TYPE_CHECKING:
    import psycopg
    from mcp.server import ServerRequestContext

    from imsg.config.schema import Config
    from imsg.mcp.audit import AuditSink
    from imsg.retrieval.access import AccessContext
    from imsg.retrieval.service import RetrievalService

SERVER_NAME = "imsg-local"

_RETRIEVAL_HANDLERS: dict[
    str, Callable[[RetrievalService, AccessContext, dict[str, Any]], dict[str, Any]]
] = {
    "search_messages": handlers.search_messages,
    "get_conversation": handlers.get_conversation,
    "list_people": handlers.list_people,
    "get_attachment_text": handlers.get_attachment_text,
}


def _to_mcp_tool(definition: ToolDefinition) -> types.Tool:
    return types.Tool(
        name=definition.name,
        description=definition.description,
        input_schema=definition.input_schema,
        annotations=types.ToolAnnotations(
            read_only_hint=definition.annotations["readOnlyHint"],
            destructive_hint=definition.annotations["destructiveHint"],
            idempotent_hint=definition.annotations["idempotentHint"],
            open_world_hint=definition.annotations["openWorldHint"],
        ),
    )


def _validated(
    definition: ToolDefinition, arguments: dict[str, Any], real_handler: Callable[[], dict[str, Any]]
) -> Callable[[], dict[str, Any]]:
    """Wrap `real_handler` so schema validation happens inside the same
    `RetrievalError`-catching path `imsg.mcp.tools.dispatch.call_tool`
    already provides — one error-formatting/audit code path for both
    "malformed arguments" and "well-formed but semantically invalid"
    (SPEC §10.1's `INVALID_ARGUMENT` covers both)."""

    def _run() -> dict[str, Any]:
        try:
            jsonschema.validate(arguments, definition.input_schema)
        except jsonschema.ValidationError as exc:
            raise InvalidArgumentError(exc.message) from exc
        return real_handler()

    return _run


@dataclass(slots=True)
class LocalMcpServer:
    """Owns the retrieval service, the audit sink, and (for
    `check_permissions` only, which is diagnostics rather than
    retrieval) config plus a raw connection — and exposes the two
    callbacks `mcp.server.lowlevel.Server` needs."""

    service: RetrievalService
    audit: AuditSink
    config: Config
    conn: psycopg.Connection

    async def on_list_tools(
        self,
        context: ServerRequestContext[None],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        del context, params  # no pagination, no session state needed
        return types.ListToolsResult(tools=[_to_mcp_tool(d) for d in TOOL_DEFINITIONS])

    async def on_call_tool(
        self, context: ServerRequestContext[None], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        del context
        name = params.name
        arguments = dict(params.arguments or {})

        definition = TOOL_DEFINITIONS_BY_NAME.get(name)
        if definition is None:
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text=f"INVALID_ARGUMENT\nunknown tool {name!r}")
                ],
                is_error=True,
            )

        real_handler: Callable[[], dict[str, Any]]
        if name == "check_permissions":

            def real_handler() -> dict[str, Any]:
                return handlers.check_permissions(config=self.config, conn=self.conn)
        else:
            tool_fn = _RETRIEVAL_HANDLERS[name]

            def real_handler() -> dict[str, Any]:
                return tool_fn(self.service, LOCAL_FULL_ACCESS, arguments)

        result = call_tool(
            self.audit,
            tool=name,
            params=arguments,
            handler=_validated(definition, arguments, real_handler),
        )

        if result.is_error:
            text = f"{result.error_code}\n{result.error_message}"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], is_error=True
            )

        payload = result.payload or {}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, default=str))],
            structured_content=payload,
            is_error=False,
        )

    def build_server(self) -> Server[None]:
        return Server(name=SERVER_NAME, on_list_tools=self.on_list_tools, on_call_tool=self.on_call_tool)


async def run_local_server(local: LocalMcpServer) -> None:
    """Run the stdio transport until the client disconnects (SPEC
    §10.3: stdio; binds nothing, no network listener)."""
    server = local.build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


__all__ = ["LocalMcpServer", "run_local_server"]
