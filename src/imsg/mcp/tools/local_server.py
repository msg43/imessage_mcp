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

**Loading on the first call.** The server answers `initialize` and
`tools/list` at once and, by default, loads no model until the first
retrieval call (`mcp.local.warm_at_start`, off: most client sessions,
each with its own server, never search, and a server that loaded at
start held a full model set regardless). That call starts the warm-up
(`imsg.retrieval.background_warm_up`), which asks the memory admission
check first, and waits for it — without blocking the event loop, so
pings and other requests are still answered — for at most
`TOOL_CALL_WARM_UP_WAIT_SECONDS`, then gets a `WARMING_UP` tool error
with an estimate of the seconds remaining. With `warm_at_start` on, the
warm-up starts as soon as the stdio transport has taken over stdout (see
`run_local_server`), as it always did before. A warm-up that failed turns
every tool call into a `WARM_UP_FAILED` error naming the cause. Both are
ordinary SPEC §10.1-style tool errors, audited like any other; arguments
that fail the schema are rejected at once, without waiting. So is
`check_permissions`, which is diagnostics rather than retrieval: it
answers straight away and carries the warm-up's own state
(`warm_up_report`), so the operator asking "what is this server doing?"
gets an answer while it is doing it.

**Unloading when idle.** With an `IdleModelUnloader`
(`imsg.retrieval.idle_unload`, `mcp.idle_unload_seconds`), the models are
dropped after that long with no retrieval tool call, and the warm-up's
state becomes `unloaded`. The next retrieval call starts the reload and
waits for it exactly as a call during the first warm-up does — up to
`TOOL_CALL_WARM_UP_WAIT_SECONDS`, then `WARMING_UP` with an estimate. Every
retrieval call is registered with the unloader for its whole span, the
wait included, so nothing unloads while one is in flight.

**Loading only when the host has room.** Every load — at startup and at
each reload — first asks `imsg.memory_admission` whether the host has the
memory for it. Refused, nothing loads and the warm-up's state becomes
`memory_busy`; a retrieval call is answered at once with the retryable
`WARMING_UP` code and a "host memory busy" message naming the numbers
(`HostMemoryBusyError`), and the next call asks again. Background work
gives way to a refused server: it starts no model load and stops after
its current unit of work, until the server loads or no call has tried
the load for two minutes (`imsg.memory_admission.LiveServerAdmission`).
The unloader also drops loaded models early when the kernel reports
critical memory pressure (`memory.local_server_release_at`).

**Exiting when the client goes.** The SDK's stdio transport ends
`server.run` when stdin reaches end-of-file, which is what an SSH client
that disconnects produces, and nothing here keeps the process alive after
that: the model thread and the idle watchdog are daemon threads, and a
call still waiting for the warm-up is cancelled with the rest of the
server's tasks. (`tests/test_mcp_idle_unload.py` closes a child server's
stdin during a held warm-up and while a call waits, and requires the exit.)
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

import anyio
import jsonschema
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from imsg.mcp.tools import handlers
from imsg.mcp.tools.dispatch import call_tool
from imsg.mcp.tools.schemas import TOOL_DEFINITIONS, TOOL_DEFINITIONS_BY_NAME, ToolDefinition
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.background_warm_up import WarmUpPhase
from imsg.retrieval.errors import InvalidArgumentError

if TYPE_CHECKING:
    import psycopg
    from mcp.server import ServerRequestContext

    from imsg.config.schema import Config
    from imsg.mcp.audit import AuditSink
    from imsg.retrieval.access import AccessContext
    from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpStatus
    from imsg.retrieval.idle_unload import IdleModelUnloader
    from imsg.retrieval.service import RetrievalService



SERVER_NAME = "imsg-local"

TOOL_CALL_WARM_UP_WAIT_SECONDS = 90.0
"""How long a tool call that arrives during warm-up waits for it before
answering `WARMING_UP`.

Chosen from the limits Claude Code documents for a stdio server's tool
calls (code.claude.com/docs/en/mcp and /env-vars, read 2026-09-17), none
of which 90 s approaches: the wall-clock limit `MCP_TOOL_TIMEOUT`
defaults to 100000000 ms (about 28 hours), stdio servers have no
per-request timer, and a call that sends no response or progress is
aborted only after `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`, 1800000 ms
(30 minutes) by default for stdio. The number that does bind is softer:
a main-conversation MCP call still running after two minutes is moved to
a background task. Waiting at most 90 s leaves the query itself room to
finish inside those two minutes (a warm query took 2.1-2.3 s on the M2
Ultra host, 2026-09-17). The warm-up took 77-185 s there the same day, so
a call made the moment the server starts either gets its answer or, when
loading runs long, a `WARMING_UP` error to retry."""

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


WARM_UP_POLL_SECONDS = 0.1
"""How often a waiting tool call re-checks the warm-up."""


def warm_up_report(status: WarmUpStatus) -> dict[str, Any]:
    """The warm-up as `check_permissions` reports it: which phase the
    server is in (`not_started`, `warming`, `ready`, `failed`, `unloaded`,
    `memory_busy`), what it is loading right now, how many steps are done,
    an estimate of the seconds left, and — when it failed — the cause, or
    — when the host had no memory for the models — why
    (`memory_detail`). Every retrieval tool is unavailable until this says
    `ready`, so this is the field that explains a `WARMING_UP` or
    `WARM_UP_FAILED` answer from any of them."""
    return {
        "state": status.phase.value,
        "loading": status.loading,
        "steps_done": status.steps_done,
        "steps_total": status.steps_total,
        "seconds_remaining": (
            None if status.settled else max(1, math.ceil(status.seconds_remaining))
        ),
        "failure": status.failure,
        "memory_detail": status.detail,
    }


def _schema_error(definition: ToolDefinition, arguments: dict[str, Any]) -> str | None:
    """The first JSON Schema violation in `arguments`, or `None`. The
    violation is raised as `InvalidArgumentError` inside the handler that
    `imsg.mcp.tools.dispatch.call_tool` runs — one error-formatting/audit
    code path for both "malformed arguments" and "well-formed but
    semantically invalid" (SPEC §10.1's `INVALID_ARGUMENT` covers both) —
    but it is found before the warm-up wait, so a malformed call is
    answered at once."""
    try:
        jsonschema.validate(arguments, definition.input_schema)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


@dataclass(slots=True)
class LocalMcpServer:
    """Owns the retrieval service, the audit sink, the model warm-up
    every tool call is gated on, and (for `check_permissions` only, which
    is diagnostics rather than retrieval) config plus a raw connection —
    and exposes the two callbacks `mcp.server.lowlevel.Server` needs."""

    service: RetrievalService
    audit: AuditSink
    config: Config
    conn: psycopg.Connection
    warm_up: BackgroundWarmUp
    warm_up_wait_seconds: float = TOOL_CALL_WARM_UP_WAIT_SECONDS
    idle_unloader: IdleModelUnloader | None = None
    """Drops the models after `mcp.idle_unload_seconds` with no retrieval
    call and reloads them on the next one; `None` keeps them loaded for
    the life of the process."""
    warm_at_start: bool = False
    """Start the warm-up as soon as the server is serving
    (`mcp.local.warm_at_start`); off, the first retrieval call starts it."""

    def begin_serving(self) -> None:
        """What happens once the transport is up: the warm-up when
        `warm_at_start`, and the watchdog that unloads idle models and
        releases them under memory pressure."""
        if self.warm_at_start:
            self.warm_up.start()
        if self.idle_unloader is not None:
            self.idle_unloader.start_watchdog()

    def _load_on_first_call(self) -> None:
        """Start the first load (`warm_at_start` off). A no-op once a load
        has begun, finished or failed; a reload after an unload, or a retry
        after a refusal, is the idle unloader's (`IdleModelUnloader.call`)."""
        if self.warm_up.status().phase is WarmUpPhase.NOT_STARTED:
            self.warm_up.start()

    def _retrieval_call(self) -> contextlib.AbstractContextManager[None]:
        """The span during which this retrieval call holds the models:
        registered with the idle unloader, when there is one."""
        if self.idle_unloader is None:
            return contextlib.nullcontext()
        return self.idle_unloader.call()

    async def on_list_tools(
        self,
        context: ServerRequestContext[None],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        del context, params  # no pagination, no session state needed
        return types.ListToolsResult(tools=[_to_mcp_tool(d) for d in TOOL_DEFINITIONS])

    async def _settled_warm_up(self) -> WarmUpStatus:
        """The warm-up's status once it is ready or failed, or after this
        call has waited `warm_up_wait_seconds` for it. Polls with
        `anyio.sleep`, so the event loop keeps answering other requests
        meanwhile and a cancelled call stops waiting at once."""
        deadline = anyio.current_time() + self.warm_up_wait_seconds
        while True:
            status = self.warm_up.status()
            left = deadline - anyio.current_time()
            if status.settled or left <= 0:
                return status
            await anyio.sleep(min(WARM_UP_POLL_SECONDS, left))

    async def on_call_tool(
        self, context: ServerRequestContext[None], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        del context
        started = monotonic()
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

        handler: Callable[[], dict[str, Any]]
        schema_error = _schema_error(definition, arguments)
        if schema_error is not None:
            violation: str = schema_error

            def handler() -> dict[str, Any]:
                raise InvalidArgumentError(violation)
        elif name == "check_permissions":
            # Diagnostics, not retrieval: it needs no model, and it is how
            # an operator asks what the server is doing — including while
            # it is still loading. Making it wait for the warm-up (or
            # answer `WARMING_UP`) would withhold the answer exactly when
            # the question is being asked, so it answers now and says how
            # the warm-up is going.
            status = self.warm_up.status()

            def handler() -> dict[str, Any]:
                payload = handlers.check_permissions(config=self.config, conn=self.conn)
                payload["warm_up"] = warm_up_report(status)
                return payload
        # Retrieval: the handler is defined below, inside the span that
        # keeps the idle unloader away from the models.
        with contextlib.ExitStack() as holding:
            if schema_error is None and name != "check_permissions":
                tool_fn = _RETRIEVAL_HANDLERS[name]
                holding.enter_context(self._retrieval_call())
                self._load_on_first_call()
                warm_up = await self._settled_warm_up()

                def handler() -> dict[str, Any]:
                    warm_up.raise_unless_ready(wait_bound_seconds=self.warm_up_wait_seconds)
                    return tool_fn(self.service, LOCAL_FULL_ACCESS, arguments)

            result = call_tool(
                self.audit, tool=name, params=arguments, handler=handler, started=started
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
    §10.3: stdio; binds nothing, no network listener). The models load in
    the background — at once with `warm_at_start`, else on the first
    retrieval call.

    Either way the warm-up starts inside `stdio_server()` on purpose. While that
    context is open the SDK writes protocol frames to a private duplicate
    of stdout and points file descriptor 1 at stderr, so anything the
    model libraries print while loading lands on stderr instead of in the
    middle of a JSON-RPC frame. PE-Core, for one, logs through an
    unconfigured structlog, whose default output is stdout. Started any
    earlier, those writes would reach the client. For the same reason
    Python's stdout buffer is flushed before the context closes and
    points file descriptor 1 back at the client: a `print` still sitting
    in it belongs on stderr too."""
    server = local.build_server()
    async with stdio_server() as (read_stream, write_stream):
        local.begin_serving()
        try:
            await server.run(read_stream, write_stream, server.create_initialization_options())
        finally:
            if local.idle_unloader is not None:
                local.idle_unloader.stop()
            with contextlib.suppress(OSError, ValueError):
                sys.stdout.flush()


__all__ = [
    "TOOL_CALL_WARM_UP_WAIT_SECONDS",
    "LocalMcpServer",
    "run_local_server",
    "warm_up_report",
]
