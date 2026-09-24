"""A real `imsg mcp local`-style stdio server whose model warm-up is held
until a file appears — the child process for the stdio test in
`tests/test_mcp_local_server_warm_up.py`.

Usage: python _run_local_mcp_server_with_held_warm_up.py RELEASE_FILE WAIT_SECONDS [IDLE_SECONDS] [--lazy]

With IDLE_SECONDS, the server also runs an idle unloader with its
watchdog thread started, as `imsg mcp local` does — so a test can show
that the watchdog does not keep a disconnected server alive either.

The warm-up starts as soon as the server is serving (`warm_at_start`),
which is what the tests of a held warm-up need; with `--lazy` it starts
on the first retrieval call instead, as `imsg mcp local` does by default.

Runs `imsg.mcp.tools.local_server.run_local_server` exactly as the CLI
does, but over a fake retrieval service (no database) and a single fake
warm-up step. While held, that step writes to stdout three ways a model
library might — `print`, an unconfigured structlog logger, and a raw
write to file descriptor 1 — none of which may reach the protocol stream.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import anyio
import structlog

from imsg.mcp.audit import MemoryAuditSink
from imsg.mcp.tools.local_server import LocalMcpServer, run_local_server
from imsg.retrieval.background_warm_up import BackgroundWarmUp, WarmUpStep
from imsg.retrieval.idle_unload import IdleModelUnloader
from imsg.retrieval.model_thread import ModelThread
from imsg.retrieval.service import SearchMessagesResult


class _FakeRetrievalService:
    def search_messages(self, context: Any, **kwargs: Any) -> SearchMessagesResult:
        return SearchMessagesResult(
            results=[], candidate_lists={"segment_fts": 0}, scan_cap_reached=False
        )


def main() -> None:
    lazy = "--lazy" in sys.argv
    args = [arg for arg in sys.argv if arg != "--lazy"]
    release_file = Path(args[1])
    wait_seconds = float(args[2])

    def held_step() -> None:
        print("stray print from a loading model")
        structlog.get_logger("held-warm-up").info("stray structlog line from a loading model")
        os.write(1, b"stray write to fd 1 from a loading model\n")
        while not release_file.exists():
            time.sleep(0.01)

    model_thread = ModelThread()
    warm_up = BackgroundWarmUp(
        [WarmUpStep("text embedder", 30.0, held_step)], model_thread=model_thread
    )
    idle_unloader = None
    if len(args) > 3:
        idle_unloader = IdleModelUnloader(
            warm_up=warm_up,
            model_thread=model_thread,
            unload=lambda: None,
            idle_seconds=float(args[3]),
        )
    local = LocalMcpServer(
        service=cast(Any, _FakeRetrievalService()),
        audit=MemoryAuditSink(),
        config=cast(Any, None),
        conn=cast(Any, None),
        warm_up=warm_up,
        warm_up_wait_seconds=wait_seconds,
        idle_unloader=idle_unloader,
        warm_at_start=not lazy,
    )
    anyio.run(run_local_server, local)


if __name__ == "__main__":
    main()
