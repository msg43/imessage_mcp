#!/usr/bin/env python3
"""Time an authorized request to the public MCP server while it is flooded
with token-less requests, and count what the flood wrote to the audit
tables. Also measures what the tool result's duplicated payload costs.

This is the QA review's verification for its first high finding
(2026-09-24): "Flood a scratch `imsg mcp public` (fake introspector) with
token-less requests at 200 per second while timing an authorized call;
the authorized call's latency must not rise, and the row count must not
track the flood."

What runs is the real transport: `build_public_asgi_app` with a real
`PublicAuthGate` (a fake introspector that knows one owner token), a
`PostgresAuditSink` against a scratch database, and a real uvicorn on a
loopback port, wired the way `imsg mcp public` wires them. The script
only uses interfaces that existed before 2026-09-24's changes, and
detects the newer ones (the pooled audit sink, the refusal tally, the
uvicorn options), so the same script measures the code before and after:

    PYTHONPATH=<checkout>/src python scripts/bench_public_flood.py \\
        --dsn postgresql://postgres@127.0.0.1:55734/imsgindex_bench

Requests carry `X-Forwarded-For`, as Tailscale's proxy adds it, so each
flood thread looks like its own internet address and the authorized
requests like the owner's client. The authorized request is an MCP
`initialize` with the owner's token: it goes through the whole guard and
the SDK without needing a session or the models. Nothing here is real:
the token, the subject and every address are fictional (RFC 5737).

`--payload` measures the tool result instead: ten search results of
about 2,000 characters, serialized the way the SDK sends them, with the
payload in both blocks (as shipped), in `structuredContent` only, and in
the text block only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import socket
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

OWNER_SUB = "900000000000000000009"
CLIENT_ID = "333333333333-fictional.apps.example"
OWNER_TOKEN = "fictional-bench-owner-token"
HOST = "mcp.fictional.example"
OWNER_ADDRESS = "198.51.100.20"

INITIALIZE = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "bench", "version": "0"},
        },
    }
).encode()


def nearest_rank(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


class Introspector:
    def introspect(self, token: str) -> Any:
        from imsg.mcp.auth import TokenIntrospection
        from imsg.mcp.errors import TokenInvalidError

        if token != OWNER_TOKEN:
            raise TokenInvalidError("unknown token")
        return TokenIntrospection(
            subject=OWNER_SUB,
            audience=CLIENT_ID,
            authorized_party=None,
            scopes=frozenset({"openid"}),
            expires_in_seconds=3600,
        )


class NoService:
    """`initialize` never reaches the retrieval service."""


def build_server(dsn: str) -> tuple[Any, dict[str, Any], list[Any]]:
    """The app, the uvicorn options, and things to stop afterwards — wired
    as `imsg mcp public` wires them in whichever code is on the path."""
    import psycopg

    from imsg.mcp import audit as audit_module
    from imsg.mcp.auth import PublicAuthGate
    from imsg.mcp.tools import public_server
    from imsg.retrieval.background_warm_up import BackgroundWarmUp
    from imsg.retrieval.model_thread import ModelThread

    stop: list[Any] = []
    gate_kwargs: dict[str, Any] = {}
    if hasattr(audit_module, "RejectionTally"):
        from imsg.db.pool import postgres_pool

        pool = postgres_pool(lambda: psycopg.connect(dsn, autocommit=True), max_size=2, name="audit")
        sink = audit_module.PostgresAuditSink(pool.lease)
        tally = audit_module.RejectionTally()
        writer = audit_module.RejectionTallyWriter(tally, sink, interval_seconds=60)
        writer.start()
        gate_kwargs["rejection_tally"] = tally
        stop.extend([writer, pool])
        wiring = "pooled audit sink, refusal tally"
    else:
        sink = audit_module.PostgresAuditSink(lambda: psycopg.connect(dsn, autocommit=True))
        wiring = "a connection per audit row"
    gate = PublicAuthGate(
        client_id=CLIENT_ID,
        owner_subject=OWNER_SUB,
        introspector=Introspector(),
        audit=sink,
        # The timed requests are the owner's, several hundred a minute:
        # the per-subject limit (60 by default) would refuse most of them
        # and measure the limiter instead of the transport.
        rate_limit_per_minute=100_000,
        **gate_kwargs,
    )
    thread = ModelThread(name="bench-flood")
    warm_up = BackgroundWarmUp([], model_thread=thread, log=lambda line: None)
    warm_up.start()
    warm_up.wait(timeout=10)
    stop.append(thread)
    server = public_server.PublicMcpServer(
        service=NoService(),  # type: ignore[arg-type]
        gate=gate,
        scope="full",
        warm_up=warm_up,
    )
    app = public_server.build_public_asgi_app(
        server,
        allowed_hosts=[HOST],
        allowed_origins=["https://assistant.fictional.example"],
        external_url=f"https://{HOST}/mcp",
    )
    options_for = getattr(public_server, "public_uvicorn_options", None)
    options = (
        options_for(host="127.0.0.1", port=0)
        if options_for is not None
        else {"host": "127.0.0.1", "port": 0, "log_level": "info"}
    )
    options["log_level"] = "warning"
    print(f"wiring: {wiring}; uvicorn options: {sorted(options)}", flush=True)
    return app, options, stop


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(conn: http.client.HTTPConnection, *, address: str, token: str | None) -> int:
    headers = {
        "Host": HOST,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "X-Forwarded-For": address,
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    conn.request("POST", "/mcp", body=INITIALIZE, headers=headers)
    response = conn.getresponse()
    response.read()
    return response.status


def time_authorized(port: int, count: int) -> list[float]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    seconds: list[float] = []
    try:
        for _ in range(count):
            started = time.perf_counter()
            status = post(conn, address=OWNER_ADDRESS, token=OWNER_TOKEN)
            seconds.append(time.perf_counter() - started)
            if status != 200:
                raise SystemExit(f"authorized request answered {status}")
            time.sleep(0.01)
    finally:
        conn.close()
    return seconds


def flood(port: int, rate: float, stop: threading.Event, statuses: Counter[int], threads: int) -> list[threading.Thread]:
    interval = threads / rate

    def worker(index: int) -> None:
        address = f"203.0.113.{index + 1}"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        next_at = time.perf_counter()
        while not stop.is_set():
            try:
                statuses[post(conn, address=address, token=None)] += 1
            except (OSError, http.client.HTTPException):
                statuses[-1] += 1
                conn.close()
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            next_at += interval
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        conn.close()

    workers = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    for w in workers:
        w.start()
    return workers


def table_counts(dsn: str) -> dict[str, int]:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM mcp_audit")
        audit = int(cur.fetchone()[0])  # type: ignore[index]
        cur.execute("SELECT to_regclass('public.mcp_audit_rollup') IS NOT NULL")
        has_rollup = bool(cur.fetchone()[0])  # type: ignore[index]
        rollup = 0
        if has_rollup:
            cur.execute("SELECT coalesce(sum(request_count), 0) FROM mcp_audit_rollup")
            rollup = int(cur.fetchone()[0])  # type: ignore[index]
    return {"mcp_audit_rows": audit, "rollup_requests": rollup}


def summary(label: str, seconds: Sequence[float]) -> dict[str, float]:
    row = {
        "n": len(seconds),
        "p50_ms": 1000 * statistics.median(seconds),
        "p95_ms": 1000 * nearest_rank(seconds, 0.95),
        "max_ms": 1000 * max(seconds),
    }
    print(
        f"{label:28s} n={row['n']:4d}  p50 {row['p50_ms']:7.1f} ms  "
        f"p95 {row['p95_ms']:7.1f} ms  max {row['max_ms']:7.1f} ms",
        flush=True,
    )
    return row


def flood_main(args: argparse.Namespace) -> dict[str, Any]:
    import uvicorn

    app, options, stop_after = build_server(args.dsn)
    port = free_port()
    options["port"] = port
    server = uvicorn.Server(uvicorn.Config(app, **options))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise SystemExit("uvicorn did not start")
        time.sleep(0.02)
    report: dict[str, Any] = {"rate_per_second": args.rate, "flood_threads": args.threads}
    try:
        time_authorized(port, 20)  # warm: the token is checked once and cached
        report["quiet"] = summary("authorized, no flood", time_authorized(port, args.requests))
        before = table_counts(args.dsn)
        statuses: Counter[int] = Counter()
        stop = threading.Event()
        started = time.perf_counter()
        workers = flood(port, args.rate, stop, statuses, args.threads)
        time.sleep(2)
        report["flooded"] = summary(
            f"authorized, {args.rate:.0f}/s flood", time_authorized(port, args.requests)
        )
        stop.set()
        for w in workers:
            w.join(timeout=30)
        flood_seconds = time.perf_counter() - started
        time.sleep(0.5)
        # Let the refusal tally write its counts (new code), as it would
        # within a minute on its own.
        for item in stop_after:
            if hasattr(item, "write_now"):
                item.write_now()
        after = table_counts(args.dsn)
        sent = sum(statuses.values())
        report["flood"] = {
            "requests": sent,
            "achieved_rate_per_second": sent / flood_seconds,
            "statuses": {str(k): v for k, v in sorted(statuses.items())},
            "mcp_audit_rows_written": after["mcp_audit_rows"] - before["mcp_audit_rows"],
            "requests_counted_in_rollup": after["rollup_requests"] - before["rollup_requests"],
        }
        print(
            f"flood: {sent} requests at {sent / flood_seconds:.0f}/s, statuses "
            f"{dict(sorted(statuses.items()))}; mcp_audit rows written "
            f"{report['flood']['mcp_audit_rows_written']}; requests counted in "
            f"mcp_audit_rollup {report['flood']['requests_counted_in_rollup']}",
            flush=True,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        for item in stop_after:
            closer = getattr(item, "stop", None) or getattr(item, "close", None)
            if callable(closer):
                closer()
    return report


# --------------------------------------------------------------------------
# --payload
# --------------------------------------------------------------------------


def payload_main(repeats: int) -> dict[str, Any]:
    import random

    import mcp.types as types

    rng = random.Random(924)
    words = (
        "ferry", "harbor", "kite", "festival", "lantern", "parade",
        "picnic", "ridge", "orchard", "cider", "market",
    )
    payload = {
        "results": [
            {
                "segment_key": f"seg_{i:064x}",
                "thread_key": f"thread_{i:060x}",
                "chat": {"thread_key": f"thread_{i:060x}", "kind": "dm", "display_name": None},
                "people": ["alice", "owner"],
                "started_at": "2024-05-01T12:00:00-04:00",
                "ended_at": "2024-05-01T12:10:00-04:00",
                "message_count": 12,
                "has_attachments": False,
                "score": 0.5 - i / 100,
                "text": "\n".join(
                    f"[2024-05-01 12:0{j % 10}] alice: "
                    + " ".join(rng.choice(words) for _ in range(20))
                    + " \U0001f6b2"
                    for j in range(14)
                ),
                "untrusted_content": True,
            }
            for i in range(10)
        ],
        "candidate_lists": {"segment_fts": 100, "segment_vector": 100},
        "scan_cap_reached": False,
    }

    def serialize(result: types.CallToolResult) -> bytes:
        response = types.JSONRPCResponse(
            jsonrpc="2.0",
            id=1,
            result=result.model_dump(by_alias=True, mode="json", exclude_none=True),
        )
        return response.model_dump_json(by_alias=True, exclude_none=True).encode()

    shapes = {
        "both blocks (shipped)": lambda: types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, default=str))],
            structured_content=payload,
            is_error=False,
        ),
        "structuredContent only": lambda: types.CallToolResult(
            content=[], structured_content=payload, is_error=False
        ),
        "text block only": lambda: types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, default=str))],
            is_error=False,
        ),
    }
    report: dict[str, Any] = {}
    for label, build in shapes.items():
        body = serialize(build())
        seconds = []
        for _ in range(repeats):
            started = time.perf_counter()
            serialize(build())
            seconds.append(time.perf_counter() - started)
        report[label] = {
            "bytes": len(body),
            "build_and_serialize_p50_ms": 1000 * statistics.median(seconds),
        }
        print(
            f"{label:24s} {len(body):8d} bytes   build+serialize p50 "
            f"{1000 * statistics.median(seconds):6.2f} ms",
            flush=True,
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsn", help="scratch database with mcp_audit (flood mode)")
    parser.add_argument("--rate", type=float, default=200.0, help="flood requests per second")
    parser.add_argument("--threads", type=int, default=20, help="flood client threads")
    parser.add_argument("--requests", type=int, default=200, help="authorized requests timed")
    parser.add_argument("--payload", action="store_true", help="measure the payload instead")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--report-json", help="write the numbers here")
    args = parser.parse_args(argv)
    if args.payload:
        report = payload_main(args.repeats)
    else:
        if not args.dsn:
            parser.error("--dsn is required for the flood")
        report = flood_main(args)
    if args.report_json:
        with open(args.report_json, "w") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
