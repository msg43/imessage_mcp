"""`imsg search-page serve`: wire the page's dependencies and listen.

**Listeners.** One socket per `search_page.listen` entry — loopback plus
the mini's local-network address — each an IP literal that config
validation has already restricted to loopback or private ranges (never a
wildcard, never a Tailscale address). Startup also refuses a port that is
the public MCP server's (`mcp.public.bind`) or the model API's, so the
page can never be put where the Funnel forwards.

**The Tailscale-private alternative** (the owner's own devices only):
listen on loopback only, add the tailnet name to `allowed_hosts` and
`https_hosts`, and run `tailscale serve --bg --https=8443
http://127.0.0.1:8710` on the mini. Tailscale Serve is reachable only
from the tailnet and terminates HTTPS, so cookies are `Secure`. Never
use `tailscale funnel` for this port: Funnel publishes to the internet.

**Memory (D10.5).** This process loads no model. Semantic search asks
the public MCP server's warm models through the internal model API;
thumbnails and conversions run in short-lived, sandboxed child processes.
"""

from __future__ import annotations

import contextlib
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import apsw
import psycopg
import uvicorn

from imsg.db.connection import connect
from imsg.db.fingerprint import verify_data_directory
from imsg.embed.fts.schema import assert_schema_current
from imsg.search_page.app import AppDeps, ConnectionPool, FtsReaders, build_app
from imsg.search_page.auth import LoginGuard, PasswordFile, SessionStore
from imsg.search_page.errors import SearchPageStartupError, SecretFileError
from imsg.search_page.media import MediaConverter
from imsg.search_page.model_api_client import ModelApiClient
from imsg.search_page.search import SearchSettings
from imsg.search_page.secret_files import data_file, ensure_private_dir

if TYPE_CHECKING:
    from imsg.config.schema import Config

FTS_BUSY_TIMEOUT_MS = 2000


def fts_path(cfg: Config) -> Path:
    return cfg.paths.data_root / "fts" / "fts.db"


def _public_port(cfg: Config) -> int | None:
    _host, _, port = cfg.mcp.public.bind.rpartition(":")
    return int(port) if port.isdigit() else None


def check_ports(cfg: Config) -> list[tuple[str, int]]:
    page = cfg.search_page
    addresses = page.listen_addresses()
    forbidden = {p for p in (_public_port(cfg), page.model_api.port) if p is not None}
    for host, port in addresses:
        if port in forbidden:
            raise SearchPageStartupError(
                f"search_page.listen {host}:{port} uses the public MCP server's or the model "
                f"API's port; the page must have its own port"
            )
    return addresses


def open_fts_reader(path: Path) -> apsw.Connection:
    if not path.is_file():
        raise SearchPageStartupError(f"the full-text sidecar {path} does not exist")
    conn = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY)
    conn.set_busy_timeout(FTS_BUSY_TIMEOUT_MS)
    assert_schema_current(conn)
    return conn


def build_deps(cfg: Config) -> AppDeps:
    """Everything the app needs, failing closed: no password file, an
    unsafe secret file, an unreachable or foreign database, or a missing
    sidecar each stops startup with a one-line reason."""
    page = cfg.search_page
    data_root = cfg.paths.data_root
    passwords = PasswordFile(data_file(data_root, page.password_file))
    try:
        passwords.load()
    except SecretFileError as exc:
        raise SearchPageStartupError(
            f"no usable owner password ({exc}); run 'imsg search-page set-password' first"
        ) from exc
    session_path = data_file(data_root, page.session_file)
    ensure_private_dir(session_path.parent)
    sessions = SessionStore(session_path, lifetime_seconds=page.session_days * 86400)
    login_guard = LoginGuard(
        passwords,
        max_failures_per_client=page.login_max_failures,
        max_failures_global=page.login_global_max_failures,
        window_seconds=page.login_window_seconds,
    )

    def connect_pg() -> psycopg.Connection:
        conn = connect(cfg.database)
        try:
            verify_data_directory(conn, data_root)
        except Exception:
            conn.close()
            raise
        return conn

    pool = ConnectionPool(connect_pg, page.db_pool_size)
    with pool.connection():
        pass
    sidecar = fts_path(cfg)
    open_fts_reader(sidecar).close()
    fts = FtsReaders(lambda: open_fts_reader(sidecar), sidecar, page.db_pool_size)
    thumbnails = data_file(data_root, page.thumbnail_dir)
    ensure_private_dir(thumbnails)
    model_api = None
    if page.model_api.enabled:
        model_api = ModelApiClient(
            port=page.model_api.port,
            secret_file=data_file(data_root, page.model_api.secret_file),
            timeout_seconds=page.model_api.timeout_seconds,
        )
    return AppDeps(
        page=page,
        settings=SearchSettings.from_config(cfg),
        data_root=data_root,
        pool=pool,
        fts=fts,
        passwords=passwords,
        sessions=sessions,
        login_guard=login_guard,
        media=MediaConverter(thumbnails),
        model_api=model_api,
    )


def bind_sockets(addresses: list[tuple[str, int]]) -> list[socket.socket]:
    sockets: list[socket.socket] = []
    try:
        for host, port in addresses:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((host, port))
            sock.listen(128)
            sock.set_inheritable(True)
            sockets.append(sock)
    except OSError:
        for sock in sockets:
            sock.close()
        raise
    return sockets


def serve(cfg: Config) -> None:
    """Run the page until interrupted."""
    if not cfg.search_page.enabled:
        raise SearchPageStartupError("search_page.enabled is false in config")
    addresses = check_ports(cfg)
    deps = build_deps(cfg)
    app = build_app(deps)
    page = cfg.search_page
    tls: dict[str, str] = {}
    if page.tls_cert_file is not None and page.tls_key_file is not None:
        tls = {
            "ssl_certfile": str(data_file(cfg.paths.data_root, page.tls_cert_file)),
            "ssl_keyfile": str(data_file(cfg.paths.data_root, page.tls_key_file)),
        }
    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        server_header=False,
        proxy_headers=False,
        timeout_keep_alive=30,
        **tls,  # type: ignore[arg-type]
    )
    server = uvicorn.Server(config)
    sockets = bind_sockets(addresses)
    listening = ", ".join(f"{h}:{p}" for h, p in addresses)
    scheme = "https" if tls else "http"
    print(
        f"search page: listening on {listening} ({scheme}); model API "
        f"{'enabled' if deps.model_api is not None else 'off'}",
        file=sys.stderr,
    )
    try:
        server.run(sockets=sockets)
    finally:
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.close()
        deps.pool.close()


__all__ = ["bind_sockets", "build_deps", "check_ports", "fts_path", "open_fts_reader", "serve"]
