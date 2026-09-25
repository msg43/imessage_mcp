"""`imsg search-page ...`: serve the page, set the owner password, create
the model API's shared secret, check a deploy. The launch agent is
rendered by `imsg install-agents --only search-page` (`imsg.agents.plists`),
under the same supervisor as every other agent.

Like `imsg.eval.cli`, this module does not import `imsg.cli` (which
imports this one); `imsg.cli` registers the sub-app with one
`add_typer` line.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from imsg.config.loader import load_config
from imsg.errors import ImsgError
from imsg.mount.guard import run_guard_mount_or_exit
from imsg.search_page.auth import (
    MIN_PASSWORD_CHARS,
    SessionStore,
    check_password_policy,
    set_password,
    write_new_secret,
)
from imsg.search_page.secret_files import check_private_file, data_file

if TYPE_CHECKING:
    from imsg.config.schema import Config

search_page_app = typer.Typer(
    name="search-page",
    help="The private local search page (D14): local network only, owner login.",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to config.yaml. Defaults to $IMSG_CONFIG, then ./config.yaml."),
]


def _config(config: Path | None) -> Config:
    try:
        return load_config(config)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@search_page_app.command("serve")
def serve_cmd(config: ConfigOption = None) -> None:
    """Serve the page on `search_page.listen` (loopback plus the local
    network) until stopped. Refuses to start without an owner password."""
    from imsg.search_page.server import serve

    cfg = _config(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    try:
        serve(cfg)
    except ImsgError as exc:
        typer.echo(f"imsg: search page not started: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@search_page_app.command("set-password")
def set_password_cmd(
    config: ConfigOption = None,
    password_stdin: Annotated[
        bool,
        typer.Option(
            "--password-stdin",
            help="Read the password from standard input (one line) instead of prompting.",
        ),
    ] = False,
) -> None:
    """Set the owner password (stored as an scrypt hash in a 0600 file on
    the encrypted volume). Ends every signed-in session."""
    cfg = _config(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    page = cfg.search_page
    if password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = typer.prompt(
            f"New search-page password (at least {MIN_PASSWORD_CHARS} characters)",
            hide_input=True,
            confirmation_prompt=True,
        )
    try:
        check_password_policy(password)
    except ValueError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    try:
        path = data_file(cfg.paths.data_root, page.password_file)
        set_password(path, password)
        sessions_path = data_file(cfg.paths.data_root, page.session_file)
        ended = SessionStore(sessions_path, lifetime_seconds=page.session_days * 86400).revoke_all()
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"search page: password set ({path}, mode 0600); {ended} session(s) ended")


@search_page_app.command("init-model-secret")
def init_model_secret_cmd(
    config: ConfigOption = None,
    rotate: Annotated[
        bool, typer.Option("--rotate", help="Replace an existing secret (restart both servers after).")
    ] = False,
) -> None:
    """Create the internal model API's shared secret (0600). Both `imsg mcp
    public` and the page read it; nothing else needs a copy."""
    cfg = _config(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    try:
        path = data_file(cfg.paths.data_root, cfg.search_page.model_api.secret_file)
        if path.exists() and not rotate:
            check_private_file(path)
            typer.echo(f"search page: model API secret already present ({path}); use --rotate to replace")
            return
        write_new_secret(path)
    except ImsgError as exc:
        typer.echo(f"imsg: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"search page: model API secret written ({path}, mode 0600)")


@search_page_app.command("check")
def check_cmd(config: ConfigOption = None) -> None:
    """Report the page's deploy state: listeners, allowed hosts, whether the
    password and model-API secret files are in place and safe, and whether
    the model API answers. Reads only."""
    from imsg.search_page.model_api_client import ModelApiClient
    from imsg.search_page.server import check_ports

    cfg = _config(config)
    run_guard_mount_or_exit(cfg.paths.data_root)
    page = cfg.search_page
    ok = True
    typer.echo(f"enabled: {page.enabled}")
    try:
        listen = check_ports(cfg)
        typer.echo("listen: " + ", ".join(f"{h}:{p}" for h, p in listen))
    except ImsgError as exc:
        ok = False
        typer.echo(f"listen: REFUSED ({exc})")
    typer.echo("allowed hosts: " + ", ".join(page.allowed_hosts))
    for label, relative in (
        ("password file", page.password_file),
        ("model API secret", page.model_api.secret_file),
    ):
        try:
            path = data_file(cfg.paths.data_root, relative)
            check_private_file(path)
            typer.echo(f"{label}: ok ({path})")
        except ImsgError as exc:
            if label == "password file" or page.model_api.enabled:
                ok = False
            typer.echo(f"{label}: MISSING OR UNSAFE ({exc})")
    if page.model_api.enabled:
        client = ModelApiClient(
            port=page.model_api.port,
            secret_file=data_file(cfg.paths.data_root, page.model_api.secret_file),
            timeout_seconds=page.model_api.timeout_seconds,
        )
        try:
            health = client.health()
            typer.echo(f"model API: reachable, phase={health.get('phase')}, ready={health.get('ready')}")
        except ImsgError as exc:
            typer.echo(f"model API: not answering ({exc}); full-text search still works")
    else:
        typer.echo("model API: off (semantic search disabled; full-text search works)")
    typer.echo(
        "launch agent: imsg install-agents --only search-page --dest <dir> renders it "
        "(review, then copy to ~/Library/LaunchAgents and launchctl bootstrap)"
    )
    raise typer.Exit(code=0 if ok else 1)


__all__ = ["search_page_app"]
