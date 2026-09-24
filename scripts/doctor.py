#!/usr/bin/env python3
"""scripts/doctor.py -- plain-English, read-only diagnostic checks.

Run this to find out why your local install isn't working, before
asking for help. Every check is read-only: this script never writes a
file, never modifies the database, and never opens (reads the
contents of) the live `chat.db` -- it only checks whether it *would be
readable* (a permissions/existence check), which is what "Full Disk
Access" verification needs. See CLAUDE.md non-negotiable #1.

It cannot check for a single common first name leaking into the repo
(that is `scripts/check_public_safety.py`'s job, and even that check
cannot catch every case) -- this script only checks environment setup.

Usage:
    uv run python scripts/doctor.py [--config PATH]

Exits 0 if every check passes, 1 if any check fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from imsg.config.schema import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
IMSG_DUMP_PATH = REPO_ROOT / "tools" / "imsg-dump" / "target" / "release" / "imsg-dump"

# Making sure `src/` is importable when this script is run directly
# (e.g. `python scripts/doctor.py` rather than through an installed
# entry point) without requiring the package to be installed.
_SRC_DIR = REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One diagnostic check's outcome.

    `next_step` is always a plain-English sentence: on failure, what to
    do to fix it; on success, a short confirmation.
    """

    name: str
    passed: bool
    next_step: str


def _fail(name: str, next_step: str) -> CheckResult:
    return CheckResult(name=name, passed=False, next_step=next_step)


def _pass(name: str, next_step: str = "OK.") -> CheckResult:
    return CheckResult(name=name, passed=True, next_step=next_step)


# --------------------------------------------------------------------------
# Individual checks. Each is self-contained and never raises -- any
# unexpected error is caught by `run_check` and reported as a FAIL.
# --------------------------------------------------------------------------


def check_uv() -> CheckResult:
    if shutil.which("uv") is None:
        return _fail(
            "uv present",
            "Do this next: install uv -- "
            "https://docs.astral.sh/uv/getting-started/installation/",
        )
    return _pass("uv present", "uv is on your PATH.")


def check_rust_shim(imsg_dump_path: Path = IMSG_DUMP_PATH) -> CheckResult:
    if not imsg_dump_path.is_file() or not os.access(imsg_dump_path, os.X_OK):
        return _fail(
            "Rust shim built (imsg-dump)",
            "Do this next: build it -- run "
            "`cd tools/imsg-dump && cargo build --release`.",
        )
    return _pass("Rust shim built (imsg-dump)", f"Found at {imsg_dump_path}.")


def _dsn_host_port(dsn: str) -> tuple[str, int] | None:
    parts = urlsplit(dsn)
    if not parts.hostname or not parts.port:
        return None
    return parts.hostname, parts.port


def check_postgres_reachable(dsn: str, *, timeout: float = 3.0) -> CheckResult:
    hostport = _dsn_host_port(dsn)
    if hostport is None:
        return _fail(
            "Postgres reachable on 5433",
            f"Do this next: fix database.dsn in your config -- "
            f"could not parse a host/port out of '{dsn}'.",
        )
    host, port = hostport
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        return _fail(
            "Postgres reachable on 5433",
            "Do this next: start your local Postgres cluster -- run "
            f"`scripts/bootstrap_local_postgres.sh <DATA_ROOT>` if you haven't "
            f"already, or start it if it's stopped (could not reach "
            f"{host}:{port}: {exc}).",
        )
    return _pass("Postgres reachable on 5433", f"Connected to {host}:{port}.")


REQUIRED_EXTENSIONS = ("vector", "pg_prewarm")


def check_extensions(dsn: str, *, timeout: float = 3.0) -> CheckResult:
    name = "pgvector + pg_prewarm present"
    try:
        import psycopg
    except ImportError:
        return _fail(name, "Do this next: run `uv sync --extra dev` (psycopg is missing).")

    try:
        with (
            psycopg.connect(dsn, connect_timeout=int(timeout), autocommit=True) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                "SELECT extname FROM pg_extension WHERE extname = ANY(%s)",
                (list(REQUIRED_EXTENSIONS),),
            )
            found = {row[0] for row in cur.fetchall()}
    except Exception as exc:
        return _fail(
            name,
            "Do this next: make sure Postgres is reachable first (see the "
            f"'Postgres reachable' check above), then re-run this. ({exc})",
        )

    missing = [ext for ext in REQUIRED_EXTENSIONS if ext not in found]
    if missing:
        joined = " and ".join(missing)
        return _fail(
            name,
            f"Do this next: connect with psql and run "
            f"`CREATE EXTENSION {joined};` (missing: {joined}).",
        )
    return _pass(name, "Both extensions are enabled.")


def check_data_volume(data_root: Path) -> CheckResult:
    name = "data volume mounted + encrypted + sentinel"
    try:
        from imsg.errors import MountGateError
        from imsg.mount.guard import guard_mount
    except ImportError as exc:
        return _fail(name, f"Do this next: run `uv sync --extra dev` ({exc}).")

    try:
        # This mirrors the exact policy the real startup gate enforces
        # (imsg.mount.guard.guard_mount) instead of re-implementing any
        # of "is this mounted / encrypted / the right volume" here --
        # doctor.py only reads the result, it never touches the mount.
        guard_mount(data_root)
    except MountGateError as exc:
        return _fail(name, f"Do this next: {exc}")
    except Exception as exc:
        return _fail(name, f"Do this next: investigate this unexpected error: {exc}")
    return _pass(name, f"'{data_root}' is a mounted, encrypted volume with its sentinel file.")


def check_config_loads(config_path: Path | None) -> tuple[CheckResult, Config | None]:
    """Returns the check result and the loaded `Config` (or `None`)."""
    name = "config loads (imsg.config.loader.load_config)"
    try:
        from imsg.config.loader import load_config
        from imsg.errors import ConfigError
    except ImportError as exc:
        return _fail(name, f"Do this next: run `uv sync --extra dev` ({exc})."), None

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        return (
            _fail(
                name,
                f"Do this next: fix your config file -- {exc}",
            ),
            None,
        )
    except Exception as exc:
        return _fail(name, f"Do this next: investigate this unexpected error: {exc}"), None
    return _pass(name, "Config loaded and validated."), config


REAL_BACKEND_PACKAGES = ("mlx", "mlx_lm")


def check_models_present(config: Config | None) -> CheckResult:
    name = "models present"
    if config is None:
        return _fail(name, "Do this next: fix your config first (see the config check above).")

    backend = getattr(getattr(config, "models", None), "backend", None)
    if backend == "fake":
        return _pass(name, "models.backend is 'fake' -- no real model runtime required.")

    missing = [pkg for pkg in REAL_BACKEND_PACKAGES if importlib.util.find_spec(pkg) is None]
    if missing:
        return _fail(
            name,
            "Do this next: run `uv sync --extra models` "
            f"(missing packages: {', '.join(missing)}).",
        )
    return _pass(name, "Model runtime packages are importable.")


def check_reranker_present(config: Config | None) -> CheckResult:
    name = "reranker converted directory present"
    if config is None:
        return _fail(name, "Do this next: fix your config first (see the config check above).")

    try:
        data_root: Path = config.paths.data_root
        reranker_model: str = config.retrieval.reranker_model
    except AttributeError as exc:
        return _fail(name, f"Do this next: investigate this unexpected error: {exc}")

    candidate = data_root / reranker_model
    if candidate.is_dir():
        return _pass(name, f"Found at {candidate}.")

    # `retrieval.reranker_model` may instead name a plain Hugging Face
    # repo id (e.g. "org/model") rather than a local conversion -- that
    # form has nothing to check locally, since the provider downloads
    # it by id. Only the local-conversion form (data-root-relative,
    # e.g. "models/qwen3-reranker-0.6b-mxfp8-e61197ed") is expected to
    # exist on disk already.
    if not reranker_model.startswith("models/"):
        return _pass(
            name,
            f"'{reranker_model}' looks like a Hugging Face repo id, not a local "
            "conversion -- nothing to check locally.",
        )

    return _fail(
        name,
        f"Do this next: convert it -- see the matching entry's `command` in "
        f"models/manifest.lock.yaml (expected directory: {candidate}).",
    )


def check_full_disk_access(chat_db_path: Path) -> CheckResult:
    """Existence/readability check ONLY. Never opens (reads the contents of) chat.db."""
    name = "Full Disk Access (chat.db readable)"
    if not chat_db_path.exists():
        return _fail(
            name,
            f"Do this next: check the path is right -- '{chat_db_path}' does not exist.",
        )
    # os.access() asks the OS for a permission verdict; it never opens
    # or reads any bytes of the file.
    if not os.access(chat_db_path, os.R_OK):
        return _fail(
            name,
            "Do this next: grant Full Disk Access to your terminal app in "
            "System Settings -> Privacy & Security -> Full Disk Access, then "
            "restart the terminal.",
        )
    return _pass(name, f"'{chat_db_path}' is readable.")


def run_check(fn: Callable[..., Any], *args: Any) -> Any:
    """Run one check, converting any surprise exception into a FAIL."""
    try:
        return fn(*args)
    except Exception as exc:
        return _fail(getattr(fn, "__name__", str(fn)), f"Do this next: investigate: {exc}")


def print_result(result: CheckResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name}")
    print(f"       {result.next_step}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only diagnostic checks for an imessage-index setup."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (defaults to $IMSG_CONFIG or ./config.yaml).",
    )
    args = parser.parse_args(argv)

    results: list[CheckResult] = []

    results.append(run_check(check_uv))
    results.append(run_check(check_rust_shim))

    config_result, config = run_check(check_config_loads, args.config)
    results.append(config_result)

    if config is not None:
        results.append(run_check(check_postgres_reachable, config.database.dsn))
        results.append(run_check(check_extensions, config.database.dsn))
        results.append(run_check(check_data_volume, config.paths.data_root))
        results.append(run_check(check_full_disk_access, config.paths.live_chat_db))
    else:
        results.append(
            _fail(
                "Postgres reachable on 5433",
                "Do this next: fix your config first (see the config check above).",
            )
        )
        results.append(
            _fail(
                "pgvector + pg_prewarm present",
                "Do this next: fix your config first (see the config check above).",
            )
        )
        results.append(
            _fail(
                "data volume mounted + encrypted + sentinel",
                "Do this next: fix your config first (see the config check above).",
            )
        )
        results.append(
            _fail(
                "Full Disk Access (chat.db readable)",
                "Do this next: fix your config first (see the config check above).",
            )
        )

    results.append(run_check(check_models_present, config))
    results.append(run_check(check_reranker_present, config))

    print("imessage-index doctor")
    print("======================")
    for result in results:
        print_result(result)
    print()

    if all(r.passed for r in results):
        print("All checks passed.")
        return 0

    failed = [r.name for r in results if not r.passed]
    print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
