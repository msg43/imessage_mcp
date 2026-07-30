"""Load and validate `config.yaml` (SPEC §6).

Usage, everywhere else in the codebase::

    from imsg.config.loader import load_config
    config = load_config(path)  # or load_config() to use $IMSG_CONFIG

This is the *only* sanctioned way to obtain a `Config` instance — load
it once near the process entry point and pass the object down, rather
than re-reading the file or re-touching `$IMSG_CONFIG` deeper in the
call stack. That keeps config access uniform for every downstream
pipeline stage.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from imsg.config.schema import Config
from imsg.config.secrets import SecretRef
from imsg.errors import ConfigError, SecretResolutionError

IMSG_CONFIG_ENV_VAR = "IMSG_CONFIG"
DEFAULT_DEV_CONFIG_PATH = Path("config.yaml")


def default_config_path() -> Path:
    """Resolve the config path the way every CLI entry point should.

    `$IMSG_CONFIG` wins if set (required for installed/production use,
    SPEC §3.2); otherwise development falls back to `./config.yaml`.
    """
    env_value = os.environ.get(IMSG_CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_DEV_CONFIG_PATH


def _format_validation_error(exc: ValidationError, source: Path) -> str:
    lines = [f"config at '{source}' failed validation ({exc.error_count()} error(s)):"]
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {loc}: {error['msg']}")
    return "\n".join(lines)


def load_config_dict(raw: dict[str, object], *, source: Path | str = "<dict>") -> Config:
    """Validate an already-parsed mapping. Mainly useful for tests."""
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, Path(source))) from exc


def load_config(path: Path | str | None = None) -> Config:
    """Read, parse, and validate `config.yaml`.

    Raises `ConfigError` (never a bare pydantic/YAML exception) on any
    failure — missing file, invalid YAML, or a failed validation rule —
    so CLI entry points can catch one exception type.
    """
    resolved = Path(path) if path is not None else default_config_path()
    if not resolved.exists():
        raise ConfigError(
            f"config file not found: '{resolved}' (set ${IMSG_CONFIG_ENV_VAR} or "
            f"pass --config)"
        )
    try:
        raw_text = resolved.read_text()
    except OSError as exc:
        raise ConfigError(f"could not read config file '{resolved}': {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file '{resolved}' is not valid YAML: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"config file '{resolved}' must be a YAML mapping at the top level, "
            f"got {type(data).__name__}"
        )

    return load_config_dict(data, source=resolved)


def collect_configured_secrets(config: Config) -> dict[str, SecretRef]:
    """Every `SecretRef`-typed field actually present in this config, by dotted name.

    Fields that are `None` (e.g. `mcp.public.oauth.owner_subject` when the
    public surface is disabled) are omitted — there is nothing to resolve.
    """
    secrets: dict[str, SecretRef] = {"database.password": config.database.password}
    if config.mcp.public.oauth.client_secret is not None:
        secrets["mcp.public.oauth.client_secret"] = config.mcp.public.oauth.client_secret
    if config.mcp.public.oauth.owner_subject is not None:
        secrets["mcp.public.oauth.owner_subject"] = config.mcp.public.oauth.owner_subject
    return secrets


def assert_secrets_resolvable(config: Config) -> None:
    """Actually resolve every configured secret reference (Keychain/env lookup).

    Kept separate from `load_config` so ordinary parsing/introspection
    (tests, `imsg migrate --status` before secrets exist, etc.) never
    has to touch the Keychain or environment. Call this explicitly
    wherever SPEC §6's "secrets resolvable" startup check applies.
    """
    failures: list[str] = []
    for name, ref in collect_configured_secrets(config).items():
        try:
            ref.resolve()
        except SecretResolutionError as exc:
            failures.append(f"{name} ({ref}): {exc}")
    if failures:
        raise ConfigError(
            "one or more secrets could not be resolved:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )


__all__ = [
    "assert_secrets_resolvable",
    "collect_configured_secrets",
    "default_config_path",
    "load_config",
    "load_config_dict",
]
