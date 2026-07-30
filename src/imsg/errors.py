"""Shared exception hierarchy for imsg.

All imsg-specific failures derive from :class:`ImsgError` so CLI entry
points can catch one type, print a clean message, and choose an exit
code — instead of letting stack traces leak filesystem paths or SQL to
whichever surface is watching (see SPEC §10.1 on error hygiene, applied
here to the CLI/operator surface too).

Downstream modules (pipeline stages, MCP tools) should raise or
subclass one of these rather than bare ``ValueError``/``RuntimeError``,
so the CLI's top-level handler can format them consistently.
"""

from __future__ import annotations


class ImsgError(Exception):
    """Base class for all imsg errors."""


class ConfigError(ImsgError):
    """Config file failed validation or could not be loaded.

    Raised for both structural pydantic failures and the
    security-relevant checks in SPEC §6 (path containment, secret
    literals, required-with-no-default fields, enum closures).
    """


class SecretResolutionError(ConfigError):
    """A ``keychain:`` or ``env:`` secret reference could not be resolved."""


class MountGateError(ImsgError):
    """The encrypted-volume mount gate refused to proceed (SPEC §5.4).

    Every CLI entry point and service start must run the mount gate
    before touching anything under ``paths.data_root``. On failure the
    caller should exit with ``EX_CONFIG`` (78), per the spec.
    """


class MigrationError(ImsgError):
    """The migration runner encountered an inconsistent or failing state."""


class ClusterFingerprintError(ImsgError):
    """The connected Postgres instance is not verifiably the dedicated
    imessage-index cluster (CLAUDE.md non-negotiable #6, SPEC §5.2)."""


class DdlLintError(ImsgError):
    """The DDL lint found a migration that violates a pgvector/index invariant."""


class StageNotImplementedError(ImsgError):
    """A pipeline stage CLI subcommand was invoked before it was built.

    Deliberately distinct from Python's built-in ``NotImplementedError``
    so the CLI can catch it and print a clean, stage-named message
    instead of a traceback, while still satisfying "raises
    NotImplementedError" in spirit for anything that imports the
    function directly.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(
            f"'{stage}' is not implemented yet — this is a CLI stub from the "
            f"foundation build. A later build adds the {stage} pipeline stage."
        )
