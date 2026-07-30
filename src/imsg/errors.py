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


class SnapshotError(ImsgError):
    """S1 snapshot of the live ``chat.db`` failed (SPEC §8 S1) — e.g.
    the live database stayed locked past the busy-timeout retry budget,
    the destination volume lacked the required free-space margin, or
    the backed-up file failed its post-backup integrity check
    (``PRAGMA quick_check`` / a missing expected core table)."""


class ExtractionError(ImsgError):
    """S2 extraction from a snapshot failed outside its normal per-row
    degrade-and-continue handling (SPEC §8 S2) — e.g. the snapshot file
    is not a valid ``chat.db``-shaped SQLite database."""


class ImsgDumpError(ExtractionError):
    """The ``tools/imsg-dump`` GPL subprocess (SPEC §4.2) could not be
    started, exited nonzero, or emitted a line that does not parse as
    the NDJSON contract S2 expects. This is about the *subprocess
    boundary* failing, not an individual message's decode failure —
    the shim itself is specified to degrade per-row (log + null body)
    rather than raise, so this error means the boundary itself broke."""


class IdentityError(ImsgError):
    """S3 identity resolution failed outside its normal per-handle
    review-stub/conflict handling (SPEC §8 S3) — e.g. the pre-S4
    invariant report found unresolved senders/participants, or Contacts
    import was requested but the framework is unavailable in a way that
    must fail loudly rather than silently degrade to raw handles."""


class SyncError(ImsgError):
    """S7 incremental sync failed outside the normal error handling of
    the stages it orchestrates (SPEC §8 S7)."""

