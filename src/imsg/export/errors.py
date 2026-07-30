"""Export-specific exceptions.

Defined inside the export package (not `imsg.errors`) so this build
touches only its own directory; all of them still derive from
:class:`imsg.errors.ImsgError` so the CLI's top-level handler formats
them like every other stage failure.

Every one of these is a *refusal*, not a crash: the safe outcome of any
uncertainty in this package is that nothing reaches GCS.
"""

from __future__ import annotations

from imsg.errors import ImsgError


class ExportError(ImsgError):
    """Base class for export-gate failures."""


class ExportPlanError(ExportError):
    """`imsg export plan` could not produce a coherent, stage-able plan."""


class ExportApprovalError(ExportError):
    """Approval was requested for a run that cannot be approved (wrong
    state, manifest mismatch, staged bytes no longer hash-verify)."""


class ExportPushError(ExportError):
    """`imsg export push` refused to run or aborted before promoting the
    plan. Nothing is uploaded after this is raised mid-verification;
    per-document failures during execution are recorded in
    `export_run_item` instead (retryable), not raised as this."""


class ExportDriftError(ExportPushError):
    """The world changed between plan/approval and push: allowlist
    edits, participant-set changes, identity merges, config changes,
    staged-byte tampering, or vanished segments/chunks. Per SPEC §11.1
    any drift voids the plan — the fix is a new `plan` + approval,
    never a bypass."""
