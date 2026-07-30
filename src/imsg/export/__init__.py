"""S8 — the export gate (SPEC §11): the only path by which any message
content can leave the encrypted volume for GCS / Discovery Engine.

This package is deliberately paranoid. The rest of the system can be
wrong and the blast radius is a bad search result; if this package is
wrong, a private message enters a corporate data store subject to
organizational retention and potential legal discovery, and cannot be
meaningfully recalled. Every trade-off here resolves toward DENY
(hard requirement 5).

The gate, in one paragraph
--------------------------
A segment exports iff its chat passes `imsg.export.eligibility`: the
chat has at least one participant, every participant is on
`allowlist_person` with `text_allowed = true` (the owner included —
`is_owner` is not a bypass), every raw participant handle is resolved
to a person, every message sender and attributed tapback sender in the
chat is resolved and text-allowed. Absence, NULLs, and malformed data
all deny. Attachments are gated separately: an attachment's content
exports only when every non-unsent message linking it into the
document has a sender with `attachments_allowed = true`. Unsent
messages and prior edit versions never export — hard-coded (D1), not
config.

Control flow (SPEC §11.1, §11.4)
--------------------------------
`plan` renders immutable staging bytes and a hash-pinned manifest;
`approve` pins the manifest sha after re-verifying the staged bytes;
`push` promotes exactly the approved bytes after re-verifying every
pin AND re-deriving eligibility from the live database — any drift
(allowlist edits, a participant added to a group, an identity merge,
a vanished segment) aborts the whole push and requires a new plan.
Approval is mandatory for the first push and whenever a new person,
new thread, new attachment MIME class, config/policy change, or any
delete enters the plan.

What revocation honestly promises
---------------------------------
`purge_person` removes the person from eligibility, plans deletion of
every now-ineligible document, and verifies absence by document id
after the push. That removes the content from the Discovery Engine
index and the GCS bucket. It does NOT and cannot un-ring the bell:
content already captured by organizational retention, backups, or
another person's screen is beyond this system's reach. The only
reliable control is the gate at export time — which is why it denies
by default.
"""

from imsg.export.errors import (
    ExportApprovalError,
    ExportDriftError,
    ExportError,
    ExportPlanError,
    ExportPushError,
)
from imsg.export.planner import plan_export
from imsg.export.purge import purge_person
from imsg.export.push import push_export
from imsg.export.review import approve_run
from imsg.export.transport import ExportTransport, FakeTransport
from imsg.export.unclassified import unclassified_summary, write_unclassified_report

__all__ = [
    "ExportApprovalError",
    "ExportDriftError",
    "ExportError",
    "ExportPlanError",
    "ExportPushError",
    "ExportTransport",
    "FakeTransport",
    "approve_run",
    "plan_export",
    "purge_person",
    "push_export",
    "unclassified_summary",
    "write_unclassified_report",
]
