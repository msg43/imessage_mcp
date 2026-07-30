"""Acceptance-test diagnostics that don't belong inside a pipeline
stage (SPEC §12): AT-2 seed completeness and the AT-3 exception-
manifest completion.

Why here and not `imsg.eval`: both AT-2 and AT-3 compare *this build's
state* (the corpus, the attachment cache) against an expectation —
they are one-shot diagnostic verdicts about pipeline completeness, not
retrieval-quality measurements over a query set. `imsg.eval` stays
scoped to SPEC §13 (queries/labels/metrics/runs); this package is the
`imsg verify-seed` / `imsg reconcile-attachments` home SPEC §8 S7 and
`docs/DECISIONS.md` D8 flag as needing one.
"""
