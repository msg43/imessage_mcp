"""S5a — attachment backfill (SPEC §8 S5a): APFS dataless-placeholder
detection and throttled materialization, plus the AT-3 reconciliation
report. `pipeline` is the only module that talks to Postgres or the
real filesystem beyond the small, injectable probes in `dataless`.
"""

from __future__ import annotations
