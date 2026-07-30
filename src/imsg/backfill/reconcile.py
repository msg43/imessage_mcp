"""AT-3 reconciliation report (SPEC §8 S5a, §10.4's AT-1/AT-2 sibling):
for every `attachment` row, does the cached file actually exist on
disk, cross-checked against `state` — a real completeness number with
every gap enumerated, not a count read only from the `state` column
(which could itself be stale — e.g. a cache file deleted out from under
the database by something else).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg

_GAP_REASONS = {
    "missing": "iCloud never materialized this file after repeated attempts",
    "error": "last materialization attempt errored",
    "dataless": "not yet materialized",
    "materializing": "materialization was in progress and never completed "
    "(likely an interrupted run)",
}


@dataclass(frozen=True, slots=True)
class ReconciliationGap:
    attachment_id: int
    attachment_key: str
    state: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    total: int
    materialized_and_present: int
    gaps: tuple[ReconciliationGap, ...]

    @property
    def completeness_ratio(self) -> float:
        if self.total == 0:
            return 1.0
        return self.materialized_and_present / self.total


def build_reconciliation_report(conn: psycopg.Connection) -> ReconciliationReport:
    """Cross-checks every attachment row's `state`/`cache_path` against
    the real filesystem. Takes no `data_root` — `cache_path` is stored
    as a full resolved path (SPEC §7.2 `attachment.cache_path` comment:
    "content-addressed copy under $DATA_ROOT/attachments"), so this
    function trusts the stored path rather than re-deriving it, and
    reports a gap if that trust turns out to be misplaced.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT attachment_id, attachment_key, state, cache_path FROM attachment")
        rows = cur.fetchall()

    present = 0
    gaps: list[ReconciliationGap] = []
    for attachment_id, attachment_key, state, cache_path in rows:
        on_disk = bool(cache_path) and Path(cache_path).is_file()
        if state == "materialized" and on_disk:
            present += 1
            continue
        if state == "materialized" and not on_disk:
            reason = "row says materialized but the cache file is missing on disk"
        else:
            reason = _GAP_REASONS.get(state, f"unrecognized state {state!r}")
        gaps.append(
            ReconciliationGap(
                attachment_id=attachment_id, attachment_key=attachment_key, state=state, reason=reason
            )
        )

    return ReconciliationReport(total=len(rows), materialized_and_present=present, gaps=tuple(gaps))


__all__ = ["ReconciliationGap", "ReconciliationReport", "build_reconciliation_report"]
