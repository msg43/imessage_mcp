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

from imsg.backfill.classify import parse_unsupported_reason

if TYPE_CHECKING:
    import psycopg

_GAP_REASONS = {
    "missing": "nothing to materialize: no source path, or iCloud never delivered the file "
    "after repeated attempts",
    "error": "last materialization attempt errored",
    "dataless": "not yet materialized",
    "materializing": "materialization was in progress and never completed "
    "(likely an interrupted run)",
}

UNSPECIFIED_UNSUPPORTED_REASON = "unspecified"
"""`unsupported_reason` for an `unsupported` row whose
`materialization_last_error` does not carry the `unsupported[...]`
prefix `imsg.backfill.classify` writes — reported as such rather than
guessed at."""


@dataclass(frozen=True, slots=True)
class ReconciliationGap:
    attachment_id: int
    attachment_key: str
    state: str
    reason: str
    unsupported_reason: str | None = None
    """For `state == 'unsupported'`, the reason class parsed from
    `materialization_last_error` (an `UnsupportedReason` value, or
    `UNSPECIFIED_UNSUPPORTED_REASON`); `None` for every other state."""


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


def _describe_gap(state: str, last_error: str | None) -> tuple[str, str | None]:
    """`(reason, unsupported_reason)` for a non-present row. Only the
    reason *class* is surfaced for `unsupported` rows — the stored
    `materialization_last_error` also carries the offending path, which
    belongs in the database, not in every rendering of the report."""
    if state == "materialized":
        return "row says materialized but the cache file is missing on disk", None
    if state == "unsupported":
        parsed = parse_unsupported_reason(last_error)
        if parsed is None:
            return (
                f"unsupported ({UNSPECIFIED_UNSUPPORTED_REASON}): can never be materialized; "
                "no reason class recorded",
                UNSPECIFIED_UNSUPPORTED_REASON,
            )
        return f"unsupported ({parsed.value}): {parsed.description}", parsed.value
    return _GAP_REASONS.get(state, f"unrecognized state {state!r}"), None


def build_reconciliation_report(conn: psycopg.Connection) -> ReconciliationReport:
    """Cross-checks every attachment row's `state`/`cache_path` against
    the real filesystem. Takes no `data_root` — `cache_path` is stored
    as a full resolved path (SPEC §7.2 `attachment.cache_path` comment:
    "content-addressed copy under $DATA_ROOT/attachments"), so this
    function trusts the stored path rather than re-deriving it, and
    reports a gap if that trust turns out to be misplaced.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, attachment_key, state, cache_path, materialization_last_error "
            "FROM attachment"
        )
        rows = cur.fetchall()

    present = 0
    gaps: list[ReconciliationGap] = []
    for attachment_id, attachment_key, state, cache_path, last_error in rows:
        on_disk = bool(cache_path) and Path(cache_path).is_file()
        if state == "materialized" and on_disk:
            present += 1
            continue
        reason, unsupported_reason = _describe_gap(state, last_error)
        gaps.append(
            ReconciliationGap(
                attachment_id=attachment_id,
                attachment_key=attachment_key,
                state=state,
                reason=reason,
                unsupported_reason=unsupported_reason,
            )
        )

    return ReconciliationReport(total=len(rows), materialized_and_present=present, gaps=tuple(gaps))


__all__ = [
    "UNSPECIFIED_UNSUPPORTED_REASON",
    "ReconciliationGap",
    "ReconciliationReport",
    "build_reconciliation_report",
]
