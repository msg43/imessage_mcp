"""A completion timestamp on a run-ledger row must be the wall clock,
never Postgres `now()`.

`now()` is frozen at transaction start. `extraction_run.finished_at`
was written with `now()` from inside the one transaction that wraps
S2's entire upsert loop, so it recorded when that transaction OPENED,
and every duration computed from the table excluded the whole write
phase (found 2026-09-14: a seed run that worked for about 2m35s
recorded 11.9s; an earlier 655k-message run recorded 7.3s).
`clock_timestamp()` is the wall clock, and is what every such stamp
now uses — `extraction_run` and `export_run` alike.

This is the mechanical guard for that class. It runs with no database
and fails on any `<completion column> = now()` anywhere under
`src/imsg`, whether or not the surrounding transaction happens to be
short today: the export ledger's stamps were statement-scoped and so
correct by accident, and a caller wrapping them in a transaction
would have reintroduced the bug silently.

`updated_at = now()` is deliberately not covered. Migration 0003's
trigger overwrites `updated_at` with `now()` on every UPDATE whatever
the statement says, and "the transaction that set this" is the
intended meaning there.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "imsg"

COMPLETION_STAMP_VIA_NOW = re.compile(r"\b(finished_at|completed_at|ended_at)\s*=\s*now\(\)")
COMPLETION_STAMP_VIA_CLOCK = re.compile(r"\bfinished_at\s*=\s*clock_timestamp\(\)")


def _python_sources() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def test_completion_timestamps_are_never_written_with_now() -> None:
    offenders: list[str] = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if COMPLETION_STAMP_VIA_NOW.search(line):
                offenders.append(
                    f"{path.relative_to(SRC_ROOT.parents[1])}:{lineno}: {line.strip()}"
                )
    assert not offenders, (
        "completion timestamps must use clock_timestamp() — now() is frozen at "
        "transaction start, so inside a long write transaction it records when the "
        "transaction opened, not when the work finished:\n" + "\n".join(offenders)
    )


def test_scan_actually_sees_the_run_ledger_writes() -> None:
    """The scan above passes vacuously if it is pointed at the wrong tree;
    make sure it can see the sites it exists to protect."""
    stamped = {
        path.name
        for path in _python_sources()
        if COMPLETION_STAMP_VIA_CLOCK.search(path.read_text(encoding="utf-8"))
    }
    assert {"extract.py", "planner.py", "push.py"} <= stamped, stamped
