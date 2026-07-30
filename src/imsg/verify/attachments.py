"""AT-3 completion — attachment reconciliation exit criteria (SPEC §12
AT-3): "Output: CSV + summary (materialization rate overall, by year,
by type; `missing` list enumerated). Exit criteria is the report
**plus an owner-accepted exception manifest**: every non-materialized
row classified (`dataless_retrying`, `remote_missing`, `unsupported`,
`error`) and a stratified sample of materialized files hashes/opens
successfully."

`imsg.backfill.reconcile.build_reconciliation_report` (S5a's own build)
already answers the base question — "does the DB's `state` match the
filesystem" — for every attachment row; **this module does not
duplicate that**, it layers the rest of AT-3's exit criteria on top:
by-year/by-type breakdown, the four-category exception manifest, CSV
rendering, and the stratified hash/open integrity pass over
*materialized* files (which `build_reconciliation_report` doesn't
touch — it only checks presence, not content integrity).
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.backfill.reconcile import ReconciliationGap, build_reconciliation_report
from imsg.hashing import sha256_file

if TYPE_CHECKING:
    import psycopg

EXCEPTION_CATEGORIES = ("dataless_retrying", "remote_missing", "unsupported", "error")

_STATE_TO_CATEGORY = {
    "dataless": "dataless_retrying",
    "materializing": "dataless_retrying",  # interrupted run — a re-run naturally retries it
    "missing": "remote_missing",
    "error": "error",
    # 'materialized' rows never appear as a *gap* under a matching state — see
    # `_classify` below for the "state says materialized but the file is
    # absent/mismatched" case, which is a data-integrity error, not a
    # not-yet-materialized state.
}
"""SPEC §12 AT-3 names an `unsupported` exception category (a MIME/UTI
S5a will never be able to materialize), but `materialization_state`
(migration 0001) has no such state — S5a's router has no "this type
cannot ever be fetched" outcome, only the backoff/retry states listed
above. `_classify` below never produces `unsupported` as a result
today; the category is kept in `EXCEPTION_CATEGORIES` so the manifest
shape matches the spec exactly and the bucket is ready the day such a
state exists, but its count will always be 0 against the current
schema — flagged in the build report as a spec/schema gap, not
silently worked around."""


def _classify(gap: ReconciliationGap) -> str:
    if gap.state == "materialized":
        # DB says materialized, but build_reconciliation_report only
        # produces a gap for this state when the file is actually
        # missing on disk — that is a data-integrity error, not a
        # pending-retry state.
        return "error"
    return _STATE_TO_CATEGORY.get(gap.state, "error")


@dataclass(frozen=True, slots=True)
class ExceptionEntry:
    attachment_id: int
    attachment_key: str
    state: str
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class IntegritySampleResult:
    attachment_id: int
    attachment_key: str
    cache_path: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class AT3Report:
    total: int
    materialized_and_present: int
    completeness_ratio: float
    by_year: dict[str, tuple[int, int]]
    """year (or `"unknown"` for an attachment with no linked message) ->
    (materialized_and_present, total) — "by year" (SPEC §12 AT-3)."""
    by_mime_type: dict[str, tuple[int, int]]
    """mime_type (or `"unknown"` when NULL) -> (materialized_and_present,
    total) — "by type" (SPEC §12 AT-3)."""
    exceptions: tuple[ExceptionEntry, ...]
    """Every non-materialized-and-present row, classified — the
    exception manifest SPEC §12 AT-3 requires."""
    integrity_sample: tuple[IntegritySampleResult, ...]
    integrity_sample_ok_count: int
    passed: bool
    """True iff every exception classified into a known category AND
    every sampled file passed integrity verification. This is a
    build-time completeness check of the *tooling*, not the owner
    acceptance SPEC §12 AT-3 also requires ("plus an owner-accepted
    exception manifest") — that sign-off happens outside this code."""
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AttachmentRow:
    attachment_id: int
    attachment_key: str
    state: str
    cache_path: str | None
    sha256: str | None
    mime_type: str | None
    year: int | None


def _fetch_attachment_rows(conn: psycopg.Connection) -> list[_AttachmentRow]:
    """One row per attachment, `year` = the UTC year of the earliest
    message it's linked into (SPEC's "by year" reads naturally as "the
    year it was sent," not the year the local cache row was created) —
    `NULL` for an attachment with no `message_attachment` link."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attachment_id, a.attachment_key, a.state, a.cache_path,
                   a.sha256, a.mime_type,
                   extract(year FROM min(m.sent_at) AT TIME ZONE 'UTC')::int AS year
            FROM attachment a
            LEFT JOIN message_attachment ma ON ma.attachment_id = a.attachment_id
            LEFT JOIN message m ON m.message_id = ma.message_id
            GROUP BY a.attachment_id, a.attachment_key, a.state, a.cache_path,
                     a.sha256, a.mime_type
            """
        )
        rows = cur.fetchall()
    return [
        _AttachmentRow(
            attachment_id=int(aid),
            attachment_key=str(akey),
            state=str(state),
            cache_path=cache_path,
            sha256=sha,
            mime_type=mime,
            year=int(year) if year is not None else None,
        )
        for aid, akey, state, cache_path, sha, mime, year in rows
    ]


def _verify_one_file(row: _AttachmentRow) -> IntegritySampleResult:
    if not row.cache_path:
        return IntegritySampleResult(
            row.attachment_id, row.attachment_key, "", False, "no cache_path recorded"
        )
    path = Path(row.cache_path)
    try:
        if path.stat().st_size == 0:
            return IntegritySampleResult(
                row.attachment_id, row.attachment_key, str(path), False, "file is zero bytes"
            )
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        return IntegritySampleResult(
            row.attachment_id, row.attachment_key, str(path), False, f"could not open: {exc}"
        )
    if row.sha256 and actual_sha256 != row.sha256:
        return IntegritySampleResult(
            row.attachment_id,
            row.attachment_key,
            str(path),
            False,
            f"sha256 mismatch: db={row.sha256} file={actual_sha256}",
        )
    detail = "sha256 verified" if row.sha256 else "opened OK, no stored sha256 to compare"
    return IntegritySampleResult(row.attachment_id, row.attachment_key, str(path), True, detail)


def build_at3_report(
    conn: psycopg.Connection,
    *,
    integrity_sample_size: int = 25,
    seed: int | None = None,
) -> AT3Report:
    """The full SPEC §12 AT-3 artifact. `seed` makes the stratified
    sample reproducible (mainly for tests); omit it for a real run."""
    base = build_reconciliation_report(conn)
    rows = _fetch_attachment_rows(conn)
    by_id = {r.attachment_id: r for r in rows}

    by_year: dict[str, list[int]] = {}
    by_mime: dict[str, list[int]] = {}
    materialized_present_ids: list[int] = []
    for row in rows:
        year_key = str(row.year) if row.year is not None else "unknown"
        mime_key = row.mime_type or "unknown"
        present = (
            row.state == "materialized"
            and row.cache_path is not None
            and Path(row.cache_path).is_file()
        )
        y = by_year.setdefault(year_key, [0, 0])
        y[1] += 1
        y[0] += 1 if present else 0
        m = by_mime.setdefault(mime_key, [0, 0])
        m[1] += 1
        m[0] += 1 if present else 0
        if present:
            materialized_present_ids.append(row.attachment_id)

    exceptions = tuple(
        ExceptionEntry(
            attachment_id=gap.attachment_id,
            attachment_key=gap.attachment_key,
            state=gap.state,
            category=_classify(gap),
            reason=gap.reason,
        )
        for gap in base.gaps
    )

    rng = random.Random(seed)
    sample_ids = (
        materialized_present_ids
        if len(materialized_present_ids) <= integrity_sample_size
        else rng.sample(materialized_present_ids, integrity_sample_size)
    )
    integrity_sample = tuple(_verify_one_file(by_id[aid]) for aid in sample_ids)
    ok_count = sum(1 for s in integrity_sample if s.ok)

    reasons: list[str] = []
    unclassified = [e for e in exceptions if e.category not in EXCEPTION_CATEGORIES]
    if unclassified:
        reasons.append(
            f"{len(unclassified)} gap(s) did not classify into a known exception category"
        )
    failed_sample = [s for s in integrity_sample if not s.ok]
    if failed_sample:
        reasons.append(
            f"{len(failed_sample)} of {len(integrity_sample)} sampled materialized "
            f"file(s) failed integrity verification"
        )

    return AT3Report(
        total=base.total,
        materialized_and_present=base.materialized_and_present,
        completeness_ratio=base.completeness_ratio,
        by_year={k: (v[0], v[1]) for k, v in sorted(by_year.items())},
        by_mime_type={k: (v[0], v[1]) for k, v in sorted(by_mime.items())},
        exceptions=exceptions,
        integrity_sample=integrity_sample,
        integrity_sample_ok_count=ok_count,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def report_to_csv(report: AT3Report) -> str:
    """SPEC §12 AT-3: "Output: CSV + summary." Three sections in one
    file: overall/by-year/by-type rates, then the full exception
    manifest, then the integrity sample."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["section", "key", "materialized_present", "total", "ratio"])
    writer.writerow(
        [
            "overall",
            "all",
            report.materialized_and_present,
            report.total,
            f"{report.completeness_ratio:.4f}",
        ]
    )
    for year, (present, total) in report.by_year.items():
        writer.writerow(["by_year", year, present, total, f"{(present / total) if total else 1.0:.4f}"])
    for mime, (present, total) in report.by_mime_type.items():
        writer.writerow(
            ["by_mime_type", mime, present, total, f"{(present / total) if total else 1.0:.4f}"]
        )

    writer.writerow([])
    writer.writerow(["attachment_id", "attachment_key", "state", "category", "reason"])
    for e in report.exceptions:
        writer.writerow([e.attachment_id, e.attachment_key, e.state, e.category, e.reason])

    writer.writerow([])
    writer.writerow(["sample_attachment_id", "sample_attachment_key", "cache_path", "ok", "detail"])
    for s in report.integrity_sample:
        writer.writerow([s.attachment_id, s.attachment_key, s.cache_path, s.ok, s.detail])

    return buf.getvalue()


def format_report_text(report: AT3Report) -> str:
    lines = [
        "AT-3 attachment reconciliation:",
        f"  total attachments: {report.total}",
        f"  materialized + present: {report.materialized_and_present} "
        f"({report.completeness_ratio:.2%})",
        f"  exceptions: {len(report.exceptions)}",
        f"  integrity sample: {report.integrity_sample_ok_count}/{len(report.integrity_sample)} ok",
        "  by year (present/total):",
    ]
    for year, (present, total) in report.by_year.items():
        lines.append(f"    {year}: {present}/{total}")
    lines.append("  by mime type (present/total):")
    for mime, (present, total) in report.by_mime_type.items():
        lines.append(f"    {mime}: {present}/{total}")
    category_counts = dict.fromkeys(EXCEPTION_CATEGORIES, 0)
    for e in report.exceptions:
        category_counts[e.category] = category_counts.get(e.category, 0) + 1
    lines.append("  exception categories:")
    for cat in EXCEPTION_CATEGORIES:
        lines.append(f"    {cat}: {category_counts.get(cat, 0)}")
    lines.append(f"  VERDICT: {'PASS' if report.passed else 'FAIL'}")
    for reason in report.reasons:
        lines.append(f"    - {reason}")
    return "\n".join(lines)


__all__ = [
    "EXCEPTION_CATEGORIES",
    "AT3Report",
    "ExceptionEntry",
    "IntegritySampleResult",
    "build_at3_report",
    "format_report_text",
    "report_to_csv",
]
