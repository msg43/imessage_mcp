"""Filling the S5b enrichment queue (SPEC §8 S5b).

The router (`imsg.enrich.router`) says which kinds an attachment needs;
until this module nothing asked it, so nothing was ever enqueued and no
attachment text was ever extracted. Two entry points put its answer in
the queue, both keyed on the queue's primary key `(attachment_id, kind)`
and both insert-only — a row already queued, in any state, is never
reset or duplicated (`done` stays `done`; reprocessing is only ever the
explicit `imsg enrich --retry-failed`):

- `enqueue_for_materialized` — S5a calls it the moment a row becomes
  `materialized`, so new attachments are queued as they arrive.
- `plan_enrichment` — `imsg enrich --plan`: every materialized attachment,
  for everything materialized before that hook existed, for rows another
  path materialized, and after a routing change (a new route reaches
  attachments that already have rows for their other kinds).

Routing is on the content-sniffed MIME type, never on `attachment.mime_type`
— that column is chat.db's claim, and a link-preview payload has none —
and a cache path is read only after it has passed the same containment
check the worker applies (under `data_root`, not a symlink, a regular
file). Sniffing the whole production index this way took about 41 s
(100,046 files, 2026-09-23), so the planner sniffs every run rather than
trusting a cache of earlier answers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.enrich.limits import check_path_containment
from imsg.enrich.mime import (
    DEFAULT_SNIFF_BATCH_SIZE,
    MimeSnifferFn,
    real_sniff_mime,
    sniff_mime_batch,
)
from imsg.enrich.queue import enqueue_pairs_by_kind
from imsg.enrich.router import ENRICHMENT_KINDS, route_for_mime
from imsg.errors import UntrustedAttachmentError
from imsg.paths import resolve_path

if TYPE_CHECKING:
    import psycopg


@dataclass
class EnrichmentPlanReport:
    """What one `plan_enrichment` pass found and did. Every count is of
    attachments, except the per-kind dictionaries, which count tasks."""

    attachments: int = 0
    """Materialized attachments examined."""
    refused: int = 0
    """Materialized rows whose cache path was never read: none recorded,
    outside `data_root`, a symlink, or not a regular file."""
    sniff_failed: int = 0
    """Readable paths `file` could not sniff; they are tried again next run."""
    enqueued: dict[str, int] = field(default_factory=dict)
    """Tasks inserted, by kind — in a dry run, tasks that would be."""
    already_queued: dict[str, int] = field(default_factory=dict)
    """Routed tasks already in the queue, in any state, by kind."""
    not_selected: dict[str, int] = field(default_factory=dict)
    """Routed tasks left out because `kinds` did not name their kind."""
    unroutable: dict[str, int] = field(default_factory=dict)
    """Attachments whose sniffed MIME type has no route, by that type."""
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class MaterializedEnqueue:
    """What `enqueue_for_materialized` did for one attachment."""

    mime_type: str | None
    routed: tuple[str, ...] = ()
    """The router's kinds for `mime_type`; empty when it has no route."""
    enqueued: tuple[str, ...] = ()
    """Of `routed`, the kinds inserted now (a kind already queued is not)."""
    error: str | None = None
    """Why nothing could be planned (the path was refused or could not be
    sniffed). `imsg enrich --plan` tries such an attachment again."""

    @property
    def unroutable(self) -> bool:
        return self.error is None and not self.routed


def _materialized_attachments(
    conn: psycopg.Connection,
) -> list[tuple[int, str | None, frozenset[str]]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attachment_id, a.cache_path,
                   coalesce(array_agg(e.kind::text) FILTER (WHERE e.kind IS NOT NULL), '{}')
            FROM attachment a
            LEFT JOIN enrichment e ON e.attachment_id = a.attachment_id
            WHERE a.state = 'materialized'
            GROUP BY a.attachment_id, a.cache_path
            ORDER BY a.attachment_id
            """
        )
        rows = cur.fetchall()
    return [(int(a), c, frozenset(str(k) for k in kinds)) for a, c, kinds in rows]


def _add(counts: dict[str, int], key: str, n: int = 1) -> None:
    counts[key] = counts.get(key, 0) + n


def plan_enrichment(
    conn: psycopg.Connection,
    *,
    data_root: Path,
    kinds: Sequence[str] | None = None,
    dry_run: bool = False,
    batch_size: int = DEFAULT_SNIFF_BATCH_SIZE,
) -> EnrichmentPlanReport:
    """Enqueue every routed kind that a materialized attachment is missing
    (`imsg enrich --plan`).

    `kinds` limits which kinds are enqueued (`--kinds`); the rest are
    counted in `not_selected` and left for a later run. `dry_run` sniffs
    and counts exactly as a real run would and writes nothing. One
    transaction holds every insert, so a run that fails part-way leaves
    the queue as it found it."""
    selected = frozenset(kinds) if kinds is not None else frozenset(ENRICHMENT_KINDS)
    rows = _materialized_attachments(conn)
    report = EnrichmentPlanReport(attachments=len(rows), dry_run=dry_run)
    root = resolve_path(data_root)

    readable: dict[int, Path] = {}
    for attachment_id, cache_path, _ in rows:
        if not cache_path:
            report.refused += 1
            continue
        try:
            readable[attachment_id] = check_path_containment(Path(cache_path), root)
        except UntrustedAttachmentError:
            report.refused += 1

    # The cache is content-addressed: identical attachments share a file,
    # which is sniffed once.
    sniffed = sniff_mime_batch(sorted(set(readable.values())), batch_size=batch_size)

    pairs: list[tuple[int, str]] = []
    for attachment_id, _, existing in rows:
        path = readable.get(attachment_id)
        if path is None:
            continue
        mime_type = sniffed.mime_by_path.get(path)
        if mime_type is None:
            report.sniff_failed += 1
            continue
        route = route_for_mime(mime_type)
        if not route:
            _add(report.unroutable, mime_type)
            continue
        for kind in route:
            if kind in existing:
                _add(report.already_queued, kind)
            elif kind not in selected:
                _add(report.not_selected, kind)
            else:
                pairs.append((attachment_id, kind))

    if dry_run:
        report.enqueued = dict(Counter(kind for _, kind in pairs))
    else:
        with conn.transaction():
            report.enqueued = enqueue_pairs_by_kind(conn, pairs)
    return report


def enqueue_for_materialized(
    conn: psycopg.Connection,
    attachment_id: int,
    cache_path: Path,
    *,
    data_root: Path,
    sniffer: MimeSnifferFn = real_sniff_mime,
) -> MaterializedEnqueue:
    """Queue the router's kinds for one attachment that has just become
    `materialized`, and commit. Called by S5a after it marks the row.

    A path that is refused or cannot be sniffed plans nothing and says
    why in `error`; it never fails the materialization, which has
    already succeeded, and `imsg enrich --plan` picks the attachment up
    later."""
    try:
        path = check_path_containment(Path(cache_path), resolve_path(data_root))
        mime_type = sniffer(path)
    except UntrustedAttachmentError as exc:
        return MaterializedEnqueue(mime_type=None, error=str(exc))
    route = route_for_mime(mime_type)
    if not route:
        return MaterializedEnqueue(mime_type=mime_type)
    inserted = enqueue_pairs_by_kind(conn, [(attachment_id, kind) for kind in route])
    conn.commit()
    return MaterializedEnqueue(
        mime_type=mime_type,
        routed=route,
        enqueued=tuple(kind for kind in route if inserted.get(kind)),
    )


__all__ = [
    "EnrichmentPlanReport",
    "MaterializedEnqueue",
    "enqueue_for_materialized",
    "plan_enrichment",
]
