"""Relevance labels from the page, in the eval harness's own format.

The page's relevant / not-relevant toggles write `relevance_label` rows
through `imsg.eval.io` — the same write path as the local
`mark_relevant` tool and `imsg eval label` — with `source =
'mark_relevant'`, so `imsg eval run` scores them with no conversion:

- A segment hit is labelled with `imsg.eval.io.label_segment_by_key`,
  anchored on the segment's first message GUID (SPEC §13.1: anchors
  survive re-segmentation; segment ids do not).
- A hit on a message that is in no segment yet is anchored on that
  message's own GUID; the runner resolves it to whichever segment holds
  the message once it is segmented.
- "Relevant" is grade 2 (`mark_relevant`'s default) and "not relevant"
  is grade 0. Clicking the active toggle again removes the page's own
  label; a label from another source (`manual`, `pool_judgment`) is
  shown but only ever replaced, never deleted, from here.

The eval query is the search text. If an `eval_query` already has
exactly this text (a curated `q001`, say), its id is used; otherwise the
page creates `adhoc:<text>` on the first label, as `mark_relevant` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from imsg.eval.io import label_segment_by_key, upsert_label, upsert_query
from imsg.eval.models import EvalQuery, RelevanceLabel
from imsg.eval.runner import compute_at4_check

if TYPE_CHECKING:
    import psycopg

    from imsg.eval.models import AT4Check

LABEL_SOURCE = "mark_relevant"
RELEVANT_GRADE = 2
NOT_RELEVANT_GRADE = 0
ADHOC_PREFIX = "adhoc:"
QUERY_NOTE = "created by the search page"

HitKind = Literal["segment", "message"]


def adhoc_query_id(query_text: str) -> str:
    return ADHOC_PREFIX + " ".join(query_text.split())


def find_query_id(pg: psycopg.Connection, query_text: str) -> str | None:
    """An existing eval query for this text: a curated one first, then the
    page's own `adhoc:` id."""
    text = " ".join(query_text.split())
    with pg.cursor() as cur:
        cur.execute(
            "SELECT query_id FROM eval_query WHERE query_text = %s "
            "ORDER BY (query_id LIKE 'adhoc:%%'), query_id LIMIT 1",
            (text,),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row[0])
        cur.execute("SELECT query_id FROM eval_query WHERE query_id = %s", (adhoc_query_id(text),))
        row = cur.fetchone()
    return str(row[0]) if row is not None else None


def ensure_query_id(pg: psycopg.Connection, query_text: str) -> str:
    existing = find_query_id(pg, query_text)
    if existing is not None:
        return existing
    query_id = adhoc_query_id(query_text)
    upsert_query(
        pg,
        EvalQuery(query_id=query_id, query_text=" ".join(query_text.split()), notes=QUERY_NOTE),
    )
    return query_id


@dataclass(frozen=True, slots=True)
class LabelCounts:
    total: int
    relevant: int
    not_relevant: int


def label_counts(pg: psycopg.Connection, query_id: str | None) -> LabelCounts:
    if query_id is None:
        return LabelCounts(0, 0, 0)
    with pg.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(*) FILTER (WHERE grade >= 1), count(*) FILTER (WHERE grade = 0) "
            "FROM relevance_label WHERE query_id = %s",
            (query_id,),
        )
        row = cur.fetchone()
    assert row is not None
    return LabelCounts(int(row[0]), int(row[1]), int(row[2]))


def baseline_progress(pg: psycopg.Connection) -> AT4Check:
    """SPEC §12 AT-4's minimums over every `local` eval query: the gate the
    labels are building toward."""
    return compute_at4_check(pg, target="local")


@dataclass(frozen=True, slots=True)
class HitLabel:
    grade: int
    source: str


def labels_for_hits(
    pg: psycopg.Connection,
    query_id: str | None,
    *,
    segment_ids: list[int],
    message_ids: list[int],
) -> tuple[dict[int, HitLabel], dict[int, HitLabel]]:
    """The label each shown hit carries for this query: per segment (any
    label anchored on a message in it, highest grade wins, as the runner
    resolves them) and per unsegmented message."""
    by_segment: dict[int, HitLabel] = {}
    by_message: dict[int, HitLabel] = {}
    if query_id is None:
        return by_segment, by_message
    with pg.cursor() as cur:
        if segment_ids:
            cur.execute(
                """
                SELECT sm.segment_id, rl.grade, rl.source
                FROM relevance_label rl
                JOIN message m ON m.source_guid = rl.anchor_guid
                JOIN segment_message sm ON sm.message_id = m.message_id
                WHERE rl.query_id = %(q)s AND sm.segment_id = ANY(%(ids)s::bigint[])
                """,
                {"q": query_id, "ids": segment_ids},
            )
            for segment_id, grade, source in cur.fetchall():
                current = by_segment.get(int(segment_id))
                if current is None or int(grade) > current.grade:
                    by_segment[int(segment_id)] = HitLabel(int(grade), str(source))
        if message_ids:
            cur.execute(
                """
                SELECT m.message_id, rl.grade, rl.source
                FROM relevance_label rl JOIN message m ON m.source_guid = rl.anchor_guid
                WHERE rl.query_id = %(q)s AND m.message_id = ANY(%(ids)s::bigint[])
                """,
                {"q": query_id, "ids": message_ids},
            )
            for message_id, grade, source in cur.fetchall():
                by_message[int(message_id)] = HitLabel(int(grade), str(source))
    return by_segment, by_message


@dataclass(frozen=True, slots=True)
class LabelOutcome:
    query_id: str
    grade: int | None
    counts: LabelCounts


def _segment_anchor_guids(pg: psycopg.Connection, segment_key: str) -> list[str]:
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT m.source_guid FROM segment s
            JOIN segment_message sm ON sm.segment_id = s.segment_id
            JOIN message m ON m.message_id = sm.message_id
            WHERE s.stable_key = %s
            """,
            (segment_key,),
        )
        return [str(r[0]) for r in cur.fetchall()]


def _message_guid(pg: psycopg.Connection, message_key: str) -> str | None:
    with pg.cursor() as cur:
        cur.execute("SELECT source_guid FROM message WHERE message_key = %s", (message_key,))
        row = cur.fetchone()
    return str(row[0]) if row is not None else None


class UnknownHitError(ValueError):
    """The labelled hit no longer exists (re-segmented, or a bad key)."""


def write_label(
    pg: psycopg.Connection,
    *,
    query_text: str,
    kind: HitKind,
    key: str,
    grade: int | None,
) -> LabelOutcome:
    """Set (`grade` 0 or 2) or clear (`None`) the page's label on one hit,
    in one transaction."""
    if grade not in (None, RELEVANT_GRADE, NOT_RELEVANT_GRADE):
        raise ValueError("grade must be 0 (not relevant), 2 (relevant) or None (clear)")
    with pg.transaction():
        query_id = ensure_query_id(pg, query_text)
        if kind == "segment":
            anchors = _segment_anchor_guids(pg, key)
            if not anchors:
                raise UnknownHitError("that conversation segment no longer exists; search again")
            if grade is None:
                with pg.cursor() as cur:
                    cur.execute(
                        "DELETE FROM relevance_label WHERE query_id = %s AND source = %s "
                        "AND anchor_guid = ANY(%s::text[])",
                        (query_id, LABEL_SOURCE, anchors),
                    )
            else:
                label_segment_by_key(
                    pg,
                    query_id=query_id,
                    segment_key=key,
                    grade=grade,
                    source=LABEL_SOURCE,
                    query_text_if_new=query_text,
                )
        else:
            guid = _message_guid(pg, key)
            if guid is None:
                raise UnknownHitError("that message no longer exists; search again")
            if grade is None:
                with pg.cursor() as cur:
                    cur.execute(
                        "DELETE FROM relevance_label WHERE query_id = %s AND source = %s "
                        "AND anchor_guid = %s",
                        (query_id, LABEL_SOURCE, guid),
                    )
            else:
                upsert_label(
                    pg,
                    RelevanceLabel(
                        query_id=query_id, anchor_guid=guid, grade=grade, source=LABEL_SOURCE
                    ),
                )
        counts = label_counts(pg, query_id)
    return LabelOutcome(query_id=query_id, grade=grade, counts=counts)


__all__ = [
    "ADHOC_PREFIX",
    "LABEL_SOURCE",
    "NOT_RELEVANT_GRADE",
    "RELEVANT_GRADE",
    "HitLabel",
    "LabelCounts",
    "LabelOutcome",
    "UnknownHitError",
    "adhoc_query_id",
    "baseline_progress",
    "ensure_query_id",
    "find_query_id",
    "label_counts",
    "labels_for_hits",
    "write_label",
]
