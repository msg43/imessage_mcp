"""Every-hit search for the local search page (D14).

The MCP tool `search_messages` answers with the top few segments after a
reranker. This page answers with **every viable hit**, grouped by
conversation, and paints full-text results before semantic ones:

1. `run_fulltext` (fast, no model):
   - every segment the full-text index matches (BM25 words, a quoted
     phrase through the trigram table, or an emoji substring), up to
     `search_page.fts_max_hits`;
   - every attachment text chunk it matches, mapped to the segment that
     holds the attachment's message;
   - messages from the last `unindexed_window_days` that are in no
     segment yet (the index lags while heavy stages are paused), matched
     directly in Postgres.
2. `run_semantic` (after the query vectors arrive from the public MCP
   server's warm models, `imsg.search_page.model_api_client`):
   - every segment whose text vector is at least
     `semantic.text_min_similarity` similar to the query;
   - every attachment text chunk above that floor;
   - every image above `semantic.multimodal_min_similarity`.
   Each is a threshold scan: rows stream out of the HNSW index in
   distance order (`hnsw.iterative_scan = strict_order`) and reading stops
   at the first row past the threshold, so the answer is "all above the
   floor", not "the nearest k".

Hits are deduplicated by segment (a segment found by several channels is
one hit carrying each channel's rank), scored by reciprocal rank fusion
over the channels that found it (the same fusion SPEC §9.4 uses), and
grouped by conversation with a count per conversation. Nothing reranks
the full list; the optional `rerank` order sends only the best
`search_page.rerank_top` hits to the reranker.

Filters (people, attachments) reuse the retrieval layer's own predicate
(`imsg.retrieval.filters.compile_predicate`), under full scope: this page
is the owner's own surface, like the local MCP surface. The page adds its
own (`PageFilters`, `page_predicate`, `message_clause`):

- dates: a segment is kept when any part of it lies in the range, and the
  message that matched must itself be in the range (the MCP tools keep
  testing when a segment started);
- who sent it ("Sent by" a person, or sent by the owner / received): the
  message that matched must be one that person sent; a segment found by
  meaning must hold at least one;
- one-to-one or group conversations, and one conversation.

A search with no words and at least one filter lists every message the
filters keep (`CHANNEL_FILTERS`), grouped by conversation, newest first.

A segment found by the full-text index is kept only when the words occur
in what people wrote (`imsg.search_page.content_match`), not only in the
header lines and labels the index also holds.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast

from imsg.embed.vector_codec import vector_literal
from imsg.retrieval.access import LOCAL_FULL_ACCESS, resolve_request_scope
from imsg.retrieval.errors import (
    DateRangeInvalidError,
    InvalidArgumentError,
    PersonAmbiguousError,
    PersonNotFoundError,
)
from imsg.retrieval.filters import (
    MAX_PEOPLE_FILTER,
    CompiledPredicate,
    SearchFilters,
    compile_predicate,
    resolve_filters,
)
from imsg.retrieval.people import resolve_people
from imsg.retrieval.query import (
    AnalyzedQuery,
    analyze_query,
    bm25_match_expression,
    like_pattern,
    trigram_match_expression,
)
from imsg.search_page.content_match import (
    ContentMatch,
    content_query,
    segments_matching_content,
    text_condition,
)
from imsg.search_page.errors import SearchInputError
from imsg.search_page.threads import is_opaque_key

if TYPE_CHECKING:
    import apsw
    import psycopg

    from imsg.config.schema import Config

SortOrder = Literal["relevance", "date", "rerank"]
AttachmentFilter = Literal["any", "with", "without"]
DirectionFilter = Literal["any", "sent", "received"]
ChatKindFilter = Literal["any", "dm", "group"]
SORT_ORDERS: tuple[str, ...] = ("relevance", "date", "rerank")
ATTACHMENT_FILTERS: tuple[str, ...] = ("any", "with", "without")
DIRECTION_FILTERS: tuple[str, ...] = ("any", "sent", "received")
CHAT_KIND_FILTERS: tuple[str, ...] = ("any", "dm", "group")
OWNER_SENDER = "me"
"""What "Sent by" takes for the owner: the same word the Timeline uses."""

CHANNEL_TEXT = "text"
CHANNEL_ATTACHMENT_TEXT = "attachment_text"
CHANNEL_UNINDEXED = "unindexed"
CHANNEL_SEMANTIC = "semantic"
CHANNEL_ATTACHMENT_SEMANTIC = "attachment_semantic"
CHANNEL_IMAGE = "image"
CHANNEL_FILTERS = "filters"
"""A search with no words: the messages the filters keep."""
FULLTEXT_CHANNELS = (CHANNEL_TEXT, CHANNEL_ATTACHMENT_TEXT, CHANNEL_UNINDEXED, CHANNEL_FILTERS)
SEMANTIC_CHANNELS = (CHANNEL_SEMANTIC, CHANNEL_ATTACHMENT_SEMANTIC, CHANNEL_IMAGE)
CHANNEL_LABELS: dict[str, str] = {
    CHANNEL_TEXT: "text",
    CHANNEL_ATTACHMENT_TEXT: "attachment text",
    CHANNEL_UNINDEXED: "not yet indexed",
    CHANNEL_SEMANTIC: "semantic",
    CHANNEL_ATTACHMENT_SEMANTIC: "attachment, semantic",
    CHANNEL_IMAGE: "image",
    CHANNEL_FILTERS: "filters",
}

MAX_QUERY_CHARS = 1000
DISTANCE_ORDER_TOLERANCE = 1e-3
"""Rows leave the HNSW index in the index's own distance order, which
differs from the recomputed `<=>` by rounding (measured at most 3.7e-5,
`imsg.retrieval.vector_search.DISTANCE_ORDER_TOLERANCE`). A threshold scan
reads this far past the floor before it stops, and keeps only rows at or
under the floor."""
THRESHOLD_FETCH_ROWS = 256
ROWS_PER_SEGMENT_OVERFETCH = 5
"""Attachment chunks and images are item-level rows collapsed to their
segment; a threshold scan may read this many rows per allowed segment."""


# --------------------------------------------------------------------------
# request, settings, hits
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """One search, as the owner typed it. `sort` is not part of it: the
    same result set is served in every order."""

    query: str
    people: tuple[str, ...] = ()
    after: str | None = None
    before: str | None = None
    attachments: AttachmentFilter = "any"
    sender: str = ""
    """A person's name, or `me`: who wrote the message."""
    direction: DirectionFilter = "any"
    """Sent by the owner, or received from anyone else."""
    chat_kind: ChatKindFilter = "any"
    thread: str = ""
    """One conversation's `thread_key`."""

    @property
    def has_filters(self) -> bool:
        return bool(
            self.people
            or self.after
            or self.before
            or self.attachments != "any"
            or self.sender.strip()
            or self.direction != "any"
            or self.chat_kind != "any"
            or self.thread
        )

    def validated(self) -> SearchRequest:
        query = self.query.strip()
        if not query and not self.has_filters:
            raise SearchInputError("Type something to search for, or pick a filter.")
        if len(query) > MAX_QUERY_CHARS:
            raise SearchInputError(f"The query is longer than {MAX_QUERY_CHARS} characters.")
        people = tuple(p.strip() for p in self.people if p.strip())
        if len(people) > MAX_PEOPLE_FILTER:
            raise SearchInputError(f"Filter by at most {MAX_PEOPLE_FILTER} people.")
        if self.attachments not in ATTACHMENT_FILTERS:
            raise SearchInputError("Unknown attachment filter.")
        if self.direction not in DIRECTION_FILTERS:
            raise SearchInputError("Unknown 'sent by' filter.")
        if self.chat_kind not in CHAT_KIND_FILTERS:
            raise SearchInputError("Unknown conversation filter.")
        sender = " ".join(self.sender.split())
        if len(sender) > 200:
            raise SearchInputError("The 'sent by' name is too long.")
        direction = self.direction
        if sender.lower() == OWNER_SENDER:
            sender, direction = "", ("sent" if direction == "any" else direction)
            if direction != "sent":
                raise SearchInputError("'Sent by me' and 'received' cannot both apply.")
        elif sender and direction == "sent":
            raise SearchInputError(f"A message sent by {sender} was not sent by you: pick one.")
        if self.thread and not is_opaque_key(self.thread):
            raise SearchInputError("Unknown conversation.")
        return SearchRequest(
            query=query,
            people=people,
            after=(self.after or None),
            before=(self.before or None),
            attachments=self.attachments,
            sender=sender,
            direction=direction,
            chat_kind=self.chat_kind,
            thread=self.thread,
        )

    def cache_key(self) -> str:
        payload = json.dumps(
            [
                self.query, list(self.people), self.after, self.before, self.attachments,
                self.sender, self.direction, self.chat_kind, self.thread,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def has_attachment(self) -> bool | None:
        return {"any": None, "with": True, "without": False}[self.attachments]


@dataclass(frozen=True, slots=True)
class SearchSettings:
    timezone: str
    index_unsent: bool
    rrf_k: int
    fts_max_hits: int
    unindexed_window_days: int
    semantic_enabled: bool
    multimodal_enabled: bool
    text_min_similarity: float
    multimodal_min_similarity: float
    max_hits_per_channel: int
    ef_search: int
    max_scan_tuples: int
    index_edit_history: bool = False
    """`policy.index_edit_history`: earlier versions of edited messages are
    in the index, so a match on one counts as a match on content."""

    @classmethod
    def from_config(cls, cfg: Config) -> SearchSettings:
        page = cfg.search_page
        return cls(
            timezone=cfg.render.timezone,
            index_unsent=cfg.policy.index_unsent,
            rrf_k=cfg.retrieval.rrf_k,
            fts_max_hits=page.fts_max_hits,
            unindexed_window_days=page.unindexed_window_days,
            semantic_enabled=page.semantic.enabled,
            multimodal_enabled=cfg.embedding.multimodal.enabled,
            text_min_similarity=page.semantic.text_min_similarity,
            multimodal_min_similarity=page.semantic.multimodal_min_similarity,
            max_hits_per_channel=page.semantic.max_hits_per_channel,
            ef_search=page.semantic.ef_search,
            max_scan_tuples=page.semantic.max_scan_tuples,
            index_edit_history=cfg.policy.index_edit_history,
        )


@dataclass(slots=True)
class Hit:
    """One viable hit: a segment, or a message not yet in any segment.

    `at` dates the hit for sorting and for its conversation's "latest":
    the latest matching message for a word match, otherwise the segment's
    start, moved into the date range when the search has one. What the
    page shows as the hit's time comes from the messages it shows."""

    key: str
    chat_id: int
    at: datetime
    ended_at: datetime
    segment_id: int | None = None
    message_id: int | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    score: float = 0.0
    rerank_score: float | None = None
    _similarity: dict[str, float] | None = None
    _matched: set[tuple[int, int]] | None = None
    """Allocated only for the hits that need them: a broad query can hold
    tens of thousands of hits in the result cache."""

    @property
    def similarity(self) -> Mapping[str, float]:
        """Cosine similarity per semantic channel that found this hit."""
        return self._similarity if self._similarity is not None else {}

    def note_similarity(self, channel: str, value: float) -> None:
        if self._similarity is None:
            self._similarity = {}
        if value > self._similarity.get(channel, -2.0):
            self._similarity[channel] = value

    @property
    def matched_attachments(self) -> AbstractSet[tuple[int, int]]:
        """`(message_id, attachment_id)` pairs whose text or image matched."""
        return self._matched if self._matched is not None else frozenset()

    def add_matched(self, pairs: AbstractSet[tuple[int, int]]) -> None:
        if not pairs:
            return
        if self._matched is None:
            self._matched = set()
        self._matched |= pairs

    @property
    def channels(self) -> list[str]:
        order = FULLTEXT_CHANNELS + SEMANTIC_CHANNELS
        return [c for c in order if c in self.ranks]

    @property
    def best_similarity(self) -> float | None:
        return max(self.similarity.values()) if self.similarity else None


def segment_hit_key(segment_id: int) -> str:
    return f"s{segment_id}"


def message_hit_key(message_id: int) -> str:
    return f"m{message_id}"


@dataclass(slots=True)
class ThreadHits:
    chat_id: int
    hits: list[Hit]
    best_score: float
    latest_at: datetime
    best_rerank: float | None = None

    @property
    def count(self) -> int:
        return len(self.hits)


@dataclass(slots=True)
class SemanticStatus:
    state: Literal["pending", "done", "unavailable", "disabled"]
    note: str | None = None
    added_hits: int = 0
    added_threads: int = 0


@dataclass(slots=True)
class SearchResult:
    request: SearchRequest
    analyzed: AnalyzedQuery | None
    """`None` for a search with no words (the filters alone)."""
    filters: PageFilters
    predicate: CompiledPredicate
    rrf_k: int
    message_predicate: CompiledPredicate | None = None
    """`predicate` without the "Sent by" test over the whole segment: for
    the channels that test each matching message itself (`None`: the same
    as `predicate`)."""
    hits: dict[str, Hit] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    capped: set[str] = field(default_factory=set)
    timings_ms: dict[str, float] = field(default_factory=dict)
    semantic: SemanticStatus = field(default_factory=lambda: SemanticStatus("pending"))
    reranked: bool = False
    non_content_segments: frozenset[int] = frozenset()
    """Segments the full-text index matched only in text nobody wrote (names
    and times in the header lines, message labels): the text channel left
    them out."""
    created_monotonic: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)
    """Held while the semantic merge or the optional rerank changes the
    hits, and by every reader that groups or pages them: requests for the
    same result run on different threads."""
    _orders: dict[str, list[ThreadHits]] = field(default_factory=dict)

    @property
    def total_hits(self) -> int:
        return len(self.hits)

    @property
    def hidden_non_content(self) -> int:
        """How many of `non_content_segments` no other channel found either."""
        return sum(1 for sid in self.non_content_segments if segment_hit_key(sid) not in self.hits)

    @property
    def per_message(self) -> CompiledPredicate:
        return self.message_predicate if self.message_predicate is not None else self.predicate

    def invalidate(self) -> None:
        self._orders.clear()

    def threads(self, sort: SortOrder) -> list[ThreadHits]:
        with self.lock:
            cached = self._orders.get(sort)
            if cached is None:
                cached = group_by_thread(list(self.hits.values()), sort)
                self._orders[sort] = cached
            return cached

    def thread(self, chat_id: int, sort: SortOrder) -> ThreadHits | None:
        for group in self.threads(sort):
            if group.chat_id == chat_id:
                return group
        return None


def _score(hit: Hit, rrf_k: int) -> float:
    return sum(1.0 / (rrf_k + rank + 1) for rank in hit.ranks.values())


def rescore(result: SearchResult) -> None:
    for hit in result.hits.values():
        hit.score = _score(hit, result.rrf_k)
    result.invalidate()


def _hit_order_key(hit: Hit, sort: SortOrder) -> tuple[float, float, float]:
    ts = hit.at.timestamp()
    if sort == "date":
        return (-ts, -hit.score, 0.0)
    if sort == "rerank" and hit.rerank_score is not None:
        return (-1.0, -hit.rerank_score, -ts)
    return (0.0, -hit.score, -ts)


def group_by_thread(hits: Sequence[Hit] | Any, sort: SortOrder) -> list[ThreadHits]:
    """Hits grouped by conversation. Relevance: conversations by their best
    hit's score. Date: by their most recent hit. Rerank: conversations
    holding reranked hits first, by the best rerank score."""
    groups: dict[int, list[Hit]] = {}
    for hit in hits:
        groups.setdefault(hit.chat_id, []).append(hit)
    threads: list[ThreadHits] = []
    for chat_id, members in groups.items():
        members.sort(key=lambda h: _hit_order_key(h, sort))
        reranks = [h.rerank_score for h in members if h.rerank_score is not None]
        threads.append(
            ThreadHits(
                chat_id=chat_id,
                hits=members,
                best_score=max(h.score for h in members),
                latest_at=max(h.at for h in members),
                best_rerank=max(reranks) if reranks else None,
            )
        )
    if sort == "date":
        threads.sort(key=lambda t: (-t.latest_at.timestamp(), -t.best_score))
    elif sort == "rerank":
        threads.sort(
            key=lambda t: (
                0 if t.best_rerank is not None else 1,
                -(t.best_rerank or 0.0),
                -t.best_score,
                -t.latest_at.timestamp(),
            )
        )
    else:
        threads.sort(key=lambda t: (-t.best_score, -t.latest_at.timestamp()))
    return threads


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

FULL_SCOPE = resolve_request_scope(None, LOCAL_FULL_ACCESS)


@dataclass(frozen=True, slots=True)
class PageFilters:
    """The retrieval layer's resolved filters (people, dates, attachments)
    plus the page's own: who sent a message, which kind of conversation,
    and one conversation. The MCP tools' `SearchFilters` is unchanged."""

    base: SearchFilters = field(default_factory=SearchFilters)
    sender: int | None = None
    """`person_id` of the person who wrote the message (never the owner)."""
    from_me: bool | None = None
    """True: sent by the owner. False: received (anyone else)."""
    chat_kind: str | None = None
    """`dm` (one-to-one, not the unfiled holding conversations) or `group`."""
    chat_id: int | None = None

    @property
    def after(self) -> datetime | None:
        return self.base.after

    @property
    def before(self) -> datetime | None:
        return self.base.before

    @property
    def people(self) -> tuple[int, ...]:
        return self.base.people

    @property
    def has_attachment(self) -> bool | None:
        return self.base.has_attachment

    @property
    def by_sender(self) -> bool:
        """A filter on who wrote each message."""
        return self.sender is not None or self.from_me is not None

    def keeps(self, *, sent_at: datetime, is_from_me: bool, sender_person_id: int | None) -> bool:
        """The per-message filters, in Python, for the messages a hit shows."""
        if self.after is not None and sent_at < self.after:
            return False
        if self.before is not None and sent_at >= self.before:
            return False
        if self.from_me is not None and is_from_me != self.from_me:
            return False
        return self.sender is None or (not is_from_me and sender_person_id == self.sender)


def _is_owner(pg: psycopg.Connection, person_id: int) -> bool:
    with pg.cursor() as cur:
        cur.execute("SELECT is_owner FROM person WHERE person_id = %s", (person_id,))
        row = cur.fetchone()
    return bool(row and row[0])


def _resolve_sender(pg: psycopg.Connection, name: str) -> int:
    person_ids = resolve_people(pg, FULL_SCOPE, [name])
    return int(person_ids[0])


def resolve_request_filters(
    pg: psycopg.Connection, request: SearchRequest, settings: SearchSettings
) -> PageFilters:
    """The retrieval layer's own filter resolution, with its errors turned
    into messages the page shows, and the page's own filters."""
    try:
        base = resolve_filters(
            pg,
            FULL_SCOPE,
            people=list(request.people),
            after=request.after,
            before=request.before,
            has_attachment=request.has_attachment,
            timezone=settings.timezone,
        )
        sender = _resolve_sender(pg, request.sender) if request.sender else None
    except PersonAmbiguousError as exc:
        names = ", ".join(f"{c.display_name} ({c.short_name})" for c in exc.candidates)
        raise SearchInputError(
            f"More than one person matches {exc.query!r}: {names}. Pick one."
        ) from exc
    except PersonNotFoundError as exc:
        raise SearchInputError(f"No person matches {exc.query!r}.") from exc
    except DateRangeInvalidError as exc:
        raise SearchInputError("The 'from' date must be before the 'to' date.") from exc
    except InvalidArgumentError as exc:
        raise SearchInputError(str(exc)) from exc
    chat_id: int | None = None
    if request.thread:
        with pg.cursor() as cur:
            cur.execute("SELECT chat_id FROM chat WHERE thread_key = %s", (request.thread,))
            row = cur.fetchone()
        if row is None:
            raise SearchInputError("That conversation is not in the index.")
        chat_id = int(row[0])
    from_me = {"any": None, "sent": True, "received": False}[request.direction]
    if sender is not None and _is_owner(pg, sender):
        if from_me is False:
            raise SearchInputError("'Sent by' names you, and 'received' excludes you: pick one.")
        sender, from_me = None, True
    elif sender is not None:
        if from_me is True:
            raise SearchInputError(f"A message sent by {request.sender} was not sent by you: pick one.")
        from_me = None  # a named person's messages are received ones by definition
    return PageFilters(
        base=base,
        sender=sender,
        from_me=from_me,
        chat_kind=None if request.chat_kind == "any" else request.chat_kind,
        chat_id=chat_id,
    )


def message_clause(filters: PageFilters, *, alias: str = "m", prefix: str = "mt") -> CompiledPredicate:
    """The per-message filters over one `message` row (`TRUE` without any):
    the message's own time, and who sent it."""
    clauses = ["TRUE"]
    params: dict[str, object] = {}
    if filters.after is not None:
        clauses.append(f"{alias}.sent_at >= %({prefix}_after)s")
        params[f"{prefix}_after"] = filters.after
    if filters.before is not None:
        clauses.append(f"{alias}.sent_at < %({prefix}_before)s")
        params[f"{prefix}_before"] = filters.before
    if filters.from_me is not None:
        clauses.append(f"{alias}.is_from_me" if filters.from_me else f"NOT {alias}.is_from_me")
    if filters.sender is not None:
        clauses.append(f"{alias}.sender_person_id = %({prefix}_sender)s AND NOT {alias}.is_from_me")
        params[f"{prefix}_sender"] = filters.sender
    return CompiledPredicate(sql=" AND ".join(clauses), params=params)


message_time_clause = message_clause
"""The earlier name, from when the only per-message filter was the date."""


def _chat_clauses(filters: PageFilters, chat_column: str, prefix: str) -> tuple[list[str], dict[str, object]]:
    clauses: list[str] = []
    params: dict[str, object] = {}
    if filters.chat_id is not None:
        clauses.append(f"{chat_column} = %({prefix}_chat)s")
        params[f"{prefix}_chat"] = filters.chat_id
    if filters.chat_kind == "group":
        clauses.append(
            f"EXISTS (SELECT 1 FROM chat __ck WHERE __ck.chat_id = {chat_column} AND __ck.kind = 'group')"
        )
    elif filters.chat_kind == "dm":
        clauses.append(
            f"EXISTS (SELECT 1 FROM chat __ck WHERE __ck.chat_id = {chat_column} "
            "AND __ck.kind = 'dm' AND __ck.unfiled_key IS NULL)"
        )
    return clauses, params


def page_predicate(filters: PageFilters, *, sender_exists: bool = True) -> CompiledPredicate:
    """The page's filter over a `segment s`: people and attachments as the
    retrieval layer compiles them; dates by overlap. A segment that began
    on 30 Nov and ended on 2 Dec is in a search for 2 Dec. Which of its
    messages count is decided per message (`message_clause`).

    With a "Sent by" filter the segment must also hold a message that
    passes every per-message filter (`sender_exists`), so a segment found
    by meaning is one the person wrote in; the channels that test each
    matching message on its own skip it."""
    base = compile_predicate(replace(filters.base, after=None, before=None), FULL_SCOPE)
    clauses = [base.sql]
    params = dict(base.params)
    if filters.after is not None:
        clauses.append("s.ended_at >= %(pg_after)s")
        params["pg_after"] = filters.after
    if filters.before is not None:
        clauses.append("s.started_at < %(pg_before)s")
        params["pg_before"] = filters.before
    chat, chat_params = _chat_clauses(filters, "s.chat_id", "pg")
    clauses += chat
    params.update(chat_params)
    if sender_exists and filters.by_sender:
        inner = message_clause(filters, alias="__pm", prefix="pe")
        clauses.append(
            "EXISTS (SELECT 1 FROM segment_message __psm JOIN message __pm "
            "ON __pm.message_id = __psm.message_id "
            f"WHERE __psm.segment_id = s.segment_id AND {inner.sql})"
        )
        params.update(inner.params)
    return CompiledPredicate(sql=" AND ".join(clauses), params=params)


def _as_filter(predicate: CompiledPredicate) -> tuple[str, Mapping[str, object]] | None:
    return None if predicate.sql == "TRUE" else (predicate.sql, predicate.params)


def _message_join(in_range: CompiledPredicate, *, on: str) -> str:
    """Join `message m` only when a per-message filter needs it: measured
    at 52 ms on a 20,000-hit attachment search that has no date."""
    return f"JOIN message m ON m.message_id = {on}" if in_range.sql != "TRUE" else ""


def _message_predicate(filters: PageFilters, *, alias: str = "m") -> CompiledPredicate:
    """The same filters, over a `message` row (for messages in no segment)."""
    own = message_clause(filters, alias=alias, prefix="mf")
    clauses: list[str] = [own.sql]
    params: dict[str, object] = dict(own.params)
    if filters.has_attachment is not None:
        clauses.append(f"{alias}.has_attachments = %(mf_att)s")
        params["mf_att"] = filters.has_attachment
    for i, person_id in enumerate(filters.people):
        clauses.append(
            f"EXISTS (SELECT 1 FROM chat_participant __mcp_{i} "
            f"WHERE __mcp_{i}.chat_id = {alias}.chat_id AND __mcp_{i}.person_id = %(mf_p{i})s)"
        )
        params[f"mf_p{i}"] = person_id
    chat, chat_params = _chat_clauses(filters, f"{alias}.chat_id", "mf")
    clauses += chat
    params.update(chat_params)
    return CompiledPredicate(sql=" AND ".join(clauses), params=params)


# --------------------------------------------------------------------------
# full-text channels
# --------------------------------------------------------------------------


def _fts_rowids(fts: apsw.Connection, table: str, match: str, cap: int) -> list[int]:
    rows = fts.execute(
        f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?", (match, cap + 1)
    ).fetchall()
    return [int(cast(int, r[0])) for r in rows]


def _match_expression(analyzed: AnalyzedQuery) -> str | None:
    if analyzed.mode == "bm25":
        return bm25_match_expression(analyzed.phrase)
    if analyzed.mode == "trigram":
        return trigram_match_expression(analyzed.phrase)
    return None


def _segment_rows(
    pg: psycopg.Connection, ids: list[int], predicate: CompiledPredicate
) -> dict[int, tuple[int, datetime, datetime]]:
    if not ids:
        return {}
    with pg.cursor() as cur:
        cur.execute(
            f"SELECT s.segment_id, s.chat_id, s.started_at, s.ended_at FROM segment s "
            f"WHERE s.segment_id = ANY(%(ids)s::bigint[]) AND ({predicate.sql})",
            {"ids": ids, **predicate.params},
        )
        return {int(r[0]): (int(r[1]), r[2], r[3]) for r in cur.fetchall()}


def _add_segment(
    result: SearchResult,
    segment_id: int,
    chat_id: int,
    started_at: datetime,
    ended_at: datetime,
    channel: str,
    rank: int,
) -> Hit:
    key = segment_hit_key(segment_id)
    hit = result.hits.get(key)
    if hit is None:
        after, before = result.filters.after, result.filters.before
        if after is not None and started_at < after:
            started_at = after
        if before is not None and ended_at >= before:
            ended_at = max(started_at, before - timedelta(microseconds=1))
        hit = Hit(key=key, chat_id=chat_id, at=started_at, ended_at=ended_at, segment_id=segment_id)
        result.hits[key] = hit
    previous = hit.ranks.get(channel)
    if previous is None or rank < previous:
        hit.ranks[channel] = rank
    return hit


def _segment_fulltext(
    pg: psycopg.Connection,
    fts: apsw.Connection,
    result: SearchResult,
    settings: SearchSettings,
) -> None:
    analyzed = result.analyzed
    assert analyzed is not None
    cap = settings.fts_max_hits
    match = _match_expression(analyzed)
    if match is not None:
        table = "seg_fts_tri" if analyzed.mode == "trigram" else "seg_fts"
        started = time.perf_counter()
        ids = _fts_rowids(fts, table, match, cap)
        result.timings_ms["fts_segments"] = (time.perf_counter() - started) * 1000
        if len(ids) > cap:
            result.capped.add(CHANNEL_TEXT)
            ids = ids[:cap]
        started = time.perf_counter()
        rows = _segment_rows(pg, ids, result.per_message)
        result.timings_ms["pg_segments"] = (time.perf_counter() - started) * 1000
        wanted = content_query(analyzed)
        matched: dict[int, ContentMatch] | None = None
        if wanted is not None:
            started = time.perf_counter()
            found = segments_matching_content(
                pg,
                [segment_id for segment_id in ids if segment_id in rows],
                wanted,
                index_unsent=settings.index_unsent,
                include_edit_history=settings.index_edit_history,
                message_filter=_as_filter(message_clause(result.filters, prefix="cmf")),
            )
            result.timings_ms["content_check"] = (time.perf_counter() - started) * 1000
            matched = found
            result.non_content_segments = frozenset(sid for sid in rows if sid not in found)
        rank = 0
        for segment_id in ids:
            row = rows.get(segment_id)
            if row is None or (matched is not None and segment_id not in matched):
                continue
            hit = _add_segment(result, segment_id, row[0], row[1], row[2], CHANNEL_TEXT, rank)
            if matched is not None:
                hit.add_matched(matched[segment_id].matched_attachments)
                hit.at = matched[segment_id].last_match
            rank += 1
        result.counts[CHANNEL_TEXT] = rank
        return

    # Emoji: unicode61 drops emoji as separators, so the index cannot
    # hold them; scan the message text itself (SPEC §7.3).
    started = time.perf_counter()
    in_range = message_clause(result.filters)
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT s.segment_id, s.chat_id, s.started_at, s.ended_at, max(m.sent_at) AS last_match
            FROM message m
            JOIN segment_message sm ON sm.message_id = m.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE m.text_original LIKE %(pattern)s ESCAPE '\\' AND ({result.per_message.sql})
              AND ({in_range.sql}) AND (%(unsent)s OR NOT m.is_unsent)
            GROUP BY s.segment_id
            ORDER BY last_match DESC
            LIMIT %(limit)s
            """,
            {
                "pattern": like_pattern(analyzed.phrase),
                "limit": cap + 1,
                "unsent": settings.index_unsent,
                **result.per_message.params,
                **in_range.params,
            },
        )
        emoji_rows = cur.fetchall()
    result.timings_ms["pg_segments"] = (time.perf_counter() - started) * 1000
    if len(emoji_rows) > cap:
        result.capped.add(CHANNEL_TEXT)
        emoji_rows = emoji_rows[:cap]
    for rank, (segment_id, chat_id, started_at, ended_at, _last) in enumerate(emoji_rows):
        _add_segment(result, int(segment_id), int(chat_id), started_at, ended_at, CHANNEL_TEXT, rank)
    result.counts[CHANNEL_TEXT] = len(emoji_rows)


def _fts_attachment_ids(fts: apsw.Connection, table: str, match: str, cap: int) -> list[int]:
    """Matching attachments in best-chunk order: chunk hits are collapsed to
    their attachment inside SQLite (`att_map` holds each chunk's
    attachment), so a many-page document counts once (SPEC §9.4 step 4)
    and Postgres maps attachments, not chunks, to segments."""
    rows = fts.execute(
        f"SELECT m.attachment_id, min(f.rank) AS best FROM {table} f "
        f"JOIN att_map m ON m.fts_rowid = f.rowid "
        f"WHERE {table} MATCH ? GROUP BY m.attachment_id ORDER BY best LIMIT ?",
        (match, cap + 1),
    ).fetchall()
    return [int(cast(int, r[0])) for r in rows]


def _attachment_fulltext(
    pg: psycopg.Connection,
    fts: apsw.Connection,
    result: SearchResult,
    settings: SearchSettings,
) -> None:
    analyzed = result.analyzed
    assert analyzed is not None
    cap = settings.fts_max_hits
    match = _match_expression(analyzed)
    started = time.perf_counter()
    if match is not None:
        table = "att_fts_tri" if analyzed.mode == "trigram" else "att_fts"
        attachment_ids = _fts_attachment_ids(fts, table, match, cap)
        result.timings_ms["fts_attachments"] = (time.perf_counter() - started) * 1000
        if len(attachment_ids) > cap:
            result.capped.add(CHANNEL_ATTACHMENT_TEXT)
            attachment_ids = attachment_ids[:cap]
        if not attachment_ids:
            result.counts[CHANNEL_ATTACHMENT_TEXT] = 0
            return
        candidate_sql = "SELECT * FROM unnest(%(ids)s::bigint[], %(ranks)s::int[])"
        params: dict[str, object] = {
            "ids": attachment_ids,
            "ranks": list(range(len(attachment_ids))),
        }
    else:
        candidate_sql = (
            "SELECT DISTINCT ON (ac.attachment_id) ac.attachment_id, 0 FROM attachment_chunk ac "
            "WHERE ac.text LIKE %(pattern)s ESCAPE '\\' LIMIT %(limit)s"
        )
        params = {"pattern": like_pattern(analyzed.phrase), "limit": cap + 1}
    started = time.perf_counter()
    in_range = message_clause(result.filters)
    with pg.cursor() as cur:
        cur.execute(
            f"""
            WITH candidate (attachment_id, rnk) AS ({candidate_sql})
            SELECT c.rnk, s.segment_id, s.chat_id, s.started_at, s.ended_at,
                   ma.message_id, ma.attachment_id
            FROM candidate c
            JOIN message_attachment ma ON ma.attachment_id = c.attachment_id
            {_message_join(in_range, on="ma.message_id")}
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE {result.per_message.sql} AND ({in_range.sql})
            """,
            {**params, **result.per_message.params, **in_range.params},
        )
        rows = cur.fetchall()
    result.timings_ms["pg_attachments"] = (time.perf_counter() - started) * 1000
    best: dict[int, tuple[int, int, datetime, datetime]] = {}
    matched: dict[int, set[tuple[int, int]]] = {}
    for rnk, segment_id, chat_id, started_at, ended_at, message_id, attachment_id in rows:
        sid = int(segment_id)
        matched.setdefault(sid, set()).add((int(message_id), int(attachment_id)))
        if sid not in best or int(rnk) < best[sid][0]:
            best[sid] = (int(rnk), int(chat_id), started_at, ended_at)
    ordered = sorted(best.items(), key=lambda kv: kv[1][0])
    for rank, (segment_id, (_rnk, chat_id, started_at, ended_at)) in enumerate(ordered):
        hit = _add_segment(
            result, segment_id, chat_id, started_at, ended_at, CHANNEL_ATTACHMENT_TEXT, rank
        )
        hit.add_matched(matched.get(segment_id, set()))
    result.counts[CHANNEL_ATTACHMENT_TEXT] = len(ordered)


def _unindexed_text_clause(analyzed: AnalyzedQuery) -> tuple[str, dict[str, object]]:
    """A text condition over `message m` that reproduces the index's
    matching for rows the index does not hold yet
    (`imsg.search_page.content_match`): whole words with case and accents
    folded for a plain search, a substring with case folded for a quoted
    phrase, a plain substring for emoji."""
    if analyzed.mode == "emoji":
        return "m.text_original LIKE %(ux_emoji)s ESCAPE '\\'", {
            "ux_emoji": like_pattern(analyzed.phrase)
        }
    wanted = content_query(analyzed)
    assert wanted is not None
    return text_condition(wanted, "coalesce(m.text_normalized, m.text_original)", prefix="ux_t")


def _unindexed_messages(
    pg: psycopg.Connection, result: SearchResult, settings: SearchSettings
) -> None:
    if settings.unindexed_window_days <= 0:
        result.counts[CHANNEL_UNINDEXED] = 0
        return
    since = datetime.now(UTC) - timedelta(days=settings.unindexed_window_days)
    assert result.analyzed is not None
    text_sql, text_params = _unindexed_text_clause(result.analyzed)
    message_predicate = _message_predicate(result.filters)
    cap = settings.fts_max_hits
    started = time.perf_counter()
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT m.message_id, m.chat_id, m.sent_at
            FROM message m
            WHERE m.sent_at >= %(ux_since)s
              AND NOT EXISTS (SELECT 1 FROM segment_message sm WHERE sm.message_id = m.message_id)
              AND (%(ux_unsent)s OR NOT m.is_unsent)
              AND ({message_predicate.sql})
              AND ({text_sql})
            ORDER BY m.sent_at DESC, m.message_id DESC
            LIMIT %(ux_limit)s
            """,
            {
                "ux_since": since,
                "ux_unsent": settings.index_unsent,
                "ux_limit": cap + 1,
                **message_predicate.params,
                **text_params,
            },
        )
        rows = cur.fetchall()
    result.timings_ms["pg_unindexed"] = (time.perf_counter() - started) * 1000
    if len(rows) > cap:
        result.capped.add(CHANNEL_UNINDEXED)
        rows = rows[:cap]
    for rank, (message_id, chat_id, sent_at) in enumerate(rows):
        key = message_hit_key(int(message_id))
        result.hits[key] = Hit(
            key=key,
            chat_id=int(chat_id),
            at=sent_at,
            ended_at=sent_at,
            message_id=int(message_id),
            ranks={CHANNEL_UNINDEXED: rank},
        )
    result.counts[CHANNEL_UNINDEXED] = len(rows)


def _filtered_messages(
    pg: psycopg.Connection, result: SearchResult, settings: SearchSettings
) -> None:
    """A search with no words: every segment holding a message the filters
    keep, dated by its latest such message, and every such message that is
    in no segment yet (of any age: no words means no index to lag behind),
    newest first, up to `search_page.fts_max_hits`."""
    cap = settings.fts_max_hits
    own = message_clause(result.filters)
    started = time.perf_counter()
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT s.segment_id, s.chat_id, s.started_at, s.ended_at, max(m.sent_at) AS last_match
            FROM message m
            JOIN segment_message sm ON sm.message_id = m.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE ({result.per_message.sql}) AND ({own.sql}) AND (%(fo_unsent)s OR NOT m.is_unsent)
            GROUP BY s.segment_id
            ORDER BY last_match DESC, s.segment_id DESC
            LIMIT %(fo_limit)s
            """,
            {
                "fo_unsent": settings.index_unsent,
                "fo_limit": cap + 1,
                **result.per_message.params,
                **own.params,
            },
            prepare=False,
        )
        segment_rows = cur.fetchall()
        loose = _message_predicate(result.filters)
        cur.execute(
            f"""
            SELECT m.message_id, m.chat_id, m.sent_at
            FROM message m
            WHERE NOT EXISTS (SELECT 1 FROM segment_message sm WHERE sm.message_id = m.message_id)
              AND (%(fo_unsent)s OR NOT m.is_unsent)
              AND ({loose.sql})
            ORDER BY m.sent_at DESC, m.message_id DESC
            LIMIT %(fo_limit)s
            """,
            {"fo_unsent": settings.index_unsent, "fo_limit": cap + 1, **loose.params},
            prepare=False,
        )
        message_rows = cur.fetchall()
    result.timings_ms["pg_filters"] = (time.perf_counter() - started) * 1000
    if len(segment_rows) > cap or len(message_rows) > cap:
        result.capped.add(CHANNEL_FILTERS)
    for rank, (segment_id, chat_id, started_at, ended_at, last) in enumerate(segment_rows[:cap]):
        hit = _add_segment(result, int(segment_id), int(chat_id), started_at, ended_at, CHANNEL_FILTERS, rank)
        hit.at = last
    for rank, (message_id, chat_id, sent_at) in enumerate(message_rows[:cap]):
        key = message_hit_key(int(message_id))
        result.hits[key] = Hit(
            key=key,
            chat_id=int(chat_id),
            at=sent_at,
            ended_at=sent_at,
            message_id=int(message_id),
            ranks={CHANNEL_FILTERS: rank},
        )
    result.counts[CHANNEL_FILTERS] = min(len(segment_rows), cap) + min(len(message_rows), cap)


def run_fulltext(
    pg: psycopg.Connection,
    fts: apsw.Connection,
    request: SearchRequest,
    settings: SearchSettings,
) -> SearchResult:
    """Every full-text hit (no model involved): the first paint. With no
    words, every message the filters keep (`_filtered_messages`)."""
    started = time.perf_counter()
    request = request.validated()
    analyzed = analyze_query(request.query) if request.query else None
    filters = resolve_request_filters(pg, request, settings)
    predicate = page_predicate(filters)
    if analyzed is None:
        semantic = SemanticStatus("disabled", note="no words to search by meaning")
    else:
        semantic = SemanticStatus("pending" if settings.semantic_enabled else "disabled")
    result = SearchResult(
        request=request,
        analyzed=analyzed,
        filters=filters,
        predicate=predicate,
        rrf_k=settings.rrf_k,
        message_predicate=page_predicate(filters, sender_exists=False) if filters.by_sender else None,
        semantic=semantic,
    )
    result.timings_ms["filters"] = (time.perf_counter() - started) * 1000
    if analyzed is None:
        _filtered_messages(pg, result, settings)
    else:
        _segment_fulltext(pg, fts, result, settings)
        _attachment_fulltext(pg, fts, result, settings)
        _unindexed_messages(pg, result, settings)
    rescore(result)
    result.timings_ms["fulltext_total"] = (time.perf_counter() - started) * 1000
    return result


# --------------------------------------------------------------------------
# semantic channels
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryVectors:
    text: list[float] | None
    multimodal: list[float] | None


@dataclass(frozen=True, slots=True)
class ThresholdRows:
    rows: list[tuple[Any, ...]]
    capped: bool


def _apply_threshold_settings(cur: Any, settings: SearchSettings) -> None:
    """The planner settings every threshold scan runs under, for the rest
    of its transaction.

    `enable_sort = off` as well as `enable_seqscan = off` (the retrieval
    layer's remedy, `imsg.retrieval.vector_search`): a threshold scan's
    `LIMIT` is a safety cap in the thousands, and at that size the planner
    prefers joining every row and sorting by distance to walking the HNSW
    index. Measured on the synthetic production-sized corpus (60,000 chunk
    vectors, 2026-09-24): `LIMIT 10000` chose a parallel hash join plus a
    sort of every vector, and the chunk scan took 520 ms; with sorting
    disabled the index is the only ordered path, and reading stops at the
    first row past the floor. Both are penalties, not prohibitions: with no
    usable index the planner still sorts."""
    cur.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
    cur.execute("SET LOCAL enable_seqscan = off")
    cur.execute("SET LOCAL enable_sort = off")
    cur.execute("SET LOCAL cursor_tuple_fraction = 1.0")
    cur.execute(
        "SELECT set_config('hnsw.ef_search', %(v)s, true)", {"v": str(int(settings.ef_search))}
    )
    cur.execute(
        "SELECT set_config('hnsw.max_scan_tuples', %(v)s, true)",
        {"v": str(int(settings.max_scan_tuples))},
    )


def threshold_scan(
    pg: psycopg.Connection,
    sql: str,
    params: dict[str, Any],
    *,
    max_distance: float,
    row_limit: int,
    settings: SearchSettings,
) -> ThresholdRows:
    """Stream `sql` (ordered by distance, distance last in each row) and
    keep every row at or under `max_distance`, stopping at the first row
    past it (plus the ordering tolerance) or at `row_limit` rows."""
    kept: list[tuple[Any, ...]] = []
    read = 0
    stopped_by_threshold = False
    with pg.transaction():
        with pg.cursor() as cur:
            _apply_threshold_settings(cur, settings)
        with pg.cursor(name="imsg_search_page_threshold") as cur:
            cur.execute(sql, {**params, "row_limit": row_limit})
            while not stopped_by_threshold:
                batch = cur.fetchmany(THRESHOLD_FETCH_ROWS)
                for row in batch:
                    read += 1
                    distance = float(row[-1])
                    if distance > max_distance + DISTANCE_ORDER_TOLERANCE:
                        stopped_by_threshold = True
                        break
                    if distance <= max_distance:
                        kept.append(tuple(row))
                if len(batch) < THRESHOLD_FETCH_ROWS:
                    break
    return ThresholdRows(rows=kept, capped=(not stopped_by_threshold and read >= row_limit))


def _merge_vector_rows(
    result: SearchResult,
    channel: str,
    rows: list[tuple[Any, ...]],
    *,
    cap: int,
    with_attachment: bool,
) -> int:
    """Collapse `(segment_id, chat_id, started_at, ended_at, [message_id,
    attachment_id,] distance)` rows to segments in best-distance order."""
    best: dict[int, tuple[float, int, datetime, datetime]] = {}
    matched: dict[int, set[tuple[int, int]]] = {}
    for row in rows:
        segment_id, chat_id, started_at, ended_at = int(row[0]), int(row[1]), row[2], row[3]
        distance = float(row[-1])
        if with_attachment:
            matched.setdefault(segment_id, set()).add((int(row[4]), int(row[5])))
        if segment_id not in best or distance < best[segment_id][0]:
            best[segment_id] = (distance, chat_id, started_at, ended_at)
    ordered = sorted(best.items(), key=lambda kv: kv[1][0])[:cap]
    for rank, (segment_id, (distance, chat_id, started_at, ended_at)) in enumerate(ordered):
        hit = _add_segment(result, segment_id, chat_id, started_at, ended_at, channel, rank)
        hit.note_similarity(channel, 1.0 - distance)
        if with_attachment:
            hit.add_matched(matched.get(segment_id, set()))
    result.counts[channel] = len(ordered)
    return len(ordered)


def run_semantic(
    pg: psycopg.Connection,
    result: SearchResult,
    vectors: QueryVectors,
    settings: SearchSettings,
) -> SemanticStatus:
    """Merge every semantic hit above the thresholds into `result`."""
    before_hits = set(result.hits)
    before_threads = {h.chat_id for h in result.hits.values()}
    cap = settings.max_hits_per_channel
    predicate = result.predicate
    per_message = result.per_message
    in_range = message_clause(result.filters)
    text_distance = 1.0 - settings.text_min_similarity
    started = time.perf_counter()
    notes: list[str] = []

    if vectors.text is not None:
        qv = vector_literal(vectors.text)
        rows = threshold_scan(
            pg,
            f"""
            SELECT s.segment_id, s.chat_id, s.started_at, s.ended_at,
                   se.vec <=> %(qv)s::halfvec AS distance
            FROM segment_embedding se
            JOIN segment s ON s.segment_id = se.segment_id
            WHERE {predicate.sql}
            ORDER BY se.vec <=> %(qv)s::halfvec
            LIMIT %(row_limit)s
            """,
            {"qv": qv, **predicate.params},
            max_distance=text_distance,
            row_limit=cap + 1,
            settings=settings,
        )
        if rows.capped or len(rows.rows) > cap:
            result.capped.add(CHANNEL_SEMANTIC)
        _merge_vector_rows(result, CHANNEL_SEMANTIC, rows.rows, cap=cap, with_attachment=False)
        result.timings_ms["pg_semantic"] = (time.perf_counter() - started) * 1000

        chunk_started = time.perf_counter()
        rows = threshold_scan(
            pg,
            f"""
            SELECT s.segment_id, s.chat_id, s.started_at, s.ended_at,
                   ma.message_id, ac.attachment_id,
                   ace.vec <=> %(qv)s::halfvec AS distance
            FROM attachment_chunk_embedding ace
            JOIN attachment_chunk ac ON ac.chunk_id = ace.chunk_id
            JOIN message_attachment ma ON ma.attachment_id = ac.attachment_id
            {_message_join(in_range, on="ma.message_id")}
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE {per_message.sql} AND ({in_range.sql})
            ORDER BY ace.vec <=> %(qv)s::halfvec
            LIMIT %(row_limit)s
            """,
            {"qv": qv, **per_message.params, **in_range.params},
            max_distance=text_distance,
            row_limit=cap * ROWS_PER_SEGMENT_OVERFETCH,
            settings=settings,
        )
        if rows.capped:
            result.capped.add(CHANNEL_ATTACHMENT_SEMANTIC)
        _merge_vector_rows(
            result, CHANNEL_ATTACHMENT_SEMANTIC, rows.rows, cap=cap, with_attachment=True
        )
        result.timings_ms["pg_attachment_semantic"] = (time.perf_counter() - chunk_started) * 1000
    else:
        notes.append("text vectors unavailable")

    if vectors.multimodal is not None and settings.multimodal_enabled:
        mm_started = time.perf_counter()
        rows = threshold_scan(
            pg,
            f"""
            SELECT s.segment_id, s.chat_id, s.started_at, s.ended_at,
                   ma.message_id, mm.attachment_id,
                   mm.vec <=> %(qv)s::halfvec AS distance
            FROM attachment_mm_embedding mm
            JOIN message_attachment ma ON ma.attachment_id = mm.attachment_id
            {_message_join(in_range, on="ma.message_id")}
            JOIN segment_message sm ON sm.message_id = ma.message_id
            JOIN segment s ON s.segment_id = sm.segment_id
            WHERE {per_message.sql} AND ({in_range.sql})
            ORDER BY mm.vec <=> %(qv)s::halfvec
            LIMIT %(row_limit)s
            """,
            {"qv": vector_literal(vectors.multimodal), **per_message.params, **in_range.params},
            max_distance=1.0 - settings.multimodal_min_similarity,
            row_limit=cap * ROWS_PER_SEGMENT_OVERFETCH,
            settings=settings,
        )
        if rows.capped:
            result.capped.add(CHANNEL_IMAGE)
        _merge_vector_rows(result, CHANNEL_IMAGE, rows.rows, cap=cap, with_attachment=True)
        result.timings_ms["pg_image"] = (time.perf_counter() - mm_started) * 1000

    rescore(result)
    result.timings_ms["semantic_total"] = (time.perf_counter() - started) * 1000
    added = set(result.hits) - before_hits
    added_threads = {result.hits[k].chat_id for k in added} - before_threads
    status = SemanticStatus(
        state="done",
        note="; ".join(notes) or None,
        added_hits=len(added),
        added_threads=len(added_threads),
    )
    result.semantic = status
    return status


# --------------------------------------------------------------------------
# optional rerank of the best few
# --------------------------------------------------------------------------


def rerank_candidates(result: SearchResult, top: int) -> list[Hit]:
    """The best `top` segment hits by fused score: the only hits the
    optional rerank order ever sends to the reranker."""
    segments = [h for h in result.hits.values() if h.segment_id is not None]
    segments.sort(key=lambda h: (-h.score, -h.at.timestamp()))
    return segments[:top]


def apply_rerank(result: SearchResult, hits: list[Hit], scores: list[float]) -> None:
    for hit in result.hits.values():
        hit.rerank_score = None
    for hit, score in zip(hits, scores, strict=True):
        hit.rerank_score = float(score)
    result.reranked = True
    result.invalidate()


def segment_texts(pg: psycopg.Connection, segment_ids: list[int], *, max_chars: int) -> dict[int, str]:
    if not segment_ids:
        return {}
    with pg.cursor() as cur:
        cur.execute(
            "SELECT segment_id, left(rendered_text, %(n)s) FROM segment "
            "WHERE segment_id = ANY(%(ids)s::bigint[])",
            {"ids": segment_ids, "n": max_chars},
        )
        return {int(sid): str(text) for sid, text in cur.fetchall()}


# --------------------------------------------------------------------------
# a small bounded cache of result sets (pagination, semantic merge)
# --------------------------------------------------------------------------


class ResultCache:
    """The last few result sets, so paging, the semantic merge and a
    change of sort order reuse them. Bounded by entries and by total hits
    held, so a burst of broad queries cannot grow the process."""

    def __init__(
        self, *, max_entries: int = 8, max_total_hits: int = 50_000, ttl_seconds: float = 900.0
    ) -> None:
        self._max_entries = max_entries
        self._max_total_hits = max_total_hits
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._entries: dict[str, SearchResult] = {}

    def get(self, key: str) -> SearchResult | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if time.monotonic() - entry.created_monotonic > self._ttl:
                del self._entries[key]
                return None
            # Move to the end: most recently used.
            del self._entries[key]
            self._entries[key] = entry
            return entry

    def put(self, key: str, result: SearchResult) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = result
            while len(self._entries) > self._max_entries or (
                len(self._entries) > 1
                and sum(e.total_hits for e in self._entries.values()) > self._max_total_hits
            ):
                oldest = next(iter(self._entries))
                del self._entries[oldest]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = [
    "ATTACHMENT_FILTERS",
    "CHANNEL_ATTACHMENT_SEMANTIC",
    "CHANNEL_ATTACHMENT_TEXT",
    "CHANNEL_FILTERS",
    "CHANNEL_IMAGE",
    "CHANNEL_LABELS",
    "CHANNEL_SEMANTIC",
    "CHANNEL_TEXT",
    "CHANNEL_UNINDEXED",
    "CHAT_KIND_FILTERS",
    "DIRECTION_FILTERS",
    "FULLTEXT_CHANNELS",
    "OWNER_SENDER",
    "SEMANTIC_CHANNELS",
    "SORT_ORDERS",
    "Hit",
    "PageFilters",
    "QueryVectors",
    "ResultCache",
    "SearchRequest",
    "SearchResult",
    "SearchSettings",
    "SemanticStatus",
    "ThreadHits",
    "apply_rerank",
    "group_by_thread",
    "message_clause",
    "message_hit_key",
    "message_time_clause",
    "page_predicate",
    "rerank_candidates",
    "rescore",
    "run_fulltext",
    "run_semantic",
    "segment_hit_key",
    "segment_texts",
    "threshold_scan",
]
