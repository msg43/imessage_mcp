"""The page's date filter tests each message's own time.

A conversation segment that began on Thu 30 Nov and ran into Sat 2 Dec
used to be found by a 30 Nov filter only, because the filter tested when
the segment started. The page now keeps every segment that overlaps the
range, counts a word match only in messages sent inside it, shows only
those messages, and labels each hit with the matching message's time.
The MCP tools' own filter (`imsg.retrieval.filters`) is unchanged.

Synthetic data only; needs the scratch Postgres."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _search_page_fixtures import (
    REACHABLE,
    SKIP_REASON,
    Corpus,
    create_scratch_db,
    drop_scratch_db,
    make_page_client,
    open_fts,
    page_settings,
    unit_vector,
)
from imsg.retrieval.access import LOCAL_FULL_ACCESS, resolve_request_scope
from imsg.retrieval.filters import SearchFilters, compile_predicate
from imsg.search_page.search import (
    CHANNEL_ATTACHMENT_TEXT,
    CHANNEL_SEMANTIC,
    QueryVectors,
    SearchRequest,
    run_fulltext,
    run_semantic,
)
from imsg.search_page.server import open_fts_reader

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_dates_test"
STARTED = datetime(2017, 11, 30, 18, 46, tzinfo=UTC)
MATCHED = datetime(2017, 12, 2, 8, 19, tzinfo=UTC)


@pytest.fixture
def corpus(tmp_path: Path) -> Iterator[Corpus]:
    conn = create_scratch_db(DB_NAME)
    root = tmp_path / "data_root"
    (root / "fts").mkdir(parents=True)
    (root / "attachments").mkdir()
    fts = open_fts(root / "fts" / "fts.db")
    try:
        yield Corpus(conn=conn, fts=fts, data_root=root)
    finally:
        fts.close()
        conn.close()
        drop_scratch_db(DB_NAME)


def run(corpus: Corpus, query: str, day_from: str, day_to_exclusive: str, **settings: Any) -> Any:
    reader = open_fts_reader(corpus.data_root / "fts" / "fts.db")
    try:
        return run_fulltext(
            corpus.conn,
            reader,
            SearchRequest(query=query, after=day_from, before=day_to_exclusive),
            page_settings(**settings),
        )
    finally:
        reader.close()


def _two_day_segment(corpus: Corpus, *, second_text: str = "the talteyorma permit came") -> Any:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(
        chat,
        [
            (STARTED, alice, "starting the plan"),
            (STARTED + timedelta(minutes=5), None, "sounds good"),
            (MATCHED, alice, second_text),
        ],
    )
    return seg


def test_a_message_is_found_by_its_own_date(corpus: Corpus) -> None:
    seg = _two_day_segment(corpus)
    on_the_day = run(corpus, "talteyorma", "2017-12-02", "2017-12-03")
    assert {h.segment_id for h in on_the_day.hits.values()} == {seg.segment_id}
    # The segment began on 30 Nov, but the word was not written that day.
    assert run(corpus, "talteyorma", "2017-11-30", "2017-12-01").total_hits == 0
    # A word written on 30 Nov is found on 30 Nov only.
    assert run(corpus, "starting", "2017-11-30", "2017-12-01").total_hits == 1
    assert run(corpus, "starting", "2017-12-02", "2017-12-03").total_hits == 0


def test_the_mcp_filter_still_tests_when_a_segment_started() -> None:
    """The page's overlap test is its own; the retrieval layer's predicate,
    which the MCP tools use, is untouched."""
    scope = resolve_request_scope(None, LOCAL_FULL_ACCESS)
    predicate = compile_predicate(SearchFilters(after=MATCHED, before=MATCHED), scope)
    assert "s.started_at >= %(f_after)s" in predicate.sql
    assert "s.started_at < %(f_before)s" in predicate.sql


def test_only_messages_inside_the_range_are_shown_and_the_hit_carries_their_time(
    corpus: Corpus,
) -> None:
    _two_day_segment(corpus)
    client = make_page_client(corpus, DB_NAME)
    body = client.get(
        "/search", params={"q": "talteyorma", "from": "2017-12-02", "to": "2017-12-02"}
    ).text
    assert "1 hit" in body
    assert "<mark>talteyorma</mark>" in body
    assert "starting the plan" not in body and "sounds good" not in body
    assert "Sat 02 Dec 2017, 08:19" in body
    assert "Thu 30 Nov 2017" not in body
    # Without a date filter the hit is labelled with the matching message's
    # time too, not the segment's first message.
    unfiltered = client.get("/search", params={"q": "talteyorma"}).text
    assert 'class="when"' in unfiltered and "Sat 02 Dec 2017, 08:19</a>" in unfiltered


def test_attachment_text_is_dated_by_its_message(corpus: Corpus) -> None:
    seg = _two_day_segment(corpus, second_text="scan attached")
    att = corpus.attachment(
        seg.messages[2], filename="scan.pdf", mime_type="application/pdf", content=b"%PDF-1.4"
    )
    corpus.chunk(att, "invoice from the quarrelsome glazier")
    on_the_day = run(corpus, "glazier", "2017-12-02", "2017-12-03")
    assert on_the_day.counts[CHANNEL_ATTACHMENT_TEXT] == 1
    assert run(corpus, "glazier", "2017-11-30", "2017-12-01").total_hits == 0


def test_a_segment_found_by_meaning_is_kept_when_it_overlaps_the_range(corpus: Corpus) -> None:
    seg = _two_day_segment(corpus)
    vector = unit_vector(2048, hot=7)
    corpus.segment_vector(seg, vector)
    result = run(corpus, "unrelated words", "2017-12-02", "2017-12-03")
    assert result.total_hits == 0
    run_semantic(corpus.conn, result, QueryVectors(text=vector, multimodal=None), page_settings())
    assert result.counts[CHANNEL_SEMANTIC] == 1
    assert {h.segment_id for h in result.hits.values()} == {seg.segment_id}
