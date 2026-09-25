"""The search page's own filters, a search by filters alone, saved searches
outside a case, and "Download results".

- "Sent by" a person (or me), "Messages" sent / received, one-to-one or
  group conversations, and one conversation. A word counts only in a
  message the filters keep, and a hit shows only such messages; a passage
  found by meaning must hold one.
- No words and at least one filter: every message the filters keep,
  grouped by conversation, newest first.
- A search saved on its own (migration 0013) is listed on /saved with
  those saved to cases; its conversations can be marked reviewed; it can
  be renamed and removed.
- "Download results" writes every result as Markdown, CSV or JSON with the
  case download's citations.

Synthetic data only; needs the scratch Postgres."""

from __future__ import annotations

import csv
import html
import io
import json
import re
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
    csrf_token_of,
    drop_scratch_db,
    make_page_client,
    open_fts,
    page_settings,
    unit_vector,
)
from imsg.search_page.errors import SearchInputError
from imsg.search_page.search import (
    CHANNEL_FILTERS,
    CHANNEL_SEMANTIC,
    QueryVectors,
    SearchRequest,
    run_fulltext,
    run_semantic,
)
from imsg.search_page.server import open_fts_reader

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_filters_test"
T0 = datetime(2023, 4, 7, 17, 0, tzinfo=UTC)


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


def _world(corpus: Corpus) -> dict[str, Any]:
    """Alice and Bob each write "zelkova" in a group; Alice writes it to the
    owner one-to-one; the owner writes it to Bob; a recent message from
    Bob is in no segment yet."""
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    group = corpus.chat([owner, alice, bob], kind="group", display_name="Garden club")
    with_alice = corpus.chat([owner, alice])
    with_bob = corpus.chat([owner, bob])
    w: dict[str, Any] = {"owner": owner, "alice": alice, "bob": bob}
    w["group"], w["with_alice"], w["with_bob"] = group, with_alice, with_bob
    w["seg_group_alice"] = corpus.segment(
        group, [(T0, alice, "the zelkova is planted"), (T0 + timedelta(minutes=3), None, "lovely")]
    )
    w["seg_group_bob"] = corpus.segment(
        group,
        [(T0 + timedelta(days=1), bob, "zelkova needs water"), (T0 + timedelta(days=1, minutes=1), alice, "ok")],
    )
    w["seg_alice"] = corpus.segment(
        with_alice, [(T0 + timedelta(days=2), alice, "zelkova photos tomorrow")]
    )
    w["seg_owner"] = corpus.segment(
        with_bob, [(T0 + timedelta(days=3), None, "my zelkova question"), (T0 + timedelta(days=3, minutes=2), bob, "sure")]
    )
    w["loose"] = corpus.message(with_bob, datetime.now(UTC) - timedelta(days=1), bob, "not yet in a segment")
    return w


def _run(corpus: Corpus, **request: Any) -> Any:
    reader = open_fts_reader(corpus.data_root / "fts" / "fts.db")
    try:
        return run_fulltext(corpus.conn, reader, SearchRequest(**request), page_settings())
    finally:
        reader.close()


def _segments(result: Any) -> set[int]:
    return {h.segment_id for h in result.hits.values() if h.segment_id is not None}


def test_sent_by_counts_a_word_only_in_what_that_person_wrote(corpus: Corpus) -> None:
    w = _world(corpus)
    everyone = _run(corpus, query="zelkova")
    assert len(_segments(everyone)) == 4
    # Alice is in three of the four conversations' segments, but wrote the
    # word in two of them: the group segment where Bob wrote it is not hers.
    by_alice = _run(corpus, query="zelkova", sender="Alice Example")
    assert _segments(by_alice) == {w["seg_group_alice"].segment_id, w["seg_alice"].segment_id}
    by_me = _run(corpus, query="zelkova", sender="me")
    assert _segments(by_me) == {w["seg_owner"].segment_id}
    # The owner's own name means the same as "me".
    assert _segments(_run(corpus, query="zelkova", sender="Owner")) == {w["seg_owner"].segment_id}
    assert _segments(_run(corpus, query="zelkova", direction="sent")) == {w["seg_owner"].segment_id}
    received = _run(corpus, query="zelkova", direction="received")
    assert w["seg_owner"].segment_id not in _segments(received) and len(_segments(received)) == 3


def test_conversation_kind_and_one_conversation(corpus: Corpus) -> None:
    w = _world(corpus)
    groups = _run(corpus, query="zelkova", chat_kind="group")
    assert {h.chat_id for h in groups.hits.values()} == {w["group"].chat_id}
    dms = _run(corpus, query="zelkova", chat_kind="dm")
    assert {h.chat_id for h in dms.hits.values()} == {w["with_alice"].chat_id, w["with_bob"].chat_id}
    only = _run(corpus, query="zelkova", thread=w["with_alice"].thread_key)
    assert _segments(only) == {w["seg_alice"].segment_id}


def test_contradicting_filters_are_refused(corpus: Corpus) -> None:
    _world(corpus)
    with pytest.raises(SearchInputError):
        _run(corpus, query="zelkova", sender="Bob Builder", direction="sent")
    with pytest.raises(SearchInputError):
        _run(corpus, query="zelkova", sender="me", direction="received")
    with pytest.raises(SearchInputError):
        _run(corpus, query="zelkova", thread="not-a-key")
    with pytest.raises(SearchInputError):
        _run(corpus, query="")


def test_no_words_lists_every_message_the_filters_keep(corpus: Corpus) -> None:
    w = _world(corpus)
    from_bob = _run(corpus, query="", sender="Bob Builder")
    assert from_bob.analyzed is None
    assert from_bob.semantic.state == "disabled"
    assert set(from_bob.counts) == {CHANNEL_FILTERS}
    assert _segments(from_bob) == {w["seg_group_bob"].segment_id, w["seg_owner"].segment_id}
    # A message in no segment yet is found too.
    assert {h.message_id for h in from_bob.hits.values() if h.message_id} == {w["loose"].message_id}
    # Newest first: each segment is dated by its latest message Bob sent.
    threads = from_bob.threads("relevance")
    assert threads[0].chat_id == w["with_bob"].chat_id
    group_hit = next(h for h in from_bob.hits.values() if h.segment_id == w["seg_group_bob"].segment_id)
    assert group_hit.at == T0 + timedelta(days=1)


def test_a_passage_found_by_meaning_must_hold_a_message_from_the_sender(corpus: Corpus) -> None:
    w = _world(corpus)
    vector = unit_vector(2048, hot=11)
    for key in ("seg_group_alice", "seg_group_bob", "seg_alice", "seg_owner"):
        corpus.segment_vector(w[key], vector)
    reader = open_fts_reader(corpus.data_root / "fts" / "fts.db")
    try:
        result = run_fulltext(
            corpus.conn, reader, SearchRequest(query="shade tree", sender="Bob Builder"), page_settings()
        )
    finally:
        reader.close()
    run_semantic(corpus.conn, result, QueryVectors(text=vector, multimodal=None), page_settings())
    found = {h.segment_id for h in result.hits.values() if CHANNEL_SEMANTIC in h.ranks}
    assert found == {w["seg_group_bob"].segment_id, w["seg_owner"].segment_id}


def test_the_page_shows_only_the_senders_messages_and_offers_one_conversation(corpus: Corpus) -> None:
    w = _world(corpus)
    client = make_page_client(corpus, DB_NAME)
    body = client.get("/search", params={"q": "zelkova", "sender": "Bob Builder"}).text
    assert "<mark>zelkova</mark> needs water" in body
    assert "the <mark>zelkova</mark> is planted" not in body
    assert "ok</" not in body.split('id="results"')[1]  # Alice's reply in Bob's segment is not shown
    assert 'name="sender" list="people-list" value="Bob Builder"' in body
    link = re.search(r'class="only-here" href="([^"]+)"', body)
    assert link is not None and "in=" in link.group(1)
    only = client.get(html.unescape(link.group(1))).text
    assert "Only in “" in only and 'class="only-here"' not in only
    # A search with no words and a filter lists the messages; no filter goes home.
    listed = client.get("/search", params={"q": "", "dir": "sent"})
    assert listed.status_code == 200 and "my zelkova question" in listed.text
    assert "Every message these filters keep" in listed.text
    dated = client.get("/search", params={"q": "", "from": "2023-04-08", "to": "2023-04-08"}).text
    assert "zelkova needs water" in dated and "/timeline?from=2023-04-08" in html.unescape(dated)
    bare = client.get("/search", params={"q": ""}, follow_redirects=False)
    assert bare.status_code == 303 and bare.headers["location"] == "/"
    wrong = client.get("/search", params={"q": "zelkova", "sender": "Bob Builder", "dir": "sent"})
    assert wrong.status_code == 400 and "pick one" in wrong.text
    # The thread page searches inside the conversation.
    thread = client.get(f"/thread/{w['with_alice'].thread_key}").text
    assert 'class="search-here"' in thread and f'name="in" value="{w["with_alice"].thread_key}"' in thread


def _post(client: Any, path: str, payload: dict[str, Any]) -> Any:
    token = csrf_token_of(client.get("/").text)
    return client.post(path, json=payload, headers={"X-CSRF-Token": token})


def _form(client: Any, path: str, fields: dict[str, str]) -> Any:
    token = csrf_token_of(client.get("/").text)
    return client.post(path, data={"csrf_token": token, **fields}, follow_redirects=False)


def test_a_search_saved_on_its_own(corpus: Corpus) -> None:
    w = _world(corpus)
    client = make_page_client(corpus, DB_NAME)
    first = client.get("/search", params={"q": "zelkova", "kind": "group"}).text
    assert 'data-endpoint="/api/saved"' in first and 'data-endpoint="/api/case/search"' in first
    saved = _post(client, "/api/saved", {"q": "zelkova", "kind": "group", "sort": "date", "people": ""})
    assert saved.status_code == 200
    search_id = saved.json()["search_id"]
    # Saving the same search again returns the same one; no case is made.
    assert _post(client, "/api/saved", {"q": " zelkova ", "kind": "group"}).json()["search_id"] == search_id
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT case_id, params FROM search_case_search")
        rows = cur.fetchall()
        cur.execute("SELECT count(*) FROM search_case")
        cases = cur.fetchone()
    assert rows == [(None, {"kind": "group"})] and cases == (0,)
    # The results now say where it is saved and carry Reviewed boxes.
    body = client.get("/search", params={"q": "zelkova", "kind": "group"}).text
    assert '<a href="/saved">Saved</a>' in body and 'class="review-box"' in body
    marked = _post(
        client, "/api/case/review", {"search_id": search_id, "thread_key": w["group"].thread_key, "reviewed": True}
    )
    assert marked.json() == {"reviewed": 1}
    # A search by filters alone can be saved too; an empty one cannot.
    assert _post(client, "/api/saved", {"q": "", "sender": "me"}).status_code == 200
    assert _post(client, "/api/saved", {"q": "", "att": "any"}).status_code == 400
    listing = client.get("/saved").text
    assert "“zelkova”" in listing and "group conversations" in listing
    assert "(filters only)" in listing and "sent by me" in listing
    assert "1 conversation marked reviewed" in listing
    assert "/search/download?" in listing
    # Rename, then remove.
    assert _form(client, f"/saved/{search_id}/rename", {"name": "Tree in the group"}).status_code == 303
    assert "Tree in the group" in client.get("/saved").text
    assert "as “Tree in the group”" in client.get("/search", params={"q": "zelkova", "kind": "group"}).text
    assert _form(client, f"/saved/{search_id}/remove", {}).headers["location"] == "/saved"
    assert "Tree in the group" not in client.get("/saved").text


def test_saved_searches_list_cases_too_and_a_case_delete_leaves_the_others(corpus: Corpus) -> None:
    _world(corpus)
    client = make_page_client(corpus, DB_NAME)
    _post(client, "/api/saved", {"q": "zelkova"})
    in_case = _post(client, "/api/case/search", {"q": "zelkova", "sender": "Alice Example"})
    assert in_case.status_code == 200
    listing = client.get("/saved").text
    assert listing.count('class="saved-search"') == 2 and "sent by Alice Example" in listing
    with corpus.conn.cursor() as cur:
        cur.execute("DELETE FROM search_case")
        cur.execute("SELECT case_id, query_text FROM search_case_search")
        assert cur.fetchall() == [(None, "zelkova")]


def test_download_results_writes_every_result_with_citations(corpus: Corpus) -> None:
    w = _world(corpus)
    client = make_page_client(corpus, DB_NAME)
    response = client.get("/search/download", params={"q": "zelkova", "dir": "received", "fmt": "csv"})
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith('attachment; filename="search-zelkova-')
    rows = list(csv.DictReader(io.StringIO(response.text)))
    texts = {r["text"] for r in rows}
    # Every received message with the word, and none of the others: not the
    # owner's, not the replies that sit beside them in a passage.
    assert texts == {"the zelkova is planted", "zelkova needs water", "zelkova photos tomorrow"}
    assert {r["found_by"] for r in rows} == {"the words"}
    assert all(r["message_id"] and r["messages_guid"] for r in rows)
    # Conversations keep their order in the results, numbered.
    assert sorted({r["conversation_result"] for r in rows}) == ["1", "2"]

    md = client.get("/search/download", params={"q": "zelkova", "dir": "received", "fmt": "md"}).text
    assert md.startswith("# Search: “zelkova”")
    assert "Filters: received." in md and "Found by: the words" in md
    assert "## 1. " in md and "Message ID" in md

    by_filters = client.get("/search/download", params={"q": "", "sender": "Bob Builder", "fmt": "json"})
    data = json.loads(by_filters.text)
    assert data["search"] == "" and data["filters"] == "sent by Bob Builder"
    assert {m["text"] for m in data["messages_found"]} == {
        "zelkova needs water", "sure", "not yet in a segment",
    }
    assert {m["found_by"] for m in data["messages_found"]} == {"the filters"}
    assert data["stopped_at_limit"] is False
    assert w["loose"].message_key in {m["message_id"] for m in data["messages_found"]}
    assert client.get("/search/download", params={"q": "zelkova", "fmt": "xml"}).status_code == 400
