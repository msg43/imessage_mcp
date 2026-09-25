"""Grading mode: grade every candidate of one search for the eval harness.

The reranker evaluation plan (2026-09-24) needs, per real query, the whole
fused candidate list down to position 30 with each candidate's first
message GUID, and a grade (0, 1, 2) for every candidate in the top 20,
shown in random order without scores. Grades are ordinary
`relevance_label` rows, so `imsg eval run` and the AT-4 check read them;
the stored list lets any reranker be scored offline later.

Synthetic data only; needs the scratch Postgres."""

from __future__ import annotations

import os
import re
import uuid
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
)
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.eval.metrics import ndcg_at_k
from imsg.eval.runner import _resolve_labels_for_query, compute_at4_check
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.search_page.model_api_client import ModelApiClient
from imsg.search_page.model_api_server import ModelAccess, ModelApiServer, ModelApiState
from imsg.search_page.search import SearchRequest, rerank_candidates, run_fulltext
from imsg.search_page.server import open_fts_reader

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_grading_test"
T0 = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
INSTRUCTION = "Given a personal message search query, retrieve relevant conversation segments"


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


@pytest.fixture
def model_api() -> Iterator[ModelApiClient]:
    access = ModelAccess(
        text_provider=FakeTextEmbeddingProvider(dim=2048),
        reranker=FakeRerankerProvider(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=1280),
        query_instruction=INSTRUCTION,
        multimodal_enabled=True,
    )
    secret = "g" * 48
    state = ModelApiState(models=access, secret=secret, port=0, warm_up=None, idle_unloader=None, log=lambda _l: None)
    server = ModelApiServer(0, state).start()
    secret_file = Path(os.environ.get("TMPDIR", "/tmp")) / f"grading-{uuid.uuid4().hex}.secret"
    fd = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, secret.encode())
    os.close(fd)
    try:
        yield ModelApiClient(port=server.port, secret_file=secret_file, timeout_seconds=5)
    finally:
        server.stop()
        secret_file.unlink(missing_ok=True)


def _corpus_with(corpus: Corpus, n: int, word: str = "pergola") -> list[Any]:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    segs = []
    for i in range(n):
        # More mentions for lower i: a known best-first fused order.
        text = " ".join([word] * max(1, 8 - i % 8)) + f" note {i}"
        segs.append(
            corpus.segment(chat, [(T0 + timedelta(hours=3 * i), alice, text), (T0 + timedelta(hours=3 * i, minutes=1), None, "ok")])
        )
    return segs


def _start(client: Any, q: str, **filters: str) -> Any:
    page = client.get("/search", params={"q": q, **filters}).text
    token = csrf_token_of(page)
    response = client.post(
        "/grade", data={"csrf_token": token, "q": q, **filters}, follow_redirects=False
    )
    return page, token, response


def _list_id(response: Any) -> int:
    assert response.status_code == 303, response.text
    match = re.fullmatch(r"/grade/(\d+)", response.headers["location"])
    assert match is not None
    return int(match.group(1))


def test_starting_needs_the_csrf_token_and_a_same_site_request(corpus: Corpus) -> None:
    _corpus_with(corpus, 3)
    client = make_page_client(corpus, DB_NAME)
    page = client.get("/search", params={"q": "pergola"}).text
    assert 'action="/grade"' in page and "Grade the top 20" in page
    token = csrf_token_of(page)
    assert client.post("/grade", data={"q": "pergola"}, follow_redirects=False).status_code == 403
    assert client.post("/grade", data={"csrf_token": "wrong", "q": "pergola"}, follow_redirects=False).status_code == 403
    cross = client.post(
        "/grade", data={"csrf_token": token, "q": "pergola"}, headers={"Origin": "http://evil.example"},
        follow_redirects=False,
    )
    assert cross.status_code == 403
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM eval_candidate_list")
        assert cur.fetchone() == (0,)


def test_the_whole_fused_list_is_stored_down_to_position_30(corpus: Corpus) -> None:
    segs = _corpus_with(corpus, 34)
    client = make_page_client(corpus, DB_NAME)
    _page, _token, response = _start(client, "pergola")
    list_id = _list_id(response)
    with corpus.conn.cursor() as cur:
        cur.execute(
            "SELECT query_id, query_text, filters, ranking FROM eval_candidate_list WHERE list_id = %s", (list_id,)
        )
        query_id, text, filters, ranking = cur.fetchone()  # type: ignore[misc]
        cur.execute(
            "SELECT fused_rank, anchor_guid, segment_key, segment_text, shown_order, channel_ranks "
            "FROM eval_candidate WHERE list_id = %s ORDER BY fused_rank",
            (list_id,),
        )
        rows = cur.fetchall()
    assert query_id == "adhoc:pergola" and text == "pergola" and filters == {}
    assert ranking["rrf_k"] == 60 and ranking["segment_hits"] == 34
    assert [r[0] for r in rows] == list(range(1, 31))
    by_key = {s.stable_key: s for s in segs}
    # The order the reranker would receive: reciprocal-rank fusion.
    reader = open_fts_reader(corpus.data_root / "fts" / "fts.db")
    try:
        result = run_fulltext(corpus.conn, reader, SearchRequest(query="pergola"), page_settings())
    finally:
        reader.close()
    expected = [h.segment_id for h in rerank_candidates(result, 30)]
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT segment_id, stable_key FROM segment")
        key_of = dict(cur.fetchall())
    assert [r[2] for r in rows] == [key_of[sid] for sid in expected]
    for _rank, anchor, key, snapshot, _shown, channels in rows:
        assert anchor == by_key[key].messages[0].source_guid  # the first message's GUID
        assert "pergola" in snapshot and snapshot.startswith("Chat: Alice Example")
        assert channels == {"text": channels["text"]}
    assert sorted(r[4] for r in rows[:20]) == list(range(1, 21))
    assert sorted(r[4] for r in rows[20:]) == list(range(21, 31))
    assert [r[4] for r in rows[:20]] != list(range(1, 21))  # shuffled


def test_the_grading_view_shows_20_in_random_order_without_scores(corpus: Corpus) -> None:
    _corpus_with(corpus, 25)
    client = make_page_client(corpus, DB_NAME)
    _page, _token, response = _start(client, "pergola")
    list_id = _list_id(response)
    view = client.get(f"/grade/{list_id}").text
    anchors = re.findall(r'<section class="candidate" data-anchor="([^"]+)"', view)
    assert len(anchors) == 20
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT anchor_guid FROM eval_candidate WHERE list_id = %s ORDER BY shown_order", (list_id,))
        in_shown_order = [r[0] for r in cur.fetchall()]
    assert anchors == in_shown_order[:20]
    cards = view[view.index('<section class="candidate"'):].lower()
    for hidden in ('class="chan', "label-btn", "score", "fused", "similar"):
        assert hidden not in cards, hidden
    assert 'class="n-graded">0</span> of 20 graded' in view
    assert view.count('class="grade-btn') == 60
    more = client.get(f"/grade/{list_id}", params={"more": "1"}).text
    assert len(re.findall(r'<section class="candidate"', more)) == 25
    assert client.get("/grade/999999").status_code == 404


def test_grades_are_eval_labels_the_harness_reads(corpus: Corpus) -> None:
    segs = _corpus_with(corpus, 5)
    client = make_page_client(corpus, DB_NAME)
    _page, token, response = _start(client, "pergola")
    list_id = _list_id(response)
    anchor = segs[1].messages[0].source_guid
    assert client.post("/api/grade", json={"list": list_id, "anchor": anchor, "grade": 2}).status_code == 403
    ok = client.post("/api/grade", json={"list": list_id, "anchor": anchor, "grade": 2}, headers={"X-CSRF-Token": token})
    assert ok.status_code == 200, ok.text
    assert ok.json() == {"grade": 2, "graded": 1, "graded_extra": 1, "total": 5}
    other = segs[3].messages[0].source_guid
    client.post("/api/grade", json={"list": list_id, "anchor": other, "grade": 0}, headers={"X-CSRF-Token": token})
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT anchor_guid, grade, source FROM relevance_label WHERE query_id = 'adhoc:pergola' ORDER BY grade")
        assert cur.fetchall() == [(other, 0, "pool_judgment"), (anchor, 2, "pool_judgment")]
    grades, unresolved = _resolve_labels_for_query(corpus.conn, "adhoc:pergola")
    assert grades == {segs[1].stable_key: 2, segs[3].stable_key: 0} and unresolved == 0
    check = compute_at4_check(corpus.conn, target="local")
    assert check.query_count == 1 and check.pooled_judgment_count == 2
    view = client.get(f"/grade/{list_id}").text
    assert 'class="n-graded">2</span> of 5 graded' in view and 'aria-pressed="true"' in view
    cleared = client.post("/api/grade", json={"list": list_id, "anchor": anchor, "grade": None}, headers={"X-CSRF-Token": token})
    assert cleared.json()["graded"] == 1
    bad = client.post("/api/grade", json={"list": list_id, "anchor": "msg-not-here", "grade": 1}, headers={"X-CSRF-Token": token})
    assert bad.status_code == 404
    assert client.post("/api/grade", json={"list": list_id, "anchor": anchor, "grade": 3}, headers={"X-CSRF-Token": token}).status_code == 400
    labels = client.get("/labels").text
    assert "Graded searches" in labels and f'href="/grade/{list_id}"' in labels


def test_a_filtered_search_gets_its_own_query_outside_the_local_eval(corpus: Corpus) -> None:
    _corpus_with(corpus, 6)
    client = make_page_client(corpus, DB_NAME)
    _page, _token, response = _start(client, "pergola", **{"from": "2024-05-01", "to": "2024-05-01"})
    list_id = _list_id(response)
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT query_id, filters FROM eval_candidate_list WHERE list_id = %s", (list_id,))
        query_id, filters = cur.fetchone()  # type: ignore[misc]
        cur.execute("SELECT targets, notes FROM eval_query WHERE query_id = %s", (query_id,))
        targets, notes = cur.fetchone()  # type: ignore[misc]
    assert query_id == "adhoc:pergola [from 2024-05-01; to 2024-05-01]"
    assert filters == {"from": "2024-05-01", "to": "2024-05-01"}
    assert targets == [] and "cannot apply the filters" in notes
    assert compute_at4_check(corpus.conn, target="local").query_count == 0
    view = client.get(f"/grade/{list_id}").text
    assert "do not count toward the first measured evaluation" in view


def test_stored_lists_score_offline_with_the_eval_metrics(corpus: Corpus) -> None:
    from imsg.search_page.grading import fused_order_metrics, load_graded_lists, reordered_metrics

    segs = _corpus_with(corpus, 12)
    client = make_page_client(corpus, DB_NAME)
    _page, token, response = _start(client, "pergola")
    list_id = _list_id(response)
    [graded] = [g for g in load_graded_lists(corpus.conn) if g.list_id == list_id]
    fused = [c.anchor_guid for c in graded.candidates]
    for grade, anchor in ((2, fused[2]), (1, fused[0]), (0, fused[5])):
        client.post("/api/grade", json={"list": list_id, "anchor": anchor, "grade": grade}, headers={"X-CSRF-Token": token})
    [graded] = [g for g in load_graded_lists(corpus.conn) if g.list_id == list_id]
    grades = {fused[2]: 2, fused[0]: 1, fused[5]: 0}
    assert fused_order_metrics(graded).ndcg_at_k == pytest.approx(ndcg_at_k(fused, grades, 10))
    # A reranker that puts the grade-2 candidate first scores higher.
    better = [fused[2]] + [a for a in fused if a != fused[2]]
    assert reordered_metrics(graded, better).ndcg_at_k > fused_order_metrics(graded).ndcg_at_k
    assert len(segs) == 12 and graded.query_text == "pergola"


def test_meaning_hits_are_in_the_fused_list(corpus: Corpus, model_api: ModelApiClient) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    query = "retaining wall"
    q = FakeTextEmbeddingProvider(dim=2048).embed_query(query, instruction=INSTRUCTION)
    corpus.segment(chat, [(T0, alice, "retaining wall blocks")])
    near = corpus.segment(chat, [(T0 + timedelta(days=2), alice, "stacked stone along the slope")])
    corpus.segment_vector(near, q)
    client = make_page_client(corpus, DB_NAME, model_api=model_api)
    _page, _token, response = _start(client, query)
    list_id = _list_id(response)
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT ranking FROM eval_candidate_list WHERE list_id = %s", (list_id,))
        (ranking,) = cur.fetchone()  # type: ignore[misc]
        cur.execute("SELECT segment_key, channel_ranks FROM eval_candidate WHERE list_id = %s", (list_id,))
        rows = dict(cur.fetchall())
    assert ranking["semantic"] == "done"
    assert rows[near.stable_key] == {"semantic": 0}


def test_a_candidate_shows_the_attachment_text_that_matched(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    bob = corpus.person("Bob Builder")
    chat = corpus.chat([owner, bob])
    seg = corpus.segment(chat, [(T0, bob, "sending the bid now")])
    att = corpus.attachment(seg.messages[0], filename="bid.pdf", mime_type="application/pdf", content=b"%PDF-1.4 bid")
    corpus.chunk(att, "Bid for the gazebo: footing and framing")
    client = make_page_client(corpus, DB_NAME)
    _page, _token, response = _start(client, "gazebo")
    view = client.get(f"/grade/{_list_id(response)}").text
    assert '<span class="muted">bid.pdf:</span> Bid for the <mark>gazebo</mark>' in view
