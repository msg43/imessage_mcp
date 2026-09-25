"""Search page against a real Postgres and a real FTS5 sidecar (D14).

Synthetic data only. Skips cleanly when no scratch Postgres is reachable
(set IMSG_TEST_PG_HOST / IMSG_TEST_PG_PORT / IMSG_TEST_PG_USER).

Covers: every-hit semantics (more hits than the MCP tool's top-k), full
text and semantic merged and deduplicated, thresholds, filters, grouping
and sort orders, the not-yet-indexed channel, the optional rerank of the
best few only, the thread view scrolling both ways with its markers,
attachment serving, and label writes read back by the eval harness.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from starlette.testclient import TestClient

from _search_page_fixtures import (
    REACHABLE,
    SKIP_REASON,
    Corpus,
    create_scratch_db,
    drop_scratch_db,
    dsn,
    open_fts,
    tiny_png,
)
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.eval.runner import _resolve_labels_for_query
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.search_page.app import AppDeps, ConnectionPool, FtsReaders, build_app
from imsg.search_page.auth import LoginGuard, PasswordFile, SessionStore, set_password
from imsg.search_page.config import SearchPageConfig
from imsg.search_page.media import MediaConverter
from imsg.search_page.model_api_client import ModelApiClient
from imsg.search_page.model_api_server import ModelAccess, ModelApiServer, ModelApiState
from imsg.search_page.search import (
    CHANNEL_ATTACHMENT_TEXT,
    CHANNEL_IMAGE,
    CHANNEL_SEMANTIC,
    CHANNEL_TEXT,
    CHANNEL_UNINDEXED,
    QueryVectors,
    SearchRequest,
    SearchSettings,
    apply_rerank,
    rerank_candidates,
    run_fulltext,
    run_semantic,
)
from imsg.search_page.server import open_fts_reader

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_search_page_test"
PASSWORD = "correct horse battery staple"
INSTRUCTION = "Given a personal message search query, retrieve relevant conversation segments"
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def pg() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(DB_NAME)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(DB_NAME)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data_root"
    (root / "fts").mkdir(parents=True)
    (root / "attachments").mkdir()
    return root


@pytest.fixture
def corpus(pg: psycopg.Connection, data_root: Path) -> Iterator[Corpus]:
    fts = open_fts(data_root / "fts" / "fts.db")
    try:
        yield Corpus(conn=pg, fts=fts, data_root=data_root)
    finally:
        fts.close()


def settings(**overrides: Any) -> SearchSettings:
    values: dict[str, Any] = {
        "timezone": "UTC",
        "index_unsent": False,
        "rrf_k": 60,
        "fts_max_hits": 20000,
        "unindexed_window_days": 60,
        "semantic_enabled": True,
        "multimodal_enabled": True,
        "text_min_similarity": 0.5,
        "multimodal_min_similarity": 0.2,
        "max_hits_per_channel": 2000,
        "ef_search": 200,
        "max_scan_tuples": 50000,
    }
    values.update(overrides)
    return SearchSettings(**values)


def run(corpus: Corpus, query: str, **request: Any) -> Any:
    reader = open_fts_reader(corpus.data_root / "fts" / "fts.db")
    try:
        return run_fulltext(corpus.conn, reader, SearchRequest(query=query, **request), settings())
    finally:
        reader.close()


def orthogonal_mix(q: list[float], similarity: float, seed: int) -> list[float]:
    """A unit vector whose cosine similarity with unit `q` is `similarity`."""
    import random

    rng = random.Random(seed)
    r = [rng.uniform(-1, 1) for _ in q]
    dot = sum(a * b for a, b in zip(r, q, strict=True))
    r = [a - dot * b for a, b in zip(r, q, strict=True)]
    norm = sum(a * a for a in r) ** 0.5
    r = [a / norm for a in r]
    other = (1 - similarity**2) ** 0.5
    return [similarity * a + other * b for a, b in zip(q, r, strict=True)]


# --------------------------------------------------------------------------
# every hit, grouping, filters, sort orders
# --------------------------------------------------------------------------


def test_every_hit_is_returned_not_a_top_k(corpus: Corpus) -> None:
    """150 matching segments: the MCP tool's channel stops at k_fts=100 and
    returns 10; the page returns all 150, grouped per conversation."""
    owner = corpus.person("Owner", owner=True)
    alice, bob, carol = corpus.person("Alice Example"), corpus.person("Bob Builder"), corpus.person("Carol Example")
    chats = [corpus.chat([owner, p]) for p in (alice, bob, carol)]
    sizes = [100, 30, 20]
    for chat, size, person in zip(chats, sizes, (alice, bob, carol), strict=True):
        for i in range(size):
            at = T0 + timedelta(hours=i * 5)
            corpus.segment(chat, [(at, person, f"the gazebo plan number {i}")])
    for i in range(20):
        corpus.segment(chats[0], [(T0 + timedelta(days=90, hours=i), alice, "nothing relevant here")])

    result = run(corpus, "gazebo")

    assert result.total_hits == 150
    assert result.counts[CHANNEL_TEXT] == 150
    threads = result.threads("relevance")
    assert sorted(t.count for t in threads) == [20, 30, 100]
    by_chat = {t.chat_id: t.count for t in threads}
    assert by_chat == {chats[0].chat_id: 100, chats[1].chat_id: 30, chats[2].chat_id: 20}


def test_filters_people_dates_and_attachments(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    with_alice = corpus.chat([owner, alice])
    with_bob = corpus.chat([owner, bob])
    early = corpus.segment(with_alice, [(T0, alice, "deck stain samples")])
    late = corpus.segment(with_alice, [(T0 + timedelta(days=40), alice, "deck stain arrived")])
    bobs = corpus.segment(with_bob, [(T0 + timedelta(days=5), bob, "deck stain quote")])
    photo_msg = late.messages[0]
    corpus.attachment(photo_msg, filename="deck.png", mime_type="image/png", content=tiny_png())

    assert run(corpus, "deck stain").total_hits == 3
    only_alice = run(corpus, "deck stain", people=(alice.short_name,))
    assert {h.segment_id for h in only_alice.hits.values()} == {early.segment_id, late.segment_id}
    dated = run(corpus, "deck stain", after="2024-03-02", before="2024-03-20")
    assert {h.segment_id for h in dated.hits.values()} == {bobs.segment_id}
    with_att = run(corpus, "deck stain", attachments="with")
    assert {h.segment_id for h in with_att.hits.values()} == {late.segment_id}
    without_att = run(corpus, "deck stain", attachments="without")
    assert late.segment_id not in {h.segment_id for h in without_att.hits.values()}


def test_sort_orders(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    old_chat = corpus.chat([owner, alice])
    new_chat = corpus.chat([owner, bob])
    # The old chat's segment mentions the words many times (better BM25);
    # the new chat's is more recent.
    corpus.segment(old_chat, [(T0, alice, "lumber lumber lumber delivery lumber delivery")])
    corpus.segment(new_chat, [(T0 + timedelta(days=300), bob, "the lumber delivery was late")])

    result = run(corpus, "lumber delivery")
    assert [t.chat_id for t in result.threads("relevance")] == [old_chat.chat_id, new_chat.chat_id]
    assert [t.chat_id for t in result.threads("date")] == [new_chat.chat_id, old_chat.chat_id]


def test_quoted_phrase_and_emoji(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    hyphen = corpus.segment(chat, [(T0, alice, "see bid-rev3.pdf attached")])
    corpus.segment(chat, [(T0 + timedelta(days=1), alice, "the bid came back")])
    emoji = corpus.segment(chat, [(T0 + timedelta(days=2), alice, "party tonight \U0001f389")])

    phrase = run(corpus, '"id-rev"')
    assert {h.segment_id for h in phrase.hits.values()} == {hyphen.segment_id}
    party = run(corpus, "\U0001f389")
    assert {h.segment_id for h in party.hits.values()} == {emoji.segment_id}


def test_attachment_text_hits_map_to_their_segment(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(chat, [(T0, alice, "here is the estimate")])
    att = corpus.attachment(seg.messages[0], filename="estimate.pdf", mime_type="application/pdf", content=b"%PDF-1.4 x")
    corpus.chunk(att, "Footing concrete and rebar, materials total 14,200")

    result = run(corpus, "rebar")
    assert result.counts[CHANNEL_ATTACHMENT_TEXT] == 1
    [hit] = result.hits.values()
    assert hit.segment_id == seg.segment_id
    assert (seg.messages[0].message_id, att.attachment_id) in hit.matched_attachments


def test_messages_not_yet_segmented_are_found(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    recent = datetime.now(UTC) - timedelta(days=2)
    fresh = corpus.message(chat, recent, alice, "the pergola permit came through")
    corpus.message(chat, recent - timedelta(days=400), alice, "old pergola note, unsegmented")
    corpus.message(chat, recent, alice, "pergolas are nice")  # different word: no whole-word match

    result = run(corpus, "pergola")
    assert result.counts[CHANNEL_UNINDEXED] == 1
    [hit] = result.hits.values()
    assert hit.message_id == fresh.message_id and hit.segment_id is None


# --------------------------------------------------------------------------
# semantic: thresholds, merge, dedupe — through the real internal model API
# --------------------------------------------------------------------------


@pytest.fixture
def model_api() -> Iterator[tuple[ModelApiServer, ModelApiClient, FakeRerankerProvider]]:
    reranker = FakeRerankerProvider()
    access = ModelAccess(
        text_provider=FakeTextEmbeddingProvider(dim=2048),
        reranker=reranker,
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=1280),
        query_instruction=INSTRUCTION,
        multimodal_enabled=True,
    )
    secret = "s" * 48
    state = ModelApiState(models=access, secret=secret, port=0, warm_up=None, idle_unloader=None, log=lambda _l: None)
    server = ModelApiServer(0, state).start()
    secret_file = Path(os.environ.get("TMPDIR", "/tmp")) / f"model-api-{uuid.uuid4().hex}.secret"
    fd = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, secret.encode())
    os.close(fd)
    client = ModelApiClient(port=server.port, secret_file=secret_file, timeout_seconds=5)
    try:
        yield server, client, reranker
    finally:
        server.stop()
        secret_file.unlink(missing_ok=True)


def test_semantic_hits_above_threshold_merge_with_full_text(
    corpus: Corpus, model_api: tuple[ModelApiServer, ModelApiClient, FakeRerankerProvider]
) -> None:
    _server, client, _reranker = model_api
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    chat_a, chat_b = corpus.chat([owner, alice]), corpus.chat([owner, bob])
    query = "backyard renovation ideas"
    q_text = FakeTextEmbeddingProvider(dim=2048).embed_query(query, instruction=INSTRUCTION)
    q_image = FakeMultimodalEmbeddingProvider(dim=1280).embed_text(query)

    both = corpus.segment(chat_a, [(T0, alice, "backyard renovation ideas and budget")])
    corpus.segment_vector(both, orthogonal_mix(q_text, 0.9, 1))
    near = [corpus.segment(chat_b, [(T0 + timedelta(days=i + 1), bob, f"patio thoughts {i}")]) for i in range(5)]
    for i, seg in enumerate(near):
        corpus.segment_vector(seg, orthogonal_mix(q_text, 0.8, 10 + i))
    far = [corpus.segment(chat_b, [(T0 + timedelta(days=i + 20), bob, f"grocery list {i}")]) for i in range(5)]
    for i, seg in enumerate(far):
        corpus.segment_vector(seg, orthogonal_mix(q_text, 0.2, 30 + i))
    photo_seg = corpus.segment(chat_a, [(T0 + timedelta(days=50), alice, "look at this")])
    photo = corpus.attachment(photo_seg.messages[0], filename="yard.png", mime_type="image/png", content=tiny_png())
    corpus.image_vector(photo, orthogonal_mix(q_image, 0.35, 99))

    result = run(corpus, query)
    assert result.total_hits == 1  # full text first: only the segment with every word

    embedded = client.embed(result.analyzed.phrase, multimodal=True)
    assert embedded.text_vector == pytest.approx(q_text, abs=1e-6)
    status = run_semantic(
        corpus.conn,
        result,
        QueryVectors(text=embedded.text_vector, multimodal=embedded.multimodal_vector),
        settings(),
    )

    segment_ids = {h.segment_id for h in result.hits.values()}
    assert segment_ids == {both.segment_id, photo_seg.segment_id, *(s.segment_id for s in near)}
    assert not segment_ids & {s.segment_id for s in far}  # below the 0.5 floor
    merged = result.hits[f"s{both.segment_id}"]
    assert set(merged.channels) >= {CHANNEL_TEXT, CHANNEL_SEMANTIC}  # one hit, two channels
    assert merged.similarity[CHANNEL_SEMANTIC] == pytest.approx(0.9, abs=0.01)
    assert result.hits[f"s{photo_seg.segment_id}"].channels == [CHANNEL_IMAGE]
    assert status.added_hits == 6
    # Relevance: the hit found by both channels ranks first.
    assert result.threads("relevance")[0].hits[0].segment_id == both.segment_id


def test_rerank_touches_only_the_best_few(
    corpus: Corpus, model_api: tuple[ModelApiServer, ModelApiClient, FakeRerankerProvider]
) -> None:
    _server, client, _reranker = model_api
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    for i in range(30):
        corpus.segment(chat, [(T0 + timedelta(hours=i), alice, f"shed roof shingles option {i}")])
    result = run(corpus, "shed roof")
    candidates = rerank_candidates(result, 5)
    assert len(candidates) == 5
    scores = client.rerank("shed roof", ["shed roof shingles"] * 5)
    apply_rerank(result, candidates, scores)
    reranked = [h for h in result.hits.values() if h.rerank_score is not None]
    assert len(reranked) == 5 and result.total_hits == 30


# --------------------------------------------------------------------------
# the web app, end to end
# --------------------------------------------------------------------------


def make_client(
    corpus: Corpus,
    tmp_path: Path,
    *,
    model_api: ModelApiClient | None = None,
    **page: Any,
) -> TestClient:
    data_root = corpus.data_root
    password_path = data_root / "private" / "search-page" / "owner-password"
    set_password(password_path, PASSWORD)
    passwords = PasswordFile(password_path)
    page_cfg = SearchPageConfig(
        enabled=True,
        allowed_hosts=["testserver"],
        **page,
    )
    deps = AppDeps(
        page=page_cfg,
        settings=settings(),
        data_root=data_root,
        pool=ConnectionPool(lambda: psycopg.connect(dsn(DB_NAME), autocommit=True), 2),
        fts=FtsReaders(
            lambda: open_fts_reader(data_root / "fts" / "fts.db"), data_root / "fts" / "fts.db", 2
        ),
        passwords=passwords,
        sessions=SessionStore(data_root / "private" / "search-page" / "sessions.json", lifetime_seconds=3600),
        login_guard=LoginGuard(passwords, max_failures_per_client=5, max_failures_global=30, window_seconds=900),
        media=MediaConverter(data_root / "search-page" / "thumbnails"),
        model_api=model_api,
    )
    client = TestClient(build_app(deps))
    page_html = client.get("/login").text
    token = re.search(r'name="login_token" value="([^"]+)"', page_html)
    assert token is not None
    response = client.post(
        "/login",
        data={"login_token": token.group(1), "password": PASSWORD, "next": "/"},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return client


def csrf_of(html: str) -> str:
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def test_search_page_renders_grouped_highlighted_and_escaped(corpus: Corpus, tmp_path: Path) -> None:
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    chat_a = corpus.chat([owner, alice])
    chat_b = corpus.chat([owner, alice, bob], kind="group", display_name="Deck project")
    corpus.segment(chat_a, [(T0, alice, "The GAZEBO is up! <script>alert(1)</script>")])
    corpus.segment(chat_b, [(T0 + timedelta(days=1), bob, "gazebo paint?"), (T0 + timedelta(days=1, minutes=1), None, "blue")])
    client = make_client(corpus, tmp_path)

    response = client.get("/search", params={"q": "gazebo"})
    assert response.status_code == 200
    body = response.text
    assert "2 hits" in body and "2 conversations" in body
    assert "Deck project" in body and "Alice Example" in body
    assert "<mark>GAZEBO</mark>" in body and "<mark>gazebo</mark>" in body
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"


def test_infinite_scroll_pages_through_every_conversation(corpus: Corpus, tmp_path: Path) -> None:
    owner = corpus.person("Owner", owner=True)
    for i in range(5):
        person = corpus.person(f"Friend {i}")
        chat = corpus.chat([owner, person])
        corpus.segment(chat, [(T0 + timedelta(days=i), person, f"trellis note {i}")])
    client = make_client(corpus, tmp_path, page_threads=2)

    first = client.get("/search", params={"q": "trellis", "sort": "date"}).text
    assert first.count('class="thread-result"') == 2
    next_urls = re.findall(r'data-next="([^"]+)"', first)
    assert next_urls
    seen = first.count('class="thread-result"')
    url = next_urls[0].replace("&amp;", "&")
    while url:
        fragment = client.get(url).text
        seen += fragment.count('class="thread-result"')
        more = re.findall(r'data-next="([^"]+)"', fragment)
        url = more[0].replace("&amp;", "&") if more else ""
    assert seen == 5
    # Without scripts, the "More conversations" link opens the next full page.
    full = re.findall(r'class="load-more"', first)
    assert full
    second = client.get("/search", params={"q": "trellis", "sort": "date", "page": 2}).text
    assert second.count('class="thread-result"') == 2 and "<html" in second.lower()


def test_semantic_api_merges_or_reports_unavailable(
    corpus: Corpus,
    tmp_path: Path,
    model_api: tuple[ModelApiServer, ModelApiClient, FakeRerankerProvider],
) -> None:
    _server, api_client, _reranker = model_api
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    query = "retaining wall"
    q = FakeTextEmbeddingProvider(dim=2048).embed_query(query, instruction=INSTRUCTION)
    text_seg = corpus.segment(chat, [(T0, alice, "retaining wall blocks")])
    corpus.segment_vector(text_seg, orthogonal_mix(q, 0.1, 5))
    sem_seg = corpus.segment(chat, [(T0 + timedelta(days=3), alice, "stacked stone along the slope")])
    corpus.segment_vector(sem_seg, orthogonal_mix(q, 0.85, 6))

    client = make_client(corpus, tmp_path, model_api=api_client)
    page = client.get("/search", params={"q": query})
    assert "semantic search running" in page.text
    answer = client.get("/api/semantic", params={"q": query}).json()
    assert answer["state"] == "done" and answer["added_hits"] == 1
    merged = client.get(answer["page1_url"]).text
    assert "stacked stone" in merged and "retaining" in merged

    closed = ModelApiClient(port=1, secret_file=tmp_path / "missing.secret", timeout_seconds=1)
    offline = make_client(corpus, tmp_path, model_api=closed)
    answer = offline.get("/api/semantic", params={"q": "stone slope"}).json()
    assert answer["state"] == "unavailable"
    assert "stone" in offline.get("/search", params={"q": "stone slope"}).text


def test_thread_view_scrolls_both_ways_with_markers(corpus: Corpus, tmp_path: Path) -> None:
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    chat = corpus.chat([owner, alice, bob], kind="group", display_name="Deck project")
    messages = []
    for i in range(200):
        sender = [alice, bob, None][i % 3]
        messages.append(corpus.message(chat, T0 + timedelta(minutes=i), sender, f"message number {i}"))
    anchor = messages[100]
    corpus.tapback(anchor, bob, "loved")
    deleted = corpus.message(chat, T0 + timedelta(minutes=100, seconds=30), alice, "oops", deleted_at=T0 + timedelta(days=1))
    holding = corpus.chat([alice], display_name="Unfiled: Alice Example", unfiled_key="sender:alice-test")
    corpus.message(holding, T0, alice, "orphan message")
    client = make_client(corpus, tmp_path)

    page = client.get(f"/thread/{chat.thread_key}", params={"anchor": anchor.message_key})
    assert page.status_code == 200
    body = page.text
    assert f'id="m-{anchor.message_key}"' in body and "msg them anchor" in body
    assert "♥ Bob Builder" in body  # reaction
    assert f'id="m-{deleted.message_key}"' in body and ">Deleted<" in body
    assert "Alice Example, Bob Builder" in body  # participants
    assert body.count('class="msg ') == 40 + 1 + 40
    older_cursor = re.search(r'data-cursor="([0-9a-f]{64})" data-dir="older"', body)
    newer_cursor = re.search(r'data-cursor="([0-9a-f]{64})" data-dir="newer"', body)
    assert older_cursor and newer_cursor
    older = client.get(
        f"/thread/{chat.thread_key}/messages", params={"cursor": older_cursor.group(1), "dir": "older"}
    ).json()
    assert "message number 59" in older["html"] and older["more"] is False
    newer = client.get(
        f"/thread/{chat.thread_key}/messages", params={"cursor": newer_cursor.group(1), "dir": "newer"}
    ).json()
    assert "message number 199" in newer["html"] and newer["more"] is False

    holding_page = client.get(f"/thread/{holding.thread_key}").text
    assert "holding chat" in holding_page and "Unfiled" in holding_page


def test_attachments_served_with_types_and_never_outside_the_cache(
    corpus: Corpus, tmp_path: Path
) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(chat, [(T0, alice, "files attached")])
    msg = seg.messages[0]
    image = corpus.attachment(msg, filename="yard.png", mime_type="image/png", content=tiny_png())
    pdf = corpus.attachment(msg, filename="bid.pdf", mime_type="application/pdf", content=b"%PDF-1.4\n%%EOF")
    page = corpus.attachment(msg, filename="evil.html", mime_type="text/html", content=b"<script>alert(1)</script>")
    missing = corpus.attachment(msg, filename="gone.jpg", mime_type="image/jpeg", content=None, state="missing")
    escape = corpus.attachment(msg, filename="x.txt", mime_type="text/plain", content=b"inside")
    # Point one row's sha at a path that climbs out of the cache.
    with corpus.conn.cursor() as cur:
        cur.execute("UPDATE attachment SET sha256 = %s WHERE attachment_id = %s", ("../../../../etc/passwd", escape.attachment_id))
    # And plant a symlink in the cache pointing outside it.
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("not for serving")
    link_sha = "ab" * 32
    (corpus.data_root / "attachments" / "ab").mkdir(parents=True, exist_ok=True)
    (corpus.data_root / "attachments" / "ab" / link_sha).symlink_to(outside)
    linked = corpus.attachment(msg, filename="linked.txt", mime_type="text/plain", content=None)
    with corpus.conn.cursor() as cur:
        cur.execute("UPDATE attachment SET sha256 = %s, state = 'materialized' WHERE attachment_id = %s", (link_sha, linked.attachment_id))
    client = make_client(corpus, tmp_path)

    img = client.get(f"/att/{image.attachment_key}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert img.headers["x-content-type-options"] == "nosniff"
    assert img.headers["content-disposition"].startswith("inline")
    assert "sandbox" in img.headers["content-security-policy"]
    doc = client.get(f"/att/{pdf.attachment_key}")
    assert doc.headers["content-type"] == "application/pdf" and doc.headers["content-disposition"].startswith("inline")
    assert "sandbox" not in doc.headers["content-security-policy"]
    html_file = client.get(f"/att/{page.attachment_key}")
    assert html_file.headers["content-type"] == "application/octet-stream"
    assert html_file.headers["content-disposition"].startswith("attachment")
    download = client.get(f"/att/{image.attachment_key}", params={"download": "1"})
    assert download.headers["content-disposition"].startswith("attachment")
    assert client.get(f"/att/{missing.attachment_key}").status_code == 404
    assert client.get(f"/att/{escape.attachment_key}").status_code == 404
    assert client.get(f"/att/{linked.attachment_key}").status_code == 404
    assert client.get("/att/..%2F..%2Fetc%2Fpasswd").status_code == 404
    ranged = client.get(f"/att/{image.attachment_key}", headers={"Range": "bytes=0-3"})
    assert ranged.status_code == 206 and ranged.content == tiny_png()[:4]


def test_labels_are_written_in_the_eval_harness_format(corpus: Corpus, tmp_path: Path) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(chat, [(T0, alice, "fence posts arrive monday"), (T0 + timedelta(minutes=2), None, "ok")])
    recent = corpus.message(chat, datetime.now(UTC) - timedelta(days=1), alice, "fence gate latch")
    client = make_client(corpus, tmp_path)
    page = client.get("/search", params={"q": "fence"}).text
    token = csrf_of(page)

    # No CSRF token: refused before anything is written.
    refused = client.post("/api/label", json={"q": "fence", "kind": "segment", "key": seg.stable_key, "grade": 2})
    assert refused.status_code == 403
    # Cross-site origin: refused even with the token.
    cross = client.post(
        "/api/label",
        json={"q": "fence", "kind": "segment", "key": seg.stable_key, "grade": 2},
        headers={"X-CSRF-Token": token, "Origin": "http://evil.example"},
    )
    assert cross.status_code == 403

    ok = client.post(
        "/api/label",
        json={"q": "fence", "kind": "segment", "key": seg.stable_key, "grade": 2},
        headers={"X-CSRF-Token": token},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["counts"] == {"total": 1, "relevant": 1, "not_relevant": 0}
    query_id = ok.json()["query_id"]
    assert query_id == "adhoc:fence"
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT anchor_guid, grade, source FROM relevance_label WHERE query_id = %s", (query_id,))
        rows = cur.fetchall()
    assert rows == [(seg.messages[0].source_guid, 2, "mark_relevant")]
    grades, unresolved = _resolve_labels_for_query(corpus.conn, query_id)
    assert grades == {seg.stable_key: 2} and unresolved == 0

    not_rel = client.post(
        "/api/label",
        json={"q": "fence", "kind": "message", "key": recent.message_key, "grade": 0},
        headers={"X-CSRF-Token": token},
    )
    assert not_rel.status_code == 200
    assert not_rel.json()["counts"] == {"total": 2, "relevant": 1, "not_relevant": 1}
    rendered = client.get("/search", params={"q": "fence"}).text
    assert "labelled for this query" in rendered and 'class="n-total">2<' in rendered
    assert 'label-btn rel on' in rendered and 'label-btn notrel on' in rendered

    cleared = client.post(
        "/api/label",
        json={"q": "fence", "kind": "segment", "key": seg.stable_key, "grade": None},
        headers={"X-CSRF-Token": token},
    )
    assert cleared.json()["counts"]["total"] == 1
    assert _resolve_labels_for_query(corpus.conn, query_id) == ({}, 1)
