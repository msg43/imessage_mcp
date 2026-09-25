"""Small review tools on the search page: citations, plain words, the
Labels page, and the paged "show all hits" list.

- Every message carries a one-line citation (time to the second with the
  zone, sender, conversation, text, message ID) for "Copy citation".
- Tooltips and notices say what happened instead of naming decision
  records ("D13"); the evaluation progress line moved off the results to
  a Labels page.
- "Show all N hits in this conversation" loads 50 at a time: it sent 6.1 MB
  of HTML for 1,976 hits in one response (design review, 2026-09-24).

Synthetic data only; needs the scratch Postgres."""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
)

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_review_tools_test"
T0 = datetime(2023, 4, 24, 18, 28, 7, tzinfo=UTC)


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


def _citations(body: str) -> list[str]:
    return [html.unescape(c) for c in re.findall(r'data-cite="([^"]*)"', body)]


def test_every_message_carries_a_one_line_citation(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    group = corpus.chat([owner, alice, bob], kind="group", display_name="Deck project")
    seg = corpus.segment(
        group,
        [
            (T0, bob, 'Deposit is refundable\nuntil "materials" are ordered'),
            (T0 + timedelta(minutes=1), None, "noted, thanks"),
        ],
    )
    client = make_page_client(corpus, DB_NAME, settings=None)
    body = client.get("/search", params={"q": "deposit"}).text
    [citation] = [c for c in _citations(body) if "Deposit" in c]
    assert citation == (
        "Mon 24 Apr 2023, 18:28:07 UTC · Bob Builder · Deck project · "
        "“Deposit is refundable until \"materials\" are ordered” · "
        f"ID {seg.messages[0].message_key}"
    )
    assert 'class="link cite-btn"' in body
    # The conversation view carries the same citation for every message.
    page = client.get(f"/thread/{group.thread_key}").text
    cited = _citations(page)
    assert citation in cited
    assert any(c.startswith("Mon 24 Apr 2023, 18:29:07 UTC · Me · Deck project") for c in cited)


def test_a_citation_names_attachments_and_says_deleted(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    msg = corpus.message(
        chat, T0, alice, None, deleted_at=T0 + timedelta(days=2)
    )
    corpus.attachment(msg, filename="receipt.pdf", mime_type="application/pdf", content=b"%PDF")
    client = make_page_client(corpus, DB_NAME)
    page = client.get(f"/thread/{chat.thread_key}").text
    [citation] = _citations(page)
    assert citation == (
        "Mon 24 Apr 2023, 18:28:07 UTC · Alice Example · Alice Example · "
        "attachment: receipt.pdf · deleted in Messages · "
        f"ID {msg.message_key}"
    )


def test_tooltips_and_notices_use_plain_words(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    corpus.message(chat, T0, alice, "draft plan", deleted_at=datetime(2023, 4, 26, 12, 3, tzinfo=UTC))
    holding = corpus.chat([alice], display_name="Unfiled: Alice Example", unfiled_key="sender:test")
    corpus.message(holding, T0, alice, "orphan note")
    client = make_page_client(corpus, DB_NAME)

    thread = client.get(f"/thread/{chat.thread_key}").text
    assert 'title="Deleted in Messages on Wed 26 Apr 2023, 12:03; kept from Recently Deleted"' in thread
    unfiled = client.get(f"/thread/{holding.thread_key}").text
    assert 'title="No conversation could be named for these messages"' in unfiled
    assert "No conversation could be named for these messages" in unfiled
    for page in (thread, unfiled):
        assert "D13" not in page and "AT-4" not in page and "holding chat" not in page


def test_the_evaluation_progress_moved_to_a_labels_page(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(chat, [(T0, alice, "fence posts arrive monday")])
    client = make_page_client(corpus, DB_NAME)
    results = client.get("/search", params={"q": "fence"}).text
    assert "eval baseline" not in results and "AT-4" not in results
    assert 'href="/labels"' in results
    token = csrf_token_of(results)
    labelled = client.post(
        "/api/label",
        json={"q": "fence", "kind": "segment", "key": seg.stable_key, "grade": 2},
        headers={"X-CSRF-Token": token},
    )
    assert labelled.status_code == 200

    labels = client.get("/labels")
    assert labels.status_code == 200
    page = labels.text
    assert "1 of the 30 queries" in page
    assert "1 of the 100 graded results" in page
    assert "1 of the 25 queries with a relevant result" in page
    assert "fence" in page and "1 relevant" in page
    assert "AT-4" not in page


def test_show_all_hits_loads_fifty_at_a_time(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    for i in range(120):
        corpus.segment(chat, [(T0 + timedelta(hours=4 * i), alice, f"trellis note {i}")])
    client = make_page_client(corpus, DB_NAME)
    results = client.get("/search", params={"q": "trellis"}).text
    assert "Show all 120 hits in this conversation" in results
    more = re.search(r'class="more-hits link" data-url="([^"]+)"', results)
    assert more is not None
    first = client.get(html.unescape(more.group(1))).text
    assert first.count('<section class="hit"') == 50
    assert 'class="thread-result"' in first
    assert "Show 50 more (50 of 120 shown)" in first
    step = re.search(r'data-url="([^"]+offset=50[^"]*)" data-append="1"', first)
    assert step is not None
    second = client.get(html.unescape(step.group(1))).text
    assert second.count('<section class="hit"') == 50
    assert 'class="thread-result"' not in second  # hits only, appended by the page script
    assert "Show 20 more (100 of 120 shown)" in second
    last_step = re.search(r'data-url="([^"]+offset=100[^"]*)" data-append="1"', second)
    assert last_step is not None
    last = client.get(html.unescape(last_step.group(1))).text
    assert last.count('<section class="hit"') == 20
    assert "more-hits" not in last
    assert client.get(html.unescape(step.group(1)).replace("offset=50", "offset=-3")).status_code == 400
