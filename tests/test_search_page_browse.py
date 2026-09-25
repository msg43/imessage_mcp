"""Browsing without a search word: the Timeline and Media views.

Before these views a search with a date or a person but no words sent the
owner back to the home page (design review 2026-09-24, change 5). The
Timeline lists every message across conversations in time order for a
day or range, with a count per day and search words highlighted; the
Media view is a grid of photos, videos, voice notes, PDFs and other
files filtered by person, date and type. Both open the conversation at
the chosen message.

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
    drop_scratch_db,
    make_page_client,
    open_fts,
    page_settings,
    tiny_png,
)
from imsg.search_page.threads import attachment_kind

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_browse_test"
DAY = datetime(2023, 4, 7, 4, 0, tzinfo=UTC)
"""Midnight in New York (EDT) on Fri 7 Apr 2023."""


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


def _client(corpus: Corpus, **settings: object) -> object:
    return make_page_client(corpus, DB_NAME, settings=page_settings(timezone="America/New_York", **settings))


def _day(corpus: Corpus) -> dict[str, object]:
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    group = corpus.chat([owner, alice, bob], kind="group", display_name="Deck project")
    with_bob = corpus.chat([owner, bob])
    with_alice = corpus.chat([owner, alice])
    m = {
        "late_before": corpus.message(with_alice, DAY - timedelta(minutes=1), alice, "night before"),
        "a": corpus.message(group, DAY + timedelta(hours=9, seconds=5), alice, "stain arrives today"),
        "b": corpus.message(with_bob, DAY + timedelta(hours=9, minutes=1), bob, "deposit sent"),
        "c": corpus.message(group, DAY + timedelta(hours=13), None, "the deck stain looks good"),
        "unsent": corpus.message(with_alice, DAY + timedelta(hours=14), alice, "oops", unsent=True),
        "d": corpus.message(with_alice, DAY + timedelta(hours=23, minutes=59), alice, "goodnight"),
        "next_day": corpus.message(with_bob, DAY + timedelta(days=1, minutes=1), bob, "next morning"),
    }
    return {"people": (owner, alice, bob), "chats": (group, with_bob, with_alice), "m": m}


def _order(body: str, texts: list[str]) -> list[int]:
    return [body.index(t) for t in texts]


def test_the_timeline_lists_a_days_messages_across_conversations_in_order(corpus: Corpus) -> None:
    data = _day(corpus)
    client = _client(corpus)
    body = client.get("/timeline", params={"from": "2023-04-07", "to": "2023-04-07"}).text  # type: ignore[attr-defined]
    positions = _order(body, ["stain arrives today", "deposit sent", "the deck stain looks good", "goodnight"])
    assert positions == sorted(positions)
    assert "night before" not in body and "next morning" not in body
    assert "oops" not in body  # unsent, hidden by policy
    assert "Fri 07 Apr 2023" in body and "4 messages" in body
    assert "09:00:05" in body  # times to the second
    m = data["m"]
    group = data["chats"][0]  # type: ignore[index]
    assert f'href="/thread/{group.thread_key}?anchor={m["a"].message_key}#m-{m["a"].message_key}"' in body  # type: ignore[index,union-attr]
    shown = client.get("/timeline", params={"from": "2023-04-07", "to": "2023-04-07"})  # type: ignore[attr-defined]
    assert shown.headers["cache-control"] == "no-store"


def test_day_counts_people_sender_and_highlights(corpus: Corpus) -> None:
    data = _day(corpus)
    _owner, alice, bob = data["people"]  # type: ignore[misc]
    client = _client(corpus)
    two_days = client.get("/timeline", params={"from": "2023-04-07", "to": "2023-04-08"}).text  # type: ignore[attr-defined]
    assert re.search(r"Fri 07 Apr 2023</a>\s*<span[^>]*>4 messages", two_days)
    assert re.search(r"Sat 08 Apr 2023</a>\s*<span[^>]*>1 message<", two_days)
    with_bob = client.get("/timeline", params={"from": "2023-04-07", "to": "2023-04-07", "people": bob.short_name}).text  # type: ignore[attr-defined]
    assert "deposit sent" in with_bob and "stain arrives today" in with_bob and "goodnight" not in with_bob
    from_bob = client.get("/timeline", params={"from": "2023-04-07", "to": "2023-04-07", "sender": bob.short_name}).text  # type: ignore[attr-defined]
    assert "deposit sent" in from_bob and "stain arrives today" not in from_bob
    mine = client.get("/timeline", params={"from": "2023-04-07", "to": "2023-04-07", "sender": "me"}).text  # type: ignore[attr-defined]
    assert "the deck stain looks good" in mine and "deposit sent" not in mine
    words = client.get("/timeline", params={"from": "2023-04-07", "to": "2023-04-07", "q": "stain"}).text  # type: ignore[attr-defined]
    assert words.count("<mark>stain</mark>") == 2 and "deposit sent" in words
    assert alice.short_name  # fixture sanity


def test_the_timeline_defaults_to_the_latest_day_and_pages_by_cursor(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    for i in range(250):
        corpus.message(chat, DAY + timedelta(minutes=i), alice, f"entry {i:03d}")
    client = _client(corpus)
    first = client.get("/timeline").text  # type: ignore[attr-defined]
    assert "Fri 07 Apr 2023" in first and "entry 000" in first and "entry 199" in first
    assert "entry 200" not in first
    next_url = re.search(r'data-next="([^"]+)"', first)
    assert next_url is not None
    rest = client.get(html.unescape(next_url.group(1))).text  # type: ignore[attr-defined]
    assert "entry 200" in rest and "entry 249" in rest and "entry 199" not in rest
    assert "data-next" not in rest
    assert client.get("/timeline", params={"from": "2023-13-01"}).status_code == 400  # type: ignore[attr-defined]
    only_to = client.get("/timeline", params={"to": "2023-04-07"}).text  # type: ignore[attr-defined]
    assert "entry 000" in only_to and "Fri 07 Apr 2023" in only_to
    backwards = client.get("/timeline", params={"from": "2023-04-08", "to": "2023-04-07"})  # type: ignore[attr-defined]
    assert backwards.status_code == 400


def test_a_search_without_words_lists_the_messages_and_links_the_timeline(corpus: Corpus) -> None:
    """A date with no words used to redirect to the Timeline; it is now a
    search by filters alone, which links the same days on the Timeline."""
    _day(corpus)
    client = _client(corpus)
    dated = client.get("/search", params={"q": "", "from": "2023-04-07", "to": "2023-04-07"}, follow_redirects=False)  # type: ignore[attr-defined]
    assert dated.status_code == 200
    assert "/timeline?from=2023-04-07&amp;to=2023-04-07" in dated.text
    bare = client.get("/search", params={"q": ""}, follow_redirects=False)  # type: ignore[attr-defined]
    assert bare.status_code == 303 and bare.headers["location"] == "/"


def _media(corpus: Corpus) -> dict[str, object]:
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    with_bob = corpus.chat([owner, bob])
    with_alice = corpus.chat([owner, alice])
    items = {}
    m1 = corpus.message(with_bob, DAY + timedelta(hours=10), bob, "photo")
    items["photo"] = corpus.attachment(m1, filename="deck.png", mime_type="image/png", content=tiny_png())
    m2 = corpus.message(with_bob, DAY + timedelta(hours=11), bob, None)
    items["voice"] = corpus.attachment(m2, filename="Audio Message.caf", mime_type="audio/x-caf", content=b"caff")
    m3 = corpus.message(with_alice, DAY + timedelta(days=3), alice, "bid")
    items["pdf"] = corpus.attachment(m3, filename="bid.pdf", mime_type="application/pdf", content=b"%PDF-1.4")
    m4 = corpus.message(with_alice, DAY + timedelta(days=4), None, "clip")
    items["video"] = corpus.attachment(m4, filename="clip.mov", mime_type="video/quicktime", content=b"moov")
    m5 = corpus.message(with_alice, DAY + timedelta(days=5), alice, "card")
    items["other"] = corpus.attachment(m5, filename="contact.vcf", mime_type="text/vcard", content=b"BEGIN:VCARD")
    m6 = corpus.message(with_alice, DAY + timedelta(days=6), alice, "sticker")
    sticker = corpus.attachment(m6, filename="sticker.png", mime_type="image/png", content=tiny_png(4, 4))
    with corpus.conn.cursor() as cur:
        cur.execute("UPDATE attachment SET is_sticker = true WHERE attachment_id = %s", (sticker.attachment_id,))
    return {"items": items, "people": (owner, alice, bob), "messages": (m1, m2, m3, m4, m5), "chats": (with_bob, with_alice)}


def test_the_media_grid_filters_by_type_person_and_date(corpus: Corpus) -> None:
    data = _media(corpus)
    items = data["items"]
    _owner, _alice, bob = data["people"]  # type: ignore[misc]
    client = _client(corpus)
    grid = client.get("/media").text  # type: ignore[attr-defined]
    keys = re.findall(r'<figure class="tile[^"]*" data-key="([0-9a-f]{64})"', grid)
    assert keys == [items[k].attachment_key for k in ("other", "video", "pdf", "voice", "photo")]  # type: ignore[index,union-attr]
    assert "sticker.png" not in grid
    assert "Photos 1" in grid and "Voice notes and audio 1" in grid and "PDFs 1" in grid
    photos = client.get("/media", params={"type": "image"}).text  # type: ignore[attr-defined]
    assert re.findall(r'data-key="([0-9a-f]{64})"', photos) == [items["photo"].attachment_key]  # type: ignore[index,union-attr]
    assert f'src="/att/{items["photo"].attachment_key}/thumb"' in photos and 'loading="lazy"' in photos  # type: ignore[index,union-attr]
    from_bob = client.get("/media", params={"sender": bob.short_name}).text  # type: ignore[attr-defined]
    assert set(re.findall(r'<figure class="tile[^"]*" data-key="([0-9a-f]{64})"', from_bob)) == {
        items["photo"].attachment_key, items["voice"].attachment_key  # type: ignore[index,union-attr]
    }
    dated = client.get("/media", params={"from": "2023-04-10", "to": "2023-04-11"}).text  # type: ignore[attr-defined]
    assert set(re.findall(r'<figure class="tile[^"]*" data-key="([0-9a-f]{64})"', dated)) == {
        items["pdf"].attachment_key, items["video"].attachment_key  # type: ignore[index,union-attr]
    }
    m1 = data["messages"][0]  # type: ignore[index]
    with_bob = data["chats"][0]  # type: ignore[index]
    assert f'href="/thread/{with_bob.thread_key}?anchor={m1.message_key}#m-{m1.message_key}"' in grid  # type: ignore[union-attr]
    assert client.get("/media", params={"type": "spreadsheet"}).status_code == 400  # type: ignore[attr-defined]


def test_the_media_grid_pages_sixty_at_a_time(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    for i in range(65):
        msg = corpus.message(chat, DAY + timedelta(minutes=i), alice, None)
        corpus.attachment(msg, filename=f"p{i:02d}.png", mime_type="image/png", content=tiny_png(2, 2 + i))
    client = _client(corpus)
    first = client.get("/media").text  # type: ignore[attr-defined]
    assert first.count('<figure class="tile') == 60
    next_url = re.search(r'data-next="([^"]+)"', first)
    assert next_url is not None
    rest = client.get(html.unescape(next_url.group(1))).text  # type: ignore[attr-defined]
    assert rest.count('<figure class="tile') == 5 and "p00.png" in rest and "data-next" not in rest


@pytest.mark.parametrize(
    ("mime", "filename"),
    [
        ("image/jpeg", "a.jpg"), (None, "IMG_1.HEIC"), ("application/pdf", "x.pdf"), (None, "x.PDF"),
        ("image/png", "scan.pdf"), ("video/quicktime", "c.mov"), (None, "c.M4V"), ("audio/x-caf", "v.caf"),
        (None, "v.amr"), ("text/vcard", "c.vcf"), (None, None), ("application/octet-stream", "notes.txt"),
        (None, "photo.jpeg.zip"),
    ],
)
def test_the_sql_kind_matches_the_page_kind(corpus: Corpus, mime: str | None, filename: str | None) -> None:
    from imsg.search_page.browse import attachment_kind_sql

    with corpus.conn.cursor() as cur:
        cur.execute(
            f"SELECT {attachment_kind_sql('a')} FROM (SELECT %s::text AS mime_type, %s::text AS filename) a",
            (mime, filename),
        )
        row = cur.fetchone()
    assert row is not None and row[0] == attachment_kind(mime, filename)
