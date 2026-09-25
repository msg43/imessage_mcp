"""Evidence cases: collect messages and files, with notes and saved
searches, and download them with exact citations.

"Add to case" on every message, file, timeline row and media tile adds to
the open case (the first add starts one). A case page lists its items in
the order they were sent, with notes; saved searches show how many of
their conversations are marked reviewed; the case downloads as Markdown,
CSV or JSON, optionally in a zip with the original files and a SHA-256
list. Items are keyed by message and attachment keys, so they survive
re-segmentation. Every POST needs the CSRF token and a same-site request.

Synthetic data only; needs the scratch Postgres."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
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
    tiny_png,
)

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_cases_test"
T0 = datetime(2023, 4, 7, 17, 58, 7, tzinfo=UTC)
"""Fri 7 Apr 2023, 13:58:07 in New York (EDT)."""
RAW_HANDLE = "+15555550199"


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
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    deck = corpus.chat([owner, alice, bob], kind="group", display_name="Deck project")
    with_bob = corpus.chat([owner, bob])
    first = corpus.segment(deck, [(T0, alice, "the deposit plan"), (T0 + timedelta(minutes=2), None, "deposit is fine")])
    later = corpus.segment(with_bob, [(T0 + timedelta(days=2), bob, "deposit refund terms attached")])
    receipt = corpus.attachment(
        later.messages[0], filename="receipt.png", mime_type="image/png", content=tiny_png(6, 6)
    )
    with corpus.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO source_handle (raw_value, service) VALUES (%s, 'imessage') RETURNING source_handle_id",
            (RAW_HANDLE,),
        )
        handle = cur.fetchone()
        assert handle is not None
        cur.execute(
            "UPDATE message SET sender_source_handle_id = %s, is_edited = true, date_edited = %s WHERE message_id = %s",
            (int(handle[0]), T0 + timedelta(days=2, seconds=40), later.messages[0].message_id),
        )
        cur.execute(
            "INSERT INTO message_version (message_id, version_idx, text, edited_at) VALUES (%s, 0, 'full refund', %s)",
            (later.messages[0].message_id, T0 + timedelta(days=2)),
        )
    return {"chats": (deck, with_bob), "first": first, "later": later, "receipt": receipt}


def _client(corpus: Corpus, **details: bool) -> Any:
    return make_page_client(
        corpus, DB_NAME, settings=page_settings(timezone="America/New_York"), details=details
    )


def _token(client: Any) -> str:
    return csrf_token_of(client.get("/case").text)


def _add(client: Any, token: str, message_key: str, attachment_key: str | None = None, add: bool = True) -> Any:
    return client.post(
        "/api/case/item",
        json={"message_key": message_key, "attachment_key": attachment_key, "add": add},
        headers={"X-CSRF-Token": token},
    )


def _only_case_id(corpus: Corpus) -> int:
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT case_id FROM search_case")
        rows = cur.fetchall()
    assert len(rows) == 1
    return int(rows[0][0])


def test_adding_needs_the_csrf_token_and_starts_a_case(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    key = world["later"].messages[0].message_key
    assert client.post("/api/case/item", json={"message_key": key, "add": True}).status_code == 403
    token = _token(client)
    cross = client.post(
        "/api/case/item", json={"message_key": key, "add": True},
        headers={"X-CSRF-Token": token, "Origin": "http://evil.example"},
    )
    assert cross.status_code == 403
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM search_case")
        assert cur.fetchone() == (0,)
    added = _add(client, token, key)
    assert added.status_code == 200, added.text
    answer = added.json()
    assert answer["in_case"] is True and answer["count"] == 1
    assert answer["case_name"].startswith("Case started ")
    results = client.get("/search", params={"q": "deposit"}).text
    assert f'class="link case-btn on" data-message="{key}"' in results
    assert "In case \u2713" in results
    assert _add(client, token, key, add=False).json()["count"] == 0
    assert _add(client, token, "0" * 64).status_code == 404
    client.cookies.clear()
    assert client.post("/api/case/item", json={"message_key": key, "add": True}).status_code == 401


def test_every_message_file_row_and_tile_can_be_added(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    msg = world["later"].messages[0]
    receipt = world["receipt"]
    thread = client.get(f"/thread/{world['chats'][1].thread_key}").text
    assert f'data-message="{msg.message_key}" aria-pressed' in thread
    assert f'data-message="{msg.message_key}" data-attachment="{receipt.attachment_key}"' in thread
    timeline = client.get("/timeline", params={"from": "2023-04-09", "to": "2023-04-09"}).text
    assert f'class="link case-btn" data-message="{msg.message_key}"' in timeline
    media = client.get("/media").text
    assert f'data-attachment="{receipt.attachment_key}"' in media
    token = _token(client)
    assert _add(client, token, msg.message_key, receipt.attachment_key).status_code == 200
    media_again = client.get("/media").text
    assert "In case \u2713" in media_again
    other = world["first"].messages[0]
    assert _add(client, token, other.message_key, receipt.attachment_key).status_code == 404  # not its file


def test_the_case_page_lists_items_in_sent_order_with_notes(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    token = _token(client)
    later, first = world["later"].messages[0], world["first"].messages[0]
    _add(client, token, later.message_key)
    _add(client, token, first.message_key)
    case_id = _only_case_id(corpus)
    page = client.get(f"/case/{case_id}").text
    assert page.index("the deposit plan") < page.index("deposit refund terms attached")
    assert "Fri 07 Apr 2023, 13:58:07 EDT (UTC\u221204:00)" in page
    assert "in <a href=" in page and "Deck project</a> (group)" in page
    assert "Bob Builder</a> (one-to-one)" in page
    assert "export" not in page.lower()
    assert client.post(f"/case/{case_id}/notes", data={"notes": "x"}, follow_redirects=False).status_code == 403
    saved = client.post(
        f"/case/{case_id}/notes", data={"csrf_token": token, "notes": "Did Bob promise a refund?"},
        follow_redirects=False,
    )
    assert saved.status_code == 303
    item_id = int(re.findall(r'id="item-(\d+)"', page)[1])
    client.post(f"/case/item/{item_id}/note", data={"csrf_token": token, "note": "the key promise"})
    page = client.get(f"/case/{case_id}").text
    assert "Did Bob promise a refund?" in page and "the key promise" in page
    client.post(f"/case/item/{item_id}/remove", data={"csrf_token": token})
    assert "deposit refund terms attached" not in client.get(f"/case/{case_id}").text


def test_items_survive_re_segmentation(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    token = _token(client)
    _add(client, token, world["first"].messages[0].message_key)
    with corpus.conn.cursor() as cur:
        cur.execute("DELETE FROM segment")  # every segment rebuilt from scratch
    page = client.get(f"/case/{_only_case_id(corpus)}").text
    assert "the deposit plan" in page and "no longer shown" not in page


def test_saved_searches_count_reviewed_conversations(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    token = _token(client)
    results = client.get("/search", params={"q": "deposit"}).text
    assert "Save this search" in results
    saved = client.post("/api/case/search", json={"q": "deposit", "att": "any"}, headers={"X-CSRF-Token": token})
    assert saved.status_code == 200
    search_id = saved.json()["search_id"]
    results = client.get("/search", params={"q": "deposit"}).text
    assert 'class="n-reviewed">0</span> of 2 conversations' in results
    assert results.count('class="review-box"') == 2
    deck = world["chats"][0]
    marked = client.post(
        "/api/case/review", json={"search_id": search_id, "thread_key": deck.thread_key, "reviewed": True},
        headers={"X-CSRF-Token": token},
    )
    assert marked.json() == {"reviewed": 1}
    results = client.get("/search", params={"q": "deposit"}).text
    assert f'data-thread="{deck.thread_key}" checked' in results
    case_page = client.get(f"/case/{_only_case_id(corpus)}").text
    assert "reviewed 1 of 2 conversations" in case_page
    # A different filter is a different search: nothing saved for it.
    assert "Save this search" in client.get("/search", params={"q": "deposit", "att": "with"}).text


def _downloaded(client: Any, case_id: int, fmt: str, **extra: str) -> Any:
    response = client.get(f"/case/{case_id}/download", params={"fmt": fmt, **extra})
    assert response.status_code == 200, response.text
    return response


def test_downloads_carry_exact_citations(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    token = _token(client)
    later = world["later"].messages[0]
    _add(client, token, later.message_key)
    case_id = _only_case_id(corpus)
    client.post(f"/case/{case_id}/rename", data={"csrf_token": token, "name": "Deck deposit dispute"})
    md = _downloaded(client, case_id, "md")
    assert md.headers["content-disposition"].startswith('attachment; filename="case-deck-deposit-dispute-')
    assert md.headers["content-disposition"].endswith('.md"')
    text = md.text
    assert text.startswith("# Deck deposit dispute")
    assert "Sun 09 Apr 2023, 13:58:07 EDT (UTC\u221204:00) \u00b7 Bob Builder \u00b7 iMessage \u00b7 in \u201cBob Builder\u201d" in text
    assert "\u201cdeposit refund terms attached\u201d" in text
    assert f"Message ID {later.message_key} \u00b7 Messages GUID {later.source_guid}" in text
    assert f"SHA-256 {world['receipt'].sha256}" in text
    assert RAW_HANDLE not in text and "full refund" not in text  # both switches off
    data = json.loads(_downloaded(client, case_id, "json").text)
    [item] = data["items"]
    assert item["message_id"] == later.message_key and item["sender_handle"] is None
    assert item["files"][0]["sha256"] == world["receipt"].sha256
    rows = list(csv.DictReader(io.StringIO(_downloaded(client, case_id, "csv").text)))
    assert rows[0]["messages_guid"] == later.source_guid and rows[0]["sent_at"].startswith("2023-04-09T17:58:07")
    for fmt in ("md", "csv", "json"):
        assert "export" not in _downloaded(client, case_id, fmt).text.lower()
    assert client.get(f"/case/{case_id}/download", params={"fmt": "docx"}).status_code == 400


def test_downloads_follow_the_details_switches(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus, show_edit_history=True, show_raw_handles=True)
    token = _token(client)
    _add(client, token, world["later"].messages[0].message_key)
    text = _downloaded(client, _only_case_id(corpus), "md").text
    assert f"Bob Builder ({RAW_HANDLE})" in text
    assert "Earlier text: \u201cfull refund\u201d" in text


def test_a_download_with_files_is_a_zip_with_a_sha256_list(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    token = _token(client)
    _add(client, token, world["later"].messages[0].message_key, world["receipt"].attachment_key)
    case_id = _only_case_id(corpus)
    response = _downloaded(client, case_id, "json", files="1")
    assert response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    [case_file] = [n for n in names if n.endswith(".json")]
    [original] = [n for n in names if n.startswith("files/")]
    assert original == f"files/{world['receipt'].sha256[:12]}-receipt.png"
    assert archive.read(original) == tiny_png(6, 6)
    sums = archive.read("SHA256SUMS.txt").decode()
    assert f"{hashlib.sha256(archive.read(original)).hexdigest()}  {original}" in sums
    assert f"{hashlib.sha256(archive.read(case_file)).hexdigest()}  {case_file}" in sums
    assert json.loads(archive.read(case_file))["items"][0]["files"][0]["path"] == original
    assert not list((corpus.data_root / "search-page" / "downloads").glob("*.zip"))  # removed after sending


def test_cases_open_one_at_a_time_and_delete_after_confirming(corpus: Corpus) -> None:
    world = _world(corpus)
    client = _client(corpus)
    token = _token(client)
    first = client.post("/case", data={"csrf_token": token, "name": "First case"}, follow_redirects=False)
    second = client.post("/case", data={"csrf_token": token, "name": "Second case"}, follow_redirects=False)
    first_id = int(first.headers["location"].rsplit("/", 1)[1])
    second_id = int(second.headers["location"].rsplit("/", 1)[1])
    assert _add(client, token, world["first"].messages[0].message_key).json()["case_id"] == second_id
    client.post(f"/case/{first_id}/activate", data={"csrf_token": token})
    assert _add(client, token, world["later"].messages[0].message_key).json()["case_id"] == first_id
    with corpus.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM search_case WHERE is_active")
        assert cur.fetchone() == (1,)
    asked = client.post(f"/case/{second_id}/delete", data={"csrf_token": token}, follow_redirects=False)
    assert asked.headers["location"] == f"/case/{second_id}?delete=1"
    assert "Delete this case" in client.get(f"/case/{second_id}", params={"delete": "1"}).text
    gone = client.post(f"/case/{second_id}/delete", data={"csrf_token": token, "confirm": "1"}, follow_redirects=False)
    assert gone.headers["location"] == "/case"
    assert client.get(f"/case/{second_id}").status_code == 404
    assert "the deposit plan" in client.get("/search", params={"q": "plan"}).text  # messages stay
    assert client.post("/case", data={"csrf_token": token, "name": "   "}, follow_redirects=False).status_code == 400
