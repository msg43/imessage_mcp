"""A search matches what people wrote, not the index's header lines.

Every indexed segment's text begins with a `Chat:` line (participant
names, the group's name) and a `Time:` line (date range, time zone), and
labels each message `[HH:MM] short-name:`. A search for a name, a year,
an hour or a time-zone word used to match every such segment although no
message held the words. These tests pin the fix: a segment hit is kept
only when a message, an attachment's text or filename, or (under
`policy.index_edit_history`) an earlier version matches, with the index's
own folding of case and accents.

Synthetic people and text only. Needs the scratch Postgres (skips
without it)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
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
)
from imsg.search_page.search import (
    CHANNEL_TEXT,
    CHANNEL_UNINDEXED,
    SearchRequest,
    run_fulltext,
)
from imsg.search_page.server import open_fts_reader

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_header_matches_test"
T0 = datetime(2024, 3, 1, 18, 58, tzinfo=UTC)
"""13:58 in New York, where one test renders its segments."""


@pytest.fixture
def pg() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(DB_NAME)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(DB_NAME)


@pytest.fixture
def corpus(pg: psycopg.Connection, tmp_path: Path) -> Iterator[Corpus]:
    root = tmp_path / "data_root"
    (root / "fts").mkdir(parents=True)
    (root / "attachments").mkdir()
    fts = open_fts(root / "fts" / "fts.db")
    try:
        yield Corpus(conn=pg, fts=fts, data_root=root)
    finally:
        fts.close()


def run(corpus: Corpus, query: str, **settings: Any) -> Any:
    reader = open_fts_reader(corpus.data_root / "fts" / "fts.db")
    try:
        return run_fulltext(
            corpus.conn, reader, SearchRequest(query=query), page_settings(**settings)
        )
    finally:
        reader.close()


def test_names_times_and_labels_in_the_segment_header_do_not_match(corpus: Corpus) -> None:
    corpus.render_timezone = "America/New_York"
    owner = corpus.person("Owner", owner=True)
    dana = corpus.person("Dana Ashdown")
    alice = corpus.person("Alice Example")
    dm = corpus.chat([owner, dana])
    group = corpus.chat([owner, dana, alice], kind="group", display_name="Deck project")
    corpus.segment(dm, [(T0, dana, "see you then"), (T0 + timedelta(minutes=3), None, "ok")])
    corpus.segment(group, [(T0 + timedelta(days=2), alice, "lumber is here")])
    mention = corpus.segment(
        dm, [(T0 + timedelta(days=9), None, "I told Ashdown about the permit")]
    )

    for header_only in ("deck", "project", "2024", "13", "58", "new york", "chat", "time", "owner"):
        result = run(corpus, header_only)
        assert result.total_hits == 0, header_only
        assert result.hidden_non_content >= 1, header_only

    result = run(corpus, "Ashdown")
    assert {h.segment_id for h in result.hits.values()} == {mention.segment_id}
    assert result.counts[CHANNEL_TEXT] == 1
    assert result.hidden_non_content == 2  # every other segment Dana is in


def test_accents_and_case_fold_both_ways(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    accented = corpus.segment(chat, [(T0, alice, "Meet at the Café")])
    plain = corpus.segment(chat, [(T0 + timedelta(days=1), alice, "CAFE closes early")])

    both = {accented.segment_id, plain.segment_id}
    assert {h.segment_id for h in run(corpus, "cafe").hits.values()} == both
    assert {h.segment_id for h in run(corpus, "café").hits.values()} == both
    assert {h.segment_id for h in run(corpus, '"CAFÉ"').hits.values()} == {accented.segment_id}


def test_attachment_filenames_and_extracted_text_are_content(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(
        chat,
        [(T0, alice, "attached")],
        extra_text='[pdf "quarzite-invoice.pdf": "Balance owed on arrival" — full text via x]',
    )
    att = corpus.attachment(
        seg.messages[0], filename="quarzite-invoice.pdf", mime_type="application/pdf", content=b"%PDF"
    )
    corpus.enrichment(att, "pdf_text", "Balance owed on arrival")
    corpus.chunk(att, "Balance owed on arrival")

    by_name = run(corpus, "quarzite")
    assert {h.segment_id for h in by_name.hits.values()} == {seg.segment_id}
    [hit] = by_name.hits.values()
    assert CHANNEL_TEXT in hit.ranks
    assert (seg.messages[0].message_id, att.attachment_id) in hit.matched_attachments
    # One word in the PDF's text: the attachment channel shows the hit, so
    # nothing is reported hidden.
    by_text = run(corpus, "owed")
    assert by_text.total_hits == 1 and by_text.hidden_non_content == 0
    # Two words, one in a message and one in the PDF's text: only the
    # segment's own index row holds both, and the check keeps it.
    across = run(corpus, "arrival attached")
    assert across.counts[CHANNEL_TEXT] == 1
    # The words the renderer wraps around every PDF snippet are not content,
    # but a word in the file's own name is ("pdf" here).
    for template_word in ("full", "text", "via"):
        assert run(corpus, template_word).total_hits == 0, template_word
    assert run(corpus, "pdf").total_hits == 1


def test_a_photo_filename_is_not_content(corpus: Corpus) -> None:
    """The renderer writes no filename for photos, video or audio, so a
    photo's name cannot be why the index matched."""
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice], kind="group", display_name="Lakehouse crew")
    seg = corpus.segment(chat, [(T0, alice, "look")])
    corpus.attachment(seg.messages[0], filename="lakehouse.jpg", mime_type="image/jpeg", content=b"jpg")
    result = run(corpus, "lakehouse")
    assert result.total_hits == 0 and result.hidden_non_content == 1


def test_earlier_versions_count_only_when_the_index_holds_them(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(
        chat, [(T0, alice, "deposit is refundable")], extra_text='[edited from: "fully vexbar"]'
    )
    with corpus.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO message_version (message_id, version_idx, text, edited_at) "
            "VALUES (%s, 0, 'fully vexbar', %s)",
            (seg.messages[0].message_id, T0 + timedelta(seconds=93)),
        )
    assert run(corpus, "vexbar").total_hits == 0
    assert run(corpus, "vexbar", index_edit_history=True).total_hits == 1


def test_unsent_messages_count_only_under_index_unsent(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    seg = corpus.segment(chat, [(T0, alice, "the plorvex plan"), (T0 + timedelta(minutes=1), None, "ok")])
    with corpus.conn.cursor() as cur:
        cur.execute(
            "UPDATE message SET is_unsent = true WHERE message_id = %s",
            (seg.messages[0].message_id,),
        )
    assert run(corpus, "plorvex").total_hits == 0
    assert run(corpus, "plorvex", index_unsent=True).total_hits == 1


def test_recent_unindexed_messages_fold_accents_too(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    recent = datetime.now(UTC) - timedelta(days=1)
    fresh = corpus.message(chat, recent, alice, "the CAFÉ permit came through")
    corpus.message(chat, recent, alice, "cafeteria menu")  # a different word

    result = run(corpus, "cafe")
    assert result.counts[CHANNEL_UNINDEXED] == 1
    [hit] = result.hits.values()
    assert hit.message_id == fresh.message_id


def test_the_page_says_how_many_header_only_matches_it_hid(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    dana = corpus.person("Dana Ashdown")
    chat = corpus.chat([owner, dana])
    for i in range(3):
        corpus.segment(chat, [(T0 + timedelta(days=i), dana, f"note {i}")])
    client = make_page_client(corpus, DB_NAME)
    body = client.get("/search", params={"q": "ashdown"}).text
    assert "0 hits" in body
    assert "Not shown: 3 matches where the words were only in names, dates, times or labels" in body
