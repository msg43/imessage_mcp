"""The per-message Details panel: where a message came from.

Every field the database already holds and the page did not show: time to
the second with the zone offset, sender and service, edit time, delete
date, whether unsent, how the message was filed into its conversation,
which sources hold it with the run that last read each, attachments with
their SHA-256, and the message's IDs. Two fields widen what the page
reveals and have their own switches, off by default:
`search_page.details.show_edit_history` (earlier text of edited messages)
and `search_page.details.show_raw_handles` (the sender's raw number or
email). Both "on" paths and both "off" paths are tested; with a switch
off, the value must not appear anywhere in the response.

Synthetic data only (the numbers are from the reserved 555-01xx range);
needs the scratch Postgres."""

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
    Msg,
    create_scratch_db,
    drop_scratch_db,
    make_page_client,
    open_fts,
    page_settings,
)
from imsg.search_page.config import SearchPageConfig

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_details_test"
SENT = datetime(2023, 4, 24, 18, 28, 7, tzinfo=UTC)
"""Mon 24 Apr 2023, 14:28:07 in New York (EDT, UTC-04:00)."""
RAW_HANDLE = "+15555550142"
EARLIER_TEXT = "Deposit is fully refundable"


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


def _run(corpus: Corpus, source: str, mode: str, finished: datetime) -> int:
    with corpus.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO extraction_run (source_name, snapshot_path, snapshot_sha256, started_at, "
            "finished_at, status, merge_mode) VALUES (%s, '/snapshots/x', 'x', %s, %s, 'ok', %s) "
            "RETURNING run_id",
            (source, finished - timedelta(minutes=2), finished, mode),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _rich_message(corpus: Corpus) -> tuple[Any, Msg, Msg]:
    """A group conversation with one edited, deleted message from Bob,
    held by the live database and an older copy, with a PDF attached."""
    owner = corpus.person("Owner", owner=True)
    alice, bob = corpus.person("Alice Example"), corpus.person("Bob Builder")
    group = corpus.chat([owner, alice, bob], kind="group", display_name="Deck project")
    seg = corpus.segment(
        group,
        [
            (SENT, bob, "Deposit is refundable until materials are ordered"),
            (SENT + timedelta(minutes=1), None, "thanks"),
        ],
    )
    edited, mine = seg.messages
    live = _run(corpus, "mini", "live", datetime(2026, 9, 24, 21, 40, tzinfo=UTC))
    older = _run(corpus, "old-phone", "seed", datetime(2026, 9, 12, 9, 10, tzinfo=UTC))
    with corpus.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO source_handle (raw_value, service) VALUES (%s, 'imessage') "
            "RETURNING source_handle_id",
            (RAW_HANDLE,),
        )
        row = cur.fetchone()
        assert row is not None
        cur.execute(
            """
            UPDATE message SET sender_source_handle_id = %s, is_edited = true, date_edited = %s,
                deleted_at = %s, chat_evidence = 'recoverable_join'
            WHERE message_id = %s
            """,
            (
                int(row[0]),
                SENT + timedelta(seconds=93),
                datetime(2023, 4, 26, 12, 3, 41, tzinfo=UTC),
                edited.message_id,
            ),
        )
        cur.execute(
            "INSERT INTO message_version (message_id, version_idx, text, edited_at) "
            "VALUES (%s, 0, %s, %s)",
            (edited.message_id, EARLIER_TEXT, SENT),
        )
        cur.execute(
            "INSERT INTO message_source (message_id, source_name, source_rowid, extraction_run_id) "
            "VALUES (%s, 'mini', 812345, %s), (%s, 'old-phone', 1234, %s)",
            (edited.message_id, live, edited.message_id, older),
        )
    att = corpus.attachment(edited, filename="deposit-terms.pdf", mime_type="application/pdf", content=b"%PDF-1.4 terms")
    with corpus.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment_source (attachment_id, source_name, source_rowid) VALUES (%s, 'mini', 77)",
            (att.attachment_id,),
        )
    return att, edited, mine


def _client(corpus: Corpus, **details: bool) -> Any:
    return make_page_client(
        corpus,
        DB_NAME,
        settings=page_settings(timezone="America/New_York"),
        details=details,
    )


def test_the_panel_shows_where_a_message_came_from(corpus: Corpus) -> None:
    att, edited, _mine = _rich_message(corpus)
    response = _client(corpus).get(f"/message/{edited.message_key}/details")
    assert response.status_code == 200
    body = response.text
    assert "Mon 24 Apr 2023, 14:28:07 EDT (UTC\u221204:00)" in body
    assert "Bob Builder" in body and "iMessage" in body
    assert "Yes, last edited Mon 24 Apr 2023, 14:29:40 EDT (UTC\u221204:00), 1 minute 33 seconds after sending" in body
    assert "Deleted in Messages on Wed 26 Apr 2023, 08:03:41 EDT (UTC\u221204:00); kept from Recently Deleted" in body
    assert "<dt>Unsent</dt><dd>No</dd>" in body
    assert "Deck project (group)" in body
    assert "filed here because Recently Deleted in the Messages database names this conversation" in body
    assert "mini: the live Messages database, row 812345, read Thu 24 Sep 2026, 17:40:00 EDT" in body
    assert "old-phone: an older or recovered copy, used only to fill gaps, row 1234" in body
    assert body.index("mini: the live") < body.index("old-phone:")
    assert "deposit-terms.pdf" in body and f"SHA-256 <code class=\"sha\">{att.sha256}</code>" in body
    assert "from mini" in body
    assert edited.message_key in body and edited.source_guid in body


def test_both_switches_off_hide_the_raw_handle_and_the_earlier_text(corpus: Corpus) -> None:
    _att, edited, _mine = _rich_message(corpus)
    body = _client(corpus).get(f"/message/{edited.message_key}/details").text
    assert "number or email hidden by setting (search_page.details.show_raw_handles)" in body
    assert "Earlier text hidden by setting (search_page.details.show_edit_history)" in body
    assert RAW_HANDLE not in body and "555" not in body
    assert EARLIER_TEXT not in body


def test_show_edit_history_lists_the_earlier_text(corpus: Corpus) -> None:
    _att, edited, _mine = _rich_message(corpus)
    body = _client(corpus, show_edit_history=True).get(f"/message/{edited.message_key}/details").text
    assert "Earlier text, oldest first:" in body
    assert f"<li>“{EARLIER_TEXT}” <span class=\"muted\">dated Mon 24 Apr 2023, 14:28:07 EDT" in body
    assert "hidden by setting (search_page.details.show_edit_history)" not in body
    assert RAW_HANDLE not in body  # the other switch is still off


def test_show_raw_handles_shows_the_senders_number(corpus: Corpus) -> None:
    _att, edited, mine = _rich_message(corpus)
    client = _client(corpus, show_raw_handles=True)
    body = client.get(f"/message/{edited.message_key}/details").text
    assert f'<span class="handle">{RAW_HANDLE}</span>' in body
    assert "hidden by setting (search_page.details.show_raw_handles)" not in body
    assert EARLIER_TEXT not in body  # the other switch is still off
    # The owner's own message carries no handle: "Me" and the service only.
    own = client.get(f"/message/{mine.message_key}/details").text
    assert "<dt>From</dt><dd>Me · iMessage</dd>" in own


def test_both_switches_on(corpus: Corpus) -> None:
    _att, edited, _mine = _rich_message(corpus)
    body = _client(corpus, show_edit_history=True, show_raw_handles=True).get(
        f"/message/{edited.message_key}/details"
    ).text
    assert RAW_HANDLE in body and EARLIER_TEXT in body
    assert "hidden by setting" not in body


def test_unsent_messages_have_a_panel_only_under_index_unsent(corpus: Corpus) -> None:
    owner = corpus.person("Owner", owner=True)
    alice = corpus.person("Alice Example")
    chat = corpus.chat([owner, alice])
    msg = corpus.message(chat, SENT, alice, "never mind", unsent=True)
    hidden = make_page_client(corpus, DB_NAME).get(f"/message/{msg.message_key}/details")
    assert hidden.status_code == 404
    shown = make_page_client(corpus, DB_NAME, settings=page_settings(index_unsent=True)).get(
        f"/message/{msg.message_key}/details"
    )
    assert shown.status_code == 200 and "<dt>Unsent</dt><dd>Yes</dd>" in shown.text


def test_every_message_has_a_details_button_and_bad_keys_are_refused(corpus: Corpus) -> None:
    _att, edited, _mine = _rich_message(corpus)
    client = _client(corpus)
    results = client.get("/search", params={"q": "deposit"}).text
    assert f'data-url="/message/{edited.message_key}/details"' in results
    assert client.get("/message/" + "0" * 64 + "/details").status_code == 404
    assert client.get("/message/not-a-key/details").status_code == 404
    client.cookies.clear()
    assert client.get(f"/message/{edited.message_key}/details").status_code == 401


def test_the_switches_default_off_and_are_strict() -> None:
    page = SearchPageConfig()
    assert page.details.show_edit_history is False
    assert page.details.show_raw_handles is False
    on = SearchPageConfig.model_validate({"details": {"show_edit_history": True, "show_raw_handles": True}})
    assert on.details.show_edit_history and on.details.show_raw_handles
    with pytest.raises(ValueError):
        SearchPageConfig.model_validate({"details": {"show_everything": True}})
    example = (Path(__file__).resolve().parents[1] / "config.example.yaml").read_text()
    assert "#     show_edit_history: false" in example
    assert "#     show_raw_handles: false" in example
