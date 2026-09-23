"""S2 keeps every source's recorded attachment path (D12, D13).

`attachment.source_path` holds one path. Under D12 a seed only fills it
when it is empty, so when two Macs recorded different paths for the same
attachment, the second Mac's path was dropped — and with it the only
pointer the fetcher had to that Mac's copy. On 2026-09-23, 668 of the
index's 978 `missing` attachments had a copy on another Mac, at exactly
the path that Mac's own chat.db recorded, which the index had never
stored.

Extraction now files each source's path in `attachment_location` under
the source's name, insert-only.

Fictional personas only (D5).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from _attachment_fetch_fixtures import requires_postgres, scratch_database
from chatdb_fixture import (
    ChatDbBuilder,
    FixtureAttachment,
    FixtureChat,
    FixtureHandle,
    FixtureMessage,
)
from imsg.stages.extract import MergeMode, run_extract
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun

pytestmark = requires_postgres

ALICE = "+15550000001"
CHAT = "chat-alice"
MSG_PHOTO = "msg-photo"
MSG_VOICE = "msg-voice"
ATT_PHOTO = "8C1D2E3F-0000-4000-8000-00000000A001"
ATT_VOICE = "8C1D2E3F-0000-4000-8000-00000000A002"

PATH_ON_MINI = f"~/Library/Messages/Attachments/0a/10/{ATT_PHOTO}/IMG_0001.heic"
PATH_ON_STUDIO = f"~/Library/Messages/Attachments/7f/15/{ATT_PHOTO}/IMG_0001.heic"


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
    yield from scratch_database("imsg_index_location_extract_test")


def _snapshot(path: Path, *, photo_path: str | None, voice_path: str | None) -> Path:
    builder = ChatDbBuilder()
    builder.add_chat(FixtureChat(guid=CHAT, rowid=1))
    builder.add_handle(FixtureHandle(raw_value=ALICE, rowid=1))
    builder.link_participant(CHAT, ALICE)
    for rowid, guid in ((1, MSG_PHOTO), (2, MSG_VOICE)):
        builder.add_message(
            FixtureMessage(guid=guid, chat_guid=CHAT, handle_raw_value=ALICE, rowid=rowid,
                           date=datetime(2024, 6, 1, 9, rowid, tzinfo=UTC))
        )
    builder.add_attachment(
        FixtureAttachment(guid=ATT_PHOTO, rowid=1, filename="IMG_0001.heic",
                          source_path=photo_path, byte_size=2048)
    )
    builder.add_attachment(
        FixtureAttachment(guid=ATT_VOICE, rowid=2, filename="Audio Message.caf",
                          source_path=voice_path, mime_type="audio/x-caf", uti="com.apple.coreaudio-format")
    )
    builder.link_attachment(MSG_PHOTO, ATT_PHOTO)
    builder.link_attachment(MSG_VOICE, ATT_VOICE)
    return builder.build(path)


def _dump() -> ImsgDumpRun:
    def message(guid: str, rowid: int) -> ImsgDumpMessage:
        return ImsgDumpMessage(
            rowid=rowid, guid=guid, chat_guid=None, handle=None, is_from_me=False, date=None,
            date_edited=None, date_retracted=None, service="iMessage", body_text=None,
            edit_history=(), is_unsent=False, tapback=None, attachment_rowids=(),
            reply_to_guid=None,
        )

    return ImsgDumpRun(messages=(message(MSG_PHOTO, 1), message(MSG_VOICE, 2)), stderr_lines=())


def _extract(conn: psycopg.Connection, snapshot: Path, source: str, mode: MergeMode,
             *, dry_run: bool = False) -> Any:
    binary = snapshot.parent / "imsg-dump"
    binary.write_text("")
    return run_extract(
        conn=conn, source_name=source, snapshot_path=snapshot, imsg_dump_binary=binary,
        run_imsg_dump_fn=lambda _b, _s, _r: _dump(), merge_mode=mode, dry_run=dry_run,
    )


def _locations(conn: psycopg.Connection) -> list[tuple[str, str, str, str, list[str]]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.source_guid, l.location, l.path, l.match_quality::text, l.reported_by
              FROM attachment_location l JOIN attachment a USING (attachment_id)
             ORDER BY a.source_guid, l.location
            """
        )
        return [tuple(r) for r in cur.fetchall()]  # type: ignore[misc]


def test_every_source_keeps_its_own_recorded_path(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    live = _snapshot(tmp_path / "mini.db", photo_path=PATH_ON_MINI, voice_path=None)
    seed = _snapshot(tmp_path / "studio.db", photo_path=PATH_ON_STUDIO, voice_path=None)
    _extract(pg_conn, live, "mini", MergeMode.LIVE)
    _extract(pg_conn, seed, "studio", MergeMode.SEED)

    # D12 still holds for the attachment row itself: the seed filled
    # nothing, so the index's one path is the live Mac's.
    with pg_conn.cursor() as cur:
        cur.execute("SELECT source_path FROM attachment WHERE source_guid = %s", (ATT_PHOTO,))
        row = cur.fetchone()
    assert row is not None and row[0] == PATH_ON_MINI

    # And the Studio's path is no longer lost.
    assert _locations(pg_conn) == [
        (ATT_PHOTO, "mini", PATH_ON_MINI, "recorded_path", ["mini"]),
        (ATT_PHOTO, "studio", PATH_ON_STUDIO, "recorded_path", ["studio"]),
    ]


def test_a_seed_that_alone_recorded_a_path_is_kept(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The live chat.db never landed the voice note (no path); another Mac
    did. The index's `source_path` is filled, and the row names the Mac."""
    voice_on_studio = f"~/Library/Messages/Attachments/3c/12/{ATT_VOICE}/Audio Message.caf"
    _extract(pg_conn, _snapshot(tmp_path / "mini.db", photo_path=PATH_ON_MINI, voice_path=None),
             "mini", MergeMode.LIVE)
    _extract(pg_conn, _snapshot(tmp_path / "studio.db", photo_path=PATH_ON_MINI,
                                voice_path=voice_on_studio), "studio", MergeMode.SEED)
    rows = _locations(pg_conn)
    assert (ATT_VOICE, "studio", voice_on_studio, "recorded_path", ["studio"]) in rows
    assert not any(r[0] == ATT_VOICE and r[1] == "mini" for r in rows)


def test_rerunning_a_source_records_nothing_new_and_rewrites_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    snapshot = _snapshot(tmp_path / "mini.db", photo_path=PATH_ON_MINI, voice_path=None)
    first = _extract(pg_conn, snapshot, "mini", MergeMode.LIVE)
    assert first.attachment_location_rows.inserted == 1
    with pg_conn.cursor() as cur:
        cur.execute("SELECT location_id, xmin::text FROM attachment_location")
        before = cur.fetchall()

    pg_conn.execute("DELETE FROM sync_state WHERE key = 'watermark.rowid.mini'")
    again = _extract(pg_conn, snapshot, "mini", MergeMode.LIVE)
    assert again.attachment_location_rows.inserted == 0
    assert again.attachment_location_rows.unchanged == 1
    assert again.table_counts()["attachment_location"] == again.attachment_location_rows
    with pg_conn.cursor() as cur:
        cur.execute("SELECT location_id, xmin::text FROM attachment_location")
        assert cur.fetchall() == before


def test_a_dry_run_records_no_path(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "mini.db", photo_path=PATH_ON_MINI, voice_path=None)
    result = _extract(pg_conn, snapshot, "mini", MergeMode.LIVE, dry_run=True)
    assert result.attachment_location_rows.inserted == 1
    assert _locations(pg_conn) == []
