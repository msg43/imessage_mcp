"""A synthetic, `chat.db`-shaped SQLite fixture builder for tests.

Not a test module itself (no `test_*` functions) — a shared helper
imported by `test_snapshot.py`, `test_extract.py`, and anything else
that needs a `chat.db`-lookalike without a real one (none is available
in this environment; see `CLAUDE.md`/the build brief). Only the columns
`imsg.stages.snapshot`/`imsg.stages.extract` actually read are
modeled — this is deliberately not a byte-exact `chat.db` schema clone.

Apple-epoch helper (`apple_ns`) matches `imsg.stages.extract.APPLE_EPOCH`
(nanoseconds since 2001-01-01 UTC) so fixture timestamps round-trip
through the real conversion code under test.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)

_SCHEMA = """
CREATE TABLE chat (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT UNIQUE,
    style INTEGER,
    display_name TEXT,
    service_name TEXT
);

CREATE TABLE handle (
    ROWID INTEGER PRIMARY KEY,
    id TEXT,
    service TEXT
);

CREATE TABLE chat_handle_join (
    chat_id INTEGER,
    handle_id INTEGER
);

CREATE TABLE message (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT UNIQUE,
    handle_id INTEGER,
    is_from_me INTEGER NOT NULL DEFAULT 0,
    date INTEGER,
    date_edited INTEGER,
    date_retracted INTEGER,
    service TEXT,
    thread_originator_guid TEXT,
    item_type INTEGER NOT NULL DEFAULT 0,
    payload_data BLOB
);

CREATE TABLE chat_message_join (
    chat_id INTEGER,
    message_id INTEGER
);

CREATE TABLE attachment (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT UNIQUE,
    filename TEXT,
    transfer_name TEXT,
    uti TEXT,
    mime_type TEXT,
    total_bytes INTEGER,
    is_sticker INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE message_attachment_join (
    message_id INTEGER,
    attachment_id INTEGER
);
"""


def apple_ns(dt: datetime) -> int:
    """Convert a real datetime to chat.db's Apple-epoch-nanoseconds encoding."""
    return int((dt.astimezone(UTC) - APPLE_EPOCH).total_seconds() * 1_000_000_000)


@dataclass
class FixtureChat:
    guid: str
    style: int = 45  # dm by default
    display_name: str | None = None
    service_name: str = "iMessage"
    rowid: int | None = None


@dataclass
class FixtureHandle:
    raw_value: str
    service: str = "iMessage"
    rowid: int | None = None


@dataclass
class FixtureMessage:
    guid: str
    chat_guid: str
    is_from_me: bool = False
    handle_raw_value: str | None = None
    date: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    date_edited: datetime | None = None
    date_retracted: datetime | None = None
    service: str = "iMessage"
    thread_originator_guid: str | None = None
    item_type: int = 0
    payload_data: bytes | None = None
    rowid: int | None = None


@dataclass
class FixtureAttachment:
    guid: str
    filename: str = "IMG_0001.jpeg"  # transfer_name (display name)
    source_path: str = "~/Library/Messages/Attachments/a/b/IMG_0001.jpeg"  # filename (disk path)
    uti: str = "public.jpeg"
    mime_type: str = "image/jpeg"
    byte_size: int = 12345
    is_sticker: bool = False
    rowid: int | None = None


class ChatDbBuilder:
    """Accumulates fixture rows, then writes them to a real SQLite file
    on disk (a stand-in `chat.db`, or already-a-"snapshot" for tests
    that skip S1)."""

    def __init__(self) -> None:
        self._chats: list[FixtureChat] = []
        self._handles: list[FixtureHandle] = []
        self._messages: list[FixtureMessage] = []
        self._attachments: list[FixtureAttachment] = []
        self._chat_handle_links: list[tuple[str, str]] = []  # (chat_guid, handle_raw_value)
        self._message_attachment_links: list[tuple[str, str]] = []  # (message_guid, attachment_guid)

    def add_chat(self, chat: FixtureChat) -> FixtureChat:
        self._chats.append(chat)
        return chat

    def add_handle(self, handle: FixtureHandle) -> FixtureHandle:
        self._handles.append(handle)
        return handle

    def add_message(self, message: FixtureMessage) -> FixtureMessage:
        self._messages.append(message)
        return message

    def add_attachment(self, attachment: FixtureAttachment) -> FixtureAttachment:
        self._attachments.append(attachment)
        return attachment

    def link_participant(self, chat_guid: str, handle_raw_value: str) -> None:
        self._chat_handle_links.append((chat_guid, handle_raw_value))

    def link_attachment(self, message_guid: str, attachment_guid: str) -> None:
        self._message_attachment_links.append((message_guid, attachment_guid))

    def build(self, path: Path) -> Path:
        conn = sqlite3.connect(str(path))
        try:
            conn.executescript(_SCHEMA)

            chat_rowid_by_guid: dict[str, int] = {}
            for i, chat in enumerate(self._chats, start=1):
                rowid = chat.rowid if chat.rowid is not None else i
                conn.execute(
                    "INSERT INTO chat (ROWID, guid, style, display_name, service_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rowid, chat.guid, chat.style, chat.display_name, chat.service_name),
                )
                chat_rowid_by_guid[chat.guid] = rowid

            handle_rowid_by_value: dict[str, int] = {}
            for i, handle in enumerate(self._handles, start=1):
                rowid = handle.rowid if handle.rowid is not None else i
                conn.execute(
                    "INSERT INTO handle (ROWID, id, service) VALUES (?, ?, ?)",
                    (rowid, handle.raw_value, handle.service),
                )
                handle_rowid_by_value[handle.raw_value] = rowid

            for chat_guid, handle_value in self._chat_handle_links:
                conn.execute(
                    "INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (?, ?)",
                    (chat_rowid_by_guid[chat_guid], handle_rowid_by_value[handle_value]),
                )

            message_rowid_by_guid: dict[str, int] = {}
            for i, message in enumerate(self._messages, start=1):
                rowid = message.rowid if message.rowid is not None else i
                handle_id = (
                    handle_rowid_by_value[message.handle_raw_value]
                    if message.handle_raw_value is not None
                    else None
                )
                conn.execute(
                    """
                    INSERT INTO message (
                        ROWID, guid, handle_id, is_from_me, date, date_edited, date_retracted,
                        service, thread_originator_guid, item_type, payload_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rowid,
                        message.guid,
                        handle_id,
                        int(message.is_from_me),
                        apple_ns(message.date),
                        apple_ns(message.date_edited) if message.date_edited else None,
                        apple_ns(message.date_retracted) if message.date_retracted else None,
                        message.service,
                        message.thread_originator_guid,
                        message.item_type,
                        message.payload_data,
                    ),
                )
                message_rowid_by_guid[message.guid] = rowid
                conn.execute(
                    "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
                    (chat_rowid_by_guid[message.chat_guid], rowid),
                )

            attachment_rowid_by_guid: dict[str, int] = {}
            for i, att in enumerate(self._attachments, start=1):
                rowid = att.rowid if att.rowid is not None else i
                conn.execute(
                    """
                    INSERT INTO attachment (
                        ROWID, guid, filename, transfer_name, uti, mime_type, total_bytes, is_sticker
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rowid,
                        att.guid,
                        att.source_path,
                        att.filename,
                        att.uti,
                        att.mime_type,
                        att.byte_size,
                        int(att.is_sticker),
                    ),
                )
                attachment_rowid_by_guid[att.guid] = rowid

            for message_guid, attachment_guid in self._message_attachment_links:
                conn.execute(
                    "INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (?, ?)",
                    (message_rowid_by_guid[message_guid], attachment_rowid_by_guid[attachment_guid]),
                )

            conn.commit()
        finally:
            conn.close()
        return path


__all__ = [
    "ChatDbBuilder",
    "FixtureAttachment",
    "FixtureChat",
    "FixtureHandle",
    "FixtureMessage",
    "apple_ns",
]
