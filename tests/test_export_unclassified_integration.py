"""The weekly unclassified-threads report (SPEC §11.5): surfaces the
right threads, and — critically — leaks no message content while doing
so (it reports on threads that are NOT allowlisted). Fictional
personas only (D5)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from _export_fixtures import (
    add_participant,
    add_raw_participant,
    admin_reachable,
    allow,
    create_scratch_db,
    drop_scratch_db,
    insert_chat,
    insert_message,
    insert_person,
)
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.export.unclassified import unclassified_summary, write_unclassified_report

TEST_DB_NAME = "imsg_index_export_unclass_test"

pytestmark = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)

_NOW = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(TEST_DB_NAME)


@pytest.fixture
def config(config_dict_factory: object) -> Config:
    return load_config_dict(config_dict_factory())  # type: ignore[operator]


def test_report_lists_active_unclassified_threads_without_content(
    db: psycopg.Connection, config: Config
) -> None:
    owner_id = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    allow(db, owner_id)
    carol_id = insert_person(db, display_name="Carol Carpenter", short_name="carol")
    # carol: NOT allowlisted, active recently
    chat_id = insert_chat(db, source_guid="chat-c", kind="group", display_name="Weekend plans")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, carol_id)
    insert_message(
        db, chat_id=chat_id, sender_person_id=carol_id, is_from_me=False,
        sent_at=_NOW - timedelta(days=3),
        text="EXTREMELY PRIVATE message body that must never appear in any report",
    )
    add_raw_participant(db, chat_id, "+15550001111")  # unresolved handle
    db.commit()

    assert unclassified_summary(db, now=_NOW) == 1
    path = write_unclassified_report(db, config, now=_NOW)
    text = path.read_text(encoding="utf-8")

    assert path.parent == config.paths.data_root / "export"
    assert "staging" not in str(path)
    assert 'group "Weekend plans"' in text
    assert "carol*" in text  # flagged as not allowlisted
    assert "+1 unresolved handle" in text
    # The leak checks: no content, no raw handles.
    assert "EXTREMELY PRIVATE" not in text
    assert "+15550001111" not in text


def test_classified_and_fully_allowlisted_threads_are_excluded(
    db: psycopg.Connection, config: Config
) -> None:
    owner_id = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    allow(db, owner_id)
    alice_id = insert_person(db, display_name="Alice Example", short_name="alice")
    allow(db, alice_id)
    business = insert_chat(db, source_guid="chat-b")
    add_participant(db, business, owner_id)
    add_participant(db, business, alice_id)
    insert_message(
        db, chat_id=business, sender_person_id=alice_id, is_from_me=False,
        sent_at=_NOW - timedelta(days=1), text="fully classified thread",
    )

    carol_id = insert_person(db, display_name="Carol Carpenter", short_name="carol")
    reviewed = insert_chat(db, source_guid="chat-r")
    add_participant(db, reviewed, owner_id)
    add_participant(db, reviewed, carol_id)
    insert_message(
        db, chat_id=reviewed, sender_person_id=carol_id, is_from_me=False,
        sent_at=_NOW - timedelta(days=1), text="reviewed personal thread",
    )
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO thread_classification (chat_id, state, reviewed_at) "
            "VALUES (%s, 'personal', now())",
            (reviewed,),
        )
    db.commit()

    assert unclassified_summary(db, now=_NOW) == 0
    text = write_unclassified_report(db, config, now=_NOW).read_text(encoding="utf-8")
    assert "Nothing to classify" in text


def test_stale_threads_outside_the_window_are_excluded(
    db: psycopg.Connection, config: Config
) -> None:
    owner_id = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    carol_id = insert_person(db, display_name="Carol Carpenter", short_name="carol")
    old_chat = insert_chat(db, source_guid="chat-old")
    add_participant(db, old_chat, owner_id)
    add_participant(db, old_chat, carol_id)
    insert_message(
        db, chat_id=old_chat, sender_person_id=carol_id, is_from_me=False,
        sent_at=_NOW - timedelta(days=200), text="ancient history",
    )
    db.commit()
    assert unclassified_summary(db, now=_NOW) == 0
