"""Identity curation must reach the index. A segment's rendered text
carries people's names — every non-owner participant's `display_name`
in the header, each sender's `short_name` on its message lines — and S4
only re-segments a chat when a message's `updated_at` has moved
(`find_dirty_chats`). So `rename_person`, `merge_persons` and
`assign_handle` mark every chat whose rendered segments name the
persons involved (`imsg.stages.identity._mark_chats_dirty_for_persons`),
and this suite drives the whole path: segment, embed, curate, observe
the chat come back dirty, re-segment, and check the new text and the
superseded embedding.

DB-gated, same scratch-instance pattern as
`test_segment_pipeline_integration.py`, whose fixture-data helpers this
reuses. Fictional personas only (Jamie Owner, Alice Example, Bob
Builder, Carol Chen).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import psycopg
import pytest

from imsg import constants
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.pipeline import run_embed
from imsg.embed.provider import FakeTextEmbeddingProvider
from imsg.errors import IdentityError
from imsg.hashing import sha256_text
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import (
    REBUILD_ALL_SENTINEL,
    find_dirty_chats,
    refresh_segment_rendering,
    run_segment_for_chat,
)
from imsg.stages.identity import assign_handle, merge_persons, rename_person
from test_segment_pipeline_integration import (
    _BASE,
    ADMIN_DSN,
    PROMPT_BYTES,
    REACHABLE,
    REAL_MIGRATIONS_DIR,
    _add_participant,
    _dsn,
    _insert_chat,
    _insert_message,
    _insert_person,
)

TEST_DB_NAME = "imsg_index_identity_dirty_test"

pytestmark = pytest.mark.skipif(
    not REACHABLE,
    reason="no reachable scratch Postgres instance (see test_segment_pipeline_integration.py)",
)


@pytest.fixture
def config(config_dict_factory: object) -> Config:
    return load_config_dict(config_dict_factory())  # type: ignore[operator]


@pytest.fixture
def scratch_db() -> Iterator[psycopg.Connection]:
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
            cur.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    finally:
        admin.close()

    conn = psycopg.connect(_dsn(TEST_DB_NAME))
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()
        admin = psycopg.connect(ADMIN_DSN, autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        finally:
            admin.close()


# --- helpers ---------------------------------------------------------------


def _segment(conn: psycopg.Connection, chat_id: int, config: Config) -> None:
    """Segment one chat from scratch and commit. The commit matters: the
    fixture connection is not autocommit, and `find_dirty_chats` compares
    `message.updated_at` (a rename's `now()`) with `segment.created_at`
    (segmentation's `now()`) — both are transaction-start times, so the
    two must be separate transactions, as they are in production."""
    run_segment_for_chat(
        conn,
        chat_id,
        config,
        FakeBoundaryProvider(),
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    conn.commit()


def _resegment_dirty(conn: psycopg.Connection, chat_id: int, config: Config) -> None:
    dirty = find_dirty_chats(conn, index_unsent=config.policy.index_unsent)
    assert chat_id in dirty, dirty
    run_segment_for_chat(
        conn,
        chat_id,
        config,
        FakeBoundaryProvider(),
        PROMPT_BYTES,
        earliest_changed_at=dirty[chat_id].earliest_changed_at,
        latest_changed_at=dirty[chat_id].latest_changed_at,
    )
    conn.commit()


def _segments(conn: psycopg.Connection, chat_id: int) -> list[tuple[int, str, str]]:
    """`(segment_id, rendered_text, rendered_sha256)` for a chat, oldest first."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT segment_id, rendered_text, rendered_sha256 FROM segment "
            "WHERE chat_id = %s ORDER BY started_at, seq_in_session",
            (chat_id,),
        )
        return [(int(s), str(t), str(h)) for s, t, h in cur.fetchall()]


def _embedding_sha(conn: psycopg.Connection, segment_id: int) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT text_sha256 FROM segment_embedding WHERE segment_id = %s", (segment_id,))
        row = cur.fetchone()
    return str(row[0]) if row else None


def _insert_handle(conn: psycopg.Connection, *, person_id: int, value: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO handle (person_id, kind, normalized_value) VALUES (%s, 'phone', %s)",
            (person_id, value),
        )


def _dirty_chat_ids(conn: psycopg.Connection, config: Config) -> set[int]:
    return set(find_dirty_chats(conn, index_unsent=config.policy.index_unsent))


def _seed_dm_with_alice(conn: psycopg.Connection) -> tuple[int, int, int]:
    """A DM between the owner and Alice with one message from each,
    committed. Returns `(chat_id, owner_id, alice_id)`."""
    owner_id = _insert_person(conn, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice_id = _insert_person(conn, display_name="Alice Example", short_name="alice")
    chat_id = _insert_chat(conn, source_guid="chat-dm-alice")
    _add_participant(conn, chat_id, owner_id)
    _add_participant(conn, chat_id, alice_id)
    _insert_message(
        conn, chat_id=chat_id, sender_person_id=owner_id, is_from_me=True, sent_at=_BASE, text="hello"
    )
    _insert_message(
        conn,
        chat_id=chat_id,
        sender_person_id=alice_id,
        is_from_me=False,
        sent_at=_BASE + timedelta(minutes=1),
        text="hi there",
    )
    conn.commit()
    return chat_id, owner_id, alice_id


# --- tests -----------------------------------------------------------------


def test_rename_after_segmentation_re_renders_and_re_embeds_the_chat(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """The whole path: segment, embed, rename a participant, see the chat
    come back dirty, re-segment, and find the new name in the rendered
    text with the old embedding superseded."""
    chat_id, _owner_id, alice_id = _seed_dm_with_alice(scratch_db)
    _segment(scratch_db, chat_id, config)

    provider = FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM)
    assert run_embed(scratch_db, provider).segments_embedded == 1
    scratch_db.commit()

    [(old_segment_id, old_text, old_sha)] = _segments(scratch_db, chat_id)
    assert "Chat: Alice Example" in old_text
    assert "] alice: hi there" in old_text
    assert _embedding_sha(scratch_db, old_segment_id) == old_sha
    assert _dirty_chat_ids(scratch_db, config) == set(), "nothing pending before the rename"

    marked = rename_person(
        scratch_db, person_id=alice_id, display_name="Alice Carter", short_name="alice-carter"
    )
    scratch_db.commit()
    assert marked == frozenset({chat_id})

    dirty = find_dirty_chats(scratch_db, index_unsent=config.policy.index_unsent)
    assert set(dirty) == {chat_id}
    assert dirty[chat_id].earliest_changed_at == _BASE, (
        "the whole chat is dirty from its first message"
    )

    _resegment_dirty(scratch_db, chat_id, config)

    [(new_segment_id, new_text, new_sha)] = _segments(scratch_db, chat_id)
    assert new_segment_id != old_segment_id
    assert "Chat: Alice Carter" in new_text
    assert "] alice-carter: hi there" in new_text
    assert "Alice Example" not in new_text
    assert "] alice:" not in new_text
    assert new_sha != old_sha
    assert new_sha == sha256_text(new_text)

    # The old segment (and, by cascade, its embedding) is gone; the new
    # segment has no embedding yet, so S6 sees exactly one pending row and
    # records the new rendered hash — the embedding is superseded, not kept.
    assert _embedding_sha(scratch_db, old_segment_id) is None
    assert _embedding_sha(scratch_db, new_segment_id) is None
    assert run_embed(scratch_db, provider).segments_embedded == 1
    scratch_db.commit()
    assert _embedding_sha(scratch_db, new_segment_id) == new_sha

    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT operation FROM search_index_event WHERE entity_kind = 'segment' "
            "AND entity_id = %s ORDER BY event_id",
            (old_segment_id,),
        )
        assert [r[0] for r in cur.fetchall()] == ["upsert", "delete"]
    assert _dirty_chat_ids(scratch_db, config) == set(), "re-segmentation settles the chat"


def test_rename_of_a_person_with_no_messages_and_no_participations_marks_nothing(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    chat_id, _owner_id, _alice_id = _seed_dm_with_alice(scratch_db)
    _segment(scratch_db, chat_id, config)
    loner_id = _insert_person(scratch_db, display_name="Carol Chen", short_name="carol")
    scratch_db.commit()

    marked = rename_person(scratch_db, person_id=loner_id, display_name="Carol Chen-Okafor")
    scratch_db.commit()

    assert marked == frozenset()
    assert _dirty_chat_ids(scratch_db, config) == set()
    with scratch_db.cursor() as cur:
        cur.execute("SELECT display_name FROM person WHERE person_id = %s", (loner_id,))
        assert cur.fetchone() == ("Carol Chen-Okafor",)


def test_merge_marks_the_absorbed_persons_chats_including_participant_only_ones(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """Bob never sent a message in chat B — he is only in its header — and
    Alice is not in chat B at all. After the merge, B's header must say
    Alice, so B is marked along with Alice's own chat A; the owner-only
    chat C stays untouched."""
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice_id = _insert_person(scratch_db, display_name="Alice Example", short_name="alice")
    bob_id = _insert_person(scratch_db, display_name="Bob Builder", short_name="bob")
    chat_a = _insert_chat(scratch_db, source_guid="chat-a")
    chat_b = _insert_chat(scratch_db, source_guid="chat-b")
    chat_c = _insert_chat(scratch_db, source_guid="chat-c")
    for chat_id in (chat_a, chat_b, chat_c):
        _add_participant(scratch_db, chat_id, owner_id)
        _insert_message(
            scratch_db,
            chat_id=chat_id,
            sender_person_id=owner_id,
            is_from_me=True,
            sent_at=_BASE,
            text="hello",
        )
    _add_participant(scratch_db, chat_a, alice_id)
    _insert_message(
        scratch_db,
        chat_id=chat_a,
        sender_person_id=alice_id,
        is_from_me=False,
        sent_at=_BASE + timedelta(minutes=1),
        text="hi from alice",
    )
    _add_participant(scratch_db, chat_b, bob_id)
    scratch_db.commit()
    for chat_id in (chat_a, chat_b, chat_c):
        _segment(scratch_db, chat_id, config)
    assert "Chat: Bob Builder" in _segments(scratch_db, chat_b)[0][1]
    assert _dirty_chat_ids(scratch_db, config) == set()

    marked = merge_persons(scratch_db, keep_person_id=alice_id, absorb_person_id=bob_id)
    scratch_db.commit()

    assert marked == frozenset({chat_a, chat_b})
    assert _dirty_chat_ids(scratch_db, config) == {chat_a, chat_b}

    _resegment_dirty(scratch_db, chat_b, config)
    [(_, text_b, _)] = _segments(scratch_db, chat_b)
    assert "Chat: Alice Example" in text_b
    assert "Bob Builder" not in text_b


def test_assign_marks_the_handles_previous_and_new_persons_chats(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    owner_id = _insert_person(scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice_id = _insert_person(scratch_db, display_name="Alice Example", short_name="alice")
    bob_id = _insert_person(scratch_db, display_name="Bob Builder", short_name="bob")
    chat_a = _insert_chat(scratch_db, source_guid="chat-a")
    chat_b = _insert_chat(scratch_db, source_guid="chat-b")
    chat_c = _insert_chat(scratch_db, source_guid="chat-c")
    for chat_id in (chat_a, chat_b, chat_c):
        _add_participant(scratch_db, chat_id, owner_id)
        _insert_message(
            scratch_db,
            chat_id=chat_id,
            sender_person_id=owner_id,
            is_from_me=True,
            sent_at=_BASE,
            text="hello",
        )
    _add_participant(scratch_db, chat_a, alice_id)
    _add_participant(scratch_db, chat_b, bob_id)
    _insert_handle(scratch_db, person_id=alice_id, value="+14155550100")
    scratch_db.commit()
    for chat_id in (chat_a, chat_b, chat_c):
        _segment(scratch_db, chat_id, config)
    assert _dirty_chat_ids(scratch_db, config) == set()

    marked = assign_handle(
        scratch_db, normalized_value="+14155550100", kind="phone", person_id=bob_id
    )
    scratch_db.commit()

    assert marked == frozenset({chat_a, chat_b})
    assert _dirty_chat_ids(scratch_db, config) == {chat_a, chat_b}
    with scratch_db.cursor() as cur:
        cur.execute("SELECT person_id FROM handle WHERE normalized_value = '+14155550100'")
        assert cur.fetchone() == (bob_id,)


def test_renaming_the_owner_is_refused_by_default_and_marks_nothing_when_allowed(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """Nothing rendered names the owner — the header lists non-owner
    participants only and the owner's lines say `owner` — so the
    allowed rename marks no chat and an in-place re-render reproduces the
    stored text byte for byte. If the renderer ever starts naming the
    owner, this is the test that fails, and
    `_mark_chats_dirty_for_persons` is the place to widen."""
    chat_id, owner_id, _alice_id = _seed_dm_with_alice(scratch_db)
    _segment(scratch_db, chat_id, config)
    [(segment_id, text_before, sha_before)] = _segments(scratch_db, chat_id)

    with pytest.raises(IdentityError, match="owner"):
        rename_person(scratch_db, person_id=owner_id, display_name="Jamie Renamed")
    scratch_db.commit()
    with scratch_db.cursor() as cur:
        cur.execute("SELECT display_name FROM person WHERE person_id = %s", (owner_id,))
        assert cur.fetchone() == ("Jamie Owner",)

    marked = rename_person(
        scratch_db,
        person_id=owner_id,
        display_name="Jamie Renamed",
        short_name="jamie",
        allow_owner=True,
    )
    scratch_db.commit()

    assert marked == frozenset()
    assert _dirty_chat_ids(scratch_db, config) == set()
    text_after, sha_after = refresh_segment_rendering(scratch_db, segment_id, config)
    scratch_db.commit()
    assert (text_after, sha_after) == (text_before, sha_before)


def test_renames_in_one_transaction_each_report_a_shared_chat(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """A batch (rematch-stubs, apply-overrides) renames many persons in
    one transaction. The message rows of a shared chat are rewritten by
    the first rename only, but every rename still reports the chat, so a
    union across the batch is the true count."""
    chat_id, _owner_id, alice_id = _seed_dm_with_alice(scratch_db)
    bob_id = _insert_person(scratch_db, display_name="Bob Builder", short_name="bob")
    _add_participant(scratch_db, chat_id, bob_id)
    scratch_db.commit()
    _segment(scratch_db, chat_id, config)

    with scratch_db.transaction():
        first = rename_person(scratch_db, person_id=alice_id, display_name="Alice Carter")
        second = rename_person(scratch_db, person_id=bob_id, display_name="Bob Mason")
    scratch_db.commit()

    assert first == second == frozenset({chat_id})
    assert _dirty_chat_ids(scratch_db, config) == {chat_id}


def test_message_updated_at_moves_for_every_message_of_a_marked_chat(
    scratch_db: psycopg.Connection, config: Config
) -> None:
    """The bump is the mechanism `find_dirty_chats` watches, and it must
    cover the owner's own messages too — the header on every segment
    changes, not just the renamed person's lines."""
    chat_id, _owner_id, alice_id = _seed_dm_with_alice(scratch_db)
    other_chat = _insert_chat(scratch_db, source_guid="chat-untouched")
    _insert_message(
        scratch_db,
        chat_id=other_chat,
        sender_person_id=_owner_id,
        is_from_me=True,
        sent_at=_BASE,
        text="elsewhere",
    )
    scratch_db.commit()

    def updated_ats() -> dict[int, Any]:
        with scratch_db.cursor() as cur:
            cur.execute("SELECT message_id, chat_id, updated_at FROM message")
            return {int(m): (int(c), u) for m, c, u in cur.fetchall()}

    before = updated_ats()
    scratch_db.commit()
    rename_person(scratch_db, person_id=alice_id, display_name="Alice Carter")
    scratch_db.commit()
    after = updated_ats()

    for message_id, (chat, stamp_before) in before.items():
        chat_after, stamp_after = after[message_id]
        assert chat_after == chat
        if chat == chat_id:
            assert stamp_after > stamp_before, message_id
        else:
            assert stamp_after == stamp_before, message_id
