"""The export gate's deny matrix (SPEC §11.2, hard requirement 5).

Every test here is a way private content could have escaped; each must
end in DENY. Fictional personas only (D5).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import psycopg
import pytest

from _export_fixtures import (
    add_participant,
    add_raw_participant,
    admin_reachable,
    allow,
    create_scratch_db,
    drop_scratch_db,
    insert_attachment,
    insert_chat,
    insert_message,
    insert_person,
    insert_segment,
    insert_tapback,
    link_attachment,
)
from imsg.export.eligibility import (
    compute_attachment_eligibility,
    compute_chat_eligibility,
    eligible_chat_ids,
)

TEST_DB_NAME = "imsg_index_export_elig_test"

pytestmark = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)

_T = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(TEST_DB_NAME)


def _owner(db: psycopg.Connection, *, allowlisted: bool = True) -> int:
    owner_id = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    if allowlisted:
        allow(db, owner_id)
    return owner_id


def _alice(db: psycopg.Connection, *, allowlisted: bool = True) -> int:
    alice_id = insert_person(db, display_name="Alice Example", short_name="alice")
    if allowlisted:
        allow(db, alice_id)
    return alice_id


def test_empty_database_yields_no_eligible_chats(db: psycopg.Connection) -> None:
    assert eligible_chat_ids(db) == set()


def test_fully_allowlisted_dm_is_eligible(db: psycopg.Connection) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    insert_message(db, chat_id=chat_id, sender_person_id=alice_id, is_from_me=False, sent_at=_T, text="hello")
    db.commit()
    assert chat_id in eligible_chat_ids(db)


def test_participant_absent_from_allowlist_denies(db: psycopg.Connection) -> None:
    """Default deny: no allowlist row at all."""
    owner_id = _owner(db)
    alice_id = _alice(db, allowlisted=False)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_participant_with_text_allowed_false_denies(db: psycopg.Connection) -> None:
    """A row that exists but says no is still no."""
    owner_id = _owner(db)
    alice_id = _alice(db, allowlisted=False)
    allow(db, alice_id, text=False, attachments=True)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_owner_not_allowlisted_denies_even_a_business_dm(db: psycopg.Connection) -> None:
    """`is_owner` is not an implicit bypass (D6)."""
    owner_id = _owner(db, allowlisted=False)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_group_with_one_outsider_is_wholly_excluded(db: psycopg.Connection) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)
    bob_id = insert_person(db, display_name="Bob Builder", short_name="bob")
    allow(db, bob_id)
    carol_id = insert_person(db, display_name="Carol Carpenter", short_name="carol")
    # carol: no allowlist row
    chat_id = insert_chat(db, source_guid="chat-g", kind="group", display_name="Acme deck crew")
    for pid in (owner_id, alice_id, bob_id, carol_id):
        add_participant(db, chat_id, pid)
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_chat_with_zero_participants_denies_not_vacuously_passes(
    db: psycopg.Connection,
) -> None:
    """The classic widening bug: 'ALL participants allowlisted' is
    vacuously true over an empty set. The gate must require a non-empty
    participant list."""
    chat_id = insert_chat(db, source_guid="chat-empty")
    db.commit()
    verdicts = compute_chat_eligibility(db)
    assert chat_id in verdicts
    assert verdicts[chat_id].participant_count == 0
    assert not verdicts[chat_id].eligible
    assert chat_id not in eligible_chat_ids(db)


def test_unresolved_raw_participant_denies(db: psycopg.Connection) -> None:
    """A `chat_participant_source` row with no resolution is an unknown
    person in the room — a NULL join must not silently pass."""
    owner_id = _owner(db)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    add_raw_participant(db, chat_id, "raw-handle-1")  # unresolved
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_resolved_source_person_missing_from_participants_still_gated(
    db: psycopg.Connection,
) -> None:
    """S3 resolved the handle to a person but never backfilled
    `chat_participant` — the person must still be allowlisted."""
    owner_id = _owner(db)
    alice_id = _alice(db)
    dana_id = insert_person(db, display_name="Dana Driver", short_name="dana")
    # dana is NOT allowlisted and NOT in chat_participant
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    add_raw_participant(db, chat_id, "raw-handle-2", resolve_to_person=dana_id)
    db.commit()
    assert chat_id not in eligible_chat_ids(db)

    allow(db, dana_id)
    db.commit()
    assert chat_id in eligible_chat_ids(db)


def test_incoming_message_with_null_sender_denies(db: psycopg.Connection) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    insert_message(db, chat_id=chat_id, sender_person_id=None, is_from_me=False, sent_at=_T, text="who am I?")
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_former_member_sender_not_allowlisted_denies(db: psycopg.Connection) -> None:
    """Someone who left the group is absent from chat_participant but
    their messages remain — they are still an outsider in the thread."""
    owner_id = _owner(db)
    alice_id = _alice(db)
    eve_id = insert_person(db, display_name="Evan Estimator", short_name="evan")
    # evan: not a participant, not allowlisted, but has a message
    chat_id = insert_chat(db, source_guid="chat-1", kind="group")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    insert_message(db, chat_id=chat_id, sender_person_id=eve_id, is_from_me=False, sent_at=_T, text="old msg")
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_unsent_message_from_outsider_still_denies_the_chat(db: psycopg.Connection) -> None:
    """The unsent message itself never exports (D1), but its sender is
    still evidence of an outsider in the thread."""
    owner_id = _owner(db)
    alice_id = _alice(db)
    eve_id = insert_person(db, display_name="Evan Estimator", short_name="evan")
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    insert_message(
        db, chat_id=chat_id, sender_person_id=eve_id, is_from_me=False,
        sent_at=_T, text="retracted", is_unsent=True,
    )
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_from_me_message_with_null_sender_uses_owner_gate(db: psycopg.Connection) -> None:
    """is_from_me rows may carry NULL sender_person_id (S3 only
    promises non-owner resolution). The effective sender is the owner —
    who must be explicitly allowlisted for the chat to pass."""
    owner_id = _owner(db)  # allowlisted
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    insert_message(db, chat_id=chat_id, sender_person_id=None, is_from_me=True, sent_at=_T, text="mine")
    db.commit()
    assert chat_id in eligible_chat_ids(db)


def test_from_me_message_with_owner_not_allowlisted_denies(db: psycopg.Connection) -> None:
    owner_id = _owner(db, allowlisted=False)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    # owner not even a participant: the content gate must still fire
    add_participant(db, chat_id, alice_id)
    insert_message(db, chat_id=chat_id, sender_person_id=None, is_from_me=True, sent_at=_T, text="mine")
    db.commit()
    assert owner_id  # (used)
    assert chat_id not in eligible_chat_ids(db)


def test_no_owner_person_at_all_denies_chats_with_from_me_content(
    db: psycopg.Connection,
) -> None:
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, alice_id)
    insert_message(db, chat_id=chat_id, sender_person_id=None, is_from_me=True, sent_at=_T, text="mine")
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_tapback_from_unresolved_or_outsider_sender_denies(db: psycopg.Connection) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    msg = insert_message(db, chat_id=chat_id, sender_person_id=alice_id, is_from_me=False, sent_at=_T, text="hi")
    insert_tapback(db, target_message_id=msg, sender_person_id=None, is_from_me=False)
    db.commit()
    assert chat_id not in eligible_chat_ids(db)


def test_tapback_from_allowlisted_participant_is_fine(db: psycopg.Connection) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-1")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    msg = insert_message(db, chat_id=chat_id, sender_person_id=alice_id, is_from_me=False, sent_at=_T, text="hi")
    insert_tapback(db, target_message_id=msg, sender_person_id=owner_id, is_from_me=True)
    db.commit()
    assert chat_id in eligible_chat_ids(db)


# --- the separate attachment gate ------------------------------------------


def _dm_with_segment(
    db: psycopg.Connection, sender_id: int, owner_id: int
) -> tuple[int, int, int]:
    chat_id = insert_chat(db, source_guid="chat-att")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, sender_id)
    msg = insert_message(db, chat_id=chat_id, sender_person_id=sender_id, is_from_me=False, sent_at=_T, text="see attached")
    segment_id = insert_segment(db, chat_id=chat_id, started_at=_T, ended_at=_T, message_ids=[msg])
    return chat_id, msg, segment_id


def test_attachment_from_text_only_person_is_denied_separately(
    db: psycopg.Connection,
) -> None:
    """A text-allowlisted business contact still sends personal photos:
    `text_allowed` alone must NOT admit attachment content."""
    owner_id = _owner(db)
    alice_id = insert_person(db, display_name="Alice Example", short_name="alice")
    allow(db, alice_id, text=True, attachments=False)
    _chat_id, msg, segment_id = _dm_with_segment(db, alice_id, owner_id)
    att = insert_attachment(db, filename="weekend-photo.jpg", mime_type="image/jpeg")
    link_attachment(db, msg, att)
    db.commit()

    elig = compute_attachment_eligibility(db, [segment_id])
    assert elig == {(segment_id, att): False}


def test_attachment_from_fully_allowed_person_is_eligible(db: psycopg.Connection) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)  # text + attachments
    _chat_id, msg, segment_id = _dm_with_segment(db, alice_id, owner_id)
    att = insert_attachment(db, filename="bid-rev3.pdf", mime_type="application/pdf")
    link_attachment(db, msg, att)
    db.commit()

    elig = compute_attachment_eligibility(db, [segment_id])
    assert elig == {(segment_id, att): True}


def test_attachment_linked_by_two_messages_needs_every_link_allowed(
    db: psycopg.Connection,
) -> None:
    """SPEC §11.2: 'via every message link through which it enters the
    document' — one disallowed link poisons the attachment."""
    owner_id = _owner(db)
    alice_id = _alice(db)  # full
    bob_id = insert_person(db, display_name="Bob Builder", short_name="bob")
    allow(db, bob_id, text=True, attachments=False)
    chat_id = insert_chat(db, source_guid="chat-2l", kind="group")
    for pid in (owner_id, alice_id, bob_id):
        add_participant(db, chat_id, pid)
    m1 = insert_message(db, chat_id=chat_id, sender_person_id=alice_id, is_from_me=False, sent_at=_T, text="photo")
    m2 = insert_message(db, chat_id=chat_id, sender_person_id=bob_id, is_from_me=False, sent_at=_T, text="same photo")
    segment_id = insert_segment(db, chat_id=chat_id, started_at=_T, ended_at=_T, message_ids=[m1, m2])
    att = insert_attachment(db, filename="site.jpg", mime_type="image/jpeg")
    link_attachment(db, m1, att)
    link_attachment(db, m2, att)
    db.commit()

    elig = compute_attachment_eligibility(db, [segment_id])
    assert elig == {(segment_id, att): False}


def test_attachment_linked_only_via_unsent_message_never_appears(
    db: psycopg.Connection,
) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-u")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    kept = insert_message(db, chat_id=chat_id, sender_person_id=alice_id, is_from_me=False, sent_at=_T, text="kept")
    unsent = insert_message(
        db, chat_id=chat_id, sender_person_id=alice_id, is_from_me=False,
        sent_at=_T, text="oops", is_unsent=True,
    )
    segment_id = insert_segment(db, chat_id=chat_id, started_at=_T, ended_at=_T, message_ids=[kept, unsent])
    att = insert_attachment(db, filename="oops.jpg", mime_type="image/jpeg")
    link_attachment(db, unsent, att)
    db.commit()

    # The pair is absent from the map entirely — callers treat absence
    # as denied, so the attachment cannot enter any document.
    elig = compute_attachment_eligibility(db, [segment_id])
    assert (segment_id, att) not in elig


def test_attachment_on_null_sender_message_denies(db: psycopg.Connection) -> None:
    owner_id = _owner(db)
    alice_id = _alice(db)
    chat_id = insert_chat(db, source_guid="chat-n")
    add_participant(db, chat_id, owner_id)
    add_participant(db, chat_id, alice_id)
    msg = insert_message(db, chat_id=chat_id, sender_person_id=None, is_from_me=False, sent_at=_T, text="?")
    segment_id = insert_segment(db, chat_id=chat_id, started_at=_T, ended_at=_T, message_ids=[msg])
    att = insert_attachment(db, filename="x.jpg", mime_type="image/jpeg")
    link_attachment(db, msg, att)
    db.commit()

    elig = compute_attachment_eligibility(db, [segment_id])
    assert elig == {(segment_id, att): False}
