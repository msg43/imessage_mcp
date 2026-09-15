"""Unit tests (fail-loud preconditions, result summary) + a live-Postgres
integration suite for `imsg.stages.identity_rematch` — naming the review
stubs a Contacts-less import left behind. Every persona, number and
address here is fictional.

The DB-gated tests borrow `test_identity.py`'s scratch-instance settings
and its S2->S3 seeding helper, run in their own database, and skip (from
the `pg_conn` fixture, so the unit tests above it still run) when no
scratch Postgres is reachable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest

from conftest import ConfigDictFactory
from imsg.db.migrations import PostgresMigrationRunner
from imsg.errors import IdentityError
from imsg.stages.identity import (
    UNNAMED_CONTACT_DISPLAY_NAME,
    ContactRecord,
    ContactsAccessDeniedError,
    merge_persons,
    rename_person,
    run_identity,
)
from imsg.stages.identity_rematch import RematchStubsResult, StubOutcome, run_rematch_stubs
from test_identity import (
    ADMIN_DSN,
    REACHABLE,
    REAL_MIGRATIONS_DIR,
    _dsn,
    _identity_config,
    _seed_extraction,
)

ALICE_PHONE = "+14155552671"
ALICE_EMAIL = "alice@example.com"
BOB_PHONE = "+14155552672"
BOB_EMAIL = "bob@example.com"
CAROL_PHONE = "+14155552673"
CAROL_EMAIL = "carol@example.com"
DANA_PHONE = "+14155552674"
DANA_EMAIL = "dana@example.com"
OWNER_PHONE = "+14155552600"
SHORT_CODE = "24273"


def _phone(value: str) -> tuple[str, str]:
    return (value, "phone")


def _email(value: str) -> tuple[str, str]:
    return (value, "email")


def _card(
    identifier: str, name: str, *identifiers: tuple[str, str], organization: str | None = None
) -> ContactRecord:
    return ContactRecord(
        identifier=identifier,
        display_name=name,
        organization=organization,
        normalized_identifiers=tuple(identifiers),
    )


# --------------------------------------------------------------------------
# unit tests: no DB
# --------------------------------------------------------------------------


class _NoDatabase:
    """A connection stand-in that fails the test on any use: the fail-loud
    preconditions must be checked before the database is touched at all."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the database must not be touched before Contacts is confirmed (conn.{name})")


def _no_database() -> psycopg.Connection:
    return cast(psycopg.Connection, _NoDatabase())


def test_rematch_refuses_to_run_with_contacts_import_disabled(
    config_dict_factory: ConfigDictFactory,
) -> None:
    config = _identity_config(config_dict_factory, contacts_import=False)

    def must_not_run(region: str) -> list[ContactRecord]:
        raise AssertionError("the importer must not be called when Contacts is disabled")

    with pytest.raises(IdentityError, match="contacts_import is false"):
        run_rematch_stubs(conn=_no_database(), config=config, contacts_importer=must_not_run)


def test_rematch_raises_when_contacts_access_is_denied(
    config_dict_factory: ConfigDictFactory,
) -> None:
    """The import catches this and degrades; the rematch must not — a run
    without Contacts has nothing to match against, and `matched=0` would
    read as "looked, found nothing"."""
    config = _identity_config(config_dict_factory)

    def denying(region: str) -> list[ContactRecord]:
        raise ContactsAccessDeniedError("simulated: Contacts access is not authorized")

    with pytest.raises(ContactsAccessDeniedError, match="not authorized") as excinfo:
        run_rematch_stubs(conn=_no_database(), config=config, contacts_importer=denying)
    assert "nothing was changed" in str(excinfo.value)


def test_rematch_raises_when_contacts_returns_no_cards(
    config_dict_factory: ConfigDictFactory,
) -> None:
    config = _identity_config(config_dict_factory)
    with pytest.raises(IdentityError, match="returned 0 cards"):
        run_rematch_stubs(conn=_no_database(), config=config, contacts_importer=lambda region: [])


def test_result_counts_and_summary_line() -> None:
    def outcome(person_id: int, status: Any) -> StubOutcome:
        return StubOutcome(person_id, ALICE_PHONE, (_phone(ALICE_PHONE),), status)

    result = RematchStubsResult(
        outcomes=(
            outcome(1, "matched"),
            outcome(2, "matched"),
            outcome(3, "ambiguous"),
            outcome(4, "unmatched"),
        ),
        contacts_loaded=9,
    )
    assert result.summary == "stubs=4 matched=2 ambiguous=1 unmatched=1"
    assert result.contacts_loaded == 9
    assert result.dry_run is False
    assert RematchStubsResult(outcomes=(), contacts_loaded=1).summary == (
        "stubs=0 matched=0 ambiguous=0 unmatched=0"
    )


# --------------------------------------------------------------------------
# live-Postgres integration suite
# --------------------------------------------------------------------------

TEST_DB_NAME = "imsg_index_identity_rematch_test"


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
    if not REACHABLE:
        pytest.skip("no reachable scratch Postgres instance (see test_identity.py)")
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


def _seed_and_resolve(
    conn: psycopg.Connection,
    tmp_path: Path,
    config_dict_factory: ConfigDictFactory,
    *,
    handles: list[str],
) -> None:
    """S2 then S3 with Contacts off: every handle becomes its own review
    stub named after its normalized value — exactly what an import run
    from a process without the Contacts grant leaves behind."""
    _seed_extraction(
        conn, tmp_path, participants=[(str(i), raw) for i, raw in enumerate(handles, start=1)]
    )
    run_identity(conn=conn, config=_identity_config(config_dict_factory, contacts_import=False))
    # The fixture connection is not autocommit, so a bare read anywhere above
    # leaves an implicit transaction open and every later `conn.transaction()`
    # nests inside it as a savepoint — `now()` would never advance, and the
    # `updated_at` assertions below could not tell a bump from no bump. Commit
    # here, as production (autocommit) does between commands.
    conn.commit()


def _person_of(conn: psycopg.Connection, value: str) -> tuple[Any, ...]:
    """`(person_id, display_name, short_name, organization, notes, needs_review, updated_at)`
    of the person holding the handle `value`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.person_id, p.display_name, p.short_name, p.organization, p.notes, "
            "p.needs_review, p.updated_at FROM handle h "
            "JOIN person p ON p.person_id = h.person_id WHERE h.normalized_value = %s",
            (value,),
        )
        row = cur.fetchone()
    assert row is not None, value
    return tuple(row)


def _state_snapshot(conn: psycopg.Connection) -> tuple[list[Any], list[Any], list[Any]]:
    """Everything a rematch may touch, `updated_at` included: a run that
    changes nothing must leave this byte-identical."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT person_id, display_name, short_name, organization, notes, needs_review, "
            "is_owner, updated_at FROM person ORDER BY person_id"
        )
        persons = cur.fetchall()
        cur.execute(
            "SELECT handle_id, person_id, normalized_value, kind FROM handle ORDER BY handle_id"
        )
        handles = cur.fetchall()
        cur.execute("SELECT message_id, sender_person_id FROM message ORDER BY message_id")
        messages = cur.fetchall()
    return persons, handles, messages


def _persons_except(snapshot: tuple[list[Any], list[Any], list[Any]], *person_ids: int) -> list[Any]:
    return [row for row in snapshot[0] if row[0] not in person_ids]


def test_rematch_names_stubs_from_a_unique_card_and_leaves_the_rest_alone(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory,
        handles=[ALICE_PHONE, BOB_PHONE, CAROL_PHONE, SHORT_CODE],
    )
    before = _state_snapshot(pg_conn)
    alice_before = _person_of(pg_conn, ALICE_PHONE)
    assert alice_before[1] == ALICE_PHONE and alice_before[5] is True  # a raw stub

    cards = [
        _card("c-alice", "Alice Example", _phone(ALICE_PHONE), organization="Acme Construction"),
        # A shared household number saved under two people who share no name
        # tokens: a genuine conflict, never guessed.
        _card("c-carol", "Carol Example", _phone(CAROL_PHONE)),
        _card("c-dave", "Dave Okafor", _phone(CAROL_PHONE)),
    ]
    config = _identity_config(config_dict_factory)
    result = run_rematch_stubs(conn=pg_conn, config=config, contacts_importer=lambda region: cards)

    assert result.summary == "stubs=4 matched=1 ambiguous=1 unmatched=2"
    assert result.contacts_loaded == 3
    assert result.dry_run is False
    by_status = {o.status: o for o in result.outcomes}
    assert by_status["matched"].person_id == alice_before[0]
    assert by_status["matched"].display_name == ALICE_PHONE
    assert by_status["matched"].handles == (_phone(ALICE_PHONE),)
    assert by_status["matched"].new_display_name == "Alice Example"
    assert by_status["matched"].new_short_name == "alice-example"
    assert by_status["ambiguous"].display_name == CAROL_PHONE
    assert {o.display_name for o in result.outcomes if o.status == "unmatched"} == {
        BOB_PHONE, SHORT_CODE,
    }

    # Named exactly as the import names a fresh handle from a unique card ...
    alice_after = _person_of(pg_conn, ALICE_PHONE)
    assert alice_after[0] == alice_before[0]
    assert alice_after[1:6] == ("Alice Example", "alice-example", "Acme Construction", None, False)
    # ... and through the rename path, so updated_at moved (re-segmentation keys on it).
    assert alice_after[6] > alice_before[6]

    # Nobody else moved: not the unmatched, not the ambiguous, not the owner.
    after = _state_snapshot(pg_conn)
    assert _persons_except(after, alice_before[0]) == _persons_except(before, alice_before[0])
    assert after[1] == before[1]  # handles
    assert after[2] == before[2]  # message attributions

    # A second run finds one stub fewer and changes nothing.
    again = run_rematch_stubs(conn=pg_conn, config=config, contacts_importer=lambda region: cards)
    assert again.summary == "stubs=3 matched=0 ambiguous=1 unmatched=2"
    assert _state_snapshot(pg_conn) == after


def test_rematch_never_touches_reviewed_persons_or_the_owner(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE])
    alice_id = int(_person_of(pg_conn, ALICE_PHONE)[0])
    bob_id = int(_person_of(pg_conn, BOB_PHONE)[0])
    # Hand-named and reviewed: curated, off limits even though Contacts disagrees.
    rename_person(pg_conn, person_id=alice_id, display_name="Alice Hand-Named")
    # Hand-named but deliberately left on the worklist: no longer raw-named, so not a stub.
    rename_person(pg_conn, person_id=bob_id, display_name="Bob Hand-Named", mark_reviewed=False)
    # The owner, in the most stub-looking state possible: a handle of their own,
    # named after it, flagged for review. `is_owner` alone must keep them out.
    with pg_conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE is_owner")
        row = cur.fetchone()
        assert row is not None
        owner_id = int(row[0])
        cur.execute(
            "INSERT INTO handle (person_id, kind, normalized_value) VALUES (%s, 'phone', %s)",
            (owner_id, OWNER_PHONE),
        )
        cur.execute(
            "UPDATE person SET display_name = %s, needs_review = true WHERE person_id = %s",
            (OWNER_PHONE, owner_id),
        )
    pg_conn.commit()
    before = _state_snapshot(pg_conn)

    cards = [
        _card("c-alice", "Alice Example", _phone(ALICE_PHONE)),
        _card("c-bob", "Bob Example", _phone(BOB_PHONE)),
        _card("c-owner", "Owner Example", _phone(OWNER_PHONE)),
    ]
    result = run_rematch_stubs(
        conn=pg_conn, config=_identity_config(config_dict_factory), contacts_importer=lambda region: cards
    )
    assert result.summary == "stubs=0 matched=0 ambiguous=0 unmatched=0"
    assert _state_snapshot(pg_conn) == before


def test_rematch_names_a_multi_handle_stub_only_when_every_handle_agrees(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """A stub holding several handles was merged by hand; Contacts has to
    agree with the human who merged it, under the same rules one handle's
    cards are held to."""
    pairs = [
        (ALICE_PHONE, ALICE_EMAIL),
        (BOB_PHONE, BOB_EMAIL),
        (CAROL_PHONE, CAROL_EMAIL),
        (DANA_PHONE, DANA_EMAIL),
    ]
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory, handles=[h for pair in pairs for h in pair]
    )
    for phone, email in pairs:
        merge_persons(
            pg_conn,
            keep_person_id=int(_person_of(pg_conn, phone)[0]),
            absorb_person_id=int(_person_of(pg_conn, email)[0]),
        )
    for phone, email in pairs:  # still raw stubs, now with two handles each
        assert _person_of(pg_conn, phone)[0] == _person_of(pg_conn, email)[0]
        assert _person_of(pg_conn, phone)[1] == phone
    before = _state_snapshot(pg_conn)

    cards = [
        # one card carrying both handles: agreement
        _card("c-alice", "Alice Example", _phone(ALICE_PHONE), _email(ALICE_EMAIL)),
        # two cards that name different people: ambiguous
        _card("c-bob-1", "Bob Example", _phone(BOB_PHONE)),
        _card("c-bob-2", "Robert Feldman", _email(BOB_EMAIL)),
        # a card for the phone only, nothing for the email: unmatched
        _card("c-carol", "Carol Example", _phone(CAROL_PHONE)),
        # the same person at two levels of completeness: agreement, fullest name wins
        _card("c-dana-1", "Dana", _phone(DANA_PHONE)),
        _card("c-dana-2", "Dana Example", _email(DANA_EMAIL)),
    ]
    result = run_rematch_stubs(
        conn=pg_conn, config=_identity_config(config_dict_factory), contacts_importer=lambda region: cards
    )
    assert result.summary == "stubs=4 matched=2 ambiguous=1 unmatched=1"
    assert {o.display_name: o.status for o in result.outcomes} == {
        ALICE_PHONE: "matched", BOB_PHONE: "ambiguous", CAROL_PHONE: "unmatched", DANA_PHONE: "matched",
    }
    matched = {o.display_name: o for o in result.outcomes if o.status == "matched"}
    assert matched[ALICE_PHONE].handles == (_phone(ALICE_PHONE), _email(ALICE_EMAIL))

    alice = _person_of(pg_conn, ALICE_PHONE)
    assert alice[1:6] == ("Alice Example", "alice-example", None, None, False)
    assert _person_of(pg_conn, ALICE_EMAIL)[0] == alice[0]  # both handles still on the one person
    dana = _person_of(pg_conn, DANA_PHONE)
    assert dana[1:3] == ("Dana Example", "dana-example")
    after = _state_snapshot(pg_conn)
    assert _persons_except(after, int(alice[0]), int(dana[0])) == _persons_except(
        before, int(alice[0]), int(dana[0])
    )
    assert after[1] == before[1] and after[2] == before[2]


def test_rematch_dry_run_reports_the_same_counts_and_writes_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE])
    before = _state_snapshot(pg_conn)
    cards = [_card("c-alice", "Alice Example", _phone(ALICE_PHONE))]
    config = _identity_config(config_dict_factory)

    preview = run_rematch_stubs(
        conn=pg_conn, config=config, contacts_importer=lambda region: cards, dry_run=True
    )
    assert preview.dry_run is True
    assert preview.summary == "stubs=2 matched=1 ambiguous=0 unmatched=1"
    assert _state_snapshot(pg_conn) == before

    real = run_rematch_stubs(conn=pg_conn, config=config, contacts_importer=lambda region: cards)
    assert real.dry_run is False
    assert real.summary == preview.summary
    assert _person_of(pg_conn, ALICE_PHONE)[1] == "Alice Example"


def test_rematch_keeps_a_stub_whose_only_card_has_no_name(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """A card with neither a name nor an organization matches uniquely, but
    naming the stub after the placeholder would clear needs_review while
    naming nobody. It stays a stub, counted as unmatched."""
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE])
    before = _state_snapshot(pg_conn)
    cards = [_card("c-nameless", UNNAMED_CONTACT_DISPLAY_NAME, _phone(ALICE_PHONE))]

    result = run_rematch_stubs(
        conn=pg_conn, config=_identity_config(config_dict_factory), contacts_importer=lambda region: cards
    )
    assert result.summary == "stubs=1 matched=0 ambiguous=0 unmatched=1"
    assert _state_snapshot(pg_conn) == before


def _message_updated_ats(conn: psycopg.Connection) -> dict[int, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT message_id, updated_at FROM message ORDER BY message_id")
        return {int(message_id): updated_at for message_id, updated_at in cur.fetchall()}


def test_rematch_marks_the_renamed_stubs_chats_for_resegmentation_once_each(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """Both stubs share the one seeded chat, so two renames mark one chat:
    the count is a union, not a sum. The mark is a bump of every message's
    `updated_at` in that chat — the owner's own message included, since the
    header on every segment changes — and a dry run rolls the bump back
    along with the renames while still reporting the count."""
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE])
    before = _message_updated_ats(pg_conn)
    assert len(before) == 3  # one message per stub plus the owner's
    cards = [
        _card("c-alice", "Alice Example", _phone(ALICE_PHONE)),
        _card("c-bob", "Bob Builder", _phone(BOB_PHONE)),
    ]
    config = _identity_config(config_dict_factory)

    preview = run_rematch_stubs(
        conn=pg_conn, config=config, contacts_importer=lambda region: cards, dry_run=True
    )
    assert preview.summary == "stubs=2 matched=2 ambiguous=0 unmatched=0"
    assert preview.chats_marked_dirty == 1
    assert _message_updated_ats(pg_conn) == before

    real = run_rematch_stubs(conn=pg_conn, config=config, contacts_importer=lambda region: cards)
    assert real.summary == preview.summary
    assert real.chats_marked_dirty == 1
    after = _message_updated_ats(pg_conn)
    assert set(after) == set(before)
    assert all(after[message_id] > before[message_id] for message_id in before)


def test_rematch_that_names_nobody_marks_no_chat(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE])
    before = _message_updated_ats(pg_conn)

    result = run_rematch_stubs(
        conn=pg_conn,
        config=_identity_config(config_dict_factory),
        contacts_importer=lambda region: [_card("c-dana", "Dana Okafor", _phone(DANA_PHONE))],
    )

    assert result.summary == "stubs=1 matched=0 ambiguous=0 unmatched=1"
    assert result.chats_marked_dirty == 0
    assert _message_updated_ats(pg_conn) == before
