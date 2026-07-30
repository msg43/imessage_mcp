"""Unit tests (handle normalization, `ContactsIndex`) + a live-Postgres
integration suite for S3 identity resolution (SPEC §8 S3). Follows the
same skipif pattern as `test_extract.py`/`test_migrations_integration.py`.

One test (`test_default_contacts_importer_degrades_loudly_when_unauthorized`)
calls the *real* `_default_contacts_importer` against the real `Contacts`
framework — this environment genuinely has no Contacts grant
(`CNAuthorizationStatusNotDetermined`, verified live while building this
module), so this is a real exercise of the loud-degrade path, not a mock.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
from conftest import ConfigDictFactory
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.errors import IdentityError
from imsg.stages.extract import run_extract
from imsg.stages.identity import (
    ContactRecord,
    ContactsAccessDeniedError,
    ContactsImportOutcome,
    ContactsIndex,
    IdentityResult,
    InvariantReport,
    _default_contacts_importer,
    assert_invariant_or_raise,
    assign_handle,
    compute_invariant_report,
    merge_persons,
    normalize_handle,
    rename_person,
    run_identity,
)
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun

# --------------------------------------------------------------------------
# unit tests: no DB, no Contacts framework
# --------------------------------------------------------------------------


def test_normalize_handle_valid_phone() -> None:
    normalized, kind = normalize_handle("(415) 555-2671", "US")
    assert kind == "phone"
    assert normalized == "+14155552671"


def test_normalize_handle_already_e164() -> None:
    normalized, kind = normalize_handle("+14155552671", "US")
    assert kind == "phone"
    assert normalized == "+14155552671"


def test_normalize_handle_email_is_lowercased() -> None:
    normalized, kind = normalize_handle("Alice.Example@ICLOUD.com", "US")
    assert kind == "email"
    assert normalized == "alice.example@icloud.com"


def test_normalize_handle_unparseable_falls_back_to_unknown() -> None:
    normalized, kind = normalize_handle("not-a-real-handle", "US")
    assert kind == "unknown"
    assert normalized == "not-a-real-handle"


def test_contacts_index_unique_match() -> None:
    alice = ContactRecord(
        identifier="c1",
        display_name="Alice Example",
        organization=None,
        normalized_identifiers=(("+14155552671", "phone"), ("alice@example.com", "email")),
    )
    index = ContactsIndex([alice])
    assert index.find_unique("+14155552671", "phone") is alice
    assert index.find_unique("alice@example.com", "email") is alice
    assert index.find_unique("+19995550000", "phone") is None


def test_contacts_index_multiple_matches_returns_none() -> None:
    shared = ("+14155552671", "phone")
    alice = ContactRecord(identifier="c1", display_name="Alice", organization=None, normalized_identifiers=(shared,))
    bob = ContactRecord(identifier="c2", display_name="Bob", organization=None, normalized_identifiers=(shared,))
    index = ContactsIndex([alice, bob])
    assert index.find_unique(*shared) is None


def test_default_contacts_importer_degrades_loudly_when_unauthorized() -> None:
    """Real exercise of the framework: this environment's Contacts
    authorization status is genuinely not `Authorized` (confirmed live
    while building this module), so this call must raise, not silently
    return an empty list."""
    with pytest.raises(ContactsAccessDeniedError):
        _default_contacts_importer("US")


# --------------------------------------------------------------------------
# live-Postgres integration suite
# --------------------------------------------------------------------------

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_identity_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def _dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


ADMIN_DSN = _dsn("postgres")


def _admin_reachable() -> bool:
    try:
        conn = psycopg.connect(ADMIN_DSN, connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


REACHABLE = _admin_reachable()

pytestmark = pytest.mark.skipif(
    not REACHABLE,
    reason=(
        "no reachable scratch Postgres instance "
        f"(tried {TEST_PG_HOST}:{TEST_PG_PORT}) — set IMSG_TEST_PG_HOST/"
        "IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
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
    runner = PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR)
    runner.apply_pending()
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


def _scalar_int(cur: psycopg.Cursor[Any]) -> int:
    """`int(cur.fetchone()[0])`, but mypy-clean about `fetchone()`
    returning `tuple | None` — every call site here is right after a
    query that is known (by test setup) to return exactly one row."""
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _identity_config(
    config_dict_factory: ConfigDictFactory, *, contacts_import: bool = True, default_region: str = "US"
) -> Config:
    """Builds a fully-valid `Config` via the shared `config_dict_factory`
    fixture (conftest.py) — reusing it rather than hand-rolling a config
    dict, since the schema's path-containment validators require the
    `messages_dir`/`data_root` redirection that fixture already sets up."""
    raw = config_dict_factory(**{
        "identity.default_region": default_region,
        "identity.contacts_import": contacts_import,
    })
    return load_config_dict(raw, source="<test>")


def _seed_extraction(
    conn: psycopg.Connection,
    tmp_path: Path,
    *,
    chat_style: int = 45,
    participants: list[tuple[str, str]] | None = None,  # [(guid_suffix, raw_handle), ...]
) -> None:
    """Runs S2 against a small synthetic snapshot so S3 has real
    `source_handle`/`message`/`chat_participant_source` rows to resolve —
    exercising the real S2->S3 handoff rather than hand-inserting rows
    that skip S2's own invariants."""
    participants = participants or [("1", "+14155552671")]
    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1", style=chat_style))
    for suffix, raw in participants:
        handle = builder.add_handle(FixtureHandle(raw_value=raw))
        builder.link_participant(chat.guid, handle.raw_value)
        builder.add_message(
            FixtureMessage(guid=f"msg-in-{suffix}", chat_guid=chat.guid, handle_raw_value=raw)
        )
    builder.add_message(FixtureMessage(guid="msg-out-1", chat_guid=chat.guid, is_from_me=True))
    snapshot_path = builder.build(tmp_path / "snapshot.db")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(
            messages=tuple(
                ImsgDumpMessage(
                    rowid=i, guid=g, chat_guid=None, handle=None, is_from_me=False, date=None,
                    date_edited=None, date_retracted=None, service="iMessage", body_text="hi",
                    edit_history=(), is_unsent=False, tapback=None, attachment_rowids=(),
                    reply_to_guid=None,
                )
                for i, g in enumerate([f"msg-in-{s}" for s, _ in participants] + ["msg-out-1"], start=1)
            ),
            stderr_lines=(),
        )

    binary = tmp_path / "imsg-dump"
    binary.write_text("")
    run_extract(
        conn=conn, source_name="mini", snapshot_path=snapshot_path,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run,
    )


def test_run_identity_resolves_incoming_message_to_stub_person(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)

    result = run_identity(conn=pg_conn, config=config)
    assert isinstance(result, IdentityResult)
    assert result.contacts == ContactsImportOutcome(
        attempted=False, contacts_loaded=0, degraded=False, degraded_reason=None
    )
    assert result.invariant.ok is True

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT p.needs_review, p.display_name FROM message m "
            "JOIN person p ON p.person_id = m.sender_person_id WHERE m.source_guid = 'msg-in-1'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] is True  # auto-created stub, needs_review stays at its default
        assert row[1] == "+14155552671"


def test_run_identity_resolves_is_from_me_to_singleton_owner(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)
    run_identity(conn=pg_conn, config=config)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM person WHERE is_owner")
        assert cur.fetchone() == (1,)

        cur.execute(
            "SELECT p.is_owner FROM message m JOIN person p ON p.person_id = m.sender_person_id "
            "WHERE m.source_guid = 'msg-out-1'"
        )
        assert cur.fetchone() == (True,)

        # Owner is inserted as a participant in every chat they sent into.
        cur.execute(
            """
            SELECT count(*) FROM chat_participant cp
            JOIN person p ON p.person_id = cp.person_id
            JOIN chat c ON c.chat_id = cp.chat_id
            WHERE p.is_owner AND c.source_guid = 'chat-1'
            """
        )
        assert cur.fetchone() == (1,)


def test_run_identity_backfills_chat_participant(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)
    run_identity(conn=pg_conn, config=config)

    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.display_name FROM chat_participant cp
            JOIN person p ON p.person_id = cp.person_id
            JOIN chat c ON c.chat_id = cp.chat_id
            WHERE c.source_guid = 'chat-1' AND NOT p.is_owner
            """
        )
        assert cur.fetchall() == [("+14155552671",)]


def test_run_identity_unique_contact_match_names_the_person(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=True)

    alice = ContactRecord(
        identifier="c1",
        display_name="Alice Example",
        organization="Acme Construction",
        normalized_identifiers=(("+14155552671", "phone"),),
    )

    def fake_importer(region: str) -> list[ContactRecord]:
        return [alice]

    result = run_identity(conn=pg_conn, config=config, contacts_importer=fake_importer)
    assert result.contacts.degraded is False
    assert result.contacts.contacts_loaded == 1

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT p.display_name, p.organization FROM message m "
            "JOIN person p ON p.person_id = m.sender_person_id WHERE m.source_guid = 'msg-in-1'"
        )
        assert cur.fetchone() == ("Alice Example", "Acme Construction")


def test_run_identity_multiple_contact_matches_falls_back_to_stub(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=True)

    shared = ("+14155552671", "phone")
    alice = ContactRecord(identifier="c1", display_name="Alice", organization=None, normalized_identifiers=(shared,))
    also_alice = ContactRecord(identifier="c2", display_name="Also Alice", organization=None, normalized_identifiers=(shared,))

    def fake_importer(region: str) -> list[ContactRecord]:
        return [alice, also_alice]

    run_identity(conn=pg_conn, config=config, contacts_importer=fake_importer)

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT p.display_name, p.notes FROM message m "
            "JOIN person p ON p.person_id = m.sender_person_id WHERE m.source_guid = 'msg-in-1'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "+14155552671"  # stub, not either contact's name
        assert row[1] is not None and "review stub" in row[1]


def test_run_identity_cross_references_same_contacts_two_handles(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """Two handles (phone + email) that both belong to the same
    CNContact must resolve to the *same* person, even though the
    schema has nowhere to store the contact's own stable identifier."""
    _seed_extraction(
        pg_conn,
        tmp_path,
        participants=[("1", "+14155552671"), ("2", "alice@example.com")],
    )
    config = _identity_config(config_dict_factory, contacts_import=True)

    alice = ContactRecord(
        identifier="c1",
        display_name="Alice Example",
        organization=None,
        normalized_identifiers=(("+14155552671", "phone"), ("alice@example.com", "email")),
    )

    def fake_importer(region: str) -> list[ContactRecord]:
        return [alice]

    run_identity(conn=pg_conn, config=config, contacts_importer=fake_importer)

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT m.sender_person_id FROM message m WHERE m.source_guid IN ('msg-in-1', 'msg-in-2')"
        )
        assert len(cur.fetchall()) == 1  # same person for both handles

        cur.execute("SELECT count(*) FROM person WHERE display_name = 'Alice Example'")
        assert cur.fetchone() == (1,)  # not duplicated


def test_run_identity_degrades_loudly_but_still_makes_progress(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=True)

    def denying_importer(region: str) -> list[ContactRecord]:
        raise ContactsAccessDeniedError("simulated: Contacts access not authorized")

    result = run_identity(conn=pg_conn, config=config, contacts_importer=denying_importer)
    assert result.contacts.attempted is True
    assert result.contacts.degraded is True
    assert result.contacts.degraded_reason is not None
    assert result.invariant.ok is True  # still fully resolved, just via stub persons

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message WHERE sender_person_id IS NULL")
        assert cur.fetchone() == (0,)


def test_run_identity_is_idempotent(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)
    run_identity(conn=pg_conn, config=config)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM person")
        first_count = cur.fetchone()

    second = run_identity(conn=pg_conn, config=config)
    assert second.source_handles_processed == 0  # nothing left unresolved
    assert second.persons_created == 0

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM person")
        assert cur.fetchone() == first_count


def test_run_identity_dry_run_writes_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM person")
        persons_before = cur.fetchone()
        cur.execute("SELECT count(*) FROM handle")
        handles_before = cur.fetchone()
        cur.execute("SELECT count(*) FROM message WHERE sender_person_id IS NOT NULL")
        resolved_before = cur.fetchone()

    result = run_identity(conn=pg_conn, config=config, dry_run=True)
    assert result.dry_run is True
    # The preview reports what a real run would do...
    assert result.persons_created > 0
    assert result.invariant.ok is True

    # ...but nothing was actually written.
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM person")
        assert cur.fetchone() == persons_before
        cur.execute("SELECT count(*) FROM handle")
        assert cur.fetchone() == handles_before
        cur.execute("SELECT count(*) FROM message WHERE sender_person_id IS NOT NULL")
        assert cur.fetchone() == resolved_before

    # A real run afterward is unaffected by the rolled-back preview and
    # produces the same counts the preview predicted.
    real_result = run_identity(conn=pg_conn, config=config)
    assert real_result.dry_run is False
    assert real_result.persons_created == result.persons_created


def test_compute_invariant_report_and_assert_raises(pg_conn: psycopg.Connection, tmp_path: Path) -> None:
    _seed_extraction(pg_conn, tmp_path)
    # Deliberately do NOT run identity resolution — everything is unresolved.
    report = compute_invariant_report(pg_conn)
    assert isinstance(report, InvariantReport)
    assert report.ok is False
    assert report.unresolved_message_senders > 0
    assert report.owner_person_count == 0

    with pytest.raises(IdentityError, match="unresolved sender_person_id"):
        assert_invariant_or_raise(report)


def test_merge_persons_repoints_everything(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(
        pg_conn, tmp_path, participants=[("1", "+14155552671"), ("2", "+14155552672")]
    )
    config = _identity_config(config_dict_factory, contacts_import=False)
    run_identity(conn=pg_conn, config=config)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE display_name = '+14155552671'")
        keep = _scalar_int(cur)
        cur.execute("SELECT person_id FROM person WHERE display_name = '+14155552672'")
        absorb = _scalar_int(cur)

    merge_persons(pg_conn, keep_person_id=keep, absorb_person_id=absorb)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM person WHERE person_id = %s", (absorb,))
        assert cur.fetchone() == (0,)
        cur.execute(
            "SELECT count(*) FROM message WHERE sender_person_id = %s", (absorb,)
        )
        assert cur.fetchone() == (0,)
        cur.execute(
            "SELECT count(*) FROM message WHERE source_guid = 'msg-in-2' AND sender_person_id = %s",
            (keep,),
        )
        assert cur.fetchone() == (1,)


def test_merge_persons_refuses_to_merge_owner(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)
    run_identity(conn=pg_conn, config=config)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE is_owner")
        owner_id = _scalar_int(cur)
        cur.execute("SELECT person_id FROM person WHERE display_name = '+14155552671'")
        other_id = _scalar_int(cur)

    with pytest.raises(IdentityError, match="owner"):
        merge_persons(pg_conn, keep_person_id=other_id, absorb_person_id=owner_id)


def test_rename_person(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)
    run_identity(conn=pg_conn, config=config)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE display_name = '+14155552671'")
        person_id = _scalar_int(cur)

    rename_person(pg_conn, person_id=person_id, display_name="Alice Example", short_name="alice")

    with pg_conn.cursor() as cur:
        cur.execute("SELECT display_name, short_name FROM person WHERE person_id = %s", (person_id,))
        assert cur.fetchone() == ("Alice Example", "alice")


def test_rename_person_missing_raises(pg_conn: psycopg.Connection) -> None:
    with pytest.raises(IdentityError, match="not found"):
        rename_person(pg_conn, person_id=999999, display_name="Nobody")


def test_assign_handle_repoints_to_new_person(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_extraction(pg_conn, tmp_path)
    config = _identity_config(config_dict_factory, contacts_import=False)
    run_identity(conn=pg_conn, config=config)

    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO person (display_name, short_name) VALUES ('New Owner', 'new-owner') RETURNING person_id"
        )
        new_person_id = _scalar_int(cur)
    pg_conn.commit()

    assign_handle(pg_conn, normalized_value="+14155552671", kind="phone", person_id=new_person_id)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT person_id FROM handle WHERE normalized_value = '+14155552671'")
        assert cur.fetchone() == (new_person_id,)


def test_assign_handle_missing_raises(pg_conn: psycopg.Connection) -> None:
    with pytest.raises(IdentityError, match="no handle found"):
        assign_handle(pg_conn, normalized_value="+10000000000", kind="phone", person_id=1)
