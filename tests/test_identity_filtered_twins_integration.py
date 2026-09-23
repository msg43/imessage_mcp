"""`imsg identity merge-filtered-twins`, and the merge rules it relies on, against a real Postgres.

Every legacy row here is made the way the production index got it: S2
extracts a snapshot whose handles carry iOS filter tags, then S3 runs with
the normalizer as it stood before 2026-08-15, which kept the tag
(`_legacy_identity_import` makes `strip_ios_filter_suffix` a no-op for that
one run). The repair then runs with today's normalizer.

The repair is imported inside each test that uses it, so on a tree without
it only those tests fail, and the merge-rule tests fail for behavioral
reasons of their own.

DB-gated, same scratch-instance pattern as `test_identity.py`. Fictional
personas and numbers only (Alice, Bob, Carol, Dana, Erin, Frank, Acme Bank).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
from conftest import ConfigDictFactory
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.errors import IdentityError
from imsg.export.eligibility import eligible_chat_ids
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import REBUILD_ALL_SENTINEL, find_dirty_chats, run_segment_for_chat
from imsg.stages import identity as identity_module
from imsg.stages.extract import run_extract
from imsg.stages.identity import assign_handle, merge_persons, rename_person, run_identity
from imsg.stages.identity_overrides import (
    IdentityOverride,
    IdentityOverridesFile,
    apply_overrides,
    export_overrides,
)
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun, TapbackInfo
from test_identity import ADMIN_DSN, REACHABLE, REAL_MIGRATIONS_DIR, _dsn, _identity_config
from test_segment_pipeline_integration import PROMPT_BYTES

TEST_DB_NAME = "imsg_index_identity_filtered_twins_test"

pytestmark = pytest.mark.skipif(
    not REACHABLE, reason="no reachable scratch Postgres instance (see test_identity.py)"
)

ALICE = "+14155552671"
BOB = "bob@example.com"
CAROL = "+14155552673"
DANA = "+14155552674"
ERIN = "+14155552675"
FRANK = "+14155552676"
BANK = "24273"

ALICE_TAGGED = f"{ALICE}(filtered)"
BOB_TAGGED = "Bob@Example.com(filtered)"
BOB_LEGACY = "bob@example.com(filtered)"  # what the old normalizer stored: lowercased, tag kept
CAROL_FILTERED = f"{CAROL}(filtered)"
CAROL_SMSFT = f"{CAROL}(smsft)"
DANA_TAGGED = f"{DANA}(smsft_rm)(smsft)"
FRANK_TAGGED = "4155552676(filtered)"
BANK_TAGGED = f"{BANK}(smsft_fi)"

_BASE = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
_TAG_PATTERN = r"\((filtered|smsft)"


@dataclass(frozen=True)
class _Chat:
    guid: str
    members: tuple[str, ...]
    """Raw handle values; each member sends one message, and the owner one."""
    group: bool = False


# One chat per form, as iOS files a filtered sender's messages apart. Alice's
# tagged form also sits in a group with Erin and reacts there. Carol has two
# tagged forms and no untagged one; Erin is the control nothing may touch.
CORPUS: tuple[_Chat, ...] = (
    _Chat("alice", (ALICE,)),
    _Chat("alice-filtered", (ALICE_TAGGED,)),
    _Chat("bob", (BOB,)),
    _Chat("bob-filtered", (BOB_TAGGED,)),
    _Chat("carol-filtered", (CAROL_FILTERED,)),
    _Chat("carol-smsft", (CAROL_SMSFT,)),
    _Chat("dana", (DANA,)),
    _Chat("dana-filtered", (DANA_TAGGED,)),
    _Chat("erin", (ERIN,)),
    _Chat("frank", (FRANK,)),
    _Chat("frank-filtered", (FRANK_TAGGED,)),
    _Chat("bank", (BANK,)),
    _Chat("bank-filtered", (BANK_TAGGED,)),
    _Chat("group", (ALICE_TAGGED, ERIN), group=True),
)
# (tapback guid, chat guid, reacting handle, target message guid)
REACTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("group/react-0", "group", ALICE_TAGGED, "group/in-1"),
)
# Chats whose rendered text names a person the repair merges or renames:
# everything except Erin's own chat.
AFFECTED_CHATS = frozenset(c.guid for c in CORPUS) - {"erin"}


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


@pytest.fixture
def config(config_dict_factory: ConfigDictFactory) -> Config:
    return _identity_config(config_dict_factory, contacts_import=False)


# --- building the legacy index ---------------------------------------------


def _dump(guid: str, rowid: int, *, tapback: TapbackInfo | None = None) -> ImsgDumpMessage:
    return ImsgDumpMessage(
        rowid=rowid, guid=guid, chat_guid=None, handle=None, is_from_me=False, date=None,
        date_edited=None, date_retracted=None, service="iMessage",
        body_text=None if tapback else f"message {rowid}", edit_history=(), is_unsent=False,
        tapback=tapback, attachment_rowids=(), reply_to_guid=None,
    )


def _extract(
    conn: psycopg.Connection,
    tmp_path: Path,
    chats: Sequence[_Chat],
    *,
    reactions: Sequence[tuple[str, str, str, str]] = (),
    source_name: str = "mini",
) -> None:
    """S2 over a synthetic snapshot: every member of a chat sends one
    message (`<chat>/in-<i>`), the owner sends one (`<chat>/out`)."""
    builder = ChatDbBuilder()
    dumped: list[ImsgDumpMessage] = []
    handles: set[str] = set()
    rowid = 0

    def message(guid: str, chat_guid: str, raw: str | None) -> None:
        nonlocal rowid
        rowid += 1
        builder.add_message(
            FixtureMessage(
                guid=guid, chat_guid=chat_guid, is_from_me=raw is None, handle_raw_value=raw,
                date=_BASE + timedelta(minutes=rowid), rowid=rowid,
            )
        )

    for chat in chats:
        builder.add_chat(FixtureChat(guid=chat.guid, style=43 if chat.group else 45))
        for raw in chat.members:
            if raw not in handles:
                service = "SMS" if "(" in raw else "iMessage"
                builder.add_handle(FixtureHandle(raw_value=raw, service=service))
                handles.add(raw)
            builder.link_participant(chat.guid, raw)
        for i, raw in enumerate(chat.members):
            message(f"{chat.guid}/in-{i}", chat.guid, raw)
            dumped.append(_dump(f"{chat.guid}/in-{i}", rowid))
        message(f"{chat.guid}/out", chat.guid, None)
        dumped.append(_dump(f"{chat.guid}/out", rowid))
    for guid, chat_guid, raw, target in reactions:
        message(guid, chat_guid, raw)
        dumped.append(_dump(guid, rowid, tapback=TapbackInfo(kind="loved", target_guid=target)))

    snapshot = builder.build(tmp_path / f"{source_name}-snapshot.db")
    binary = tmp_path / "imsg-dump"
    binary.write_text("")

    def fake_run(binary_path: Path, snap: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(messages=tuple(dumped), stderr_lines=())

    run_extract(
        conn=conn, source_name=source_name, snapshot_path=snapshot,
        imsg_dump_binary=binary, run_imsg_dump_fn=fake_run,
    )
    conn.commit()


def _legacy_identity_import(
    conn: psycopg.Connection, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S3 exactly as it ran before 2026-08-15: no tag stripping."""
    with monkeypatch.context() as patch:
        patch.setattr(identity_module, "strip_ios_filter_suffix", lambda raw_value: raw_value)
        run_identity(conn=conn, config=config)
    conn.commit()


def _legacy_index(
    conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _extract(conn, tmp_path, CORPUS, reactions=REACTIONS)
    _legacy_identity_import(conn, config, monkeypatch)
    # The premise every test here rests on: one person per tagged form.
    assert _handle_kind(conn, ALICE_TAGGED) == "unknown"
    assert _handle_kind(conn, BOB_LEGACY) == "email"
    assert _person_of(conn, ALICE) != _person_of(conn, ALICE_TAGGED)
    assert _count(conn, f"SELECT count(*) FROM handle WHERE normalized_value ~* '{_TAG_PATTERN}'") == 7


def _repair(conn: psycopg.Connection, *, dry_run: bool = False) -> Any:
    from imsg.stages.identity_filtered_twins import run_merge_filtered_twins

    result = run_merge_filtered_twins(conn=conn, default_region="US", dry_run=dry_run)
    conn.commit()
    return result


# --- reading the index -------------------------------------------------------


def _count(conn: psycopg.Connection, sql: str, *params: Any) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _handle_kind(conn: psycopg.Connection, value: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT kind::text FROM handle WHERE normalized_value = %s", (value,))
        row = cur.fetchone()
    return None if row is None else str(row[0])


def _person_of(conn: psycopg.Connection, value: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM handle WHERE normalized_value = %s", (value,))
        row = cur.fetchone()
    assert row is not None, value
    return int(row[0])


def _person_of_source(conn: psycopg.Connection, raw_value: str) -> int:
    """The person a raw source handle resolves to, through its canonical handle."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT h.person_id FROM source_handle sh "
            "JOIN source_handle_resolution shr ON shr.source_handle_id = sh.source_handle_id "
            "JOIN handle h ON h.handle_id = shr.handle_id WHERE sh.raw_value = %s",
            (raw_value,),
        )
        rows = {int(r[0]) for r in cur.fetchall()}
    assert len(rows) == 1, (raw_value, rows)
    return rows.pop()


def _name(conn: psycopg.Connection, person_id: int) -> tuple[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT display_name, short_name FROM person WHERE person_id = %s", (person_id,))
        row = cur.fetchone()
    assert row is not None, person_id
    return str(row[0]), str(row[1])


def _sender(conn: psycopg.Connection, guid: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT sender_person_id FROM message WHERE source_guid = %s", (guid,))
        row = cur.fetchone()
    assert row is not None, guid
    return int(row[0])


def _chat_id(conn: psycopg.Connection, guid: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT chat_id FROM chat WHERE source_guid = %s", (guid,))
        row = cur.fetchone()
    assert row is not None, guid
    return int(row[0])


def _participants(conn: psycopg.Connection, chat_guid: str) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT cp.person_id FROM chat_participant cp JOIN person p USING (person_id) "
            "WHERE cp.chat_id = %s AND NOT p.is_owner",
            (_chat_id(conn, chat_guid),),
        )
        return {int(r[0]) for r in cur.fetchall()}


def _owner(conn: psycopg.Connection) -> int:
    return _count(conn, "SELECT person_id FROM person WHERE is_owner")


def _state(conn: psycopg.Connection) -> dict[str, list[Any]]:
    """Every row the repair may touch, `updated_at` included."""
    queries = {
        "person": "SELECT person_id, display_name, short_name, needs_review, organization, notes, "
        "updated_at FROM person ORDER BY person_id",
        "handle": "SELECT handle_id, person_id, normalized_value, kind FROM handle ORDER BY handle_id",
        "source_handle": "SELECT source_handle_id, raw_value, service FROM source_handle "
        "ORDER BY source_handle_id",
        "resolution": "SELECT source_handle_id, handle_id FROM source_handle_resolution "
        "ORDER BY source_handle_id",
        "message": "SELECT message_id, sender_person_id, updated_at FROM message ORDER BY message_id",
        "tapback": "SELECT tapback_id, sender_person_id FROM tapback ORDER BY tapback_id",
        "chat_participant": "SELECT chat_id, person_id FROM chat_participant ORDER BY 1, 2",
        "allowlist": "SELECT * FROM allowlist_person ORDER BY person_id",
        "chat": "SELECT chat_id, source_guid, display_name FROM chat ORDER BY chat_id",
    }
    state: dict[str, list[Any]] = {}
    with conn.cursor() as cur:
        for key, sql in queries.items():
            cur.execute(sql)
            state[key] = cur.fetchall()
    return state


def _allow(conn: psycopg.Connection, person_id: int, *, text: bool, attachments: bool) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO allowlist_person (person_id, text_allowed, attachments_allowed) "
            "VALUES (%s, %s, %s)",
            (person_id, text, attachments),
        )
    conn.commit()


def _allowlist_row(conn: psycopg.Connection, person_id: int) -> tuple[bool, bool] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT text_allowed, attachments_allowed FROM allowlist_person WHERE person_id = %s",
            (person_id,),
        )
        row = cur.fetchone()
    return None if row is None else (bool(row[0]), bool(row[1]))


# --- the repair --------------------------------------------------------------


def test_repair_merges_every_twin_and_gives_a_twinless_handle_its_clean_value(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    before = _state(pg_conn)
    persons_before = len(before["person"])
    alice = _person_of(pg_conn, ALICE)

    result = _repair(pg_conn)

    assert (result.tagged_source_handles, result.legacy_source_handles, result.legacy_handles) == (7, 7, 7)
    assert result.persons_merged == 6  # Alice, Bob, Dana, Frank, the bank, and Carol's second form
    assert result.handles_rewritten == 1  # Carol has no untagged handle
    assert result.legacy_handles_merged == 6
    assert result.source_handles_repointed == 6
    assert result.stubs_renamed == 1
    assert result.refused == ()
    assert result.other_stale_source_handles == 0
    assert result.invariant.ok
    assert result.chats_marked_dirty == len(AFFECTED_CHATS)

    # One person per sender, whichever form their messages came in.
    for clean, tagged in (
        (ALICE, ALICE_TAGGED), (BOB, BOB_TAGGED), (DANA, DANA_TAGGED),
        (FRANK, FRANK_TAGGED), (BANK, BANK_TAGGED),
    ):
        assert _person_of_source(pg_conn, tagged) == _person_of_source(pg_conn, clean), tagged
    assert _person_of_source(pg_conn, CAROL_FILTERED) == _person_of_source(pg_conn, CAROL_SMSFT)
    assert len(_state(pg_conn)["person"]) == persons_before - 6

    # Senders, reactions and chat membership followed the merge.
    assert _sender(pg_conn, "alice-filtered/in-0") == alice
    assert _sender(pg_conn, "group/in-0") == alice
    assert _count(
        pg_conn, "SELECT count(*) FROM tapback WHERE source_guid = 'group/react-0' "
        "AND sender_person_id = %s", alice,
    ) == 1
    assert _participants(pg_conn, "alice-filtered") == {alice}
    assert _participants(pg_conn, "group") == {alice, _person_of(pg_conn, ERIN)}

    # Carol's first tagged handle became her clean handle, and her stub took the clean name.
    assert _handle_kind(pg_conn, CAROL) == "phone"
    assert _name(pg_conn, _person_of(pg_conn, CAROL)) == (CAROL, "14155552673")
    # No canonical handle carries a tag any more; raw provenance is untouched.
    assert _count(pg_conn, f"SELECT count(*) FROM handle WHERE normalized_value ~* '{_TAG_PATTERN}'") == 0
    after = _state(pg_conn)
    for table in ("source_handle", "chat"):
        assert after[table] == before[table], table
    assert [m[0] for m in after["message"]] == [m[0] for m in before["message"]]
    assert [t[0] for t in after["tapback"]] == [t[0] for t in before["tapback"]]
    assert len(after["resolution"]) == len(before["resolution"])
    # Erin is in none of it.
    erin = _person_of(pg_conn, ERIN)
    assert _name(pg_conn, erin) == (ERIN, "14155552675")


def test_repair_keeps_a_curated_name_from_either_side(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    alice = _person_of(pg_conn, ALICE)
    dana_tagged = _person_of(pg_conn, f"{DANA}(smsft_rm)(smsft)")
    dana_clean = _person_of(pg_conn, DANA)
    frank = _person_of(pg_conn, FRANK)
    rename_person(pg_conn, person_id=alice, display_name="Alice Example", short_name="alice-example")
    rename_person(pg_conn, person_id=dana_tagged, display_name="Dana Example", short_name="dana-example")
    rename_person(pg_conn, person_id=frank, display_name="Frank Example")
    rename_person(pg_conn, person_id=_person_of(pg_conn, FRANK_TAGGED), display_name="frank example")
    pg_conn.commit()

    result = _repair(pg_conn)

    assert result.persons_merged == 6
    assert result.refused == ()
    # The twin was named: it is kept, name and slug unchanged.
    assert _person_of_source(pg_conn, ALICE_TAGGED) == alice
    assert _name(pg_conn, alice) == ("Alice Example", "alice-example")
    # Only the tagged side was named: that person is kept, and holds the clean handle.
    assert _person_of(pg_conn, DANA) == dana_tagged
    assert _name(pg_conn, dana_tagged) == ("Dana Example", "dana-example")
    assert _count(pg_conn, "SELECT count(*) FROM person WHERE person_id = %s", dana_clean) == 0
    # The same curated name on both sides is one person: merged, the twin kept.
    assert _person_of_source(pg_conn, FRANK_TAGGED) == frank
    assert _name(pg_conn, frank)[0] == "Frank Example"


def test_repair_refuses_different_curated_names_and_the_owner_and_changes_nothing_about_them(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    bob, bob_tagged = _person_of(pg_conn, BOB), _person_of(pg_conn, BOB_LEGACY)
    rename_person(pg_conn, person_id=bob, display_name="Bob Builder")
    rename_person(pg_conn, person_id=bob_tagged, display_name="Robert Builder")
    owner = _owner(pg_conn)
    bank_tagged = _person_of(pg_conn, BANK_TAGGED)
    assign_handle(pg_conn, normalized_value=BANK, kind="unknown", person_id=owner)
    pg_conn.commit()
    before = _state(pg_conn)

    result = _repair(pg_conn)

    assert {(p.reason, p.legacy_person_id, p.target_person_id) for p in result.refused} == {
        ("curated_conflict", bob_tagged, bob),
        ("owner", bank_tagged, owner),
    }
    conflict = next(p for p in result.refused if p.reason == "curated_conflict")
    assert (conflict.legacy_display_name, conflict.target_display_name) == ("Robert Builder", "Bob Builder")
    assert (conflict.clean_value, conflict.clean_kind) == (BOB, "email")
    assert result.persons_merged == 4  # Alice, Dana, Frank, Carol's second form
    after = _state(pg_conn)
    for person_id in (bob, bob_tagged, bank_tagged, owner):
        assert [p for p in after["person"] if p[0] == person_id] == [
            p for p in before["person"] if p[0] == person_id
        ]
    for value in (BOB_LEGACY, BANK_TAGGED):  # the legacy handles stay where they were
        assert _handle_kind(pg_conn, value) is not None
    assert _person_of_source(pg_conn, BOB_TAGGED) == bob_tagged
    assert _sender(pg_conn, "bob-filtered/in-0") == bob_tagged
    assert _sender(pg_conn, "bank-filtered/in-0") == bank_tagged

    # Refusals are stable: a second run refuses the same pairs and writes nothing.
    settled = _state(pg_conn)
    again = _repair(pg_conn)
    assert again.refused == result.refused
    assert again.persons_merged == 0
    assert _state(pg_conn) == settled


def test_repair_marks_exactly_the_chats_whose_segments_name_the_merged_persons(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    rename_person(pg_conn, person_id=_person_of(pg_conn, ALICE), display_name="Alice Example")
    pg_conn.commit()
    guids = {c.guid: _chat_id(pg_conn, c.guid) for c in CORPUS}
    for chat_id in guids.values():
        run_segment_for_chat(
            pg_conn, chat_id, config, FakeBoundaryProvider(), PROMPT_BYTES,
            earliest_changed_at=REBUILD_ALL_SENTINEL,
        )
        pg_conn.commit()
    assert find_dirty_chats(pg_conn, index_unsent=False) == {}
    group_text = _rendered(pg_conn, guids["group"])
    assert ALICE_TAGGED in group_text  # the header names the legacy stub

    result = _repair(pg_conn)

    dirty = find_dirty_chats(pg_conn, index_unsent=False)
    assert set(dirty) == {guids[g] for g in AFFECTED_CHATS}
    assert guids["erin"] not in dirty
    assert result.chats_marked_dirty == len(AFFECTED_CHATS)

    span = dirty[guids["group"]]
    run_segment_for_chat(
        pg_conn, guids["group"], config, FakeBoundaryProvider(), PROMPT_BYTES,
        earliest_changed_at=span.earliest_changed_at, latest_changed_at=span.latest_changed_at,
    )
    pg_conn.commit()
    group_text = _rendered(pg_conn, guids["group"])
    assert "Alice Example" in group_text
    assert "filtered" not in group_text


def _rendered(conn: psycopg.Connection, chat_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT string_agg(rendered_text, E'\\n' ORDER BY started_at, seq_in_session) "
            "FROM segment WHERE chat_id = %s",
            (chat_id,),
        )
        row = cur.fetchone()
    assert row is not None and row[0] is not None
    return str(row[0])


def test_dry_run_reports_the_real_counts_and_writes_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    before = _state(pg_conn)

    dry = _repair(pg_conn, dry_run=True)

    assert dry.dry_run is True
    assert _state(pg_conn) == before
    real = _repair(pg_conn)
    assert replace(dry, dry_run=False) == real
    assert real.persons_merged == 6


def test_second_run_changes_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    _repair(pg_conn)
    settled = _state(pg_conn)

    again = _repair(pg_conn)

    assert (again.legacy_handles, again.persons_merged, again.chats_marked_dirty) == (0, 0, 0)
    assert again.tagged_source_handles == 7
    assert _state(pg_conn) == settled


def test_new_tagged_and_clean_handles_join_the_existing_person_after_the_repair(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carol was only ever seen tagged, so before the repair no handle held
    her clean number: the next message from it, tagged or not, would have
    become a new stub. After the repair both forms reach her person."""
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    _repair(pg_conn)
    carol = _person_of(pg_conn, CAROL)
    alice = _person_of(pg_conn, ALICE)
    persons = _count(pg_conn, "SELECT count(*) FROM person")

    later = (
        _Chat("carol-later-tagged", (f"{CAROL}(smsft_or)",)),
        _Chat("carol-later-clean", (CAROL,)),
        _Chat("carol-later-national", ("4155552673",)),
        _Chat("alice-later-tagged", (f"{ALICE} (smsft_rm) (smsft)",)),
    )
    _extract(pg_conn, tmp_path, later, source_name="studio")
    resolved = run_identity(conn=pg_conn, config=config)
    pg_conn.commit()

    assert resolved.persons_created == 0
    assert _count(pg_conn, "SELECT count(*) FROM person") == persons
    assert _sender(pg_conn, "carol-later-tagged/in-0") == carol
    assert _sender(pg_conn, "carol-later-clean/in-0") == carol
    assert _sender(pg_conn, "carol-later-national/in-0") == carol
    assert _sender(pg_conn, "alice-later-tagged/in-0") == alice


def test_curated_decisions_replay_the_same_after_the_repair(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's decisions file names some senders by their tagged form.
    Replaying it after the repair must apply nothing new, report the same
    outcome for every decision, and find every name where it was."""
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    decisions = (
        IdentityOverride(value=ALICE_TAGGED, kind="unknown", name="Alice Example"),
        IdentityOverride(value=ALICE, kind="phone", name="Alice Example"),
        IdentityOverride(value=BOB, kind="email", name="Bob Builder"),
        # Disagrees with the decision above about the same sender: a
        # conflict on every run, before the repair and after it.
        IdentityOverride(value=BOB_LEGACY, kind="email", name="Robert Builder"),
        IdentityOverride(value=BANK, kind="unknown", name="Acme Bank Alerts"),
        IdentityOverride(value=CAROL_FILTERED, kind="unknown", name="Carol Chen"),
    )
    apply_overrides(pg_conn, overrides=decisions, default_region="US")
    pg_conn.commit()
    before = apply_overrides(pg_conn, overrides=decisions, default_region="US")
    pg_conn.commit()
    assert before.summary == "applied=0 already=5 unmatched=0 conflicts=1"

    _repair(pg_conn)

    after = apply_overrides(pg_conn, overrides=decisions, default_region="US")
    pg_conn.commit()
    assert after.summary == before.summary
    assert [o.status for o in after.outcomes] == [o.status for o in before.outcomes]
    assert after.chats_marked_dirty == 0
    for value, name in (
        (ALICE, "Alice Example"), (BOB, "Bob Builder"), (BANK, "Acme Bank Alerts"),
        (CAROL, "Carol Chen"),
    ):
        assert _name(pg_conn, _person_of(pg_conn, value))[0] == name
    for raw, clean in (
        (ALICE_TAGGED, ALICE), (BOB_TAGGED, BOB), (BANK_TAGGED, BANK),
        (CAROL_FILTERED, CAROL), (CAROL_SMSFT, CAROL),
    ):
        assert _person_of_source(pg_conn, raw) == _person_of(pg_conn, clean), raw

    # With the legacy rows gone, a decision recorded against a tagged form
    # follows the clean handle when the file is regenerated.
    exported = export_overrides(
        pg_conn, default_region="US", previous=IdentityOverridesFile(overrides=decisions)
    )
    assert IdentityOverride(value=CAROL, kind="phone", name="Carol Chen") in exported.file.overrides
    assert not any(o.value == CAROL_FILTERED for o in exported.file.overrides)


def test_merged_stubs_stay_nameable_by_contacts_and_out_of_the_decisions_file(
    pg_conn: psycopg.Connection,
    tmp_path: Path,
    config: Config,
    config_dict_factory: ConfigDictFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the emptied legacy handle is removed rather than left on the kept
    person. `rematch-stubs` names a stub only when every handle it holds is
    on the card, and no card carries a tagged value; `export-overrides`
    treats a person holding two handles as a hand merge. A leftover tagged
    row would break both for every merged stub."""
    from imsg.stages.identity import ContactRecord
    from imsg.stages.identity_rematch import run_rematch_stubs

    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    _repair(pg_conn)
    dana = _person_of(pg_conn, DANA)

    exported = export_overrides(pg_conn, default_region="US")
    assert exported.file.overrides == ()  # every person is still an unreviewed one-handle stub

    card = ContactRecord(
        identifier="card-dana", display_name="Dana Example", organization=None,
        normalized_identifiers=((DANA, "phone"),),
    )
    rematched = run_rematch_stubs(
        conn=pg_conn,
        config=_identity_config(config_dict_factory, contacts_import=True),
        contacts_importer=lambda region: [card],
    )
    pg_conn.commit()
    assert [(o.person_id, o.status) for o in rematched.outcomes if o.status == "matched"] == [
        (dana, "matched")
    ]
    assert _name(pg_conn, dana)[0] == "Dana Example"


def test_repair_never_widens_allowlist_access(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    alice, alice_tagged = _person_of(pg_conn, ALICE), _person_of(pg_conn, ALICE_TAGGED)
    _allow(pg_conn, _owner(pg_conn), text=True, attachments=True)
    _allow(pg_conn, alice_tagged, text=True, attachments=True)
    assert eligible_chat_ids(pg_conn) == {_chat_id(pg_conn, "alice-filtered")}

    result = _repair(pg_conn)

    assert (result.allowlist_rows_absorbed, result.allowlist_rows_narrowed) == (1, 0)
    assert _allowlist_row(pg_conn, alice) is None
    assert eligible_chat_ids(pg_conn) == set()


def test_repair_refuses_a_legacy_handle_that_an_untagged_source_also_resolves_to(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cannot arise from the old normalizer; if it ever does, guessing which
    person the untagged sender belongs to is not the repair's call."""
    _legacy_index(pg_conn, tmp_path, config, monkeypatch)
    with pg_conn.cursor() as cur:
        cur.execute(
            "UPDATE source_handle_resolution SET handle_id = "
            "(SELECT handle_id FROM handle WHERE normalized_value = %s) "
            "WHERE source_handle_id = (SELECT source_handle_id FROM source_handle WHERE raw_value = %s)",
            (ALICE_TAGGED, ERIN),
        )
    pg_conn.commit()
    before = _state(pg_conn)

    with pytest.raises(IdentityError, match="refuses to guess"):
        _repair(pg_conn)
    pg_conn.rollback()

    assert _state(pg_conn) == before


# --- merge_persons: allowlist rows follow a merge without widening access ----


def _two_people(
    conn: psycopg.Connection, tmp_path: Path, config: Config
) -> tuple[int, int, int, int]:
    """Alice and Bob, one 1:1 chat each: `(alice, bob, alice_chat, bob_chat)`."""
    _extract(conn, tmp_path, (_Chat("alice", (ALICE,)), _Chat("bob", (BOB,))))
    run_identity(conn=conn, config=config)
    conn.commit()
    return (
        _person_of(conn, ALICE), _person_of(conn, BOB),
        _chat_id(conn, "alice"), _chat_id(conn, "bob"),
    )


def test_merge_never_copies_the_absorbed_persons_flags_onto_the_kept_person(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    alice, bob, _alice_chat, bob_chat = _two_people(pg_conn, tmp_path, config)
    _allow(pg_conn, _owner(pg_conn), text=True, attachments=True)
    _allow(pg_conn, bob, text=True, attachments=True)
    assert eligible_chat_ids(pg_conn) == {bob_chat}

    merge_persons(pg_conn, keep_person_id=alice, absorb_person_id=bob)
    pg_conn.commit()

    assert _allowlist_row(pg_conn, alice) is None
    assert _allowlist_row(pg_conn, bob) is None
    assert eligible_chat_ids(pg_conn) == set()


def test_merge_narrows_the_kept_row_by_the_absorbed_row_and_keeps_it_otherwise(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    alice, bob, _alice_chat, _bob_chat = _two_people(pg_conn, tmp_path, config)
    _allow(pg_conn, alice, text=True, attachments=True)
    _allow(pg_conn, bob, text=True, attachments=False)

    merge_persons(pg_conn, keep_person_id=alice, absorb_person_id=bob)
    pg_conn.commit()

    assert _allowlist_row(pg_conn, alice) == (True, False)
    assert _allowlist_row(pg_conn, bob) is None


def test_merge_keeps_the_kept_row_when_the_absorbed_person_has_none(
    pg_conn: psycopg.Connection, tmp_path: Path, config: Config
) -> None:
    alice, bob, _alice_chat, _bob_chat = _two_people(pg_conn, tmp_path, config)
    _allow(pg_conn, alice, text=True, attachments=True)

    merge_persons(pg_conn, keep_person_id=alice, absorb_person_id=bob)
    pg_conn.commit()

    assert _allowlist_row(pg_conn, alice) == (True, True)
