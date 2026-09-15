"""Unit tests (file schema, identifier forms, summary) + a live-Postgres
integration suite for replaying and exporting curated identity decisions
(`imsg.stages.identity_overrides`). Every persona, number and address
here is fictional.

The DB-gated tests borrow `test_identity.py`'s scratch-instance settings
and its S2->S3 seeding helper, but run in their own database; they skip
(from the `pg_conn` fixture, so the unit tests above it still run) when
no scratch Postgres is reachable.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import ConfigDictFactory
from imsg.db.migrations import PostgresMigrationRunner
from imsg.errors import IdentityError
from imsg.stages.identity import (
    InvariantReport,
    assign_handle,
    merge_persons,
    rename_person,
    run_identity,
)
from imsg.stages.identity_overrides import (
    DEFAULT_NOTE,
    ApplyOverridesResult,
    IdentityOverride,
    IdentityOverridesFile,
    OverrideOutcome,
    apply_overrides,
    export_overrides,
    identifier_forms,
    load_overrides,
    parse_overrides,
    write_overrides,
)
from test_identity import (
    ADMIN_DSN,
    REACHABLE,
    REAL_MIGRATIONS_DIR,
    _dsn,
    _identity_config,
    _seed_extraction,
)

# --------------------------------------------------------------------------
# unit tests: no DB
# --------------------------------------------------------------------------

ALICE_PHONE = "+14155552671"
BOB_PHONE = "+14155552672"
ALICE_EMAIL = "alice@example.com"
SHORT_CODE = "24273"


def _document(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"note": "fixture", "decided": "2026-08-15", "overrides": list(entries)}


def test_parse_overrides_accepts_the_documented_schema() -> None:
    parsed = parse_overrides(
        _document(
            {
                "value": f"  {ALICE_PHONE} ",
                "kind": "phone",
                "name": " Alice Example ",
                "was": ["Contacts Conflict", " "],
                "why": "the household number; Alice is the one who texts",
            },
            {"value": SHORT_CODE, "kind": "unknown", "name": "Acme Bank Alerts"},
        )
    )
    assert parsed.note == "fixture"
    assert parsed.decided == "2026-08-15"
    assert parsed.overrides == (
        IdentityOverride(
            value=ALICE_PHONE,
            kind="phone",
            name="Alice Example",
            was=("Contacts Conflict",),
            why="the household number; Alice is the one who texts",
        ),
        IdentityOverride(value=SHORT_CODE, kind="unknown", name="Acme Bank Alerts"),
    )


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (["not", "an", "object"], "top level must be a JSON object"),
        ({"overrides": {}}, "'overrides' must be a list"),
        ({"overrides": [], "extra": 1}, "unknown top-level key"),
        ({"overrides": [], "note": 5}, "'note' must be a string"),
        ({"overrides": ["x"]}, "overrides[0]: each override must be a JSON object"),
        ({"overrides": [{"value": ALICE_PHONE, "kind": "phone"}]}, "overrides[0]: 'name' must be"),
        ({"overrides": [{"value": " ", "kind": "phone", "name": "A"}]}, "'value' must be"),
        (
            {"overrides": [{"value": ALICE_PHONE, "kind": "fax", "name": "A"}]},
            "'kind' must be one of",
        ),
        (
            {"overrides": [{"value": ALICE_PHONE, "kind": "phone", "name": "A", "was": "x"}]},
            "'was' must be a list",
        ),
        (
            {"overrides": [{"value": ALICE_PHONE, "kind": "phone", "name": "A", "why": 3}]},
            "'why' must be a string",
        ),
        (
            {"overrides": [{"value": ALICE_PHONE, "kind": "phone", "name": "A", "op": "rename"}]},
            "unknown key(s) ['op']",
        ),
        (
            {
                "overrides": [
                    {"value": ALICE_PHONE, "kind": "phone", "name": "A"},
                    {"value": ALICE_PHONE, "kind": "phone", "name": "B"},
                ]
            },
            "overrides[1]: repeats the identifier of overrides[0]",
        ),
    ],
)
def test_parse_overrides_rejects_malformed_documents(document: object, message: str) -> None:
    with pytest.raises(IdentityError, match=re.escape(message)):
        parse_overrides(document, source="fixture")


def test_load_overrides_reports_unreadable_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(IdentityError, match="cannot read"):
        load_overrides(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(IdentityError, match="not valid JSON"):
        load_overrides(broken)


def test_write_then_load_round_trips_and_keeps_key_order(tmp_path: Path) -> None:
    path = tmp_path / "private" / "identity-overrides.json"
    file = IdentityOverridesFile(
        overrides=(
            IdentityOverride(
                value=ALICE_PHONE,
                kind="phone",
                name="Alice Example",
                was=("Old Name",),
                why="because",
            ),
            IdentityOverride(value=SHORT_CODE, kind="unknown", name="Acme Bank Alerts"),
        ),
        note="keep me",
        decided="2026-08-15",
    )
    write_overrides(path, file)

    assert load_overrides(path) == file
    text = path.read_text()
    assert text.endswith("}\n")
    assert not (tmp_path / "private" / "identity-overrides.json.tmp").exists()
    raw = json.loads(text)
    assert list(raw) == ["note", "decided", "overrides"]
    assert list(raw["overrides"][0]) == ["value", "kind", "name", "was", "why"]
    assert list(raw["overrides"][1]) == ["value", "kind", "name", "was"]  # no why -> key omitted


def test_write_overrides_fills_note_and_decided_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    write_overrides(path, IdentityOverridesFile(overrides=()))
    raw = json.loads(path.read_text())
    assert raw["note"] == DEFAULT_NOTE
    dt.date.fromisoformat(raw["decided"])  # today's date, well-formed
    assert raw["overrides"] == []


def test_identifier_forms_add_the_normalized_form_when_it_differs() -> None:
    # Recorded before the iOS filter-tag fix: the tagged number sat in kind='unknown'.
    tagged = IdentityOverride(
        value=f"{ALICE_PHONE}(filtered)", kind="unknown", name="Alice Example"
    )
    assert identifier_forms(tagged, "US") == (
        (f"{ALICE_PHONE}(filtered)", "unknown"),
        (ALICE_PHONE, "phone"),
    )
    clean = IdentityOverride(value=ALICE_PHONE, kind="phone", name="Alice Example")
    assert identifier_forms(clean, "US") == ((ALICE_PHONE, "phone"),)
    short_code = IdentityOverride(value=SHORT_CODE, kind="unknown", name="Acme Bank Alerts")
    assert identifier_forms(short_code, "US") == ((SHORT_CODE, "unknown"),)


def test_result_counts_and_summary_line() -> None:
    o = IdentityOverride(value=ALICE_PHONE, kind="phone", name="Alice Example")
    result = ApplyOverridesResult(
        outcomes=(
            OverrideOutcome(o, "applied", "renamed"),
            OverrideOutcome(o, "applied", "renamed", forced=True),
            OverrideOutcome(o, "already", "-"),
            OverrideOutcome(o, "unmatched", "-"),
            OverrideOutcome(o, "conflict", "-"),
            OverrideOutcome(o, "conflict", "-"),
        ),
        invariant=InvariantReport(0, 0, 0, 1),
    )
    assert result.summary == "applied=2 already=1 unmatched=1 conflicts=2"
    assert result.forced == 1
    assert result.dry_run is False


# --------------------------------------------------------------------------
# live-Postgres integration suite
# --------------------------------------------------------------------------

TEST_DB_NAME = "imsg_index_identity_overrides_test"


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
    stub named after its normalized value, which is exactly the state an
    identity rebuild leaves behind."""
    _seed_extraction(
        conn, tmp_path, participants=[(str(i), raw) for i, raw in enumerate(handles, start=1)]
    )
    run_identity(conn=conn, config=_identity_config(config_dict_factory, contacts_import=False))


def _person_of(conn: psycopg.Connection, value: str) -> tuple[int, str, bool]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.person_id, p.display_name, p.needs_review FROM handle h "
            "JOIN person p ON p.person_id = h.person_id WHERE h.normalized_value = %s",
            (value,),
        )
        row = cur.fetchone()
    assert row is not None, value
    return int(row[0]), str(row[1]), bool(row[2])


def _sender_of(conn: psycopg.Connection, guid: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT sender_person_id FROM message WHERE source_guid = %s", (guid,))
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _state_snapshot(conn: psycopg.Connection) -> tuple[list[Any], list[Any], list[Any]]:
    """Everything a replay may touch, including `updated_at`: a second run
    that changes nothing must leave this byte-identical."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT person_id, display_name, short_name, needs_review, notes, updated_at "
            "FROM person ORDER BY person_id"
        )
        persons = cur.fetchall()
        cur.execute(
            "SELECT handle_id, person_id, normalized_value, kind FROM handle ORDER BY handle_id"
        )
        handles = cur.fetchall()
        cur.execute("SELECT message_id, sender_person_id FROM message ORDER BY message_id")
        messages = cur.fetchall()
    return persons, handles, messages


def _count(conn: psycopg.Connection, sql: str, *params: Any) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def test_apply_renames_stubs_and_unifies_handles_that_share_a_name(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE, ALICE_EMAIL]
    )
    decisions = [
        IdentityOverride(
            value=ALICE_PHONE, kind="phone", name="Alice Example", why="she signs her texts"
        ),
        IdentityOverride(value=BOB_PHONE, kind="phone", name="Bob Example"),
        IdentityOverride(value=ALICE_EMAIL, kind="email", name="Alice Example"),
    ]

    result = apply_overrides(pg_conn, overrides=decisions, default_region="US")

    assert [o.status for o in result.outcomes] == ["applied", "applied", "applied"]
    assert result.summary == "applied=3 already=0 unmatched=0 conflicts=0"
    assert result.invariant.ok is True
    assert result.dry_run is False
    assert "renamed" in result.outcomes[0].detail
    assert "merged" in result.outcomes[2].detail  # the email stub folds into Alice

    alice_id, alice_name, alice_review = _person_of(pg_conn, ALICE_PHONE)
    assert (alice_name, alice_review) == ("Alice Example", False)
    assert _person_of(pg_conn, ALICE_EMAIL)[0] == alice_id  # one person, both handles
    assert _sender_of(pg_conn, "msg-in-1") == alice_id
    assert _sender_of(pg_conn, "msg-in-3") == alice_id  # the email's messages moved with it
    assert _person_of(pg_conn, BOB_PHONE)[1:] == ("Bob Example", False)
    assert _count(pg_conn, "SELECT count(*) FROM person WHERE display_name = 'Alice Example'") == 1
    assert (
        _count(pg_conn, "SELECT count(*) FROM person WHERE NOT is_owner") == 2
    )  # 3 stubs -> 2 persons


def test_apply_is_idempotent(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE, ALICE_EMAIL]
    )
    decisions = [
        IdentityOverride(value=ALICE_PHONE, kind="phone", name="Alice Example"),
        IdentityOverride(value=BOB_PHONE, kind="phone", name="Bob Example"),
        IdentityOverride(value=ALICE_EMAIL, kind="email", name="Alice Example"),
    ]
    first = apply_overrides(pg_conn, overrides=decisions, default_region="US")
    assert first.applied == 3
    before = _state_snapshot(pg_conn)

    second = apply_overrides(pg_conn, overrides=decisions, default_region="US")

    assert second.summary == "applied=0 already=3 unmatched=0 conflicts=0"
    assert second.invariant.ok is True
    assert _state_snapshot(pg_conn) == before  # nothing written, not even updated_at

    # Case-only drift in the file is not a reason to write either.
    shouted = [IdentityOverride(value=o.value, kind=o.kind, name=o.name.upper()) for o in decisions]
    third = apply_overrides(pg_conn, overrides=shouted, default_region="US")
    assert third.already == 3
    assert _state_snapshot(pg_conn) == before


def test_apply_dry_run_reports_the_real_outcomes_and_writes_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, ALICE_EMAIL])
    decisions = [
        IdentityOverride(value=ALICE_PHONE, kind="phone", name="Alice Example"),
        IdentityOverride(value=ALICE_EMAIL, kind="email", name="Alice Example"),
    ]
    before = _state_snapshot(pg_conn)

    preview = apply_overrides(pg_conn, overrides=decisions, default_region="US", dry_run=True)

    assert preview.dry_run is True
    # The second decision only merges because the preview really ran the first.
    assert [o.status for o in preview.outcomes] == ["applied", "applied"]
    assert "merged" in preview.outcomes[1].detail
    assert preview.invariant.ok is True
    assert _state_snapshot(pg_conn) == before

    real = apply_overrides(pg_conn, overrides=decisions, default_region="US")
    assert real.summary == preview.summary
    assert [o.detail for o in real.outcomes] == [o.detail for o in preview.outcomes]


def test_apply_reports_unmatched_identifiers_without_inventing_anything(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE])
    persons_before = _count(pg_conn, "SELECT count(*) FROM person")
    handles_before = _count(pg_conn, "SELECT count(*) FROM handle")

    result = apply_overrides(
        pg_conn,
        overrides=[
            IdentityOverride(value=BOB_PHONE, kind="phone", name="Bob Example"),
            IdentityOverride(value="carol@example.com", kind="email", name="Carol Example"),
        ],
        default_region="US",
    )

    assert result.summary == "applied=0 already=0 unmatched=2 conflicts=0"
    assert "no handle in the index" in result.outcomes[0].detail
    assert _count(pg_conn, "SELECT count(*) FROM person") == persons_before
    assert _count(pg_conn, "SELECT count(*) FROM handle") == handles_before
    assert (
        _count(
            pg_conn,
            "SELECT count(*) FROM person WHERE display_name IN ('Bob Example', 'Carol Example')",
        )
        == 0
    )


def test_apply_matches_a_handle_normalized_after_the_file_was_written(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """The file recorded the sender as iOS wrote it — tagged, and therefore
    `kind='unknown'` at the time. S3 now strips the tag, so the rebuilt
    index only has the clean phone handle; the decision must still find it."""
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[f"{ALICE_PHONE}(filtered)"])
    assert _person_of(pg_conn, ALICE_PHONE)[1] == ALICE_PHONE  # clean stub, no tagged handle exists
    assert _count(pg_conn, "SELECT count(*) FROM handle WHERE kind = 'unknown'") == 0

    result = apply_overrides(
        pg_conn,
        overrides=[
            IdentityOverride(value=f"{ALICE_PHONE}(filtered)", kind="unknown", name="Alice Example")
        ],
        default_region="US",
    )

    assert result.summary == "applied=1 already=0 unmatched=0 conflicts=0"
    assert _person_of(pg_conn, ALICE_PHONE)[1:] == ("Alice Example", False)


def test_apply_skips_a_person_named_since_unless_forced(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE])
    person_id = _person_of(pg_conn, ALICE_PHONE)[0]
    # A later hand-curation the file knows nothing about.
    rename_person(pg_conn, person_id=person_id, display_name="Carol Example")
    stale = IdentityOverride(value=ALICE_PHONE, kind="phone", name="Alice Example")

    result = apply_overrides(pg_conn, overrides=[stale], default_region="US")
    assert result.summary == "applied=0 already=0 unmatched=0 conflicts=1"
    assert "named 'Carol Example'" in result.outcomes[0].detail
    assert "--force" in result.outcomes[0].detail
    assert _person_of(pg_conn, ALICE_PHONE)[1] == "Carol Example"  # untouched

    forced = apply_overrides(pg_conn, overrides=[stale], default_region="US", force=True)
    assert forced.summary == "applied=1 already=0 unmatched=0 conflicts=0"
    assert forced.outcomes[0].forced is True
    assert forced.forced == 1
    assert _person_of(pg_conn, ALICE_PHONE)[1] == "Alice Example"


def test_apply_treats_a_name_listed_in_was_as_not_yet_decided(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """`was` records what the person was called when the decision was made
    (a Contacts name, a placeholder). Finding that name is the pre-decision
    state, not a conflict."""
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE])
    person_id = _person_of(pg_conn, ALICE_PHONE)[0]
    rename_person(pg_conn, person_id=person_id, display_name="Contacts Conflict")

    result = apply_overrides(
        pg_conn,
        overrides=[
            IdentityOverride(
                value=ALICE_PHONE, kind="phone", name="Alice Example", was=("contacts conflict",)
            )
        ],
        default_region="US",
    )

    assert result.summary == "applied=1 already=0 unmatched=0 conflicts=0"
    assert result.outcomes[0].forced is False
    assert _person_of(pg_conn, ALICE_PHONE)[1] == "Alice Example"


def test_apply_assigns_only_the_handle_when_its_person_keeps_others(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE, ALICE_EMAIL]
    )
    # Dana holds two handles (a merge somebody made), Alice already exists by name.
    dana_id = _person_of(pg_conn, BOB_PHONE)[0]
    merge_persons(
        pg_conn, keep_person_id=dana_id, absorb_person_id=_person_of(pg_conn, ALICE_EMAIL)[0]
    )
    rename_person(pg_conn, person_id=dana_id, display_name="Dana Example")
    alice_id = _person_of(pg_conn, ALICE_PHONE)[0]
    rename_person(pg_conn, person_id=alice_id, display_name="Alice Example")

    result = apply_overrides(
        pg_conn,
        overrides=[
            IdentityOverride(
                value=ALICE_EMAIL, kind="email", name="Alice Example", was=("Dana Example",)
            )
        ],
        default_region="US",
    )

    assert result.summary == "applied=1 already=0 unmatched=0 conflicts=0"
    assert result.outcomes[0].detail.startswith("assigned email")
    assert _person_of(pg_conn, ALICE_EMAIL)[0] == alice_id
    assert _person_of(pg_conn, BOB_PHONE) == (
        dana_id,
        "Dana Example",
        False,
    )  # Dana keeps her phone
    assert _sender_of(pg_conn, "msg-in-3") == alice_id  # the email's messages re-attributed too


def test_apply_never_guesses_between_two_persons_with_the_decided_name(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE, ALICE_EMAIL]
    )
    # Two different people who share a first name — must never be fused.
    rename_person(
        pg_conn, person_id=_person_of(pg_conn, ALICE_PHONE)[0], display_name="Alex Example"
    )
    rename_person(pg_conn, person_id=_person_of(pg_conn, BOB_PHONE)[0], display_name="Alex Example")
    decision = IdentityOverride(value=ALICE_EMAIL, kind="email", name="Alex Example")
    before = _state_snapshot(pg_conn)

    for force in (False, True):
        result = apply_overrides(pg_conn, overrides=[decision], default_region="US", force=force)
        assert result.summary == "applied=0 already=0 unmatched=0 conflicts=1", force
        assert "2 persons are already named 'Alex Example'" in result.outcomes[0].detail
        assert _state_snapshot(pg_conn) == before

    # ...unless the file itself says which one: the person holding another
    # handle the file lists under the same name is the file's own target.
    result = apply_overrides(
        pg_conn,
        overrides=[IdentityOverride(value=BOB_PHONE, kind="phone", name="Alex Example"), decision],
        default_region="US",
    )
    assert result.summary == "applied=1 already=1 unmatched=0 conflicts=0"
    assert _person_of(pg_conn, ALICE_EMAIL)[0] == _person_of(pg_conn, BOB_PHONE)[0]


def test_apply_refuses_to_touch_the_owner_even_when_forced(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE])
    with pg_conn.cursor() as cur:
        cur.execute("SELECT person_id FROM person WHERE is_owner")
        row = cur.fetchone()
    assert row is not None
    owner_id = int(row[0])
    assign_handle(pg_conn, normalized_value=ALICE_PHONE, kind="phone", person_id=owner_id)
    decision = IdentityOverride(value=ALICE_PHONE, kind="phone", name="Not The Owner")

    for force in (False, True):
        result = apply_overrides(pg_conn, overrides=[decision], default_region="US", force=force)
        assert result.conflicts == 1, force
        assert "owner" in result.outcomes[0].detail
    assert _person_of(pg_conn, ALICE_PHONE)[0] == owner_id
    assert _count(pg_conn, "SELECT count(*) FROM person WHERE display_name = 'Not The Owner'") == 0


def test_export_writes_named_handles_and_carries_a_previous_file_forward(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE, ALICE_EMAIL]
    )
    alice_id = _person_of(pg_conn, ALICE_PHONE)[0]
    merge_persons(
        pg_conn, keep_person_id=alice_id, absorb_person_id=_person_of(pg_conn, ALICE_EMAIL)[0]
    )
    rename_person(pg_conn, person_id=alice_id, display_name="Alice Example")
    # Bob stays an unreviewed single-handle stub: not a decision, not exported.
    previous = IdentityOverridesFile(
        overrides=(
            IdentityOverride(
                value=ALICE_PHONE,
                kind="phone",
                name="Alicia Example",
                was=("Contacts Conflict",),
                why="kept",
            ),
            IdentityOverride(
                value="+14155552699", kind="phone", name="Gone Example", why="left the index"
            ),
        ),
        note="the owner's note",
        decided="2026-08-15",
    )

    exported = export_overrides(
        pg_conn, default_region="US", previous=previous, today=dt.date(2026, 9, 15)
    )

    assert (exported.exported, exported.carried_forward) == (2, 1)
    assert exported.file.note == "the owner's note"
    assert exported.file.decided == "2026-09-15"
    assert exported.file.overrides == (
        IdentityOverride(value=ALICE_EMAIL, kind="email", name="Alice Example"),
        IdentityOverride(
            value=ALICE_PHONE,
            kind="phone",
            name="Alice Example",
            was=("Alicia Example", "Contacts Conflict"),  # renamed since: old name recorded
            why="kept",
        ),
        IdentityOverride(
            value="+14155552699", kind="phone", name="Gone Example", why="left the index"
        ),
    )

    # What export wrote, apply accepts as already done — the round trip is a no-op.
    replay = apply_overrides(pg_conn, overrides=exported.file.overrides, default_region="US")
    assert replay.summary == "applied=0 already=2 unmatched=1 conflicts=0"


def test_export_without_a_previous_file_uses_defaults(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE])
    rename_person(pg_conn, person_id=_person_of(pg_conn, BOB_PHONE)[0], display_name="Bob Example")

    exported = export_overrides(pg_conn, default_region="US")

    assert exported.file.note == DEFAULT_NOTE
    assert exported.file.overrides == (
        IdentityOverride(value=BOB_PHONE, kind="phone", name="Bob Example"),
    )
    assert exported.carried_forward == 0


def test_export_then_rebuild_then_apply_restores_every_name(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """The scenario the feature exists for: a rebuild recreates the stubs
    and drops every curated name; replaying the exported file puts the
    person table back, including the merge, with the invariant clean."""
    _seed_and_resolve(
        pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE, BOB_PHONE, ALICE_EMAIL]
    )
    alice_id = _person_of(pg_conn, ALICE_PHONE)[0]
    merge_persons(
        pg_conn, keep_person_id=alice_id, absorb_person_id=_person_of(pg_conn, ALICE_EMAIL)[0]
    )
    rename_person(pg_conn, person_id=alice_id, display_name="Alice Example")
    rename_person(pg_conn, person_id=_person_of(pg_conn, BOB_PHONE)[0], display_name="Bob Example")
    exported = export_overrides(pg_conn, default_region="US")
    assert exported.exported == 3

    # A rebuild: drop everything S3 derived, then resolve from scratch.
    with pg_conn.transaction(), pg_conn.cursor() as cur:
        cur.execute("DELETE FROM source_handle_resolution")
        cur.execute("DELETE FROM chat_participant")
        cur.execute("UPDATE message SET sender_person_id = NULL")
        cur.execute("UPDATE tapback SET sender_person_id = NULL")
        cur.execute("DELETE FROM allowlist_person")
        cur.execute("DELETE FROM handle")
        cur.execute("DELETE FROM person")
    rebuilt = run_identity(
        conn=pg_conn, config=_identity_config(config_dict_factory, contacts_import=False)
    )
    assert rebuilt.persons_created == 3  # three stubs (the owner is re-created before the count)
    assert _count(pg_conn, "SELECT count(*) FROM person") == 4
    assert _person_of(pg_conn, ALICE_PHONE)[1] == ALICE_PHONE  # the names are gone

    result = apply_overrides(pg_conn, overrides=exported.file.overrides, default_region="US")

    assert result.summary == "applied=3 already=0 unmatched=0 conflicts=0"
    assert result.invariant.ok is True
    restored_alice = _person_of(pg_conn, ALICE_PHONE)
    assert restored_alice[1:] == ("Alice Example", False)
    assert _person_of(pg_conn, ALICE_EMAIL)[0] == restored_alice[0]
    assert _person_of(pg_conn, BOB_PHONE)[1:] == ("Bob Example", False)
    assert _count(pg_conn, "SELECT count(*) FROM person WHERE NOT is_owner") == 2
    assert _count(pg_conn, "SELECT count(*) FROM person WHERE needs_review") == 0
    # And a second pass is the no-op it must be.
    assert apply_overrides(
        pg_conn, overrides=exported.file.overrides, default_region="US"
    ).summary == ("applied=0 already=3 unmatched=0 conflicts=0")


def test_export_moves_a_legacy_prior_to_the_clean_handle_only_once_the_legacy_row_is_gone(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory
) -> None:
    """Before the filter-tag fix a tagged number and its clean form were two
    handles on two persons. A decision recorded against the tagged row must
    not be copied onto the clean row while both exist — they may disagree —
    but once a rebuild leaves only the clean row, the decision follows it."""
    tagged = f"{ALICE_PHONE}(filtered)"
    _seed_and_resolve(pg_conn, tmp_path, config_dict_factory, handles=[ALICE_PHONE])
    with pg_conn.cursor() as cur:  # the legacy row, as the pre-fix index held it
        cur.execute(
            "INSERT INTO person (display_name, short_name) VALUES (%s, %s) RETURNING person_id",
            (tagged, "legacy-stub"),
        )
        row = cur.fetchone()
        assert row is not None
        legacy_id = int(row[0])
        cur.execute(
            "INSERT INTO handle (person_id, kind, normalized_value) VALUES (%s, 'unknown', %s)",
            (legacy_id, tagged),
        )
    pg_conn.commit()
    rename_person(pg_conn, person_id=legacy_id, display_name="Alice Example")
    rename_person(
        pg_conn, person_id=_person_of(pg_conn, ALICE_PHONE)[0], display_name="Carol Example"
    )
    previous = IdentityOverridesFile(
        overrides=(
            IdentityOverride(
                value=tagged,
                kind="unknown",
                name="Alice Example",
                was=("Contacts Conflict",),
                why="kept",
            ),
        )
    )

    both = export_overrides(pg_conn, default_region="US", previous=previous)
    assert (both.exported, both.carried_forward) == (2, 0)
    assert both.file.overrides == (
        IdentityOverride(
            value=tagged,
            kind="unknown",
            name="Alice Example",
            was=("Contacts Conflict",),
            why="kept",
        ),
        IdentityOverride(
            value=ALICE_PHONE, kind="phone", name="Carol Example"
        ),  # untouched by the prior
    )

    with pg_conn.transaction(), pg_conn.cursor() as cur:  # the rebuild collapses the legacy row
        cur.execute("DELETE FROM handle WHERE person_id = %s", (legacy_id,))
        cur.execute("DELETE FROM person WHERE person_id = %s", (legacy_id,))

    clean_only = export_overrides(pg_conn, default_region="US", previous=previous)
    assert (clean_only.exported, clean_only.carried_forward) == (1, 0)
    assert clean_only.file.overrides == (
        IdentityOverride(
            value=ALICE_PHONE,
            kind="phone",
            name="Carol Example",
            was=("Alice Example", "Contacts Conflict"),  # the prior followed the clean handle
            why="kept",
        ),
    )
