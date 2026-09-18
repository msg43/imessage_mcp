"""S2: what happens when `imsg-dump` returns no record for a message
the SQL half did return (2026-09-17).

INVARIANT UNDER TEST: a message must never be committed with a body
that the corpus has no way to fill in later. Extraction either gets the
body, or leaves the message re-targetable by failing the run — it never
seals a NULL.

The merge policy added the same day (`text_original` is
`Merge.PRESENT`) already stops a silent shim from blanking a body an
earlier run stored; `test_extract_poorer_source_never_overwrites_
integration.py` is where that is asserted. This file is about the case
that policy cannot reach: a message being extracted for the FIRST time
has no stored value to protect, so it is inserted bodiless — and the
first test below shows that nothing ever comes back for it, because the
watermark has moved past it and only an edit or a retraction re-selects
an older row.

Fictional personas only (D5): Alice Example, and a documentation-range
phone number.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
from imsg.db.migrations import PostgresMigrationRunner
from imsg.errors import ExtractionError
from imsg.stages.extract import (
    MIN_MISSING_BODIES_TO_FAIL,
    ExtractResult,
    run_extract,
)
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_extract_missing_bodies_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

SOURCE = "studio"
ALICE_HANDLE = "+15550000001"
CHAT_GUID = "chat-trip"
_BASE = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)

BODIES = [
    "are we still on for saturday",
    "yes, ten sharp",
    "bring boots",
    "the trail is muddy",
    "i have the map",
    "parking is five dollars",
    "see you at the gate",
    "running ten minutes late",
    "no rush",
    "just arrived",
]
GUIDS = [f"msg-{i:02d}" for i in range(1, len(BODIES) + 1)]


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

    conn = psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
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


# --- the corpus ------------------------------------------------------------


def _snapshot(path: Path, *, count: int = len(BODIES)) -> Path:
    builder = ChatDbBuilder()
    builder.add_chat(
        FixtureChat(guid=CHAT_GUID, rowid=1, style=43, display_name="Trip", service_name="iMessage")
    )
    builder.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    builder.link_participant(CHAT_GUID, ALICE_HANDLE)
    for i in range(count):
        builder.add_message(
            FixtureMessage(
                guid=GUIDS[i], chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
                rowid=i + 1, date=_BASE + timedelta(minutes=i),
            )
        )
    return builder.build(path)


def _dump_message(*, guid: str, rowid: int, body_text: str | None) -> ImsgDumpMessage:
    return ImsgDumpMessage(
        rowid=rowid, guid=guid, chat_guid=None, handle=None, is_from_me=False,
        date=None, date_edited=None, date_retracted=None, service="iMessage",
        body_text=body_text, edit_history=(), is_unsent=False,
        tapback=None, attachment_rowids=(), reply_to_guid=None,
    )


def _dump(*, omit: Collection[str] = (), count: int = len(BODIES)) -> ImsgDumpRun:
    """The shim's output, minus the guids it returned no record for at
    all — not a record with a null body, which is a different case the
    corpus can tell apart."""
    return ImsgDumpRun(
        messages=tuple(
            _dump_message(guid=GUIDS[i], rowid=i + 1, body_text=BODIES[i])
            for i in range(count)
            if GUIDS[i] not in omit
        ),
        stderr_lines=(),
    )


def _extract(
    conn: psycopg.Connection,
    tmp_path: Path,
    snapshot_path: Path,
    dump: ImsgDumpRun,
    *,
    source_name: str = SOURCE,
    max_missing_body_fraction: float = 0.10,
) -> ExtractResult:
    binary = tmp_path / "imsg-dump"
    binary.write_text("")
    return run_extract(
        conn=conn,
        source_name=source_name,
        snapshot_path=snapshot_path,
        imsg_dump_binary=binary,
        run_imsg_dump_fn=lambda _b, _s, _r: dump,
        max_missing_body_fraction=max_missing_body_fraction,
    )


# --- probes ----------------------------------------------------------------


def _body(conn: psycopg.Connection, guid: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = %s", (guid,))
        row = cur.fetchone()
    assert row is not None, f"no message row for {guid}"
    return row[0]


def _message_count(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message")
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _watermark(conn: psycopg.Connection) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM sync_state WHERE key = %s", (f"watermark.rowid.{SOURCE}",))
        row = cur.fetchone()
    return None if row is None else str(row[0])


def _run_statuses(conn: psycopg.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM extraction_run ORDER BY run_id")
        return [str(r[0]) for r in cur.fetchall()]


# --- why a missing record cannot be left to degrade ------------------------


def test_a_body_missed_on_first_extraction_is_never_revisited(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The reason the run has to fail at all, asserted rather than
    assumed. `Merge.PRESENT` protects a body an earlier run stored, but
    a message's FIRST extraction has nothing stored to protect: it is
    inserted with NULL, the watermark advances to the snapshot's max
    ROWID regardless, and `fetch_target_messages` re-selects an older
    row only when its `date_edited`/`date_retracted` moved. A merely
    bodiless message therefore falls out of scope permanently — a later
    run whose shim decodes it perfectly does not look at it.

    Note the second run reports `bodies_missing=0`: the counter goes
    quiet on exactly the run that could have healed the gap, which is
    why an operator watching that number would never see this.

    The bound is disabled here (`1.0`) so the degrade path is
    observable at all; every other test runs the real policy.
    """
    snapshot = _snapshot(tmp_path / "snap.db", count=2)

    first = _extract(
        pg_conn, tmp_path, snapshot, _dump(omit={GUIDS[1]}, count=2),
        max_missing_body_fraction=1.0,
    )
    assert first.bodies_missing == 1
    assert first.watermark_before == 0
    assert first.watermark_after == 2
    assert _body(pg_conn, GUIDS[0]) == BODIES[0]
    assert _body(pg_conn, GUIDS[1]) is None

    # Same snapshot, and this time the shim decodes everything.
    second = _extract(
        pg_conn, tmp_path, snapshot, _dump(count=2), max_missing_body_fraction=1.0
    )
    assert second.bodies_missing == 0, "nothing is missing because nothing was targeted"
    assert second.messages_upserted == 0, "the watermark has already moved past both rows"
    assert _body(pg_conn, GUIDS[1]) is None, (
        "the gap is permanent: no later run re-targets a merely bodiless message"
    )


# --- the policy ------------------------------------------------------------


def test_a_run_missing_too_many_bodies_commits_nothing(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """A shim that crashed early, was given the wrong arguments, or had
    its output truncated returns records for only part of what was
    asked. Committing that writes permanently bodiless rows (the test
    above), so the whole run is refused instead — and because every
    write in `_do_extract` shares one transaction, refusing it leaves
    no message rows and an unmoved watermark."""
    snapshot = _snapshot(tmp_path / "snap.db")
    omitted = set(GUIDS[:6])

    with pytest.raises(ExtractionError, match="6 of 10 targeted messages"):
        _extract(pg_conn, tmp_path, snapshot, _dump(omit=omitted))

    assert _message_count(pg_conn) == 0, "a refused run must write no message rows at all"
    assert _watermark(pg_conn) is None, (
        "the watermark must not move, or the next run cannot retry the same range"
    )
    assert _run_statuses(pg_conn) == ["failed"], (
        "the attempt is still recorded — rolled back, not invisible"
    )


def test_the_refused_range_is_healed_by_the_next_run(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The payoff, and the whole reason failing beats degrading: the
    rows the refused run would have sealed are still in scope, so a
    later run with a working shim lands every body. Retrying a whole
    run is affordable because the 2026-09-17 state-idempotence fix
    stopped unchanged rows from being rewritten."""
    snapshot = _snapshot(tmp_path / "snap.db")

    with pytest.raises(ExtractionError):
        _extract(pg_conn, tmp_path, snapshot, _dump(omit=set(GUIDS[:6])))

    healed = _extract(pg_conn, tmp_path, snapshot, _dump())

    assert healed.bodies_missing == 0
    assert healed.message_upserts.inserted == len(BODIES)
    for guid, body in zip(GUIDS, BODIES, strict=True):
        assert _body(pg_conn, guid) == body
    assert _run_statuses(pg_conn) == ["failed", "ok"]


def test_one_undecodable_body_does_not_fail_the_run(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """The other side of the bound. A typedstream blob that is simply
    gone fails to decode on every future run, so a rule that stopped on
    any missing record would stall extraction for that source forever.
    Below the floor the row is warned about and the run proceeds."""
    snapshot = _snapshot(tmp_path / "snap.db")

    result = _extract(pg_conn, tmp_path, snapshot, _dump(omit={GUIDS[3]}))

    assert result.bodies_missing == 1
    assert result.message_upserts.inserted == len(BODIES)
    assert _body(pg_conn, GUIDS[3]) is None
    assert _body(pg_conn, GUIDS[4]) == BODIES[4]
    assert _run_statuses(pg_conn) == ["ok"]


def test_the_floor_outranks_the_fraction_on_a_small_run(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """A run targeting three messages that decodes one of them is at
    67% — far over the fraction, and still under the floor. Incremental
    runs are routinely this small (a measured nightly on the real
    corpus targeted 972 of 673,113 rows), so the fraction alone would
    turn an ordinary quiet night into a failed sync."""
    snapshot = _snapshot(tmp_path / "snap.db", count=3)
    assert MIN_MISSING_BODIES_TO_FAIL > 2, "this test only means something below the floor"

    result = _extract(pg_conn, tmp_path, snapshot, _dump(omit=set(GUIDS[:2]), count=3))

    assert result.bodies_missing == 2
    assert result.message_upserts.inserted == 3
    assert _run_statuses(pg_conn) == ["ok"]


def test_a_missing_dump_record_does_not_clear_a_stored_retraction(
    pg_conn: psycopg.Connection, tmp_path: Path
) -> None:
    """`is_unsent` and `is_edited` fall back to the SQL-derived values
    when there is no dump record, and those SQL columns are explicitly
    not authoritative — the module docstring's "Correction" note
    records that the crate derives unsent status from the typedstream,
    with no retract-shaped column to read. So the fallback for a
    witness that never received the retraction is `False`: exactly what
    a message that was never retracted reports.

    `Merge.POSITIVE` is what stops that `False` from landing, and this
    asserts it on the missing-record path specifically — where the
    value comes from `date_retracted` rather than from the shim, which
    is a different branch of `_upsert_message` than the one
    `test_a_source_that_never_saw_the_retraction_keeps_is_unsent`
    covers. A second source is used because each source carries its own
    watermark, which is what puts an already-extracted message back in
    scope.
    """
    rich = ChatDbBuilder()
    rich.add_chat(
        FixtureChat(guid=CHAT_GUID, rowid=1, style=43, display_name="Trip", service_name="iMessage")
    )
    rich.add_handle(FixtureHandle(raw_value=ALICE_HANDLE, rowid=1))
    rich.link_participant(CHAT_GUID, ALICE_HANDLE)
    rich.add_message(
        FixtureMessage(
            guid=GUIDS[0], chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=1, date=_BASE, date_retracted=_BASE + timedelta(minutes=1),
        )
    )
    rich.add_message(
        FixtureMessage(
            guid=GUIDS[1], chat_guid=CHAT_GUID, handle_raw_value=ALICE_HANDLE,
            rowid=2, date=_BASE + timedelta(minutes=2),
        )
    )
    rich_path = rich.build(tmp_path / "rich.db")

    retracted = ImsgDumpMessage(
        rowid=1, guid=GUIDS[0], chat_guid=None, handle=None, is_from_me=False,
        date=None, date_edited=None, date_retracted=None, service="iMessage",
        body_text=BODIES[0], edit_history=(), is_unsent=True,
        tapback=None, attachment_rowids=(), reply_to_guid=None,
    )
    _extract(
        pg_conn, tmp_path, rich_path,
        ImsgDumpRun(
            messages=(retracted, _dump_message(guid=GUIDS[1], rowid=2, body_text=BODIES[1])),
            stderr_lines=(),
        ),
    )
    with pg_conn.cursor() as cur:
        cur.execute("SELECT is_unsent FROM message WHERE source_guid = %s", (GUIDS[0],))
        row = cur.fetchone()
    assert row is not None and row[0] is True, "the baseline must start retracted"

    # A second witness: same two messages, no retraction in its SQL, and a
    # shim that returns no record for the retracted one at all.
    poor_path = _snapshot(tmp_path / "poor.db", count=2)
    result = _extract(
        pg_conn, tmp_path, poor_path, _dump(omit={GUIDS[0]}, count=2), source_name="mini"
    )

    assert result.bodies_missing == 1
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT is_unsent, text_original FROM message WHERE source_guid = %s", (GUIDS[0],)
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] is True, "a witness with no dump record must not un-retract the message"
    assert row[1] == BODIES[0], "and must not blank the body either"
