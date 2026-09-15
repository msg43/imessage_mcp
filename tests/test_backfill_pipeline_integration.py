"""Postgres integration tests for S5a (SPEC §8 S5a) — skips cleanly
when no scratch Postgres is reachable, same pattern as
`tests/test_migrations_integration.py`."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from imsg.backfill.classify import NO_SOURCE_PATH_ERROR
from imsg.backfill.pipeline import MAX_MATERIALIZATION_ATTEMPTS, run_backfill
from imsg.backfill.reconcile import build_reconciliation_report
from imsg.backfill.throttle import RateThrottle
from imsg.db.migrations import PostgresMigrationRunner

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_backfill_test"

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
def scratch_db() -> Iterator[psycopg.Connection]:
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


def _insert_attachment(
    conn: psycopg.Connection,
    *,
    source_path: str | None,
    state: str = "dataless",
    attempts: int = 0,
    next_attempt_at: datetime | None = None,
    last_error: str | None = None,
) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, source_path, state,
                                    materialization_attempts, materialization_next_attempt_at,
                                    materialization_last_error)
            VALUES (%s, %s, %s, %s, %s, COALESCE(%s, now()), %s) RETURNING attachment_id
            """,
            (guid, f"key-{guid}", source_path, state, attempts, next_attempt_at, last_error),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _fetch_state(conn: psycopg.Connection, attachment_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT state FROM attachment WHERE attachment_id = %s", (attachment_id,))
        row = cur.fetchone()
        assert row is not None
        return str(row[0])


def _fetch_row(conn: psycopg.Connection, attachment_id: int) -> tuple[str, int, datetime, str | None]:
    """(state, attempts, next_attempt_at, last_error)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state, materialization_attempts, materialization_next_attempt_at, "
            "materialization_last_error FROM attachment WHERE attachment_id = %s",
            (attachment_id,),
        )
        row = cur.fetchone()
        assert row is not None
        return str(row[0]), int(row[1]), row[2], row[3]


def _state_counts(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT state, count(*) FROM attachment GROUP BY state")
        return {str(state): int(count) for state, count in cur.fetchall()}


def _future() -> datetime:
    return datetime.now(UTC) + timedelta(hours=6)


@pytest.fixture
def no_sleep_throttle() -> RateThrottle:
    return RateThrottle(10_000, sleep_fn=lambda _: None)


def test_run_backfill_materializes_real_files(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    f1 = attachments_root / "a" / "photo.jpg"
    f1.parent.mkdir(parents=True)
    f1.write_bytes(b"fake jpeg bytes")
    att_id = _insert_attachment(scratch_db, source_path=str(f1))

    report = run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=no_sleep_throttle,
    )

    assert report.materialized == 1
    assert report.errored == 0
    assert _fetch_state(scratch_db, att_id) == "materialized"

    with scratch_db.cursor() as cur:
        cur.execute("SELECT sha256, byte_size, cache_path FROM attachment WHERE attachment_id = %s", (att_id,))
        sha, byte_size, cache_path = cur.fetchone()  # type: ignore[misc]
    assert byte_size == len(b"fake jpeg bytes")
    assert Path(cache_path).is_file()
    assert Path(cache_path).read_bytes() == b"fake jpeg bytes"
    assert cache_path.endswith(f"{sha[:2]}/{sha}")


def test_run_backfill_dry_run_writes_nothing(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    f1 = attachments_root / "a" / "photo.jpg"
    f1.parent.mkdir(parents=True)
    f1.write_bytes(b"fake jpeg bytes")
    att_id = _insert_attachment(scratch_db, source_path=str(f1))

    report = run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=no_sleep_throttle,
        dry_run=True,
    )

    assert report.dry_run is True
    assert report.considered == 1
    assert report.detected_already_local == 1
    # Outcome can't be known without attempting it — always 0 in dry-run.
    assert report.materialized == 0
    assert report.errored == 0
    assert report.marked_missing == 0

    # Nothing was actually written: state untouched, no file copied.
    assert _fetch_state(scratch_db, att_id) == "dataless"
    assert not (data_root / "attachments").exists()

    # A real run afterward materializes it normally.
    real_report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert real_report.materialized == 1
    assert _fetch_state(scratch_db, att_id) == "materialized"


def test_trial_gate_caps_first_run_at_default_limit(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    for i in range(15):
        f = attachments_root / f"file{i}.bin"
        f.write_bytes(f"content-{i}".encode())
        _insert_attachment(scratch_db, source_path=str(f))

    report = run_backfill(scratch_db, data_root, attachments_root, throttle=no_sleep_throttle)

    assert report.trial_gate_capped is True
    assert report.materialized == 12  # DEFAULT_TRIAL_LIMIT
    assert report.considered == 12

    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM attachment WHERE state = 'materialized'")
        (materialized_count,) = cur.fetchone()  # type: ignore[misc]
    assert materialized_count == 12


def test_yes_full_run_bypasses_trial_gate(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    for i in range(15):
        f = attachments_root / f"file{i}.bin"
        f.write_bytes(f"content-{i}".encode())
        _insert_attachment(scratch_db, source_path=str(f))

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )

    assert report.trial_gate_capped is False
    assert report.materialized == 15


def test_second_run_skips_already_materialized(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    f = attachments_root / "once.bin"
    f.write_bytes(b"data")
    _insert_attachment(scratch_db, source_path=str(f))

    run_backfill(scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle)
    second_report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )

    assert second_report.considered == 0
    assert second_report.materialized == 0


# --------------------------------------------------------------------------
# deterministic failures -> `unsupported` (migration 0003; imsg.backfill.classify)
# --------------------------------------------------------------------------


def test_source_path_escaping_attachments_root_is_refused_as_unsupported(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"should never be read")
    data_root = tmp_path / "data_root"

    att_id = _insert_attachment(scratch_db, source_path=str(outside))

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )

    # Still refused — the containment check is load-bearing — but the
    # outcome is terminal, not "errored, retry later".
    assert report.materialized == 0
    assert report.errored == 0
    assert report.marked_unsupported == 1
    state, attempts, next_attempt_at, err = _fetch_row(scratch_db, att_id)
    assert state == "unsupported"
    assert attempts == 0  # not on the retry ladder
    assert next_attempt_at <= datetime.now(UTC)  # no backoff scheduled
    assert err is not None
    assert err.startswith("unsupported[")
    assert "attachments root" in err
    assert not (data_root / "attachments").exists()  # never read, never copied

    # And it is never considered again.
    again = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert again.considered == 0


def test_out_of_root_reason_classes_are_named(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    # None of these needs to exist: classification is by path shape, and
    # nothing outside the root is ever stat'd or opened.
    temp_dir_id = _insert_attachment(
        scratch_db, source_path="/private/var/folders/ab/cdef0123/T/com.example.app/blob.bin"
    )
    sticker_id = _insert_attachment(
        scratch_db,
        source_path=str(tmp_path / "home" / "Library" / "Messages" / "StickerCache" / "s.heic"),
    )
    other_id = _insert_attachment(scratch_db, source_path="/nonexistent-volume/elsewhere/x.pdf")

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert report.marked_unsupported == 3
    assert report.errored == 0

    for att_id, expected_class in (
        (temp_dir_id, "temp-directory-path"),
        (sticker_id, "sticker-cache-path"),
        (other_id, "out-of-root-path"),
    ):
        state, _attempts, _next, err = _fetch_row(scratch_db, att_id)
        assert state == "unsupported"
        assert err is not None and err.startswith(f"unsupported[{expected_class}]: ")


def test_directory_and_overlong_name_are_unsupported_not_error(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    a_dir = attachments_root / "not-a-file"
    a_dir.mkdir()
    dir_id = _insert_attachment(scratch_db, source_path=str(a_dir))
    # 300 characters exceeds NAME_MAX (255) on every filesystem this runs
    # on; the path is inside the root, so it IS attempted — and the
    # open() fails with an errno that no retry can change.
    long_id = _insert_attachment(scratch_db, source_path=str(attachments_root / ("x" * 300 + ".bin")))

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert report.considered == 2
    assert report.marked_unsupported == 2
    assert report.errored == 0
    assert report.marked_missing == 0

    state, _attempts, next_attempt_at, err = _fetch_row(scratch_db, dir_id)
    assert state == "unsupported"
    assert err is not None and err.startswith("unsupported[is-a-directory]: ")
    assert next_attempt_at <= datetime.now(UTC)

    state, _attempts, _next, err = _fetch_row(scratch_db, long_id)
    assert state == "unsupported"
    assert err is not None and err.startswith("unsupported[file-name-too-long]: ")
    # The recorded detail is the read that failed, not a cleanup error
    # on the cache's own temp file (whose name must stay short).
    assert ".partial" not in err
    tmp_dir = data_root / "attachments" / ".tmp"
    assert not tmp_dir.exists() or not any(tmp_dir.iterdir())  # no stray partial left behind


def test_missing_file_backs_off_then_becomes_missing_after_three_attempts(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    ghost = attachments_root / "ghost.bin"  # never actually created
    att_id = _insert_attachment(scratch_db, source_path=str(ghost))

    for attempt in range(1, MAX_MATERIALIZATION_ATTEMPTS + 1):
        report = run_backfill(
            scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
        )
        assert report.considered == 1
        if attempt < MAX_MATERIALIZATION_ATTEMPTS:
            assert report.errored == 1
            # A second run at the same instant must NOT reconsider it — the
            # failure just backed next_attempt_at off into the future.
            immediate_rerun = run_backfill(
                scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
            )
            assert immediate_rerun.considered == 0
            # Force the backoff clock forward so the *next* real attempt is eligible.
            with scratch_db.cursor() as cur:
                cur.execute(
                    "UPDATE attachment SET materialization_next_attempt_at = now() "
                    "WHERE attachment_id = %s",
                    (att_id,),
                )
        else:
            assert report.marked_missing == 1

    assert _fetch_state(scratch_db, att_id) == "missing"
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT materialization_attempts FROM attachment WHERE attachment_id = %s", (att_id,)
        )
        (attempts,) = cur.fetchone()  # type: ignore[misc]
    assert attempts == MAX_MATERIALIZATION_ATTEMPTS


# --------------------------------------------------------------------------
# reclassification pre-passes: an older index heals on its next run
# --------------------------------------------------------------------------


def test_rows_without_a_source_path_are_reclassified_missing(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    # Rows an S2 built before it assigned `missing` at insert time would leave behind.
    dataless_id = _insert_attachment(scratch_db, source_path=None)
    interrupted_id = _insert_attachment(scratch_db, source_path=None, state="materializing")
    # Already right: must be counted as nothing, touched as nothing.
    already_missing_id = _insert_attachment(
        scratch_db, source_path=None, state="missing", last_error="kept as is"
    )

    dry = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True,
        throttle=no_sleep_throttle, dry_run=True,
    )
    assert dry.reclassified_missing_no_source == 2
    assert dry.considered == 0
    assert _fetch_state(scratch_db, dataless_id) == "dataless"  # dry run wrote nothing

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert report.reclassified_missing_no_source == 2
    assert report.considered == 0
    for att_id in (dataless_id, interrupted_id):
        state, _attempts, next_attempt_at, err = _fetch_row(scratch_db, att_id)
        assert state == "missing"
        assert err == NO_SOURCE_PATH_ERROR
        assert next_attempt_at <= datetime.now(UTC)
    assert _fetch_row(scratch_db, already_missing_id)[3] == "kept as is"

    # Idempotent: the next run has nothing left to heal.
    assert run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    ).reclassified_missing_no_source == 0


def test_failed_rows_with_out_of_root_paths_are_reclassified_without_retry(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    # What a pre-0003 pipeline left behind: deterministic refusals parked
    # on the retry ladder, backed off into the future or given up on.
    backed_off_id = _insert_attachment(
        scratch_db,
        source_path=str(tmp_path / "elsewhere" / "blob.bin"),
        state="error", attempts=2, next_attempt_at=_future(), last_error="old refusal text",
    )
    given_up_id = _insert_attachment(
        scratch_db,
        source_path=str(tmp_path / "home" / "Library" / "Messages" / "StickerCache" / "s.heic"),
        state="missing", attempts=3, next_attempt_at=_future(), last_error="old refusal text",
    )
    # An in-root failed row is NOT touched by this pass: only an actual
    # retry can say what happens to it.
    in_root_error_id = _insert_attachment(
        scratch_db,
        source_path=str(attachments_root / "ghost.bin"),
        state="error", attempts=1, next_attempt_at=_future(), last_error="[Errno 2] gone",
    )

    dry = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True,
        throttle=no_sleep_throttle, dry_run=True,
    )
    assert dry.reclassified_unsupported == 2
    assert _fetch_state(scratch_db, backed_off_id) == "error"  # dry run wrote nothing

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert report.reclassified_unsupported == 2
    assert report.considered == 0  # the in-root error row is still backed off
    assert report.marked_unsupported == 0

    state, attempts, next_attempt_at, err = _fetch_row(scratch_db, backed_off_id)
    assert state == "unsupported"
    assert attempts == 2  # untouched: not a retry
    assert next_attempt_at <= datetime.now(UTC)  # no longer "retrying at <future>"
    assert err is not None and err.startswith("unsupported[temp-directory-path]: ")

    state, _attempts, _next, err = _fetch_row(scratch_db, given_up_id)
    assert state == "unsupported"
    assert err is not None and err.startswith("unsupported[sticker-cache-path]: ")

    assert _fetch_row(scratch_db, in_root_error_id) == (
        "error", 1, _fetch_row(scratch_db, in_root_error_id)[2], "[Errno 2] gone"
    )
    assert not (data_root / "attachments").exists()


# --------------------------------------------------------------------------
# retry_failed (`imsg backfill-attachments --retry-failed`)
# --------------------------------------------------------------------------


def test_retry_failed_puts_error_and_missing_rows_back_on_the_ladder(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    now_ok = attachments_root / "now-ok.bin"
    now_ok.write_bytes(b"arrived since the last run")
    backed_off_id = _insert_attachment(
        scratch_db, source_path=str(now_ok), state="error", attempts=2,
        next_attempt_at=_future(), last_error="[Errno 5] transient",
    )
    also_ok = attachments_root / "also-ok.bin"
    also_ok.write_bytes(b"gave up too early")
    given_up_id = _insert_attachment(
        scratch_db, source_path=str(also_ok), state="missing", attempts=3,
        next_attempt_at=_future(), last_error="[Errno 2] never arrived",
    )
    # Never reset: the reason is a property of the row.
    unsupported_id = _insert_attachment(
        scratch_db, source_path=str(attachments_root / "a-dir"), state="unsupported",
        last_error="unsupported[is-a-directory]: x",
    )
    # Never reset: nothing to read.
    no_path_id = _insert_attachment(scratch_db, source_path=None, state="missing")

    # Without the flag, both failed rows stay parked.
    parked = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert parked.considered == 0
    assert parked.retry_reset == 0

    # Dry run: reports what the reset would touch and what would then be
    # considered — and writes neither.
    dry = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True,
        throttle=no_sleep_throttle, dry_run=True, retry_failed=True,
    )
    assert dry.retry_reset == 2
    assert dry.considered == 2
    assert dry.detected_already_local == 2
    assert _fetch_row(scratch_db, backed_off_id)[:2] == ("error", 2)
    assert _fetch_row(scratch_db, given_up_id)[:2] == ("missing", 3)

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True,
        throttle=no_sleep_throttle, retry_failed=True,
    )
    assert report.retry_reset == 2
    assert report.considered == 2
    assert report.materialized == 2
    for att_id in (backed_off_id, given_up_id):
        state, attempts, _next, err = _fetch_row(scratch_db, att_id)
        assert state == "materialized"
        assert attempts == 0
        assert err is None
    assert _fetch_state(scratch_db, unsupported_id) == "unsupported"
    assert _fetch_state(scratch_db, no_path_id) == "missing"


# --------------------------------------------------------------------------
# the printed report agrees with the database, state for state
# --------------------------------------------------------------------------


def _mixed_outcome_corpus(
    conn: psycopg.Connection, attachments_root: Path, tmp_path: Path
) -> dict[str, int]:
    """One row per outcome a single run can produce; returns name -> id."""
    ids: dict[str, int] = {}
    ok1 = attachments_root / "ok1.bin"
    ok1.write_bytes(b"one")
    ok2 = attachments_root / "ok2.bin"
    ok2.write_bytes(b"two")
    ids["materialized_1"] = _insert_attachment(conn, source_path=str(ok1))
    ids["materialized_2"] = _insert_attachment(conn, source_path=str(ok2))
    # unsupported, decided in the loop: refused before reading / EISDIR on read
    ids["unsupported_refused"] = _insert_attachment(conn, source_path=str(tmp_path / "outside.bin"))
    a_dir = attachments_root / "a-dir"
    a_dir.mkdir()
    ids["unsupported_directory"] = _insert_attachment(conn, source_path=str(a_dir))
    # error: first transient failure; missing: third transient failure
    ids["error_first_attempt"] = _insert_attachment(
        conn, source_path=str(attachments_root / "ghost1.bin")
    )
    ids["missing_third_attempt"] = _insert_attachment(
        conn, source_path=str(attachments_root / "ghost2.bin"), state="error",
        attempts=MAX_MATERIALIZATION_ATTEMPTS - 1, last_error="[Errno 2] gone",
    )
    # reclassified by the pre-passes, never read
    ids["missing_no_source"] = _insert_attachment(conn, source_path=None)
    ids["unsupported_healed"] = _insert_attachment(
        conn, source_path=str(tmp_path / "elsewhere.bin"), state="error", attempts=1,
        next_attempt_at=_future(), last_error="old refusal",
    )
    return ids


def test_report_counts_match_database_state_counts_for_every_terminal_state(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    """Pins the invariant the CLI prints from: every counter on the
    report is a row the database now holds in that state — including
    the third failed attempt, which lands in `missing`, not `errored`
    (a run once printed errored=440 over a database holding 234
    missing + 211 error, because the containment branch ignored
    `_mark_failure`'s return value)."""
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"
    ids = _mixed_outcome_corpus(scratch_db, attachments_root, tmp_path)

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    counts = _state_counts(scratch_db)

    assert report.considered == 6
    assert counts == {"materialized": 2, "unsupported": 3, "missing": 2, "error": 1}
    assert report.materialized == counts["materialized"]
    assert report.marked_unsupported + report.reclassified_unsupported == counts["unsupported"]
    assert report.marked_missing + report.reclassified_missing_no_source == counts["missing"]
    assert report.errored == counts["error"]
    # ...and each counter came from the row it claims to describe.
    assert report.marked_unsupported == 2
    assert report.reclassified_unsupported == 1
    assert report.marked_missing == 1
    assert report.reclassified_missing_no_source == 1
    assert _fetch_state(scratch_db, ids["missing_third_attempt"]) == "missing"
    assert _fetch_state(scratch_db, ids["error_first_attempt"]) == "error"
    assert _fetch_state(scratch_db, ids["unsupported_healed"]) == "unsupported"

    # Second pass: nothing left that this run can decide.
    again = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle
    )
    assert (
        again.considered, again.reclassified_unsupported, again.reclassified_missing_no_source
    ) == (0, 0, 0)
    assert _state_counts(scratch_db) == counts


def test_dry_run_counts_only_what_is_decidable_without_reading(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"
    _mixed_outcome_corpus(scratch_db, attachments_root, tmp_path)
    before = _state_counts(scratch_db)

    report = run_backfill(
        scratch_db, data_root, attachments_root, yes_full_run=True,
        throttle=no_sleep_throttle, dry_run=True,
    )

    assert report.dry_run is True
    assert report.considered == 6  # same candidate set a real run would attempt
    assert report.marked_unsupported == 1  # the refusal is decidable; the EISDIR is not
    assert report.reclassified_unsupported == 1
    assert report.reclassified_missing_no_source == 1
    assert (report.materialized, report.errored, report.marked_missing) == (0, 0, 0)
    assert _state_counts(scratch_db) == before
    assert not (data_root / "attachments").exists()


def test_low_disk_space_halts_the_run(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    f = attachments_root / "f.bin"
    f.write_bytes(b"data")
    _insert_attachment(scratch_db, source_path=str(f))

    report = run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=no_sleep_throttle,
        disk_free_fn=lambda _p: 0,  # always "out of space"
    )

    assert report.halted_low_disk_space is True
    assert report.materialized == 0


def test_reconciliation_report_enumerates_gaps(
    scratch_db: psycopg.Connection, tmp_path: Path, no_sleep_throttle: RateThrottle
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    data_root = tmp_path / "data_root"

    ok_file = attachments_root / "ok.bin"
    ok_file.write_bytes(b"ok")
    _insert_attachment(scratch_db, source_path=str(ok_file))
    _insert_attachment(scratch_db, source_path=str(attachments_root / "never.bin"))
    _insert_attachment(scratch_db, source_path=str(tmp_path / "outside.bin"))

    run_backfill(scratch_db, data_root, attachments_root, yes_full_run=True, throttle=no_sleep_throttle)

    report = build_reconciliation_report(scratch_db)
    assert report.total == 3
    assert report.materialized_and_present == 1
    assert len(report.gaps) == 2
    by_state = {gap.state: gap for gap in report.gaps}
    assert set(by_state) == {"error", "unsupported"}
    assert by_state["error"].unsupported_reason is None
    assert by_state["unsupported"].unsupported_reason == "temp-directory-path"
    assert by_state["unsupported"].reason.startswith("unsupported (temp-directory-path): ")
    assert "outside.bin" not in by_state["unsupported"].reason  # the path stays in the database
    assert 0.0 < report.completeness_ratio < 1.0
