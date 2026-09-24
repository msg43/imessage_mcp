"""Heavy background work stops cleanly between units of work, against a
real Postgres (`imsg.background_gate`): the unit in progress finishes and
commits, nothing more is started, no enrichment lease is left behind, and
the rest is still there for the next run.

Skips cleanly when no scratch Postgres is reachable, like every other
integration file. Fictional personas only.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from conftest import ConfigDictFactory
from imsg import constants
from imsg.backfill.pipeline import run_backfill
from imsg.backfill.throttle import RateThrottle
from imsg.background_gate import BackgroundWorkDeferred, DeferralKind, StopReason
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.enrichment_yield_locks import YieldReport
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.pipeline import EmbedRunReport, run_embed
from imsg.embed.provider import FakeTextEmbeddingProvider
from imsg.enrich.pipeline import EnrichmentProviders
from imsg.enrich.queue import EnrichmentTask, claim_tasks, complete_task, enqueue, release_task
from imsg.enrich.worker import run_enrich_worker
from imsg.segment.boundaries import FakeBoundaryProvider
from imsg.segment.pipeline import find_dirty_chats, run_segment

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_memory_stops_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
PAUSED = StopReason(DeferralKind.PAUSED, "heavy background work is paused: test")
PRESSURE = StopReason(DeferralKind.MEMORY, "the kernel reports critical memory pressure")


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


pytestmark = pytest.mark.skipif(
    not _admin_reachable(),
    reason=(
        "no reachable scratch Postgres instance "
        f"(tried {TEST_PG_HOST}:{TEST_PG_PORT}) — set IMSG_TEST_PG_HOST/"
        "IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    """A migrated scratch database on an autocommit connection — the shape
    production's `imsg.db.connection.connect` gives every stage."""
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


@pytest.fixture
def config(config_dict_factory: ConfigDictFactory) -> Config:
    return load_config_dict(config_dict_factory())


class StopAfter:
    """A stop check that says go `allowed` times, then gives `reason`."""

    def __init__(self, allowed: int, reason: StopReason) -> None:
        self.allowed = allowed
        self.reason = reason
        self.asked = 0

    def __call__(self) -> StopReason | None:
        self.asked += 1
        return None if self.asked <= self.allowed else self.reason


def _one(conn: psycopg.Connection, sql: str, params: tuple[object, ...] = ()) -> object:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    assert row is not None
    return row[0]


# --------------------------------------------------------------------------
# embedding: stops between batches
# --------------------------------------------------------------------------


def _insert_segments(conn: psycopg.Connection, count: int) -> None:
    guid = f"chat-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') RETURNING chat_id",
            (guid, f"thread-{guid}"),
        )
        chat_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
            "VALUES (%s, now(), now(), 3.0) RETURNING session_id",
            (chat_id,),
        )
        session_id = cur.fetchone()[0]  # type: ignore[index]
        for seq in range(count):
            cur.execute(
                """
                INSERT INTO segment (
                    stable_key, chat_id, session_id, seq_in_session, started_at, ended_at,
                    message_count, token_count, rendered_text, rendered_sha256, seg_config_hash
                ) VALUES (%s, %s, %s, %s, now(), now(), 1, 10, %s, 'x', 'cfg-hash')
                """,
                (f"stable-{uuid.uuid4()}", chat_id, session_id, seq, f"Alice asks about kites {seq}"),
            )


def test_embedding_stops_between_batches_keeping_what_committed(db: psycopg.Connection) -> None:
    _insert_segments(db, 3)
    provider = FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM)

    with pytest.raises(BackgroundWorkDeferred) as stopped:
        run_embed(db, provider, batch_size=1, stop_check=StopAfter(1, PRESSURE))

    assert stopped.value.reason == PRESSURE
    partial = stopped.value.partial
    assert isinstance(partial, EmbedRunReport)
    assert partial.segments_embedded == 1
    assert _one(db, "SELECT count(*) FROM segment_embedding") == 1  # the first batch committed

    # The rest were left pending, and the next run picks them up.
    report = run_embed(db, provider, batch_size=1)
    assert report.segments_embedded == 2
    assert _one(db, "SELECT count(*) FROM segment_embedding") == 3


def test_embedding_that_is_never_stopped_runs_to_the_end(db: psycopg.Connection) -> None:
    _insert_segments(db, 3)
    provider = FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM)
    check = StopAfter(1000, PRESSURE)
    report = run_embed(db, provider, batch_size=1, stop_check=check)
    assert report.segments_embedded == 3
    assert check.asked == 3  # asked before every batch


# --------------------------------------------------------------------------
# segmentation: stops between chats
# --------------------------------------------------------------------------


def _insert_chat_with_messages(conn: psycopg.Connection, owner_id: int, other_id: int) -> int:
    guid = f"chat-{uuid.uuid4()}"
    base = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') RETURNING chat_id",
            (guid, f"thread-{guid}"),
        )
        chat_id = int(cur.fetchone()[0])  # type: ignore[index]
        for person in (owner_id, other_id):
            cur.execute(
                "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)",
                (chat_id, person),
            )
        for i in range(3):
            message_guid = f"msg-{uuid.uuid4()}"
            cur.execute(
                """
                INSERT INTO message (
                    source_guid, message_key, chat_id, sender_person_id,
                    is_from_me, sent_at, service, text_original, text_normalized
                ) VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s)
                """,
                (
                    message_guid,
                    f"key-{message_guid}",
                    chat_id,
                    owner_id if i % 2 == 0 else other_id,
                    i % 2 == 0,
                    base + timedelta(minutes=i),
                    f"message {i}",
                    f"message {i}",
                ),
            )
    return chat_id


def _person(conn: psycopg.Connection, display_name: str, short_name: str, *, owner: bool) -> int:
    return int(
        _one(  # type: ignore[arg-type]
            conn,
            "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
            "VALUES (%s, %s, %s, false) RETURNING person_id",
            (display_name, short_name, owner),
        )
    )


def test_segmentation_stops_between_chats(db: psycopg.Connection, config: Config) -> None:
    owner = _person(db, "Jamie Owner", "owner", owner=True)
    alice = _person(db, "Alice Example", "alice", owner=False)
    bob = _person(db, "Bob Builder", "bob", owner=False)
    chats = {_insert_chat_with_messages(db, owner, alice), _insert_chat_with_messages(db, owner, bob)}
    assert set(find_dirty_chats(db, index_unsent=False)) == chats

    with pytest.raises(BackgroundWorkDeferred) as stopped:
        run_segment(
            db, config, FakeBoundaryProvider(), b"prompt", stop_check=StopAfter(1, PAUSED)
        )

    assert stopped.value.reason == PAUSED
    done = stopped.value.partial
    assert isinstance(done, list) and len(done) == 1
    still_dirty = set(find_dirty_chats(db, index_unsent=False))
    assert len(still_dirty) == 1 and still_dirty < chats  # one chat committed, one left
    assert _one(db, "SELECT count(DISTINCT chat_id) FROM segment") == 1

    run_segment(db, config, FakeBoundaryProvider(), b"prompt")
    assert find_dirty_chats(db, index_unsent=False) == {}


# --------------------------------------------------------------------------
# the enrichment worker: stops between tasks, holding no lease
# --------------------------------------------------------------------------


class NeverYields:
    def wait_until_clear(self) -> YieldReport:
        return YieldReport(paused=False, waited_seconds=0.0)


def _attachment(conn: psycopg.Connection) -> int:
    guid = f"att-{uuid.uuid4()}"
    return int(
        _one(  # type: ignore[arg-type]
            conn,
            "INSERT INTO attachment (source_guid, attachment_key) VALUES (%s, %s) "
            "RETURNING attachment_id",
            (guid, f"key-{guid}"),
        )
    )


def _states(conn: psycopg.Connection) -> list[tuple[str, str | None]]:
    with conn.cursor() as cur:
        cur.execute("SELECT state::text, locked_by FROM enrichment ORDER BY attachment_id")
        return [(str(state), locked_by) for state, locked_by in cur.fetchall()]


def _completing(conn: psycopg.Connection) -> object:
    def process(
        connection: psycopg.Connection, cfg: Config, providers: EnrichmentProviders, task: EnrichmentTask
    ) -> str:
        complete_task(connection, task.attachment_id, task.kind, model="m", model_version=None, text="t")
        return "done"

    return process


def test_the_worker_stops_between_tasks_and_leaves_no_lease(
    db: psycopg.Connection, config: Config
) -> None:
    for _ in range(3):
        enqueue(db, _attachment(db), ("ocr",))

    report = run_enrich_worker(
        db,
        config,
        providers=None,  # type: ignore[arg-type]
        worker_id="worker-test",
        claim_order=("ocr",),
        limit=100,
        yield_gate=NeverYields(),
        stop_check=StopAfter(1, PRESSURE),
        process=_completing(db),  # type: ignore[arg-type]
    )

    assert report.processed == 1
    assert report.stopped == PRESSURE
    assert _states(db) == [("pending", None), ("pending", None), ("done", None)]
    assert _one(db, "SELECT count(*) FROM enrichment WHERE state = 'running'") == 0


def test_a_task_interrupted_mid_processing_is_handed_back_unfinished(
    db: psycopg.Connection, config: Config
) -> None:
    enqueue(db, _attachment(db), ("ocr",))

    def interrupted(*args: object) -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_enrich_worker(
            db,
            config,
            providers=None,  # type: ignore[arg-type]
            worker_id="worker-test",
            claim_order=("ocr",),
            limit=10,
            yield_gate=NeverYields(),
            process=interrupted,
        )

    assert _states(db) == [("pending", None)]
    assert _one(db, "SELECT attempts FROM enrichment") == 0  # not counted against it


def test_releasing_leaves_a_task_leased_to_another_worker_alone(db: psycopg.Connection) -> None:
    attachment = _attachment(db)
    enqueue(db, attachment, ("ocr",))
    [task] = claim_tasks(db, worker_id="worker-other", limit=1, kinds=("ocr",))
    assert release_task(db, task.attachment_id, task.kind, worker_id="worker-test") is False
    assert _states(db) == [("running", "worker-other")]
    assert release_task(db, task.attachment_id, task.kind, worker_id="worker-other") is True
    assert _states(db) == [("pending", None)]


# --------------------------------------------------------------------------
# attachment backfill: stops between files
# --------------------------------------------------------------------------


def test_backfill_stops_between_files(db: psycopg.Connection, tmp_path: Path) -> None:
    attachments_root = tmp_path / "Attachments"
    data_root = tmp_path / "data_root"
    ids = []
    for i in range(3):
        source = attachments_root / f"{i}" / "photo.jpg"
        source.parent.mkdir(parents=True)
        source.write_bytes(f"fake jpeg {i}".encode())
        ids.append(
            _one(
                db,
                "INSERT INTO attachment (source_guid, attachment_key, source_path, state) "
                "VALUES (%s, %s, %s, 'dataless') RETURNING attachment_id",
                (f"att-{i}", f"key-{i}", str(source)),
            )
        )

    report = run_backfill(
        db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=RateThrottle(10_000, sleep_fn=lambda _: None),
        stop_check=StopAfter(1, PAUSED),
    )

    assert report.stopped == PAUSED
    assert report.materialized == 1
    assert any(note.startswith("stopped after 1 file(s): deferred: paused") for note in report.notes)
    with db.cursor() as cur:
        cur.execute("SELECT state::text FROM attachment ORDER BY attachment_id")
        states = sorted(str(row[0]) for row in cur.fetchall())
    assert states == ["dataless", "dataless", "materialized"]
