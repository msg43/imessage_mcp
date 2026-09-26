"""`imsg sync`'s segmentation and embedding wait while a search is in
flight, against a real Postgres, through the CLI (`imsg.search_yield`).

The search is the real marker the MCP servers and the search page's
model API hold (`QueryInFlightMarker`), on its own connection. A watcher
releases it once `pg_locks` shows the step waiting for it (or after a
timeout, so a build that never waits fails instead of hanging), and the
fake models record whether the search was still in flight at each call.

- Segmentation: no boundary-model call while the search holds the
  marker; the step carries on once it is released.
- Embedding: likewise for every batch, with the search starting while
  segmentation is running.
- The bound: with the search never ending, each unit waits the bound and
  then runs, and the sync finishes its work.
- Nothing in flight: the sync runs as before and says nothing.
- The search page's model API holds the marker while it computes an
  embedding or a rerank, as the background steps see it.

Skips cleanly when no scratch Postgres is reachable, like every other
integration file. Fictional personas only.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

import imsg.cli as cli_module
import imsg.stages.sync as sync_module
from imsg import constants
from imsg.cli import app
from imsg.db.enrichment_yield_locks import QUERY_IN_FLIGHT_LOCK_KEY, QueryInFlightMarker
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.mount.guard import MountInfo
from imsg.segment.boundaries import FakeBoundaryProvider
from test_cli import _write_config

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_search_yield_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

NAMESPACE = 0x696D7367
"""The high 32 bits of every imsg advisory key (`pg_locks.classid`)."""
SEGMENT_YIELDING_OBJID = 3
EMBED_YIELDING_OBJID = 4
"""The low 32 bits (`pg_locks.objid`) of the keys a segmentation or an
embedding step holds while it waits: written out here rather than
imported, so this file runs, and fails for what it checks, against a
build that has no such keys."""

runner = CliRunner()


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
def dsn() -> Iterator[str]:
    """A migrated scratch database, dropped afterwards."""
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        admin.close()
    conn = psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
    finally:
        conn.close()
    yield _dsn(TEST_DB_NAME)
    admin = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
    finally:
        admin.close()


def _one(conn: psycopg.Connection, sql: str, params: tuple[object, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    assert row is not None
    return row[0]


def _seed_one_dirty_chat(dsn: str, messages: int = 12) -> None:
    """One chat, one session of `messages` short messages: more than
    `segmentation.topical_min_messages` (10), so the boundary model is
    asked, and within one window, so it is asked exactly once."""
    conn = psycopg.connect(dsn, autocommit=True)
    try:
        people = []
        for name, short, owner in (("Jamie Owner", "owner", True), ("Alice Example", "alice", False)):
            people.append(
                _one(
                    conn,
                    "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
                    "VALUES (%s, %s, %s, false) RETURNING person_id",
                    (name, short, owner),
                )
            )
        guid = f"chat-{uuid.uuid4()}"
        chat_id = _one(
            conn,
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') RETURNING chat_id",
            (guid, f"thread-{guid}"),
        )
        for person in people:
            conn.execute(
                "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)", (chat_id, person)
            )
        base = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
        for i in range(messages):
            message_guid = f"msg-{uuid.uuid4()}"
            conn.execute(
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
                    people[i % 2],
                    i % 2 == 0,
                    base + timedelta(minutes=i),
                    f"Alice asks about the kite order, part {i}",
                    f"Alice asks about the kite order, part {i}",
                ),
            )
    finally:
        conn.close()


@dataclass
class Search:
    """A search in flight: the query-in-flight marker, held on its own
    connection exactly as an MCP server or the model API holds it, and
    released by a watcher once `pg_locks` shows a step waiting for it."""

    dsn: str
    marker: QueryInFlightMarker = field(init=False)
    started: bool = False
    released_at: float | None = None
    saw_waiting: dict[int, bool] = field(default_factory=dict)
    _watcher: threading.Thread | None = None
    _closing: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.marker = QueryInFlightMarker(lambda: psycopg.connect(self.dsn, autocommit=True))

    def start(self) -> None:
        self.marker.__enter__()
        assert self.marker.is_marking
        self.started = True

    def release_once_waiting(self, objid: int, *, hold_after: float = 0.3, timeout: float = 30.0) -> None:
        """Watch for a step holding its yielding key; then keep the search
        going `hold_after` seconds more and end it. Ends it after
        `timeout`, or when the test closes the search, whatever happens,
        so a build that never waits fails its assertions rather than
        hanging."""

        def watch() -> None:
            reader = psycopg.connect(self.dsn, autocommit=True)
            try:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and not self._closing.is_set():
                    held = _one(
                        reader,
                        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "
                        "AND classid = %s AND objid = %s",
                        (NAMESPACE, objid),
                    )
                    if held:
                        self.saw_waiting[objid] = True
                        time.sleep(hold_after)
                        break
                    time.sleep(0.01)
            finally:
                reader.close()
                self.end()

        self._watcher = threading.Thread(target=watch, daemon=True)
        self._watcher.start()

    def end(self) -> None:
        if self.marker.is_marking:
            self.marker.__exit__(None, None, None)
            self.released_at = time.monotonic()

    def close(self) -> None:
        self._closing.set()
        if self._watcher is not None:
            self._watcher.join(timeout=10)
        self.end()
        self.marker.close()

    @property
    def in_flight(self) -> bool:
        return self.marker.is_marking


class RecordingBoundary(FakeBoundaryProvider):
    """The fake boundary model, noting whether a search was in flight at
    each call; `on_call` runs inside the call (a search can start there)."""

    def __init__(self, search: Search, on_call: Callable[[], None] | None = None) -> None:
        super().__init__(messages_per_segment=5)
        self.search = search
        self.on_call = on_call
        self.search_in_flight_at_call: list[bool] = []

    def detect_boundaries(self, window: Any) -> list[int]:
        self.search_in_flight_at_call.append(self.search.in_flight)
        if self.on_call is not None:
            self.on_call()
        return super().detect_boundaries(window)


class RecordingEmbedder(FakeTextEmbeddingProvider):
    def __init__(self, search: Search) -> None:
        super().__init__(dim=constants.PRIMARY_EMBEDDING_DIM)
        self.search = search
        self.search_in_flight_at_batch: list[bool] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.search_in_flight_at_batch.append(self.search.in_flight)
        return super().embed_documents(texts)


@pytest.fixture
def env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dsn: str
) -> Iterator[dict[str, Any]]:
    """`imsg sync` on the fake model backend against the scratch database:
    snapshot, extract and identity are stubbed (there is no chat.db), and
    segmentation and embedding run for real."""
    from test_sync import (
        _fake_extract_result,
        _fake_identity_result,
        _fake_snapshot_result,
        _ok_invariant,
    )

    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")
    prompt = data_root / "prompts" / "segment_boundaries.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("segment this")
    config = tmp_path / "config.yaml"
    _write_config(config, data_root, messages_dir)
    config.write_text(config.read_text() + "enrichment:\n  yield_poll_interval_seconds: 0.02\n")

    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(
        cli_module,
        "connect",
        lambda database, **kw: psycopg.connect(dsn, autocommit=kw.get("autocommit", True)),
    )
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    monkeypatch.setattr(sync_module, "guard_mount", lambda data_root: None)
    real_all_sources = sync_module.run_sync_all_sources
    monkeypatch.setattr(
        cli_module,
        "run_sync_all_sources",
        lambda **kw: real_all_sources(
            **kw,
            run_snapshot_fn=lambda **k: _fake_snapshot_result(data_root / "snapshot.db"),
            run_extract_fn=lambda **k: _fake_extract_result(),
            run_identity_fn=lambda **k: _fake_identity_result(_ok_invariant()),
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "build_multimodal_provider",
        lambda cfg: FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM),
    )
    _seed_one_dirty_chat(dsn)
    search = Search(dsn)
    yield {"config": config, "dsn": dsn, "search": search, "monkeypatch": monkeypatch}
    search.close()


def _use_models(env: dict[str, Any], boundary: RecordingBoundary, embedder: RecordingEmbedder) -> None:
    monkeypatch: pytest.MonkeyPatch = env["monkeypatch"]
    monkeypatch.setattr(cli_module, "build_boundary_provider", lambda cfg, prompt: boundary)
    monkeypatch.setattr(cli_module, "build_text_provider", lambda cfg: embedder)


def _sync(env: dict[str, Any]) -> Any:
    return runner.invoke(app, ["sync", "--config", str(env["config"])])


def _counts(dsn: str) -> tuple[int, int]:
    conn = psycopg.connect(dsn, autocommit=True)
    try:
        return (
            _one(conn, "SELECT count(*) FROM segment"),
            _one(conn, "SELECT count(*) FROM segment_embedding"),
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------
# each step waits while a search is in flight, and carries on after it
# --------------------------------------------------------------------------


def test_sync_segmentation_waits_while_a_search_is_in_flight(env: dict[str, Any]) -> None:
    search: Search = env["search"]
    boundary = RecordingBoundary(search)
    embedder = RecordingEmbedder(search)
    _use_models(env, boundary, embedder)
    search.start()
    search.release_once_waiting(SEGMENT_YIELDING_OBJID)

    result = _sync(env)

    assert result.exit_code == 0, result.output
    assert search.saw_waiting.get(SEGMENT_YIELDING_OBJID), "segmentation never waited"
    assert boundary.search_in_flight_at_call == [False], (
        "the boundary model ran while a search was in flight"
    )
    assert "sync: segmentation yielded to in-flight searches 1 time(s)" in result.output
    assert "embedding yielded" not in result.output
    segments, embedded = _counts(env["dsn"])
    assert segments > 0 and embedded == segments  # the sync finished its work


def test_sync_embedding_waits_while_a_search_is_in_flight(env: dict[str, Any]) -> None:
    """The search starts while segmentation is asking the boundary model
    (segmentation had already checked, so it goes on): embedding's first
    batch must wait for it."""
    search: Search = env["search"]

    def search_arrives() -> None:
        search.start()
        search.release_once_waiting(EMBED_YIELDING_OBJID)

    boundary = RecordingBoundary(search, on_call=search_arrives)
    embedder = RecordingEmbedder(search)
    _use_models(env, boundary, embedder)

    result = _sync(env)

    assert result.exit_code == 0, result.output
    assert search.saw_waiting.get(EMBED_YIELDING_OBJID), "embedding never waited"
    assert embedder.search_in_flight_at_batch, "nothing was embedded"
    assert not any(embedder.search_in_flight_at_batch), (
        "an embedding batch ran while a search was in flight"
    )
    assert "sync: embedding yielded to in-flight searches 1 time(s)" in result.output
    assert "segmentation yielded" not in result.output
    segments, embedded = _counts(env["dsn"])
    assert segments > 0 and embedded == segments


# --------------------------------------------------------------------------
# the bound, and the case with no search at all
# --------------------------------------------------------------------------


def test_a_search_that_never_ends_delays_each_unit_by_the_bound_only(env: dict[str, Any]) -> None:
    env["config"].write_text(
        env["config"].read_text().replace(
            "enrichment:\n", "enrichment:\n  yield_max_pause_seconds: 0.3\n"
        )
    )
    search: Search = env["search"]
    boundary = RecordingBoundary(search)
    embedder = RecordingEmbedder(search)
    _use_models(env, boundary, embedder)
    search.start()  # and never released while the sync runs

    started = time.monotonic()
    result = _sync(env)
    elapsed = time.monotonic() - started

    assert result.exit_code == 0, result.output
    batches = len(embedder.search_in_flight_at_batch)
    assert boundary.search_in_flight_at_call == [True]  # it ran anyway, after the bound
    assert batches >= 1 and all(embedder.search_in_flight_at_batch)
    assert (
        "sync: segmentation yielded to in-flight searches 1 time(s), 0.3s total; went ahead "
        "after the 0.3 s limit 1 time(s)"
    ) in result.output
    assert f"sync: embedding yielded to in-flight searches {batches} time(s)" in result.output
    assert elapsed >= 0.3 * (1 + batches)
    segments, embedded = _counts(env["dsn"])
    assert segments > 0 and embedded == segments


def test_with_no_search_in_flight_the_sync_runs_as_before(env: dict[str, Any]) -> None:
    search: Search = env["search"]
    boundary = RecordingBoundary(search)
    embedder = RecordingEmbedder(search)
    _use_models(env, boundary, embedder)

    result = _sync(env)

    assert result.exit_code == 0, result.output
    assert "yielded" not in result.output
    assert boundary.search_in_flight_at_call == [False]
    segments, embedded = _counts(env["dsn"])
    assert segments > 0 and embedded == segments


# --------------------------------------------------------------------------
# status, and a waiting step never holds a search up
# --------------------------------------------------------------------------


def test_status_reads_which_step_is_yielding_without_taking_a_lock(dsn: str) -> None:
    from imsg.db.enrichment_yield_locks import read_yield_state

    reader = psycopg.connect(dsn, autocommit=True)
    holder = psycopg.connect(dsn, autocommit=True)
    try:
        idle = read_yield_state(reader)
        assert (idle.segment_yielding, idle.embed_yielding) == (False, False)  # type: ignore[attr-defined]
        holder.execute("SELECT pg_advisory_lock_shared(%s)", ((NAMESPACE << 32) | EMBED_YIELDING_OBJID,))
        state = read_yield_state(reader)
        assert (state.segment_yielding, state.embed_yielding) == (False, True)  # type: ignore[attr-defined]
        holder.execute(
            "SELECT pg_advisory_lock_shared(%s)", ((NAMESPACE << 32) | SEGMENT_YIELDING_OBJID,)
        )
        assert read_yield_state(reader).segment_yielding is True  # type: ignore[attr-defined]
        assert read_yield_state(reader).enrichment_paused is False
        holder.close()  # a step that dies takes its "yielding" with it
        gone = read_yield_state(reader)
        assert (gone.segment_yielding, gone.embed_yielding) == (False, False)  # type: ignore[attr-defined]
    finally:
        reader.close()
        if not holder.closed:
            holder.close()


def test_a_step_checking_for_searches_never_makes_a_search_wait(dsn: str) -> None:
    """A search's marker is a try-lock, and a step's probe holds the
    conflicting lock for one statement only: with a step checking every
    5 ms (production checks once per unit, and every 0.25 s while it
    waits), every search still takes its marker at once, and nearly all
    of them are marked. A probe that kept the lock would leave every
    search after it unmarked."""
    from imsg.db.enrichment_yield_locks import QueryYieldGate  # type: ignore[attr-defined]

    searches = QueryInFlightMarker(lambda: psycopg.connect(dsn, autocommit=True))
    step_conn = psycopg.connect(dsn, autocommit=True)
    stop = threading.Event()
    gate = QueryYieldGate(step_conn, step="embed", poll_interval_seconds=0.005, max_pause_seconds=30.0)

    def step() -> None:
        while not stop.is_set():
            gate.wait_until_clear(interrupt=stop.is_set)
            time.sleep(0.005)

    checker = threading.Thread(target=step, daemon=True)
    checker.start()
    try:
        slowest, marked = 0.0, 0
        for _ in range(300):
            began = time.perf_counter()
            with searches:
                marked += int(searches.is_marking)
            slowest = max(slowest, time.perf_counter() - began)
        assert slowest < 1.0, f"a search waited {slowest * 1000:.0f} ms for its marker"
        assert marked >= 240, f"only {marked} of 300 searches were marked"
    finally:
        stop.set()
        checker.join(timeout=5)
        searches.close()
        step_conn.close()


# --------------------------------------------------------------------------
# the search page's model API counts as a search while it computes
# --------------------------------------------------------------------------


def test_the_model_api_holds_the_marker_while_it_embeds_and_reranks(dsn: str) -> None:
    """The model API runs inside the public server and shares its marker
    (`imsg mcp public` passes the same one). Seen from another session the
    way a waiting background step probes it: taken while each embedding
    and rerank computes, free before and after."""
    import http.client
    import json

    from imsg.retrieval.reranker import FakeRerankerProvider
    from imsg.search_page.model_api_server import (
        BIND_HOST,
        ModelAccess,
        ModelApiServer,
        ModelApiState,
    )

    probe = psycopg.connect(dsn, autocommit=True)

    def search_in_flight() -> bool:
        """The background steps' probe: the exclusive try-lock fails
        exactly while a search holds the shared one."""
        taken = bool(_one(probe, "SELECT pg_try_advisory_lock(%s)", (QUERY_IN_FLIGHT_LOCK_KEY,)))
        if taken:
            probe.execute("SELECT pg_advisory_unlock(%s)", (QUERY_IN_FLIGHT_LOCK_KEY,))
        return not taken

    seen: dict[str, bool] = {}

    class Text(FakeTextEmbeddingProvider):
        def embed_query(self, text: str, *, instruction: str) -> list[float]:
            seen["embed"] = search_in_flight()
            return super().embed_query(text, instruction=instruction)

    class Multimodal(FakeMultimodalEmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            seen["multimodal"] = search_in_flight()
            return super().embed_text(text)

    class Reranker(FakeRerankerProvider):
        def score(self, query: str, documents: list[str]) -> list[float]:
            seen["rerank"] = search_in_flight()
            return super().score(query, documents)

    marker = QueryInFlightMarker(lambda: psycopg.connect(dsn, autocommit=True))
    secret = "s" * 48
    state = ModelApiState(
        models=ModelAccess(
            text_provider=Text(dim=constants.PRIMARY_EMBEDDING_DIM),
            reranker=Reranker(),
            multimodal_provider=Multimodal(dim=constants.MULTIMODAL_EMBEDDING_DIM),
            query_instruction="retrieve",
            multimodal_enabled=True,
            query_marker=marker,
        ),
        secret=secret,
        port=0,
        warm_up=None,
        idle_unloader=None,
        log=lambda _line: None,
    )
    server = ModelApiServer(0, state).start()

    def post(path: str, body: dict[str, Any]) -> int:
        conn = http.client.HTTPConnection(BIND_HOST, server.port, timeout=10)
        try:
            conn.request(
                "POST",
                path,
                body=json.dumps(body).encode(),
                headers={
                    "Host": f"127.0.0.1:{server.port}",
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                },
            )
            return conn.getresponse().status
        finally:
            conn.close()

    try:
        assert search_in_flight() is False
        assert post("/v1/embed", {"text": "kite order", "multimodal": True}) == 200
        assert post("/v1/rerank", {"query": "kite order", "documents": ["kites", "boats"]}) == 200
        assert seen == {"embed": True, "multimodal": True, "rerank": True}
        assert search_in_flight() is False  # released when each call ended
    finally:
        server.stop()
        marker.close()
        probe.close()
