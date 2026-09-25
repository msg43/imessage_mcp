"""A search's five candidate channels run side by side on separate
connections, and nothing about a result changes (QA review 2026-09-24,
latency item "the five candidate channels run one after another on one
connection").

Real Postgres (pgvector, HNSW, iterative scans) and real SQLite FTS5
throughout, same skip pattern as `tests/test_retrieval_integration.py`.
The corpus is generated from a fixed seed and fictional vocabulary: six
chats, 160 segments, attachments with text chunks, chunk embeddings and
multimodal embeddings, so every channel has something to find.

- **Equality.** The same service answers a battery of searches — BM25,
  quoted phrases (trigram), an emoji, people/date/attachment filters,
  every scope — three ways: shared connections (the old sequential
  path), pooled with room for every channel, and pooled with a single
  connection (every channel falls back to the call's own). All three
  produce the same `SearchMessagesResult`, and the same bytes once
  serialized the way the public server serializes them.
- **Concurrency is real.** With each channel slowed by a fixed delay,
  the pooled search takes about one delay and the shared one five, and
  the channels run on the pool's worker threads.
- **Calls no longer queue behind each other.** A `get_conversation`
  finishes while a search is still inside a channel, on the pooled
  build; the shared build makes it wait.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import apsw
import psycopg
import pytest

from imsg import constants
from imsg.db.migrations import PostgresMigrationRunner
from imsg.db.pool import postgres_pool
from imsg.embed.fts.schema import create_schema
from imsg.embed.fts.sync import upsert_chunk_row, upsert_segment_row
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.embed.vector_codec import vector_literal
from imsg.keys import message_key as derive_message_key
from imsg.keys import thread_key as derive_thread_key
from imsg.mcp.tools import handlers
from imsg.retrieval import fts_search, vector_search
from imsg.retrieval.access import LOCAL_FULL_ACCESS, AccessContext
from imsg.retrieval.connections import (
    CHANNEL_THREAD_PREFIX,
    RetrievalConnections,
    fts_pool,
    open_fts_reader,
)
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.service import RetrievalService, SearchMessagesResult

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_concurrent_channels_test"

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


class _Cfg:
    """The fields `RetrievalService` reads (as in
    `tests/test_retrieval_integration.py`)."""

    class _Retrieval:
        k_fts = 100
        k_vector = 100
        rrf_k = 60
        rerank_top = 20
        default_limit = 10
        hnsw_ef_search = 1000

    class _Render:
        timezone = "America/New_York"
        attachment_snippet_chars = 200

    class _Embedding:
        query_instruction = "search"

        class _Multimodal:
            enabled = True

        multimodal = _Multimodal()

    class _Policy:
        index_unsent = False
        index_edit_history = False

    retrieval = _Retrieval()
    render = _Render()
    embedding = _Embedding()
    policy = _Policy()


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

WORDS: tuple[str, ...] = (
    "kite", "festival", "ferry", "schedule", "harbor", "bake", "sale", "receipt", "lantern",
    "parade", "picnic", "ridge", "orchard", "cider", "market", "ticket", "rehearsal", "choir",
    "bicycle", "repair", "lighthouse", "tide", "pool", "museum", "pottery", "class", "garden",
    "plot", "compost", "soup", "quilt", "raffle", "trivia", "night", "canoe", "portage",
    "trail", "map", "sunrise", "hike",
)
EMOJI = ("\U0001f6b2", "\U0001f389", "\U0001f3ae")  # bicycle, party popper, game pad


@dataclasses.dataclass
class Corpus:
    people: dict[str, str]  # display name -> short name
    thread_keys: list[str]
    segment_count: int


@pytest.fixture(scope="module")
def scratch_db() -> Iterator[psycopg.Connection]:
    """One database for the module: every test only reads the corpus."""
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


@pytest.fixture(scope="module")
def fts_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("concurrent-channels") / "fts.db"


def _sentence(rng: random.Random, n: int) -> str:
    words = [rng.choice(WORDS) for _ in range(n)]
    if rng.random() < 0.15:
        words.insert(rng.randrange(len(words)), rng.choice(EMOJI))
    return " ".join(words)


def _one(cur: psycopg.Cursor[Any]) -> int:
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def build_corpus(conn: psycopg.Connection, fts: apsw.Connection) -> Corpus:
    rng = random.Random(20260924)
    text_embedder = FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM)
    mm_embedder = FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM)
    people: dict[str, int] = {}
    short_names: dict[str, str] = {}
    with conn.cursor() as cur:
        for name, is_owner in (
            ("Owner Example", True),
            ("Alice Example", False),
            ("Bob Example", False),
            ("Carol Example", False),
            ("Dave Example", False),
        ):
            short = name.lower().split()[0]
            cur.execute(
                "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
                "VALUES (%s, %s, %s, false) RETURNING person_id",
                (name, short, is_owner),
            )
            people[name] = _one(cur)
            short_names[name] = short
        # Allowlist Alice and Bob (and the owner), so `allowlist` scope sees
        # their direct chats and nothing with Carol or Dave in it.
        for name in ("Owner Example", "Alice Example", "Bob Example"):
            cur.execute(
                "INSERT INTO allowlist_person (person_id, text_allowed) VALUES (%s, true)",
                (people[name],),
            )

        chat_members = [
            ("dm", None, ["Owner Example", "Alice Example"]),
            ("dm", None, ["Owner Example", "Bob Example"]),
            ("dm", None, ["Owner Example", "Carol Example"]),
            ("dm", None, ["Owner Example", "Dave Example"]),
            ("group", "Harbor crew", ["Owner Example", "Alice Example", "Bob Example"]),
            ("group", "Orchard club", ["Owner Example", "Carol Example", "Dave Example"]),
        ]
        chats: list[tuple[int, list[int]]] = []
        thread_keys: list[str] = []
        for kind, display, members in chat_members:
            guid = f"chat-{uuid.UUID(int=rng.getrandbits(128))}"
            tkey = derive_thread_key(guid)
            cur.execute(
                "INSERT INTO chat (source_guid, thread_key, kind, display_name) "
                "VALUES (%s, %s, %s, %s) RETURNING chat_id",
                (guid, tkey, kind, display),
            )
            chat_id = _one(cur)
            member_ids = [people[m] for m in members]
            for person_id in member_ids:
                cur.execute(
                    "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)",
                    (chat_id, person_id),
                )
            chats.append((chat_id, member_ids))
            thread_keys.append(tkey)

        start = datetime(2023, 1, 1, 15, 0, tzinfo=UTC)
        segment_count = 160
        for i in range(segment_count):
            chat_id, member_ids = chats[i % len(chats)]
            began = start + timedelta(days=i * 4, hours=rng.randrange(0, 8))
            lines = [_sentence(rng, rng.randrange(4, 12)) for _ in range(rng.randrange(1, 4))]
            rendered = "\n".join(lines)
            cur.execute(
                "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
                "VALUES (%s, %s, %s, 3.0) RETURNING session_id",
                (chat_id, began, began + timedelta(minutes=len(lines))),
            )
            session_id = _one(cur)
            stable_key = f"stable-{uuid.UUID(int=rng.getrandbits(128))}"
            cur.execute(
                """
                INSERT INTO segment (
                    stable_key, chat_id, session_id, seq_in_session, started_at, ended_at,
                    message_count, token_count, rendered_text, rendered_sha256, seg_config_hash
                ) VALUES (%s, %s, %s, 0, %s, %s, %s, 10, %s, 'x', 'cfg-hash')
                RETURNING segment_id
                """,
                (
                    stable_key,
                    chat_id,
                    session_id,
                    began,
                    began + timedelta(minutes=len(lines)),
                    len(lines),
                    rendered,
                ),
            )
            segment_id = _one(cur)
            upsert_segment_row(fts, segment_id, stable_key, rendered)
            vec = text_embedder.embed_documents([rendered])[0]
            cur.execute(
                "INSERT INTO segment_embedding (segment_id, model, text_sha256, vec) "
                "VALUES (%s, 'fake', %s, %s::halfvec)",
                (segment_id, f"sha-{segment_id}", vector_literal(vec)),
            )
            for n, line in enumerate(lines):
                sender = member_ids[(n + i) % len(member_ids)]
                guid = f"msg-{uuid.UUID(int=rng.getrandbits(128))}"
                has_attachment = n == 0 and i % 3 == 0
                cur.execute(
                    """
                    INSERT INTO message (
                        source_guid, message_key, chat_id, sender_person_id, is_from_me,
                        sent_at, service, text_original, text_normalized, has_attachments
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s, %s)
                    RETURNING message_id
                    """,
                    (
                        guid,
                        derive_message_key(guid),
                        chat_id,
                        sender,
                        sender == people["Owner Example"],
                        began + timedelta(minutes=n),
                        line,
                        line,
                        has_attachment,
                    ),
                )
                message_id = _one(cur)
                cur.execute(
                    "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
                    (segment_id, message_id),
                )
                if not has_attachment:
                    continue
                att_guid = f"att-{uuid.UUID(int=rng.getrandbits(128))}"
                cur.execute(
                    "INSERT INTO attachment (source_guid, attachment_key, filename, mime_type, state) "
                    "VALUES (%s, %s, %s, 'application/pdf', 'materialized') RETURNING attachment_id",
                    (att_guid, derive_message_key(att_guid), f"notes-{i}.pdf"),
                )
                attachment_id = _one(cur)
                cur.execute(
                    "INSERT INTO message_attachment (message_id, attachment_id, ordinal) "
                    "VALUES (%s, %s, 0)",
                    (message_id, attachment_id),
                )
                for seq in range(rng.randrange(1, 4)):
                    chunk_text = _sentence(rng, rng.randrange(8, 20))
                    cur.execute(
                        "INSERT INTO attachment_chunk (attachment_id, kind, seq, text) "
                        "VALUES (%s, 'pdf_text', %s, %s) RETURNING chunk_id",
                        (attachment_id, seq, chunk_text),
                    )
                    chunk_id = _one(cur)
                    upsert_chunk_row(fts, chunk_id, attachment_id, chunk_text)
                    chunk_vec = text_embedder.embed_documents([chunk_text])[0]
                    cur.execute(
                        "INSERT INTO attachment_chunk_embedding (chunk_id, model, text_sha256, vec) "
                        "VALUES (%s, 'fake', %s, %s::halfvec)",
                        (chunk_id, f"sha-c{chunk_id}", vector_literal(chunk_vec)),
                    )
                mm_vec = mm_embedder.embed_images([Path(f"/fictional/notes-{i}.png")])[0]
                cur.execute(
                    "INSERT INTO attachment_mm_embedding (attachment_id, model, media_sha256, vec) "
                    "VALUES (%s, 'fake', %s, %s::halfvec)",
                    (attachment_id, f"sha-m{attachment_id}", vector_literal(mm_vec)),
                )
    return Corpus(
        people=dict(short_names),
        thread_keys=thread_keys,
        segment_count=segment_count,
    )


@pytest.fixture(scope="module")
def corpus(scratch_db: psycopg.Connection, fts_path: Path) -> Iterator[Corpus]:
    fts = apsw.Connection(str(fts_path))
    create_schema(fts)
    with fts:
        built = build_corpus(scratch_db, fts)
    fts.close()
    yield built


def _open_pg() -> psycopg.Connection:
    return psycopg.connect(_dsn(TEST_DB_NAME), autocommit=True)


@pytest.fixture
def shared_fts(fts_path: Path, corpus: Corpus) -> Iterator[apsw.Connection]:
    del corpus
    conn = apsw.Connection(str(fts_path))
    yield conn
    conn.close()


def make_service(
    scratch_db: psycopg.Connection,
    fts: apsw.Connection,
    connections: RetrievalConnections | None = None,
) -> RetrievalService:
    return RetrievalService(
        pg_conn=scratch_db,
        fts_conn=fts,
        config=_Cfg(),  # type: ignore[arg-type]
        text_provider=FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM),
        reranker=FakeRerankerProvider(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM),
        connections=connections,
    )


@pytest.fixture
def pools_factory(fts_path: Path) -> Iterator[Callable[[int, int], RetrievalConnections]]:
    made: list[RetrievalConnections] = []

    def make(pg_size: int, fts_size: int) -> RetrievalConnections:
        pools = RetrievalConnections(
            pg=postgres_pool(_open_pg, max_size=pg_size, name="test-retrieval"),
            fts=fts_pool(lambda: open_fts_reader(fts_path), max_size=fts_size),
        )
        made.append(pools)
        return pools

    yield make
    for pools in made:
        pools.close()


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


def _search_battery(corpus: Corpus) -> list[tuple[AccessContext, dict[str, Any]]]:
    alice = corpus.people["Alice Example"]
    carol = corpus.people["Carol Example"]
    public_full = AccessContext(surface="public", scope="full", subject="1")
    allowlist = AccessContext(surface="public", scope="allowlist", subject="1")
    queries: list[dict[str, Any]] = [
        {"query": "kite festival"},
        {"query": "ferry schedule harbor", "limit": 20},
        {"query": "bake sale receipt", "limit": 5},
        {"query": "lighthouse"},
        {"query": '"estival"'},  # quoted: the trigram path, a mid-word substring
        {"query": '"sale rec"'},
        {"query": EMOJI[0]},  # the emoji path (LIKE on Postgres)
        {"query": "cider market", "people": [alice]},
        {"query": "orchard trail", "people": [carol]},
        {"query": "picnic ridge", "after": "2023-06-01", "before": "2024-01-01"},
        {"query": "quilt raffle", "has_attachment": True},
        {"query": "canoe portage", "has_attachment": False},
        {"query": "nothing matches zzyzx"},
    ]
    battery: list[tuple[AccessContext, dict[str, Any]]] = []
    for context in (LOCAL_FULL_ACCESS, public_full, allowlist):
        battery.extend((context, q) for q in queries)
    return battery


def _answer(
    service: RetrievalService, context: AccessContext, params: dict[str, Any]
) -> tuple[SearchMessagesResult | str, bytes]:
    """The result and the bytes the public server would send for it — or,
    for a search that is refused (a person the scope cannot see), the
    error, which has to match just as exactly."""
    try:
        result = service.search_messages(context, **params)
        payload = handlers.search_messages(service, context, params)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return error, error.encode()
    return result, json.dumps(payload, default=str).encode()


def test_pooled_channels_return_exactly_what_sequential_channels_return(
    scratch_db: psycopg.Connection,
    shared_fts: apsw.Connection,
    corpus: Corpus,
    pools_factory: Callable[[int, int], RetrievalConnections],
) -> None:
    sequential = make_service(scratch_db, shared_fts)
    roomy = make_service(scratch_db, shared_fts, pools_factory(10, 4))
    starved = make_service(scratch_db, shared_fts, pools_factory(1, 1))

    compared = 0
    nonempty = 0
    refused = 0
    for context, params in _search_battery(corpus):
        expected, expected_bytes = _answer(sequential, context, params)
        for label, service in (("roomy", roomy), ("starved", starved)):
            got, got_bytes = _answer(service, context, params)
            assert got == expected, (label, context, params)
            assert got_bytes == expected_bytes, (label, context, params)
        compared += 1
        if isinstance(expected, str):
            refused += 1
        else:
            nonempty += bool(expected.results)
    assert compared == 39
    # The battery exercised real results, not a pile of empty answers...
    assert nonempty >= 20, (nonempty, refused)
    # ...and every channel contributed somewhere.
    lists = []
    for q in ("kite festival", '"estival"', "receipt ticket"):
        answer = _answer(sequential, LOCAL_FULL_ACCESS, {"query": q})[0]
        assert isinstance(answer, SearchMessagesResult)
        lists.append(answer.candidate_lists)
    for channel in (
        "segment_fts",
        "attachment_fts",
        "segment_vector",
        "attachment_vector",
        "multimodal_vector",
    ):
        assert any(cl[channel] > 0 for cl in lists), channel


def test_pooled_get_conversation_list_people_and_attachment_text_match_too(
    scratch_db: psycopg.Connection,
    shared_fts: apsw.Connection,
    corpus: Corpus,
    pools_factory: Callable[[int, int], RetrievalConnections],
) -> None:
    sequential = make_service(scratch_db, shared_fts)
    pooled = make_service(scratch_db, shared_fts, pools_factory(4, 2))
    with scratch_db.cursor() as cur:
        cur.execute("SELECT attachment_key FROM attachment ORDER BY attachment_id LIMIT 3")
        attachment_keys = [str(r[0]) for r in cur.fetchall()]
    for context in (LOCAL_FULL_ACCESS, AccessContext(surface="public", scope="allowlist", subject="1")):
        for thread_key in corpus.thread_keys:
            args = {"thread_id": thread_key, "window": 5}
            try:
                expected: object = sequential.get_conversation(context, **args)
            except Exception as exc:  # NOT_FOUND under allowlist is an answer too
                expected = type(exc).__name__
            try:
                got: object = pooled.get_conversation(context, **args)
            except Exception as exc:
                got = type(exc).__name__
            assert got == expected
        assert pooled.list_people(context, limit=50) == sequential.list_people(context, limit=50)
        for key in attachment_keys:
            try:
                want: object = sequential.get_attachment_text(context, attachment_key=key)
            except Exception as exc:
                want = type(exc).__name__
            try:
                have: object = pooled.get_attachment_text(context, attachment_key=key)
            except Exception as exc:
                have = type(exc).__name__
            assert have == want


# ---------------------------------------------------------------------------
# Concurrency is real
# ---------------------------------------------------------------------------

CHANNEL_DELAY_SECONDS = 0.25

_CHANNELS: tuple[tuple[Any, str], ...] = (
    (fts_search, "search_segment_fts"),
    (fts_search, "search_attachment_chunk_fts"),
    (vector_search, "search_segment_vector"),
    (vector_search, "search_attachment_chunk_vector"),
    (vector_search, "search_multimodal_vector"),
)


@pytest.fixture
def slow_channels(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every channel sleeps a fixed delay first, and records the thread it
    ran on. The service calls channels through their modules, as the bench
    script's instrumentation relies on."""
    threads: list[str] = []
    lock = threading.Lock()
    for module, name in _CHANNELS:
        real = getattr(module, name)

        def slowed(*args: Any, _real: Any = real, **kwargs: Any) -> Any:
            with lock:
                threads.append(threading.current_thread().name)
            time.sleep(CHANNEL_DELAY_SECONDS)
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, slowed)
    return threads


def test_the_channels_run_side_by_side_on_worker_threads(
    scratch_db: psycopg.Connection,
    shared_fts: apsw.Connection,
    corpus: Corpus,
    pools_factory: Callable[[int, int], RetrievalConnections],
    slow_channels: list[str],
) -> None:
    del corpus
    sequential = make_service(scratch_db, shared_fts)
    pooled = make_service(scratch_db, shared_fts, pools_factory(10, 4))

    started = time.perf_counter()
    expected = sequential.search_messages(LOCAL_FULL_ACCESS, query="kite festival")
    sequential_seconds = time.perf_counter() - started
    sequential_threads = list(slow_channels)
    slow_channels.clear()

    started = time.perf_counter()
    got = pooled.search_messages(LOCAL_FULL_ACCESS, query="kite festival")
    pooled_seconds = time.perf_counter() - started

    assert got == expected
    assert sequential_seconds >= 5 * CHANNEL_DELAY_SECONDS
    assert pooled_seconds < 2 * CHANNEL_DELAY_SECONDS
    assert len(set(sequential_threads)) == 1  # all on the caller's thread
    assert len(slow_channels) == 5
    assert all(name.startswith(CHANNEL_THREAD_PREFIX) for name in slow_channels)
    assert len(set(slow_channels)) == 5  # five channels, five threads at once


def test_a_starved_pool_runs_every_channel_on_the_calls_own_connection(
    scratch_db: psycopg.Connection,
    shared_fts: apsw.Connection,
    corpus: Corpus,
    pools_factory: Callable[[int, int], RetrievalConnections],
    slow_channels: list[str],
) -> None:
    """One connection in the pool: the call holds it, so no channel can
    borrow another — each runs on the call's own, in order, and the search
    completes rather than waiting on itself."""
    del corpus
    pooled = make_service(scratch_db, shared_fts, pools_factory(1, 1))
    caller = threading.current_thread().name

    result = pooled.search_messages(LOCAL_FULL_ACCESS, query="kite festival")

    assert result.results
    assert slow_channels == [caller] * 5


def test_a_call_no_longer_waits_for_another_calls_search(
    scratch_db: psycopg.Connection,
    shared_fts: apsw.Connection,
    corpus: Corpus,
    pools_factory: Callable[[int, int], RetrievalConnections],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pooled: a `get_conversation` answers while a search is held inside a
    channel. Shared: the same call waits for the whole search, because the
    two share one connection behind the lock. The hold is released by a
    timer after `HOLD_SECONDS`."""
    hold_seconds = 1.0
    held = threading.Event()
    release = threading.Event()
    real = vector_search.search_segment_vector

    def holding(*args: Any, **kwargs: Any) -> Any:
        held.set()
        assert release.wait(timeout=30), "the search was never released"
        return real(*args, **kwargs)

    monkeypatch.setattr(vector_search, "search_segment_vector", holding)

    def seconds_to_answer_during_a_search(service: RetrievalService) -> float:
        held.clear()
        release.clear()
        searcher = threading.Thread(
            target=lambda: service.search_messages(LOCAL_FULL_ACCESS, query="kite festival")
        )
        searcher.start()
        assert held.wait(timeout=30)
        timer = threading.Timer(hold_seconds, release.set)
        timer.start()
        started = time.perf_counter()
        service.get_conversation(LOCAL_FULL_ACCESS, thread_id=corpus.thread_keys[0], window=3)
        elapsed = time.perf_counter() - started
        timer.join()
        searcher.join(timeout=30)
        return elapsed

    pooled = seconds_to_answer_during_a_search(
        make_service(scratch_db, shared_fts, pools_factory(10, 4))
    )
    shared = seconds_to_answer_during_a_search(make_service(scratch_db, shared_fts))

    assert pooled < hold_seconds / 2
    assert shared >= hold_seconds * 0.8
