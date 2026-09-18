"""Live-database integration tests for the retrieval service (SPEC
§9.4) — real Postgres *and* real SQLite FTS5 throughout, same skip
pattern as `tests/test_migrations_integration.py`/
`tests/test_fts_sync_integration.py`.

Three properties the build task calls out explicitly get dedicated
tests here:

1. **Query-side normalization matches ingest-time normalization
   exactly** (SPEC §9.2, D2) — an apostrophe typed one way in the
   corpus and another way in the query must still match, exercised
   against the real FTS5 engine (`test_bm25_search_survives_apostrophe_
   style_mismatch`).
2. **The trigram/exact-phrase path is genuinely distinct from BM25**
   (SPEC §7.3/D2) — a quoted query finds a mid-word substring BM25
   cannot (`test_trigram_finds_a_substring_bm25_cannot`).
3. **A selective filter cannot silently starve the result set** (SPEC
   §9.4 step 5, D6 — the exact bug a naive top-K-then-filter
   implementation has) — `test_filtered_retrieval_does_not_starve_
   on_a_selective_filter` constructs the starvation scenario directly
   and shows the overfetch defeats it.
"""

from __future__ import annotations

import math
import os
import random
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import apsw
import psycopg
import pytest

from imsg import constants
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.fts.schema import create_schema
from imsg.embed.fts.sync import upsert_segment_row
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.embed.vector_codec import vector_literal
from imsg.keys import message_key as derive_message_key
from imsg.keys import thread_key as derive_thread_key
from imsg.retrieval.access import LOCAL_FULL_ACCESS, AccessContext, resolve_request_scope
from imsg.retrieval.errors import (
    DateRangeInvalidError,
    NotEnrichedError,
    NotFoundError,
    PersonAmbiguousError,
    PersonNotFoundError,
)
from imsg.retrieval.filters import SearchFilters, compile_predicate, resolve_filters
from imsg.retrieval.fts_search import search_segment_fts
from imsg.retrieval.people import resolve_person
from imsg.retrieval.query import analyze_query
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.service import RetrievalService
from imsg.retrieval.vector_search import _best_distance_per_segment, search_multimodal_vector

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_retrieval_test"
FULL = resolve_request_scope(None, LOCAL_FULL_ACCESS)

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


@pytest.fixture
def fts_conn(tmp_path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    return conn


class _Cfg:
    """A minimal stand-in for `imsg.config.schema.Config` carrying only
    the fields `RetrievalService` actually reads — avoids constructing
    a full validated `Config` (which needs secrets/paths that don't
    exist in this test environment) just to exercise retrieval logic."""

    class _Retrieval:
        k_fts = 100
        k_vector = 100
        rrf_k = 60
        rerank_top = 50
        default_limit = 10
        hnsw_ef_search = 1000

    class _Render:
        timezone = "UTC"
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


# --- Postgres-side fixture helpers ------------------------------------


def _insert_person(
    conn: psycopg.Connection, display_name: str, *, is_owner: bool = False
) -> tuple[int, str]:
    short_name = f"{display_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
            "VALUES (%s, %s, %s, false) RETURNING person_id",
            (display_name, short_name, is_owner),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0]), short_name


def _insert_chat(
    conn: psycopg.Connection, *, participants: list[int], kind: str = "dm", display_name: str | None = None
) -> tuple[int, str]:
    guid = f"chat-{uuid.uuid4()}"
    tkey = derive_thread_key(guid)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind, display_name) "
            "VALUES (%s, %s, %s, %s) RETURNING chat_id",
            (guid, tkey, kind, display_name),
        )
        row = cur.fetchone()
        assert row is not None
        chat_id = int(row[0])
        for person_id in participants:
            cur.execute(
                "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)",
                (chat_id, person_id),
            )
    return chat_id, tkey


def _insert_session(
    conn: psycopg.Connection, chat_id: int, *, started_at: datetime, ended_at: datetime
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
            "VALUES (%s, %s, %s, 3.0) RETURNING session_id",
            (chat_id, started_at, ended_at),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_segment(
    conn: psycopg.Connection,
    *,
    chat_id: int,
    session_id: int,
    rendered_text: str,
    started_at: datetime,
    ended_at: datetime,
    seq: int = 0,
) -> tuple[int, str]:
    stable_key = f"stable-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO segment (
                stable_key, chat_id, session_id, seq_in_session, started_at, ended_at,
                message_count, token_count, rendered_text, rendered_sha256, seg_config_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, 1, 10, %s, 'x', 'cfg-hash')
            RETURNING segment_id
            """,
            (stable_key, chat_id, session_id, seq, started_at, ended_at, rendered_text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0]), stable_key


def _insert_message(
    conn: psycopg.Connection,
    *,
    chat_id: int,
    segment_id: int,
    sender_person_id: int,
    is_from_me: bool,
    sent_at: datetime,
    text: str,
    is_unsent: bool = False,
) -> tuple[int, str]:
    guid = f"msg-{uuid.uuid4()}"
    mkey = derive_message_key(guid)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO message (
                source_guid, message_key, chat_id, sender_person_id, is_from_me, sent_at,
                service, text_original, text_normalized, is_unsent
            ) VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s, %s)
            RETURNING message_id
            """,
            (guid, mkey, chat_id, sender_person_id, is_from_me, sent_at, text, text, is_unsent),
        )
        row = cur.fetchone()
        assert row is not None
        message_id = int(row[0])
        cur.execute(
            "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
            (segment_id, message_id),
        )
    return message_id, mkey


def _make_segment(
    conn: psycopg.Connection,
    fts_conn: apsw.Connection,
    *,
    chat_id: int,
    rendered_text: str,
    started_at: datetime,
    person_id: int,
    is_from_me: bool = False,
) -> tuple[int, str]:
    """Convenience: one session + one segment + one linked message +
    FTS indexing, in one call — most tests below need exactly this
    shape and nothing more elaborate."""
    session_id = _insert_session(conn, chat_id, started_at=started_at, ended_at=started_at)
    segment_id, stable_key = _insert_segment(
        conn,
        chat_id=chat_id,
        session_id=session_id,
        rendered_text=rendered_text,
        started_at=started_at,
        ended_at=started_at,
    )
    _insert_message(
        conn,
        chat_id=chat_id,
        segment_id=segment_id,
        sender_person_id=person_id,
        is_from_me=is_from_me,
        sent_at=started_at,
        text=rendered_text,
    )
    upsert_segment_row(fts_conn, segment_id, stable_key, rendered_text)
    return segment_id, stable_key


def _messages(out: dict[str, object]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], out["messages"])


def _people(out: dict[str, object]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], out["people"])


def _allowlist(conn: psycopg.Connection, person_id: int, *, text_allowed: bool = True) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO allowlist_person (person_id, text_allowed) VALUES (%s, %s)",
            (person_id, text_allowed),
        )


# ==========================================================================
# 1. Query normalization trap — real SQLite FTS5
# ==========================================================================


def test_bm25_search_survives_apostrophe_style_mismatch(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    """SPEC §9.2/D2: the corpus was typed on iOS (curly U+2019
    apostrophe); the searcher types a straight U+0027 apostrophe. If
    query-side normalization ever drifted from ingest-time
    normalization, this would return zero hits."""
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])

    ios_curly_apostrophe = "\u2019"  # RIGHT SINGLE QUOTATION MARK, via escape to avoid RUF001
    rendered = f"I can{ios_curly_apostrophe}t make it Friday"
    segment_id, _ = _make_segment(
        scratch_db,
        fts_conn,
        chat_id=chat_id,
        rendered_text=rendered,
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        person_id=alice_id,
    )

    query = analyze_query("I can't make it Friday")  # straight apostrophe, as typed
    assert query.mode == "bm25"
    predicate = compile_predicate(SearchFilters(), FULL)
    result = search_segment_fts(fts_conn, scratch_db, query, predicate, k=10)

    assert segment_id in result.segment_ids


# ==========================================================================
# 2. Trigram/exact-phrase path is genuinely distinct from BM25
# ==========================================================================


def test_trigram_finds_a_substring_bm25_cannot(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])

    hyphenated_id, _ = _make_segment(
        scratch_db,
        fts_conn,
        chat_id=chat_id,
        rendered_text="attached bid-rev3.pdf for review",
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        person_id=alice_id,
    )
    plain_id, _ = _make_segment(
        scratch_db,
        fts_conn,
        chat_id=chat_id,
        rendered_text="let's discuss the bid over lunch",
        started_at=datetime(2024, 1, 2, tzinfo=UTC),
        person_id=alice_id,
    )

    predicate = compile_predicate(SearchFilters(), FULL)

    # unicode61 tokenizes "bid-rev3" into "bid"/"rev3" — the literal
    # substring "id-rev" spans a token boundary and is never itself a
    # token, so a BM25 (word-tokenized) search for it finds nothing.
    bm25_query = analyze_query("id-rev")
    assert bm25_query.mode == "bm25"
    bm25_result = search_segment_fts(fts_conn, scratch_db, bm25_query, predicate, k=10)
    assert hyphenated_id not in bm25_result.segment_ids
    assert plain_id not in bm25_result.segment_ids

    # The trigram tokenizer indexes at the character level, so the same
    # substring, quoted (routing to trigram per D2), *is* found — and
    # only in the segment that actually contains it.
    trigram_query = analyze_query('"id-rev"')
    assert trigram_query.mode == "trigram"
    trigram_result = search_segment_fts(fts_conn, scratch_db, trigram_query, predicate, k=10)
    assert hyphenated_id in trigram_result.segment_ids
    assert plain_id not in trigram_result.segment_ids


# ==========================================================================
# 3. Filtered retrieval must not starve on a selective filter
# ==========================================================================


def test_filtered_retrieval_does_not_starve_on_a_selective_filter(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    """SPEC §9.4 step 5 / D6: "v1.0's post-filtering could silently
    starve `k`." Builds the exact failure shape a naive
    fetch-top-k-then-filter implementation has: many segments match
    the FTS query about equally well; only one of them is eligible
    under a restrictive person filter, and that one is made to rank
    *worse* than a naive small top-k cutoff. `search_segment_fts`'s
    overfetch (fetch up to `scan_cap`, authorize the whole batch, take
    the first `k` eligible) must still surface it; a bare
    "fetch k, then filter" implementation would not.
    """
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    bob_id, bob_short = _insert_person(scratch_db, "Bob Builder")
    other_ids = [_insert_person(scratch_db, f"Person {i}")[0] for i in range(9)]

    eligible_chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, bob_id])
    eligible_segment_id, _ = _make_segment(
        scratch_db,
        fts_conn,
        chat_id=eligible_chat_id,
        # Extra filler dilutes BM25 term-frequency ratio relative to the
        # other (short, dense) segments below, pushing this segment's
        # raw rank toward the back of the pack deterministically.
        rendered_text=(
            "deck project notes and miscellaneous filler words padding "
            "this message out considerably so its relevance score is "
            "lower than the terse ones deck deck"
        ),
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        person_id=bob_id,
    )

    for i, other_id in enumerate(other_ids):
        chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, other_id])
        _make_segment(
            scratch_db,
            fts_conn,
            chat_id=chat_id,
            rendered_text="deck deck deck",  # short and dense: outranks the eligible one
            started_at=datetime(2024, 1, 2 + i, tzinfo=UTC),
            person_id=other_id,
        )

    query = analyze_query("deck")
    k = 3

    # --- Reproduce the bug: fetch only k raw candidates, then authorize.
    from imsg.retrieval.fts_search import _authorize_segment_ids, _raw_segment_candidates

    people_filter = resolve_filters(
        scratch_db,
        FULL,
        people=[bob_short],
        after=None,
        before=None,
        has_attachment=None,
        timezone="UTC",
    )
    predicate = compile_predicate(people_filter, FULL)

    naive_raw = _raw_segment_candidates(fts_conn, query, k)  # no overfetch at all
    naive_eligible = _authorize_segment_ids(scratch_db, naive_raw, predicate)
    assert eligible_segment_id not in naive_eligible, (
        "test setup didn't actually reproduce the starvation shape — the "
        "eligible segment must rank outside the naive top-k"
    )

    # --- This module's real implementation must not have that bug.
    result = search_segment_fts(fts_conn, scratch_db, query, predicate, k=k)
    assert eligible_segment_id in result.segment_ids
    assert len(result.segment_ids) <= k


def test_scan_cap_reached_is_reported_when_the_pool_is_exhausted(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When even the overfetched pool doesn't contain k eligible
    results, `scan_cap_reached` must be True (SPEC §9.4 step 5: "emit a
    metric") rather than silently returning fewer with no signal.

    Since `scan_cap = max(k * MULTIPLIER, MIN_SCAN_CAP)` is always
    `>= k` by construction, a *full-scope* search can never observe
    "capped raw fetch but still short of k" — every raw candidate is
    eligible under full scope, so hitting the cap always yields >= k
    eligible too. The cap can only bind visibly behind a restrictive
    filter: this test makes `cap` real matches exist, none of which
    are eligible under a `people` filter that excludes all of them, so
    the raw fetch is capped (there could be more matches beyond it)
    *and* eligible is 0 < k.
    """
    import imsg.retrieval.fts_search as fts_search_module

    monkeypatch.setattr(fts_search_module, "MIN_SCAN_CAP", 3)
    monkeypatch.setattr(fts_search_module, "SCAN_CAP_MULTIPLIER", 1)

    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    _, bob_short = _insert_person(scratch_db, "Bob Builder")
    excluded_id, _ = _insert_person(scratch_db, "Excluded Person")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, excluded_id])
    k = 3  # cap = max(k*1, 3) = 3
    for i in range(k):
        _make_segment(
            scratch_db,
            fts_conn,
            chat_id=chat_id,
            rendered_text="deck deck deck",
            started_at=datetime(2024, 1, 1 + i, tzinfo=UTC),
            person_id=excluded_id,
        )

    query = analyze_query("deck")
    people_filter = resolve_filters(
        scratch_db, FULL, people=[bob_short], after=None, before=None,
        has_attachment=None, timezone="UTC",
    )
    predicate = compile_predicate(people_filter, FULL)

    result = search_segment_fts(fts_conn, scratch_db, query, predicate, k=k)
    assert result.segment_ids == ()
    assert result.scan_cap_reached is True


# ==========================================================================
# Channel C streams its overfetch and stops early — same answer as reading it all
# ==========================================================================


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def test_multimodal_channel_early_stop_matches_collapsing_every_row(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real pgvector HNSW scan, real server-side cursor: bursts of
    near-identical photos per segment (the shape that makes the 500-row
    overfetch mostly duplicates), fetched a few rows at a time so the
    early stop has to decide across several round trips."""
    import imsg.retrieval.vector_search as vector_search_module

    fetched: list[int] = []
    real_fetchmany = psycopg.ServerCursor.fetchmany

    def counting_fetchmany(self: psycopg.ServerCursor[Any], size: int = 0) -> list[Any]:
        rows = real_fetchmany(self, size)
        fetched.append(len(rows))
        return rows

    monkeypatch.setattr(vector_search_module, "COLLAPSE_FETCH_ROWS", 4)
    monkeypatch.setattr(psycopg.ServerCursor, "fetchmany", counting_fetchmany)

    rng = random.Random(20260916)
    dim = constants.MULTIMODAL_EMBEDDING_DIM
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id])
    base = datetime(2025, 6, 1, tzinfo=UTC)
    for i in range(40):
        started = base + timedelta(hours=i)
        segment_id, _ = _make_segment(
            scratch_db,
            fts_conn,
            chat_id=chat_id,
            rendered_text=f"photo burst {i}",
            started_at=started,
            person_id=owner_id,
        )
        message_id, _ = _insert_message(
            scratch_db,
            chat_id=chat_id,
            segment_id=segment_id,
            sender_person_id=owner_id,
            is_from_me=True,
            sent_at=started + timedelta(minutes=1),
            text="",
        )
        centre = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        with scratch_db.cursor() as cur:
            for burst in range(rng.randint(1, 6)):
                cur.execute(
                    "INSERT INTO attachment (source_guid, attachment_key) VALUES (%s, %s) "
                    "RETURNING attachment_id",
                    (f"att-{uuid.uuid4()}", f"attachment-key-{uuid.uuid4().hex}"),
                )
                row = cur.fetchone()
                assert row is not None
                attachment_id = int(row[0])
                cur.execute(
                    "INSERT INTO message_attachment (message_id, attachment_id, ordinal) "
                    "VALUES (%s, %s, %s)",
                    (message_id, attachment_id, burst),
                )
                vec = _unit([c + rng.gauss(0.0, 0.05) for c in centre])
                cur.execute(
                    "INSERT INTO attachment_mm_embedding (attachment_id, model, media_sha256, vec) "
                    "VALUES (%s, 'fake/multimodal@test', %s, %s::halfvec)",
                    (attachment_id, uuid.uuid4().hex, vector_literal(vec)),
                )

    predicate = compile_predicate(SearchFilters(), FULL)
    for k in (3, 10, 60):
        query = _unit([rng.gauss(0.0, 1.0) for _ in range(dim)])
        fetched.clear()
        streamed = search_multimodal_vector(scratch_db, query, predicate, k)
        row_limit = max(k * 5, 50)
        with scratch_db.transaction(), scratch_db.cursor() as cur:
            # The same scan settings the channel itself applies, so the two
            # run the same plan and any difference is the streaming, not the
            # planner (imsg.retrieval.vector_search).
            cur.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                """
                SELECT s.segment_id, mm.vec <=> %(qv)s::halfvec AS distance
                FROM attachment_mm_embedding mm
                JOIN message_attachment ma ON ma.attachment_id = mm.attachment_id
                JOIN segment_message sm ON sm.message_id = ma.message_id
                JOIN segment s ON s.segment_id = sm.segment_id
                ORDER BY mm.vec <=> %(qv)s::halfvec
                LIMIT %(n)s
                """,
                {"qv": vector_literal(query), "n": row_limit},
            )
            every_row = [(int(r[0]), float(r[1])) for r in cur.fetchall()]
        assert streamed == _best_distance_per_segment(every_row, k)
        if k < 40:
            assert len(streamed.segment_ids) == k
            assert sum(fetched) < len(every_row), "a small k settles before the overfetch runs out"
        else:
            assert streamed.scan_cap_reached is True  # only 40 segments exist


# ==========================================================================
# Person resolution ladder (SPEC §9.4 step 1, D6)
# ==========================================================================


def test_resolve_person_exact_short_name(scratch_db: psycopg.Connection) -> None:
    person_id, short_name = _insert_person(scratch_db, "Alice Example")
    assert resolve_person(scratch_db, FULL, short_name) == person_id


def test_resolve_person_exact_display_name(scratch_db: psycopg.Connection) -> None:
    person_id, _ = _insert_person(scratch_db, "Unique Display Name")
    assert (
        resolve_person(scratch_db, FULL, "Unique Display Name") == person_id
    )


def test_resolve_person_not_found(scratch_db: psycopg.Connection) -> None:
    with pytest.raises(PersonNotFoundError):
        resolve_person(scratch_db, FULL, "nobody-like-this-exists-zzz")


def test_resolve_person_fuzzy_never_silently_picks(scratch_db: psycopg.Connection) -> None:
    _insert_person(scratch_db, "Alice Example")
    with pytest.raises(PersonAmbiguousError) as exc_info:
        resolve_person(scratch_db, FULL, "Alise Example")  # one-letter typo
    assert len(exc_info.value.candidates) >= 1


def test_resolve_person_ambiguous_on_duplicate_display_names(scratch_db: psycopg.Connection) -> None:
    _insert_person(scratch_db, "Duplicate Name")
    _insert_person(scratch_db, "Duplicate Name")
    with pytest.raises(PersonAmbiguousError):
        resolve_person(scratch_db, FULL, "Duplicate Name")


# ==========================================================================
# RetrievalService.search_messages — hybrid flow end to end
# ==========================================================================


@pytest.fixture
def service(scratch_db: psycopg.Connection, fts_conn: apsw.Connection) -> RetrievalService:
    return RetrievalService(
        pg_conn=scratch_db,
        fts_conn=fts_conn,
        config=_Cfg(),  # type: ignore[arg-type]
        text_provider=FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM),
        reranker=FakeRerankerProvider(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM),
    )


def test_search_messages_returns_a_matching_segment_with_expected_shape(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice Example")
    chat_id, thread_key = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _, stable_key = _make_segment(
        scratch_db,
        fts_conn,
        chat_id=chat_id,
        rendered_text="the deck rebuild bid came in at fourteen thousand",
        started_at=datetime(2024, 4, 12, 13, 58, tzinfo=UTC),
        person_id=alice_id,
    )

    result = service.search_messages(LOCAL_FULL_ACCESS, query="deck rebuild bid")

    assert len(result.results) == 1
    hit = result.results[0]
    assert hit["segment_key"] == stable_key
    assert hit["thread_key"] == thread_key
    assert hit["untrusted_content"] is True
    assert str(hit["text"]).startswith("the deck rebuild bid")
    assert result.candidate_lists["segment_fts"] >= 1
    assert isinstance(result.scan_cap_reached, bool)


def test_search_messages_reranks_at_least_limit_candidates_when_rerank_top_is_smaller(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    """`retrieval.rerank_top` is a latency setting, not a result cap: with
    a pool of 2 and `limit=5`, all five matching segments come back, and
    every one of them went through the reranker."""
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id])
    for i in range(6):
        _make_segment(
            scratch_db,
            fts_conn,
            chat_id=chat_id,
            rendered_text=f"harbor kite festival notes part {i}",
            started_at=datetime(2024, 5, 1 + i, 9, 0, tzinfo=UTC),
            person_id=owner_id,
        )

    scored: list[int] = []

    class _CountingReranker(FakeRerankerProvider):
        def score(self, query: str, documents: list[str]) -> list[float]:
            scored.append(len(documents))
            return super().score(query, documents)

    class _SmallPoolCfg(_Cfg):
        class _Retrieval(_Cfg._Retrieval):
            rerank_top = 2

        retrieval = _Retrieval()

    small_pool = RetrievalService(
        pg_conn=scratch_db,
        fts_conn=fts_conn,
        config=_SmallPoolCfg(),  # type: ignore[arg-type]
        text_provider=FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM),
        reranker=_CountingReranker(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM),
    )
    assert len(small_pool.search_messages(LOCAL_FULL_ACCESS, query="kite festival", limit=5).results) == 5
    assert scored == [5]
    # at or below rerank_top, the pool is rerank_top
    scored.clear()
    assert len(small_pool.search_messages(LOCAL_FULL_ACCESS, query="kite festival", limit=1).results) == 1
    assert scored == [2]


def test_search_messages_empty_result_is_success_not_error(
    service: RetrievalService,
) -> None:
    result = service.search_messages(LOCAL_FULL_ACCESS, query="absolutely nothing matches this")
    assert result.results == []


def test_search_messages_date_range_invalid_raises(service: RetrievalService) -> None:
    with pytest.raises(DateRangeInvalidError):
        service.search_messages(
            LOCAL_FULL_ACCESS, query="x", after="2024-06-01", before="2024-01-01"
        )


def test_search_messages_scope_allowlist_excludes_ineligible_thread(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice Example")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _make_segment(
        scratch_db,
        fts_conn,
        chat_id=chat_id,
        rendered_text="totally unique needle phrase xyzzyplugh",
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        person_id=alice_id,
    )
    # Neither owner nor alice is allowlisted — the thread is ineligible.
    allowlist_ctx = AccessContext(surface="public", scope="allowlist", subject="12345")

    result = service.search_messages(allowlist_ctx, query="xyzzyplugh")
    assert result.results == []

    # Once both participants are allowlisted, it becomes visible.
    _allowlist(scratch_db, owner_id)
    _allowlist(scratch_db, alice_id)
    result = service.search_messages(allowlist_ctx, query="xyzzyplugh")
    assert len(result.results) == 1


# ==========================================================================
# get_conversation
# ==========================================================================


def test_get_conversation_by_segment_key_defaults_anchor_to_segment_start(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    session_id = _insert_session(
        scratch_db, chat_id, started_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        ended_at=datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
    )
    segment_id, stable_key = _insert_segment(
        scratch_db, chat_id=chat_id, session_id=session_id,
        rendered_text="...", started_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        ended_at=datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
    )
    _, mkey1 = _insert_message(
        scratch_db, chat_id=chat_id, segment_id=segment_id, sender_person_id=alice_id,
        is_from_me=False, sent_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC), text="hello",
    )
    _, mkey2 = _insert_message(
        scratch_db, chat_id=chat_id, segment_id=segment_id, sender_person_id=owner_id,
        is_from_me=True, sent_at=datetime(2024, 1, 1, 12, 5, tzinfo=UTC), text="hi back",
    )

    out = service.get_conversation(LOCAL_FULL_ACCESS, thread_id=stable_key)
    keys = [m["message_key"] for m in _messages(out)]
    assert mkey1 in keys
    assert mkey2 in keys


def test_get_conversation_anchor_by_message_key_and_window(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    session_id = _insert_session(
        scratch_db, chat_id, started_at=datetime(2024, 1, 1, tzinfo=UTC),
        ended_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    segment_id, stable_key = _insert_segment(
        scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="...",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), ended_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    keys = []
    base = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    for i in range(7):
        _, mkey = _insert_message(
            scratch_db, chat_id=chat_id, segment_id=segment_id, sender_person_id=alice_id,
            is_from_me=False, sent_at=base + timedelta(minutes=i), text=f"message {i}",
        )
        keys.append(mkey)

    # Anchor on message 3 (0-indexed), window=2 -> messages 1..5.
    out = service.get_conversation(
        LOCAL_FULL_ACCESS, thread_id=stable_key, anchor=keys[3], window=2
    )
    got_keys = {m["message_key"] for m in _messages(out)}
    assert got_keys == set(keys[1:6])


def test_get_conversation_unknown_thread_raises_not_found(service: RetrievalService) -> None:
    with pytest.raises(NotFoundError):
        service.get_conversation(LOCAL_FULL_ACCESS, thread_id="no-such-thread-or-segment")


def test_get_conversation_allowlist_scope_hides_ineligible_thread(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, thread_key = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="secret",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    allowlist_ctx = AccessContext(surface="public", scope="allowlist", subject="12345")
    with pytest.raises(NotFoundError):
        service.get_conversation(allowlist_ctx, thread_id=thread_key)


# ==========================================================================
# list_people
# ==========================================================================


def test_list_people_query_filter_and_message_counts(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, alice_short = _insert_person(scratch_db, "Alice Example")
    _insert_person(scratch_db, "Bob Builder")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="hi",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )

    out = service.list_people(LOCAL_FULL_ACCESS, query="Alice")
    people = _people(out)
    names = [p["short_name"] for p in people]
    assert names == [alice_short]
    assert people[0]["message_count"] == 1
    assert "handles" not in people[0]


def test_list_people_include_handles_local_only(
    scratch_db: psycopg.Connection, service: RetrievalService
) -> None:
    person_id, short_name = _insert_person(scratch_db, "Alice Example")
    with scratch_db.cursor() as cur:
        cur.execute(
            "INSERT INTO handle (person_id, kind, normalized_value) VALUES (%s, 'email', %s)",
            (person_id, "alice@example.com"),
        )
    out = service.list_people(LOCAL_FULL_ACCESS, query=short_name, include_handles=True)
    assert _people(out)[0]["handles"] == ["alice@example.com"]


def test_list_people_allowlist_scope_hides_non_allowlisted(
    scratch_db: psycopg.Connection, service: RetrievalService
) -> None:
    _insert_person(scratch_db, "Not Allowlisted Person")
    allowlist_ctx = AccessContext(surface="public", scope="allowlist", subject="12345")
    out = service.list_people(allowlist_ctx, query="Not Allowlisted")
    assert out["people"] == []


# ==========================================================================
# get_attachment_text
# ==========================================================================


def _insert_attachment(conn: psycopg.Connection) -> tuple[int, str]:
    guid = f"att-{uuid.uuid4()}"
    key = f"attkey-{uuid.uuid4().hex}" + "0" * 16
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key, filename, mime_type) "
            "VALUES (%s, %s, 'bid.pdf', 'application/pdf') RETURNING attachment_id",
            (guid, key),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0]), key


def _link_attachment_to_message(conn: psycopg.Connection, *, message_id: int, attachment_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO message_attachment (message_id, attachment_id, ordinal) VALUES (%s, %s, 0)",
            (message_id, attachment_id),
        )


def test_get_attachment_text_returns_done_enrichment(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    segment_id, _ = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="see attached",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    with scratch_db.cursor() as cur:
        cur.execute("SELECT message_id FROM segment_message WHERE segment_id = %s", (segment_id,))
        message_id = cur.fetchone()[0]  # type: ignore[index]

    attachment_id, attachment_key = _insert_attachment(scratch_db)
    _link_attachment_to_message(scratch_db, message_id=message_id, attachment_id=attachment_id)
    with scratch_db.cursor() as cur:
        cur.execute(
            "INSERT INTO enrichment (attachment_id, kind, state, model, text) "
            "VALUES (%s, 'pdf_text', 'done', 'pdftotext', %s)",
            (attachment_id, "Deck rebuild, materials $14,200"),
        )

    out = service.get_attachment_text(LOCAL_FULL_ACCESS, attachment_key=attachment_key)
    assert out["attachment_key"] == attachment_key
    assert out["texts"] == [{"kind": "pdf_text", "model": "pdftotext", "text": "Deck rebuild, materials $14,200"}]
    assert out["untrusted_content"] is True


def test_get_attachment_text_not_enriched_yet(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    segment_id, _ = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="see attached",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    with scratch_db.cursor() as cur:
        cur.execute("SELECT message_id FROM segment_message WHERE segment_id = %s", (segment_id,))
        message_id = cur.fetchone()[0]  # type: ignore[index]
    attachment_id, attachment_key = _insert_attachment(scratch_db)
    _link_attachment_to_message(scratch_db, message_id=message_id, attachment_id=attachment_id)

    with pytest.raises(NotEnrichedError):
        service.get_attachment_text(LOCAL_FULL_ACCESS, attachment_key=attachment_key)


def test_get_attachment_text_unknown_key_not_found(service: RetrievalService) -> None:
    with pytest.raises(NotFoundError):
        service.get_attachment_text(LOCAL_FULL_ACCESS, attachment_key="no-such-attachment-key-at-all")


def test_get_attachment_text_allowlist_scope_unauthorized_is_not_found(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    """D6: an unauthorized key returns NOT_FOUND, not SCOPE_DENIED — no
    existence oracle."""
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    segment_id, _ = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="see attached",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    with scratch_db.cursor() as cur:
        cur.execute("SELECT message_id FROM segment_message WHERE segment_id = %s", (segment_id,))
        message_id = cur.fetchone()[0]  # type: ignore[index]
    attachment_id, attachment_key = _insert_attachment(scratch_db)
    _link_attachment_to_message(scratch_db, message_id=message_id, attachment_id=attachment_id)
    with scratch_db.cursor() as cur:
        cur.execute(
            "INSERT INTO enrichment (attachment_id, kind, state, model, text) "
            "VALUES (%s, 'pdf_text', 'done', 'pdftotext', 'text')",
            (attachment_id,),
        )

    allowlist_ctx = AccessContext(surface="public", scope="allowlist", subject="12345")
    with pytest.raises(NotFoundError):
        service.get_attachment_text(allowlist_ctx, attachment_key=attachment_key)


# ==========================================================================
# Two threads, one service (the public MCP surface's worker threads)
# ==========================================================================


CONCURRENT_SEARCH_THREADS = 4
SEARCHES_PER_THREAD = 3


def test_concurrent_searches_never_touch_a_connection_from_two_threads(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    """`imsg.mcp.tools.public_server` dispatches each tool call on a
    worker thread, so several can reach one `RetrievalService` at once.
    Neither connection it holds tolerates that, and both say so loudly:
    psycopg raises `OutOfOrderTransactionNesting` when two
    `conn.transaction()` blocks of different lengths exit out of order
    (and leaves the connection mid-transaction, breaking every request
    after it), and apsw raises `ThreadingViolationError` outright. The
    service serializes calls for exactly this reason.

    Written against the real drivers rather than fakes because both
    failures are driver behaviour: a fake connection would only assert
    what this test already assumes."""
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice Example")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    for day in range(6):
        _make_segment(
            scratch_db,
            fts_conn,
            chat_id=chat_id,
            rendered_text=f"the kite festival on the ridge, day {day}",
            started_at=datetime(2024, 5, 1 + day, 12, 0, tzinfo=UTC),
            person_id=alice_id,
        )

    failures: list[str] = []
    hit_counts: list[int] = []
    start = threading.Barrier(CONCURRENT_SEARCH_THREADS, timeout=30)

    def search_repeatedly() -> None:
        try:
            start.wait()
            for _ in range(SEARCHES_PER_THREAD):
                result = service.search_messages(LOCAL_FULL_ACCESS, query="kite festival")
                hit_counts.append(len(result.results))
        except BaseException as exc:  # the failure *is* the finding
            failures.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=search_repeatedly) for _ in range(CONCURRENT_SEARCH_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert failures == []
    assert len(hit_counts) == CONCURRENT_SEARCH_THREADS * SEARCHES_PER_THREAD
    # Every search saw the whole corpus: a query that ran with another
    # thread's transaction (or its `SET LOCAL` settings) underneath it
    # would not have to.
    assert set(hit_counts) == {6}
    # And nothing was left mid-transaction for the next request to trip on.
    assert scratch_db.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def test_concurrent_calls_across_different_tools_are_serialized(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    """The lock covers every entry point, not just `search_messages` —
    a `list_people` running against the same connection as a search is
    the same violation."""
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice Example")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _make_segment(
        scratch_db,
        fts_conn,
        chat_id=chat_id,
        rendered_text="the kite festival on the ridge",
        started_at=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
        person_id=alice_id,
    )

    failures: list[str] = []
    start = threading.Barrier(2, timeout=30)

    def run(call: Callable[[], object]) -> Callable[[], None]:
        def go() -> None:
            try:
                start.wait()
                for _ in range(SEARCHES_PER_THREAD):
                    call()
            except BaseException as exc:  # the failure *is* the finding
                failures.append(f"{type(exc).__name__}: {exc}")

        return go

    threads = [
        threading.Thread(
            target=run(lambda: service.search_messages(LOCAL_FULL_ACCESS, query="kite festival"))
        ),
        threading.Thread(target=run(lambda: service.list_people(LOCAL_FULL_ACCESS, limit=10))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert failures == []
    assert scratch_db.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
