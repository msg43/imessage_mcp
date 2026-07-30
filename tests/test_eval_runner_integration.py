"""Live-Postgres integration tests for the eval harness (SPEC §13):
the canonical-store I/O (`imsg.eval.io`), the local backend wired to a
real `RetrievalService`, the `GeminiEvalBackend` doc-id-to-segment_key
resolution, the runner end to end, and the AT-4 minimums check.

Same skip pattern as `tests/test_retrieval_integration.py` — real
Postgres and real SQLite FTS5, skipped cleanly when no scratch instance
is reachable.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import apsw
import psycopg
import pytest

from imsg import constants
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.fts.schema import create_schema
from imsg.embed.fts.sync import upsert_segment_row
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.eval.backend import GeminiEvalBackend, GeminiSearchClient, LocalEvalBackend
from imsg.eval.io import (
    label_segment_by_key,
    load_labels,
    load_queries,
    upsert_query,
)
from imsg.eval.models import EvalQuery
from imsg.eval.runner import compute_at4_check, run_eval
from imsg.keys import message_key as derive_message_key
from imsg.keys import thread_key as derive_thread_key
from imsg.retrieval.access import LOCAL_FULL_ACCESS
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.service import RetrievalService

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_eval_test"

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
    class _Retrieval:
        k_fts = 100
        k_vector = 100
        rrf_k = 60
        rerank_top = 50
        default_limit = 10

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


# --- fixture helpers (same shape as test_retrieval_integration.py) -------


def _insert_person(conn: psycopg.Connection, display_name: str, *, is_owner: bool = False) -> tuple[int, str]:
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


def _insert_chat(conn: psycopg.Connection, *, participants: list[int]) -> tuple[int, str]:
    guid = f"chat-{uuid.uuid4()}"
    tkey = derive_thread_key(guid)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind) VALUES (%s, %s, 'dm') RETURNING chat_id",
            (guid, tkey),
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


def _make_segment(
    conn: psycopg.Connection,
    fts_conn: apsw.Connection,
    *,
    chat_id: int,
    rendered_text: str,
    started_at: datetime,
    person_id: int,
) -> tuple[int, str, str]:
    """Returns (segment_id, stable_key, first_message_guid)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
            "VALUES (%s, %s, %s, 3.0) RETURNING session_id",
            (chat_id, started_at, started_at),
        )
        row = cur.fetchone()
        assert row is not None
        session_id = int(row[0])
        stable_key = f"stable-{uuid.uuid4()}"
        cur.execute(
            """
            INSERT INTO segment (
                stable_key, chat_id, session_id, seq_in_session, started_at, ended_at,
                message_count, token_count, rendered_text, rendered_sha256, seg_config_hash
            ) VALUES (%s, %s, %s, 0, %s, %s, 1, 10, %s, 'x', 'cfg-hash')
            RETURNING segment_id
            """,
            (stable_key, chat_id, session_id, started_at, started_at, rendered_text),
        )
        row = cur.fetchone()
        assert row is not None
        segment_id = int(row[0])

        guid = f"msg-{uuid.uuid4()}"
        mkey = derive_message_key(guid)
        cur.execute(
            """
            INSERT INTO message (
                source_guid, message_key, chat_id, sender_person_id, is_from_me, sent_at,
                service, text_original, text_normalized, is_unsent
            ) VALUES (%s, %s, %s, %s, false, %s, 'imessage', %s, %s, false)
            RETURNING message_id
            """,
            (guid, mkey, chat_id, person_id, started_at, rendered_text, rendered_text),
        )
        row = cur.fetchone()
        assert row is not None
        message_id = int(row[0])
        cur.execute(
            "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
            (segment_id, message_id),
        )
    upsert_segment_row(fts_conn, segment_id, stable_key, rendered_text)
    return segment_id, stable_key, guid


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


# ==========================================================================
# label_segment_by_key — anchors on the segment's first message GUID
# ==========================================================================


def test_label_segment_by_key_anchors_on_first_message_guid(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _, stable_key, first_guid = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="the deck bid",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="deck bid"))

    label = label_segment_by_key(scratch_db, query_id="q001", segment_key=stable_key, grade=2)
    assert label.anchor_guid == first_guid

    stored = load_labels(scratch_db, query_id="q001")
    assert len(stored) == 1
    assert stored[0].anchor_guid == first_guid
    assert stored[0].grade == 2


def test_label_segment_by_key_creates_query_when_query_text_if_new_given(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _, stable_key, _ = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="lunch plans",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    label_segment_by_key(
        scratch_db, query_id="adhoc:lunch", segment_key=stable_key, grade=1,
        query_text_if_new="lunch plans",
    )
    queries = load_queries(scratch_db)
    assert any(q.query_id == "adhoc:lunch" and q.query_text == "lunch plans" for q in queries)


def test_label_segment_by_key_unknown_segment_raises(scratch_db: psycopg.Connection) -> None:
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="x"))
    with pytest.raises(ValueError, match="does not resolve"):
        label_segment_by_key(scratch_db, query_id="q001", segment_key="no-such-segment", grade=1)


# ==========================================================================
# run_eval end to end, local target
# ==========================================================================


def test_run_eval_scores_a_query_against_its_label(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    _, stable_key, _ = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id,
        rendered_text="the deck rebuild bid came in at fourteen thousand",
        started_at=datetime(2024, 4, 12, tzinfo=UTC), person_id=alice_id,
    )
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="deck rebuild bid"))
    label_segment_by_key(scratch_db, query_id="q001", segment_key=stable_key, grade=2)

    backend = LocalEvalBackend(service=service, context=LOCAL_FULL_ACCESS)
    result = run_eval(scratch_db, backend, target="local", config_sha256="deadbeef", k=10)

    assert len(result.per_query) == 1
    qr = result.per_query[0]
    assert qr.query_id == "q001"
    assert stable_key in qr.ranked_segment_keys
    assert qr.resolved_grades == {stable_key: 2}
    assert qr.unresolved_label_count == 0
    assert qr.ndcg_at_k == pytest.approx(1.0)
    assert qr.recall_at_k == pytest.approx(1.0)
    assert qr.reciprocal_rank == pytest.approx(1.0)
    assert qr.success_at_k is True
    assert result.ndcg_at_k_mean == pytest.approx(1.0)


def test_run_eval_only_scores_queries_targeting_this_target(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    upsert_query(scratch_db, EvalQuery(query_id="q_local", query_text="a", targets=("local",)))
    upsert_query(scratch_db, EvalQuery(query_id="q_gemini_only", query_text="b", targets=("gemini",)))

    backend = LocalEvalBackend(service=service, context=LOCAL_FULL_ACCESS)
    result = run_eval(scratch_db, backend, target="local", config_sha256="x", k=10)
    assert [q.query_id for q in result.per_query] == ["q_local"]


def test_run_eval_reports_unresolved_labels_separately(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection, service: RetrievalService
) -> None:
    """A label whose anchor message no longer belongs to any segment
    (deleted / never segmented) must not silently vanish — it's counted
    as unresolved, not scored as a miss (SPEC §13.1)."""
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="ghost"))
    with scratch_db.cursor() as cur:
        cur.execute(
            "INSERT INTO relevance_label (query_id, anchor_guid, grade, source) "
            "VALUES ('q001', 'no-such-message-guid', 2, 'manual')"
        )
    backend = LocalEvalBackend(service=service, context=LOCAL_FULL_ACCESS)
    result = run_eval(scratch_db, backend, target="local", config_sha256="x", k=10)
    qr = result.per_query[0]
    assert qr.unresolved_label_count == 1
    assert qr.resolved_grades == {}
    assert qr.ndcg_at_k is None  # no resolved labels for this query


# ==========================================================================
# AT-4 minimums (SPEC §12)
# ==========================================================================


def test_at4_fails_with_too_few_queries(scratch_db: psycopg.Connection) -> None:
    upsert_query(scratch_db, EvalQuery(query_id="q001", query_text="x"))
    check = compute_at4_check(scratch_db, target="local")
    assert check.passed is False
    assert check.query_count == 1
    assert any("queries recorded" in r for r in check.reasons)


def test_at4_passes_when_all_minimums_met(scratch_db: psycopg.Connection) -> None:
    # 30 queries, each with >= 1 positive label; 100+ total pooled judgments;
    # at least one negative judgment somewhere.
    with scratch_db.cursor() as cur:
        for i in range(30):
            qid = f"q{i:03d}"
            cur.execute(
                "INSERT INTO eval_query (query_id, query_text, targets) "
                "VALUES (%s, %s, ARRAY['local'])",
                (qid, f"query {i}"),
            )
            # Each query gets 4 judgments: 3 positive-ish + 1 negative,
            # so pooled_judgment_count = 120 >= 100 and every query has
            # >= 1 positive.
            for j in range(4):
                grade = 0 if j == 0 else (j % 2) + 1
                cur.execute(
                    "INSERT INTO relevance_label (query_id, anchor_guid, grade, source) "
                    "VALUES (%s, %s, %s, 'manual')",
                    (qid, f"guid-{qid}-{j}", grade),
                )
    check = compute_at4_check(scratch_db, target="local")
    assert check.query_count == 30
    assert check.pooled_judgment_count == 120
    assert check.queries_with_a_positive == 30
    assert check.has_any_positive is True
    assert check.has_any_negative is True
    assert check.passed is True
    assert check.reasons == ()


# ==========================================================================
# GeminiEvalBackend — doc id -> segment_key resolution and de-dup
# ==========================================================================


class _FakeGeminiSearchClient:
    def __init__(self, doc_ids: list[str]) -> None:
        self._doc_ids = doc_ids

    def search(self, query_text: str, *, page_size: int) -> list[str]:
        del query_text, page_size
        return self._doc_ids


def _insert_export_document(conn: psycopg.Connection, *, document_id: str, segment_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO export_document (document_id, kind, segment_id, state)
            VALUES (%s, 'segment', %s, 'pushed')
            """,
            (document_id, segment_id),
        )


def test_gemini_backend_resolves_document_ids_to_segment_keys(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    segment_id, stable_key, _ = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="hi",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    document_id = "d" + "a" * 62
    _insert_export_document(scratch_db, document_id=document_id, segment_id=segment_id)

    backend = GeminiEvalBackend(conn=scratch_db, client=_FakeGeminiSearchClient([document_id]))
    results = backend.search("hi", k=10)
    assert results == [stable_key]


def test_gemini_backend_dedupes_multiple_docs_pointing_to_same_segment(
    scratch_db: psycopg.Connection, fts_conn: apsw.Connection
) -> None:
    """A segment document and one of its attachment-chunk documents can
    both appear in a result page — they must fold to the same
    segment_key and count once (SPEC §13.3)."""
    owner_id, _ = _insert_person(scratch_db, "Owner", is_owner=True)
    alice_id, _ = _insert_person(scratch_db, "Alice")
    chat_id, _ = _insert_chat(scratch_db, participants=[owner_id, alice_id])
    segment_id, stable_key, _ = _make_segment(
        scratch_db, fts_conn, chat_id=chat_id, rendered_text="hi",
        started_at=datetime(2024, 1, 1, tzinfo=UTC), person_id=alice_id,
    )
    seg_doc = "d" + "a" * 62
    chunk_doc = "d" + "b" * 62
    _insert_export_document(scratch_db, document_id=seg_doc, segment_id=segment_id)
    _insert_export_document(scratch_db, document_id=chunk_doc, segment_id=segment_id)

    backend = GeminiEvalBackend(conn=scratch_db, client=_FakeGeminiSearchClient([seg_doc, chunk_doc]))
    results = backend.search("hi", k=10)
    assert results == [stable_key]  # deduped to one


def test_gemini_backend_unresolvable_doc_id_is_skipped(scratch_db: psycopg.Connection) -> None:
    backend = GeminiEvalBackend(conn=scratch_db, client=_FakeGeminiSearchClient(["d" + "z" * 62]))
    assert backend.search("nothing", k=10) == []


def test_gemini_search_client_protocol_conformance() -> None:
    # Purely a static-typing sanity check exercised at runtime: any
    # object with a matching `search` method satisfies the Protocol.
    client: GeminiSearchClient = _FakeGeminiSearchClient(["d" + "a" * 62])
    assert client.search("x", page_size=1) == ["d" + "a" * 62]
