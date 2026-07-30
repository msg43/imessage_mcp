"""Postgres integration tests for the S6 embedding pipeline (SPEC §8
S6) — skips cleanly when no scratch Postgres is reachable, same
pattern as `tests/test_migrations_integration.py`."""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from imsg import constants
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.pipeline import run_embed
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.errors import EmbeddingError

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_embed_pipeline_test"

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
def text_provider() -> FakeTextEmbeddingProvider:
    return FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM)


@pytest.fixture
def mm_provider() -> FakeMultimodalEmbeddingProvider:
    return FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM)


def _insert_chat_and_session(conn: psycopg.Connection) -> tuple[int, int]:
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
    return chat_id, session_id


def _insert_segment(conn: psycopg.Connection, *, chat_id: int, session_id: int, rendered_text: str, seq: int = 0) -> int:
    stable_key = f"stable-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO segment (
                stable_key, chat_id, session_id, seq_in_session, started_at, ended_at,
                message_count, token_count, rendered_text, rendered_sha256, seg_config_hash
            ) VALUES (%s, %s, %s, %s, now(), now(), 1, 10, %s, 'x', 'cfg-hash')
            RETURNING segment_id
            """,
            (stable_key, chat_id, session_id, seq, rendered_text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_attachment_and_chunk(conn: psycopg.Connection, *, text: str) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key) VALUES (%s, %s) RETURNING attachment_id",
            (guid, f"key-{guid}"),
        )
        attachment_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO attachment_chunk (attachment_id, kind, seq, text, token_count) "
            "VALUES (%s, 'pdf_text', 0, %s, 10) RETURNING chunk_id",
            (attachment_id, text),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_image_attachment(conn: psycopg.Connection, *, cache_path: Path) -> int:
    import hashlib

    guid = f"att-{uuid.uuid4()}"
    sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, cache_path, mime_type, sha256, state)
            VALUES (%s, %s, %s, 'image/png', %s, 'materialized')
            RETURNING attachment_id
            """,
            (guid, f"key-{guid}", str(cache_path), sha),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


# --- tests -------------------------------------------------------------


def test_embeds_segments_and_chunks(
    scratch_db: psycopg.Connection, text_provider: FakeTextEmbeddingProvider
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="hello world")
    chunk_id = _insert_attachment_and_chunk(scratch_db, text="chunk text")

    report = run_embed(scratch_db, text_provider)

    assert report.segments_embedded == 1
    assert report.chunks_embedded == 1

    with scratch_db.cursor() as cur:
        cur.execute("SELECT model, dim, text_sha256 FROM segment_embedding WHERE segment_id = %s", (segment_id,))
        model, dim, text_sha = cur.fetchone()  # type: ignore[misc]
    assert model == text_provider.model_id
    assert dim == constants.PRIMARY_EMBEDDING_DIM
    assert text_sha is not None

    with scratch_db.cursor() as cur:
        cur.execute("SELECT model, dim FROM attachment_chunk_embedding WHERE chunk_id = %s", (chunk_id,))
        model, dim = cur.fetchone()  # type: ignore[misc]
    assert model == text_provider.model_id
    assert dim == constants.PRIMARY_EMBEDDING_DIM


def test_second_run_with_unchanged_text_is_a_noop(
    scratch_db: psycopg.Connection, text_provider: FakeTextEmbeddingProvider
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="stable content")

    first = run_embed(scratch_db, text_provider)
    second = run_embed(scratch_db, text_provider)

    assert first.segments_embedded == 1
    assert second.segments_embedded == 0


def test_changed_text_triggers_reembedding(
    scratch_db: psycopg.Connection, text_provider: FakeTextEmbeddingProvider
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    segment_id = _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="version one")
    run_embed(scratch_db, text_provider)

    with scratch_db.cursor() as cur:
        cur.execute("UPDATE segment SET rendered_text = %s WHERE segment_id = %s", ("version two", segment_id))

    report = run_embed(scratch_db, text_provider)
    assert report.segments_embedded == 1

    with scratch_db.cursor() as cur:
        cur.execute("SELECT text_sha256 FROM segment_embedding WHERE segment_id = %s", (segment_id,))
        (text_sha,) = cur.fetchone()  # type: ignore[misc]
    import hashlib

    assert text_sha == hashlib.sha256(b"version two").hexdigest()


def test_batching_embeds_more_segments_than_one_batch(
    scratch_db: psycopg.Connection, text_provider: FakeTextEmbeddingProvider
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    for i in range(10):
        _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text=f"segment {i}", seq=i)

    report = run_embed(scratch_db, text_provider, batch_size=3)  # forces multiple batches

    assert report.segments_embedded == 10
    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM segment_embedding")
        (count,) = cur.fetchone()  # type: ignore[misc]
    assert count == 10


class _NeverCallMeTextProvider:
    """Wraps `FakeTextEmbeddingProvider` but fails the test if
    `embed_documents` is ever called — proves `run_embed(dry_run=True)`
    genuinely never invokes the provider (SPEC §8: avoid wasted compute
    on a preview), not just that it skips writing the result."""

    model_id = "never-call-me/text"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed_documents must not be called in dry-run mode")

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        raise AssertionError("embed_query must not be called in dry-run mode")


def test_run_embed_dry_run_writes_nothing_and_never_calls_the_provider(
    scratch_db: psycopg.Connection,
) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="hello world")
    _insert_attachment_and_chunk(scratch_db, text="chunk text")

    spy_provider = _NeverCallMeTextProvider(dim=constants.PRIMARY_EMBEDDING_DIM)
    report = run_embed(scratch_db, spy_provider, dry_run=True)

    assert report.dry_run is True
    assert report.segments_embedded == 1
    assert report.chunks_embedded == 1

    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM segment_embedding")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM attachment_chunk_embedding")
        assert cur.fetchone() == (0,)

    # A real run afterward embeds normally.
    real_provider = FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM)
    real_report = run_embed(scratch_db, real_provider)
    assert real_report.dry_run is False
    assert real_report.segments_embedded == 1
    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM segment_embedding")
        assert cur.fetchone() == (1,)


def test_wrong_dimension_provider_raises(scratch_db: psycopg.Connection) -> None:
    chat_id, session_id = _insert_chat_and_session(scratch_db)
    _insert_segment(scratch_db, chat_id=chat_id, session_id=session_id, rendered_text="hello")

    bad_provider = FakeTextEmbeddingProvider(dim=17)  # not PRIMARY_EMBEDDING_DIM
    with pytest.raises(EmbeddingError):
        run_embed(scratch_db, bad_provider)


def test_multimodal_image_embedding(
    scratch_db: psycopg.Connection,
    tmp_path: Path,
    text_provider: FakeTextEmbeddingProvider,
    mm_provider: FakeMultimodalEmbeddingProvider,
) -> None:
    image_path = tmp_path / "photo.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=8x8", "-frames:v", "1", str(image_path)],
        check=True, capture_output=True, timeout=15,
    )
    attachment_id = _insert_image_attachment(scratch_db, cache_path=image_path)

    report = run_embed(scratch_db, text_provider, multimodal_provider=mm_provider)

    assert report.attachments_embedded == 1
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT model, dim, media_sha256 FROM attachment_mm_embedding WHERE attachment_id = %s",
            (attachment_id,),
        )
        model, dim, media_sha = cur.fetchone()  # type: ignore[misc]
    assert model == mm_provider.model_id
    assert dim == constants.MULTIMODAL_EMBEDDING_DIM
    assert media_sha is not None


def test_multimodal_disabled_by_default_when_no_provider_given(
    scratch_db: psycopg.Connection, tmp_path: Path, text_provider: FakeTextEmbeddingProvider
) -> None:
    image_path = tmp_path / "photo.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=8x8", "-frames:v", "1", str(image_path)],
        check=True, capture_output=True, timeout=15,
    )
    _insert_image_attachment(scratch_db, cache_path=image_path)

    report = run_embed(scratch_db, text_provider)  # no multimodal_provider

    assert report.attachments_embedded == 0
    with scratch_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM attachment_mm_embedding")
        (count,) = cur.fetchone()  # type: ignore[misc]
    assert count == 0


def test_multimodal_second_run_with_unchanged_image_is_a_noop(
    scratch_db: psycopg.Connection,
    tmp_path: Path,
    text_provider: FakeTextEmbeddingProvider,
    mm_provider: FakeMultimodalEmbeddingProvider,
) -> None:
    image_path = tmp_path / "photo.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=yellow:s=8x8", "-frames:v", "1", str(image_path)],
        check=True, capture_output=True, timeout=15,
    )
    _insert_image_attachment(scratch_db, cache_path=image_path)

    first = run_embed(scratch_db, text_provider, multimodal_provider=mm_provider)
    second = run_embed(scratch_db, text_provider, multimodal_provider=mm_provider)

    assert first.attachments_embedded == 1
    assert second.attachments_embedded == 0
