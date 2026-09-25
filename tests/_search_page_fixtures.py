"""Synthetic corpus builder for the search page tests.

Fictional people and made-up text only (the repo is public). Every row is
written through plain SQL against a scratch database that has the real
migrations applied, and every segment and attachment chunk is also
written into a real FTS5 sidecar, so the page's queries run against the
same schema they meet in production. Segments are rendered by the
production renderer (`imsg.segment.render.render_segment`), header lines
and message labels included, because that is the text the index holds.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import apsw
import psycopg

from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.fts.schema import create_schema
from imsg.embed.fts.sync import upsert_chunk_row, upsert_segment_row
from imsg.embed.vector_codec import vector_literal
from imsg.keys import attachment_key, message_key, thread_key
from imsg.segment.models import MessageForSegmentation, SegmentDraft
from imsg.segment.render import render_segment
from imsg.textnorm import normalize_text

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def dsn(dbname: str) -> str:
    return f"postgresql://{TEST_PG_USER}@/{dbname}?host={TEST_PG_HOST}&port={TEST_PG_PORT}"


def admin_reachable() -> bool:
    try:
        conn = psycopg.connect(dsn("postgres"), connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


REACHABLE = admin_reachable()
SKIP_REASON = (
    f"no reachable scratch Postgres instance (tried {TEST_PG_HOST}:{TEST_PG_PORT}) — set "
    "IMSG_TEST_PG_HOST/IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
)


def create_scratch_db(name: str) -> psycopg.Connection:
    admin = psycopg.connect(dsn("postgres"), autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {name}")
            cur.execute(f"CREATE DATABASE {name}")
    finally:
        admin.close()
    conn = psycopg.connect(dsn(name), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR).apply_pending()
    return conn


def drop_scratch_db(name: str) -> None:
    admin = psycopg.connect(dsn("postgres"), autocommit=True)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
    finally:
        admin.close()


def scratch_db(name: str) -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(name)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(name)


RENDER_TIMEZONE = "UTC"
"""The zone segments are rendered in (their `Time:` line); a test may
set `Corpus.render_timezone` to another."""


@dataclass(frozen=True, slots=True)
class Person:
    person_id: int
    short_name: str
    display_name: str
    is_owner: bool = False


@dataclass(frozen=True, slots=True)
class Chat:
    chat_id: int
    thread_key: str
    kind: str = "dm"
    display_name: str | None = None
    participant_names: tuple[str, ...] = ()
    """Everyone but the owner, as the renderer's `Chat:` line lists them."""
    unfiled: bool = False


@dataclass(frozen=True, slots=True)
class Msg:
    message_id: int
    message_key: str
    source_guid: str


@dataclass(frozen=True, slots=True)
class Att:
    attachment_id: int
    attachment_key: str
    sha256: str | None


@dataclass(slots=True)
class Seg:
    segment_id: int
    stable_key: str
    messages: list[Msg] = field(default_factory=list)


@dataclass
class Corpus:
    conn: psycopg.Connection
    fts: apsw.Connection
    data_root: Path
    render_timezone: str = RENDER_TIMEZONE

    def person(self, display_name: str, *, owner: bool = False) -> Person:
        short = f"{display_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
                "VALUES (%s, %s, %s, false) RETURNING person_id",
                (display_name, short, owner),
            )
            row = cur.fetchone()
        assert row is not None
        return Person(int(row[0]), short, display_name, owner)

    def chat(
        self,
        participants: Sequence[Person],
        *,
        kind: str = "dm",
        display_name: str | None = None,
        unfiled_key: str | None = None,
    ) -> Chat:
        guid = f"chat-{uuid.uuid4()}"
        tkey = thread_key(guid)
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat (source_guid, thread_key, kind, display_name, unfiled_key) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING chat_id",
                (guid, tkey, kind, display_name, unfiled_key),
            )
            row = cur.fetchone()
            assert row is not None
            chat_id = int(row[0])
            for person in participants:
                cur.execute(
                    "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)",
                    (chat_id, person.person_id),
                )
        others = tuple(sorted(p.display_name for p in participants if not p.is_owner))
        return Chat(chat_id, tkey, kind, display_name, others, unfiled_key is not None)

    def message(
        self,
        chat: Chat,
        sent_at: datetime,
        sender: Person | None,
        text: str | None,
        *,
        deleted_at: datetime | None = None,
        unsent: bool = False,
        edited: bool = False,
        reply_to_guid: str | None = None,
    ) -> Msg:
        guid = f"msg-{uuid.uuid4()}"
        mkey = message_key(guid)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO message (source_guid, message_key, chat_id, sender_person_id,
                    is_from_me, sent_at, service, text_original, text_normalized, is_unsent,
                    is_edited, deleted_at, reply_to_guid)
                VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s, %s, %s, %s, %s)
                RETURNING message_id
                """,
                (
                    guid,
                    mkey,
                    chat.chat_id,
                    sender.person_id if sender else None,
                    sender is None,
                    sent_at,
                    text,
                    normalize_text(text.replace("\ufffc", "")) if text is not None else None,
                    unsent,
                    edited,
                    deleted_at,
                    reply_to_guid,
                ),
            )
            row = cur.fetchone()
        assert row is not None
        return Msg(int(row[0]), mkey, guid)

    def segment(
        self,
        chat: Chat,
        lines: Sequence[tuple[datetime, Person | None, str]],
        *,
        index: bool = True,
        extra_text: str = "",
    ) -> Seg:
        """A session with one segment holding `lines` (sender None = the
        owner), rendered as production renders it and indexed in the FTS
        sidecar. `extra_text` is appended to the rendered text (standing
        in for attachment snippets)."""
        started, ended = lines[0][0], lines[-1][0]
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO session (chat_id, started_at, ended_at, gap_hours) "
                "VALUES (%s, %s, %s, 3.0) RETURNING session_id",
                (chat.chat_id, started, ended),
            )
            row = cur.fetchone()
            assert row is not None
            session_id = int(row[0])
        messages = [self.message(chat, at, person, text) for at, person, text in lines]
        draft = SegmentDraft(
            session_started_at=started,
            seq_in_session=0,
            messages=tuple(
                MessageForSegmentation(
                    message_id=msg.message_id,
                    source_guid=msg.source_guid,
                    chat_id=chat.chat_id,
                    sent_at=at,
                    is_from_me=person is None,
                    sender_short_name=person.short_name if person else "owner",
                    text=text,
                    is_unsent=False,
                    is_edited=False,
                    has_attachments=False,
                )
                for msg, (at, person, text) in zip(messages, lines, strict=True)
            ),
        )
        rendered = render_segment(
            draft,
            participants=chat.participant_names,
            chat_kind=chat.kind,
            chat_display_name=chat.display_name,
            timezone=self.render_timezone,
            attachment_snippet_chars=200,
            unfiled=chat.unfiled,
        )
        rendered = f"{rendered}{(' ' + extra_text) if extra_text else ''}"
        stable = hashlib.sha256(f"seg:{uuid.uuid4()}".encode()).hexdigest()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO segment (stable_key, chat_id, session_id, seq_in_session, started_at,
                    ended_at, message_count, token_count, rendered_text, rendered_sha256,
                    seg_config_hash)
                VALUES (%s, %s, %s, 0, %s, %s, %s, 10, %s, 'x', 'cfg') RETURNING segment_id
                """,
                (stable, chat.chat_id, session_id, started, ended, len(lines), rendered),
            )
            row = cur.fetchone()
            assert row is not None
            seg = Seg(int(row[0]), stable)
        for msg in messages:
            self.link(seg, msg)
        if index:
            upsert_segment_row(self.fts, seg.segment_id, stable, rendered)
        return seg

    def link(self, seg: Seg, msg: Msg) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
                (seg.segment_id, msg.message_id),
            )
        seg.messages.append(msg)

    def attachment(
        self,
        msg: Msg,
        *,
        filename: str,
        mime_type: str | None,
        content: bytes | None,
        state: str = "materialized",
    ) -> Att:
        guid = f"att-{uuid.uuid4()}"
        akey = attachment_key(guid)
        sha = hashlib.sha256(content).hexdigest() if content is not None else None
        if content is not None and sha is not None:
            path = self.data_root / "attachments" / sha[:2] / sha
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attachment (source_guid, attachment_key, filename, mime_type,
                    byte_size, sha256, state)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING attachment_id
                """,
                (
                    guid,
                    akey,
                    filename,
                    mime_type,
                    len(content) if content is not None else None,
                    sha,
                    state,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            att_id = int(row[0])
            cur.execute(
                "INSERT INTO message_attachment (message_id, attachment_id, ordinal) VALUES (%s, %s, 0)",
                (msg.message_id, att_id),
            )
            cur.execute(
                "UPDATE message SET has_attachments = true WHERE message_id = %s", (msg.message_id,)
            )
        return Att(att_id, akey, sha)

    def chunk(self, att: Att, text: str, *, seq: int = 0, kind: str = "pdf_text") -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO attachment_chunk (attachment_id, kind, seq, text) "
                "VALUES (%s, %s, %s, %s) RETURNING chunk_id",
                (att.attachment_id, kind, seq, text),
            )
            row = cur.fetchone()
        assert row is not None
        chunk_id = int(row[0])
        upsert_chunk_row(self.fts, chunk_id, att.attachment_id, text)
        return chunk_id

    def enrichment(self, att: Att, kind: str, text: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO enrichment (attachment_id, kind, state, text) VALUES (%s, %s, 'done', %s)",
                (att.attachment_id, kind, text),
            )

    def tapback(self, target: Msg, sender: Person | None, kind: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tapback (source_guid, target_source_guid, target_message_id,
                    sender_person_id, is_from_me, kind, acted_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                """,
                (
                    f"tb-{uuid.uuid4()}",
                    target.source_guid,
                    target.message_id,
                    sender.person_id if sender else None,
                    sender is None,
                    kind,
                ),
            )

    def segment_vector(self, seg: Seg, vector: Sequence[float]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO segment_embedding (segment_id, model, dim, text_sha256, vec) "
                "VALUES (%s, 'fake', %s, 'x', %s::halfvec)",
                (seg.segment_id, len(vector), vector_literal(vector)),
            )

    def chunk_vector(self, chunk_id: int, vector: Sequence[float]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO attachment_chunk_embedding (chunk_id, model, dim, text_sha256, vec) "
                "VALUES (%s, 'fake', %s, 'x', %s::halfvec)",
                (chunk_id, len(vector), vector_literal(vector)),
            )

    def image_vector(self, att: Att, vector: Sequence[float]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO attachment_mm_embedding (attachment_id, model, dim, media_sha256, vec) "
                "VALUES (%s, 'fake', %s, 'x', %s::halfvec)",
                (att.attachment_id, len(vector), vector_literal(vector)),
            )


def open_fts(path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(path))
    create_schema(conn)
    return conn


def unit_vector(dim: int, *, hot: int, mix: Sequence[tuple[int, float]] = ()) -> list[float]:
    """A unit vector mostly along axis `hot`, tilted toward `mix` axes, so
    its cosine similarity with the pure `hot` axis is predictable."""
    values = [0.0] * dim
    values[hot] = 1.0
    for axis, weight in mix:
        values[axis] += weight
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def tiny_png(width: int = 8, height: int = 8, rgb: tuple[int, int, int] = (40, 120, 200)) -> bytes:
    """A valid solid-colour PNG, built by hand (no imaging library)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def page_settings(**overrides: Any) -> Any:
    """`SearchSettings` for tests: UTC, every channel on."""
    from imsg.search_page.search import SearchSettings

    values: dict[str, Any] = {
        "timezone": "UTC",
        "index_unsent": False,
        "rrf_k": 60,
        "fts_max_hits": 20000,
        "unindexed_window_days": 60,
        "semantic_enabled": True,
        "multimodal_enabled": True,
        "text_min_similarity": 0.5,
        "multimodal_min_similarity": 0.2,
        "max_hits_per_channel": 2000,
        "ef_search": 200,
        "max_scan_tuples": 50000,
    }
    values.update(overrides)
    return SearchSettings(**values)


def make_page_client(
    corpus: Corpus,
    dbname: str,
    *,
    settings: Any = None,
    model_api: Any = None,
    **page: Any,
) -> Any:
    """A Starlette test client for the page, already signed in: the owner
    password is a random value nobody types, and the session is minted
    server-side with the page's own `SessionStore` and set as a cookie
    (no login form is filled in)."""
    import psycopg as _psycopg
    from starlette.testclient import TestClient

    from imsg.search_page.app import AppDeps, ConnectionPool, FtsReaders, build_app
    from imsg.search_page.auth import LoginGuard, PasswordFile, SessionStore, hash_password
    from imsg.search_page.config import SearchPageConfig
    from imsg.search_page.media import MediaConverter
    from imsg.search_page.secret_files import write_private_file
    from imsg.search_page.server import open_fts_reader

    root = corpus.data_root
    private = root / "private" / "search-page"
    password_path = private / "owner-password"
    write_private_file(
        password_path, (hash_password(secrets.token_urlsafe(24), n=2**14) + "\n").encode("utf-8")
    )
    passwords = PasswordFile(password_path)
    sessions = SessionStore(private / "sessions.json", lifetime_seconds=3600)
    raw_token, _session = sessions.create(passwords.fingerprint())
    deps = AppDeps(
        page=SearchPageConfig(enabled=True, allowed_hosts=["testserver"], **page),
        settings=settings if settings is not None else page_settings(),
        data_root=root,
        pool=ConnectionPool(lambda: _psycopg.connect(dsn(dbname), autocommit=True), 2),
        fts=FtsReaders(
            lambda: open_fts_reader(root / "fts" / "fts.db"), root / "fts" / "fts.db", 2
        ),
        passwords=passwords,
        sessions=sessions,
        login_guard=LoginGuard(
            passwords, max_failures_per_client=5, max_failures_global=30, window_seconds=900
        ),
        media=MediaConverter(root / "search-page" / "thumbnails"),
        model_api=model_api,
    )
    client = TestClient(build_app(deps))
    client.cookies.set("imsg_session", raw_token)
    return client


def csrf_token_of(html: str) -> str:
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    assert match is not None
    return match.group(1)


__all__ = [
    "REACHABLE",
    "SKIP_REASON",
    "Att",
    "Chat",
    "Corpus",
    "Msg",
    "Person",
    "Seg",
    "cosine",
    "create_scratch_db",
    "csrf_token_of",
    "drop_scratch_db",
    "dsn",
    "make_page_client",
    "open_fts",
    "page_settings",
    "scratch_db",
    "tiny_png",
    "unit_vector",
]
