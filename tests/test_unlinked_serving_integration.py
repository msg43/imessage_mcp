"""What the two MCP scopes and export do with D13's rows (owner decision
D13, 2026-09-23).

Extraction now files every message `chat.db` holds, including rows with no
chat link: into Apple's "Recently Deleted" chat with a delete date
(`message.deleted_at`), or into a holding chat (`chat.unfiled_key`) when no
real chat can be named. The owner's rule for serving them:

  * `full` scope (the local surface, and the public target for Gemini):
    every message is searchable, holding chats and deleted messages
    included, and a deleted message renders as `[deleted]`.
  * `allowlist` scope: holding chats are denied outright, by a rule in the
    eligibility module export and retrieval share; deleted messages never
    show, the way unsent ones never do.
  * Export (S8): deleted messages are left out like unsent ones, and
    holding chats are never eligible.

Every test here fails on the code from before D13 (core 14ee55c): the
holding-chat and delete-date columns do not exist there, and the last test
drives extraction itself, which did not file these rows at all.

Fictional personas only (D5).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import apsw
import psycopg
import pytest

from _export_fixtures import (
    add_participant,
    admin_reachable,
    allow,
    create_scratch_db,
    drop_scratch_db,
    insert_attachment,
    insert_chat,
    insert_message,
    insert_person,
    insert_segment,
    link_attachment,
)
from imsg import constants
from imsg.embed.fts.schema import create_schema
from imsg.embed.fts.sync import upsert_segment_row
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.export.eligibility import compute_chat_eligibility, eligible_chat_ids
from imsg.export.planner import build_desired_documents
from imsg.retrieval.access import LOCAL_FULL_ACCESS, AccessContext
from imsg.retrieval.errors import NotFoundError
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.service import RetrievalService

TEST_DB_NAME = "imsg_index_unlinked_serving_test"

pytestmark = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)

PUBLIC = AccessContext(surface="public", scope="allowlist", subject="owner-subject")
PUBLIC_FULL = AccessContext(surface="public", scope="full", subject="owner-subject")
NEEDLE = "servingneedle"
_T0 = datetime(2024, 6, 3, 9, 0, tzinfo=UTC)
_DELETED_AT = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)


class _Cfg:
    """The fields `RetrievalService`, `refresh_segment_rendering` and the
    export planner read."""

    class _Retrieval:
        k_fts = 100
        k_vector = 100
        rrf_k = 60
        rerank_top = 50
        default_limit = 10
        hnsw_ef_search = 100

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


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    conn.commit()
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(TEST_DB_NAME)


@pytest.fixture
def fts(tmp_path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(tmp_path / "fts.db"))
    create_schema(conn)
    return conn


def _service(db: psycopg.Connection, fts: apsw.Connection) -> RetrievalService:
    return RetrievalService(
        pg_conn=db,
        fts_conn=fts,
        config=cast(Any, _Cfg()),
        text_provider=FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM),
        reranker=FakeRerankerProvider(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM),
    )


@dataclass(frozen=True)
class People:
    owner: int
    alice: int


def _people(db: psycopg.Connection) -> People:
    """Both allowlisted, so nothing but the rule under test can deny."""
    owner = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice = insert_person(db, display_name="Alice Example", short_name="alice")
    allow(db, owner)
    allow(db, alice)
    return People(owner=owner, alice=alice)


@dataclass(frozen=True)
class Chat:
    chat_id: int
    thread_key: str
    segment_id: int
    message_ids: tuple[int, ...]


def _chat(
    db: psycopg.Connection,
    fts: apsw.Connection,
    name: str,
    p: People,
    *,
    lines: list[str],
    unfiled_key: str | None = None,
    deleted: frozenset[int] = frozenset(),
) -> Chat:
    """A chat with one segment of Alice's messages, one per line, rendered
    by the segment pipeline itself and indexed the way S6 would. Messages
    whose position is in `deleted` carry a delete date; `unfiled_key`
    makes the chat a holding chat."""
    from imsg.segment.pipeline import refresh_segment_rendering

    chat_id = insert_chat(db, source_guid=f"chat-{name}")
    if unfiled_key is not None:
        db.execute("UPDATE chat SET unfiled_key = %s WHERE chat_id = %s", (unfiled_key, chat_id))
    add_participant(db, chat_id, p.owner)
    add_participant(db, chat_id, p.alice)
    message_ids = [
        insert_message(
            db, chat_id=chat_id, sender_person_id=p.alice, is_from_me=False,
            sent_at=_T0 + timedelta(minutes=i), text=line,
        )
        for i, line in enumerate(lines)
    ]
    for i in deleted:
        db.execute(
            "UPDATE message SET deleted_at = %s WHERE message_id = %s", (_DELETED_AT, message_ids[i])
        )
    segment_id = insert_segment(
        db, chat_id=chat_id, started_at=_T0,
        ended_at=_T0 + timedelta(minutes=len(lines) - 1),
        message_ids=message_ids, stable_key=f"segment-{name}",
    )
    stored, _ = refresh_segment_rendering(db, segment_id, cast(Any, _Cfg()))
    upsert_segment_row(fts, segment_id, f"segment-{name}", stored)
    return Chat(chat_id, f"thread-chat-{name}", segment_id, tuple(message_ids))


def _conversation(service: RetrievalService, context: AccessContext, chat: Chat) -> list[dict[str, Any]]:
    out = service.get_conversation(context, thread_id=chat.thread_key)
    return cast(list[dict[str, Any]], out["messages"])


def _hit_texts(service: RetrievalService, context: AccessContext) -> dict[str, str]:
    results = service.search_messages(context, query=NEEDLE, limit=50).results
    return {str(r["thread_key"]): str(r["text"]) for r in results}


# --- allowlist: holding chats are denied outright ------------------------------


def test_allowlist_denies_a_holding_chat_even_when_everyone_in_it_is_allowlisted(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    from imsg.export.models import DENY_HOLDING_CHAT  # here, so the file collects on older code

    p = _people(db)
    control = _chat(db, fts, "real-dm", p, lines=[f"{NEEDLE} real thread"])
    holding = _chat(
        db, fts, "holding", p, lines=[f"{NEEDLE} unfiled thread"], unfiled_key="sender:alice",
    )
    service = _service(db, fts)

    assert eligible_chat_ids(db) == {control.chat_id}
    verdicts = compute_chat_eligibility(db)
    assert verdicts[holding.chat_id].deny_reasons == {DENY_HOLDING_CHAT}
    assert verdicts[control.chat_id].eligible

    assert set(_hit_texts(service, PUBLIC)) == {control.thread_key}
    with pytest.raises(NotFoundError):
        service.get_conversation(PUBLIC, thread_id=holding.thread_key)

    # Full scope serves it, on the local surface and on the public one.
    for context in (LOCAL_FULL_ACCESS, PUBLIC_FULL):
        assert set(_hit_texts(service, context)) == {control.thread_key, holding.thread_key}
        [line] = _conversation(service, context, holding)
        assert "unfiled thread" in str(line["text"])


# --- deleted messages: labelled under full scope, absent under allowlist -------


def test_deleted_messages_are_labelled_under_full_scope_and_absent_under_allowlist(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    p = _people(db)
    dm = _chat(
        db, fts, "with-deleted", p,
        lines=[f"{NEEDLE} kept wording", "deleted wording"],
        deleted=frozenset({1}),
    )
    service = _service(db, fts)

    for context in (LOCAL_FULL_ACCESS, PUBLIC_FULL):
        hit = _hit_texts(service, context)[dm.thread_key]
        assert "[deleted] deleted wording" in hit
        lines = _conversation(service, context, dm)
        assert [line["is_deleted"] for line in lines] == [False, True]
        assert "[deleted] deleted wording" in str(lines[1]["text"])

    public_hit = _hit_texts(service, PUBLIC)[dm.thread_key]
    assert "kept wording" in public_hit
    assert "deleted wording" not in public_hit
    public_lines = _conversation(service, PUBLIC, dm)
    assert [str(line["text"]) for line in public_lines] == [
        f"[09:00] alice: {NEEDLE} kept wording"
    ]

    people = {
        str(person["short_name"]): person
        for person in cast(list[dict[str, Any]], service.list_people(PUBLIC)["people"])
    }
    assert people["alice"]["message_count"] == 1


# --- export: deleted messages stay home ----------------------------------------


def test_export_leaves_deleted_messages_out_the_way_it_leaves_unsent_ones(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    p = _people(db)
    dm = _chat(
        db, fts, "export", p,
        lines=["kept wording", "deleted wording"],
        deleted=frozenset({1}),
    )
    photo = insert_attachment(db, filename="deleted-photo.jpeg", mime_type="image/jpeg")
    link_attachment(db, dm.message_ids[1], photo)
    holding = _chat(db, fts, "holding-export", p, lines=["unfiled wording"], unfiled_key="owner")

    documents = build_desired_documents(db, cast(Any, _Cfg()))
    texts = [doc.text for doc in documents.values()]

    assert any("kept wording" in text for text in texts)
    assert not any("deleted wording" in text for text in texts)
    assert not any("deleted-photo" in text for text in texts)
    assert not any("unfiled wording" in text for text in texts)
    assert {doc.upsert.chat_id for doc in documents.values()} == {dm.chat_id}
    assert holding.chat_id not in eligible_chat_ids(db)


# --- end to end: extraction's own rows ----------------------------------------


def test_full_scope_finds_what_extraction_filed_and_allowlist_withholds_it(
    db: psycopg.Connection, fts: apsw.Connection, tmp_path: Path, config_dict_factory: object
) -> None:
    """Extraction files a deleted message into its real chat and an
    unlinked one into a holding chat; after identity and segmentation,
    full scope searches both and allowlist scope serves neither, with
    every person in the index allowlisted."""
    from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
    from imsg.config.loader import load_config_dict
    from imsg.segment.boundaries import FakeBoundaryProvider
    from imsg.segment.pipeline import run_segment
    from imsg.stages.extract import MergeMode, run_extract
    from imsg.stages.identity import ContactRecord, run_identity
    from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun

    alice, chat_guid = "+15550000001", "iMessage;-;+15550000001"
    bodies = {
        "msg-kept": f"{NEEDLE} kept", "msg-deleted": f"{NEEDLE} deleted",
        "msg-unfiled": f"{NEEDLE} unfiled",
    }
    builder = ChatDbBuilder()
    builder.add_chat(FixtureChat(guid=chat_guid, rowid=1))
    builder.add_handle(FixtureHandle(raw_value=alice, rowid=1))
    builder.link_participant(chat_guid, alice)
    builder.add_message(FixtureMessage(guid="msg-kept", chat_guid=chat_guid, rowid=1,
                                       handle_raw_value=alice, date=_T0))
    builder.add_message(FixtureMessage(guid="msg-deleted", chat_guid=None, rowid=2,
                                       handle_raw_value=alice, date=_T0 + timedelta(minutes=1),
                                       recoverable_chat_guid=chat_guid, delete_date=_DELETED_AT))
    builder.add_message(FixtureMessage(guid="msg-unfiled", chat_guid=None, rowid=3,
                                       handle_raw_value=alice, date=_T0 + timedelta(minutes=2)))
    snapshot = builder.build(tmp_path / "snapshot.db")
    binary = tmp_path / "imsg-dump"
    binary.write_text("")
    dump = ImsgDumpRun(
        messages=tuple(
            ImsgDumpMessage(
                rowid=i, guid=guid, chat_guid=None, handle=None, is_from_me=False, date=None,
                date_edited=None, date_retracted=None, service="iMessage", body_text=body,
                edit_history=(), is_unsent=False, tapback=None, attachment_rowids=(),
                reply_to_guid=None,
            )
            for i, (guid, body) in enumerate(bodies.items(), start=1)
        ),
        stderr_lines=(),
    )
    run_extract(
        conn=db, source_name="mini", snapshot_path=snapshot, imsg_dump_binary=binary,
        run_imsg_dump_fn=lambda _b, _s, _since: dump, merge_mode=MergeMode.LIVE,
    )
    config = load_config_dict(config_dict_factory())  # type: ignore[operator]
    def no_contacts(default_region: str) -> list[ContactRecord]:
        return []

    run_identity(conn=db, config=config, contacts_importer=no_contacts)
    run_segment(db, config, FakeBoundaryProvider(), b"prompt")
    for person_id, in db.execute("SELECT person_id FROM person").fetchall():
        allow(db, int(person_id))
    for segment_id, stable_key, text in db.execute(
        "SELECT segment_id, stable_key, rendered_text FROM segment"
    ).fetchall():
        upsert_segment_row(fts, int(segment_id), str(stable_key), str(text))
    service = _service(db, fts)

    local = " ".join(_hit_texts(service, LOCAL_FULL_ACCESS).values())
    assert f"[deleted] {NEEDLE} deleted" in local
    assert f"{NEEDLE} unfiled" in local
    assert "(Unfiled: no chat recorded)" in local

    public = " ".join(_hit_texts(service, PUBLIC).values())
    assert f"{NEEDLE} kept" in public
    assert "deleted" not in public
    assert "unfiled" not in public
