"""Public `allowlist` scope serves exactly what export would ship.

SPEC §10.3a: "Public `allowlist` scope applies the same eligibility
predicate to `search_messages`, `get_conversation`, `list_people`,
`get_attachment_text`, and any future tool". Every test here drives
`RetrievalService` the way the public MCP server does — with
`AccessContext(surface="public", scope="allowlist")` — and checks what
comes back against export's rule in `imsg.export.eligibility`.

The tests use only the service's public methods, so this file also runs
unchanged against the code from before the public surface shared
export's rule. There, every test fails except the two marked as
controls.

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
    add_edit_history,
    add_enrichment,
    add_participant,
    add_raw_participant,
    admin_reachable,
    allow,
    create_scratch_db,
    drop_scratch_db,
    insert_attachment,
    insert_chat,
    insert_message,
    insert_person,
    insert_segment,
    insert_tapback,
    link_attachment,
)
from imsg import constants
from imsg.embed.fts.schema import create_schema
from imsg.embed.fts.sync import upsert_segment_row
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.export.eligibility import compute_chat_eligibility, eligible_chat_ids
from imsg.retrieval.access import LOCAL_FULL_ACCESS, AccessContext
from imsg.retrieval.errors import (
    InvalidArgumentError,
    NotFoundError,
    PersonAmbiguousError,
    PersonNotFoundError,
)
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.retrieval.service import RetrievalService

TEST_DB_NAME = "imsg_index_allowlist_parity_test"

pytestmark = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)

PUBLIC = AccessContext(surface="public", scope="allowlist", subject="owner-subject")
NEEDLE = "parityneedle"
WITHHELD = "[attachment withheld]"
_T0 = datetime(2024, 6, 3, 9, 0, tzinfo=UTC)


class _Cfg:
    """The fields `RetrievalService` and `refresh_segment_rendering` read."""

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


class _ShowEverythingCfg(_Cfg):
    """`policy.*` set to show unsent messages and edit history — which
    `allowlist` scope must ignore (D1)."""

    class _Policy:
        index_unsent = True
        index_edit_history = True

    policy = _Policy()


class _CountingTextProvider(FakeTextEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dim=constants.PRIMARY_EMBEDDING_DIM)
        self.queries: list[str] = []

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        self.queries.append(text)
        return super().embed_query(text, instruction=instruction)


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    conn.commit()
    conn.autocommit = True  # as `imsg.db.connection.connect` opens it for the servers
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


def _service(
    db: psycopg.Connection,
    fts: apsw.Connection,
    cfg: object | None = None,
    text_provider: FakeTextEmbeddingProvider | None = None,
) -> RetrievalService:
    return RetrievalService(
        pg_conn=db,
        fts_conn=fts,
        config=cast(Any, cfg if cfg is not None else _Cfg()),
        text_provider=text_provider or FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM),
        reranker=FakeRerankerProvider(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=constants.MULTIMODAL_EMBEDDING_DIM),
    )


# --- the cast ---------------------------------------------------------------


@dataclass(frozen=True)
class People:
    owner: int
    alice: int  # text + attachments allowed
    bob: int  # text allowed, attachments NOT allowed
    frank: int  # text allowed; a former member where he appears
    carol: int  # no allowlist row
    evan: int  # no allowlist row; a former member where he appears
    gina: int  # no allowlist row; reached only through a raw participant handle
    dana: int  # an allowlist row that says no


def _people(db: psycopg.Connection, *, owner: bool = True) -> People:
    owner_id = (
        insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
        if owner
        else insert_person(db, display_name="Not The Owner", short_name="notowner")
    )
    ids = {
        "alice": insert_person(db, display_name="Alice Example", short_name="alice"),
        "bob": insert_person(db, display_name="Bob Builder", short_name="bob"),
        "frank": insert_person(db, display_name="Frank Framer", short_name="frank"),
        "carol": insert_person(db, display_name="Carol Carpenter", short_name="carol"),
        "evan": insert_person(db, display_name="Evan Estimator", short_name="evan"),
        "gina": insert_person(db, display_name="Gina Glazier", short_name="gina"),
        "dana": insert_person(db, display_name="Dana Driver", short_name="dana"),
    }
    if owner:
        allow(db, owner_id)
    allow(db, ids["alice"])
    allow(db, ids["bob"], text=True, attachments=False)
    allow(db, ids["frank"], text=True, attachments=False)
    allow(db, ids["dana"], text=False, attachments=False)
    return People(owner=owner_id, **ids)


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
    *,
    participants: list[int],
    senders: list[tuple[int | None, bool]],
    kind: str = "dm",
) -> Chat:
    """A chat with one segment: one message per `(sender_person_id,
    is_from_me)` in `senders`, each carrying `NEEDLE` and the chat's name,
    indexed in the FTS sidecar the way S6 would."""
    chat_id = insert_chat(db, source_guid=f"chat-{name}", kind=kind)
    for person_id in participants:
        add_participant(db, chat_id, person_id)
    message_ids = [
        insert_message(
            db,
            chat_id=chat_id,
            sender_person_id=sender,
            is_from_me=from_me,
            sent_at=_T0 + timedelta(minutes=i),
            text=f"{NEEDLE} {name} line {i}",
        )
        for i, (sender, from_me) in enumerate(senders)
    ]
    text = "\n".join(f"{NEEDLE} {name} line {i}" for i in range(len(senders)))
    segment_id = insert_segment(
        db,
        chat_id=chat_id,
        started_at=_T0,
        ended_at=_T0 + timedelta(minutes=max(len(senders) - 1, 0)),
        message_ids=message_ids,
        stable_key=f"segment-{name}",
        rendered_text=text,
    )
    upsert_segment_row(fts, segment_id, f"segment-{name}", text)
    return Chat(chat_id, f"thread-chat-{name}", segment_id, tuple(message_ids))


def _served_by_conversation(service: RetrievalService, chat: Chat) -> bool:
    try:
        service.get_conversation(PUBLIC, thread_id=chat.thread_key)
    except NotFoundError:
        return False
    return True


def _served_by_search(service: RetrievalService, query: str = NEEDLE) -> set[str]:
    results = service.search_messages(PUBLIC, query=query, limit=50).results
    return {str(r["thread_key"]) for r in results}


def _assert_denied_everywhere(
    db: psycopg.Connection, service: RetrievalService, denied: Chat, control: Chat
) -> None:
    """`denied` is refused by export's rule and by every public read path;
    `control` (every person allowlisted) is served by both."""
    assert _served_by_conversation(service, control)
    assert not _served_by_conversation(service, denied)
    assert _served_by_search(service) == {control.thread_key}
    assert eligible_chat_ids(db) == {control.chat_id}


def _eligible_dm(db: psycopg.Connection, fts: apsw.Connection, p: People) -> Chat:
    return _chat(
        db, fts, "eligible-dm",
        participants=[p.owner, p.alice],
        senders=[(p.alice, False), (None, True)],  # the owner's row, sender unresolved
    )


# --- the chat rule, case by case ----------------------------------------------


def test_chat_with_no_participants_is_not_served(db: psycopg.Connection, fts: apsw.Connection) -> None:
    """'Every participant allowlisted' is vacuously true of an empty set;
    export requires at least one participant."""
    p = _people(db)
    control = _eligible_dm(db, fts, p)
    empty = _chat(db, fts, "no-participants", participants=[], senders=[(p.alice, False)])
    _assert_denied_everywhere(db, _service(db, fts), empty, control)


def test_former_member_who_is_not_allowlisted_denies_the_chat(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """Evan left the group, so he has no `chat_participant` row, but his
    old messages are still in it."""
    p = _people(db)
    control = _eligible_dm(db, fts, p)
    group = _chat(
        db, fts, "former-member", kind="group",
        participants=[p.owner, p.alice],
        senders=[(p.alice, False), (p.evan, False)],
    )
    _assert_denied_everywhere(db, _service(db, fts), group, control)


def test_unresolved_sender_denies_the_chat(db: psycopg.Connection, fts: apsw.Connection) -> None:
    p = _people(db)
    control = _eligible_dm(db, fts, p)
    chat = _chat(
        db, fts, "unresolved-sender",
        participants=[p.owner, p.alice],
        senders=[(p.alice, False), (None, False)],
    )
    _assert_denied_everywhere(db, _service(db, fts), chat, control)


def test_unresolved_source_participant_denies_the_chat(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    p = _people(db)
    control = _eligible_dm(db, fts, p)
    chat = _chat(
        db, fts, "unresolved-handle",
        participants=[p.owner, p.alice],
        senders=[(p.alice, False)],
    )
    add_raw_participant(db, chat.chat_id, "raw-handle-unresolved")
    _assert_denied_everywhere(db, _service(db, fts), chat, control)


def test_unallowlisted_source_participant_denies_the_chat(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """The raw handle resolves to Gina, who never made it into
    `chat_participant` and is not allowlisted."""
    p = _people(db)
    control = _eligible_dm(db, fts, p)
    chat = _chat(
        db, fts, "unallowlisted-handle",
        participants=[p.owner, p.alice],
        senders=[(p.alice, False)],
    )
    add_raw_participant(db, chat.chat_id, "raw-handle-gina", resolve_to_person=p.gina)
    _assert_denied_everywhere(db, _service(db, fts), chat, control)


@pytest.mark.parametrize("tapback_sender", ["carol", "unresolved"])
def test_unallowlisted_or_unresolved_tapback_sender_denies_the_chat(
    db: psycopg.Connection, fts: apsw.Connection, tapback_sender: str
) -> None:
    p = _people(db)
    control = _eligible_dm(db, fts, p)
    chat = _chat(
        db, fts, f"tapback-{tapback_sender}",
        participants=[p.owner, p.alice],
        senders=[(p.alice, False)],
    )
    insert_tapback(
        db,
        target_message_id=chat.message_ids[0],
        sender_person_id=p.carol if tapback_sender == "carol" else None,
        is_from_me=False,
    )
    _assert_denied_everywhere(db, _service(db, fts), chat, control)


def test_chat_where_every_person_is_allowlisted_is_served(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """CONTROL — passes before and after the change: the rule must not
    over-deny. Current participants, a former member, the owner's own rows
    and a tapback, all allowlisted."""
    p = _people(db)
    group = _chat(
        db, fts, "all-allowlisted", kind="group",
        participants=[p.owner, p.alice, p.bob],
        senders=[(p.alice, False), (p.frank, False), (None, True), (p.bob, False)],
    )
    insert_tapback(db, target_message_id=group.message_ids[0], sender_person_id=p.bob)
    insert_tapback(db, target_message_id=group.message_ids[1], sender_person_id=None, is_from_me=True)
    service = _service(db, fts)

    assert eligible_chat_ids(db) == {group.chat_id}
    assert _served_by_conversation(service, group)
    assert _served_by_search(service) == {group.thread_key}


def test_no_owner_person_denies_chats_with_owner_content(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """Fail closed: with no owner person, an `is_from_me` row has no
    effective sender, and its chat is denied even though every listed
    participant is allowlisted."""
    p = _people(db, owner=False)
    control = _chat(db, fts, "alice-only", participants=[p.alice], senders=[(p.alice, False)])
    mine = _chat(
        db, fts, "owner-rows", participants=[p.alice], senders=[(p.alice, False), (None, True)]
    )
    _assert_denied_everywhere(db, _service(db, fts), mine, control)


# --- parity with export, on one fixture holding every case --------------------


def test_served_chats_equal_export_eligible_chats(db: psycopg.Connection, fts: apsw.Connection) -> None:
    p = _people(db)
    chats: dict[str, Chat] = {}
    chats["eligible-dm"] = _eligible_dm(db, fts, p)
    chats["all-allowlisted-group"] = _chat(
        db, fts, "all-allowlisted-group", kind="group",
        participants=[p.owner, p.alice, p.bob],
        senders=[(p.alice, False), (p.frank, False), (None, True)],
    )
    chats["owner-content-only-participant-alice"] = _chat(
        db, fts, "owner-content-only-participant-alice",
        participants=[p.alice], senders=[(None, True), (p.alice, False)],
    )
    chats["no-participants"] = _chat(
        db, fts, "no-participants", participants=[], senders=[(p.alice, False)]
    )
    chats["participant-not-allowlisted"] = _chat(
        db, fts, "participant-not-allowlisted",
        participants=[p.owner, p.carol], senders=[(p.carol, False)],
    )
    chats["participant-says-no"] = _chat(
        db, fts, "participant-says-no",
        participants=[p.owner, p.dana], senders=[(p.dana, False)],
    )
    chats["former-member"] = _chat(
        db, fts, "former-member", kind="group",
        participants=[p.owner, p.alice], senders=[(p.alice, False), (p.evan, False)],
    )
    chats["unsent-outsider"] = _chat(
        db, fts, "unsent-outsider", participants=[p.owner, p.alice], senders=[(p.alice, False)]
    )
    insert_message(
        db, chat_id=chats["unsent-outsider"].chat_id, sender_person_id=p.evan,
        is_from_me=False, sent_at=_T0, text="retracted", is_unsent=True,
    )
    chats["unresolved-sender"] = _chat(
        db, fts, "unresolved-sender",
        participants=[p.owner, p.alice], senders=[(p.alice, False), (None, False)],
    )
    chats["unresolved-handle"] = _chat(
        db, fts, "unresolved-handle", participants=[p.owner, p.alice], senders=[(p.alice, False)]
    )
    add_raw_participant(db, chats["unresolved-handle"].chat_id, "raw-unresolved")
    chats["unallowlisted-handle"] = _chat(
        db, fts, "unallowlisted-handle",
        participants=[p.owner, p.alice], senders=[(p.alice, False)],
    )
    add_raw_participant(db, chats["unallowlisted-handle"].chat_id, "raw-gina", resolve_to_person=p.gina)
    chats["allowlisted-handle"] = _chat(
        db, fts, "allowlisted-handle", participants=[p.owner, p.alice], senders=[(p.alice, False)]
    )
    add_raw_participant(db, chats["allowlisted-handle"].chat_id, "raw-alice", resolve_to_person=p.alice)
    chats["tapback-outsider"] = _chat(
        db, fts, "tapback-outsider", participants=[p.owner, p.alice], senders=[(p.alice, False)]
    )
    insert_tapback(db, target_message_id=chats["tapback-outsider"].message_ids[0], sender_person_id=p.carol)
    service = _service(db, fts)

    designed = {"eligible-dm", "all-allowlisted-group", "owner-content-only-participant-alice",
                "allowlisted-handle"}
    by_thread = {chat.thread_key: name for name, chat in chats.items()}
    eligible = eligible_chat_ids(db)
    eligible_names = {name for name, chat in chats.items() if chat.chat_id in eligible}

    # What the public surface serves, observed from outside.
    served_by_conversation = {
        name for name, chat in chats.items() if _served_by_conversation(service, chat)
    }
    served_by_search = {by_thread[key] for key in _served_by_search(service)}
    assert served_by_conversation == eligible_names
    assert served_by_search == eligible_names
    assert eligible_names == designed

    # Export's two evaluations of the rule agree, whole-corpus and per chat.
    assert eligible == {c for c, v in compute_chat_eligibility(db).items() if v.eligible}
    for chat in chats.values():
        assert eligible_chat_ids(db, among=[chat.chat_id]) == {chat.chat_id} & eligible
        verdict = compute_chat_eligibility(db, among=[chat.chat_id]).get(chat.chat_id)
        assert verdict is not None
        assert verdict.eligible == (chat.chat_id in eligible)


# --- attachment text: the separate gate ----------------------------------------


def _attachment(
    db: psycopg.Connection, message_id: int, *, caption: str, filename: str = "photo.jpg"
) -> str:
    attachment_id = insert_attachment(db, filename=filename, mime_type="image/jpeg")
    link_attachment(db, message_id, attachment_id)
    add_enrichment(db, attachment_id, kind="caption", text=caption)
    with db.cursor() as cur:
        cur.execute("SELECT attachment_key FROM attachment WHERE attachment_id = %s", (attachment_id,))
        row = cur.fetchone()
        assert row is not None
        return str(row[0])


def test_attachment_text_needs_attachments_allowed_on_every_link(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """SPEC §11.2: an attachment's text ships only if its segment is
    eligible AND every message linking it into that segment has a sender
    with `attachments_allowed`. Bob is text-only."""
    p = _people(db)
    group = _chat(
        db, fts, "attachments", kind="group",
        participants=[p.owner, p.alice, p.bob],
        senders=[(p.alice, False), (p.bob, False), (p.alice, False)],
    )
    alice_msg, bob_msg, alice_again = group.message_ids
    from_alice = _attachment(db, alice_msg, caption="site survey sketch")
    from_bob = _attachment(db, bob_msg, caption="weekend beach photo")
    shared = insert_attachment(db, filename="shared.jpg", mime_type="image/jpeg")
    link_attachment(db, alice_again, shared)
    link_attachment(db, bob_msg, shared)
    add_enrichment(db, shared, kind="caption", text="forwarded twice")
    with db.cursor() as cur:
        cur.execute("SELECT attachment_key FROM attachment WHERE attachment_id = %s", (shared,))
        row = cur.fetchone()
        assert row is not None
        shared_key = str(row[0])
    service = _service(db, fts)
    assert eligible_chat_ids(db) == {group.chat_id}

    with pytest.raises(NotFoundError) as denied:
        service.get_attachment_text(PUBLIC, attachment_key=from_bob)
    with pytest.raises(NotFoundError):
        service.get_attachment_text(PUBLIC, attachment_key=shared_key)
    allowed = service.get_attachment_text(PUBLIC, attachment_key=from_alice)
    assert allowed["texts"] == [{"kind": "caption", "model": "", "text": "site survey sketch"}]

    # Denied reads exactly like a key that does not exist (existence oracle).
    unknown_key = "no-such-attachment-" + "0" * 20
    with pytest.raises(NotFoundError) as unknown:
        service.get_attachment_text(PUBLIC, attachment_key=unknown_key)
    assert str(denied.value).replace(from_bob, "KEY") == str(unknown.value).replace(unknown_key, "KEY")

    # The local surface is unaffected.
    local = service.get_attachment_text(LOCAL_FULL_ACCESS, attachment_key=from_bob)
    assert local["texts"][0]["text"] == "weekend beach photo"


def _conversation_text(service: RetrievalService, context: AccessContext, chat: Chat) -> str:
    out = service.get_conversation(context, thread_id=chat.thread_key, window=50)
    return "\n".join(str(m["text"]) for m in cast(list[dict[str, Any]], out["messages"]))


def test_conversation_withholds_attachment_content_the_gate_denies(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    p = _people(db)
    group = _chat(
        db, fts, "conversation-attachments", kind="group",
        participants=[p.owner, p.alice, p.bob],
        senders=[(p.alice, False), (p.bob, False)],
    )
    _attachment(db, group.message_ids[0], caption="site survey sketch")
    _attachment(db, group.message_ids[1], caption="weekend beach photo", filename="beach.jpg")
    service = _service(db, fts)

    public = _conversation_text(service, PUBLIC, group)
    assert "weekend beach photo" not in public
    assert "beach.jpg" not in public
    assert WITHHELD in public
    assert "site survey sketch" in public

    local = _conversation_text(service, LOCAL_FULL_ACCESS, group)
    assert "weekend beach photo" in local
    assert WITHHELD not in local


def test_search_text_withholds_attachment_content_the_gate_denies(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """The stored `segment.rendered_text` carries every attachment's
    snippet — the pipeline renders it without the allowlist — so the
    public surface must not return it as stored."""
    from imsg.segment.pipeline import refresh_segment_rendering

    p = _people(db)
    group = _chat(
        db, fts, "search-attachments", kind="group",
        participants=[p.owner, p.alice, p.bob],
        senders=[(p.alice, False), (p.bob, False)],
    )
    _attachment(db, group.message_ids[0], caption="site survey sketch")
    _attachment(db, group.message_ids[1], caption="weekend beach photo")
    stored, _ = refresh_segment_rendering(db, group.segment_id, cast(Any, _Cfg()))
    upsert_segment_row(fts, group.segment_id, "segment-search-attachments", stored)
    assert "weekend beach photo" in stored  # the stored text really does carry it
    service = _service(db, fts)

    [hit] = service.search_messages(PUBLIC, query=NEEDLE).results
    assert "weekend beach photo" not in str(hit["text"])
    assert WITHHELD in str(hit["text"])
    assert "site survey sketch" in str(hit["text"])

    [local_hit] = service.search_messages(LOCAL_FULL_ACCESS, query=NEEDLE).results
    assert local_hit["text"] == stored


def test_public_re_render_matches_the_pipeline_when_nothing_is_withheld(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """CONTROL — passes before and after the change. Guards the public
    re-render against drifting from the segment pipeline's own renderer:
    with every attachment allowed and default policy, the two agree."""
    from imsg.segment.pipeline import refresh_segment_rendering

    p = _people(db)
    group = _chat(
        db, fts, "render-parity", kind="group",
        participants=[p.owner, p.alice, p.frank],
        senders=[(p.alice, False), (None, True), (p.frank, False)],
    )
    _attachment(db, group.message_ids[0], caption="site survey sketch")
    insert_tapback(db, target_message_id=group.message_ids[0], sender_person_id=None, is_from_me=True)
    stored, _ = refresh_segment_rendering(db, group.segment_id, cast(Any, _Cfg()))
    upsert_segment_row(fts, group.segment_id, "segment-render-parity", stored)

    [hit] = _service(db, fts).search_messages(PUBLIC, query=NEEDLE).results
    assert hit["text"] == stored


def test_allowlist_never_serves_unsent_messages_or_edit_history(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """SPEC §11.2 / D1: unsent messages and prior edit versions never
    export, "regardless of `policy.*` flags". With the flags on, the
    local surface shows both; the public surface under `allowlist` shows
    neither."""
    from imsg.segment.pipeline import refresh_segment_rendering

    p = _people(db)
    dm = _chat(db, fts, "d1", participants=[p.owner, p.alice], senders=[(p.alice, False)])
    add_edit_history(db, dm.message_ids[0], version_idx=0, text="first draft wording")
    unsent = insert_message(
        db, chat_id=dm.chat_id, sender_person_id=p.alice, is_from_me=False,
        sent_at=_T0 + timedelta(minutes=5), text="retracted wording", is_unsent=True,
    )
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO segment_message (segment_id, message_id) VALUES (%s, %s)",
            (dm.segment_id, unsent),
        )
    stored, _ = refresh_segment_rendering(db, dm.segment_id, cast(Any, _ShowEverythingCfg()))
    upsert_segment_row(fts, dm.segment_id, "segment-d1", stored)
    assert "retracted wording" in stored and "first draft wording" in stored
    service = _service(db, fts, cfg=_ShowEverythingCfg())

    public_conversation = _conversation_text(service, PUBLIC, dm)
    assert "retracted wording" not in public_conversation
    assert "first draft wording" not in public_conversation
    [hit] = service.search_messages(PUBLIC, query=NEEDLE).results
    assert "retracted wording" not in str(hit["text"])
    assert "first draft wording" not in str(hit["text"])

    local_conversation = _conversation_text(service, LOCAL_FULL_ACCESS, dm)
    assert "retracted wording" in local_conversation
    assert "first draft wording" in local_conversation


# --- people ---------------------------------------------------------------------


def test_list_people_lists_only_people_in_eligible_chats_counting_only_their_messages(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    p = _people(db)
    _eligible_dm(db, fts, p)  # one message from alice
    _chat(  # ineligible: carol is not allowlisted
        db, fts, "with-carol", kind="group",
        participants=[p.owner, p.alice, p.carol],
        senders=[(p.alice, False), (p.alice, False), (p.carol, False)],
    )
    _chat(  # ineligible; frank is allowlisted but appears nowhere eligible
        db, fts, "frank-and-carol", participants=[p.frank, p.carol], senders=[(p.frank, False)]
    )
    listed = {
        str(person["short_name"]): person
        for person in cast(list[dict[str, Any]], _service(db, fts).list_people(PUBLIC)["people"])
    }

    assert set(listed) == {"owner", "alice"}
    assert listed["alice"]["message_count"] == 1
    assert listed["owner"]["message_count"] == 1  # the owner's is_from_me row in the eligible DM


def test_person_filter_resolves_only_people_the_scope_can_see(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """An exact name outside the scope is `PERSON_NOT_FOUND`, like a name
    that exists nowhere — not a silent empty success that confirms the
    person exists."""
    p = _people(db)
    _eligible_dm(db, fts, p)
    _chat(db, fts, "with-carol", participants=[p.owner, p.carol], senders=[(p.carol, False)])
    service = _service(db, fts)

    with pytest.raises(PersonNotFoundError):
        service.search_messages(PUBLIC, query=NEEDLE, people=["carol"])
    with pytest.raises(PersonNotFoundError):
        service.search_messages(PUBLIC, query=NEEDLE, people=["Carol Carpenter"])
    with pytest.raises((PersonAmbiguousError, PersonNotFoundError)) as near:
        service.search_messages(PUBLIC, query=NEEDLE, people=["Carl Carpenter"])
    assert all(c.short_name != "carol" for c in near.value.candidates)
    assert len(service.search_messages(PUBLIC, query=NEEDLE, people=["alice"]).results) == 1


def test_include_handles_is_refused_off_the_local_surface(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    p = _people(db)
    _eligible_dm(db, fts, p)
    with pytest.raises(InvalidArgumentError):
        _service(db, fts).list_people(PUBLIC, include_handles=True)


# --- existence oracle and the empty allowlist -------------------------------------


def test_denied_thread_reads_exactly_like_an_unknown_one(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """The public server returns an error's text verbatim, so the two
    NOT_FOUND messages must be identical (SPEC §10.2)."""
    p = _people(db)
    denied = _chat(db, fts, "with-carol", participants=[p.owner, p.carol], senders=[(p.carol, False)])
    service = _service(db, fts)

    with pytest.raises(NotFoundError) as unknown:
        service.get_conversation(PUBLIC, thread_id="thread-that-does-not-exist")
    with pytest.raises(NotFoundError) as refused:
        service.get_conversation(PUBLIC, thread_id=denied.thread_key)
    with pytest.raises(NotFoundError) as refused_by_segment:
        service.get_conversation(PUBLIC, thread_id="segment-with-carol")
    assert str(unknown.value) == str(refused.value) == str(refused_by_segment.value)


def test_nothing_eligible_answers_empty_without_touching_the_models(
    db: psycopg.Connection, fts: apsw.Connection
) -> None:
    """The live state today: an empty allowlist. Every channel would
    return nothing, so the search must not pay for a query embedding."""
    carol = insert_person(db, display_name="Carol Carpenter", short_name="carol")
    _chat(db, fts, "nobody-allowlisted", participants=[carol], senders=[(carol, False)])
    counting = _CountingTextProvider()

    result = _service(db, fts, text_provider=counting).search_messages(PUBLIC, query=NEEDLE)

    assert result.results == []
    assert counting.queries == []
    assert set(result.candidate_lists.values()) == {0}
