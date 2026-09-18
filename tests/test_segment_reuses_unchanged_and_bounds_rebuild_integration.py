"""Postgres integration tests for the two bounds on how much work one
re-segmentation run is allowed to do (SPEC §8 S4):

- **Fix A, reuse.** A recomputed segment identical to the stored one is
  left in place — no DELETE, no INSERT, no `search_index_event`, and
  the `segment_id` survives so `segment_embedding` is not cascaded
  away and S6 never re-embeds it.
- **Fix B, an end to the range.** The rebuild stops at the first
  persisted session starting more than one session gap after the last
  change, so sessions past there are not even recomputed.

Both were measured on the live corpus before the fix: 1,582 genuinely
changed messages re-segmented 236,300 messages and re-wrote 32,473
segments, every one of them re-embedded.

Skips cleanly when no scratch Postgres is reachable, same pattern as
`tests/test_segment_pipeline_integration.py`. Fictional personas only
(D5): Alice Example / Bob Builder, never real names.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.embed.vector_codec import vector_literal
from imsg.errors import BoundaryDetectionError
from imsg.segment.hashing import compute_seg_config_hash
from imsg.segment.models import MessageForSegmentation
from imsg.segment.pipeline import (
    REBUILD_ALL_SENTINEL,
    fetch_persisted_sessions,
    find_config_stale_chat_ids,
    find_dirty_chats,
    run_segment,
    run_segment_for_chat,
)
from imsg.segment.sessionize import compute_recompute_end

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_resegment_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
PROMPT_BYTES = b"fixed boundary-detection prompt for tests"
EMBEDDING_DIM = 2048


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

    conn = psycopg.connect(_dsn(TEST_DB_NAME))
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
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
def config(config_dict_factory: object) -> Config:
    return load_config_dict(config_dict_factory())  # type: ignore[operator]


@pytest.fixture
def split_config(config: Config) -> Config:
    """Thresholds low enough that a session of a few dozen messages
    becomes several segments — needed to tell "this session was
    rebuilt" apart from "this *segment* was rebuilt"."""
    return config.model_copy(
        update={"segmentation": config.segmentation.model_copy(update={"topical_min_messages": 5})}
    )


class RecordingBoundaryProvider:
    """`FakeBoundaryProvider` that also remembers which messages it was
    asked about — the direct evidence for Fix B, since a bounded run
    must never even *score* a session it is not rebuilding. Boundary
    detection is the expensive call on the real pipeline too."""

    model_id = "fake/boundary-detector@test"

    def __init__(self, *, messages_per_segment: int = 6) -> None:
        self._messages_per_segment = messages_per_segment
        self.calls = 0
        self.seen_message_ids: set[int] = set()

    def detect_boundaries(self, window: Sequence[MessageForSegmentation]) -> list[int]:
        self.calls += 1
        self.seen_message_ids.update(m.message_id for m in window)
        return list(range(self._messages_per_segment, len(window), self._messages_per_segment))


class AlwaysFailingBoundaryProvider:
    """Proves a claim about what was *not* recomputed: any session this
    is asked about raises, and `segment_session` falls back to
    session-as-segment, which shows up as `fallback_sessions`."""

    model_id = "fake/boundary-detector@test"

    def detect_boundaries(self, window: Sequence[MessageForSegmentation]) -> list[int]:
        raise BoundaryDetectionError("configured to fail")


# --- fixture-data helpers -------------------------------------------------


def _insert_person(
    conn: psycopg.Connection, *, display_name: str, short_name: str, is_owner: bool = False
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO person (display_name, short_name, is_owner, needs_review) "
            "VALUES (%s, %s, %s, false) RETURNING person_id",
            (display_name, short_name, is_owner),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _insert_chat(conn: psycopg.Connection, *, source_guid: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat (source_guid, thread_key, kind, display_name) "
            "VALUES (%s, %s, 'dm', NULL) RETURNING chat_id",
            (source_guid, f"thread-{source_guid}"),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _add_participant(conn: psycopg.Connection, chat_id: int, person_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_participant (chat_id, person_id) VALUES (%s, %s)",
            (chat_id, person_id),
        )


def _insert_messages(
    conn: psycopg.Connection,
    *,
    chat_id: int,
    senders: Sequence[tuple[int, bool]],
    sent_ats: Sequence[datetime],
) -> list[int]:
    """One round trip for the whole batch — the scale fixture inserts a
    few thousand rows and a per-row `execute` dominates its runtime."""
    rows = []
    for (person_id, is_from_me), sent_at in zip(senders, sent_ats, strict=True):
        guid = f"msg-{uuid.uuid4()}"
        rows.append((guid, f"key-{guid}", chat_id, person_id, is_from_me, sent_at, f"body {guid}"))
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO message (
                source_guid, message_key, chat_id, sender_person_id,
                is_from_me, sent_at, service, text_original, text_normalized
            ) VALUES (%s, %s, %s, %s, %s, %s, 'imessage', %s, %s)
            """,
            [(*r, r[6]) for r in rows],
        )
        cur.execute(
            "SELECT message_id FROM message WHERE chat_id = %s ORDER BY sent_at, message_id",
            (chat_id,),
        )
        return [int(r[0]) for r in cur.fetchall()]


@pytest.fixture
def dm_chat(scratch_db: psycopg.Connection) -> tuple[int, int, int]:
    """Returns (chat_id, owner_person_id, alice_person_id)."""
    owner_id = _insert_person(
        scratch_db, display_name="Jamie Owner", short_name="owner", is_owner=True
    )
    alice_id = _insert_person(scratch_db, display_name="Alice Example", short_name="alice")
    chat_id = _insert_chat(scratch_db, source_guid="chat-dm-resegment")
    _add_participant(scratch_db, chat_id, owner_id)
    _add_participant(scratch_db, chat_id, alice_id)
    scratch_db.commit()
    return chat_id, owner_id, alice_id


_BASE = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


def _build_sessions(
    conn: psycopg.Connection,
    *,
    chat_id: int,
    owner_id: int,
    alice_id: int,
    session_count: int,
    messages_per_session: int,
    session_stride_hours: int = 24,
) -> list[list[int]]:
    """`session_stride_hours` apart is far beyond the 3h default gap, so
    each block is its own sealed session. Returns the message ids
    grouped by session, in time order."""
    senders: list[tuple[int, bool]] = []
    sent_ats: list[datetime] = []
    for s in range(session_count):
        for i in range(messages_per_session):
            senders.append((owner_id, True) if i % 2 == 0 else (alice_id, False))
            sent_ats.append(_BASE + timedelta(hours=s * session_stride_hours, minutes=i))
    ids = _insert_messages(conn, chat_id=chat_id, senders=senders, sent_ats=sent_ats)
    conn.commit()
    return [
        ids[s * messages_per_session : (s + 1) * messages_per_session] for s in range(session_count)
    ]


# --- observation helpers --------------------------------------------------


def _segment_rows(conn: psycopg.Connection, chat_id: int) -> list[tuple[Any, ...]]:
    """Every column of every segment that a rewrite would disturb, in a
    stable order — `created_at` included, which is what tells "left
    alone" apart from "deleted and re-inserted identically"."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT seg.segment_id, seg.session_id, seg.seq_in_session, seg.stable_key,
                   seg.started_at, seg.ended_at, seg.message_count, seg.token_count,
                   seg.rendered_sha256, seg.seg_config_hash, seg.created_at
            FROM segment seg
            JOIN session sess ON sess.session_id = seg.session_id
            WHERE seg.chat_id = %s
            ORDER BY sess.started_at, seg.seq_in_session
            """,
            (chat_id,),
        )
        return list(cur.fetchall())


def _session_rows(conn: psycopg.Connection, chat_id: int) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_id, started_at, ended_at, gap_hours FROM session "
            "WHERE chat_id = %s ORDER BY started_at",
            (chat_id,),
        )
        return list(cur.fetchall())


def _events(conn: psycopg.Connection) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT entity_id, operation FROM search_index_event "
            "WHERE entity_kind = 'segment' ORDER BY event_id"
        )
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def _clear_events(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM search_index_event")
    conn.commit()


def _fake_embed_everything(conn: psycopg.Connection, chat_id: int) -> dict[int, datetime]:
    """Give every segment a `segment_embedding` row, so a later
    assertion can show the vectors survived. That row is what costs
    hours to recompute, and it disappears by FK cascade the moment its
    `segment_id` does — `search_index_event` is the FTS sidecar's
    concern, not S6's."""
    literal = vector_literal([0.0] * EMBEDDING_DIM)
    with conn.cursor() as cur:
        cur.execute("SELECT segment_id FROM segment WHERE chat_id = %s", (chat_id,))
        segment_ids = [int(r[0]) for r in cur.fetchall()]
        cur.executemany(
            "INSERT INTO segment_embedding (segment_id, model, text_sha256, vec) "
            "VALUES (%s, 'fake/embedder@test', 'sha', %s::halfvec)",
            [(sid, literal) for sid in segment_ids],
        )
        cur.execute(
            "SELECT segment_id, embedded_at FROM segment_embedding "
            "WHERE segment_id = ANY(%s)",
            (segment_ids,),
        )
        embedded = {int(r[0]): r[1] for r in cur.fetchall()}
    conn.commit()
    return embedded


def _embedding_rows(conn: psycopg.Connection, chat_id: int) -> dict[int, datetime]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT e.segment_id, e.embedded_at FROM segment_embedding e "
            "JOIN segment s USING (segment_id) WHERE s.chat_id = %s",
            (chat_id,),
        )
        return {int(r[0]): r[1] for r in cur.fetchall()}


def _edit(conn: psycopg.Connection, message_id: int, text: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE message SET text_original = %s, text_normalized = %s, "
            "is_edited = true, updated_at = now() WHERE message_id = %s",
            (text, text, message_id),
        )
    conn.commit()


def _resegment(
    conn: psycopg.Connection,
    chat_id: int,
    config: Config,
    provider: object,
) -> Any:
    """Re-segment exactly the way `run_segment` does: both ends of the
    dirty span, nothing hand-picked."""
    dirty = find_dirty_chats(conn, index_unsent=config.policy.index_unsent)
    assert chat_id in dirty, f"chat {chat_id} is not dirty: {dirty}"
    report = run_segment_for_chat(
        conn,
        chat_id,
        config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=dirty[chat_id].earliest_changed_at,
        latest_changed_at=dirty[chat_id].latest_changed_at,
    )
    conn.commit()
    return report


# --- Fix A: identical segments keep their rows ----------------------------


def test_an_edit_rebuilds_only_its_own_segment_inside_the_session(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """One session, five segments, one edited message in the first of
    them. The other four re-render byte-for-byte, so their rows — ids,
    `created_at`, embeddings — must survive untouched and emit no index
    events.

    This isolates Fix A from Fix B: everything here is inside a single
    session, so the end bound cannot help. Against the pre-fix code the
    whole session is deleted and re-inserted, so all five segments get
    new ids, new `created_at`, a delete + an upsert event each, and
    five cascaded-away embeddings.
    """
    chat_id, owner_id, alice_id = dm_chat
    [session_message_ids] = _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=1,
        messages_per_session=30,
    )

    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    before = _segment_rows(scratch_db, chat_id)
    assert len(before) == 5, "fixture must produce several segments in ONE session"
    sessions_before = _session_rows(scratch_db, chat_id)
    assert len(sessions_before) == 1
    embeddings_before = _fake_embed_everything(scratch_db, chat_id)
    _clear_events(scratch_db)

    # An edit in the *first* segment (message index 1 of 30, segments of 6).
    _edit(scratch_db, session_message_ids[1], "corrected body")

    report = _resegment(scratch_db, chat_id, split_config, provider)

    after = _segment_rows(scratch_db, chat_id)
    assert len(after) == 5

    # The four later segments are the same rows, not equal-looking new ones.
    assert after[1:] == before[1:], (
        "segments after the edited one were rewritten: ids and/or created_at moved"
    )
    assert after[0][0] != before[0][0], "the edited segment must be rebuilt"
    assert after[0][8] != before[0][8], "the edited segment's rendered_sha256 must change"

    # The session row itself is reused, so the survivors keep pointing at it.
    assert _session_rows(scratch_db, chat_id) == sessions_before

    assert report.skipped_unchanged == 4
    assert report.segments_written == 1
    assert report.segments_deleted == 1
    assert report.sessions_written == 0

    # Exactly two events, both about the one segment that changed.
    events = _events(scratch_db)
    assert events == [(before[0][0], "delete"), (after[0][0], "upsert")]

    # And the expensive artifact: four of five vectors never moved.
    embeddings_after = _embedding_rows(scratch_db, chat_id)
    survivors = {row[0] for row in before[1:]}
    assert set(embeddings_after) == survivors
    assert all(embeddings_after[sid] == embeddings_before[sid] for sid in survivors)


# --- Fix B: the rebuild has an end ---------------------------------------


def test_sessions_a_gap_past_the_last_change_are_never_even_recomputed(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """Six sealed sessions a day apart, one edit in the first. The
    rebuild must stop at the second session.

    The evidence is deliberately *not* "the later rows are unchanged" —
    Fix A alone would produce that. It is that the boundary provider,
    the expensive call, is never shown a message from any later session
    at all. Against the pre-fix code it sees all six.
    """
    chat_id, owner_id, alice_id = dm_chat
    session_message_ids = _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=6,
        messages_per_session=20,
    )

    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    before = _segment_rows(scratch_db, chat_id)
    sessions_before = _session_rows(scratch_db, chat_id)
    assert len(sessions_before) == 6
    embeddings_before = _fake_embed_everything(scratch_db, chat_id)
    _clear_events(scratch_db)

    _edit(scratch_db, session_message_ids[0][3], "corrected body")

    # The bound the run will use, stated up front.
    dirty = find_dirty_chats(scratch_db, index_unsent=split_config.policy.index_unsent)
    spans = fetch_persisted_sessions(scratch_db, chat_id)
    bound = compute_recompute_end(
        spans,
        dirty[chat_id].latest_changed_at,
        split_config.segmentation.session_gap_hours,
    )
    assert bound == spans[1].started_at

    provider.calls = 0
    provider.seen_message_ids = set()
    report = _resegment(scratch_db, chat_id, split_config, provider)

    beyond_the_bound = {mid for ids in session_message_ids[1:] for mid in ids}
    assert provider.seen_message_ids
    assert not (provider.seen_message_ids & beyond_the_bound), (
        "the boundary provider was shown messages from sessions past the bound — "
        "those sessions were recomputed, which is the multi-hour cost this bound removes"
    )
    assert provider.seen_message_ids <= set(session_message_ids[0])

    # Nothing past the bound was written, deleted or re-embedded.
    after = _segment_rows(scratch_db, chat_id)
    first_session_id = sessions_before[0][0]
    assert [r for r in after if r[1] != first_session_id] == [
        r for r in before if r[1] != first_session_id
    ]
    assert _session_rows(scratch_db, chat_id) == sessions_before
    assert _embedding_rows(scratch_db, chat_id).items() >= {
        (sid, ts) for sid, ts in embeddings_before.items() if sid not in {r[0] for r in before if r[1] == first_session_id}
    }

    touched = {entity_id for entity_id, _ in _events(scratch_db)}
    untouched_segment_ids = {r[0] for r in before if r[1] != first_session_id}
    assert not (touched & untouched_segment_ids)
    assert report.segments_written == 1
    assert report.segments_deleted == 1
    assert report.skipped_unchanged == 3  # the other 3 segments of session 1


def test_an_unbounded_run_is_still_possible_when_the_change_reaches_the_tail(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """The bound must not fire when it cannot be justified: a message
    appended to the tail session leaves no persisted session starting a
    full gap later, so there is no bound and the run behaves exactly as
    it did before."""
    chat_id, owner_id, alice_id = dm_chat
    _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=2,
        messages_per_session=8,
    )
    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    _insert_messages(
        scratch_db,
        chat_id=chat_id,
        senders=[(alice_id, False)],
        sent_ats=[_BASE + timedelta(hours=24, minutes=20)],  # inside session 2's gap
    )
    scratch_db.commit()

    dirty = find_dirty_chats(scratch_db, index_unsent=split_config.policy.index_unsent)
    spans = fetch_persisted_sessions(scratch_db, chat_id)
    assert (
        compute_recompute_end(
            spans,
            dirty[chat_id].latest_changed_at,
            split_config.segmentation.session_gap_hours,
        )
        is None
    )

    _resegment(scratch_db, chat_id, split_config, provider)
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM segment_message sm JOIN segment s USING (segment_id) "
            "WHERE s.chat_id = %s",
            (chat_id,),
        )
        assert cur.fetchone() == (17,)  # 8 + 8 + the new one, each in exactly one segment
    assert chat_id not in find_dirty_chats(
        scratch_db, index_unsent=split_config.policy.index_unsent
    )


# --- the freeze mechanism still wins -------------------------------------


def test_a_seg_config_hash_change_rebuilds_every_segment(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """The sharpest case for D4's freeze: only `boundary_revision`
    changes, so the segmentation outcome and every rendered byte are
    identical and *only* `seg_config_hash` (and, through it,
    `stable_key`) differs. Reuse must still match nothing."""
    chat_id, owner_id, alice_id = dm_chat
    _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=3,
        messages_per_session=12,
    )
    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    before = _segment_rows(scratch_db, chat_id)
    assert len(before) == 6
    _fake_embed_everything(scratch_db, chat_id)
    _clear_events(scratch_db)

    new_config = split_config.model_copy(
        update={
            "segmentation": split_config.segmentation.model_copy(
                update={"boundary_revision": "0000000000000000000000000000000000000000"}
            )
        }
    )
    new_hash = compute_seg_config_hash(
        session_gap_hours=new_config.segmentation.session_gap_hours,
        topical_min_messages=new_config.segmentation.topical_min_messages,
        max_messages=new_config.segmentation.max_messages,
        max_tokens=new_config.segmentation.max_tokens,
        boundary_model=new_config.segmentation.boundary_model,
        boundary_revision=new_config.segmentation.boundary_revision,
        boundary_prompt_bytes=PROMPT_BYTES,
        index_unsent=new_config.policy.index_unsent,
        index_edit_history=new_config.policy.index_edit_history,
    )
    assert new_hash != before[0][9]
    assert find_config_stale_chat_ids(scratch_db, current_seg_config_hash=new_hash) == {chat_id}

    report = run_segment_for_chat(
        scratch_db,
        chat_id,
        new_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    after = _segment_rows(scratch_db, chat_id)
    assert report.skipped_unchanged == 0, "a config change must reuse nothing"
    assert report.segments_written == len(before)
    assert report.segments_deleted == len(before)
    assert not ({r[0] for r in after} & {r[0] for r in before}), "every segment_id must be new"
    assert {r[9] for r in after} == {new_hash}
    assert find_config_stale_chat_ids(scratch_db, current_seg_config_hash=new_hash) == set()

    events = _events(scratch_db)
    assert sorted(e for e in events if e[1] == "delete") == sorted(
        (r[0], "delete") for r in before
    )
    assert len([e for e in events if e[1] == "upsert"]) == len(before)

    # The session rows are rewritten too, so `gap_hours` can never be
    # left describing a threshold the segments no longer use.
    assert not ({r[0] for r in _session_rows(scratch_db, chat_id)} & {r[1] for r in before})


# --- nothing changed means nothing happens -------------------------------


def test_a_chat_with_no_changes_is_left_completely_untouched(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """Both fixes together. Forcing a full-range recompute over a chat
    nothing has touched must produce zero writes and zero index events
    — every recomputed segment matches, so `skipped_unchanged` accounts
    for all of them."""
    chat_id, owner_id, alice_id = dm_chat
    _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=4,
        messages_per_session=14,
    )
    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    segments_before = _segment_rows(scratch_db, chat_id)
    sessions_before = _session_rows(scratch_db, chat_id)
    embeddings_before = _fake_embed_everything(scratch_db, chat_id)
    _clear_events(scratch_db)

    # Nothing is dirty, so the real entry point does no work at all.
    assert find_dirty_chats(scratch_db, index_unsent=split_config.policy.index_unsent) == {}
    assert (
        run_segment(scratch_db, split_config, provider, PROMPT_BYTES) == []  # type: ignore[arg-type]
    )
    scratch_db.commit()

    # And forcing the whole chat through the recompute anyway still
    # writes nothing — the skip, not the dirty check, is what proves it.
    report = run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    assert report.skipped_unchanged == len(segments_before)
    assert report.segments_written == 0
    assert report.segments_deleted == 0
    assert report.sessions_written == 0
    assert _segment_rows(scratch_db, chat_id) == segments_before
    assert _session_rows(scratch_db, chat_id) == sessions_before
    assert _embedding_rows(scratch_db, chat_id) == embeddings_before
    assert _events(scratch_db) == []


def test_segmenting_twice_in_a_row_writes_nothing_the_second_time(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """Idempotency through the top-level entry point, including the case
    that used to hide it: an edit, a real rebuild, and then a second run
    that must find nothing left to do."""
    chat_id, owner_id, alice_id = dm_chat
    session_message_ids = _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=3,
        messages_per_session=14,
    )
    provider = RecordingBoundaryProvider(messages_per_segment=6)

    first = run_segment(scratch_db, split_config, provider, PROMPT_BYTES)  # type: ignore[arg-type]
    scratch_db.commit()
    assert [r.segments_written for r in first] == [9]  # 3 sessions x (6, 6, 2)

    second = run_segment(scratch_db, split_config, provider, PROMPT_BYTES)  # type: ignore[arg-type]
    scratch_db.commit()
    assert second == [], "a second run must not even find the chat dirty"

    _edit(scratch_db, session_message_ids[1][2], "corrected body")
    _clear_events(scratch_db)
    _resegment(scratch_db, chat_id, split_config, provider)

    segments_after_edit = _segment_rows(scratch_db, chat_id)
    sessions_after_edit = _session_rows(scratch_db, chat_id)
    events_after_edit = _events(scratch_db)
    assert len(events_after_edit) == 2

    third = run_segment(scratch_db, split_config, provider, PROMPT_BYTES)  # type: ignore[arg-type]
    scratch_db.commit()
    assert third == [], "the rebuild must have cleared the dirty flag, not carried it forward"
    assert _segment_rows(scratch_db, chat_id) == segments_after_edit
    assert _session_rows(scratch_db, chat_id) == sessions_after_edit
    assert _events(scratch_db) == events_after_edit


def test_a_change_that_renders_identically_still_clears_the_dirty_flag(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """The livelock the reuse check has to avoid. A message can be
    bumped without changing a rendered byte — `date_edited` moving, a
    sender re-pointed to a person with the same `short_name`, a
    re-extraction touching a non-rendered column. Reuse keeps the old
    `created_at`, and `find_dirty_chats` compares `message.updated_at`
    against exactly that, so reusing such a segment would leave the chat
    dirty forever and re-run the recompute every night.

    `_segment_is_unchanged` therefore refuses to reuse a segment that
    holds a flagged message, even when it re-renders identically.
    """
    chat_id, owner_id, alice_id = dm_chat
    session_message_ids = _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=2,
        messages_per_session=14,
    )
    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment(scratch_db, split_config, provider, PROMPT_BYTES)  # type: ignore[arg-type]
    scratch_db.commit()

    before = _segment_rows(scratch_db, chat_id)
    _clear_events(scratch_db)

    # A bump with no rendered consequence at all.
    with scratch_db.cursor() as cur:
        cur.execute(
            "UPDATE message SET updated_at = now() WHERE message_id = %s",
            (session_message_ids[0][2],),
        )
    scratch_db.commit()
    assert chat_id in find_dirty_chats(scratch_db, index_unsent=split_config.policy.index_unsent)

    report = _resegment(scratch_db, chat_id, split_config, provider)
    after = _segment_rows(scratch_db, chat_id)

    # The holding segment is rewritten (same content, new row) ...
    assert report.segments_written == 1
    assert after[0][0] != before[0][0]
    assert after[0][8] == before[0][8], "content really was identical"
    # ... its siblings are not ...
    assert after[1:] == before[1:]
    # ... and the chat is clean, which is the whole point.
    assert find_dirty_chats(scratch_db, index_unsent=split_config.policy.index_unsent) == {}


# --- the combined effect, at a size where it shows -----------------------


def test_one_early_edit_in_a_large_chat_touches_only_a_handful_of_segments(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """The shape of the incident, shrunk to fit a test: a chat of a few
    thousand messages across many sessions, one edit early on.

    Before the two fixes this rebuilt every session from the edit to the
    end of the chat — all ~50 sessions, all ~400 segments, every one of
    them re-embedded. Now it rebuilds one segment.
    """
    chat_id, owner_id, alice_id = dm_chat
    session_count = 50
    session_message_ids = _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=session_count,
        messages_per_session=48,
    )
    total_messages = session_count * 48

    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        provider,  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    before = _segment_rows(scratch_db, chat_id)
    _clear_events(scratch_db)
    assert len(before) >= 300, "fixture must be big enough for the effect to be visible"

    # One edit, in the third session of fifty.
    _edit(scratch_db, session_message_ids[2][5], "corrected body")

    provider.seen_message_ids = set()
    report = _resegment(scratch_db, chat_id, split_config, provider)
    after = _segment_rows(scratch_db, chat_id)

    rebuilt = [r for r in after if r[0] not in {b[0] for b in before}]
    # What the pre-fix code would have done: rebuild every session from
    # the edited one to the end of the chat, and re-embed all of it.
    edited_session_start = _BASE + timedelta(hours=2 * 24)
    unbounded_segments = [r for r in before if r[4] >= edited_session_start]
    print(
        f"\nlarge-chat fixture: {total_messages} messages, {session_count} sessions, "
        f"{len(before)} segments\n"
        f"  messages recomputed : {len(provider.seen_message_ids)} "
        f"(unbounded: {total_messages - 2 * 48})\n"
        f"  segments rebuilt    : {len(rebuilt)} "
        f"(unbounded: {len(unbounded_segments)})\n"
        f"  segments reused     : {report.skipped_unchanged}\n"
        f"  index events emitted: {len(_events(scratch_db))} "
        f"(unbounded: {2 * len(unbounded_segments)})\n"
    )

    assert len(rebuilt) == 1
    assert report.segments_written == 1
    assert report.segments_deleted == 1
    assert len(_events(scratch_db)) == 2
    # Everything past the third session was not even scored.
    beyond = {mid for ids in session_message_ids[3:] for mid in ids}
    assert not (provider.seen_message_ids & beyond)
    assert chat_id not in find_dirty_chats(
        scratch_db, index_unsent=split_config.policy.index_unsent
    )


def test_a_rebuilt_session_that_falls_back_proves_the_others_were_skipped(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """A second, independent read on "which sessions were recomputed":
    swap in a boundary provider that always fails, so every session the
    run actually touches shows up as a fallback. One edit in the first
    of six sessions must produce exactly one fallback."""
    chat_id, owner_id, alice_id = dm_chat
    session_message_ids = _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=6,
        messages_per_session=20,
    )
    run_segment_for_chat(
        scratch_db,
        chat_id,
        split_config,
        RecordingBoundaryProvider(messages_per_segment=6),  # type: ignore[arg-type]
        PROMPT_BYTES,
        earliest_changed_at=REBUILD_ALL_SENTINEL,
    )
    scratch_db.commit()

    _edit(scratch_db, session_message_ids[0][1], "corrected body")
    report = _resegment(scratch_db, chat_id, split_config, AlwaysFailingBoundaryProvider())
    assert report.fallback_sessions == 1, (
        "more than one session fell back, so more than one session was recomputed"
    )


def test_a_message_that_merges_two_sessions_moves_cleanly_between_segments(
    scratch_db: psycopg.Connection,
    dm_chat: tuple[int, int, int],
    split_config: Config,
) -> None:
    """The case where reuse and rewriting have to coexist inside one
    transaction. A message landing in the hole between two sessions
    narrows both gaps enough to merge them: the first session's row is
    reused (its `started_at` is unchanged) and its early segments match,
    the second session disappears, and its messages move into new
    segments of the first.

    `segment_message` has a UNIQUE index on `message_id` alone, so a
    message can belong to exactly one segment at any instant — the
    deletes therefore have to land before the inserts inside the same
    transaction, which is what this exercises.
    """
    chat_id, owner_id, alice_id = dm_chat
    # Two sessions four hours apart: beyond the 3h gap, so they are
    # separate, but close enough that one message in between closes it.
    session_message_ids = _build_sessions(
        scratch_db,
        chat_id=chat_id,
        owner_id=owner_id,
        alice_id=alice_id,
        session_count=2,
        messages_per_session=14,
        session_stride_hours=4,
    )
    provider = RecordingBoundaryProvider(messages_per_segment=6)
    run_segment(scratch_db, split_config, provider, PROMPT_BYTES)  # type: ignore[arg-type]
    scratch_db.commit()

    before = _segment_rows(scratch_db, chat_id)
    sessions_before = _session_rows(scratch_db, chat_id)
    assert len(sessions_before) == 2
    _clear_events(scratch_db)

    # Session 1 ends at +13min; session 2 starts at +4h. A message at
    # +2h is within 3h of both, so the three become one session.
    _insert_messages(
        scratch_db,
        chat_id=chat_id,
        senders=[(alice_id, False)],
        sent_ats=[_BASE + timedelta(hours=2)],
    )
    scratch_db.commit()

    _resegment(scratch_db, chat_id, split_config, provider)

    after_sessions = _session_rows(scratch_db, chat_id)
    assert len(after_sessions) == 1, "the two sessions must have merged into one"
    assert after_sessions[0][0] == sessions_before[0][0], "the first session's row is reused"
    assert after_sessions[0][2] == _BASE + timedelta(hours=4, minutes=13)

    # Every message is in exactly one segment, and nothing is dirty.
    every_message = [*session_message_ids[0], *session_message_ids[1]]
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(DISTINCT message_id) FROM segment_message sm "
            "JOIN segment s USING (segment_id) WHERE s.chat_id = %s",
            (chat_id,),
        )
        assert cur.fetchone() == (len(every_message) + 1, len(every_message) + 1)
    assert find_dirty_chats(scratch_db, index_unsent=split_config.policy.index_unsent) == {}

    # The first session's opening segments were untouched by all of that.
    after = _segment_rows(scratch_db, chat_id)
    assert after[0] == before[0]
    assert after[1] == before[1]
