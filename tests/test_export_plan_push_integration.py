"""End-to-end tests of the export control flow (SPEC §11.1, §11.4):
plan → review → approve → push, with every drift/abort path exercised
against a live scratch Postgres and the FakeTransport (nothing here can
reach a network). Fictional personas only (D5)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from _export_fixtures import (
    add_chunk,
    add_edit_history,
    add_enrichment,
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
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.export.documents import segment_document_id
from imsg.export.errors import ExportDriftError, ExportPushError
from imsg.export.planner import plan_export
from imsg.export.purge import purge_person
from imsg.export.push import push_export
from imsg.export.review import (
    APPROVAL_CONFIG_CHANGED,
    APPROVAL_DELETES,
    APPROVAL_FIRST_PUSH,
    APPROVAL_NEW_PERSON,
    APPROVAL_NEW_THREAD,
    approve_run,
)
from imsg.export.transport import FakeTransport

TEST_DB_NAME = "imsg_index_export_plan_test"

pytestmark = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)

_T = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(TEST_DB_NAME)


@pytest.fixture
def config(config_dict_factory: object) -> Config:
    return load_config_dict(config_dict_factory())  # type: ignore[operator]


@dataclass
class Corpus:
    owner_id: int
    alice_id: int
    bob_id: int
    dm_chat_id: int
    dm_segment_id: int
    dm_stable_key: str
    dm_message_edited_id: int
    group_chat_id: int
    group_segment_id: int
    group_stable_key: str
    group_message_id: int
    pdf_attachment_id: int
    image_attachment_id: int

    @property
    def dm_doc_id(self) -> str:
        return segment_document_id(self.dm_stable_key)

    @property
    def group_doc_id(self) -> str:
        return segment_document_id(self.group_stable_key)


@pytest.fixture
def corpus(db: psycopg.Connection) -> Corpus:
    """owner + alice (fully allowlisted), bob (text-only).

    - DM owner/alice: two messages; the second was edited (prior text
      'SECRET OLD DRAFT'); an unsent message ('RETRACTED CONTENT') is a
      member of the same segment; alice sent a PDF (enriched, 2 chunks).
    - Group owner/alice/bob: bob sent an image with a caption — bob is
      text-allowed but NOT attachments-allowed.
    """
    owner_id = insert_person(db, display_name="Jamie Owner", short_name="owner", is_owner=True)
    alice_id = insert_person(db, display_name="Alice Example", short_name="alice")
    bob_id = insert_person(db, display_name="Bob Builder", short_name="bob")
    allow(db, owner_id, text=True, attachments=True)
    allow(db, alice_id, text=True, attachments=True)
    allow(db, bob_id, text=True, attachments=False)

    dm_chat_id = insert_chat(db, source_guid="chat-dm")
    add_participant(db, dm_chat_id, owner_id)
    add_participant(db, dm_chat_id, alice_id)
    m1 = insert_message(
        db, chat_id=dm_chat_id, sender_person_id=owner_id, is_from_me=True,
        sent_at=_T, text="did the revised bid come through?",
    )
    m2 = insert_message(
        db, chat_id=dm_chat_id, sender_person_id=alice_id, is_from_me=False,
        sent_at=_T + timedelta(minutes=5), text="yes, see attached",
    )
    add_edit_history(db, m2, version_idx=0, text="SECRET OLD DRAFT")
    m3 = insert_message(
        db, chat_id=dm_chat_id, sender_person_id=alice_id, is_from_me=False,
        sent_at=_T + timedelta(minutes=6), text="RETRACTED CONTENT", is_unsent=True,
    )
    dm_stable_key = "stable-dm-1"
    dm_segment_id = insert_segment(
        db, chat_id=dm_chat_id, started_at=_T, ended_at=_T + timedelta(minutes=6),
        message_ids=[m1, m2, m3], stable_key=dm_stable_key,
    )
    pdf = insert_attachment(db, filename="bid-rev3.pdf", mime_type="application/pdf")
    link_attachment(db, m2, pdf)
    add_enrichment(db, pdf, kind="pdf_text", text="Deck rebuild, materials $14,200")
    add_chunk(db, pdf, kind="pdf_text", seq=0, text="Deck rebuild, materials $14,200")
    add_chunk(db, pdf, kind="pdf_text", seq=1, text="labor and permits, $9,800")

    group_chat_id = insert_chat(
        db, source_guid="chat-group", kind="group", display_name="Acme deck crew"
    )
    for pid in (owner_id, alice_id, bob_id):
        add_participant(db, group_chat_id, pid)
    g1 = insert_message(
        db, chat_id=group_chat_id, sender_person_id=bob_id, is_from_me=False,
        sent_at=_T, text="progress photo",
    )
    group_stable_key = "stable-group-1"
    group_segment_id = insert_segment(
        db, chat_id=group_chat_id, started_at=_T, ended_at=_T,
        message_ids=[g1], stable_key=group_stable_key,
    )
    image = insert_attachment(db, filename="weekend-lake-trip.jpg", mime_type="image/jpeg")
    link_attachment(db, g1, image)
    add_enrichment(db, image, kind="caption", text="PRIVATE CAPTION about the lake")
    add_chunk(db, image, kind="caption", seq=0, text="PRIVATE CAPTION about the lake")

    db.commit()
    return Corpus(
        owner_id=owner_id,
        alice_id=alice_id,
        bob_id=bob_id,
        dm_chat_id=dm_chat_id,
        dm_segment_id=dm_segment_id,
        dm_stable_key=dm_stable_key,
        dm_message_edited_id=m2,
        group_chat_id=group_chat_id,
        group_segment_id=group_segment_id,
        group_stable_key=group_stable_key,
        group_message_id=g1,
        pdf_attachment_id=pdf,
        image_attachment_id=image,
    )


def _staged_doc(staging_dir: str, document_id: str) -> str:
    return (Path(staging_dir) / "docs" / f"{document_id}.txt").read_text(encoding="utf-8")


def _happy_path(
    db: psycopg.Connection, config: Config, transport: FakeTransport
) -> None:
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()
    result = push_export(db, config, plan.run_id, transport)
    db.commit()
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_first_plan_stages_gated_docs_and_requires_approval(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    plan = plan_export(db, config)
    db.commit()

    assert plan.approval_required
    assert plan.approval_reasons == (APPROVAL_FIRST_PUSH,)
    # 2 segment docs + 2 pdf chunk docs; bob's image chunk is gated out
    assert plan.upsert_count == 4
    assert plan.delete_count == 0

    dm_text = _staged_doc(plan.staging_dir, corpus.dm_doc_id)
    assert "did the revised bid come through?" in dm_text
    assert "yes, see attached" in dm_text
    # D1, hard-coded: no unsent text, no prior edit versions — ever.
    assert "RETRACTED CONTENT" not in dm_text
    assert "SECRET OLD DRAFT" not in dm_text
    # alice's PDF snippet is present (she is attachments-allowed)
    assert 'bid-rev3.pdf' in dm_text

    group_text = _staged_doc(plan.staging_dir, corpus.group_doc_id)
    # bob is text-allowed: his words export...
    assert "progress photo" in group_text
    # ...but his attachment is withheld with zero metadata (separate gate)
    assert "[attachment not exported]" in group_text
    assert "weekend-lake-trip" not in group_text
    assert "PRIVATE CAPTION" not in group_text

    report = Path(plan.report_path).read_text(encoding="utf-8")
    assert "OWNER APPROVAL REQUIRED" in report
    assert "sample:" in report
    # The withheld caption never reaches the report either.
    assert "PRIVATE CAPTION" not in report


def test_unsent_only_segment_plans_no_document(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    ghost = insert_message(
        db, chat_id=corpus.dm_chat_id, sender_person_id=corpus.alice_id,
        is_from_me=False, sent_at=_T + timedelta(hours=2),
        text="FULLY RETRACTED", is_unsent=True,
    )
    ghost_key = "stable-ghost"
    insert_segment(
        db, chat_id=corpus.dm_chat_id, started_at=_T + timedelta(hours=2),
        ended_at=_T + timedelta(hours=2), message_ids=[ghost], stable_key=ghost_key,
    )
    db.commit()

    plan = plan_export(db, config)
    db.commit()
    ghost_doc = Path(plan.staging_dir) / "docs" / f"{segment_document_id(ghost_key)}.txt"
    assert not ghost_doc.exists()


# ---------------------------------------------------------------------------
# Approval + push
# ---------------------------------------------------------------------------


def test_push_refuses_without_approval_and_uploads_nothing(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    plan = plan_export(db, config)
    db.commit()
    transport = FakeTransport()
    with pytest.raises(ExportPushError, match="requires owner approval"):
        push_export(db, config, plan.run_id, transport)
    assert transport.calls == []
    assert transport.imported == {}


def test_approved_push_promotes_exactly_the_plan(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    transport = FakeTransport()
    result = push_export(db, config, plan.run_id, transport)
    db.commit()

    assert result.status == "ok"
    assert result.pushed == 4
    assert set(transport.imported) == {
        item[1] for item in transport.calls if item[0] == "import"
    }
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM export_document WHERE state = 'pushed'")
        assert cur.fetchone() == (4,)
        cur.execute(
            "SELECT status FROM export_run WHERE export_run_id = %s", (plan.run_id,)
        )
        assert cur.fetchone() == ("ok",)


def test_content_change_updates_the_same_document_no_duplicates(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    transport = FakeTransport()
    _happy_path(db, config, transport)

    with db.cursor() as cur:
        cur.execute(
            "UPDATE message SET text_original = %s, updated_at = now() WHERE message_id = %s",
            ("yes, revised numbers attached", corpus.dm_message_edited_id),
        )
    db.commit()

    plan2 = plan_export(db, config)
    db.commit()
    # Only the DM segment doc changed; same external id (D6), no approval
    # needed (no new person/thread/mime/config, no deletes).
    assert plan2.upsert_count == 1
    assert plan2.delete_count == 0
    assert not plan2.approval_required

    result = push_export(db, config, plan2.run_id, transport)
    db.commit()
    assert result.status == "ok"
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM export_document")
        assert cur.fetchone() == (4,)  # updated in place, not duplicated
    assert "revised numbers" in transport.objects[f"segments/{corpus.dm_doc_id}.txt"].decode()


def test_unchanged_replan_is_an_empty_plan(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    transport = FakeTransport()
    _happy_path(db, config, transport)
    plan2 = plan_export(db, config)
    db.commit()
    assert plan2.upsert_count == 0
    assert plan2.delete_count == 0
    assert plan2.unchanged_count == 4
    assert not plan2.approval_required


# ---------------------------------------------------------------------------
# TOCTOU: the world changes between approval and push
# ---------------------------------------------------------------------------


def test_participant_added_after_approval_aborts_push(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    """The hash pin proves the bytes didn't change — it cannot prove
    the thread didn't. A new, unclassified member voids the plan."""
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    carol_id = insert_person(db, display_name="Carol Carpenter", short_name="carol")
    add_participant(db, corpus.dm_chat_id, carol_id)  # NOT allowlisted
    db.commit()

    transport = FakeTransport()
    with pytest.raises(ExportDriftError, match="no longer export-eligible"):
        push_export(db, config, plan.run_id, transport)
    assert transport.calls == []
    with db.cursor() as cur:
        cur.execute("SELECT status FROM export_run WHERE export_run_id = %s", (plan.run_id,))
        assert cur.fetchone() == ("planned",)


def test_allowlist_change_after_approval_aborts_push(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    allow(db, corpus.bob_id, text=False, attachments=False)
    db.commit()

    transport = FakeTransport()
    with pytest.raises(ExportDriftError, match="allowlist_person changed"):
        push_export(db, config, plan.run_id, transport)
    assert transport.calls == []


def test_identity_merge_after_approval_aborts_push(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    """An S3 identity merge can silently change who a message belongs
    to without touching the allowlist table — push must notice."""
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    mallory_id = insert_person(db, display_name="Mallory Mason", short_name="mallory")
    with db.cursor() as cur:
        cur.execute(
            "UPDATE message SET sender_person_id = %s WHERE message_id = %s",
            (mallory_id, corpus.dm_message_edited_id),
        )
    db.commit()

    transport = FakeTransport()
    with pytest.raises(ExportDriftError):
        push_export(db, config, plan.run_id, transport)
    assert transport.calls == []


def test_staged_tampering_after_approval_aborts_push(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    staged = Path(plan.staging_dir) / "docs" / f"{corpus.dm_doc_id}.txt"
    staged.write_text(staged.read_text() + "INJECTED LINE\n", encoding="utf-8")

    transport = FakeTransport()
    with pytest.raises(ExportDriftError, match="staging was modified"):
        push_export(db, config, plan.run_id, transport)
    assert transport.calls == []


def test_config_change_after_plan_aborts_push(
    db: psycopg.Connection, config: Config, corpus: Corpus,
    config_dict_factory: object,
) -> None:
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    changed = load_config_dict(
        config_dict_factory(**{"render.attachment_snippet_chars": 33})  # type: ignore[operator]
    )
    transport = FakeTransport()
    with pytest.raises(ExportDriftError, match="config changed"):
        push_export(db, changed, plan.run_id, transport)
    assert transport.calls == []


def test_segment_rebuilt_after_approval_aborts_push(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    with db.cursor() as cur:
        cur.execute(
            "UPDATE segment SET stable_key = 'stable-dm-REBUILT' WHERE segment_id = %s",
            (corpus.dm_segment_id,),
        )
    db.commit()

    transport = FakeTransport()
    with pytest.raises(ExportDriftError, match=r"re-segmented|stable key|no longer derives"):
        push_export(db, config, plan.run_id, transport)
    assert transport.calls == []


def test_attachment_gate_drift_after_approval_aborts_push(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    """attachments_allowed flips are caught by the allowlist pin; a new
    attachment LINK appearing in a planned segment is caught by the
    eligible-attachment-set comparison."""
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    extra = insert_attachment(db, filename="late-add.jpg", mime_type="image/jpeg")
    link_attachment(db, corpus.dm_message_edited_id, extra)
    db.commit()

    transport = FakeTransport()
    with pytest.raises(ExportDriftError, match="eligible-attachment set"):
        push_export(db, config, plan.run_id, transport)
    assert transport.calls == []


# ---------------------------------------------------------------------------
# Approval-delta rules on subsequent pushes
# ---------------------------------------------------------------------------


def test_new_thread_and_new_person_require_fresh_approval(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    transport = FakeTransport()
    _happy_path(db, config, transport)

    carol_id = insert_person(db, display_name="Carol Carpenter", short_name="carol")
    allow(db, carol_id)
    chat3 = insert_chat(db, source_guid="chat-new")
    add_participant(db, chat3, corpus.owner_id)
    add_participant(db, chat3, carol_id)
    m = insert_message(
        db, chat_id=chat3, sender_person_id=carol_id, is_from_me=False,
        sent_at=_T, text="new business thread",
    )
    insert_segment(db, chat_id=chat3, started_at=_T, ended_at=_T, message_ids=[m])
    db.commit()

    plan2 = plan_export(db, config)
    db.commit()
    assert plan2.approval_required
    assert APPROVAL_NEW_PERSON in plan2.approval_reasons
    assert APPROVAL_NEW_THREAD in plan2.approval_reasons

    with pytest.raises(ExportPushError, match="requires owner approval"):
        push_export(db, config, plan2.run_id, FakeTransport())


def test_config_change_requires_fresh_approval(
    db: psycopg.Connection, config: Config, corpus: Corpus,
    config_dict_factory: object,
) -> None:
    transport = FakeTransport()
    _happy_path(db, config, transport)

    changed = load_config_dict(
        config_dict_factory(**{"render.attachment_snippet_chars": 33})  # type: ignore[operator]
    )
    plan2 = plan_export(db, changed)
    db.commit()
    assert APPROVAL_CONFIG_CHANGED in plan2.approval_reasons


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


def test_purge_person_deletes_and_verifies_absence(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    transport = FakeTransport()
    _happy_path(db, config, transport)
    assert corpus.group_doc_id in transport.imported

    plan = purge_person(db, config, "bob")
    db.commit()
    assert plan.mode == "purge"
    assert plan.delete_count == 1  # the group segment doc
    assert APPROVAL_DELETES in plan.approval_reasons

    approve_run(db, config, plan.run_id)
    db.commit()
    result = push_export(db, config, plan.run_id, transport)
    db.commit()

    assert result.status == "ok"
    assert result.deleted == 1
    assert corpus.group_doc_id not in transport.imported
    assert f"segments/{corpus.group_doc_id}.txt" not in transport.objects
    assert ("verify-absent", corpus.group_doc_id) in transport.calls
    with db.cursor() as cur:
        cur.execute(
            "SELECT state FROM export_document WHERE document_id = %s",
            (corpus.group_doc_id,),
        )
        assert cur.fetchone() == ("purged",)
        # bob's allowlist row is retained for audit, flags off
        cur.execute(
            "SELECT text_allowed, attachments_allowed FROM allowlist_person "
            "WHERE person_id = %s",
            (corpus.bob_id,),
        )
        assert cur.fetchone() == (False, False)
    # the DM thread is untouched
    assert corpus.dm_doc_id in transport.imported


def test_purge_with_lingering_document_stays_failed_not_falsely_purged(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    """If the store still reports the document after deletion, the
    system must say so — 'purged' is only recorded on verified absence."""
    transport = FakeTransport()
    _happy_path(db, config, transport)

    transport.linger_after_delete_ids.add(corpus.group_doc_id)
    plan = purge_person(db, config, "bob")
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()
    result = push_export(db, config, plan.run_id, transport)
    db.commit()

    assert result.status == "failed"
    assert result.failed == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT state FROM export_document WHERE document_id = %s",
            (corpus.group_doc_id,),
        )
        assert cur.fetchone() == ("pushed",)  # honest: NOT purged
        cur.execute(
            "SELECT result_state, error FROM export_run_item "
            "WHERE export_run_id = %s AND document_id = %s",
            (plan.run_id, corpus.group_doc_id),
        )
        state, error = cur.fetchone()  # type: ignore[misc]
        assert state == "failed"
        assert "absence NOT verified" in error


def test_purge_of_unknown_person_refuses(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    from imsg.export.errors import ExportPlanError

    with pytest.raises(ExportPlanError, match="no person"):
        purge_person(db, config, "nobody-by-this-name")


# ---------------------------------------------------------------------------
# Partial failure + retry
# ---------------------------------------------------------------------------


def test_partial_import_failure_is_recorded_and_retryable(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()

    transport = FakeTransport(fail_import_ids={corpus.dm_doc_id})
    result = push_export(db, config, plan.run_id, transport)
    db.commit()
    assert result.status == "failed"
    assert result.failed == 1
    assert result.pushed == 3
    with db.cursor() as cur:
        cur.execute(
            "SELECT state FROM export_document WHERE document_id = %s",
            (corpus.dm_doc_id,),
        )
        assert cur.fetchone() is None  # never recorded as pushed

    transport.fail_import_ids.clear()
    retry = push_export(db, config, plan.run_id, transport)
    db.commit()
    assert retry.status == "ok"
    assert retry.pushed == 1
    assert retry.skipped_already_done == 3
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM export_document WHERE state = 'pushed'")
        assert cur.fetchone() == (4,)


def test_push_of_unknown_or_finished_run_refuses(
    db: psycopg.Connection, config: Config, corpus: Corpus
) -> None:
    with pytest.raises(ExportPushError, match="does not exist"):
        push_export(db, config, 424242, FakeTransport())

    transport = FakeTransport()
    plan = plan_export(db, config)
    db.commit()
    approve_run(db, config, plan.run_id)
    db.commit()
    push_export(db, config, plan.run_id, transport)
    db.commit()
    with pytest.raises(ExportPushError, match="'ok'"):
        push_export(db, config, plan.run_id, transport)
