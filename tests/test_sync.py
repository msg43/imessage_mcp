"""S7 sync orchestration tests (SPEC §8 S7).

Two layers: pure orchestration/ordering/error-wrapping/gating tests
against fake `run_snapshot_fn`/`run_extract_fn`/`run_identity_fn` (no
Postgres, no SQLite, no subprocess — these are the tests that run
everywhere), plus a smaller live-Postgres integration suite exercising
the real S1→S2→S3 chain end to end (skips cleanly with no reachable
scratch Postgres, same pattern as `test_extract.py`/`test_identity.py`).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from chatdb_fixture import ChatDbBuilder, FixtureChat, FixtureHandle, FixtureMessage
from conftest import ConfigDictFactory
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.errors import ExtractionError, IdentityError, ImsgError, SyncError
from imsg.stages.extract import ExtractResult
from imsg.stages.identity import ContactsImportOutcome, IdentityResult, InvariantReport
from imsg.stages.imsg_dump import ImsgDumpMessage, ImsgDumpRun
from imsg.stages.snapshot import SnapshotResult
from imsg.stages.sync import SyncResult, run_sync, run_sync_all_sources

# --------------------------------------------------------------------------
# fakes for the pure orchestration suite
# --------------------------------------------------------------------------


def _fake_snapshot_result(path: Path) -> SnapshotResult:
    return SnapshotResult(path=path, sha256="a" * 64, byte_size=1024, reused_existing=False)


def _ok_invariant() -> InvariantReport:
    return InvariantReport(
        unresolved_message_senders=0,
        unresolved_tapback_senders=0,
        unresolved_chat_participants=0,
        owner_person_count=1,
    )


def _bad_invariant() -> InvariantReport:
    return InvariantReport(
        unresolved_message_senders=3,
        unresolved_tapback_senders=0,
        unresolved_chat_participants=0,
        owner_person_count=1,
    )


def _fake_extract_result(**overrides: object) -> ExtractResult:
    base: dict[str, object] = {
        "run_id": 1, "watermark_before": 0, "watermark_after": 1, "chats_upserted": 1,
        "handles_upserted": 1, "messages_upserted": 1, "tapbacks_upserted": 0,
        "system_messages_skipped": 0, "attachments_upserted": 0, "link_previews_upserted": 0,
        "bodies_missing": 0, "dump_stderr_line_count": 0,
    }
    base.update(overrides)
    return ExtractResult(**base)  # type: ignore[arg-type]


def _fake_identity_result(invariant: InvariantReport) -> IdentityResult:
    return IdentityResult(
        source_handles_processed=1, persons_created=1, handles_created=1, messages_resolved=1,
        tapbacks_resolved=0, chat_participants_resolved=1,
        contacts=ContactsImportOutcome(attempted=False, contacts_loaded=0, degraded=False, degraded_reason=None),
        invariant=invariant,
    )


class _Spy:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def _stub_guard_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every orchestration test below fakes S1/S2/S3, so the mount gate
    (which would otherwise need a real `diskutil`-visible encrypted
    volume) is stubbed too — `run_sync`'s own responsibility to *call*
    the gate is what matters here, not `imsg.mount.guard`'s own
    correctness (that has its own test suite, `test_mount_gate.py`)."""
    import imsg.stages.sync as sync_mod

    monkeypatch.setattr(sync_mod, "guard_mount", lambda data_root: None)


def _minimal_config(config_dict_factory: ConfigDictFactory, **overrides: object) -> Config:
    return load_config_dict(config_dict_factory(**overrides), source="<test>")


def test_run_sync_happy_path_skips_s4_s6_when_not_wired(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = _minimal_config(config_dict_factory)

    snapshot_calls = _Spy()

    def fake_snapshot(**kwargs: object) -> SnapshotResult:
        snapshot_calls(**kwargs)
        return _fake_snapshot_result(tmp_path / "snapshot.db")

    extract_calls = _Spy()

    def fake_extract(**kwargs: object) -> ExtractResult:
        extract_calls(**kwargs)
        return _fake_extract_result()

    identity_calls = _Spy()

    def fake_identity(**kwargs: object) -> IdentityResult:
        identity_calls(**kwargs)
        return _fake_identity_result(_ok_invariant())

    result = run_sync(
        conn=object(),  # type: ignore[arg-type]
        config=config,
        source_name="mini",
        imsg_dump_binary=tmp_path / "imsg-dump",
        run_snapshot_fn=fake_snapshot,
        run_extract_fn=fake_extract,
        run_identity_fn=fake_identity,
    )

    assert isinstance(result, SyncResult)
    assert result.source_name == "mini"
    assert result.snapshot is not None
    assert result.segment_ran is False
    assert result.embed_ran is False
    assert len(snapshot_calls.calls) == 1
    assert len(extract_calls.calls) == 1
    assert len(identity_calls.calls) == 1


def test_run_sync_calls_stages_in_order(config_dict_factory: ConfigDictFactory, tmp_path: Path) -> None:
    config = _minimal_config(config_dict_factory)
    order: list[str] = []

    def fake_snapshot(**kwargs: object) -> SnapshotResult:
        order.append("snapshot")
        return _fake_snapshot_result(tmp_path / "snapshot.db")

    def fake_extract(**kwargs: object) -> ExtractResult:
        order.append("extract")
        return _fake_extract_result()

    def fake_identity(**kwargs: object) -> IdentityResult:
        order.append("identity")
        return _fake_identity_result(_ok_invariant())

    def fake_segment(conn: object, cfg: object) -> None:
        order.append("segment")

    def fake_embed(conn: object, cfg: object) -> None:
        order.append("embed")

    run_sync(
        conn=object(),  # type: ignore[arg-type]
        config=config,
        source_name="mini",
        imsg_dump_binary=tmp_path / "imsg-dump",
        run_snapshot_fn=fake_snapshot,
        run_extract_fn=fake_extract,
        run_identity_fn=fake_identity,
        segment_fn=fake_segment,
        embed_fn=fake_embed,
    )

    assert order == ["snapshot", "extract", "identity", "segment", "embed"]


def test_run_sync_reports_segment_and_embed_ran_when_wired(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = _minimal_config(config_dict_factory)

    result = run_sync(
        conn=object(),  # type: ignore[arg-type]
        config=config,
        source_name="mini",
        imsg_dump_binary=tmp_path / "imsg-dump",
        run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
        run_extract_fn=lambda **kw: _fake_extract_result(),
        run_identity_fn=lambda **kw: _fake_identity_result(_ok_invariant()),
        segment_fn=lambda conn, cfg: "segmented",
        embed_fn=lambda conn, cfg: "embedded",
    )
    assert result.segment_ran is True
    assert result.embed_ran is True


def test_run_sync_stops_before_s4_when_invariant_fails(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = _minimal_config(config_dict_factory)
    segment_spy = _Spy()

    with pytest.raises(SyncError, match="stopped before S4"):
        run_sync(
            conn=object(),  # type: ignore[arg-type]
            config=config,
            source_name="mini",
            imsg_dump_binary=tmp_path / "imsg-dump",
            run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
            run_extract_fn=lambda **kw: _fake_extract_result(),
            run_identity_fn=lambda **kw: _fake_identity_result(_bad_invariant()),
            segment_fn=segment_spy,
        )
    assert segment_spy.calls == []  # S4 must never have been reached


def test_run_sync_wraps_s1_failure(config_dict_factory: ConfigDictFactory, tmp_path: Path) -> None:
    config = _minimal_config(config_dict_factory)

    def boom(**kwargs: object) -> SnapshotResult:
        raise ImsgError("simulated S1 failure")

    with pytest.raises(SyncError, match="S1 snapshot"):
        run_sync(
            conn=object(),  # type: ignore[arg-type]
            config=config,
            source_name="mini",
            imsg_dump_binary=tmp_path / "imsg-dump",
            run_snapshot_fn=boom,
        )


def test_run_sync_wraps_s2_failure(config_dict_factory: ConfigDictFactory, tmp_path: Path) -> None:
    config = _minimal_config(config_dict_factory)

    def boom(**kwargs: object) -> ExtractResult:
        raise ExtractionError("simulated S2 failure")

    with pytest.raises(SyncError, match="S2 extract"):
        run_sync(
            conn=object(),  # type: ignore[arg-type]
            config=config,
            source_name="mini",
            imsg_dump_binary=tmp_path / "imsg-dump",
            run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
            run_extract_fn=boom,
        )


def test_run_sync_wraps_s3_failure(config_dict_factory: ConfigDictFactory, tmp_path: Path) -> None:
    config = _minimal_config(config_dict_factory)

    def boom(**kwargs: object) -> IdentityResult:
        raise IdentityError("simulated S3 failure")

    with pytest.raises(SyncError, match="S3 identity"):
        run_sync(
            conn=object(),  # type: ignore[arg-type]
            config=config,
            source_name="mini",
            imsg_dump_binary=tmp_path / "imsg-dump",
            run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
            run_extract_fn=lambda **kw: _fake_extract_result(),
            run_identity_fn=boom,
        )


def test_run_sync_wraps_s4_failure(config_dict_factory: ConfigDictFactory, tmp_path: Path) -> None:
    config = _minimal_config(config_dict_factory)

    def boom(conn: object, cfg: object) -> None:
        raise ImsgError("simulated S4 failure")

    with pytest.raises(SyncError, match="S4 segment"):
        run_sync(
            conn=object(),  # type: ignore[arg-type]
            config=config,
            source_name="mini",
            imsg_dump_binary=tmp_path / "imsg-dump",
            run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
            run_extract_fn=lambda **kw: _fake_extract_result(),
            run_identity_fn=lambda **kw: _fake_identity_result(_ok_invariant()),
            segment_fn=boom,
        )


def test_run_sync_snapshot_override_skips_s1(config_dict_factory: ConfigDictFactory, tmp_path: Path) -> None:
    """The studio-seed one-shot path (SPEC §8 S7)."""
    config = _minimal_config(config_dict_factory)
    snapshot_spy = _Spy()
    already_transferred = tmp_path / "studio-snapshot.db"
    already_transferred.write_text("")

    extract_kwargs: dict[str, object] = {}

    def fake_extract(**kwargs: object) -> ExtractResult:
        extract_kwargs.update(kwargs)
        return _fake_extract_result()

    result = run_sync(
        conn=object(),  # type: ignore[arg-type]
        config=config,
        source_name="studio-seed",
        imsg_dump_binary=tmp_path / "imsg-dump",
        snapshot_override=already_transferred,
        run_snapshot_fn=snapshot_spy,  # type: ignore[arg-type]  # must never be called
        run_extract_fn=fake_extract,
        run_identity_fn=lambda **kw: _fake_identity_result(_ok_invariant()),
    )

    assert snapshot_spy.calls == []
    assert result.snapshot is None
    assert extract_kwargs["snapshot_path"] == already_transferred
    assert extract_kwargs["snapshot_sha256"] is None


def test_run_sync_calls_mount_guard(config_dict_factory: ConfigDictFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import imsg.stages.sync as sync_mod

    config = _minimal_config(config_dict_factory)
    guard_calls: list[Path] = []
    monkeypatch.setattr(sync_mod, "guard_mount", lambda data_root: guard_calls.append(data_root))

    run_sync(
        conn=object(),  # type: ignore[arg-type]
        config=config,
        source_name="mini",
        imsg_dump_binary=tmp_path / "imsg-dump",
        run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
        run_extract_fn=lambda **kw: _fake_extract_result(),
        run_identity_fn=lambda **kw: _fake_identity_result(_ok_invariant()),
    )
    assert guard_calls == [config.paths.data_root]


def test_run_sync_raises_when_source_not_configured(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = _minimal_config(config_dict_factory)
    with pytest.raises(SyncError, match=r"not found in config\.sync\.sources"):
        run_sync(
            conn=object(),  # type: ignore[arg-type]
            config=config,
            source_name="nonexistent-source",
            imsg_dump_binary=tmp_path / "imsg-dump",
            run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
        )


def test_run_sync_all_sources_syncs_each_configured_source(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = _minimal_config(
        config_dict_factory,
        **{"sync.sources": [{"name": "mini", "chat_db": "/tmp/a"}, {"name": "studio", "chat_db": "/tmp/b"}]},
    )
    seen_sources: list[str] = []

    def fake_extract(**kwargs: object) -> ExtractResult:
        seen_sources.append(str(kwargs["source_name"]))
        return _fake_extract_result()

    results = run_sync_all_sources(
        conn=object(),  # type: ignore[arg-type]
        config=config,
        imsg_dump_binary=tmp_path / "imsg-dump",
        run_snapshot_fn=lambda **kw: _fake_snapshot_result(tmp_path / "snapshot.db"),
        run_extract_fn=fake_extract,
        run_identity_fn=lambda **kw: _fake_identity_result(_ok_invariant()),
    )
    assert seen_sources == ["mini", "studio"]
    assert [r.source_name for r in results] == ["mini", "studio"]


def test_run_sync_all_sources_stops_at_first_failure(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = _minimal_config(
        config_dict_factory,
        **{"sync.sources": [{"name": "mini", "chat_db": "/tmp/a"}, {"name": "studio", "chat_db": "/tmp/b"}]},
    )

    def flaky_snapshot(**kwargs: object) -> SnapshotResult:
        if kwargs.get("live_chat_db") == Path("/tmp/a"):
            raise ImsgError("mini is broken")
        return _fake_snapshot_result(tmp_path / "snapshot.db")

    with pytest.raises(SyncError):
        run_sync_all_sources(
            conn=object(),  # type: ignore[arg-type]
            config=config,
            imsg_dump_binary=tmp_path / "imsg-dump",
            run_snapshot_fn=flaky_snapshot,
            run_extract_fn=lambda **kw: _fake_extract_result(),
            run_identity_fn=lambda **kw: _fake_identity_result(_ok_invariant()),
        )


# --------------------------------------------------------------------------
# live-Postgres integration: the real S1->S2->S3 chain end to end
# --------------------------------------------------------------------------

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_sync_test"

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

pg_skipif = pytest.mark.skipif(
    not REACHABLE,
    reason=(
        "no reachable scratch Postgres instance "
        f"(tried {TEST_PG_HOST}:{TEST_PG_PORT}) — set IMSG_TEST_PG_HOST/"
        "IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
    ),
)


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
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
    runner = PostgresMigrationRunner(conn, REAL_MIGRATIONS_DIR)
    runner.apply_pending()
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


@pg_skipif
def test_run_sync_real_end_to_end_snapshot_extract_identity(
    pg_conn: psycopg.Connection, tmp_path: Path, config_dict_factory: ConfigDictFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the *real* S1 (`run_snapshot`) + S2 (`run_extract`) +
    S3 (`run_identity`) defaults together, with only the `imsg-dump`
    subprocess faked (no compiled Rust binary needed for this test) —
    the genuine end-to-end wiring this build's spec asks S7 to own."""
    import imsg.stages.sync as sync_mod

    monkeypatch.setattr(sync_mod, "guard_mount", lambda data_root: None)

    builder = ChatDbBuilder()
    chat = builder.add_chat(FixtureChat(guid="chat-1"))
    handle = builder.add_handle(FixtureHandle(raw_value="+14155552671"))
    builder.link_participant(chat.guid, handle.raw_value)
    builder.add_message(
        FixtureMessage(guid="msg-1", chat_guid=chat.guid, handle_raw_value=handle.raw_value)
    )
    live_chat_db = builder.build(tmp_path / "chat.db")

    # `config_dict_factory`'s own `data_root` fixture already created
    # `tmp_path / "data_root"` and pointed `paths.data_root` at it — no
    # need to build a second one here.
    config = load_config_dict(
        config_dict_factory(**{
            "sync.sources": [{"name": "mini", "chat_db": str(live_chat_db)}],
        }),
        source="<test>",
    )

    def fake_run_imsg_dump(binary_path: Path, snapshot_path: Path, since_rowid: int) -> ImsgDumpRun:
        return ImsgDumpRun(
            messages=(
                ImsgDumpMessage(
                    rowid=1, guid="msg-1", chat_guid=None, handle=None, is_from_me=False, date=None,
                    date_edited=None, date_retracted=None, service="iMessage", body_text="hello there",
                    edit_history=(), is_unsent=False, tapback=None, attachment_rowids=(), reply_to_guid=None,
                ),
            ),
            stderr_lines=(),
        )

    binary = tmp_path / "imsg-dump"
    binary.write_text("")

    result = run_sync(
        conn=pg_conn,
        config=config,
        source_name="mini",
        imsg_dump_binary=binary,
        run_imsg_dump_fn=fake_run_imsg_dump,
    )

    assert result.snapshot is not None
    assert result.snapshot.path.is_file()
    assert result.extract.messages_upserted == 1
    assert result.identity.invariant.ok is True
    assert result.segment_ran is False
    assert result.embed_ran is False

    with pg_conn.cursor() as cur:
        cur.execute("SELECT text_original FROM message WHERE source_guid = 'msg-1'")
        assert cur.fetchone() == ("hello there",)
