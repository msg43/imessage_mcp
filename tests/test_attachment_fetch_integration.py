"""The attachment fetcher (owner decision D13), end to end through
`imsg backfill-attachments` against a real Postgres.

Before this, S5a tried one path per attachment — the one this host's
chat.db recorded — on this host only. An attachment with no local path
stayed `missing`, one whose path pointed into a temporary directory stayed
`unsupported` for good, and a copy on another Mac, a drive or a NAS share
was never looked at. These tests put a candidate copy in
`attachment_location` and check that the backfill finds it, in the right
order, verifies it against what its location reported, and never takes a
copy matched on its name alone.

The copies other hosts supply are placed where they arrive: the staging
directory (`imsg push-attachments` writes there), or, for the SSH pull, a
directory reached through a stand-in `ssh` that runs rsync's remote half
locally, so the real rsync runs.

Fictional content only (D5).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner, Result

import imsg.cli as cli_module
from _attachment_fetch_fixtures import (
    RSYNC,
    attachment_row,
    dsn,
    insert_attachment,
    insert_location,
    location_row,
    requires_postgres,
    requires_rsync,
    scratch_database,
    sha256_bytes,
    tree_state,
    write_fake_ssh,
    write_file,
)
from imsg.cli import app

pytestmark = requires_postgres

DB_NAME = "imsg_index_attachment_fetch_test"
PREFIX = "~/Library/Messages/Attachments/"
GUID_A = "5A0C3B21-0000-4000-8000-0000000000A1"
GUID_B = "5A0C3B21-0000-4000-8000-0000000000B2"
GUID_C = "5A0C3B21-0000-4000-8000-0000000000C3"

runner = CliRunner()


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
    yield from scratch_database(DB_NAME)


@dataclass
class Env:
    tmp: Path
    config: Path
    data_root: Path
    live_chat_db: Path
    attachments_root: Path
    """This host's Messages attachments folder."""
    staging: Path
    """Where another host's pushed copies land."""

    def write_config(self, attachments_yaml: str = "") -> None:
        self.config.write_text(
            f"""
paths:
  data_root: {self.data_root}
  live_chat_db: {self.live_chat_db}
database:
  dsn: postgresql://imsg@127.0.0.1:5433/imsgindex
  password: env:IMSG_TEST_PG_PASSWORD
sync:
  interval_seconds: 900
  sources:
    - name: mini
      chat_db: {self.live_chat_db}
embedding:
  revision: deadbeef
  query_instruction: "test instruction"
  multimodal:
    revision: cafef00d
retrieval:
  reranker_revision: f00dcafe
models:
  backend: fake
mcp:
  public:
    scope: allowlist
export:
  gcp_project: example-project
  gcs_bucket: example-bucket
  data_store_id: example-datastore
{attachments_yaml}
"""
        )


@pytest.fixture
def env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pg_conn: psycopg.Connection
) -> Env:
    home = tmp_path / "home"
    messages = home / "Library" / "Messages"
    (messages / "Attachments").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")
    live = messages / "chat.db"
    live.write_text("")
    environment = Env(
        tmp=tmp_path,
        config=tmp_path / "config.yaml",
        data_root=data_root,
        live_chat_db=live,
        attachments_root=messages / "Attachments",
        staging=data_root / "attachment-staging",
    )
    environment.write_config()
    monkeypatch.setattr(cli_module, "run_guard_mount_or_exit", lambda data_root: None)
    monkeypatch.setattr(
        cli_module, "connect", lambda database, **kw: psycopg.connect(dsn(DB_NAME), autocommit=True)
    )
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    return environment


def _backfill(env: Env, *extra: str) -> Result:
    result = runner.invoke(
        app,
        ["backfill-attachments", "--yes-full-run", "--rate", "600000", "--config",
         str(env.config), *extra],
    )
    assert result.exit_code == 0, result.output
    return result


def _cache_files(data_root: Path) -> set[str]:
    cache = data_root / "attachments"
    if not cache.is_dir():
        return set()
    return {p.name for p in cache.rglob("*") if p.is_file() and ".tmp" not in p.parts}


# --- the required behaviours -------------------------------------------------


def test_an_unsupported_attachment_is_fetched_once_a_copy_appears(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    """`unsupported` was terminal: its recorded path points into a temporary
    directory and is refused without reading, forever. A copy found
    elsewhere is a new reason to try."""
    data = b"fictional photo bytes: a deck at dusk"
    aid = insert_attachment(
        pg_conn,
        guid=GUID_A,
        source_path="/private/var/folders/zz/abc/T/IMG_0001.jpeg",
        state="unsupported",
        last_error="unsupported[temp-directory-path]: refused",
        byte_size=len(data),
    )
    rel = f"7f/15/{GUID_A}/IMG_0001.jpeg"
    lid = insert_location(
        pg_conn, attachment_id=aid, location="studio", path=PREFIX + rel,
        byte_size=len(data), sha256=sha256_bytes(data), reported_by="listing:studio.tsv",
    )
    write_file(env.staging / "studio", rel, data)

    _backfill(env)

    row = attachment_row(pg_conn, aid)
    assert row["state"] == "materialized"
    assert row["sha256"] == sha256_bytes(data)
    assert Path(row["cache_path"]).read_bytes() == data
    location = location_row(pg_conn, lid)
    assert location["outcome"] == "fetched"
    assert location["fetched_at"] is not None


def test_a_pushed_copy_is_verified_materialized_and_released(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    """Staging to cache: a `missing` attachment (no path on this host) whose
    copy another host pushed. The staged copy is deleted once its content is
    in the cache; the index's own staging directory is the only thing the
    fetcher ever deletes from."""
    data = b"fictional voice note bytes"
    aid = insert_attachment(pg_conn, guid=GUID_B, filename="Audio Message.caf",
                            byte_size=len(data), mime_type="audio/x-caf")
    rel = f"3c/12/{GUID_B}/Audio Message.caf"
    insert_location(pg_conn, attachment_id=aid, location="studio", path=PREFIX + rel,
                    byte_size=len(data), sha256=sha256_bytes(data))
    staged = write_file(env.staging / "studio", rel, data)

    result = _backfill(env)

    row = attachment_row(pg_conn, aid)
    assert row["state"] == "materialized"
    cache_path = env.data_root / "attachments" / sha256_bytes(data)[:2] / sha256_bytes(data)
    assert Path(row["cache_path"]) == cache_path
    assert cache_path.read_bytes() == data
    assert not staged.exists()
    assert (env.staging / "studio").is_dir()
    assert "staged/studio/recorded_path=1" in result.output


def test_a_copy_whose_size_or_hash_differs_is_rejected_and_not_retried(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    right = b"the bytes the listing hashed"
    wrong_size = b"short"
    wrong_hash = b"the bytes another file holds"  # same length as `right`
    assert len(wrong_hash) == len(right)

    by_size = insert_attachment(pg_conn, guid=GUID_A, byte_size=len(right))
    by_hash = insert_attachment(pg_conn, guid=GUID_B, byte_size=len(right))
    rel_size = f"11/22/{GUID_A}/IMG_0001.jpeg"
    rel_hash = f"33/44/{GUID_B}/IMG_0001.jpeg"
    lid_size = insert_location(pg_conn, attachment_id=by_size, location="studio",
                               path=PREFIX + rel_size, byte_size=len(right),
                               sha256=sha256_bytes(right))
    lid_hash = insert_location(pg_conn, attachment_id=by_hash, location="studio",
                               path=PREFIX + rel_hash, byte_size=len(right),
                               sha256=sha256_bytes(right))
    write_file(env.staging / "studio", rel_size, wrong_size)
    write_file(env.staging / "studio", rel_hash, wrong_hash)

    result = _backfill(env)

    for aid, lid, what in ((by_size, lid_size, "size"), (by_hash, lid_hash, "sha256")):
        assert attachment_row(pg_conn, aid)["state"] == "missing"
        location = location_row(pg_conn, lid)
        assert location["outcome"] == "rejected"
        assert f"{what} " in location["error"]
        assert location["fetched_at"] is None
    assert "studio/size=1" in result.output and "studio/sha256=1" in result.output
    # Nothing that failed verification reached the cache.
    assert sha256_bytes(wrong_hash) not in _cache_files(env.data_root)
    assert sha256_bytes(wrong_size) not in _cache_files(env.data_root)

    # A rejected copy stays closed: the same wrong bytes staged again are
    # not re-read on the next run.
    tried = location_row(pg_conn, lid_hash)["tried_at"]
    write_file(env.staging / "studio", rel_hash, wrong_hash)
    _backfill(env)
    assert location_row(pg_conn, lid_hash)["tried_at"] == tried
    assert attachment_row(pg_conn, by_hash)["state"] == "missing"


def test_a_name_only_match_is_never_stored_or_fetched(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    """D13: never on a name alone. The listing's file has the attachment's
    name but not its size, in a folder that is not its GUID's."""
    aid = insert_attachment(pg_conn, guid=GUID_C, filename="IMG_0009.jpeg", byte_size=1234)
    rel = "ee/ff/0F0F0F0F-0000-4000-8000-000000000000/IMG_0009.jpeg"
    listing = env.tmp / "studio-listing.tsv"
    listing.write_text(f"{rel}\t999\t1700000000\t{'ab' * 32}\n")
    staged = write_file(env.staging / "studio", rel, b"x" * 999)

    located = runner.invoke(
        app,
        ["locate-attachments", "--listing", f"studio={listing}", "--no-local-walk",
         "--config", str(env.config)],
    )
    assert located.exit_code == 0, located.output
    assert "never stored, never fetched: studio=1" in located.output
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM attachment_location")
        count = cur.fetchone()
    assert count is not None and count[0] == 0

    _backfill(env)
    assert attachment_row(pg_conn, aid)["state"] == "missing"
    assert staged.read_bytes() == b"x" * 999

    # And the schema itself has no way to store one.
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        insert_location(pg_conn, attachment_id=aid, location="studio", path=PREFIX + rel,
                        match="name_only")


def test_copies_are_tried_best_first(pg_conn: psycopg.Connection, env: Env) -> None:
    """Identity matches (a recorded path, a GUID folder) come before a
    name-and-size match, and among equals this host's own copy comes
    before one another host pushed."""
    local_bytes = b"this host's own copy"
    pushed_bytes = b"the other Mac's copy"

    # A: both copies are identity matches; this host's is tried first.
    a = insert_attachment(pg_conn, guid=GUID_A)
    rel_local = f"0a/10/{GUID_A}/IMG_0001.jpeg"
    rel_pushed = f"7f/15/{GUID_A}/IMG_0001.jpeg"
    a_local = insert_location(pg_conn, attachment_id=a, location="mini", path=PREFIX + rel_local)
    a_pushed = insert_location(pg_conn, attachment_id=a, location="studio",
                               path=PREFIX + rel_pushed, byte_size=len(pushed_bytes),
                               sha256=sha256_bytes(pushed_bytes))
    write_file(env.attachments_root, rel_local, local_bytes)
    write_file(env.staging / "studio", rel_pushed, pushed_bytes)

    # B: this host's copy matches on name and size only; the pushed copy
    # sits at the path a chat.db recorded, and wins although it is farther.
    b_local_bytes = b"same name and size, unknown content!"
    b_pushed_bytes = b"the recorded copy, content verified"
    b = insert_attachment(pg_conn, guid=GUID_B, byte_size=len(b_local_bytes))
    rel_b_local = "aa/bb/77777777-0000-4000-8000-000000000000/IMG_0001.jpeg"
    rel_b_pushed = f"cc/dd/{GUID_B}/IMG_0001.jpeg"
    b_local = insert_location(pg_conn, attachment_id=b, location="mini", path=PREFIX + rel_b_local,
                              match="name_size", byte_size=len(b_local_bytes))
    b_pushed = insert_location(pg_conn, attachment_id=b, location="studio",
                               path=PREFIX + rel_b_pushed, byte_size=len(b_pushed_bytes),
                               sha256=sha256_bytes(b_pushed_bytes))
    write_file(env.attachments_root, rel_b_local, b_local_bytes)
    write_file(env.staging / "studio", rel_b_pushed, b_pushed_bytes)

    result = _backfill(env)

    assert attachment_row(pg_conn, a)["sha256"] == sha256_bytes(local_bytes)
    assert location_row(pg_conn, a_local)["outcome"] == "fetched"
    assert location_row(pg_conn, a_pushed)["tried_at"] is None
    assert attachment_row(pg_conn, b)["sha256"] == sha256_bytes(b_pushed_bytes)
    assert location_row(pg_conn, b_pushed)["outcome"] == "fetched"
    assert location_row(pg_conn, b_local)["tried_at"] is None
    assert "(flagged name+size=0)" in result.output


def test_a_flagged_copy_waits_while_a_certain_one_is_on_its_way(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    """A name-and-size copy here, and a recorded-path copy on another Mac
    that has not pushed yet: the probable copy is held back for the certain
    one rather than taken because it happens to be closer."""
    env.write_config("attachments:\n  push_locations: [studio]\n")
    flagged = b"a same-size lookalike"
    certain = b"the real one, verified"
    aid = insert_attachment(pg_conn, guid=GUID_C, byte_size=len(flagged))
    rel_flagged = "aa/bb/88888888-0000-4000-8000-000000000000/IMG_0001.jpeg"
    rel_certain = f"cc/dd/{GUID_C}/IMG_0001.jpeg"
    insert_location(pg_conn, attachment_id=aid, location="mini", path=PREFIX + rel_flagged,
                    match="name_size", byte_size=len(flagged))
    insert_location(pg_conn, attachment_id=aid, location="studio", path=PREFIX + rel_certain,
                    byte_size=len(certain), sha256=sha256_bytes(certain))
    write_file(env.attachments_root, rel_flagged, flagged)

    first = _backfill(env)
    assert attachment_row(pg_conn, aid)["state"] == "missing"
    assert "still waiting on a push: studio=1" in first.output
    assert "held back for a pending push=1" in first.output

    write_file(env.staging / "studio", rel_certain, certain)  # the push arrives
    _backfill(env)
    assert attachment_row(pg_conn, aid)["sha256"] == sha256_bytes(certain)


def test_content_already_in_the_cache_needs_no_copy(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    """A listing hashed a copy on a drive nobody has attached, and another
    attachment row already materialized those bytes: the cache has them."""
    data = b"a photo sent twice, into two chats"
    digest = sha256_bytes(data)
    cached = write_file(env.data_root / "attachments" / digest[:2], digest, data)
    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key, state, sha256, byte_size, "
            "cache_path) VALUES ('earlier-copy', 'key-earlier-copy', 'materialized', %s, %s, %s)",
            (digest, len(data), str(cached)),
        )
    aid = insert_attachment(pg_conn, guid=GUID_A, byte_size=len(data))
    lid = insert_location(pg_conn, attachment_id=aid, location="D-TEST1",
                          path=f"Backups/{GUID_A}/IMG_0001.jpeg", match="guid_folder",
                          byte_size=len(data), sha256=digest)

    result = _backfill(env)

    row = attachment_row(pg_conn, aid)
    assert row["state"] == "materialized" and row["sha256"] == digest
    assert Path(row["cache_path"]) == cached
    assert location_row(pg_conn, lid)["outcome"] == "fetched"
    assert "cache/D-TEST1/guid_folder=1" in result.output


@requires_rsync
def test_a_nas_copy_is_pulled_read_only_verified_and_materialized(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    remote_root = env.tmp / "nas" / "homes"
    data = b"a scan kept on the NAS"
    rel = "alice/Pictures/IMG_0003.jpeg"
    write_file(remote_root, rel, data)
    fake_ssh = write_fake_ssh(env.tmp)
    env.write_config(
        "attachments:\n"
        "  pull:\n"
        f"    - location: N-TEST1\n      ssh_host: nashost\n      root: {remote_root}\n"
        f"  ssh_command: {fake_ssh}\n"
        f"  rsync_command: {RSYNC}\n"
    )
    aid = insert_attachment(pg_conn, guid=GUID_B, filename="IMG_0003.jpeg", byte_size=len(data))
    lid = insert_location(pg_conn, attachment_id=aid, location="N-TEST1", path=rel,
                          match="name_size", byte_size=len(data))
    before = tree_state(remote_root)

    result = _backfill(env)

    assert tree_state(remote_root) == before  # the source was only read
    row = attachment_row(pg_conn, aid)
    assert row["state"] == "materialized" and row["sha256"] == sha256_bytes(data)
    location = location_row(pg_conn, lid)
    assert location["outcome"] == "fetched" and location["match"] == "name_size"
    assert "pull/N-TEST1/name_size=1" in result.output
    assert "(flagged name+size=1)" in result.output
    assert not (env.staging / "N-TEST1" / rel).exists()

    # `--no-pull` skips the SSH copy entirely.
    other = insert_attachment(pg_conn, filename="IMG_0004.jpeg", byte_size=4)
    other_lid = insert_location(pg_conn, attachment_id=other, location="N-TEST1",
                                path="alice/Pictures/IMG_0004.jpeg", match="name_size",
                                byte_size=4)
    skipped = _backfill(env, "--no-pull")
    assert location_row(pg_conn, other_lid)["tried_at"] is None
    assert "pulls not run (--no-pull): N-TEST1=1" in skipped.output


def test_a_dry_run_reads_nothing_and_writes_nothing(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    data = b"fictional bytes"
    aid = insert_attachment(pg_conn, guid=GUID_A, byte_size=len(data))
    rel = f"7f/15/{GUID_A}/IMG_0001.jpeg"
    lid = insert_location(pg_conn, attachment_id=aid, location="studio", path=PREFIX + rel,
                          byte_size=len(data), sha256=sha256_bytes(data))
    staged = write_file(env.staging / "studio", rel, data)

    result = _backfill(env, "--dry-run")

    assert "staged/studio/recorded_path=1" in result.output
    assert attachment_row(pg_conn, aid)["state"] == "missing"
    assert location_row(pg_conn, lid)["tried_at"] is None
    assert staged.read_bytes() == data
    assert _cache_files(env.data_root) == set()


def test_a_path_the_first_pass_just_failed_to_read_is_not_read_again_that_run(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    """The attachment's own path has its own retry ladder (`error`, then
    `missing`). The same path recorded as a location is not read a second
    time in the run that just failed on it; another copy still is."""
    rel = f"0a/10/{GUID_A}/IMG_0001.jpeg"
    unreadable = write_file(env.attachments_root, rel, b"locked")
    unreadable.chmod(0o000)
    pushed = b"the pushed copy"
    try:
        aid = insert_attachment(pg_conn, guid=GUID_A, source_path=PREFIX + rel, state="dataless",
                                byte_size=6)
        same_path = insert_location(pg_conn, attachment_id=aid, location="mini",
                                    path=PREFIX + rel)
        rel_pushed = f"7f/15/{GUID_A}/IMG_0001.jpeg"
        other = insert_location(pg_conn, attachment_id=aid, location="studio",
                                path=PREFIX + rel_pushed, byte_size=len(pushed),
                                sha256=sha256_bytes(pushed))
        write_file(env.staging / "studio", rel_pushed, pushed)

        result = _backfill(env)
    finally:
        unreadable.chmod(0o600)

    assert "errored=1" in result.output  # the first pass's own attempt
    assert location_row(pg_conn, same_path)["tried_at"] is None
    assert location_row(pg_conn, other)["outcome"] == "fetched"
    assert attachment_row(pg_conn, aid)["sha256"] == sha256_bytes(pushed)


def test_the_first_run_trial_gate_also_caps_the_location_phase(
    pg_conn: psycopg.Connection, env: Env
) -> None:
    """SPEC §8 S5a's trial gate ("test on a dozen before ten years") holds
    for copies fetched from elsewhere too: on a first run, without
    --yes-full-run, only the allowance is attempted."""
    from imsg.backfill.fetch import LocationAccess, LocationFetchSettings
    from imsg.backfill.pipeline import run_backfill
    from imsg.backfill.throttle import RateThrottle

    for i in range(3):
        data = f"staged copy {i}".encode()
        guid = f"7E57AB1E-0000-4000-8000-00000000000{i}"
        aid = insert_attachment(pg_conn, guid=guid, byte_size=len(data))
        rel = f"aa/0{i}/{guid}/IMG_000{i}.jpeg"
        insert_location(pg_conn, attachment_id=aid, location="studio", path=PREFIX + rel,
                        byte_size=len(data), sha256=sha256_bytes(data))
        write_file(env.staging / "studio", rel, data)
    settings = LocationFetchSettings(access=LocationAccess(
        local_location="mini", attachments_root=env.attachments_root, staging_root=env.staging,
        push_locations=("studio",),
    ))

    report = run_backfill(
        pg_conn, env.data_root, env.attachments_root, trial_limit=2, locations=settings,
        throttle=RateThrottle(600000, sleep_fn=lambda _s: None),
        disk_free_fn=lambda _p: 10**15,
    )

    assert report.locations is not None
    assert report.locations.materialized_total == 2
    assert report.trial_gate_capped
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM attachment WHERE state = 'materialized'")
        count = cur.fetchone()
    assert count is not None and count[0] == 2
