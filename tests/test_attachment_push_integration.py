"""`imsg push-attachments` (D13): copying attachment files the index is
missing from a host the index host cannot reach, read-only, into the index
host's staging directory — and the guarantee under all of it, that no step
of a fetch writes to a source.

The real rsync runs, with a stand-in `ssh` that executes the remote half
locally; the pushing host's folders and drives are directories under the
test's temporary directory, snapshotted before and after to prove nothing
in them changed.

Fictional content only (D5).
"""

from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner

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
from imsg.backfill.fetch import LocationAccess, LocationFetchSettings, fetch_from_locations
from imsg.backfill.locations import LocationOutcome
from imsg.backfill.push import (
    PushResult,
    PushRoot,
    build_push_plan,
    parse_plan,
    parse_results,
    plan_to_lines,
    record_push_results,
    results_to_lines,
    run_push,
)
from imsg.backfill.transfer import run_copy
from imsg.cli import app

pytestmark = requires_postgres

DB_NAME = "imsg_index_attachment_push_test"
PREFIX = "~/Library/Messages/Attachments/"
GUIDS = [f"D00DFEED-0000-4000-8000-00000000000{i}" for i in range(6)]


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
    yield from scratch_database(DB_NAME)


@pytest.fixture
def index_host(tmp_path: Path) -> LocationAccess:
    """The index host's side: its own (empty) Messages folder and staging."""
    root = tmp_path / "index" / "home" / "Library" / "Messages" / "Attachments"
    root.mkdir(parents=True)
    return LocationAccess(
        local_location="mini",
        attachments_root=root,
        staging_root=tmp_path / "index" / "data_root" / "attachment-staging",
        push_locations=("studio", "V-TEST2"),
    )


def _data_root(access: LocationAccess) -> Path:
    return access.staging_root.parent


class PushingHost:
    """The other Mac: its Messages folder and one attached drive."""

    def __init__(self, tmp_path: Path) -> None:
        self.messages = tmp_path / "studio" / "home" / "Library" / "Messages" / "Attachments"
        self.drive = tmp_path / "studio" / "Volumes" / "Drive 1"
        self.messages.mkdir(parents=True)
        self.drive.mkdir(parents=True)
        self.roots = [PushRoot("studio", self.messages), PushRoot("V-TEST2", self.drive)]

    def state(self) -> tuple[dict[str, tuple[int, int, int, str]], ...]:
        return tree_state(self.messages), tree_state(self.drive)


def _world(conn: psycopg.Connection, host: PushingHost) -> dict[str, tuple[int, bytes]]:
    """Six attachments, each missing a copy here, each testing one path
    through the push."""
    world: dict[str, tuple[int, bytes]] = {}

    def attachment(key: str, guid: str, data: bytes) -> int:
        aid = insert_attachment(conn, guid=guid, byte_size=len(data))
        world[key] = (aid, data)
        return aid

    # present on the other Mac at its recorded path
    a = attachment("present", GUIDS[0], b"present on the other Mac")
    insert_location(conn, attachment_id=a, location="studio",
                    path=f"{PREFIX}0a/10/{GUIDS[0]}/IMG_0001.jpeg", byte_size=24,
                    sha256=sha256_bytes(b"present on the other Mac"))
    write_file(host.messages, f"0a/10/{GUIDS[0]}/IMG_0001.jpeg", b"present on the other Mac")

    # gone from the other Mac's folder, but on its drive (a GUID folder)
    b = attachment("on_drive", GUIDS[1], b"kept on the attached drive")
    insert_location(conn, attachment_id=b, location="studio",
                    path=f"{PREFIX}0b/11/{GUIDS[1]}/IMG_0002.jpeg", byte_size=26)
    insert_location(conn, attachment_id=b, location="V-TEST2",
                    path=f"Backup/{GUIDS[1]}/IMG_0002.jpeg", match="guid_folder", byte_size=26)
    write_file(host.drive, f"Backup/{GUIDS[1]}/IMG_0002.jpeg", b"kept on the attached drive")

    # there, but not the size its listing said: not sent
    c = attachment("wrong_size", GUIDS[2], b"listed at another size")
    insert_location(conn, attachment_id=c, location="studio",
                    path=f"{PREFIX}0c/12/{GUIDS[2]}/IMG_0003.jpeg", byte_size=999)
    write_file(host.messages, f"0c/12/{GUIDS[2]}/IMG_0003.jpeg", b"listed at another size")

    # a symlink where the file should be: never followed
    d = attachment("symlink", GUIDS[3], b"elsewhere")
    insert_location(conn, attachment_id=d, location="studio",
                    path=f"{PREFIX}0d/13/{GUIDS[3]}/IMG_0004.jpeg")
    target = write_file(host.drive, "not-an-attachment.txt", b"elsewhere")
    (host.messages / f"0d/13/{GUIDS[3]}").mkdir(parents=True)
    os.symlink(target, host.messages / f"0d/13/{GUIDS[3]}/IMG_0004.jpeg")

    # only on a drive this Mac does not have
    e = attachment("elsewhere", GUIDS[4], b"on a drive in a drawer")
    insert_location(conn, attachment_id=e, location="D-DRAWER",
                    path=f"Old/{GUIDS[4]}/IMG_0005.jpeg", match="guid_folder")

    # a path no root can read (a temporary directory a chat.db recorded)
    f = attachment("temp_path", GUIDS[5], b"gone with the temp dir")
    insert_location(conn, attachment_id=f, location="studio",
                    path="/private/var/folders/zz/T/IMG_0006.jpeg")
    return world


@requires_rsync
def test_a_push_copies_only_what_checks_out_and_changes_nothing_at_the_source(
    pg_conn: psycopg.Connection, index_host: LocationAccess, tmp_path: Path
) -> None:
    host = PushingHost(tmp_path)
    world = _world(pg_conn, host)
    before = host.state()

    plan = build_push_plan(pg_conn, locations=["studio", "V-TEST2"], access=index_host,
                           data_root=_data_root(index_host))
    planned = {item.attachment_id for item in plan.items}
    assert planned == {world[k][0] for k in ("present", "on_drive", "wrong_size", "symlink")}
    # Round trip through the wire format, as the pushing host reads it.
    plan = parse_plan(plan_to_lines(plan))

    report, results, rsync_log = run_push(
        plan, host.roots, ssh_host="indexhost", ssh=str(write_fake_ssh(tmp_path)),
        rsync=str(RSYNC), runner=run_copy,
    )

    assert host.state() == before  # the pushing host's files were only read
    staged_present = index_host.staging_root / "studio" / f"0a/10/{GUIDS[0]}/IMG_0001.jpeg"
    staged_drive = index_host.staging_root / "V-TEST2" / f"Backup/{GUIDS[1]}/IMG_0002.jpeg"
    assert staged_present.read_bytes() == b"present on the other Mac"
    assert staged_drive.read_bytes() == b"kept on the attached drive"
    assert stat.S_IMODE(staged_present.stat().st_mode) == 0o600
    staged = {p for p in index_host.staging_root.rglob("*") if p.is_file()}
    assert staged == {staged_present, staged_drive}
    assert dict(report.selected) == {"studio": 1, "V-TEST2": 1}
    assert report.copy_exit_codes == {"studio": 0, "V-TEST2": 0}
    assert dict(report.not_sent) == {
        ("studio", "absent"): 1, ("studio", "rejected"): 1, ("studio", "refused"): 1,
    }
    assert report.unserved == 2

    counts = record_push_results(
        pg_conn, parse_results(results_to_lines(results, rsync_log))[0],
        log_dir=_data_root(index_host) / "logs",
    )
    assert dict(counts) == {"staged": 2, "absent": 1, "rejected": 1, "refused": 1}

    # The index host verifies and materializes what arrived.
    fetched = fetch_from_locations(pg_conn, _data_root(index_host),
                                   LocationFetchSettings(access=index_host))
    for key in ("present", "on_drive"):
        aid, data = world[key]
        row = attachment_row(pg_conn, aid)
        assert row["state"] == "materialized" and row["sha256"] == sha256_bytes(data)
    for key in ("wrong_size", "symlink", "elsewhere", "temp_path"):
        assert attachment_row(pg_conn, world[key][0])["state"] == "missing"
    assert dict(fetched.no_access) == {"D-DRAWER": 1}
    assert fetched.refused["studio"] == 1  # the temporary-directory path, closed

    # Nothing is planned twice: the next plan holds only what is still open.
    replan = build_push_plan(pg_conn, locations=["studio", "V-TEST2"], access=index_host,
                             data_root=_data_root(index_host))
    assert replan.items == ()


def test_a_plan_skips_what_the_index_host_can_already_do(
    pg_conn: psycopg.Connection, index_host: LocationAccess, tmp_path: Path
) -> None:
    data_root = _data_root(index_host)
    data = b"already cached under another row"
    digest = sha256_bytes(data)
    write_file(data_root / "attachments" / digest[:2], digest, data)
    cached = insert_attachment(pg_conn, guid=GUIDS[0], byte_size=len(data))
    insert_location(pg_conn, attachment_id=cached, location="studio",
                    path=f"{PREFIX}1/2/{GUIDS[0]}/IMG_0001.jpeg", sha256=digest)
    waiting = insert_attachment(pg_conn, guid=GUIDS[1])
    insert_location(pg_conn, attachment_id=waiting, location="studio",
                    path=f"{PREFIX}3/4/{GUIDS[1]}/IMG_0002.jpeg")
    write_file(index_host.staging_root / "studio", f"3/4/{GUIDS[1]}/IMG_0002.jpeg", b"here")
    unasked = insert_attachment(pg_conn, guid=GUIDS[2])
    insert_location(pg_conn, attachment_id=unasked, location="V-OTHER",
                    path=f"x/{GUIDS[2]}/IMG_0003.jpeg", match="guid_folder")
    wanted = insert_attachment(pg_conn, guid=GUIDS[3])
    insert_location(pg_conn, attachment_id=wanted, location="studio",
                    path=f"{PREFIX}5/6/{GUIDS[3]}/IMG_0004.jpeg")

    plan = build_push_plan(pg_conn, locations=["studio"], access=index_host,
                           data_root=data_root)
    assert [item.attachment_id for item in plan.items] == [wanted]
    assert plan.items[0].candidates[0].rel == f"5/6/{GUIDS[3]}/IMG_0004.jpeg"


# --- no fetch step writes to a source ----------------------------------------


def test_recording_push_results_touches_only_the_attempt_columns(
    pg_conn: psycopg.Connection,
) -> None:
    aid = insert_attachment(pg_conn, guid=GUIDS[0])
    open_row = insert_location(pg_conn, attachment_id=aid, location="studio",
                               path=f"{PREFIX}a/{GUIDS[0]}/IMG_0001.jpeg")
    done_row = insert_location(pg_conn, attachment_id=aid, location="V-TEST2",
                               path=f"b/{GUIDS[0]}/IMG_0001.jpeg", match="guid_folder")
    pg_conn.execute("UPDATE attachment_location SET fetched_at = now(), last_outcome = 'fetched' "
                    "WHERE location_id = %s", (done_row,))
    with pg_conn.cursor() as cur:
        cur.execute("SELECT xmin::text FROM attachment WHERE attachment_id = %s", (aid,))
        before = cur.fetchone()

    counts = record_push_results(pg_conn, [
        PushResult(open_row, LocationOutcome.ABSENT, "absent on the pushing host"),
        PushResult(done_row, LocationOutcome.ABSENT, "a late report"),
    ])

    assert dict(counts) == {"absent": 1}
    assert location_row(pg_conn, open_row)["outcome"] == "absent"
    assert location_row(pg_conn, done_row)["outcome"] == "fetched"
    with pg_conn.cursor() as cur:
        cur.execute("SELECT xmin::text FROM attachment WHERE attachment_id = %s", (aid,))
        assert cur.fetchone() == before


# --- the command, with the index host played in-process ----------------------


@requires_rsync
def test_push_attachments_runs_plan_copy_and_record_end_to_end(
    pg_conn: psycopg.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`imsg push-attachments` on the pushing host, with the index host's two
    commands (`push-attachments-plan`, `push-attachments-record`) run
    in-process against the scratch database where SSH would have run them,
    and the real rsync copying through the stand-in ssh."""
    home = tmp_path / "index-home"
    messages = home / "Library" / "Messages"
    (messages / "Attachments").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")
    (messages / "chat.db").write_text("")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  data_root: {data_root}
  live_chat_db: {messages / 'chat.db'}
database:
  dsn: postgresql://imsg@127.0.0.1:5433/imsgindex
  password: env:IMSG_TEST_PG_PASSWORD
sync:
  sources:
    - name: mini
      chat_db: {messages / 'chat.db'}
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
"""
    )
    monkeypatch.setattr(cli_module, "run_guard_mount_or_exit", lambda data_root: None)
    monkeypatch.setattr(
        cli_module, "connect", lambda database, **kw: psycopg.connect(dsn(DB_NAME), autocommit=True)
    )
    monkeypatch.setattr(cli_module, "verify_data_directory", lambda conn, root: Path(str(root)))

    studio_root = tmp_path / "studio" / "Attachments"
    data = b"a photo only the other Mac kept"
    aid = insert_attachment(pg_conn, guid=GUIDS[0], byte_size=len(data))
    lid = insert_location(pg_conn, attachment_id=aid, location="studio",
                          path=f"{PREFIX}0a/10/{GUIDS[0]}/IMG_0001.jpeg", byte_size=len(data),
                          sha256=sha256_bytes(data))
    write_file(studio_root, f"0a/10/{GUIDS[0]}/IMG_0001.jpeg", data)
    before = tree_state(studio_root)

    runner = CliRunner()
    remote_calls: list[list[str]] = []

    def fake_remote(ssh: str, host: str, argv: list[str], *, stdin: bytes | None,
                    timeout: float) -> subprocess.CompletedProcess[bytes]:
        remote_calls.append(argv)
        assert host == "indexhost" and argv[0] == "/opt/imsg/bin/imsg"
        result = runner.invoke(app, argv[1:], input=stdin)
        return subprocess.CompletedProcess(argv, result.exit_code, result.stdout_bytes,
                                           result.stderr_bytes)

    monkeypatch.setattr(cli_module, "_remote_imsg", fake_remote)
    pushed = runner.invoke(
        app,
        ["push-attachments", "--ssh-host", "indexhost", "--remote-imsg", "/opt/imsg/bin/imsg",
         "--remote-config", str(config), "--root", f"studio={studio_root}",
         "--ssh", str(write_fake_ssh(tmp_path)), "--rsync", str(RSYNC)],
    )
    assert pushed.exit_code == 0, pushed.output
    assert "sent: studio=1" in pushed.stdout
    assert "index host: push-attachments-record: recorded staged=1" in pushed.stdout
    assert [c[1] for c in remote_calls] == ["push-attachments-plan", "push-attachments-record"]
    assert tree_state(studio_root) == before
    # Nothing the pushing host printed names a file.
    assert "IMG_0001" not in pushed.output and GUIDS[0] not in pushed.output
    assert location_row(pg_conn, lid)["outcome"] == "staged"

    backfilled = runner.invoke(
        app, ["backfill-attachments", "--yes-full-run", "--config", str(config)]
    )
    assert backfilled.exit_code == 0, backfilled.output
    assert attachment_row(pg_conn, aid)["sha256"] == sha256_bytes(data)
