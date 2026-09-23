"""`imsg locate-attachments` (D13): finding candidate copies and recording
how each matched, against a real Postgres.

Match rules under test, as measured on the production corpus 2026-09-23
(a GUID folder plus the name: right every time; the same name and size in
another folder: right 99.5% of the time; a name alone: never accepted):

- a path a chat.db recorded, found again in a listing, a drive catalog
  (below a copied Messages folder) or this host's own folder;
- a folder named after the attachment's GUID holding one of its names;
- one of its names at its byte size anywhere else (flagged);
- a name alone, or a GUID folder with another name: counted, never stored.

Fictional content only (D5).
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from _attachment_fetch_fixtures import (
    insert_attachment,
    insert_location,
    location_row,
    requires_postgres,
    scratch_database,
    sha256_bytes,
    write_file,
)
from chatdb_fixture import (
    ChatDbBuilder,
    FixtureAttachment,
    FixtureChat,
    FixtureHandle,
    FixtureMessage,
)
from imsg.backfill.fetch import LocationAccess, PullLocation
from imsg.backfill.locate import (
    MAX_UNRECORDED_ROWS_PER_LOCATION,
    LocatedFile,
    LocateError,
    catalog_code_for,
    run_locate,
)

pytestmark = requires_postgres

PREFIX = "~/Library/Messages/Attachments/"
GUID_REC = "A1B2C3D4-0000-4000-8000-000000000001"
GUID_FOLDER = "A1B2C3D4-0000-4000-8000-000000000002"
GUID_SIZE = "A1B2C3D4-0000-4000-8000-000000000003"
GUID_WEAK = "A1B2C3D4-0000-4000-8000-000000000004"
SHA = "c" * 64


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
    yield from scratch_database("imsg_index_attachment_locate_test")


@pytest.fixture
def access(tmp_path: Path) -> LocationAccess:
    root = tmp_path / "home" / "Library" / "Messages" / "Attachments"
    root.mkdir(parents=True)
    return LocationAccess(
        local_location="mini",
        attachments_root=root,
        staging_root=tmp_path / "data_root" / "attachment-staging",
        push_locations=("studio", "V-TEST2"),
        pulls=(PullLocation("N-TEST3", "nashost", "/volume1/homes"),),
    )


def _rows(conn: psycopg.Connection) -> list[tuple[str, str, str, str, int | None, str | None]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.source_guid, l.location, l.path, l.match_quality::text, l.byte_size, l.sha256
              FROM attachment_location l JOIN attachment a USING (attachment_id)
             ORDER BY a.source_guid, l.location, l.path
            """
        )
        return [tuple(r) for r in cur.fetchall()]  # type: ignore[misc]


def _targets(conn: psycopg.Connection) -> dict[str, int]:
    return {
        GUID_REC: insert_attachment(
            conn, guid=GUID_REC, filename="IMG_0001.heic",
            source_path=f"{PREFIX}0a/10/{GUID_REC}/IMG_0001.heic", byte_size=100,
        ),
        GUID_FOLDER: insert_attachment(conn, guid=GUID_FOLDER, filename="IMG_0002.jpeg",
                                       byte_size=200),
        GUID_SIZE: insert_attachment(conn, guid=GUID_SIZE, filename="Scan.pdf", byte_size=300,
                                     mime_type="application/pdf", uti="com.adobe.pdf"),
        GUID_WEAK: insert_attachment(conn, guid=GUID_WEAK, filename="IMG_0004.jpeg",
                                     byte_size=400),
    }


def test_a_listing_is_matched_by_recorded_path_guid_folder_and_name_size(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    _targets(pg_conn)
    listing = tmp_path / "studio.tsv"
    listing.write_text(
        "\n".join(
            [
                f"0a/10/{GUID_REC}/IMG_0001.heic\t100\t1\t{SHA}",
                f"55/66/{GUID_FOLDER}/img_0002.JPEG\t201\t1\t{'d' * 64}",  # name compares case-blind
                "77/88/99999999-0000-4000-8000-000000000000/Scan.pdf\t300\t1\t",
                "77/88/99999999-0000-4000-8000-000000000001/IMG_0004.jpeg\t999\t1\t",  # name only
                f"12/34/{GUID_WEAK}/Other.jpeg\t400\t1\t",  # its GUID folder, another name
            ]
        )
        + "\n"
    )

    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path / "data_root",
                           listings=[LocatedFile("studio", listing)], walk_local=False)

    assert _rows(pg_conn) == [
        (GUID_REC, "studio", f"{PREFIX}0a/10/{GUID_REC}/IMG_0001.heic", "recorded_path", 100, SHA),
        (GUID_FOLDER, "studio", f"{PREFIX}55/66/{GUID_FOLDER}/img_0002.JPEG", "guid_folder", 201,
         "d" * 64),
        (GUID_SIZE, "studio", f"{PREFIX}77/88/99999999-0000-4000-8000-000000000000/Scan.pdf",
         "name_size", 300, None),
    ]
    assert report.weak["studio"] == 2
    assert report.stored[("studio", "recorded_path", "inserted")] == 1


def test_a_drive_catalog_gives_its_code_and_finds_a_copied_messages_folder(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    _targets(pg_conn)
    run_dir = tmp_path / "runs" / "20260911T000000Z_Host-0a1b2c_V-TEST2_0123abcd"
    run_dir.mkdir(parents=True)
    catalog = run_dir / "catalog.files.tsv.gz"
    with gzip.open(catalog, "wt", encoding="utf-8") as f:
        f.write("# path\tsize\tmtime_epoch\text\n")
        # A backup of a home folder: the part below the Messages folder is
        # the recorded path, although the folder above it differs.
        f.write(f"Backups/2024-01-01/Users/alice/Library/Messages/Attachments/0a/10/{GUID_REC}/"
                f"IMG_0001.heic\t100\t1\theic\n")
        f.write(f"Photos/{GUID_FOLDER}/IMG_0002.jpeg\t200\t1\tjpeg\n")
    flat = tmp_path / "D-TEST1.files.2.tsv.gz"
    with gzip.open(flat, "wt", encoding="utf-8") as f:
        f.write("Scans/Scan.pdf\t300\t1\tpdf\n")

    assert catalog_code_for(catalog) == "V-TEST2"
    assert catalog_code_for(flat) == "D-TEST1"
    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path / "data_root",
                           catalogs=[str(tmp_path / "runs"), str(flat)], walk_local=False)

    assert report.catalog_files == 2
    assert _rows(pg_conn) == [
        (GUID_REC, "V-TEST2", f"Backups/2024-01-01/Users/alice/Library/Messages/Attachments/0a/10/"
         f"{GUID_REC}/IMG_0001.heic", "recorded_path", 100, None),
        (GUID_FOLDER, "V-TEST2", f"Photos/{GUID_FOLDER}/IMG_0002.jpeg", "guid_folder", 200, None),
        (GUID_SIZE, "D-TEST1", "Scans/Scan.pdf", "name_size", 300, None),
    ]


def test_a_catalog_with_no_code_in_its_name_is_skipped_unless_one_is_given(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    _targets(pg_conn)
    unnamed = tmp_path / "files.tsv"
    unnamed.write_text("Scans/Scan.pdf\t300\t1\tpdf\n")
    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path, catalogs=[str(unnamed)],
                           walk_local=False)
    assert report.catalogs_skipped == 1 and _rows(pg_conn) == []
    run_locate(pg_conn, access=access, data_root=tmp_path, catalogs=[f"D-GIVEN={unnamed}"],
               walk_local=False)
    assert [r[1] for r in _rows(pg_conn)] == ["D-GIVEN"]


def test_name_and_size_copies_are_capped_per_location(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    insert_attachment(pg_conn, guid=GUID_SIZE, filename="Scan.pdf", byte_size=300)
    catalog = tmp_path / "D-TEST1.files.tsv"
    catalog.write_text("".join(f"copy{i}/Scan.pdf\t300\t1\tpdf\n" for i in range(25)))
    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path,
                           catalogs=[str(catalog)], walk_local=False)
    assert len(_rows(pg_conn)) == MAX_UNRECORDED_ROWS_PER_LOCATION
    assert report.capped == 25 - MAX_UNRECORDED_ROWS_PER_LOCATION


def _seed_db(path: Path, guid: str, recorded: str) -> Path:
    builder = ChatDbBuilder()
    builder.add_chat(FixtureChat(guid="chat-1", rowid=1))
    builder.add_handle(FixtureHandle(raw_value="+15550000001", rowid=1))
    builder.link_participant("chat-1", "+15550000001")
    builder.add_message(FixtureMessage(guid="msg-1", chat_guid="chat-1",
                                       handle_raw_value="+15550000001", rowid=1))
    builder.add_attachment(FixtureAttachment(guid=guid, rowid=1, filename="IMG_0002.jpeg",
                                             source_path=recorded))
    builder.link_attachment("msg-1", guid)
    return builder.build(path)


def test_a_seed_database_path_is_stored_and_then_found_in_a_listing(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    """A Mac extracted before extraction kept every source's path: its
    chat.db copy supplies the path, and a listing of that Mac's folder then
    finds the file at exactly that path although the folder is not the
    attachment's GUID."""
    _targets(pg_conn)
    recorded = f"{PREFIX}f0/0d/AB12CD34-EF56-4000-8000-00000000FFFF/IMG_0002.jpeg"
    seed = _seed_db(tmp_path / "studio-chat.db", GUID_FOLDER, recorded)
    listing = tmp_path / "studio.tsv"
    listing.write_text(f"f0/0d/AB12CD34-EF56-4000-8000-00000000FFFF/IMG_0002.jpeg\t777\t1\t{SHA}\n")

    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path,
                           seeds=[LocatedFile("studio", seed)],
                           listings=[LocatedFile("studio", listing)], walk_local=False)

    assert report.seed_rows == 1
    # One row: the seed's recorded path and the listing's copy are the same
    # file on the same Mac, so the listing adds its size and hash to it.
    assert _rows(pg_conn) == [(GUID_FOLDER, "studio", recorded, "recorded_path", 777, SHA)]
    with pg_conn.cursor() as cur:
        cur.execute("SELECT reported_by FROM attachment_location")
        reporters = cur.fetchone()
    assert reporters is not None and reporters[0] == ["studio", "listing:studio.tsv"]


def test_a_seed_database_with_unfolded_write_ahead_log_is_refused(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    seed = _seed_db(tmp_path / "copy.db", GUID_FOLDER, f"{PREFIX}a/b/IMG_0002.jpeg")
    Path(f"{seed}-wal").write_bytes(b"\0" * 4096)
    with pytest.raises(LocateError, match="write-ahead log"):
        run_locate(pg_conn, access=access, data_root=tmp_path,
                   seeds=[LocatedFile("studio", seed)], walk_local=False)


def test_this_hosts_own_folder_is_walked(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    _targets(pg_conn)
    write_file(access.attachments_root, f"0a/10/{GUID_REC}/IMG_0001.heic", b"x" * 100)
    write_file(access.attachments_root, f"99/aa/{GUID_FOLDER}/IMG_0002.jpeg", b"y" * 5)
    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path)
    assert report.local_files == 2
    assert _rows(pg_conn) == [
        (GUID_REC, "mini", f"{PREFIX}0a/10/{GUID_REC}/IMG_0001.heic", "recorded_path", 100, None),
        (GUID_FOLDER, "mini", f"{PREFIX}99/aa/{GUID_FOLDER}/IMG_0002.jpeg", "guid_folder", 5,
         None),
    ]


def test_new_evidence_reopens_a_closed_location_and_the_same_evidence_does_not(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    ids = _targets(pg_conn)
    path = f"{PREFIX}0a/10/{GUID_REC}/IMG_0001.heic"
    lid = insert_location(pg_conn, attachment_id=ids[GUID_REC], location="studio", path=path,
                          byte_size=100, sha256=SHA)
    pg_conn.execute(
        "UPDATE attachment_location SET last_tried_at = now(), last_outcome = 'rejected', "
        "last_error = 'rejected: sha256 mismatch' WHERE location_id = %s", (lid,)
    )
    listing = tmp_path / "studio.tsv"

    listing.write_text(f"0a/10/{GUID_REC}/IMG_0001.heic\t100\t1\t{SHA}\n")
    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path,
                           listings=[LocatedFile("studio", listing)], walk_local=False)
    assert report.stored[("studio", "recorded_path", "unchanged")] == 1
    assert location_row(pg_conn, lid)["outcome"] == "rejected"

    fresh = sha256_bytes(b"the file was replaced since")
    listing.write_text(f"0a/10/{GUID_REC}/IMG_0001.heic\t100\t1\t{fresh}\n")
    report, _ = run_locate(pg_conn, access=access, data_root=tmp_path,
                           listings=[LocatedFile("studio", listing)], walk_local=False)
    assert report.stored[("studio", "recorded_path", "updated")] == 1
    reopened = location_row(pg_conn, lid)
    assert reopened["outcome"] is None and reopened["tried_at"] is None
    assert reopened["sha256"] == fresh


def test_a_dry_run_counts_everything_and_stores_nothing(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    _targets(pg_conn)
    listing = tmp_path / "studio.tsv"
    listing.write_text(f"0a/10/{GUID_REC}/IMG_0001.heic\t100\t1\t{SHA}\n")
    report, coverage = run_locate(pg_conn, access=access, data_root=tmp_path,
                                  listings=[LocatedFile("studio", listing)], walk_local=False,
                                  dry_run=True)
    assert report.stored[("studio", "recorded_path", "inserted")] == 1
    assert coverage.best["push:studio"] == 1
    assert _rows(pg_conn) == []


def test_coverage_says_where_each_next_copy_is_and_what_has_none(
    pg_conn: psycopg.Connection, access: LocationAccess, tmp_path: Path
) -> None:
    ids = _targets(pg_conn)
    link = insert_attachment(pg_conn, filename="Preview.pluginPayloadAttachment",
                             uti="com.apple.messages.pluginPayloadAttachment", mime_type=None)
    insert_location(pg_conn, attachment_id=ids[GUID_REC], location="studio",
                    path=f"{PREFIX}0a/10/{GUID_REC}/IMG_0001.heic")
    insert_location(pg_conn, attachment_id=ids[GUID_FOLDER], location="N-TEST3",
                    path="alice/IMG_0002.jpeg", match="name_size", byte_size=200)
    insert_location(pg_conn, attachment_id=ids[GUID_SIZE], location="D-NOWHERE",
                    path="Scans/Scan.pdf", match="name_size", byte_size=300)
    closed = insert_location(pg_conn, attachment_id=ids[GUID_WEAK], location="studio",
                             path=f"{PREFIX}1/2/{GUID_WEAK}/IMG_0004.jpeg")
    pg_conn.execute("UPDATE attachment_location SET last_tried_at = now(), "
                    "last_outcome = 'absent' WHERE location_id = %s", (closed,))

    _, coverage = run_locate(pg_conn, access=access, data_root=tmp_path, walk_local=False)

    assert dict(coverage.best) == {"push:studio": 1, "pull:N-TEST3": 1, "no-access:D-NOWHERE": 1}
    assert coverage.best_is_flagged == 2
    assert dict(coverage.exhausted) == {("missing", "image"): 1}
    assert dict(coverage.no_candidate) == {("missing", "link-preview"): 1}
    assert coverage.unfetchable == 2
    assert link  # the link-preview payload no host has a copy of
