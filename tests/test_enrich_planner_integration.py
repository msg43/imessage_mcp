"""Filling the S5b enrichment queue (SPEC §8 S5b): every materialized
attachment gets the router's kinds for its content-sniffed MIME type,
once — when S5a materializes it, and through `imsg enrich --plan` for
everything materialized before that hook existed.

Postgres integration tests; skip cleanly when no scratch Postgres is
reachable. Real files, real `file` sniffing, fictional personas only.
"""

from __future__ import annotations

import hashlib
import os
import struct
import uuid
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from _pdf_fixtures import write_minimal_pdf
from imsg.backfill.pipeline import run_backfill
from imsg.backfill.throttle import RateThrottle
from imsg.db.migrations import PostgresMigrationRunner
from imsg.enrich.queue import complete_task, enqueue

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_enrich_planner_test"

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


# --------------------------------------------------------------------------
# fixture content: each is the real format `file` recognizes
# --------------------------------------------------------------------------


def _tiny_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _pdf_bytes(tmp_path: Path) -> bytes:
    pdf = tmp_path / f"{uuid.uuid4()}.pdf"
    write_minimal_pdf(pdf, ["Deck rebuild estimate for Acme Construction"])
    return pdf.read_bytes()


_VCARD = (
    b"BEGIN:VCARD\r\nVERSION:3.0\r\nN:Example;Alice;;;\r\nFN:Alice Example\r\n"
    b"ORG:Acme Construction;\r\nTEL;type=CELL:+1 555 0100\r\nEND:VCARD\r\n"
)

# 'caff', version 1, flags 0, then a 'desc' chunk header: what `file`
# 5.41 calls application/octet-stream and every voice message starts with.
_CAF = b"caff\x00\x01\x00\x00desc" + (32).to_bytes(8, "big") + bytes(32)


def _zip_bytes(tmp_path: Path) -> bytes:
    archive = tmp_path / f"{uuid.uuid4()}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("readme.txt", "an archive the router has no route for")
    return archive.read_bytes()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _materialized(
    conn: psycopg.Connection, data_root: Path, content: bytes, *, mime_type: str | None = None
) -> int:
    """An attachment S5a already materialized: its bytes in the
    content-addressed cache under `data_root`, the row `materialized`."""
    sha = hashlib.sha256(content).hexdigest()
    cache = data_root / "attachments" / sha[:2] / sha
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(content)
    return _insert(conn, state="materialized", cache_path=str(cache), sha256=sha, mime_type=mime_type)


def _insert(
    conn: psycopg.Connection,
    *,
    state: str,
    cache_path: str | None = None,
    sha256: str | None = None,
    mime_type: str | None = None,
    source_path: str | None = None,
) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attachment (source_guid, attachment_key, state, cache_path, sha256,
                                    mime_type, source_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING attachment_id
            """,
            (guid, f"key-{guid}", state, cache_path, sha256, mime_type, source_path),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _queued(conn: psycopg.Connection) -> dict[int, set[str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT attachment_id, kind::text FROM enrichment")
        rows = cur.fetchall()
    out: dict[int, set[str]] = {}
    for attachment_id, kind in rows:
        out.setdefault(int(attachment_id), set()).add(str(kind))
    return out


def _count(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM enrichment")
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data_root"
    root.mkdir()
    return root


# --------------------------------------------------------------------------
# on materialization (S5a)
# --------------------------------------------------------------------------


def test_materialization_enqueues_the_router_kinds_for_the_sniffed_content(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    attachments_root = tmp_path / "Attachments"
    (attachments_root / "a").mkdir(parents=True)
    files = {
        "photo.jpg": _tiny_png(),  # the extension lies; the content is a PNG
        "bid.pdf": _pdf_bytes(tmp_path),
        "card.vcf": _VCARD,
        "voice.caf": _CAF,
        "notes.txt": b"Pour moved to Thursday, tell Bob.\n",
        "bundle.zip": _zip_bytes(tmp_path),
    }
    ids: dict[str, int] = {}
    for name, content in files.items():
        path = attachments_root / "a" / name
        path.write_bytes(content)
        ids[name] = _insert(scratch_db, state="dataless", source_path=str(path))

    report = run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=RateThrottle(10_000, sleep_fn=lambda _: None),
    )

    assert report.materialized == len(files)
    queued = _queued(scratch_db)
    assert queued.get(ids["photo.jpg"]) == {"ocr", "caption"}
    assert queued.get(ids["bid.pdf"]) == {"pdf_text"}
    assert queued.get(ids["card.vcf"]) == {"doc_text"}
    assert queued.get(ids["voice.caf"]) == {"transcript"}
    assert queued.get(ids["notes.txt"]) == {"doc_text"}
    assert ids["bundle.zip"] not in queued
    assert report.enrichment_enqueued == 6
    assert report.enrichment_unroutable == 1
    assert report.enrichment_plan_errors == 0


def test_a_backfill_dry_run_enqueues_nothing(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    path = attachments_root / "card.vcf"
    path.write_bytes(_VCARD)
    _insert(scratch_db, state="dataless", source_path=str(path))

    run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=RateThrottle(10_000, sleep_fn=lambda _: None),
        dry_run=True,
    )

    assert _count(scratch_db) == 0


def test_the_planner_adds_nothing_after_the_materialization_hook(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    for name, content in (("p.png", _tiny_png()), ("c.vcf", _VCARD)):
        (attachments_root / name).write_bytes(content)
        _insert(scratch_db, state="dataless", source_path=str(attachments_root / name))
    run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=RateThrottle(10_000, sleep_fn=lambda _: None),
    )
    before = _count(scratch_db)

    report = plan_enrichment(scratch_db, data_root=data_root)

    assert before == 3
    assert _count(scratch_db) == before
    assert report.enqueued == {}
    assert report.already_queued == {"ocr": 1, "caption": 1, "doc_text": 1}


# --------------------------------------------------------------------------
# the planner (`imsg enrich --plan`)
# --------------------------------------------------------------------------


def test_the_planner_enqueues_every_missing_kind_exactly_once(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    photo = _materialized(scratch_db, data_root, _tiny_png(), mime_type="image/png")
    pdf = _materialized(scratch_db, data_root, _pdf_bytes(tmp_path), mime_type="application/pdf")
    card = _materialized(scratch_db, data_root, _VCARD, mime_type="text/vcard")
    # The photo's OCR already ran: it must be neither duplicated nor reset.
    enqueue(scratch_db, photo, ("ocr",))
    complete_task(scratch_db, photo, "ocr", model="m", model_version=None, text="OCR TEXT")

    first = plan_enrichment(scratch_db, data_root=data_root)
    second = plan_enrichment(scratch_db, data_root=data_root)

    assert first.enqueued == {"caption": 1, "pdf_text": 1, "doc_text": 1}
    assert first.already_queued == {"ocr": 1}
    assert second.enqueued == {}
    assert _queued(scratch_db) == {
        photo: {"ocr", "caption"},
        pdf: {"pdf_text"},
        card: {"doc_text"},
    }
    with scratch_db.cursor() as cur:
        cur.execute(
            "SELECT state, text FROM enrichment WHERE attachment_id = %s AND kind = 'ocr'", (photo,)
        )
        assert cur.fetchone() == ("done", "OCR TEXT")


def test_the_kinds_filter_limits_what_the_planner_enqueues(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    photo = _materialized(scratch_db, data_root, _tiny_png())
    pdf = _materialized(scratch_db, data_root, _pdf_bytes(tmp_path))
    card = _materialized(scratch_db, data_root, _VCARD)

    cheap = plan_enrichment(scratch_db, data_root=data_root, kinds=("ocr", "pdf_text"))

    assert cheap.enqueued == {"ocr": 1, "pdf_text": 1}
    assert cheap.not_selected == {"caption": 1, "doc_text": 1}
    assert _queued(scratch_db) == {photo: {"ocr"}, pdf: {"pdf_text"}}

    later = plan_enrichment(scratch_db, data_root=data_root, kinds=("caption",))

    assert later.enqueued == {"caption": 1}
    assert _queued(scratch_db) == {photo: {"ocr", "caption"}, pdf: {"pdf_text"}}
    assert card not in _queued(scratch_db)


def test_a_planner_dry_run_counts_per_kind_and_writes_nothing(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    _materialized(scratch_db, data_root, _tiny_png())
    _materialized(scratch_db, data_root, _CAF)
    _materialized(scratch_db, data_root, _zip_bytes(tmp_path))

    report = plan_enrichment(scratch_db, data_root=data_root, dry_run=True)

    assert report.dry_run is True
    assert report.attachments == 3
    assert report.enqueued == {"ocr": 1, "caption": 1, "transcript": 1}
    assert report.unroutable == {"application/zip": 1}
    assert _count(scratch_db) == 0


def test_the_planner_routes_on_content_not_on_the_stored_mime_type(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    """chat.db's MIME type is the sender's claim. A link-preview payload
    carries none at all, and a file named like a photo can be a PDF."""
    link_preview = _materialized(scratch_db, data_root, _tiny_png(), mime_type=None)
    fake_photo = _materialized(
        scratch_db, data_root, _pdf_bytes(tmp_path), mime_type="image/jpeg"
    )
    voice = _materialized(scratch_db, data_root, _CAF, mime_type=None)

    from imsg.enrich.planner import plan_enrichment

    plan_enrichment(scratch_db, data_root=data_root)

    assert _queued(scratch_db) == {
        link_preview: {"ocr", "caption"},
        fake_photo: {"pdf_text"},
        voice: {"transcript"},
    }


def test_unroutable_attachments_are_reported_by_sniffed_type(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    _materialized(scratch_db, data_root, _zip_bytes(tmp_path))
    _materialized(scratch_db, data_root, b"")

    report = plan_enrichment(scratch_db, data_root=data_root)

    assert report.unroutable == {"application/zip": 1, "inode/x-empty": 1}
    assert _count(scratch_db) == 0


def test_only_materialized_attachments_are_planned(
    scratch_db: psycopg.Connection, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    for state in ("dataless", "missing", "error", "unsupported"):
        _insert(scratch_db, state=state)

    report = plan_enrichment(scratch_db, data_root=data_root)

    assert report.attachments == 0
    assert _count(scratch_db) == 0


def test_a_cache_path_outside_data_root_is_refused_unread(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    outside = tmp_path / "elsewhere" / "card"
    outside.parent.mkdir()
    outside.write_bytes(_VCARD)
    _insert(scratch_db, state="materialized", cache_path=str(outside))
    link = data_root / "attachments" / "ab" / "link"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    _insert(scratch_db, state="materialized", cache_path=str(link))
    _insert(scratch_db, state="materialized", cache_path=None)

    report = plan_enrichment(scratch_db, data_root=data_root)

    assert report.refused == 3
    assert report.enqueued == {}
    assert _count(scratch_db) == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file")
def test_a_file_that_cannot_be_sniffed_is_counted_and_the_rest_still_plan(
    scratch_db: psycopg.Connection, data_root: Path
) -> None:
    from imsg.enrich.planner import plan_enrichment

    unreadable_id = _materialized(scratch_db, data_root, b"locked away\n")
    card = _materialized(scratch_db, data_root, _VCARD)
    with scratch_db.cursor() as cur:
        cur.execute("SELECT cache_path FROM attachment WHERE attachment_id = %s", (unreadable_id,))
        row = cur.fetchone()
    assert row is not None
    locked = Path(row[0])
    locked.chmod(0)
    try:
        report = plan_enrichment(scratch_db, data_root=data_root)
    finally:
        locked.chmod(0o600)

    assert report.sniff_failed == 1
    assert _queued(scratch_db) == {card: {"doc_text"}}


def test_an_attachment_the_fetcher_materializes_from_another_copy_is_queued_too(
    scratch_db: psycopg.Connection, tmp_path: Path, data_root: Path
) -> None:
    """D13's fetcher materializes rows through its own path (a copy on
    another Mac, a drive or the NAS), not S5a's. Those rows need their
    enrichment queued just the same."""
    from _attachment_fetch_fixtures import (
        insert_attachment,
        insert_location,
        sha256_bytes,
        write_file,
    )
    from imsg.backfill.fetch import LocationAccess, LocationFetchSettings

    attachments_root = tmp_path / "Attachments"
    attachments_root.mkdir()
    guid = "5A0C3B21-0000-4000-8000-00000000E001"
    aid = insert_attachment(
        scratch_db,
        guid=guid,
        filename="Alice Example.vcf",
        source_path="/private/var/folders/zz/abc/T/Alice Example.vcf",
        state="unsupported",
        last_error="unsupported[temp-directory-path]: refused",
        byte_size=len(_VCARD),
        mime_type="text/vcard",
        uti="public.vcard",
    )
    rel = f"7f/15/{guid}/Alice Example.vcf"
    insert_location(
        scratch_db,
        attachment_id=aid,
        location="studio",
        path="~/Library/Messages/Attachments/" + rel,
        byte_size=len(_VCARD),
        sha256=sha256_bytes(_VCARD),
    )
    staging = (data_root / "attachment-staging").resolve()
    write_file(staging / "studio", rel, _VCARD)
    settings = LocationFetchSettings(
        access=LocationAccess(
            local_location="mini", attachments_root=attachments_root, staging_root=staging
        )
    )

    report = run_backfill(
        scratch_db,
        data_root,
        attachments_root,
        yes_full_run=True,
        throttle=RateThrottle(10_000, sleep_fn=lambda _: None),
        locations=settings,
    )

    assert report.locations is not None
    assert report.locations.materialized_total == 1
    assert _queued(scratch_db) == {aid: {"doc_text"}}
    assert report.enrichment_enqueued == 1
