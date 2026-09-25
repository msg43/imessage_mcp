"""Attacker-supplied attachments through the real enrichment pipeline
(`imsg.enrich.pipeline.process_one_task`) against a real Postgres queue
(SPEC §8 S5b, D6; QA review 2026-09-24).

Each test feeds one synthetic hostile file (`tests/_hostile_attachments.py`)
through a claimed task and reads back what the queue recorded. A ceiling
hit must end the task `failed` on its first attempt, with `last_error`
naming the ceiling: a typed permanent failure, not a hang, not a retry.

Only names that existed before this change are imported at module level,
so the file runs against the unfixed code and fails there for what it
checks rather than for an import. Skips cleanly without a reachable
scratch Postgres, like every integration file; the decoder tests also
need macOS `sandbox-exec`, poppler and ffmpeg. Fictional personas only.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from _hostile_attachments import (
    png_size,
    write_dense_page_pdf,
    write_huge_text_pdf,
    write_pdf,
    write_png_bomb,
)
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.migrations import PostgresMigrationRunner
from imsg.enrich.pipeline import EnrichmentProviders, process_one_task
from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider
from imsg.enrich.queue import claim_tasks, enqueue

TEST_PG_HOST = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
TEST_PG_PORT = os.environ.get("IMSG_TEST_PG_PORT", "55432")
TEST_PG_USER = os.environ.get("IMSG_TEST_PG_USER", "postgres")
TEST_DB_NAME = "imsg_index_enrich_hostile_test"

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

PILLOW_BOMB_CEILING = 178_956_970
"""`enrichment.limits.max_image_pixels`'s default, written out so this
file imports nothing the change added."""

DECODERS = frozenset({"pdfinfo", "pdftotext", "pdftoppm", "ffmpeg", "ffprobe"})


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


pytestmark = [
    pytest.mark.skipif(
        not _admin_reachable(),
        reason=(
            "no reachable scratch Postgres instance "
            f"(tried {TEST_PG_HOST}:{TEST_PG_PORT}) — set IMSG_TEST_PG_HOST/"
            "IMSG_TEST_PG_PORT/IMSG_TEST_PG_USER to point at one"
        ),
    ),
    pytest.mark.skipif(
        not Path("/usr/bin/sandbox-exec").exists()
        or any(shutil.which(tool) is None for tool in DECODERS),
        reason="needs macOS sandbox-exec, poppler and ffmpeg",
    ),
]


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
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


@pytest.fixture
def config(config_dict_factory: Any) -> Config:
    return load_config_dict(config_dict_factory())


def _with_limits(config: Config, **limits: int) -> Config:
    return config.model_copy(
        update={
            "enrichment": config.enrichment.model_copy(
                update={"limits": config.enrichment.limits.model_copy(update=limits)}
            )
        }
    )


class RecordingOcr(FakeOcrProvider):
    """The fake OCR provider, remembering the pixel size of every image it
    was handed (read from the PNG header)."""

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def recognize_text(self, image_path: Path) -> str:
        self.sizes.append(png_size(image_path))
        return super().recognize_text(image_path)


def _providers(ocr: Any = None) -> EnrichmentProviders:
    return EnrichmentProviders(
        ocr=ocr if ocr is not None else FakeOcrProvider(),
        caption=FakeCaptionProvider(),
        transcription=FakeTranscriptionProvider(),
    )


def _attachment(conn: psycopg.Connection, path: Path) -> int:
    guid = f"att-{uuid.uuid4()}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO attachment (source_guid, attachment_key, filename, cache_path, state) "
            "VALUES (%s, %s, %s, %s, 'materialized') RETURNING attachment_id",
            (guid, f"key-{guid}", path.name, str(path)),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _run(
    conn: psycopg.Connection, config: Config, path: Path, kind: str, providers: EnrichmentProviders
) -> tuple[str, str, int, str | None, Any]:
    """Enqueue `kind` for `path`, claim it, process it; return the
    outcome and what the queue recorded: state, attempts, last_error,
    detail."""
    attachment_id = _attachment(conn, path)
    enqueue(conn, attachment_id, (kind,))
    (task,) = claim_tasks(conn, worker_id="hostile-test", limit=1, kinds=(kind,))
    outcome = process_one_task(conn, config, providers, task)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state::text, attempts, last_error, detail FROM enrichment "
            "WHERE attachment_id = %s AND kind = %s",
            (attachment_id, kind),
        )
        row = cur.fetchone()
    assert row is not None
    state, attempts, last_error, detail = row
    return outcome, str(state), int(attempts), last_error, detail


def _assert_failed_permanently(
    result: tuple[str, str, int, str | None, Any], *, naming: str
) -> str:
    outcome, state, attempts, last_error, _ = result
    assert (outcome, state, attempts) == ("failed", "failed", 1), (
        f"expected a permanent failure on the first attempt, got outcome={outcome} "
        f"state={state} attempts={attempts} last_error={last_error!r}"
    )
    assert last_error is not None and naming in last_error, last_error
    return last_error


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every argv a subprocess is started with during the test.
    `subprocess.run` starts its process through this same class."""
    seen: list[list[str]] = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):  # type: ignore[valid-type,misc]
        def __init__(self, args: Any, *rest: Any, **kwargs: Any) -> None:
            seen.append([os.fspath(a) for a in args] if not isinstance(args, str) else [args])
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    return seen


def _media(path: Path, *args: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", *args, str(path)], check=True, capture_output=True, timeout=60
    )
    return path


# --------------------------------------------------------------------------
# PDFs
# --------------------------------------------------------------------------


def test_a_pdf_whose_text_is_enormous_fails_at_the_output_ceiling(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    """1,000 pages of about 80 KiB of text each, from a 158 KB file: 78 MiB
    of text, over the 64 MiB ceiling. It used to be read into memory
    whole and indexed; now `pdftotext` is stopped at the ceiling."""
    pdf = write_huge_text_pdf(data_root / "huge-text.pdf", pages=1000)
    result = _run(db, config, pdf, "pdf_text", _providers())
    _assert_failed_permanently(result, naming="decoder output ceiling")
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM attachment_chunk")
        assert cur.fetchone() == (0,)


def test_a_task_past_its_deadline_fails_permanently_instead_of_retrying(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    """Extracting the enormous text layer takes about 11 s; the task is
    allowed 2. A timeout used to be an ordinary failure, retried with
    backoff up to five times."""
    pdf = write_huge_text_pdf(data_root / "slow.pdf", pages=1000)
    result = _run(db, _with_limits(config, task_timeout_seconds=2), pdf, "pdf_text", _providers())
    _assert_failed_permanently(result, naming="task_timeout_seconds")


def test_the_page_count_is_checked_before_any_page_renders(
    db: psycopg.Connection, data_root: Path, config: Config, launched: list[list[str]]
) -> None:
    pdf = write_pdf(data_root / "long.pdf", pages=40)
    result = _run(db, _with_limits(config, max_pdf_pages=3), pdf, "ocr", _providers())
    _assert_failed_permanently(result, naming="max_pdf_pages")
    rendered = [argv for argv in launched if any(Path(a).name == "pdftoppm" for a in argv)]
    assert rendered == [], f"pages were rendered before the page count was checked: {rendered}"


def test_a_huge_page_is_rendered_within_the_pixel_ceiling(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    """A 50-inch-square page is 225 megapixels at 300 dpi, over the 179
    megapixel ceiling; it is rendered at 267 dpi instead and still read."""
    pdf = write_pdf(data_root / "poster.pdf", media_box=(0, 0, 3600, 3600))
    ocr = RecordingOcr()
    outcome, state, _, last_error, detail = _run(db, config, pdf, "ocr", _providers(ocr))
    assert (outcome, state) == ("done", "done"), last_error
    assert len(ocr.sizes) == 1
    width, height = ocr.sizes[0]
    assert width * height <= PILLOW_BOMB_CEILING, f"OCR was handed {width}x{height}"
    assert detail["downscaled_pages"] == [1]


def test_a_pdf_page_image_over_the_temp_ceiling_fails_the_task(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    pdf = write_dense_page_pdf(data_root / "dense.pdf")  # about 95 KB as a 300 dpi PNG
    result = _run(db, _with_limits(config, temp_bytes_per_task=64 * 1024), pdf, "ocr", _providers())
    last_error = _assert_failed_permanently(result, naming="temp_bytes_per_task")
    assert "pdftoppm" in last_error


def test_a_decoder_over_the_memory_ceiling_fails_the_task(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    """A 40-inch page at 300 dpi (144 megapixels, under the pixel ceiling)
    needs a bitmap of several hundred megabytes; with the decoder memory
    ceiling at 128 MiB, `pdftoppm` is stopped."""
    pdf = write_pdf(data_root / "big.pdf", media_box=(0, 0, 2880, 2880))
    limited = _with_limits(config, max_decoder_memory_bytes=128 * 2**20)
    last_error = _assert_failed_permanently(
        _run(db, limited, pdf, "ocr", _providers()), naming="max_decoder_memory_bytes"
    )
    assert "pdftoppm" in last_error


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------


def _vision_runtime_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in ("Foundation", "Quartz", "Vision")
    )


@pytest.mark.skipif(
    not _vision_runtime_available(), reason="needs pyobjc's Vision and Quartz (the models extra)"
)
def test_a_decompression_bomb_is_refused_before_vision_decodes_it(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    """A 20,000 x 20,000 PNG (400 megapixels) of a few tens of kilobytes,
    through the real OCR provider and the real ImageIO header read. Only
    Vision's recognition step is replaced, by one that records what it
    was asked to decode: without the ceiling, real Vision spent 17.4 s on
    this file (measured 2026-09-24)."""
    from imsg.enrich.vision_ocr import AppleVisionOcrProvider

    class RecordingVision(AppleVisionOcrProvider):
        def __init__(self) -> None:
            super().__init__()
            self.decoded: list[Path] = []

        def _recognize(self, vision: Any, foundation: Any, image_path: Path) -> list[Any]:
            self.decoded.append(image_path)
            return []

    bomb = write_png_bomb(data_root / "bomb.png", 20_000, 20_000)
    ocr = RecordingVision()
    result = _run(db, config, bomb, "ocr", _providers(ocr))
    _assert_failed_permanently(result, naming="max_image_pixels")
    assert ocr.decoded == [], "the bomb reached Vision"


# --------------------------------------------------------------------------
# audio and video
# --------------------------------------------------------------------------


def test_audio_decoded_past_the_temp_ceiling_fails_the_task(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    """A minute of audio decodes to about 1.9 MB of 16 kHz WAV; the task
    may hold 256 KB."""
    clip = _media(
        data_root / "memo.m4a", "-f", "lavfi", "-i", "sine=frequency=330:duration=60",
        "-c:a", "aac", "-b:a", "32k",
    )
    limited = _with_limits(config, temp_bytes_per_task=256 * 1024)
    last_error = _assert_failed_permanently(
        _run(db, limited, clip, "transcript", _providers()), naming="temp_bytes_per_task"
    )
    assert "ffmpeg" in last_error


def test_every_decoder_the_pipeline_runs_is_sandboxed(
    db: psycopg.Connection, data_root: Path, config: Config, launched: list[list[str]]
) -> None:
    """PDF text, PDF OCR, audio transcription, and video frame OCR and
    captions: every `pdfinfo`/`pdftotext`/`pdftoppm`/`ffprobe`/`ffmpeg`
    runs under `sandbox-exec` with the no-network profile, confined to a
    work directory under `data_root`. (`file`, the MIME sniffer, is not a
    decoder and is outside this check.)"""
    text_pdf = write_pdf(data_root / "letter.pdf", content=b"BT /F1 12 Tf 72 712 Td (Dear Alice, the kite shop opens at nine and closes at five on weekdays) Tj ET")
    scanned = write_pdf(data_root / "scan.pdf", content=b"BT /F1 12 Tf 72 712 Td (x) Tj ET")
    memo = _media(data_root / "memo.m4a", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac")
    clip = _media(
        data_root / "clip.mp4", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=2", "-pix_fmt", "yuv420p", "-shortest",
    )
    launched.clear()  # only what the pipeline starts from here on
    for path, kind in [
        (text_pdf, "pdf_text"),
        (scanned, "ocr"),
        (memo, "transcript"),
        (clip, "frame_ocr"),
        (clip, "caption"),
        (clip, "transcript"),
    ]:
        outcome, state, _, last_error, _ = _run(db, config, path, kind, _providers())
        assert (outcome, state) == ("done", "done"), (kind, path.name, last_error)

    decoder_runs = [argv for argv in launched if any(Path(a).name in DECODERS for a in argv)]
    assert decoder_runs, "no decoder ran at all"
    work_root = (data_root / "artifacts" / "enrich-work").resolve()
    for argv in decoder_runs:
        assert Path(argv[0]).name == "sandbox-exec", f"ran unsandboxed: {argv}"
        assert argv[1] == "-D" and argv[2].startswith("WORK_DIR="), argv
        assert Path(argv[2].split("=", 1)[1]).parent == work_root, argv[2]
        assert argv[3] == "-p" and "(deny network*)" in argv[4] and "(deny file-write*)" in argv[4]
    # Every decoder was among them, so the check above covered them all.
    assert {Path(a).name for argv in decoder_runs for a in argv} >= DECODERS


# --------------------------------------------------------------------------
# where the work happens
# --------------------------------------------------------------------------


def test_task_work_dirs_live_under_data_root_and_are_removed(
    db: psycopg.Connection, data_root: Path, config: Config
) -> None:
    from imsg.enrich.pipeline import enrich_work_root, sweep_stale_work_dirs

    pdf = write_pdf(data_root / "note.pdf", content=b"BT /F1 12 Tf 72 712 Td (Bob says the boat is ready for Saturday morning at the east dock) Tj ET")
    outcome, *_ = _run(db, config, pdf, "pdf_text", _providers())
    assert outcome == "done"
    root = enrich_work_root(data_root)
    assert root == (data_root / "artifacts" / "enrich-work").resolve()
    assert list(root.iterdir()) == [], "the task's work directory was left behind"

    # A killed worker's directory is removed; a live process's is not.
    dead = subprocess.Popen(["/usr/bin/true"])
    dead.wait()
    stale = root / f"{dead.pid}-12-ocr-abc"
    (stale / "frames").mkdir(parents=True)
    (stale / "frames" / "frame_0001.png").write_bytes(b"x")
    live = root / f"{os.getpid()}-13-ocr-def"
    live.mkdir()
    foreign = root / "not-a-worker"
    foreign.mkdir()
    assert sweep_stale_work_dirs(data_root) == 1
    assert not stale.exists() and live.exists() and foreign.exists()
