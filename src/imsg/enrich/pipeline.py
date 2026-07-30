"""S5b Postgres + filesystem orchestration (SPEC §8 S5b): claims tasks
from the enrichment queue, sniffs MIME (never trusts the extension),
enforces the untrusted-attachment resource ceilings, dispatches to the
real subprocess tooling (`pdftotext`/`pdftoppm`/`ffmpeg`) plus the
model-backed providers (OCR/caption/transcription), writes
`attachment_chunk` rows, re-renders every current parent segment
(`imsg.segment.pipeline.refresh_segment_rendering`), and emits the FTS
outbox events for both segment and attachment_chunk content.

Takes an already-open `psycopg.Connection`, never owns its lifecycle.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imsg.config.schema import Config
from imsg.enrich import audio, mime, pdf_render, pdf_text, video
from imsg.enrich.chunking import chunk_text
from imsg.enrich.limits import check_file_size, check_media_duration, check_pdf_page_count
from imsg.enrich.provider import CaptionProvider, OcrProvider, TranscriptionProvider
from imsg.enrich.queue import (
    EnrichmentTask,
    complete_task,
    enqueue,
    fail_task,
    fail_task_permanently,
    skip_task,
)
from imsg.enrich.router import PDF_MIME
from imsg.errors import EnrichmentError, UnsupportedEnrichmentTypeError, UntrustedAttachmentError
from imsg.hashing import sha256_text
from imsg.paths import is_contained_in, resolve_path
from imsg.segment.pipeline import find_segment_ids_for_attachment, refresh_segment_rendering
from imsg.tokens import estimate_tokens

if TYPE_CHECKING:
    import psycopg


@dataclass(frozen=True, slots=True)
class EnrichmentProviders:
    ocr: OcrProvider
    caption: CaptionProvider
    transcription: TranscriptionProvider


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    model: str
    model_version: str | None
    text: str
    detail: dict[str, object] | None = None
    follow_up_kinds: tuple[str, ...] = ()
    """Additional `enrichment_kind`s to enqueue as a result of this
    one (SPEC §8 S5b: a scanned PDF's `pdf_text` run also enqueues
    `ocr`)."""


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    attachment_id: int
    attachment_key: str
    cache_path: str | None
    state: str


def _fetch_attachment(conn: psycopg.Connection, attachment_id: int) -> AttachmentRecord:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT attachment_id, attachment_key, cache_path, state FROM attachment "
            "WHERE attachment_id = %s",
            (attachment_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise EnrichmentError(f"attachment_id {attachment_id} not found")
    return AttachmentRecord(*row)


def _persist_sniffed_mime_type(conn: psycopg.Connection, attachment_id: int, mime_type: str) -> None:
    """Content-sniffed MIME (D6: "the extension is not trusted") is the
    most trustworthy value this pipeline ever computes for
    `attachment.mime_type` — persist it so S4's rendering
    (`imsg.segment.pipeline._classify_attachment_kind`) can tell a PDF
    from an image from a generic file, instead of relying on whatever
    S2 guessed from `chat.db`/extension at extraction time. Runs
    autocommit-adjacent to the main task transaction (a stale
    mime_type is a rendering cosmetic issue, not a correctness one, so
    this is deliberately not rolled back if the dispatch step below
    fails).
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE attachment SET mime_type = %s, updated_at = now() "
            "WHERE attachment_id = %s AND mime_type IS DISTINCT FROM %s",
            (mime_type, attachment_id, mime_type),
        )


# --------------------------------------------------------------------------
# per-kind dispatch — each enforces its own limits, then runs real tooling
# and/or a model-backed provider
# --------------------------------------------------------------------------


def _run_pdf_text(cache_path: Path, config: Config) -> EnrichmentResult:
    limits_cfg = config.enrichment.limits
    check_file_size(cache_path, max_bytes=limits_cfg.max_file_bytes)
    result = pdf_text.extract_pdf_text(cache_path, timeout_seconds=limits_cfg.task_timeout_seconds)
    check_pdf_page_count(result.page_count, max_pages=limits_cfg.max_pdf_pages)
    scanned = pdf_text.is_scanned(
        result, threshold_chars_per_page=config.enrichment.pdf_scanned_threshold_chars_per_page
    )
    return EnrichmentResult(
        model="pdftotext",
        model_version=None,
        text=result.full_text,
        detail={
            "page_count": result.page_count,
            "avg_chars_per_page": result.avg_chars_per_page,
            "scanned": scanned,
        },
        follow_up_kinds=("ocr",) if scanned else (),
    )


def _run_ocr(
    cache_path: Path,
    mime_type: str,
    config: Config,
    providers: EnrichmentProviders,
    work_dir: Path,
) -> EnrichmentResult:
    limits_cfg = config.enrichment.limits
    check_file_size(cache_path, max_bytes=limits_cfg.max_file_bytes)
    if mime_type == PDF_MIME:
        pages = pdf_render.render_pdf_pages_to_png(
            cache_path, work_dir, timeout_seconds=limits_cfg.task_timeout_seconds
        )
        check_pdf_page_count(len(pages), max_pages=limits_cfg.max_pdf_pages)
        texts = [providers.ocr.recognize_text(p) for p in pages]
        return EnrichmentResult(
            model=providers.ocr.model_id,
            model_version=None,
            text="\n\n".join(texts),
            detail={"page_count": len(pages)},
        )
    if mime_type.startswith("image/"):
        return EnrichmentResult(
            model=providers.ocr.model_id,
            model_version=None,
            text=providers.ocr.recognize_text(cache_path),
        )
    raise UnsupportedEnrichmentTypeError(
        f"'ocr' has no route for mime type {mime_type!r} (only PDF and image/*)"
    )


def _run_caption_image(
    cache_path: Path, config: Config, providers: EnrichmentProviders
) -> EnrichmentResult:
    check_file_size(cache_path, max_bytes=config.enrichment.limits.max_file_bytes)
    return EnrichmentResult(
        model=providers.caption.model_id,
        model_version=None,
        text=providers.caption.caption(cache_path),
    )


def frames_dir_for_attachment(data_root: Path, attachment_id: int) -> Path:
    """SPEC §5.3 filesystem layout: `artifacts/` holds "OCR page images,
    sampled video frames (transient, 30-day GC)". Sampled keyframes must
    live *here*, never in a task-scoped temp dir that vanishes the
    moment `process_one_task` returns — S6's multimodal embedding
    (`imsg.embed.pipeline`, D3a) reads these same frame files back
    later, as a logically separate pipeline run."""
    return data_root / "artifacts" / "frames" / str(attachment_id)


def _sample_and_run(
    cache_path: Path,
    config: Config,
    frames_dir: Path,
    per_frame_fn: Callable[[Path], str],
) -> tuple[list[dict[str, object]], float]:
    limits_cfg = config.enrichment.limits
    check_file_size(cache_path, max_bytes=limits_cfg.max_file_bytes)
    duration = audio.probe_duration_seconds(cache_path, timeout_seconds=limits_cfg.task_timeout_seconds)
    check_media_duration(duration, max_seconds=limits_cfg.max_media_seconds)
    frames = video.sample_keyframes(
        cache_path,
        frames_dir,
        max_frames=config.enrichment.video_max_frames,
        timeout_seconds=limits_cfg.task_timeout_seconds,
    )
    per_frame: list[dict[str, object]] = [
        {"timestamp_seconds": f.timestamp_seconds, "text": per_frame_fn(f.path), "path": str(f.path)}
        for f in frames
    ]
    return per_frame, duration


def _run_frame_ocr(
    cache_path: Path, config: Config, providers: EnrichmentProviders, attachment_id: int
) -> EnrichmentResult:
    frames_dir = frames_dir_for_attachment(config.paths.data_root, attachment_id)
    per_frame, duration = _sample_and_run(
        cache_path, config, frames_dir, providers.ocr.recognize_text
    )
    text = "\n\n".join(f"[{f['timestamp_seconds']:.1f}s] {f['text']}" for f in per_frame)
    return EnrichmentResult(
        model=providers.ocr.model_id,
        model_version=None,
        text=text,
        detail={"frames": per_frame, "duration_seconds": duration},
    )


def _run_caption_video(
    cache_path: Path, config: Config, providers: EnrichmentProviders, attachment_id: int
) -> EnrichmentResult:
    frames_dir = frames_dir_for_attachment(config.paths.data_root, attachment_id)
    per_frame, duration = _sample_and_run(cache_path, config, frames_dir, providers.caption.caption)
    text = "\n\n".join(f"[{f['timestamp_seconds']:.1f}s] {f['text']}" for f in per_frame)
    return EnrichmentResult(
        model=providers.caption.model_id,
        model_version=None,
        text=text,
        detail={"frames": per_frame, "duration_seconds": duration},
    )


def _run_transcript(
    cache_path: Path, config: Config, providers: EnrichmentProviders, work_dir: Path
) -> EnrichmentResult:
    """Shared by audio *and* video attachments — `ffmpeg` extracts the
    audio track from a video input exactly the same way it normalizes a
    standalone audio file, so no branching on mime type is needed here."""
    limits_cfg = config.enrichment.limits
    check_file_size(cache_path, max_bytes=limits_cfg.max_file_bytes)
    duration = audio.probe_duration_seconds(cache_path, timeout_seconds=limits_cfg.task_timeout_seconds)
    check_media_duration(duration, max_seconds=limits_cfg.max_media_seconds)
    wav_path = work_dir / "audio.wav"
    audio.convert_to_whisper_wav(cache_path, wav_path, timeout_seconds=limits_cfg.task_timeout_seconds)
    return EnrichmentResult(
        model=providers.transcription.model_id,
        model_version=None,
        text=providers.transcription.transcribe(wav_path),
        detail={"duration_seconds": duration},
    )


def _dispatch(
    kind: str,
    cache_path: Path,
    mime_type: str,
    config: Config,
    providers: EnrichmentProviders,
    work_dir: Path,
    attachment_id: int,
) -> EnrichmentResult:
    if kind == "pdf_text":
        return _run_pdf_text(cache_path, config)
    if kind == "ocr":
        return _run_ocr(cache_path, mime_type, config, providers, work_dir)
    if kind == "caption":
        if mime_type.startswith("video/"):
            return _run_caption_video(cache_path, config, providers, attachment_id)
        if mime_type.startswith("image/"):
            return _run_caption_image(cache_path, config, providers)
        raise UnsupportedEnrichmentTypeError(
            f"'caption' has no route for mime type {mime_type!r}"
        )
    if kind == "transcript":
        if mime_type.startswith("audio/") or mime_type.startswith("video/"):
            return _run_transcript(cache_path, config, providers, work_dir)
        raise UnsupportedEnrichmentTypeError(
            f"'transcript' has no route for mime type {mime_type!r}"
        )
    if kind == "frame_ocr":
        if mime_type.startswith("video/"):
            return _run_frame_ocr(cache_path, config, providers, attachment_id)
        raise UnsupportedEnrichmentTypeError(f"'frame_ocr' has no route for mime type {mime_type!r}")
    raise EnrichmentError(f"no dispatch handler for enrichment kind {kind!r}")


# --------------------------------------------------------------------------
# post-success bookkeeping: chunks, outbox events, parent-segment refresh
# --------------------------------------------------------------------------


def _replace_chunks_and_emit_events(
    conn: psycopg.Connection, attachment_id: int, kind: str, text: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id FROM attachment_chunk WHERE attachment_id = %s AND kind = %s",
            (attachment_id, kind),
        )
        old_chunk_ids = [row[0] for row in cur.fetchall()]
        if old_chunk_ids:
            cur.executemany(
                "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
                "VALUES ('attachment_chunk', %s, 'delete', NULL)",
                [(cid,) for cid in old_chunk_ids],
            )
            cur.execute(
                "DELETE FROM attachment_chunk WHERE attachment_id = %s AND kind = %s",
                (attachment_id, kind),
            )

        for seq, chunk in enumerate(chunk_text(text)):
            cur.execute(
                "INSERT INTO attachment_chunk (attachment_id, kind, seq, text, token_count) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING chunk_id",
                (attachment_id, kind, seq, chunk, estimate_tokens(chunk)),
            )
            row = cur.fetchone()
            if row is None:  # pragma: no cover - INSERT ... RETURNING always returns a row
                raise EnrichmentError("attachment_chunk insert did not return a chunk_id")
            cur.execute(
                "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
                "VALUES ('attachment_chunk', %s, 'upsert', %s)",
                (row[0], sha256_text(chunk)),
            )


def _refresh_parent_segments(conn: psycopg.Connection, attachment_id: int, config: Config) -> int:
    segment_ids = find_segment_ids_for_attachment(conn, attachment_id)
    for segment_id in segment_ids:
        _, rendered_sha = refresh_segment_rendering(conn, segment_id, config)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO search_index_event (entity_kind, entity_id, operation, content_sha256) "
                "VALUES ('segment', %s, 'upsert', %s)",
                (segment_id, rendered_sha),
            )
    return len(segment_ids)


# --------------------------------------------------------------------------
# top-level: process exactly one already-claimed task
# --------------------------------------------------------------------------


def process_one_task(
    conn: psycopg.Connection,
    config: Config,
    providers: EnrichmentProviders,
    task: EnrichmentTask,
    *,
    mime_sniffer: mime.MimeSnifferFn = mime.real_sniff_mime,
) -> str:
    """Process one already-claimed task end-to-end. Returns
    `'done' | 'skipped' | 'retry' | 'failed'` — the caller
    (`imsg enrich`'s worker loop, later CLI wiring) uses this only for
    logging/metrics; the queue table is already the durable record.
    """
    try:
        record = _fetch_attachment(conn, task.attachment_id)
        if record.state != "materialized" or not record.cache_path:
            raise EnrichmentError(
                f"attachment {task.attachment_id} is not materialized yet "
                f"(state={record.state!r}) — S5a must run first; will retry"
            )
        cache_path = resolve_path(Path(record.cache_path))
        if not is_contained_in(cache_path, resolve_path(config.paths.data_root)):
            raise UntrustedAttachmentError(
                f"attachment {task.attachment_id}'s cache_path does not resolve under data_root "
                f"— refusing to read it"
            )
        mime_type = mime_sniffer(cache_path)
        _persist_sniffed_mime_type(conn, task.attachment_id, mime_type)

        with tempfile.TemporaryDirectory(
            prefix=f"imsg-enrich-{task.attachment_id}-{task.kind}-"
        ) as tmp:
            result = _dispatch(
                task.kind, cache_path, mime_type, config, providers, Path(tmp), task.attachment_id
            )

            with conn.transaction():
                complete_task(
                    conn,
                    task.attachment_id,
                    task.kind,
                    model=result.model,
                    model_version=result.model_version,
                    text=result.text,
                    detail=result.detail,
                )
                _replace_chunks_and_emit_events(conn, task.attachment_id, task.kind, result.text)
                if result.follow_up_kinds:
                    enqueue(conn, task.attachment_id, result.follow_up_kinds)
                _refresh_parent_segments(conn, task.attachment_id, config)
        return "done"

    except UnsupportedEnrichmentTypeError as exc:
        with conn.transaction():
            skip_task(conn, task.attachment_id, task.kind, reason=str(exc))
        return "skipped"
    except UntrustedAttachmentError as exc:
        with conn.transaction():
            fail_task_permanently(conn, task.attachment_id, task.kind, error=str(exc))
        return "failed"
    except EnrichmentError as exc:
        with conn.transaction():
            became_permanent = fail_task(
                conn,
                task.attachment_id,
                task.kind,
                error=str(exc),
                max_attempts=config.enrichment.max_attempts,
            )
        return "failed" if became_permanent else "retry"


__all__ = [
    "AttachmentRecord",
    "EnrichmentProviders",
    "EnrichmentResult",
    "frames_dir_for_attachment",
    "process_one_task",
]
