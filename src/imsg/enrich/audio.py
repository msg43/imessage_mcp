"""Real `ffmpeg`/`ffprobe` audio handling for the transcription path
(SPEC §8 S5b: "ffmpeg -> 16 kHz mono WAV -> mlx-whisper large-v3").

Both run sandboxed and inside the task's budget
(`imsg.enrich.sandboxed_decoder`, 2026-09-24): no network, writes only
in the task's work directory, and the WAV they write counts against
`enrichment.limits.temp_bytes_per_task` while it is written.
"""

from __future__ import annotations

import os
from pathlib import Path

from imsg.enrich.sandboxed_decoder import DecoderBudget, run_decoder
from imsg.errors import EnrichmentError

WHISPER_SAMPLE_RATE = 16_000

MAX_PROBE_OUTPUT_BYTES = 1024 * 1024
"""`ffprobe` prints a number or a stream list here; a megabyte is far
more than either."""


def convert_to_whisper_wav(
    source_path: Path,
    output_path: Path,
    *,
    budget: DecoderBudget,
    max_seconds: int | None = None,
) -> Path:
    """Decode `source_path` to a 16 kHz mono WAV at `output_path`, which
    must be inside the budget's work directory. `max_seconds` stops the
    output at that duration, so a container that under-reports its length
    cannot produce more audio than `enrichment.limits.max_media_seconds`
    allows."""
    if not budget.contains(output_path):
        raise EnrichmentError(
            f"ffmpeg output '{output_path}' is outside the task's work directory "
            f"'{budget.work_dir}', the only place a decoder may write"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        os.fspath(source_path.absolute()),
    ]
    if max_seconds is not None:
        argv += ["-t", str(max_seconds)]
    argv += [
        "-ac",
        "1",
        "-ar",
        str(WHISPER_SAMPLE_RATE),
        "-f",
        "wav",
        os.fspath(output_path.absolute()),
    ]
    run = run_decoder(argv, budget=budget, name="ffmpeg", subject=source_path)
    if run.returncode != 0:
        raise EnrichmentError(f"ffmpeg failed converting '{source_path}': {run.stderr_tail()}")
    if not output_path.is_file():
        raise EnrichmentError(f"ffmpeg reported success but produced no output for '{source_path}'")
    return output_path


def _ffprobe(path: Path, args: list[str], *, budget: DecoderBudget) -> str:
    run = run_decoder(
        ["ffprobe", "-v", "error", *args, os.fspath(path.absolute())],
        budget=budget,
        name="ffprobe",
        subject=path,
        stdout_name="ffprobe.txt",
        max_stdout_bytes=MAX_PROBE_OUTPUT_BYTES,
    )
    if run.returncode != 0:
        raise EnrichmentError(f"ffprobe failed on '{path}': {run.stderr_tail()}")
    assert run.stdout_path is not None
    return run.stdout_path.read_bytes().decode("utf-8", errors="replace")


def probe_duration_seconds(path: Path, *, budget: DecoderBudget) -> float:
    out = _ffprobe(
        path,
        ["-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1"],
        budget=budget,
    ).strip()
    if not out:
        raise EnrichmentError(f"ffprobe reported no duration for '{path}'")
    try:
        return float(out)
    except ValueError as exc:
        raise EnrichmentError(
            f"ffprobe returned a non-numeric duration for '{path}': {out[:200]!r}"
        ) from exc


def has_stream(path: Path, stream_type: str, *, budget: DecoderBudget) -> bool:
    """Whether `path` holds at least one stream of `stream_type` (`a` for
    audio, `v` for video), by `ffprobe`. `file` calls an audio-only MP4
    `video/mp4`, and a screen recording or a muted clip has no audio
    track; ffmpeg exits 234 on either rather than producing nothing, so
    without this check such a task burns its whole retry budget and ends
    `failed` when the honest outcome is `skipped`."""
    out = _ffprobe(
        path,
        ["-select_streams", stream_type, "-show_entries", "stream=index", "-of", "csv=p=0"],
        budget=budget,
    )
    return any(line.strip() for line in out.splitlines())


__all__ = [
    "MAX_PROBE_OUTPUT_BYTES",
    "WHISPER_SAMPLE_RATE",
    "convert_to_whisper_wav",
    "has_stream",
    "probe_duration_seconds",
]
