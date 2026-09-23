"""Real `ffmpeg`/`ffprobe` audio handling for the transcription path
(SPEC §8 S5b: "ffmpeg -> 16 kHz mono WAV -> mlx-whisper large-v3").
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from imsg.errors import EnrichmentError

WHISPER_SAMPLE_RATE = 16_000


def convert_to_whisper_wav(source_path: Path, output_path: Path, *, timeout_seconds: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                str(WHISPER_SAMPLE_RATE),
                "-f",
                "wav",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnrichmentError(
            f"ffmpeg timed out after {timeout_seconds}s converting '{source_path}'"
        ) from exc
    except OSError as exc:
        raise EnrichmentError(f"ffmpeg could not run: {exc}") from exc
    if proc.returncode != 0:
        raise EnrichmentError(
            f"ffmpeg failed converting '{source_path}': {proc.stderr.strip()[-500:]}"
        )
    if not output_path.is_file():
        raise EnrichmentError(f"ffmpeg reported success but produced no output for '{source_path}'")
    return output_path


def probe_duration_seconds(path: Path, *, timeout_seconds: int) -> float:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnrichmentError(f"ffprobe timed out after {timeout_seconds}s on '{path}'") from exc
    except OSError as exc:
        raise EnrichmentError(f"ffprobe could not run: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout.strip():
        raise EnrichmentError(f"ffprobe failed on '{path}': {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise EnrichmentError(
            f"ffprobe returned a non-numeric duration for '{path}': {proc.stdout!r}"
        ) from exc


def has_stream(path: Path, stream_type: str, *, timeout_seconds: int) -> bool:
    """Whether `path` holds at least one stream of `stream_type` (`a` for
    audio, `v` for video), by `ffprobe`. `file` calls an audio-only MP4
    `video/mp4`, and a screen recording or a muted clip has no audio
    track; ffmpeg exits 234 on either rather than producing nothing, so
    without this check such a task burns its whole retry budget and ends
    `failed` when the honest outcome is `skipped`."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                stream_type,
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnrichmentError(f"ffprobe timed out after {timeout_seconds}s on '{path}'") from exc
    except OSError as exc:
        raise EnrichmentError(f"ffprobe could not run: {exc}") from exc
    if proc.returncode != 0:
        raise EnrichmentError(f"ffprobe failed on '{path}': {proc.stderr.strip()}")
    return any(line.strip() for line in proc.stdout.splitlines())


__all__ = ["WHISPER_SAMPLE_RATE", "convert_to_whisper_wav", "has_stream", "probe_duration_seconds"]
