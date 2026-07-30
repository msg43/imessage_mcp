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


__all__ = ["WHISPER_SAMPLE_RATE", "convert_to_whisper_wav", "probe_duration_seconds"]
