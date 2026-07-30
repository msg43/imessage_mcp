"""Real `ffmpeg` scene-change keyframe sampling for video enrichment
(SPEC §8 S5b: "ffmpeg scene-change sampling (gt(scene,0.3), cap
video_max_frames)"). Frame timestamps come from the `showinfo` filter's
stderr output, matched positionally to the numbered PNG files ffmpeg
writes for the frames that passed the scene-change filter.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from imsg.errors import EnrichmentError

SCENE_CHANGE_THRESHOLD = 0.3
_PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    path: Path
    timestamp_seconds: float


def sample_keyframes(
    video_path: Path, output_dir: Path, *, max_frames: int, timeout_seconds: int
) -> list[VideoFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%04d.png"
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"select='gt(scene,{SCENE_CHANGE_THRESHOLD})',showinfo",
                "-vsync",
                "vfr",
                "-frames:v",
                str(max_frames),
                str(pattern),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnrichmentError(
            f"ffmpeg timed out after {timeout_seconds}s sampling '{video_path}'"
        ) from exc
    except OSError as exc:
        raise EnrichmentError(f"ffmpeg could not run: {exc}") from exc
    if proc.returncode != 0:
        raise EnrichmentError(
            f"ffmpeg failed sampling '{video_path}': {proc.stderr.strip()[-500:]}"
        )

    timestamps = [float(m.group(1)) for m in _PTS_TIME_RE.finditer(proc.stderr)]
    frame_paths = sorted(output_dir.glob("frame_*.png"))

    return [
        VideoFrame(path=path, timestamp_seconds=(timestamps[i] if i < len(timestamps) else float("nan")))
        for i, path in enumerate(frame_paths)
    ]


__all__ = ["SCENE_CHANGE_THRESHOLD", "VideoFrame", "sample_keyframes"]
