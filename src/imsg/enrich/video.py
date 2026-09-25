"""Real `ffmpeg` scene-change keyframe sampling for video enrichment
(SPEC §8 S5b: "ffmpeg scene-change sampling (gt(scene,0.3), cap
video_max_frames)"). Frame timestamps come from the `showinfo` filter's
stderr output, matched positionally to the numbered PNG files ffmpeg
writes for the frames that passed the scene-change filter.

The first frame is always kept. A scene-change filter alone keeps
nothing from a clip with no hard cut — one continuous handheld shot, the
usual home video — and frame OCR and the video caption then finish
`done` with empty text, which reads as success (checked 2026-09-23 on a
continuous synthetic 1080p clip: 0 frames without the first-frame term,
1 with it).

**Sandboxed (2026-09-24).** `ffmpeg` runs inside the task's budget
(`imsg.enrich.sandboxed_decoder`) and may write only into the task's
work directory, so it samples into `<work dir>/frames`, and the frames
are moved into `output_dir` (the attachment's durable frames directory,
which S6 reads back) once it has exited. The frames an earlier run left
in `output_dir` are removed first, so a re-run that keeps fewer frames
leaves no stale ones behind.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from imsg.enrich.sandboxed_decoder import DecoderBudget, run_decoder
from imsg.errors import EnrichmentError

SCENE_CHANGE_THRESHOLD = 0.3
_PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")
_FRAME_GLOB = "frame_*.png"


@dataclass(frozen=True, slots=True)
class VideoFrame:
    path: Path
    timestamp_seconds: float


def sample_keyframes(
    video_path: Path, output_dir: Path, *, max_frames: int, budget: DecoderBudget
) -> list[VideoFrame]:
    staging = budget.work_dir / "frames"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    run = run_decoder(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-y",
            "-i",
            os.fspath(video_path.absolute()),
            "-vf",
            f"select='eq(n,0)+gt(scene,{SCENE_CHANGE_THRESHOLD})',showinfo",
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(max_frames),
            os.fspath(staging / "frame_%04d.png"),
        ],
        budget=budget,
        name="ffmpeg",
        subject=video_path,
    )
    if run.returncode != 0:
        raise EnrichmentError(f"ffmpeg failed sampling '{video_path}': {run.stderr_tail()}")

    timestamps = [
        float(match.group(1))
        for line in run.stderr_lines()
        for match in _PTS_TIME_RE.finditer(line)
    ]
    staged = sorted(staging.glob(_FRAME_GLOB))
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob(_FRAME_GLOB):
        stale.unlink()
    frames: list[VideoFrame] = []
    for i, frame in enumerate(staged):
        kept = output_dir / frame.name
        shutil.move(frame, kept)
        frames.append(
            VideoFrame(
                path=kept,
                timestamp_seconds=(timestamps[i] if i < len(timestamps) else float("nan")),
            )
        )
    return frames


__all__ = ["SCENE_CHANGE_THRESHOLD", "VideoFrame", "sample_keyframes"]
