"""Real `ffmpeg` scene-change keyframe sampling (SPEC §8 S5b). Builds a
real two-color-block video with `ffmpeg`'s `lavfi` synthetic source
(no external fixture files, no mocking of the subprocess boundary)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from imsg.enrich.sandboxed_decoder import DecoderBudget
from imsg.enrich.video import sample_keyframes
from imsg.errors import EnrichmentError


def _budget(tmp_path: Path) -> DecoderBudget:
    return DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)


def _write_two_scene_video(path: Path) -> None:
    """One second of solid red, then one second of solid blue — a
    single, unambiguous scene change at t=1s."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:d=1,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=1,format=yuv420p",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,fps=5",
            "-t",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )


def test_scene_change_produces_at_least_one_frame(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_two_scene_video(video)
    out_dir = tmp_path / "frames"

    frames = sample_keyframes(video, out_dir, max_frames=20, budget=_budget(tmp_path))

    assert len(frames) >= 1
    for f in frames:
        assert f.path.is_file()
        assert f.path.stat().st_size > 0


def test_frame_count_never_exceeds_max_frames(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_two_scene_video(video)
    out_dir = tmp_path / "frames"

    frames = sample_keyframes(video, out_dir, max_frames=1, budget=_budget(tmp_path))
    assert len(frames) <= 1


def test_nonexistent_video_raises(tmp_path: Path) -> None:
    with pytest.raises(EnrichmentError):
        sample_keyframes(tmp_path / "missing.mp4", tmp_path / "out", max_frames=5, budget=_budget(tmp_path))


def test_a_clip_with_no_scene_change_still_yields_its_first_frame(tmp_path: Path) -> None:
    """One continuous shot, the usual home video: a scene-change filter
    alone keeps no frame at all, and frame OCR and the video caption
    would then finish `done` with empty text."""
    video = tmp_path / "continuous.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10",
            "-t", "2", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    frames = sample_keyframes(video, tmp_path / "frames", max_frames=20, budget=_budget(tmp_path))

    assert len(frames) >= 1
    assert frames[0].timestamp_seconds == 0.0
