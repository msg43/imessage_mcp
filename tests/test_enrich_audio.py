"""Real `ffmpeg`/`ffprobe` audio handling (SPEC §8 S5b: "ffmpeg -> 16
kHz mono WAV"). Builds a real stereo 44.1kHz WAV with Python's stdlib
`wave` module (no fixtures/mocking of the subprocess boundary) and
converts it for real, sandboxed inside a task budget
(`imsg.enrich.sandboxed_decoder`)."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from imsg.enrich.audio import WHISPER_SAMPLE_RATE, convert_to_whisper_wav, probe_duration_seconds
from imsg.enrich.sandboxed_decoder import DecoderBudget
from imsg.errors import EnrichmentError


def _budget(tmp_path: Path) -> DecoderBudget:
    return DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)


def _write_stereo_wav(path: Path, *, seconds: float, sample_rate: int = 44_100) -> None:
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<hh", (i % 1000) - 500, (i % 700) - 350) for i in range(n_frames)
        )
        w.writeframes(frames)


def test_convert_produces_16khz_mono_wav(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_stereo_wav(source, seconds=1.0)
    budget = _budget(tmp_path)
    output = budget.work_dir / "out.wav"

    convert_to_whisper_wav(source, output, budget=budget)

    assert output.is_file()
    with wave.open(str(output), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == WHISPER_SAMPLE_RATE


def test_probe_duration_matches_written_length(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_stereo_wav(source, seconds=2.0)
    duration = probe_duration_seconds(source, budget=_budget(tmp_path))
    assert 1.9 < duration < 2.1


def test_convert_nonexistent_source_raises(tmp_path: Path) -> None:
    with pytest.raises(EnrichmentError):
        budget = _budget(tmp_path)
        convert_to_whisper_wav(tmp_path / "missing.wav", budget.work_dir / "out.wav", budget=budget)


def test_probe_nonexistent_source_raises(tmp_path: Path) -> None:
    with pytest.raises(EnrichmentError):
        probe_duration_seconds(tmp_path / "missing.wav", budget=_budget(tmp_path))
