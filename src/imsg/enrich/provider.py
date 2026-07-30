"""Model-backed enrichment providers (SPEC §4.1, §8 S5b): OCR,
captioning, and transcription are the genuine model-inference steps in
S5b (Apple Vision, the local captioning VLM, Whisper). Everything else
in this package — `pdftotext`, `pdftoppm`, `ffmpeg` audio/video
handling — is ordinary deterministic subprocess tooling, not a model,
and is implemented for real rather than behind a provider interface.

No model weights exist in this build environment, and there is no
corpus to run them against. `Fake*Provider` implementations are
deterministic stand-ins used by every test in this build; the real
Vision/MLX-backed implementations are Phase 3/5 work and drop in
behind the same Protocols — nothing above this layer should need to
change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class OcrProvider(Protocol):
    model_id: str

    def recognize_text(self, image_path: Path) -> str:
        """Recognize visible text in an image (SPEC §4.1: Apple Vision
        `VNRecognizeTextRequest`, accurate mode)."""
        ...


class CaptionProvider(Protocol):
    model_id: str

    def caption(self, image_path: Path) -> str:
        """Describe an image's content (SPEC §4.1: local VLM, fixed
        prompt `prompts/caption.txt`, temperature 0)."""
        ...


class TranscriptionProvider(Protocol):
    model_id: str

    def transcribe(self, audio_wav_path: Path) -> str:
        """Transcribe a 16kHz mono WAV (SPEC §4.1: Whisper large-v3 via
        mlx-whisper)."""
        ...


def _deterministic_text(prefix: str, path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"[fake {prefix} of {path.name}, content-hash {digest}]"


class FakeOcrProvider:
    model_id = "fake/ocr@test"

    def recognize_text(self, image_path: Path) -> str:
        return _deterministic_text("ocr text", image_path)


class FakeCaptionProvider:
    model_id = "fake/caption@test"

    def caption(self, image_path: Path) -> str:
        return _deterministic_text("caption", image_path)


class FakeTranscriptionProvider:
    model_id = "fake/whisper@test"

    def transcribe(self, audio_wav_path: Path) -> str:
        return _deterministic_text("transcript", audio_wav_path)


__all__ = [
    "CaptionProvider",
    "FakeCaptionProvider",
    "FakeOcrProvider",
    "FakeTranscriptionProvider",
    "OcrProvider",
    "TranscriptionProvider",
]
