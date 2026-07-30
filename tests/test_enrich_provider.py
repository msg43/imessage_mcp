"""Fake model-backed enrichment providers (SPEC §4.1, §8 S5b) —
deterministic stand-ins for OCR/caption/transcription; no model weights
in this build environment."""

from __future__ import annotations

from pathlib import Path

from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider


def test_fake_ocr_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "image.bin"
    f.write_bytes(b"same content")
    provider = FakeOcrProvider()
    assert provider.recognize_text(f) == provider.recognize_text(f)


def test_fake_ocr_differs_by_content(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"content a")
    b.write_bytes(b"content b")
    provider = FakeOcrProvider()
    assert provider.recognize_text(a) != provider.recognize_text(b)


def test_fake_caption_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "image.bin"
    f.write_bytes(b"a photo")
    provider = FakeCaptionProvider()
    assert provider.caption(f) == provider.caption(f)


def test_fake_transcription_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "audio.wav"
    f.write_bytes(b"pretend audio bytes")
    provider = FakeTranscriptionProvider()
    assert provider.transcribe(f) == provider.transcribe(f)


def test_providers_have_stable_model_ids() -> None:
    assert FakeOcrProvider.model_id
    assert FakeCaptionProvider.model_id
    assert FakeTranscriptionProvider.model_id
