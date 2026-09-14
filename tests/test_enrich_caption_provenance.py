"""Caption provenance (SPEC §4.1: a fixed prompt is part of what produced
a caption). `MlxVlmCaptionProvider` exposes `prompt_sha256` for exactly
this, but until 2026-09-14 nothing recorded it — `enrichment.detail`
now carries it, next to `model`, whenever the provider has one. The
deterministic fakes have no prompt, so their captions record nothing
extra. No database: `_run_caption_image` is the pure dispatch step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from imsg.config.loader import load_config_dict
from imsg.enrich.pipeline import EnrichmentProviders, _caption_provenance, _run_caption_image
from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider

pytestmark = pytest.mark.usefixtures("messages_dir")

PROMPT_SHA = "ab" * 32


class _PromptedCaption:
    """The shape of the real provider: a model id plus the prompt hash."""

    model_id = "example-org/vlm@rev1"
    prompt_sha256 = PROMPT_SHA

    def caption(self, image_path: Path) -> str:
        return "a caption"


def _providers(caption: Any) -> EnrichmentProviders:
    return EnrichmentProviders(
        ocr=FakeOcrProvider(), caption=caption, transcription=FakeTranscriptionProvider()
    )


def test_caption_detail_records_the_prompt_sha256(config_dict_factory: Any, tmp_path: Path) -> None:
    cfg = load_config_dict(config_dict_factory())
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-ish bytes")

    result = _run_caption_image(image, cfg, _providers(_PromptedCaption()))

    assert result.model == "example-org/vlm@rev1"
    assert result.text == "a caption"
    assert result.detail == {"prompt_sha256": PROMPT_SHA}


def test_fake_caption_provider_has_no_prompt_to_record(config_dict_factory: Any, tmp_path: Path) -> None:
    cfg = load_config_dict(config_dict_factory())
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-ish bytes")

    result = _run_caption_image(image, cfg, _providers(FakeCaptionProvider()))

    assert result.detail is None


def test_video_captions_carry_the_same_provenance_key() -> None:
    """`_run_caption_video` merges the same mapping into its frames detail."""
    assert _caption_provenance(_providers(_PromptedCaption())) == {"prompt_sha256": PROMPT_SHA}
    assert _caption_provenance(_providers(FakeCaptionProvider())) == {}
