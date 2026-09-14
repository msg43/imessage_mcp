"""Apple Vision OCR provider (`imsg.enrich.vision_ocr`), driven through
a fake `Vision`/`Foundation` pair stood in `sys.modules` — no pyobjc
framework, no macOS, no image decoding involved."""

from __future__ import annotations

import platform
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _model_runtime_stubs import block_module
from imsg.enrich.model_runtime import ModelRuntimeUnavailableError
from imsg.enrich.provider import OcrProvider
from imsg.enrich.vision_ocr import AppleVisionOcrProvider, RecognizedLine, order_lines
from imsg.errors import EnrichmentError

ACCURATE_LEVEL = object()  # the stub's VNRequestTextRecognitionLevelAccurate constant


def _box(x: float, y: float, width: float, height: float) -> Any:
    """The CGRect shape pyobjc hands back from `boundingBox()`."""
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y), size=SimpleNamespace(width=width, height=height)
    )


class _Candidate:
    def __init__(self, text: str) -> None:
        self._text = text

    def string(self) -> str:
        return self._text


class _Observation:
    """A `VNRecognizedTextObservation`; `text=None` models an observation
    with no candidates at all."""

    def __init__(self, text: str | None, box: Any) -> None:
        self._text = text
        self._box = box

    def topCandidates_(self, count: int) -> list[_Candidate]:
        return [] if self._text is None else [_Candidate(self._text)][:count]

    def boundingBox(self) -> Any:
        return self._box


@dataclass
class VisionStub:
    """What the fake framework will do, and what it saw."""

    observations: list[_Observation] | None = field(default_factory=list)
    perform_result: tuple[bool, Any] = (True, None)
    perform_exception: Exception | None = None
    requests: list[Any] = field(default_factory=list)
    handlers: list[Any] = field(default_factory=list)
    file_url_paths: list[str] = field(default_factory=list)


@pytest.fixture
def vision(monkeypatch: pytest.MonkeyPatch) -> VisionStub:
    stub = VisionStub()

    class Request:
        def __init__(self) -> None:
            self.level: Any = None
            self.language_correction: bool | None = None
            self.languages: list[str] | None = None
            self.auto_detect: bool | None = None
            self.minimum_text_height: float | None = None
            self.recognized: list[_Observation] | None = None

        @classmethod
        def alloc(cls) -> Request:
            return cls()

        def init(self) -> Request:
            stub.requests.append(self)
            return self

        def setRecognitionLevel_(self, level: Any) -> None:
            self.level = level

        def setUsesLanguageCorrection_(self, flag: bool) -> None:
            self.language_correction = flag

        def setRecognitionLanguages_(self, languages: list[str]) -> None:
            self.languages = list(languages)

        def setAutomaticallyDetectsLanguage_(self, flag: bool) -> None:
            self.auto_detect = flag

        def setMinimumTextHeight_(self, height: float) -> None:
            self.minimum_text_height = height

        def results(self) -> list[_Observation] | None:
            return self.recognized

    class Handler:
        def __init__(self) -> None:
            self.url: Any = None
            self.options: Any = None

        @classmethod
        def alloc(cls) -> Handler:
            return cls()

        def initWithURL_options_(self, url: Any, options: Any) -> Handler:
            self.url = url
            self.options = options
            stub.handlers.append(self)
            return self

        def performRequests_error_(self, requests: list[Any], error: Any) -> tuple[bool, Any]:
            if stub.perform_exception is not None:
                raise stub.perform_exception
            for request in requests:
                request.recognized = stub.observations
            return stub.perform_result

    class NSURL:
        @staticmethod
        def fileURLWithPath_(path: str) -> tuple[str, str]:
            stub.file_url_paths.append(path)
            return ("file-url", path)

    vision_module = types.ModuleType("Vision")
    vision_module.VNImageRequestHandler = Handler  # type: ignore[attr-defined]
    vision_module.VNRecognizeTextRequest = Request  # type: ignore[attr-defined]
    vision_module.VNRequestTextRecognitionLevelAccurate = ACCURATE_LEVEL  # type: ignore[attr-defined]
    foundation_module = types.ModuleType("Foundation")
    foundation_module.NSURL = NSURL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "Vision", vision_module)
    monkeypatch.setitem(sys.modules, "Foundation", foundation_module)
    return stub


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    path.write_bytes(b"not really a png")
    return path


# --------------------------------------------------------------------------
# reading order (pure)
# --------------------------------------------------------------------------


def _line(
    text: str, x: float, y: float, width: float = 0.2, height: float = 0.04
) -> RecognizedLine:
    return RecognizedLine(text=text, x=x, y=y, width=width, height=height)


def test_order_lines_reads_rows_top_to_bottom_then_left_to_right() -> None:
    # Vision's y axis points up: a larger y is nearer the top of the image.
    scrambled = [
        _line("$42.00", x=0.70, y=0.10),
        _line("Receipt", x=0.05, y=0.80),
        _line("Total", x=0.05, y=0.10),
        _line("Price", x=0.70, y=0.50),
        _line("Acme Hardware", x=0.30, y=0.90, width=0.4, height=0.05),
        _line("Item", x=0.05, y=0.50),
    ]
    assert [line.text for line in order_lines(scrambled)] == [
        "Acme Hardware",
        "Receipt",
        "Item",
        "Price",
        "Total",
        "$42.00",
    ]


def test_order_lines_keeps_slightly_offset_boxes_on_one_row() -> None:
    # Same visual line, baselines a few pixels apart: still left-to-right.
    lines = [_line("right", x=0.6, y=0.505), _line("left", x=0.1, y=0.50)]
    assert [line.text for line in order_lines(lines)] == ["left", "right"]


def test_order_lines_starts_a_new_row_for_a_box_that_merely_touches() -> None:
    # Consecutive paragraph lines whose boxes touch: two rows, not one.
    lines = [_line("second", x=0.5, y=0.46), _line("first", x=0.1, y=0.50)]
    assert [line.text for line in order_lines(lines)] == ["first", "second"]


def test_order_lines_anchors_rows_on_their_topmost_line() -> None:
    # A tall box on the left spans two short lines on the right; all three
    # share a row, read left to right, the short lines staying top-down.
    lines = [
        _line("lower-right", x=0.6, y=0.82, height=0.03),
        _line("upper-right", x=0.6, y=0.86, height=0.03),
        _line("TITLE", x=0.0, y=0.80, height=0.10),
    ]
    assert [line.text for line in order_lines(lines)] == ["TITLE", "upper-right", "lower-right"]


def test_order_lines_handles_empty_input() -> None:
    assert order_lines([]) == []


# --------------------------------------------------------------------------
# the provider against the fake framework
# --------------------------------------------------------------------------


def test_recognize_text_joins_lines_in_reading_order(vision: VisionStub, image: Path) -> None:
    vision.observations = [
        _Observation("$42.00", _box(0.70, 0.10, 0.2, 0.04)),
        _Observation("Total", _box(0.05, 0.10, 0.2, 0.04)),
        _Observation("Acme Hardware", _box(0.30, 0.90, 0.4, 0.05)),
    ]
    text = AppleVisionOcrProvider().recognize_text(image)
    assert text == "Acme Hardware\nTotal\n$42.00"


def test_recognize_text_returns_empty_string_when_nothing_is_recognised(
    vision: VisionStub, image: Path
) -> None:
    provider = AppleVisionOcrProvider()
    vision.observations = []
    assert provider.recognize_text(image) == ""
    vision.observations = None  # pyobjc returns None for a nil results array
    assert provider.recognize_text(image) == ""


def test_recognize_text_drops_observations_without_candidates_or_text(
    vision: VisionStub, image: Path
) -> None:
    vision.observations = [
        _Observation(None, _box(0.1, 0.9, 0.2, 0.04)),
        _Observation("   ", _box(0.1, 0.8, 0.2, 0.04)),
        _Observation("  kept  ", _box(0.1, 0.7, 0.2, 0.04)),
    ]
    assert AppleVisionOcrProvider().recognize_text(image) == "kept"


def test_recognize_text_configures_accurate_level_and_language_correction(
    vision: VisionStub, image: Path
) -> None:
    AppleVisionOcrProvider().recognize_text(image)
    (request,) = vision.requests
    assert request.level is ACCURATE_LEVEL
    assert request.language_correction is True
    # No languages configured means "detect" (the config contract for
    # `ocr_languages: null`), which Vision does NOT do by default — its
    # default is a fixed en-US list with automaticallyDetectsLanguage off.
    assert request.languages is None
    assert request.auto_detect is True
    assert request.minimum_text_height is None


def test_recognize_text_passes_languages_and_minimum_height_when_given(
    vision: VisionStub, image: Path
) -> None:
    provider = AppleVisionOcrProvider(
        recognition_languages=["en-US", "es-ES"], minimum_text_height=0.02
    )
    provider.recognize_text(image)
    (request,) = vision.requests
    assert request.languages == ["en-US", "es-ES"]
    assert request.auto_detect is None  # an explicit list is honoured as given
    assert request.minimum_text_height == 0.02


def test_handler_is_built_from_the_image_file_url(vision: VisionStub, image: Path) -> None:
    AppleVisionOcrProvider().recognize_text(image)
    assert vision.file_url_paths == [str(image)]
    (handler,) = vision.handlers
    assert handler.url == ("file-url", str(image))
    assert handler.options == {}


def test_model_id_embeds_the_macos_product_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.4.1", ("", "", ""), "arm64"))
    assert AppleVisionOcrProvider().model_id == "apple/vision-recognize-text@15.4.1"


def test_model_id_falls_back_when_no_macos_version_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert AppleVisionOcrProvider().model_id == "apple/vision-recognize-text@unknown"


def test_failed_perform_is_an_enrichment_error(vision: VisionStub, image: Path) -> None:
    vision.perform_result = (False, "The file could not be decoded")
    with pytest.raises(EnrichmentError) as excinfo:
        AppleVisionOcrProvider().recognize_text(image)
    assert "The file could not be decoded" in str(excinfo.value)
    assert str(image) in str(excinfo.value)


def test_framework_exception_is_wrapped_as_enrichment_error(
    vision: VisionStub, image: Path
) -> None:
    vision.perform_exception = RuntimeError("objc bridge exploded")
    with pytest.raises(EnrichmentError) as excinfo:
        AppleVisionOcrProvider().recognize_text(image)
    assert "objc bridge exploded" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_missing_image_is_an_enrichment_error_without_touching_the_framework(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    block_module(monkeypatch, "Vision")
    with pytest.raises(EnrichmentError):
        AppleVisionOcrProvider().recognize_text(tmp_path / "missing.png")


def test_missing_vision_runtime_is_a_clear_error_naming_the_package(
    monkeypatch: pytest.MonkeyPatch, image: Path
) -> None:
    block_module(monkeypatch, "Vision")
    provider = AppleVisionOcrProvider()  # construction never imports the framework
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        provider.recognize_text(image)
    assert "pyobjc-framework-Vision" in str(excinfo.value)
    assert not isinstance(excinfo.value, EnrichmentError)


def test_satisfies_the_ocr_provider_protocol() -> None:
    provider: OcrProvider = AppleVisionOcrProvider()
    assert isinstance(provider.model_id, str)
    assert provider.model_id.startswith("apple/vision-recognize-text@")
