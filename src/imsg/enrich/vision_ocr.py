"""Apple Vision OCR (SPEC §4.1: `VNRecognizeTextRequest`, accurate mode)
behind the `OcrProvider` Protocol, through pyobjc's `Vision` bridge.

Vision reports one `VNRecognizedTextObservation` per detected line, in
no documented order, each with a bounding box normalized to [0, 1]
whose origin is the image's *bottom-left* corner. `order_lines` turns
that into reading order: observations are grouped into visual rows by
vertical overlap and each row is read left to right, so a receipt's
two-column "item ... price" lines and slightly skewed scans come out
the way a person reads them, not in detection order. Side-by-side
paragraphs (true multi-column text) still interleave line by line —
column detection is deliberately out of scope for a search index.

The frameworks are imported on first use (see
`imsg.enrich.model_runtime`); constructing the provider needs nothing
installed.
"""

from __future__ import annotations

import platform
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imsg.enrich.model_runtime import import_runtime_module
from imsg.errors import EnrichmentError, ImsgError
from imsg.textnorm import strip_nul

_INSTALL_HINT = (
    "install `pyobjc-framework-Vision` (it brings the Quartz and CoreML bridges with it) "
    "and run on macOS"
)

ROW_OVERLAP_FRACTION = 0.5
"""Two boxes share a row when their vertical overlap exceeds this
fraction of the shorter box's height — enough that a box merely touching
the line above it starts a new row, while boxes on one visual line with
slightly different baselines stay together."""


@dataclass(frozen=True, slots=True)
class RecognizedLine:
    """One recognized line and its Vision bounding box: normalized to
    [0, 1], origin bottom-left, `y` the bottom edge — exactly as
    `VNRecognizedTextObservation.boundingBox` reports it."""

    text: str
    x: float
    y: float
    width: float
    height: float

    @property
    def top(self) -> float:
        return self.y + self.height


def _same_row(anchor: RecognizedLine, line: RecognizedLine) -> bool:
    overlap = min(anchor.top, line.top) - max(anchor.y, line.y)
    return overlap > ROW_OVERLAP_FRACTION * min(anchor.height, line.height)


def order_lines(lines: Iterable[RecognizedLine]) -> list[RecognizedLine]:
    """Top-to-bottom, left-to-right reading order.

    Lines are visited from the top of the image down; each joins the
    current row when it vertically overlaps the row's first (topmost)
    line by more than `ROW_OVERLAP_FRACTION` of the shorter of the two
    heights, and otherwise starts a new row. Rows are then read left to
    right. Anchoring on the row's first line rather than its latest
    member keeps a skewed scan from drifting one row into the next.
    """
    rows: list[list[RecognizedLine]] = []
    for line in sorted(lines, key=lambda item: (-item.top, item.x)):
        if rows and _same_row(rows[-1][0], line):
            rows[-1].append(line)
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda item: item.x)]


def _macos_product_version() -> str:
    version = platform.mac_ver()[0]
    return version or "unknown"


class AppleVisionOcrProvider:
    """`OcrProvider` backed by `VNRecognizeTextRequest` in accurate mode
    with language correction on.

    `model_id` is `apple/vision-recognize-text@<macOS product version>`:
    the recognizer ships with the OS, so the OS version is the only
    identifier of which model actually ran. `recognition_languages`
    (BCP-47 tags, in priority order) and `minimum_text_height` (a
    fraction of the image height) are passed straight through to the
    request when given. With no languages configured the request's
    `automaticallyDetectsLanguage` is switched on — Vision's own default
    is a fixed en-US list with detection off, which is not what
    `enrichment.ocr_languages: null` promises.
    """

    def __init__(
        self,
        *,
        recognition_languages: list[str] | None = None,
        minimum_text_height: float | None = None,
    ) -> None:
        self.recognition_languages = list(recognition_languages) if recognition_languages else None
        self.minimum_text_height = minimum_text_height
        self.model_id = f"apple/vision-recognize-text@{_macos_product_version()}"

    def recognize_text(self, image_path: Path) -> str:
        """Every recognized line, one per line, in reading order; the
        empty string when Vision finds no text at all."""
        if not image_path.is_file():
            raise EnrichmentError(f"OCR input is not a file: '{image_path}'")
        vision = import_runtime_module("Vision", install_hint=_INSTALL_HINT)
        foundation = import_runtime_module("Foundation", install_hint=_INSTALL_HINT)
        try:
            lines = self._recognize(vision, foundation, image_path)
        except ImsgError:
            raise
        except Exception as exc:
            raise EnrichmentError(
                f"Apple Vision text recognition failed on '{image_path}': {exc}"
            ) from exc
        return strip_nul("\n".join(line.text for line in order_lines(lines)))

    def _recognize(self, vision: Any, foundation: Any, image_path: Path) -> list[RecognizedLine]:
        url = foundation.NSURL.fileURLWithPath_(str(image_path))
        handler = vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        request = vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        if self.recognition_languages is not None:
            request.setRecognitionLanguages_(self.recognition_languages)
        else:
            # Vision's own default is a fixed ['en-US'] with automatic
            # detection OFF (read back from a live VNRecognizeTextRequest,
            # pyobjc 12.2.2 on macOS 26, 2026-09-14). The config contract
            # for `enrichment.ocr_languages: null` is detection, so ask for
            # it explicitly (macOS 13+, the manifest's floor).
            request.setAutomaticallyDetectsLanguage_(True)
        if self.minimum_text_height is not None:
            request.setMinimumTextHeight_(self.minimum_text_height)

        # pyobjc hands the trailing `NSError **` out-parameter back as a
        # second tuple element — the same shape `imsg.stages.identity`
        # reads from `enumerateContactsWithFetchRequest:error:usingBlock:`.
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise EnrichmentError(
                f"VNImageRequestHandler could not run text recognition on '{image_path}': {error}"
            )

        lines: list[RecognizedLine] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            text = str(candidates[0].string()).strip()
            if not text:
                continue
            box = observation.boundingBox()
            lines.append(
                RecognizedLine(
                    text=text,
                    x=float(box.origin.x),
                    y=float(box.origin.y),
                    width=float(box.size.width),
                    height=float(box.size.height),
                )
            )
        return lines


__all__ = ["ROW_OVERLAP_FRACTION", "AppleVisionOcrProvider", "RecognizedLine", "order_lines"]
