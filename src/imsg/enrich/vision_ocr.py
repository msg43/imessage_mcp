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

**A pixel ceiling before decoding (2026-09-24, QA review).** Vision
decodes whatever the file declares, and a PNG of a few hundred kilobytes
can declare 50,000 x 50,000 pixels, which decodes to gigabytes. So
`recognize_text` first reads the image's dimensions from its header with
ImageIO, the framework Vision itself decodes with, without decoding any
pixels (`read_image_dimensions`: a 2.5-gigapixel bomb read in 6 ms with
about 1 MiB of memory growth, measured 2026-09-24), and refuses an image
over `max_image_pixels` (`enrichment.limits.max_image_pixels`) with a
typed permanent failure. The default is the ceiling Pillow already puts
on the caption path (twice `PIL.Image.MAX_IMAGE_PIXELS`: 178,956,970), so
OCR and captioning refuse the same images. An image ImageIO cannot read
the size of is refused too: Vision could not decode it either.

**A time limit on every recognition (2026-10-29).** On the production host
`performRequests:error:` never returned for one JPEG: the worker sat at
0 % CPU for 7 hours, and the caption runner and the scheduled sync waited
behind it for the heavy-model lock. So each recognition runs on its own
thread, and the caller waits at most `timeout_seconds`. Past that the
request is cancelled (`VNRequest.cancel`) and given `CANCEL_GRACE_SECONDS`
to return. If it does, the image fails permanently and work goes on. If
not, the thread cannot be stopped from Python, so the provider marks
itself unusable and raises `OcrHungError`: the task is recorded failed and
`imsg enrich` exits, so the next run starts with a fresh process.
"""

from __future__ import annotations

import contextlib
import os
import platform
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imsg import constants
from imsg.enrich.limits import check_image_pixels
from imsg.enrich.model_runtime import import_runtime_module
from imsg.errors import EnrichmentError, ImsgError, OcrHungError, UntrustedAttachmentError
from imsg.textnorm import strip_nul

ImageDimensionsFn = Callable[[Path], tuple[int, int]]

_INSTALL_HINT = (
    "install `pyobjc-framework-Vision` (it brings the Quartz and CoreML bridges with it) "
    "and run on macOS"
)

DEFAULT_TIMEOUT_SECONDS = 300.0
"""`enrichment.limits.ocr_timeout_seconds`'s default (config schema)."""

CANCEL_GRACE_SECONDS = 10.0
"""How long a timed-out recognition gets to return after it is cancelled."""

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


def read_image_dimensions(image_path: Path) -> tuple[int, int]:
    """The width and height of the image Vision would decode from
    `image_path`, read from the file's header by ImageIO without decoding
    its pixels. For a file holding several images (HEIC, TIFF, GIF) the
    larger of the first and the primary image is reported. Raises
    `UntrustedAttachmentError` when ImageIO cannot read a size at all."""
    quartz = import_runtime_module("Quartz", install_hint=_INSTALL_HINT)
    options = {quartz.kCGImageSourceShouldCache: False}
    provider = quartz.CGDataProviderCreateWithFilename(os.fsencode(image_path))
    source = (
        quartz.CGImageSourceCreateWithDataProvider(provider, options)
        if provider is not None
        else None
    )
    count = quartz.CGImageSourceGetCount(source) if source is not None else 0
    if source is None or count < 1:
        raise UntrustedAttachmentError(
            f"'{image_path}' is not an image ImageIO can read; not handing it to Vision"
        )
    indices = {0, int(quartz.CGImageSourceGetPrimaryImageIndex(source))}
    best: tuple[int, int] | None = None
    for index in sorted(i for i in indices if 0 <= i < count):
        props = quartz.CGImageSourceCopyPropertiesAtIndex(source, index, options)
        width = props.get(quartz.kCGImagePropertyPixelWidth) if props is not None else None
        height = props.get(quartz.kCGImagePropertyPixelHeight) if props is not None else None
        if width is None or height is None:
            raise UntrustedAttachmentError(
                f"ImageIO reports no pixel size for image {index} of '{image_path}'; "
                f"not handing it to Vision"
            )
        size = (int(width), int(height))
        if best is None or size[0] * size[1] > best[0] * best[1]:
            best = size
    assert best is not None
    return best


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

    `max_image_pixels` is the ceiling checked before Vision sees an image
    (module docstring); `None` turns it off. `image_dimensions` reads an
    image's size and defaults to `read_image_dimensions`.
    `timeout_seconds` bounds each recognition (module docstring).
    """

    def __init__(
        self,
        *,
        recognition_languages: list[str] | None = None,
        minimum_text_height: float | None = None,
        max_image_pixels: int | None = constants.DEFAULT_MAX_IMAGE_PIXELS,
        image_dimensions: ImageDimensionsFn | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if max_image_pixels is not None and max_image_pixels < 1:
            raise ValueError(f"max_image_pixels must be positive or None, got {max_image_pixels}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.timeout_seconds = float(timeout_seconds)
        self.hung: str | None = None
        """Set once a recognition could not be stopped; every later call
        refuses at once, since Vision is stuck in this process."""
        self.recognition_languages = list(recognition_languages) if recognition_languages else None
        self.minimum_text_height = minimum_text_height
        self.max_image_pixels = max_image_pixels
        self._image_dimensions = image_dimensions or read_image_dimensions
        self.model_id = f"apple/vision-recognize-text@{_macos_product_version()}"

    def recognize_text(self, image_path: Path) -> str:
        """Every recognized line, one per line, in reading order; the
        empty string when Vision finds no text at all. An image over the
        pixel ceiling is refused before Vision is involved."""
        if self.hung is not None:
            raise OcrHungError(
                f"Apple Vision is stuck in this process ({self.hung}); not starting another "
                f"recognition on '{image_path}'"
            )
        if not image_path.is_file():
            raise EnrichmentError(f"OCR input is not a file: '{image_path}'")
        if self.max_image_pixels is not None:
            try:
                width, height = self._image_dimensions(image_path)
            except ImsgError:
                raise
            except Exception as exc:
                raise EnrichmentError(
                    f"could not read the pixel size of '{image_path}' before OCR: {exc}"
                ) from exc
            check_image_pixels(width, height, max_pixels=self.max_image_pixels, subject=image_path)
        vision = import_runtime_module("Vision", install_hint=_INSTALL_HINT)
        foundation = import_runtime_module("Foundation", install_hint=_INSTALL_HINT)
        try:
            lines = self._recognize_within_limit(vision, foundation, image_path)
        except ImsgError:
            raise
        except Exception as exc:
            raise EnrichmentError(
                f"Apple Vision text recognition failed on '{image_path}': {exc}"
            ) from exc
        return strip_nul("\n".join(line.text for line in order_lines(lines)))

    def _recognize_within_limit(
        self, vision: Any, foundation: Any, image_path: Path
    ) -> list[RecognizedLine]:
        """`_recognize` on its own thread, waited for at most
        `timeout_seconds`, then cancelled (module docstring)."""
        outcome: dict[str, Any] = {}
        requests: list[Any] = []

        def run() -> None:
            try:
                outcome["lines"] = self._recognize(
                    vision, foundation, image_path, on_request=requests.append
                )
            except BaseException as exc:  # handed to the waiting thread
                outcome["error"] = exc

        worker = threading.Thread(target=run, name="imsg-vision-ocr", daemon=True)
        worker.start()
        worker.join(self.timeout_seconds)
        if worker.is_alive():
            for request in requests:
                # A request that cannot cancel is waited out below.
                with contextlib.suppress(Exception):
                    request.cancel()
            worker.join(CANCEL_GRACE_SECONDS)
            limit = f"{self.timeout_seconds:g} s (enrichment.limits.ocr_timeout_seconds)"
            if worker.is_alive():
                self.hung = f"text recognition on '{image_path}' passed {limit} and ignored cancel"
                raise OcrHungError(
                    f"Apple Vision did not finish text recognition on '{image_path}' within "
                    f"{limit} and did not stop when cancelled"
                )
            raise UntrustedAttachmentError(
                f"Apple Vision text recognition on '{image_path}' passed {limit}; cancelled"
            )
        if "error" in outcome:
            raise outcome["error"]
        lines: list[RecognizedLine] = outcome["lines"]
        return lines

    def _recognize(
        self,
        vision: Any,
        foundation: Any,
        image_path: Path,
        *,
        on_request: Callable[[Any], object] | None = None,
    ) -> list[RecognizedLine]:
        url = foundation.NSURL.fileURLWithPath_(str(image_path))
        # An empty Foundation dictionary, never a Python `{}`: on macOS 27
        # with pyobjc 12.2.2 the bridged empty Python dict makes Vision
        # raise "NSInvalidArgumentException - key does not exist" for every
        # image (2026-10-28, production host), while an NSDictionary works.
        handler = vision.VNImageRequestHandler.alloc().initWithURL_options_(
            url, foundation.NSDictionary.dictionary()
        )
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
        if on_request is not None:
            on_request(request)

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


__all__ = [
    "CANCEL_GRACE_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "ROW_OVERLAP_FRACTION",
    "AppleVisionOcrProvider",
    "ImageDimensionsFn",
    "RecognizedLine",
    "order_lines",
    "read_image_dimensions",
]
