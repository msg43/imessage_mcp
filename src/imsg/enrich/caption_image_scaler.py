"""The caption provider's image scaler, run as its own process under the
enrichment sandbox (`imsg.enrich.sandboxed_decoder`, 2026-09-29).

`python -m imsg.enrich.caption_image_scaler <image> <scaled> <max_side>`
opens `<image>` with PIL and, when its longest side is over `max_side`,
writes a copy scaled to fit to `<scaled>` as PNG: aspect ratio kept, EXIF
orientation applied first. It writes nothing when the image already fits
or PIL cannot open it, and exits 0 in all three cases; the caller tells
them apart by whether `<scaled>` exists. Any other exit is a failure.

Decoding the attachment here, rather than in the enrichment worker,
puts PIL's parsing of bytes a stranger chose inside the same fence as
every other decoder: no network, writes only in the task's work
directory, and the task's wall-clock, temp-space and memory ceilings.
Imports nothing from `imsg`, so the process starts quickly.
"""

from __future__ import annotations

import contextlib
import importlib
import sys

EXIT_USAGE = 2


def scale_to_fit(image_path: str, scaled_path: str, max_side: int) -> bool:
    """Write the scaled copy and return True, or return False when the
    image already fits or PIL cannot open it."""
    # Imported by name: PIL and pillow-heif come with the `models` extra.
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    # Without pillow-heif, HEIC is merely unreadable here, as before.
    with contextlib.suppress(Exception):
        importlib.import_module("pillow_heif").register_heif_opener()
    try:
        with image_module.open(image_path) as img:
            if max(img.size) <= max_side:
                return False
            upright = image_ops.exif_transpose(img).convert("RGB")
    except (OSError, ValueError, SyntaxError, image_module.DecompressionBombError):
        return False
    upright.thumbnail((max_side, max_side), image_module.Resampling.LANCZOS)
    upright.save(scaled_path, "PNG")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: caption_image_scaler <image> <scaled> <max_side>", file=sys.stderr)
        return EXIT_USAGE
    image_path, scaled_path, max_side_text = argv
    try:
        max_side = int(max_side_text)
    except ValueError:
        print(f"max_side must be an integer, got {max_side_text!r}", file=sys.stderr)
        return EXIT_USAGE
    scale_to_fit(image_path, scaled_path, max_side)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
