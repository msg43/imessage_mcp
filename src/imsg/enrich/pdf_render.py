"""Real `pdftoppm` (poppler) PDF-page rasterization for the scanned-OCR
path (SPEC §8 S5b: "pdftoppm 300 dpi -> Apple Vision per page").
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from imsg.errors import EnrichmentError

DEFAULT_DPI = 300


def render_pdf_pages_to_png(
    pdf_path: Path, output_dir: Path, *, dpi: int = DEFAULT_DPI, timeout_seconds: int
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"
    try:
        proc = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnrichmentError(
            f"pdftoppm timed out after {timeout_seconds}s on '{pdf_path}'"
        ) from exc
    except OSError as exc:
        raise EnrichmentError(f"pdftoppm could not run: {exc}") from exc
    if proc.returncode != 0:
        raise EnrichmentError(f"pdftoppm failed on '{pdf_path}': {proc.stderr.strip()}")

    pages = sorted(output_dir.glob("page-*.png"))
    if not pages:
        raise EnrichmentError(f"pdftoppm produced no page images for '{pdf_path}'")
    return pages


__all__ = ["DEFAULT_DPI", "render_pdf_pages_to_png"]
