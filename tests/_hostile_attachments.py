"""Synthetic hostile attachments for the untrusted-input tests (SPEC §8
S5b, D6): decompression-bomb images, PDFs with enormous pages, PDFs whose
text layer inflates to tens of megabytes, and pages that render large.
Everything is generated from constants; no real file is involved. Leading
underscore keeps pytest from collecting this as a test module.
"""

from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, *, shade: int = 200) -> Path:
    """A small, ordinary 8-bit grayscale PNG of one flat shade."""
    raw = b"".join(b"\x00" + bytes([shade]) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)
    return path


def write_png_bomb(path: Path, width: int, height: int) -> Path:
    """A complete, valid 1-bit PNG of `width` x `height` all-black pixels:
    a few hundred kilobytes on disk that decode to `width * height` pixels
    (50,000 x 50,000 is 303,851 bytes and 2.5 gigapixels). The rows are
    compressed one at a time, so building it never holds the image."""
    row = b"\x00" * (1 + (width + 7) // 8)
    packer = zlib.compressobj(9)
    parts = [packer.compress(row) for _ in range(height)]
    parts.append(packer.flush())
    ihdr = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    with path.open("wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_png_chunk(b"IHDR", ihdr))
        fh.write(_png_chunk(b"IDAT", b"".join(parts)))
        fh.write(_png_chunk(b"IEND", b""))
    return path


def png_size(path: Path) -> tuple[int, int]:
    """Width and height from a PNG's IHDR, without decoding it."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def write_pdf(
    path: Path,
    *,
    pages: int = 1,
    media_box: tuple[float, float, float, float] = (0, 0, 612, 792),
    crop_box: tuple[float, float, float, float] | None = None,
    content: bytes = b"BT /F1 24 Tf 72 712 Td (Synthetic page) Tj ET",
    compress: bool = False,
) -> Path:
    """A PDF of `pages` pages that all share one content stream (so a
    thousand pages cost a few kilobytes), with the given media box and
    optional crop box. The smallest structure poppler parses cleanly."""

    def box(values: tuple[float, float, float, float]) -> bytes:
        return ("[" + " ".join(f"{v:g}" for v in values) + "]").encode()

    stream = zlib.compress(content, 9) if compress else content
    flate = b" /Filter /FlateDecode" if compress else b""
    objects: dict[int, bytes] = {
        2: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        3: b"<< /Length %d%s >>\nstream\n" % (len(stream), flate) + stream + b"\nendstream",
    }
    crop = b" /CropBox " + box(crop_box) if crop_box is not None else b""
    kids = []
    for i in range(pages):
        number = 4 + i
        kids.append(b"%d 0 R" % number)
        objects[number] = (
            b"<< /Type /Page /Parent 1 0 R /MediaBox "
            + box(media_box)
            + crop
            + b" /Resources << /Font << /F1 2 0 R >> >> /Contents 3 0 R >>"
        )
    catalog = 4 + pages
    objects[1] = b"<< /Type /Pages /Kids [" + b" ".join(kids) + b"] /Count %d >>" % pages
    objects[catalog] = b"<< /Type /Catalog /Pages 1 0 R >>"
    out = bytearray(b"%PDF-1.7\n")
    offsets: dict[int, int] = {}
    for number in range(1, catalog + 1):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (catalog + 1)
    for number in range(1, catalog + 1):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        catalog + 1,
        catalog,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


_SENTENCE = (
    b"Synthetic filler text for an enormous text layer, repeated to inflate extraction output"
)


def write_huge_text_pdf(path: Path, *, pages: int) -> Path:
    """A PDF whose text layer extracts to thousands of times its size:
    every page (5000 x 792 pt) shows the same compressed content stream of
    77 numbered lines of 8-point text, each twelve copies of one sentence
    long. `pdftotext -layout` keeps about 80 KiB of text per page (poppler
    26.08, measured 2026-09-24; closer or smaller lines are partly dropped
    as overlapping text), so 1,000 pages come to 78.0 MiB, over the
    64 MiB decoder output ceiling, from a PDF of 157,916 bytes."""
    lines = b"".join(
        b"(%05d " % i + b" ".join([_SENTENCE] * 12) + b") Tj T* " for i in range(77)
    )
    content = b"BT /F1 8 Tf 10 TL 5 782 Td " + lines + b"ET"
    return write_pdf(path, pages=pages, media_box=(0, 0, 5000, 792), content=content, compress=True)


def write_dense_page_pdf(path: Path, *, cell_pt: int = 6, seed: int = 7) -> Path:
    """One letter page tiled with `cell_pt`-point squares of random gray,
    which compresses badly as a PNG: at 300 dpi the 6-point version renders
    to about 95 KB (poppler 26.08, measured 2026-09-24), against about
    36 KB for a page with one line of text."""
    rng = random.Random(seed)
    cells = [
        b"%.2f g %d %d %d %d re f" % (rng.random(), x, y, cell_pt, cell_pt)
        for y in range(0, 792, cell_pt)
        for x in range(0, 612, cell_pt)
    ]
    return write_pdf(path, content=b"\n".join(cells), compress=True)


__all__ = [
    "png_size",
    "write_dense_page_pdf",
    "write_huge_text_pdf",
    "write_pdf",
    "write_png",
    "write_png_bomb",
]
