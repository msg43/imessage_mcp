"""Hand-built minimal PDF bytes for real `pdftotext`/`pdftoppm` tests
(SPEC §8 S5b) — no `reportlab`/`fpdf` dependency needed. Leading
underscore keeps pytest from collecting this as a test module.
"""

from __future__ import annotations

from pathlib import Path


def _page_object_bytes(obj_num: int, content_obj_num: int, font_obj_num: int, parent_obj_num: int) -> bytes:
    return (
        f"{obj_num} 0 obj\n"
        f"<< /Type /Page /Parent {parent_obj_num} 0 R /MediaBox [0 0 612 792] "
        f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> /Contents {content_obj_num} 0 R >>\n"
        f"endobj\n"
    ).encode()


def _content_object_bytes(obj_num: int, text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 712 Td ({text}) Tj ET".encode()
    return (
        f"{obj_num} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
        + stream
        + b"\nendstream\nendobj\n"
    )


def write_minimal_pdf(path: Path, page_texts: list[str]) -> None:
    """Write a minimal, valid, single- or multi-page text PDF at `path`,
    one line of text per entry in `page_texts`. Verified against real
    `pdftotext -layout` / `pdftoppm` during development (see
    `docs`-adjacent scratch notes) — this is deliberately the smallest
    PDF structure poppler will parse cleanly, not a general-purpose
    writer.
    """
    n_pages = len(page_texts)
    font_obj = 2
    pages_obj = 1
    catalog_obj = n_pages * 2 + 3  # after: pages_obj(1) + font(2) + n_pages*(page+content)
    # Object numbering: 1=Pages, 2=Font, then for each page i (0-indexed):
    #   page_obj = 3 + 2*i, content_obj = 4 + 2*i
    # catalog is the last object.
    objects: dict[int, bytes] = {}
    page_obj_nums = []
    for i, text in enumerate(page_texts):
        page_obj = 3 + 2 * i
        content_obj = 4 + 2 * i
        page_obj_nums.append(page_obj)
        objects[page_obj] = _page_object_bytes(page_obj, content_obj, font_obj, pages_obj)
        objects[content_obj] = _content_object_bytes(content_obj, text)

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects[pages_obj] = (
        f"{pages_obj} 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>\nendobj\n"
    ).encode()
    objects[font_obj] = (
        f"{font_obj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ).encode()
    objects[catalog_obj] = (
        f"{catalog_obj} 0 obj\n<< /Type /Catalog /Pages {pages_obj} 0 R >>\nendobj\n"
    ).encode()

    max_obj = catalog_obj
    pdf = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for num in range(1, max_obj + 1):
        offsets[num] = len(pdf)
        pdf += objects[num]

    xref_offset = len(pdf)
    pdf += f"xref\n0 {max_obj + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for num in range(1, max_obj + 1):
        pdf += f"{offsets[num]:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {max_obj + 1} /Root {catalog_obj} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode()

    path.write_bytes(pdf)
