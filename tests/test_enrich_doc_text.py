"""Text extraction for text-bearing attachments (`doc_text`, SPEC §8 S5b):
contact cards, plain text, HTML/SVG, Office and RTF documents (through
macOS `textutil`, sandboxed), and OOXML spreadsheets and slides.

Fictional personas only. Every fixture is generated here; nothing is
read from a real corpus.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

import imsg.enrich.doc_text as doc_text
from imsg.enrich.doc_text import (
    TEXTUTIL_SANDBOX_PROFILE,
    decode_text_bytes,
    extract_document_text,
    html_to_text,
    textutil_command,
    vcard_to_text,
)
from imsg.errors import (
    EnrichmentError,
    UnsupportedEnrichmentTypeError,
    UntrustedAttachmentError,
)

_ON_MACOS = shutil.which("textutil") is not None and shutil.which("sandbox-exec") is not None

_ALICE_CARD = (
    "BEGIN:VCARD\r\n"
    "VERSION:3.0\r\n"
    "N:Example;Alice;Q.;Dr.;\r\n"
    "FN:Alice Example\r\n"
    "ORG:Acme Construction;Estimating\r\n"
    "TITLE:Project Manager\r\n"
    "TEL;type=CELL;type=VOICE;type=pref:+1 555 0100\r\n"
    "TEL;type=WORK:+1 555 0199\r\n"
    "EMAIL;type=INTERNET;type=WORK:alice@example.com\r\n"
    "ADR;type=WORK:;;1 Main Street;Springfield;IL;62701;USA\r\n"
    "NOTE:Met at the site walk\\, bring the\\nrevised bid\r\n"
    "PHOTO;ENCODING=b;TYPE=JPEG:/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgK\r\n"
    " DBQNDAsLDBkSEw8UHRofHiQYGhsbGyEeGBsfIxsd\r\n"
    "END:VCARD\r\n"
)


# --------------------------------------------------------------------------
# vCard
# --------------------------------------------------------------------------


def test_vcard_renders_the_fields_a_person_would_search_for() -> None:
    text = vcard_to_text(_ALICE_CARD)
    assert "Alice Example" in text
    assert "Dr. Alice Q. Example" in text
    assert "Acme Construction" in text
    assert "Project Manager" in text
    assert "+1 555 0100" in text
    assert "+1 555 0199" in text
    assert "alice@example.com" in text
    assert "1 Main Street, Springfield, IL, 62701, USA" in text
    assert "Met at the site walk, bring the\nrevised bid" in text


def test_vcard_drops_embedded_photos_including_their_folded_lines() -> None:
    text = vcard_to_text(_ALICE_CARD)
    assert "/9j/4AAQ" not in text
    assert "DBQNDAsL" not in text
    assert "PHOTO" not in text


def test_vcard_labels_phone_and_email_types() -> None:
    text = vcard_to_text(_ALICE_CARD)
    assert "Phone (cell): +1 555 0100" in text
    assert "Phone (work): +1 555 0199" in text
    assert "Email (work): alice@example.com" in text


def test_vcard_21_quoted_printable_values_are_decoded() -> None:
    card = (
        "BEGIN:VCARD\r\n"
        "VERSION:2.1\r\n"
        "FN;CHARSET=UTF-8;ENCODING=QUOTED-PRINTABLE:Bob Caf=C3=A9\r\n"
        "NOTE;ENCODING=QUOTED-PRINTABLE;CHARSET=UTF-8:Deck rebuild=0D=0Aon Thurs=\r\n"
        "day\r\n"
        "END:VCARD\r\n"
    )
    text = vcard_to_text(card)
    assert "Bob Café" in text
    assert "Deck rebuild\r\non Thursday" in text or "Deck rebuild\non Thursday" in text


def test_location_card_keeps_the_map_link_under_its_label() -> None:
    card = (
        "BEGIN:VCARD\r\n"
        "VERSION:3.0\r\n"
        "N:;Current Location;;;\r\n"
        "FN:Current Location\r\n"
        "item1.URL;type=pref:http://maps.apple.com/?ll=39.78\\,-89.65&q=39.78\\,-89.65\r\n"
        "item1.X-ABLabel:map url\r\n"
        "END:VCARD\r\n"
    )
    text = vcard_to_text(card)
    assert "Current Location" in text
    assert "map url: http://maps.apple.com/?ll=39.78,-89.65&q=39.78,-89.65" in text


def test_every_card_in_a_multi_card_file_is_kept() -> None:
    bob = _ALICE_CARD.replace("Alice", "Bob").replace("alice@", "bob@")
    text = vcard_to_text(_ALICE_CARD + bob)
    assert "Alice Example" in text
    assert "Bob Example" in text
    assert "bob@example.com" in text


def test_vcard_route_reads_a_file_end_to_end(tmp_path: Path) -> None:
    card = tmp_path / "card"
    card.write_bytes(_ALICE_CARD.encode("utf-8"))
    result = extract_document_text(card, "vcard", work_dir=tmp_path, timeout_seconds=30)
    assert result.extractor == "vcard"
    assert "Alice Example" in result.text
    assert "/9j/4AAQ" not in result.text
    assert result.truncated is False


# --------------------------------------------------------------------------
# plain text and markup
# --------------------------------------------------------------------------


def test_utf16_with_a_byte_order_mark_is_decoded() -> None:
    assert decode_text_bytes("Bid for Acme".encode("utf-16")) == "Bid for Acme"


def test_bytes_that_are_not_utf8_still_decode() -> None:
    assert decode_text_bytes("Caf\xe9 at noon".encode("cp1252")) == "Café at noon"


def test_nul_characters_are_removed() -> None:
    assert decode_text_bytes(b"deck\x00 rebuild") == "deck rebuild"


def test_html_keeps_visible_text_and_drops_script_and_style() -> None:
    raw = (
        "<html><head><title>Bid summary</title><style>p {color: red}</style>"
        "<script>var secret = 1;</script></head>"
        "<body><h1>Acme &amp; Sons</h1><p>Total: 14&nbsp;000 USD</p><p>Call Bob</p></body></html>"
    )
    text = html_to_text(raw)
    assert "Bid summary" in text
    assert "Acme & Sons" in text
    assert "Total: 14\xa0000 USD" in text or "Total: 14 000 USD" in text
    assert "Call Bob" in text
    assert "color: red" not in text
    assert "secret" not in text


def test_html_block_elements_become_separate_lines() -> None:
    text = html_to_text("<div>first</div><div>second</div><p>third<br>fourth</p>")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines == ["first", "second", "third", "fourth"]


def test_svg_text_nodes_are_extracted(tmp_path: Path) -> None:
    svg = tmp_path / "drawing"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><style>.a{fill:red}</style>'
        '<text x="1" y="9">Lot 12 site plan</text></svg>'
    )
    result = extract_document_text(svg, "html", work_dir=tmp_path, timeout_seconds=30)
    assert result.text.strip() == "Lot 12 site plan"


def test_plain_text_route_reads_a_file(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.write_text("Pour moved to Thursday.\r\nBring the level.\r\n")
    result = extract_document_text(notes, "plain", work_dir=tmp_path, timeout_seconds=30)
    assert result.extractor == "text"
    assert result.text == "Pour moved to Thursday.\nBring the level."


# --------------------------------------------------------------------------
# ceilings: input size is a permanent failure, output length truncates
# --------------------------------------------------------------------------


def test_oversized_document_is_a_permanent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doc_text, "MAX_DOCUMENT_INPUT_BYTES", 16)
    big = tmp_path / "big"
    big.write_text("x" * 17)
    with pytest.raises(UntrustedAttachmentError, match="ceiling"):
        extract_document_text(big, "plain", work_dir=tmp_path, timeout_seconds=30)


def test_extracted_text_is_truncated_at_the_ceiling_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doc_text, "MAX_DOCUMENT_TEXT_CHARS", 10)
    notes = tmp_path / "notes"
    notes.write_text("0123456789ABCDEF")
    result = extract_document_text(notes, "plain", work_dir=tmp_path, timeout_seconds=30)
    assert result.text == "0123456789"
    assert result.truncated is True


def test_a_long_text_file_is_read_up_to_the_ceiling_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-process parsing runs outside any subprocess timeout, so a long
    file is read only as far as the text that would be kept. The read
    stops at 4 bytes per kept character; here that lands inside the
    sixth three-byte euro sign, and the cut must not turn the rest into
    Windows-1252 mojibake."""
    monkeypatch.setattr(doc_text, "MAX_DOCUMENT_TEXT_CHARS", 4)
    prices = tmp_path / "prices"
    prices.write_bytes(("\u20ac" * 50).encode("utf-8"))  # 150 bytes; the read stops at 16
    result = extract_document_text(prices, "plain", work_dir=tmp_path, timeout_seconds=30)
    assert result.text == "\u20ac" * 4
    assert result.truncated is True


def test_unknown_format_is_skipped_not_retried(tmp_path: Path) -> None:
    blob = tmp_path / "blob"
    blob.write_bytes(b"\x00\x01")
    with pytest.raises(UnsupportedEnrichmentTypeError):
        extract_document_text(blob, "pages", work_dir=tmp_path, timeout_seconds=30)


# --------------------------------------------------------------------------
# Office and RTF through textutil, sandboxed
# --------------------------------------------------------------------------


def test_textutil_command_is_sandboxed_shell_free_and_format_bound(tmp_path: Path) -> None:
    target = tmp_path / "-looks-like-an-option"
    argv = textutil_command(target, "docx")
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[1] == "-p"
    assert argv[2] == TEXTUTIL_SANDBOX_PROFILE
    assert "(deny network*)" in TEXTUTIL_SANDBOX_PROFILE
    assert "(deny file-write*)" in TEXTUTIL_SANDBOX_PROFILE
    assert argv[3] == "/usr/bin/textutil"
    # The sniffed type decides the reader, never the file's own claims.
    assert argv[argv.index("-format") + 1] == "docx"
    # A path can never be read as an option.
    assert argv[-2:] == ["--", str(target)]


@pytest.mark.skipif(not _ON_MACOS, reason="needs macOS textutil and sandbox-exec")
@pytest.mark.parametrize("fmt", ["docx", "doc", "rtf", "odt"])
def test_textutil_extracts_office_and_rtf_documents(tmp_path: Path, fmt: str) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Quarterly plan for Acme Construction\n\nAlice Example reviews the bid.\n")
    made = tmp_path / f"made.{fmt}"
    subprocess.run(
        ["textutil", "-convert", fmt, "-output", str(made), str(source)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    cached = tmp_path / "0f1e2d3c"  # the content-addressed cache has no extension
    cached.write_bytes(made.read_bytes())
    work = tmp_path / "work"
    work.mkdir()

    result = extract_document_text(cached, fmt, work_dir=work, timeout_seconds=60)

    assert result.extractor == "textutil"
    assert "Quarterly plan for Acme Construction" in result.text
    assert "Alice Example reviews the bid." in result.text


@pytest.mark.skipif(not _ON_MACOS, reason="needs macOS textutil and sandbox-exec")
def test_textutil_failure_is_an_enrichment_error(tmp_path: Path) -> None:
    fake = tmp_path / "not-really-docx"
    fake.write_text("this is plain text pretending to be a Word document")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(EnrichmentError) as info:
        extract_document_text(fake, "docx", work_dir=work, timeout_seconds=60)
    # A decoder that exits non-zero follows the existing decoders: an
    # ordinary failure, retried with backoff, then `failed`.
    assert type(info.value) is EnrichmentError


@pytest.mark.skipif(not _ON_MACOS, reason="needs macOS sandbox-exec and curl")
def test_the_textutil_sandbox_profile_blocks_network_access() -> None:
    """The profile is what enforces SPEC §8 S5b's "never fetch embedded
    URLs" for Cocoa's document readers. Checked against a listener on
    loopback, so the test needs no internet: without the profile the
    connection succeeds, under it the connection is refused."""
    if shutil.which("curl") is None:
        pytest.skip("needs curl")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    port = server.getsockname()[1]
    stop = threading.Event()

    def serve() -> None:
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok")
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    try:
        open_run = subprocess.run(
            ["curl", "-s", "-m", "5", url], capture_output=True, text=True, timeout=30
        )
        sandboxed_run = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", TEXTUTIL_SANDBOX_PROFILE, "curl", "-s", "-m", "5", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        stop.set()
        thread.join(timeout=5)
        server.close()
    assert open_run.stdout == "ok"
    assert sandboxed_run.returncode != 0
    assert sandboxed_run.stdout == ""


# --------------------------------------------------------------------------
# OOXML spreadsheets and slides (textutil reads neither)
# --------------------------------------------------------------------------

_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DRAW_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_PRES_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _write_xlsx(path: Path, *, shared_strings: list[str], inline: str) -> None:
    sst = "".join(f"<si><t>{s}</t></si>" for s in shared_strings)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{_SHEET_NS}"><sheets><sheet name="Bids" sheetId="1"/>'
            f'<sheet name="Crew" sheetId="2"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{_SHEET_NS}" count="{len(shared_strings)}">{sst}</sst>',
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{_SHEET_NS}"><sheetData><row r="1">'
            f'<c r="A1" t="s"><v>0</v></c>'
            f'<c r="B1" t="inlineStr"><is><t>{inline}</t></is></c>'
            f'<c r="C1"><v>14000</v></c>'
            f"</row></sheetData></worksheet>",
        )


def _slide(texts: list[str]) -> str:
    paragraphs = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in texts)
    return (
        f'<p:sld xmlns:p="{_PRES_NS}" xmlns:a="{_DRAW_NS}"><p:cSld><p:spTree><p:sp><p:txBody>'
        f"{paragraphs}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )


def test_xlsx_sheet_names_shared_and_inline_strings_are_extracted(tmp_path: Path) -> None:
    book = tmp_path / "book"
    _write_xlsx(book, shared_strings=["Deck rebuild", "Acme Construction"], inline="Alice Example")
    result = extract_document_text(book, "xlsx", work_dir=tmp_path, timeout_seconds=30)
    assert result.extractor == "ooxml"
    for expected in ("Bids", "Crew", "Deck rebuild", "Acme Construction", "Alice Example"):
        assert expected in result.text


def test_pptx_slides_are_read_in_slide_number_order(tmp_path: Path) -> None:
    deck = tmp_path / "deck"
    with zipfile.ZipFile(deck, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("ppt/slides/slide10.xml", _slide(["Tenth: closing"]))
        zf.writestr("ppt/slides/slide2.xml", _slide(["Second: budget", "14,000 USD"]))
        zf.writestr("ppt/slides/slide1.xml", _slide(["First: Acme kickoff"]))
        zf.writestr("ppt/notesSlides/notesSlide1.xml", _slide(["Speaker note for Bob"]))
    result = extract_document_text(deck, "pptx", work_dir=tmp_path, timeout_seconds=30)
    text = result.text
    assert text.index("First: Acme kickoff") < text.index("Second: budget") < text.index("Tenth: closing")
    assert "14,000 USD" in text
    assert "Speaker note for Bob" in text


def test_an_oversized_ooxml_member_is_refused_before_it_is_inflated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doc_text, "MAX_OOXML_MEMBER_BYTES", 1024)
    book = tmp_path / "bomb"
    _write_xlsx(book, shared_strings=["x" * 4096], inline="small")
    with pytest.raises(UntrustedAttachmentError, match="ceiling"):
        extract_document_text(book, "xlsx", work_dir=tmp_path, timeout_seconds=30)


def test_a_broken_ooxml_container_is_an_enrichment_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(EnrichmentError) as info:
        extract_document_text(broken, "xlsx", work_dir=tmp_path, timeout_seconds=30)
    assert type(info.value) is EnrichmentError
