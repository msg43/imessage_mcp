"""`doc_text`: text from text-bearing attachments that are not PDFs (SPEC
§8 S5b) — contact cards, plain text, HTML and SVG, RTF and Word
documents, and OOXML spreadsheets and slides.

Every reader here is local and none needs a model:

- **plain text, vCard, HTML/SVG** are decoded and parsed in-process with
  the standard library. Nothing is fetched: `html.parser` has no loader.
- **RTF, DOC, DOCX, ODT** go through macOS `textutil`, the Cocoa text
  system's converter. It runs without a shell, under `sandbox-exec` with a
  profile that denies all network access and every file write, in the
  task's temp directory, with the reader chosen by `-format` from the
  sniffed MIME type (never from the file), a wall-clock ceiling, and an
  output ceiling enforced while it runs. Cocoa's document readers are the
  one decoder here that could otherwise resolve a document's remote
  references; the sandbox is what enforces SPEC §8 S5b's "never fetch
  embedded URLs" for them.
- **XLSX, PPTX**, which `textutil` cannot read, are opened with
  `zipfile` and parsed with ElementTree (expat 2.6, which refuses
  entity-expansion bombs), each member's declared and actual size checked
  against a ceiling before and while it is inflated.

A ceiling hit is a typed permanent failure (`UntrustedAttachmentError`),
per SPEC §8 S5b. A reader that fails on malformed input raises a plain
`EnrichmentError`, which the queue retries with backoff and then marks
`failed`, exactly as it does for `pdftotext` and `ffmpeg`.

Contact cards drop their embedded photos and keys: a base64 blob in a
chunk would be indexed as noise and embedded as nonsense.
"""

from __future__ import annotations

import codecs
import quopri
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from imsg.enrich.sandboxed_decoder import MAX_DECODER_OUTPUT_BYTES
from imsg.errors import EnrichmentError, UnsupportedEnrichmentTypeError, UntrustedAttachmentError
from imsg.textnorm import strip_nul

MAX_DOCUMENT_INPUT_BYTES = 128 * 1024 * 1024
"""Largest attachment `doc_text` will open. Far above any text document
in the corpus (the 54 Word documents on the production index total
10.6 MB); a file above it is refused, permanently, before it is read."""

MAX_DOCUMENT_TEXT_CHARS = 1_000_000
"""Longest text kept from one document, about a 400-page book. Beyond it
the text is cut and `detail.truncated` says so; every chunk still gets
embedded, so this bounds the embedding work one attachment can cause."""

# `textutil` is stopped once its output passes `MAX_DECODER_OUTPUT_BYTES`,
# the ceiling every decoder's read-back output shares
# (`imsg.enrich.sandboxed_decoder`; `pdftotext` has it too).

MAX_OOXML_MEMBER_BYTES = 64 * 1024 * 1024
MAX_OOXML_MEMBERS = 20_000

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
TEXTUTIL = "/usr/bin/textutil"
TEXTUTIL_SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)(deny file-write*)"
"""No network, no file writes. `textutil -stdout` writes only to the pipe
it inherits, so it needs neither."""

_IN_PROCESS_FORMATS = frozenset({"plain", "vcard", "html"})
_TEXTUTIL_FORMATS = frozenset({"rtf", "doc", "docx", "odt"})
_OOXML_FORMATS = frozenset({"xlsx", "pptx"})
_POLL_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class DocumentText:
    text: str
    extractor: str
    """`text`, `vcard`, `html`, `textutil` or `ooxml` — recorded as the
    enrichment row's `model`."""
    truncated: bool


def extract_document_text(
    path: Path, fmt: str, *, work_dir: Path, timeout_seconds: int
) -> DocumentText:
    """Read `path` as the document format `fmt`
    (`imsg.enrich.router.document_format_for_mime`)."""
    size = path.stat().st_size
    if size > MAX_DOCUMENT_INPUT_BYTES:
        raise UntrustedAttachmentError(
            f"'{path}' is {size} bytes, over the doc_text input ceiling "
            f"({MAX_DOCUMENT_INPUT_BYTES} bytes)"
        )
    cut = False
    if fmt in _IN_PROCESS_FORMATS:
        raw, cut = _read_prefix(path)
        decoded = decode_text_bytes(raw, partial=cut)
        if fmt == "plain":
            text, extractor = decoded, "text"
        elif fmt == "vcard":
            text, extractor = vcard_to_text(decoded), "vcard"
        else:
            text, extractor = html_to_text(decoded), "html"
    elif fmt in _TEXTUTIL_FORMATS:
        text = run_textutil(path, fmt, work_dir=work_dir, timeout_seconds=timeout_seconds)
        extractor = "textutil"
    elif fmt in _OOXML_FORMATS:
        text, extractor = ooxml_text(path, fmt), "ooxml"
    else:
        raise UnsupportedEnrichmentTypeError(f"doc_text has no reader for document format {fmt!r}")

    text = _tidy(text)
    truncated = cut or len(text) > MAX_DOCUMENT_TEXT_CHARS
    text = text[:MAX_DOCUMENT_TEXT_CHARS]
    return DocumentText(text=text, extractor=extractor, truncated=truncated)


def _read_prefix(path: Path) -> tuple[bytes, bool]:
    """At most four bytes per character `MAX_DOCUMENT_TEXT_CHARS` keeps
    (UTF-8's widest), and whether the file went on past that. The
    in-process parsers run outside any subprocess timeout, so this is
    what bounds their time; a longer file is read, and indexed, up to
    there rather than refused."""
    limit = 4 * MAX_DOCUMENT_TEXT_CHARS
    with path.open("rb") as fh:
        data = fh.read(limit + 1)
    return data[:limit], len(data) > limit


def _tidy(text: str) -> str:
    """One newline convention, no trailing spaces, at most one blank line
    in a row, nothing leading or trailing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(out).strip()


# --------------------------------------------------------------------------
# plain text
# --------------------------------------------------------------------------

_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def decode_text_bytes(data: bytes, *, partial: bool = False) -> str:
    """Bytes to text: a byte-order mark decides when there is one, then
    UTF-8, then Windows-1252 (which maps almost every byte, so an old
    Western-European text file still reads). NUL is removed; Postgres
    `text` cannot hold it. `partial` says `data` is a prefix, which may
    end inside a UTF-8 character; up to three trailing bytes are then
    dropped rather than letting that one cut character turn the whole
    text into Windows-1252."""
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return strip_nul(data.decode(encoding, errors="replace"))
    for trim in range(4 if partial else 1):
        try:
            return strip_nul(data[: len(data) - trim].decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return strip_nul(data.decode("cp1252", errors="replace"))


# --------------------------------------------------------------------------
# vCard (2.1, 3.0, 4.0), including Apple's shared-location cards
# --------------------------------------------------------------------------

_BINARY_ENCODINGS = frozenset({"B", "BASE64"})
_BARE_ENCODINGS = frozenset({"B", "BASE64", "QUOTED-PRINTABLE", "8BIT"})
_DROPPED_PROPERTIES = frozenset({"PHOTO", "LOGO", "SOUND", "KEY"})
_PROPERTY_LABELS: dict[str, str] = {
    "NICKNAME": "Nickname",
    "TITLE": "Title",
    "ROLE": "Role",
    "URL": "URL",
    "NOTE": "Note",
    "BDAY": "Birthday",
    "ANNIVERSARY": "Anniversary",
    "IMPP": "Messaging",
    "GEO": "Location",
    "X-SOCIALPROFILE": "Social profile",
    "X-ABRELATEDNAMES": "Related name",
    "X-ABDATE": "Date",
}
_TYPED_PROPERTIES: dict[str, str] = {"TEL": "Phone", "EMAIL": "Email", "ADR": "Address"}
_TYPE_WORDS: dict[str, str] = {
    "cell": "cell",
    "mobile": "mobile",
    "iphone": "iPhone",
    "home": "home",
    "work": "work",
    "main": "main",
    "fax": "fax",
    "homefax": "home fax",
    "workfax": "work fax",
    "pager": "pager",
    "other": "other",
}
_APPLE_LABEL_RE = re.compile(r"^_\$!<(.*)>!\$_$")
_ESCAPE_RE = re.compile(r"\\(.)")


@dataclass(frozen=True, slots=True)
class _ContentLine:
    group: str | None
    name: str
    params: dict[str, list[str]]
    value: str


def _split_unquoted(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        if ch == sep and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _parse_content_line(line: str) -> _ContentLine | None:
    head, colon, value = "", "", ""
    quoted = False
    for index, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == ":" and not quoted:
            head, colon, value = line[:index], ":", line[index + 1 :]
            break
    if not colon:
        return None
    pieces = _split_unquoted(head, ";")
    group_and_name = pieces[0]
    group, _, name = group_and_name.rpartition(".")
    params: dict[str, list[str]] = {}
    for piece in pieces[1:]:
        key, eq, raw = piece.partition("=")
        if eq:
            values = [v.strip().strip('"') for v in raw.split(",") if v.strip()]
            params.setdefault(key.strip().upper(), []).extend(values)
        elif piece.strip().upper() in _BARE_ENCODINGS:  # vCard 2.1: `NOTE;QUOTED-PRINTABLE:`
            params.setdefault("ENCODING", []).append(piece.strip())
        elif piece.strip():  # vCard 2.1 bare type: `TEL;CELL;VOICE:`
            params.setdefault("TYPE", []).append(piece.strip())
    return _ContentLine(group=group or None, name=name.strip().upper(), params=params, value=value)


def _logical_lines(raw: str) -> list[str]:
    """Physical lines joined back into content lines: RFC 6350 folding
    (a line starting with a space or tab continues the one before) and
    vCard 2.1 quoted-printable soft breaks (a value ending in `=`)."""
    physical = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: list[str] = []
    for line in physical:
        if lines and _continues_quoted_printable(lines[-1]):
            lines[-1] = lines[-1][:-1] + line.lstrip(" \t")
            continue
        if line[:1] in (" ", "\t") and lines:
            lines[-1] += line[1:]
            continue
        lines.append(line)
    return [line for line in lines if line.strip()]


def _continues_quoted_printable(line: str) -> bool:
    if not line.endswith("="):
        return False
    head = line.split(":", 1)[0].upper()
    return "QUOTED-PRINTABLE" in head


def _unescape(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        ch = match.group(1)
        return "\n" if ch in ("n", "N") else ch

    return _ESCAPE_RE.sub(repl, value)


def _components(value: str) -> list[str]:
    """Split a structured value (N, ADR, ORG) on unescaped semicolons,
    then unescape each component."""
    parts: list[str] = []
    current: list[str] = []
    escaped = False
    for ch in value:
        if escaped:
            current.append("\\" + ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ";":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [_unescape(p).strip() for p in parts]


def _decode_value(line: _ContentLine) -> str | None:
    """The property's text, or `None` when it is binary."""
    encodings = {e.upper() for e in line.params.get("ENCODING", [])}
    if encodings & _BINARY_ENCODINGS:
        return None
    if any(v.upper() == "BINARY" for v in line.params.get("VALUE", [])):
        return None
    if "QUOTED-PRINTABLE" in encodings:
        charset = (line.params.get("CHARSET") or ["utf-8"])[0]
        raw = quopri.decodestring(line.value.encode("ascii", errors="replace"))
        try:
            decoded = raw.decode(charset, errors="replace")
        except LookupError:
            decoded = raw.decode("utf-8", errors="replace")
        return decoded.replace("\r\n", "\n")
    return line.value


def _type_label(line: _ContentLine, group_labels: dict[str, str]) -> str:
    if line.group and line.group in group_labels:
        return group_labels[line.group]
    for raw in line.params.get("TYPE", []):
        for word in raw.split(","):
            label = _TYPE_WORDS.get(word.strip().lower())
            if label:
                return label
    return ""


def _apple_label(value: str) -> str:
    match = _APPLE_LABEL_RE.match(value.strip())
    return (match.group(1) if match else value).strip()


def _format_name(parts: list[str]) -> str:
    family, given, additional, prefix, suffix = (parts + [""] * 5)[:5]
    return " ".join(p for p in (prefix, given, additional, family, suffix) if p)


def _render_card(lines: list[_ContentLine]) -> list[str]:
    group_labels = {
        line.group: _apple_label(line.value)
        for line in lines
        if line.group and line.name == "X-ABLABEL" and line.value.strip()
    }
    formatted_name = ""
    for line in lines:
        if line.name == "FN":
            value = _decode_value(line)
            formatted_name = _unescape(value).strip() if value is not None else ""
            break
    out: list[str] = []
    if formatted_name:
        out.append(f"Name: {formatted_name}")
    for line in lines:
        if line.name in _DROPPED_PROPERTIES or line.name in ("FN", "X-ABLABEL"):
            continue
        value = _decode_value(line)
        if value is None:
            continue
        if line.name == "N":
            full = _format_name(_components(value))
            if full and full != formatted_name:
                out.append(f"Full name: {full}" if formatted_name else f"Name: {full}")
        elif line.name == "ORG":
            org = ", ".join(p for p in _components(value) if p)
            if org:
                out.append(f"Organization: {org}")
        elif line.name in _TYPED_PROPERTIES:
            if line.name == "ADR":
                text = ", ".join(p for p in _components(value) if p)
            else:
                text = _unescape(value).strip()
            if text:
                label = _type_label(line, group_labels)
                noun = _TYPED_PROPERTIES[line.name]
                out.append(f"{noun} ({label}): {text}" if label else f"{noun}: {text}")
        elif line.name in _PROPERTY_LABELS:
            text = _unescape(value).strip()
            if text:
                label = group_labels.get(line.group or "") or _PROPERTY_LABELS[line.name]
                out.append(f"{label}: {text}")
    return out


def vcard_to_text(raw: str) -> str:
    """Every card in `raw`, as labelled lines a person would search for:
    names, organization, title, phones, emails, addresses, links and
    notes. Embedded photos, logos, sounds and keys are dropped, and so is
    any property whose value is binary. Cards are separated by a blank
    line."""
    cards: list[list[_ContentLine]] = []
    current: list[_ContentLine] | None = None
    for text in _logical_lines(raw):
        parsed = _parse_content_line(text)
        if parsed is None:
            continue
        if parsed.name == "BEGIN" and parsed.value.strip().upper() == "VCARD":
            current = []
            continue
        if parsed.name == "END" and parsed.value.strip().upper() == "VCARD":
            if current is not None:
                cards.append(current)
            current = None
            continue
        if current is not None:
            current.append(parsed)
    if current:  # an unterminated last card still counts
        cards.append(current)
    rendered = ["\n".join(lines) for lines in (_render_card(c) for c in cards) if lines]
    return "\n\n".join(rendered)


# --------------------------------------------------------------------------
# HTML and SVG
# --------------------------------------------------------------------------

_SKIPPED_ELEMENTS = frozenset({"script", "style", "noscript", "template"})
_BLOCK_ELEMENTS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
        "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "td", "text",
        "th", "title", "tr", "ul",
    }
)


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_ELEMENTS:
            self._skip_depth += 1
        elif tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_ELEMENTS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    """Visible text of an HTML or SVG document: script, style and
    template content dropped, entities decoded, block elements on lines of
    their own. Parsing only; nothing the document references is loaded."""
    collector = _TextCollector()
    collector.feed(raw)
    collector.close()
    joined = "".join(collector.parts).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in joined.split("\n")]
    return _tidy("\n".join(lines))


# --------------------------------------------------------------------------
# RTF, DOC, DOCX, ODT: macOS textutil, sandboxed
# --------------------------------------------------------------------------


def textutil_command(path: Path, fmt: str) -> list[str]:
    """The exact argv: `sandbox-exec` with the no-network, no-write
    profile, `textutil` told which reader to use, and `--` so a path is
    never read as an option. No shell anywhere."""
    return [
        SANDBOX_EXEC,
        "-p",
        TEXTUTIL_SANDBOX_PROFILE,
        TEXTUTIL,
        "-convert",
        "txt",
        "-format",
        fmt,
        "-encoding",
        "UTF-8",
        "-stdout",
        "--",
        str(path),
    ]


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL cannot be caught or ignored, so the wait always returns."""
    proc.kill()
    proc.wait()


def run_textutil(path: Path, fmt: str, *, work_dir: Path, timeout_seconds: int) -> str:
    """Run `textutil_command` in `work_dir` and return its text.

    `textutil` exits 0 even when it cannot read the file (checked on
    macOS 26: it prints "Error reading <name>. The file isn't in the
    correct format." on stderr), so an `Error` line on stderr is a failure
    too. Output goes to a file in `work_dir`, never into memory
    unbounded, and the run is stopped if that file passes
    `MAX_DECODER_OUTPUT_BYTES` or the run passes `timeout_seconds`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "textutil.out"
    err_path = work_dir / "textutil.err"
    with out_path.open("wb") as out, err_path.open("wb") as err:
        try:
            proc = subprocess.Popen(
                textutil_command(path, fmt),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=work_dir,
            )
        except OSError as exc:
            raise EnrichmentError(f"textutil could not run: {exc}") from exc
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                returncode = proc.wait(timeout=_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            if out_path.stat().st_size > MAX_DECODER_OUTPUT_BYTES:
                _stop(proc)
                raise UntrustedAttachmentError(
                    f"textutil output for '{path}' passed the decoder output ceiling "
                    f"({MAX_DECODER_OUTPUT_BYTES} bytes)"
                )
            if time.monotonic() > deadline:
                _stop(proc)
                raise UntrustedAttachmentError(
                    f"textutil ran past the {timeout_seconds}s task ceiling "
                    f"(enrichment.limits.task_timeout_seconds) on '{path}'"
                )
    stderr = err_path.read_text(encoding="utf-8", errors="replace").strip()
    if returncode != 0 or any(line.startswith("Error") for line in stderr.splitlines()):
        raise EnrichmentError(
            f"textutil could not read '{path}' as {fmt} (exit {returncode}): {stderr[-500:]}"
        )
    if out_path.stat().st_size > MAX_DECODER_OUTPUT_BYTES:
        raise UntrustedAttachmentError(
            f"textutil output for '{path}' passed the decoder output ceiling "
            f"({MAX_DECODER_OUTPUT_BYTES} bytes)"
        )
    return decode_text_bytes(out_path.read_bytes())


# --------------------------------------------------------------------------
# XLSX, PPTX
# --------------------------------------------------------------------------

_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_SHEET_PART_RE = re.compile(r"^xl/worksheets/sheet(\d+)\.xml$")
_SLIDE_PART_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
_NOTES_PART_RE = re.compile(r"^ppt/notesSlides/notesSlide(\d+)\.xml$")


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_OOXML_MEMBER_BYTES:
        raise UntrustedAttachmentError(
            f"OOXML part '{name}' declares {info.file_size} bytes, over the "
            f"{MAX_OOXML_MEMBER_BYTES}-byte member ceiling"
        )
    with zf.open(info) as fh:
        data = fh.read(MAX_OOXML_MEMBER_BYTES + 1)
    if len(data) > MAX_OOXML_MEMBER_BYTES:
        raise UntrustedAttachmentError(
            f"OOXML part '{name}' inflates past the {MAX_OOXML_MEMBER_BYTES}-byte member ceiling"
        )
    return data


def _numbered_parts(zf: zipfile.ZipFile, pattern: re.Pattern[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for name in zf.namelist():
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), name))
    return [name for _, name in sorted(found)]


def _joined_text(element: ElementTree.Element, tag: str) -> str:
    return "".join(t.text or "" for t in element.iter(tag))


def _xlsx_text(zf: zipfile.ZipFile) -> str:
    lines: list[str] = []
    workbook = _read_member(zf, "xl/workbook.xml")
    if workbook is not None:
        for sheet in ElementTree.fromstring(workbook).iter(f"{_SHEET_NS}sheet"):
            name = sheet.get("name")
            if name:
                lines.append(f"Sheet: {name}")
    shared = _read_member(zf, "xl/sharedStrings.xml")
    if shared is not None:
        for item in ElementTree.fromstring(shared).iter(f"{_SHEET_NS}si"):
            text = _joined_text(item, f"{_SHEET_NS}t").strip()
            if text:
                lines.append(text)
    for part in _numbered_parts(zf, _SHEET_PART_RE):
        data = _read_member(zf, part)
        if data is None:
            continue
        for cell in ElementTree.fromstring(data).iter(f"{_SHEET_NS}c"):
            kind = cell.get("t")
            if kind == "inlineStr":
                text = _joined_text(cell, f"{_SHEET_NS}t").strip()
            elif kind == "str":  # a formula's string result
                value = cell.find(f"{_SHEET_NS}v")
                text = (value.text or "").strip() if value is not None else ""
            else:
                continue
            if text:
                lines.append(text)
    return "\n".join(lines)


def _slide_text(data: bytes) -> str:
    paragraphs: list[str] = []
    for paragraph in ElementTree.fromstring(data).iter(f"{_DRAWING_NS}p"):
        text = _joined_text(paragraph, f"{_DRAWING_NS}t").strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _pptx_text(zf: zipfile.ZipFile) -> str:
    blocks: list[str] = []
    for pattern in (_SLIDE_PART_RE, _NOTES_PART_RE):
        for part in _numbered_parts(zf, pattern):
            data = _read_member(zf, part)
            if data is not None:
                text = _slide_text(data)
                if text:
                    blocks.append(text)
    return "\n\n".join(blocks)


def ooxml_text(path: Path, fmt: str) -> str:
    """Sheet names and every string cell of a spreadsheet, or every
    paragraph of every slide (in slide-number order) and then the speaker
    notes of a deck."""
    try:
        with zipfile.ZipFile(path) as zf:
            if len(zf.infolist()) > MAX_OOXML_MEMBERS:
                raise UntrustedAttachmentError(
                    f"'{path}' holds more than {MAX_OOXML_MEMBERS} parts, over the OOXML ceiling"
                )
            return _xlsx_text(zf) if fmt == "xlsx" else _pptx_text(zf)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, ElementTree.ParseError, EOFError) as exc:
        raise EnrichmentError(f"could not read '{path}' as {fmt}: {exc}") from exc


__all__ = [
    "MAX_DECODER_OUTPUT_BYTES",
    "MAX_DOCUMENT_INPUT_BYTES",
    "MAX_DOCUMENT_TEXT_CHARS",
    "MAX_OOXML_MEMBERS",
    "MAX_OOXML_MEMBER_BYTES",
    "TEXTUTIL_SANDBOX_PROFILE",
    "DocumentText",
    "decode_text_bytes",
    "extract_document_text",
    "html_to_text",
    "ooxml_text",
    "run_textutil",
    "textutil_command",
    "vcard_to_text",
]
