"""Real content-based MIME sniffing via `file` (SPEC §8 S5b, D6: "the
extension is not trusted")."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from _pdf_fixtures import write_minimal_pdf
from imsg.enrich.mime import real_sniff_mime, sniff_mime_batch
from imsg.errors import UntrustedAttachmentError


def test_sniffs_pdf_by_content_even_with_wrong_extension(tmp_path: Path) -> None:
    misnamed = tmp_path / "totally_an_image.jpg"
    write_minimal_pdf(misnamed, ["Hello"])
    assert real_sniff_mime(misnamed) == "application/pdf"


def test_sniffs_plain_text(tmp_path: Path) -> None:
    f = tmp_path / "notes.dat"
    f.write_text("just some plain text content here\n")
    assert real_sniff_mime(f) == "text/plain"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UntrustedAttachmentError):
        real_sniff_mime(tmp_path / "does-not-exist")


# --------------------------------------------------------------------------
# Core Audio Format: `file` 5.41 (macOS 26) calls a voice message
# `application/octet-stream`; the signature says otherwise
# --------------------------------------------------------------------------


def _minimal_caf_header() -> bytes:
    """The first bytes of every CAF file: 'caff', version 1, flags 0,
    then a 'desc' chunk header. Enough for signature sniffing; not a
    playable file."""
    return b"caff" + b"\x00\x01" + b"\x00\x00" + b"desc" + (32).to_bytes(8, "big") + bytes(32)


def test_caf_signature_is_sniffed_as_audio(tmp_path: Path) -> None:
    voice = tmp_path / "0123abcd"  # content-addressed cache names carry no extension
    voice.write_bytes(_minimal_caf_header())
    assert real_sniff_mime(voice) == "audio/x-caf"


def test_a_real_voice_message_container_is_sniffed_as_audio(tmp_path: Path) -> None:
    """A CAF written by Core Audio itself, the container Messages uses for
    voice messages. Without the signature check this routes nowhere and
    the voice message is never transcribed."""
    if shutil.which("afconvert") is None or shutil.which("ffmpeg") is None:
        pytest.skip("needs macOS afconvert and ffmpeg")
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(wav)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    caf = tmp_path / "voice"
    subprocess.run(
        ["afconvert", "-f", "caff", "-d", "aac", str(wav), str(caf)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert real_sniff_mime(caf) == "audio/x-caf"


def test_octet_stream_without_the_caf_signature_stays_octet_stream(tmp_path: Path) -> None:
    blob = tmp_path / "blob"
    blob.write_bytes(b"cafe" + bytes(range(256)) * 4)
    assert real_sniff_mime(blob) == "application/octet-stream"


# --------------------------------------------------------------------------
# batch sniffing: one `file` process per batch, not per attachment
# --------------------------------------------------------------------------


def test_batch_sniff_agrees_with_single_sniff(tmp_path: Path) -> None:
    pdf = tmp_path / "a"
    write_minimal_pdf(pdf, ["Hello"])
    text = tmp_path / "b"
    text.write_text("plain words for Alice\n")
    caf = tmp_path / "c"
    caf.write_bytes(_minimal_caf_header())
    paths = [pdf, text, caf]

    result = sniff_mime_batch(paths, batch_size=2)

    assert result.errors == {}
    assert result.mime_by_path == {p: real_sniff_mime(p) for p in paths}
    assert result.mime_by_path[caf] == "audio/x-caf"


def test_batch_sniff_reports_a_missing_file_without_losing_the_others(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.write_text("still here\n")
    missing = tmp_path / "missing"

    result = sniff_mime_batch([present, missing])

    assert result.mime_by_path == {present: "text/plain"}
    assert set(result.errors) == {missing}


def test_batch_sniff_of_nothing_runs_nothing() -> None:
    result = sniff_mime_batch([])
    assert result.mime_by_path == {}
    assert result.errors == {}
