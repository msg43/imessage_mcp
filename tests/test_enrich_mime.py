"""Real content-based MIME sniffing via `file` (SPEC §8 S5b, D6: "the
extension is not trusted")."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from _pdf_fixtures import write_minimal_pdf
from imsg.enrich.mime import sniff_mime, sniff_mime_batch
from imsg.enrich.sandboxed_decoder import DecoderBudget
from imsg.errors import UntrustedAttachmentError


def real_sniff_mime(path: Path) -> str:
    """The sandboxed sniffer inside a budget of its own, next to `path`."""
    budget = DecoderBudget.start(
        path.parent / "sniff-work", timeout_seconds=60, max_temp_bytes=2**30
    )
    return sniff_mime(path, budget=budget)


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

    result = sniff_mime_batch(paths, data_root=tmp_path, batch_size=2)

    assert result.errors == {}
    assert result.mime_by_path == {p: real_sniff_mime(p) for p in paths}
    assert result.mime_by_path[caf] == "audio/x-caf"


def test_batch_sniff_reports_a_missing_file_without_losing_the_others(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.write_text("still here\n")
    missing = tmp_path / "missing"

    result = sniff_mime_batch([present, missing], data_root=tmp_path)

    assert result.mime_by_path == {present: "text/plain"}
    assert set(result.errors) == {missing}


def test_batch_sniff_of_nothing_runs_nothing(tmp_path: Path) -> None:
    result = sniff_mime_batch([], data_root=tmp_path)
    assert result.mime_by_path == {}
    assert result.errors == {}


# --------------------------------------------------------------------------
# `file` parses attacker-chosen bytes, so it runs sandboxed and bounded
# like every other decoder (QA review 2026-09-24, fixed 2026-09-29)
# --------------------------------------------------------------------------

needs_sandbox = pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").exists(), reason="needs macOS sandbox-exec"
)


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every argv a subprocess is started with during the test
    (`subprocess.run` starts its process through this same class)."""
    seen: list[list[str]] = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):  # type: ignore[valid-type,misc]
        def __init__(self, args: Any, *rest: Any, **kwargs: Any) -> None:
            seen.append([os.fspath(a) for a in args] if not isinstance(args, str) else [args])
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    return seen


def _file_runs(launched: list[list[str]]) -> list[list[str]]:
    return [argv for argv in launched if any(Path(a).name == "file" for a in argv)]


def _assert_sandboxed_in(argv: list[str], work_dir: Path) -> None:
    assert Path(argv[0]).name == "sandbox-exec", f"ran unsandboxed: {argv}"
    assert argv[1] == "-D" and argv[2] == f"WORK_DIR={work_dir}", argv
    assert argv[3] == "-p" and "(deny network*)" in argv[4] and "(deny file-write*)" in argv[4]
    assert Path(argv[5]).is_absolute() and Path(argv[5]).name == "file", argv


def _fake_file(bin_dir: Path, body: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a `file` that runs `body` first on PATH."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "file"
    fake.write_text(f"#!/bin/sh\n{body}\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@needs_sandbox
def test_the_single_sniff_runs_file_under_the_sandbox_in_the_budgets_work_dir(
    tmp_path: Path, launched: list[list[str]]
) -> None:
    note = tmp_path / "note"
    note.write_text("plain words for Alice\n")
    budget = DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)

    assert sniff_mime(note, budget=budget) == "text/plain"

    (argv,) = _file_runs(launched)
    _assert_sandboxed_in(argv, budget.work_dir)
    assert argv[-2:] == ["--", str(note)]
    assert list(budget.work_dir.iterdir()) == [], "the sniff left its output behind"


@needs_sandbox
def test_the_batch_sniff_runs_file_under_the_sandbox_under_data_root(
    tmp_path: Path, launched: list[list[str]]
) -> None:
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / name
        path.write_text(f"words for Bob, part {name}\n")
        paths.append(path)

    result = sniff_mime_batch(paths, data_root=tmp_path, batch_size=2)

    assert result.mime_by_path == dict.fromkeys(paths, "text/plain")
    runs = _file_runs(launched)
    assert len(runs) == 2  # one `file` per batch of two
    work_root = (tmp_path / "artifacts" / "enrich-work").resolve()
    for argv in runs:
        work_dir = Path(argv[2].split("=", 1)[1])
        assert work_dir.parent == work_root and work_dir.name.startswith(f"{os.getpid()}-sniff-")
        _assert_sandboxed_in(argv, work_dir)
    assert list(work_root.iterdir()) == [], "a scratch work directory was left behind"


@needs_sandbox
def test_a_sniff_past_its_time_bound_is_stopped_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_file(tmp_path / "bin", "exec /bin/sleep 30", monkeypatch)
    note = tmp_path / "note"
    note.write_text("plain words\n")
    budget = DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)

    started = time.monotonic()
    with pytest.raises(UntrustedAttachmentError, match=r"ran past its 0\.5s bound"):
        sniff_mime(note, budget=budget, max_seconds=0.5)
    assert time.monotonic() - started < 10


@needs_sandbox
def test_a_sniff_past_the_tasks_deadline_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_file(tmp_path / "bin", "exec /bin/sleep 30", monkeypatch)
    note = tmp_path / "note"
    note.write_text("plain words\n")
    budget = DecoderBudget.start(tmp_path / "work", timeout_seconds=0.5, max_temp_bytes=2**30)

    with pytest.raises(UntrustedAttachmentError, match="task_timeout_seconds"):
        sniff_mime(note, budget=budget)


@needs_sandbox
def test_a_write_outside_the_work_dir_is_refused_and_fails_the_sniff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `file` that tries to write beside the attachment: the same
    script unsandboxed does write there, so the refusal is the sandbox's."""
    target = tmp_path / "escaped"
    _fake_file(
        tmp_path / "bin",
        f"echo pwned > '{target}' || exit 1\necho text/plain",
        monkeypatch,
    )
    note = tmp_path / "note"
    note.write_text("plain words\n")
    subprocess.run([str(tmp_path / "bin" / "file")], check=True, capture_output=True)
    assert target.exists(), "the probe cannot write even unsandboxed"
    target.unlink()
    budget = DecoderBudget.start(tmp_path / "work", timeout_seconds=60, max_temp_bytes=2**30)

    with pytest.raises(UntrustedAttachmentError, match="'file' exited 1"):
        sniff_mime(note, budget=budget)
    assert not target.exists()


@needs_sandbox
def test_the_enqueue_hook_sniffs_sandboxed_under_data_root(
    tmp_path: Path, launched: list[list[str]]
) -> None:
    """S5a's hook (`enqueue_for_materialized`) runs outside any task; its
    sniff gets a scratch work directory under `data_root`. An unidentified
    binary routes nowhere, so no database is touched."""
    from imsg.enrich.planner import enqueue_for_materialized

    blob = tmp_path / "blob"
    blob.write_bytes(b"cafe" + bytes(range(256)) * 4)

    outcome = enqueue_for_materialized(None, 1, blob, data_root=tmp_path)  # type: ignore[arg-type]

    assert outcome.error is None and outcome.mime_type == "application/octet-stream"
    (argv,) = _file_runs(launched)
    work_dir = Path(argv[2].split("=", 1)[1])
    assert work_dir.parent == (tmp_path / "artifacts" / "enrich-work").resolve()
    _assert_sandboxed_in(argv, work_dir)
    assert not work_dir.exists()
