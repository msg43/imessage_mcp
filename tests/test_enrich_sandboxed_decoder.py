"""The sandboxed decoder runner (`imsg.enrich.sandboxed_decoder`), with
the real `sandbox-exec`, `ffmpeg` and poppler binaries.

Each sandbox test first runs the same command unsandboxed and checks it
does reach the network or write outside the work directory, so the
sandboxed run's "nothing happened" is a result and not a broken probe.
Every input is synthetic (`tests/_hostile_attachments.py`, ffmpeg's
`lavfi` sources)."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from _hostile_attachments import write_huge_text_pdf, write_pdf
from imsg.enrich.sandboxed_decoder import (
    SANDBOX_EXEC,
    TASK_SANDBOX_PROFILE,
    WORK_DIR_PARAM,
    DecoderBudget,
    decoder_command,
    directory_bytes,
    resolve_executable,
    run_decoder,
)
from imsg.errors import EnrichmentError, UntrustedAttachmentError

_HAVE_TOOLS = (
    Path(SANDBOX_EXEC).exists()
    and shutil.which("ffmpeg") is not None
    and shutil.which("pdftoppm") is not None
    and shutil.which("pdftotext") is not None
)
needs_tools = pytest.mark.skipif(
    not _HAVE_TOOLS, reason="needs macOS sandbox-exec, ffmpeg and poppler"
)


def _budget(tmp_path: Path, **overrides: float) -> DecoderBudget:
    settings: dict[str, float] = {
        "timeout_seconds": 60,
        "max_temp_bytes": 2**30,
        "max_memory_bytes": 4 * 2**30,
    }
    settings.update(overrides)
    return DecoderBudget.start(
        tmp_path / "work",
        timeout_seconds=settings["timeout_seconds"],
        max_temp_bytes=int(settings["max_temp_bytes"]),
        max_memory_bytes=int(settings["max_memory_bytes"]),
    )


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    path = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:a", "aac", str(path)],
        check=True, capture_output=True, timeout=30,
    )
    return path


class _Listener:
    """A TCP listener on 127.0.0.1 that counts accepted connections."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        self._stop = False
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            conn.close()

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=5)
        self.sock.close()


@pytest.fixture
def listener() -> Iterator[_Listener]:
    server = _Listener()
    try:
        yield server
    finally:
        server.close()


# --------------------------------------------------------------------------
# the command and the budget (no binaries needed)
# --------------------------------------------------------------------------


def test_the_command_is_sandbox_exec_with_the_work_dir_as_a_parameter(tmp_path: Path) -> None:
    argv = decoder_command(["/usr/bin/true", "--flag"], tmp_path)
    assert argv[:5] == [SANDBOX_EXEC, "-D", f"{WORK_DIR_PARAM}={tmp_path}", "-p", TASK_SANDBOX_PROFILE]
    assert argv[5:] == ["/usr/bin/true", "--flag"]
    assert "(deny network*)" in TASK_SANDBOX_PROFILE
    assert "(deny file-write*)" in TASK_SANDBOX_PROFILE
    assert f'(allow file-write* (subpath (param "{WORK_DIR_PARAM}")))' in TASK_SANDBOX_PROFILE
    # The path is a parameter, never spliced into the profile text.
    assert str(tmp_path) not in TASK_SANDBOX_PROFILE


def test_a_missing_decoder_is_an_ordinary_enrichment_error() -> None:
    with pytest.raises(EnrichmentError) as excinfo:
        resolve_executable("imsg-no-such-decoder-anywhere")
    assert not isinstance(excinfo.value, UntrustedAttachmentError)  # retried, not permanent


def test_the_budget_resolves_the_work_dir_the_way_the_kernel_sees_it(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    budget = DecoderBudget.start(tmp_path / "link" / "work", timeout_seconds=5, max_temp_bytes=100)
    assert budget.work_dir == (real / "work").resolve()


def test_the_deadline_covers_the_whole_task_not_each_call(tmp_path: Path) -> None:
    now = [100.0]
    budget = DecoderBudget.start(
        tmp_path / "work", timeout_seconds=30, max_temp_bytes=100, clock=lambda: now[0]
    )
    budget.check_deadline("first decoder")
    now[0] = 129.0
    budget.check_deadline("second decoder")  # 29 s in: still inside
    now[0] = 131.0
    with pytest.raises(UntrustedAttachmentError, match="task_timeout_seconds"):
        budget.check_deadline("third decoder")


def test_directory_bytes_counts_files_below_the_root(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"x" * 32)
    (tmp_path / "link").symlink_to(tmp_path / "a")  # not followed, not counted twice
    assert directory_bytes(tmp_path) == 42


# --------------------------------------------------------------------------
# the sandbox, with real decoders
# --------------------------------------------------------------------------


@needs_tools
def test_a_sandboxed_decoder_opens_no_network_connection(
    tmp_path: Path, listener: _Listener
) -> None:
    url = f"http://127.0.0.1:{listener.port}/remote.wav"
    budget = _budget(tmp_path)
    out = budget.work_dir / "fetched.wav"

    # The probe can fail: the same command, unsandboxed, does connect.
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", url, str(tmp_path / "control.wav")],
        capture_output=True, timeout=30, check=False,
    )
    assert listener.connections >= 1, "the unsandboxed control never reached the listener"
    before = listener.connections

    run = run_decoder(
        ["ffmpeg", "-nostdin", "-y", "-i", url, str(out)],
        budget=budget, name="ffmpeg", subject=Path(url),
    )
    time.sleep(0.3)  # let the listener thread see any late connection
    assert listener.connections == before, "the sandboxed decoder reached the network"
    assert run.returncode != 0
    assert "Operation not permitted" in run.stderr_tail()


@needs_tools
def test_a_sandboxed_decoder_cannot_write_outside_the_work_dir(tmp_path: Path, tone: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    budget = _budget(tmp_path)

    control = outside / "control.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(tone), str(control)],
        capture_output=True, timeout=30, check=True,
    )
    assert control.is_file(), "the unsandboxed control could not write there either"

    escaped = outside / "escaped.wav"
    run = run_decoder(
        ["ffmpeg", "-nostdin", "-y", "-i", str(tone), str(escaped)],
        budget=budget, name="ffmpeg", subject=tone,
    )
    assert run.returncode != 0
    assert not escaped.exists()
    assert "Operation not permitted" in run.stderr_tail()

    pdf = write_pdf(tmp_path / "doc.pdf")
    run = run_decoder(
        ["pdftoppm", "-png", "-r", "10", "-singlefile", str(pdf), str(outside / "page")],
        budget=budget, name="pdftoppm", subject=pdf,
    )
    assert run.returncode != 0
    assert not (outside / "page.png").exists()


@needs_tools
def test_a_sandboxed_decoder_writes_inside_the_work_dir(tmp_path: Path, tone: Path) -> None:
    budget = _budget(tmp_path)
    out = budget.work_dir / "tone.wav"
    run = run_decoder(
        ["ffmpeg", "-nostdin", "-y", "-i", str(tone), str(out)],
        budget=budget, name="ffmpeg", subject=tone,
    )
    assert run.returncode == 0, run.stderr_tail()
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------
# the ceilings, each stopping a decoder that is still running
# --------------------------------------------------------------------------


def _endless_wav(out: Path) -> list[str]:
    """ffmpeg writing a ten-minute tone in real time (`-re`): about 32 KB of
    WAV a second, for far longer than any of these tests waits."""
    return [
        "ffmpeg", "-nostdin", "-y", "-re", "-f", "lavfi", "-i", "sine=frequency=440:duration=600",
        "-ac", "1", "-ar", "16000", "-f", "wav", str(out),
    ]


@needs_tools
def test_a_decoder_that_keeps_writing_is_stopped_at_the_temp_ceiling(tmp_path: Path) -> None:
    budget = _budget(tmp_path, max_temp_bytes=96 * 1024)
    out = budget.work_dir / "endless.wav"
    started = time.monotonic()
    with pytest.raises(UntrustedAttachmentError, match="temp_bytes_per_task") as excinfo:
        run_decoder(_endless_wav(out), budget=budget, name="ffmpeg", subject=Path("tone"))
    elapsed = time.monotonic() - started
    assert elapsed < 30, f"stopped only after {elapsed:.1f} s"
    assert "ffmpeg" in str(excinfo.value)
    size = out.stat().st_size
    time.sleep(0.5)
    assert out.stat().st_size == size, "the decoder kept writing after it was stopped"
    assert size < 1024 * 1024


@needs_tools
def test_a_decoder_running_past_the_task_deadline_is_stopped(tmp_path: Path) -> None:
    budget = _budget(tmp_path, timeout_seconds=1.0)
    started = time.monotonic()
    with pytest.raises(UntrustedAttachmentError, match="task_timeout_seconds"):
        run_decoder(
            _endless_wav(budget.work_dir / "endless.wav"),
            budget=budget, name="ffmpeg", subject=Path("tone"),
        )
    assert time.monotonic() - started < 10


@needs_tools
def test_a_decoder_over_the_memory_ceiling_is_stopped(tmp_path: Path) -> None:
    """A 40-inch page at 600 dpi is a 1.7 GB bitmap; the decoder is stopped
    at 256 MiB, long before it gets there."""
    pdf = write_pdf(tmp_path / "poster.pdf", media_box=(0, 0, 2880, 2880))
    budget = _budget(tmp_path, max_memory_bytes=256 * 2**20)
    started = time.monotonic()
    with pytest.raises(UntrustedAttachmentError, match="max_decoder_memory_bytes"):
        run_decoder(
            ["pdftoppm", "-png", "-r", "600", "-singlefile", str(pdf), str(budget.work_dir / "p")],
            budget=budget, name="pdftoppm", subject=pdf,
        )
    assert time.monotonic() - started < 30
    assert not (budget.work_dir / "p.png").exists()


@needs_tools
def test_output_past_its_ceiling_stops_the_decoder(tmp_path: Path) -> None:
    pdf = write_huge_text_pdf(tmp_path / "text.pdf", pages=20)
    budget = _budget(tmp_path)
    with pytest.raises(UntrustedAttachmentError, match="decoder output ceiling"):
        run_decoder(
            ["pdftotext", "-layout", str(pdf), "-"],
            budget=budget, name="pdftotext", subject=pdf,
            stdout_name="out.txt", max_stdout_bytes=256 * 1024,
        )


@needs_tools
def test_the_stderr_tail_is_the_end_of_the_log_and_bounded(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(os.urandom(4096))
    run = run_decoder(
        ["ffmpeg", "-nostdin", "-y", "-i", str(junk), str(budget.work_dir / "x.wav")],
        budget=budget, name="ffmpeg", subject=junk,
    )
    assert run.returncode != 0
    tail = run.stderr_tail(limit=120)
    assert 0 < len(tail) <= 120
