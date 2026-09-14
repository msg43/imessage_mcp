"""mlx-whisper transcription provider
(`imsg.enrich.mlx_whisper_transcription`), driven through a fake
`mlx_whisper` module stood in `sys.modules` — no weights, no network."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from _model_runtime_stubs import block_module, install_hub_stub
from imsg.enrich.mlx_whisper_transcription import (
    WHISPER_TEMPERATURE_SCHEDULE,
    MlxWhisperTranscriptionProvider,
    transcript_text,
)
from imsg.enrich.model_runtime import ModelRuntimeUnavailableError
from imsg.enrich.provider import TranscriptionProvider
from imsg.errors import EnrichmentError

REPO = "example-org/whisper-large-v3-mlx"


@dataclass
class WhisperStub:
    result: Any = field(default_factory=lambda: {"text": "", "segments": [], "language": "en"})
    exception: Exception | None = None
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)


@pytest.fixture
def whisper(monkeypatch: pytest.MonkeyPatch) -> WhisperStub:
    stub = WhisperStub()

    def transcribe(*args: Any, **kwargs: Any) -> Any:
        stub.calls.append((args, kwargs))
        if stub.exception is not None:
            raise stub.exception
        return stub.result

    module = types.ModuleType("mlx_whisper")
    module.transcribe = transcribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    return stub


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


# --------------------------------------------------------------------------
# result flattening (pure)
# --------------------------------------------------------------------------


def test_transcript_text_concatenates_segments_and_normalises_whitespace() -> None:
    result = {
        "text": " ignored when segments exist",
        "segments": [
            {"text": " Hello there,"},
            {"text": "  this is\n\ta   voice memo. "},
            {"text": ""},
            {"text": " Bye."},
        ],
    }
    assert transcript_text(result) == "Hello there, this is a voice memo. Bye."


def test_transcript_text_falls_back_to_top_level_text_without_segments() -> None:
    assert transcript_text({"text": "  only\n text  "}) == "only text"


def test_transcript_text_is_empty_for_silence() -> None:
    assert transcript_text({"text": "", "segments": []}) == ""
    assert transcript_text({"text": None, "segments": [{"text": None}]}) == ""


# --------------------------------------------------------------------------
# the provider against the fake runtime
# --------------------------------------------------------------------------


def test_model_id_is_repo_at_revision_with_main_as_the_default() -> None:
    assert MlxWhisperTranscriptionProvider(REPO, None).model_id == f"{REPO}@main"
    assert MlxWhisperTranscriptionProvider(REPO, "abc123").model_id == f"{REPO}@abc123"


def test_transcribe_returns_the_flattened_segments(whisper: WhisperStub, wav: Path) -> None:
    whisper.result = {"segments": [{"text": " Testing,"}, {"text": " one two. "}]}
    assert MlxWhisperTranscriptionProvider(REPO, None).transcribe(wav) == "Testing, one two."


def test_transcribe_passes_wav_path_repo_and_decode_parameters(
    whisper: WhisperStub, wav: Path
) -> None:
    MlxWhisperTranscriptionProvider(REPO, None, language="en").transcribe(wav)
    ((args, kwargs),) = whisper.calls
    assert args == (str(wav),)
    assert kwargs["path_or_hf_repo"] == REPO  # unpinned: the runtime resolves `main` itself
    assert kwargs["language"] == "en"
    assert kwargs["temperature"] == WHISPER_TEMPERATURE_SCHEDULE
    assert kwargs["temperature"][0] == 0.0  # the first pass is greedy; sampling is fallback only
    assert kwargs["word_timestamps"] is False


def test_language_none_is_passed_through_for_auto_detection(
    whisper: WhisperStub, wav: Path
) -> None:
    MlxWhisperTranscriptionProvider(REPO, None).transcribe(wav)
    ((_, kwargs),) = whisper.calls
    assert "language" in kwargs
    assert kwargs["language"] is None


def test_temperature_override_is_passed_through(whisper: WhisperStub, wav: Path) -> None:
    MlxWhisperTranscriptionProvider(REPO, None, temperature=0.0).transcribe(wav)
    ((_, kwargs),) = whisper.calls
    assert kwargs["temperature"] == 0.0


def test_pinned_revision_resolves_a_local_snapshot_once(
    monkeypatch: pytest.MonkeyPatch, whisper: WhisperStub, wav: Path, tmp_path: Path
) -> None:
    hub = install_hub_stub(monkeypatch, tmp_path / "snapshot")
    provider = MlxWhisperTranscriptionProvider(REPO, "abc123")
    provider.transcribe(wav)
    provider.transcribe(wav)
    assert hub.calls == [{"repo_id": REPO, "revision": "abc123"}]
    assert [kwargs["path_or_hf_repo"] for _, kwargs in whisper.calls] == [
        str(tmp_path / "snapshot"),
        str(tmp_path / "snapshot"),
    ]


def test_local_model_directory_is_passed_through(
    monkeypatch: pytest.MonkeyPatch, whisper: WhisperStub, wav: Path, tmp_path: Path
) -> None:
    block_module(monkeypatch, "huggingface_hub")
    local = tmp_path / "whisper-weights"
    local.mkdir()
    MlxWhisperTranscriptionProvider(str(local), "abc123").transcribe(wav)
    ((_, kwargs),) = whisper.calls
    assert kwargs["path_or_hf_repo"] == str(local)


def test_unresolvable_pinned_revision_is_runtime_unavailable(
    monkeypatch: pytest.MonkeyPatch, whisper: WhisperStub, wav: Path, tmp_path: Path
) -> None:
    hub = install_hub_stub(monkeypatch, tmp_path)
    hub.exception = OSError("offline")
    with pytest.raises(ModelRuntimeUnavailableError):
        MlxWhisperTranscriptionProvider(REPO, "abc123").transcribe(wav)
    assert whisper.calls == []


def test_runtime_failure_is_wrapped_as_enrichment_error(
    monkeypatch: pytest.MonkeyPatch, whisper: WhisperStub, wav: Path, tmp_path: Path
) -> None:
    install_hub_stub(monkeypatch, tmp_path)
    whisper.exception = RuntimeError("Metal out of memory")
    with pytest.raises(EnrichmentError) as excinfo:
        MlxWhisperTranscriptionProvider(REPO, "abc123").transcribe(wav)
    message = str(excinfo.value)
    assert "Metal out of memory" in message
    assert f"{REPO}@abc123" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_unexpected_result_shape_is_an_enrichment_error(whisper: WhisperStub, wav: Path) -> None:
    whisper.result = ["not", "a", "dict"]
    with pytest.raises(EnrichmentError) as excinfo:
        MlxWhisperTranscriptionProvider(REPO, None).transcribe(wav)
    assert "list" in str(excinfo.value)


def test_missing_wav_is_an_enrichment_error_without_touching_the_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    block_module(monkeypatch, "mlx_whisper")
    with pytest.raises(EnrichmentError):
        MlxWhisperTranscriptionProvider(REPO, None).transcribe(tmp_path / "missing.wav")


def test_missing_runtime_is_a_clear_error_naming_the_package(
    monkeypatch: pytest.MonkeyPatch, wav: Path
) -> None:
    block_module(monkeypatch, "mlx_whisper")
    provider = MlxWhisperTranscriptionProvider(REPO, None)  # construction never imports it
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        provider.transcribe(wav)
    assert "mlx-whisper" in str(excinfo.value)
    assert not isinstance(excinfo.value, EnrichmentError)


def test_satisfies_the_transcription_provider_protocol() -> None:
    provider: TranscriptionProvider = MlxWhisperTranscriptionProvider(REPO, None)
    assert provider.model_id == f"{REPO}@main"
