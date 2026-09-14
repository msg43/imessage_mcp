"""Whisper transcription through `mlx_whisper` (SPEC §4.1, §8 S5b:
"ffmpeg -> 16 kHz mono WAV -> mlx-whisper large-v3") behind the
`TranscriptionProvider` Protocol. The WAV arrives already normalized by
`imsg.enrich.audio.convert_to_whisper_wav`; this module only runs the
model and flattens its output.

Decoding uses Whisper's standard temperature-fallback schedule
(`WHISPER_TEMPERATURE_SCHEDULE`): the first pass is greedy (temperature
0, deterministic), and higher temperatures are tried for a window only
when Whisper's own quality checks — compression ratio, mean log
probability — reject the greedy result. That fallback is Whisper's
defence against the repetition loops and hallucinated text greedy
decoding produces on noisy or long audio, which describes most voice
memos; dropping it for bit-exact reproducibility would trade transcript
quality for a property nothing downstream needs. Passing
`temperature=0.0` restores fully deterministic decoding if that trade is
ever wanted.

`mlx_whisper` caches the loaded model per model path between calls (its
`ModelHolder`), so one provider serving a whole queue loads the weights
once. The runtime is imported on first use (see
`imsg.enrich.model_runtime`); constructing the provider needs nothing
installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from imsg.enrich.model_runtime import import_runtime_module, resolve_model_snapshot
from imsg.errors import EnrichmentError
from imsg.textnorm import strip_nul

_INSTALL_HINT = "install `mlx-whisper` (Apple silicon only)"

WHISPER_TEMPERATURE_SCHEDULE: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
"""Whisper's own default: greedy first, sampling only as a fallback."""


def collapse_whitespace(text: str) -> str:
    """Collapse every run of whitespace (Whisper's per-segment leading
    spaces, newlines) to one ASCII space and strip the ends."""
    return " ".join(text.split())


def transcript_text(result: Mapping[str, Any]) -> str:
    """Flatten an `mlx_whisper.transcribe` result to one
    whitespace-normalized string: the segment texts in order, or the
    top-level `text` when the result carries no `segments` list. Silence
    flattens to the empty string — a legitimate transcript, not an
    error."""
    segments = result.get("segments")
    if segments is None:
        pieces = [str(result.get("text") or "")]
    else:
        pieces = [str(segment.get("text") or "") for segment in segments]
    return strip_nul(collapse_whitespace(" ".join(pieces)))


class MlxWhisperTranscriptionProvider:
    """`TranscriptionProvider` over `mlx_whisper.transcribe`.

    `model_repo` is a Hugging Face repo id or a local model directory;
    `revision` pins the Hub revision (see
    `imsg.enrich.model_runtime.resolve_model_snapshot`) and is part of
    `model_id` (`<repo>@<revision or 'main'>`). `language` is a Whisper
    language code; `None` lets Whisper detect it per file.
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        *,
        language: str | None = None,
        temperature: float | tuple[float, ...] = WHISPER_TEMPERATURE_SCHEDULE,
    ) -> None:
        self.model_repo = model_repo
        self.revision = revision
        self.language = language
        self.temperature = temperature
        self.model_id = f"{model_repo}@{revision or 'main'}"
        self._model_path: str | None = None

    def _resolved_model_path(self) -> str:
        model_path = self._model_path
        if model_path is None:
            model_path = resolve_model_snapshot(self.model_repo, self.revision)
            self._model_path = model_path
        return model_path

    def transcribe(self, audio_wav_path: Path) -> str:
        if not audio_wav_path.is_file():
            raise EnrichmentError(f"transcription input is not a file: '{audio_wav_path}'")
        mlx_whisper = import_runtime_module("mlx_whisper", install_hint=_INSTALL_HINT)
        model_path = self._resolved_model_path()
        try:
            result = mlx_whisper.transcribe(
                str(audio_wav_path),
                path_or_hf_repo=model_path,
                language=self.language,
                temperature=self.temperature,
                word_timestamps=False,
            )
        except Exception as exc:
            raise EnrichmentError(
                f"mlx_whisper ({self.model_id}) failed on '{audio_wav_path}': {exc}"
            ) from exc
        if not isinstance(result, Mapping):
            raise EnrichmentError(
                f"mlx_whisper ({self.model_id}) returned {type(result).__name__}, not a result "
                f"dict, for '{audio_wav_path}'"
            )
        return transcript_text(result)


__all__ = [
    "WHISPER_TEMPERATURE_SCHEDULE",
    "MlxWhisperTranscriptionProvider",
    "collapse_whitespace",
    "transcript_text",
]
