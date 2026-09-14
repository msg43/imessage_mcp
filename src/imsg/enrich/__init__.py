"""S5b — enrichment queue worker (SPEC §8 S5b): routes attachments by
sniffed MIME type to pdftotext/pdftoppm/ffmpeg subprocess tooling
(implemented for real — see each module's docstring) and to the
model-backed OCR/caption/transcription providers (`provider.py` holds
the Protocols and deterministic fakes; `vision_ocr`, `mlx_vlm_caption`
and `mlx_whisper_transcription` are the real ones, importing their
runtimes on first use). `pipeline` is the only module that talks to
Postgres.
"""

from __future__ import annotations
