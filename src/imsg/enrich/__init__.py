"""S5b — enrichment queue worker (SPEC §8 S5b): routes attachments by
sniffed MIME type to pdftotext/pdftoppm/ffmpeg subprocess tooling
(implemented for real — see each module's docstring) and to the
model-backed OCR/caption/transcription providers (`provider.py` —
Protocol + deterministic fake, no model weights in this build).
`pipeline` is the only module that talks to Postgres.
"""

from __future__ import annotations
