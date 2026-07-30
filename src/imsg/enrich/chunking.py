"""Attachment-chunk splitting (SPEC §8 S5b): enrichment text (a long
PDF's extracted body, a transcript) is chunked to ~1500 tokens with a
150-token overlap before becoming `attachment_chunk` rows — the same
per-chunk shape S6 embeds and indexes.

Splits on paragraph boundaries when the text has them; falls back to
sentence boundaries for paragraph-less text (transcripts typically have
none); a single unit that alone exceeds the target token budget (one
enormous run-on block) is hard-sliced by an approximate character
budget as a last resort, so no chunk is ever unbounded.
"""

from __future__ import annotations

import re

from imsg.tokens import estimate_tokens

CHUNK_TARGET_TOKENS = 1500
CHUNK_OVERLAP_TOKENS = 150

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


def _hard_slice(unit: str, max_tokens: int) -> list[str]:
    max_chars = max_tokens * 4  # matches imsg.tokens's chars-per-token heuristic
    if len(unit) <= max_chars:
        return [unit]
    return [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)]


def chunk_text(
    text: str,
    *,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split `text` into chunks of roughly `target_tokens`, each
    (after the first) beginning with up to `overlap_tokens` of trailing
    context carried over from the previous chunk. Empty/whitespace-only
    input returns `[]`."""
    stripped = text.strip()
    if not stripped:
        return []

    units: list[str] = []
    for unit in _split_into_units(stripped):
        if estimate_tokens(unit) > target_tokens:
            units.extend(_hard_slice(unit, target_tokens))
        else:
            units.append(unit)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target_tokens:
            chunks.append("\n\n".join(current))
            # Carry trailing units from the just-closed chunk into the next
            # one's overlap window, but never force in a unit that alone
            # already exceeds the overlap budget (a hard-sliced oversized
            # unit, e.g.) — doing so would make `current` start already at
            # or past `target_tokens` before the loop even adds new
            # content, growing chunks without bound. An empty overlap is
            # fine: that just means "no meaningful trailing context to
            # carry", which is already true once a single unit fills the
            # whole overlap budget on its own.
            overlap: list[str] = []
            overlap_tokens_acc = 0
            for prev in reversed(current):
                t = estimate_tokens(prev)
                if overlap_tokens_acc + t > overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_tokens_acc += t
            current = overlap
            current_tokens = overlap_tokens_acc
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


__all__ = ["CHUNK_OVERLAP_TOKENS", "CHUNK_TARGET_TOKENS", "chunk_text"]
