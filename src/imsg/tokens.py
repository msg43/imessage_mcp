"""Approximate token counting shared by S4 (segmentation windowing/caps)
and S5b (attachment-chunk sizing).

No tokenizer dependency is pinned yet — the real count at Phase 3/5
comes from the actual Qwen3 tokenizer loaded alongside the embedding/
boundary models, which this build never loads (see the provider
abstractions in `imsg.segment.boundaries` / `imsg.embed.provider`).
Until then this is a deliberately simple, dependency-free
chars-per-token heuristic (~4 chars/token is the commonly cited
English-text average for BPE-family tokenizers) — good enough to keep
windowing/cap logic exercising realistic boundaries in tests, not
precise enough to be load-bearing for a real token budget. Replace the
implementation, not the call sites, when the real tokenizer lands.
"""

from __future__ import annotations

import math

_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Rough token count for `text`. Empty string -> 0; anything else -> >= 1."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


__all__ = ["estimate_tokens"]
