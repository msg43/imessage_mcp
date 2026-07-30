"""Text normalization for indexed copies (SPEC §9.2, D2).

Applied to `message.text_normalized` (S2's concern, not this module's
caller) and to segment `rendered_text` immediately before FTS insertion
and embedding (S4/S6, this build's scope) — never to the stored,
returned copy. `text_original` / `segment.rendered_text` are always
kept verbatim; this function only ever touches a transient copy used to
build an index key or an embedding input.

Rules, in order (D2):
1. NFC normalize.
2. Fold curly quotes/apostrophes to ASCII (iOS types U+2019, searchers
   type U+0027 — folding both sides is what makes exact-phrase FTS work
   without tokenizer exotica).
3. Strip Unicode variation selectors (U+FE00-FE0F) — these attach to
   emoji/ideographs and are noise for text search.
4. Collapse whitespace (including newlines) to single ASCII spaces, and
   trim. This is safe for FTS purposes: FTS5 tokenizes on whitespace
   anyway, so newline-vs-space carries no token-boundary information,
   and the *returned* text for display is always the untouched
   original/rendered copy, never this normalized one.
"""

from __future__ import annotations

import re
import unicodedata

_CURLY_TO_ASCII: dict[str, str] = {
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK (iOS apostrophe)
    "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201f": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
}

_VARIATION_SELECTOR_LOW = 0xFE00
_VARIATION_SELECTOR_HIGH = 0xFE0F

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return the normalized, FTS/embedding-ready copy of `text`.

    Deterministic and pure — safe to call repeatedly (idempotent:
    `normalize_text(normalize_text(x)) == normalize_text(x)`).
    """
    folded = unicodedata.normalize("NFC", text)
    folded = "".join(_CURLY_TO_ASCII.get(ch, ch) for ch in folded)
    folded = "".join(
        ch
        for ch in folded
        if not (_VARIATION_SELECTOR_LOW <= ord(ch) <= _VARIATION_SELECTOR_HIGH)
    )
    return _WHITESPACE_RE.sub(" ", folded).strip()


__all__ = ["normalize_text"]
