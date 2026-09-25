"""Matched-term highlighting that agrees with how the index matched.

The full-text index (`imsg.embed.fts.schema`) tokenizes with FTS5's
`unicode61 remove_diacritics 2`: a token is a run of letters and digits,
case and diacritics are ignored, and a query word matches only whole
tokens. The quoted-phrase path uses FTS5's trigram tokenizer:
case-insensitive substrings, diacritics significant. Emoji queries use a
plain `LIKE` substring. `QueryMatcher` reproduces each rule on display
text, so what is highlighted is what matched.

Folding (case, diacritics) is done one character at a time, keeping a map
from each folded position back to the original character, so a match
found in the folded text highlights exactly the original characters.
Everything returned as HTML is escaped first; the only markup this
module ever emits is `<mark>`.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from imsg.retrieval.query import AnalyzedQuery

_TOKEN_RE = re.compile(r"[^\W_]+")
_OBJECT_REPLACEMENT = "￼"
MAX_TERMS = 32


def _fold_token_char(ch: str) -> str:
    decomposed = unicodedata.normalize("NFKD", ch)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _fold_case_char(ch: str) -> str:
    return ch.casefold()


def _identity_char(ch: str) -> str:
    return ch


def _fold(text: str, fold_char: Callable[[str], str]) -> tuple[str, list[int]]:
    """The folded text and, for each folded character, the index of the
    original character it came from."""
    pieces: list[str] = []
    origin: list[int] = []
    for index, ch in enumerate(text):
        folded = fold_char(ch)
        pieces.append(folded)
        origin.extend([index] * len(folded))
    return "".join(pieces), origin


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True, slots=True)
class _Term:
    label: str
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class QueryMatcher:
    """Finds a query's terms in display text, the way the index would."""

    mode: str
    terms: tuple[_Term, ...] = field(default_factory=tuple)

    @classmethod
    def for_query(cls, analyzed: AnalyzedQuery) -> QueryMatcher:
        if analyzed.mode == "bm25":
            terms: list[_Term] = []
            seen: set[str] = set()
            for word in analyzed.phrase.split():
                folded, _ = _fold(word, _fold_token_char)
                subtokens = _TOKEN_RE.findall(folded)
                if not subtokens:
                    continue
                label = " ".join(subtokens)
                if label in seen:
                    continue
                seen.add(label)
                body = r"[\W_]+".join(re.escape(t) for t in subtokens)
                terms.append(_Term(label, re.compile(rf"(?<![^\W_]){body}(?![^\W_])")))
                if len(terms) >= MAX_TERMS:
                    break
            return cls(mode="bm25", terms=tuple(terms))
        if analyzed.mode == "trigram":
            folded, _ = _fold(analyzed.phrase, _fold_case_char)
            return cls(mode="trigram", terms=(_Term(folded, re.compile(re.escape(folded))),))
        return cls(
            mode="emoji", terms=(_Term(analyzed.phrase, re.compile(re.escape(analyzed.phrase))),)
        )

    def _fold_for_mode(self, text: str) -> tuple[str, list[int]]:
        if self.mode == "bm25":
            return _fold(text, _fold_token_char)
        if self.mode == "trigram":
            return _fold(text, _fold_case_char)
        return _fold(text, _identity_char)

    def spans(self, text: str | None) -> list[tuple[int, int]]:
        """Merged `(start, end)` spans of `text` that match any term."""
        if not text or not self.terms:
            return []
        folded, origin = self._fold_for_mode(_normalize_quotes(text))
        spans: list[tuple[int, int]] = []
        for term in self.terms:
            for match in term.pattern.finditer(folded):
                if match.end() <= match.start():
                    continue
                spans.append((origin[match.start()], origin[match.end() - 1] + 1))
        return _merge_spans(spans)

    def matched_terms(self, text: str | None) -> int:
        """How many distinct terms occur in `text`."""
        if not text or not self.terms:
            return 0
        folded, _ = self._fold_for_mode(_normalize_quotes(text))
        return sum(1 for term in self.terms if term.pattern.search(folded))

    def highlight(self, text: str | None) -> str:
        """`text` HTML-escaped, with every match wrapped in `<mark>`."""
        if not text:
            return ""
        clean = display_text(text)
        spans = self.spans(clean)
        if not spans:
            return html.escape(clean)
        out: list[str] = []
        cursor = 0
        for start, end in spans:
            out.append(html.escape(clean[cursor:start]))
            out.append(f"<mark>{html.escape(clean[start:end])}</mark>")
            cursor = end
        out.append(html.escape(clean[cursor:]))
        return "".join(out)

    def snippet(self, text: str | None, *, width: int = 240) -> str:
        """A highlighted excerpt of about `width` characters around the
        first match (or the start, when nothing matches)."""
        if not text:
            return ""
        clean = " ".join(display_text(text).split())
        spans = self.spans(clean)
        if len(clean) <= width:
            return self.highlight(clean)
        start = 0
        if spans:
            first = spans[0][0]
            start = max(0, first - width // 3)
        end = min(len(clean), start + width)
        start = max(0, end - width) if end == len(clean) else start
        excerpt = clean[start:end]
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(clean) else ""
        return html.escape(prefix) + self.highlight(excerpt) + html.escape(suffix)


_QUOTE_FOLDS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)


def _normalize_quotes(text: str) -> str:
    """The index matched `normalize_text`'s copy, which folds curly quotes
    to ASCII. Folding them here is a one-to-one character mapping, so the
    spans still index the original text."""
    return text.translate(_QUOTE_FOLDS)


def display_text(text: str) -> str:
    """Message text as shown: the attachment placeholder character that
    iMessage leaves in the body is dropped."""
    return text.replace(_OBJECT_REPLACEMENT, "").strip()


__all__ = ["QueryMatcher", "display_text"]
