"""Query-side text analysis and routing (SPEC §9.2, §9.4 step 3, D2).

**Query normalization MUST match ingest-time normalization exactly**
(SPEC §9.2, D2) — the indexed copy (`text_normalized`/`rendered_text`)
is NFC-normalized with curly quotes/apostrophes folded to ASCII and
variation selectors stripped before it ever reaches FTS or the
embedder; if the query side skipped that same folding, an iOS-typed
apostrophe (U+2019) in the corpus would never match a straight
apostrophe (U+0027) typed into a search box, and exact-phrase search
would silently return nothing. This module normalizes every query
through the exact same function (`imsg.textnorm.normalize_text`) the
ingest path uses — no separate implementation to drift.

Routing (D2): quoted phrases -> the trigram tables (exact-substring,
>= 3 chars); emoji-bearing queries -> a bounded `LIKE` scan on
`text_original` (unicode61 drops emoji as separators, so they are
untokenizable in the BM25 tables); everything else -> BM25.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from imsg.textnorm import normalize_text

QueryMode = Literal["bm25", "trigram", "emoji"]

TRIGRAM_MIN_CHARS = 3
"""FTS5's own trigram-tokenizer floor (SPEC §7.3/D2: "trigram matches
need >= 3 characters")."""

# Emoji-ish codepoint ranges, deliberately generous rather than exhaustive:
# false positives here just mean an emoji-free-but-symbol-heavy query takes
# the LIKE path instead of BM25, which is still correct (just not maximally
# fast) — false negatives (an emoji query treated as BM25 and therefore
# silently dropped by unicode61) are the failure mode this exists to avoid.
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F1E6, 0x1F1FF),  # regional indicators (flag pairs)
    (0x1F300, 0x1FAFF),  # misc symbols/pictographs .. symbols & pictographs ext-A
    (0x1F000, 0x1F0FF),  # mahjong / dominoes / playing cards
    (0x2600, 0x27BF),  # misc symbols, dingbats
    (0x2300, 0x23FF),  # misc technical (e.g. watch/hourglass)
    (0x2B00, 0x2BFF),  # misc symbols and arrows (e.g. star)
    (0x2190, 0x21FF),  # arrows (rarely search text, harmless to include)
)


def _contains_emoji(text: str) -> bool:
    return any(
        any(lo <= ord(ch) <= hi for lo, hi in _EMOJI_RANGES) for ch in text
    )


def _is_fully_quoted(text: str) -> bool:
    return len(text) >= 2 and text[0] == '"' and text[-1] == '"'


@dataclass(frozen=True, slots=True)
class AnalyzedQuery:
    """The result of routing one raw query string. `phrase` is always
    already `imsg.textnorm.normalize_text`-normalized — callers never
    need to normalize again."""

    raw: str
    mode: QueryMode
    phrase: str


def analyze_query(raw_query: str) -> AnalyzedQuery:
    """Normalize `raw_query` and pick its FTS routing mode.

    Only a *fully*-quoted whole query (`'"exact phrase"'`) routes to
    trigram — a query that merely contains an embedded quote elsewhere
    is treated as an ordinary BM25 query with the quote characters
    stripped (SPEC §9.4 step 3 documents "quoted phrases -> trigram"
    without specifying partial-quote handling; this build's reading,
    flagged as a judgment call, is that only a whole-query quoted
    phrase is unambiguous enough to route specially).
    """
    normalized = normalize_text(raw_query)
    if _contains_emoji(normalized):
        return AnalyzedQuery(raw=raw_query, mode="emoji", phrase=normalized)

    if _is_fully_quoted(normalized):
        inner = normalized[1:-1].strip()
        if len(inner) >= TRIGRAM_MIN_CHARS:
            return AnalyzedQuery(raw=raw_query, mode="trigram", phrase=inner)
        # Too short for the trigram floor — fall through to BM25 on the
        # dequoted text rather than issuing a query FTS5 would reject.
        normalized = inner or normalized

    return AnalyzedQuery(raw=raw_query, mode="bm25", phrase=normalized)


def bm25_match_expression(phrase: str) -> str:
    """A safe FTS5 `MATCH` string for `phrase`: every whitespace-
    separated token individually quoted (embedded `"` doubled per
    FTS5's own escaping rule) and joined with FTS5's implicit `AND` —
    so raw user text (which may contain `-`, `OR`, unbalanced quotes,
    etc.) can never be interpreted as FTS5 query-syntax operators."""
    tokens = phrase.split()
    if not tokens:
        return '""'
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def trigram_match_expression(phrase: str) -> str:
    """A single FTS5 phrase query against a trigram-tokenized table —
    the documented mechanism for accelerated exact-substring matching
    (SPEC §7.3/D2)."""
    return '"' + phrase.replace('"', '""') + '"'


def like_pattern(phrase: str) -> str:
    """A bounded `LIKE` pattern for the emoji path (SPEC §7.3: "route
    to a `LIKE '%…%'` scan"), escaping SQL `LIKE` metacharacters in the
    (already user-supplied) phrase so they match literally."""
    escaped = phrase.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


__all__ = [
    "TRIGRAM_MIN_CHARS",
    "AnalyzedQuery",
    "QueryMode",
    "analyze_query",
    "bm25_match_expression",
    "like_pattern",
    "trigram_match_expression",
]
