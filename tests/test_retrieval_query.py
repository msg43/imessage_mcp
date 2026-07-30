"""Unit tests for query-side text analysis and routing (SPEC §9.2,
§9.4 step 3, D2) — no database required.

The single most important property tested here (per the build task):
query-side normalization must produce *exactly* the same text ingest-
time normalization would have produced for the same underlying string,
or exact-phrase search silently breaks — apostrophe folding is the
concrete, real-world failure mode (iOS types U+2019, most keyboards
type U+0027).
"""

from __future__ import annotations

from imsg.retrieval.query import (
    TRIGRAM_MIN_CHARS,
    analyze_query,
    bm25_match_expression,
    like_pattern,
    trigram_match_expression,
)
from imsg.textnorm import normalize_text


def test_query_normalization_matches_ingest_time_normalization_exactly() -> None:
    """The trap: a segment's `rendered_text` is normalized via
    `imsg.textnorm.normalize_text` before indexing (SPEC §9.2). If the
    query side used a *different* normalization (or none), a query
    typed with a straight apostrophe would never match corpus text
    typed with iOS's curly one."""
    ios_curly_apostrophe = "\u2019"  # RIGHT SINGLE QUOTATION MARK, via escape to avoid RUF001
    ingested_raw = f"I can{ios_curly_apostrophe}t make it Friday"  # iOS curly apostrophe
    query_raw = "I can't make it Friday"  # straight apostrophe, as typed

    ingested_normalized = normalize_text(ingested_raw)
    analyzed = analyze_query(query_raw)

    assert analyzed.phrase == ingested_normalized
    assert "'" in analyzed.phrase  # both folded to the same ASCII apostrophe


def test_analyze_query_normalizes_via_the_same_function_ingest_uses() -> None:
    # NFC + curly-quote folding + variation-selector stripping, exactly
    # per imsg.textnorm.normalize_text's own docstring/behavior. Built
    # from explicit code points rather than embedding invisible
    # characters directly in the source, for robustness.
    left_double_quote = "“"
    right_double_quote = "”"
    variation_selector = "️"
    raw = f"cafe {left_double_quote}deal{right_double_quote}{variation_selector}"
    analyzed = analyze_query(raw)
    assert analyzed.phrase == normalize_text(raw)
    assert variation_selector not in analyzed.phrase
    assert analyzed.phrase == 'cafe "deal"'


def test_fully_quoted_query_routes_to_trigram() -> None:
    analyzed = analyze_query('"exact phrase"')
    assert analyzed.mode == "trigram"
    assert analyzed.phrase == "exact phrase"


def test_quoted_but_too_short_for_trigram_falls_back_to_bm25() -> None:
    analyzed = analyze_query('"hi"')
    assert len("hi") < TRIGRAM_MIN_CHARS
    assert analyzed.mode == "bm25"
    assert analyzed.phrase == "hi"


def test_partial_quote_is_not_treated_as_a_whole_query_phrase() -> None:
    # Judgment call documented in the module: only a *fully*-quoted whole
    # query routes to trigram.
    analyzed = analyze_query('find "the deck" please')
    assert analyzed.mode == "bm25"


def test_emoji_query_routes_to_emoji_mode() -> None:
    analyzed = analyze_query("🎉 party")
    assert analyzed.mode == "emoji"


def test_plain_query_routes_to_bm25() -> None:
    analyzed = analyze_query("deck project bid")
    assert analyzed.mode == "bm25"
    assert analyzed.phrase == "deck project bid"


def test_bm25_match_expression_quotes_every_token() -> None:
    expr = bm25_match_expression("deck project")
    assert expr == '"deck" "project"'


def test_bm25_match_expression_escapes_embedded_quotes() -> None:
    expr = bm25_match_expression('say "hi"')
    # Each whitespace-separated token individually quoted; an embedded
    # quote is doubled per FTS5's own escaping rule.
    assert '""hi""' in expr


def test_bm25_match_expression_neutralizes_fts5_operators() -> None:
    # Raw text containing FTS5 syntax characters must never be
    # interpreted as operators (SPEC-adjacent safety property, not
    # explicitly required by the spec text but necessary for any
    # trustworthy implementation of it).
    expr = bm25_match_expression("NOT alice OR bob")
    assert expr == '"NOT" "alice" "OR" "bob"'


def test_trigram_match_expression_wraps_as_one_phrase() -> None:
    assert trigram_match_expression("bid-rev3") == '"bid-rev3"'


def test_like_pattern_escapes_sql_wildcards() -> None:
    pattern = like_pattern("50%_off")
    assert pattern == "%50\\%\\_off%"
