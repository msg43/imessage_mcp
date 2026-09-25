"""Highlighting marks what the index matched, and never lets message text
become markup."""

from __future__ import annotations

from imsg.retrieval.query import analyze_query
from imsg.search_page.highlight import QueryMatcher, display_text


def _matcher(query: str) -> QueryMatcher:
    return QueryMatcher.for_query(analyze_query(query))


def test_whole_words_case_and_diacritics_insensitive() -> None:
    m = _matcher("cafe deck")
    assert m.highlight("Meet at the Café by the DECK") == (
        "Meet at the <mark>Café</mark> by the <mark>DECK</mark>"
    )
    # Whole tokens only, as unicode61 matches: no match inside "decking".
    assert m.highlight("decking and cafeteria") == "decking and cafeteria"
    assert m.matched_terms("the deck") == 1


def test_curly_apostrophes_match_straight_ones() -> None:
    m = _matcher("can't")
    assert m.highlight("I can\u2019t make it") == "I <mark>can\u2019t</mark> make it"


def test_quoted_phrase_matches_substrings() -> None:
    m = _matcher('"id-rev"')
    assert m.mode == "trigram"
    assert m.highlight("see BID-REV3.pdf") == "see B<mark>ID-REV</mark>3.pdf"


def test_emoji_query() -> None:
    m = _matcher("\U0001f389")
    assert m.highlight("party \U0001f389 tonight") == "party <mark>\U0001f389</mark> tonight"


def test_markup_in_messages_is_escaped() -> None:
    m = _matcher("alert")
    out = m.highlight('<img src=x onerror="alert(1)"> <script>alert(2)</script>')
    assert "<img" not in out and "<script>" not in out
    assert "&lt;img src=x onerror=&quot;<mark>alert</mark>(1)&quot;&gt;" in out
    assert out.count("<mark>") == 2


def test_snippet_centres_on_the_first_match() -> None:
    m = _matcher("rebar")
    text = "lorem " * 100 + "footing rebar spacing" + " ipsum" * 100
    snippet = m.snippet(text, width=80)
    assert "<mark>rebar</mark>" in snippet and snippet.startswith("…") and snippet.endswith("…")


def test_object_replacement_character_is_dropped() -> None:
    assert display_text("￼ look at this") == "look at this"
