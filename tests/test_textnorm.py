"""Text normalization for indexed copies (SPEC §9.2, D2)."""

from __future__ import annotations

from imsg.textnorm import normalize_text


def test_nfc_normalizes_combining_characters() -> None:
    # 'e' + combining acute accent (NFD) -> precomposed 'é' (NFC)
    decomposed = "café"
    assert normalize_text(decomposed) == "café"


def test_folds_curly_single_quotes_to_ascii() -> None:
    assert normalize_text("don\u2019t") == "don't"
    assert normalize_text("\u2018hello\u2019") == "'hello'"


def test_folds_curly_double_quotes_to_ascii() -> None:
    assert normalize_text("“hello”") == '"hello"'


def test_folds_low_and_reversed_quote_variants() -> None:
    assert normalize_text("\u201a\u201b\u201e\u201f") == "''\"\""


def test_strips_variation_selectors() -> None:
    # U+FE0F is the emoji-presentation variation selector.
    assert normalize_text("hello️ world") == "hello world"


def test_collapses_whitespace_including_newlines() -> None:
    assert normalize_text("line one\n\nline   two\t\ttabbed") == "line one line two tabbed"


def test_trims_leading_and_trailing_whitespace() -> None:
    assert normalize_text("   padded text   ") == "padded text"


def test_leaves_plain_ascii_text_unchanged() -> None:
    assert normalize_text("did the revised bid come through?") == "did the revised bid come through?"


def test_is_idempotent() -> None:
    text = "  \u2018don\u2019t\u2019   say  \n\n \u201chello\u201d\ufe0f  "
    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice


def test_empty_string() -> None:
    assert normalize_text("") == ""
