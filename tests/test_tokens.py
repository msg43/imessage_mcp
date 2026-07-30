"""Approximate token counting (chars/4 heuristic; see module docstring
for why this is a deliberate placeholder pending the real tokenizer)."""

from __future__ import annotations

from imsg.tokens import estimate_tokens


def test_empty_string_is_zero_tokens() -> None:
    assert estimate_tokens("") == 0


def test_short_string_is_at_least_one_token() -> None:
    assert estimate_tokens("hi") >= 1


def test_scales_roughly_with_length() -> None:
    short = estimate_tokens("a" * 40)
    long = estimate_tokens("a" * 400)
    assert long > short
    assert long == 10 * short


def test_matches_the_documented_four_chars_per_token_ratio() -> None:
    assert estimate_tokens("a" * 400) == 100


def test_never_returns_zero_for_nonempty_text() -> None:
    assert estimate_tokens("a") >= 1
