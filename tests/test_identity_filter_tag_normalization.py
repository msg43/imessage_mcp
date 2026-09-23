"""iOS filter tags on raw handles (`strip_ios_filter_suffix`, `normalize_handle`).

iOS records a sender whose messages it filtered as `<id>(filtered)` or
`<id>(smsft…)`. The tagged and untagged forms are one sender and must
normalize to one canonical handle, or S3 gives them two persons.

The production index (measured read-only 2026-09-23) holds 3,414
`(filtered)` and 113 `(smsft…)` source handles: `(smsft)`, `(smsft_rm)`,
`(smsft_or)`, `(smsft_fi)` and stacked pairs, all without whitespace. The
first test pins every one of those forms. The others pin forms the corpus
does not contain yet (whitespace between or inside tags, a longer `smsft`
suffix) and the guard that keeps a bare tag from normalizing to an empty
identifier.

Fictional numbers only (public repo).
"""

from __future__ import annotations

import pytest

from imsg.stages.identity import normalize_handle, strip_ios_filter_suffix

PHONE = "+14155552671"
SHORT_CODE = "24273"


@pytest.mark.parametrize(
    "tagged",
    [
        f"{PHONE}(filtered)",
        f"{PHONE}(smsft)",
        f"{PHONE}(smsft_rm)",
        f"{PHONE}(smsft_or)",
        f"{PHONE}(smsft_fi)",
        f"{PHONE}(smsft_rm)(smsft)",
        f"{PHONE}(FILTERED)",
        f"  {PHONE}(filtered)  ",
        f"{PHONE} (filtered)",
        "4155552671(filtered)",
    ],
)
def test_every_tag_form_in_the_corpus_normalizes_to_the_clean_phone(tagged: str) -> None:
    assert normalize_handle(tagged, "US") == (PHONE, "phone")


def test_tagged_short_codes_and_emails_normalize_to_their_clean_form() -> None:
    assert normalize_handle(f"{SHORT_CODE}(smsft_fi)", "US") == (SHORT_CODE, "unknown")
    assert normalize_handle("Alice@Example.com(filtered)", "US") == ("alice@example.com", "email")


@pytest.mark.parametrize(
    "tagged",
    [
        f"{PHONE}(smsft_rm) (smsft)",
        f"{PHONE}( filtered )",
        f"{PHONE} ( smsft_rm ) ( smsft ) ",
        f"{PHONE}(smsft_abc)",
    ],
)
def test_whitespace_inside_and_between_tags_and_long_suffixes_are_stripped(tagged: str) -> None:
    """Not in the measured corpus. Each of these left a tag behind before
    2026-09-23, which made the value unparseable and split the sender."""
    assert normalize_handle(tagged, "US") == (PHONE, "phone")


def test_a_value_that_is_only_a_tag_is_not_emptied() -> None:
    """Stripping `(filtered)` from `(filtered)` would leave `''`, and every
    such sender would share the one empty handle."""
    assert strip_ios_filter_suffix("(filtered)") == "(filtered)"
    assert normalize_handle(" (filtered) ", "US") == ("(filtered)", "unknown")


def test_a_real_number_containing_parens_is_not_touched() -> None:
    assert strip_ios_filter_suffix("(800) 555-0199") == "(800) 555-0199"
    assert strip_ios_filter_suffix("+1 (800) 555-0199") == "+1 (800) 555-0199"
    assert strip_ios_filter_suffix("Acme (Support)") == "Acme (Support)"


def test_has_ios_filter_suffix_tells_tagged_from_untagged() -> None:
    from imsg.stages.identity import has_ios_filter_suffix

    assert has_ios_filter_suffix(f"{PHONE}(filtered)")
    assert has_ios_filter_suffix(f"{SHORT_CODE}(smsft_rm)(smsft)")
    assert has_ios_filter_suffix(f"{PHONE} ( filtered ) ")
    assert not has_ios_filter_suffix(PHONE)
    assert not has_ios_filter_suffix(f" {PHONE} ")
    assert not has_ios_filter_suffix("(800) 555-0199")
    assert not has_ios_filter_suffix("(filtered)")
