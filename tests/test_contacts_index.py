"""ContactsIndex matching — deliberately NOT Postgres-gated.

These live outside `test_identity.py` because that module skips wholesale
without a live Postgres, and `ContactsIndex` is pure in-memory logic. The
first version of these tests was written into the gated file and silently
skipped — a test that cannot run is not a test.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 2026-08-15 — duplicate accounts are not a conflict.
#
# macOS surfaces one human once per configured account, so anyone in both a
# Google and an iCloud address book produced two CNContact identifiers for the
# same number and was refused as "ambiguous". Measured: 1,338 of 1,375 real
# identifiers had every card agreeing on the name; only 37 truly disagreed.
# The refusal fell hardest on the highest-volume correspondents.
# --------------------------------------------------------------------------


def test_find_unique_collapses_duplicate_cards_that_agree_on_the_name() -> None:
    from imsg.stages.identity import ContactRecord, ContactsIndex

    ident = ("+15551234567", "phone")
    google = ContactRecord(
        identifier="google-abc", display_name="Jane Doe", organization=None,
        normalized_identifiers=(ident,),
    )
    icloud = ContactRecord(
        identifier="icloud-xyz", display_name="Jane  Doe", organization=None,
        normalized_identifiers=(ident,),
    )
    match = ContactsIndex([google, icloud]).find_unique(*ident)
    assert match is not None, "same person in two accounts must not read as a conflict"
    assert match.display_name.strip() in ("Jane Doe", "Jane  Doe")


def test_find_unique_still_refuses_a_genuine_name_conflict() -> None:
    from imsg.stages.identity import ContactRecord, ContactsIndex

    ident = ("+15559998888", "phone")
    a = ContactRecord(identifier="a", display_name="Jason Haim", organization=None,
                      normalized_identifiers=(ident,))
    b = ContactRecord(identifier="b", display_name="Laura Greer Haim", organization=None,
                      normalized_identifiers=(ident,))
    assert ContactsIndex([a, b]).find_unique(*ident) is None


def test_find_unique_returns_none_when_absent() -> None:
    from imsg.stages.identity import ContactsIndex

    assert ContactsIndex([]).find_unique("+15550000000", "phone") is None


def test_subset_names_collapse_to_the_most_complete() -> None:
    from imsg.stages.identity import ContactRecord, ContactsIndex

    ident = ("+15551110000", "phone")
    short = ContactRecord(identifier="a", display_name="Noel", organization=None,
                          normalized_identifiers=(ident,))
    full = ContactRecord(identifier="b", display_name="Noel Painter", organization=None,
                         normalized_identifiers=(ident,))
    m = ContactsIndex([short, full]).find_unique(*ident)
    assert m is not None and m.display_name == "Noel Painter"


def test_shared_surname_with_different_first_names_is_still_a_conflict() -> None:
    """Real case: a company and its owner on one number ("Nexon Pool" /
    "Roberto Pool"). Neither token set contains the other, so this must NOT
    collapse — the same laxity would fuse two siblings."""
    from imsg.stages.identity import ContactRecord, ContactsIndex

    ident = ("+15552220000", "phone")
    a = ContactRecord(identifier="a", display_name="Nexon Pool", organization=None,
                      normalized_identifiers=(ident,))
    b = ContactRecord(identifier="b", display_name="Roberto Pool", organization=None,
                      normalized_identifiers=(ident,))
    assert ContactsIndex([a, b]).find_unique(*ident) is None


def test_emoji_decoration_is_not_a_different_person() -> None:
    from imsg.stages.identity import ContactRecord, ContactsIndex

    ident = ("+15553330000", "phone")
    a = ContactRecord(identifier="a", display_name="Melissa\U0001F41D?", organization=None,
                      normalized_identifiers=(ident,))
    b = ContactRecord(identifier="b", display_name="Melissa\U0001F41D\U0001F41D",
                      organization=None, normalized_identifiers=(ident,))
    assert ContactsIndex([a, b]).find_unique(*ident) is not None


def test_nickname_equivalence_is_deliberately_not_inferred() -> None:
    """Joe/Joseph is the same person; Chris/Christina is not, and no rule
    distinguishes them. These stay review stubs on purpose."""
    from imsg.stages.identity import ContactRecord, ContactsIndex

    ident = ("+15554440000", "phone")
    a = ContactRecord(identifier="a", display_name="Joe Rubinsztain", organization=None,
                      normalized_identifiers=(ident,))
    b = ContactRecord(identifier="b", display_name="Joseph Rubinsztain", organization=None,
                      normalized_identifiers=(ident,))
    assert ContactsIndex([a, b]).find_unique(*ident) is None


def test_ios_filter_suffixes_are_stripped() -> None:
    """iOS tags a filtered sender as `<id>(filtered)` / `(smsft_*)`. Same
    sender — but untagged it parses as a phone and tagged it does not, so
    the two resolved to different persons. 2,803 persons collapse once this
    is normalized."""
    from imsg.stages.identity import normalize_handle

    plain = normalize_handle("+12402610473", "US")
    for tagged in (
        "+12402610473(filtered)",
        "+12402610473(smsft)",
        "+12402610473(smsft_fi)",
        "+12402610473(smsft_rm)(smsft)",
    ):
        assert normalize_handle(tagged, "US") == plain, tagged
    assert plain[1] == "phone"


def test_short_code_filter_suffix_is_stripped() -> None:
    from imsg.stages.identity import normalize_handle

    assert normalize_handle("24273(smsft_fi)", "US") == normalize_handle("24273", "US")


def test_a_real_number_containing_parens_is_not_corrupted() -> None:
    """This corpus contains the literal handle `(800) 275-2273`. Cutting at
    the first paren would destroy it, so the strip is allowlisted."""
    from imsg.stages.identity import strip_ios_filter_suffix

    assert strip_ios_filter_suffix("(800) 275-2273") == "(800) 275-2273"
    assert strip_ios_filter_suffix("+1 (800) 275-2273") == "+1 (800) 275-2273"
