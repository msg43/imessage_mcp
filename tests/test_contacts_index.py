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
