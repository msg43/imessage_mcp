"""Tests for scripts/check_public_safety.py, the guard added after real
contact names, a real phone number, and real message snippets were found
in this repo's public history and had to be rewritten out (2026-09-24).

Every fixture below uses fictional stand-ins shaped like the real leaks
described in the readiness review, never real data: a non-555 `+1` number,
a three-word display name, an emoji-decorated first name, and a
brand/bank/recipient commit-message line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_public_safety.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_public_safety", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cps() -> ModuleType:
    return _load_module()


def _hard(findings: list[Any]) -> list[Any]:
    return [f for f in findings if f.hard]


def _soft(findings: list[Any]) -> list[Any]:
    return [f for f in findings if not f.hard]


# ---------------------------------------------------------------------------
# Must-fail seeds (hard findings)
# ---------------------------------------------------------------------------


def test_plus1_number_with_non_555_exchange_is_a_hard_finding(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.txt", "reach the reporter at +1 240 867 5309 any time")
    hard = _hard(findings)
    assert any(f.category == "phone" for f in hard)


def test_three_word_display_name_fixture_is_flagged(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.py", 'display_name="Jordan Alexander Reyes"')
    assert any(f.category == "display-name-three-word" for f in findings)


def test_emoji_decorated_first_name_is_flagged(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.py", 'sender = "Taylor \U0001f389"')
    assert any(f.category == "emoji-name" for f in findings)


def test_brand_bank_recipient_commit_line_is_a_hard_finding(cps: ModuleType) -> None:
    message = "PayPal: paid $75 yesterday\nWells Fargo: outgoing wire, recipient Alice Example\n"
    findings = cps.check_commit_message_shape("deadbeef", message)
    assert findings
    assert all(f.hard for f in findings)
    assert findings[0].category == "commit-message-shape"


def test_email_outside_allowlisted_domain_is_a_hard_finding(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.txt", "contact reporter@realnewsroom.example.net directly")
    hard = _hard(findings)
    assert any(f.category == "email" for f in hard)


def test_users_path_with_unknown_persona_is_a_hard_finding(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.txt", "config lives at /Users/jordanreyes/config.yaml")
    hard = _hard(findings)
    assert any(f.category == "home-path" for f in hard)


# ---------------------------------------------------------------------------
# Must-pass fictional fixtures (this repo's own conventions must stay clean)
# ---------------------------------------------------------------------------


def test_fictional_555_number_area_code_form_is_clean(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.txt", "call +1 555 222 0000 for support")
    assert _hard(findings) == []


def test_fictional_555_number_exchange_form_is_clean(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.txt", "call (415) 555-2671 for support")
    assert _hard(findings) == []


def test_apple_support_number_is_allowlisted(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.txt", "Apple support: (800) 275-2273")
    assert _hard(findings) == []


def test_persona_email_on_a_real_provider_domain_is_clean(cps: ModuleType) -> None:
    # Mirrors tests/test_identity.py's own fixture: a fictional persona's
    # local part paired with a real mail provider, used to test handle
    # normalization case-insensitivity.
    findings = cps.scan_text("fixture.py", 'normalize_handle("Alice.Example@ICLOUD.com", "US")')
    assert _hard(findings) == []


def test_example_domain_email_is_clean(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.py", 'owner_email = "owner@fictional.example"')
    assert _hard(findings) == []


def test_noreply_attribution_email_is_clean(cps: ModuleType) -> None:
    findings = cps.scan_text("commit.txt", "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>")
    assert _hard(findings) == []


def test_persona_users_path_is_clean(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.py", 'home = "/Users/alice/Library/Messages"')
    assert _hard(findings) == []


def test_allowlisted_full_name_pair_produces_no_warning(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.py", 'name="Bob Feldman"')
    assert findings == []


def test_unknown_capitalized_pair_is_report_only_not_hard(cps: ModuleType) -> None:
    findings = cps.scan_text("fixture.py", "the handler calls Jordan Reyes directly")
    hard = _hard(findings)
    soft = _soft(findings)
    assert hard == []
    assert any(f.category == "name-pair" for f in soft)


# ---------------------------------------------------------------------------
# Current-repo acceptance: the actual tree and the real commit range
# ---------------------------------------------------------------------------


def test_all_files_in_this_repo_have_no_hard_findings(cps: ModuleType) -> None:
    repo_root = SCRIPT_PATH.parents[1]
    files = cps.git_ls_files(repo_root)
    findings: list[Any] = []
    for f in files:
        findings.extend(cps.scan_file(repo_root / f))
    assert _hard(findings) == []


def test_main_exits_zero_on_all_files(cps: ModuleType) -> None:
    repo_root = SCRIPT_PATH.parents[1]
    assert cps.main(["--all-files", "--repo-root", str(repo_root)]) == 0


def test_main_exits_zero_on_recent_commit_range(cps: ModuleType) -> None:
    repo_root = SCRIPT_PATH.parents[1]
    assert cps.main(["--commits", "HEAD~20..HEAD", "--repo-root", str(repo_root)]) == 0


def test_main_requires_something_to_scan(cps: ModuleType) -> None:
    with pytest.raises(SystemExit):
        cps.main([])


def test_main_exits_nonzero_for_a_seeded_bad_file(tmp_path: Path, cps: ModuleType) -> None:
    bad = tmp_path / "leak.txt"
    bad.write_text("call the source at +1 240 867 5309")
    assert cps.main([str(bad)]) == 1


def test_strict_promotes_warnings_to_failures(tmp_path: Path, cps: ModuleType) -> None:
    fixture = tmp_path / "names.py"
    fixture.write_text("the handler calls Jordan Reyes directly")
    assert cps.main([str(fixture)]) == 0
    assert cps.main([str(fixture), "--strict"]) == 1
