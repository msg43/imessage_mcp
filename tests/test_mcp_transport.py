"""Unit tests for `imsg.mcp.transport` — the pure Host/Origin/duplicate-
header functions `imsg.mcp.auth`'s own docstring assigns outside that
module (SPEC §10.4). No ASGI server, no socket, no `mcp` SDK involved."""

from __future__ import annotations

from imsg.mcp.transport import find_duplicate_header, validate_host, validate_origin


def _h(*pairs: tuple[str, str]) -> list[tuple[bytes, bytes]]:
    return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in pairs]


# ---------------------------------------------------------------------------
# find_duplicate_header
# ---------------------------------------------------------------------------


def test_no_duplicates_returns_none() -> None:
    raw = _h(("host", "mcp.fictional.example"), ("authorization", "Bearer x"), ("accept", "*/*"))
    assert find_duplicate_header(raw) is None


def test_duplicate_authorization_is_caught() -> None:
    raw = _h(("authorization", "Bearer a"), ("authorization", "Bearer b"))
    assert find_duplicate_header(raw) == "authorization"


def test_duplicate_host_is_caught() -> None:
    raw = _h(("host", "mcp.fictional.example"), ("host", "evil.fictional.example"))
    assert find_duplicate_header(raw) == "host"


def test_duplicate_origin_is_caught() -> None:
    raw = _h(("origin", "https://a.fictional.example"), ("origin", "https://b.fictional.example"))
    assert find_duplicate_header(raw) == "origin"


def test_duplicate_detection_is_case_insensitive_on_header_name() -> None:
    raw = _h(("Authorization", "Bearer a"), ("authorization", "Bearer b"))
    assert find_duplicate_header(raw) == "authorization"


def test_duplicate_of_an_unguarded_header_is_not_reported() -> None:
    # Accept-Language repeating is unremarkable and not this function's concern.
    raw = _h(("accept-language", "en"), ("accept-language", "fr"))
    assert find_duplicate_header(raw) is None


def test_three_guarded_headers_each_once_is_fine() -> None:
    raw = _h(
        ("authorization", "Bearer x"),
        ("host", "mcp.fictional.example"),
        ("origin", "https://vertexaisearch.fictional.example"),
    )
    assert find_duplicate_header(raw) is None


def test_empty_headers_is_fine() -> None:
    assert find_duplicate_header([]) is None


# ---------------------------------------------------------------------------
# validate_host
# ---------------------------------------------------------------------------


def test_host_missing_is_rejected() -> None:
    assert validate_host(None, ["mcp.fictional.example"]) is False
    assert validate_host("", ["mcp.fictional.example"]) is False


def test_host_exact_match_is_allowed() -> None:
    assert validate_host("mcp.fictional.example", ["mcp.fictional.example"]) is True


def test_host_mismatch_is_rejected() -> None:
    assert validate_host("evil.fictional.example", ["mcp.fictional.example"]) is False


def test_host_match_is_case_insensitive() -> None:
    assert validate_host("MCP.Fictional.EXAMPLE", ["mcp.fictional.example"]) is True


def test_host_no_wildcard_support() -> None:
    """A short, owner-curated allowlist only — no suffix/wildcard match,
    which would widen the DNS-rebinding surface this check exists to
    close (SPEC §10.4)."""
    assert validate_host("sub.mcp.fictional.example", ["mcp.fictional.example"]) is False
    assert validate_host("mcp.fictional.example", ["*.fictional.example"]) is False


def test_host_empty_allowlist_rejects_everything() -> None:
    assert validate_host("mcp.fictional.example", []) is False


# ---------------------------------------------------------------------------
# validate_origin
# ---------------------------------------------------------------------------


def test_origin_absent_is_allowed() -> None:
    """SPEC §10.4: "any present Origin" — absence is legitimate for
    non-browser, server-to-server calls (Gemini Enterprise's own)."""
    assert validate_origin(None, ["https://vertexaisearch.fictional.example"]) is True


def test_origin_present_and_matching_is_allowed() -> None:
    allowed = ["https://vertexaisearch.fictional.example"]
    assert validate_origin("https://vertexaisearch.fictional.example", allowed) is True


def test_origin_present_and_not_allowlisted_is_rejected() -> None:
    allowed = ["https://vertexaisearch.fictional.example"]
    assert validate_origin("https://evil.fictional.example", allowed) is False


def test_origin_empty_string_is_present_and_rejected_unless_listed() -> None:
    # An empty Origin header value is a present-but-empty value, not an
    # absent header — treat it like any other non-matching value.
    assert validate_origin("", ["https://vertexaisearch.fictional.example"]) is False


def test_origin_match_is_exact_not_prefix() -> None:
    allowed = ["https://vertexaisearch.fictional.example"]
    assert validate_origin("https://vertexaisearch.fictional.example.evil.example", allowed) is False
