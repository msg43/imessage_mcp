"""Owner login building blocks: scrypt hashing, 0600 secret files,
sessions that survive restarts and end on a password change, login
throttling that refuses before hashing, and the same-site check."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from imsg.search_page import auth as auth_module
from imsg.search_page.auth import (
    LoginGuard,
    PasswordFile,
    SessionStore,
    cookie_header,
    cookie_is_secure,
    hash_password,
    same_site_request,
    session_cookie_name,
    set_password,
    tokens_match,
    verify_password,
)
from imsg.search_page.errors import SecretFileError
from imsg.search_page.secret_files import data_file, read_private_file, write_private_file

FAST = {"n": 2**14, "r": 8, "p": 1}


def test_scrypt_hash_verifies_only_the_right_password() -> None:
    encoded = hash_password("a long enough password", **FAST)
    assert encoded.startswith("scrypt$n=16384,r=8,p=1$")
    assert verify_password("a long enough password", encoded)
    assert not verify_password("a long enough passwore", encoded)
    assert hash_password("a long enough password", **FAST) != encoded  # salted


@pytest.mark.parametrize(
    "tampered",
    [
        "scrypt$n=1073741824,r=8,p=1$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAA==",
        "scrypt$n=16384,r=99,p=1$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAA==",
        "plain-text-password",
        "",
    ],
)
def test_tampered_hashes_verify_nothing(tampered: str) -> None:
    assert not verify_password("anything at all", tampered)


def test_private_files_are_0600_and_checked(tmp_path: Path) -> None:
    path = tmp_path / "private" / "secret"
    write_private_file(path, b"value")
    assert (path.stat().st_mode & 0o777) == 0o600
    assert (path.parent.stat().st_mode & 0o777) == 0o700
    assert read_private_file(path) == b"value"
    os.chmod(path, 0o644)
    with pytest.raises(SecretFileError, match="0600"):
        read_private_file(path)
    os.chmod(path, 0o600)
    link = tmp_path / "private" / "link"
    link.symlink_to(path)
    with pytest.raises(SecretFileError, match="symlink"):
        read_private_file(link)


def test_data_file_refuses_escapes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "private").symlink_to(outside)
    with pytest.raises(SecretFileError, match="outside"):
        data_file(root, Path("private/owner-password"))
    assert data_file(root, Path("search-page/x")) == (root / "search-page" / "x").resolve()


def test_password_file_reloads_on_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_module, "SCRYPT_N", 2**14)
    path = tmp_path / "pw"
    set_password(path, "first password here")
    file = PasswordFile(path)
    encoded, first = file.load()
    assert verify_password("first password here", encoded)
    set_password(path, "second password here")
    encoded, second = file.load()
    assert second != first and verify_password("second password here", encoded)


def test_short_passwords_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        set_password(tmp_path / "pw", "short")


def test_sessions_persist_expire_and_end_with_the_password(tmp_path: Path) -> None:
    now = [1000.0]
    store_path = tmp_path / "sessions.json"
    store = SessionStore(store_path, lifetime_seconds=100, clock=lambda: now[0])
    raw, session = store.create("fp-1")
    assert store.get(raw, "fp-1") == session
    assert (store_path.stat().st_mode & 0o777) == 0o600
    assert raw not in store_path.read_text()  # only the hash is stored

    reopened = SessionStore(store_path, lifetime_seconds=100, clock=lambda: now[0])
    assert reopened.get(raw, "fp-1") is not None  # survives a restart
    assert reopened.get(raw, "fp-2") is None  # a new password ends it
    assert reopened.get(raw, "fp-1") is None

    raw2, _ = store.create("fp-1")
    now[0] += 101
    assert store.get(raw2, "fp-1") is None  # expired
    raw3, _ = store.create("fp-1")
    store.revoke(raw3)
    assert store.get(raw3, "fp-1") is None
    assert store.get("guess", "fp-1") is None


def test_corrupt_session_store_grants_nothing(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    write_private_file(path, b"{not json")
    assert len(SessionStore(path, lifetime_seconds=100)) == 0


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_login_throttle_refuses_before_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_module, "SCRYPT_N", 2**14)
    path = tmp_path / "pw"
    set_password(path, "the right password")
    clock = _Clock()
    guard = LoginGuard(
        PasswordFile(path),
        max_failures_per_client=3,
        max_failures_global=5,
        window_seconds=60,
        clock=clock,
    )
    calls = []
    real_verify = auth_module.verify_password

    def counting_verify(password: str, encoded: str) -> bool:
        calls.append(password)
        return real_verify(password, encoded)

    monkeypatch.setattr(auth_module, "verify_password", counting_verify)
    for _ in range(3):
        assert guard.attempt("10.0.0.9", "wrong")[0] == "bad_password"
    assert guard.attempt("10.0.0.9", "the right password")[0] == "throttled"
    assert len(calls) == 3  # the throttled attempt never hashed
    # Another address still gets in, until the global budget is spent.
    verdict, fingerprint = guard.attempt("10.0.0.10", "the right password")
    assert verdict == "ok" and fingerprint
    assert guard.attempt("10.0.0.10", "wrong")[0] == "bad_password"
    assert guard.attempt("10.0.0.11", "wrong")[0] == "bad_password"
    assert guard.attempt("10.0.0.12", "the right password")[0] == "throttled"  # 5 global failures
    clock.now += 61
    assert guard.attempt("10.0.0.9", "the right password")[0] == "ok"


@pytest.mark.parametrize(
    ("origin", "fetch_site", "ok"),
    [
        (None, None, True),
        ("http://127.0.0.1:8710", "same-origin", True),
        ("https://mini.example.ts.net", None, False),
        ("http://evil.example", None, False),
        ("null", None, False),
        (None, "cross-site", False),
    ],
)
def test_same_site_request(origin: str | None, fetch_site: str | None, ok: bool) -> None:
    assert same_site_request(host="127.0.0.1:8710", origin=origin, sec_fetch_site=fetch_site) is ok


def test_tokens_match_is_strict() -> None:
    assert tokens_match("abc", "abc")
    assert not tokens_match("abc", "abd")
    assert not tokens_match("", "")
    assert not tokens_match(None, "abc")


def test_cookies_are_httponly_strict_and_secure_over_https() -> None:
    header = cookie_header(session_cookie_name(True), "tok", max_age=60, secure=True)
    assert header.startswith("__Host-imsg_session=tok")
    assert "HttpOnly" in header and "SameSite=Strict" in header and "Secure" in header
    plain = cookie_header(session_cookie_name(False), "tok", max_age=60, secure=False)
    assert "Secure" not in plain and "HttpOnly" in plain
    assert cookie_is_secure("auto", scheme="https", host="a", https_hosts=[])
    assert cookie_is_secure("auto", scheme="http", host="mini.ts.net", https_hosts=["mini.ts.net"])
    assert not cookie_is_secure("auto", scheme="http", host="192.168.1.2:8710", https_hosts=[])
    assert cookie_is_secure("always", scheme="http", host="x", https_hosts=[])
