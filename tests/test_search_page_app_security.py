"""Access control on the web app, without a database: the Host
allowlist runs before everything, every data route refuses a request
without a session, the login is CSRF-protected and rate-limited, and
state-changing requests need the session's CSRF token.

The connection pool here fails the test if any refused request reaches
the database."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import psycopg
import pytest
from starlette.testclient import TestClient

from imsg.search_page import auth as auth_module
from imsg.search_page.app import AppDeps, ConnectionPool, FtsReaders, build_app
from imsg.search_page.auth import LoginGuard, PasswordFile, SessionStore, set_password
from imsg.search_page.config import SearchPageConfig
from imsg.search_page.media import MediaConverter
from imsg.search_page.search import SearchSettings

PASSWORD = "correct horse battery staple"


def _no_database() -> psycopg.Connection:
    raise AssertionError("this request must not reach the database")


def _no_fts() -> Any:
    raise AssertionError("this request must not reach the full-text index")


def make_app_client(tmp_path: Path, **page: Any) -> TestClient:
    root = tmp_path / "root"
    password_path = root / "private" / "search-page" / "owner-password"
    set_password(password_path, PASSWORD)
    passwords = PasswordFile(password_path)
    deps = AppDeps(
        page=SearchPageConfig(
            **{
                "enabled": True,
                "allowed_hosts": ["testserver", "127.0.0.1:8710"],
                "login_max_failures": 3,
                "login_global_max_failures": 10,
                **page,
            }
        ),
        settings=SearchSettings(
            timezone="UTC",
            index_unsent=False,
            rrf_k=60,
            fts_max_hits=100,
            unindexed_window_days=0,
            semantic_enabled=False,
            multimodal_enabled=False,
            text_min_similarity=0.5,
            multimodal_min_similarity=0.2,
            max_hits_per_channel=10,
            ef_search=40,
            max_scan_tuples=1000,
        ),
        data_root=root,
        pool=ConnectionPool(_no_database, 1),
        fts=FtsReaders(_no_fts, root / "fts" / "fts.db", 1),
        passwords=passwords,
        sessions=SessionStore(root / "private" / "search-page" / "sessions.json", lifetime_seconds=3600),
        login_guard=LoginGuard(passwords, max_failures_per_client=3, max_failures_global=10, window_seconds=900),
        media=MediaConverter(root / "thumbs"),
        model_api=None,
    )
    return TestClient(build_app(deps))


@pytest.fixture(autouse=True)
def _fast_scrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_module, "SCRYPT_N", 2**14)


@pytest.fixture
def app_client(tmp_path: Path) -> TestClient:
    return make_app_client(tmp_path)


def _login_token(client: TestClient) -> str:
    match = re.search(r'name="login_token" value="([^"]+)"', client.get("/login").text)
    assert match is not None
    return match.group(1)


def _login(client: TestClient, password: str = PASSWORD, **extra: Any) -> Any:
    token = _login_token(client)
    return client.post(
        "/login",
        data={"login_token": token, "password": password, "next": "/"},
        follow_redirects=False,
        **extra,
    )


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1:8700", "192.168.1.20:8710", ""])
def test_unknown_host_is_refused_before_anything_else(app_client: TestClient, host: str) -> None:
    for path in ("/", "/login", "/static/app.js", "/att/" + "a" * 64):
        response = app_client.get(path, headers={"Host": host})
        assert response.status_code == 403, (path, host)


def test_duplicate_host_header_is_refused(app_client: TestClient) -> None:
    response = app_client.get(
        "/login", headers=[("Host", "testserver"), ("Host", "evil.example")]  # type: ignore[arg-type]
    )
    assert response.status_code in (400, 403)


@pytest.mark.parametrize(
    "path",
    [
        "/search?q=anything",
        "/search/page?q=anything",
        "/search/thread?q=anything&thread=" + "a" * 64,
        "/api/semantic?q=anything",
        "/api/people?q=al",
        "/thread/" + "a" * 64,
        "/thread/" + "a" * 64 + "/messages?cursor=" + "b" * 64,
        "/att/" + "a" * 64,
        "/att/" + "a" * 64 + "/thumb",
        "/att/" + "a" * 64 + "/view",
        "/att/" + "a" * 64 + "/audio",
        "/att/" + "a" * 64 + "/poster",
        "/",
        "/labels",
        "/message/" + "a" * 64 + "/details",
        "/grade/1",
        "/timeline",
        "/timeline/page?cursor=1.1",
        "/media",
        "/media/page?cursor=1.1.1",
        "/case",
        "/case/1",
        "/case/1/download?fmt=md",
        "/saved",
        "/search/download?q=anything&fmt=csv",
        "/search?sender=me",
    ],
)
def test_every_data_route_refuses_without_a_session(app_client: TestClient, path: str) -> None:
    response = app_client.get(path, follow_redirects=False)
    assert response.status_code in (303, 401), path
    if response.status_code == 303:
        assert response.headers["location"].startswith("/login?next=")


def test_label_write_without_a_session_is_refused(app_client: TestClient) -> None:
    response = app_client.post("/api/label", json={"q": "x", "kind": "segment", "key": "abc", "grade": 2})
    assert response.status_code == 401


def test_login_sets_a_strict_httponly_session_cookie(app_client: TestClient) -> None:
    response = _login(app_client)
    assert response.status_code == 303 and response.headers["location"] == "/"
    cookies = response.headers.get_list("set-cookie")
    session = next(c for c in cookies if c.startswith("imsg_session="))
    assert "HttpOnly" in session and "SameSite=Strict" in session and "Path=/" in session
    assert "Secure" not in session  # plain http on the LAN
    assert app_client.get("/", follow_redirects=False).status_code == 200


def test_login_over_https_uses_a_secure_host_cookie(tmp_path: Path) -> None:
    client = make_app_client(tmp_path)
    client.base_url = client.base_url.copy_with(scheme="https")
    response = _login(client)
    assert response.status_code == 303
    session = next(c for c in response.headers.get_list("set-cookie") if "imsg_session=" in c and "Max-Age=0" not in c)
    assert session.startswith("__Host-imsg_session=") and "Secure" in session


def test_login_needs_the_form_token(app_client: TestClient) -> None:
    app_client.get("/login")
    response = app_client.post(
        "/login", data={"login_token": "forged", "password": PASSWORD, "next": "/"}, follow_redirects=False
    )
    assert response.status_code == 403
    assert "imsg_session=" not in " ".join(response.headers.get_list("set-cookie"))


def test_cross_site_login_post_is_refused(app_client: TestClient) -> None:
    response = _login(app_client, headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_wrong_password_then_rate_limited(app_client: TestClient) -> None:
    for _ in range(3):
        assert _login(app_client, password="not the password").status_code == 401
    response = _login(app_client)  # right password, but the address is throttled
    assert response.status_code == 429
    assert app_client.get("/", follow_redirects=False).status_code == 303


def test_open_redirects_are_refused(tmp_path: Path) -> None:
    for i, target in enumerate(("https://evil.example/", "//evil.example/", "/\\evil.example")):
        client = make_app_client(tmp_path / str(i))
        response = client.post(
            "/login",
            data={"login_token": _login_token(client), "password": PASSWORD, "next": target},
            follow_redirects=False,
        )
        assert response.status_code == 303 and response.headers["location"] == "/"


def test_state_changes_need_the_csrf_token(app_client: TestClient) -> None:
    _login(app_client)
    page = app_client.get("/").text
    token = re.search(r'<meta name="csrf-token" content="([^"]+)"', page)
    assert token is not None
    # No token and a wrong token are refused before the database is touched.
    body = {"q": "x", "kind": "segment", "key": "abc", "grade": 2}
    assert app_client.post("/api/label", json=body).status_code == 403
    assert app_client.post("/api/label", json=body, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert (
        app_client.post(
            "/api/label",
            json=body,
            headers={"X-CSRF-Token": token.group(1), "Sec-Fetch-Site": "cross-site"},
        ).status_code
        == 403
    )
    # Logout needs it too.
    assert app_client.post("/logout", data={"csrf_token": "wrong"}, follow_redirects=False).status_code == 403
    response = app_client.post("/logout", data={"csrf_token": token.group(1)}, follow_redirects=False)
    assert response.status_code == 303
    assert app_client.get("/", follow_redirects=False).status_code == 303


def test_security_headers_on_every_response(app_client: TestClient) -> None:
    for path in ("/login", "/static/app.css"):
        response = app_client.get(path)
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "same-origin"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "'unsafe-inline'" not in response.headers["content-security-policy"]
    assert "server" not in {k.lower() for k in app_client.get("/login").headers}


def test_static_files_are_only_the_two_assets(app_client: TestClient) -> None:
    assert app_client.get("/static/app.js").status_code == 200
    assert app_client.get("/static/..%2Fapp.py").status_code == 404
    assert app_client.get("/static/app.py").status_code == 404


def _signed_in(client: TestClient) -> str:
    """A session minted server-side with the page's own `SessionStore`
    (no password is posted); returns its CSRF token."""
    deps = client.app._app.state.deps  # type: ignore[attr-defined]
    raw, session = deps.sessions.create(deps.passwords.fingerprint())
    client.cookies.set("imsg_session", raw)
    return str(session.csrf_token)


STATE_CHANGING_FORMS = [
    "/grade",
    "/case",
    "/case/1/activate",
    "/case/1/rename",
    "/case/1/notes",
    "/case/1/delete",
    "/case/item/1/note",
    "/case/item/1/remove",
    "/case/search/1/remove",
    "/saved/1/rename",
    "/saved/1/remove",
]
STATE_CHANGING_APIS = [
    "/api/grade", "/api/case/item", "/api/case/search", "/api/case/review", "/api/saved",
]


@pytest.mark.parametrize("path", STATE_CHANGING_FORMS + STATE_CHANGING_APIS)
def test_every_new_state_change_needs_a_session(app_client: TestClient, path: str) -> None:
    response = app_client.post(path, data={"q": "x"}, follow_redirects=False)
    assert response.status_code in (303, 401), path


@pytest.mark.parametrize("path", STATE_CHANGING_FORMS)
def test_every_new_form_needs_the_csrf_token_and_the_same_site(app_client: TestClient, path: str) -> None:
    """Refused before the database is touched (the pool here fails the test
    on any connection)."""
    token = _signed_in(app_client)
    assert app_client.post(path, data={"q": "x"}, follow_redirects=False).status_code == 403
    assert app_client.post(path, data={"csrf_token": "wrong"}, follow_redirects=False).status_code == 403
    cross = app_client.post(
        path,
        data={"csrf_token": token},
        headers={"Origin": "http://evil.example"},
        follow_redirects=False,
    )
    assert cross.status_code == 403


@pytest.mark.parametrize("path", STATE_CHANGING_APIS)
def test_every_new_api_needs_the_csrf_header_and_the_same_site(app_client: TestClient, path: str) -> None:
    token = _signed_in(app_client)
    assert app_client.post(path, json={}).status_code == 403
    assert app_client.post(path, json={}, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    cross = app_client.post(
        path, json={}, headers={"X-CSRF-Token": token, "Sec-Fetch-Site": "cross-site"}
    )
    assert cross.status_code == 403
