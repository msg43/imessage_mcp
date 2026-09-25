"""Access control for the search page (D14): one owner login.

The layers, outermost first (wired together in `imsg.search_page.app`):

1. **Host allowlist.** A request whose `Host` header is not exactly one of
   `search_page.allowed_hosts`, or that repeats `Host`, is refused 403/400
   before anything else runs. This is the DNS-rebinding defence: a hostile
   web page cannot point its own name at the mini and read answers.
2. **Session.** Every route except the login page and the static CSS/JS
   needs a valid session cookie. A session is a 256-bit random token; the
   store keeps only its SHA-256, in a 0600 file on the encrypted volume, so
   sessions survive a restart and a copied store holds no usable token.
   Changing the password ends every session (each records the password
   file's fingerprint). Cookies are `HttpOnly`, `SameSite=Strict`, and
   `Secure` whenever the request arrived over HTTPS (TLS on the listener,
   or a host listed in `https_hosts`, such as the Tailscale Serve name).
3. **CSRF.** Every state-changing request (`POST`) must carry the
   session's CSRF token (a header for script calls, a form field for
   forms), compared in constant time, and must not come from another site
   (`Origin` must name this host; `Sec-Fetch-Site: cross-site` is refused).
   The login form uses a double-submit token, since no session exists yet.
4. **Rate-limited login.** Failed logins are counted per client address
   and across all addresses in a sliding window; over either budget the
   login is refused before the password is even hashed, and only one
   hash runs at a time (scrypt is memory-hard by design, so an unthrottled
   flood would also be a memory flood on a memory-sensitive host).

The password is stored as an scrypt hash (N=2^16, r=8, p=1: 64 MiB and
roughly a tenth of a second per check), with its parameters recorded so
they can be raised later. The hash file is 0600 and set only by
`imsg search-page set-password`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from imsg.mcp.ratelimit import SlidingWindowLimiter
from imsg.search_page.errors import SecretFileError
from imsg.search_page.secret_files import check_private_file, read_private_file, write_private_file

# --------------------------------------------------------------------------
# password hashing
# --------------------------------------------------------------------------

SCRYPT_N = 2**16
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SCRYPT_SALT_BYTES = 16
SCRYPT_MAXMEM = 256 * 1024 * 1024
MIN_PASSWORD_CHARS = 12
MAX_PASSWORD_CHARS = 1024

_N_BOUNDS = (2**14, 2**20)
_R_BOUNDS = (1, 16)
_P_BOUNDS = (1, 4)
"""Bounds a stored hash's parameters must sit inside, so a tampered or
corrupt file cannot make a login allocate gigabytes."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def check_password_policy(password: str) -> None:
    if len(password) < MIN_PASSWORD_CHARS:
        raise ValueError(f"the password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(password) > MAX_PASSWORD_CHARS:
        raise ValueError(f"the password must be at most {MAX_PASSWORD_CHARS} characters")


def hash_password(
    password: str, *, n: int | None = None, r: int | None = None, p: int | None = None
) -> str:
    """`scrypt$n=<N>,r=<r>,p=<p>$<salt b64>$<hash b64>`, with the module's
    parameters unless others are given."""
    n = SCRYPT_N if n is None else n
    r = SCRYPT_R if r is None else r
    p = SCRYPT_P if p is None else p
    salt = secrets.token_bytes(SCRYPT_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_DKLEN
    )
    return f"scrypt$n={n},r={r},p={p}${_b64(salt)}${_b64(derived)}"


@dataclass(frozen=True, slots=True)
class _ScryptParams:
    n: int
    r: int
    p: int
    salt: bytes
    derived: bytes


def _parse_hash(encoded: str) -> _ScryptParams:
    parts = encoded.strip().split("$")
    if len(parts) != 4 or parts[0] != "scrypt":
        raise ValueError("not an scrypt password hash")
    fields = dict(item.split("=", 1) for item in parts[1].split(","))
    n, r, p = int(fields["n"]), int(fields["r"]), int(fields["p"])
    if not (_N_BOUNDS[0] <= n <= _N_BOUNDS[1] and n & (n - 1) == 0):
        raise ValueError("scrypt N out of bounds")
    if not (_R_BOUNDS[0] <= r <= _R_BOUNDS[1] and _P_BOUNDS[0] <= p <= _P_BOUNDS[1]):
        raise ValueError("scrypt r/p out of bounds")
    salt, derived = _unb64(parts[2]), _unb64(parts[3])
    if len(salt) < 16 or len(derived) < 16:
        raise ValueError("scrypt salt or hash too short")
    return _ScryptParams(n=n, r=r, p=p, salt=salt, derived=derived)


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time comparison of `password` against a stored hash. A
    malformed hash verifies nothing."""
    try:
        params = _parse_hash(encoded)
    except (ValueError, KeyError):
        return False
    if len(password) > MAX_PASSWORD_CHARS:
        return False
    candidate = hashlib.scrypt(
        password.encode("utf-8"),
        salt=params.salt,
        n=params.n,
        r=params.r,
        p=params.p,
        maxmem=SCRYPT_MAXMEM,
        dklen=len(params.derived),
    )
    return hmac.compare_digest(candidate, params.derived)


class PasswordFile:
    """The owner's password hash, read from its 0600 file and re-read
    whenever the file changes (so `set-password` takes effect, and ends
    every session, without a restart)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._stamp: tuple[int, int, int] | None = None
        self._encoded = ""
        self._fingerprint = ""

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> tuple[str, str]:
        """`(encoded hash, fingerprint)`; raises `SecretFileError` when the
        file is missing or unsafe, or holds no valid hash."""
        info = check_private_file(self._path)
        stamp = (info.st_ino, info.st_size, info.st_mtime_ns)
        with self._lock:
            if stamp != self._stamp:
                raw = read_private_file(self._path)
                encoded = raw.decode("utf-8", errors="strict").strip()
                try:
                    _parse_hash(encoded)
                except (ValueError, KeyError) as exc:
                    raise SecretFileError(
                        f"{self._path} does not hold a valid password hash; run "
                        f"'imsg search-page set-password'"
                    ) from exc
                self._encoded = encoded
                self._fingerprint = hashlib.sha256(raw).hexdigest()
                self._stamp = stamp
            return self._encoded, self._fingerprint

    def fingerprint(self) -> str:
        return self.load()[1]


def set_password(path: Path, password: str) -> None:
    """Hash and store `password` (the `set-password` command's work)."""
    check_password_policy(password)
    write_private_file(path, (hash_password(password) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

MAX_SESSIONS = 50
TOKEN_BYTES = 32


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Session:
    token_hash: str
    csrf_token: str
    created_at: float
    expires_at: float
    password_fingerprint: str


class SessionStore:
    """Server-side sessions, persisted to a 0600 JSON file. Only token
    hashes are stored; the raw token lives in the owner's cookie."""

    def __init__(
        self,
        path: Path,
        *,
        lifetime_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._lifetime = lifetime_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = read_private_file(self._path)
        except SecretFileError:
            if self._path.exists() or self._path.is_symlink():
                raise
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            rows = payload.get("sessions", []) if isinstance(payload, dict) else []
            loaded = [Session(**row) for row in rows if isinstance(row, dict)]
        except (ValueError, TypeError):
            # A corrupt store ends every session; it never grants one.
            loaded = []
        now = self._clock()
        self._sessions = {s.token_hash: s for s in loaded if s.expires_at > now}

    def _save(self) -> None:
        payload = {"version": 1, "sessions": [asdict(s) for s in self._sessions.values()]}
        write_private_file(self._path, json.dumps(payload, sort_keys=True).encode("utf-8"))

    def create(self, password_fingerprint: str) -> tuple[str, Session]:
        raw = secrets.token_urlsafe(TOKEN_BYTES)
        now = self._clock()
        session = Session(
            token_hash=_token_hash(raw),
            csrf_token=secrets.token_urlsafe(TOKEN_BYTES),
            created_at=now,
            expires_at=now + self._lifetime,
            password_fingerprint=password_fingerprint,
        )
        with self._lock:
            self._sessions = {k: s for k, s in self._sessions.items() if s.expires_at > now}
            while len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions.values(), key=lambda s: s.created_at)
                del self._sessions[oldest.token_hash]
            self._sessions[session.token_hash] = session
            self._save()
        return raw, session

    def get(self, raw_token: str | None, password_fingerprint: str) -> Session | None:
        if not raw_token or len(raw_token) > 200:
            return None
        key = _token_hash(raw_token)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            if session.expires_at <= self._clock() or not hmac.compare_digest(
                session.password_fingerprint, password_fingerprint
            ):
                del self._sessions[key]
                self._save()
                return None
            return session

    def revoke(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        with self._lock:
            if self._sessions.pop(_token_hash(raw_token), None) is not None:
                self._save()

    def revoke_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions = {}
            self._save()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


# --------------------------------------------------------------------------
# login throttling
# --------------------------------------------------------------------------

LoginVerdict = Literal["ok", "bad_password", "throttled"]
GLOBAL_KEY = "*"


class LoginGuard:
    """Checks a login attempt: throttle first, then verify the password,
    one verification at a time."""

    def __init__(
        self,
        password_file: PasswordFile,
        *,
        max_failures_per_client: int,
        max_failures_global: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._password_file = password_file
        self._per_client = SlidingWindowLimiter(
            max_failures_per_client, window_seconds=window_seconds, clock=clock
        )
        self._global = SlidingWindowLimiter(
            max_failures_global, window_seconds=window_seconds, clock=clock
        )
        self._verify_lock = threading.Lock()

    def attempt(self, client: str, password: str) -> tuple[LoginVerdict, str]:
        """`(verdict, password fingerprint)`; the fingerprint is empty
        unless the verdict is `ok`."""
        if not self._per_client.would_allow(client) or not self._global.would_allow(GLOBAL_KEY):
            return "throttled", ""
        encoded, fingerprint = self._password_file.load()
        with self._verify_lock:
            ok = verify_password(password, encoded)
        if ok:
            return "ok", fingerprint
        self._per_client.note(client)
        self._global.note(GLOBAL_KEY)
        return "bad_password", ""


# --------------------------------------------------------------------------
# CSRF and request-origin checks
# --------------------------------------------------------------------------


def tokens_match(expected: str | None, provided: str | None) -> bool:
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def same_site_request(*, host: str, origin: str | None, sec_fetch_site: str | None) -> bool:
    """False when the browser says this state-changing request came from
    another site. A request with neither header (curl, a script) passes
    here and still needs the CSRF token."""
    if sec_fetch_site is not None and sec_fetch_site.lower() == "cross-site":
        return False
    if origin is None:
        return True
    if origin == "null":
        return False
    scheme, sep, rest = origin.partition("://")
    if not sep or scheme not in ("http", "https"):
        return False
    return rest.lower() == host.lower()


def new_login_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


# --------------------------------------------------------------------------
# cookies
# --------------------------------------------------------------------------

SESSION_COOKIE = "imsg_session"
SECURE_SESSION_COOKIE = "__Host-imsg_session"
LOGIN_COOKIE = "imsg_login"
SECURE_LOGIN_COOKIE = "__Host-imsg_login"


def cookie_is_secure(
    mode: Literal["auto", "always", "never"], *, scheme: str, host: str, https_hosts: list[str]
) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return scheme == "https" or host.lower() in https_hosts


def session_cookie_name(secure: bool) -> str:
    return SECURE_SESSION_COOKIE if secure else SESSION_COOKIE


def login_cookie_name(secure: bool) -> str:
    return SECURE_LOGIN_COOKIE if secure else LOGIN_COOKIE


def cookie_header(name: str, value: str, *, max_age: int, secure: bool) -> str:
    """One `Set-Cookie` value: HttpOnly, SameSite=Strict, Path=/, and
    Secure when the request came over HTTPS. `__Host-` names require
    Secure, Path=/ and no Domain, which this always satisfies."""
    parts = [f"{name}={value}", "Path=/", "HttpOnly", "SameSite=Strict", f"Max-Age={max_age}"]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def generate_secret() -> str:
    """A fresh shared secret for the internal model API."""
    return secrets.token_urlsafe(48)


def write_new_secret(path: Path) -> None:
    write_private_file(path, (generate_secret() + "\n").encode("ascii"))


def read_secret(path: Path) -> str:
    value = read_private_file(path).decode("utf-8").strip()
    if len(value) < 32:
        raise SecretFileError(f"{path} holds a secret shorter than 32 characters")
    return value


def client_label(client: tuple[str, int] | None) -> str:
    return client[0] if client else "unknown"


__all__ = [
    "LOGIN_COOKIE",
    "MIN_PASSWORD_CHARS",
    "SECURE_LOGIN_COOKIE",
    "SECURE_SESSION_COOKIE",
    "SESSION_COOKIE",
    "LoginGuard",
    "PasswordFile",
    "Session",
    "SessionStore",
    "check_password_policy",
    "client_label",
    "cookie_header",
    "cookie_is_secure",
    "generate_secret",
    "hash_password",
    "login_cookie_name",
    "new_login_token",
    "read_secret",
    "same_site_request",
    "session_cookie_name",
    "set_password",
    "tokens_match",
    "verify_password",
    "write_new_secret",
]

