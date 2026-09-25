"""The `search_page:` config section (D14).

Everything defaults to off: an existing `config.yaml` without this key
loads exactly as before, and the page and the internal model API each
need an explicit `enabled: true`.

Three rules are enforced here, at config load, rather than left to the
operator:

- **Listeners are loopback or local-network addresses only.** Each
  `listen` entry is an IP literal and a port. A wildcard (`0.0.0.0`,
  `::`), a public address, or an address in the Tailscale range
  (100.64.0.0/10) is rejected, so the page can never be bound to an
  interface the tailnet or the internet reaches. The Tailscale-private
  alternative binds loopback and lets `tailscale serve` proxy to it (see
  `imsg.search_page.server`).
- **Ports never collide** with the internal model API's port, and
  `imsg search-page serve` also refuses the public MCP server's port
  (`mcp.public.bind`, 127.0.0.1:8700 by default) at startup, so neither
  the page nor the model API can ever sit on the port the Funnel
  forwards to.
- **Every file path is relative to `paths.data_root`**, with no `..`, so
  the password hash, sessions, model-API secret and thumbnails all land
  on the encrypted volume. Use-time checks (`imsg.search_page.secret_files`)
  also resolve symlinks, so a link planted under data_root cannot point
  outside it.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_PAGE_PORT = 8710
DEFAULT_MODEL_API_PORT = 8711

_TAILSCALE_RANGE = ipaddress.ip_network("100.64.0.0/10")
_HOST_RE = re.compile(r"^[a-z0-9.\-]+(:[0-9]{1,5})?$|^\[[0-9a-f:.]+\](:[0-9]{1,5})?$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _relative_data_path(value: Path, field: str) -> Path:
    """A path that can only resolve beneath `paths.data_root`."""
    text = str(value)
    if value.is_absolute() or text.startswith("~"):
        raise ValueError(f"{field} must be relative to paths.data_root, got {text!r}")
    if ".." in value.parts:
        raise ValueError(f"{field} must not contain '..', got {text!r}")
    if not text.strip() or text == ".":
        raise ValueError(f"{field} must name a file or directory under paths.data_root")
    return value


def parse_listen_address(value: str) -> tuple[str, int]:
    """Parse one `listen` entry (`192.0.2.10:8710`, `127.0.0.1:8710`,
    `[::1]:8710`) and refuse anything but a loopback or local-network IP
    literal. Returns `(host, port)` with IPv6 hosts unbracketed."""
    if value.startswith("["):
        host, sep, port_text = value[1:].partition("]:")
        if not sep:
            raise ValueError(f"listen entry {value!r} must look like '[addr]:port'")
    else:
        host, sep, port_text = value.rpartition(":")
        if not sep or not host:
            raise ValueError(f"listen entry {value!r} must look like 'address:port'")
    if not port_text.isdigit() or not (1 <= int(port_text) <= 65535):
        raise ValueError(f"listen entry {value!r} has no valid port")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            f"listen entry {value!r}: the address must be an IP literal (a host name could "
            f"resolve to an interface nobody meant to expose)"
        ) from exc
    if address.is_unspecified:
        raise ValueError(
            f"listen entry {value!r} is a wildcard address, which would also listen on the "
            f"Tailscale interface and any other network; name the local-network address"
        )
    if address.version == 4 and address in _TAILSCALE_RANGE:
        raise ValueError(
            f"listen entry {value!r} is a Tailscale address; for tailnet access bind loopback "
            f"and use 'tailscale serve' instead (never 'tailscale funnel')"
        )
    if address.is_multicast:
        raise ValueError(f"listen entry {value!r} is a multicast address")
    if not (address.is_loopback or address.is_private or address.is_link_local):
        raise ValueError(
            f"listen entry {value!r} is not a loopback or local-network address; the search "
            f"page is local-network only (D14)"
        )
    return host, int(port_text)


class SemanticSearchConfig(_Strict):
    """Semantic hits: every segment whose vector is at least this similar
    to the query, not a fixed top-k. Similarity is cosine similarity
    (1 - pgvector's `<=>` distance)."""

    enabled: bool = True
    text_min_similarity: float = Field(default=0.45, ge=-1.0, le=1.0)
    """Floor for the text vectors (segments and attachment text chunks).
    A starting value, not a measured one: tune it against labelled
    queries (the relevant / not-relevant toggles build that set)."""
    multimodal_min_similarity: float = Field(default=0.2, ge=-1.0, le=1.0)
    """Floor for the image vectors (PE-Core text-to-image scores run far
    lower than text-to-text ones). Also a starting value."""
    max_hits_per_channel: int = Field(default=2000, ge=1, le=20000)
    """A safety cap per vector channel; the page says when it was reached."""
    ef_search: int = Field(default=200, ge=1, le=1000)
    max_scan_tuples: int = Field(default=50000, ge=1000, le=1_000_000)
    """pgvector's iterative-scan budget per channel: how many index
    tuples one threshold search may visit before it stops."""


class ModelApiConfig(_Strict):
    """The internal model API: hosted by `imsg mcp public`, bound to
    127.0.0.1 only, authenticated by a shared secret in a 0600 file. The
    bind address is not configurable on purpose."""

    enabled: bool = False
    port: int = Field(default=DEFAULT_MODEL_API_PORT, ge=1024, le=65535)
    secret_file: Path = Field(default=Path("private/search-page/model-api.secret"))
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    """How long the page waits for an embedding or rerank answer."""

    @field_validator("secret_file")
    @classmethod
    def _relative(cls, v: Path) -> Path:
        return _relative_data_path(v, "search_page.model_api.secret_file")


class SearchPageConfig(_Strict):
    enabled: bool = False
    listen: list[str] = Field(default_factory=lambda: [f"127.0.0.1:{DEFAULT_PAGE_PORT}"])
    """Addresses to bind: loopback plus the mini's local-network address,
    e.g. `["127.0.0.1:8710", "192.0.2.10:8710"]` (IP literals only)."""
    allowed_hosts: list[str] = Field(
        default_factory=lambda: [f"127.0.0.1:{DEFAULT_PAGE_PORT}", f"localhost:{DEFAULT_PAGE_PORT}"]
    )
    """Exact `Host` header values the page answers; anything else gets 403
    before any other processing (DNS-rebinding defence). Include the
    port whenever it is not 80/443, exactly as a browser sends it."""
    https_hosts: list[str] = Field(default_factory=list)
    """Hosts reached through an HTTPS proxy that forwards plain HTTP to
    loopback (Tailscale Serve). Requests for these hosts get `Secure`
    cookies under `cookie_secure: auto`. Each must also be in
    `allowed_hosts`."""
    cookie_secure: Literal["auto", "always", "never"] = "auto"
    tls_cert_file: Path | None = None
    tls_key_file: Path | None = None
    """Optional TLS for the listeners themselves (for example a
    locally-trusted certificate); both or neither."""
    password_file: Path = Field(default=Path("private/search-page/owner-password"))
    session_file: Path = Field(default=Path("private/search-page/sessions.json"))
    session_days: int = Field(default=30, ge=1, le=365)
    login_max_failures: int = Field(default=5, ge=1, le=100)
    """Failed logins one client address may make per window."""
    login_global_max_failures: int = Field(default=30, ge=1, le=10000)
    """Failed logins all addresses together may make per window."""
    login_window_seconds: int = Field(default=900, ge=10, le=86400)
    thumbnail_dir: Path = Field(default=Path("search-page/thumbnails"))
    fts_max_hits: int = Field(default=20000, ge=100, le=200000)
    """Safety cap on full-text matches per channel; the page says when a
    query reached it. At 20,000 it only binds for near-stopwords."""
    page_threads: int = Field(default=25, ge=1, le=200)
    hits_per_thread: int = Field(default=3, ge=1, le=50)
    """Hits shown per conversation before 'show all'."""
    unindexed_window_days: int = Field(default=60, ge=0, le=3650)
    """Also search messages from the last N days that are not yet in any
    segment (the index lags while heavy stages are paused). 0 disables."""
    rerank_top: int = Field(default=20, ge=1, le=50)
    """How many of the best hits the optional 'rerank' sort sends to the
    reranker. Never the full list."""
    db_pool_size: int = Field(default=4, ge=1, le=16)
    semantic: SemanticSearchConfig = Field(default_factory=SemanticSearchConfig)
    model_api: ModelApiConfig = Field(default_factory=ModelApiConfig)

    @field_validator("listen")
    @classmethod
    def _listen_addresses(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("search_page.listen needs at least one address")
        seen: set[tuple[str, int]] = set()
        for entry in v:
            parsed = parse_listen_address(entry)
            if parsed in seen:
                raise ValueError(f"search_page.listen lists {entry!r} twice")
            seen.add(parsed)
        return v

    @field_validator("allowed_hosts", "https_hosts")
    @classmethod
    def _hosts(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for entry in v:
            lowered = entry.strip().lower()
            if not _HOST_RE.match(lowered):
                raise ValueError(
                    f"host entry {entry!r} must be an exact 'host' or 'host:port' value "
                    f"(no scheme, path or wildcard)"
                )
            out.append(lowered)
        return out

    @field_validator("password_file", "session_file", "thumbnail_dir")
    @classmethod
    def _relative_paths(cls, v: Path) -> Path:
        return _relative_data_path(v, "search_page path")

    @field_validator("tls_cert_file", "tls_key_file")
    @classmethod
    def _relative_tls(cls, v: Path | None) -> Path | None:
        return None if v is None else _relative_data_path(v, "search_page TLS file")

    @model_validator(mode="after")
    def _consistent(self) -> SearchPageConfig:
        if self.enabled and not self.allowed_hosts:
            raise ValueError("search_page.enabled requires a non-empty allowed_hosts list")
        missing = [h for h in self.https_hosts if h not in self.allowed_hosts]
        if missing:
            raise ValueError(f"search_page.https_hosts entries must also be allowed_hosts: {missing}")
        if (self.tls_cert_file is None) != (self.tls_key_file is None):
            raise ValueError("search_page.tls_cert_file and tls_key_file go together")
        ports = {parse_listen_address(entry)[1] for entry in self.listen}
        if self.model_api.port in ports:
            raise ValueError(
                f"search_page.model_api.port {self.model_api.port} collides with a "
                f"search_page.listen port"
            )
        return self

    def listen_addresses(self) -> list[tuple[str, int]]:
        return [parse_listen_address(entry) for entry in self.listen]


__all__ = [
    "DEFAULT_MODEL_API_PORT",
    "DEFAULT_PAGE_PORT",
    "ModelApiConfig",
    "SearchPageConfig",
    "SemanticSearchConfig",
    "parse_listen_address",
]
