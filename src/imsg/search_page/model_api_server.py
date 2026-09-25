"""The internal model API: lets the search page use the public MCP
server's already-loaded models instead of loading its own (D10.5: one
model set per host, not one per process).

`imsg mcp public` starts it (one call, `start_model_api_if_enabled`) when
`search_page.model_api.enabled` is true. It is a separate listener,
entirely apart from the public MCP app:

- **Loopback only, on its own port.** It binds `127.0.0.1` and nothing
  else — the bind address is not configurable — on
  `search_page.model_api.port` (8711 by default), which config validation
  keeps off the page's ports and startup keeps off the public server's
  port. The Funnel forwards only to 127.0.0.1:8700, so nothing outside the
  host can reach this port; a peer address that is not loopback is refused
  anyway.
- **Shared secret.** Every request must carry `Authorization: Bearer
  <secret>`, the contents of a 0600 file on the encrypted volume
  (`search_page.model_api.secret_file`), compared in constant time.
  Requests with an `Origin` header (browsers) and requests whose `Host` is
  not `127.0.0.1:<port>` / `localhost:<port>` are refused, so a web page
  cannot drive it through DNS rebinding.
- **Nothing here can weaken the public surface.** It shares no code path,
  middleware or port with `imsg.mcp.tools.public_server`; it only calls the
  model providers, on the same model thread the MCP tools use and inside
  the same "query in flight" marker, so enrichment yields to it exactly as
  it yields to MCP searches. A call registers with the idle unloader, so
  the models are never dropped mid-call.
- **It never makes the public server fail.** A missing or unsafe secret
  file, or a port already in use, is logged and the API stays off; the MCP
  server carries on.

When the models are not ready (warming up, unloaded after memory
pressure, refused by memory admission) the API answers 503 at once with
the warm-up phase, and the page shows its full-text results alone.

Endpoints (JSON; vectors travel as base64 of little-endian float32):

- `GET /v1/health` → `{"ready": bool, "phase": str, ...}`
- `POST /v1/embed` `{"text": str, "multimodal": bool}` →
  `{"text_vector": b64, "text_dim": int, "multimodal_vector": b64|null, ...}`
- `POST /v1/rerank` `{"query": str, "documents": [str]}` → `{"scores": [float]}`

It is a standard-library `ThreadingHTTPServer` on a daemon thread: no
event loop shared with uvicorn, nothing to install.
"""

from __future__ import annotations

import base64
import contextlib
import hmac
import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any

from imsg.search_page.auth import read_secret
from imsg.search_page.errors import SearchPageStartupError, SecretFileError
from imsg.search_page.secret_files import data_file

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from imsg.config.schema import Config
    from imsg.embed.provider import MultimodalEmbeddingProvider, TextEmbeddingProvider
    from imsg.retrieval.background_warm_up import BackgroundWarmUp
    from imsg.retrieval.idle_unload import IdleModelUnloader
    from imsg.retrieval.model_thread import ModelThread
    from imsg.retrieval.reranker import RerankerProvider

BIND_HOST = "127.0.0.1"
MAX_BODY_BYTES = 512 * 1024
MAX_QUERY_CHARS = 1000
MAX_RERANK_DOCUMENTS = 50
MAX_DOCUMENT_CHARS = 8000
REQUEST_TIMEOUT_SECONDS = 30.0
READY_WAIT_SECONDS = 0.5
"""How long a call waits for a warm-up that is about to finish before it
answers 503; the page never waits on a cold load."""


def encode_vector(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def decode_vector(text: str) -> list[float]:
    raw = base64.b64decode(text.encode("ascii"), validate=True)
    if len(raw) % 4:
        raise ValueError("vector byte length is not a multiple of 4")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


@dataclass(frozen=True, slots=True)
class ModelAccess:
    """What the API may call: the public server's providers, its model
    thread and its query-in-flight marker."""

    text_provider: TextEmbeddingProvider
    reranker: RerankerProvider
    multimodal_provider: MultimodalEmbeddingProvider | None
    query_instruction: str
    multimodal_enabled: bool
    model_thread: ModelThread | None = None
    query_marker: AbstractContextManager[object] | None = None

    def run[T](self, call: Callable[[], T]) -> T:
        marker = self.query_marker if self.query_marker is not None else contextlib.nullcontext()
        with marker:
            if self.model_thread is None:
                return call()
            return self.model_thread.run(call)


class ModelApiState:
    """Everything a request handler needs; shared by all handler threads."""

    def __init__(
        self,
        *,
        models: ModelAccess,
        secret: str,
        port: int,
        warm_up: BackgroundWarmUp | None,
        idle_unloader: IdleModelUnloader | None,
        log: Callable[[str], None],
    ) -> None:
        self.models = models
        self.secret = secret.encode("utf-8")
        self.port = port
        self.warm_up = warm_up
        self.idle_unloader = idle_unloader
        self.log = log
        self.allowed_hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})

    def bind_port(self, port: int) -> None:
        self.port = port
        self.allowed_hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})

    def authorized(self, header: str | None) -> bool:
        if not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[len("Bearer ") :].encode("utf-8"), self.secret)

    def readiness(self) -> dict[str, Any]:
        if self.warm_up is None:
            return {"ready": True, "phase": "ready"}
        status = self.warm_up.status()
        if not status.settled and status.phase.value == "warming":
            status = self.warm_up.wait(READY_WAIT_SECONDS)
        return {
            "ready": status.phase.value == "ready",
            "phase": status.phase.value,
            "seconds_remaining": round(status.seconds_remaining, 1),
            "loading": status.loading,
        }

    def call(self) -> AbstractContextManager[None]:
        if self.idle_unloader is None:
            return contextlib.nullcontext()
        return self.idle_unloader.call()


class _Handler(BaseHTTPRequestHandler):
    server_version = "imsg-model-api"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> ModelApiState:
        state = getattr(self.server, "state", None)
        assert isinstance(state, ModelApiState)
        return state

    def log_message(self, format: str, *args: Any) -> None:
        # Request lines would carry nothing sensitive, but stay quiet: the
        # MCP server's stderr is for its own events.
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _guard(self) -> bool:
        """Transport checks before anything else: loopback peer, exact
        Host, no browser Origin, the shared secret."""
        peer = self.client_address[0] if self.client_address else ""
        try:
            loopback = ip_address(peer).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            self._send(HTTPStatus.FORBIDDEN, {"error": "FORBIDDEN"})
            return False
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0].lower() not in self.state.allowed_hosts:
            self._send(HTTPStatus.FORBIDDEN, {"error": "FORBIDDEN"})
            return False
        if self.headers.get("Origin") is not None:
            self._send(HTTPStatus.FORBIDDEN, {"error": "FORBIDDEN"})
            return False
        auth = self.headers.get_all("Authorization") or []
        if len(auth) != 1 or not self.state.authorized(auth[0]):
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "UNAUTHORIZED"})
            return False
        return True

    def _body(self) -> dict[str, Any] | None:
        length_header = self.headers.get("Content-Length")
        if length_header is None or not length_header.isdigit():
            self._send(HTTPStatus.LENGTH_REQUIRED, {"error": "LENGTH_REQUIRED"})
            return None
        length = int(length_header)
        if length > MAX_BODY_BYTES:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "TOO_LARGE"})
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "INVALID_JSON"})
            return None
        if not isinstance(payload, dict):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "INVALID_JSON"})
            return None
        return payload

    def do_GET(self) -> None:
        if not self._guard():
            return
        if self.path != "/v1/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        self._send(HTTPStatus.OK, self.state.readiness())

    def do_POST(self) -> None:
        if not self._guard():
            return
        if self.path not in ("/v1/embed", "/v1/rerank"):
            self._send(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        payload = self._body()
        if payload is None:
            return
        state = self.state
        with state.call():
            readiness = state.readiness()
            if not readiness["ready"]:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "WARMING_UP", **readiness})
                return
            try:
                if self.path == "/v1/embed":
                    self._embed(payload)
                else:
                    self._rerank(payload)
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "INVALID_ARGUMENT", "detail": str(exc)})
            except Exception as exc:  # a model failure must not kill the thread
                state.log(f"model api: {type(exc).__name__} during {self.path}")
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "INTERNAL"})

    def _embed(self, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_QUERY_CHARS:
            raise ValueError(f"'text' must be 1-{MAX_QUERY_CHARS} characters")
        want_multimodal = bool(payload.get("multimodal", True))
        models = self.state.models
        started = time.perf_counter()
        text_vector = models.run(
            lambda: models.text_provider.embed_query(text, instruction=models.query_instruction)
        )
        text_ms = (time.perf_counter() - started) * 1000
        multimodal_vector: list[float] | None = None
        mm_ms = 0.0
        provider = models.multimodal_provider
        if want_multimodal and models.multimodal_enabled and provider is not None:
            started = time.perf_counter()
            multimodal_vector = models.run(lambda: provider.embed_text(text))
            mm_ms = (time.perf_counter() - started) * 1000
        self._send(
            HTTPStatus.OK,
            {
                "text_vector": encode_vector(list(text_vector)),
                "text_dim": len(text_vector),
                "multimodal_vector": (
                    encode_vector(list(multimodal_vector)) if multimodal_vector is not None else None
                ),
                "multimodal_dim": len(multimodal_vector) if multimodal_vector is not None else 0,
                "timings_ms": {"text": round(text_ms, 1), "multimodal": round(mm_ms, 1)},
            },
        )

    def _rerank(self, payload: dict[str, Any]) -> None:
        query = payload.get("query")
        documents = payload.get("documents")
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"'query' must be 1-{MAX_QUERY_CHARS} characters")
        if (
            not isinstance(documents, list)
            or not documents
            or len(documents) > MAX_RERANK_DOCUMENTS
            or not all(isinstance(d, str) for d in documents)
        ):
            raise ValueError(f"'documents' must be 1-{MAX_RERANK_DOCUMENTS} strings")
        docs = [d[:MAX_DOCUMENT_CHARS] for d in documents]
        models = self.state.models
        started = time.perf_counter()
        scores = models.run(lambda: models.reranker.score(query, docs))
        self._send(
            HTTPStatus.OK,
            {
                "scores": [float(s) for s in scores],
                "timings_ms": {"rerank": round((time.perf_counter() - started) * 1000, 1)},
            },
        )


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, port: int, state: ModelApiState) -> None:
        self.state = state
        super().__init__((BIND_HOST, port), _Handler)

    def get_request(self) -> tuple[socket.socket, Any]:
        sock, address = super().get_request()
        sock.settimeout(REQUEST_TIMEOUT_SECONDS)
        return sock, address


class ModelApiServer:
    """A running internal model API; `stop()` shuts it down."""

    def __init__(self, port: int, state: ModelApiState) -> None:
        self._server = _Server(port, state)
        # The Host check follows the port actually bound (port 0 picks one).
        state.bind_port(int(self._server.server_address[1]))
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="imsg-model-api", daemon=True
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> ModelApiServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _public_port(cfg: Config) -> int | None:
    _host, _, port = cfg.mcp.public.bind.rpartition(":")
    return int(port) if port.isdigit() else None


def build_model_api(
    cfg: Config,
    *,
    models: ModelAccess,
    warm_up: BackgroundWarmUp | None,
    idle_unloader: IdleModelUnloader | None,
    log: Callable[[str], None],
) -> ModelApiServer:
    """Build (not start) the API; raises `SearchPageStartupError` or
    `SecretFileError` when it must stay off."""
    api = cfg.search_page.model_api
    if api.port == _public_port(cfg):
        raise SearchPageStartupError(
            f"search_page.model_api.port {api.port} is the public MCP server's port"
        )
    secret = read_secret(data_file(cfg.paths.data_root, api.secret_file))
    state = ModelApiState(
        models=models,
        secret=secret,
        port=api.port,
        warm_up=warm_up,
        idle_unloader=idle_unloader,
        log=log,
    )
    return ModelApiServer(api.port, state)


def start_model_api_if_enabled(
    cfg: Config,
    *,
    text_provider: TextEmbeddingProvider,
    reranker: RerankerProvider,
    multimodal_provider: MultimodalEmbeddingProvider | None,
    model_thread: ModelThread | None,
    query_marker: AbstractContextManager[object] | None,
    warm_up: BackgroundWarmUp | None,
    idle_unloader: IdleModelUnloader | None,
    log: Callable[[str], None],
) -> ModelApiServer | None:
    """The one call `imsg mcp public` makes. Returns the running API, or
    `None` when it is disabled or cannot start safely (logged, never
    raised: the public server must not fail because of the page)."""
    if not cfg.search_page.model_api.enabled:
        return None
    models = ModelAccess(
        text_provider=text_provider,
        reranker=reranker,
        multimodal_provider=multimodal_provider,
        query_instruction=cfg.embedding.query_instruction,
        multimodal_enabled=cfg.embedding.multimodal.enabled,
        model_thread=model_thread,
        query_marker=query_marker,
    )
    try:
        server = build_model_api(
            cfg, models=models, warm_up=warm_up, idle_unloader=idle_unloader, log=log
        ).start()
    except (SearchPageStartupError, SecretFileError, OSError) as exc:
        log(f"model api: not started ({exc})")
        return None
    except Exception as exc:  # anything else: still never the public server's problem
        log(f"model api: not started ({type(exc).__name__})")
        return None
    log(f"model api: listening on {BIND_HOST}:{server.port} for the search page")
    return server


__all__ = [
    "BIND_HOST",
    "ModelAccess",
    "ModelApiServer",
    "ModelApiState",
    "build_model_api",
    "decode_vector",
    "encode_vector",
    "start_model_api_if_enabled",
]
