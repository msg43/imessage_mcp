"""The internal model API hosted by `imsg mcp public` for the search page:
loopback only, shared secret required, browser and rebinding requests
refused, answers 503 while the models are not ready, and never makes the
public server fail to start."""

from __future__ import annotations

import http.client
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from imsg.config.schema import Config
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.retrieval.background_warm_up import WarmUpPhase, WarmUpStatus
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.search_page.errors import ModelApiUnavailable
from imsg.search_page.model_api_client import ModelApiClient
from imsg.search_page.model_api_server import (
    BIND_HOST,
    ModelAccess,
    ModelApiServer,
    ModelApiState,
    decode_vector,
    encode_vector,
    start_model_api_if_enabled,
)
from imsg.search_page.secret_files import write_private_file

SECRET = "x" * 48
INSTRUCTION = "retrieve"


class _Marker:
    def __init__(self) -> None:
        self.entered = 0

    def __enter__(self) -> _Marker:
        self.entered += 1
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _WarmUp:
    def __init__(self, phase: WarmUpPhase) -> None:
        self.phase = phase

    def status(self) -> WarmUpStatus:
        return WarmUpStatus(
            phase=self.phase,
            loading="text embedder" if self.phase is WarmUpPhase.WARMING else None,
            steps_done=0,
            steps_total=3,
            elapsed_seconds=1.0,
            seconds_remaining=30.0 if self.phase is WarmUpPhase.WARMING else 0.0,
            failure=None,
        )

    def wait(self, timeout: float) -> WarmUpStatus:
        return self.status()


class _Unloader:
    def __init__(self) -> None:
        self.calls = 0

    @contextmanager
    def call(self) -> Iterator[None]:
        self.calls += 1
        yield


def _access(marker: _Marker | None = None, *, multimodal: bool = True) -> ModelAccess:
    return ModelAccess(
        text_provider=FakeTextEmbeddingProvider(dim=2048),
        reranker=FakeRerankerProvider(),
        multimodal_provider=FakeMultimodalEmbeddingProvider(dim=1280),
        query_instruction=INSTRUCTION,
        multimodal_enabled=multimodal,
        query_marker=marker,
    )


@pytest.fixture
def api() -> Iterator[tuple[ModelApiServer, _Marker, _Unloader]]:
    marker, unloader = _Marker(), _Unloader()
    state = ModelApiState(
        models=_access(marker),
        secret=SECRET,
        port=0,
        warm_up=None,
        idle_unloader=unloader,  # type: ignore[arg-type]
        log=lambda _line: None,
    )
    server = ModelApiServer(0, state).start()
    try:
        yield server, marker, unloader
    finally:
        server.stop()


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection(BIND_HOST, port, timeout=5)
    payload = json.dumps(body).encode() if body is not None else None
    base = {"Host": f"127.0.0.1:{port}", "Authorization": f"Bearer {SECRET}"}
    if payload is not None:
        base["Content-Type"] = "application/json"
    base.update(headers or {})
    base = {k: v for k, v in base.items() if v != ""}
    conn.request(method, path, body=payload, headers=base)
    response = conn.getresponse()
    data = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, data


def test_binds_loopback_only(api: tuple[ModelApiServer, _Marker, _Unloader]) -> None:
    server, _m, _u = api
    assert BIND_HOST == "127.0.0.1"
    assert server._server.server_address[0] == "127.0.0.1"


def test_shared_secret_is_required(api: tuple[ModelApiServer, _Marker, _Unloader]) -> None:
    server, _m, _u = api
    assert _request(server.port, "GET", "/v1/health", headers={"Authorization": ""})[0] == 401
    assert _request(server.port, "GET", "/v1/health", headers={"Authorization": "Bearer wrong"})[0] == 401
    assert _request(server.port, "GET", "/v1/health", headers={"Authorization": SECRET})[0] == 401
    status, body = _request(server.port, "GET", "/v1/health")
    assert status == 200 and body["ready"] is True


def test_browser_and_rebinding_requests_are_refused(api: tuple[ModelApiServer, _Marker, _Unloader]) -> None:
    server, _m, _u = api
    assert _request(server.port, "GET", "/v1/health", headers={"Host": "evil.example"})[0] == 403
    assert _request(server.port, "GET", "/v1/health", headers={"Host": "127.0.0.1:8700"})[0] == 403
    assert _request(server.port, "GET", "/v1/health", headers={"Origin": "http://127.0.0.1:8710"})[0] == 403


def test_embed_uses_the_shared_models_inside_the_query_marker(
    api: tuple[ModelApiServer, _Marker, _Unloader],
) -> None:
    server, marker, unloader = api
    status, body = _request(server.port, "POST", "/v1/embed", body={"text": "deck stain", "multimodal": True})
    assert status == 200
    expected = FakeTextEmbeddingProvider(dim=2048).embed_query("deck stain", instruction=INSTRUCTION)
    assert decode_vector(body["text_vector"]) == pytest.approx(expected, abs=1e-6)
    assert body["text_dim"] == 2048 and body["multimodal_dim"] == 1280
    assert marker.entered == 2  # one per model call: enrichment yields to both
    assert unloader.calls == 1  # the models cannot be dropped mid-call
    status, body = _request(server.port, "POST", "/v1/embed", body={"text": "deck", "multimodal": False})
    assert status == 200 and body["multimodal_vector"] is None


def test_rerank_and_argument_limits(api: tuple[ModelApiServer, _Marker, _Unloader]) -> None:
    server, _m, _u = api
    status, body = _request(server.port, "POST", "/v1/rerank", body={"query": "deck", "documents": ["deck stain", "fish"]})
    assert status == 200 and body["scores"][0] > body["scores"][1]
    assert _request(server.port, "POST", "/v1/rerank", body={"query": "deck", "documents": ["x"] * 51})[0] == 400
    assert _request(server.port, "POST", "/v1/embed", body={"text": ""})[0] == 400
    assert _request(server.port, "POST", "/v1/embed", body={"text": "x" * 1001})[0] == 400
    assert _request(server.port, "POST", "/v1/nothing", body={})[0] == 404


def test_not_ready_answers_503_at_once() -> None:
    state = ModelApiState(
        models=_access(),
        secret=SECRET,
        port=0,
        warm_up=_WarmUp(WarmUpPhase.WARMING),  # type: ignore[arg-type]
        idle_unloader=None,
        log=lambda _line: None,
    )
    server = ModelApiServer(0, state).start()
    try:
        status, body = _request(server.port, "POST", "/v1/embed", body={"text": "deck"})
        assert status == 503 and body["error"] == "WARMING_UP" and body["phase"] == "warming"
    finally:
        server.stop()


def test_client_round_trip_and_failures(api: tuple[ModelApiServer, _Marker, _Unloader], tmp_path: Path) -> None:
    server, _m, _u = api
    secret_file = tmp_path / "secret"
    write_private_file(secret_file, SECRET.encode())
    client = ModelApiClient(port=server.port, secret_file=secret_file, timeout_seconds=5)
    result = client.embed("deck", multimodal=True)
    assert len(result.text_vector) == 2048 and result.multimodal_vector is not None
    assert len(client.rerank("deck", ["a deck", "b"])) == 2

    wrong = tmp_path / "wrong"
    write_private_file(wrong, b"y" * 48)
    with pytest.raises(ModelApiUnavailable, match="refused the shared secret"):
        ModelApiClient(port=server.port, secret_file=wrong, timeout_seconds=5).health()
    with pytest.raises(ModelApiUnavailable, match="secret file"):
        ModelApiClient(port=server.port, secret_file=tmp_path / "missing", timeout_seconds=5).health()
    with pytest.raises(ModelApiUnavailable, match="not reachable"):
        ModelApiClient(port=1, secret_file=secret_file, timeout_seconds=1).health()


def test_vectors_round_trip() -> None:
    values = [0.5, -0.25, 1.0, 0.0]
    assert decode_vector(encode_vector(values)) == values


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _config(config_dict_factory: Any, **model_api: Any) -> Config:
    raw = config_dict_factory()
    raw["search_page"] = {"model_api": {"enabled": True, **model_api}}
    return Config.model_validate(raw)


def _start(cfg: Config, log: list[str]) -> ModelApiServer | None:
    return start_model_api_if_enabled(
        cfg,
        text_provider=FakeTextEmbeddingProvider(dim=2048),
        reranker=FakeRerankerProvider(),
        multimodal_provider=None,
        model_thread=None,
        query_marker=None,
        warm_up=None,
        idle_unloader=None,
        log=log.append,
    )


def test_start_is_off_unless_enabled(config_dict_factory: Any) -> None:
    raw = config_dict_factory()
    log: list[str] = []
    assert _start(Config.model_validate(raw), log) is None and log == []


def test_start_never_raises_on_a_missing_or_unsafe_secret(config_dict_factory: Any, data_root: Path) -> None:
    log: list[str] = []
    assert _start(_config(config_dict_factory, port=_free_port()), log) is None
    assert "not started" in log[0]
    secret = data_root / "private" / "search-page" / "model-api.secret"
    write_private_file(secret, SECRET.encode())
    secret.chmod(0o644)
    log.clear()
    assert _start(_config(config_dict_factory, port=_free_port()), log) is None
    assert "0600" in log[0]
    secret.unlink()
    write_private_file(secret, b"\xff" * 64)  # not text at all
    log.clear()
    assert _start(_config(config_dict_factory, port=_free_port()), log) is None
    assert "not started" in log[0]


def test_start_refuses_the_public_port(config_dict_factory: Any, data_root: Path) -> None:
    write_private_file(data_root / "private" / "search-page" / "model-api.secret", SECRET.encode())
    log: list[str] = []
    assert _start(_config(config_dict_factory, port=8700), log) is None
    assert "public MCP" in log[0]


def test_start_serves_when_enabled(config_dict_factory: Any, data_root: Path) -> None:
    write_private_file(data_root / "private" / "search-page" / "model-api.secret", SECRET.encode())
    log: list[str] = []
    server = _start(_config(config_dict_factory, port=_free_port()), log)
    assert server is not None
    try:
        assert "listening on 127.0.0.1" in log[0]
        assert _request(server.port, "GET", "/v1/health")[0] == 200
    finally:
        server.stop()
