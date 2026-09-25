"""The search page's client for the internal model API
(`imsg.search_page.model_api_server`), over loopback HTTP.

Every failure — the public server down, its models warming up or
unloaded, a wrong secret, a timeout — becomes `ModelApiUnavailable` with
a short reason the page shows next to its full-text results. Nothing here
retries: the owner can search again, and a retry loop would only delay
the answer that is already on screen.
"""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imsg.search_page.auth import read_secret
from imsg.search_page.errors import ModelApiUnavailable, SecretFileError
from imsg.search_page.model_api_server import BIND_HOST, decode_vector


@dataclass(frozen=True, slots=True)
class EmbedResult:
    text_vector: list[float]
    multimodal_vector: list[float] | None
    timings_ms: dict[str, float]


class ModelApiClient:
    def __init__(self, *, port: int, secret_file: Path, timeout_seconds: float) -> None:
        self._port = port
        self._secret_file = secret_file
        self._timeout = timeout_seconds
        self._secret: str | None = None

    def _authorization(self) -> str:
        if self._secret is None:
            try:
                self._secret = read_secret(self._secret_file)
            except SecretFileError as exc:
                raise ModelApiUnavailable(
                    "the model API secret file is missing or unsafe", retryable=False
                ) from exc
        return f"Bearer {self._secret}"

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Host": f"{BIND_HOST}:{self._port}",
            "Authorization": self._authorization(),
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection(BIND_HOST, self._port, timeout=self._timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            status = response.status
        except TimeoutError as exc:
            raise ModelApiUnavailable("the model API did not answer in time") from exc
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            raise ModelApiUnavailable(
                "the public MCP server's model API is not reachable (is it running?)"
            ) from exc
        finally:
            conn.close()
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, ValueError) as exc:
            raise ModelApiUnavailable("the model API sent an unreadable answer") from exc
        if status == 503:
            phase = data.get("phase", "warming up")
            remaining = data.get("seconds_remaining")
            detail = f"the models are not loaded ({phase}"
            if isinstance(remaining, (int, float)) and remaining > 0:
                detail += f", about {int(remaining)} s left"
            raise ModelApiUnavailable(detail + ")")
        if status == 401:
            self._secret = None
            raise ModelApiUnavailable("the model API refused the shared secret", retryable=False)
        if status != 200 or not isinstance(data, dict):
            raise ModelApiUnavailable(f"the model API answered {status}")
        return data

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health", None)

    def embed(self, text: str, *, multimodal: bool) -> EmbedResult:
        data = self._request("POST", "/v1/embed", {"text": text, "multimodal": multimodal})
        try:
            text_vector = decode_vector(str(data["text_vector"]))
            mm_raw = data.get("multimodal_vector")
            mm_vector = decode_vector(str(mm_raw)) if mm_raw else None
        except (KeyError, ValueError) as exc:
            raise ModelApiUnavailable("the model API sent a malformed vector") from exc
        timings = data.get("timings_ms") or {}
        return EmbedResult(
            text_vector=text_vector,
            multimodal_vector=mm_vector,
            timings_ms={k: float(v) for k, v in timings.items() if isinstance(v, (int, float))},
        )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        data = self._request("POST", "/v1/rerank", {"query": query, "documents": documents})
        scores = data.get("scores")
        if not isinstance(scores, list) or len(scores) != len(documents):
            raise ModelApiUnavailable("the model API sent a malformed rerank answer")
        return [float(s) for s in scores]


__all__ = ["EmbedResult", "ModelApiClient"]
