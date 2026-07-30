"""Unit tests for `imsg.eval.gemini_client.DiscoveryEngineSearchClient`
(SPEC §13.3, target=`gemini`) — no network access: `search_client` is a
hand-written stand-in implementing only `serving_config_path` and
`search`. See the module's own docstring: unverified against the live
API, this only checks request construction and response mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from google.api_core.exceptions import PermissionDenied

from imsg.eval.gemini_client import DiscoveryEngineSearchClient, GeminiSearchError


@dataclass
class _FakeSearchResult:
    document_id: str

    @property
    def document(self) -> _FakeDocument:
        return _FakeDocument(id=self.document_id)


@dataclass
class _FakeDocument:
    id: str


@dataclass
class _FakeSearchServiceClient:
    result_ids: list[str] = field(default_factory=list)
    raise_on_search: Exception | None = None
    captured_requests: list[object] = field(default_factory=list)

    def serving_config_path(
        self, *, project: str, location: str, data_store: str, serving_config: str
    ) -> str:
        return (
            f"projects/{project}/locations/{location}/dataStores/{data_store}/"
            f"servingConfigs/{serving_config}"
        )

    def search(self, *, request: object) -> list[_FakeSearchResult]:
        if self.raise_on_search is not None:
            raise self.raise_on_search
        self.captured_requests.append(request)
        return [_FakeSearchResult(document_id=doc_id) for doc_id in self.result_ids]


def _make_client(search_client: _FakeSearchServiceClient) -> DiscoveryEngineSearchClient:
    return DiscoveryEngineSearchClient(
        gcp_project="example-project",
        data_store_id="example-datastore",
        search_client=search_client,  # type: ignore[arg-type]
    )


def test_search_maps_results_to_document_ids_in_order() -> None:
    search_client = _FakeSearchServiceClient(result_ids=["d" + "a" * 62, "d" + "b" * 62])
    client = _make_client(search_client)
    ids = client.search("deck rebuild", page_size=10)
    assert list(ids) == ["d" + "a" * 62, "d" + "b" * 62]
    assert len(search_client.captured_requests) == 1


def test_search_empty_results() -> None:
    search_client = _FakeSearchServiceClient(result_ids=[])
    client = _make_client(search_client)
    assert list(client.search("nothing matches", page_size=10)) == []


def _permission_denied(message: str) -> PermissionDenied:
    # google.api_core.exceptions.PermissionDenied's __init__ carries no
    # type annotations even though the api_core package ships py.typed.
    return PermissionDenied(message)  # type: ignore[no-untyped-call]


def test_search_wraps_api_error() -> None:
    search_client = _FakeSearchServiceClient(raise_on_search=_permission_denied("nope"))
    client = _make_client(search_client)
    with pytest.raises(GeminiSearchError, match="search failed"):
        client.search("deck rebuild", page_size=10)
