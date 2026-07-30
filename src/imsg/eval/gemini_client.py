"""Real `GeminiSearchClient` implementation: Discovery Engine
`servingConfigs.search` (SPEC §13.3 target=`gemini`).

**UNVERIFIED AGAINST THE LIVE API.** Like `imsg.export.gcp_transport`,
this has never been exercised against a real Discovery Engine data
store — its first real exercise is Phase 8 (SPEC §15's side-by-side
eval phase), which is gated behind Phase 7's export having actually
pushed content. Nothing in this build's test suite calls Google APIs
over the network: `tests/test_eval_gemini_client.py` only tests the
pure `_search_result_to_document_id` mapping and construction, with a
stand-in client object.

Construction requires an already-resolved `google.auth.credentials.
Credentials` — reuse `imsg.export.gcp_transport.resolve_gcp_credentials`
(same env/Keychain-only contract; never a literal, never config.yaml).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google.api_core.exceptions import GoogleAPIError
from google.cloud import discoveryengine_v1 as discoveryengine

if TYPE_CHECKING:
    from imsg.eval.backend import GeminiSearchClient  # noqa: F401  (documented conformance only)


class GeminiSearchError(Exception):
    """A Discovery Engine search call failed. Distinct from
    `imsg.export.transport.TransportError` because this is a read-only
    eval-time query, not part of the export gate's write path."""


@dataclass
class DiscoveryEngineSearchClient:
    """Implements `imsg.eval.backend.GeminiSearchClient` against the
    real Discovery Engine `SearchServiceClient` (google-cloud-
    discoveryengine). See module docstring: not yet exercised live.
    """

    gcp_project: str
    data_store_id: str
    search_client: discoveryengine.SearchServiceClient
    location: str = "global"
    serving_config: str = "default_search"

    def _serving_config_path(self) -> str:
        return self.search_client.serving_config_path(
            project=self.gcp_project,
            location=self.location,
            data_store=self.data_store_id,
            serving_config=self.serving_config,
        )

    def search(self, query_text: str, *, page_size: int) -> Sequence[str]:
        request = discoveryengine.SearchRequest(
            serving_config=self._serving_config_path(),
            query=query_text,
            page_size=page_size,
        )
        try:
            pager = self.search_client.search(request=request)
            return [str(result.document.id) for result in pager]
        except GoogleAPIError as exc:
            raise GeminiSearchError(
                f"Discovery Engine search failed for query {query_text!r}: {exc}"
            ) from exc

__all__ = ["DiscoveryEngineSearchClient", "GeminiSearchError"]
