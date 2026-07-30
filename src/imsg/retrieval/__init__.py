"""The retrieval service (SPEC §9): segment rendering is
`imsg.segment.render`'s job (built by the indexing agent); this
package is the hybrid query flow (§9.4) — FTS5/BM25, the primary text
vector, and the secondary multimodal vector, fused via RRF and
reranked — plus the domain logic behind every SPEC §10.2 tool.

`imsg.mcp.tools` is the only expected caller; it adapts
`RetrievalService`'s plain-Python methods to MCP wire types and the
SPEC §10.1 error model.
"""

from imsg.retrieval.access import LOCAL_FULL_ACCESS, AccessContext, Scope, Surface
from imsg.retrieval.errors import (
    DateRangeInvalidError,
    InvalidArgumentError,
    NotEnrichedError,
    NotFoundError,
    PersonAmbiguousError,
    PersonCandidate,
    PersonNotFoundError,
    RetrievalError,
    ScopeDeniedError,
)
from imsg.retrieval.reranker import FakeRerankerProvider, RerankerProvider
from imsg.retrieval.service import RetrievalService, SearchMessagesResult

__all__ = [
    "LOCAL_FULL_ACCESS",
    "AccessContext",
    "DateRangeInvalidError",
    "FakeRerankerProvider",
    "InvalidArgumentError",
    "NotEnrichedError",
    "NotFoundError",
    "PersonAmbiguousError",
    "PersonCandidate",
    "PersonNotFoundError",
    "RerankerProvider",
    "RetrievalError",
    "RetrievalService",
    "Scope",
    "ScopeDeniedError",
    "SearchMessagesResult",
    "Surface",
]
