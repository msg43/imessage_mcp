"""Retrieval-service error hierarchy — SPEC §10.1's error model.

Every code that appears in `RetrievalError.code` below is one of the
machine codes SPEC §10.1 defines: "Tool errors return MCP
tool-error content with a stable machine code first line:
`INVALID_ARGUMENT | PERSON_NOT_FOUND | PERSON_AMBIGUOUS |
DATE_RANGE_INVALID | NOT_FOUND | NOT_ENRICHED | SCOPE_DENIED |
RATE_LIMITED | INTERNAL`." `RATE_LIMITED` is a transport-boundary
concern (the public gate in `imsg.mcp.auth`, out of this module's
scope) and is not raised from here.

Two codes are additions that SPEC §10.1's list does not name yet:
`WARMING_UP` and `WARM_UP_FAILED`, raised on the local surface while its
models load in the background after the server has started answering
(`imsg.retrieval.background_warm_up`). Neither fits an existing code: the
first is transient and worth retrying, the second lasts until the server
is restarted, and `INTERNAL` would say neither.

Retrieval code raises one of these; `imsg.mcp.tools` (the MCP-surface
adapter) catches `RetrievalError` and formats the stable code + message
into MCP tool-error content, and treats anything else as `INTERNAL`
(never letting a raw exception/traceback reach a tool response — SPEC
§10.1: "Public errors never include filesystem paths, SQL, raw
handles, or stack traces").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from imsg.errors import ImsgError


class RetrievalError(ImsgError):
    """Base class for every retrieval-service error. `code` is one of
    the closed SPEC §10.1 machine codes; subclasses set it as a class
    attribute so `imsg.mcp.tools` never has to guess a mapping."""

    code: str = "INTERNAL"


class InvalidArgumentError(RetrievalError):
    code = "INVALID_ARGUMENT"


class DateRangeInvalidError(RetrievalError):
    """`after` is not strictly before `before` (SPEC §9.4 step 1: `after`
    inclusive at local midnight, `before` exclusive — a range where
    `after >= before` can never match anything)."""

    code = "DATE_RANGE_INVALID"


@dataclass(frozen=True, slots=True)
class PersonCandidate:
    """One near-match surfaced by `PERSON_NOT_FOUND`/`PERSON_AMBIGUOUS`
    (SPEC §10.1: "include up to 5 near-matches"; §9.4 step 1: person
    resolution is exact `short_name` -> exact `display_name` -> fuzzy
    candidates, "a non-unique fuzzy match returns `PERSON_AMBIGUOUS`
    with candidates, never a silent pick")."""

    short_name: str
    display_name: str


class PersonNotFoundError(RetrievalError):
    code = "PERSON_NOT_FOUND"

    def __init__(self, query: str, candidates: tuple[PersonCandidate, ...] = ()) -> None:
        self.query = query
        self.candidates = candidates
        super().__init__(f"no person matches {query!r}")


class PersonAmbiguousError(RetrievalError):
    code = "PERSON_AMBIGUOUS"

    def __init__(self, query: str, candidates: tuple[PersonCandidate, ...]) -> None:
        self.query = query
        self.candidates = candidates
        super().__init__(
            f"{query!r} matches more than one person: "
            + ", ".join(c.short_name for c in candidates)
        )


class NotFoundError(RetrievalError):
    """Also the answer for an unauthorized key on the public surface
    (SPEC §10.2 D6: "an unauthorized key returns `NOT_FOUND`, not
    `SCOPE_DENIED`, to avoid an existence oracle") — callers must not
    branch on "exists but unauthorized" vs "does not exist" once this
    is raised; the whole point is that they are indistinguishable from
    outside."""

    code = "NOT_FOUND"


class NotEnrichedError(RetrievalError):
    """Attachment exists (and is authorized) but has no `enrichment`
    row in state `done` yet — SPEC §10.2: "exists, enrichment
    pending/failed (message says which)"."""

    code = "NOT_ENRICHED"


class ScopeDeniedError(RetrievalError):
    """Reserved for the closed SPEC §10.1 code set; not raised by the
    local surface (always full scope) — kept here so a future public
    surface build, and `imsg.mcp.tools`'s error formatter, has exactly
    one place this code is defined, matching every other code in this
    module."""

    code = "SCOPE_DENIED"


class WarmingUpError(RetrievalError):
    """The models are still loading and did not finish within the time
    this call was allowed to wait for them. Transient: the same call
    succeeds once loading finishes."""

    code = "WARMING_UP"

    def __init__(
        self, *, seconds_remaining: int, loading: str | None, wait_bound_seconds: float
    ) -> None:
        self.seconds_remaining = seconds_remaining
        self.loading = loading
        self.wait_bound_seconds = wait_bound_seconds
        what = f"loading the {loading}" if loading else "its models have not started loading"
        super().__init__(
            f"the index is warming up ({what}): about {seconds_remaining} s remaining "
            f"(estimate). Retry this call; each call waits up to {wait_bound_seconds:g} s "
            f"for the models to finish loading"
        )


class HostMemoryBusyError(WarmingUpError):
    """The server did not load its models because the host did not have
    the memory for them (`imsg.memory_admission`). The same retryable
    `WARMING_UP` code, so a client that retries a warming server retries
    this too; the message says the host's memory is busy and when the
    server tries again. `detail` is the admission's own summary: on the
    public surface only numbers and the pressure level, never the text
    of an error (SPEC §10.1: no paths in public errors)."""

    code = "WARMING_UP"

    def __init__(self, *, detail: str, retry_seconds: float, wait_bound_seconds: float) -> None:
        self.seconds_remaining = max(1, math.ceil(retry_seconds))
        self.loading = None
        self.wait_bound_seconds = wait_bound_seconds
        self.detail = detail
        RetrievalError.__init__(
            self,
            f"host memory busy — the index has not loaded its models: {detail}. Retry "
            f"this call later; the server tries loading them again at most every "
            f"{retry_seconds:g} s",
        )


class WarmUpFailedError(RetrievalError):
    """A model failed to load when the server started, so no tool call
    can be answered until the cause is fixed and the server restarts."""

    code = "WARM_UP_FAILED"

    def __init__(self, cause: str) -> None:
        self.cause = cause
        super().__init__(
            f"the index cannot answer: its models failed to load when the server "
            f"started ({cause}). Fix the cause, then restart the MCP server"
        )


__all__ = [
    "DateRangeInvalidError",
    "HostMemoryBusyError",
    "InvalidArgumentError",
    "NotEnrichedError",
    "NotFoundError",
    "PersonAmbiguousError",
    "PersonCandidate",
    "PersonNotFoundError",
    "RetrievalError",
    "ScopeDeniedError",
    "WarmUpFailedError",
    "WarmingUpError",
]
