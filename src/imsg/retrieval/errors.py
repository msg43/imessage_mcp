"""Retrieval-service error hierarchy — SPEC §10.1's error model.

Every code that appears in `RetrievalError.code` below is one of the
eight machine codes SPEC §10.1 defines: "Tool errors return MCP
tool-error content with a stable machine code first line:
`INVALID_ARGUMENT | PERSON_NOT_FOUND | PERSON_AMBIGUOUS |
DATE_RANGE_INVALID | NOT_FOUND | NOT_ENRICHED | SCOPE_DENIED |
RATE_LIMITED | INTERNAL`." `RATE_LIMITED` is a transport-boundary
concern (the public gate in `imsg.mcp.auth`, out of this module's
scope) and is not raised from here.

Retrieval code raises one of these; `imsg.mcp.tools` (the MCP-surface
adapter) catches `RetrievalError` and formats the stable code + message
into MCP tool-error content, and treats anything else as `INTERNAL`
(never letting a raw exception/traceback reach a tool response — SPEC
§10.1: "Public errors never include filesystem paths, SQL, raw
handles, or stack traces").
"""

from __future__ import annotations

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


__all__ = [
    "DateRangeInvalidError",
    "InvalidArgumentError",
    "NotEnrichedError",
    "NotFoundError",
    "PersonAmbiguousError",
    "PersonCandidate",
    "PersonNotFoundError",
    "RetrievalError",
    "ScopeDeniedError",
]
