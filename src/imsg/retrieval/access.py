"""Scope enforcement (SPEC §10.3a, D6) — the single place the
authorization predicate is built, for every retrieval-service entry
point.

"The retrieval repository accepts a non-optional `AccessContext`
(`surface`, subject, effective scope) on every call. Public
`allowlist` scope applies the same eligibility predicate to
`search_messages`, `get_conversation`, `list_people`,
`get_attachment_text`, and any future tool — enforcement lives in one
place, not per-tool." This build's own scope is the local surface
(always `full`), but the retrieval service itself is shared with a
later public-transport build, so `AccessContext` and the predicate
builder are written generically now rather than retrofitted later.

Eligibility predicate for `scope='allowlist'`: SPEC §11.2 defines it
for export ("a segment exports iff every `chat_participant` of its
chat has `allowlist_person.text_allowed = true`") and §10.3a says the
MCP public surface reuses "the same eligibility predicate" — so this
module is that predicate's one implementation, shared by both. `imsg.
export` (a parallel agent's module this wave, not built yet in this
checkout) is expected to converge on the same rule when it lands; this
module does not import from or depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Surface = Literal["local", "public"]
Scope = Literal["full", "allowlist"]


@dataclass(frozen=True, slots=True)
class AccessContext:
    """Non-optional on every retrieval-service call (SPEC §10.3a).

    `subject` is `None` on the local surface (no OAuth subject exists
    there) and the pinned owner subject string on the public surface —
    carried through only for audit logging, never branched on inside
    the retrieval service itself (the *scope*, not the subject, decides
    what is visible; §10.3a: "No repository method callable from the
    public entry point has a default/full context").
    """

    surface: Surface
    scope: Scope
    subject: str | None = None

    def __post_init__(self) -> None:
        if self.surface == "local" and self.scope != "full":
            raise ValueError(
                "AccessContext: the local surface is always full-corpus scope "
                "(SPEC §10.3: 'Full corpus scope') — a local context with "
                "scope='allowlist' is not a supported configuration"
            )


LOCAL_FULL_ACCESS = AccessContext(surface="local", scope="full", subject=None)
"""The only `AccessContext` the local MCP surface ever constructs."""


# --------------------------------------------------------------------------
# SQL predicate fragments
# --------------------------------------------------------------------------

# Every predicate below is a self-contained boolean SQL expression over a
# `segment s` (aliased exactly `s`) already joined into the query, or over a
# `chat_id` parameter — callers pick whichever shape fits their query.

_CHAT_ELIGIBLE_ALLOWLIST_SQL = """
    NOT EXISTS (
        SELECT 1 FROM chat_participant cp
        WHERE cp.chat_id = {chat_id_expr}
          AND NOT EXISTS (
              SELECT 1 FROM allowlist_person ap
              WHERE ap.person_id = cp.person_id AND ap.text_allowed
          )
    )
"""
"""SPEC §11.2: "A segment exports iff every `chat_participant` of its
chat has `allowlist_person.text_allowed = true`." Expressed as a
double-negative EXISTS so an empty `chat_participant` set (should never
happen post-S3, but defensively) reads as eligible rather than needing
a separate `COUNT(*) = COUNT(*) FILTER(...)` comparison."""


def segment_eligibility_predicate(context: AccessContext, *, chat_id_expr: str = "s.chat_id") -> str:
    """A boolean SQL expression, safe to AND into any query that has
    `chat_id_expr` in scope (default: a `segment s` alias). Returns the
    literal `TRUE` for full scope — callers can always AND this in
    unconditionally without a separate `if scope == "full"` branch at
    every call site.
    """
    if context.scope == "full":
        return "TRUE"
    return _CHAT_ELIGIBLE_ALLOWLIST_SQL.format(chat_id_expr=chat_id_expr)


__all__ = [
    "LOCAL_FULL_ACCESS",
    "AccessContext",
    "Scope",
    "Surface",
    "segment_eligibility_predicate",
]
