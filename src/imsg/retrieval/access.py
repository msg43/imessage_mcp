"""Scope enforcement (SPEC §10.3a, D6) — the single place every
retrieval-service entry point learns what the caller may read.

"The retrieval repository accepts a non-optional `AccessContext`
(`surface`, subject, effective scope) on every call. Public
`allowlist` scope applies the same eligibility predicate to
`search_messages`, `get_conversation`, `list_people`,
`get_attachment_text`, and any future tool — enforcement lives in one
place, not per-tool."

Each service method turns its `AccessContext` into a `RequestScope` once,
at the start of the request, with `resolve_request_scope`. Every later
step — person resolution, the candidate predicate, conversation windows,
segment text, attachment text — asks that object, never
`context.scope` directly.

**`allowlist` scope is export's rule, not a copy of it.** SPEC §11.2
defines eligibility for export, and §10.3a says the public surface uses
"the same eligibility predicate", so the chats come from
`imsg.export.eligibility.eligible_chat_ids` and attachment text from
`imsg.export.eligibility.compute_attachment_eligibility`. Under
`allowlist` the public surface serves what export would ship:

- only chats `eligible_chat_ids` returns. A chat is denied if any current
  participant, raw participant handle, message sender (a former member
  included) or tapback sender is unresolved or not `text_allowed`, and a
  chat with no participants is denied;
- attachment text — the snippets inside segment text and conversation
  lines, and `get_attachment_text` — only for a (segment, attachment)
  pair whose every non-unsent message link in that segment has a sender
  with `attachments_allowed`. Anything else shows a content-free
  placeholder (`AttachmentSnippet.withheld`);
- no unsent messages and no prior edit versions, whatever `policy.*` says
  (D1, the same unconditional exclusions §11.2 imposes on export);
- only people who appear in an eligible chat, in `list_people` and in
  person-filter resolution and its near-match suggestions (D6).

**Evaluated per request, never cached.** A cached "eligible" can go
stale in ways no cheap key detects — an identity merge repoints
`message.sender_person_id`, a revoked allowlist row, a new message from
an outsider — so the rule runs fresh at the start of every request.
Measured on the real corpus (2026-09-22, read-only, allowlists
simulated, warm runs): 17 ms with the allowlist empty, 12 ms and 21-31
ms for allowlists leaving 2 and 53 eligible chats, and 99-192 ms for
1,061 eligible chats or with every person allowlisted. The candidate
queries then filter on `chat_id = ANY(<eligible ids>)`, which the planner
turns into an exact scan by chat when few chats are eligible.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from imsg.export.eligibility import (
    compute_attachment_eligibility,
    effective_sender_sql,
    eligible_chat_ids,
    owner_person_id,
)

if TYPE_CHECKING:
    import psycopg

Surface = Literal["local", "public"]
Scope = Literal["full", "allowlist"]

_SURFACES: frozenset[str] = frozenset({"local", "public"})
_SCOPES: frozenset[str] = frozenset({"full", "allowlist"})


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
        # The Literal types say this already; checked at runtime too, so
        # a stray value from config or a test fails here rather than
        # falling through to whichever branch a later `if` happens to take.
        if self.surface not in _SURFACES:
            raise ValueError(f"AccessContext: unknown surface {self.surface!r}")
        if self.scope not in _SCOPES:
            raise ValueError(f"AccessContext: unknown scope {self.scope!r}")
        if self.surface == "local" and self.scope != "full":
            raise ValueError(
                "AccessContext: the local surface is always full-corpus scope "
                "(SPEC §10.3: 'Full corpus scope') — a local context with "
                "scope='allowlist' is not a supported configuration"
            )


LOCAL_FULL_ACCESS = AccessContext(surface="local", scope="full", subject=None)
"""The only `AccessContext` the local MCP surface ever constructs."""


SCOPE_CHAT_IDS_PARAM = "scope_eligible_chat_ids"
"""The bind-parameter name `RequestScope.chat_predicate` uses — distinct
from every other parameter the candidate queries bind."""


@dataclass(frozen=True, slots=True)
class ScopePredicate:
    """A boolean SQL expression and the parameters it binds, safe to AND
    into any query that has its chat-id expression in scope."""

    sql: str
    params: dict[str, object]


@dataclass(frozen=True, slots=True)
class RequestScope:
    """One request's authorization, with the allowlist rule already
    evaluated. Build it with `resolve_request_scope`."""

    context: AccessContext
    eligible_chat_ids: frozenset[int] | None
    """`None` under `full` scope: every chat. Under `allowlist`, the
    chats export would ship, evaluated when this request started. A chat
    outside the set is denied — including one that was never evaluated,
    which is how a restricted evaluation (`among_chat_ids`) stays closed."""
    owner_person_id: int | None = None
    """Bound as `%(owner)s` wherever this scope computes an effective
    sender (`imsg.export.eligibility.effective_sender_sql`)."""

    @property
    def is_full(self) -> bool:
        return self.eligible_chat_ids is None

    @property
    def admits_nothing(self) -> bool:
        return self.eligible_chat_ids is not None and not self.eligible_chat_ids

    @property
    def withholds_ineligible_content(self) -> bool:
        """True under `allowlist`: attachment text passes the separate
        attachment gate, and unsent messages and edit history never show.
        Stored `segment.rendered_text` was rendered without either rule,
        so it is re-rendered under such a scope rather than returned."""
        return not self.is_full

    def admits_chat(self, chat_id: int) -> bool:
        return self.eligible_chat_ids is None or chat_id in self.eligible_chat_ids

    def chat_predicate(self, chat_id_expr: str) -> ScopePredicate:
        """A boolean SQL expression over `chat_id_expr`: `TRUE` under
        `full` scope, so callers can always AND it in; under `allowlist`,
        membership in the eligible set."""
        if self.eligible_chat_ids is None:
            return ScopePredicate(sql="TRUE", params={})
        return ScopePredicate(
            sql=f"{chat_id_expr} = ANY(%({SCOPE_CHAT_IDS_PARAM})s::bigint[])",
            params={SCOPE_CHAT_IDS_PARAM: sorted(self.eligible_chat_ids)},
        )

    def shows_unsent(self, policy_index_unsent: bool) -> bool:
        """`policy.index_unsent` under `full` scope; never under
        `allowlist` (D1)."""
        return policy_index_unsent and self.is_full

    def shows_edit_history(self, policy_index_edit_history: bool) -> bool:
        """`policy.index_edit_history` under `full` scope; never under
        `allowlist` (D1)."""
        return policy_index_edit_history and self.is_full


def resolve_request_scope(
    conn: psycopg.Connection | None,
    context: AccessContext,
    *,
    among_chat_ids: Collection[int] | None = None,
) -> RequestScope:
    """The scope one request runs under. `full` needs no database (so
    `conn` may be `None`); anything else evaluates export's chat rule now.

    `among_chat_ids` limits the evaluation to the chats a request can
    touch at all — the one thread `get_conversation` resolved, the parents
    of one attachment — so a single-thread request does not pay for the
    whole corpus. Chats outside it are denied."""
    if context.scope == "full":
        return RequestScope(context=context, eligible_chat_ids=None)
    if conn is None:
        raise ValueError("resolve_request_scope: allowlist scope needs a database connection")
    return RequestScope(
        context=context,
        eligible_chat_ids=frozenset(eligible_chat_ids(conn, among=among_chat_ids)),
        owner_person_id=owner_person_id(conn),
    )


def visible_person_ids(conn: psycopg.Connection, scope: RequestScope) -> frozenset[int] | None:
    """The people a request may see by name: `None` (everyone) under
    `full` scope; under `allowlist`, every participant of an eligible
    chat and every effective sender of a non-unsent message in one. The
    chat rule has already required each of them to be `text_allowed`, so
    this set is the allowlisted people the caller can find content from —
    an allowlisted person with no eligible chat is not in it."""
    if scope.eligible_chat_ids is None:
        return None
    if not scope.eligible_chat_ids:
        return frozenset()
    sender = effective_sender_sql("m")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT cp.person_id FROM chat_participant cp
            WHERE cp.chat_id = ANY(%(chats)s::bigint[])
            UNION
            SELECT {sender} FROM message m
            WHERE m.chat_id = ANY(%(chats)s::bigint[])
              AND NOT m.is_unsent
              AND {sender} IS NOT NULL
            """,
            {"chats": sorted(scope.eligible_chat_ids), "owner": scope.owner_person_id},
        )
        return frozenset(int(person_id) for (person_id,) in cur.fetchall())


@dataclass(frozen=True, slots=True)
class AttachmentGate:
    """Which (segment, attachment) pairs may show attachment content —
    filename, kind and enrichment text — under one request's scope."""

    allowed_pairs: frozenset[tuple[int, int]] | None
    """`None` under `full` scope: every pair."""

    def allows(self, segment_id: int | None, attachment_id: int) -> bool:
        if self.allowed_pairs is None:
            return True
        if segment_id is None:
            # A message in no segment yet: export has no document for it
            # to enter, so there is no verdict to honor. Deny.
            return False
        return (segment_id, attachment_id) in self.allowed_pairs


def attachment_gate(
    conn: psycopg.Connection, scope: RequestScope, segment_ids: Iterable[int]
) -> AttachmentGate:
    """The attachment gate for these segments' attachments. Under
    `allowlist`, a pair is allowed only if its segment's chat is eligible
    and `compute_attachment_eligibility` — export's per-pair rule — says
    so; a pair it does not return (linked only through unsent messages)
    is denied."""
    if scope.is_full:
        return AttachmentGate(allowed_pairs=None)
    requested = sorted({int(s) for s in segment_ids})
    if not requested or scope.admits_nothing:
        return AttachmentGate(allowed_pairs=frozenset())
    with conn.cursor() as cur:
        cur.execute(
            "SELECT segment_id, chat_id FROM segment WHERE segment_id = ANY(%(ids)s::bigint[])",
            {"ids": requested},
        )
        admitted = sorted(
            int(segment_id) for segment_id, chat_id in cur.fetchall() if scope.admits_chat(int(chat_id))
        )
    verdicts = compute_attachment_eligibility(conn, admitted)
    return AttachmentGate(
        allowed_pairs=frozenset(pair for pair, allowed in verdicts.items() if allowed)
    )


__all__ = [
    "LOCAL_FULL_ACCESS",
    "SCOPE_CHAT_IDS_PARAM",
    "AccessContext",
    "AttachmentGate",
    "RequestScope",
    "Scope",
    "ScopePredicate",
    "Surface",
    "attachment_gate",
    "resolve_request_scope",
    "visible_person_ids",
]
