"""Opaque, sha256-derived entity keys (SPEC §10.2, D6).

MCP responses carry `segment_key` / `thread_key` / `message_key` /
`attachment_key` instead of raw database integers or source `guid`s —
"opaque keys (D6): sha256-derived opaque strings — never enumerable
database integers and never source GUIDs" (SPEC §10.2). The point is
two-fold: keys must not leak ordering/volume information the way a
sequential bigint would, and they must not double as the same value
`chat.db` itself uses (which would make the opaque key reversible to a
real GUID by anyone who also has a `chat.db`).

This module is the single place that derivation happens, so every
producer (S2 for `message_key`/`thread_key`/`attachment_key`; a later
S4 build for `segment_key`) agrees on the same formula and salts.
`derive_key` is intentionally generic — it is not specific to any one
entity kind — so new opaque-keyed entities can reuse it without a new
hashing scheme.

This is a one-way derivation (sha256 of a namespaced string), not
encryption: it is "opaque" in the sense of "not a raw database id and
not the source GUID," not in the sense of being secret-safe against an
attacker who also has row-level database access. That matches the
spec's own framing (existence-oracle avoidance on the public MCP
surface — SPEC §10.1/§10.2 — not confidentiality of the key itself).
"""

from __future__ import annotations

from imsg.hashing import sha256_text


def derive_key(kind: str, source_guid: str) -> str:
    """The opaque key for one `(kind, source_guid)` pair.

    `kind` namespaces the hash so the same `chat.db` guid used for two
    different entity kinds (unlikely, but chat.db guids are just
    strings) can never collide across kinds.
    """
    if not kind:
        raise ValueError("derive_key: 'kind' must be non-empty")
    if not source_guid:
        raise ValueError("derive_key: 'source_guid' must be non-empty")
    return sha256_text(f"{kind}:{source_guid}")


def message_key(source_guid: str) -> str:
    """Opaque key for a `message` row, derived from `message.guid`."""
    return derive_key("message", source_guid)


def thread_key(source_guid: str) -> str:
    """Opaque key for a `chat` row, derived from `chat.guid`.

    Named `thread_key` (not `chat_key`) to match the column name in
    `chat.thread_key` and the MCP-facing term (SPEC §7.2, §10.2).
    """
    return derive_key("thread", source_guid)


def attachment_key(source_guid: str) -> str:
    """Opaque key for an `attachment` row, derived from `attachment.guid`."""
    return derive_key("attachment", source_guid)


__all__ = ["attachment_key", "derive_key", "message_key", "thread_key"]
