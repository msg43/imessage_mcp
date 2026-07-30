"""Serialize Python float vectors to pgvector's `halfvec` text input
format, for binding as a plain string parameter with an explicit
`::halfvec` cast (SPEC §7.2/§7.4) — this build does not depend on the
`pgvector` psycopg adapter package (not in SPEC §4's dependency table;
only the Postgres *extension* is), so vectors cross the wire as
ordinary bound strings like any other value.
"""

from __future__ import annotations

from collections.abc import Sequence


def vector_literal(values: Sequence[float]) -> str:
    """`[0.1,0.2,...]` — the text form `halfvec`'s input function
    accepts. Bind this as a normal string parameter alongside an
    explicit `%s::halfvec` cast in the SQL (see `imsg.embed.pipeline`).
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


__all__ = ["vector_literal"]
