"""Pull the search path's relations into the buffer pool (`pg_prewarm`,
migration 0004) and report how big they are against `shared_buffers`.

**Why.** An HNSW search touches a few thousand pages of an index far
larger than a default buffer pool, and the pages it wants are scattered:
on the live index (2026-09-16/17) a segment-vector search took 300-900 ms
with cold pages and 7-24 ms once they were cached. Both halves of the fix
live outside the code — `shared_buffers` and `pg_prewarm.autoprewarm` in
the instance's `postgresql.conf` — but the dump the autoprewarm worker
reloads is missing or stale exactly when it matters most (a first start,
a restore, a newly built index), so a server that has just started warms
the pool itself: :func:`prewarm_query_path`, called from
`RetrievalService.warm_up()`.

**What is prewarmed.** Every HNSW index, found in the catalog rather than
listed here (a later migration's index is covered without editing this
file), plus the tables and indexes the query path reads — measured, not
assumed: the 20 fictional benchmark queries were run through the real
service on the live index (2026-09-17) and `pg_statio_user_tables` /
`pg_statio_user_indexes` diffed around them. Every relation with real
traffic is in the lists below, each table with its TOAST relation and
TOAST index (`segment_embedding`'s vectors live in 642 MiB of TOAST, read
on every segment-vector search).

**What is deliberately left out.** `message`'s heap (629 MiB): those 20
queries read 17 MiB of it, so prewarming it would spend a quarter of the
pool on pages the OS page cache serves well enough. Its primary key,
which the summary fetch hits hard, IS prewarmed. Indexes that only ever
saw their metapage touched (`message_sent_idx` and friends) are out for
the same reason.

Sizes as measured on 2026-09-17: prewarming this set moves 2,296 MiB,
against a 3 GB pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import psycopg

QUERY_PATH_TABLES: tuple[str, ...] = (
    "segment_embedding",
    "attachment_mm_embedding",
    "attachment_chunk_embedding",
    "segment",
    "segment_message",
    "message_attachment",
    "chat",
    "chat_participant",
    "person",
)
"""Tables the query path reads, prewarmed with their TOAST relation and
TOAST index. `message` is missing on purpose (module docstring)."""

QUERY_PATH_INDEXES: tuple[str, ...] = (
    "segment_embedding_pkey",
    "attachment_mm_embedding_pkey",
    "attachment_chunk_embedding_pkey",
    "segment_pkey",
    "segment_chat_time_idx",
    "segment_message_pkey",
    "segment_message_single",
    "message_pkey",
    "message_attachment_pkey",
    "message_attachment_attachment_idx",
    "chat_pkey",
    "chat_participant_pkey",
    "chat_participant_person_idx",
    "person_pkey",
)
"""Indexes the query path reads, other than the HNSW ones (those come from
the catalog, `hnsw_index_bytes` / `_hnsw_indexes_sql`)."""

MIB = float(2**20)

_HNSW_INDEXES_SQL = """
SELECT c.oid::regclass::text, pg_relation_size(c.oid), c.oid
FROM pg_class c
JOIN pg_am am ON am.oid = c.relam
JOIN pg_index i ON i.indexrelid = c.oid
WHERE am.amname = 'hnsw'
ORDER BY 1
"""

_NAMED_RELATIONS_SQL = """
SELECT c.relname,
       c.oid::regclass::text,
       pg_relation_size(c.oid),
       t.oid::regclass::text,
       pg_relation_size(t.oid),
       ti.indexrelid::regclass::text,
       pg_relation_size(ti.indexrelid),
       c.oid,
       t.oid,
       ti.indexrelid
FROM pg_class c
LEFT JOIN pg_class t ON t.oid = c.reltoastrelid
LEFT JOIN pg_index ti ON ti.indrelid = t.oid
WHERE c.relname = ANY(%(names)s) AND c.relnamespace = 'public'::regnamespace
"""

_PREWARM_AVAILABLE_SQL = """
SELECT EXISTS (
    SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE p.proname = 'pg_prewarm'
)
"""
"""Whether the extension's function exists at all, by name. NOT
`to_regprocedure('pg_prewarm(regclass)')`, which answers NULL even with
the extension installed: `pg_prewarm` takes five arguments, four of them
defaulted, and that lookup wants an exact signature. The wrong probe
reported "not installed" against an instance where `\\dx` listed
pg_prewarm 1.2 — found by running it against the live database
(2026-09-17)."""

_SHARED_BUFFERS_SQL = """
SELECT setting::bigint * current_setting('block_size')::bigint
FROM pg_settings WHERE name = 'shared_buffers'
"""

PREWARM_UNAVAILABLE = (
    "pg_prewarm is not installed in this database (migration 0004) — the buffer pool is "
    "left to fill itself, so the first searches after a restart read from disk"
)


@dataclass(frozen=True, slots=True)
class PrewarmRelation:
    """One relation to pull into the pool: `name` for reports
    (schema-qualified for TOAST), `size_bytes` from the catalog, and
    `oid`, which is what `pg_prewarm` is given.

    By OID, not by name: turning the name `pg_toast.pg_toast_1234` back
    into a relation checks USAGE on the `pg_toast` schema, which only a
    superuser has. Run by the database's owner without superuser (the
    least-privilege role, QA review 2026-09-24), a name-based prewarm
    skipped every TOAST relation and its index with "permission denied
    for schema pg_toast" (measured on a scratch cluster, 6 of 27
    relations). Casting an OID to `regclass` looks nothing up;
    `pg_prewarm` then checks only SELECT on the relation, which the
    owner of the table holds on its TOAST relation too."""

    name: str
    size_bytes: int
    oid: int = 0


@dataclass(frozen=True, slots=True)
class PrewarmReport:
    relations: int
    blocks: int
    bytes_prewarmed: int
    seconds: float
    error: str | None = None
    """Set when nothing (or not everything) could be prewarmed; the caller
    logs it and carries on — a cold pool is slow, not broken."""
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        """One line for the warm-up log: what was pulled in, and how long."""
        if self.error is not None and self.relations == 0:
            return self.error
        line = (
            f"{self.bytes_prewarmed / MIB:,.0f} MiB in {self.relations} relation(s), "
            f"{self.seconds:.1f} s"
        )
        return f"{line}; {self.error}" if self.error else line


def hnsw_indexes(conn: psycopg.Connection) -> list[PrewarmRelation]:
    """Every HNSW index in the database, with its size — read from the
    catalog so an index a later migration adds is covered."""
    with conn.cursor() as cur:
        cur.execute(_HNSW_INDEXES_SQL)
        return [
            PrewarmRelation(str(name), int(size), int(oid)) for name, size, oid in cur.fetchall()
        ]


def hnsw_index_bytes(conn: psycopg.Connection) -> int:
    """Total size of every HNSW index — what `shared_buffers` must at the
    very least hold for vector search to stay fast (`imsg status`)."""
    return sum(index.size_bytes for index in hnsw_indexes(conn))


def shared_buffers_bytes(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(_SHARED_BUFFERS_SQL)
        row = cur.fetchone()
    return int(row[0]) if row is not None else 0


def query_path_relations(conn: psycopg.Connection) -> list[PrewarmRelation]:
    """Everything :func:`prewarm_query_path` pulls in, in the order it does:
    the HNSW indexes first (the pages a cold vector search waits on), then
    the named tables with their TOAST, then the named indexes. Relations
    that do not exist yet are simply absent."""
    relations = list(hnsw_indexes(conn))
    seen = {relation.name for relation in relations}
    with conn.cursor() as cur:
        cur.execute(_NAMED_RELATIONS_SQL, {"names": list(QUERY_PATH_TABLES)})
        by_name = {str(row[0]): row for row in cur.fetchall()}
        for name in QUERY_PATH_TABLES:
            row = by_name.get(name)
            if row is None:
                continue
            for relation, size, oid in (
                (row[1], row[2], row[7]),
                (row[3], row[4], row[8]),
                (row[5], row[6], row[9]),
            ):
                if relation is None or str(relation) in seen:
                    continue
                seen.add(str(relation))
                relations.append(PrewarmRelation(str(relation), int(size or 0), int(oid)))
        cur.execute(_NAMED_RELATIONS_SQL, {"names": list(QUERY_PATH_INDEXES)})
        index_rows = {str(row[0]): row for row in cur.fetchall()}
    for name in QUERY_PATH_INDEXES:
        row = index_rows.get(name)
        if row is None or str(row[1]) in seen:
            continue
        seen.add(str(row[1]))
        relations.append(PrewarmRelation(str(row[1]), int(row[2] or 0), int(row[7])))
    return relations


def prewarm_query_path(
    conn: psycopg.Connection, *, clock: Callable[[], float] = perf_counter
) -> PrewarmReport:
    """Read every :func:`query_path_relations` relation into the buffer
    pool and report the blocks, bytes and seconds it took.

    Never raises for a database-side problem: a missing extension or a
    relation that cannot be prewarmed is reported in the result, because
    an unwarmed pool makes searches slow, not wrong, and a server that
    refuses to start over it would be worse than one that is briefly
    cold.
    """
    started = clock()
    with conn.cursor() as cur:
        cur.execute(_PREWARM_AVAILABLE_SQL)
        row = cur.fetchone()
        if row is None or not row[0]:
            return PrewarmReport(0, 0, 0, clock() - started, error=PREWARM_UNAVAILABLE)
    relations = query_path_relations(conn)
    blocks = 0
    done = 0
    failures: list[str] = []
    for relation in relations:
        if relation.size_bytes == 0:
            continue  # an empty table has nothing to read
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_prewarm(%(oid)s::oid::regclass)", {"oid": relation.oid})
                result = cur.fetchone()
            blocks += int(result[0]) if result and result[0] is not None else 0
            done += 1
        except Exception as exc:  # one relation's problem is not the server's
            failures.append(f"{relation.name}: {type(exc).__name__}: {exc}")
    with conn.cursor() as cur:
        cur.execute("SELECT current_setting('block_size')::bigint")
        row = cur.fetchone()
    block_size = int(row[0]) if row is not None else 8192
    error = (
        f"{len(failures)} relation(s) could not be prewarmed: {failures[0]}" if failures else None
    )
    return PrewarmReport(
        relations=done,
        blocks=blocks,
        bytes_prewarmed=blocks * block_size,
        seconds=clock() - started,
        error=error,
        failures=tuple(failures),
    )


__all__ = [
    "MIB",
    "PREWARM_UNAVAILABLE",
    "QUERY_PATH_INDEXES",
    "QUERY_PATH_TABLES",
    "PrewarmRelation",
    "PrewarmReport",
    "hnsw_index_bytes",
    "hnsw_indexes",
    "prewarm_query_path",
    "query_path_relations",
    "shared_buffers_bytes",
]
