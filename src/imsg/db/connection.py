"""Open a connection to the dedicated imessage-index Postgres instance.

This is the only place in the codebase that should call
`psycopg.connect` directly — everything else takes a `Config` or an
already-open connection, so there is exactly one place that knows how
to turn `database.dsn` + `database.password` into a live connection.
"""

from __future__ import annotations

import psycopg

from imsg.config.schema import DatabaseConfig


def connect(database: DatabaseConfig, *, autocommit: bool = True) -> psycopg.Connection:
    """Connect using `database.dsn`, resolving `database.password` first.

    Never logs or includes the resolved password in any exception —
    `psycopg` itself is careful about this, and we do not touch the
    resolved value except to hand it straight to `psycopg.connect`.

    **`autocommit` defaults to True, and that is load-bearing (2026-08-15).**

    With `autocommit=False`, psycopg3 implicitly `BEGIN`s on the first
    statement of any kind — including a bare `SELECT`. Every stage in this
    codebase opens with a read (dirty-chat scans, pending-segment queries,
    invariant reports), so by the time it reached its `with
    conn.transaction():` write block, a transaction was already open and
    that block degraded to a **SAVEPOINT**. Since no CLI command called
    `conn.commit()`, the `finally: conn.close()` in each command rolled the
    whole thing back — while the command printed its success summary and
    exited 0.

    Measured on the real corpus: `extract` reported
    `messages_upserted=655494` with `n_tup_ins=1511805, n_live_tup=0,
    n_tup_del=0`. Inserted, then discarded. `segment`, `embed`, `enrich`
    and `sync`'s S4/S6 half had the identical defect.

    An earlier attempt fixed this per-stage by issuing `conn.rollback()`
    after each stage's opening read. That was whack-a-mole: it only closed
    the transaction the fingerprint check opened, and any stage whose first
    statement is a bare read re-opened one immediately.

    `autocommit=True` is the structural fix. Reads no longer start
    transactions, so `conn.transaction()` is the *only* transaction
    mechanism and always produces a genuine top-level transaction that
    commits on exit. Verified:

        autocommit=False: read -> INTRANS; transaction() -> INTRANS (lost at close)
        autocommit=True : read -> IDLE;    transaction() -> IDLE (committed)

    Consequence to know: any write NOT wrapped in `conn.transaction()` now
    commits on its own. That is correct for the enrichment queue's
    single-statement `complete_task`/`fail_task` updates, and every
    multi-statement write in this codebase is already inside a
    `conn.transaction()` block, which keeps its atomicity.
    """
    password = database.password.resolve()
    return psycopg.connect(database.dsn, password=password, autocommit=autocommit)


__all__ = ["connect"]
