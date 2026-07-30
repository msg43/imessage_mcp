"""Open a connection to the dedicated imessage-index Postgres instance.

This is the only place in the codebase that should call
`psycopg.connect` directly — everything else takes a `Config` or an
already-open connection, so there is exactly one place that knows how
to turn `database.dsn` + `database.password` into a live connection.
"""

from __future__ import annotations

import psycopg

from imsg.config.schema import DatabaseConfig


def connect(database: DatabaseConfig, *, autocommit: bool = False) -> psycopg.Connection:
    """Connect using `database.dsn`, resolving `database.password` first.

    Never logs or includes the resolved password in any exception —
    `psycopg` itself is careful about this, and we do not touch the
    resolved value except to hand it straight to `psycopg.connect`.
    """
    password = database.password.resolve()
    return psycopg.connect(database.dsn, password=password, autocommit=autocommit)


__all__ = ["connect"]
